from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ChargeSanityConfig:
    raw_root: Path = Path("nmc_samsung_inr_18650_2ah_raw/NMC SAMSUNG INR 18650 2Ah")
    output_dir: Path = Path("paper_results")
    output_prefix: str = "nmc_charge_conservation_sanity"
    train_profiles: tuple[str, ...] = ("DST", "US06")
    valid_profiles: tuple[str, ...] = ("VALIDATION",)
    test_profiles: tuple[str, ...] = ("FUDS",)


def _parse_csv_tuple(text: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in text.split(",") if x.strip())


def _temp_from_label(value: object) -> float:
    return float(str(value).replace("C", "").strip())


def _files(root: Path) -> list[Path]:
    files = sorted(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files under {root}")
    return files


def estimate_train_capacity(files: list[Path], cfg: ChargeSanityConfig) -> pd.DataFrame:
    rows = []
    for path in files:
        df = pd.read_csv(path, usecols=["TempLabel", "Profile", "SOC_CC", "Qnet_denom(Ah)"])
        profile = str(df["Profile"].iloc[0])
        if profile not in cfg.train_profiles:
            continue
        temp = _temp_from_label(df["TempLabel"].iloc[0])
        soc0 = float(df["SOC_CC"].iloc[0])
        q_avail = float(df["Qnet_denom(Ah)"].iloc[0])
        rows.append(
            {
                "temperature_C": temp,
                "profile": profile,
                "soc0": soc0,
                "q_available_to_cutoff_Ah": q_avail,
                "q_full_equivalent_Ah": q_avail / max(soc0, 1e-9),
            }
        )
    out = pd.DataFrame(rows).sort_values(["temperature_C", "profile"]).reset_index(drop=True)
    if out.empty:
        raise RuntimeError("No train capacity rows found.")
    return out


def _integrate_positive_discharge(df: pd.DataFrame) -> np.ndarray:
    t = pd.to_numeric(df["Step_Time(s)"], errors="coerce").to_numpy(np.float64)
    i_dis = -pd.to_numeric(df["Current(A)"], errors="coerce").to_numpy(np.float64)
    dt = np.diff(t, prepend=t[0])
    ok = np.isfinite(dt) & (dt > 0)
    default_dt = float(np.nanmedian(dt[ok])) if np.any(ok) else 1.0
    dt[~ok] = default_dt
    return np.cumsum(i_dis * dt / 3600.0)


def _eval_q_full(df: pd.DataFrame, q_full: float) -> tuple[float, float, float]:
    y = pd.to_numeric(df["SOC_CC"], errors="coerce").to_numpy(np.float64)
    soc0 = float(y[0])
    ah = _integrate_positive_discharge(df)
    pred = np.clip(soc0 - ah / max(float(q_full), 1e-9), 0.0, 1.0)
    err = pred - y
    return (
        float(np.mean(np.abs(err)) * 100.0),
        float(np.mean(err) * 100.0),
        float(pred[-1]),
    )


def evaluate(files: list[Path], cfg: ChargeSanityConfig, cap: pd.DataFrame) -> pd.DataFrame:
    q_temp = cap.groupby("temperature_C")["q_full_equivalent_Ah"].mean().to_dict()
    q_global = float(cap["q_full_equivalent_Ah"].mean())
    rows = []
    eval_profiles = set(cfg.valid_profiles) | set(cfg.test_profiles)
    for path in files:
        df = pd.read_csv(path, usecols=["TempLabel", "Profile", "Step_Time(s)", "Current(A)", "SOC_CC"])
        profile = str(df["Profile"].iloc[0])
        if profile not in eval_profiles:
            continue
        temp = _temp_from_label(df["TempLabel"].iloc[0])
        split = "valid" if profile in cfg.valid_profiles else "test"
        y = pd.to_numeric(df["SOC_CC"], errors="coerce").to_numpy(np.float64)
        soc0 = float(y[0])
        ah = _integrate_positive_discharge(df)
        variants = {
            "nominal_2Ah": 2.0,
            "train_global_Qfull": q_global,
            "train_temp_Qfull": float(q_temp[temp]),
        }
        for name, q_full in variants.items():
            pred = np.clip(soc0 - ah / max(q_full, 1e-9), 0.0, 1.0)
            err = pred - y
            rows.append(
                {
                    "split": split,
                    "profile": profile,
                    "temperature_C": temp,
                    "model_name": name,
                    "q_full_used_Ah": float(q_full),
                    "n_points": int(len(df)),
                    "MAE_pct": float(np.mean(np.abs(err)) * 100.0),
                    "RMSE_pct": float(np.sqrt(np.mean(err**2)) * 100.0),
                    "bias_pct": float(np.mean(err) * 100.0),
                    "pred_end_soc": float(pred[-1]),
                    "true_end_soc": float(y[-1]),
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "model_name", "temperature_C"]).reset_index(drop=True)


def validation_capacity_selector(files: list[Path], cfg: ChargeSanityConfig, cap: pd.DataFrame) -> pd.DataFrame:
    q_temp = cap.groupby("temperature_C")["q_full_equivalent_Ah"].mean().to_dict()
    frames: dict[tuple[str, float], pd.DataFrame] = {}
    for path in files:
        head = pd.read_csv(path, nrows=2)
        profile = str(head["Profile"].iloc[0])
        temp = _temp_from_label(head["TempLabel"].iloc[0])
        if profile in set(cfg.valid_profiles) | set(cfg.test_profiles):
            frames[(profile, temp)] = pd.read_csv(
                path,
                usecols=["TempLabel", "Profile", "Step_Time(s)", "Current(A)", "SOC_CC"],
            )
    rows = []
    for temp, q_base in sorted(q_temp.items()):
        valid_frames = [frames[(p, temp)] for p in cfg.valid_profiles if (p, temp) in frames]
        test_frames = [frames[(p, temp)] for p in cfg.test_profiles if (p, temp) in frames]
        if not valid_frames or not test_frames:
            continue
        q_grid = np.linspace(float(q_base) * 0.88, float(q_base) * 1.12, 241)

        def mean_mae(split_frames: list[pd.DataFrame], q: float) -> float:
            return float(np.mean([_eval_q_full(frame, q)[0] for frame in split_frames]))

        valid_mae = np.asarray([mean_mae(valid_frames, q) for q in q_grid], dtype=np.float64)
        test_mae = np.asarray([mean_mae(test_frames, q) for q in q_grid], dtype=np.float64)
        valid_idx = int(np.argmin(valid_mae))
        test_idx = int(np.argmin(test_mae))
        q_valid = float(q_grid[valid_idx])
        q_oracle = float(q_grid[test_idx])
        base_test = mean_mae(test_frames, float(q_base))
        valid_selected_test = mean_mae(test_frames, q_valid)
        oracle_test = mean_mae(test_frames, q_oracle)
        _, valid_bias_on_test, valid_pred_end = _eval_q_full(test_frames[0], q_valid)
        rows.append(
            {
                "temperature_C": float(temp),
                "q_train_avg_Ah": float(q_base),
                "q_valid_selected_Ah": q_valid,
                "q_oracle_test_Ah": q_oracle,
                "valid_MAE_at_valid_selected_pct": float(valid_mae[valid_idx]),
                "test_MAE_at_train_avg_pct": float(base_test),
                "test_MAE_at_valid_selected_pct": float(valid_selected_test),
                "test_MAE_at_oracle_pct": float(oracle_test),
                "test_bias_at_valid_selected_pct": float(valid_bias_on_test),
                "test_pred_end_at_valid_selected": float(valid_pred_end),
                "selector_helps_test": bool(valid_selected_test < base_test),
            }
        )
    return pd.DataFrame(rows)


def profile_rotation_sanity(files: list[Path]) -> pd.DataFrame:
    frames: dict[tuple[str, float], pd.DataFrame] = {}
    profiles: list[str] = []
    for path in files:
        head = pd.read_csv(path, nrows=2)
        profile = str(head["Profile"].iloc[0])
        temp = _temp_from_label(head["TempLabel"].iloc[0])
        profiles.append(profile)
        frames[(profile, temp)] = pd.read_csv(
            path,
            usecols=["Step_Time(s)", "Current(A)", "SOC_CC", "Qnet_denom(Ah)"],
        )
    profiles = sorted(set(profiles))
    temps = sorted({temp for _, temp in frames})
    targets = {0.0: 1.0, 25.0: 0.7, 45.0: 0.3}
    rows = []
    for test_profile in profiles:
        remaining = [p for p in profiles if p != test_profile]
        for valid_profile in remaining:
            train_profiles = [p for p in remaining if p != valid_profile]
            row = {
                "test_profile": test_profile,
                "valid_profile": valid_profile,
                "train_profiles": ",".join(train_profiles),
            }
            worst = 0.0
            failures = []
            for temp in temps:
                q_vals = []
                for profile in train_profiles:
                    frame = frames[(profile, temp)]
                    q_vals.append(float(frame["Qnet_denom(Ah)"].iloc[0]) / float(frame["SOC_CC"].iloc[0]))
                q_train = float(np.mean(q_vals))
                mae, _bias, _end = _eval_q_full(frames[(test_profile, temp)], q_train)
                row[f"{temp:g}C_Qtrain_Ah"] = q_train
                row[f"{temp:g}C_MAE_pct"] = mae
                if temp in targets:
                    worst = max(worst, mae / targets[temp])
                    if mae >= targets[temp]:
                        failures.append(f"{temp:g}C {mae:.3f}>={targets[temp]:.3f}")
            row["target_met"] = not failures
            row["target_norm_worst"] = float(worst)
            row["failure_detail"] = "; ".join(failures) if failures else "pass"
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["target_norm_worst", "test_profile", "valid_profile"]).reset_index(drop=True)


def focus_table(metrics: pd.DataFrame) -> pd.DataFrame:
    targets = {0.0: 1.0, 25.0: 0.7, 45.0: 0.3}
    test = metrics[metrics["split"].eq("test")].copy()
    rows = []
    for model, g in test.groupby("model_name"):
        details = []
        worst = 0.0
        for temp, target in targets.items():
            row = g[np.isclose(g["temperature_C"].astype(float), temp)]
            if row.empty:
                details.append(f"{temp:g}C missing")
                worst = np.inf
                continue
            mae = float(row["MAE_pct"].iloc[0])
            worst = max(worst, mae / target)
            if mae >= target:
                details.append(f"{temp:g}C {mae:.3f}>={target:.3f}")
        rows.append(
            {
                "model_name": model,
                "target_met": not details,
                "target_norm_worst": float(worst),
                "failure_detail": "; ".join(details) if details else "pass",
            }
        )
    return pd.DataFrame(rows)


def write_report(
    cfg: ChargeSanityConfig,
    cap: pd.DataFrame,
    metrics: pd.DataFrame,
    focus: pd.DataFrame,
    selector: pd.DataFrame,
    rotation: pd.DataFrame,
) -> None:
    lines = [
        "# NMC Charge-Conservation Sanity Check",
        "",
        "## Purpose",
        "",
        "This is not a strict NoCC experiment and not a learned deep model. It is a causal current-integration sanity check used to locate the boundary between NoCC failure and observer-style SOC estimation.",
        "",
        "The baseline estimates a full-equivalent capacity from train profiles only, then integrates measured current online from the known initial SOC.",
        "",
        "## Train Capacity Estimates",
        "",
        cap.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Validation/Test Metrics",
        "",
        metrics.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## FUDS Target Verdict",
        "",
        focus.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Validation-Selected Capacity Check",
        "",
        "This check asks whether a paper-safe VALIDATION validation selector can choose a better temperature capacity for the hidden FUDS profile.",
        "",
        selector.to_markdown(index=False, floatfmt=".3f") if not selector.empty else "(no selector rows)",
        "",
        "Key point: at 0C, VALIDATION and FUDS require opposite capacity corrections. Selecting Q by VALIDATION makes FUDS worse, not better.",
        "",
        "## Profile-Rotation Sanity",
        "",
        "This rotates the held-out test profile and estimates temperature-level Q from two train profiles.",
        "",
        rotation.to_markdown(index=False, floatfmt=".3f") if not rotation.empty else "(no rotation rows)",
        "",
        "Key point: simple train-temperature Q can pass for some held-out profiles, but FUDS/VALIDATION remain fragile at 0C. This is a profile-coverage problem, not just an epoch-selection problem.",
        "",
        "## Interpretation",
        "",
        "- Temperature-level charge conservation nearly solves 25C and 45C FUDS, but 0C remains above the 1.0% target.",
        "- This supports using a causal observer main track if the paper explicitly declares current integration and initial SOC.",
        "- It does not rescue the strict NoCC claim; it shows why NoCC should remain an ablation/failure-analysis track.",
        "- The next logical main-track model should learn a voltage-residual/capacity correction around this causal charge path, not use Stage2 cold/hot correction after a 25C-selected checkpoint.",
        "- A VALIDATION-only validation selector is not sufficient for universal FUDS adoption; it can be actively misleading at 0C.",
        "",
    ]
    (cfg.output_dir / f"{cfg.output_prefix}.md").write_text("\n".join(lines), encoding="utf-8")


def run(cfg: ChargeSanityConfig) -> dict[str, pd.DataFrame]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    files = _files(cfg.raw_root)
    cap = estimate_train_capacity(files, cfg)
    metrics = evaluate(files, cfg, cap)
    focus = focus_table(metrics)
    selector = validation_capacity_selector(files, cfg, cap)
    rotation = profile_rotation_sanity(files)
    cap.to_csv(cfg.output_dir / f"{cfg.output_prefix}_capacity.csv", index=False)
    metrics.to_csv(cfg.output_dir / f"{cfg.output_prefix}_metrics.csv", index=False)
    focus.to_csv(cfg.output_dir / f"{cfg.output_prefix}_focus.csv", index=False)
    selector.to_csv(cfg.output_dir / f"{cfg.output_prefix}_validation_selector.csv", index=False)
    rotation.to_csv(cfg.output_dir / f"{cfg.output_prefix}_profile_rotation.csv", index=False)
    write_report(cfg, cap, metrics, focus, selector, rotation)
    print(metrics.to_string(index=False))
    print(focus.to_string(index=False))
    print(selector.to_string(index=False))
    print(rotation.to_string(index=False))
    return {"capacity": cap, "metrics": metrics, "focus": focus, "selector": selector, "rotation": rotation}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NMC causal charge-conservation baseline sanity check.")
    p.add_argument("--raw-root", default=ChargeSanityConfig.raw_root)
    p.add_argument("--output-dir", default=ChargeSanityConfig.output_dir)
    p.add_argument("--output-prefix", default=ChargeSanityConfig.output_prefix)
    p.add_argument("--train-profiles", default="DST,US06")
    p.add_argument("--valid-profiles", default="VALIDATION")
    p.add_argument("--test-profiles", default="FUDS")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ChargeSanityConfig(
        raw_root=Path(args.raw_root),
        output_dir=Path(args.output_dir),
        output_prefix=args.output_prefix,
        train_profiles=_parse_csv_tuple(args.train_profiles),
        valid_profiles=_parse_csv_tuple(args.valid_profiles),
        test_profiles=_parse_csv_tuple(args.test_profiles),
    )
    run(cfg)


if __name__ == "__main__":
    main()
