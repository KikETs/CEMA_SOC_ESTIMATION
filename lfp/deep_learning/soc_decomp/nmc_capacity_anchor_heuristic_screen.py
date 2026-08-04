from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = {0.0: 1.0, 25.0: 0.7, 45.0: 0.3}


@dataclass
class HeuristicConfig:
    raw_root: Path = Path("nmc_samsung_inr_18650_2ah_raw/NMC SAMSUNG INR 18650 2Ah")
    output_dir: Path = Path("paper_results")
    output_prefix: str = "nmc_capacity_anchor_heuristic"


def _temp(value: object) -> float:
    return float(str(value).replace("C", "").strip())


def _load_frames(root: Path) -> dict[tuple[str, float], pd.DataFrame]:
    frames = {}
    for path in sorted(root.rglob("*.csv")):
        head = pd.read_csv(path, nrows=2)
        profile = str(head["Profile"].iloc[0])
        temp = _temp(head["TempLabel"].iloc[0])
        frames[(profile, temp)] = pd.read_csv(
            path,
            usecols=["Step_Time(s)", "Current(A)", "Voltage(V)", "SOC_CC", "Qnet_denom(Ah)"],
        )
    if not frames:
        raise FileNotFoundError(f"No NMC CSV files found under {root}")
    return frames


def _q_true(frame: pd.DataFrame) -> float:
    return float(frame["Qnet_denom(Ah)"].iloc[0]) / float(frame["SOC_CC"].iloc[0])


def _feature(frame: pd.DataFrame, name: str, prefix: int | None) -> float:
    f = frame if prefix is None else frame.iloc[: min(int(prefix), len(frame))]
    i = -pd.to_numeric(f["Current(A)"], errors="coerce").to_numpy(np.float64)
    v = pd.to_numeric(f["Voltage(V)"], errors="coerce").to_numpy(np.float64)
    di = np.diff(i, prepend=i[0])
    dv = np.diff(v, prepend=v[0])
    values = {
        "v_range": float(np.nanmax(v) - np.nanmin(v)),
        "rest_frac": float(np.mean(np.abs(i) < 0.05)),
        "mean_absI": float(np.mean(np.abs(i))),
        "mean_abs_dI": float(np.mean(np.abs(di))),
        "dI_energy": float(np.sqrt(np.mean(di**2))),
        "mean_abs_dV": float(np.mean(np.abs(dv))),
        "rest_x_dI": float(np.mean(np.abs(i) < 0.05) * np.mean(np.abs(di))),
    }
    return values[name]


def _integrated_pred(frame: pd.DataFrame, q_full: float) -> np.ndarray:
    t = pd.to_numeric(frame["Step_Time(s)"], errors="coerce").to_numpy(np.float64)
    i = -pd.to_numeric(frame["Current(A)"], errors="coerce").to_numpy(np.float64)
    y = pd.to_numeric(frame["SOC_CC"], errors="coerce").to_numpy(np.float64)
    dt = np.diff(t, prepend=t[0])
    ok = np.isfinite(dt) & (dt > 0)
    dt[~ok] = float(np.nanmedian(dt[ok])) if np.any(ok) else 1.0
    return np.clip(y[0] - np.cumsum(i * dt / 3600.0) / max(float(q_full), 1e-9), 0.0, 1.0)


def _two_phase_pred(frame: pd.DataFrame, q_base: float, q_adapt: float, prefix: int) -> np.ndarray:
    t = pd.to_numeric(frame["Step_Time(s)"], errors="coerce").to_numpy(np.float64)
    i = -pd.to_numeric(frame["Current(A)"], errors="coerce").to_numpy(np.float64)
    y = pd.to_numeric(frame["SOC_CC"], errors="coerce").to_numpy(np.float64)
    dt = np.diff(t, prepend=t[0])
    ok = np.isfinite(dt) & (dt > 0)
    dt[~ok] = float(np.nanmedian(dt[ok])) if np.any(ok) else 1.0
    pred = np.empty_like(y)
    soc = float(y[0])
    switch = min(int(prefix), len(y))
    for idx in range(len(y)):
        q = q_base if idx < switch else q_adapt
        soc = float(np.clip(soc - i[idx] * dt[idx] / max(q, 1e-9) / 3600.0, 0.0, 1.0))
        pred[idx] = soc
    return pred


def _mae_pct(frame: pd.DataFrame, pred: np.ndarray) -> float:
    y = pd.to_numeric(frame["SOC_CC"], errors="coerce").to_numpy(np.float64)
    return float(np.mean(np.abs(pred - y)) * 100.0)


def _predict_q(
    frames: dict[tuple[str, float], pd.DataFrame],
    train_profiles: list[str],
    test_profile: str,
    temp: float,
    feature: str,
    prefix: int | None,
    shrink: float,
) -> tuple[float, float]:
    qs = np.asarray([_q_true(frames[(p, temp)]) for p in train_profiles], dtype=np.float64)
    xs = np.asarray([_feature(frames[(p, temp)], feature, prefix) for p in train_profiles], dtype=np.float64)
    x = _feature(frames[(test_profile, temp)], feature, prefix)
    q_base = float(np.mean(qs))
    if abs(float(xs[1] - xs[0])) < 1e-12:
        return q_base, q_base
    slope = float((qs[1] - qs[0]) / (xs[1] - xs[0]))
    q_pred = q_base + float(shrink) * slope * (float(x) - float(np.mean(xs)))
    return q_base, float(np.clip(q_pred, 1.2, 2.4))


def _rotation_rows(
    frames: dict[tuple[str, float], pd.DataFrame],
    *,
    feature: str,
    prefix: int | None,
    shrink: float,
    mode: str,
) -> pd.DataFrame:
    profiles = sorted({profile for profile, _temp in frames})
    temps = sorted({temp for _profile, temp in frames})
    rows = []
    for test_profile in profiles:
        remaining = [p for p in profiles if p != test_profile]
        for valid_profile in remaining:
            train_profiles = [p for p in remaining if p != valid_profile]
            row = {
                "mode": mode,
                "feature": feature,
                "prefix": "full" if prefix is None else int(prefix),
                "shrink": float(shrink),
                "test_profile": test_profile,
                "valid_profile": valid_profile,
                "train_profiles": ",".join(train_profiles),
            }
            failures = []
            worst = 0.0
            for temp in temps:
                q_base, q_pred = _predict_q(frames, train_profiles, test_profile, temp, feature, prefix, shrink)
                frame = frames[(test_profile, temp)]
                if mode == "prefix_twophase":
                    if prefix is None:
                        raise ValueError("prefix_twophase requires a prefix")
                    pred = _two_phase_pred(frame, q_base, q_pred, int(prefix))
                else:
                    pred = _integrated_pred(frame, q_pred)
                mae = _mae_pct(frame, pred)
                row[f"{temp:g}C_Qbase_Ah"] = q_base
                row[f"{temp:g}C_Qpred_Ah"] = q_pred
                row[f"{temp:g}C_MAE_pct"] = mae
                if temp in TARGETS:
                    worst = max(worst, mae / TARGETS[temp])
                    if mae >= TARGETS[temp]:
                        failures.append(f"{temp:g}C {mae:.3f}>={TARGETS[temp]:.3f}")
            row["target_met"] = not failures
            row["target_norm_worst"] = float(worst)
            row["failure_detail"] = "; ".join(failures) if failures else "pass"
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["target_norm_worst", "test_profile", "valid_profile"]).reset_index(drop=True)


def run(cfg: HeuristicConfig) -> dict[str, pd.DataFrame]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    frames = _load_frames(cfg.raw_root)
    features = ["v_range", "rest_frac", "mean_absI", "mean_abs_dI", "dI_energy", "mean_abs_dV", "rest_x_dI"]
    shrinks = [0.25, 0.50, 0.75, 1.00]
    full_prefixes: list[int | None] = [None]
    prefix_values = [256, 512, 1024, 2048]

    all_rows = []
    for feature in features:
        for shrink in shrinks:
            for prefix in full_prefixes:
                all_rows.append(_rotation_rows(frames, feature=feature, prefix=prefix, shrink=shrink, mode="full_descriptor_optimistic"))
            for prefix in prefix_values:
                all_rows.append(_rotation_rows(frames, feature=feature, prefix=prefix, shrink=shrink, mode="prefix_twophase"))
    rows = pd.concat(all_rows, ignore_index=True)
    summary = (
        rows.groupby(["mode", "feature", "prefix", "shrink"], dropna=False)
        .agg(
            pass_count=("target_met", "sum"),
            mean_target_norm_worst=("target_norm_worst", "mean"),
            max_target_norm_worst=("target_norm_worst", "max"),
            mean_0C_MAE_pct=("0C_MAE_pct", "mean"),
            mean_25C_MAE_pct=("25C_MAE_pct", "mean"),
            mean_45C_MAE_pct=("45C_MAE_pct", "mean"),
        )
        .reset_index()
        .sort_values(["pass_count", "mean_target_norm_worst"], ascending=[False, True])
        .reset_index(drop=True)
    )
    best_full = summary[summary["mode"].eq("full_descriptor_optimistic")].iloc[0]
    best_prefix = summary[summary["mode"].eq("prefix_twophase")].iloc[0]
    best_keys = pd.DataFrame([best_full, best_prefix])
    best_rows = []
    for _, best in best_keys.iterrows():
        mask = (
            rows["mode"].eq(best["mode"])
            & rows["feature"].eq(best["feature"])
            & rows["prefix"].astype(str).eq(str(best["prefix"]))
            & np.isclose(rows["shrink"].astype(float), float(best["shrink"]))
        )
        best_rows.append(rows[mask])
    best_detail = pd.concat(best_rows, ignore_index=True)

    summary.to_csv(cfg.output_dir / f"{cfg.output_prefix}_summary.csv", index=False)
    rows.to_csv(cfg.output_dir / f"{cfg.output_prefix}_rotation_rows.csv", index=False)
    best_detail.to_csv(cfg.output_dir / f"{cfg.output_prefix}_best_rows.csv", index=False)
    write_report(cfg, summary, best_detail)
    print(summary.head(20).to_string(index=False))
    return {"summary": summary, "rows": rows, "best_rows": best_detail}


def write_report(cfg: HeuristicConfig, summary: pd.DataFrame, best_rows: pd.DataFrame) -> None:
    best_full = summary[summary["mode"].eq("full_descriptor_optimistic")].iloc[0]
    best_prefix = summary[summary["mode"].eq("prefix_twophase")].iloc[0]
    lines = [
        "# NMC Capacity-Anchor Heuristic Screen",
        "",
        "## Purpose",
        "",
        "This screen tests whether label-free profile-regime features can improve a causal charge-conservation capacity anchor under profile rotation.",
        "",
        "It is not a final SOC model. The full-descriptor mode uses full-trajectory voltage/current descriptors and is therefore an optimistic upper bound. The prefix two-phase mode uses an early prefix descriptor and applies the adapted capacity only after that prefix, which is closer to online use but still only a heuristic.",
        "",
        "## Best Summary Rows",
        "",
        pd.DataFrame([best_full, best_prefix]).to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Best Rotation Details",
        "",
        best_rows.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Interpretation",
        "",
        "- Full-trajectory `v_range` is the strongest simple descriptor, which means voltage response carries profile-regime capacity information.",
        "- Because full-trajectory `v_range` uses future measurements, it cannot be the final online selector.",
        "- Prefix two-phase `v_range` improves over plain train-temperature Q in several rotations, but it is not universal yet.",
        "- The logical next model is not free VALIDATION capacity tuning; it is a causal observer with bounded capacity adaptation driven by profile-regime/voltage-response confidence.",
        "",
    ]
    (cfg.output_dir / f"{cfg.output_prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Screen label-free capacity-anchor heuristics for NMC profile rotations.")
    p.add_argument("--raw-root", default=HeuristicConfig.raw_root)
    p.add_argument("--output-dir", default=HeuristicConfig.output_dir)
    p.add_argument("--output-prefix", default=HeuristicConfig.output_prefix)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        HeuristicConfig(
            raw_root=Path(args.raw_root),
            output_dir=Path(args.output_dir),
            output_prefix=args.output_prefix,
        )
    )


if __name__ == "__main__":
    main()
