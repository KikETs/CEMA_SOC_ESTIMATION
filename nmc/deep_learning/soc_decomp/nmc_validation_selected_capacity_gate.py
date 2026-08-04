from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = {0.0: 1.0, 25.0: 0.7, 45.0: 0.3}


@dataclass
class ValidationGateConfig:
    raw_root: Path = Path("nmc_samsung_inr_18650_2ah_raw/NMC SAMSUNG INR 18650 2Ah")
    output_dir: Path = Path("paper_results")
    output_prefix: str = "nmc_validation_selected_capacity_gate"


def _temp(value: object) -> float:
    return float(str(value).replace("C", "").strip())


def _load(root: Path) -> dict[tuple[str, float], dict[str, np.ndarray | float]]:
    data: dict[tuple[str, float], dict[str, np.ndarray | float]] = {}
    for path in sorted(root.rglob("*.csv")):
        head = pd.read_csv(path, nrows=2)
        profile = str(head["Profile"].iloc[0])
        temp = _temp(head["TempLabel"].iloc[0])
        df = pd.read_csv(path, usecols=["Step_Time(s)", "Current(A)", "Voltage(V)", "SOC_CC", "Qnet_denom(Ah)"])
        t = pd.to_numeric(df["Step_Time(s)"], errors="coerce").to_numpy(np.float64)
        i_dis = -pd.to_numeric(df["Current(A)"], errors="coerce").to_numpy(np.float64)
        v = pd.to_numeric(df["Voltage(V)"], errors="coerce").to_numpy(np.float64)
        y = pd.to_numeric(df["SOC_CC"], errors="coerce").to_numpy(np.float64)
        dt = np.diff(t, prepend=t[0])
        ok = np.isfinite(dt) & (dt > 0)
        dt[~ok] = float(np.nanmedian(dt[ok])) if np.any(ok) else 1.0
        dah = i_dis * dt / 3600.0
        data[(profile, temp)] = {
            "i_dis": i_dis,
            "v": v,
            "y": y,
            "dah": dah,
            "ah": np.cumsum(dah),
            "q_true": float(df["Qnet_denom(Ah)"].iloc[0]) / max(float(y[0]), 1e-9),
        }
    if not data:
        raise FileNotFoundError(f"No CSV files under {root}")
    return data


def _prefix(rec: dict[str, np.ndarray | float], prefix: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v = rec["v"]
    i = rec["i_dis"]
    ah = rec["ah"]
    y = rec["y"]
    assert isinstance(v, np.ndarray) and isinstance(i, np.ndarray) and isinstance(ah, np.ndarray) and isinstance(y, np.ndarray)
    n = min(max(int(prefix), 2), len(v))
    return v[:n], i[:n], ah[:n], y[:n]


def _feature(rec: dict[str, np.ndarray | float], prefix: int, name: str) -> float:
    v, i, ah, _y = _prefix(rec, prefix)
    abs_i = np.abs(i)
    d_i = np.diff(i, prepend=i[0])
    ah_end = max(abs(float(ah[-1])), 1e-6)
    if name == "v_range":
        value = float(np.nanmax(v) - np.nanmin(v))
    elif name == "v_drop":
        value = float(v[0] - v[-1])
    elif name == "v_end":
        value = float(v[-1])
    elif name == "v_mean":
        value = float(np.nanmean(v))
    elif name == "v_std":
        value = float(np.nanstd(v))
    elif name == "v_range_per_ah":
        value = float((np.nanmax(v) - np.nanmin(v)) / ah_end)
    elif name == "v_drop_per_ah":
        value = float((v[0] - v[-1]) / ah_end)
    elif name == "ah_prefix":
        value = float(ah[-1])
    elif name == "abs_i_mean":
        value = float(np.nanmean(abs_i))
    elif name == "i_rms":
        value = float(np.sqrt(np.nanmean(i**2)))
    elif name == "di_rms":
        value = float(np.sqrt(np.nanmean(d_i**2)))
    elif name == "rest_frac":
        value = float(np.nanmean(abs_i < 0.05))
    else:
        raise ValueError(f"Unknown feature: {name}")
    return value if np.isfinite(value) else 0.0


def _q_values(data: dict[tuple[str, float], dict[str, np.ndarray | float]], profiles: list[str], temp: float) -> np.ndarray:
    return np.asarray([float(data[(profile, temp)]["q_true"]) for profile in profiles], dtype=np.float64)


def _predict_q(
    data: dict[tuple[str, float], dict[str, np.ndarray | float]],
    *,
    train_profiles: list[str],
    target_profile: str,
    temp: float,
    feature_name: str,
    prefix: int,
    shrink: float,
    margin: float,
    gate: str,
) -> tuple[float, float, float, float]:
    qs = _q_values(data, train_profiles, temp)
    xs = np.asarray([_feature(data[(profile, temp)], prefix, feature_name) for profile in train_profiles], dtype=np.float64)
    x = _feature(data[(target_profile, temp)], prefix, feature_name)
    q_base = float(np.mean(qs))
    span = abs(float(xs[1] - xs[0]))
    if span < 1e-12:
        q_raw = q_base
    else:
        q_raw = q_base + float(shrink) * float((qs[1] - qs[0]) / (xs[1] - xs[0])) * (x - float(np.mean(xs)))
    extra = float(margin) * abs(float(qs[1] - qs[0]))
    q_bound = float(np.clip(q_raw, float(np.min(qs) - extra), float(np.max(qs) + extra)))
    lo = float(np.min(xs))
    hi = float(np.max(xs))
    outside = max(lo - x, x - hi, 0.0)
    if gate == "none":
        confidence = 1.0
    elif gate == "inside_only":
        confidence = 0.0 if outside > 0.0 else 1.0
    elif gate == "smooth_1span":
        confidence = float(np.exp(-((outside / max(span, 1e-9)) ** 2)))
    elif gate == "smooth_0p5span":
        confidence = float(np.exp(-((outside / max(0.5 * span, 1e-9)) ** 2)))
    else:
        raise ValueError(f"Unknown gate: {gate}")
    q_pred = q_base + confidence * (q_bound - q_base)
    return q_base, float(q_pred), float(x), confidence


def _twophase_mae(rec: dict[str, np.ndarray | float], q_base: float, q_adapt: float, prefix: int) -> float:
    y = rec["y"]
    ah = rec["ah"]
    assert isinstance(y, np.ndarray) and isinstance(ah, np.ndarray)
    sw = min(max(int(prefix), 1), len(y))
    pred = np.empty_like(y)
    pred[:sw] = np.clip(float(y[0]) - ah[:sw] / max(float(q_base), 1e-9), 0.0, 1.0)
    if sw < len(y):
        pred[sw:] = np.clip(pred[sw - 1] - (ah[sw:] - ah[sw - 1]) / max(float(q_adapt), 1e-9), 0.0, 1.0)
    return float(np.mean(np.abs(pred - y)) * 100.0)


def _eval_candidate(
    data: dict[tuple[str, float], dict[str, np.ndarray | float]],
    *,
    train_profiles: list[str],
    profile: str,
    feature_name: str,
    prefix: int,
    shrink: float,
    margin: float,
    gate: str,
) -> dict[str, float | bool | str]:
    temps = sorted({temp for _profile, temp in data})
    out: dict[str, float | bool | str] = {}
    failures = []
    worst = 0.0
    mean_terms = []
    conf_terms = []
    for temp in temps:
        q_base, q_pred, x, confidence = _predict_q(
            data,
            train_profiles=train_profiles,
            target_profile=profile,
            temp=temp,
            feature_name=feature_name,
            prefix=prefix,
            shrink=shrink,
            margin=margin,
            gate=gate,
        )
        mae = _twophase_mae(data[(profile, temp)], q_base, q_pred, prefix)
        out[f"{temp:g}C_Qbase_Ah"] = q_base
        out[f"{temp:g}C_Qpred_Ah"] = q_pred
        out[f"{temp:g}C_feature"] = x
        out[f"{temp:g}C_confidence"] = confidence
        out[f"{temp:g}C_MAE_pct"] = mae
        mean_terms.append(mae / TARGETS.get(temp, 1.0))
        conf_terms.append(confidence)
        if temp in TARGETS:
            worst = max(worst, mae / TARGETS[temp])
            if mae >= TARGETS[temp]:
                failures.append(f"{temp:g}C {mae:.3f}>={TARGETS[temp]:.3f}")
    out["target_met"] = not failures
    out["target_norm_worst"] = float(worst)
    out["target_norm_mean"] = float(np.mean(mean_terms))
    out["mean_confidence"] = float(np.mean(conf_terms))
    out["failure_detail"] = "; ".join(failures) if failures else "pass"
    return out


def _candidate_grid() -> list[dict[str, float | int | str]]:
    features = [
        "v_range",
        "v_drop",
        "v_end",
        "v_mean",
        "v_std",
        "v_range_per_ah",
        "v_drop_per_ah",
        "ah_prefix",
        "abs_i_mean",
        "i_rms",
        "di_rms",
        "rest_frac",
    ]
    rows: list[dict[str, float | int | str]] = []
    for feature in features:
        for prefix in [128, 256, 512]:
            for shrink in [0.25, 0.50, 0.75, 1.00]:
                for margin in [0.0, 0.5, 1.0, 2.0]:
                    for gate in ["none", "inside_only", "smooth_1span", "smooth_0p5span"]:
                        rows.append(
                            {
                                "feature_name": feature,
                                "prefix": int(prefix),
                                "shrink": float(shrink),
                                "margin": float(margin),
                                "gate": gate,
                            }
                        )
    return rows


def run(cfg: ValidationGateConfig) -> dict[str, pd.DataFrame]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    data = _load(cfg.raw_root)
    profiles = sorted({profile for profile, _temp in data})
    candidates = _candidate_grid()
    candidate_rows = []
    selected_rows = []
    oracle_rows = []
    fixed_rows = []

    fixed = {
        "feature_name": "v_range",
        "prefix": 256,
        "shrink": 0.75,
        "margin": 0.5,
        "gate": "none",
    }
    for test_profile in profiles:
        remaining = [p for p in profiles if p != test_profile]
        for valid_profile in remaining:
            train_profiles = [p for p in remaining if p != valid_profile]
            rotation = {
                "test_profile": test_profile,
                "valid_profile": valid_profile,
                "train_profiles": ",".join(train_profiles),
            }
            for cand in candidates:
                valid = _eval_candidate(data, train_profiles=train_profiles, profile=valid_profile, **cand)
                test = _eval_candidate(data, train_profiles=train_profiles, profile=test_profile, **cand)
                row = {**rotation, **cand}
                row.update({f"valid_{k}": v for k, v in valid.items()})
                row.update({f"test_{k}": v for k, v in test.items()})
                candidate_rows.append(row)
            frame = pd.DataFrame(candidate_rows[-len(candidates) :])
            selected = frame.sort_values(
                ["valid_target_norm_worst", "valid_target_norm_mean", "test_profile", "valid_profile"],
                ascending=[True, True, True, True],
            ).iloc[0]
            selected_rows.append({"selection": "valid_selected", **selected.to_dict()})
            oracle = frame.sort_values(
                ["test_target_norm_worst", "test_target_norm_mean", "valid_target_norm_worst"],
                ascending=[True, True, True],
            ).iloc[0]
            oracle_rows.append({"selection": "test_oracle_upper_bound", **oracle.to_dict()})
            valid_fixed = _eval_candidate(data, train_profiles=train_profiles, profile=valid_profile, **fixed)
            test_fixed = _eval_candidate(data, train_profiles=train_profiles, profile=test_profile, **fixed)
            fixed_row = {**rotation, **fixed}
            fixed_row.update({f"valid_{k}": v for k, v in valid_fixed.items()})
            fixed_row.update({f"test_{k}": v for k, v in test_fixed.items()})
            fixed_rows.append({"selection": "fixed_bounded_vrange", **fixed_row})

    candidates_df = pd.DataFrame(candidate_rows)
    selected_df = pd.DataFrame([*fixed_rows, *selected_rows, *oracle_rows])
    summary = (
        selected_df.groupby("selection", dropna=False)
        .agg(
            pass_count=("test_target_met", "sum"),
            mean_test_target_norm_worst=("test_target_norm_worst", "mean"),
            max_test_target_norm_worst=("test_target_norm_worst", "max"),
            mean_test_0C_MAE_pct=("test_0C_MAE_pct", "mean"),
            mean_test_25C_MAE_pct=("test_25C_MAE_pct", "mean"),
            mean_test_45C_MAE_pct=("test_45C_MAE_pct", "mean"),
            mean_valid_target_norm_worst=("valid_target_norm_worst", "mean"),
        )
        .reset_index()
        .sort_values(["pass_count", "mean_test_target_norm_worst"], ascending=[False, True])
    )
    global_upper = (
        candidates_df.groupby(["feature_name", "prefix", "shrink", "margin", "gate"], dropna=False)
        .agg(
            pass_count=("test_target_met", "sum"),
            mean_test_target_norm_worst=("test_target_norm_worst", "mean"),
            max_test_target_norm_worst=("test_target_norm_worst", "max"),
            mean_test_0C_MAE_pct=("test_0C_MAE_pct", "mean"),
            mean_test_25C_MAE_pct=("test_25C_MAE_pct", "mean"),
            mean_test_45C_MAE_pct=("test_45C_MAE_pct", "mean"),
            mean_valid_target_norm_worst=("valid_target_norm_worst", "mean"),
        )
        .reset_index()
        .sort_values(
            ["pass_count", "mean_test_target_norm_worst", "max_test_target_norm_worst"],
            ascending=[False, True, True],
        )
    )
    feature_counts = (
        selected_df[selected_df["selection"].eq("valid_selected")]
        .groupby(["feature_name", "prefix", "shrink", "margin", "gate"], dropna=False)
        .size()
        .reset_index(name="selected_count")
        .sort_values(["selected_count", "feature_name"], ascending=[False, True])
    )

    candidates_df.to_csv(cfg.output_dir / f"{cfg.output_prefix}_candidate_rows.csv", index=False)
    selected_df.to_csv(cfg.output_dir / f"{cfg.output_prefix}_selected_rows.csv", index=False)
    summary.to_csv(cfg.output_dir / f"{cfg.output_prefix}_summary.csv", index=False)
    global_upper.to_csv(cfg.output_dir / f"{cfg.output_prefix}_global_upper_bound.csv", index=False)
    feature_counts.to_csv(cfg.output_dir / f"{cfg.output_prefix}_valid_selection_counts.csv", index=False)
    write_report(cfg, summary, selected_df, feature_counts, global_upper)
    print(summary.to_string(index=False))
    print("\nGlobal fixed-candidate upper bound")
    print(global_upper.head(20).to_string(index=False))
    print("\nValid-selected feature counts")
    print(feature_counts.head(20).to_string(index=False))
    return {
        "candidate_rows": candidates_df,
        "selected_rows": selected_df,
        "summary": summary,
        "global_upper_bound": global_upper,
        "feature_counts": feature_counts,
    }


def write_report(
    cfg: ValidationGateConfig,
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    feature_counts: pd.DataFrame,
    global_upper: pd.DataFrame,
) -> None:
    details = selected[
        selected["selection"].isin(["fixed_bounded_vrange", "valid_selected", "test_oracle_upper_bound"])
    ].copy()
    cols = [
        "selection",
        "test_profile",
        "valid_profile",
        "train_profiles",
        "feature_name",
        "prefix",
        "shrink",
        "margin",
        "gate",
        "valid_target_norm_worst",
        "test_0C_MAE_pct",
        "test_25C_MAE_pct",
        "test_45C_MAE_pct",
        "test_target_met",
        "test_target_norm_worst",
        "test_failure_detail",
    ]
    e1 = details[
        details["test_profile"].eq("FUDS")
        & details["valid_profile"].eq("VALIDATION")
        & details["train_profiles"].eq("DST,US06")
    ]
    lines = [
        "# NMC Validation-Selected Capacity Gate",
        "",
        "## Purpose",
        "",
        "This is a causal current-integration observer screen, not strict NoCC.",
        "",
        "The candidate capacity gate uses only prefix features from the target trajectory. The candidate set is selected by the held-out validation profile, then evaluated on the hidden test profile.",
        "",
        "This checks whether a paper-safe validation selector can replace the earlier test-picked `v_range` heuristic.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## E1 FUDS Diagnostic",
        "",
        e1[cols].to_markdown(index=False, floatfmt=".3f") if not e1.empty else "(missing)",
        "",
        "## Valid-Selected Feature Counts",
        "",
        feature_counts.head(30).to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Global Fixed-Candidate Upper Bound",
        "",
        "This is diagnostic only because it ranks candidates by test rotations. It shows whether the candidate family contains a stronger single fixed rule.",
        "",
        global_upper.head(30).to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Rotation Details",
        "",
        details[cols].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Interpretation",
        "",
        "- `valid_selected` is the paper-safe row: no test profile label is used for candidate selection.",
        "- `test_oracle_upper_bound` is diagnostic only; it shows whether the candidate family contains a good rule if the test profile were known.",
        "- If oracle is much better than valid-selected, the issue is adoption/coverage, not only model expressiveness.",
        "- This screen still uses explicit current integration and known initial SOC, so it belongs to the causal observer track, not the strict NoCC track.",
        "",
    ]
    (cfg.output_dir / f"{cfg.output_prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-selected causal prefix capacity-gate screen.")
    parser.add_argument("--raw-root", default=ValidationGateConfig.raw_root)
    parser.add_argument("--output-dir", default=ValidationGateConfig.output_dir)
    parser.add_argument("--output-prefix", default=ValidationGateConfig.output_prefix)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        ValidationGateConfig(
            raw_root=Path(args.raw_root),
            output_dir=Path(args.output_dir),
            output_prefix=str(args.output_prefix),
        )
    )


if __name__ == "__main__":
    main()
