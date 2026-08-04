from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CombinedAbstentionConfig:
    coverage_rows: Path = Path("paper_results/nmc_capacity_coverage_audit_rows.csv")
    residual_rows: Path = Path("paper_results/nmc_voltage_residual_ood_audit_rows.csv")
    output_dir: Path = Path("paper_results")
    output_prefix: str = "nmc_combined_abstention_audit"


KEYS = ["test_profile", "valid_profile", "train_profiles"]
LABEL_COLS = [
    "target_met",
    "target_norm_worst",
    "0C_MAE_pct",
    "25C_MAE_pct",
    "45C_MAE_pct",
    "failure_detail",
]
SCORE_COLS = [
    "max_feature_outside_norm",
    "mean_feature_outside_norm",
    "prefix256_max_resid_p95_mV",
    "prefix512_max_resid_p95_mV",
    "prefix1024_max_resid_p95_mV",
    "prefix256_max_mean_resid_ratio",
    "prefix512_max_mean_resid_ratio",
    "prefix1024_max_mean_resid_ratio",
    "prefix256_max_p95_resid_ratio",
    "prefix512_max_p95_resid_ratio",
    "prefix1024_max_p95_resid_ratio",
]


def _load_merged(cfg: CombinedAbstentionConfig) -> pd.DataFrame:
    coverage = pd.read_csv(cfg.coverage_rows)
    residual = pd.read_csv(cfg.residual_rows)
    cov_cols = [*KEYS, *LABEL_COLS, "max_feature_outside_norm", "mean_feature_outside_norm", "inside_all_temps"]
    residual_score_cols = [c for c in SCORE_COLS if c in residual.columns and c not in coverage.columns]
    out = coverage[cov_cols].merge(residual[[*KEYS, *residual_score_cols]], on=KEYS, how="inner")
    if len(out) != 12:
        raise RuntimeError(f"Expected 12 profile rotations, got {len(out)}.")
    return out


def _threshold_values(values: pd.Series) -> list[float]:
    unique = sorted(set(float(v) for v in values.astype(float).round(6)))
    if not unique:
        return [float("inf")]
    mids = []
    for a, b in zip(unique[:-1], unique[1:]):
        mids.append((a + b) / 2.0)
    return [unique[0], *mids, unique[-1], float("inf")]


def _accepted(frame: pd.DataFrame, rule: dict[str, object]) -> pd.Series:
    a = frame[str(rule["metric_a"])].astype(float) <= float(rule["threshold_a"])
    metric_b = str(rule["metric_b"])
    if metric_b == "":
        return a
    b = frame[metric_b].astype(float) <= float(rule["threshold_b"])
    if rule["operator"] == "and":
        return a & b
    if rule["operator"] == "or":
        return a | b
    raise ValueError(f"Unknown operator: {rule['operator']}")


def _evaluate_rule(frame: pd.DataFrame, rule: dict[str, object]) -> dict[str, object]:
    accepted = _accepted(frame, rule)
    accepted_rows = frame[accepted]
    rejected_rows = frame[~accepted]
    accepted_count = int(len(accepted_rows))
    accepted_pass = int(accepted_rows["target_met"].astype(bool).sum()) if accepted_count else 0
    accepted_fail = int(accepted_count - accepted_pass)
    total_fail = int((~frame["target_met"].astype(bool)).sum())
    rejected_fail = int((~rejected_rows["target_met"].astype(bool)).sum()) if len(rejected_rows) else 0
    return {
        **rule,
        "accepted_count": accepted_count,
        "accepted_pass": accepted_pass,
        "accepted_fail": accepted_fail,
        "accepted_precision": float(accepted_pass / accepted_count) if accepted_count else np.nan,
        "rejected_count": int(len(rejected_rows)),
        "rejected_fail": rejected_fail,
        "failure_recall_by_rejection": float(rejected_fail / total_fail) if total_fail else np.nan,
        "accepted_mean_target_norm_worst": float(accepted_rows["target_norm_worst"].mean()) if accepted_count else np.nan,
        "accepted_max_target_norm_worst": float(accepted_rows["target_norm_worst"].max()) if accepted_count else np.nan,
    }


def _candidate_rules(frame: pd.DataFrame) -> list[dict[str, object]]:
    metrics = [col for col in SCORE_COLS if col in frame.columns]
    threshold_map = {metric: _threshold_values(frame[metric]) for metric in metrics}
    rules: list[dict[str, object]] = []
    for metric in metrics:
        for threshold in threshold_map[metric]:
            rules.append(
                {
                    "operator": "single",
                    "metric_a": metric,
                    "threshold_a": threshold,
                    "metric_b": "",
                    "threshold_b": np.nan,
                }
            )
    coverage_metrics = ["max_feature_outside_norm", "mean_feature_outside_norm"]
    residual_metrics = [m for m in metrics if m not in coverage_metrics]
    for cov_metric in coverage_metrics:
        for residual_metric in residual_metrics:
            for cov_threshold in threshold_map[cov_metric]:
                for residual_threshold in threshold_map[residual_metric]:
                    for op in ["and", "or"]:
                        rules.append(
                            {
                                "operator": op,
                                "metric_a": cov_metric,
                                "threshold_a": cov_threshold,
                                "metric_b": residual_metric,
                                "threshold_b": residual_threshold,
                            }
                        )
    return rules


def _global_sweep(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [_evaluate_rule(frame, rule) for rule in _candidate_rules(frame)]
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["accepted_precision", "accepted_count", "failure_recall_by_rejection", "accepted_max_target_norm_worst"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def _select_rule(train: pd.DataFrame, min_accepted: int = 1) -> dict[str, object]:
    sweep = _global_sweep(train)
    no_fail = sweep[(sweep["accepted_fail"].eq(0)) & (sweep["accepted_count"].ge(int(min_accepted)))].copy()
    if no_fail.empty:
        return sweep.iloc[0].to_dict()
    no_fail = no_fail.sort_values(
        ["accepted_count", "failure_recall_by_rejection", "accepted_mean_target_norm_worst"],
        ascending=[False, False, True],
    )
    return no_fail.iloc[0].to_dict()


def _leave_one_rotation(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, heldout in frame.iterrows():
        train = frame.drop(index=idx).reset_index(drop=True)
        selected = _select_rule(train)
        accepted = bool(_accepted(frame.iloc[[idx]], selected).iloc[0])
        row = {
            "heldout_test_profile": heldout["test_profile"],
            "heldout_valid_profile": heldout["valid_profile"],
            "heldout_train_profiles": heldout["train_profiles"],
            "heldout_target_met": bool(heldout["target_met"]),
            "heldout_target_norm_worst": float(heldout["target_norm_worst"]),
            "heldout_0C_MAE_pct": float(heldout["0C_MAE_pct"]),
            "heldout_25C_MAE_pct": float(heldout["25C_MAE_pct"]),
            "heldout_45C_MAE_pct": float(heldout["45C_MAE_pct"]),
            "heldout_failure_detail": str(heldout["failure_detail"]),
            "heldout_accepted": accepted,
            "heldout_accepted_fail": bool(accepted and not bool(heldout["target_met"])),
            "heldout_accepted_pass": bool(accepted and bool(heldout["target_met"])),
        }
        for key, value in selected.items():
            row[f"selected_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _summary(frame: pd.DataFrame, global_sweep: pd.DataFrame, loo: pd.DataFrame) -> pd.DataFrame:
    best = global_sweep.iloc[0]
    loo_accepted = loo[loo["heldout_accepted"].astype(bool)]
    rows = [
        {
            "mode": "global_oracle_best_rule",
            "accepted_count": int(best["accepted_count"]),
            "accepted_pass": int(best["accepted_pass"]),
            "accepted_fail": int(best["accepted_fail"]),
            "accepted_precision": float(best["accepted_precision"]),
            "failure_recall_by_rejection": float(best["failure_recall_by_rejection"]),
            "accepted_max_target_norm_worst": float(best["accepted_max_target_norm_worst"]),
            "note": "Diagnostic upper bound; thresholds are chosen using all rotation labels.",
        },
        {
            "mode": "leave_one_rotation_calibrated",
            "accepted_count": int(len(loo_accepted)),
            "accepted_pass": int(loo["heldout_accepted_pass"].astype(bool).sum()),
            "accepted_fail": int(loo["heldout_accepted_fail"].astype(bool).sum()),
            "accepted_precision": float(loo["heldout_accepted_pass"].astype(bool).sum() / len(loo_accepted))
            if len(loo_accepted)
            else np.nan,
            "failure_recall_by_rejection": float(
                ((~loo["heldout_target_met"].astype(bool)) & (~loo["heldout_accepted"].astype(bool))).sum()
                / max((~loo["heldout_target_met"].astype(bool)).sum(), 1)
            ),
            "accepted_max_target_norm_worst": float(loo_accepted["heldout_target_norm_worst"].max())
            if len(loo_accepted)
            else np.nan,
            "note": "Each heldout rotation is evaluated by a zero-fail rule selected on the other rotations.",
        },
        {
            "mode": "no_abstention_fixed_capacity_anchor",
            "accepted_count": int(len(frame)),
            "accepted_pass": int(frame["target_met"].astype(bool).sum()),
            "accepted_fail": int((~frame["target_met"].astype(bool)).sum()),
            "accepted_precision": float(frame["target_met"].astype(bool).mean()),
            "failure_recall_by_rejection": 0.0,
            "accepted_max_target_norm_worst": float(frame["target_norm_worst"].max()),
            "note": "Baseline fixed bounded v_range capacity anchor without abstention.",
        },
    ]
    return pd.DataFrame(rows)


def run(cfg: CombinedAbstentionConfig) -> dict[str, pd.DataFrame]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_merged(cfg)
    global_sweep = _global_sweep(frame)
    loo = _leave_one_rotation(frame)
    summary = _summary(frame, global_sweep, loo)
    frame.to_csv(cfg.output_dir / f"{cfg.output_prefix}_merged_rows.csv", index=False)
    global_sweep.to_csv(cfg.output_dir / f"{cfg.output_prefix}_global_sweep.csv", index=False)
    loo.to_csv(cfg.output_dir / f"{cfg.output_prefix}_leave_one_rotation.csv", index=False)
    summary.to_csv(cfg.output_dir / f"{cfg.output_prefix}_summary.csv", index=False)
    write_report(cfg, frame, global_sweep, loo, summary)
    print(summary.to_string(index=False))
    print("\nTop global rules")
    print(global_sweep.head(20).to_string(index=False))
    print("\nLeave-one-rotation")
    print(loo.to_string(index=False))
    return {
        "merged_rows": frame,
        "global_sweep": global_sweep,
        "leave_one_rotation": loo,
        "summary": summary,
    }


def write_report(
    cfg: CombinedAbstentionConfig,
    frame: pd.DataFrame,
    global_sweep: pd.DataFrame,
    loo: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    detail_cols = [
        *KEYS,
        "0C_MAE_pct",
        "25C_MAE_pct",
        "45C_MAE_pct",
        "target_met",
        "target_norm_worst",
        "max_feature_outside_norm",
        "prefix512_max_resid_p95_mV",
        "prefix1024_max_resid_p95_mV",
        "failure_detail",
    ]
    loo_cols = [
        "heldout_test_profile",
        "heldout_valid_profile",
        "heldout_train_profiles",
        "heldout_target_met",
        "heldout_target_norm_worst",
        "heldout_accepted",
        "heldout_accepted_pass",
        "heldout_accepted_fail",
        "selected_operator",
        "selected_metric_a",
        "selected_threshold_a",
        "selected_metric_b",
        "selected_threshold_b",
        "selected_accepted_count",
        "selected_accepted_fail",
        "heldout_failure_detail",
    ]
    lines = [
        "# NMC Combined Abstention Audit",
        "",
        "## Purpose",
        "",
        "This combines label-free prefix coverage and voltage-residual OOD scores to test whether a safer adoption/abstention rule can be built for the fixed bounded capacity anchor.",
        "",
        "The global sweep is diagnostic only because it uses all rotation labels to choose thresholds. The leave-one-rotation result is the more realistic calibration check.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Top Global Rules",
        "",
        global_sweep.head(30).to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Leave-One-Rotation Calibration",
        "",
        loo[loo_cols].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Rotation Scores",
        "",
        frame[detail_cols].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Interpretation",
        "",
        "- A combined rule can produce high-precision accepted subsets in the global diagnostic sweep.",
        "- The key question is whether those thresholds remain reliable when calibrated without the held-out rotation.",
        "- If leave-one-rotation accepts few or no rotations, the rule is too conservative for a main SOC estimator.",
        "- If it accepts failures, the adoption rule is not paper-safe enough.",
        "",
    ]
    (cfg.output_dir / f"{cfg.output_prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combined coverage + voltage residual abstention audit.")
    parser.add_argument("--coverage-rows", default=CombinedAbstentionConfig.coverage_rows)
    parser.add_argument("--residual-rows", default=CombinedAbstentionConfig.residual_rows)
    parser.add_argument("--output-dir", default=CombinedAbstentionConfig.output_dir)
    parser.add_argument("--output-prefix", default=CombinedAbstentionConfig.output_prefix)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        CombinedAbstentionConfig(
            coverage_rows=Path(args.coverage_rows),
            residual_rows=Path(args.residual_rows),
            output_dir=Path(args.output_dir),
            output_prefix=str(args.output_prefix),
        )
    )


if __name__ == "__main__":
    main()
