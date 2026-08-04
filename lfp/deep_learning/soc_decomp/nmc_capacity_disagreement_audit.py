from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DisagreementAuditConfig:
    candidate_rows: Path = Path("paper_results/nmc_validation_selected_capacity_gate_candidate_rows.csv")
    selected_rows: Path = Path("paper_results/nmc_validation_selected_capacity_gate_selected_rows.csv")
    output_dir: Path = Path("paper_results")
    output_prefix: str = "nmc_capacity_disagreement_audit"


KEYS = ["test_profile", "valid_profile", "train_profiles"]
TEMPS = ["0C", "25C", "45C"]
METRICS = [
    "max_qpred_std_pct",
    "max_qdelta_iqr_pct",
    "max_qdelta_range_pct",
    "max_qdelta_abs_median_pct",
    "mean_conf_mean",
    "max_conf_std",
]


def _build_rows(candidates: pd.DataFrame, fixed: pd.DataFrame) -> pd.DataFrame:
    labels = fixed[fixed["selection"].eq("fixed_bounded_vrange")][
        [
            *KEYS,
            "test_target_met",
            "test_target_norm_worst",
            "test_0C_MAE_pct",
            "test_25C_MAE_pct",
            "test_45C_MAE_pct",
            "test_failure_detail",
        ]
    ].copy()
    rows = []
    for key, group in candidates.groupby(KEYS):
        row = dict(zip(KEYS, key))
        for temp in TEMPS:
            q = group[f"test_{temp}_Qpred_Ah"].astype(float).to_numpy()
            base = group[f"test_{temp}_Qbase_Ah"].astype(float).to_numpy()
            conf = group[f"test_{temp}_confidence"].astype(float).to_numpy()
            q_norm = q / np.maximum(base, 1e-9)
            q_delta = q_norm - 1.0
            row[f"{temp}_qpred_std_pct"] = float(np.std(q_norm) * 100.0)
            row[f"{temp}_qdelta_iqr_pct"] = float((np.percentile(q_delta, 75) - np.percentile(q_delta, 25)) * 100.0)
            row[f"{temp}_qdelta_range_pct"] = float((np.max(q_delta) - np.min(q_delta)) * 100.0)
            row[f"{temp}_qdelta_abs_median_pct"] = float(np.median(np.abs(q_delta)) * 100.0)
            row[f"{temp}_conf_mean"] = float(np.mean(conf))
            row[f"{temp}_conf_std"] = float(np.std(conf))
        for stat in [
            "qpred_std_pct",
            "qdelta_iqr_pct",
            "qdelta_range_pct",
            "qdelta_abs_median_pct",
            "conf_mean",
            "conf_std",
        ]:
            cols = [f"{temp}_{stat}" for temp in TEMPS]
            row[f"max_{stat}"] = float(np.max([row[col] for col in cols]))
            row[f"mean_{stat}"] = float(np.mean([row[col] for col in cols]))
        rows.append(row)
    out = pd.DataFrame(rows).merge(labels, on=KEYS, how="left")
    return out.sort_values("test_target_norm_worst").reset_index(drop=True)


def _correlations(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for metric in METRICS:
        out.append(
            {
                "metric": metric,
                "pearson_corr_with_target_norm_worst": float(rows[metric].corr(rows["test_target_norm_worst"])),
                "mean_pass": float(rows[rows["test_target_met"].astype(bool)][metric].mean()),
                "mean_fail": float(rows[~rows["test_target_met"].astype(bool)][metric].mean()),
            }
        )
    return pd.DataFrame(out)


def _thresholds(rows: pd.DataFrame) -> pd.DataFrame:
    total_fail = int((~rows["test_target_met"].astype(bool)).sum())
    out = []
    for metric in METRICS:
        values = sorted(set(float(x) for x in rows[metric].round(6)))
        values.append(float("inf"))
        for direction in ["accept_low", "accept_high"]:
            for threshold in values:
                if direction == "accept_low":
                    accepted = rows[metric].astype(float) <= float(threshold)
                    threshold_text = "inf" if np.isinf(threshold) else float(threshold)
                else:
                    if np.isinf(threshold):
                        continue
                    accepted = rows[metric].astype(float) >= float(threshold)
                    threshold_text = float(threshold)
                accepted_rows = rows[accepted]
                rejected_rows = rows[~accepted]
                if accepted_rows.empty:
                    continue
                accepted_pass = int(accepted_rows["test_target_met"].astype(bool).sum())
                accepted_fail = int(len(accepted_rows) - accepted_pass)
                rejected_fail = int((~rejected_rows["test_target_met"].astype(bool)).sum()) if len(rejected_rows) else 0
                out.append(
                    {
                        "metric": metric,
                        "policy": direction,
                        "threshold": threshold_text,
                        "accepted_count": int(len(accepted_rows)),
                        "accepted_pass": accepted_pass,
                        "accepted_fail": accepted_fail,
                        "accepted_precision": float(accepted_pass / len(accepted_rows)),
                        "rejected_count": int(len(rejected_rows)),
                        "rejected_fail": rejected_fail,
                        "failure_recall_by_rejection": float(rejected_fail / total_fail) if total_fail else np.nan,
                        "accepted_mean_target_norm_worst": float(accepted_rows["test_target_norm_worst"].mean()),
                        "accepted_max_target_norm_worst": float(accepted_rows["test_target_norm_worst"].max()),
                    }
                )
    return pd.DataFrame(out)


def run(cfg: DisagreementAuditConfig) -> dict[str, pd.DataFrame]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(cfg.candidate_rows)
    fixed = pd.read_csv(cfg.selected_rows)
    rows = _build_rows(candidates, fixed)
    corrs = _correlations(rows)
    thresholds = _thresholds(rows)
    rows.to_csv(cfg.output_dir / f"{cfg.output_prefix}_rows.csv", index=False)
    corrs.to_csv(cfg.output_dir / f"{cfg.output_prefix}_correlations.csv", index=False)
    thresholds.to_csv(cfg.output_dir / f"{cfg.output_prefix}_thresholds.csv", index=False)
    write_report(cfg, rows, corrs, thresholds)
    print(corrs.to_string(index=False))
    print("\nBest threshold rows")
    print(
        thresholds.sort_values(
            ["accepted_precision", "accepted_count", "failure_recall_by_rejection"],
            ascending=[False, False, False],
        )
        .head(20)
        .to_string(index=False)
    )
    return {"rows": rows, "correlations": corrs, "thresholds": thresholds}


def write_report(
    cfg: DisagreementAuditConfig,
    rows: pd.DataFrame,
    corrs: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> None:
    detail_cols = [
        *KEYS,
        "test_target_met",
        "test_target_norm_worst",
        "test_0C_MAE_pct",
        "test_25C_MAE_pct",
        "test_45C_MAE_pct",
        "max_qpred_std_pct",
        "max_qdelta_iqr_pct",
        "max_qdelta_range_pct",
        "max_qdelta_abs_median_pct",
        "mean_conf_mean",
        "max_conf_std",
        "test_failure_detail",
    ]
    best_thresholds = thresholds.sort_values(
        ["accepted_precision", "accepted_count", "failure_recall_by_rejection"],
        ascending=[False, False, False],
    ).head(30)
    low_disagreement_failures = rows[
        (~rows["test_target_met"].astype(bool))
        & (rows["max_qpred_std_pct"] <= rows["max_qpred_std_pct"].median())
    ].copy()
    high_disagreement_passes = rows[
        rows["test_target_met"].astype(bool)
        & (rows["max_qpred_std_pct"] >= rows["max_qpred_std_pct"].median())
    ].copy()
    lines = [
        "# NMC Capacity Candidate Disagreement Audit",
        "",
        "## Purpose",
        "",
        "This checks whether disagreement across label-free capacity-gate candidates can serve as an OOD or abstention signal for the best fixed bounded `v_range` rule.",
        "",
        "If the idea worked, failing rotations would have large candidate disagreement and passing rotations would have small disagreement.",
        "",
        "## Correlations",
        "",
        corrs.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Best Threshold Rows",
        "",
        best_thresholds.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Low-Disagreement Failures",
        "",
        low_disagreement_failures[detail_cols].to_markdown(index=False, floatfmt=".3f")
        if not low_disagreement_failures.empty
        else "(none)",
        "",
        "## High-Disagreement Passes",
        "",
        high_disagreement_passes[detail_cols].to_markdown(index=False, floatfmt=".3f")
        if not high_disagreement_passes.empty
        else "(none)",
        "",
        "## Rotation Details",
        "",
        rows[detail_cols].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Interpretation",
        "",
        "- Simple candidate disagreement is not a reliable failure detector here.",
        "- Several failing rotations have low candidate disagreement, meaning the candidate family can agree and still be wrong.",
        "- Several passing rotations have high disagreement, so high spread alone would reject useful cases.",
        "- The next observer needs calibrated uncertainty tied to profile coverage and physical residuals, not raw candidate spread alone.",
        "",
    ]
    (cfg.output_dir / f"{cfg.output_prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit candidate disagreement as OOD signal for capacity anchors.")
    parser.add_argument("--candidate-rows", default=DisagreementAuditConfig.candidate_rows)
    parser.add_argument("--selected-rows", default=DisagreementAuditConfig.selected_rows)
    parser.add_argument("--output-dir", default=DisagreementAuditConfig.output_dir)
    parser.add_argument("--output-prefix", default=DisagreementAuditConfig.output_prefix)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        DisagreementAuditConfig(
            candidate_rows=Path(args.candidate_rows),
            selected_rows=Path(args.selected_rows),
            output_dir=Path(args.output_dir),
            output_prefix=str(args.output_prefix),
        )
    )


if __name__ == "__main__":
    main()
