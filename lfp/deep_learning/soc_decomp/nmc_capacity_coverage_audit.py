from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .nmc_validation_selected_capacity_gate import TARGETS, _eval_candidate, _feature, _load


@dataclass
class CoverageAuditConfig:
    raw_root: Path = Path("nmc_samsung_inr_18650_2ah_raw/NMC SAMSUNG INR 18650 2Ah")
    output_dir: Path = Path("paper_results")
    output_prefix: str = "nmc_capacity_coverage_audit"
    feature_name: str = "v_range"
    prefix: int = 256
    shrink: float = 0.75
    margin: float = 0.5
    gate: str = "none"


def _rotation_rows(cfg: CoverageAuditConfig) -> pd.DataFrame:
    data = _load(cfg.raw_root)
    profiles = sorted({profile for profile, _temp in data})
    temps = sorted({temp for _profile, temp in data})
    rows = []
    candidate = {
        "feature_name": cfg.feature_name,
        "prefix": int(cfg.prefix),
        "shrink": float(cfg.shrink),
        "margin": float(cfg.margin),
        "gate": str(cfg.gate),
    }
    for test_profile in profiles:
        remaining = [p for p in profiles if p != test_profile]
        for valid_profile in remaining:
            train_profiles = [p for p in remaining if p != valid_profile]
            eval_row = _eval_candidate(data, train_profiles=train_profiles, profile=test_profile, **candidate)
            row = {
                "test_profile": test_profile,
                "valid_profile": valid_profile,
                "train_profiles": ",".join(train_profiles),
                **candidate,
                **eval_row,
            }
            outside_values = []
            inside_values = []
            for temp in temps:
                xs = np.asarray(
                    [_feature(data[(profile, temp)], int(cfg.prefix), str(cfg.feature_name)) for profile in train_profiles],
                    dtype=np.float64,
                )
                x = _feature(data[(test_profile, temp)], int(cfg.prefix), str(cfg.feature_name))
                lo = float(np.min(xs))
                hi = float(np.max(xs))
                span = max(float(hi - lo), 1e-9)
                outside = max(lo - x, x - hi, 0.0)
                outside_norm = float(outside / span)
                inside = bool(outside <= 0.0)
                outside_values.append(outside_norm)
                inside_values.append(inside)
                row[f"{temp:g}C_train_feature_min"] = lo
                row[f"{temp:g}C_train_feature_max"] = hi
                row[f"{temp:g}C_test_feature"] = x
                row[f"{temp:g}C_feature_outside_norm"] = outside_norm
                row[f"{temp:g}C_feature_inside_train_span"] = inside
            row["max_feature_outside_norm"] = float(np.max(outside_values))
            row["mean_feature_outside_norm"] = float(np.mean(outside_values))
            row["inside_all_temps"] = bool(all(inside_values))
            row["inside_any_temp"] = bool(any(inside_values))
            rows.append(row)
    return pd.DataFrame(rows)


def _threshold_table(rows: pd.DataFrame) -> pd.DataFrame:
    thresholds = [0.0, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00, np.inf]
    out = []
    total_fail = int((~rows["target_met"].astype(bool)).sum())
    for threshold in thresholds:
        accepted = rows["max_feature_outside_norm"].astype(float) <= float(threshold)
        accepted_rows = rows[accepted]
        rejected_rows = rows[~accepted]
        accepted_count = int(len(accepted_rows))
        accepted_pass = int(accepted_rows["target_met"].astype(bool).sum()) if accepted_count else 0
        accepted_fail = int(accepted_count - accepted_pass)
        rejected_fail = int((~rejected_rows["target_met"].astype(bool)).sum()) if len(rejected_rows) else 0
        out.append(
            {
                "max_outside_norm_threshold": "inf" if not np.isfinite(threshold) else float(threshold),
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
        )
    return pd.DataFrame(out)


def run(cfg: CoverageAuditConfig) -> dict[str, pd.DataFrame]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _rotation_rows(cfg)
    thresholds = _threshold_table(rows)
    rows.to_csv(cfg.output_dir / f"{cfg.output_prefix}_rows.csv", index=False)
    thresholds.to_csv(cfg.output_dir / f"{cfg.output_prefix}_thresholds.csv", index=False)
    write_report(cfg, rows, thresholds)
    print(thresholds.to_string(index=False))
    return {"rows": rows, "thresholds": thresholds}


def write_report(cfg: CoverageAuditConfig, rows: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    detail_cols = [
        "test_profile",
        "valid_profile",
        "train_profiles",
        "0C_MAE_pct",
        "25C_MAE_pct",
        "45C_MAE_pct",
        "target_met",
        "target_norm_worst",
        "max_feature_outside_norm",
        "mean_feature_outside_norm",
        "inside_all_temps",
        "failure_detail",
    ]
    lines = [
        "# NMC Capacity Coverage Audit",
        "",
        "## Purpose",
        "",
        "This checks whether a simple label-free coverage/OOD gate can identify failure rotations for the best fixed capacity anchor.",
        "",
        "The gate uses only prefix feature coverage: whether the test profile's prefix feature lies inside the two train-profile feature values.",
        "",
        "## Fixed Capacity Anchor",
        "",
        f"- feature: `{cfg.feature_name}`",
        f"- prefix: `{cfg.prefix}`",
        f"- shrink: `{cfg.shrink}`",
        f"- margin: `{cfg.margin}`",
        f"- gate during prediction: `{cfg.gate}`",
        "",
        "## Coverage Threshold Table",
        "",
        thresholds.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Rotation Details",
        "",
        rows[detail_cols].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Interpretation",
        "",
        "- A useful coverage gate should accept many passing rotations while rejecting most failing rotations.",
        "- If accepted precision remains poor or accepted count is tiny, prefix feature coverage alone is not enough for a deployable adoption rule.",
        "- This audit is label-free at inference, but the threshold analysis is diagnostic and should be validated by profile rotation before any claim.",
        "",
    ]
    (cfg.output_dir / f"{cfg.output_prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coverage/OOD audit for fixed causal capacity anchor.")
    parser.add_argument("--raw-root", default=CoverageAuditConfig.raw_root)
    parser.add_argument("--output-dir", default=CoverageAuditConfig.output_dir)
    parser.add_argument("--output-prefix", default=CoverageAuditConfig.output_prefix)
    parser.add_argument("--feature-name", default=CoverageAuditConfig.feature_name)
    parser.add_argument("--prefix", type=int, default=CoverageAuditConfig.prefix)
    parser.add_argument("--shrink", type=float, default=CoverageAuditConfig.shrink)
    parser.add_argument("--margin", type=float, default=CoverageAuditConfig.margin)
    parser.add_argument("--gate", default=CoverageAuditConfig.gate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        CoverageAuditConfig(
            raw_root=Path(args.raw_root),
            output_dir=Path(args.output_dir),
            output_prefix=str(args.output_prefix),
            feature_name=str(args.feature_name),
            prefix=int(args.prefix),
            shrink=float(args.shrink),
            margin=float(args.margin),
            gate=str(args.gate),
        )
    )


if __name__ == "__main__":
    main()
