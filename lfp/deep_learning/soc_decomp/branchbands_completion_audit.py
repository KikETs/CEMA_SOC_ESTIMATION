from __future__ import annotations

from pathlib import Path
import argparse
import json

import pandas as pd

from .branchbands_improvement_experiment import FOLDS, VARIANTS, run_paths, BranchBandsImprovementConfig


REQUIRED_ARTIFACTS = [
    "branchbands_leakage_audit.csv",
    "branchbands_input_schema.csv",
    "branchband_residual_mode_results.csv",
    "branchbands_improvement_results.csv",
    "branchbands_improvement_by_temperature.csv",
    "branchbands_improvement_focus.csv",
    "branchbands_improvement_decision_summary.csv",
    "branchbands_improvement_report.md",
    "branchbands_improvement_promotion.csv",
    "branchbands_observability_metrics.csv",
    "branchbands_observability_vs_error.csv",
    "branchbands_cold_expert_fusion_results.csv",
    "branchbands_cold_expert_fusion_by_temperature.csv",
    "branchbands_cold_expert_fusion_focus.csv",
    "branchbands_cold_expert_weights.csv",
]


def read_status(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "unreadable", "error": repr(exc)}


def add(rows: list[dict], item: str, status: str, detail: str, evidence: str = "") -> None:
    rows.append({
        "audit_item": item,
        "status": status,
        "detail": detail,
        "evidence": evidence,
    })


def table_or_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


def csv_has_rows(path: Path) -> tuple[bool, int]:
    if not path.exists() or path.stat().st_size <= 1:
        return False, 0
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return False, 0
    return len(df) > 0, int(len(df))


def update_remaining_runs(base_dir: Path, cfg: BranchBandsImprovementConfig) -> pd.DataFrame:
    rows = []
    order = 0
    for seed in cfg.seeds:
        for fold in cfg.folds:
            for variant in cfg.variants:
                order += 1
                paths = run_paths(cfg, str(variant), str(fold), int(seed))
                status = read_status(paths["status"])
                done = status.get("status") == "done" and paths["pred"].exists()
                rows.append({
                    "run_order": order,
                    "variant": variant,
                    "fold_name": fold,
                    "seed": int(seed),
                    "status": "done" if done else "remaining",
                    "will_skip_on_resume": bool(done),
                    "status_path": str(paths["status"].relative_to(base_dir)),
                    "prediction_path": str(paths["pred"].relative_to(base_dir)),
                })
    out = pd.DataFrame(rows)
    out.to_csv(base_dir / "branchbands_remaining_runs.csv", index=False)
    return out


def audit(base_dir: Path, *, seeds: tuple[int, ...] = (0,), output_prefix: str = "branchbands_improvement") -> tuple[pd.DataFrame, dict]:
    base_dir = Path(base_dir)
    cfg = BranchBandsImprovementConfig(
        base_dir=base_dir,
        output_prefix=output_prefix,
        folds=FOLDS,
        variants=tuple(VARIANTS.keys()),
        seeds=tuple(int(s) for s in seeds),
    )
    expected_runs = len(cfg.folds) * len(cfg.variants) * len(cfg.seeds)
    rows: list[dict] = []

    remaining = update_remaining_runs(base_dir, cfg)
    done_count = int(remaining["status"].eq("done").sum())
    add(
        rows,
        "run_status_coverage",
        "PASS" if done_count == expected_runs else "INCOMPLETE",
        f"{done_count}/{expected_runs} runs have status==done and prediction_rows.csv.gz",
        "branchbands_remaining_runs.csv",
    )

    for artifact in REQUIRED_ARTIFACTS:
        path = base_dir / artifact
        add(
            rows,
            f"artifact_exists:{artifact}",
            "PASS" if path.exists() and path.stat().st_size > 0 else "MISSING",
            f"size={path.stat().st_size if path.exists() else 0}",
            artifact,
        )

    audit_path = base_dir / "branchbands_leakage_audit.csv"
    if audit_path.exists():
        leak = pd.read_csv(audit_path)
        bad = leak[~leak["status"].astype(str).eq("PASS")]
        add(
            rows,
            "leakage_audit_all_pass",
            "PASS" if bad.empty else "FAIL",
            f"non-pass rows={len(bad)}",
            "branchbands_leakage_audit.csv",
        )
    else:
        add(rows, "leakage_audit_all_pass", "MISSING", "branchbands_leakage_audit.csv missing")

    schema_path = base_dir / "branchbands_input_schema.csv"
    if schema_path.exists():
        schema = pd.read_csv(schema_path)
        forbidden = schema[schema.get("forbidden_token_hit", False).astype(bool)]
        expected_variants = set(cfg.variants)
        actual_variants = set(schema["variant"].astype(str).unique()) if "variant" in schema else set()
        add(
            rows,
            "input_schema_forbidden_scan",
            "PASS" if forbidden.empty else "FAIL",
            f"forbidden rows={len(forbidden)}",
            "branchbands_input_schema.csv",
        )
        add(
            rows,
            "input_schema_variant_coverage",
            "PASS" if actual_variants == expected_variants else "INCOMPLETE",
            f"actual={len(actual_variants)} expected={len(expected_variants)}",
            ",".join(sorted(expected_variants - actual_variants)),
        )
    else:
        add(rows, "input_schema_forbidden_scan", "MISSING", "branchbands_input_schema.csv missing")
        add(rows, "input_schema_variant_coverage", "MISSING", "branchbands_input_schema.csv missing")

    strict_bad = []
    for rec in remaining[remaining["status"].eq("done")].itertuples(index=False):
        paths = run_paths(cfg, str(rec.variant), str(rec.fold_name), int(rec.seed))
        status = read_status(paths["status"])
        if status.get("strict_no_soc_input") is not True:
            strict_bad.append((rec.variant, rec.fold_name, rec.seed, "strict_no_soc_input"))
        if status.get("strict_no_cumulative_input") is not True:
            strict_bad.append((rec.variant, rec.fold_name, rec.seed, "strict_no_cumulative_input"))
        if status.get("explicit_current_integration_state_update") is not False:
            strict_bad.append((rec.variant, rec.fold_name, rec.seed, "explicit_current_integration_state_update"))
    add(
        rows,
        "done_run_strict_flags",
        "PASS" if not strict_bad and done_count > 0 else ("INCOMPLETE" if done_count == 0 else "FAIL"),
        f"bad strict flag rows={len(strict_bad)} among completed runs",
        repr(strict_bad[:10]),
    )

    results_path = base_dir / "branchbands_improvement_results.csv"
    results = pd.DataFrame()
    if results_path.exists():
        results = pd.read_csv(results_path)
        expected_pairs = {(v, f, int(s)) for v in cfg.variants for f in cfg.folds for s in cfg.seeds}
        actual_pairs = {
            (str(r.model_name), str(r.fold_name), int(r.seed))
            for r in results.itertuples(index=False)
            if hasattr(r, "model_name") and hasattr(r, "fold_name") and hasattr(r, "seed")
        }
        missing = sorted(expected_pairs - actual_pairs)
        add(
            rows,
            "results_target_fold_coverage",
            "PASS" if not missing else "INCOMPLETE",
            f"actual={len(actual_pairs)} expected={len(expected_pairs)} missing={len(missing)}",
            repr(missing[:10]),
        )
        metric_cols = [
            "MAE_pct",
            "RMSE_pct",
            "jitter_ratio",
            "catastrophic_error_rate_5pct",
            "plateau_20_80_MAE_pct",
            "low_current_MAE_pct",
        ]
        missing_metric_cols = [c for c in metric_cols if c not in results.columns]
        finite_ok = True
        if not missing_metric_cols and len(results):
            finite_ok = results[metric_cols].notna().all().all()
        add(
            rows,
            "results_required_metrics_present",
            "PASS" if not missing_metric_cols and finite_ok else "FAIL",
            f"missing_cols={missing_metric_cols} finite_ok={finite_ok}",
            "branchbands_improvement_results.csv",
        )
    else:
        add(rows, "results_target_fold_coverage", "MISSING", "branchbands_improvement_results.csv missing")
        add(rows, "results_required_metrics_present", "MISSING", "branchbands_improvement_results.csv missing")

    by_temp_path = base_dir / "branchbands_improvement_by_temperature.csv"
    if by_temp_path.exists():
        by_temp = pd.read_csv(by_temp_path)
        required = {"model_name", "fold_name", "seed", "temperature_C", "MAE_pct", "RMSE_pct"}
        missing_cols = sorted(required - set(by_temp.columns))
        if not missing_cols and len(by_temp):
            coverage = by_temp.groupby(["model_name", "fold_name", "seed"])["temperature_C"].nunique()
            min_temps = int(coverage.min()) if len(coverage) else 0
        else:
            min_temps = 0
        add(
            rows,
            "by_temperature_coverage",
            "PASS" if not missing_cols and min_temps >= 8 and len(by_temp) >= expected_runs * 8 else "INCOMPLETE",
            f"rows={len(by_temp)} min_temps_per_run={min_temps} missing_cols={missing_cols}",
            "branchbands_improvement_by_temperature.csv",
        )
    else:
        add(rows, "by_temperature_coverage", "MISSING", "branchbands_improvement_by_temperature.csv missing")

    residual_path = base_dir / "branchband_residual_mode_results.csv"
    if residual_path.exists():
        residual = pd.read_csv(residual_path)
        expected_residual_models = {
            "BandTCN_w150_base",
            "BandTCN_w150_residual_windowLocal",
            "BandTCN_w150_residual_hybrid",
        }
        pairs = set()
        if {"model_name", "fold_name", "seed"}.issubset(residual.columns):
            pairs = {
                (str(r.model_name), str(r.fold_name), int(r.seed))
                for r in residual.itertuples(index=False)
            }
        expected_residual = {(m, f, int(s)) for m in expected_residual_models for f in cfg.folds for s in cfg.seeds}
        add(
            rows,
            "residual_mode_results_coverage",
            "PASS" if expected_residual.issubset(pairs) else "INCOMPLETE",
            f"actual={len(pairs)} expected={len(expected_residual)} missing={len(expected_residual - pairs)}",
            repr(sorted(expected_residual - pairs)[:10]),
        )
    else:
        add(rows, "residual_mode_results_coverage", "MISSING", "branchband_residual_mode_results.csv missing")

    report_path = base_dir / "branchbands_improvement_report.md"
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        required_phrases = [
            "Current is used only as instantaneous excitation",
            "Do not claim NoCC proves current integration unnecessary",
            "temperature extrapolation",
        ]
        missing_phrases = [p for p in required_phrases if p not in text]
        add(
            rows,
            "report_safe_wording",
            "PASS" if not missing_phrases else "FAIL",
            f"missing_phrases={missing_phrases}",
            "branchbands_improvement_report.md",
        )
    else:
        add(rows, "report_safe_wording", "MISSING", "branchbands_improvement_report.md missing")

    if results.empty:
        add(
            rows,
            "optional_fusion_trigger_and_outputs",
            "INCOMPLETE",
            "Target-fold results are required before optional fusion trigger can be evaluated.",
        )
    else:
        base_name = "BandTCN_w150_base"
        required_folds = set(cfg.folds)
        base = results[results["model_name"].eq(base_name)]
        cold_models = sorted(m for m in results["model_name"].astype(str).unique() if "coldEMA" in m)
        full_enough = set(base["fold_name"].astype(str)) == required_folds and bool(cold_models)
        for model in cold_models:
            full_enough = full_enough and set(results[results["model_name"].eq(model)]["fold_name"].astype(str)) == required_folds
        if not full_enough:
            add(
                rows,
                "optional_fusion_trigger_and_outputs",
                "INCOMPLETE",
                "Base and coldEMA/coldHead models need all target folds before optional fusion trigger can be evaluated.",
                f"cold_models={cold_models}",
            )
        else:
            base_all = float(base["MAE_pct"].mean())
            base_outside = float(base[base["fold_name"].isin(["Omit N10", "Omit 50"])]["MAE_pct"].mean())
            triggered = []
            for model in cold_models:
                g = results[results["model_name"].eq(model)]
                candidate_all = float(g["MAE_pct"].mean())
                candidate_outside = float(g[g["fold_name"].isin(["Omit N10", "Omit 50"])]["MAE_pct"].mean())
                if candidate_outside < base_outside and candidate_all > base_all:
                    triggered.append({
                        "model_name": model,
                        "outside_gain_pctp": base_outside - candidate_outside,
                        "all_avg_degradation_pctp": candidate_all - base_all,
                    })
            fusion_rows_ok, fusion_rows = csv_has_rows(base_dir / "branchbands_cold_expert_fusion_results.csv")
            weights_ok, weight_rows = csv_has_rows(base_dir / "branchbands_cold_expert_weights.csv")
            if triggered:
                add(
                    rows,
                    "optional_fusion_trigger_and_outputs",
                    "PASS" if fusion_rows_ok and weights_ok else "FAIL",
                    f"triggered={triggered}; fusion_rows={fusion_rows}; weight_rows={weight_rows}",
                    "branchbands_cold_expert_fusion_results.csv,branchbands_cold_expert_weights.csv",
                )
            else:
                add(
                    rows,
                    "optional_fusion_trigger_and_outputs",
                    "PASS",
                    "Optional fusion was not triggered by the rule; empty fusion files are acceptable.",
                    f"base_all={base_all:.6f}; base_outside={base_outside:.6f}; cold_models={cold_models}",
                )

    audit_df = pd.DataFrame(rows)
    blocking = audit_df["status"].isin(["FAIL", "MISSING", "INCOMPLETE"]).any()
    summary = {
        "status": "complete" if not blocking else "incomplete",
        "expected_runs": int(expected_runs),
        "completed_runs": int(done_count),
        "remaining_runs": int(expected_runs - done_count),
        "audit_rows": int(len(audit_df)),
        "non_pass_rows": int(audit_df[~audit_df["status"].eq("PASS")].shape[0]),
    }
    audit_df.to_csv(base_dir / "branchbands_completion_audit.csv", index=False)
    (base_dir / "branchbands_completion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# BranchBands Completion Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Completed runs: {summary['completed_runs']}/{summary['expected_runs']}",
        f"- Remaining runs: {summary['remaining_runs']}",
        "",
        "## Audit Rows",
        table_or_text(audit_df),
        "",
    ]
    if summary["remaining_runs"]:
        lines += [
            "## Next Remaining Runs",
            table_or_text(remaining[remaining["status"].eq("remaining")].head(10)),
            "",
        ]
    (base_dir / "branchbands_completion_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return audit_df, summary


def parse_args():
    p = argparse.ArgumentParser(description="Audit BranchBands strict NoCC screening completion.")
    p.add_argument("--base-dir", type=Path, default=Path("."))
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--output-prefix", default="branchbands_improvement")
    p.add_argument("--fail-on-incomplete", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    audit_df, summary = audit(args.base_dir, seeds=tuple(args.seeds), output_prefix=args.output_prefix)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    non_pass = audit_df[~audit_df["status"].eq("PASS")]
    if len(non_pass):
        print(non_pass.to_string(index=False))
    if args.fail_on_incomplete and summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
