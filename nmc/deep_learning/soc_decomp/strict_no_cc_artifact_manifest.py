from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


DEFAULT_FOLDS = ("Exp A", "Exp B", "Exp C", "Exp D", "Omit N10", "Omit 50")
DEFAULT_VARIANTS = (
    "BandTCN_w150",
    "BandTCN_w300",
    "BandTCN_multiscale_50_150_300",
    "BandTCN_multiscale_50_150_500",
    "BandTCN_REX",
    "BandTCN_GroupDRO",
    "BandTCN_REX_GroupDRO",
    "BandTCN_highT",
)
DEFAULT_SEEDS = (0, 1, 2)


ARTIFACTS = (
    ("branch_bands_audit_json", "branch_bands_strict_no_cc_audit.json"),
    ("branch_bands_audit_csv", "branch_bands_strict_no_cc_audit.csv"),
    ("branch_bands_audit_report", "branch_bands_strict_no_cc_audit.md"),
    ("bandtcn_results", "no_cc_bandtcn_results.csv"),
    ("bandtcn_by_temperature", "no_cc_bandtcn_by_temperature.csv"),
    ("bandtcn_focus", "no_cc_bandtcn_focus.csv"),
    ("bandtcn_predictions", "no_cc_bandtcn_prediction_rows.csv.gz"),
    ("robust_domain_summary", "no_cc_bandtcn_domain_robust_summary.csv"),
    ("robust_rex_results", "no_cc_bandtcn_rex_results.csv"),
    ("robust_groupdro_results", "no_cc_bandtcn_groupdro_results.csv"),
    ("thermoguard_results", "no_cc_thermoguard_results.csv"),
    ("thermoguard_by_temperature", "no_cc_thermoguard_by_temperature.csv"),
    ("thermoguard_focus", "no_cc_thermoguard_focus.csv"),
    ("thermoguard_weights", "no_cc_thermoguard_expert_weights.csv"),
    ("observability_metrics", "no_cc_observability_metrics.csv"),
    ("observability_vs_error", "no_cc_observability_vs_error.csv"),
    ("observability_uncertainty_report", "no_cc_uncertainty_report.md"),
    ("strict_observability_report", "strict_no_cc_observability_report.md"),
    ("bandtcn_report", "no_cc_bandtcn_report.md"),
    ("thermoguard_report", "no_cc_thermoguard_report.md"),
    ("final_comparison_csv", "no_cc_final_comparison.csv"),
    ("legacy_no_cc_final_comparison_report", "no_cc_final_comparison_report.md"),
    ("final_comparison_report", "strict_no_cc_final_comparison_report.md"),
)


def fold_key(fold: str) -> str:
    return fold.replace(" ", "_").replace("-", "N")


def run_key(prefix: str, variant: str, fold: str, seed: int) -> str:
    return f"{prefix}_{variant}_{fold_key(fold)}_seed{seed}"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv_len(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return None


def config_from_metadata(base_dir: Path) -> dict:
    meta = read_json(base_dir / "no_cc_bandtcn_metadata.json")
    return {
        "output_prefix": meta.get("output_prefix", "no_cc_bandtcn"),
        "folds": tuple(meta.get("folds", DEFAULT_FOLDS)),
        "variants": tuple(meta.get("variants", DEFAULT_VARIANTS)),
        "seeds": tuple(int(s) for s in meta.get("seeds", DEFAULT_SEEDS)),
        "metadata": meta,
    }


def completion_table(base_dir: Path, cfg: dict) -> pd.DataFrame:
    rows = []
    run_root = base_dir / "no_cc_bandtcn_runs"
    for seed in cfg["seeds"]:
        for fold in cfg["folds"]:
            for variant in cfg["variants"]:
                key = run_key(cfg["output_prefix"], variant, fold, int(seed))
                status_path = run_root / key / "status.json"
                pred_path = run_root / key / "prediction_rows.csv.gz"
                status = read_json(status_path)
                rows.append({
                    "variant": variant,
                    "fold_name": fold,
                    "seed": int(seed),
                    "status": status.get("status", "missing"),
                    "prediction_rows_exists": pred_path.exists(),
                    "duration_s": status.get("duration_s", np.nan),
                    "train_windows": status.get("train_windows", np.nan),
                    "test_windows": status.get("test_windows", np.nan),
                })
    return pd.DataFrame(rows)


def artifact_table(base_dir: Path) -> pd.DataFrame:
    rows = []
    for name, rel in ARTIFACTS:
        path = base_dir / rel
        rows.append({
            "artifact": name,
            "path": rel,
            "exists": path.exists(),
            "row_count": read_csv_len(path) if path.suffix in {".csv", ".gz"} else None,
            "size_bytes": int(path.stat().st_size) if path.exists() else 0,
        })
    return pd.DataFrame(rows)


def covered_folds(df: pd.DataFrame) -> list[str]:
    if df.empty or "fold_name" not in df.columns:
        return []
    return sorted(str(v) for v in df["fold_name"].dropna().unique())


def best_rows(path: Path, n: int = 12) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty or "MAE_pct" not in df.columns:
        return pd.DataFrame()
    cols = [
        c for c in [
            "model_name",
            "seed",
            "fold_name",
            "target_temperature_C",
            "temperature_C",
            "target_type",
            "comparison_status",
            "source",
            "MAE_pct",
            "RMSE_pct",
            "n_samples",
        ] if c in df.columns
    ]
    return df.sort_values("MAE_pct")[cols].head(n)


def outside_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty or "fold_name" not in df.columns:
        return pd.DataFrame()
    out = df[df["fold_name"].isin(["Omit N10", "Omit 50"])].copy()
    if out.empty:
        return out
    cols = [c for c in ["model_name", "seed", "fold_name", "comparison_status", "source", "MAE_pct", "RMSE_pct", "n_samples"] if c in out.columns]
    return out.sort_values("MAE_pct")[cols].head(20)


def observability_snapshot(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "no_cc_observability_vs_error.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    preferred = [
        "model_name",
        "seed",
        "fold_name",
        "observability_feature",
        "observability_bin",
        "observability_decile",
        "mean_observability_score",
        "MAE_pct",
        "RMSE_pct",
        "abs_error_spearman",
        "spearman_observability_vs_abs_error",
        "n_samples",
        "source",
    ]
    cols = [c for c in preferred if c in df.columns]
    sort_cols = [
        c for c in [
            "model_name",
            "fold_name",
            "observability_feature",
            "observability_bin",
            "observability_decile",
        ] if c in df.columns
    ]
    return df.sort_values(sort_cols)[cols].head(24)


def write_observability_report(base_dir: Path) -> None:
    metrics_path = base_dir / "no_cc_observability_metrics.csv"
    vs_path = base_dir / "no_cc_observability_vs_error.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    vs = pd.read_csv(vs_path) if vs_path.exists() else pd.DataFrame()

    lines = [
        "# Strict No-CC Observability And Uncertainty Report",
        "",
        "## Scope",
        "- This report is generated from saved strict No-CC prediction rows.",
        "- Observability features are derived from V/I/T window response, branch-band energy, and prediction behavior.",
        "- SOC, cumulative Ah, Q_ref/q_cutoff, absolute timestep, trajectory progress, and trajectory ID are not used as model inputs.",
        "- Window-local normalized position may be used; absolute start/end timestep and trajectory progress are forbidden.",
        "- Current is allowed only as instantaneous excitation, not as a current-integrated SOC state.",
        "",
    ]

    if not metrics.empty:
        cols = [c for c in [
            "model_name",
            "seed",
            "fold_name",
            "observability_bin",
            "mean_branch_band_energy",
            "MAE_pct",
            "RMSE_pct",
            "catastrophic_error_rate_5pct",
            "n_samples",
        ] if c in metrics.columns]
        lines += [
            "## Branch-Band Observability Bins",
            metrics.sort_values([c for c in ["model_name", "fold_name", "observability_bin"] if c in metrics.columns])[cols].head(60).to_markdown(index=False),
            "",
        ]
        if {"model_name", "observability_bin", "MAE_pct", "n_samples"}.issubset(metrics.columns):
            summary = (
                metrics.groupby(["model_name", "observability_bin"], observed=False)[["MAE_pct", "RMSE_pct", "n_samples"]]
                .mean(numeric_only=True)
                .reset_index()
                .sort_values(["model_name", "observability_bin"])
            )
            lines += ["## Mean Error By Observability Bin", summary.to_markdown(index=False), ""]
    else:
        lines += ["## Branch-Band Observability Bins", "No `no_cc_observability_metrics.csv` rows are available yet.", ""]

    if not vs.empty:
        cols = [c for c in [
            "model_name",
            "seed",
            "fold_name",
            "observability_feature",
            "observability_bin",
            "observability_decile",
            "mean_observability_score",
            "MAE_pct",
            "RMSE_pct",
            "abs_error_spearman",
            "spearman_observability_vs_abs_error",
            "n_samples",
            "source",
        ] if c in vs.columns]
        lines += [
            "## Observability Versus Error",
            vs.sort_values([c for c in ["model_name", "fold_name", "observability_feature", "observability_bin", "observability_decile"] if c in vs.columns])[cols].head(80).to_markdown(index=False),
            "",
        ]
        corr_col = "abs_error_spearman" if "abs_error_spearman" in vs.columns else "spearman_observability_vs_abs_error" if "spearman_observability_vs_abs_error" in vs.columns else None
        if corr_col and {"model_name", corr_col}.issubset(vs.columns):
            group_cols = [c for c in ["model_name", "observability_feature"] if c in vs.columns]
            if group_cols:
                corr = vs.groupby(group_cols)[corr_col].mean().reset_index().sort_values(corr_col, ascending=False)
                lines += ["## Correlation Summary", corr.to_markdown(index=False), ""]
    else:
        lines += ["## Observability Versus Error", "No `no_cc_observability_vs_error.csv` rows are available yet.", ""]

    lines += [
        "## Interpretation",
        "- Higher MAE in lower-observability bins indicates regions where strict No-CC inference is weakly constrained by voltage/current response.",
        "- Positive rank correlation means the label-free observability/risk proxy tends to increase with absolute error.",
        "- This diagnostic is not a performance claim; it is a failure-region analysis for the strict No-CC rebuild.",
    ]
    text = "\n".join(lines)
    (base_dir / "no_cc_uncertainty_report.md").write_text(text, encoding="utf-8")
    (base_dir / "strict_no_cc_observability_report.md").write_text(text, encoding="utf-8")


def json_clean(value):
    if isinstance(value, dict):
        return {k: json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def write_reports(base_dir: Path, completion: pd.DataFrame, artifacts: pd.DataFrame) -> dict:
    expected = int(len(completion))
    completed = int((completion["status"] == "done").sum()) if expected else 0
    complete = expected > 0 and completed == expected

    band_results = pd.read_csv(base_dir / "no_cc_bandtcn_results.csv") if (base_dir / "no_cc_bandtcn_results.csv").exists() else pd.DataFrame()
    tg_results = pd.read_csv(base_dir / "no_cc_thermoguard_results.csv") if (base_dir / "no_cc_thermoguard_results.csv").exists() else pd.DataFrame()
    comp = pd.read_csv(base_dir / "no_cc_final_comparison.csv") if (base_dir / "no_cc_final_comparison.csv").exists() else pd.DataFrame()

    summary = {
        "expected_bandtcn_runs": expected,
        "completed_bandtcn_runs": completed,
        "bandtcn_sweep_complete": complete,
        "bandtcn_result_rows": int(len(band_results)),
        "thermoguard_result_rows": int(len(tg_results)),
        "final_comparison_rows": int(len(comp)),
        "bandtcn_covered_folds": covered_folds(band_results),
        "thermoguard_covered_folds": covered_folds(tg_results),
        "final_report_status": "final" if complete else "partial_pending_sweep_completion",
    }

    artifacts.to_csv(base_dir / "strict_no_cc_artifact_manifest.csv", index=False)
    manifest_payload = json_clean({"summary": summary, "artifacts": artifacts.to_dict(orient="records")})
    (base_dir / "strict_no_cc_artifact_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    completion_summary = (
        completion.groupby(["fold_name", "seed"])["status"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
        if not completion.empty else pd.DataFrame()
    )
    best_band = best_rows(base_dir / "no_cc_bandtcn_results.csv")
    best_tg = best_rows(base_dir / "no_cc_thermoguard_results.csv")
    best_comp = best_rows(base_dir / "no_cc_final_comparison.csv")
    outside = outside_rows(base_dir / "no_cc_final_comparison.csv")
    obs = observability_snapshot(base_dir)

    lines = [
        "# Strict No-CC Final Comparison Report",
        "",
        f"Status: **{summary['final_report_status']}**",
        "",
        "## Retraction Policy",
        "- Previous CC-assisted NeuralECM/ThermoGuard rows that advanced SOC through current integration are withdrawn as main results.",
        "- The active comparison set is strict No-CC: no SOC input, no cumulative Ah/current-integrated SOC state, no Q_ref/q_cutoff input, no absolute timestep/trajectory progress input.",
        "- Window-local normalized position is allowed, but absolute start/end timestep and trajectory progress are not model inputs.",
        "- Instantaneous current excitation (`I_raw`, `dI`, `absI`) remains allowed unless explicitly running a no-current ablation.",
        "- Branch-band features pass the strict No-CC input audit but remain causal/precomputed voltage-decomposition features, not raw-voltage-only inputs.",
        "",
        "## Sweep Completion",
        f"- Completed BandTCN runs: {completed}/{expected}",
        f"- Covered BandTCN folds: {', '.join(summary['bandtcn_covered_folds']) if summary['bandtcn_covered_folds'] else 'none'}",
        f"- Covered NoCC-ThermoGuard folds: {', '.join(summary['thermoguard_covered_folds']) if summary['thermoguard_covered_folds'] else 'none'}",
        "",
    ]
    if not completion_summary.empty:
        lines += [completion_summary.to_markdown(index=False), ""]

    lines += ["## Artifact Manifest", artifacts.to_markdown(index=False), ""]

    if not best_band.empty:
        lines += ["## Current Best BandTCN Target Rows", best_band.to_markdown(index=False), ""]
    if not best_tg.empty:
        lines += ["## Current NoCC-ThermoGuard Target Rows", best_tg.to_markdown(index=False), ""]
    if not outside.empty:
        lines += ["## Outside-Range Rows", outside.to_markdown(index=False), ""]
    else:
        lines += ["## Outside-Range Rows", "No Omit N10/Omit 50 rows are available yet in the current strict No-CC BandTCN/ThermoGuard artifacts.", ""]
    if not best_comp.empty:
        lines += ["## Current Final-Comparison Snapshot", best_comp.to_markdown(index=False), ""]
    if not obs.empty:
        lines += ["## Observability Snapshot", obs.to_markdown(index=False), ""]

    lines += [
        "## Interpretation Guardrails",
        "- If `Status` is `partial_pending_sweep_completion`, do not cite the report as final performance.",
        "- Robust objectives should be judged by worst target-fold MAE and outside-range average/worst MAE after all expected runs complete.",
        "- NoCC-ThermoGuard should be accepted only if its label-free gate improves strict No-CC outside-range behavior without using test labels.",
        "- A sub-0.7% claim is not supported unless the strict No-CC artifacts show it after completion.",
    ]
    (base_dir / "strict_no_cc_final_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Write strict No-CC artifact manifest and comparison status report.")
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()
    summary = generate_manifest(Path(args.base_dir))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def generate_manifest(base_dir: Path) -> dict:
    base_dir = Path(base_dir)
    cfg = config_from_metadata(base_dir)
    completion = completion_table(base_dir, cfg)
    write_observability_report(base_dir)
    artifacts = artifact_table(base_dir)
    completion.to_csv(base_dir / "strict_no_cc_bandtcn_completion.csv", index=False)
    summary = write_reports(base_dir, completion, artifacts)
    return summary


if __name__ == "__main__":
    main()
