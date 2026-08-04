from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import pandas as pd

from .strict_no_cc_artifact_manifest import generate_manifest


REQUIRED_ARTIFACTS = {
    "branch_bands_audit_json": "branch_bands_strict_no_cc_audit.json",
    "branch_bands_audit_csv": "branch_bands_strict_no_cc_audit.csv",
    "branch_bands_audit_report": "branch_bands_strict_no_cc_audit.md",
    "bandtcn_results": "no_cc_bandtcn_results.csv",
    "bandtcn_by_temperature": "no_cc_bandtcn_by_temperature.csv",
    "bandtcn_focus": "no_cc_bandtcn_focus.csv",
    "bandtcn_predictions": "no_cc_bandtcn_prediction_rows.csv.gz",
    "robust_domain_summary": "no_cc_bandtcn_domain_robust_summary.csv",
    "thermoguard_results": "no_cc_thermoguard_results.csv",
    "thermoguard_by_temperature": "no_cc_thermoguard_by_temperature.csv",
    "thermoguard_focus": "no_cc_thermoguard_focus.csv",
    "thermoguard_weights": "no_cc_thermoguard_expert_weights.csv",
    "observability_metrics": "no_cc_observability_metrics.csv",
    "observability_vs_error": "no_cc_observability_vs_error.csv",
    "observability_uncertainty_report": "no_cc_uncertainty_report.md",
    "final_comparison_csv": "no_cc_final_comparison.csv",
    "final_comparison_report": "strict_no_cc_final_comparison_report.md",
}

REQUIRED_FOLDS = {"Exp A", "Exp B", "Exp C", "Exp D", "Omit N10", "Omit 50"}
REQUIRED_ROBUST_MODELS = {"BandTCN_REX", "BandTCN_GroupDRO", "BandTCN_REX_GroupDRO"}


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def file_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def csv_nonempty(path: Path) -> bool:
    if not file_nonempty(path):
        return False
    try:
        return len(pd.read_csv(path)) > 0
    except Exception:
        return False


def table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def append_check(rows: list[dict], requirement: str, passed: bool, evidence: str):
    rows.append({
        "requirement": requirement,
        "passed": bool(passed),
        "evidence": evidence,
    })


def audit(base_dir: Path) -> tuple[dict, pd.DataFrame]:
    summary = generate_manifest(base_dir)
    manifest = read_json(base_dir / "strict_no_cc_artifact_manifest.json")
    artifacts = {row.get("artifact"): row for row in manifest.get("artifacts", [])}
    rows: list[dict] = []

    append_check(
        rows,
        "BandTCN sweep complete",
        bool(summary.get("bandtcn_sweep_complete")),
        f"{summary.get('completed_bandtcn_runs')}/{summary.get('expected_bandtcn_runs')} runs complete",
    )

    for name, rel in REQUIRED_ARTIFACTS.items():
        path = base_dir / rel
        ok = csv_nonempty(path) if path.suffix in {".csv", ".gz"} else file_nonempty(path)
        manifest_row = artifacts.get(name, {})
        append_check(
            rows,
            f"required artifact exists and is nonempty: {name}",
            ok,
            f"{rel}; manifest_exists={manifest_row.get('exists')}; rows={manifest_row.get('row_count')}; bytes={manifest_row.get('size_bytes')}",
        )

    audit_json = read_json(base_dir / "branch_bands_strict_no_cc_audit.json")
    append_check(
        rows,
        "branch_bands strict No-CC audit passes",
        bool(audit_json.get("strict_no_cc_input_pass")),
        f"selected_forbidden={audit_json.get('selected_forbidden_features')}; missing={audit_json.get('missing_selected_features_after_runtime_derivations')}",
    )
    append_check(
        rows,
        "branch_bands selected inputs have no forbidden columns",
        not audit_json.get("selected_forbidden_features"),
        f"selected_forbidden={audit_json.get('selected_forbidden_features')}",
    )
    absolute_policy = str(audit_json.get("absolute_time_policy", ""))
    append_check(
        rows,
        "strict No-CC time policy forbids absolute timestep while allowing window-local position",
        "not selected as model input" in absolute_policy and "window-local normalized position is allowed" in absolute_policy,
        f"absolute_time_policy={absolute_policy}",
    )

    band = table(base_dir / "no_cc_bandtcn_results.csv")
    band_folds = set(band.get("fold_name", pd.Series(dtype=str)).dropna().astype(str).unique())
    append_check(
        rows,
        "BandTCN target results cover all folds",
        REQUIRED_FOLDS.issubset(band_folds),
        f"covered={sorted(band_folds)}",
    )

    robust = table(base_dir / "no_cc_bandtcn_domain_robust_summary.csv")
    robust_models = set(robust.get("model_name", pd.Series(dtype=str)).dropna().astype(str).unique())
    robust_folds = set(robust.get("fold_name", pd.Series(dtype=str)).dropna().astype(str).unique())
    append_check(
        rows,
        "robust BandTCN rows include REX/GroupDRO/REX_GroupDRO",
        REQUIRED_ROBUST_MODELS.issubset(robust_models),
        f"models={sorted(robust_models)}",
    )
    append_check(
        rows,
        "robust BandTCN rows cover all folds",
        REQUIRED_FOLDS.issubset(robust_folds),
        f"covered={sorted(robust_folds)}",
    )

    thermo = table(base_dir / "no_cc_thermoguard_results.csv")
    thermo_folds = set(thermo.get("fold_name", pd.Series(dtype=str)).dropna().astype(str).unique())
    append_check(
        rows,
        "NoCC-ThermoGuard target rows cover all folds",
        REQUIRED_FOLDS.issubset(thermo_folds),
        f"covered={sorted(thermo_folds)}",
    )

    obs = table(base_dir / "no_cc_observability_metrics.csv")
    obs_folds = set(obs.get("fold_name", pd.Series(dtype=str)).dropna().astype(str).unique())
    append_check(
        rows,
        "observability diagnostics cover all folds",
        REQUIRED_FOLDS.issubset(obs_folds),
        f"covered={sorted(obs_folds)}",
    )

    final_text_path = base_dir / "strict_no_cc_final_comparison_report.md"
    final_text = final_text_path.read_text(encoding="utf-8") if final_text_path.exists() else ""
    append_check(
        rows,
        "final report records CC-assisted withdrawal policy",
        "withdrawn as main results" in final_text and "strict No-CC" in final_text,
        "checked report text for withdrawal and strict No-CC policy",
    )
    append_check(
        rows,
        "final report records absolute-timestep exclusion policy",
        "absolute start/end timestep and trajectory progress are not model inputs" in final_text,
        "checked report text for absolute timestep exclusion and window-local position policy",
    )
    append_check(
        rows,
        "final report is not partial",
        "partial_pending_sweep_completion" not in final_text and summary.get("final_report_status") == "final",
        f"final_report_status={summary.get('final_report_status')}",
    )

    checks = pd.DataFrame(rows)
    passed = bool(len(checks) and checks["passed"].all())
    result = {
        "strict_no_cc_goal_complete": passed,
        "n_checks": int(len(checks)),
        "n_failed": int((~checks["passed"]).sum()) if len(checks) else 0,
        "failed_requirements": checks.loc[~checks["passed"], "requirement"].tolist() if len(checks) else [],
    }
    return result, checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit whether strict No-CC goal artifacts are complete.")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--allow-partial-exit-zero", action="store_true")
    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    result, checks = audit(base_dir)
    checks.to_csv(base_dir / "strict_no_cc_completion_audit.csv", index=False)
    (base_dir / "strict_no_cc_completion_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["strict_no_cc_goal_complete"] or args.allow_partial_exit_zero:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
