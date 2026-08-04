from __future__ import annotations

from pathlib import Path
import argparse
import json

import pandas as pd


HANDOFF_FILES = [
    "branchbands_completion_summary.json",
    "branchbands_completion_audit.csv",
    "branchbands_remaining_runs.csv",
    "branchbands_leakage_audit.csv",
    "branchbands_input_schema.csv",
    "branchbands_improvement_results.csv",
    "branchbands_improvement_by_temperature.csv",
    "branchbands_improvement_focus.csv",
    "branchbands_improvement_decision_summary.csv",
    "branchbands_improvement_promotion.csv",
    "branchband_residual_mode_results.csv",
    "branchbands_observability_metrics.csv",
    "branchbands_observability_vs_error.csv",
    "branchbands_cold_expert_fusion_results.csv",
    "branchbands_cold_expert_weights.csv",
    "branchbands_improvement_report.md",
    "BranchBands_TCN_NoCC_resume_screening.ipynb",
    "check_branchbands_status.sh",
    "run_branchbands_resume.sh",
    "sync_branchbands_results_from_remote.sh",
]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    out = df.head(max_rows) if max_rows else df
    try:
        return out.to_markdown(index=False)
    except ImportError:
        return "```\n" + out.to_string(index=False) + "\n```"


def artifact_table(base_dir: Path) -> pd.DataFrame:
    rows = []
    for name in HANDOFF_FILES:
        path = base_dir / name
        rows.append({
            "artifact": name,
            "exists": path.exists(),
            "size_bytes": int(path.stat().st_size) if path.exists() else 0,
        })
    return pd.DataFrame(rows)


def generate_handoff(base_dir: Path, output: Path) -> Path:
    base_dir = Path(base_dir)
    output = Path(output)
    summary = read_json(base_dir / "branchbands_completion_summary.json")
    remaining = read_csv(base_dir / "branchbands_remaining_runs.csv")
    audit = read_csv(base_dir / "branchbands_completion_audit.csv")
    decision = read_csv(base_dir / "branchbands_improvement_decision_summary.csv")
    results = read_csv(base_dir / "branchbands_improvement_results.csv")
    promotion = read_csv(base_dir / "branchbands_improvement_promotion.csv")
    by_temp = read_csv(base_dir / "branchbands_improvement_by_temperature.csv")

    completed = summary.get("completed_runs", 0)
    expected = summary.get("expected_runs", 42)
    status = summary.get("status", "unknown")
    next_run = pd.DataFrame()
    if not remaining.empty and "status" in remaining:
        next_run = remaining[remaining["status"].eq("remaining")].head(1)

    non_pass = pd.DataFrame()
    if not audit.empty and "status" in audit:
        non_pass = audit[~audit["status"].eq("PASS")].copy()

    result_cols = [
        "model_name",
        "fold_name",
        "target_temperature_C",
        "MAE_pct",
        "RMSE_pct",
        "jitter_ratio",
        "catastrophic_error_rate_5pct",
        "plateau_20_80_MAE_pct",
        "low_current_MAE_pct",
    ]
    if not results.empty:
        show_cols = [c for c in result_cols if c in results.columns]
        results_show = results[show_cols].sort_values(["fold_name", "MAE_pct"])
    else:
        results_show = pd.DataFrame()

    temp_show = pd.DataFrame()
    if not by_temp.empty:
        show_cols = [c for c in ["model_name", "fold_name", "temperature_C", "MAE_pct", "RMSE_pct"] if c in by_temp.columns]
        temp_show = by_temp[show_cols].sort_values(["fold_name", "model_name", "temperature_C"]).tail(24)

    lines = [
        "# BranchBands TCN Strict NoCC GPT Handoff",
        "",
        "## Status",
        f"- Completion status: `{status}`",
        f"- Completed runs: {completed}/{expected}",
        f"- Remaining runs: {summary.get('remaining_runs', 'unknown')}",
        "- Strict policy: no SOC input, no window-start SOC, no SOC_CC, no cumulative Ah, no absolute time/trajectory progress, no explicit current-integration SOC state update.",
        "- Current is used only as instantaneous excitation, not integrated into SOC state.",
        "",
        "## Next Run",
        table(next_run),
        "",
        "## How To Resume On Remote",
        "```bash",
        "/home/lab/바탕화면/LSTM_STATELESS_DECOMP_SOC_branchbands_resume/run_branchbands_resume.sh /home/lab/바탕화면/LSTM_STATELESS_DECOMP_SOC_branchbands_resume",
        "```",
        "",
        "## How To Pull Results Back",
        "```bash",
        "./sync_branchbands_results_from_remote.sh /home/user/바탕화면/DL/LSTM_STATELESS_DECOMP_SOC",
        "```",
        "",
        "## Completion Audit Non-Pass Rows",
        table(non_pass),
        "",
        "## Decision Summary",
        table(decision),
        "",
        "## Target-Fold Results",
        table(results_show, max_rows=80),
        "",
        "## Promotion Table",
        table(promotion),
        "",
        "## Latest By-Temperature Rows",
        table(temp_show, max_rows=40),
        "",
        "## Artifacts",
        table(artifact_table(base_dir)),
        "",
        "## Safe Interpretation",
        "- Treat current results as partial until completion audit status is `complete`.",
        "- Do not claim NoCC proves current integration unnecessary.",
        "- Do not claim NoCC solves temperature extrapolation.",
        "- If promoted candidates appear, they are one-seed screening candidates only until seed 0/1/2 confirmation is complete.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def parse_args():
    p = argparse.ArgumentParser(description="Generate a GPT handoff summary for BranchBands NoCC screening.")
    p.add_argument("--base-dir", type=Path, default=Path("."))
    p.add_argument("--output", type=Path, default=Path("branchbands_gpt_handoff.md"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = generate_handoff(args.base_dir, args.output)
    print(out)


if __name__ == "__main__":
    main()
