from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import json

import pandas as pd


FOCUS_COLS = [
    "model_name",
    "scope",
    "average_MAE_pct",
    "worst_MAE_pct",
    "average_RMSE_pct",
    "worst_RMSE_pct",
    "average_catastrophic_error_rate_5pct",
    "n_folds",
]


RESULT_COLS = [
    "model_name",
    "seed",
    "fold_name",
    "target_temperature_C",
    "target_type",
    "MAE_pct",
    "RMSE_pct",
    "catastrophic_error_rate_5pct",
    "plateau_20_80_MAE_pct",
    "low_current_MAE_pct",
    "n_samples",
]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def table(df: pd.DataFrame, cols: list[str] | None = None, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows available._"
    out = df.copy()
    if cols is not None:
        out = out[[c for c in cols if c in out.columns]]
    if max_rows is not None:
        out = out.head(max_rows)
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(4)
    return out.to_markdown(index=False)


def best_row(df: pd.DataFrame, scope: str) -> pd.Series | None:
    if df.empty or "scope" not in df.columns or "average_MAE_pct" not in df.columns:
        return None
    sub = df[df["scope"].eq(scope)].copy()
    if sub.empty:
        return None
    return sub.sort_values("average_MAE_pct").iloc[0]


def load_metadata(base_dir: Path) -> dict:
    path = base_dir / "no_cc_bandtcn_metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def make_handoff(base_dir: Path, out_path: Path) -> Path:
    focus = read_csv(base_dir / "no_cc_bandtcn_focus.csv")
    results = read_csv(base_dir / "no_cc_bandtcn_results.csv")
    by_temp = read_csv(base_dir / "no_cc_bandtcn_by_temperature.csv")
    tg_focus = read_csv(base_dir / "no_cc_thermoguard_focus.csv")
    tg_results = read_csv(base_dir / "no_cc_thermoguard_results.csv")
    final_comp = read_csv(base_dir / "no_cc_final_comparison.csv")
    obs = read_csv(base_dir / "no_cc_observability_metrics.csv")
    obs_error = read_csv(base_dir / "no_cc_observability_vs_error.csv")
    weights = read_csv(base_dir / "no_cc_thermoguard_expert_weights.csv")
    metadata = load_metadata(base_dir)

    run_count = 0
    seeds = []
    if not results.empty:
        run_count = len(results[["model_name", "seed", "fold_name"]].drop_duplicates())
        seeds = sorted(results["seed"].dropna().astype(int).unique().tolist()) if "seed" in results else []

    all_best = best_row(focus, "all_target_folds")
    outside_best = best_row(focus, "outside_range")
    omitted_best = best_row(focus, "omitted_A_B_C")

    lines: list[str] = [
        "# Strict NoCC BandTCN GPT Handoff",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Base directory: `{base_dir}`",
        f"- Completed result rows: `{run_count}`",
        f"- Included seeds in current aggregate: `{seeds}`",
        "",
        "## Strict No-CC Policy",
        "",
        "- No SOC input, no usable SOC input, no SOC_CC input.",
        "- No cumulative Ah, integrated-current feature, trajectory progress, absolute time index, trajectory ID, or shifted target input.",
        "- No explicit SOC state update of the form `SOC_{t+1}=SOC_t-I_t*dt/Q_eff`.",
        "- Current is used only as instantaneous excitation through `I_raw`, `dI`, `absI`, and voltage-response features.",
        "- Target label is physical_smoothQ SOC.",
        "- Old NeuralECM/ThermoGuard current-integration results are invalidated as main evidence.",
        "",
        "## Implementation Snapshot",
        "",
        "- Main runner: `python -m soc_decomp.no_cc_bandtcn_experiment`",
        "- Model family: stateless fixed-window causal BandTCN endpoint SOC prediction.",
        "- Fusion: NoCC-ThermoGuard BandTCN uses label-free weights from temperature, expert disagreement, branch-band/OOD proxy, and recent prediction jitter.",
        f"- Feature count: `{len(metadata.get('feature_cols', []))}`",
        f"- AMP used: `{metadata.get('amp_used', False)}`",
        f"- Device recorded: `{metadata.get('device', 'unknown')}`",
        "",
        "## Best Rows",
        "",
    ]
    if all_best is not None:
        lines += [
            f"- Best all-target BandTCN: `{all_best['model_name']}` with average MAE `{all_best['average_MAE_pct']:.4f}%` and worst MAE `{all_best['worst_MAE_pct']:.4f}%`.",
        ]
    if outside_best is not None:
        lines += [
            f"- Best outside-range BandTCN: `{outside_best['model_name']}` with average MAE `{outside_best['average_MAE_pct']:.4f}%` and worst MAE `{outside_best['worst_MAE_pct']:.4f}%`.",
        ]
    if omitted_best is not None:
        lines += [
            f"- Best omitted A/B/C BandTCN: `{omitted_best['model_name']}` with average MAE `{omitted_best['average_MAE_pct']:.4f}%` and worst MAE `{omitted_best['worst_MAE_pct']:.4f}%`.",
        ]

    lines += [
        "",
        "## BandTCN Focus Table",
        "",
        table(focus.sort_values(["scope", "average_MAE_pct"]) if not focus.empty else focus, FOCUS_COLS),
        "",
        "## NoCC-ThermoGuard Focus Table",
        "",
        table(tg_focus.sort_values(["scope", "average_MAE_pct"]) if not tg_focus.empty else tg_focus, FOCUS_COLS),
        "",
        "## Target Fold Results",
        "",
        table(results.sort_values(["fold_name", "MAE_pct"]) if not results.empty else results, RESULT_COLS),
        "",
        "## Outside Fold Detail",
        "",
    ]
    if not results.empty:
        outside = results[results["fold_name"].isin(["Omit N10", "Omit 50"])].sort_values(["fold_name", "MAE_pct"])
        lines.append(table(outside, RESULT_COLS))
    else:
        lines.append("_No outside rows available._")

    lines += [
        "",
        "## Temperature Table Preview",
        "",
        table(by_temp.sort_values(["model_name", "fold_name", "temperature_C"]) if not by_temp.empty else by_temp, max_rows=80),
        "",
        "## Observability Diagnostics",
        "",
    ]
    if not obs.empty:
        obs_summary = (
            obs.groupby(["model_name", "observability_bin"], observed=False)
            [["MAE_pct", "RMSE_pct", "n_samples"]]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values(["model_name", "observability_bin"])
        )
        lines.append(table(obs_summary, max_rows=80))
    else:
        lines.append("_No observability metrics available._")

    lines += [
        "",
        "## Observability Vs Error Preview",
        "",
    ]
    if not obs_error.empty:
        corr = (
            obs_error.groupby(["model_name", "observability_feature"])["abs_error_spearman"]
            .mean()
            .reset_index()
            .sort_values(["model_name", "abs_error_spearman"], ascending=[True, False])
        )
        lines.append(table(corr, max_rows=80))
    else:
        lines.append("_No observability-vs-error rows available._")

    lines += [
        "",
        "## ThermoGuard Weight Preview",
        "",
        table(weights.describe().reset_index() if not weights.empty else weights),
        "",
        "## Final Comparison Preview",
        "",
        table(final_comp, max_rows=80),
        "",
        "## Direct Conclusions",
        "",
        "1. Removing explicit current integration substantially degrades outside-temperature performance in the current strict NoCC setup.",
        "2. Omit -10 and Omit 50 are the most affected folds; the best outside MAE remains several percent, not near 0.7%.",
        "3. Branch-band representation helps included/intermediate conditions more than true outside temperature extrapolation.",
        "4. REx/GroupDRO/highT do not reliably improve outside worst-case in the current one-seed aggregate.",
        "5. NoCC-ThermoGuard does not beat the best simple BandTCN outside-range row in the current aggregate.",
        "6. Safe paper wording: current is used only as instantaneous excitation, not integrated into SOC state. These results do not prove current integration is unnecessary and do not prove temperature extrapolation is solved.",
        "",
        "## Files To Attach Or Reference",
        "",
        "- `branch_bands_leakage_audit.md`",
        "- `delta_start_time_audit.md`",
        "- `band_feature_schema.csv`",
        "- `no_cc_bandtcn_results.csv`",
        "- `no_cc_bandtcn_by_temperature.csv`",
        "- `no_cc_bandtcn_focus.csv`",
        "- `no_cc_thermoguard_results.csv`",
        "- `no_cc_thermoguard_focus.csv`",
        "- `no_cc_observability_metrics.csv`",
        "- `no_cc_observability_vs_error.csv`",
        "- `no_cc_uncertainty_report.md`",
        "- `no_cc_final_comparison_report.md`",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    parser = ArgumentParser(description="Build a copy-pasteable strict NoCC GPT handoff markdown file.")
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("no_cc_gpt_handoff.md"))
    args = parser.parse_args()
    path = make_handoff(args.base_dir.resolve(), args.out if args.out.is_absolute() else args.base_dir / args.out)
    print(path)


if __name__ == "__main__":
    main()
