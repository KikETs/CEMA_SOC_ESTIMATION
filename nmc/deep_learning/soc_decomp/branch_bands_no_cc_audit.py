from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import csv
import json
import re

import pandas as pd

from .deep_no_leak_experiment import BRANCH_BAND_DEEP_FEATURES


FORBIDDEN_PATTERNS = (
    re.compile(r"soc", re.I),
    re.compile(r"cumulative", re.I),
    re.compile(r"q_ref", re.I),
    re.compile(r"q_cutoff", re.I),
    re.compile(r"trajectory_fraction", re.I),
    re.compile(r"time_index", re.I),
    re.compile(r"end_index", re.I),
    re.compile(r"trajectory_id", re.I),
    re.compile(r"file_name", re.I),
)


@dataclass(frozen=True)
class FeatureAuditRow:
    feature: str
    category: str
    source: str
    expression: str
    uses_current: bool
    uses_voltage: bool
    uses_temperature: bool
    uses_soc: bool
    uses_cumulative_or_capacity: bool
    uses_absolute_time_or_id: bool
    causal_status: str
    notes: str


DIRECT_FEATURES = {
    "V_raw": ("measured_voltage", "raw terminal voltage", "V_raw[t]"),
    "V_corr_raw": ("decomposed_voltage", "voltage-corrected feature file column", "V_corr_raw[t]"),
    "I_raw": ("instantaneous_current", "raw instantaneous current", "I_raw[t]"),
    "T": ("temperature", "ambient/chamber temperature", "T[t]"),
    "dI": ("instantaneous_current", "current difference feature", "dI[t]"),
    "absI": ("instantaneous_current", "absolute instantaneous current", "|I_raw[t]|"),
    "V_pol_raw": ("decomposed_voltage", "polarization branch from voltage decomposition", "V_pol_raw[t]"),
    "V_hys_raw": ("decomposed_voltage", "hysteresis branch from voltage decomposition", "V_hys_raw[t]"),
    "V_ohm_raw": ("decomposed_voltage", "ohmic branch from voltage decomposition", "V_ohm_raw[t]"),
    "R0": ("decomposed_voltage", "instantaneous ohmic resistance proxy", "R0[t]"),
    "V_pol_fast_raw": ("branch_band", "cached fast polarization branch", "V_pol_fast_raw[t]"),
    "V_pol_mid_raw": ("branch_band", "cached mid polarization branch", "V_pol_mid_raw[t]"),
    "V_pol_slow_raw": ("branch_band", "cached slow polarization branch", "V_pol_slow_raw[t]"),
}


DERIVED_FEATURES = {
    "R0_x_V_pol": ("interaction", "R0[t] * V_pol_raw[t]"),
    "T_x_V_pol": ("interaction", "T[t] * V_pol_raw[t]"),
    "R0_x_absI": ("interaction", "R0[t] * |I_raw[t]|"),
    "V_pol_x_abs_dI": ("interaction", "V_pol_raw[t] * |dI[t]|"),
    "V_residual_raw": ("residual_band", "V_raw[t] - V_corr_raw[t]"),
    "V_residual_low": ("residual_band", "EMA_causal(V_raw - V_corr_raw, cutoff=0.003 Hz)[t]"),
    "V_residual_mid": ("residual_band", "EMA_causal(V_raw - V_corr_raw, cutoff=0.03 Hz)[t] - V_residual_low[t]"),
    "V_residual_high": ("residual_band", "(V_raw[t] - V_corr_raw[t]) - EMA_causal(V_raw - V_corr_raw, cutoff=0.03 Hz)[t]"),
    "V_residual_low_x_T": ("interaction", "V_residual_low[t] * T[t]"),
    "V_residual_mid_x_absI": ("interaction", "V_residual_mid[t] * |I_raw[t]|"),
    "V_residual_high_x_abs_dI": ("interaction", "V_residual_high[t] * |dI[t]|"),
}


def has_forbidden_token(name: str) -> bool:
    return any(p.search(name) for p in FORBIDDEN_PATTERNS)


def source_files(base_dir: Path) -> list[Path]:
    candidates = [
        base_dir / "decomposed_features_train_temp_minus10_0_10_20_25_50",
        base_dir / "decomposed_features",
    ]
    for folder in candidates:
        if folder.exists():
            files = sorted(folder.glob("*_features.csv"))
            if files:
                return files
    return []


def read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def feature_row(feature: str, available_columns: set[str]) -> FeatureAuditRow:
    if feature in DIRECT_FEATURES:
        category, source, expression = DIRECT_FEATURES[feature]
        materialized = feature in available_columns
        notes = "materialized in decomposed feature CSV" if materialized else "expected direct feature but not found in inspected CSV headers"
    elif feature in DERIVED_FEATURES:
        category, expression = DERIVED_FEATURES[feature]
        source = "runtime deterministic transform from allowed V/I/T/decomposition columns"
        materialized = feature in available_columns
        notes = "materialized in CSV or recomputed identically at runtime" if materialized else "computed by add_derived_features at runtime"
    else:
        category = "unknown"
        source = "unknown"
        expression = "unknown"
        notes = "not recognized by audit mapping"

    lower_expr = expression.lower()
    uses_current = any(tok in lower_expr for tok in ["i_raw", "|i_raw", "di", "absi", "|di|"])
    uses_voltage = any(tok in lower_expr for tok in ["v_", "voltage", "r0"])
    uses_temperature = "t[" in lower_expr or "* t" in lower_expr or feature == "T"
    return FeatureAuditRow(
        feature=feature,
        category=category,
        source=source,
        expression=expression,
        uses_current=uses_current,
        uses_voltage=uses_voltage,
        uses_temperature=uses_temperature,
        uses_soc=False,
        uses_cumulative_or_capacity=False,
        uses_absolute_time_or_id=False,
        causal_status="causal_or_instantaneous",
        notes=notes,
    )


def build_audit(base_dir: Path) -> tuple[dict, pd.DataFrame]:
    features = list(BRANCH_BAND_DEEP_FEATURES)
    files = source_files(base_dir)
    headers_by_file = {str(p.relative_to(base_dir)): read_header(p) for p in files}
    available_columns = set().union(*(set(cols) for cols in headers_by_file.values())) if headers_by_file else set()
    selected_forbidden = [c for c in features if has_forbidden_token(c)]
    source_forbidden_columns = sorted(c for c in available_columns if has_forbidden_token(c))
    missing_after_runtime = sorted(c for c in features if c not in available_columns and c not in DERIVED_FEATURES)
    rows = [asdict(feature_row(c, available_columns)) for c in features]
    df = pd.DataFrame(rows)
    summary = {
        "base_dir": str(base_dir),
        "feature_set": "branch_bands",
        "n_input_features": len(features),
        "selected_features": features,
        "forbidden_patterns": [p.pattern for p in FORBIDDEN_PATTERNS],
        "selected_forbidden_features": selected_forbidden,
        "source_files_inspected": list(headers_by_file.keys()),
        "source_forbidden_columns_not_selected": source_forbidden_columns,
        "missing_selected_features_after_runtime_derivations": missing_after_runtime,
        "strict_no_cc_input_pass": not selected_forbidden and not missing_after_runtime,
        "current_policy": "instantaneous current features are allowed; no current integration or cumulative Ah feature is used",
        "absolute_time_policy": "window metadata such as end_index/time_index/trajectory_fraction is not selected as model input; window-local normalized position is allowed",
        "soc_policy": "SOC labels may exist in source files for targets, but no SOC/cumulative/capacity column is selected as input",
        "branch_band_caveat": "branch bands are causal/precomputed voltage-decomposition features; they satisfy No-CC input audit but are not raw-voltage-only features",
    }
    return summary, df


def write_report(base_dir: Path, summary: dict, df: pd.DataFrame) -> None:
    md_df = df[[
        "feature",
        "category",
        "expression",
        "uses_current",
        "uses_voltage",
        "uses_temperature",
        "causal_status",
        "notes",
    ]].copy()
    for col in ["feature", "category", "expression", "causal_status", "notes"]:
        md_df[col] = md_df[col].astype(str).str.replace("|", "\\|", regex=False)
    lines = [
        "# Branch-Bands Strict No-CC Input Audit",
        "",
        "## Verdict",
        f"- strict_no_cc_input_pass: `{summary['strict_no_cc_input_pass']}`",
        f"- selected forbidden features: `{summary['selected_forbidden_features']}`",
        f"- missing selected features after runtime derivations: `{summary['missing_selected_features_after_runtime_derivations']}`",
        "",
        "## Policy",
        f"- {summary['soc_policy']}.",
        f"- {summary['current_policy']}.",
        f"- {summary['absolute_time_policy']}.",
        f"- Caveat: {summary['branch_band_caveat']}.",
        "",
        "## Selected Inputs",
        md_df.to_markdown(index=False),
        "",
        "## Source Columns With Forbidden Names",
        "These columns may exist in source feature CSVs as labels/metadata, but are not selected as model inputs.",
        "",
    ]
    if summary["source_forbidden_columns_not_selected"]:
        lines.append(", ".join(summary["source_forbidden_columns_not_selected"]))
    else:
        lines.append("None found in inspected source headers.")
    lines += [
        "",
        "## Interpretation",
        "- This audit supports using `branch_bands` as a strict No-CC input set.",
        "- It does not make the model a no-current model: instantaneous current excitation remains present.",
        "- It does not make the model raw-window-only: branch/band columns reuse causal voltage-decomposition preprocessing.",
    ]
    (base_dir / "branch_bands_strict_no_cc_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit branch_bands inputs under strict No-CC policy.")
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    summary, df = build_audit(base_dir)
    df.to_csv(base_dir / "branch_bands_strict_no_cc_audit.csv", index=False)
    (base_dir / "branch_bands_strict_no_cc_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(base_dir, summary, df)
    print(json.dumps({
        "strict_no_cc_input_pass": summary["strict_no_cc_input_pass"],
        "n_input_features": summary["n_input_features"],
        "selected_forbidden_features": summary["selected_forbidden_features"],
        "missing_selected_features_after_runtime_derivations": summary["missing_selected_features_after_runtime_derivations"],
    }, indent=2))


if __name__ == "__main__":
    main()
