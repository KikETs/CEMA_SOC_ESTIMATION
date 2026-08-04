from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = {"0C": 1.0, "25C": 0.7, "45C": 0.3}
FORBIDDEN_SOURCE_TERMS = (
    "lfp",
    "capacity_anchor",
    "charge_conservation",
    "causal_observer",
    "online_capacity",
    "three_profile_capacity",
)


@dataclass(frozen=True)
class VerifierPaths:
    base_dir: Path = Path(".")
    gate_rows: Path = Path("paper_results/universal_protocol_gate_rows.csv")
    shift_guard_by_temp: Path = Path(
        "remote_result_summaries/"
        "nmc_shift_guard_fast_temperature_profile_soc_balanced_e90_E1_DST_US06_to_FUDS_seed0_by_temperature.csv"
    )
    out_csv: Path = Path("paper_results/nmc_strict_nocc_clean_adoption_verifier.csv")
    out_summary: Path = Path("paper_results/nmc_strict_nocc_clean_adoption_summary.csv")
    out_md: Path = Path("paper_results/nmc_strict_nocc_clean_adoption_verifier.md")


def _norm_source(text: object) -> str:
    return str(text).lower().replace("-", "_")


def _has_forbidden_source(row: pd.Series) -> bool:
    haystack = " ".join(_norm_source(row.get(c, "")) for c in ("source_file", "protocol", "case"))
    return any(term in haystack for term in FORBIDDEN_SOURCE_TERMS)


def _target_met(mae_0: float, mae_25: float, mae_45: float) -> bool:
    return bool(mae_0 < TARGETS["0C"] and mae_25 < TARGETS["25C"] and mae_45 < TARGETS["45C"])


def _add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_ratio_0C"] = out["MAE_0C"] / TARGETS["0C"]
    out["target_ratio_25C"] = out["MAE_25C"] / TARGETS["25C"]
    out["target_ratio_45C"] = out["MAE_45C"] / TARGETS["45C"]
    out["target_norm_worst"] = out[["target_ratio_0C", "target_ratio_25C", "target_ratio_45C"]].max(axis=1)
    out["target_norm_mean"] = out[["target_ratio_0C", "target_ratio_25C", "target_ratio_45C"]].mean(axis=1)
    out["target_met"] = [
        _target_met(float(a), float(b), float(c))
        for a, b, c in zip(out["MAE_0C"], out["MAE_25C"], out["MAE_45C"])
    ]
    ratio_cols = ["target_ratio_0C", "target_ratio_25C", "target_ratio_45C"]
    worst_idx = out[ratio_cols].to_numpy().argmax(axis=1)
    out["worst_temp"] = np.array(["0C", "25C", "45C"], dtype=object)[worst_idx]
    return out


def _load_gate_rows(paths: VerifierPaths) -> pd.DataFrame:
    path = paths.base_dir / paths.gate_rows
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    keep = [
        "source_file",
        "protocol",
        "case",
        "seed",
        "uses_stage2_correction",
        "is_base_first_protocol",
        "is_predeclared_selected_row",
        "MAE_0C",
        "MAE_25C",
        "MAE_45C",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["evidence_source"] = str(paths.gate_rows)
    return df.drop_duplicates()


def _load_shift_guard(paths: VerifierPaths) -> pd.DataFrame:
    path = paths.base_dir / paths.shift_guard_by_temp
    if not path.exists():
        return pd.DataFrame()
    by_temp = pd.read_csv(path)
    values = {}
    for temp, key in [(0.0, "MAE_0C"), (25.0, "MAE_25C"), (45.0, "MAE_45C")]:
        sub = by_temp[np.isclose(by_temp["temperature_C"], temp)]
        values[key] = float(sub["MAE_pct"].iloc[0]) if len(sub) else np.nan
    row = {
        "source_file": paths.shift_guard_by_temp.name,
        "protocol": "shift_guard_tcn_vcorr_it_only_valid_score",
        "case": "E1_DST_US06_to_FUDS",
        "seed": 0,
        "uses_stage2_correction": False,
        "is_base_first_protocol": True,
        "is_predeclared_selected_row": True,
        "evidence_source": str(paths.shift_guard_by_temp),
        **values,
    }
    return pd.DataFrame([row])


def _classify(df: pd.DataFrame) -> pd.DataFrame:
    out = _add_target_columns(df)
    out["forbidden_source_term"] = out.apply(_has_forbidden_source, axis=1)
    out["clean_base_first_eligible"] = (
        (~out["forbidden_source_term"])
        & (~out["uses_stage2_correction"].astype(bool))
        & out["is_base_first_protocol"].astype(bool)
        & out["is_predeclared_selected_row"].astype(bool)
    )
    out["paper_main_accepted"] = out["clean_base_first_eligible"] & out["target_met"].astype(bool)

    reason = []
    for row in out.to_dict("records"):
        if row["forbidden_source_term"]:
            reason.append("reject_forbidden_source_family")
        elif bool(row["uses_stage2_correction"]) and not bool(row["is_base_first_protocol"]):
            reason.append("not_main_stage2_before_base_pass")
        elif not bool(row["is_predeclared_selected_row"]):
            reason.append("diagnostic_or_test_peek_row")
        elif not bool(row["is_base_first_protocol"]):
            reason.append("not_base_first_protocol")
        elif bool(row["target_met"]):
            reason.append("accepted_clean_base_first")
        else:
            reason.append(f"reject_base_first_miss_{row['worst_temp']}")
    out["decision_reason"] = reason
    order = [
        "paper_main_accepted",
        "clean_base_first_eligible",
        "target_norm_worst",
        "target_norm_mean",
        "protocol",
    ]
    return out.sort_values(order, ascending=[False, False, True, True, True]).reset_index(drop=True)


def _fmt(x: object) -> str:
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def _table_md(df: pd.DataFrame, cols: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    view = df.loc[:, [c for c in cols if c in df.columns]].head(max_rows)
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in view.to_numpy()]
    return "\n".join([header, sep, *rows])


def _write_report(df: pd.DataFrame, summary: pd.DataFrame, paths: VerifierPaths) -> None:
    accepted = df[df["paper_main_accepted"]]
    eligible = df[df["clean_base_first_eligible"]]
    best_eligible = eligible.sort_values("target_norm_worst").head(10)
    best_stage2 = df[
        (~df["forbidden_source_term"])
        & df["uses_stage2_correction"].astype(bool)
        & df["is_predeclared_selected_row"].astype(bool)
    ].sort_values("target_norm_worst").head(8)
    miss_by_temp = (
        eligible.groupby("worst_temp", dropna=False)
        .size()
        .rename("n_clean_base_first_misses")
        .reset_index()
        .sort_values("n_clean_base_first_misses", ascending=False)
    )

    lines = [
        "# NMC Strict NoCC Clean Adoption Verifier",
        "",
        "## Scope",
        "",
        "This verifier only supports the current NMC strict NoCC goal.",
        "",
        "It rejects or de-prioritizes rows that are not clean base-first candidates:",
        "",
        "- Stage2/correction rows before a base model has passed.",
        "- Diagnostic or test-peek rows.",
        "- Any source family containing LFP, capacity-anchor, charge-conservation, or causal-observer evidence.",
        "",
        "Current is allowed only as instantaneous/local excitation, not integrated into SOC state.",
        "",
        "## Verdict",
        "",
        f"- Rows scanned: {len(df)}",
        f"- Clean base-first eligible rows: {len(eligible)}",
        f"- Paper-main accepted rows: {len(accepted)}",
        "",
    ]
    if len(accepted):
        lines.append("A strict NoCC main-model candidate passed the gate.")
    else:
        lines.extend(
            [
                "No strict NoCC main-model candidate passed the gate.",
                "",
                "The objective is therefore not complete yet. The current evidence supports only an ablation/failure-analysis claim.",
            ]
        )

    lines.extend(
        [
            "",
            "## Best Clean Base-First Rows",
            "",
            _table_md(
                best_eligible,
                [
                    "protocol",
                    "case",
                    "MAE_0C",
                    "MAE_25C",
                    "MAE_45C",
                    "target_norm_worst",
                    "worst_temp",
                    "decision_reason",
                ],
                max_rows=10,
            ),
            "",
            "## Why Stage2-Looking Rows Are Not Main Evidence",
            "",
            "Rows below may have better numbers, but they are not clean NoCC main evidence because Stage2/correction is used before a passing Stage1 base exists.",
            "",
            _table_md(
                best_stage2,
                [
                    "protocol",
                    "case",
                    "MAE_0C",
                    "MAE_25C",
                    "MAE_45C",
                    "target_norm_worst",
                    "worst_temp",
                    "decision_reason",
                ],
                max_rows=8,
            ),
            "",
            "## Clean Base-First Failure Distribution",
            "",
            _table_md(miss_by_temp, ["worst_temp", "n_clean_base_first_misses"], max_rows=10),
            "",
            "## Interpretation",
            "",
            "The closest eligible row is still above the target-normalized gate. The main bottleneck remains clean Stage1 profile robustness, especially around 25C and sometimes 45C depending on the feature family.",
            "",
            "A defensible next run should therefore keep the adoption rule fixed first, then improve the base representation. Another cold/hot correction is not a logical next step until a base-first candidate passes.",
            "",
            "## Generated Files",
            "",
            f"- `{paths.out_csv}`",
            f"- `{paths.out_summary}`",
            f"- `{paths.out_md}`",
        ]
    )
    (paths.base_dir / paths.out_md).write_text("\n".join(lines), encoding="utf-8")


def run(paths: VerifierPaths = VerifierPaths()) -> dict[str, pd.DataFrame]:
    frames = [_load_gate_rows(paths), _load_shift_guard(paths)]
    frames = [f for f in frames if len(f)]
    if not frames:
        raise FileNotFoundError("No verifier inputs found.")

    df = pd.concat(frames, ignore_index=True, sort=False)
    for col in ("uses_stage2_correction", "is_base_first_protocol", "is_predeclared_selected_row"):
        df[col] = df[col].fillna(False).astype(bool)
    df = _classify(df)

    summary = pd.DataFrame(
        [
            {"metric": "rows_scanned", "value": int(len(df))},
            {"metric": "clean_base_first_eligible_rows", "value": int(df["clean_base_first_eligible"].sum())},
            {"metric": "paper_main_accepted_rows", "value": int(df["paper_main_accepted"].sum())},
            {
                "metric": "best_clean_base_first_target_norm_worst",
                "value": float(
                    df.loc[df["clean_base_first_eligible"], "target_norm_worst"].min()
                    if df["clean_base_first_eligible"].any()
                    else np.nan
                ),
            },
        ]
    )

    out_csv = paths.base_dir / paths.out_csv
    out_summary = paths.base_dir / paths.out_summary
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    summary.to_csv(out_summary, index=False)
    _write_report(df, summary, paths)
    return {"rows": df, "summary": summary}


def main() -> None:
    run()


if __name__ == "__main__":
    main()
