from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = {"0.0": 1.0, "25.0": 0.7, "45.0": 0.3}
CASE_RE = re.compile(r"(?:^|_)(E\d+_[A-Z0-9]+(?:_[A-Z0-9]+)*?_to_[A-Z0-9]+)")
PROFILE_ROTATION_CASES = {"E1_DST_US06_to_FUDS", "E2_FUDS_US06_to_DST", "E3_DST_FUDS_to_US06", "E4_DST_US06_to_VALIDATION"}
INTERNAL_VALID_3TRAIN_CASES = {
    "E5_VALIDATION_DST_US06_to_FUDS",
    "E6_VALIDATION_FUDS_US06_to_DST",
    "E7_VALIDATION_DST_FUDS_to_US06",
    "E8_DST_FUDS_US06_to_VALIDATION",
}


def _read_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    protocol = path.name.split("nmc_universal_", 1)[-1]
    protocol = protocol.rsplit("_test_summary.csv", 1)[0]
    protocol = protocol.rsplit("_stage2_rule_selected", 1)[0]
    case_match = CASE_RE.search(path.name)
    case_id = case_match.group(1) if case_match else ("E1_DST_US06_to_FUDS" if "_e1_" in path.name.lower() else "unknown")
    protocol = protocol.replace(f"_{case_id}", "")
    protocol = re.sub(r"_seed[0-9,]+$", "", protocol)
    if protocol.endswith("_e1"):
        protocol = protocol[:-3]

    rows = []
    for _, row in df.iterrows():
        seed = int(row["seed"]) if "seed" in row and pd.notna(row["seed"]) else -1
        mae = {k: float(row[k]) if k in row and pd.notna(row[k]) else np.nan for k in TARGETS}
        ratios = {k: mae[k] / TARGETS[k] if pd.notna(mae[k]) else np.nan for k in TARGETS}
        rows.append(
            {
                "source_file": path.name,
                "protocol": protocol,
                "case": case_id,
                "seed": seed,
                "uses_stage2_correction": bool("stage2_rule_selected" in path.name or str(row.get("variant", "")).find("_stage2") >= 0),
                "is_base_first_protocol": protocol.startswith("base_"),
                "is_predeclared_selected_row": bool(
                    "stage2_rule_selected" in path.name
                    or "selected_seed" in str(row.get("variant", ""))
                ),
                "MAE_0C": mae["0.0"],
                "MAE_25C": mae["25.0"],
                "MAE_45C": mae["45.0"],
                "target_ratio_0C": ratios["0.0"],
                "target_ratio_25C": ratios["25.0"],
                "target_ratio_45C": ratios["45.0"],
                "target_norm_mean": float(np.nanmean(list(ratios.values()))),
                "target_norm_worst": float(np.nanmax(list(ratios.values()))),
                "worst_temp": max(ratios, key=lambda k: -np.inf if pd.isna(ratios[k]) else ratios[k]).replace(".0", "C"),
                "target_met": bool(
                    mae["0.0"] < TARGETS["0.0"]
                    and mae["25.0"] < TARGETS["25.0"]
                    and mae["45.0"] < TARGETS["45.0"]
                ),
            }
        )
    return pd.DataFrame(rows)


def collect_rows(result_dir: Path) -> pd.DataFrame:
    paths = sorted(result_dir.glob("nmc_universal_*_stage2_rule_selected_test_summary.csv"))
    no_stage2_paths = sorted(result_dir.glob("nmc_universal_*_test_summary.csv"))
    stage2_roots = {p.name.replace("_stage2_rule_selected_test_summary.csv", "") for p in paths}
    for path in no_stage2_paths:
        root = path.name.replace("_test_summary.csv", "")
        if root not in stage2_roots:
            paths.append(path)
    frames = [_read_summary(path) for path in paths]
    frames = [df for df in frames if not df.empty]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(
            [
                {
                    "protocol": "missing",
                    "decision": "no_results_found",
                    "notes": "No nmc_universal_* summary files were found.",
                }
            ]
        )
    summary_rows = []
    for protocol, g in rows.groupby("protocol"):
        decision_g = g[g["is_predeclared_selected_row"].astype(bool)].copy()
        if decision_g.empty:
            decision_g = g.copy()
        cases = sorted(str(x) for x in g["case"].dropna().unique())
        seeds = sorted(int(x) for x in g["seed"].dropna().unique())
        is_base = bool(g["is_base_first_protocol"].any())
        selected_cases = sorted(str(x) for x in decision_g["case"].dropna().unique())
        selected_case_set = set(selected_cases)
        all_cases_available = len(selected_case_set & PROFILE_ROTATION_CASES) == 4
        all_internal_valid_cases_available = len(selected_case_set & INTERNAL_VALID_3TRAIN_CASES) == 4
        e1 = decision_g[decision_g["case"].eq("E1_DST_US06_to_FUDS")]
        e5 = decision_g[decision_g["case"].eq("E5_VALIDATION_DST_US06_to_FUDS")]
        mean_25 = float(decision_g["MAE_25C"].mean())
        worst_25 = float(decision_g["MAE_25C"].max())
        worst_target = float(decision_g["target_norm_worst"].max())
        pass_count = int(decision_g["target_met"].sum())
        total_rows = int(len(decision_g))
        diagnostic_best_25c = float(g["MAE_25C"].min())

        if is_base:
            if all_internal_valid_cases_available:
                if worst_25 < 0.8 and worst_target < 1.15:
                    decision = "promote_internalvalid3_to_3seed"
                    notes = "3-profile train/internal-validation profile holdout passes seed0; promote to 3 seeds before correction."
                else:
                    decision = "reject_internalvalid3_rotation"
                    notes = "3-profile train/internal-validation holdout rotation does not pass the profile-robust gate."
            elif all_cases_available:
                if worst_25 < 0.8:
                    decision = "promote_to_corrected_3seed"
                    notes = "Base-only profile-rotation 25C worst is below 0.8; correction may be tested after base-first gate."
                else:
                    decision = "reject_base_first"
                    notes = "Base-only profile-rotation 25C worst is not below 0.8; do not add correction as a main claim."
            elif not e5.empty and float(e5["MAE_25C"].mean()) < 0.7:
                decision = "run_internalvalid3_rotation"
                notes = "3-profile train/internal-validation FUDS holdout passes 25C; run E6-E8 before adding correction."
            elif not e5.empty:
                decision = "reject_internalvalid3_e5"
                notes = "3-profile train/internal-validation FUDS holdout 25C does not pass 0.7; do not promote."
            elif not e1.empty and float(e1["MAE_25C"].mean()) < 0.7:
                decision = "run_full_base_rotation"
                notes = "Base-only E1 passes 25C; run E2-E4 before adding correction."
            elif not e1.empty:
                decision = "reject_base_first_e1"
                notes = "Base-only E1 25C does not pass 0.7; do not run correction as a main-model candidate."
            else:
                decision = "needs_base_e1_screening"
                notes = "Base-only E1 evidence is missing; run E1 before correction or profile rotation."
        else:
            if all_cases_available and pass_count == total_rows:
                decision = "candidate_main_after_3seed_confirmation"
                notes = "All available profile-rotation rows pass; still require base-first evidence and 3 seeds."
            elif pass_count == total_rows and total_rows >= 3:
                decision = "passes_e1_only_needs_rotation"
                notes = "E1 passes, but profile rotation is not complete."
            else:
                decision = "reject_or_diagnostic_only"
                notes = "Clean protocol does not pass all available cases/seeds."

        summary_rows.append(
            {
                "protocol": protocol,
                "is_base_first_protocol": is_base,
                "uses_stage2_correction": bool(g["uses_stage2_correction"].any()),
                "cases": ",".join(cases),
                "seeds": ",".join(str(s) for s in seeds),
                "n_rows": int(len(g)),
                "n_selected_rows_for_decision": total_rows,
                "pass_count": pass_count,
                "MAE_0C_mean": float(decision_g["MAE_0C"].mean()),
                "MAE_25C_mean": mean_25,
                "MAE_45C_mean": float(decision_g["MAE_45C"].mean()),
                "MAE_25C_worst": worst_25,
                "MAE_25C_diagnostic_best": diagnostic_best_25c,
                "target_norm_worst": worst_target,
                "all_four_rotation_cases_available": all_cases_available,
                "all_internal_valid_cases_available": all_internal_valid_cases_available,
                "decision": decision,
                "notes": notes,
            }
        )
    return pd.DataFrame(summary_rows).sort_values(["decision", "MAE_25C_mean", "protocol"])


def write_report(out_dir: Path, rows: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Universal Protocol Gate Report",
        "",
        "This gate exists to avoid the suspected shortcut: choosing a 25C-friendly base checkpoint and then using cold/hot correction for the other temperatures.",
        "",
        "Base-first rule:",
        "",
        "- Run base-only protocols before corrected protocols.",
        "- If diagnostic rows include many epochs, use the predeclared selected row for decisions, not the best test epoch.",
        "- If only E1 is available, predeclared base-only 25C must be below 0.7 before running full rotation.",
        "- If E1-E4 are available, predeclared base-only 25C worst must be below 0.8 before adding correction as a main-model candidate.",
        "- If base-only E1 25C is not below 0.7, reject the base-first candidate immediately instead of adding Stage 2 correction.",
        "- Corrected results without base-first evidence remain diagnostic only.",
        "",
        "## Protocol Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".3f") if not summary.empty else "No summary rows.",
        "",
        "## Row-Level Evidence",
        "",
        rows.to_markdown(index=False, floatfmt=".3f") if not rows.empty else "No result rows found.",
        "",
        "## Next Commands",
        "",
        "Run base-first screening before corrected variants:",
        "",
        "```bash",
        "RUN=1 PROTOCOL=base_fixed_last5 SEEDS=0 CASES=E1,E2,E3,E4 bash /home/user/바탕화면/DL/LSTM_STATELESS_DECOMP_SOC/run_nmc_universal_profile_protocol_remote.sh",
        "RUN=1 PROTOCOL=base_mmd20_profile_balanced SEEDS=0 CASES=E1,E2,E3,E4 bash /home/user/바탕화면/DL/LSTM_STATELESS_DECOMP_SOC/run_nmc_universal_profile_protocol_remote.sh",
        "RUN=1 PROTOCOL=base_supcon_profile_balanced SUPCON_LAMBDA=0.05 SEEDS=0 CASES=E1,E2,E3,E4 bash /home/user/바탕화면/DL/LSTM_STATELESS_DECOMP_SOC/run_nmc_universal_profile_protocol_remote.sh",
        "RUN=1 PROTOCOL=base_currcons_profile_balanced SEEDS=0 CASES=E1 bash /home/user/바탕화면/DL/LSTM_STATELESS_DECOMP_SOC/run_nmc_universal_profile_protocol_remote.sh",
        "```",
    ]
    (out_dir / "universal_protocol_gate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(result_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(result_dir)
    summary = summarize(rows)
    rows.to_csv(out_dir / "universal_protocol_gate_rows.csv", index=False)
    summary.to_csv(out_dir / "universal_protocol_gate_summary.csv", index=False)
    write_report(out_dir, rows, summary)
    print(f"Wrote {out_dir / 'universal_protocol_gate_summary.csv'}")
    print(f"Wrote {out_dir / 'universal_protocol_gate_report.md'}")
    if not summary.empty:
        print(summary[["protocol", "MAE_25C_mean", "MAE_25C_worst", "pass_count", "decision"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply base-first universal SOC adoption gates to nmc_universal results.")
    parser.add_argument("--result-dir", default="remote_result_summaries")
    parser.add_argument("--out-dir", default="paper_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    run(result_dir, out_dir)


if __name__ == "__main__":
    main()
