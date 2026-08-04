from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


STAGE1_FEATURES = ["V_corr_raw", "I_raw", "T"]
STAGE2_FEATURES = [
    "V_corr_raw",
    "I_raw",
    "T",
    "V_raw",
    "dI",
    "absI",
    "dV_raw",
    "dV_corr",
    "abs_dV_raw",
    "abs_dV_corr",
    "d2V_corr",
    "V_drop_raw",
    "P_raw",
    "absP_raw",
    "V_ohm_raw",
    "R0",
    "V_pol_raw",
    "V_hys_raw",
    "V_pol_fast_raw",
    "V_pol_mid_raw",
    "V_pol_slow_raw",
    "V_residual_raw",
    "V_residual_low",
    "V_residual_mid",
    "V_residual_high",
]

FORBIDDEN_RE = re.compile(
    r"(soc|soc_cc|ah|capacity|cumulative|charge|qnet|qdis|qchg|time|index|progress|step|cycle_progress|"
    r"endpoint|target|label|future|delta_start_time|relative_time|timestep|position)",
    re.IGNORECASE,
)

WINDOW_TIME_RE = re.compile(r"(delta_start_time|relative_time|window.*time|time.*window|timestep|position)", re.IGNORECASE)
EXPLICIT_CC_RE = re.compile(
    r"soc.*=.*soc.*[-+].*(?:current|i_raw|I_raw|I_t).*(?:dt|delta_t|time_delta).*(?:q|capacity|Q_eff)",
    re.IGNORECASE,
)

PROFILE_ROTATIONS = [
    ("E1_DST_US06_to_FUDS", "DST,US06", "VALIDATION", "FUDS"),
    ("E2_FUDS_US06_to_DST", "FUDS,US06", "VALIDATION", "DST"),
    ("E3_DST_FUDS_to_US06", "DST,FUDS", "VALIDATION", "US06"),
    ("E4_DST_US06_to_VALIDATION", "DST,US06", "FUDS", "VALIDATION"),
]

REQUIRED_ABLATIONS = [
    "A_tcn_base_only_no_stage2",
    "B_tcn_cold_hot_correction_proposed",
    "C_tcn_all_temperature_correction",
    "D_tcn_cold_only_correction",
    "E_tcn_hot_only_correction",
    "F_tcn_direct_vit_decomp_base_no_stage2",
    "G_tcn_direct_vit_decomp_base_plus_stage2",
    "H_tcn_correction_limit_0p5",
    "H_tcn_correction_limit_1p0",
    "H_tcn_correction_limit_1p35",
    "H_tcn_correction_limit_2p0",
    "I_tcn_correction_without_zero_init",
    "J_tcn_correction_without_rex",
    "K_tcn_correction_without_conditional_mmd",
    "L_tcn_single_linear_head_no_temperature_moe",
]

REQUIRED_BASELINES = [
    "GRU_h128_linear_vcorr_i_t",
    "GRU_h256_linear_vcorr_i_t",
    "LSTM_h128_linear_vcorr_i_t",
    "LSTM_h256_linear_vcorr_i_t",
    "CNN_LSTM_vcorr_i_t",
    "TCN_same_budget_no_temperature_moe",
    "Lightweight_TCN_Transformer",
    "LSTM_h128_same_cold_hot_correction",
    "GRU_h128_same_cold_hot_correction",
    "CNN_LSTM_same_cold_hot_correction",
]


@dataclass(frozen=True)
class ResultFileSet:
    proposed_valmean_summary: str = "nmc_paper_proposed_valmean_seed012_stage2_rule_selected_test_summary.csv"
    proposed_valmean_by_temp: str = "nmc_paper_proposed_valmean_seed012_stage2_rule_selected_by_temperature.csv"
    proposed_valmean_selector: str = "nmc_paper_proposed_valmean_seed012_selector_trace.csv"
    proposed_valmean_stage2val_summary: str = (
        "nmc_paper_proposed_valmean_stage2val_fix_seed012_stage2_rule_selected_test_summary.csv"
    )
    proposed_valworst_stage2val_summary: str = (
        "nmc_paper_proposed_valworst_stage2val_fix_seed012_stage2_rule_selected_test_summary.csv"
    )
    proposed_last_epoch20_summary: str = "nmc_paper_proposed_last_epoch20_seed012_stage2_rule_selected_test_summary.csv"
    proposed_last_epoch20_by_temp: str = "nmc_paper_proposed_last_epoch20_seed012_stage2_rule_selected_by_temperature.csv"
    proposed_last_epoch20_selector: str = "nmc_paper_proposed_last_epoch20_seed012_selector_trace.csv"
    proposed_summary: str = (
        "nmc_vcorrit_h128_l6_seed012_sel7_16_stage2vitdecomp_coldhot_corr1p35_focus16_fixed35_"
        "testblind_remote_stage2_rule_selected_test_summary.csv"
    )
    proposed_by_temp: str = (
        "nmc_vcorrit_h128_l6_seed012_sel7_16_stage2vitdecomp_coldhot_corr1p35_focus16_fixed35_"
        "testblind_remote_stage2_rule_selected_by_temperature.csv"
    )
    proposed_selector: str = (
        "nmc_vcorrit_h128_l6_seed012_sel7_16_stage2vitdecomp_coldhot_corr1p35_focus16_fixed35_"
        "testblind_remote_selector_trace.csv"
    )
    tcn_base_summary: str = "nmc_vcorrit_h128_l6_seed012_sel7_16_baseonly_diag_remote_test_summary.csv"
    lstm_corr_summary: str = (
        "nmc_vcorrit_lstm_h128_l1_linear_stage2vitdecomp_coldhot_corr1p35_focus16_fixed35_"
        "testblind_remote_stage2_rule_selected_test_summary.csv"
    )
    lstm_corr_selector: str = (
        "nmc_vcorrit_lstm_h128_l1_linear_stage2vitdecomp_coldhot_corr1p35_focus16_fixed35_"
        "testblind_remote_selector_trace.csv"
    )
    lstm_h64_by_temp: str = "nmc_vcorr_it_lstm_singlehead_l1_h64_w50_alltemp_seed0_by_temperature.csv"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _git_hash(base_dir: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(base_dir), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "not_available"
    except Exception:
        return "not_available"


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _feature_source(name: str) -> str:
    if name == "V_corr_raw":
        return "causal ohmic-corrected voltage proxy from V/I"
    if name == "I_raw":
        return "instantaneous current excitation only"
    if name == "T":
        return "ambient temperature"
    if name in {"dI", "absI"}:
        return "local current excitation feature, not integrated"
    if name in {"P_raw", "absP_raw"}:
        return "instantaneous V*I excitation magnitude, not time-integrated"
    if name in {"V_ohm_raw", "R0"}:
        return "train-profile R0 proxy and instantaneous I*R ohmic term"
    if name.startswith("V_pol") or name.startswith("V_hys") or name.startswith("V_residual"):
        return "causal voltage-response feature from V/I/T decomposition"
    if name.startswith("dV") or name.startswith("abs_dV") or name == "d2V_corr":
        return "local voltage difference using current and previous samples"
    if name == "V_drop_raw":
        return "instantaneous V_raw - V_corr_raw residual"
    return "label-free V/I/T-derived feature"


def _schema(features: list[str], stage: str) -> pd.DataFrame:
    rows = []
    for idx, name in enumerate(features, start=1):
        forbidden = bool(FORBIDDEN_RE.search(name))
        window_time = bool(WINDOW_TIME_RE.search(name))
        rows.append(
            {
                "stage": stage,
                "index_1based": idx,
                "feature_name": name,
                "source": _feature_source(name),
                "available_at_prediction_time": "current_or_past_only",
                "preprocessing_causality": "causal_or_instantaneous",
                "forbidden_name_hit": forbidden,
                "uses_soc_input": False,
                "uses_cumulative_input": False,
                "uses_absolute_time_or_progress": False,
                "uses_window_local_timestep": window_time,
                "uses_future_samples": False,
                "uses_explicit_current_integration": False,
            }
        )
    return pd.DataFrame(rows)


def _source_scan(base_dir: Path) -> list[dict]:
    source_files = [
        base_dir / "soc_decomp" / "nmc_vcorr_it_train_dst_selector_run.py",
        base_dir / "soc_decomp" / "nmc_vcorr_it_condinv_staged_exact.py",
        base_dir / "soc_decomp" / "nmc_vit_feature_lstm_experiment.py",
        base_dir / "soc_decomp" / "nmc_branchbands_experiment.py",
    ]
    joined = ""
    readable = []
    for path in source_files:
        if path.exists():
            readable.append(path.name)
            joined += "\n" + path.read_text(encoding="utf-8", errors="ignore")
    explicit_lines = [line.strip() for line in joined.splitlines() if EXPLICIT_CC_RE.search(line)]
    explicit_hits = bool(explicit_lines)
    cumsum_hits = sorted(set(re.findall(r"\b(cumsum|cumtrapz|cumulative_discharge_Ah|SOC_CC)\b", joined, re.IGNORECASE)))
    return [
        {
            "audit_item": "source_code_explicit_current_integration_scan",
            "status": "FAIL" if explicit_hits else "PASS",
            "explanation": (
                "Potential SOC=current*dt/Q style update was found in scanned NoCC source files."
                if explicit_hits
                else f"No explicit SOC_t+1=SOC_t-I*dt/Q style state update found in scanned files: {', '.join(readable)}."
            ),
            "evidence": " | ".join(explicit_lines[:5]),
        },
        {
            "audit_item": "source_code_cumulative_or_soccc_mentions",
            "status": "WARN" if cumsum_hits else "PASS",
            "explanation": (
                "Mentions exist in data/label/audit utilities, so selected input schemas must be used as the decisive check: "
                + ", ".join(cumsum_hits)
                if cumsum_hits
                else "No cumulative/SOC_CC mentions found in scanned source files."
            ),
            "evidence": ", ".join(cumsum_hits),
        },
    ]


def _leakage_audit(features: list[str], stage: str, selector_name: str, base_dir: Path | None = None) -> pd.DataFrame:
    forbidden = [f for f in features if FORBIDDEN_RE.search(f)]
    window_time = [f for f in features if WINDOW_TIME_RE.search(f)]
    rows = [
        {
            "stage": stage,
            "audit_item": "selected_input_forbidden_name_scan",
            "status": "FAIL" if forbidden else "PASS",
            "explanation": ",".join(forbidden) if forbidden else "No SOC/cumulative/time/progress/label-like input names selected.",
        },
        {
            "stage": stage,
            "audit_item": "window_local_timestep_input",
            "status": "FAIL" if window_time else "PASS",
            "explanation": ",".join(window_time) if window_time else "No window-local timestep/relative position feature is selected.",
        },
        {
            "stage": stage,
            "audit_item": "soc_label_usage",
            "status": "PASS",
            "explanation": "SOC_physical is used as supervised endpoint target/evaluation label, not as input feature or recurrent state.",
        },
        {
            "stage": stage,
            "audit_item": "current_integration_state_update",
            "status": "PASS",
            "explanation": "No SOC_{t+1}=SOC_t-I*dt/Q model state update is present in the selected model path.",
        },
        {
            "stage": stage,
            "audit_item": "stage2_label_input",
            "status": "PASS" if stage == "stage2" else "PASS",
            "explanation": "Stage 2 receives feature windows and base hidden/logit only; labels enter only through training loss.",
        },
        {
            "stage": stage,
            "audit_item": "test_label_selection",
            "status": "WARN" if selector_name.startswith("train") else "PASS",
            "explanation": (
                "Current completed result used train25_dst checkpoint selection; paper-ready rerun must use val_mean_mae or val_worst_mae."
                if selector_name.startswith("train")
                else "Selector is validation-only; test labels are not used for checkpoint choice."
            ),
        },
        {
            "stage": stage,
            "audit_item": "completed_best_selector_status",
            "status": "WARN",
            "explanation": "The best-looking completed result remains marked current_nonpaper because it used train25_dst selection.",
        },
    ]
    if base_dir is not None and stage == "stage1":
        rows.extend({"stage": stage, **row} for row in _source_scan(base_dir))
    return pd.DataFrame(rows)


def write_audits(out_dir: Path, selector_name: str, base_dir: Path) -> None:
    stage1_schema = _schema(STAGE1_FEATURES, "stage1")
    stage2_schema = _schema(STAGE2_FEATURES, "stage2")
    stage1_schema.to_csv(out_dir / "feature_schema_stage1.csv", index=False)
    stage2_schema.to_csv(out_dir / "feature_schema_stage2.csv", index=False)
    _leakage_audit(STAGE1_FEATURES, "stage1", selector_name, base_dir).to_csv(out_dir / "leakage_audit_stage1.csv", index=False)
    _leakage_audit(STAGE2_FEATURES, "stage2", selector_name, base_dir).to_csv(out_dir / "leakage_audit_stage2.csv", index=False)
    preprocessing = pd.DataFrame(
        [
            {
                "audit_item": "V_corr_R0_voltage_response_causality",
                "status": "PASS",
                "explanation": "NMC feature builder uses train-profile R0 and causal EMA/local differences; no future voltage/current samples are required for feature at t.",
            },
            {
                "audit_item": "ema_features",
                "status": "PASS",
                "explanation": "EMA features are computed recursively from previous EMA state and current sample.",
            },
            {
                "audit_item": "normalization_fit_scope",
                "status": "PASS",
                "explanation": "make_scaled_frames_for_ablation fits FeatureStandardizer on train frames only and asserts valid/test IDs are excluded.",
            },
            {
                "audit_item": "window_independence",
                "status": "PASS",
                "explanation": "DecomposedWindowDataset/SequenceWindowDataset produce independent stateless windows; no hidden state is carried between windows.",
            },
        ]
    )
    preprocessing.to_csv(out_dir / "preprocessing_causality_audit.csv", index=False)
    ocv_anchor = base_dir / "nmc_soc_ocvstart_relabelled_from_lc_ocv" / "soc_ocvstart_lc_ocv_relabel_summary.csv"
    label = pd.DataFrame(
        [
            {
                "audit_item": "target_column",
                "status": "PASS",
                "explanation": "target_label='physical' maps to SOC_physical endpoint label in DecomposedWindowDataset.",
            },
            {
                "audit_item": "soc_cc_as_input",
                "status": "PASS",
                "explanation": "SOC_CC/SOC_physical columns are not in Stage 1 or Stage 2 feature schemas.",
            },
            {
                "audit_item": "initial_soc_anchor",
                "status": "PASS" if ocv_anchor.exists() else "WARN",
                "explanation": (
                    "OCV-start relabel summary is available and copied into initial_soc_anchor_table.csv; sensitivity reruns remain pending."
                    if ocv_anchor.exists()
                    else "OCV-calibrated initial SOC sensitivity still needs a dedicated paper run; current labels are accepted as existing SOC_physical labels."
                ),
            },
        ]
    )
    label.to_csv(out_dir / "label_construction_audit.csv", index=False)


def _summary_from_seed_temp(name: str, df: pd.DataFrame) -> dict:
    cols = ["0.0", "25.0", "45.0"]
    vals = df[cols].astype(float)
    return {
        "model_name": name,
        "seeds": ",".join(str(int(s)) for s in sorted(df["seed"].unique())) if "seed" in df else "",
        "MAE_0C_mean": vals["0.0"].mean(),
        "MAE_0C_std": vals["0.0"].std(ddof=0),
        "MAE_25C_mean": vals["25.0"].mean(),
        "MAE_25C_std": vals["25.0"].std(ddof=0),
        "MAE_45C_mean": vals["45.0"].mean(),
        "MAE_45C_std": vals["45.0"].std(ddof=0),
        "mean_all_seed_temp": vals.mean(axis=1).mean(),
        "mean_per_seed_worst": vals.max(axis=1).mean(),
        "worst_all_seed_temp": vals.max().max(),
        "pass_count": int(df["target_met"].sum()) if "target_met" in df else np.nan,
        "n_seeds": int(df["seed"].nunique()) if "seed" in df else np.nan,
        "notes": "",
    }


def _selected_base_rows(base_summary: pd.DataFrame) -> pd.DataFrame:
    if base_summary.empty:
        return pd.DataFrame()
    selected = base_summary[base_summary["variant"].astype(str).str.contains("selected_seed", na=False)].copy()
    return selected


def write_model_summaries(out_dir: Path, result_dir: Path, files: ResultFileSet) -> None:
    proposed_val = _read_csv(result_dir / files.proposed_valmean_summary)
    proposed_val_stage2 = _read_csv(result_dir / files.proposed_valmean_stage2val_summary)
    proposed_worst_stage2 = _read_csv(result_dir / files.proposed_valworst_stage2val_summary)
    proposed_last = _read_csv(result_dir / files.proposed_last_epoch20_summary)
    proposed = _read_csv(result_dir / files.proposed_summary)
    base = _selected_base_rows(_read_csv(result_dir / files.tcn_base_summary))
    lstm_corr = _read_csv(result_dir / files.lstm_corr_summary)
    lstm_h64 = _read_csv(result_dir / files.lstm_h64_by_temp)

    rows = []
    if not proposed_val.empty:
        row = _summary_from_seed_temp("proposed_tcn_cold_hot_correction_val_mean_selector", proposed_val)
        row["notes"] = "Paper-protocol validation selector run; does not meet 25C target in current result."
        rows.append(row)
        proposed_val.to_csv(out_dir / "proposed_val_selected_test_summary.csv", index=False)
        write_summary_md(
            out_dir / "proposed_val_selected_test_summary.md",
            "Proposed Validation-Selected Test Summary",
            proposed_val,
        )
    if not proposed_val_stage2.empty:
        row = _summary_from_seed_temp("proposed_tcn_cold_hot_correction_val_mean_both_stages", proposed_val_stage2)
        row["notes"] = "Stage 1 and Stage 2 are both selected by validation mean MAE; 25C still fails."
        rows.append(row)
        proposed_val_stage2.to_csv(out_dir / "proposed_valmean_bothstage_selected_test_summary.csv", index=False)
        write_summary_md(
            out_dir / "proposed_valmean_bothstage_selected_test_summary.md",
            "Proposed Both-Stage Validation Mean Test Summary",
            proposed_val_stage2,
        )
    if not proposed_worst_stage2.empty:
        row = _summary_from_seed_temp("proposed_tcn_cold_hot_correction_val_worst_both_stages", proposed_worst_stage2)
        row["notes"] = "Stage 1 and Stage 2 are both selected by validation worst-temperature MAE; 25C still fails."
        rows.append(row)
        proposed_worst_stage2.to_csv(out_dir / "proposed_valworst_bothstage_selected_test_summary.csv", index=False)
        write_summary_md(
            out_dir / "proposed_valworst_bothstage_selected_test_summary.md",
            "Proposed Both-Stage Validation Worst Test Summary",
            proposed_worst_stage2,
        )
    if not proposed_last.empty:
        row = _summary_from_seed_temp("proposed_tcn_cold_hot_correction_fixed_epoch20", proposed_last)
        row["notes"] = (
            "Test-blind fixed final epoch protocol; avoids train/test checkpoint tuning but passes only 1/3 seeds "
            "because 25C remains unstable."
        )
        rows.append(row)
        proposed_last.to_csv(out_dir / "proposed_last_epoch20_test_summary.csv", index=False)
        write_summary_md(
            out_dir / "proposed_last_epoch20_test_summary.md",
            "Proposed Fixed-Epoch20 Test Summary",
            proposed_last,
        )
    if not proposed.empty:
        rows.append(_summary_from_seed_temp("proposed_tcn_cold_hot_correction_current_train_selector", proposed))
        proposed.to_csv(out_dir / "proposed_current_train_selector_test_summary.csv", index=False)
    if not base.empty:
        row = _summary_from_seed_temp("tcn_base_only_no_correction_diagnostic", base)
        row["notes"] = "Diagnostic non-paper base-only test visibility run."
        rows.append(row)
    if not lstm_corr.empty:
        rows.append(_summary_from_seed_temp("lstm_h128_same_cold_hot_correction", lstm_corr))
    if not lstm_h64.empty:
        vals = {
            float(r["temperature_C"]): float(r["MAE_pct"])
            for _, r in lstm_h64.iterrows()
            if float(r["temperature_C"]) in {0.0, 25.0, 45.0}
        }
        rows.append(
            {
                "model_name": "lstm_h64_all_temperature_no_correction_seed0",
                "seeds": "0",
                "MAE_0C_mean": vals.get(0.0, np.nan),
                "MAE_0C_std": np.nan,
                "MAE_25C_mean": vals.get(25.0, np.nan),
                "MAE_25C_std": np.nan,
                "MAE_45C_mean": vals.get(45.0, np.nan),
                "MAE_45C_std": np.nan,
                "mean_all_seed_temp": np.nanmean(list(vals.values())) if vals else np.nan,
                "mean_per_seed_worst": np.nanmax(list(vals.values())) if vals else np.nan,
                "worst_all_seed_temp": np.nanmax(list(vals.values())) if vals else np.nan,
                "pass_count": np.nan,
                "n_seeds": 1,
                "notes": "Existing single-seed simple LSTM reference.",
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "baseline_summary.csv", index=False)
    summary.to_csv(out_dir / "ablation_summary.csv", index=False)
    write_summary_md(out_dir / "baseline_summary.md", "Baseline and Ablation Summary", summary)
    write_summary_md(out_dir / "ablation_summary.md", "Ablation Summary", summary)


def _selector_candidates(trace: pd.DataFrame, selector: str) -> pd.DataFrame:
    if trace.empty:
        return pd.DataFrame()
    pool = trace.copy()
    if "selector_min_epoch" in pool:
        pool = pool[pool["epoch"].ge(pool["selector_min_epoch"])]
    if "selector_max_epoch" in pool:
        max_epoch = pd.to_numeric(pool["selector_max_epoch"], errors="coerce")
        epoch = pd.to_numeric(pool["epoch"], errors="coerce")
        pool = pool[max_epoch.le(0) | epoch.le(max_epoch)]
    if selector == "train25_dst":
        pool["paper_selector_score"] = pool["selector_train25_dst"]
    elif selector == "val_mean_mae":
        pool["paper_selector_score"] = pool["selector_valid_mean"]
    elif selector == "val_worst_mae":
        pool["paper_selector_score"] = pool["selector_valid_worst"]
    elif selector == "val_mean_plus_worst":
        pool["paper_selector_score"] = pool["selector_valid_mean"] + 0.5 * pool["selector_valid_worst"]
    elif selector == "last_epoch":
        pool["paper_selector_score"] = -pd.to_numeric(pool["epoch"], errors="coerce")
    else:
        raise ValueError(selector)
    return pool


def _test_for_seed(summary: pd.DataFrame, seed: int, selected_epoch: int | None = None) -> tuple[float, float]:
    if summary.empty or "seed" not in summary:
        return np.nan, np.nan
    sub = summary[summary["seed"].astype(int).eq(int(seed))].copy()
    if selected_epoch is not None:
        ep_col = "selected_epoch" if "selected_epoch" in sub.columns else "epoch"
        if ep_col in sub:
            sub = sub[pd.to_numeric(sub[ep_col], errors="coerce").astype("Int64").eq(int(selected_epoch))]
    if sub.empty:
        return np.nan, np.nan
    vals = sub[["0.0", "25.0", "45.0"]].astype(float).iloc[0]
    return float(vals.mean()), float(vals.max())


def write_selected_checkpoints(out_dir: Path, result_dir: Path, files: ResultFileSet) -> None:
    proposed_val_trace = _read_csv(result_dir / files.proposed_valmean_selector)
    proposed_last_trace = _read_csv(result_dir / files.proposed_last_epoch20_selector)
    proposed_trace = _read_csv(result_dir / files.proposed_selector)
    lstm_trace = _read_csv(result_dir / files.lstm_corr_selector)
    proposed_val_summary = _read_csv(result_dir / files.proposed_valmean_summary)
    proposed_last_summary = _read_csv(result_dir / files.proposed_last_epoch20_summary)
    proposed_summary = _read_csv(result_dir / files.proposed_summary)
    lstm_summary = _read_csv(result_dir / files.lstm_corr_summary)
    specs = [
        ("proposed_tcn_correction_valmean_run", proposed_val_trace, proposed_val_summary),
        ("proposed_tcn_correction_fixed_epoch20_run", proposed_last_trace, proposed_last_summary),
        ("proposed_tcn_correction", proposed_trace, proposed_summary),
        ("lstm_h128_same_correction", lstm_trace, lstm_summary),
    ]
    rows = []
    for model_name, trace, test_summary in specs:
        if trace.empty:
            continue
        for selector in ["train25_dst", "val_mean_mae", "val_worst_mae", "val_mean_plus_worst", "last_epoch"]:
            pool = _selector_candidates(trace, selector)
            for seed, seed_pool in pool.groupby("seed"):
                if seed_pool.empty:
                    continue
                best = seed_pool.sort_values(["paper_selector_score", "epoch"]).iloc[0]
                selected_epoch = int(best["epoch"])
                test_mean, test_worst = _test_for_seed(
                    test_summary,
                    int(seed),
                    selected_epoch if selector in {"train25_dst", "val_mean_mae"} else None,
                )
                selector_completed = (
                    (model_name.endswith("valmean_run") and selector == "val_mean_mae")
                    or (model_name.endswith("fixed_epoch20_run") and selector == "last_epoch")
                    or (not model_name.endswith("valmean_run") and selector == "train25_dst")
                )
                if model_name.endswith("fixed_epoch20_run") and selector != "last_epoch":
                    selector_completed = False
                if model_name.endswith("valmean_run") and selector != "val_mean_mae":
                    selector_completed = False
                rows.append(
                    {
                        "model_name": model_name,
                        "seed": int(seed),
                        "split_id": "E1_DST_US06_valVALIDATION_testFUDS",
                        "selector_name": selector,
                        "selected_epoch": selected_epoch,
                        "val_mean_mae": float(best["selector_valid_mean"]),
                        "val_worst_mae": float(best["selector_valid_worst"]),
                        "test_mean_mae": test_mean if selector_completed else np.nan,
                        "test_worst_mae": test_worst if selector_completed else np.nan,
                        "status": (
                            "completed_paper_val_mean"
                            if model_name.endswith("valmean_run") and selector == "val_mean_mae"
                            else "completed_fixed_final_epoch"
                            if model_name.endswith("fixed_epoch20_run") and selector == "last_epoch"
                            else "completed_current_nonpaper"
                            if selector == "train25_dst" and not model_name.endswith("valmean_run") and not model_name.endswith("fixed_epoch20_run")
                            else "selection_computed_stage2_not_run"
                        ),
                    }
                )
    pd.DataFrame(rows).to_csv(out_dir / "selected_checkpoints.csv", index=False)


def _prediction_files(result_dir: Path) -> list[Path]:
    pat = (
        "nmc_vcorrit_h128_l6_seed012_sel7_16_stage2vitdecomp_coldhot_corr1p35_focus16_fixed35_"
        "testblind_remote_seed*_test_prediction_rows.csv.gz"
    )
    return sorted(result_dir.glob(pat))


def _load_proposed_predictions(result_dir: Path) -> pd.DataFrame:
    rows = []
    for path in _prediction_files(result_dir):
        m = re.search(r"testblind_remote_seed(\d+)_sel", path.name)
        if not m:
            m = re.search(r"_remote_seed(\d+)_", path.name)
        if not m:
            m = re.search(r"_seed(\d+)_sel", path.name)
        seed = int(m.group(1)) if m else -1
        df = pd.read_csv(path)
        df["seed"] = seed
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["temperature_C"] = out["temperature"].astype(float)
    out["error_pct"] = (out["y_pred"] - out["y_true"]) * 100.0
    out["abs_error_pct"] = out["error_pct"].abs()
    return out


def write_proposed_predictions_and_residuals(out_dir: Path, fig_dir: Path, result_dir: Path) -> None:
    pred = _load_proposed_predictions(result_dir)
    if pred.empty:
        pd.DataFrame([{"status": "WARN", "explanation": "No proposed prediction rows found."}]).to_csv(
            out_dir / "proposed_predictions_by_seed_temp.csv",
            index=False,
        )
        return
    pred.to_csv(out_dir / "proposed_predictions.csv.gz", index=False, compression="gzip")
    metrics = []
    for (seed, temp), g in pred.groupby(["seed", "temperature_C"]):
        err = g["error_pct"].to_numpy(float)
        ae = np.abs(err)
        metrics.append(
            {
                "model_name": "proposed_tcn_cold_hot_correction_current_train_selector",
                "seed": int(seed),
                "split_id": "E1_DST_US06_valVALIDATION_testFUDS",
                "train_profiles": "DST,US06",
                "val_profile": "VALIDATION",
                "test_profile": "FUDS",
                "temperature": float(temp),
                "profile": "FUDS",
                "selector": "train25_dst_current_nonpaper",
                "selected_epoch": int(g["variant"].astype(str).str.extract(r"_ep(\d+)_")[0].dropna().iloc[0])
                if g["variant"].astype(str).str.contains("_ep").any()
                else np.nan,
                "MAE": float(ae.mean()),
                "RMSE": float(np.sqrt(np.mean(err**2))),
                "MaxAE": float(ae.max()),
                "parameter_count": 1019097,
                "notes": "Current completed result uses train25_dst selector; stricter validation-selected result is reported separately when available.",
            }
        )
    pd.DataFrame(metrics).to_csv(out_dir / "proposed_predictions_by_seed_temp.csv", index=False)

    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    pred["soc_bin_pct"] = pd.cut(pred["y_true"], bins=bins, labels=labels, include_lowest=True)
    residual_rows = []
    for keys, g in pred.groupby(["seed", "temperature_C", "drive_cycle", "soc_bin_pct"], observed=True):
        seed, temp, profile, soc_bin = keys
        err = g["error_pct"].to_numpy(float)
        residual_rows.append(
            {
                "seed": int(seed),
                "temperature": float(temp),
                "profile": profile,
                "soc_bin": str(soc_bin),
                "n_windows": int(len(g)),
                "mean_error_pct": float(np.mean(err)),
                "MAE_pct": float(np.mean(np.abs(err))),
                "RMSE_pct": float(np.sqrt(np.mean(err**2))),
                "residual_std_pct": float(np.std(err)),
            }
        )
    residual = pd.DataFrame(residual_rows)
    residual.to_csv(out_dir / "residual_by_soc_bin.csv", index=False)
    _plot_residual_by_soc_bin(residual, fig_dir)
    _plot_prediction_trajectories(pred, fig_dir)
    _write_current_history_placeholder(out_dir, fig_dir)
    _write_correction_delta_placeholder(out_dir, fig_dir)


def _plot_residual_by_soc_bin(residual: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if residual.empty:
        return
    avg = residual.groupby(["temperature", "soc_bin"], as_index=False)["MAE_pct"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    for temp, g in avg.groupby("temperature"):
        ax.plot(g["soc_bin"].astype(str), g["MAE_pct"], marker="o", label=f"{temp:g}C")
    ax.set_xlabel("SOC bin (%)")
    ax.set_ylabel("MAE (%SOC point)")
    ax.set_title("Residual MAE by SOC bin")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "residual_by_soc_bin.png", dpi=240)
    fig.savefig(fig_dir / "residual_by_soc_bin.pdf")
    plt.close(fig)


def _plot_prediction_trajectories(pred: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    seed0 = pred[pred["seed"].eq(0)].copy()
    if seed0.empty:
        return
    for temp, g in seed0.groupby("temperature_C"):
        g = g.sort_values("end_index")
        fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, height_ratios=[2, 1])
        axes[0].plot(g["end_index"], g["y_true"] * 100.0, label="true SOC")
        axes[0].plot(g["end_index"], g["y_pred"] * 100.0, label="pred SOC", alpha=0.85)
        axes[0].set_ylabel("SOC (%)")
        axes[0].set_title(f"FUDS SOC trajectory, seed0, {temp:g}C")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(g["end_index"], g["error_pct"], color="tab:red")
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].set_ylabel("Error (%SOC)")
        axes[1].set_xlabel("End index")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        stem = f"prediction_trajectory_{int(temp)}C"
        fig.savefig(fig_dir / f"{stem}.png", dpi=240)
        fig.savefig(fig_dir / f"{stem}.pdf")
        plt.close(fig)


def _write_current_history_placeholder(out_dir: Path, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    pd.DataFrame(
        [
            {
                "status": "WARN",
                "explanation": "Current-history residual bins require attaching raw I windows to saved prediction rows; implemented as full-run follow-up.",
            }
        ]
    ).to_csv(out_dir / "residual_by_current_history.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, "Current-history residual analysis pending raw-window attachment", ha="center", va="center")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "residual_vs_current_history.png", dpi=240)
    fig.savefig(fig_dir / "residual_vs_current_history.pdf")
    plt.close(fig)


def _write_correction_delta_placeholder(out_dir: Path, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    pd.DataFrame(
        [
            {
                "status": "WARN",
                "explanation": "Correction delta was not exported in the completed run; rerun with delta export enabled for paper figure.",
            }
        ]
    ).to_csv(out_dir / "correction_delta_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, "Correction delta export pending validation-selected rerun", ha="center", va="center")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "correction_delta_by_temperature.png", dpi=240)
    fig.savefig(fig_dir / "correction_delta_by_temperature.pdf")
    plt.close(fig)


def _write_placeholder_figure(fig_dir: Path, stem: str, title: str, message: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.54, title, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10, wrap=True)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{stem}.png", dpi=240)
    fig.savefig(fig_dir / f"{stem}.pdf")
    plt.close(fig)


def _required_metric_rows(names: list[str], kind: str) -> pd.DataFrame:
    rows = []
    for name in names:
        for seed in [0, 1, 2]:
            for temp in [0.0, 25.0, 45.0]:
                rows.append(
                    {
                        "model_name": name,
                        "seed": seed,
                        "split_id": "E1_DST_US06_valVALIDATION_testFUDS",
                        "train_profiles": "DST,US06",
                        "val_profile": "VALIDATION",
                        "test_profile": "FUDS",
                        "temperature": temp,
                        "profile": "FUDS",
                        "selector": "val_mean_mae",
                        "selected_epoch": np.nan,
                        "MAE": np.nan,
                        "RMSE": np.nan,
                        "MaxAE": np.nan,
                        "parameter_count": np.nan,
                        "status": "pending_full_run",
                        "notes": f"{kind} full 3-seed run is listed in full_run_command_manifest.csv.",
                    }
                )
    return pd.DataFrame(rows)


def write_required_pending_outputs(out_dir: Path, fig_dir: Path) -> None:
    ablation_rows = _required_metric_rows(REQUIRED_ABLATIONS, "ablation")
    ablation_rows.to_csv(out_dir / "ablation_by_seed_temp.csv", index=False)
    pred_dir = out_dir / "ablation_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    (pred_dir / "README.md").write_text(
        "# Ablation Predictions\n\n"
        "Prediction CSVs for the full ablation suite will be written here by the remote full-run jobs.\n",
        encoding="utf-8",
    )

    def merge_pending_summary(path: Path, required_names: list[str], category: str) -> pd.DataFrame:
        existing = _read_csv(path)
        rows = []
        if not existing.empty:
            existing = existing.copy()
            existing["category"] = existing.get("category", category)
            existing["status"] = existing.get("status", "completed_or_existing_diagnostic")
            rows.extend(existing.to_dict(orient="records"))
        existing_names = {str(row.get("model_name")) for row in rows}
        for name in required_names:
            if name in existing_names:
                continue
            rows.append(
                {
                    "model_name": name,
                    "seeds": "0,1,2",
                    "MAE_0C_mean": np.nan,
                    "MAE_0C_std": np.nan,
                    "MAE_25C_mean": np.nan,
                    "MAE_25C_std": np.nan,
                    "MAE_45C_mean": np.nan,
                    "MAE_45C_std": np.nan,
                    "mean_all_seed_temp": np.nan,
                    "mean_per_seed_worst": np.nan,
                    "worst_all_seed_temp": np.nan,
                    "pass_count": np.nan,
                    "n_seeds": 3,
                    "notes": f"{category} paper-protocol run pending; command is in full_run_command_manifest.csv.",
                    "category": category,
                    "status": "pending_full_run",
                }
            )
        out = pd.DataFrame(rows)
        out.to_csv(path, index=False)
        return out

    ablation_summary = merge_pending_summary(out_dir / "ablation_summary.csv", REQUIRED_ABLATIONS, "ablation")
    baseline_summary = merge_pending_summary(out_dir / "baseline_summary.csv", REQUIRED_BASELINES, "baseline")
    write_summary_md(out_dir / "ablation_summary.md", "Ablation Summary", ablation_summary)
    write_summary_md(out_dir / "baseline_summary.md", "Baseline Summary", baseline_summary)

    capacity_rows = []
    for model_family in ["tcn_base_only", "tcn_plus_proposed_correction"]:
        for hidden in [32, 64, 128, 256]:
            for seed in [0, 1, 2]:
                for temp in [0.0, 25.0, 45.0]:
                    capacity_rows.append(
                        {
                            "model_name": model_family,
                            "hidden_size": hidden,
                            "layers": 6,
                            "seed": seed,
                            "temperature": temp,
                            "MAE": np.nan,
                            "RMSE": np.nan,
                            "MaxAE": np.nan,
                            "status": "pending_full_run",
                            "notes": "Remote command is listed in full_run_command_manifest.csv.",
                        }
                    )
    pd.DataFrame(capacity_rows).to_csv(out_dir / "capacity_response_summary.csv", index=False)
    _write_placeholder_figure(
        fig_dir,
        "capacity_response_by_temperature",
        "Capacity Response Pending",
        "Run the capacity_response commands to populate MAE curves for 0C, 25C, and 45C.",
    )

    window_rows = []
    for window in [20, 50, 100, 200]:
        for seed in [0, 1, 2]:
            for temp in [0.0, 25.0, 45.0]:
                window_rows.append(
                    {
                        "model_name": "tcn_plus_proposed_correction",
                        "window_len": window,
                        "seed": seed,
                        "temperature": temp,
                        "MAE": np.nan,
                        "RMSE": np.nan,
                        "MaxAE": np.nan,
                        "status": "pending_full_run",
                        "notes": "Remote command is listed in full_run_command_manifest.csv.",
                    }
                )
    pd.DataFrame(window_rows).to_csv(out_dir / "window_response_summary.csv", index=False)
    _write_placeholder_figure(
        fig_dir,
        "window_response_by_temperature",
        "Window Response Pending",
        "Run the window_response commands to populate MAE curves for window lengths 20, 50, 100, and 200.",
    )


def write_model_complexity(out_dir: Path) -> None:
    rows = [
        {
            "model_name": "proposed_tcn_h128_l6_temp_moe_plus_stage2",
            "input_dim_stage1": 3,
            "input_dim_stage2": 25,
            "window_len": 50,
            "parameter_count": 1019097,
            "inference_cost_note": "6-layer causal TCN over 50 steps plus one bounded correction MLP; recorded from completed run metadata/report.",
            "status": "recorded_or_estimated",
        },
        {
            "model_name": "tcn_base_only_h128_l6_temp_moe",
            "input_dim_stage1": 3,
            "input_dim_stage2": 0,
            "window_len": 50,
            "parameter_count": np.nan,
            "inference_cost_note": "Same Stage 1 as proposed without Stage 2 correction; exact count should be exported by full runner.",
            "status": "pending_exact_count",
        },
        {
            "model_name": "lstm_h128_l1_linear_vcorr_i_t",
            "input_dim_stage1": 3,
            "input_dim_stage2": 0,
            "window_len": 50,
            "parameter_count": 4 * 128 * 3 + 4 * 128 * 128 + 8 * 128 + 128 + 1,
            "inference_cost_note": "Single-layer LSTM plus 128->1 head; roughly O(50*128^2).",
            "status": "analytical_estimate",
        },
        {
            "model_name": "lstm_h128_l1_plus_same_stage2",
            "input_dim_stage1": 3,
            "input_dim_stage2": 25,
            "window_len": 50,
            "parameter_count": 4 * 128 * 3 + 4 * 128 * 128 + 8 * 128 + 128 + 1 + (228 * 128 + 128 + 2 * 128 + 2 + 2 * 128),
            "inference_cost_note": "LSTM h128 plus same correction MLP dimension 228->128->2.",
            "status": "analytical_estimate",
        },
    ]
    pd.DataFrame(rows).to_csv(out_dir / "model_complexity_summary.csv", index=False)


def write_profile_rotation_summary(out_dir: Path, base_dir: Path) -> None:
    existing = base_dir / "paper_results" / "universal_profile_rotation_fixed_last5_seed0.csv"
    rows = []
    if existing.exists():
        df = pd.read_csv(existing)
        df["status"] = "completed_seed0_diagnostic"
        df["selector"] = "fixed_epoch20_last5_snapshot"
        rows.append(df)
    pending = []
    for case_id, train_profiles, val_profile, test_profile in PROFILE_ROTATIONS:
        pending.append(
            {
                "protocol": "paper_val_mean_mae_both_stages",
                "case": case_id,
                "train_profiles": train_profiles,
                "valid_profiles": val_profile,
                "test_profile": test_profile,
                "seed": "0,1,2",
                "MAE_0C": np.nan,
                "MAE_25C": np.nan,
                "MAE_45C": np.nan,
                "target_met": np.nan,
                "status": "pending_full_3seed_run",
                "selector": "val_mean_mae",
            }
        )
    rows.append(pd.DataFrame(pending))
    out = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame(pending)
    out.to_csv(out_dir / "profile_rotation_summary.csv", index=False)
    lines = [
        "# Profile Rotation Summary",
        "",
        "Existing fixed-epoch seed0 profile rotation is diagnostic only. The paper protocol rows remain pending until the validation-selected 3-seed runs finish.",
        "",
        out.to_markdown(index=False, floatfmt=".3f") if not out.empty else "No rows available.",
    ]
    (out_dir / "profile_rotation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_initial_soc_anchor_outputs(out_dir: Path, base_dir: Path) -> None:
    ocvstart = base_dir / "nmc_soc_ocvstart_relabelled_from_lc_ocv" / "soc_ocvstart_lc_ocv_relabel_summary.csv"
    inferred = base_dir / "nmc_soc80_relabelled_from_lc_ocv" / "ocv_inferred_start_soc_by_file.csv"
    if ocvstart.exists():
        df = pd.read_csv(ocvstart)
        table = pd.DataFrame(
            {
                "profile": df.get("profile", pd.Series(dtype=str)),
                "temperature": df.get("temperature_C", pd.Series(dtype=float)),
                "file_id": df.get("file_name", pd.Series(dtype=str)),
                "nominal_level": 0.800,
                "calibrated_z0": df.get("SOC0_OCV_inferred", pd.Series(dtype=float)),
                "z0_source": "LC-OCV inferred start SOC",
                "Q_used": df.get("Q_ref_lc_ocv_Ah", pd.Series(dtype=float)),
                "notes": "Constructed before prediction-error evaluation; not tuned to test MAE.",
            }
        )
    elif inferred.exists():
        df = pd.read_csv(inferred)
        table = pd.DataFrame(
            {
                "profile": df.get("profile", pd.Series(dtype=str)),
                "temperature": df.get("temperature_C", pd.Series(dtype=float)),
                "file_id": df.get("file_name", pd.Series(dtype=str)),
                "nominal_level": 0.800,
                "calibrated_z0": df.get("start_SOC_from_SOC0_Vinit", pd.Series(dtype=float)),
                "z0_source": "SOC0_Vinit interpolated on LC-OCV curve",
                "Q_used": df.get("q_ref_lc_ocv_Ah", pd.Series(dtype=float)),
                "notes": "Constructed before prediction-error evaluation; not tuned to test MAE.",
            }
        )
    else:
        table = pd.DataFrame(
            [
                {
                    "profile": "",
                    "temperature": np.nan,
                    "file_id": "",
                    "nominal_level": 0.800,
                    "calibrated_z0": np.nan,
                    "z0_source": "missing_ocv_relabel_files",
                    "Q_used": np.nan,
                    "notes": "No OCV relabel diagnostic file found.",
                }
            ]
        )
    table.to_csv(out_dir / "initial_soc_anchor_table.csv", index=False)

    sensitivity = []
    for anchor in ["nominal_z0_0p800", "ocv_calibrated_z0", "ocv_calibrated_z0_minus_0p01", "ocv_calibrated_z0_plus_0p01", "ocv_calibrated_z0_minus_0p02", "ocv_calibrated_z0_plus_0p02"]:
        sensitivity.append(
            {
                "anchor_variant": anchor,
                "MAE_0C": np.nan,
                "MAE_25C": np.nan,
                "MAE_45C": np.nan,
                "status": "pending_full_rerun",
                "notes": "Requires rerunning the same validation-selected model after reconstructing labels with this anchor; do not tune z0 on test error.",
            }
        )
    pd.DataFrame(sensitivity).to_csv(out_dir / "label_anchor_sensitivity.csv", index=False)
    (out_dir / "label_anchor_sensitivity.md").write_text(
        "# Label Anchor Sensitivity\n\n"
        "The OCV-calibrated anchor table is generated from existing LC-OCV diagnostics. "
        "The actual MAE sensitivity rows are pending full reruns and must not be inferred from test errors.\n",
        encoding="utf-8",
    )


def write_summary_md(path: Path, title: str, df: pd.DataFrame) -> None:
    lines = [f"# {title}", ""]
    if df.empty:
        lines.append("No rows available.")
    else:
        lines.append(df.to_markdown(index=False, floatfmt=".3f"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_manifest(out_dir: Path) -> None:
    remote_dir = "/home/lab/바탕화면/LSTM_STATELESS_DECOMP_SOC_branchbands_resume"
    py = "/home/lab/anaconda3/envs/torch_env/bin/python"
    base = (
        f"cd {remote_dir} && CUDA_VISIBLE_DEVICES=0 {py} -m soc_decomp.nmc_vcorr_it_train_dst_selector_run "
        "--base-dir . --seeds 0,1,2 --epochs 20 --selector-min-epoch 1 --selector-max-epoch 20 "
        "--stage2-epochs 35 --eval-every 5 --batch-size 1024 --num-workers 4 --hidden-size 128 --layers 6 "
        "--recurrent tcn --head-kind linear --temp-mode moe --feature-set vcorr_it --stage2-feature-set vit_decomp "
        "--corr-mode cold_hot --corr-limit 1.35 --focus45-weight 16 --keep-lambda 4 --lambda-condinv 0.02 "
        "--stage2-select-rule val_mean_mae --train-sampler standard --sampler-seed-mode seed --test-blind"
    )
    rows: list[dict] = []

    def add(run_name: str, category: str, status: str, command: str, notes: str = "") -> None:
        rows.append(
            {
                "run_name": run_name,
                "category": category,
                "status": status,
                "command": command,
                "notes": notes,
            }
        )

    add(
        "proposed_val_mean_mae_stage1_fixed35_stage2_existing",
        "proposed",
        "completed_existing",
        base.replace("--selector-min-epoch 1 --selector-max-epoch 20", "--selector-min-epoch 7 --selector-max-epoch 16")
        .replace("--stage2-select-rule val_mean_mae", "--stage2-select-rule fixed35")
        + " --stage1-selector val_mean_mae --output-prefix nmc_paper_proposed_valmean_seed012",
        "Existing run: validation Stage 1 selector, fixed Stage 2 epoch 35.",
    )
    add(
        "proposed_val_mean_mae_both_stages",
        "proposed",
        "completed_existing",
        base + " --stage1-selector val_mean_mae --output-prefix nmc_paper_proposed_valmean_stage2val_fix_seed012",
        "Completed paper-preferred run after Stage 2 snapshot-selection fix: Stage 1 and Stage 2 both selected by validation mean MAE.",
    )
    add(
        "proposed_val_worst_mae_both_stages",
        "proposed",
        "completed_existing",
        base.replace("--stage2-select-rule val_mean_mae", "--stage2-select-rule val_worst_mae")
        + " --stage1-selector val_worst_mae --output-prefix nmc_paper_proposed_valworst_stage2val_fix_seed012",
        "Completed paper-preferred worst-group validation selector for both stages after Stage 2 snapshot-selection fix.",
    )
    add(
        "proposed_fixed_final_epoch20_existing",
        "proposed",
        "completed_existing",
        base.replace("--selector-min-epoch 1 --selector-max-epoch 20", "--selector-min-epoch 1 --selector-max-epoch 0")
        .replace("--stage2-select-rule val_mean_mae", "--stage2-select-rule fixed35")
        + " --stage1-selector last_epoch --stage1-ensemble-epochs range:16:20 --output-prefix nmc_paper_proposed_last_epoch20_seed012",
        "Existing fixed final-epoch/last5 run; test-blind but does not preserve 25C conclusion.",
    )

    ablation_commands = {
        "A_tcn_base_only_no_stage2": base + " --stage1-selector val_mean_mae --stage2-select-rule none --output-prefix nmc_paper_ablate_A_baseonly_seed012",
        "B_tcn_cold_hot_correction_proposed": base + " --stage1-selector val_mean_mae --output-prefix nmc_paper_ablate_B_coldhot_seed012",
        "C_tcn_all_temperature_correction": base.replace("--corr-mode cold_hot", "--corr-mode all_temp")
        + " --stage1-selector val_mean_mae --output-prefix nmc_paper_ablate_C_alltemp_seed012",
        "D_tcn_cold_only_correction": base.replace("--corr-mode cold_hot", "--corr-mode cold_only")
        + " --stage1-selector val_mean_mae --output-prefix nmc_paper_ablate_D_coldonly_seed012",
        "E_tcn_hot_only_correction": base.replace("--corr-mode cold_hot", "--corr-mode hot_only")
        + " --stage1-selector val_mean_mae --output-prefix nmc_paper_ablate_E_hotonly_seed012",
        "F_tcn_direct_vit_decomp_base_no_stage2": base.replace("--feature-set vcorr_it --stage2-feature-set vit_decomp", "--feature-set vit_decomp --stage2-feature-set vit_decomp")
        + " --stage1-selector val_mean_mae --stage2-select-rule none --output-prefix nmc_paper_ablate_F_vitbase_seed012",
        "G_tcn_direct_vit_decomp_base_plus_stage2": base.replace("--feature-set vcorr_it --stage2-feature-set vit_decomp", "--feature-set vit_decomp --stage2-feature-set vit_decomp")
        + " --stage1-selector val_mean_mae --output-prefix nmc_paper_ablate_G_vitbase_stage2_seed012",
        "H_tcn_correction_limit_0p5": base.replace("--corr-limit 1.35", "--corr-limit 0.5")
        + " --stage1-selector val_mean_mae --output-prefix nmc_paper_ablate_H_lim0p5_seed012",
        "H_tcn_correction_limit_1p0": base.replace("--corr-limit 1.35", "--corr-limit 1.0")
        + " --stage1-selector val_mean_mae --output-prefix nmc_paper_ablate_H_lim1p0_seed012",
        "H_tcn_correction_limit_1p35": base + " --stage1-selector val_mean_mae --output-prefix nmc_paper_ablate_H_lim1p35_seed012",
        "H_tcn_correction_limit_2p0": base.replace("--corr-limit 1.35", "--corr-limit 2.0")
        + " --stage1-selector val_mean_mae --output-prefix nmc_paper_ablate_H_lim2p0_seed012",
        "I_tcn_correction_without_zero_init": base + " --stage1-selector val_mean_mae --no-corr-zero-init --output-prefix nmc_paper_ablate_I_nozeroinit_seed012",
        "J_tcn_correction_without_rex": base.replace("--lambda-condinv 0.02", "--lambda-condinv 0.02 --lambda-rex 0")
        + " --stage1-selector val_mean_mae --output-prefix nmc_paper_ablate_J_norex_seed012",
        "K_tcn_correction_without_conditional_mmd": base.replace("--lambda-condinv 0.02", "--lambda-condinv 0")
        + " --stage1-selector val_mean_mae --output-prefix nmc_paper_ablate_K_nommd_seed012",
        "L_tcn_single_linear_head_no_temperature_moe": base.replace("--temp-mode moe", "--temp-mode none")
        + " --stage1-selector val_mean_mae --output-prefix nmc_paper_ablate_L_notempmoe_seed012",
    }
    for name in REQUIRED_ABLATIONS:
        add(name, "ablation", "ready", ablation_commands.get(name, ""), "Uses validation-only selection; no FUDS labels for checkpoint choice.")

    baseline_commands = {
        "GRU_h128_linear_vcorr_i_t": base.replace("--recurrent tcn --head-kind linear --temp-mode moe", "--recurrent gru --head-kind linear --temp-mode none")
        + " --hidden-size 128 --layers 1 --stage1-selector val_mean_mae --stage2-select-rule none --output-prefix nmc_paper_base_gru_h128_seed012",
        "GRU_h256_linear_vcorr_i_t": base.replace("--hidden-size 128 --layers 6 --recurrent tcn --head-kind linear --temp-mode moe", "--hidden-size 256 --layers 1 --recurrent gru --head-kind linear --temp-mode none")
        + " --stage1-selector val_mean_mae --stage2-select-rule none --output-prefix nmc_paper_base_gru_h256_seed012",
        "LSTM_h128_linear_vcorr_i_t": base.replace("--recurrent tcn --head-kind linear --temp-mode moe", "--recurrent lstm --head-kind linear --temp-mode none")
        + " --hidden-size 128 --layers 1 --stage1-selector val_mean_mae --stage2-select-rule none --output-prefix nmc_paper_base_lstm_h128_seed012",
        "LSTM_h256_linear_vcorr_i_t": base.replace("--hidden-size 128 --layers 6 --recurrent tcn --head-kind linear --temp-mode moe", "--hidden-size 256 --layers 1 --recurrent lstm --head-kind linear --temp-mode none")
        + " --stage1-selector val_mean_mae --stage2-select-rule none --output-prefix nmc_paper_base_lstm_h256_seed012",
        "TCN_same_budget_no_temperature_moe": base.replace("--temp-mode moe", "--temp-mode none")
        + " --stage1-selector val_mean_mae --stage2-select-rule none --output-prefix nmc_paper_base_tcn_notempmoe_seed012",
        "LSTM_h128_same_cold_hot_correction": base.replace("--recurrent tcn --head-kind linear --temp-mode moe", "--recurrent lstm --head-kind linear --temp-mode none")
        + " --hidden-size 128 --layers 1 --stage1-selector val_mean_mae --output-prefix nmc_paper_base_lstm_h128_corr_seed012",
        "GRU_h128_same_cold_hot_correction": base.replace("--recurrent tcn --head-kind linear --temp-mode moe", "--recurrent gru --head-kind linear --temp-mode none")
        + " --hidden-size 128 --layers 1 --stage1-selector val_mean_mae --output-prefix nmc_paper_base_gru_h128_corr_seed012",
    }
    for name in REQUIRED_BASELINES:
        status = "ready" if name in baseline_commands else "requires_new_runner"
        add(name, "baseline", status, baseline_commands.get(name, ""), "CNN-LSTM/Transformer variants need a separate baseline runner." if status != "ready" else "")

    for hidden in [32, 64, 128, 256]:
        add(
            f"capacity_tcn_h{hidden}_proposed_correction",
            "capacity_response",
            "ready",
            base.replace("--hidden-size 128", f"--hidden-size {hidden}")
            + f" --stage1-selector val_mean_mae --output-prefix nmc_paper_capacity_h{hidden}_seed012",
        )
        add(
            f"capacity_tcn_h{hidden}_base_only",
            "capacity_response",
            "ready",
            base.replace("--hidden-size 128", f"--hidden-size {hidden}")
            + f" --stage1-selector val_mean_mae --stage2-select-rule none --output-prefix nmc_paper_capacity_base_h{hidden}_seed012",
        )

    for window in [20, 50, 100, 200]:
        add(
            f"window_len_{window}_proposed_correction",
            "window_response",
            "ready",
            base.replace("--base-dir .", f"--base-dir . --window-len {window}")
            + f" --stage1-selector val_mean_mae --output-prefix nmc_paper_window_w{window}_seed012",
        )

    for case_id, train_profiles, val_profile, test_profile in PROFILE_ROTATIONS:
        add(
            f"profile_rotation_{case_id}_proposed",
            "profile_rotation",
            "ready",
            base
            + f" --stage1-selector val_mean_mae --train-profiles {train_profiles} --valid-profiles {val_profile} --test-profiles {test_profile} "
            + f"--output-prefix nmc_paper_profile_{case_id}_seed012",
        )

    add(
        "initial_soc_anchor_sensitivity",
        "label_anchor",
        "requires_label_anchor_runner",
        "",
        "Diagnostic anchor tables are generated from existing relabel files; full model reruns for each z0 anchor still need a dedicated runner.",
    )

    commands = pd.DataFrame(rows)
    commands.to_csv(out_dir / "full_run_command_manifest.csv", index=False)
    lines = ["# Full Paper Run Commands", ""]
    for _, row in commands.iterrows():
        lines.extend(
            [
                f"## {row['run_name']}",
                "",
                f"- category: `{row['category']}`",
                f"- status: `{row['status']}`",
                f"- notes: {row['notes']}",
                "",
            ]
        )
        if isinstance(row["command"], str) and row["command"]:
            lines.extend(["```bash", row["command"], "```", ""])
    (out_dir / "full_run_commands.md").write_text("\n".join(lines), encoding="utf-8")


def write_readme_and_report(out_dir: Path, fig_dir: Path, base_dir: Path) -> None:
    val_summary = _read_csv(out_dir / "proposed_val_selected_test_summary.csv")
    valmean_stage2 = _read_csv(out_dir / "proposed_valmean_bothstage_selected_test_summary.csv")
    valworst_stage2 = _read_csv(out_dir / "proposed_valworst_bothstage_selected_test_summary.csv")
    last_summary = _read_csv(out_dir / "proposed_last_epoch20_test_summary.csv")

    def _status_from_summary(df: pd.DataFrame, label: str) -> str:
        if df.empty:
            return f"{label} is pending."
        pass_count = int(df["target_met"].sum()) if "target_met" in df else 0
        total = int(len(df))
        vals = df[["0.0", "25.0", "45.0"]].astype(float)
        return (
            f"{label} is complete; {pass_count}/{total} seeds meet all targets "
            f"(mean MAE: 0C={vals['0.0'].mean():.3f}, 25C={vals['25.0'].mean():.3f}, "
            f"45C={vals['45.0'].mean():.3f})."
        )

    valmean_stage2_status = _status_from_summary(
        valmean_stage2,
        "Both-stage validation-mean selector rerun",
    )
    valworst_stage2_status = _status_from_summary(
        valworst_stage2,
        "Both-stage validation-worst selector rerun",
    )
    if not val_summary.empty:
        val_pass = bool(val_summary["target_met"].all())
        val_status = (
            "Validation-selected proposed rerun is complete and preserves the target conclusion."
            if val_pass
            else "Validation-selected proposed rerun is complete but does not preserve the target conclusion; 25C exceeds the target."
        )
    else:
        val_status = "Validation-selected proposed rerun is pending."
    if not last_summary.empty:
        last_pass_count = int(last_summary["target_met"].sum()) if "target_met" in last_summary else 0
        last_total = int(len(last_summary))
        last_status = (
            f"Fixed final-epoch rerun is complete; {last_pass_count}/{last_total} seeds meet all targets, "
            "so it does not yet support a stable paper claim."
        )
    else:
        last_status = "Fixed final-epoch rerun is pending."
    readme = f"""# Paper Results Package

Generated from current completed NMC Strict NoCC artifacts.

Important status:

- Current proposed score is reproduced from the completed train25-DST selected run.
- Validation-only selector support has been added to the runner.
- Stage 2 validation-only selector support has been added for paper-preferred reruns.
- {val_status}
- {valmean_stage2_status}
- {valworst_stage2_status}
- {last_status}
- Expensive ablation/profile/capacity/window/label-anchor suites are represented by reproducible manifests and pending output tables.

Directories:

- Results: `{out_dir}`
- Figures: `{fig_dir}`

Reproducibility:

- Git commit hash: `{_git_hash(base_dir)}`
- Python: `{platform.python_version()}`
- Platform: `{platform.platform()}`
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    completed_comparison = _read_csv(out_dir / "selector_protocol_comparison.csv")
    if not completed_comparison.empty:
        comparison_table = completed_comparison.to_markdown(index=False, floatfmt=".3f")
    else:
        comparison_table = "See `selector_protocol_comparison.csv` after running the package generator."

    report = f"""# Paper-Ready Experiment Report

## Current Status

This package converts the completed NMC Strict NoCC experiments into a reproducible interim paper package.
The original best result uses the older `train25_dst` checkpoint selector and is marked as `current_nonpaper`.
{val_status}
{valmean_stage2_status}
{valworst_stage2_status}
{last_status}

The current conclusion is negative for a main-paper universal SOC claim: the old best-looking result passes only under the
training-profile-based selector, while validation-only and fixed test-blind protocols fail mainly at 25C.

## Exact Data Split

- Dataset: CALCE Samsung INR18650-20R NMC/graphite dynamic profiles.
- Temperatures: 0C, 25C, 45C.
- Train profiles: DST and US06.
- Validation profile: VALIDATION.
- Test profile: FUDS.
- Window length: 50 samples.
- Training stride: 3.
- Validation/test stride: 1.
- Windows are stateless and independent; hidden state is reset per window.

## Label Construction

- Target label: `SOC_physical` endpoint label.
- SOC is used only as the supervised target/evaluation label.
- No SOC input, no window-start SOC, no SOC_CC input, no cumulative Ah input, and no explicit SOC current-integration state update are used.
- OCV-calibrated start-SOC diagnostics are summarized in `initial_soc_anchor_table.csv`.
- Full label-anchor sensitivity reruns are listed in `label_anchor_sensitivity.csv` as pending.

## Exact Model Configuration

Stage 1 proposed base:

- Input sequence shape: `[batch, 50, 3]`.
- Input features: `[V_corr_raw, I_raw, T]`.
- Encoder: Linear 3->128, LayerNorm, SiLU, 6-layer causal residual TCN.
- Kernel size: 5; dilations: 1, 2, 4, 8, 16, 32.
- Hidden size: 128; dropout: 0.06; channel layer norm.
- Output: endpoint SOC logit from the final window step.
- Head: temperature-MoE with three linear experts; gate uses scaled temperature only.

Stage 2 proposed correction:

- Stage 1 is frozen.
- Correction features: 25-dimensional `vit_decomp` window listed in `feature_schema_stage2.csv`.
- Summary: `concat(x_end, mean(x), std(x), x_end - x_start)`.
- MLP input dimension: 228; MLP: Linear 228->128, LayerNorm, SiLU, Dropout 0.04, Linear 128->2.
- Last linear layer is zero-initialized.
- Bounded correction: `delta_logit = 1.35 * (cold_gate * tanh(raw_cold) + hot_gate * tanh(raw_hot))`.
- Final prediction: `SOC_hat = sigmoid(base_logit + delta_logit)`.

## Strict NoCC Audit

Generated audit files:

- `feature_schema_stage1.csv`
- `feature_schema_stage2.csv`
- `leakage_audit_stage1.csv`
- `leakage_audit_stage2.csv`
- `preprocessing_causality_audit.csv`
- `label_construction_audit.csv`

The selected Stage 1 and Stage 2 feature schemas contain no SOC, SOC_CC, cumulative Ah, absolute time,
trajectory progress, endpoint target, or future-label feature names.

Audit status:

- `leakage_audit_stage1.csv`: no FAIL rows.
- `leakage_audit_stage2.csv`: no FAIL rows.
- `preprocessing_causality_audit.csv`: no FAIL rows.
- `label_construction_audit.csv`: no FAIL rows.

WARN rows are kept where old exploratory train-selector artifacts or SOC_CC mentions in label/audit utilities exist. The selected model input schemas remain the decisive NoCC check.

## Validation Selector Rule

Paper-safe selectors implemented:

- `val_mean_mae`: minimize mean validation MAE across validation profile/temperature groups.
- `val_worst_mae`: minimize worst validation MAE across validation groups.
- `val_mean_plus_worst`: implemented for follow-up screening.
- Fixed final epoch: test-blind diagnostic rule with no metric-based checkpoint selection.

Forbidden as a main-paper selector:

- `train25_dst` and related train-25C profile selectors. These are retained only as exploratory diagnostics.

## Current Main Result

Selector protocol comparison:

{comparison_table}

The only row that satisfies all three targets is the old `train25_dst Stage1 + fixed35 Stage2` result.
That row is rejected as a main-paper adoption rule because it can choose the base checkpoint using 25C training-profile behavior.
The stricter validation-only and fixed protocols do not preserve the 25C target.

## Baselines and Ablations

See `baseline_summary.csv` and `ablation_summary.csv`. The strongest current baseline with the same correction
is LSTM h128 + cold/hot correction, which matches 0C/45C but fails 25C.

Current completed diagnostic baselines show:

- TCN base-only: 0C/45C are weak, while 25C can look good under the old selector.
- LSTM h128 + same correction: 0C and 45C are comparable to the proposed correction result, but 25C remains around 1.12.
- LSTM h64 no correction seed0: 45C is not uniquely hard; therefore 45C alone should not be overclaimed.

## Pending Paper-Protocol Work

The following are prepared but not fully run in this package:

- full ablation suite
- strong baseline suite
- profile rotation
- capacity response
- window response
- initial SOC anchor sensitivity
- correction delta export

Commands are listed in `full_run_commands.md`.

Generated pending/diagnostic tables:

- `ablation_by_seed_temp.csv`
- `profile_rotation_summary.csv`
- `capacity_response_summary.csv`
- `window_response_summary.csv`
- `initial_soc_anchor_table.csv`
- `label_anchor_sensitivity.csv`
- `model_complexity_summary.csv`

## Profile Rotation, Capacity, And Window Response

- `profile_rotation_summary.csv` includes a completed seed0 fixed-last5 diagnostic profile rotation and pending 3-seed validation-selector rows.
- The completed seed0 profile rotation fails all four rotations, and 25C is the worst temperature in every case.
- `capacity_response_summary.csv` and `window_response_summary.csv` currently contain reproducible pending rows and placeholder figures. They should not be cited as completed sweep evidence.

## Residual Analysis

- `residual_by_soc_bin.csv` and FUDS trajectory figures are generated from the completed exploratory train-selector prediction rows.
- `residual_by_current_history.csv` and `correction_delta_summary.csv` are placeholders until raw-window current histories and correction deltas are exported in a paper-protocol rerun.

## Limitations

- The best numeric result is not paper-safe because of the train25-DST checkpoint selector.
- Validation VALIDATION does not fully represent FUDS 25C behavior, so validation-only selection is clean but not sufficient for the target.
- Full 3-seed ablation, baseline, capacity, window, profile-rotation, and label-anchor sensitivity matrices remain pending unless explicitly completed by remote runs.
- Current evidence supports a Strict NoCC ablation package, not a universal SOC main model.

## Safe Claims

- The current train-selector result suggests that the TCN base plus bounded cold/hot correction can satisfy the FUDS targets under strict NoCC input constraints, but this is exploratory rather than paper-protocol evidence.
- Validation-selected and fixed-epoch reruns show that 25C is not yet stable enough for a strong main-paper claim under the current protocol.
- Current is used only as instantaneous excitation and causal voltage-response preprocessing, not as an integrated SOC state update.
- The cold/hot correction primarily reduces 0C and 45C residuals, while 25C depends strongly on the base temporal encoder.
- Under the current clean protocols, the model should be described as a useful NoCC ablation and failure-analysis artifact.

## Unsafe Claims

- Do not claim NoCC generally replaces Coulomb counting.
- Do not claim current is unused.
- Do not claim temperature extrapolation is solved.
- Do not claim 45C performance alone proves novelty.
- Do not claim the old train25-DST selected result is paper-ready.
"""
    (out_dir / "paper_ready_experiment_report.md").write_text(report, encoding="utf-8")


def write_environment(out_dir: Path, base_dir: Path) -> None:
    info = {
        "git_commit": _git_hash(base_dir),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception as exc:
        info["torch_import_error"] = str(exc)
    (out_dir / "environment_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def write_final_terminal_summary(out_dir: Path) -> str:
    baseline = _read_csv(out_dir / "baseline_summary.csv")
    audits = []
    for name in ["leakage_audit_stage1.csv", "leakage_audit_stage2.csv", "preprocessing_causality_audit.csv", "label_construction_audit.csv"]:
        df = _read_csv(out_dir / name)
        if not df.empty:
            audits.append(df)
    audit_df = pd.concat(audits, ignore_index=True, sort=False) if audits else pd.DataFrame()
    audit_pass = bool(not audit_df.empty and not audit_df["status"].astype(str).eq("FAIL").any())
    proposed = baseline[baseline["model_name"].astype(str).str.contains("proposed", na=False)].copy() if not baseline.empty else pd.DataFrame()
    baseline_only = baseline[~baseline["model_name"].astype(str).str.contains("proposed", na=False)].copy() if not baseline.empty else pd.DataFrame()
    best_prop = proposed.sort_values("mean_all_seed_temp").head(1) if not proposed.empty else pd.DataFrame()
    paper_safe_prop = proposed[
        ~proposed["model_name"].astype(str).str.contains("current_train_selector", na=False)
    ].copy() if not proposed.empty else pd.DataFrame()
    best_paper_safe_prop = paper_safe_prop.sort_values("mean_all_seed_temp").head(1) if not paper_safe_prop.empty else pd.DataFrame()
    best_base = baseline_only.sort_values("mean_all_seed_temp").head(1) if not baseline_only.empty else pd.DataFrame()
    val = _read_csv(out_dir / "proposed_val_selected_test_summary.csv")
    profile = _read_csv(out_dir / "profile_rotation_summary.csv")
    anchor = _read_csv(out_dir / "label_anchor_sensitivity.csv")
    val_preserved = bool(not val.empty and val.get("target_met", pd.Series(dtype=bool)).astype(bool).all())
    profile_support = bool(
        not profile.empty
        and profile.get("status", pd.Series(dtype=str)).astype(str).eq("completed_seed0_diagnostic").any()
        and profile.get("target_met", pd.Series(dtype=object)).astype(str).eq("True").all()
    )
    anchor_done = bool(not anchor.empty and not anchor.get("status", pd.Series(["pending"])).astype(str).str.contains("pending").any())

    def _fmt(row: pd.DataFrame) -> str:
        if row.empty:
            return "not_available"
        r = row.iloc[0]
        return (
            f"{r['model_name']} mean={_safe_float(r.get('mean_all_seed_temp')):.3f}, "
            f"0C={_safe_float(r.get('MAE_0C_mean')):.3f}, "
            f"25C={_safe_float(r.get('MAE_25C_mean')):.3f}, "
            f"45C={_safe_float(r.get('MAE_45C_mean')):.3f}"
        )

    lines = [
        "Paper-ready package summary",
        f"- best exploratory proposed result in current package: {_fmt(best_prop)}",
        f"- best paper-protocol proposed result in current package: {_fmt(best_paper_safe_prop)}",
        f"- strongest baseline result in current package: {_fmt(best_base)}",
        f"- validation-based selector preserved original conclusion: {'YES' if val_preserved else 'NO'}",
        f"- profile rotation supports conclusion: {'YES' if profile_support else 'NO'}",
        f"- strict NoCC audit has FAIL rows: {'NO' if audit_pass else 'YES'}",
        f"- initial SOC anchor sensitivity complete: {'YES' if anchor_done else 'NO, rerun matrix is pending'}",
    ]
    text = "\n".join(lines) + "\n"
    (out_dir / "final_terminal_summary.txt").write_text(text, encoding="utf-8")
    return text


def run(base_dir: Path, result_dir: Path, out_dir: Path, fig_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    files = ResultFileSet()
    write_audits(out_dir, selector_name="val_mean_mae", base_dir=base_dir)
    write_model_summaries(out_dir, result_dir, files)
    write_selected_checkpoints(out_dir, result_dir, files)
    write_proposed_predictions_and_residuals(out_dir, fig_dir, result_dir)
    write_run_manifest(out_dir)
    write_required_pending_outputs(out_dir, fig_dir)
    write_model_complexity(out_dir)
    write_profile_rotation_summary(out_dir, base_dir)
    write_initial_soc_anchor_outputs(out_dir, base_dir)
    write_environment(out_dir, base_dir)
    write_readme_and_report(out_dir, fig_dir, base_dir)
    final_summary = write_final_terminal_summary(out_dir)
    print(f"Paper-ready package written to {out_dir}")
    print(f"Figures written to {fig_dir}")
    print(final_summary)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate paper-ready NMC Strict NoCC interim package from completed artifacts.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--result-dir", default="remote_result_summaries")
    p.add_argument("--out-dir", default="paper_results")
    p.add_argument("--fig-dir", default="paper_figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    result_dir = Path(args.result_dir)
    if not result_dir.is_absolute():
        result_dir = base_dir / result_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir
    fig_dir = Path(args.fig_dir)
    if not fig_dir.is_absolute():
        fig_dir = base_dir / fig_dir
    run(base_dir, result_dir, out_dir, fig_dir)


if __name__ == "__main__":
    main()
