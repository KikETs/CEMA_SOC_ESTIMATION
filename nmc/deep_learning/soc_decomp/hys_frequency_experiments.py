from dataclasses import fields
from pathlib import Path

import pandas as pd

from .config import CFG, make_cfg
from .runtime import configure_torch_runtime
from .smooth_corrector import (
    STATE_SUMMARY_COLS,
    add_state_summary_features,
    branch_frequency_energy_frame,
    make_smooth_corrector_cfg,
    smooth_corrector_voltage_reconstruction,
    train_smooth_corrector_and_extract_features,
)
from .training import (
    ABLATIONS,
    attach_prediction_features,
    build_prediction_feature_lookup,
    train_one_lstm_ablation,
)
from .variance_control import (
    HYBRID_RAW_COLS,
    HYBRID_SUMMARY_COLS,
    R5_GATED_FEATURES,
    _summary_with_temp20_focus,
    load_feature_frame_dict_from_csv,
    train_variance_model_from_spec,
)

try:
    from IPython.display import display
except Exception:
    display = print


def clone_cfg(cfg: CFG | None = None) -> CFG:
    src = make_cfg() if cfg is None else cfg
    out = make_cfg()
    for f in fields(CFG):
        setattr(out, f.name, getattr(src, f.name))
    return out


def exp_c_cfg(cfg: CFG | None = None) -> CFG:
    out = clone_cfg(cfg)
    out.smoke_mode = False
    out.use_existing_soc_cc_if_available = False
    out.use_existing_usable_if_available = False
    out.train_temps = ("N10", "0", "10", "25", "50")
    out.eval_temps = ("N10", "0", "10", "20", "25", "30", "40", "50")
    out.train_drives = ("DST", "US06")
    out.eval_drive = "FUDS"
    return out


def _train_prediction_rows(feature_frames, cfg: CFG, model_specs):
    feature_lookup = build_prediction_feature_lookup(feature_frames)
    rows = []
    histories = {}
    gates = []
    for model_name, spec in model_specs:
        print(f"\n=== hys/frequency SOC model: {model_name} ===")
        if isinstance(spec, list):
            _, hist, _, pred_test, _, _ = train_one_lstm_ablation(
                feature_frames,
                spec,
                "physical",
                cfg,
                model_name,
            )
            gate_df = pd.DataFrame()
        else:
            _, hist, pred_test, gate_df = train_variance_model_from_spec(feature_frames, model_name, spec, cfg)
        histories[model_name] = hist
        pred_test = pred_test.assign(split="test", ablation=model_name)
        rows.append(attach_prediction_features(pred_test, feature_lookup, ablation_name=model_name, target_label="physical"))
        if len(gate_df):
            gates.append(gate_df)
    pred_rows = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    gate_rows = pd.concat(gates, ignore_index=True) if gates else pd.DataFrame()
    return pred_rows, histories, gate_rows


def _write_summary_outputs(pred_rows, cfg: CFG, prefix: str):
    summary, by_temp, focus, temp20 = _summary_with_temp20_focus(pred_rows)
    pred_rows.to_csv(cfg.output_dir / f"{prefix}_prediction_rows.csv", index=False)
    summary.to_csv(cfg.output_dir / f"{prefix}_results.csv", index=False)
    by_temp.to_csv(cfg.output_dir / f"{prefix}_by_temperature.csv", index=False)
    focus.to_csv(cfg.output_dir / f"{prefix}_focus_metrics.csv", index=False)
    temp20.to_csv(cfg.output_dir / f"temp20_{prefix}_jitter.csv", index=False)
    return {
        "prediction_rows": pred_rows,
        "summary": summary,
        "by_temperature": by_temp,
        "focus_metrics": focus,
        "temp20_jitter": temp20,
    }


def _evaluate_corrector_feature_models(feature_frames, cfg: CFG, prefix: str):
    weak_aug_spec = {
        "features": R5_GATED_FEATURES,
        "kind": "gated_seq",
        "lambda_gate_smooth": 0.03,
        "lambda_aug": 0.03,
        "component_noise_std": 0.005,
        "component_dropout_p": 0.05,
    }
    model_specs = [
        ("R5_raw_I_T_all_components", ABLATIONS["R5_raw_I_T_all_components"]),
        ("R5_GATED", ABLATIONS["R5_GATED"]),
        ("R5_GATED_AUG_WEAK", weak_aug_spec),
    ]
    pred_rows, histories, gates = _train_prediction_rows(feature_frames, cfg, model_specs)
    out = _write_summary_outputs(pred_rows, cfg, prefix)
    if len(gates):
        gates.to_csv(cfg.output_dir / f"{prefix}_component_gate_by_temperature.csv", index=False)
    out["histories"] = histories
    out["component_gates"] = gates
    return out


def _prepare_corrector_cfg(
    cfg: CFG,
    *,
    output_dir_name: str,
    strong_hys: bool = False,
    frequency_routing: bool = False,
):
    ccfg = exp_c_cfg(cfg)
    make_smooth_corrector_cfg(ccfg, output_dir_name=output_dir_name)
    ccfg.corrector_batch_by_length = True
    ccfg.corrector_profile_batch_size = int(getattr(cfg, "corrector_profile_batch_size", 8))
    ccfg.corrector_train_segment_len = getattr(cfg, "corrector_train_segment_len", 4096)
    ccfg.corrector_segments_per_profile_per_epoch = int(getattr(cfg, "corrector_segments_per_profile_per_epoch", 4))
    if strong_hys:
        ccfg.lambda_hys_hf = float(getattr(cfg, "lambda_hys_hf", 0.03) or 0.03)
        ccfg.lambda_hys_tv = float(getattr(cfg, "lambda_hys_tv", 0.05) or 0.05)
        ccfg.lambda_hys_slope = float(getattr(cfg, "lambda_hys_slope", 0.01) or 0.01)
        ccfg.hys_limit_scale = float(getattr(cfg, "hys_limit_scale", 0.5) or 0.5)
    if frequency_routing:
        ccfg.lambda_pol_hf = float(getattr(cfg, "lambda_pol_hf", 0.005) or 0.005)
        ccfg.lambda_pol_slow_hf = float(getattr(cfg, "lambda_pol_slow_hf", 0.01) or 0.01)
        ccfg.lambda_hys_hf = float(getattr(cfg, "lambda_hys_hf", 0.03) or 0.03)
        ccfg.lambda_R0_hf = float(getattr(cfg, "lambda_R0_hf", 0.01) or 0.01)
        ccfg.lambda_frequency_route = float(getattr(cfg, "lambda_frequency_route", 0.005) or 0.005)
        ccfg.lambda_R0_smooth = float(getattr(cfg, "lambda_R0_smooth", 0.05) or 0.05)
    return ccfg


def run_strong_hys_corrector_experiment(cfg: CFG | None = None):
    cfg = exp_c_cfg(cfg)
    ccfg = _prepare_corrector_cfg(
        cfg,
        output_dir_name="decomposed_features_strong_hys_corrector",
        strong_hys=True,
        frequency_routing=False,
    )
    smooth = train_smooth_corrector_and_extract_features(ccfg)
    smooth["component_hf"].to_csv(ccfg.output_dir / "strong_hys_corrector_component_hf.csv", index=False)
    recon = smooth_corrector_voltage_reconstruction(smooth["feature_frames"], ccfg)
    recon.to_csv(ccfg.output_dir / "strong_hys_corrector_reconstruction.csv", index=False)
    soc = _evaluate_corrector_feature_models(smooth["feature_frames"], ccfg, "strong_hys_corrector")
    soc["summary"].to_csv(ccfg.output_dir / "strong_hys_corrector_ablation_results.csv", index=False)
    soc["temp20_jitter"].to_csv(ccfg.output_dir / "strong_hys_corrector_temp20_jitter.csv", index=False)
    return {"cfg": ccfg, "corrector": smooth, "soc": soc}


def run_frequency_routing_corrector_experiment(cfg: CFG | None = None):
    cfg = exp_c_cfg(cfg)
    ccfg = _prepare_corrector_cfg(
        cfg,
        output_dir_name="decomposed_features_frequency_routing_corrector",
        strong_hys=True,
        frequency_routing=True,
    )
    smooth = train_smooth_corrector_and_extract_features(ccfg)
    component_energy = smooth["component_hf"]
    branch_energy = branch_frequency_energy_frame(smooth["feature_frames"])
    component_energy.to_csv(ccfg.output_dir / "frequency_routing_component_energy.csv", index=False)
    branch_energy.to_csv(ccfg.output_dir / "frequency_routing_branch_energy.csv", index=False)
    soc = _evaluate_corrector_feature_models(smooth["feature_frames"], ccfg, "frequency_routing")
    soc["summary"].to_csv(ccfg.output_dir / "frequency_routing_ablation_results.csv", index=False)
    return {"cfg": ccfg, "corrector": smooth, "branch_energy": branch_energy, "soc": soc}


def _load_exp_c_frames(cfg: CFG):
    cfg = exp_c_cfg(cfg)
    decomposed_dir = cfg.output_dir / "decomposed_features_train_temp_minus10_0_10_25_50"
    return load_feature_frame_dict_from_csv(cfg, decomposed_dir=decomposed_dir)


def hybrid_summary_specs():
    features = [*HYBRID_RAW_COLS, *HYBRID_SUMMARY_COLS]
    return [
        ("R5_HYBRID_SUMMARY", {
            "features": features,
            "kind": "hybrid_summary",
            "raw_cols": HYBRID_RAW_COLS,
            "summary_cols": HYBRID_SUMMARY_COLS,
            "lambda_seq": 0.5,
        }),
        ("R5_GATED_HYBRID_SUMMARY", {
            "features": features,
            "kind": "hybrid_summary",
            "raw_cols": HYBRID_RAW_COLS,
            "summary_cols": HYBRID_SUMMARY_COLS,
            "gated": True,
            "lambda_seq": 0.5,
            "lambda_gate_smooth": 0.03,
            "lambda_aug": 0.03,
            "component_noise_std": 0.005,
            "component_dropout_p": 0.05,
        }),
    ]


def monotonicity_specs():
    features = [*HYBRID_RAW_COLS, *HYBRID_SUMMARY_COLS]
    return [
        ("R5_GATED_MONO", {
            "features": R5_GATED_FEATURES,
            "kind": "gated_seq",
            "lambda_seq": 0.5,
            "lambda_gate_smooth": 0.03,
            "lambda_mono": 0.03,
            "mono_eps": 0.003,
        }),
        ("R5_GATED_HYBRID_SUMMARY_MONO", {
            "features": features,
            "kind": "hybrid_summary",
            "raw_cols": HYBRID_RAW_COLS,
            "summary_cols": HYBRID_SUMMARY_COLS,
            "gated": True,
            "lambda_seq": 0.5,
            "lambda_gate_smooth": 0.03,
            "lambda_aug": 0.03,
            "component_noise_std": 0.005,
            "component_dropout_p": 0.05,
            "lambda_mono": 0.03,
            "mono_eps": 0.003,
        }),
    ]


def run_hybrid_summary_experiment(cfg: CFG | None = None, *, feature_frames=None, output_prefix="hybrid_summary"):
    cfg = exp_c_cfg(cfg)
    configure_torch_runtime()
    frames = _load_exp_c_frames(cfg) if feature_frames is None else feature_frames
    frames = add_state_summary_features(frames, window_len=cfg.window_len)
    pred_rows, histories, gates = _train_prediction_rows(frames, cfg, hybrid_summary_specs())
    out = _write_summary_outputs(pred_rows, cfg, output_prefix)
    if len(gates):
        gates.to_csv(cfg.output_dir / f"{output_prefix}_component_gate_by_temperature.csv", index=False)
    out["histories"] = histories
    out["component_gates"] = gates
    # Compatibility filenames requested by the notebook notes.
    if output_prefix == "hybrid_summary":
        out["summary"].to_csv(cfg.output_dir / "hybrid_summary_results.csv", index=False)
        out["by_temperature"].to_csv(cfg.output_dir / "hybrid_summary_by_temperature.csv", index=False)
        out["temp20_jitter"].to_csv(cfg.output_dir / "temp20_hybrid_summary_jitter.csv", index=False)
    return out


def run_monotonicity_regularization_experiment(cfg: CFG | None = None, *, feature_frames=None):
    cfg = exp_c_cfg(cfg)
    configure_torch_runtime()
    frames = _load_exp_c_frames(cfg) if feature_frames is None else feature_frames
    frames = add_state_summary_features(frames, window_len=cfg.window_len)
    pred_rows, histories, gates = _train_prediction_rows(frames, cfg, monotonicity_specs())
    out = _write_summary_outputs(pred_rows, cfg, "monotonicity_regularization")
    out["summary"].to_csv(cfg.output_dir / "monotonicity_regularization_results.csv", index=False)
    out["by_temperature"].to_csv(cfg.output_dir / "monotonicity_by_temperature.csv", index=False)
    out["temp20_jitter"].to_csv(cfg.output_dir / "temp20_monotonicity_jitter.csv", index=False)
    if len(gates):
        gates.to_csv(cfg.output_dir / "monotonicity_component_gate_by_temperature.csv", index=False)
    out["histories"] = histories
    out["component_gates"] = gates
    return out


def run_hys_frequency_followup_experiments(cfg: CFG | None = None):
    cfg = exp_c_cfg(cfg)
    strong = run_strong_hys_corrector_experiment(cfg)
    freq = run_frequency_routing_corrector_experiment(cfg)
    hybrid = run_hybrid_summary_experiment(cfg)
    mono = run_monotonicity_regularization_experiment(cfg)
    return {"strong_hys": strong, "frequency_routing": freq, "hybrid_summary": hybrid, "monotonicity": mono}
