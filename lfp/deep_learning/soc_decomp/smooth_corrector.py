from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import CFG, make_cfg
from .runtime import configure_torch_runtime, device
from .data import load_and_prepare_data
from .models import build_corrector
from .corrector import run_corrector_pretraining, tensor_profile
from .features import extract_all_feature_frames
from .training import (
    ABLATIONS,
    attach_prediction_features,
    build_prediction_feature_lookup,
    train_one_lstm_ablation,
)
from .variance_control import (
    R5_GATED_FEATURES,
    _summary_with_temp20_focus,
    _trajectory_jitter_rows,
    load_feature_frame_dict_from_csv,
    train_variance_model_from_spec,
)

try:
    from IPython.display import display
except Exception:
    display = print


COMPONENT_HF_COLUMNS = ["V_pol_raw", "V_hys_raw", "R0"]
STATE_SUMMARY_COLS = [
    "V_pol_fast_last", "V_pol_mid_last", "V_pol_slow_last",
    "V_pol_fast_mean", "V_pol_mid_mean", "V_pol_slow_mean",
    "V_pol_fast_rms", "V_pol_mid_rms", "V_pol_slow_rms",
    "V_pol_mean", "V_pol_rms", "V_pol_hf_energy",
    "V_hys_last", "V_hys_mean", "V_hys_rms", "V_hys_hf_energy",
    "R0_mean", "R0_slope",
    "V_ohm_rms",
]
BRANCH_HF_COLUMNS = [
    "V_pol_fast_raw", "V_pol_mid_raw", "V_pol_slow_raw",
    "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0",
]


def high_frequency_energy_np(values):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    d1 = np.diff(x)
    d2 = np.diff(d1)
    return float(np.mean(d2 ** 2))


def _component_hf_frame(feature_frames):
    rows = []
    for split, frames in feature_frames.items():
        for frame in frames:
            if frame.empty:
                continue
            row = {
                "split": split,
                "trajectory_id": frame["trajectory_id"].iloc[0],
                "drive_cycle": frame["drive_cycle"].iloc[0],
                "temperature_C": float(frame["temperature"].iloc[0]),
                "n_rows": int(len(frame)),
            }
            for col in COMPONENT_HF_COLUMNS:
                if col in frame.columns:
                    row[f"{col}_hf_energy"] = high_frequency_energy_np(frame[col].to_numpy())
                    row[f"{col}_diff_energy"] = float(np.mean(np.diff(frame[col].to_numpy(np.float64)) ** 2)) if len(frame) > 1 else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def branch_frequency_energy_frame(feature_frames):
    rows = []
    for split, frames in feature_frames.items():
        for frame in frames:
            if frame.empty:
                continue
            row = {
                "split": split,
                "trajectory_id": frame["trajectory_id"].iloc[0],
                "drive_cycle": frame["drive_cycle"].iloc[0],
                "temperature_C": float(frame["temperature"].iloc[0]),
                "n_rows": int(len(frame)),
            }
            for col in BRANCH_HF_COLUMNS:
                if col not in frame.columns:
                    continue
                x = frame[col].to_numpy(np.float64)
                row[f"{col}_hf_energy"] = high_frequency_energy_np(x)
                row[f"{col}_diff_energy"] = float(np.mean(np.diff(x) ** 2)) if len(x) > 1 else float("nan")
                row[f"{col}_rms"] = float(np.sqrt(np.mean(x ** 2)))
            rows.append(row)
    return pd.DataFrame(rows)


def run_component_high_frequency_diagnostic(cfg: CFG | None = None, *, decomposed_dir=None):
    cfg = make_cfg() if cfg is None else cfg
    configure_torch_runtime()
    decomposed_dir = Path(decomposed_dir or cfg.output_dir / "decomposed_features_train_temp_minus10_0_10_25_50")
    feature_frames = load_feature_frame_dict_from_csv(cfg, decomposed_dir=decomposed_dir)
    hf = _component_hf_frame(feature_frames)
    hf.to_csv(cfg.output_dir / "component_high_frequency_energy.csv", index=False)

    hf_cols = [c for c in hf.columns if c.endswith("_hf_energy") or c.endswith("_diff_energy")]
    by_temp = (
        hf.groupby(["split", "drive_cycle", "temperature_C"])[hf_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    by_temp.columns = ["_".join([str(x) for x in c if str(x)]) for c in by_temp.columns]
    by_temp.to_csv(cfg.output_dir / "component_hf_by_temperature.csv", index=False)

    pred_path = cfg.output_dir / "train_temp_minus10_0_10_25_50_prediction_rows.csv"
    if pred_path.exists():
        pred = pd.read_csv(pred_path)
        pred = pred[(pred["label_type"] == "physical") & (pred["end_index"] >= 2)].copy()
        jitter = _trajectory_jitter_rows(pred)
        joined = jitter.merge(
            hf[hf["split"].eq("test")][["trajectory_id", *hf_cols]],
            on="trajectory_id",
            how="left",
        )
        corr_rows = []
        for model, g in joined.groupby("model_name"):
            for col in hf_cols:
                if col not in g or g[col].notna().sum() < 3:
                    continue
                for metric in ["jitter_ratio", "high_frequency_error_energy", "delta_soc_mae", "curvature_mae"]:
                    corr = g[[col, metric]].corr().iloc[0, 1]
                    corr_rows.append({
                        "model_name": model,
                        "component_hf_metric": col,
                        "prediction_jitter_metric": metric,
                        "pearson_corr": float(corr),
                        "n_trajectories": int(g[[col, metric]].dropna().shape[0]),
                    })
        hf_vs_jitter = pd.DataFrame(corr_rows)
    else:
        hf_vs_jitter = pd.DataFrame()
    hf_vs_jitter.to_csv(cfg.output_dir / "component_hf_vs_prediction_jitter.csv", index=False)

    print("Component high-frequency energy:")
    display(hf)
    print("Component HF by temperature:")
    display(by_temp)
    print("Component HF vs prediction jitter correlation:")
    display(hf_vs_jitter)
    return {"component_hf": hf, "by_temperature": by_temp, "hf_vs_prediction_jitter": hf_vs_jitter}


@torch.no_grad()
def smooth_corrector_voltage_reconstruction(feature_frames, cfg: CFG):
    rows = []
    for split, frames in feature_frames.items():
        for frame in frames:
            if frame.empty:
                continue
            comp_sum = frame["V_pol_raw"] + frame["V_hys_raw"] + frame["V_ohm_raw"]
            if "g_corr" in frame.columns:
                drop_uncapped = frame["g_corr"] * comp_sum
            else:
                drop_uncapped = comp_sum
            drop_actual = frame["V_corr_raw"] - frame["V_raw"]
            component_recon_error = drop_actual - drop_uncapped
            rows.append({
                "split": split,
                "trajectory_id": frame["trajectory_id"].iloc[0],
                "drive_cycle": frame["drive_cycle"].iloc[0],
                "temperature_C": float(frame["temperature"].iloc[0]),
                "component_reconstruction_MAE": float(np.mean(np.abs(component_recon_error))),
                "component_reconstruction_RMSE": float(np.sqrt(np.mean(component_recon_error ** 2))),
                "drop_capping_MAE": float(np.mean(np.abs(component_recon_error))),
                "corrected_voltage_shift_MAE": float(np.mean(np.abs(drop_actual))),
                "corrected_voltage_shift_RMSE": float(np.sqrt(np.mean(drop_actual ** 2))),
                "vcorr_min": float(frame["V_corr_raw"].min()),
                "vcorr_max": float(frame["V_corr_raw"].max()),
            })
    out = pd.DataFrame(rows)
    out.to_csv(cfg.output_dir / "smooth_corrector_voltage_reconstruction.csv", index=False)
    return out


def make_smooth_corrector_cfg(cfg: CFG, *, output_dir_name="decomposed_features_smooth_corrector"):
    scfg = cfg
    scfg.corrector_variant = "smooth_decomp"
    scfg.decomposed_dir = scfg.output_dir / output_dir_name
    scfg.save_decomposed_features = True
    scfg.reuse_cached_decomposed_features = False
    scfg.run_corrector_pretraining = True
    scfg.lambda_pol_hf = float(getattr(scfg, "lambda_pol_hf", 0.005) or 0.005)
    scfg.lambda_hys_hf = float(getattr(scfg, "lambda_hys_hf", 0.001) or 0.001)
    scfg.lambda_R0_smooth = float(getattr(scfg, "lambda_R0_smooth", 0.05) or 0.05)
    scfg.lambda_hys_smooth = float(getattr(scfg, "lambda_hys_smooth", 0.01) or 0.01)
    scfg.lambda_timescale_sep = float(getattr(scfg, "lambda_timescale_sep", 0.001) or 0.001)
    scfg.decomposed_dir.mkdir(parents=True, exist_ok=True)
    return scfg


def train_smooth_corrector_and_extract_features(cfg: CFG | None = None):
    cfg = make_cfg() if cfg is None else cfg
    cfg = make_smooth_corrector_cfg(cfg)
    configure_torch_runtime()
    data = load_and_prepare_data(cfg)
    corrector = build_corrector(cfg, device)
    history = run_corrector_pretraining(corrector, data["train_profiles"], cfg, data["v_scaler"])
    feature_frames = extract_all_feature_frames(
        corrector,
        data["train_profiles"],
        data["valid_profiles"],
        data["test_profiles"],
        cfg,
        data["v_scaler"],
    )
    hf = _component_hf_frame(feature_frames)
    hf.to_csv(cfg.output_dir / "smooth_corrector_component_hf.csv", index=False)
    voltage_recon = smooth_corrector_voltage_reconstruction(feature_frames, cfg)
    history.to_csv(cfg.output_dir / "smooth_corrector_training_history.csv", index=False)
    return {
        "cfg": cfg,
        "data": data,
        "corrector": corrector,
        "history": history,
        "feature_frames": feature_frames,
        "component_hf": hf,
        "voltage_reconstruction": voltage_recon,
    }


def _train_lstm_prediction_rows(feature_frames, cfg: CFG, names):
    feature_lookup = build_prediction_feature_lookup(feature_frames)
    rows = []
    histories = {}
    for name in names:
        print(f"\n=== smooth feature SOC model: {name} ===")
        model, hist, _, pred_test, _, _ = train_one_lstm_ablation(
            feature_frames,
            ABLATIONS[name],
            "physical",
            cfg,
            name,
        )
        histories[name] = hist
        pred_test = pred_test.assign(split="test", ablation=name)
        rows.append(attach_prediction_features(pred_test, feature_lookup, ablation_name=name, target_label="physical"))
    return rows, histories


def _train_aug_prediction_rows(feature_frames, cfg: CFG):
    feature_lookup = build_prediction_feature_lookup(feature_frames)
    specs = [
        ("R5_GATED_AUG_np01_dp1_lp05", {
            "features": R5_GATED_FEATURES,
            "kind": "gated_seq",
            "lambda_gate_smooth": 0.03,
            "lambda_aug": 0.05,
            "component_noise_std": 0.01,
            "component_dropout_p": 0.10,
        }),
        ("R5_GATED_AUG_WEAK_np005_dp05_lp03", {
            "features": R5_GATED_FEATURES,
            "kind": "gated_seq",
            "lambda_gate_smooth": 0.03,
            "lambda_aug": 0.03,
            "component_noise_std": 0.005,
            "component_dropout_p": 0.05,
        }),
    ]
    rows = []
    histories = {}
    for name, spec in specs:
        print(f"\n=== smooth feature SOC model: {name} ===")
        _, hist, pred_test, _ = train_variance_model_from_spec(feature_frames, name, spec, cfg)
        histories[name] = hist
        pred_test = pred_test.assign(split="test", ablation=name)
        rows.append(attach_prediction_features(pred_test, feature_lookup, ablation_name=name, target_label="physical"))
    return rows, histories


def evaluate_smooth_corrector_soc_models(feature_frames, cfg: CFG):
    base_rows, base_histories = _train_lstm_prediction_rows(
        feature_frames,
        cfg,
        ["R5_raw_I_T_all_components", "R5_GATED"],
    )
    aug_rows, aug_histories = _train_aug_prediction_rows(feature_frames, cfg)
    pred_rows = pd.concat(base_rows + aug_rows, ignore_index=True)
    pred_rows.to_csv(cfg.output_dir / "smooth_corrector_prediction_rows.csv", index=False)
    summary, by_temp, focus, temp20 = _summary_with_temp20_focus(pred_rows)
    summary.to_csv(cfg.output_dir / "smooth_corrector_ablation_results.csv", index=False)
    by_temp.to_csv(cfg.output_dir / "smooth_corrector_by_temperature.csv", index=False)
    focus.to_csv(cfg.output_dir / "smooth_corrector_focus_metrics.csv", index=False)
    temp20.to_csv(cfg.output_dir / "smooth_corrector_temp20_jitter.csv", index=False)
    print("Smooth corrector SOC results:")
    display(summary)
    return {
        "prediction_rows": pred_rows,
        "summary": summary,
        "by_temperature": by_temp,
        "focus_metrics": focus,
        "temp20_jitter": temp20,
        "histories": {**base_histories, **aug_histories},
    }


def add_state_summary_features(feature_frames, window_len=50):
    out = {}
    for split, frames in feature_frames.items():
        out[split] = []
        for frame in frames:
            f = frame.copy()
            w = int(window_len)
            if "V_pol_fast_raw" not in f:
                f["V_pol_fast_raw"] = f["V_pol_raw"]
                f["V_pol_mid_raw"] = 0.0
                f["V_pol_slow_raw"] = 0.0
            f["V_pol_fast_last"] = f["V_pol_fast_raw"]
            f["V_pol_mid_last"] = f["V_pol_mid_raw"]
            f["V_pol_slow_last"] = f["V_pol_slow_raw"]
            f["V_hys_last"] = f["V_hys_raw"]
            for src, mean_col, rms_col, hf_col in [
                ("V_pol_raw", "V_pol_mean", "V_pol_rms", "V_pol_hf_energy"),
                ("V_hys_raw", "V_hys_mean", "V_hys_rms", "V_hys_hf_energy"),
                ("V_ohm_raw", None, "V_ohm_rms", None),
                ("V_pol_fast_raw", "V_pol_fast_mean", "V_pol_fast_rms", None),
                ("V_pol_mid_raw", "V_pol_mid_mean", "V_pol_mid_rms", None),
                ("V_pol_slow_raw", "V_pol_slow_mean", "V_pol_slow_rms", None),
            ]:
                if src not in f.columns:
                    continue
                roll = f[src].rolling(w, min_periods=1)
                if mean_col:
                    f[mean_col] = roll.mean()
                f[rms_col] = np.sqrt(roll.apply(lambda x: float(np.mean(np.asarray(x) ** 2)), raw=False))
                if hf_col:
                    f[hf_col] = roll.apply(high_frequency_energy_np, raw=True).fillna(0.0)
            f["R0_mean"] = f["R0"].rolling(w, min_periods=1).mean()
            f["R0_slope"] = f["R0"].diff().rolling(w, min_periods=1).mean().fillna(0.0)
            out[split].append(f)
    return out


def run_state_summary_experiment(feature_frames, cfg: CFG):
    frames = add_state_summary_features(feature_frames, window_len=cfg.window_len)
    feature_lookup = build_prediction_feature_lookup(frames)
    specs = {
        "R5_STATE_SUMMARY": ["V_raw", "V_corr_raw", "I_raw", "T", "dI", "absI", *STATE_SUMMARY_COLS],
        "R5_RAW_PLUS_STATE_SUMMARY": [
            "V_raw", "V_corr_raw", "I_raw", "T", "dI", "absI",
            "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0", *STATE_SUMMARY_COLS,
        ],
        "R5_GATED_STATE_SUMMARY": [
            "V_raw", "V_corr_raw", "I_raw", "T", "dI", "absI",
            "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0", *STATE_SUMMARY_COLS,
        ],
    }
    rows = []
    histories = {}
    for name, cols in specs.items():
        print(f"\n=== state-summary SOC model: {name} ===")
        model, hist, _, pred_test, _, _ = train_one_lstm_ablation(frames, cols, "physical", cfg, name)
        histories[name] = hist
        pred_test = pred_test.assign(split="test", ablation=name)
        rows.append(attach_prediction_features(pred_test, feature_lookup, ablation_name=name, target_label="physical"))
    pred_rows = pd.concat(rows, ignore_index=True)
    pred_rows.to_csv(cfg.output_dir / "state_summary_prediction_rows.csv", index=False)
    summary, by_temp, focus, temp20 = _summary_with_temp20_focus(pred_rows)
    summary.to_csv(cfg.output_dir / "state_summary_results.csv", index=False)
    by_temp.to_csv(cfg.output_dir / "state_summary_by_temperature.csv", index=False)
    focus.to_csv(cfg.output_dir / "state_summary_focus_metrics.csv", index=False)
    temp20.to_csv(cfg.output_dir / "temp20_state_summary_jitter.csv", index=False)
    print("State-summary SOC results:")
    display(summary)
    return {
        "prediction_rows": pred_rows,
        "summary": summary,
        "by_temperature": by_temp,
        "focus_metrics": focus,
        "temp20_jitter": temp20,
        "histories": histories,
    }


def run_smooth_corrector_experiment(cfg: CFG | None = None):
    cfg = make_cfg() if cfg is None else cfg
    smooth = train_smooth_corrector_and_extract_features(cfg)
    soc = evaluate_smooth_corrector_soc_models(smooth["feature_frames"], smooth["cfg"])
    state = run_state_summary_experiment(smooth["feature_frames"], smooth["cfg"])
    return {"smooth": smooth, "soc": soc, "state_summary": state}
