from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .config import CFG, make_cfg
from .runtime import configure_torch_runtime, device
from .data import load_and_prepare_data
from .models import build_corrector
from .corrector import run_corrector_pretraining
from .features import extract_all_feature_frames
from .diagnostics import run_shortcut_diagnostics
from .temp10_diagnostics import (
    component_temperature_distance,
    temp10_error_diagnostic,
)
from .training import (
    ABLATIONS,
    ablation_feature_cols,
    attach_prediction_features,
    build_prediction_feature_lookup,
    metrics_by_group,
    metrics_by_soc_bin,
    plot_component_gate_summary,
    summarize_component_gates,
    summarize_overall_metrics,
    train_one_lstm_ablation,
)

try:
    from IPython.display import display
except Exception:
    display = print


def _drive_split(cfg: CFG, drive):
    drive = str(drive).upper()
    if drive == str(getattr(cfg, "eval_drive", "FUDS")).upper():
        return "test"
    if drive in {str(d).upper() for d in getattr(cfg, "train_drives", ("DST", "US06"))}:
        return "train"
    return None


def load_feature_frame_dict_from_csv(cfg: CFG, decomposed_dir=None):
    decomposed_dir = Path(decomposed_dir or cfg.decomposed_dir)
    feature_frames = {"train": [], "valid": [], "test": []}
    for path in sorted(decomposed_dir.glob("*_features.csv")):
        frame = pd.read_csv(path)
        if frame.empty or "drive_cycle" not in frame.columns:
            continue
        split = _drive_split(cfg, frame["drive_cycle"].iloc[0])
        if split is not None:
            feature_frames[split].append(frame)
    if not feature_frames["train"] or not feature_frames["test"]:
        raise ValueError(f"Could not load train/test feature frames from {decomposed_dir}")
    return feature_frames


def _copy_if_exists(src, dst):
    src = Path(src)
    if src.exists():
        shutil.copy2(src, dst)


def _combined_predictions_for_csv(prediction_rows):
    if not prediction_rows:
        return pd.DataFrame()
    out = pd.concat(prediction_rows, ignore_index=True)
    ordered = [
        "model_name", "label_type", "trajectory_id", "temperature_C", "drive_cycle",
        "time_index", "end_index", "y_true", "y_pred", "error", "abs_error",
        "V_raw", "V_corr_raw", "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0",
        "cumulative_Ah", "SOC_bin", "is_plateau_20_80", "is_cutoff_last10",
    ]
    cols = [c for c in ordered if c in out.columns] + [c for c in out.columns if c not in ordered]
    return out[cols]


def focus_metrics(pred_rows, *, temperature=None):
    if pred_rows.empty:
        return pd.DataFrame()
    df = pred_rows.copy()
    if temperature is not None:
        df = df[np.isclose(df["temperature_C"].astype(float), float(temperature))].copy()
    rows = []
    for (label, model), g in df.groupby(["label_type", "model_name"]):
        if g.empty:
            continue
        plateau = g[g["is_plateau_20_80"]]
        cutoff = g[g["is_cutoff_last10"]]
        final_rows = []
        for _, tg in g.groupby("trajectory_id"):
            final_rows.append(tg.sort_values("end_index").iloc[-1])
        final = pd.DataFrame(final_rows)
        err = g["error"].to_numpy(np.float32)
        rows.append({
            "label_type": label,
            "model_name": model,
            "temperature_C": temperature if temperature is not None else "all",
            "n_windows": int(len(g)),
            "MAE": float(g["abs_error"].mean()),
            "MAE_pct": float(g["abs_error"].mean() * 100.0),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "RMSE_pct": float(np.sqrt(np.mean(err ** 2)) * 100.0),
            "plateau_20_80_MAE": float(plateau["abs_error"].mean()) if len(plateau) else float("nan"),
            "plateau_20_80_MAE_pct": float(plateau["abs_error"].mean() * 100.0) if len(plateau) else float("nan"),
            "cutoff_last10_MAE": float(cutoff["abs_error"].mean()) if len(cutoff) else float("nan"),
            "cutoff_last10_MAE_pct": float(cutoff["abs_error"].mean() * 100.0) if len(cutoff) else float("nan"),
            "final_error_mean": float(final["error"].mean()) if len(final) else float("nan"),
            "final_error_mean_pct": float(final["error"].mean() * 100.0) if len(final) else float("nan"),
            "final_abs_error_mean": float(final["abs_error"].mean()) if len(final) else float("nan"),
            "final_abs_error_mean_pct": float(final["abs_error"].mean() * 100.0) if len(final) else float("nan"),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["label_type", "MAE", "model_name"]).reset_index(drop=True)


def cutoff_exclusion_metrics_for_models(pred_rows):
    if pred_rows.empty:
        return pd.DataFrame()
    rows = []
    conditions = {
        "all_test": lambda g, max_idx: np.ones(len(g), dtype=bool),
        "drop_last_5pct": lambda g, max_idx: g["end_index"].to_numpy() < 0.95 * max_idx,
        "drop_last_10pct": lambda g, max_idx: g["end_index"].to_numpy() < 0.90 * max_idx,
        "drop_V_raw_lt_2p2": lambda g, max_idx: g["V_raw"].to_numpy() >= 2.2,
        "drop_V_raw_lt_2p3": lambda g, max_idx: g["V_raw"].to_numpy() >= 2.3,
    }
    for (label, model, tid), g in pred_rows.groupby(["label_type", "model_name", "trajectory_id"]):
        max_idx = float(g["end_index"].max())
        for condition, fn in conditions.items():
            mask = fn(g, max_idx)
            gg = g.loc[mask]
            if gg.empty:
                continue
            err = gg["error"].to_numpy(np.float32)
            rows.append({
                "label_type": label,
                "model_name": model,
                "trajectory_id": tid,
                "condition": condition,
                "n_windows": int(len(gg)),
                "MAE": float(gg["abs_error"].mean()),
                "MAE_pct": float(gg["abs_error"].mean() * 100.0),
                "RMSE": float(np.sqrt(np.mean(err ** 2))),
                "RMSE_pct": float(np.sqrt(np.mean(err ** 2)) * 100.0),
            })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    return (
        detail.groupby(["label_type", "model_name", "condition"])[["n_windows", "MAE", "MAE_pct", "RMSE", "RMSE_pct"]]
        .agg({"n_windows": "sum", "MAE": "mean", "MAE_pct": "mean", "RMSE": "mean", "RMSE_pct": "mean"})
        .reset_index()
        .sort_values(["label_type", "condition", "MAE", "model_name"])
    )


RBF_TEMPERATURE_CENTERS = (-10.0, 0.0, 10.0, 20.0, 25.0, 30.0, 40.0, 50.0)


def _rbf_col_name(center):
    return f"T_rbf_m{int(abs(center))}" if center < 0 else f"T_rbf_{int(center)}"


def add_temperature_rbf_features(feature_frames, *, centers=RBF_TEMPERATURE_CENTERS, sigma=10.0):
    out = {}
    sigma = max(float(sigma), 1e-6)
    for split, frames in feature_frames.items():
        out_frames = []
        for frame in frames:
            g = frame.copy()
            temp = g["T"].to_numpy(dtype=np.float32)
            for center in centers:
                g[_rbf_col_name(center)] = np.exp(-0.5 * ((temp - float(center)) / sigma) ** 2).astype(np.float32)
            out_frames.append(g)
        out[split] = out_frames
    return out


def temperature_focus_metrics(pred_rows, *, train_temperatures=None):
    if pred_rows.empty:
        return pd.DataFrame()
    train_temperatures = set(float(t) for t in (train_temperatures or []))
    rows = []
    for (label, model, temp), g in pred_rows.groupby(["label_type", "model_name", "temperature_C"]):
        plateau = g[g["is_plateau_20_80"]]
        mid = g[(g["trajectory_fraction"] >= 1 / 3) & (g["trajectory_fraction"] <= 2 / 3)]
        cutoff = g[g["is_cutoff_last10"]]
        err = g["error"].to_numpy(np.float32)
        rows.append({
            "label_type": label,
            "model_name": model,
            "temperature_C": float(temp),
            "is_train_temperature": bool(float(temp) in train_temperatures),
            "n_windows": int(len(g)),
            "MAE": float(g["abs_error"].mean()),
            "MAE_pct": float(g["abs_error"].mean() * 100.0),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "RMSE_pct": float(np.sqrt(np.mean(err ** 2)) * 100.0),
            "plateau_20_80_MAE": float(plateau["abs_error"].mean()) if len(plateau) else float("nan"),
            "plateau_20_80_MAE_pct": float(plateau["abs_error"].mean() * 100.0) if len(plateau) else float("nan"),
            "mid_trajectory_MAE": float(mid["abs_error"].mean()) if len(mid) else float("nan"),
            "mid_trajectory_MAE_pct": float(mid["abs_error"].mean() * 100.0) if len(mid) else float("nan"),
            "cutoff_last10_MAE": float(cutoff["abs_error"].mean()) if len(cutoff) else float("nan"),
            "cutoff_last10_MAE_pct": float(cutoff["abs_error"].mean() * 100.0) if len(cutoff) else float("nan"),
            "mean_error": float(g["error"].mean()),
            "mean_error_pct": float(g["error"].mean() * 100.0),
        })
    return pd.DataFrame(rows).sort_values(["label_type", "model_name", "temperature_C"]).reset_index(drop=True)


def _load_exp_a_comparison_rows(cfg: CFG):
    path = cfg.output_dir / "temp_variant_prediction_rows.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "label_type" in df.columns:
        df = df[df["label_type"] == "physical"].copy()
    keep_models = {"R5_raw_I_T_all_components", "R5_GATED"}
    if "model_name" in df.columns:
        df = df[df["model_name"].isin(keep_models)].copy()
    if df.empty:
        return pd.DataFrame()
    out = temperature_focus_metrics(df, train_temperatures=[-10, 0, 25, 50])
    out["experiment"] = "Exp A"
    out["train_temps"] = "[-10,0,25,50]"
    out["source"] = "temp_variant_prediction_rows.csv"
    return out


def _manual_exp_b_rows():
    rows = []
    for temp, mae, rmse in [
        (10.0, 0.96, 1.35),
        (0.0, 10.54, 14.10),
    ]:
        rows.append({
            "label_type": "physical",
            "model_name": "R5_raw_I_T_all_components",
            "temperature_C": temp,
            "is_train_temperature": bool(temp in {-10.0, 10.0, 25.0, 50.0}),
            "n_windows": np.nan,
            "MAE": mae / 100.0,
            "MAE_pct": mae,
            "RMSE": rmse / 100.0,
            "RMSE_pct": rmse,
            "plateau_20_80_MAE": np.nan,
            "plateau_20_80_MAE_pct": np.nan,
            "mid_trajectory_MAE": np.nan,
            "mid_trajectory_MAE_pct": np.nan,
            "cutoff_last10_MAE": np.nan,
            "cutoff_last10_MAE_pct": np.nan,
            "mean_error": np.nan,
            "mean_error_pct": np.nan,
            "experiment": "Exp B",
            "train_temps": "[-10,10,25,50]",
            "source": "user_observed_summary",
        })
    return pd.DataFrame(rows)


def write_temperature_coverage_comparison(cfg: CFG, exp_c_focus):
    rows = []
    exp_a = _load_exp_a_comparison_rows(cfg)
    if len(exp_a):
        rows.append(exp_a)
    rows.append(_manual_exp_b_rows())
    exp_c = exp_c_focus[exp_c_focus["label_type"] == "physical"].copy()
    exp_c["experiment"] = "Exp C"
    exp_c["train_temps"] = "[-10,0,10,25,50]"
    exp_c["source"] = "train_temp_minus10_0_10_25_50_temp_focus.csv"
    rows.append(exp_c)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    front = ["experiment", "train_temps", "source"]
    cols = front + [c for c in out.columns if c not in front]
    out = out[cols]
    out.to_csv(cfg.output_dir / "train_temp_coverage_comparison.csv", index=False)
    return out


def _run_targeted_lstm_set(feature_frames, cfg: CFG, *, ablation_names, target_labels):
    results_rows = []
    soc_bin_rows = []
    temp_rows = []
    prediction_rows = []
    gate_rows = []
    all_predictions = {}
    feature_lookup = build_prediction_feature_lookup(feature_frames)

    for target_label in target_labels:
        for ablation_name in ablation_names:
            if ablation_name not in ABLATIONS:
                raise ValueError(f"Unknown ablation: {ablation_name}")
            print(f"\n=== targeted target={target_label} | ablation={ablation_name} ===")
            cols = ablation_feature_cols(ablation_name, cfg)
            model, hist, pred_valid, pred_test, scaler, test_loader = train_one_lstm_ablation(
                feature_frames,
                cols,
                target_label,
                cfg,
                ablation_name,
            )
            pred_test = pred_test.assign(split="test", ablation=ablation_name)
            all_predictions[(target_label, ablation_name)] = {"valid": pred_valid, "test": pred_test}
            pred_with_features = attach_prediction_features(
                pred_test,
                feature_lookup,
                ablation_name=ablation_name,
                target_label=target_label,
            )
            prediction_rows.append(pred_with_features)
            results_rows.append(summarize_overall_metrics(pred_test, ablation_name))
            soc_bin_rows.append(metrics_by_soc_bin(pred_test, ablation_name))
            temp_rows.append(metrics_by_group(pred_test, "temperature", ablation_name))
            gate_summary = summarize_component_gates(model, test_loader, target_label, ablation_name, cfg)
            if len(gate_summary):
                gate_rows.append(gate_summary)

    ablation_results = pd.concat(results_rows, ignore_index=True) if results_rows else pd.DataFrame()
    soc_bin = pd.concat(soc_bin_rows, ignore_index=True) if soc_bin_rows else pd.DataFrame()
    by_temp = pd.concat(temp_rows, ignore_index=True) if temp_rows else pd.DataFrame()
    predictions = _combined_predictions_for_csv(prediction_rows)
    gates = pd.concat(gate_rows, ignore_index=True) if gate_rows else pd.DataFrame()
    return {
        "ablation_results": ablation_results,
        "metrics_by_soc_bin": soc_bin,
        "metrics_by_temperature": by_temp,
        "prediction_rows": predictions,
        "component_gate_by_temperature": gates,
        "all_predictions": all_predictions,
    }


def run_train_temp_coverage_experiment(cfg: CFG | None = None, *, make_plots=True):
    cfg = make_cfg() if cfg is None else cfg
    configure_torch_runtime()
    cfg.corrector_variant = "base"
    cfg.train_temps = ("N10", "0", "10", "25", "50")
    cfg.eval_temps = ("N10", "0", "10", "20", "25", "30", "40", "50")
    cfg.train_drives = ("DST", "US06")
    cfg.eval_drive = "FUDS"
    cfg.decomposed_dir = cfg.output_dir / "decomposed_features_train_temp_minus10_0_10_25_50"
    cfg.decomposed_dir.mkdir(parents=True, exist_ok=True)
    cfg.ablation_names_to_run = (
        "A3_V_raw_I_T",
        "R5_raw_I_T_all_components",
        "R5_GATED",
        "R5_GATED_RBF",
    )
    cfg.target_labels_to_run = ("physical",)

    print("Running Exp C: train temps [-10, 0, 10, 25, 50], train DST/US06, test FUDS.")
    data = load_and_prepare_data(cfg)
    corrector = build_corrector(cfg, device)
    corrector_history = run_corrector_pretraining(corrector, data["train_profiles"], cfg, data["v_scaler"])
    feature_frames = extract_all_feature_frames(
        corrector,
        data["train_profiles"],
        data["valid_profiles"],
        data["test_profiles"],
        cfg,
        data["v_scaler"],
    )
    feature_frames = add_temperature_rbf_features(feature_frames)
    outputs = _run_targeted_lstm_set(
        feature_frames,
        cfg,
        ablation_names=cfg.ablation_names_to_run,
        target_labels=cfg.target_labels_to_run,
    )
    results_path = cfg.output_dir / "train_temp_minus10_0_10_25_50_results.csv"
    by_temp_path = cfg.output_dir / "train_temp_minus10_0_10_25_50_by_temperature.csv"
    focus_path = cfg.output_dir / "train_temp_minus10_0_10_25_50_temp_focus.csv"
    prediction_path = cfg.output_dir / "train_temp_minus10_0_10_25_50_prediction_rows.csv"

    outputs["ablation_results"].to_csv(results_path, index=False)
    outputs["metrics_by_temperature"].to_csv(by_temp_path, index=False)
    outputs["prediction_rows"].to_csv(prediction_path, index=False)
    exp_c_focus = temperature_focus_metrics(outputs["prediction_rows"], train_temperatures=[-10, 0, 10, 25, 50])
    exp_c_focus.to_csv(focus_path, index=False)
    comparison = write_temperature_coverage_comparison(cfg, exp_c_focus)

    if len(outputs["component_gate_by_temperature"]):
        outputs["component_gate_by_temperature"].to_csv(
            cfg.output_dir / "train_temp_minus10_0_10_25_50_component_gate_by_temperature.csv",
            index=False,
        )
        plot_component_gate_summary(outputs["component_gate_by_temperature"], cfg)
        _copy_if_exists(
            cfg.output_dir / "component_gate_by_temperature.png",
            cfg.output_dir / "train_temp_minus10_0_10_25_50_component_gate_by_temperature.png",
        )

    print("Exp C focused temperature metrics:")
    display(exp_c_focus)
    print("Temperature coverage comparison:")
    display(comparison)
    return {
        **outputs,
        "corrector_history": corrector_history,
        "feature_frames": feature_frames,
        "temp_focus": exp_c_focus,
        "coverage_comparison": comparison,
    }


def _write_variant_outputs(outputs, cfg: CFG, *, prefix="temp_variant"):
    outputs["ablation_results"].to_csv(cfg.output_dir / f"{prefix}_results.csv", index=False)
    outputs["prediction_rows"].to_csv(cfg.output_dir / f"{prefix}_prediction_rows.csv", index=False)
    # Keep the canonical prediction-row file populated for diagnostics and notebook reuse.
    outputs["prediction_rows"].to_csv(cfg.output_dir / "all_predictions_fixed_labels.csv", index=False)
    temp10 = focus_metrics(outputs["prediction_rows"], temperature=10)
    cold = focus_metrics(outputs["prediction_rows"], temperature=-10)
    overall = focus_metrics(outputs["prediction_rows"], temperature=None)
    cutoff = cutoff_exclusion_metrics_for_models(outputs["prediction_rows"])
    temp10.to_csv(cfg.output_dir / "temp10_focus_metrics.csv", index=False)
    cold.to_csv(cfg.output_dir / "cold_minus10_focus_metrics.csv", index=False)
    overall.to_csv(cfg.output_dir / f"{prefix}_overall_focus_metrics.csv", index=False)
    cutoff.to_csv(cfg.output_dir / f"{prefix}_cutoff_exclusion_test_fixed.csv", index=False)
    gates = outputs["component_gate_by_temperature"]
    gates.to_csv(cfg.output_dir / "component_gate_by_temperature.csv", index=False)
    if len(gates):
        plot_component_gate_summary(gates, cfg)
    return temp10, cold, overall


def run_reused_feature_temperature_variants(cfg: CFG | None = None, *, make_plots=True):
    cfg = make_cfg() if cfg is None else cfg
    configure_torch_runtime()
    feature_frames = load_feature_frame_dict_from_csv(cfg)
    cfg.ablation_names_to_run = (
        "R5_raw_I_T_all_components",
        "R5_GATED",
        "R5_TEMP_HEAD",
        "R5_GATED_TEMP_HEAD",
    )
    cfg.target_labels_to_run = ("physical", "usable")
    outputs = _run_targeted_lstm_set(
        feature_frames,
        cfg,
        ablation_names=cfg.ablation_names_to_run,
        target_labels=cfg.target_labels_to_run,
    )
    temp10, cold, overall = _write_variant_outputs(outputs, cfg, prefix="temp_variant")
    feature_df = pd.concat([f.assign(split="test") for f in feature_frames["test"]], ignore_index=True)
    temp10_error_diagnostic(feature_df, outputs["prediction_rows"], cfg, ablation_for_error="R5_raw_I_T_all_components", make_plots=make_plots)
    print("TEMP_HEAD/GATED focused 10C metrics:")
    display(temp10)
    print("TEMP_HEAD/GATED focused -10C metrics:")
    display(cold)
    return {
        **outputs,
        "temp10_focus_metrics": temp10,
        "cold_minus10_focus_metrics": cold,
        "overall_focus_metrics": overall,
        "feature_frames": feature_frames,
    }


def component_summary_by_temperature(feature_frames, cfg: CFG, *, out_name):
    rows = []
    for frame in feature_frames["test"]:
        temp = float(frame["temperature"].iloc[0])
        row = {
            "temperature_C": temp,
            "trajectory_id": frame["trajectory_id"].iloc[0],
            "drive_cycle": frame["drive_cycle"].iloc[0],
        }
        for col in ["V_corr_raw", "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0"]:
            values = frame[col].to_numpy(dtype=float)
            row[f"{col}_mean"] = float(np.nanmean(values))
            row[f"{col}_var"] = float(np.nanvar(values))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("temperature_C")
    out.to_csv(cfg.output_dir / out_name, index=False)
    return out


def run_temp_tau_temperature_pipeline(cfg: CFG | None = None, *, make_plots=True):
    cfg = make_cfg() if cfg is None else cfg
    configure_torch_runtime()
    cfg.corrector_variant = "temp_tau"
    cfg.decomposed_dir = cfg.output_dir / "decomposed_features_temp_tau"
    cfg.decomposed_dir.mkdir(parents=True, exist_ok=True)
    cfg.ablation_names_to_run = (
        "R5_raw_I_T_all_components",
        "LSTM_R5_TEMP_TAU",
        "R5_TEMP_TAU_GATED",
    )
    cfg.target_labels_to_run = ("physical", "usable")

    print("Running CorrectorTempTau pipeline with fresh corrector training and fresh feature extraction.")
    data = load_and_prepare_data(cfg)
    corrector = build_corrector(cfg, device)
    corrector_history = run_corrector_pretraining(corrector, data["train_profiles"], cfg, data["v_scaler"])
    feature_frames = extract_all_feature_frames(
        corrector,
        data["train_profiles"],
        data["valid_profiles"],
        data["test_profiles"],
        cfg,
        data["v_scaler"],
    )

    component_summary = component_summary_by_temperature(
        feature_frames,
        cfg,
        out_name="temp_tau_component_summary.csv",
    )
    feature_df = pd.concat([f.assign(split="test") for f in feature_frames["test"]], ignore_index=True)
    distance = component_temperature_distance(
        feature_df,
        cfg,
        make_plots=make_plots,
        csv_name="temp_tau_component_temperature_distance.csv",
        heatmap_name="temp_tau_component_temperature_distance_heatmap.png",
    )
    outputs = _run_targeted_lstm_set(
        feature_frames,
        cfg,
        ablation_names=cfg.ablation_names_to_run,
        target_labels=cfg.target_labels_to_run,
    )
    outputs["ablation_results"].to_csv(cfg.output_dir / "temp_tau_ablation_results.csv", index=False)
    outputs["prediction_rows"].to_csv(cfg.output_dir / "temp_tau_prediction_rows.csv", index=False)
    temp10 = focus_metrics(outputs["prediction_rows"], temperature=10)
    temp10.to_csv(cfg.output_dir / "temp_tau_temp10_focus_metrics.csv", index=False)
    cutoff = cutoff_exclusion_metrics_for_models(outputs["prediction_rows"])
    cutoff.to_csv(cfg.output_dir / "temp_tau_cutoff_exclusion_test_fixed.csv", index=False)

    diagnostics = run_shortcut_diagnostics(
        feature_frames,
        outputs["all_predictions"],
        outputs["ablation_results"],
        cfg,
        make_plots=make_plots,
    )
    _copy_if_exists(cfg.output_dir / "same_voltage_soc_spread_fixed.csv", cfg.output_dir / "temp_tau_same_voltage_soc_spread_fixed.csv")
    _copy_if_exists(cfg.output_dir / "same_voltage_error_comparison_fixed.csv", cfg.output_dir / "temp_tau_same_voltage_error_comparison_fixed.csv")

    temp10_error_diagnostic(
        feature_df,
        outputs["prediction_rows"],
        cfg,
        ablation_for_error="R5_TEMP_TAU_GATED",
        make_plots=make_plots,
    )
    _copy_if_exists(cfg.output_dir / "temp10_error_summary.csv", cfg.output_dir / "temp_tau_temp10_error_summary.csv")
    _copy_if_exists(cfg.output_dir / "temp10_error_classification.txt", cfg.output_dir / "temp_tau_temp10_error_classification.txt")

    print("TEMP_TAU component summary:")
    display(component_summary)
    print("TEMP_TAU 10C focus metrics:")
    display(temp10)
    return {
        **outputs,
        "corrector_history": corrector_history,
        "feature_frames": feature_frames,
        "component_summary": component_summary,
        "component_temperature_distance": distance,
        "temp10_focus_metrics": temp10,
        "diagnostics": diagnostics,
    }
