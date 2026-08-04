from collections import OrderedDict
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import CFG
from .runtime import device
from .data import FeatureStandardizer
from .models import DecomposedWindowDataset, build_lstm_soc_model, assert_stateless_lstm, collate_meta_to_frame

# Training, evaluation, metrics, and CSV writers
def _cuda_non_blocking(cfg: CFG):
    return bool(getattr(cfg, "cuda_non_blocking", True)) and device.type == "cuda"


def _move_float(x, cfg: CFG):
    return x.to(device=device, dtype=torch.float32, non_blocking=_cuda_non_blocking(cfg))


def _loader_num_workers(cfg: CFG):
    requested = int(getattr(cfg, "dataloader_num_workers", 0))
    if requested < 0:
        return max(0, min(8, (os.cpu_count() or 1) // 2))
    return max(0, requested)


def make_data_loader(dataset, cfg: CFG, *, shuffle: bool):
    if dataset is None or len(dataset) == 0:
        return None
    num_workers = _loader_num_workers(cfg)
    kwargs = {
        "batch_size": int(cfg.batch_size),
        "shuffle": bool(shuffle),
        "num_workers": num_workers,
        "pin_memory": bool(getattr(cfg, "dataloader_pin_memory", True)) and device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(getattr(cfg, "dataloader_persistent_workers", True))
        kwargs["prefetch_factor"] = int(getattr(cfg, "dataloader_prefetch_factor", 2))
    return DataLoader(dataset, **kwargs)


def make_scaled_frames_for_ablation(feature_frames, feature_cols):
    train_ids = [f["trajectory_id"].iloc[0] for f in feature_frames["train"]]
    scaler = FeatureStandardizer().fit(feature_frames["train"], feature_cols, fit_ids=train_ids)
    valid_ids = {f["trajectory_id"].iloc[0] for f in feature_frames["valid"]}
    test_ids = {f["trajectory_id"].iloc[0] for f in feature_frames["test"]}
    assert scaler.fit_ids.isdisjoint(valid_ids), "Scaler leakage: valid IDs included in fit"
    assert scaler.fit_ids.isdisjoint(test_ids), "Scaler leakage: test IDs included in fit"
    scaled = {
        split: [scaler.transform_frame(f) for f in frames]
        for split, frames in feature_frames.items()
    }
    return scaled, scaler


def train_one_lstm_ablation(feature_frames, feature_cols, target_label, cfg: CFG, ablation_name: str):
    scaled_frames, scaler = make_scaled_frames_for_ablation(feature_frames, feature_cols)
    train_ds = DecomposedWindowDataset(scaled_frames["train"], feature_cols, cfg.window_len, cfg.stride, target_label=target_label)
    test_ds = DecomposedWindowDataset(scaled_frames["test"], feature_cols, cfg.window_len, cfg.stride, target_label=target_label)
    if len(train_ds) == 0:
        raise ValueError("No train windows. Reduce cfg.window_len or check data length.")

    output_dim = 2 if target_label == "multi" else 1
    model = build_lstm_soc_model(feature_cols, output_dim, cfg, ablation_name).to(device)
    assert_stateless_lstm(model)

    train_loader = make_data_loader(train_ds, cfg, shuffle=True)
    test_loader = make_data_loader(test_ds, cfg, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lstm_lr, weight_decay=cfg.lstm_weight_decay)
    history = []
    for ep in range(1, int(cfg.lstm_epochs) + 1):
        model.train()
        losses = []
        for x, y, _ in train_loader:
            x = _move_float(x, cfg)
            y = _move_float(y, cfg)
            pred = model(x)
            loss = F.smooth_l1_loss(pred, y, beta=0.02)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(losses))
        history.append({"epoch": ep, "train_loss": train_loss})
        print_every = max(1, int(getattr(cfg, "lstm_print_every", 1)))
        if ep == 1 or ep == int(cfg.lstm_epochs) or ep % print_every == 0:
            print(f"epoch={ep} train_loss={train_loss:.5f}")
    pred_valid = pd.DataFrame()
    pred_test = predict_loader(model, test_loader, target_label, cfg) if test_loader is not None else pd.DataFrame()
    return model, pd.DataFrame(history), pred_valid, pred_test, scaler, test_loader


@torch.no_grad()
def evaluate_loader_mae(model, loader, cfg: CFG):
    if loader is None:
        return float("nan")
    model.eval()
    errors = []
    for x, y, _ in loader:
        x = _move_float(x, cfg)
        y = _move_float(y, cfg)
        pred = model(x)
        errors.append(torch.abs(pred - y).detach().cpu().numpy())
    if not errors:
        return float("nan")
    return float(np.mean(np.concatenate(errors)))


@torch.no_grad()
def predict_loader(model, loader, target_label, cfg: CFG):
    if loader is None:
        return pd.DataFrame()
    model.eval()
    rows = []
    for x, y, meta in loader:
        pred = model(_move_float(x, cfg)).detach().cpu().numpy()
        yy = y.numpy()
        mdf = collate_meta_to_frame(meta)
        if target_label == "multi":
            for j, name in enumerate(["physical", "usable"]):
                tmp = mdf.copy()
                tmp["target_label"] = name
                tmp["y_true"] = yy[:, j]
                tmp["y_pred"] = pred[:, j]
                rows.append(tmp)
        else:
            mdf["target_label"] = target_label
            mdf["y_true"] = yy[:, 0]
            mdf["y_pred"] = pred[:, 0]
            rows.append(mdf)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out):
        out["error"] = out["y_pred"] - out["y_true"]
        out["abs_error"] = np.abs(out["error"])
    return out


def build_prediction_feature_lookup(feature_frames):
    rows = []
    keep = [
        "trajectory_id", "end_index", "V_raw", "V_corr_raw", "V_pol_raw", "V_hys_raw",
        "V_ohm_raw", "R0", "R0_x_V_pol", "T_x_V_pol", "R0_x_absI", "V_pol_x_abs_dI",
        "cumulative_discharge_Ah", "SOC_physical", "SOC_usable_cutoff",
    ]
    for split, frames in feature_frames.items():
        for frame in frames:
            cols = [c for c in keep if c in frame.columns]
            rows.append(frame[cols].assign(feature_split=split))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=keep + ["feature_split"])


def attach_prediction_features(pred_df, feature_lookup, *, ablation_name=None, target_label=None):
    if pred_df.empty:
        return pred_df.copy()
    out = pred_df.copy()
    if ablation_name is not None:
        out["model_name"] = ablation_name
    elif "ablation" in out.columns:
        out["model_name"] = out["ablation"]
    if target_label is not None:
        out["label_type"] = target_label
    elif "target_label" in out.columns:
        out["label_type"] = out["target_label"]
    out["temperature_C"] = out["temperature"]
    out["time_index"] = out["end_index"]
    keep = [
        "trajectory_id", "end_index", "V_raw", "V_corr_raw", "V_pol_raw", "V_hys_raw",
        "V_ohm_raw", "R0", "R0_x_V_pol", "T_x_V_pol", "R0_x_absI", "V_pol_x_abs_dI",
        "cumulative_discharge_Ah", "SOC_physical", "SOC_usable_cutoff",
    ]
    if feature_lookup is not None and len(feature_lookup):
        lookup = feature_lookup[[c for c in keep if c in feature_lookup.columns]].drop_duplicates(
            ["trajectory_id", "end_index"]
        )
        out = out.merge(lookup, on=["trajectory_id", "end_index"], how="left", validate="many_to_one")
    out["cumulative_Ah"] = out.get("cumulative_discharge_Ah", np.nan)
    out["is_plateau_20_80"] = (out["y_true"] >= 0.2) & (out["y_true"] <= 0.8)
    out["SOC_bin"] = np.select(
        [out["y_true"] < 0.2, out["y_true"] <= 0.8],
        ["0-20", "20-80"],
        default="80-100",
    )
    max_index = out.groupby("trajectory_id")["end_index"].transform("max").replace(0, np.nan)
    out["trajectory_fraction"] = out["end_index"] / max_index
    out["is_cutoff_last10"] = out["trajectory_fraction"] >= 0.9
    required_order = [
        "model_name", "label_type", "trajectory_id", "temperature_C", "drive_cycle",
        "time_index", "end_index", "y_true", "y_pred", "error", "abs_error",
        "V_raw", "V_corr_raw", "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0",
        "R0_x_V_pol", "T_x_V_pol", "R0_x_absI", "V_pol_x_abs_dI",
        "cumulative_Ah", "cumulative_discharge_Ah", "SOC_bin", "is_plateau_20_80",
        "is_cutoff_last10", "target_label", "ablation", "split", "file_name", "soc_bin",
    ]
    ordered = [c for c in required_order if c in out.columns]
    rest = [c for c in out.columns if c not in ordered]
    return out[ordered + rest]


@torch.no_grad()
def summarize_component_gates(model, loader, target_label, ablation_name, cfg: CFG):
    if loader is None or not hasattr(model, "component_gates"):
        return pd.DataFrame()
    model.eval()
    rows = []
    for x, _, meta in loader:
        x_dev = _move_float(x, cfg)
        gates = model.component_gates(x_dev).detach().cpu().numpy()
        gate_window_mean = gates.mean(axis=1)
        mdf = collate_meta_to_frame(meta)
        mdf["target_label"] = target_label
        mdf["ablation"] = ablation_name
        mdf["gate_pol"] = gate_window_mean[:, 0]
        mdf["gate_hys"] = gate_window_mean[:, 1]
        mdf["gate_ohm"] = gate_window_mean[:, 2]
        rows.append(mdf)
    detail = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if detail.empty:
        return detail
    summary = (
        detail.groupby(["target_label", "ablation", "temperature"])[["gate_pol", "gate_hys", "gate_ohm"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(c) for c in col if c != ""]).rstrip("_")
        if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    return summary


def plot_component_gate_summary(gate_df, cfg: CFG):
    if gate_df.empty:
        return
    for (target, ablation), g in gate_df.groupby(["target_label", "ablation"]):
        gg = g.sort_values("temperature")
        plt.figure(figsize=(7, 3))
        plt.plot(gg["temperature"], gg["gate_pol_mean"], marker="o", label="polarization-like gate")
        plt.plot(gg["temperature"], gg["gate_hys_mean"], marker="o", label="hysteresis-like gate")
        plt.plot(gg["temperature"], gg["gate_ohm_mean"], marker="o", label="ohmic-like gate")
        plt.ylim(0.0, 1.0)
        plt.xlabel("temperature (C)")
        plt.ylabel("window-mean gate")
        plt.title(f"Component reliability gate by temperature | {target} | {ablation}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(cfg.output_dir / f"component_gate_by_temperature_{target}_{ablation}.png", dpi=180)
        plt.close()
    latest = gate_df.sort_values(["target_label", "ablation", "temperature"])
    if len(latest):
        plt.figure(figsize=(8, 3))
        for col, label in [
            ("gate_pol_mean", "polarization-like gate"),
            ("gate_hys_mean", "hysteresis-like gate"),
            ("gate_ohm_mean", "ohmic-like gate"),
        ]:
            g = latest.groupby("temperature")[col].mean().reset_index()
            plt.plot(g["temperature"], g[col], marker="o", label=label)
        plt.ylim(0.0, 1.0)
        plt.xlabel("temperature (C)")
        plt.ylabel("mean gate across gated ablations")
        plt.title("Component reliability gate by temperature")
        plt.legend()
        plt.tight_layout()
        plt.savefig(cfg.output_dir / "component_gate_by_temperature.png", dpi=180)
        plt.close()


def summarize_overall_metrics(pred_df, ablation_name):
    if pred_df.empty:
        return pd.DataFrame()
    def mean_abs_in_range(g, lo, hi, *, include_hi=False):
        if include_hi:
            sub = g[(g["y_true"] >= lo) & (g["y_true"] <= hi)]
        else:
            sub = g[(g["y_true"] >= lo) & (g["y_true"] < hi)]
        return float(sub["abs_error"].mean()) if len(sub) else float("nan")

    rows = []
    for label, g in pred_df.groupby("target_label"):
        err = g["error"].to_numpy(np.float32)
        abs_err = np.abs(err)
        plateau = g[(g["y_true"] >= 0.2) & (g["y_true"] <= 0.8)]
        cutoff_rows = []
        for tid, tg in g.groupby("trajectory_id"):
            mx = tg["end_index"].max()
            cutoff_rows.append(tg[tg["end_index"] >= 0.9 * mx])
        cutoff = pd.concat(cutoff_rows, ignore_index=True) if cutoff_rows else pd.DataFrame()
        rows.append({
            "ablation": ablation_name,
            "target_label": label,
            "n_windows": int(len(g)),
            "MAE": float(abs_err.mean()),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "Max_error": float(abs_err.max()),
            "soc_0_20_MAE": mean_abs_in_range(g, 0.0, 0.2),
            "soc_20_80_MAE": mean_abs_in_range(g, 0.2, 0.8, include_hi=True),
            "soc_80_100_MAE": mean_abs_in_range(g, 0.8, 1.000001, include_hi=True),
            "soc_0_20_MAE_pct": mean_abs_in_range(g, 0.0, 0.2) * 100.0,
            "soc_20_80_MAE_pct": mean_abs_in_range(g, 0.2, 0.8, include_hi=True) * 100.0,
            "soc_80_100_MAE_pct": mean_abs_in_range(g, 0.8, 1.000001, include_hi=True) * 100.0,
            "MAE_pct": float(abs_err.mean() * 100.0),
            "RMSE_pct": float(np.sqrt(np.mean(err ** 2)) * 100.0),
            "Max_error_pct": float(abs_err.max() * 100.0),
            "plateau_20_80_MAE": float(plateau["abs_error"].mean()) if len(plateau) else float("nan"),
            "plateau_20_80_MAE_pct": float(plateau["abs_error"].mean() * 100.0) if len(plateau) else float("nan"),
            "cutoff_last_10pct_MAE": float(cutoff["abs_error"].mean()) if len(cutoff) else float("nan"),
            "cutoff_last_10pct_MAE_pct": float(cutoff["abs_error"].mean() * 100.0) if len(cutoff) else float("nan"),
        })
    return pd.DataFrame(rows)


def metrics_by_soc_bin(pred_df, ablation_name):
    if pred_df.empty:
        return pd.DataFrame()
    out = pred_df.copy()
    out["soc_range_bin"] = pd.cut(
        out["y_true"],
        bins=[-1e-6, 0.2, 0.8, 1.000001],
        labels=["0-20%", "20-80% LFP plateau", "80-100%"],
        include_lowest=True,
    )
    return (
        out.groupby(["target_label", "soc_range_bin"], observed=True)["abs_error"]
        .agg(MAE="mean", count="size")
        .reset_index()
        .assign(ablation=ablation_name, MAE_pct=lambda d: d["MAE"] * 100.0)
    )


def metrics_by_group(pred_df, group_col, ablation_name):
    if pred_df.empty:
        return pd.DataFrame()
    return (
        pred_df.groupby(["target_label", group_col])["abs_error"]
        .agg(MAE="mean", count="size")
        .reset_index()
        .assign(ablation=ablation_name, MAE_pct=lambda d: d["MAE"] * 100.0)
    )


def final_soc_error_by_trajectory(pred_df, ablation_name):
    if pred_df.empty:
        return pd.DataFrame()
    rows = []
    for (label, tid), g in pred_df.groupby(["target_label", "trajectory_id"]):
        last = g.sort_values("end_index").iloc[-1]
        rows.append({
            "ablation": ablation_name,
            "target_label": label,
            "trajectory_id": tid,
            "file_name": last["file_name"],
            "temperature": last["temperature"],
            "drive_cycle": last["drive_cycle"],
            "final_true": last["y_true"],
            "final_pred": last["y_pred"],
            "final_error": last["error"],
            "final_error_pct": last["error"] * 100.0,
        })
    return pd.DataFrame(rows)


def cutoff_physical_soc_summary(frames):
    rows = []
    for f in frames:
        last = f.iloc[-1]
        cutoff_physical = float(last["SOC_physical"])
        cutoff_usable = float(last["SOC_usable_cutoff"])
        rows.append({
            "trajectory_id": last["trajectory_id"],
            "file_name": last["file_name"],
            "temperature": last["temperature"],
            "drive_cycle": last["drive_cycle"],
            "cutoff_voltage": float(last["V_raw"]),
            "cutoff_physical_SOC": cutoff_physical,
            "cutoff_usable_SOC": cutoff_usable,
            "physical_minus_usable_at_cutoff": cutoff_physical - cutoff_usable,
            "cumulative_Ah_at_cutoff": float(last["cumulative_discharge_Ah"]),
            "Q_ref_Ah": float(last["Q_ref_Ah"]),
            "physical_not_zero_at_cutoff": bool(abs(cutoff_physical) > 0.02),
        })
    return pd.DataFrame(rows)


# Shortcut-defense ablation definitions and execution
ABLATIONS = OrderedDict({
    "A1_V_raw_only": ["V_raw"],
    "A2_V_corr_only": ["V_corr_raw"],
    "A3_V_raw_I_T": ["V_raw", "I_raw", "T"],
    "A4_V_corr_I_T": ["V_corr_raw", "I_raw", "T"],
    "A5_I_T_only": ["I_raw", "T"],
    "A6_I_T_dI_absI_only": ["I_raw", "T", "dI", "absI"],
    "F1_full_decomposed": ["V_corr_raw", "I_raw", "T", "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0"],
    "F2_raw_plus_full_decomposed": ["V_raw", "V_corr_raw", "I_raw", "T", "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0"],
    "R1_raw_I_T_pol_like": ["V_raw", "I_raw", "T", "V_pol_raw"],
    "R2_raw_I_T_hys_like": ["V_raw", "I_raw", "T", "V_hys_raw"],
    "R3_raw_I_T_ohmic_like": ["V_raw", "I_raw", "T", "V_ohm_raw", "R0"],
    "R4_raw_I_T_pol_hys_like": ["V_raw", "I_raw", "T", "V_pol_raw", "V_hys_raw"],
    "R5_raw_I_T_all_components": ["V_raw", "I_raw", "T", "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0"],
    "R5_GATED": ["V_raw", "V_corr_raw", "I_raw", "T", "dI", "absI", "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0"],
    "LSTM_R5_TEMP_TAU": ["V_raw", "I_raw", "T", "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0"],
    "R5_TEMP_TAU_GATED": ["V_raw", "V_corr_raw", "I_raw", "T", "dI", "absI", "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0"],
    "R5_TEMP_HEAD": ["V_raw", "I_raw", "T", "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0"],
    "R5_GATED_TEMP_HEAD": ["V_raw", "V_corr_raw", "I_raw", "T", "dI", "absI", "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0"],
    "R5_GATED_RBF": [
        "V_raw", "V_corr_raw", "I_raw", "T", "dI", "absI",
        "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0",
        "T_rbf_m10", "T_rbf_0", "T_rbf_10", "T_rbf_20",
        "T_rbf_25", "T_rbf_30", "T_rbf_40", "T_rbf_50",
    ],
})


def ablation_feature_cols(name, cfg: CFG):
    cols = list(ABLATIONS[name])
    if cfg.include_dI_absI_in_ablation_windows:
        for c in ["dI", "absI"]:
            if c not in cols:
                cols.append(c)
    return cols


def run_ablation_experiments(feature_frames, cfg: CFG):
    results_rows = []
    soc_bin_rows = []
    temp_rows = []
    cycle_rows = []
    final_rows = []
    gate_rows = []
    prediction_rows = []
    all_predictions = {}
    all_histories = {}
    feature_lookup = build_prediction_feature_lookup(feature_frames)

    missing_ab = [name for name in cfg.ablation_names_to_run if name not in ABLATIONS]
    assert not missing_ab, f"Unknown ablations requested: {missing_ab}"

    for target_label in cfg.target_labels_to_run:
        for ablation_name in cfg.ablation_names_to_run:
            print("\n=== target:", target_label, "| ablation:", ablation_name, "===")
            if "TEMP_TAU" in ablation_name and str(getattr(cfg, "corrector_variant", "base")).lower() not in {"temp_tau", "correctortemptau", "temperature_tau"}:
                warnings.warn(
                    f"{ablation_name} is named as a TEMP_TAU model, but cfg.corrector_variant="
                    f"{getattr(cfg, 'corrector_variant', 'base')!r}. Run a separate cfg.corrector_variant='temp_tau' pipeline for a true TEMP_TAU comparison."
                )
            cols = ablation_feature_cols(ablation_name, cfg)
            model_lstm, hist, pred_valid, pred_test, scaler, test_loader = train_one_lstm_ablation(
                feature_frames, cols, target_label, cfg, ablation_name
            )
            all_histories[(target_label, ablation_name)] = hist
            pred_test = pred_test.assign(split="test", ablation=ablation_name)
            pred_valid = pred_valid.assign(split="valid", ablation=ablation_name) if len(pred_valid) else pred_valid
            all_predictions[(target_label, ablation_name)] = {"valid": pred_valid, "test": pred_test}
            if bool(getattr(cfg, "save_prediction_rows", True)) and len(pred_test):
                prediction_rows.append(
                    attach_prediction_features(
                        pred_test,
                        feature_lookup,
                        ablation_name=ablation_name,
                        target_label=target_label,
                    )
                )
            if bool(getattr(cfg, "gate_summary_enabled", True)):
                gate_summary = summarize_component_gates(model_lstm, test_loader, target_label, ablation_name, cfg)
                if len(gate_summary):
                    gate_rows.append(gate_summary)

            met = summarize_overall_metrics(pred_test, ablation_name)
            results_rows.append(met)
            soc_bin_rows.append(metrics_by_soc_bin(pred_test, ablation_name))
            temp_rows.append(metrics_by_group(pred_test, "temperature", ablation_name))
            cycle_rows.append(metrics_by_group(pred_test, "drive_cycle", ablation_name))
            final_rows.append(final_soc_error_by_trajectory(pred_test, ablation_name))

    ablation_results = pd.concat(results_rows, ignore_index=True) if results_rows else pd.DataFrame()
    metrics_by_soc_bin_df = pd.concat(soc_bin_rows, ignore_index=True) if soc_bin_rows else pd.DataFrame()
    metrics_by_temperature_df = pd.concat(temp_rows, ignore_index=True) if temp_rows else pd.DataFrame()
    metrics_by_cycle_df = pd.concat(cycle_rows, ignore_index=True) if cycle_rows else pd.DataFrame()
    final_soc_error_df = pd.concat(final_rows, ignore_index=True) if final_rows else pd.DataFrame()
    component_gate_df = pd.concat(gate_rows, ignore_index=True) if gate_rows else pd.DataFrame()
    all_prediction_rows_df = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()

    ablation_results.to_csv(cfg.output_dir / "ablation_results_fixed_labels.csv", index=False)
    ablation_results.to_csv(cfg.output_dir / "ablation_results.csv", index=False)
    ablation_results[ablation_results["target_label"] == "physical"].to_csv(
        cfg.output_dir / "physical_soc_ablation_results.csv", index=False
    )
    ablation_results[ablation_results["target_label"] == "usable"].to_csv(
        cfg.output_dir / "usable_cutoff_ablation_results.csv", index=False
    )
    metrics_by_soc_bin_df.to_csv(cfg.output_dir / "metrics_by_soc_bin_fixed_labels.csv", index=False)
    metrics_by_soc_bin_df.to_csv(cfg.output_dir / "metrics_by_soc_bin.csv", index=False)
    metrics_by_temperature_df.to_csv(cfg.output_dir / "metrics_by_temperature_fixed_labels.csv", index=False)
    metrics_by_temperature_df.to_csv(cfg.output_dir / "metrics_by_temperature.csv", index=False)
    metrics_by_cycle_df.to_csv(cfg.output_dir / "metrics_by_cycle_fixed_labels.csv", index=False)
    metrics_by_cycle_df.to_csv(cfg.output_dir / "metrics_by_cycle.csv", index=False)
    final_soc_error_df.to_csv(cfg.output_dir / "trajectory_final_soc_error_fixed_labels.csv", index=False)
    final_soc_error_df.to_csv(cfg.output_dir / "trajectory_final_soc_error.csv", index=False)
    component_gate_df.to_csv(cfg.output_dir / "component_gate_by_temperature.csv", index=False)
    if len(component_gate_df):
        plot_component_gate_summary(component_gate_df, cfg)
    if bool(getattr(cfg, "save_prediction_rows", True)) and len(all_prediction_rows_df):
        all_prediction_rows_df.to_csv(cfg.output_dir / "all_predictions_fixed_labels.csv", index=False)

    voltage_only_names = ["A1_V_raw_only", "A2_V_corr_only"]
    current_only_names = ["A5_I_T_only", "A6_I_T_dI_absI_only"]
    full_names = ["F1_full_decomposed", "F2_raw_plus_full_decomposed"]

    voltage_only_baseline_table = ablation_results[
        ablation_results["ablation"].isin(voltage_only_names + ["A3_V_raw_I_T", "A4_V_corr_I_T"])
    ].copy()
    current_only_baseline_table = ablation_results[ablation_results["ablation"].isin(current_only_names)].copy()
    full_decomposed_ablation_table = ablation_results[ablation_results["ablation"].isin(full_names)].copy()
    plateau_20_80_mae_table = ablation_results[[
        "target_label", "ablation", "plateau_20_80_MAE", "plateau_20_80_MAE_pct",
        "MAE_pct", "RMSE_pct", "Max_error_pct"
    ]].copy()
    cutoff_last10_mae_table = ablation_results[[
        "target_label", "ablation", "cutoff_last_10pct_MAE", "cutoff_last_10pct_MAE_pct",
        "MAE_pct", "RMSE_pct", "Max_error_pct"
    ]].copy()

    voltage_only_baseline_table.to_csv(cfg.output_dir / "voltage_only_baseline_table.csv", index=False)
    current_only_baseline_table.to_csv(cfg.output_dir / "current_only_baseline_table.csv", index=False)
    full_decomposed_ablation_table.to_csv(cfg.output_dir / "full_decomposed_ablation_table.csv", index=False)
    plateau_20_80_mae_table.to_csv(cfg.output_dir / "plateau_20_80_mae_table_fixed.csv", index=False)
    plateau_20_80_mae_table.to_csv(cfg.output_dir / "plateau_20_80_mae_table.csv", index=False)
    cutoff_last10_mae_table.to_csv(cfg.output_dir / "cutoff_last10_mae_table_fixed.csv", index=False)

    core_compare = ablation_results[
        ablation_results["ablation"].isin(voltage_only_names + current_only_names + full_names)
    ].copy()

    return {
        "all_predictions": all_predictions,
        "all_histories": all_histories,
        "ablation_results": ablation_results,
        "metrics_by_soc_bin_df": metrics_by_soc_bin_df,
        "metrics_by_temperature_df": metrics_by_temperature_df,
        "metrics_by_cycle_df": metrics_by_cycle_df,
        "final_soc_error_df": final_soc_error_df,
        "component_gate_df": component_gate_df,
        "voltage_only_baseline_table": voltage_only_baseline_table,
        "current_only_baseline_table": current_only_baseline_table,
        "full_decomposed_ablation_table": full_decomposed_ablation_table,
        "plateau_20_80_mae_table": plateau_20_80_mae_table,
        "cutoff_last10_mae_table": cutoff_last10_mae_table,
        "core_compare": core_compare,
    }
