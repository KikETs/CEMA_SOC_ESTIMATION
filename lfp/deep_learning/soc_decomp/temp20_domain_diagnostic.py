from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import CFG, make_cfg
from .runtime import configure_torch_runtime, device
from .data import FeatureStandardizer, load_and_prepare_data
from .corrector import run_corrector_pretraining
from .features import extract_all_feature_frames
from .models import (
    DecomposedWindowDataset,
    build_corrector,
    build_lstm_soc_model,
    collate_meta_to_frame,
)
from .training import (
    ABLATIONS,
    attach_prediction_features,
    build_prediction_feature_lookup,
    make_data_loader,
    make_scaled_frames_for_ablation,
    train_one_lstm_ablation,
)
from .variance_control import (
    R5_GATED_FEATURES,
    _summary_with_temp20_focus,
    load_feature_frame_dict_from_csv,
    train_variance_model_from_spec,
)

try:
    from IPython.display import display
except Exception:
    display = print


FEATURE_SETS = {
    "S1_raw_VIT": ["V_raw", "I_raw", "T"],
    "S2_decomp": ["V_corr_raw", "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0"],
    "S3_raw_plus_decomp": ["V_raw", "I_raw", "T", "V_corr_raw", "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0"],
}


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


def exp_d_cfg(cfg: CFG | None = None) -> CFG:
    out = exp_c_cfg(cfg)
    out.train_temps = ("N10", "0", "10", "20", "25", "50")
    out.decomposed_dir = out.output_dir / "decomposed_features_train_temp_minus10_0_10_20_25_50"
    out.decomposed_dir.mkdir(parents=True, exist_ok=True)
    return out


def load_exp_c_features(cfg: CFG | None = None):
    cfg = exp_c_cfg(cfg)
    decomposed_dir = cfg.output_dir / "decomposed_features_train_temp_minus10_0_10_25_50"
    return load_feature_frame_dict_from_csv(cfg, decomposed_dir=decomposed_dir)


def concat_feature_frames(feature_frames):
    rows = []
    for split, frames in feature_frames.items():
        for frame in frames:
            rows.append(frame.assign(split=split))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def add_domain_bins(df):
    out = df.copy()
    out["temperature_C"] = out["temperature"].astype(float)
    out["SOC_bin"] = pd.cut(
        out["SOC_physical"].astype(float),
        bins=[-np.inf, 0.2, 0.8, np.inf],
        labels=["0-20", "20-80", "80-100"],
    ).astype(str)
    max_idx = out.groupby("trajectory_id")["end_index"].transform("max").replace(0, np.nan)
    frac = (out["end_index"] / max_idx).fillna(0.0)
    out["trajectory_fraction"] = frac
    out["phase_bin"] = pd.cut(
        frac,
        bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=["early", "mid", "late"],
    ).astype(str)
    q = out["absI"].quantile([0.5, 0.9, 0.98]).to_numpy()
    out["abs_dI"] = np.abs(out["dI"].astype(float))
    d = out["abs_dI"].to_numpy()
    qd = np.nanquantile(d, [0.5, 0.9, 0.98])
    out["dI_event_bin"] = pd.cut(
        out["abs_dI"],
        bins=[-np.inf, qd[0], qd[1], qd[2], np.inf],
        labels=["low", "mid", "high", "extreme"],
    ).astype(str)
    out["absI_bin"] = pd.cut(
        out["absI"].astype(float),
        bins=[-np.inf, q[0], q[1], q[2], np.inf],
        labels=["low", "mid", "high", "extreme"],
    ).astype(str)
    return out


def _standardize_train_test(train, test):
    mu = np.nanmean(train, axis=0)
    sd = np.nanstd(train, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (train - mu) / sd, (test - mu) / sd, mu, sd


def _cov_inv(x):
    cov = np.cov(x, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    reg = 1e-3 * np.trace(cov) / max(1, cov.shape[0])
    cov = cov + np.eye(cov.shape[0]) * max(reg, 1e-6)
    return np.linalg.pinv(cov)


def _mahalanobis(x, mu, inv_cov):
    delta = x - mu
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, inv_cov, delta), 0.0))


def _rbf_mmd(x, y, max_samples=1200):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    rng = np.random.default_rng()
    if len(x) > max_samples:
        x = x[rng.choice(len(x), max_samples, replace=False)]
    if len(y) > max_samples:
        y = y[rng.choice(len(y), max_samples, replace=False)]
    z = np.vstack([x, y])
    if len(z) > 500:
        zz = z[rng.choice(len(z), 500, replace=False)]
    else:
        zz = z
    pd2 = np.sum((zz[:, None, :] - zz[None, :, :]) ** 2, axis=-1)
    med = np.median(pd2[pd2 > 0]) if np.any(pd2 > 0) else 1.0
    gamma = 1.0 / max(med, 1e-6)
    kxx = np.exp(-gamma * np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)).mean()
    kyy = np.exp(-gamma * np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1)).mean()
    kxy = np.exp(-gamma * np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)).mean()
    return float(kxx + kyy - 2.0 * kxy)


def _feature_distance_rows(df, feature_set_name, cols, *, split_filter=True):
    use_cols = [c for c in cols if c in df.columns]
    train_df = df[df["split"].eq("train")].copy()
    test20 = df[(df["split"].eq("test")) & np.isclose(df["temperature_C"].astype(float), 20.0)].copy()
    train_x = train_df[use_cols].to_numpy(np.float64)
    test_x = test20[use_cols].to_numpy(np.float64)
    train_z, test_z, _, _ = _standardize_train_test(train_x, test_x)
    centroid_rows = []
    centroids = []
    for temp, g in train_df.groupby("temperature_C"):
        z = train_z[g.index.to_numpy() - train_df.index.min()] if False else None
    # Avoid relying on original index ordering after filtering.
    train_df = train_df.reset_index(drop=True)
    test20 = test20.reset_index(drop=True)
    train_x = train_df[use_cols].to_numpy(np.float64)
    test_x = test20[use_cols].to_numpy(np.float64)
    train_z, test_z, _, _ = _standardize_train_test(train_x, test_x)
    temp_to_idx = {}
    for temp, idx in train_df.groupby("temperature_C").groups.items():
        idx = np.asarray(list(idx), dtype=int)
        temp_to_idx[float(temp)] = idx
        c = train_z[idx].mean(axis=0)
        centroids.append((float(temp), c))
    centroid_stack = np.stack([c for _, c in centroids], axis=0)
    dmat = np.sqrt(((test_z[:, None, :] - centroid_stack[None, :, :]) ** 2).sum(axis=-1))
    nearest_idx = dmat.argmin(axis=1)
    mu = train_z.mean(axis=0)
    inv_cov = _cov_inv(train_z)
    maha = _mahalanobis(test_z, mu, inv_cov)
    out = test20[[
        "trajectory_id", "drive_cycle", "temperature_C", "end_index", "SOC_physical",
        "SOC_bin", "phase_bin", "dI_event_bin", "absI_bin", "trajectory_fraction",
    ]].copy()
    out["feature_set"] = feature_set_name
    out["nearest_train_centroid_temp"] = [centroids[i][0] for i in nearest_idx]
    out["nearest_train_centroid_distance"] = dmat[np.arange(len(dmat)), nearest_idx]
    out["mahalanobis_distance"] = maha
    for temp, c in centroids:
        out[f"centroid_distance_to_{temp:g}C"] = np.sqrt(((test_z - c) ** 2).sum(axis=1))
    for train_temp in [10.0, 25.0]:
        idx = temp_to_idx.get(train_temp)
        out[f"mmd_to_{train_temp:g}C_global"] = _rbf_mmd(test_z, train_z[idx]) if idx is not None else np.nan
    return out


def _aggregate_distance(distance_rows, group_cols):
    metric_cols = [
        "nearest_train_centroid_distance",
        "mahalanobis_distance",
        "mmd_to_10C_global",
        "mmd_to_25C_global",
    ]
    rows = []
    for keys, g in distance_rows.groupby(["feature_set", *group_cols]):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {"feature_set": keys[0]}
        for col, val in zip(group_cols, keys[1:]):
            row[col] = val
        row["n"] = int(len(g))
        for m in metric_cols:
            if m in g:
                row[f"{m}_mean"] = float(g[m].mean())
                row[f"{m}_p90"] = float(g[m].quantile(0.9))
        rows.append(row)
    return pd.DataFrame(rows)


def _train_latent_gated_model(feature_frames, cfg: CFG):
    feature_cols = ABLATIONS["R5_GATED"]
    scaled, _ = make_scaled_frames_for_ablation(feature_frames, feature_cols)
    train_ds = DecomposedWindowDataset(scaled["train"], feature_cols, cfg.window_len, cfg.stride, target_label="physical")
    test_ds = DecomposedWindowDataset(scaled["test"], feature_cols, cfg.window_len, cfg.stride, target_label="physical")
    train_loader = make_data_loader(train_ds, cfg, shuffle=True)
    test_loader = make_data_loader(test_ds, cfg, shuffle=False)
    model = build_lstm_soc_model(feature_cols, 1, cfg, "R5_GATED").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lstm_lr, weight_decay=cfg.lstm_weight_decay)
    for ep in range(1, int(cfg.lstm_epochs) + 1):
        model.train()
        losses = []
        for x, y, _ in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            y = y.to(device=device, dtype=torch.float32, non_blocking=True)
            pred = model(x)
            loss = F.smooth_l1_loss(pred, y, beta=0.02)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print_every = max(1, int(getattr(cfg, "lstm_print_every", 25)))
        if ep == 1 or ep == int(cfg.lstm_epochs) or ep % print_every == 0:
            print(f"R5_GATED latent model epoch={ep} train_loss={np.mean(losses):.5f}")
    return model, feature_cols, scaled, train_loader, test_loader


@torch.no_grad()
def _latent_frame(model, loader, cfg: CFG):
    rows = []
    h_rows = []
    model.eval()
    for x, y, meta in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        if hasattr(model, "apply_component_gates"):
            x_eff, _ = model.apply_component_gates(x)
        else:
            x_eff = x
        h = model.encode_last(x_eff).detach().cpu().numpy()
        pred = model(x).detach().cpu().numpy()[:, 0]
        yy = y.numpy()[:, 0]
        mdf = collate_meta_to_frame(meta)
        mdf["y_true"] = yy
        mdf["y_pred"] = pred
        mdf["error"] = pred - yy
        mdf["abs_error"] = np.abs(mdf["error"])
        rows.append(mdf)
        h_rows.append(h)
    meta_df = pd.concat(rows, ignore_index=True)
    h = np.vstack(h_rows)
    return meta_df, h


def _latent_distance_rows(model, train_loader, test_loader, cfg: CFG):
    train_meta, train_h = _latent_frame(model, train_loader, cfg)
    test_meta, test_h = _latent_frame(model, test_loader, cfg)
    train_meta["temperature_C"] = train_meta["temperature"].astype(float)
    test_meta["temperature_C"] = test_meta["temperature"].astype(float)
    test20 = test_meta[np.isclose(test_meta["temperature_C"], 20.0)].reset_index(drop=True)
    test20_h = test_h[np.isclose(test_meta["temperature_C"].to_numpy(float), 20.0)]
    train_z, test_z, _, _ = _standardize_train_test(train_h, test20_h)
    train_meta = train_meta.reset_index(drop=True)
    centroids = []
    temp_to_idx = {}
    for temp, idx in train_meta.groupby("temperature_C").groups.items():
        idx = np.asarray(list(idx), dtype=int)
        temp_to_idx[float(temp)] = idx
        centroids.append((float(temp), train_z[idx].mean(axis=0)))
    centroid_stack = np.stack([c for _, c in centroids], axis=0)
    dmat = np.sqrt(((test_z[:, None, :] - centroid_stack[None, :, :]) ** 2).sum(axis=-1))
    nearest_idx = dmat.argmin(axis=1)
    inv_cov = _cov_inv(train_z)
    out = test20[["trajectory_id", "drive_cycle", "temperature_C", "end_index"]].copy()
    out["feature_set"] = "S4_R5_GATED_latent"
    out["nearest_train_centroid_temp"] = [centroids[i][0] for i in nearest_idx]
    out["nearest_train_centroid_distance"] = dmat[np.arange(len(dmat)), nearest_idx]
    out["mahalanobis_distance"] = _mahalanobis(test_z, train_z.mean(axis=0), inv_cov)
    for train_temp in [10.0, 25.0]:
        idx = temp_to_idx.get(train_temp)
        out[f"mmd_to_{train_temp:g}C_global"] = _rbf_mmd(test_z, train_z[idx]) if idx is not None else np.nan
    return out


def plot_distance_heatmap(by_soc_phase, cfg: CFG):
    data = by_soc_phase[by_soc_phase["feature_set"].eq("S3_raw_plus_decomp")].copy()
    if data.empty:
        return
    pivot = data.pivot_table(
        index="SOC_bin",
        columns="phase_bin",
        values="mahalanobis_distance_mean",
        aggfunc="mean",
    )
    plt.figure(figsize=(6, 3.6))
    im = plt.imshow(pivot.to_numpy(), aspect="auto", cmap="magma")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(im, label="mean Mahalanobis distance")
    plt.title("20C FUDS train-manifold distance | S3")
    plt.tight_layout()
    plt.savefig(cfg.output_dir / "temp20_distance_heatmap.png", dpi=180)
    plt.close()


def run_temp20_train_manifold_distance_diagnostic(cfg: CFG | None = None, *, include_latent=True):
    cfg = exp_c_cfg(cfg)
    configure_torch_runtime()
    feature_frames = load_exp_c_features(cfg)
    df = add_domain_bins(concat_feature_frames(feature_frames))
    rows = []
    for name, cols in FEATURE_SETS.items():
        rows.append(_feature_distance_rows(df, name, cols))
    distance = pd.concat(rows, ignore_index=True)
    if include_latent:
        model, _, _, train_loader, test_loader = _train_latent_gated_model(feature_frames, cfg)
        latent = _latent_distance_rows(model, train_loader, test_loader, cfg)
        bins = df[["trajectory_id", "end_index", "SOC_physical", "SOC_bin", "phase_bin", "dI_event_bin", "absI_bin", "trajectory_fraction"]]
        latent = latent.merge(bins, on=["trajectory_id", "end_index"], how="left")
        distance = pd.concat([distance, latent], ignore_index=True)
    distance.to_csv(cfg.output_dir / "temp20_train_manifold_distance.csv", index=False)
    by_soc = _aggregate_distance(distance, ["SOC_bin"])
    by_phase = _aggregate_distance(distance, ["phase_bin"])
    by_event = _aggregate_distance(distance, ["dI_event_bin"])
    by_soc_phase = _aggregate_distance(distance, ["SOC_bin", "phase_bin"])
    by_soc.to_csv(cfg.output_dir / "temp20_distance_by_soc_bin.csv", index=False)
    by_phase.to_csv(cfg.output_dir / "temp20_distance_by_phase.csv", index=False)
    by_event.to_csv(cfg.output_dir / "temp20_distance_by_dI_event_bin.csv", index=False)
    by_soc_phase.to_csv(cfg.output_dir / "temp20_distance_by_soc_phase.csv", index=False)
    plot_distance_heatmap(by_soc_phase, cfg)
    print("20C train-manifold distance by SOC bin:")
    display(by_soc)
    return {
        "distance": distance,
        "by_soc": by_soc,
        "by_phase": by_phase,
        "by_event": by_event,
        "by_soc_phase": by_soc_phase,
        "feature_frames": feature_frames,
    }


def _train_single_model_prediction(feature_frames, cfg, model_name, spec=None):
    feature_lookup = build_prediction_feature_lookup(feature_frames)
    if spec is None:
        model, hist, _, pred_test, _, test_loader = train_one_lstm_ablation(
            feature_frames,
            ABLATIONS[model_name],
            "physical",
            cfg,
            model_name,
        )
    else:
        model, hist, pred_test, gates = train_variance_model_from_spec(feature_frames, model_name, spec, cfg)
        test_loader = None
    pred_test = pred_test.assign(split="test", ablation=model_name)
    pred = attach_prediction_features(pred_test, feature_lookup, ablation_name=model_name, target_label="physical")
    return model, hist, pred, test_loader


@torch.no_grad()
def _gate_entropy_rows(model, loader, cfg: CFG):
    if loader is None or not hasattr(model, "component_gates"):
        return pd.DataFrame()
    rows = []
    model.eval()
    for x, _, meta in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        gates = model.component_gates(x)
        if gates is None:
            continue
        g = gates[:, -1, :].detach().cpu().numpy()
        ent = -(g * np.log(g + 1e-8) + (1.0 - g) * np.log(1.0 - g + 1e-8)).mean(axis=1)
        mdf = collate_meta_to_frame(meta)
        mdf["gate_entropy"] = ent
        rows.append(mdf[["trajectory_id", "end_index", "gate_entropy"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _ensemble_aggregate(member_rows):
    pred = pd.concat(member_rows, ignore_index=True)
    keys = ["model_name", "trajectory_id", "temperature_C", "drive_cycle", "end_index"]
    agg = (
        pred.groupby(keys)
        .agg(
            y_true=("y_true", "first"),
            y_pred_mean=("y_pred", "mean"),
            pred_variance=("y_pred", "var"),
            n_members=("y_pred", "count"),
            V_raw=("V_raw", "first"),
            V_corr_raw=("V_corr_raw", "first"),
            V_pol_raw=("V_pol_raw", "first"),
            V_hys_raw=("V_hys_raw", "first"),
            V_ohm_raw=("V_ohm_raw", "first"),
            R0=("R0", "first"),
            is_plateau_20_80=("is_plateau_20_80", "first"),
            is_cutoff_last10=("is_cutoff_last10", "first"),
            trajectory_fraction=("trajectory_fraction", "first"),
        )
        .reset_index()
    )
    agg["pred_variance"] = agg["pred_variance"].fillna(0.0)
    agg["error"] = agg["y_pred_mean"] - agg["y_true"]
    agg["abs_error"] = np.abs(agg["error"])
    return agg


def run_temp20_uncertainty_diagnostic(cfg: CFG | None = None, *, ensemble_size=3):
    cfg = exp_c_cfg(cfg)
    configure_torch_runtime()
    feature_frames = load_exp_c_features(cfg)
    model_specs = [
        ("R5_raw_I_T_all_components", None),
        ("R5_GATED", None),
        ("R5_GATED_AUG_np01_dp1_lp05", {
            "features": R5_GATED_FEATURES,
            "kind": "gated_seq",
            "lambda_gate_smooth": 0.03,
            "lambda_aug": 0.05,
            "component_noise_std": 0.01,
            "component_dropout_p": 0.10,
        }),
    ]
    member_rows = []
    gate_rows = []
    for member in range(int(ensemble_size)):
        for model_name, spec in model_specs:
            print(f"\n=== uncertainty ensemble member={member + 1} | model={model_name} ===")
            model, _, pred, test_loader = _train_single_model_prediction(feature_frames, cfg, model_name, spec=spec)
            pred["ensemble_member"] = member
            member_rows.append(pred)
            if model_name == "R5_GATED":
                ge = _gate_entropy_rows(model, test_loader, cfg)
                if len(ge):
                    ge["model_name"] = model_name
                    ge["ensemble_member"] = member
                    gate_rows.append(ge)
    agg = _ensemble_aggregate(member_rows)
    gate = pd.concat(gate_rows, ignore_index=True) if gate_rows else pd.DataFrame()
    if len(gate):
        gate_agg = gate.groupby(["model_name", "trajectory_id", "end_index"])["gate_entropy"].mean().reset_index()
        agg = agg.merge(gate_agg, on=["model_name", "trajectory_id", "end_index"], how="left")
    distance_path = cfg.output_dir / "temp20_train_manifold_distance.csv"
    if distance_path.exists():
        dist = pd.read_csv(distance_path)
        s3 = dist[dist["feature_set"].eq("S3_raw_plus_decomp")][[
            "trajectory_id", "end_index", "nearest_train_centroid_distance", "mahalanobis_distance",
        ]].copy()
        s3 = s3.rename(columns={
            "nearest_train_centroid_distance": "S3_nearest_train_centroid_distance",
            "mahalanobis_distance": "S3_mahalanobis_distance",
        })
        agg = agg.merge(s3, on=["trajectory_id", "end_index"], how="left")
    temp20 = agg[np.isclose(agg["temperature_C"].astype(float), 20.0)].copy()
    rows = []
    for model, g in temp20.groupby("model_name"):
        row = {
            "model_name": model,
            "n_points": int(len(g)),
            "MAE_pct": float(g["abs_error"].mean() * 100.0),
            "RMSE_pct": float(np.sqrt(np.mean(g["error"].to_numpy(float) ** 2)) * 100.0),
            "prediction_variance_mean": float(g["pred_variance"].mean()),
            "corr_abs_error_pred_variance": float(g[["abs_error", "pred_variance"]].corr().iloc[0, 1]),
        }
        for col in ["S3_nearest_train_centroid_distance", "S3_mahalanobis_distance", "gate_entropy"]:
            if col in g and g[col].notna().sum() > 2:
                row[f"corr_abs_error_{col}"] = float(g[["abs_error", col]].corr().iloc[0, 1])
                row[f"{col}_mean"] = float(g[col].mean())
        rows.append(row)
    summary = pd.DataFrame(rows)
    temp20.to_csv(cfg.output_dir / "temp20_uncertainty_rows.csv", index=False)
    summary.to_csv(cfg.output_dir / "temp20_uncertainty_diagnostic.csv", index=False)
    _plot_uncertainty(temp20, cfg)
    print("20C uncertainty diagnostic:")
    display(summary)
    return {"rows": temp20, "summary": summary}


def _plot_uncertainty(temp20, cfg: CFG):
    if temp20.empty:
        return
    plt.figure(figsize=(6, 4))
    for model, g in temp20.groupby("model_name"):
        plt.scatter(g["pred_variance"], g["abs_error"], s=5, alpha=0.25, label=model)
    plt.xlabel("prediction variance")
    plt.ylabel("absolute error")
    plt.title("20C uncertainty vs error")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(cfg.output_dir / "uncertainty_vs_error.png", dpi=180)
    plt.close()
    if "S3_mahalanobis_distance" in temp20:
        plt.figure(figsize=(6, 4))
        for model, g in temp20.groupby("model_name"):
            plt.scatter(g["S3_mahalanobis_distance"], g["abs_error"], s=5, alpha=0.25, label=model)
        plt.xlabel("S3 train-manifold Mahalanobis distance")
        plt.ylabel("absolute error")
        plt.title("20C train-manifold distance vs error")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(cfg.output_dir / "distance_vs_error.png", dpi=180)
        plt.close()


def run_expD_train_with_20_experiment(cfg: CFG | None = None):
    cfg = exp_d_cfg(cfg)
    configure_torch_runtime()
    print("Running Exp D: train temps [-10, 0, 10, 20, 25, 50], train DST/US06, test FUDS.")
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
    feature_lookup = build_prediction_feature_lookup(feature_frames)
    pred_rows = []
    specs = [
        ("R5_raw_I_T_all_components", None),
        ("R5_GATED", None),
        ("R5_GATED_AUG_np01_dp1_lp05", {
            "features": R5_GATED_FEATURES,
            "kind": "gated_seq",
            "lambda_gate_smooth": 0.03,
            "lambda_aug": 0.05,
            "component_noise_std": 0.01,
            "component_dropout_p": 0.10,
        }),
    ]
    for model_name, spec in specs:
        print(f"\n=== Exp D model={model_name} ===")
        if spec is None:
            _, hist, _, pred_test, _, _ = train_one_lstm_ablation(
                feature_frames,
                ABLATIONS[model_name],
                "physical",
                cfg,
                model_name,
            )
        else:
            _, hist, pred_test, _ = train_variance_model_from_spec(feature_frames, model_name, spec, cfg)
        pred_test = pred_test.assign(split="test", ablation=model_name)
        pred_rows.append(attach_prediction_features(pred_test, feature_lookup, ablation_name=model_name, target_label="physical"))
    pred = pd.concat(pred_rows, ignore_index=True)
    summary, by_temp, focus, temp20 = _summary_with_temp20_focus(pred)
    pred.to_csv(cfg.output_dir / "expD_train_with_20_prediction_rows.csv", index=False)
    summary.to_csv(cfg.output_dir / "expD_train_with_20_results.csv", index=False)
    by_temp.to_csv(cfg.output_dir / "expD_by_temperature.csv", index=False)
    temp20.to_csv(cfg.output_dir / "expD_temp20_jitter.csv", index=False)
    focus.to_csv(cfg.output_dir / "expD_focus_metrics.csv", index=False)
    history.to_csv(cfg.output_dir / "expD_corrector_history.csv", index=False)
    print("Exp D 20C jitter:")
    display(temp20)
    return {
        "feature_frames": feature_frames,
        "prediction_rows": pred,
        "summary": summary,
        "by_temperature": by_temp,
        "temp20_jitter": temp20,
        "corrector_history": history,
    }


def coral_loss(source, target):
    source = source - source.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    cs = (source.T @ source) / max(1, source.size(0) - 1)
    ct = (target.T @ target) / max(1, target.size(0) - 1)
    return (cs - ct).pow(2).mean()


def run_temp20_uda_diagnostic(cfg: CFG | None = None, *, lambda_coral=0.01):
    """Transductive UDA diagnostic: uses unlabeled 20C FUDS features during training."""
    cfg = exp_c_cfg(cfg)
    configure_torch_runtime()
    feature_frames = load_exp_c_features(cfg)
    feature_cols = R5_GATED_FEATURES
    scaled, _ = make_scaled_frames_for_ablation(feature_frames, feature_cols)
    train_ds = DecomposedWindowDataset(scaled["train"], feature_cols, cfg.window_len, cfg.stride, target_label="physical")
    target20_frames = [f for f in scaled["test"] if np.isclose(float(f["temperature"].iloc[0]), 20.0)]
    target_ds = DecomposedWindowDataset(target20_frames, feature_cols, cfg.window_len, cfg.stride, target_label="physical")
    test_ds = DecomposedWindowDataset(scaled["test"], feature_cols, cfg.window_len, cfg.stride, target_label="physical")
    train_loader = make_data_loader(train_ds, cfg, shuffle=True)
    target_loader = make_data_loader(target_ds, cfg, shuffle=True)
    test_loader = make_data_loader(test_ds, cfg, shuffle=False)
    feature_lookup = build_prediction_feature_lookup(feature_frames)

    def train_model(use_uda):
        model = build_lstm_soc_model(feature_cols, 1, cfg, "R5_GATED").to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lstm_lr, weight_decay=cfg.lstm_weight_decay)
        target_iter = iter(target_loader)
        for ep in range(1, int(cfg.lstm_epochs) + 1):
            model.train()
            for xs, ys, _ in train_loader:
                xs = xs.to(device=device, dtype=torch.float32, non_blocking=True)
                ys = ys.to(device=device, dtype=torch.float32, non_blocking=True)
                pred = model(xs)
                loss = F.smooth_l1_loss(pred, ys, beta=0.02)
                if use_uda:
                    try:
                        xt, _, _ = next(target_iter)
                    except StopIteration:
                        target_iter = iter(target_loader)
                        xt, _, _ = next(target_iter)
                    xt = xt.to(device=device, dtype=torch.float32, non_blocking=True)
                    xs_eff, _ = model.apply_component_gates(xs)
                    xt_eff, _ = model.apply_component_gates(xt)
                    hs = model.encode_last(xs_eff)
                    ht = model.encode_last(xt_eff)
                    loss = loss + float(lambda_coral) * coral_loss(hs, ht)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
                opt.step()
        return model

    rows = []
    for name, use_uda in [("R5_GATED_noadapt", False), ("R5_GATED_UDA_CORAL_20C_transductive", True)]:
        print(f"\n=== temp20 UDA diagnostic: {name} ===")
        model = train_model(use_uda)
        pred = []
        model.eval()
        with torch.no_grad():
            for x, y, meta in test_loader:
                x = x.to(device=device, dtype=torch.float32, non_blocking=True)
                yy = y.numpy()[:, 0]
                yp = model(x).detach().cpu().numpy()[:, 0]
                mdf = collate_meta_to_frame(meta)
                mdf["target_label"] = "physical"
                mdf["y_true"] = yy
                mdf["y_pred"] = yp
                pred.append(mdf)
        pred = pd.concat(pred, ignore_index=True)
        pred["error"] = pred["y_pred"] - pred["y_true"]
        pred["abs_error"] = np.abs(pred["error"])
        pred = pred.assign(split="test", ablation=name)
        rows.append(attach_prediction_features(pred, feature_lookup, ablation_name=name, target_label="physical"))
    out = pd.concat(rows, ignore_index=True)
    summary, by_temp, focus, temp20 = _summary_with_temp20_focus(out)
    out.to_csv(cfg.output_dir / "temp20_uda_prediction_rows.csv", index=False)
    summary.to_csv(cfg.output_dir / "temp20_uda_results.csv", index=False)
    temp20.to_csv(cfg.output_dir / "temp20_uda_vs_noadapt.csv", index=False)
    return {"prediction_rows": out, "summary": summary, "temp20": temp20}
