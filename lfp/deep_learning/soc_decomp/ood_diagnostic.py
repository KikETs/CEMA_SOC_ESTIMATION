from dataclasses import fields
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG, make_cfg
from .runtime import configure_torch_runtime, device
from .models import DecomposedWindowDataset, build_lstm_soc_model, collate_meta_to_frame
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
    load_feature_frame_dict_from_csv,
    train_variance_model_from_spec,
)

try:
    from IPython.display import display
except Exception:
    display = print


FEATURE_SETS = {
    "S1": ["V_raw", "I_raw", "T"],
    "S2": ["V_corr_raw", "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0"],
    "S3": ["V_raw", "I_raw", "T", "V_corr_raw", "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0"],
}

KNN_MAX_TRAIN_SAMPLES = 8000

MODEL_SPECS = {
    "R5_raw_I_T_all_components": None,
    "R5_GATED": None,
    "R5_GATED_AUG_np01_dp1_lp05": {
        "features": R5_GATED_FEATURES,
        "kind": "gated_seq",
        "lambda_gate_smooth": 0.03,
        "lambda_aug": 0.05,
        "component_noise_std": 0.01,
        "component_dropout_p": 0.10,
    },
}

EXPERIMENTS = {
    "Exp A": {
        "train_temps": ("N10", "0", "25", "50"),
        "omitted_temp_C": 10.0,
        "prediction_rows": "temp_variant_prediction_rows.csv",
        "feature_dir": "decomposed_features",
    },
    "Exp B": {
        "train_temps": ("N10", "10", "25", "50"),
        "omitted_temp_C": 0.0,
        "prediction_rows": None,
        "feature_dir": "decomposed_features",
    },
    "Exp C": {
        "train_temps": ("N10", "0", "10", "25", "50"),
        "omitted_temp_C": 20.0,
        "prediction_rows": "train_temp_minus10_0_10_25_50_prediction_rows.csv",
        "feature_dir": "decomposed_features_train_temp_minus10_0_10_25_50",
    },
    "Exp D": {
        "train_temps": ("N10", "0", "10", "20", "25", "50"),
        "omitted_temp_C": None,
        "prediction_rows": "expD_train_with_20_prediction_rows.csv",
        "feature_dir": "decomposed_features_train_temp_minus10_0_10_20_25_50",
    },
}


def clone_cfg(cfg: CFG | None = None) -> CFG:
    src = make_cfg() if cfg is None else cfg
    out = make_cfg()
    for f in fields(CFG):
        setattr(out, f.name, getattr(src, f.name))
    return out


def ood_cfg(cfg: CFG | None = None, experiment: str = "Exp C") -> CFG:
    out = clone_cfg(cfg)
    spec = EXPERIMENTS[experiment]
    out.smoke_mode = False
    out.use_existing_soc_cc_if_available = False
    out.use_existing_usable_if_available = False
    out.train_temps = spec["train_temps"]
    out.eval_temps = ("N10", "0", "10", "20", "25", "30", "40", "50")
    out.train_drives = ("DST", "US06")
    out.eval_drive = "FUDS"
    out.window_len = int(getattr(out, "ood_window_len", 50))
    out.stride = int(getattr(out, "ood_stride", 1))
    out.decomposed_dir = out.output_dir / spec["feature_dir"]
    out.decomposed_dir.mkdir(parents=True, exist_ok=True)
    return out


def _temp_key_to_celsius(v):
    if isinstance(v, str) and v.upper().startswith("N"):
        return -float(v[1:])
    return float(v)


def _assert_no_split_leakage(feature_frames):
    train_ids = {f["trajectory_id"].iloc[0] for f in feature_frames.get("train", [])}
    test_ids = {f["trajectory_id"].iloc[0] for f in feature_frames.get("test", [])}
    valid_ids = {f["trajectory_id"].iloc[0] for f in feature_frames.get("valid", [])}
    assert train_ids.isdisjoint(test_ids), "Train/test leakage: same trajectory_id appears in both splits"
    assert train_ids.isdisjoint(valid_ids), "Train/valid leakage: same trajectory_id appears in both splits"
    return {"train_ids": train_ids, "valid_ids": valid_ids, "test_ids": test_ids}


def _concat_frames(feature_frames):
    rows = []
    for split, frames in feature_frames.items():
        for f in frames:
            rows.append(f.copy().assign(feature_split=split))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _add_prediction_regions(df):
    out = df.copy()
    if "SOC_true" not in out and "y_true" in out:
        out["SOC_true"] = out["y_true"]
    if "SOC_pred" not in out and "y_pred" in out:
        out["SOC_pred"] = out["y_pred"]
    if "squared_error" not in out and "error" in out:
        out["squared_error"] = out["error"].astype(float) ** 2
    if "is_plateau_20_80" not in out:
        out["is_plateau_20_80"] = (out["SOC_true"] >= 0.2) & (out["SOC_true"] <= 0.8)
    if "trajectory_fraction" not in out:
        max_idx = out.groupby("trajectory_id")["end_index"].transform("max").replace(0, np.nan)
        out["trajectory_fraction"] = (out["end_index"] / max_idx).fillna(0.0)
    out["is_mid_trajectory"] = (out["trajectory_fraction"] >= 1 / 3) & (out["trajectory_fraction"] <= 2 / 3)
    if "is_cutoff_last10" not in out:
        out["is_cutoff_last10"] = out["trajectory_fraction"] >= 0.9
    out["phase_bin"] = pd.cut(
        out["trajectory_fraction"],
        bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=["early", "mid", "late"],
    ).astype(str)
    out["SOC_bin"] = np.select(
        [out["SOC_true"] < 0.2, out["SOC_true"] <= 0.8],
        ["0-20", "20-80"],
        default="80-100",
    )
    return out


def _standardize_from_train(train_x, other_x):
    mu = np.nanmean(train_x, axis=0)
    sd = np.nanstd(train_x, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (train_x - mu) / sd, (other_x - mu) / sd, mu, sd


def _cov_inv(train_z):
    try:
        from sklearn.covariance import LedoitWolf

        cov = LedoitWolf().fit(train_z).covariance_
    except Exception:
        cov = np.cov(train_z, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        reg = 1e-3 * np.trace(cov) / max(1, cov.shape[0])
        cov = cov + np.eye(cov.shape[0]) * max(reg, 1e-6)
    return np.linalg.pinv(cov)


def _mahalanobis(x, mu, inv_cov):
    delta = x - mu
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, inv_cov, delta), 0.0))


def _knn_distance(train_z, test_z, k=5):
    k = int(k)
    if len(train_z) > KNN_MAX_TRAIN_SAMPLES:
        # kNN is only an OOD score reference. Centroid/covariance still use all train windows.
        idx = np.linspace(0, len(train_z) - 1, KNN_MAX_TRAIN_SAMPLES).round().astype(int)
        train_z = train_z[idx]
    try:
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=min(k, len(train_z)), algorithm="auto")
        nn_model.fit(train_z)
        d, _ = nn_model.kneighbors(test_z, return_distance=True)
        return d.mean(axis=1)
    except Exception:
        # Chunked fallback avoids materializing a full test x train distance matrix.
        out = np.empty(len(test_z), dtype=np.float64)
        for start in range(0, len(test_z), 1024):
            chunk = test_z[start:start + 1024]
            d2 = ((chunk[:, None, :] - train_z[None, :, :]) ** 2).sum(axis=-1)
            part = np.partition(np.sqrt(d2), kth=min(k - 1, d2.shape[1] - 1), axis=1)[:, :k]
            out[start:start + len(chunk)] = part.mean(axis=1)
        return out


def _train_centroids(train_df, train_z):
    centroids = {}
    for temp, idx in train_df.groupby("temperature").groups.items():
        idx = np.asarray(list(idx), dtype=int)
        centroids[float(temp)] = train_z[idx].mean(axis=0)
    return centroids


def _distance_scores_for_set(train_df, test_df, cols, prefix):
    cols = [c for c in cols if c in train_df.columns and c in test_df.columns]
    if not cols:
        raise ValueError(f"No usable columns for {prefix}")
    train_x = train_df[cols].to_numpy(np.float64)
    test_x = test_df[cols].to_numpy(np.float64)
    assert np.isfinite(train_x).all(), f"Non-finite train features in {prefix}"
    train_z, test_z, _, _ = _standardize_from_train(train_x, test_x)
    centroids = _train_centroids(train_df.reset_index(drop=True), train_z)
    c_stack = np.stack(list(centroids.values()), axis=0)
    c_temps = list(centroids.keys())
    dmat = np.sqrt(((test_z[:, None, :] - c_stack[None, :, :]) ** 2).sum(axis=-1))
    nearest = dmat.argmin(axis=1)
    inv_cov = _cov_inv(train_z)
    out = pd.DataFrame(index=test_df.index)
    out[f"distance_{prefix}_centroid"] = dmat[np.arange(len(test_z)), nearest]
    out[f"nearest_temp_{prefix}"] = [c_temps[i] for i in nearest]
    out[f"distance_{prefix}_global_centroid"] = np.sqrt(((test_z - train_z.mean(axis=0)) ** 2).sum(axis=1))
    out[f"mahalanobis_{prefix}"] = _mahalanobis(test_z, train_z.mean(axis=0), inv_cov)
    for k in (5, 10, 20):
        out[f"knn_{prefix}_k{k}"] = _knn_distance(train_z, test_z, k=k)
    train_scores = pd.DataFrame({
        f"distance_{prefix}_centroid": np.min(
            np.sqrt(((train_z[:, None, :] - c_stack[None, :, :]) ** 2).sum(axis=-1)),
            axis=1,
        ),
        f"mahalanobis_{prefix}": _mahalanobis(train_z, train_z.mean(axis=0), inv_cov),
    })
    return out, train_scores


def _load_prediction_rows(cfg: CFG, experiment="Exp C", preferred_model="R5_GATED"):
    spec = EXPERIMENTS[experiment]
    path = cfg.output_dir / spec["prediction_rows"] if spec["prediction_rows"] else None
    if path is None or not path.exists():
        raise FileNotFoundError(f"Prediction rows for {experiment} are unavailable: {path}")
    pred = pd.read_csv(path)
    pred = pred[pred.get("label_type", pred.get("target_label", "physical")).eq("physical")].copy()
    if "model_name" not in pred:
        pred["model_name"] = pred.get("ablation", preferred_model)
    if preferred_model in set(pred["model_name"]):
        pred = pred[pred["model_name"].eq(preferred_model)].copy()
    pred["SOC_true"] = pred["y_true"]
    pred["SOC_pred"] = pred["y_pred"]
    pred["squared_error"] = pred["error"].astype(float) ** 2
    return _add_prediction_regions(pred)


def load_ood_feature_frames(cfg: CFG | None = None, experiment="Exp C"):
    cfg = ood_cfg(cfg, experiment)
    feature_frames = load_feature_frame_dict_from_csv(cfg, decomposed_dir=cfg.decomposed_dir)
    _assert_no_split_leakage(feature_frames)
    return cfg, feature_frames


def _train_latent_model(feature_frames, cfg: CFG, model_name="R5_GATED", epochs=None):
    cfg = clone_cfg(cfg)
    cfg.target_labels_to_run = ("physical",)
    if epochs is not None:
        cfg.lstm_epochs = int(epochs)
    feature_cols = ABLATIONS.get(model_name, ABLATIONS["R5_GATED"])
    scaled, _ = make_scaled_frames_for_ablation(feature_frames, feature_cols)
    train_ds = DecomposedWindowDataset(scaled["train"], feature_cols, cfg.window_len, cfg.stride, target_label="physical")
    test_ds = DecomposedWindowDataset(scaled["test"], feature_cols, cfg.window_len, cfg.stride, target_label="physical")
    train_loader = make_data_loader(train_ds, cfg, shuffle=True)
    test_loader = make_data_loader(test_ds, cfg, shuffle=False)
    model = build_lstm_soc_model(feature_cols, 1, cfg, model_name).to(device)
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
            print(f"OOD latent model {model_name}: epoch={ep} train_loss={np.mean(losses):.5f}")
    return model, train_loader, test_loader, feature_cols


@torch.no_grad()
def _latent_rows(model, loader, cfg: CFG):
    meta_rows, h_rows, pred_rows, gate_rows = [], [], [], []
    model.eval()
    for x, y, meta in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        if hasattr(model, "apply_component_gates"):
            x_eff, gates = model.apply_component_gates(x)
            g = gates.detach().cpu().numpy()
            gate_last = g[:, -1, :]
            gate_var = g.var(axis=1).mean(axis=1)
            gate_entropy = -(gate_last * np.log(gate_last + 1e-8) + (1.0 - gate_last) * np.log(1.0 - gate_last + 1e-8)).mean(axis=1)
            gate_variation = np.abs(np.diff(g, axis=1)).mean(axis=(1, 2)) if g.shape[1] > 1 else np.zeros(g.shape[0])
        else:
            x_eff = x
            gate_last = gate_var = gate_entropy = gate_variation = None
        h = model.encode_last(x_eff).detach().cpu().numpy()
        yp = model(x).detach().cpu().numpy()[:, 0]
        yy = y.numpy()[:, 0]
        mdf = collate_meta_to_frame(meta)
        mdf["SOC_true"] = yy
        mdf["SOC_pred_latent_model"] = yp
        mdf["latent_model_error"] = yp - yy
        meta_rows.append(mdf)
        h_rows.append(h)
        pred_rows.append(yp)
        if gate_last is not None:
            gate_df = mdf[["trajectory_id", "end_index"]].copy()
            gate_df["gate_pol"] = gate_last[:, 0]
            gate_df["gate_hys"] = gate_last[:, 1]
            gate_df["gate_ohm"] = gate_last[:, 2]
            gate_df["gate_entropy"] = gate_entropy
            gate_df["gate_variation"] = gate_variation
            gate_df["gate_variance"] = gate_var
            gate_rows.append(gate_df)
    meta_df = pd.concat(meta_rows, ignore_index=True) if meta_rows else pd.DataFrame()
    h = np.vstack(h_rows) if h_rows else np.empty((0, 0))
    gates = pd.concat(gate_rows, ignore_index=True) if gate_rows else pd.DataFrame()
    return meta_df, h, gates


def _latent_distance_scores(train_meta, train_h, test_meta, test_h):
    train_z, test_z, _, _ = _standardize_from_train(train_h, test_h)
    train_meta = train_meta.reset_index(drop=True)
    centroids = _train_centroids(train_meta, train_z)
    c_stack = np.stack(list(centroids.values()), axis=0)
    c_temps = list(centroids.keys())
    dmat = np.sqrt(((test_z[:, None, :] - c_stack[None, :, :]) ** 2).sum(axis=-1))
    nearest = dmat.argmin(axis=1)
    inv_cov = _cov_inv(train_z)
    out = test_meta[["trajectory_id", "end_index"]].copy()
    out["distance_S4_latent"] = dmat[np.arange(len(test_z)), nearest]
    out["nearest_temp_S4"] = [c_temps[i] for i in nearest]
    out["mahalanobis_S4"] = _mahalanobis(test_z, train_z.mean(axis=0), inv_cov)
    for k in (5, 10, 20):
        out[f"knn_S4_k{k}"] = _knn_distance(train_z, test_z, k=k)
    train_scores = pd.DataFrame({
        "distance_S4_latent": np.min(
            np.sqrt(((train_z[:, None, :] - c_stack[None, :, :]) ** 2).sum(axis=-1)),
            axis=1,
        ),
        "mahalanobis_S4": _mahalanobis(train_z, train_z.mean(axis=0), inv_cov),
    })
    return out, train_scores, (train_meta, train_h, test_meta, test_h)


def compute_ood_feature_distance_scores(
    cfg: CFG | None = None,
    *,
    experiment="Exp C",
    model_name="R5_GATED",
    include_latent=True,
    latent_epochs=60,
    output_prefix="",
):
    cfg, feature_frames = load_ood_feature_frames(cfg, experiment)
    pred = _load_prediction_rows(cfg, experiment, preferred_model=model_name)
    frame_df = _concat_frames(feature_frames)
    train_df = frame_df[frame_df["feature_split"].eq("train")].reset_index(drop=True)
    test_df = frame_df[frame_df["feature_split"].eq("test")].reset_index(drop=True)
    test_key = test_df[["trajectory_id", "end_index", *sorted({c for cols in FEATURE_SETS.values() for c in cols if c in test_df.columns})]].copy()
    score_base = pred.merge(test_key, on=["trajectory_id", "end_index"], how="left", suffixes=("", "_feature"))
    if score_base[[c for c in FEATURE_SETS["S3"] if c in score_base.columns]].isna().any().any():
        warnings.warn("Some prediction rows did not find matching cached feature rows.")

    test_for_scores = score_base.copy()
    score_cols = []
    train_score_parts = []
    for prefix, cols in FEATURE_SETS.items():
        scores, train_scores = _distance_scores_for_set(train_df, test_for_scores, cols, prefix)
        for c in scores.columns:
            test_for_scores[c] = scores[c].to_numpy()
            score_cols.append(c)
        train_scores["score_prefix"] = prefix
        train_score_parts.append(train_scores)

    latent_bundle = None
    gate_rows = pd.DataFrame()
    if include_latent:
        model, train_loader, test_loader, _ = _train_latent_model(feature_frames, cfg, "R5_GATED", epochs=latent_epochs)
        train_meta, train_h, _ = _latent_rows(model, train_loader, cfg)
        test_meta, test_h, gate_rows = _latent_rows(model, test_loader, cfg)
        latent_scores, latent_train_scores, latent_bundle = _latent_distance_scores(train_meta, train_h, test_meta, test_h)
        test_for_scores = test_for_scores.merge(latent_scores, on=["trajectory_id", "end_index"], how="left")
        train_score_parts.append(latent_train_scores.assign(score_prefix="S4"))
        if len(gate_rows):
            test_for_scores = test_for_scores.merge(gate_rows, on=["trajectory_id", "end_index"], how="left")

    test_for_scores["model_name"] = model_name
    test_for_scores["label_type"] = "physical"
    test_for_scores = _add_prediction_regions(test_for_scores)
    required = [
        "trajectory_id", "temperature_C", "drive_cycle", "model_name", "label_type",
        "time_index", "end_index", "SOC_true", "SOC_pred", "abs_error", "squared_error",
        "is_plateau_20_80", "is_mid_trajectory", "is_cutoff_last10",
        "distance_S1_centroid", "distance_S2_centroid", "distance_S3_centroid",
        "distance_S4_latent", "mahalanobis_S1", "mahalanobis_S2", "mahalanobis_S3",
        "knn_S3_k5", "knn_S4_k5",
    ]
    ordered = [c for c in required if c in test_for_scores.columns]
    rest = [c for c in test_for_scores.columns if c not in ordered]
    out = test_for_scores[ordered + rest]
    out.to_csv(cfg.output_dir / f"{output_prefix}ood_feature_distance_scores.csv", index=False)
    pd.concat(train_score_parts, ignore_index=True).to_csv(cfg.output_dir / f"{output_prefix}ood_train_score_reference.csv", index=False)
    _summarize_distance_scores(out, cfg, output_prefix=output_prefix)
    if latent_bundle is not None:
        _plot_latent_pca(latent_bundle, out, cfg)
    return {"scores": out, "feature_frames": feature_frames, "latent_bundle": latent_bundle}


def _summarize_distance_scores(scores, cfg: CFG, output_prefix=""):
    score_cols = [c for c in scores.columns if c.startswith("distance_") or c.startswith("mahalanobis_") or c.startswith("knn_")]
    by_temp = scores.groupby("temperature_C")[score_cols + ["abs_error"]].mean(numeric_only=True).reset_index()
    by_soc = scores.groupby("SOC_bin")[score_cols + ["abs_error"]].mean(numeric_only=True).reset_index()
    phase_col = "phase_bin" if "phase_bin" in scores.columns else "is_mid_trajectory"
    by_phase = scores.groupby(phase_col)[score_cols + ["abs_error"]].mean(numeric_only=True).reset_index()
    by_temp.to_csv(cfg.output_dir / f"{output_prefix}ood_distance_by_temperature.csv", index=False)
    by_soc.to_csv(cfg.output_dir / f"{output_prefix}ood_distance_by_soc_bin.csv", index=False)
    by_phase.to_csv(cfg.output_dir / f"{output_prefix}ood_distance_by_phase.csv", index=False)


def _model_prediction_with_optional_spec(feature_frames, cfg, model_name, spec=None):
    feature_lookup = build_prediction_feature_lookup(feature_frames)
    if spec is None:
        model, hist, _, pred_test, _, test_loader = train_one_lstm_ablation(
            feature_frames, ABLATIONS[model_name], "physical", cfg, model_name
        )
    else:
        model, hist, pred_test, _ = train_variance_model_from_spec(feature_frames, model_name, spec, cfg)
        feature_cols = spec["features"]
        scaled, _ = make_scaled_frames_for_ablation(feature_frames, feature_cols)
        test_ds = DecomposedWindowDataset(scaled["test"], feature_cols, cfg.window_len, cfg.stride, target_label="physical")
        test_loader = make_data_loader(test_ds, cfg, shuffle=False)
    pred_test = pred_test.assign(split="test", ablation=model_name)
    pred = attach_prediction_features(pred_test, feature_lookup, ablation_name=model_name, target_label="physical")
    return model, pred, test_loader


@torch.no_grad()
def _mc_dropout_std(model, loader, cfg: CFG, samples=20):
    if loader is None or float(getattr(cfg, "lstm_dropout", 0.0)) <= 0.0:
        return pd.DataFrame()
    rows = []
    model.train()
    for x, y, meta in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        preds = []
        for _ in range(int(samples)):
            preds.append(model(x).detach().cpu().numpy()[:, 0])
        arr = np.stack(preds, axis=0)
        mdf = collate_meta_to_frame(meta)
        mdf["mc_dropout_std"] = arr.std(axis=0)
        rows.append(mdf[["trajectory_id", "end_index", "mc_dropout_std"]])
    model.eval()
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


@torch.no_grad()
def _gate_uncertainty(model, loader, cfg: CFG):
    if loader is None or not hasattr(model, "component_gates"):
        return pd.DataFrame()
    rows = []
    model.eval()
    for x, _, meta in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        g = model.component_gates(x).detach().cpu().numpy()
        last = g[:, -1, :]
        ent = -(last * np.log(last + 1e-8) + (1.0 - last) * np.log(1.0 - last + 1e-8)).mean(axis=1)
        variation = np.abs(np.diff(g, axis=1)).mean(axis=(1, 2)) if g.shape[1] > 1 else np.zeros(g.shape[0])
        mdf = collate_meta_to_frame(meta)
        mdf["gate_pol"] = last[:, 0]
        mdf["gate_hys"] = last[:, 1]
        mdf["gate_ohm"] = last[:, 2]
        mdf["gate_entropy"] = ent
        mdf["gate_variation"] = variation
        rows.append(mdf)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compute_prediction_uncertainty(
    cfg: CFG | None = None,
    *,
    experiment="Exp C",
    ensemble_size=3,
    ensemble_epochs=40,
    mc_samples=20,
    model_names=("R5_raw_I_T_all_components", "R5_GATED", "R5_GATED_AUG_np01_dp1_lp05"),
):
    cfg, feature_frames = load_ood_feature_frames(cfg, experiment)
    cfg = clone_cfg(cfg)
    cfg.lstm_epochs = int(ensemble_epochs)
    member_rows, gate_rows, mc_rows = [], [], []
    for member in range(int(ensemble_size)):
        torch.manual_seed(member)
        np.random.seed(member)
        for model_name in model_names:
            spec = MODEL_SPECS.get(model_name)
            print(f"\n=== OOD uncertainty ensemble member={member + 1}/{ensemble_size} | {model_name} ===")
            model, pred, test_loader = _model_prediction_with_optional_spec(feature_frames, cfg, model_name, spec=spec)
            pred["ensemble_member"] = member
            member_rows.append(pred)
            gates = _gate_uncertainty(model, test_loader, cfg)
            if len(gates):
                gates["model_name"] = model_name
                gates["ensemble_member"] = member
                gate_rows.append(gates)
            mc = _mc_dropout_std(model, test_loader, cfg, samples=mc_samples)
            if len(mc):
                mc["model_name"] = model_name
                mc["ensemble_member"] = member
                mc_rows.append(mc)
    pred_all = pd.concat(member_rows, ignore_index=True)
    keys = ["model_name", "trajectory_id", "temperature_C", "drive_cycle", "end_index"]
    agg = (
        pred_all.groupby(keys)
        .agg(
            SOC_true=("y_true", "first"),
            SOC_pred=("y_pred", "mean"),
            ensemble_std=("y_pred", "std"),
            ensemble_min=("y_pred", "min"),
            ensemble_max=("y_pred", "max"),
            n_members=("y_pred", "count"),
            is_plateau_20_80=("is_plateau_20_80", "first"),
            is_cutoff_last10=("is_cutoff_last10", "first"),
            trajectory_fraction=("trajectory_fraction", "first"),
        )
        .reset_index()
    )
    agg["ensemble_std"] = agg["ensemble_std"].fillna(0.0)
    agg["prediction_range"] = agg["ensemble_max"] - agg["ensemble_min"]
    agg["ensemble_abs_deviation"] = (
        pred_all.merge(agg[keys + ["SOC_pred"]], on=keys, how="left")
        .assign(dev=lambda d: np.abs(d["y_pred"] - d["SOC_pred"]))
        .groupby(keys)["dev"].mean()
        .reset_index(drop=True)
    )
    agg["error"] = agg["SOC_pred"] - agg["SOC_true"]
    agg["abs_error"] = np.abs(agg["error"])
    agg = _add_prediction_regions(agg)
    gate = pd.concat(gate_rows, ignore_index=True) if gate_rows else pd.DataFrame()
    if len(gate):
        gate_summary = (
            gate.groupby(["model_name", "trajectory_id", "end_index"])
            .agg(
                gate_pol=("gate_pol", "mean"),
                gate_hys=("gate_hys", "mean"),
                gate_ohm=("gate_ohm", "mean"),
                gate_entropy=("gate_entropy", "mean"),
                gate_variation=("gate_variation", "mean"),
            )
            .reset_index()
        )
        agg = agg.merge(gate_summary, on=["model_name", "trajectory_id", "end_index"], how="left")
        gate_summary.to_csv(cfg.output_dir / "ood_gate_uncertainty.csv", index=False)
    else:
        pd.DataFrame().to_csv(cfg.output_dir / "ood_gate_uncertainty.csv", index=False)
    mc = pd.concat(mc_rows, ignore_index=True) if mc_rows else pd.DataFrame()
    if len(mc):
        mc_summary = mc.groupby(["model_name", "trajectory_id", "end_index"])["mc_dropout_std"].mean().reset_index()
        agg = agg.merge(mc_summary, on=["model_name", "trajectory_id", "end_index"], how="left")
    if "mc_dropout_std" not in agg:
        agg["mc_dropout_std"] = np.nan
    dist_path = cfg.output_dir / "ood_feature_distance_scores.csv"
    if dist_path.exists():
        dist = pd.read_csv(dist_path)
        keep = [
            "trajectory_id", "end_index", "distance_S4_latent", "mahalanobis_S3",
            "knn_S4_k5", "distance_S3_centroid",
        ]
        agg = agg.merge(dist[[c for c in keep if c in dist.columns]].drop_duplicates(["trajectory_id", "end_index"]),
                        on=["trajectory_id", "end_index"], how="left")
    ordered = [
        "model_name", "temperature_C", "trajectory_id", "end_index", "SOC_true", "SOC_pred",
        "abs_error", "ensemble_std", "mc_dropout_std", "prediction_range",
        "gate_entropy", "gate_variation", "distance_S4_latent",
    ]
    agg[[c for c in ordered if c in agg.columns] + [c for c in agg.columns if c not in ordered]].to_csv(
        cfg.output_dir / "ood_prediction_uncertainty.csv", index=False
    )
    return agg


def _corr(x, y):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 3 or df["x"].std() == 0 or df["y"].std() == 0:
        return float("nan")
    return float(df["x"].corr(df["y"]))


def _corr_strength(v):
    if not np.isfinite(v):
        return "not available"
    av = abs(v)
    if av >= 0.5:
        return "useful OOD indicator"
    if av >= 0.2:
        return "weak-to-moderate signal"
    return "weak signal"


def _read_csv_if_nonempty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def compute_ood_error_correlations(cfg: CFG | None = None):
    cfg = clone_cfg(cfg)
    score_path = cfg.output_dir / "ood_feature_distance_scores.csv"
    unc_path = cfg.output_dir / "ood_prediction_uncertainty.csv"
    if not score_path.exists():
        raise FileNotFoundError(score_path)
    scores = pd.read_csv(score_path)
    unc = _read_csv_if_nonempty(unc_path)
    if len(unc):
        merge_cols = ["model_name", "trajectory_id", "end_index"]
        scores = scores.merge(
            unc[[c for c in [
                *merge_cols, "ensemble_std", "mc_dropout_std", "prediction_range",
                "gate_entropy", "gate_variation"
            ] if c in unc.columns]],
            on=merge_cols,
            how="left",
            suffixes=("", "_unc"),
        )
        for c in ["ensemble_std", "mc_dropout_std", "prediction_range", "gate_entropy", "gate_variation"]:
            uc = f"{c}_unc"
            if uc in scores:
                scores[c] = scores[c].where(scores[c].notna(), scores[uc]) if c in scores else scores[uc]
    cols = [
        "distance_S3_centroid", "distance_S4_latent", "mahalanobis_S3", "knn_S4_k5",
        "ensemble_std", "mc_dropout_std", "gate_entropy", "gate_variation",
    ]
    rows = []
    for col in cols:
        if col in scores:
            v = _corr(scores["abs_error"], scores[col])
            rows.append({"scope": "overall", "score": col, "corr_abs_error": v, "interpretation": _corr_strength(v)})
    overall = pd.DataFrame(rows)
    by_temp = []
    for temp, g in scores.groupby("temperature_C"):
        for col in cols:
            if col in g:
                v = _corr(g["abs_error"], g[col])
                by_temp.append({"temperature_C": temp, "score": col, "corr_abs_error": v, "interpretation": _corr_strength(v)})
    by_region = []
    regions = {
        "0-20% SOC": scores["SOC_true"] < 0.2,
        "20-80% plateau": (scores["SOC_true"] >= 0.2) & (scores["SOC_true"] <= 0.8),
        "80-100% SOC": scores["SOC_true"] > 0.8,
        "mid-trajectory": scores.get("is_mid_trajectory", False),
        "cutoff last10": scores.get("is_cutoff_last10", False),
    }
    for name, mask in regions.items():
        g = scores[mask].copy()
        for col in cols:
            if col in g:
                v = _corr(g["abs_error"], g[col])
                by_region.append({"region": name, "score": col, "corr_abs_error": v, "interpretation": _corr_strength(v)})
    overall.to_csv(cfg.output_dir / "ood_error_correlation.csv", index=False)
    pd.DataFrame(by_temp).to_csv(cfg.output_dir / "ood_error_correlation_by_temperature.csv", index=False)
    pd.DataFrame(by_region).to_csv(cfg.output_dir / "ood_error_correlation_by_region.csv", index=False)
    return {"overall": overall, "by_temperature": pd.DataFrame(by_temp), "by_region": pd.DataFrame(by_region)}


def _jitter_metrics(g):
    if g.empty:
        return {"retained_jitter_ratio": np.nan, "retained_plateau_MAE": np.nan}
    ratios = []
    for _, tg in g.sort_values("end_index").groupby("trajectory_id"):
        pred_j = np.abs(np.diff(tg["SOC_pred"].to_numpy(float))).mean() if len(tg) > 1 else np.nan
        true_j = np.abs(np.diff(tg["SOC_true"].to_numpy(float))).mean() if len(tg) > 1 else np.nan
        if np.isfinite(pred_j) and np.isfinite(true_j) and true_j > 1e-12:
            ratios.append(pred_j / true_j)
    plateau = g[g["is_plateau_20_80"].astype(bool)]
    return {
        "retained_jitter_ratio": float(np.nanmean(ratios)) if ratios else np.nan,
        "retained_plateau_MAE": float(plateau["abs_error"].mean()) if len(plateau) else np.nan,
    }


def compute_ood_abstention_curve(cfg: CFG | None = None):
    cfg = clone_cfg(cfg)
    path = cfg.output_dir / "ood_feature_distance_scores.csv"
    scores = pd.read_csv(path)
    unc_path = cfg.output_dir / "ood_prediction_uncertainty.csv"
    unc = _read_csv_if_nonempty(unc_path)
    if len(unc):
        scores = scores.merge(
            unc[[c for c in ["model_name", "trajectory_id", "end_index", "ensemble_std", "gate_entropy"] if c in unc.columns]],
            on=["model_name", "trajectory_id", "end_index"],
            how="left",
            suffixes=("", "_unc"),
        )
        for col in ["ensemble_std", "gate_entropy"]:
            uc = f"{col}_unc"
            if uc in scores:
                scores[col] = scores[col].where(scores[col].notna(), scores[uc]) if col in scores else scores[uc]

    ref = _read_csv_if_nonempty(cfg.output_dir / "ood_train_score_reference.csv")
    def z(col):
        if col in scores and len(ref):
            if col.startswith("distance_S4"):
                rr = ref[ref["score_prefix"].eq("S4")][col]
            elif "S3" in col:
                rr = ref[ref["score_prefix"].eq("S3")][col]
            else:
                rr = pd.Series(dtype=float)
            mu = rr.mean() if len(rr) else scores[col].mean()
            sd = rr.std() if len(rr) and rr.std() > 1e-12 else scores[col].std()
        else:
            mu, sd = scores[col].mean(), scores[col].std()
        return (scores[col] - mu) / max(float(sd), 1e-12)

    pieces = []
    for col in ["distance_S4_latent", "ensemble_std", "gate_entropy"]:
        if col in scores:
            pieces.append(z(col).fillna(0.0))
    scores["combined_ood_score"] = np.sum(pieces, axis=0) if pieces else scores.get("mahalanobis_S3", 0.0)
    score_cols = [c for c in ["distance_S4_latent", "mahalanobis_S3", "knn_S4_k5", "ensemble_std", "mc_dropout_std", "combined_ood_score"] if c in scores]
    rows, temp_rows = [], []
    for score_col in score_cols:
        s = scores[score_col].replace([np.inf, -np.inf], np.nan)
        valid = scores[s.notna()].copy()
        valid["_risk_score"] = s[s.notna()]
        for coverage in [1.0, 0.95, 0.90, 0.80, 0.70, 0.50]:
            if len(valid) == 0:
                continue
            threshold = valid["_risk_score"].quantile(coverage)
            retained = valid[valid["_risk_score"] <= threshold]
            rejected = valid[valid["_risk_score"] > threshold]
            jm = _jitter_metrics(retained)
            err = retained["SOC_pred"] - retained["SOC_true"]
            rows.append({
                "score": score_col,
                "coverage": coverage,
                "threshold_score_percentile": coverage,
                "retained_n": int(len(retained)),
                "rejected_n": int(len(rejected)),
                "retained_MAE": float(retained["abs_error"].mean()) if len(retained) else np.nan,
                "rejected_MAE": float(rejected["abs_error"].mean()) if len(rejected) else np.nan,
                "retained_RMSE": float(np.sqrt(np.mean(err.to_numpy(float) ** 2))) if len(retained) else np.nan,
                "retained_max_error": float(retained["abs_error"].max()) if len(retained) else np.nan,
                **jm,
            })
            for temp, g in valid.groupby("temperature_C"):
                temp_rows.append({
                    "score": score_col,
                    "coverage": coverage,
                    "temperature_C": temp,
                    "n_total": int(len(g)),
                    "n_rejected": int((g["_risk_score"] > threshold).sum()),
                    "rejected_fraction": float((g["_risk_score"] > threshold).mean()),
                })
    curve = pd.DataFrame(rows)
    by_temp = pd.DataFrame(temp_rows)
    curve.to_csv(cfg.output_dir / "ood_abstention_curve.csv", index=False)
    by_temp.to_csv(cfg.output_dir / "ood_abstention_by_temperature.csv", index=False)
    _plot_abstention(curve, cfg)
    return {"curve": curve, "by_temperature": by_temp, "scores": scores}


def _plot_abstention(curve, cfg: CFG):
    if curve.empty:
        return
    plt.figure(figsize=(6, 4))
    for score, g in curve.groupby("score"):
        plt.plot(g["coverage"], g["retained_MAE"] * 100.0, marker="o", label=score)
    plt.gca().invert_xaxis()
    plt.xlabel("retained coverage")
    plt.ylabel("retained MAE (%)")
    plt.title("OOD abstention curve")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(cfg.output_dir / "ood_abstention_plot.png", dpi=180)
    plt.savefig(cfg.output_dir / "ood_risk_coverage_plot.png", dpi=180)
    plt.close()


def summarize_omitted_temperature_ood(cfg: CFG | None = None):
    cfg = clone_cfg(cfg)
    rows = []
    coverage_path = cfg.output_dir / "train_temp_coverage_comparison.csv"
    coverage = _read_csv_if_nonempty(coverage_path)
    ood_scores = _read_csv_if_nonempty(cfg.output_dir / "ood_feature_distance_scores.csv")
    for exp, spec in EXPERIMENTS.items():
        omitted = spec["omitted_temp_C"]
        if exp == "Exp D":
            omitted = 20.0
        err_rows = coverage[coverage["experiment"].eq(exp)].copy() if len(coverage) else pd.DataFrame()
        if len(err_rows) and omitted is not None:
            err_rows = err_rows[np.isclose(err_rows["temperature_C"].astype(float), float(omitted))]
        for _, er in err_rows.iterrows():
            row = {
                "experiment": exp,
                "train_temps": ",".join(str(_temp_key_to_celsius(t)) for t in spec["train_temps"]),
                "focus_temperature_C": omitted,
                "model_name": er.get("model_name"),
                "MAE_pct": er.get("MAE_pct"),
                "RMSE_pct": er.get("RMSE_pct"),
                "mid_trajectory_MAE_pct": er.get("mid_trajectory_MAE_pct"),
                "cutoff_last10_MAE_pct": er.get("cutoff_last10_MAE_pct"),
                "is_train_temperature": er.get("is_train_temperature"),
            }
            if exp == "Exp C" and len(ood_scores):
                og = ood_scores[np.isclose(ood_scores["temperature_C"].astype(float), float(omitted))]
                row.update({
                    "ood_score_mean_distance_S4": og.get("distance_S4_latent", pd.Series(dtype=float)).mean(),
                    "ood_score_mean_mahalanobis_S3": og.get("mahalanobis_S3", pd.Series(dtype=float)).mean(),
                    "ood_score_p90_distance_S4": og.get("distance_S4_latent", pd.Series(dtype=float)).quantile(0.9),
                })
            rows.append(row)
        if exp == "Exp D" and err_rows.empty:
            expd_by_temp = _read_csv_if_nonempty(cfg.output_dir / "expD_by_temperature.csv")
            expd_ood_by_temp = _read_csv_if_nonempty(cfg.output_dir / "expD_ood_distance_by_temperature.csv")
            expd_ood20 = expd_ood_by_temp[
                np.isclose(expd_ood_by_temp["temperature_C"].astype(float), 20.0)
            ] if len(expd_ood_by_temp) else pd.DataFrame()
            expd_maha_s3 = float(expd_ood20["mahalanobis_S3"].iloc[0]) if len(expd_ood20) and "mahalanobis_S3" in expd_ood20 else np.nan
            if len(expd_by_temp):
                focus = expd_by_temp[np.isclose(expd_by_temp["temperature_C"].astype(float), 20.0)].copy()
                for _, er in focus.iterrows():
                    rows.append({
                        "experiment": exp,
                        "train_temps": ",".join(str(_temp_key_to_celsius(t)) for t in spec["train_temps"]),
                        "focus_temperature_C": 20.0,
                        "model_name": er.get("model_name"),
                        "MAE_pct": er.get("MAE_pct"),
                        "RMSE_pct": er.get("RMSE_pct"),
                        "jitter_ratio": er.get("jitter_ratio"),
                        "mid_trajectory_MAE_pct": np.nan,
                        "cutoff_last10_MAE_pct": np.nan,
                        "is_train_temperature": True,
                        "ood_score_mean_distance_S4": np.nan,
                        "ood_score_mean_mahalanobis_S3": expd_maha_s3,
                        "ood_score_p90_distance_S4": np.nan,
                    })
    out = pd.DataFrame(rows)
    out.to_csv(cfg.output_dir / "omitted_temperature_ood_summary.csv", index=False)
    out.to_csv(cfg.output_dir / "expA_B_C_D_ood_comparison.csv", index=False)
    _plot_omitted_summary(out, cfg)
    return out


def _plot_omitted_summary(out, cfg: CFG):
    if out.empty or "MAE_pct" not in out:
        return
    data = out[out["model_name"].isin(["R5_GATED", "R5_raw_I_T_all_components", "R5_GATED_AUG_np01_dp1_lp05"])].copy()
    if data.empty:
        data = out.copy()
    plt.figure(figsize=(7, 4))
    for model, g in data.groupby("model_name"):
        plt.plot(g["experiment"], g["MAE_pct"], marker="o", label=model)
    plt.ylabel("focus temperature MAE (%)")
    plt.title("Omitted/included temperature diagnostic")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(cfg.output_dir / "omitted_temperature_ood_plot.png", dpi=180)
    plt.close()


def _plot_latent_pca(latent_bundle, scores, cfg: CFG):
    train_meta, train_h, test_meta, test_h = latent_bundle
    try:
        from sklearn.decomposition import PCA
    except Exception:
        warnings.warn("sklearn PCA is unavailable; latent PCA plots skipped.")
        return
    meta = pd.concat([
        train_meta.assign(split="train"),
        test_meta.assign(split="test"),
    ], ignore_index=True)
    h = np.vstack([train_h, test_h])
    z = PCA(n_components=2).fit_transform(h)
    meta["pca1"] = z[:, 0]
    meta["pca2"] = z[:, 1]
    score_small = scores[["trajectory_id", "end_index", "abs_error", "distance_S4_latent", "SOC_bin"]].drop_duplicates(["trajectory_id", "end_index"])
    meta = meta.merge(score_small, on=["trajectory_id", "end_index"], how="left")

    def scatter(col, path, cmap="viridis"):
        plt.figure(figsize=(6, 5))
        vals = meta[col]
        if vals.dtype.kind in "biufc":
            sc = plt.scatter(meta["pca1"], meta["pca2"], c=vals, s=4, alpha=0.45, cmap=cmap)
            plt.colorbar(sc, label=col)
        else:
            for key, g in meta.groupby(col):
                plt.scatter(g["pca1"], g["pca2"], s=4, alpha=0.45, label=str(key))
            plt.legend(markerscale=3, fontsize=7)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"R5_GATED latent PCA by {col}")
        plt.tight_layout()
        plt.savefig(cfg.output_dir / path, dpi=180)
        plt.close()

    scatter("temperature", "latent_pca_by_temperature.png")
    scatter("abs_error", "latent_pca_by_error.png", cmap="magma")
    scatter("distance_S4_latent", "latent_pca_by_ood_score.png", cmap="magma")
    meta.to_csv(cfg.output_dir / "latent_pca_rows.csv", index=False)
    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=0)
        zz = reducer.fit_transform(h)
        meta["umap1"] = zz[:, 0]
        meta["umap2"] = zz[:, 1]
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(meta["umap1"], meta["umap2"], c=meta["temperature"], s=4, alpha=0.45, cmap="viridis")
        plt.colorbar(sc, label="temperature")
        plt.title("R5_GATED latent UMAP by temperature")
        plt.tight_layout()
        plt.savefig(cfg.output_dir / "latent_umap_by_temperature.png", dpi=180)
        plt.close()
    except Exception:
        pass


def make_ood_aware_predictions(cfg: CFG | None = None):
    cfg = clone_cfg(cfg)
    scores = pd.read_csv(cfg.output_dir / "ood_feature_distance_scores.csv")
    unc_path = cfg.output_dir / "ood_prediction_uncertainty.csv"
    unc = _read_csv_if_nonempty(unc_path)
    if len(unc):
        scores = scores.merge(
            unc[[c for c in ["model_name", "trajectory_id", "end_index", "ensemble_std", "gate_entropy"] if c in unc.columns]],
            on=["model_name", "trajectory_id", "end_index"],
            how="left",
            suffixes=("", "_unc"),
        )
        for col in ["ensemble_std", "gate_entropy"]:
            uc = f"{col}_unc"
            if uc in scores:
                scores[col] = scores[col].where(scores[col].notna(), scores[uc]) if col in scores else scores[uc]
    pieces = []
    for col in ["distance_S4_latent", "ensemble_std", "gate_entropy"]:
        if col in scores:
            sd = scores[col].std()
            pieces.append(((scores[col] - scores[col].mean()) / max(float(sd), 1e-12)).fillna(0.0))
    scores["OOD_score"] = np.sum(pieces, axis=0) if pieces else scores.get("mahalanobis_S3", 0.0)
    scores["uncertainty_score"] = scores.get("ensemble_std", np.nan)
    q70 = scores["OOD_score"].quantile(0.70)
    q90 = scores["OOD_score"].quantile(0.90)
    scores["risk_flag"] = np.select(
        [scores["OOD_score"] > q90, scores["OOD_score"] > q70],
        ["high risk", "medium risk"],
        default="low risk",
    )
    out_cols = [
        "model_name", "trajectory_id", "temperature_C", "drive_cycle", "end_index",
        "SOC_true", "SOC_pred", "OOD_score", "uncertainty_score", "risk_flag",
        "abs_error", "distance_S4_latent", "mahalanobis_S3", "knn_S4_k5",
    ]
    out = scores[[c for c in out_cols if c in scores.columns]].copy()
    out.to_csv(cfg.output_dir / "ood_aware_predictions.csv", index=False)
    return out


def write_ood_summary_report(cfg: CFG | None = None):
    cfg = clone_cfg(cfg)
    corr = _read_csv_if_nonempty(cfg.output_dir / "ood_error_correlation.csv")
    abst = _read_csv_if_nonempty(cfg.output_dir / "ood_abstention_curve.csv")
    omitted = _read_csv_if_nonempty(cfg.output_dir / "omitted_temperature_ood_summary.csv")
    scores = _read_csv_if_nonempty(cfg.output_dir / "ood_feature_distance_scores.csv")
    lines = [
        "# OOD / Uncertainty Diagnostic Summary",
        "",
        "## 1. Omitted-temperature failure summary",
        "Representative temperature coverage stabilizes FUDS SOC mapping, while omitted transition temperatures can produce unstable predictions.",
        "",
    ]
    if len(omitted):
        lines.append(omitted.head(30).to_markdown(index=False))
        lines.append("")
    lines.extend([
        "## 2. OOD score and error correlation",
        "Correlation labels: >=0.5 useful, 0.2-0.5 weak-to-moderate, <0.2 weak.",
        "",
    ])
    if len(corr):
        lines.append(corr.to_markdown(index=False))
        lines.append("")
    if len(scores) and "temperature_C" in scores:
        t20 = scores[np.isclose(scores["temperature_C"].astype(float), 20.0)]
        if len(t20):
            lines.extend([
                "## 3. 20C omitted condition",
                f"20C mean abs error in the scored model: {t20['abs_error'].mean() * 100.0:.3f}%.",
                f"20C mean S3 Mahalanobis score: {t20.get('mahalanobis_S3', pd.Series(dtype=float)).mean():.3f}.",
                f"20C mean S4 latent distance: {t20.get('distance_S4_latent', pd.Series(dtype=float)).mean():.3f}.",
                "",
            ])
    lines.extend([
        "## 4. Risk-coverage curve",
    ])
    if len(abst):
        best = abst.sort_values(["score", "coverage"]).groupby("score").head(1)
        lines.append(best.to_markdown(index=False))
        lines.append("")
    lines.extend([
        "## 5. Safe interpretation",
        "- The model does not guarantee pure temperature extrapolation.",
        "- OOD diagnostics can flag some high-risk omitted-temperature predictions, but should be evaluated by score/error correlation and risk-coverage curves.",
        "- Temperature coverage remains critical for stable SOC mapping.",
        "- Learned dynamic voltage components can be useful under representative temperature coverage but are insufficient alone for guaranteed unseen-temperature robustness.",
        "",
        "## 6. Forbidden interpretation",
        "- The model solves unseen-temperature extrapolation.",
        "- OOD score perfectly detects all extrapolation failures.",
        "- UDA/CORAL solves the omitted-temperature issue.",
        "- Learned voltage components are true physical polarization or true hysteresis.",
    ])
    path = cfg.output_dir / "ood_summary_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_ood_diagnostic_suite(
    cfg: CFG | None = None,
    *,
    experiment="Exp C",
    model_name="R5_GATED",
    include_latent=True,
    latent_epochs=60,
    run_uncertainty=True,
    ensemble_size=2,
    ensemble_epochs=40,
):
    configure_torch_runtime()
    cfg = ood_cfg(cfg, experiment)
    print(f"Running OOD diagnostic suite for {experiment} with model={model_name}")
    scores = compute_ood_feature_distance_scores(
        cfg,
        experiment=experiment,
        model_name=model_name,
        include_latent=include_latent,
        latent_epochs=latent_epochs,
    )
    if run_uncertainty:
        compute_prediction_uncertainty(
            cfg,
            experiment=experiment,
            ensemble_size=ensemble_size,
            ensemble_epochs=ensemble_epochs,
        )
    else:
        warnings.warn("Prediction uncertainty training skipped; correlation/abstention will use feature-space scores only.")
        pd.DataFrame().to_csv(cfg.output_dir / "ood_prediction_uncertainty.csv", index=False)
        pd.DataFrame().to_csv(cfg.output_dir / "ood_gate_uncertainty.csv", index=False)
    correlations = compute_ood_error_correlations(cfg)
    abstention = compute_ood_abstention_curve(cfg)
    omitted = summarize_omitted_temperature_ood(cfg)
    aware = make_ood_aware_predictions(cfg)
    report = write_ood_summary_report(cfg)
    print("OOD diagnostic files written to:", cfg.output_dir.resolve())
    display(correlations["overall"])
    return {
        "scores": scores["scores"],
        "correlations": correlations,
        "abstention": abstention,
        "omitted_summary": omitted,
        "ood_aware_predictions": aware,
        "report_path": report,
    }
