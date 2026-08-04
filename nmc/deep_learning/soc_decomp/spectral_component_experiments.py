from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import CFG, make_cfg
from .runtime import configure_torch_runtime, device
from .models import DecomposedWindowDataset, build_lstm_soc_model, collate_meta_to_frame
from .training import (
    attach_prediction_features,
    build_prediction_feature_lookup,
    make_scaled_frames_for_ablation,
)
from .variance_control import _overall_metrics, variance_by_temperature
from .dft_diagnostic import _rfft_metrics, _trajectory_label


SOURCE_FEATURE_DIR = "decomposed_features_train_temp_minus10_0_10_20_25_50"
STRICT_OMITTED_BRANCH_FEATURE_DIR = "decomposed_features_exp_c_shift_tau_hybrid"

SPECTRAL_EXPERIMENTS = {
    "Exp A": {
        "train_temps": ("N10", "0", "25", "50"),
        "focus_temperature_C": 10.0,
        "source_dir": STRICT_OMITTED_BRANCH_FEATURE_DIR,
        "note": "FUDS omitted 10C. Branch feature cache is used for post-hoc spectral role diagnostics.",
    },
    "Exp B": {
        "train_temps": ("N10", "10", "25", "50"),
        "focus_temperature_C": 0.0,
        "source_dir": STRICT_OMITTED_BRANCH_FEATURE_DIR,
        "note": "FUDS omitted 0C. Branch feature cache is used for post-hoc spectral role diagnostics.",
    },
    "Exp C": {
        "train_temps": ("N10", "0", "10", "25", "50"),
        "focus_temperature_C": 20.0,
        "source_dir": STRICT_OMITTED_BRANCH_FEATURE_DIR,
        "note": "FUDS omitted 20C with branch cache that does not include 20C DST/US06 feature files.",
    },
    "Exp D": {
        "train_temps": ("N10", "0", "10", "20", "25", "50"),
        "focus_temperature_C": 20.0,
        "source_dir": SOURCE_FEATURE_DIR,
        "note": "FUDS 20C included in DST/US06 training coverage.",
    },
    "Omit N10": {
        "train_temps": ("0", "10", "25", "50"),
        "focus_temperature_C": -10.0,
        "source_dir": STRICT_OMITTED_BRANCH_FEATURE_DIR,
        "note": "FUDS outside low-temperature fold.",
    },
    "Omit 50": {
        "train_temps": ("N10", "0", "10", "25"),
        "focus_temperature_C": 50.0,
        "source_dir": STRICT_OMITTED_BRANCH_FEATURE_DIR,
        "note": "FUDS outside high-temperature fold.",
    },
}

REORDER_ABLATIONS = {
    "SR_Vcorr_I_T": ["V_corr_raw", "I_raw", "T"],
    "SR_lowfreq": ["V_corr_raw", "I_raw", "T", "V_pol_lowfreq"],
    "SR_midfreq": ["V_corr_raw", "I_raw", "T", "V_pol_midfreq"],
    "SR_highfreq": ["V_corr_raw", "I_raw", "T", "V_pol_highfreq"],
    "SR_low_mid": ["V_corr_raw", "I_raw", "T", "V_pol_lowfreq", "V_pol_midfreq"],
    "SR_mid_high": ["V_corr_raw", "I_raw", "T", "V_pol_midfreq", "V_pol_highfreq"],
    "SR_all_reordered_pol": ["V_corr_raw", "I_raw", "T", "V_pol_lowfreq", "V_pol_midfreq", "V_pol_highfreq"],
    "SR_full_reordered": [
        "V_raw", "V_corr_raw", "I_raw", "T",
        "V_pol_lowfreq", "V_pol_midfreq", "V_pol_highfreq",
        "V_hys_raw", "V_ohm_raw", "R0",
    ],
}

DETERMINISTIC_ABLATIONS = {
    "DB_Vraw_I_T": ["V_raw", "I_raw", "T"],
    "DB_Vcorr_I_T": ["V_corr_raw", "I_raw", "T"],
    "DB_low": ["V_corr_raw", "I_raw", "T", "residual_low"],
    "DB_mid": ["V_corr_raw", "I_raw", "T", "residual_mid"],
    "DB_high": ["V_corr_raw", "I_raw", "T", "residual_high"],
    "DB_all_bands": ["V_corr_raw", "I_raw", "T", "residual_low", "residual_mid", "residual_high"],
    "DB_raw_corr_all_bands": ["V_raw", "V_corr_raw", "I_raw", "T", "residual_low", "residual_mid", "residual_high"],
}


def clone_cfg(cfg: CFG | None = None) -> CFG:
    src = cfg or make_cfg()
    out = make_cfg()
    for f in fields(CFG):
        setattr(out, f.name, getattr(src, f.name))
    return out


def temp_key_to_float(v) -> float:
    s = str(v).strip().upper()
    return -float(s[1:]) if s.startswith("N") else float(s)


def _move(x, cfg: CFG):
    return x.to(
        device=device,
        dtype=torch.float32,
        non_blocking=bool(getattr(cfg, "cuda_non_blocking", True)) and device.type == "cuda",
    )


def _load_all_frames(feature_dir: Path) -> list[pd.DataFrame]:
    frames = []
    for path in sorted(feature_dir.glob("*_features.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["_source_feature_file"] = path.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No *_features.csv files found in {feature_dir}")
    return frames


def _split_frames(
    frames: list[pd.DataFrame],
    *,
    train_temps: tuple[str, ...],
    eval_drive: str = "FUDS",
) -> dict[str, list[pd.DataFrame]]:
    train_temp_c = {temp_key_to_float(t) for t in train_temps}
    train, test = [], []
    for frame in frames:
        temp = float(frame["temperature"].iloc[0])
        drive = str(frame["drive_cycle"].iloc[0]).upper()
        if drive in {"DST", "US06"} and temp in train_temp_c:
            train.append(frame)
        elif drive == eval_drive.upper():
            test.append(frame)
    train_ids = {f["trajectory_id"].iloc[0] for f in train}
    test_ids = {f["trajectory_id"].iloc[0] for f in test}
    assert train_ids.isdisjoint(test_ids), "Train/test leakage in spectral component split"
    if not train or not test:
        raise ValueError(f"Empty train/test split: train={len(train)} test={len(test)}")
    return {"train": train, "valid": [], "test": test}


def _safe_rfft_centroid(x: np.ndarray) -> tuple[float, float]:
    _, summary, _ = _rfft_metrics(np.asarray(x, dtype=np.float64), dt_sec=1.0)
    return (
        float(summary.get("spectral_centroid_Hz", np.nan)),
        float(summary.get("high_0p05_Nyquist_power_fraction", np.nan)),
    )


def build_spectral_reordered_feature_dir(
    source_dir: Path,
    out_dir: Path,
    *,
    force: bool = False,
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir.parent / "spectral_reordered_components.csv"
    summary_path = out_dir.parent / "spectral_reorder_summary.csv"
    if combined_path.exists() and summary_path.exists() and list(out_dir.glob("*_features.csv")) and not force:
        return out_dir, pd.read_csv(combined_path), pd.read_csv(summary_path)

    frames = _load_all_frames(source_dir)
    combined = []
    rows = []
    required = ["V_pol_fast_raw", "V_pol_mid_raw", "V_pol_slow_raw"]
    for frame in frames:
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise KeyError(f"{frame['_source_feature_file'].iloc[0]} missing branch columns {missing}")
        f = frame.copy()
        branch_stats = []
        for col in required:
            centroid, high_frac = _safe_rfft_centroid(f[col].to_numpy(np.float64))
            branch_stats.append((col, centroid, high_frac))
        ordered = sorted(branch_stats, key=lambda x: (np.nan_to_num(x[1], nan=np.inf), x[0]))
        mapping = {
            "V_pol_lowfreq": ordered[0][0],
            "V_pol_midfreq": ordered[1][0],
            "V_pol_highfreq": ordered[2][0],
        }
        for new_col, src_col in mapping.items():
            f[new_col] = f[src_col].astype(np.float32)
        meta = {
            "trajectory_id": str(f["trajectory_id"].iloc[0]),
            "temperature_C": float(f["temperature"].iloc[0]),
            "drive_cycle": str(f["drive_cycle"].iloc[0]).upper(),
            "source_feature_file": str(f["_source_feature_file"].iloc[0]),
        }
        for rank_name, (src_col, centroid, high_frac) in zip(["lowfreq", "midfreq", "highfreq"], ordered):
            rows.append({
                **meta,
                "rank": rank_name,
                "source_branch": src_col,
                "spectral_centroid_Hz": centroid,
                "high_0p05_Nyquist_power_fraction": high_frac,
            })
        out_file = out_dir / str(f["_source_feature_file"].iloc[0])
        f.drop(columns=["_source_feature_file"], errors="ignore").to_csv(out_file, index=False)
        combined.append(f.drop(columns=["_source_feature_file"], errors="ignore"))
    combined_df = pd.concat(combined, ignore_index=True)
    summary_df = pd.DataFrame(rows)
    combined_df.to_csv(combined_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    return out_dir, combined_df, summary_df


def _ema_causal(x: np.ndarray, cutoff_hz: float, dt_sec: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x
    alpha = float(np.exp(-2.0 * np.pi * float(cutoff_hz) * float(dt_sec)))
    alpha = min(max(alpha, 0.0), 0.999999)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * y[i - 1] + (1.0 - alpha) * x[i]
    return y


def build_deterministic_band_feature_dir(
    source_dir: Path,
    out_dir: Path,
    *,
    slow_cutoff_hz: float = 0.003,
    mid_cutoff_hz: float = 0.03,
    residual_source: str = "V_raw_minus_V_corr",
    force: bool = False,
) -> tuple[Path, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir.parent / "deterministic_band_decomp_feature_summary.csv"
    if summary_path.exists() and list(out_dir.glob("*_features.csv")) and not force:
        return out_dir, pd.read_csv(summary_path)
    frames = _load_all_frames(source_dir)
    rows = []
    for frame in frames:
        f = frame.copy()
        if residual_source == "V_raw_minus_V_corr":
            residual = f["V_raw"].to_numpy(np.float64) - f["V_corr_raw"].to_numpy(np.float64)
        elif residual_source == "component_sum":
            residual = (
                f["V_pol_raw"].to_numpy(np.float64)
                + f["V_hys_raw"].to_numpy(np.float64)
                + f["V_ohm_raw"].to_numpy(np.float64)
            )
        else:
            raise ValueError(f"Unknown residual source: {residual_source}")
        lp_slow = _ema_causal(residual, slow_cutoff_hz)
        lp_mid = _ema_causal(residual, mid_cutoff_hz)
        low = lp_slow
        mid = lp_mid - lp_slow
        high = residual - lp_mid
        f["residual_raw_for_band"] = residual.astype(np.float32)
        f["residual_low"] = low.astype(np.float32)
        f["residual_mid"] = mid.astype(np.float32)
        f["residual_high"] = high.astype(np.float32)
        for col in ["residual_low", "residual_mid", "residual_high"]:
            centroid, high_frac = _safe_rfft_centroid(f[col].to_numpy(np.float64))
            rows.append({
                "trajectory_id": str(f["trajectory_id"].iloc[0]),
                "temperature_C": float(f["temperature"].iloc[0]),
                "drive_cycle": str(f["drive_cycle"].iloc[0]).upper(),
                "source_feature_file": str(f["_source_feature_file"].iloc[0]),
                "residual_source": residual_source,
                "slow_cutoff_Hz": float(slow_cutoff_hz),
                "mid_cutoff_Hz": float(mid_cutoff_hz),
                "component": col,
                "spectral_centroid_Hz": centroid,
                "high_0p05_Nyquist_power_fraction": high_frac,
                "rms": float(np.sqrt(np.mean(np.asarray(f[col], dtype=np.float64) ** 2))),
            })
        f.drop(columns=["_source_feature_file"], errors="ignore").to_csv(out_dir / str(f["_source_feature_file"].iloc[0]), index=False)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(summary_path, index=False)
    return out_dir, summary_df


@torch.no_grad()
def _predict(model, loader, cfg: CFG) -> pd.DataFrame:
    rows = []
    model.eval()
    for x, y, meta in loader:
        yp = model(_move(x, cfg)).detach().cpu().numpy()[:, 0]
        yy = y.numpy()[:, 0]
        mdf = collate_meta_to_frame(meta)
        mdf["target_label"] = "physical"
        mdf["y_true"] = yy
        mdf["y_pred"] = yp
        rows.append(mdf)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out):
        out["error"] = out["y_pred"] - out["y_true"]
        out["abs_error"] = np.abs(out["error"])
    return out


def _train_generic_lstm(
    feature_frames: dict[str, list[pd.DataFrame]],
    feature_cols: list[str],
    cfg: CFG,
    *,
    model_name: str,
    experiment: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaled, _ = make_scaled_frames_for_ablation(feature_frames, feature_cols)
    train_ds = DecomposedWindowDataset(scaled["train"], feature_cols, cfg.window_len, cfg.stride, target_label="physical")
    test_ds = DecomposedWindowDataset(scaled["test"], feature_cols, cfg.window_len, cfg.stride, target_label="physical")
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise ValueError(f"Empty windows for {experiment} {model_name}")
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(getattr(cfg, "dataloader_num_workers", 0)),
        pin_memory=bool(getattr(cfg, "dataloader_pin_memory", True)) and device.type == "cuda",
    )
    test_loader = DataLoader(test_ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)
    # These are feature-ablation diagnostics, so use the same plain stateless
    # LSTM head for every feature set.  The gated head expects the original
    # V_pol/V_hys/V_ohm triplet and would confound post-hoc reordered columns.
    model = build_lstm_soc_model(feature_cols, 1, cfg, "A3_V_raw_I_T").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lstm_lr), weight_decay=float(cfg.lstm_weight_decay))
    history = []
    early_stop = bool(getattr(cfg, "lstm_early_stop", True))
    warmup = int(getattr(cfg, "lstm_plateau_warmup_epochs", 50))
    patience = int(getattr(cfg, "lstm_plateau_patience", 35))
    min_delta = float(getattr(cfg, "lstm_plateau_min_delta", 1e-4))
    best = float("inf")
    bad = 0
    for ep in range(1, int(cfg.lstm_epochs) + 1):
        model.train()
        losses = []
        for x, y, _ in train_loader:
            x = _move(x, cfg)
            y = _move(y, cfg)
            pred = model(x)
            loss = F.smooth_l1_loss(pred, y, beta=0.02)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(getattr(cfg, "grad_clip", 1.0)))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(losses))
        row = {
            "experiment": experiment,
            "model_name": model_name,
            "epoch": ep,
            "train_loss": train_loss,
        }
        history.append(row)
        print_every = max(1, int(getattr(cfg, "lstm_print_every", 25)))
        if ep == 1 or ep == int(cfg.lstm_epochs) or ep % print_every == 0:
            print(f"{experiment} {model_name} epoch={ep} train_loss={train_loss:.5f}")
        if train_loss < best - min_delta:
            best = train_loss
            bad = 0
        else:
            bad += 1
        if early_stop and ep >= warmup and bad >= patience:
            row["stopped_early"] = True
            row["stop_reason"] = f"plateau monitor=train_loss best={best:.6f} patience={patience} min_delta={min_delta}"
            history[-1] = row
            print(f"{experiment} {model_name} early-stop at epoch={ep}: {row['stop_reason']}")
            break
    pred = _predict(model, test_loader, cfg)
    pred["model_name"] = model_name
    pred["experiment"] = experiment
    return pd.DataFrame(history), pred


def _summarize_predictions(pred: pd.DataFrame, focus_temp: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if pred.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    results = _overall_metrics(pred)
    by_temp = variance_by_temperature(pred)
    focus_rows = []
    for model, g in by_temp.groupby("model_name"):
        omitted = g[np.isclose(g["temperature_C"].astype(float), float(focus_temp))]
        seen = g[~np.isclose(g["temperature_C"].astype(float), float(focus_temp))]
        focus_rows.append({
            "model_name": model,
            "focus_temperature_C": float(focus_temp),
            "focus_MAE_pct": float(omitted["MAE_pct"].iloc[0]) if len(omitted) else np.nan,
            "focus_RMSE_pct": float(omitted["RMSE_pct"].iloc[0]) if len(omitted) else np.nan,
            "focus_jitter_ratio": float(omitted["jitter_ratio"].iloc[0]) if len(omitted) else np.nan,
            "focus_high_frequency_error_energy": float(omitted["high_frequency_error_energy"].iloc[0]) if len(omitted) else np.nan,
            "seen_MAE_pct": float(seen["MAE_pct"].mean()) if len(seen) else np.nan,
            "worst_temperature_MAE_pct": float(g["MAE_pct"].max()) if len(g) else np.nan,
            "temperature_MAE_variance": float(g["MAE_pct"].var(ddof=0)) if len(g) else np.nan,
        })
    focus = pd.DataFrame(focus_rows)
    return results, by_temp, focus


def _prepare_cfg(cfg: CFG | None = None) -> CFG:
    out = clone_cfg(cfg)
    out.smoke_mode = False
    out.window_len = int(getattr(out, "window_len", 50))
    out.stride = int(getattr(out, "stride", 1))
    out.lstm_epochs = int(getattr(out, "lstm_epochs", 300))
    out.batch_size = int(getattr(out, "batch_size", 8192))
    out.lstm_print_every = int(getattr(out, "lstm_print_every", 25))
    out.lstm_early_stop = bool(getattr(out, "lstm_early_stop", True))
    out.lstm_plateau_warmup_epochs = int(getattr(out, "lstm_plateau_warmup_epochs", 50))
    out.lstm_plateau_patience = int(getattr(out, "lstm_plateau_patience", 35))
    out.lstm_plateau_min_delta = float(getattr(out, "lstm_plateau_min_delta", 1e-4))
    return out


def _run_ablation_grid(
    *,
    feature_dir: Path,
    experiments: tuple[str, ...],
    ablations: dict[str, list[str]],
    output_prefix: str,
    cfg: CFG | None = None,
) -> dict[str, pd.DataFrame]:
    cfg = _prepare_cfg(cfg)
    paths = {
        "history": Path(f"{output_prefix}_history.csv"),
        "predictions": Path(f"{output_prefix}_prediction_rows.csv"),
        "results": Path(f"{output_prefix}_results.csv"),
        "by_temperature": Path(f"{output_prefix}_by_temperature.csv"),
        "focus": Path(f"{output_prefix}_focus.csv"),
    }
    all_pred = [pd.read_csv(paths["predictions"])] if paths["predictions"].exists() else []
    all_hist = [pd.read_csv(paths["history"])] if paths["history"].exists() else []
    all_res = [pd.read_csv(paths["results"])] if paths["results"].exists() else []
    all_by_temp = [pd.read_csv(paths["by_temperature"])] if paths["by_temperature"].exists() else []
    all_focus = [pd.read_csv(paths["focus"])] if paths["focus"].exists() else []
    done_models = set()
    if all_focus and len(all_focus[0]) and "model_name" in all_focus[0]:
        done_models = set(all_focus[0]["model_name"].dropna().astype(str))
    for experiment in experiments:
        spec = SPECTRAL_EXPERIMENTS[experiment]
        frames = _load_all_frames(feature_dir)
        feature_frames = _split_frames(frames, train_temps=spec["train_temps"], eval_drive="FUDS")
        lookup = build_prediction_feature_lookup(feature_frames)
        for model_name, cols in ablations.items():
            full_model_name = f"{model_name}_{experiment.replace(' ', '_')}"
            if full_model_name in done_models:
                print(f"skip completed {experiment} {full_model_name}")
                continue
            missing = sorted({c for c in cols if c not in feature_frames["train"][0].columns})
            if missing:
                print(f"skip {experiment} {model_name}: missing {missing}")
                continue
            hist, pred_raw = _train_generic_lstm(
                feature_frames,
                cols,
                cfg,
                model_name=full_model_name,
                experiment=experiment,
            )
            attached = attach_prediction_features(
                pred_raw.assign(split="test", ablation=pred_raw["model_name"]),
                lookup,
                ablation_name=pred_raw["model_name"].iloc[0],
                target_label="physical",
            )
            attached["experiment"] = experiment
            res, by_temp, focus = _summarize_predictions(attached, spec["focus_temperature_C"])
            for df in (res, by_temp, focus):
                if len(df):
                    df["experiment"] = experiment
                    df["feature_dir"] = str(feature_dir)
                    df["experiment_note"] = spec.get("note", "")
            all_hist.append(hist)
            all_pred.append(attached)
            all_res.append(res)
            all_by_temp.append(by_temp)
            all_focus.append(focus)
            done_models.add(full_model_name)
            # Incremental writes make long runs restartable enough for notebook use.
            pd.concat(all_hist, ignore_index=True).to_csv(f"{output_prefix}_history.csv", index=False)
            pd.concat(all_pred, ignore_index=True).to_csv(f"{output_prefix}_prediction_rows.csv", index=False)
            pd.concat(all_res, ignore_index=True).to_csv(f"{output_prefix}_results.csv", index=False)
            pd.concat(all_by_temp, ignore_index=True).to_csv(f"{output_prefix}_by_temperature.csv", index=False)
            pd.concat(all_focus, ignore_index=True).to_csv(f"{output_prefix}_focus.csv", index=False)
    return {
        "history": pd.concat(all_hist, ignore_index=True) if all_hist else pd.DataFrame(),
        "predictions": pd.concat(all_pred, ignore_index=True) if all_pred else pd.DataFrame(),
        "results": pd.concat(all_res, ignore_index=True) if all_res else pd.DataFrame(),
        "by_temperature": pd.concat(all_by_temp, ignore_index=True) if all_by_temp else pd.DataFrame(),
        "focus": pd.concat(all_focus, ignore_index=True) if all_focus else pd.DataFrame(),
    }


def run_spectral_reorder_soc(
    cfg: CFG | None = None,
    *,
    experiments: tuple[str, ...] = ("Exp A", "Exp B", "Exp C", "Exp D", "Omit N10", "Omit 50"),
    source_dir: str | Path = STRICT_OMITTED_BRANCH_FEATURE_DIR,
    force_features: bool = False,
) -> dict[str, pd.DataFrame]:
    configure_torch_runtime()
    source_dir = Path(source_dir)
    out_dir = Path("decomposed_features_spectral_reordered")
    build_spectral_reordered_feature_dir(source_dir, out_dir, force=force_features)
    return _run_ablation_grid(
        feature_dir=out_dir,
        experiments=experiments,
        ablations=REORDER_ABLATIONS,
        output_prefix="spectral_reorder_soc",
        cfg=cfg,
    )


def run_deterministic_band_soc(
    cfg: CFG | None = None,
    *,
    experiments: tuple[str, ...] = ("Exp A", "Exp B", "Exp C", "Exp D", "Omit N10", "Omit 50"),
    source_dir: str | Path = STRICT_OMITTED_BRANCH_FEATURE_DIR,
    residual_source: str = "V_raw_minus_V_corr",
    slow_cutoff_hz: float = 0.003,
    mid_cutoff_hz: float = 0.03,
    force_features: bool = False,
) -> dict[str, pd.DataFrame]:
    configure_torch_runtime()
    source_dir = Path(source_dir)
    out_dir = Path(f"decomposed_features_deterministic_band_{residual_source}_{slow_cutoff_hz:g}_{mid_cutoff_hz:g}".replace(".", "p"))
    build_deterministic_band_feature_dir(
        source_dir,
        out_dir,
        slow_cutoff_hz=slow_cutoff_hz,
        mid_cutoff_hz=mid_cutoff_hz,
        residual_source=residual_source,
        force=force_features,
    )
    return _run_ablation_grid(
        feature_dir=out_dir,
        experiments=experiments,
        ablations=DETERMINISTIC_ABLATIONS,
        output_prefix="deterministic_band_decomp",
        cfg=cfg,
    )


def drivecycle_spectral_domain_diagnostic(
    prediction_csv: str | Path = "rex_prediction_rows.csv",
    dft_summary_csv: str | Path = "dft_component_diagnostic/dft_component_summary_by_temp_cycle.csv",
    output_dir: str | Path = "spectral_domain_error_plots",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = Path(prediction_csv)
    dft_path = Path(dft_summary_csv)
    if not pred_path.exists() or not dft_path.exists():
        raise FileNotFoundError(f"Need {pred_path} and {dft_path}")
    pred = pd.read_csv(pred_path)
    if "temperature_C" not in pred and "temperature" in pred:
        pred["temperature_C"] = pred["temperature"]
    dft = pd.read_csv(dft_path)
    keep_components = ["I_raw", "V_raw", "V_pol_raw", "V_ohm_raw", "V_hys_raw", "R0"]
    if "I_raw" not in dft["component"].unique():
        # DFT diagnostic did not include I_raw originally; compute its metrics from the feature cache.
        rows = []
        for p in Path(SOURCE_FEATURE_DIR).glob("*_features.csv"):
            f = pd.read_csv(p)
            if "I_raw" not in f:
                continue
            _, summ, _ = _rfft_metrics(f["I_raw"].to_numpy(np.float64), dt_sec=1.0)
            rows.append({
                "trajectory_id": str(f["trajectory_id"].iloc[0]),
                "temperature_C": float(f["temperature"].iloc[0]),
                "drive_cycle": str(f["drive_cycle"].iloc[0]).upper(),
                "component": "I_raw",
                **summ,
            })
        if rows:
            dft = pd.concat([dft, pd.DataFrame(rows)], ignore_index=True, sort=False)
    spec_metrics = dft[dft["component"].isin(keep_components)].copy()
    wide = spec_metrics.pivot_table(
        index=["temperature_C", "drive_cycle"],
        columns="component",
        values=["spectral_centroid_Hz", "high_0p05_Nyquist_power_fraction"],
        aggfunc="mean",
    )
    wide.columns = [f"{metric}_{comp}" for metric, comp in wide.columns]
    wide = wide.reset_index()
    metric_rows = []
    for (model, temp, drive, tid), g in pred.groupby(["model_name", "temperature_C", "drive_cycle", "trajectory_id"]):
        if len(g) < 4:
            continue
        err = g["y_pred"].to_numpy(np.float64) - g["y_true"].to_numpy(np.float64)
        dyp = np.diff(g.sort_values("end_index")["y_pred"].to_numpy(np.float64))
        dyt = np.diff(g.sort_values("end_index")["y_true"].to_numpy(np.float64))
        metric_rows.append({
            "model_name": model,
            "temperature_C": float(temp),
            "drive_cycle": str(drive).upper(),
            "trajectory_id": tid,
            "MAE_pct": float(np.mean(np.abs(err)) * 100.0),
            "RMSE_pct": float(np.sqrt(np.mean(err**2)) * 100.0),
            "jitter_ratio": float(np.mean(np.abs(dyp)) / max(np.mean(np.abs(dyt)), 1e-12)),
            "error_std_pct": float(np.std(err) * 100.0),
        })
    metrics = pd.DataFrame(metric_rows)
    merged = metrics.merge(wide, on=["temperature_C", "drive_cycle"], how="left")
    merged.to_csv("drivecycle_spectral_domain_metrics.csv", index=False)
    corr_rows = []
    y_cols = ["MAE_pct", "RMSE_pct", "jitter_ratio", "error_std_pct"]
    x_cols = [c for c in merged.columns if c.startswith("spectral_centroid") or c.startswith("high_")]
    for model, g in merged.groupby("model_name"):
        for y in y_cols:
            for x in x_cols:
                gg = g[[x, y]].dropna()
                if len(gg) < 4 or gg[x].std() <= 1e-12 or gg[y].std() <= 1e-12:
                    corr = np.nan
                else:
                    corr = float(np.corrcoef(gg[x], gg[y])[0, 1])
                corr_rows.append({"model_name": model, "error_metric": y, "spectral_metric": x, "correlation": corr})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv("spectral_domain_error_correlation.csv", index=False)
    return merged, corr


def write_component_separation_report() -> Path:
    lines = ["# Component Separation Report\n"]
    def add_table(title, path, cols=None, n=20):
        lines.append(f"\n## {title}\n")
        p = Path(path)
        if not p.exists():
            lines.append(f"`{path}` not available.\n")
            return
        df = pd.read_csv(p)
        if cols:
            df = df[[c for c in cols if c in df.columns]]
        lines.append(df.head(n).to_markdown(index=False, floatfmt=".4f"))
        lines.append("\n")
    add_table(
        "Spectral Reorder Mapping",
        "spectral_reorder_summary.csv",
        ["trajectory_id", "temperature_C", "drive_cycle", "rank", "source_branch", "spectral_centroid_Hz", "high_0p05_Nyquist_power_fraction"],
        n=30,
    )
    add_table(
        "Spectral Reorder SOC Focus",
        "spectral_reorder_soc_focus.csv",
        ["experiment", "model_name", "focus_temperature_C", "focus_MAE_pct", "focus_RMSE_pct", "focus_jitter_ratio", "worst_temperature_MAE_pct"],
        n=40,
    )
    add_table(
        "Deterministic Band SOC Focus",
        "deterministic_band_decomp_focus.csv",
        ["experiment", "model_name", "focus_temperature_C", "focus_MAE_pct", "focus_RMSE_pct", "focus_jitter_ratio", "worst_temperature_MAE_pct"],
        n=40,
    )
    if Path("spectral_domain_error_correlation.csv").exists():
        corr = pd.read_csv("spectral_domain_error_correlation.csv")
        corr["abs_correlation"] = corr["correlation"].abs()
        lines.append("\n## Strongest Spectral Domain/Error Correlations\n")
        lines.append(corr.sort_values("abs_correlation", ascending=False).head(25).to_markdown(index=False, floatfmt=".4f"))
        lines.append("\n")
    lines.append("\n## Interpretation Guardrails\n")
    lines.append("- Nominal fast/mid/slow branch names are architectural labels unless a time-scale constrained corrector is used.\n")
    lines.append("- Post-hoc spectral reorder tests SOC usefulness by actual DFT role, not by branch name.\n")
    lines.append("- Deterministic causal bands test whether simple frequency features can replace learned components.\n")
    lines.append("- DFT does not prove physical component identification.\n")
    lines.append("- Learned voltage components should be described as dynamic voltage features, not true polarization/hysteresis/R0.\n")
    lines.append("\n## Pending/Conditional\n")
    lines.append("- Slow-only or slow+mid SOC estimation should be run only after a constrained corrector shows actual low/mid/high separation.\n")
    path = Path("component_separation_report.md")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["features", "reorder_soc", "deterministic_soc", "spectral_domain", "report", "all"], default="features")
    parser.add_argument("--experiments", default="Exp A,Exp B,Exp C,Exp D,Omit N10,Omit 50")
    parser.add_argument("--source-dir", default=STRICT_OMITTED_BRANCH_FEATURE_DIR)
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()

    experiments = tuple(x.strip() for x in args.experiments.split(",") if x.strip())
    cfg = make_cfg()
    cfg.smoke_mode = False
    cfg.window_len = 50
    cfg.stride = 1
    cfg.lstm_epochs = 300
    cfg.batch_size = 8192
    cfg.lstm_print_every = 25
    cfg.lstm_early_stop = True
    cfg.lstm_plateau_warmup_epochs = 50
    cfg.lstm_plateau_patience = 35
    cfg.lstm_plateau_min_delta = 1e-4

    if args.phase in ("features", "all"):
        build_spectral_reordered_feature_dir(Path(args.source_dir), Path("decomposed_features_spectral_reordered"), force=args.force_features)
        build_deterministic_band_feature_dir(Path(args.source_dir), Path("decomposed_features_deterministic_band_V_raw_minus_V_corr_0p003_0p03"), force=args.force_features)
    if args.phase in ("reorder_soc", "all"):
        run_spectral_reorder_soc(cfg, experiments=experiments, source_dir=args.source_dir, force_features=False)
    if args.phase in ("deterministic_soc", "all"):
        run_deterministic_band_soc(cfg, experiments=experiments, source_dir=args.source_dir, force_features=False)
    if args.phase in ("spectral_domain", "all"):
        drivecycle_spectral_domain_diagnostic()
    if args.phase in ("report", "all"):
        write_component_separation_report()


if __name__ == "__main__":
    main()
