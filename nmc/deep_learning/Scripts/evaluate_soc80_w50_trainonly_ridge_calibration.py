#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from soc_decomp.nmc_vcorr_it_condinv_staged_exact import CondInvVariant  # noqa: E402
from soc_decomp.nmc_vcorr_it_train_dst_selector_run import (  # noqa: E402
    TrainDSTSelectorConfig,
    _cached_feature_frames,
    _cached_find_csv_files,
    _cached_r0_by_temperature,
    _collect_eval_tensors,
    _make_eval_loader,
    _make_stage1_model,
    _prepare_seed_shared_assets,
    _predict_from_tensor,
    _selected_feature_columns,
    _split_train_profile_blocks_for_validation,
)
from soc_decomp.runtime import configure_torch_runtime, device  # noqa: E402


OUT_DIR = PROJECT_ROOT / "nmc_goal_vcorr_it_train_dst_selector_results"
TAG = "soc80_w50_lopo_w1p4_lam1_trainonly_ridge_calibration_v1"
BY_TEMP_OUT = OUT_DIR / f"{TAG}_by_temp.csv"
AGG_OUT = OUT_DIR / f"{TAG}_0c_agg.csv"
MANIFEST_OUT = OUT_DIR / f"{TAG}_manifest.csv"
REPORT_OUT = OUT_DIR / f"{TAG}.md"

RUN_PREFIXES = (
    "soc80_goal_w50_deterministic_VALIDATION_w1p4_lam1_seed02_v1_w1p4_w25x2p2_under0p5_lam1p0_seed02_complete_holdoutVALIDATION_seeds02_b2048_e200",
    "soc80_goal_w50_deterministic_VALIDATION_seed1_nearbest_grid_v1_grid_nearbest_w1p4_w25x2p2_under0p5_lam1p0_holdoutVALIDATION_seeds1_b2048_e200",
    "soc80_goal_w50_deterministic_lopo_w1p4_lam1_seed012_v1_w1p4_w25x2p2_under0p5_lam1p0_fixedtau75_candidate_holdoutdst_seeds012_b2048_e200",
    "soc80_goal_w50_deterministic_lopo_w1p4_lam1_seed012_v1_w1p4_w25x2p2_under0p5_lam1p0_fixedtau75_candidate_holdoutfuds_seeds012_b2048_e200",
    "soc80_goal_w50_deterministic_lopo_w1p4_lam1_seed012_v1_w1p4_w25x2p2_under0p5_lam1p0_fixedtau75_candidate_holdoutus06_seeds012_b2048_e200",
)

RIDGE_ALPHA = 1.0
CLIP_FRACTION = 0.008


def _cfg_from_metadata(prefix: str) -> TrainDSTSelectorConfig:
    meta_path = OUT_DIR / f"{prefix}_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    allowed = {f.name for f in fields(TrainDSTSelectorConfig)}
    kwargs = {k: v for k, v in meta.items() if k in allowed}
    for key in (
        "seeds",
        "train_profiles",
        "valid_profiles",
        "test_profiles",
        "train_temperatures",
        "valid_temperatures",
        "test_temperatures",
        "finetune_temperatures",
        "stage2_train_temperatures",
        "stage2_valid_temperatures",
        "stage2_test_temperatures",
    ):
        if key in kwargs:
            kwargs[key] = tuple(kwargs[key])
    for key in ("base_dir", "raw_root"):
        if key in kwargs:
            p = Path(kwargs[key])
            kwargs[key] = p if p.is_absolute() else PROJECT_ROOT / p
    return TrainDSTSelectorConfig(**kwargs)


def _variant_from_cfg(cfg: TrainDSTSelectorConfig) -> CondInvVariant:
    return CondInvVariant(
        f"condinv_{cfg.recurrent}{cfg.layers}_{cfg.temp_mode}_{cfg.head_kind}_mmd0p02_trainDST25_selector_base",
        recurrent=str(cfg.recurrent),
        layers=int(cfg.layers),
        kernel_size=int(cfg.kernel_size),
        head_kind=str(cfg.head_kind),
        temp_mode=str(cfg.temp_mode),
        dropout=float(cfg.dropout),
        lambda_rex=float(cfg.lambda_rex),
        lambda_condinv=float(cfg.lambda_condinv),
        weight_0=float(cfg.weight_0),
        weight_25=float(cfg.weight_25),
        weight_45=float(cfg.weight_45),
        lr=float(cfg.lr),
        sequence_training=bool(cfg.sequence_training),
    )


def _prepare_loaders(cfg: TrainDSTSelectorConfig, prefix: str):
    files_ = _cached_find_csv_files(Path(cfg.raw_root))
    r0_df = _cached_r0_by_temperature(cfg, files_)
    frames_ = _cached_feature_frames(cfg, files_, r0_df)
    frames_ = _split_train_profile_blocks_for_validation(
        frames_,
        cfg,
        OUT_DIR / f"{prefix}_ridge_internal_valid_split_audit.csv",
    )
    shared = _prepare_seed_shared_assets(cfg, frames_, OUT_DIR)
    train_loader = _make_eval_loader(shared["train_ds"], cfg)
    test_loader = shared["test_loader"]
    return shared, train_loader, test_loader


def _load_model(cfg: TrainDSTSelectorConfig, prefix: str, seed: int, feature_cols: list[str]):
    variant = _variant_from_cfg(cfg)
    model = _make_stage1_model(cfg, variant, input_dim=len(feature_cols), feature_cols=feature_cols)
    weight_path = OUT_DIR / f"{prefix}_seed{seed}_final_epoch{cfg.epochs}_weights.pt"
    payload = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, weight_path


def _calib_features(x: torch.Tensor, pred: np.ndarray, feature_cols: list[str]) -> np.ndarray:
    arr = x.detach().cpu().numpy().astype(np.float64)
    idx = {name: i for i, name in enumerate(feature_cols)}

    def col(name: str) -> np.ndarray:
        return arr[:, :, idx[name]]

    v = col("V_corr_raw")
    i = col("I_raw")
    temp = col("T")
    if "dI" in idx:
        di = col("dI")
    else:
        di = np.diff(i, axis=1, prepend=i[:, :1])
    vcorr_span = np.nanmax(v, axis=1) - np.nanmin(v, axis=1)
    v_mean = np.nanmean(v, axis=1)
    i_mean = np.nanmean(i, axis=1)
    i_std = np.nanstd(i, axis=1)
    abs_i_mean = np.nanmean(np.abs(i), axis=1)
    di_energy = np.nanmean(di * di, axis=1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)

    feats = [
        pred,
        pred * pred,
        temp[:, -1],
        v[:, -1],
        v_mean,
        v_mean - v[:, -1],
        vcorr_span,
        i[:, -1],
        i_mean,
        i_std,
        abs_i_mean,
        di_energy,
    ]
    for extra in (
        "V_corr_raw_ema50",
        "V_corr_raw_ema200",
        "V_corr_raw_dev_ema50",
        "V_corr_raw_dev_ema200",
        "V_corr_raw_span50",
        "I_raw_std50",
        "V_corr_raw_ema50_minus_ema800",
    ):
        if extra in idx:
            feats.append(col(extra)[:, -1])
    return np.column_stack(feats)


def _fit_ridge(x_train: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=0)
    sig = x_train.std(axis=0)
    sig[sig < 1e-8] = 1.0
    z = (x_train - mu) / sig
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1]) * float(RIDGE_ALPHA)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ residual)
    return beta, mu, sig


def _apply_ridge(x: np.ndarray, beta: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    z = (x - mu) / sig
    design = np.column_stack([np.ones(len(z)), z])
    corr = design @ beta
    return np.clip(corr, -float(CLIP_FRACTION), float(CLIP_FRACTION))


def _metric_rows(meta: pd.DataFrame, y_true: np.ndarray, pred: np.ndarray, mode: str, prefix: str, seed: int, weight_path: Path) -> list[dict]:
    d = meta.copy()
    d["y_true"] = y_true
    d["y_pred"] = pred
    d["error"] = d["y_pred"] - d["y_true"]
    d["abs_error"] = d["error"].abs()
    rows = []
    for temp, g in d.groupby("temperature", sort=True):
        rows.append(
            {
                "prefix": prefix,
                "weight_path": weight_path.name,
                "holdout": str(g["drive_cycle"].iloc[0]).upper(),
                "seed": int(seed),
                "mode": mode,
                "temperature_C": float(temp),
                "n_windows": int(len(g)),
                "mae_pct": float(g["abs_error"].mean() * 100.0),
                "bias_pct": float(g["error"].mean() * 100.0),
            }
        )
    rows.append(
        {
            "prefix": prefix,
            "weight_path": weight_path.name,
            "holdout": str(d["drive_cycle"].iloc[0]).upper(),
            "seed": int(seed),
            "mode": mode,
            "temperature_C": "ALL",
            "n_windows": int(len(d)),
            "mae_pct": float(d["abs_error"].mean() * 100.0),
            "bias_pct": float(d["error"].mean() * 100.0),
        }
    )
    return rows


def main() -> None:
    configure_torch_runtime()
    all_rows: list[dict] = []
    manifest: list[dict] = []
    for prefix in RUN_PREFIXES:
        cfg = _cfg_from_metadata(prefix)
        feature_cols = _selected_feature_columns(str(cfg.feature_set))
        shared, train_loader, test_loader = _prepare_loaders(cfg, prefix)
        x_train, y_train, _meta_train = _collect_eval_tensors(train_loader)
        x_test, y_test, meta_test = _collect_eval_tensors(test_loader)
        for seed in cfg.seeds:
            model, weight_path = _load_model(cfg, prefix, int(seed), feature_cols)
            pred_train = _predict_from_tensor(model, x_train, int(cfg.batch_size))
            pred_test = _predict_from_tensor(model, x_test, int(cfg.batch_size))
            residual = y_train.astype(np.float64) - pred_train.astype(np.float64)
            xcal_train = _calib_features(x_train, pred_train, feature_cols)
            xcal_test = _calib_features(x_test, pred_test, feature_cols)
            beta, mu, sig = _fit_ridge(xcal_train, residual)
            corr_test = _apply_ridge(xcal_test, beta, mu, sig)
            pred_corr = np.clip(pred_test.astype(np.float64) + corr_test, 0.0, 1.0)
            all_rows.extend(_metric_rows(meta_test, y_test, pred_test, "raw_loaded", prefix, int(seed), weight_path))
            all_rows.extend(_metric_rows(meta_test, y_test, pred_corr, "trainonly_ridge", prefix, int(seed), weight_path))
            manifest.append(
                {
                    "prefix": prefix,
                    "seed": int(seed),
                    "weight_path": weight_path.name,
                    "ridge_alpha": float(RIDGE_ALPHA),
                    "correction_clip_pct": float(CLIP_FRACTION * 100.0),
                    "train_profiles": ";".join(cfg.train_profiles),
                    "test_profiles": ";".join(cfg.test_profiles),
                    "train_windows": int(len(y_train)),
                    "test_windows": int(len(y_test)),
                    "uses_test_labels_for_fit": False,
                    "uses_soc_as_input_feature": False,
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del shared
        if device.type == "cuda":
            torch.cuda.empty_cache()

    by_temp = pd.DataFrame(all_rows)
    by_temp.to_csv(BY_TEMP_OUT, index=False)
    m = by_temp[by_temp["temperature_C"].astype(str).eq("0.0")].copy()
    agg = (
        m.groupby(["holdout", "mode"], sort=True)
        .agg(
            mean_0c_mae=("mae_pct", "mean"),
            max_seed_0c_mae=("mae_pct", "max"),
            min_seed_0c_mae=("mae_pct", "min"),
            mean_0c_bias=("bias_pct", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
        .sort_values(["mode", "max_seed_0c_mae"])
    )
    agg.to_csv(AGG_OUT, index=False)
    pd.DataFrame(manifest).to_csv(MANIFEST_OUT, index=False)
    report = [
        "# SOC80 w50 train-only ridge calibration v1",
        "",
        "This diagnostic loads the existing w1p4_lam1 final weights, fits a fixed ridge residual calibrator on train windows only, and evaluates on the held-out profile.",
        "Inputs to the calibrator are model prediction plus scaled V/I/T-derived window statistics. SOC is not used as an input feature.",
        f"Fixed settings: ridge_alpha={RIDGE_ALPHA}, correction_clip={CLIP_FRACTION * 100.0:.3f}% SOC.",
        "",
        "## 0C aggregate",
        agg.to_markdown(index=False),
        "",
        "## Outputs",
        f"- `{BY_TEMP_OUT.name}`",
        f"- `{AGG_OUT.name}`",
        f"- `{MANIFEST_OUT.name}`",
    ]
    REPORT_OUT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {BY_TEMP_OUT}")
    print(f"wrote {AGG_OUT}")
    print(f"wrote {MANIFEST_OUT}")
    print(f"wrote {REPORT_OUT}")


if __name__ == "__main__":
    main()
