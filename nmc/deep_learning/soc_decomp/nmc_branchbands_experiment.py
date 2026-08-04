from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import least_squares

from .config import make_cfg
from .deep_no_leak_experiment import (
    AugmentedSequenceWindowDataset,
    AugmentedWindowDataset,
    DeepNoLeakTCN,
    add_derived_features,
    augmented_input_dim,
    feature_columns,
    make_eval_loader,
    predict,
    temp_balanced_rex_loss,
)
from .extrapolation_robustness import temperature_balanced_loader
from .runtime import configure_torch_runtime, device
from .training import attach_prediction_features, build_prediction_feature_lookup, make_scaled_frames_for_ablation
from .variance_control import _overall_metrics, variance_by_temperature


FORBIDDEN_INPUT_PATTERNS = (
    "SOC",
    "soc",
    "Qnet",
    "Qdis",
    "Qchg",
    "Capacity",
    "capacity",
    "progress",
    "t_global",
    "Test_Time",
    "Step_Time",
    "Data_Point",
    "cumulative",
    "Ah",
)


@dataclass
class NMCBranchBandsConfig:
    base_dir: Path = Path(".")
    raw_root: Path = Path("nmc_samsung_inr_18650_2ah_raw/NMC SAMSUNG INR 18650 2Ah")
    output_prefix: str = "nmc_branchbands_w150_s3_h128_l6_alltemps_trainProfiles_to_FUDS_seed0"
    seed: int = 0
    train_profiles: tuple[str, ...] = ("VALIDATION", "DST", "US06")
    test_profiles: tuple[str, ...] = ("FUDS",)
    window_len: int = 150
    stride: int = 3
    epochs: int = 300
    batch_size: int = 1024
    lr: float = 8e-4
    weight_decay: float = 1e-4
    hidden_size: int = 128
    layers: int = 6
    kernel_size: int = 5
    norm_kind: str = "channel"
    dropout: float = 0.04
    lambda_rex: float = 2.0
    rex_group: str = "temperature_drive"
    loss_kind: str = "huber"
    huber_beta: float = 0.02
    lambda_smooth: float = 0.0
    endpoint_loss_weight: float = 0.0
    lambda_worst: float = 0.0
    window_feature_mode: str = "delta_start_time"
    num_workers: int = 4
    prefetch_factor: int = 4
    print_every: int = 10
    low_current_threshold_A: float = 0.05
    v_corr_tau_s: float = 120.0
    v_pol_mid_tau_s: float = 60.0
    v_pol_slow_tau_s: float = 600.0
    v_hys_tau_s: float = 1200.0
    v_corr_variant: str = "orig_ohm_ema120"
    v_corr_tail_gate_v: float = 3.30
    v_corr_tail_gate_s: float = 0.15
    v_corr_tail_tau_s: float = 40.0
    v_corr_asym_down_tau_s: float = 20.0
    v_corr_asym_up_tau_s: float = 160.0
    v_corr_lv_gate_v: float = 3.35
    v_corr_lv_gate_s: float = 0.15
    v_corr_ocv_low_current_a: float = 0.05
    v_corr_ocv_bin_width_soc: float = 0.01
    v_corr_ocv_min_bin_points: int = 5
    v_corr_ocv_tau_fast_s: float = 20.0
    v_corr_ocv_tau_slow_s: float = 200.0
    v_corr_ocv_ridge: float = 1e-4
    v_corr_ocv_coef_limit: float = 0.30
    v_corr_ocv_intercept_limit_v: float = 0.20
    dynr_ridge: float = 1e-3
    dynr_log_bound: float = 0.6931471805599453
    lfpstyle_fit_stride: int = 5
    lfpstyle_fit_max_nfev: int = 250
    lfpstyle_v_floor_raw: float = 2.45
    lfpstyle_vfloor_beta: float = 20.0
    lfpstyle_r0_max_ohm: float = 0.22
    lfpstyle_r0_min_ohm: float = 0.0
    lfpstyle_g_corr_scale: float = 1.35
    lfpstyle_pol_fast_limit_v: float = 0.16
    lfpstyle_pol_mid_limit_v: float = 0.12
    lfpstyle_pol_slow_limit_v: float = 0.10
    lfpstyle_pol_limit_v: float = 0.25
    lfpstyle_hys_limit_v: float = 0.20


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def parse_temp_label(value) -> float:
    text = str(value).strip()
    m = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not m:
        raise ValueError(f"Cannot parse temperature from {value!r}")
    return float(m.group(1))


def parse_profile(path: Path, df: pd.DataFrame | None = None) -> str:
    if df is not None and "Profile" in df.columns and len(df):
        return str(df["Profile"].iloc[0])
    parts = path.stem.split("_")
    return parts[-1]


def parse_temperature(path: Path, df: pd.DataFrame | None = None) -> float:
    if df is not None and "TempLabel" in df.columns and len(df):
        return parse_temp_label(df["TempLabel"].iloc[0])
    for part in path.parts:
        if part.endswith("C"):
            return parse_temp_label(part)
    parts = path.stem.split("_")
    if len(parts) >= 2:
        return parse_temp_label(parts[1])
    raise ValueError(f"Cannot parse temperature from path {path}")


def find_csv_files(raw_root: Path) -> list[Path]:
    files = sorted(Path(raw_root).rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No NMC CSV files found under {raw_root}")
    return files


def write_start_audit(files: list[Path], out_path: Path) -> pd.DataFrame:
    rows = []
    for p in files:
        df = pd.read_csv(p)
        first = df.iloc[0]
        last = df.iloc[-1]
        soc = pd.to_numeric(df.get("SOC_CC"), errors="coerce")
        first_soc = float(first.get("SOC_CC", np.nan))
        first_soc_pct = float(first.get("SOC_CC(%)", np.nan))
        rows.append(
            {
                "file_name": p.name,
                "temperature_C": parse_temperature(p, df),
                "profile": parse_profile(p, df),
                "rows": int(len(df)),
                "first_data_point": int(first.get("Data_Point", -1)),
                "first_test_time_s": float(first.get("Test_Time(s)", np.nan)),
                "first_step_time_s": float(first.get("Step_Time(s)", np.nan)),
                "first_step_index": int(first.get("Step_Index", -1)),
                "drive_step_index": int(first.get("DriveStepIndex", -1)),
                "first_voltage_v": float(first.get("Voltage(V)", np.nan)),
                "first_current_a": float(first.get("Current(A)", np.nan)),
                "soc0_used": float(first.get("SOC0_used", np.nan)),
                "soc0_vinit_v": float(first.get("SOC0_Vinit(V)", np.nan)),
                "soc0_rest_step": int(first.get("SOC0_restStep", -1)),
                "qnet_denom_Ah": float(first.get("Qnet_denom(Ah)", np.nan)),
                "first_soc_cc": first_soc,
                "first_soc_pct": first_soc_pct,
                "last_soc_cc": float(last.get("SOC_CC", np.nan)),
                "min_soc_cc": float(np.nanmin(soc)),
                "max_soc_cc": float(np.nanmax(soc)),
                "starts_at_80pct": bool(abs(first_soc - 0.8) < 1e-5 or abs(first_soc_pct - 80.0) < 1e-3),
                "soc_range_is_0_to_80pct": bool(np.nanmin(soc) >= -1e-6 and np.nanmax(soc) <= 0.800001),
                "already_trimmed_to_drive_step": bool(
                    int(first.get("Step_Index", -1)) == int(first.get("DriveStepIndex", -2))
                ),
            }
        )
    out = pd.DataFrame(rows).sort_values(["temperature_C", "profile"]).reset_index(drop=True)
    out.to_csv(out_path, index=False)
    return out


def causal_ema(values: np.ndarray, times_s: np.ndarray, tau_s: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    t = np.asarray(times_s, dtype=np.float64)
    if len(x) == 0:
        return x.astype(np.float32)
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    if len(x) == 1:
        return y.astype(np.float32)
    dt_default = float(np.nanmedian(np.diff(t)[np.isfinite(np.diff(t)) & (np.diff(t) > 0)]))
    if not np.isfinite(dt_default) or dt_default <= 0:
        dt_default = 1.0
    for i in range(1, len(x)):
        dt = t[i] - t[i - 1]
        if not np.isfinite(dt) or dt <= 0:
            dt = dt_default
        alpha = float(np.exp(-dt / max(float(tau_s), 1e-6)))
        alpha = min(max(alpha, 0.0), 0.999999)
        y[i] = alpha * y[i - 1] + (1.0 - alpha) * x[i]
    return y.astype(np.float32)


def causal_asymmetric_ema(
    values: np.ndarray,
    times_s: np.ndarray,
    tau_down_s: float,
    tau_up_s: float,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    t = np.asarray(times_s, dtype=np.float64)
    if len(x) == 0:
        return x.astype(np.float32)
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    if len(x) == 1:
        return y.astype(np.float32)
    dt_default = float(np.nanmedian(np.diff(t)[np.isfinite(np.diff(t)) & (np.diff(t) > 0)]))
    if not np.isfinite(dt_default) or dt_default <= 0:
        dt_default = 1.0
    for i in range(1, len(x)):
        dt = t[i] - t[i - 1]
        if not np.isfinite(dt) or dt <= 0:
            dt = dt_default
        tau = tau_down_s if x[i] < y[i - 1] else tau_up_s
        alpha = float(np.exp(-dt / max(float(tau), 1e-6)))
        alpha = min(max(alpha, 0.0), 0.999999)
        y[i] = alpha * y[i - 1] + (1.0 - alpha) * x[i]
    return y.astype(np.float32)


def estimate_r0_by_temperature(files: list[Path], train_profiles: tuple[str, ...], quantile: float = 0.5) -> pd.DataFrame:
    q = min(max(float(quantile), 0.0), 1.0)
    rows = []
    all_ratios: list[float] = []
    for p in files:
        df_head = pd.read_csv(p, nrows=2)
        profile = parse_profile(p, df_head)
        if profile not in train_profiles:
            continue
        temp = parse_temperature(p, df_head)
        df = pd.read_csv(p, usecols=["Current(A)", "Voltage(V)"])
        i_raw = df["Current(A)"].to_numpy(np.float64)
        v_raw = df["Voltage(V)"].to_numpy(np.float64)
        d_i = np.diff(i_raw, prepend=i_raw[0])
        d_v = np.diff(v_raw, prepend=v_raw[0])
        mask = np.isfinite(d_i) & np.isfinite(d_v) & (np.abs(d_i) > 0.05) & (np.abs(d_v) > 1e-5)
        ratio = d_v[mask] / d_i[mask]
        ratio = ratio[np.isfinite(ratio) & (ratio > 0.001) & (ratio < 0.5)]
        for r in ratio:
            rows.append({"temperature_C": float(temp), "profile": profile, "file_name": p.name, "r0_event_ohm": float(r)})
        all_ratios.extend([float(r) for r in ratio])
    event_df = pd.DataFrame(rows)
    if event_df.empty:
        raise RuntimeError("Could not estimate R0 from train profiles.")
    fallback = float(np.quantile(all_ratios, q))
    summary = (
        event_df.groupby("temperature_C")["r0_event_ohm"]
        .agg(
            r0_ohm=lambda s: float(np.quantile(s, q)),
            n_events="count",
            r0_p20_ohm=lambda s: float(np.percentile(s, 20)),
            r0_p80_ohm=lambda s: float(np.percentile(s, 80)),
        )
        .reset_index()
    )
    present = set(summary["temperature_C"].astype(float))
    all_temps = sorted({parse_temperature(p, pd.read_csv(p, nrows=2)) for p in files})
    for temp in all_temps:
        if float(temp) not in present:
            summary = pd.concat(
                [
                    summary,
                    pd.DataFrame(
                        [{"temperature_C": float(temp), "r0_ohm": fallback, "n_events": 0, "r0_p20_ohm": np.nan, "r0_p80_ohm": np.nan}]
                    ),
                ],
                ignore_index=True,
            )
    return summary.sort_values("temperature_C").reset_index(drop=True)


def estimate_r0_by_temperature_profile(files: list[Path], quantile: float = 0.5) -> pd.DataFrame:
    q = min(max(float(quantile), 0.0), 1.0)
    rows = []
    for p in files:
        head = pd.read_csv(p, nrows=2)
        profile = parse_profile(p, head)
        temp = parse_temperature(p, head)
        df = pd.read_csv(p, usecols=["Current(A)", "Voltage(V)"])
        i_raw = df["Current(A)"].to_numpy(np.float64)
        v_raw = df["Voltage(V)"].to_numpy(np.float64)
        d_i = np.diff(i_raw, prepend=i_raw[0])
        d_v = np.diff(v_raw, prepend=v_raw[0])
        mask = np.isfinite(d_i) & np.isfinite(d_v) & (np.abs(d_i) > 0.05) & (np.abs(d_v) > 1e-5)
        ratio = d_v[mask] / d_i[mask]
        ratio = ratio[np.isfinite(ratio) & (ratio > 0.001) & (ratio < 0.5)]
        if len(ratio) == 0:
            continue
        rows.append(
            {
                "temperature_C": float(temp),
                "profile": profile,
                "r0_ohm": float(np.quantile(ratio, q)),
                "n_events": int(len(ratio)),
                "r0_quantile": float(q),
                "r0_p20_ohm": float(np.percentile(ratio, 20)),
                "r0_p80_ohm": float(np.percentile(ratio, 80)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("Could not estimate profile-local R0 from observed V/I steps.")
    return out.sort_values(["temperature_C", "profile"]).reset_index(drop=True)


def _ocv_calibrated_variant(v_corr_variant: str) -> bool:
    return str(v_corr_variant) in {"ocvcal_dynfit_v1", "train_ocvcal_dynfit_v1"}


def _dynamic_r_variant(v_corr_variant: str) -> bool:
    return str(v_corr_variant) in {"dynr_vit_event_ema10", "dynr_vit_event"}


def _rtvar_lowv_variant(v_corr_variant: str) -> bool:
    return str(v_corr_variant) in {"rtvar_lowv_ema120", "endzero_rtvar_lowv_ema120"}


def _lfp_style_variant(v_corr_variant: str) -> bool:
    return str(v_corr_variant) in {"lfpstyle_ocvfit_v1", "lfpstyle_ocvfit"}


def _nmc_tailored_minimax_variant(v_corr_variant: str) -> bool:
    return str(v_corr_variant) in {
        "nmc_tailored_minimax_v1",
        "nmc_tailored_minimax",
        "nmc_tailored_minimax_aligned_v1",
        "nmc_tailored_minimax_aligned",
    }


def _nmc_tailored_aligned_variant(v_corr_variant: str) -> bool:
    return str(v_corr_variant) in {"nmc_tailored_minimax_aligned_v1", "nmc_tailored_minimax_aligned"}


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _logit_np(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def _soft_floor_np(v: np.ndarray, floor: float, beta: float) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    beta = max(float(beta), 1e-6)
    z = (arr - float(floor)) * beta
    out = np.empty_like(arr, dtype=np.float64)
    hi = z > 40.0
    lo = z < -40.0
    mid = ~(hi | lo)
    out[hi] = arr[hi]
    out[lo] = float(floor) + np.exp(z[lo]) / beta
    out[mid] = float(floor) + np.log1p(np.exp(z[mid])) / beta
    return out


def _soc01_from_frame(df: pd.DataFrame) -> np.ndarray:
    soc = pd.to_numeric(df["SOC_CC"], errors="coerce").to_numpy(np.float64)
    if np.nanmax(soc) > 1.5:
        soc = soc / 100.0
    return np.clip(soc, 0.0, 1.0)


def _interp_ocv_from_calibration(calibration: dict, temp: float, soc: np.ndarray) -> np.ndarray:
    curves = calibration.get("curves", {}) if calibration else {}
    if not curves:
        return np.full_like(np.asarray(soc, dtype=np.float64), np.nan, dtype=np.float64)
    key = float(temp)
    if key not in curves:
        key = min(curves, key=lambda k: abs(float(k) - float(temp)))
    x, y = curves[key]
    s = np.asarray(soc, dtype=np.float64)
    return np.interp(s, np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), left=float(y[0]), right=float(y[-1]))


def _fit_ridge_dynamic_coefficients(x: np.ndarray, y: np.ndarray, cfg: NMCBranchBandsConfig) -> np.ndarray:
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x_fit = x[mask]
    y_fit = y[mask]
    if len(y_fit) < 20:
        return np.zeros(3, dtype=np.float64)
    keep = np.ones(len(y_fit), dtype=bool)
    beta = np.zeros(3, dtype=np.float64)
    ridge = max(float(getattr(cfg, "v_corr_ocv_ridge", 1e-4)), 0.0)
    for _ in range(3):
        xf = x_fit[keep]
        yf = y_fit[keep]
        reg = np.eye(xf.shape[1], dtype=np.float64) * ridge
        reg[0, 0] = ridge * 0.01
        try:
            beta = np.linalg.solve(xf.T @ xf + reg, xf.T @ yf)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(xf, yf, rcond=None)[0]
        resid = np.abs(y_fit - x_fit @ beta)
        q = float(np.nanquantile(resid, 0.90))
        keep = np.isfinite(resid) & (resid <= max(q, 1e-9))
        if int(keep.sum()) < 20:
            break
    beta[0] = float(np.clip(beta[0], -float(cfg.v_corr_ocv_intercept_limit_v), float(cfg.v_corr_ocv_intercept_limit_v)))
    limit = float(getattr(cfg, "v_corr_ocv_coef_limit", 0.30))
    beta[1:] = np.clip(beta[1:], -limit, limit)
    return beta.astype(np.float64)


def _dynamic_r_feature_matrix(
    v_raw: np.ndarray,
    i_raw: np.ndarray,
    d_i: np.ndarray,
    times: np.ndarray,
    temp: float,
) -> np.ndarray:
    abs_i = np.abs(i_raw).astype(np.float64)
    abs_di = np.abs(d_i).astype(np.float64)
    abs_i_ema60 = causal_ema(abs_i, times, 60.0).astype(np.float64)
    abs_di_ema20 = causal_ema(abs_di, times, 20.0).astype(np.float64)
    v_ema120 = causal_ema(v_raw, times, 120.0).astype(np.float64)
    v_dev = np.asarray(v_raw, dtype=np.float64) - v_ema120
    t_raw = np.full_like(abs_i, float(temp), dtype=np.float64)
    return np.column_stack(
        [
            np.asarray(v_raw, dtype=np.float64),
            np.asarray(v_raw, dtype=np.float64) ** 2,
            np.asarray(i_raw, dtype=np.float64),
            abs_i,
            abs_i_ema60,
            abs_di,
            abs_di_ema20,
            v_dev,
            t_raw,
            t_raw * t_raw,
        ]
    ).astype(np.float64)


def _fit_dynamic_r_coefficients(x: np.ndarray, y: np.ndarray, cfg: NMCBranchBandsConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x_fit = x[mask]
    y_fit = y[mask]
    if len(y_fit) < 50:
        mu = np.zeros(x.shape[1], dtype=np.float64)
        sig = np.ones(x.shape[1], dtype=np.float64)
        return np.zeros(x.shape[1] + 1, dtype=np.float64), mu, sig
    mu = np.nanmean(x_fit, axis=0)
    sig = np.nanstd(x_fit, axis=0)
    sig = np.where(sig > 1e-9, sig, 1.0)
    xs = (x_fit - mu) / sig
    X = np.column_stack([np.ones(len(xs), dtype=np.float64), xs])
    bound = float(getattr(cfg, "dynr_log_bound", np.log(2.0)))
    y_fit = np.clip(y_fit, -bound, bound)
    keep = np.ones(len(y_fit), dtype=bool)
    beta = np.zeros(X.shape[1], dtype=np.float64)
    ridge = max(float(getattr(cfg, "dynr_ridge", 1e-3)), 0.0)
    for _ in range(3):
        xf = X[keep]
        yf = y_fit[keep]
        reg = np.eye(X.shape[1], dtype=np.float64) * ridge
        reg[0, 0] = ridge * 0.01
        try:
            beta = np.linalg.solve(xf.T @ xf + reg, xf.T @ yf)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(xf, yf, rcond=None)[0]
        resid = np.abs(y_fit - X @ beta)
        q = float(np.nanquantile(resid, 0.90))
        keep = np.isfinite(resid) & (resid <= max(q, 1e-9))
        if int(keep.sum()) < 50:
            break
    return beta.astype(np.float64), mu.astype(np.float64), sig.astype(np.float64)


def estimate_dynamic_r_calibration(
    files: list[Path],
    train_profiles: tuple[str, ...],
    r0_df: pd.DataFrame,
    cfg: NMCBranchBandsConfig,
) -> dict:
    r0_lookup = {float(r["temperature_C"]): float(r["r0_ohm"]) for _, r in r0_df.iterrows()}
    feats: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    rows = []
    for p in files:
        head = pd.read_csv(p, nrows=2)
        profile = parse_profile(p, head)
        if profile not in train_profiles:
            continue
        temp = float(parse_temperature(p, head))
        r0 = float(r0_lookup[temp])
        df = pd.read_csv(p)
        time_col = "Step_Time(s)" if "Step_Time(s)" in df.columns else "Test_Time(s)"
        times = pd.to_numeric(df[time_col], errors="coerce").to_numpy(np.float64)
        v_raw = pd.to_numeric(df["Voltage(V)"], errors="coerce").to_numpy(np.float64)
        i_raw = pd.to_numeric(df["Current(A)"], errors="coerce").to_numpy(np.float64)
        d_i = np.diff(i_raw, prepend=i_raw[0])
        d_v = np.diff(v_raw, prepend=v_raw[0])
        mask = np.isfinite(d_i) & np.isfinite(d_v) & (np.abs(d_i) > 0.05) & (np.abs(d_v) > 1e-5)
        ratio = np.full_like(d_v, np.nan, dtype=np.float64)
        np.divide(d_v, d_i, out=ratio, where=np.abs(d_i) > 1e-12)
        mask &= np.isfinite(ratio) & (ratio > 0.001) & (ratio < 0.5)
        if not np.any(mask):
            continue
        x = _dynamic_r_feature_matrix(v_raw, i_raw, d_i, times, temp)
        y = np.log(np.clip(ratio[mask] / max(r0, 1e-9), 1e-6, 1e6))
        feats.append(x[mask])
        targets.append(y.astype(np.float64))
        rows.append({"temperature_C": temp, "profile": profile, "n_event_points": int(mask.sum()), "r0_base_ohm": r0})
    if not feats:
        raise RuntimeError("Could not estimate dynamic R calibration from train profiles.")
    x_all = np.vstack(feats)
    y_all = np.concatenate(targets)
    beta, mu, sig = _fit_dynamic_r_coefficients(x_all, y_all, cfg)
    bound = float(getattr(cfg, "dynr_log_bound", np.log(2.0)))
    pred = np.clip(np.column_stack([np.ones(len(x_all)), (x_all - mu) / sig]) @ beta, -bound, bound)
    params_df = pd.DataFrame(
        [
            {
                "variant": "dynr_vit_event_ema10",
                "n_event_points": int(len(y_all)),
                "dynr_ridge": float(getattr(cfg, "dynr_ridge", 1e-3)),
                "dynr_log_bound": bound,
                "train_logr_mae_before": float(np.nanmean(np.abs(y_all))),
                "train_logr_mae_after": float(np.nanmean(np.abs(y_all - pred))),
                "train_ratio_median": float(np.nanmedian(np.exp(y_all))),
                "pred_ratio_median": float(np.nanmedian(np.exp(pred))),
                "pred_ratio_p05": float(np.nanquantile(np.exp(pred), 0.05)),
                "pred_ratio_p95": float(np.nanquantile(np.exp(pred), 0.95)),
            }
        ]
    )
    coef_rows = [{"term": "intercept", "beta": float(beta[0]), "mu": np.nan, "sigma": np.nan}]
    names = ["V_raw", "V_raw_sq", "I_raw", "absI", "absI_ema60", "absdI", "absdI_ema20", "V_dev_ema120", "T", "T_sq"]
    for name, b, m, s in zip(names, beta[1:], mu, sig):
        coef_rows.append({"term": name, "beta": float(b), "mu": float(m), "sigma": float(s)})
    return {
        "beta": beta,
        "mu": mu,
        "sig": sig,
        "r0_lookup": r0_lookup,
        "params_df": params_df,
        "coef_df": pd.DataFrame(coef_rows),
        "event_df": pd.DataFrame(rows).sort_values(["temperature_C", "profile"]).reset_index(drop=True),
    }


def apply_dynamic_r(
    v_raw: np.ndarray,
    i_raw: np.ndarray,
    d_i: np.ndarray,
    times: np.ndarray,
    temp: float,
    r0: float,
    calibration: dict,
    cfg: NMCBranchBandsConfig,
) -> np.ndarray:
    x = _dynamic_r_feature_matrix(v_raw, i_raw, d_i, times, temp)
    beta = np.asarray(calibration.get("beta"), dtype=np.float64)
    mu = np.asarray(calibration.get("mu"), dtype=np.float64)
    sig = np.asarray(calibration.get("sig"), dtype=np.float64)
    if beta.size != x.shape[1] + 1 or mu.size != x.shape[1] or sig.size != x.shape[1]:
        return np.full(len(v_raw), float(r0), dtype=np.float32)
    z = np.column_stack([np.ones(len(x), dtype=np.float64), (x - mu) / np.where(sig > 1e-9, sig, 1.0)])
    bound = float(getattr(cfg, "dynr_log_bound", np.log(2.0)))
    eta = np.clip(z @ beta, -bound, bound)
    r_eff = float(r0) * np.exp(eta)
    return r_eff.astype(np.float32)


def _rtvar_local_variance_r(
    v_raw: np.ndarray,
    i_raw: np.ndarray,
    times: np.ndarray,
    r0: float,
    tau_s: float,
    rmax: float,
) -> np.ndarray:
    v = np.asarray(v_raw, dtype=np.float64)
    i = np.asarray(i_raw, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    ev = causal_ema(v, t, float(tau_s))
    ei = causal_ema(i, t, float(tau_s))
    evi = causal_ema(v * i, t, float(tau_s))
    ei2 = causal_ema(i * i, t, float(tau_s))
    cov = evi - ev * ei
    var_i = ei2 - ei * ei
    r = np.full(len(v), float(r0), dtype=np.float64)
    np.divide(cov, var_i, out=r, where=np.isfinite(var_i) & (var_i > 1e-5))
    r = np.clip(r, 0.0, float(rmax))
    return causal_ema(r, t, max(float(tau_s) * 0.5, 10.0)).astype(np.float64)


def _rtvar_hf_residual(values: np.ndarray, times: np.ndarray, tau_s: float = 600.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr - causal_ema(arr, np.asarray(times, dtype=np.float64), float(tau_s))


def _rtvar_base_record(path: Path, r0: float, cfg: NMCBranchBandsConfig) -> dict:
    df = pd.read_csv(path)
    temp = float(parse_temperature(path, df))
    profile = parse_profile(path, df)
    time_col = "Step_Time(s)" if "Step_Time(s)" in df.columns else "Test_Time(s)"
    times = pd.to_numeric(df[time_col], errors="coerce").to_numpy(np.float64)
    v_raw = pd.to_numeric(df["Voltage(V)"], errors="coerce").to_numpy(np.float64)
    i_raw = pd.to_numeric(df["Current(A)"], errors="coerce").to_numpy(np.float64)
    soc = _soc01_from_frame(df)
    valid = np.isfinite(times) & np.isfinite(v_raw) & np.isfinite(i_raw) & np.isfinite(soc)
    times = times[valid]
    v_raw = v_raw[valid]
    i_raw = i_raw[valid]
    soc = soc[valid]
    base_ohm_free = v_raw - i_raw * float(r0)
    base_vcorr = causal_ema(base_ohm_free, times, float(getattr(cfg, "v_corr_tau_s", 120.0)))
    return {
        "path": str(path),
        "temperature_C": temp,
        "profile": profile,
        "time_s": times,
        "V_raw": v_raw,
        "I_raw": i_raw,
        "SOC_frac": soc,
        "V_corr_base": base_vcorr.astype(np.float64),
    }


def _rtvar_score_candidate(record: dict, v_corr: np.ndarray, r_eff: np.ndarray) -> float:
    soc = np.asarray(record["SOC_frac"], dtype=np.float64)
    times = np.asarray(record["time_s"], dtype=np.float64)
    base = np.asarray(record["V_corr_base"], dtype=np.float64)
    low = soc <= 0.10
    mid = (soc > 0.20) & (soc <= 0.80)
    if not np.any(low):
        return float("inf")
    low_score = float(np.nanstd(_rtvar_hf_residual(v_corr[low], times[low])) * 1000.0)
    penalty = 0.0
    if np.any(mid):
        base_mid = float(np.nanstd(_rtvar_hf_residual(base[mid], times[mid])) * 1000.0)
        dyn_mid = float(np.nanstd(_rtvar_hf_residual(v_corr[mid], times[mid])) * 1000.0)
        penalty += max(0.0, dyn_mid - base_mid) * 0.5
    penalty += max(0.0, float(np.nanpercentile(r_eff, 95)) - 0.25) * 50.0
    return low_score + penalty


def estimate_rtvar_lowv_calibration(
    files: list[Path],
    train_profiles: tuple[str, ...],
    r0_df: pd.DataFrame,
    cfg: NMCBranchBandsConfig,
) -> dict:
    r0_lookup = {float(r["temperature_C"]): float(r["r0_ohm"]) for _, r in r0_df.iterrows()}
    records_by_temp: dict[float, list[dict]] = {}
    for p in files:
        head = pd.read_csv(p, nrows=2)
        profile = parse_profile(p, head)
        if profile not in train_profiles:
            continue
        temp = float(parse_temperature(p, head))
        if temp not in r0_lookup:
            continue
        records_by_temp.setdefault(temp, []).append(_rtvar_base_record(p, r0_lookup[temp], cfg))

    if not records_by_temp:
        raise RuntimeError("Could not estimate RT-variance V_corr calibration from train profiles.")

    tau_candidates = (30.0, 60.0, 120.0)
    rmax_candidates = (0.18, 0.25)
    gate_s_candidates = (0.06, 0.10)
    alpha_candidates = (0.50, 1.00)
    params: dict[float, dict[str, float]] = {}
    param_rows = []
    selection_rows = []
    for temp, records in sorted(records_by_temp.items()):
        r0 = float(r0_lookup[float(temp)])
        all_base = np.concatenate([np.asarray(rec["V_corr_base"], dtype=np.float64) for rec in records])
        low_base_parts = [
            np.asarray(rec["V_corr_base"], dtype=np.float64)[np.asarray(rec["SOC_frac"], dtype=np.float64) <= 0.10]
            for rec in records
        ]
        low_base_parts = [x for x in low_base_parts if len(x)]
        low_base = np.concatenate(low_base_parts) if low_base_parts else all_base
        gate_candidates = sorted(set(float(x) for x in np.nanpercentile(low_base, [70, 90]) if np.isfinite(x)))
        if not gate_candidates:
            gate_candidates = [float(np.nanmedian(all_base))]

        r_cache: dict[tuple[int, float, float], np.ndarray] = {}
        for idx, rec in enumerate(records):
            for tau_r in tau_candidates:
                for rmax in rmax_candidates:
                    r_cache[(idx, tau_r, rmax)] = _rtvar_local_variance_r(
                        rec["V_raw"],
                        rec["I_raw"],
                        rec["time_s"],
                        r0,
                        tau_s=tau_r,
                        rmax=rmax,
                    )

        for tau_r in tau_candidates:
            for rmax in rmax_candidates:
                for gate_v in gate_candidates:
                    for gate_s in gate_s_candidates:
                        for alpha in alpha_candidates:
                            scores = []
                            r_p95 = []
                            for idx, rec in enumerate(records):
                                times = np.asarray(rec["time_s"], dtype=np.float64)
                                base = np.asarray(rec["V_corr_base"], dtype=np.float64)
                                r_local = r_cache[(idx, tau_r, rmax)]
                                gate = _sigmoid_np((float(gate_v) - base) / max(float(gate_s), 1e-6))
                                gate_state = causal_ema(gate, times, 60.0)
                                r_eff = float(r0) + float(alpha) * gate_state * (r_local - float(r0))
                                r_eff = np.clip(r_eff, 0.0, float(rmax))
                                v_corr = causal_ema(
                                    np.asarray(rec["V_raw"], dtype=np.float64) - np.asarray(rec["I_raw"], dtype=np.float64) * r_eff,
                                    times,
                                    float(getattr(cfg, "v_corr_tau_s", 120.0)),
                                )
                                scores.append(_rtvar_score_candidate(rec, v_corr, r_eff))
                                r_p95.append(float(np.nanpercentile(r_eff, 95)))
                            selection_rows.append(
                                {
                                    "temperature_C": float(temp),
                                    "tau_r": float(tau_r),
                                    "rmax": float(rmax),
                                    "gate_v": float(gate_v),
                                    "gate_s": float(gate_s),
                                    "alpha": float(alpha),
                                    "train_score_mean": float(np.nanmean(scores)),
                                    "train_score_max": float(np.nanmax(scores)),
                                    "r_eff_p95_mean_ohm": float(np.nanmean(r_p95)),
                                }
                            )

        temp_table = pd.DataFrame([r for r in selection_rows if float(r["temperature_C"]) == float(temp)])
        temp_table = temp_table.sort_values(["train_score_mean", "train_score_max", "rmax", "alpha"]).reset_index(drop=True)
        best = temp_table.iloc[0].to_dict()
        p = {
            "temperature_C": float(temp),
            "r0_ohm": float(r0),
            "tau_r": float(best["tau_r"]),
            "rmax": float(best["rmax"]),
            "gate_v": float(best["gate_v"]),
            "gate_s": float(best["gate_s"]),
            "alpha": float(best["alpha"]),
            "train_score_mean": float(best["train_score_mean"]),
            "train_score_max": float(best["train_score_max"]),
            "tau_corr": float(getattr(cfg, "v_corr_tau_s", 120.0)),
            "objective": "train low-SOC corrected-voltage high-frequency variance; no OCV target",
        }
        params[float(temp)] = p
        param_rows.append(p)

    return {
        "params": params,
        "params_df": pd.DataFrame(param_rows).sort_values(["temperature_C"]).reset_index(drop=True),
        "selection_df": pd.DataFrame(selection_rows).sort_values(["temperature_C", "train_score_mean", "train_score_max"]).reset_index(drop=True),
    }


def apply_rtvar_lowv_vcorr(
    v_raw: np.ndarray,
    i_raw: np.ndarray,
    times: np.ndarray,
    temp: float,
    r0: float,
    calibration: dict,
    cfg: NMCBranchBandsConfig,
) -> tuple[np.ndarray, np.ndarray]:
    params = calibration.get("params", {}) if calibration else {}
    if not params:
        raise RuntimeError("RT-variance V_corr requested, but calibration has no params.")
    key = float(temp)
    if key not in params:
        key = min(params, key=lambda k: abs(float(k) - float(temp)))
    p = params[key]
    tau_corr = float(p.get("tau_corr", getattr(cfg, "v_corr_tau_s", 120.0)))
    r0_p = float(p.get("r0_ohm", r0))
    r_local = _rtvar_local_variance_r(
        v_raw,
        i_raw,
        times,
        r0_p,
        tau_s=float(p["tau_r"]),
        rmax=float(p["rmax"]),
    )
    base = causal_ema(np.asarray(v_raw, dtype=np.float64) - np.asarray(i_raw, dtype=np.float64) * r0_p, times, tau_corr)
    gate = _sigmoid_np((float(p["gate_v"]) - base) / max(float(p["gate_s"]), 1e-6))
    gate_state = causal_ema(gate, np.asarray(times, dtype=np.float64), 60.0)
    r_eff = r0_p + float(p["alpha"]) * gate_state * (r_local - r0_p)
    r_eff = np.clip(r_eff, 0.0, float(p["rmax"]))
    v_corr = causal_ema(np.asarray(v_raw, dtype=np.float64) - np.asarray(i_raw, dtype=np.float64) * r_eff, times, tau_corr)
    return v_corr.astype(np.float32), r_eff.astype(np.float32)


def estimate_vcorr_ocv_calibration(
    files: list[Path],
    train_profiles: tuple[str, ...],
    r0_df: pd.DataFrame,
    cfg: NMCBranchBandsConfig,
) -> dict:
    r0_lookup = {float(r["temperature_C"]): float(r["r0_ohm"]) for _, r in r0_df.iterrows()}
    low_rows: dict[float, list[tuple[np.ndarray, np.ndarray, str]]] = {}
    train_frames: dict[float, list[dict[str, np.ndarray]]] = {}
    low_current = float(getattr(cfg, "v_corr_ocv_low_current_a", 0.05))

    for p in files:
        head = pd.read_csv(p, nrows=2)
        profile = parse_profile(p, head)
        if profile not in train_profiles:
            continue
        temp = float(parse_temperature(p, head))
        r0 = float(r0_lookup[temp])
        df = pd.read_csv(p)
        time_col = "Step_Time(s)" if "Step_Time(s)" in df.columns else "Test_Time(s)"
        times = pd.to_numeric(df[time_col], errors="coerce").to_numpy(np.float64)
        v_raw = pd.to_numeric(df["Voltage(V)"], errors="coerce").to_numpy(np.float64)
        i_raw = pd.to_numeric(df["Current(A)"], errors="coerce").to_numpy(np.float64)
        soc = _soc01_from_frame(df)
        v_ohm_removed = v_raw - i_raw * r0
        finite = np.isfinite(soc) & np.isfinite(v_ohm_removed) & np.isfinite(i_raw)
        low_mask = finite & (np.abs(i_raw) <= low_current) & (soc >= 0.0) & (soc <= 0.800001)
        if np.any(low_mask):
            low_rows.setdefault(temp, []).append((soc[low_mask], v_ohm_removed[low_mask], profile))
        train_frames.setdefault(temp, []).append(
            {
                "times": times,
                "soc": soc,
                "i_raw": i_raw,
                "v_ohm_removed": v_ohm_removed,
                "profile": np.array([profile], dtype=object),
            }
        )

    curves: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    curve_audit_rows: list[dict] = []
    bin_width = max(float(getattr(cfg, "v_corr_ocv_bin_width_soc", 0.01)), 1e-4)
    min_points = max(int(getattr(cfg, "v_corr_ocv_min_bin_points", 5)), 1)
    for temp, parts in low_rows.items():
        soc_all = np.concatenate([p[0] for p in parts])
        v_all = np.concatenate([p[1] for p in parts])
        edges = np.arange(0.0, 0.800001 + bin_width, bin_width)
        xs: list[float] = []
        ys: list[float] = []
        counts: list[int] = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (soc_all >= lo) & (soc_all < hi if hi < 0.8 else soc_all <= hi)
            if int(mask.sum()) >= min_points:
                xs.append(float(np.nanmedian(soc_all[mask])))
                ys.append(float(np.nanmedian(v_all[mask])))
                counts.append(int(mask.sum()))
        if len(xs) < 6:
            order = np.argsort(soc_all)
            soc_sorted = soc_all[order]
            v_sorted = v_all[order]
            step = max(len(soc_sorted) // 80, 1)
            xs = [float(np.nanmedian(soc_sorted[i : i + step])) for i in range(0, len(soc_sorted), step)]
            ys = [float(np.nanmedian(v_sorted[i : i + step])) for i in range(0, len(v_sorted), step)]
            counts = [int(min(step, len(soc_sorted) - i)) for i in range(0, len(soc_sorted), step)]
        x_arr = np.asarray(xs, dtype=np.float64)
        y_arr = np.asarray(ys, dtype=np.float64)
        order = np.argsort(x_arr)
        x_arr = x_arr[order]
        y_arr = np.maximum.accumulate(y_arr[order])
        curves[float(temp)] = (x_arr.astype(np.float32), y_arr.astype(np.float32))
        for x, y, n in zip(x_arr, y_arr, np.asarray(counts)[order]):
            curve_audit_rows.append({"temperature_C": float(temp), "soc": float(x), "ocv_voltage": float(y), "n_low_current_points": int(n)})

    params: dict[float, dict[str, float]] = {}
    param_rows: list[dict] = []
    for temp, parts in train_frames.items():
        if not curves:
            beta = np.zeros(3, dtype=np.float64)
            n_fit = 0
            mae_before = np.nan
            mae_after = np.nan
        else:
            targets = []
            feats = []
            for part in parts:
                soc = part["soc"]
                v_ohm_removed = part["v_ohm_removed"]
                times = part["times"]
                i_raw = part["i_raw"]
                ocv = _interp_ocv_from_calibration({"curves": curves}, float(temp), soc)
                e_fast = causal_ema(i_raw, times, float(cfg.v_corr_ocv_tau_fast_s)).astype(np.float64)
                e_slow = causal_ema(i_raw, times, float(cfg.v_corr_ocv_tau_slow_s)).astype(np.float64)
                finite = np.isfinite(v_ohm_removed) & np.isfinite(ocv) & np.isfinite(e_fast) & np.isfinite(e_slow)
                if np.any(finite):
                    targets.append((v_ohm_removed[finite] - ocv[finite]).astype(np.float64))
                    feats.append(np.column_stack([np.ones(int(finite.sum())), e_fast[finite], e_slow[finite]]).astype(np.float64))
            if targets:
                y = np.concatenate(targets)
                x = np.vstack(feats)
                beta = _fit_ridge_dynamic_coefficients(x, y, cfg)
                resid_before = y
                resid_after = y - x @ beta
                n_fit = int(len(y))
                mae_before = float(np.nanmean(np.abs(resid_before)) * 1000.0)
                mae_after = float(np.nanmean(np.abs(resid_after)) * 1000.0)
            else:
                beta = np.zeros(3, dtype=np.float64)
                n_fit = 0
                mae_before = np.nan
                mae_after = np.nan
        params[float(temp)] = {"intercept_v": float(beta[0]), "a_fast_ohm": float(beta[1]), "a_slow_ohm": float(beta[2])}
        param_rows.append(
            {
                "temperature_C": float(temp),
                "intercept_v": float(beta[0]),
                "a_fast_ohm": float(beta[1]),
                "a_slow_ohm": float(beta[2]),
                "tau_fast_s": float(cfg.v_corr_ocv_tau_fast_s),
                "tau_slow_s": float(cfg.v_corr_ocv_tau_slow_s),
                "n_fit_points": int(n_fit),
                "train_mae_before_mV": mae_before,
                "train_mae_after_mV": mae_after,
            }
        )

    return {
        "curves": curves,
        "params": params,
        "curve_df": pd.DataFrame(curve_audit_rows),
        "params_df": pd.DataFrame(param_rows).sort_values("temperature_C").reset_index(drop=True) if param_rows else pd.DataFrame(),
    }


def _lfp_style_features(
    v_raw: np.ndarray,
    i_raw: np.ndarray,
    times: np.ndarray,
    scales: dict[str, float],
    cfg: NMCBranchBandsConfig,
) -> dict[str, np.ndarray]:
    # NMC raw current is negative during discharge, while the LFP corrector
    # convention adds a positive drop to recover an OCV-like voltage.
    i_used = -np.asarray(i_raw, dtype=np.float64)
    d_i = np.diff(i_used, prepend=i_used[0]).astype(np.float64)
    i_scale = max(float(scales.get("i_scale", 1.0)), 1e-6)
    di_scale = max(float(scales.get("di_scale", 1.0)), 1e-6)
    vmin = float(scales.get("vmin", np.nanmin(v_raw)))
    vmax = float(scales.get("vmax", np.nanmax(v_raw)))
    vrange = max(vmax - vmin, 1e-6)
    fast = causal_ema(i_used, times, float(getattr(cfg, "v_corr_ocv_tau_fast_s", 20.0))).astype(np.float64)
    mid = causal_ema(i_used, times, float(getattr(cfg, "v_pol_mid_tau_s", 60.0))).astype(np.float64)
    slow = causal_ema(i_used, times, float(getattr(cfg, "v_pol_slow_tau_s", 600.0))).astype(np.float64)
    hys_drive = np.sign(i_used) * np.sqrt(np.abs(i_used) + 1e-9)
    hys = causal_ema(hys_drive, times, float(getattr(cfg, "v_hys_tau_s", 1200.0))).astype(np.float64)
    return {
        "i_used": i_used,
        "d_i_used": d_i,
        "V_s": np.clip(2.0 * (np.asarray(v_raw, dtype=np.float64) - vmin) / vrange - 1.0, -3.0, 3.0),
        "absI_s": np.clip(np.abs(i_used) / i_scale, 0.0, 5.0),
        "absdI_s": np.clip(np.abs(d_i) / di_scale, 0.0, 5.0),
        "fast_s": np.clip(fast / i_scale, -5.0, 5.0),
        "mid_s": np.clip(mid / i_scale, -5.0, 5.0),
        "slow_s": np.clip(slow / i_scale, -5.0, 5.0),
        "hys_s": np.clip(hys / np.sqrt(i_scale), -5.0, 5.0),
    }


def _predict_lfp_style_vcorr(
    p: np.ndarray,
    v_raw: np.ndarray,
    i_raw: np.ndarray,
    times: np.ndarray,
    scales: dict[str, float],
    cfg: NMCBranchBandsConfig,
    return_parts: bool = False,
):
    feat = _lfp_style_features(v_raw, i_raw, times, scales, cfg)
    p = np.asarray(p, dtype=np.float64)
    v_pf = float(cfg.lfpstyle_pol_fast_limit_v) * np.tanh(p[0] * feat["fast_s"])
    v_pm = float(cfg.lfpstyle_pol_mid_limit_v) * np.tanh(p[1] * feat["mid_s"])
    v_ps = float(cfg.lfpstyle_pol_slow_limit_v) * np.tanh(p[2] * feat["slow_s"])
    v_pol = np.clip(v_pf + v_pm + v_ps, -float(cfg.lfpstyle_pol_limit_v), float(cfg.lfpstyle_pol_limit_v))
    v_hys = float(cfg.lfpstyle_hys_limit_v) * np.tanh(p[3] * feat["hys_s"])
    r0_min = float(getattr(cfg, "lfpstyle_r0_min_ohm", 0.0))
    r0_max = float(getattr(cfg, "lfpstyle_r0_max_ohm", 0.22))
    r0 = r0_min + (r0_max - r0_min) * _sigmoid_np(p[4] + p[5] * feat["absI_s"] + p[6] * feat["absdI_s"])
    v_ohm_drop = r0 * feat["i_used"]
    g_corr = float(cfg.lfpstyle_g_corr_scale) * _sigmoid_np(
        p[7] + p[8] * feat["V_s"] + p[9] * feat["absI_s"] + p[10] * feat["absdI_s"]
    )
    v_drop = g_corr * (v_pol + v_hys + v_ohm_drop)
    v_corr = _soft_floor_np(
        np.asarray(v_raw, dtype=np.float64) + v_drop,
        floor=float(cfg.lfpstyle_v_floor_raw),
        beta=float(cfg.lfpstyle_vfloor_beta),
    ).astype(np.float32)
    if not return_parts:
        return v_corr
    return v_corr, {
        "R0": r0.astype(np.float32),
        "v_ohm_drop": v_ohm_drop.astype(np.float32),
        "v_pol_raw": v_pol.astype(np.float32),
        "v_hys_raw": v_hys.astype(np.float32),
        "v_drop_raw": v_drop.astype(np.float32),
        "g_corr": g_corr.astype(np.float32),
    }


def estimate_lfp_style_vcorr_calibration(
    files: list[Path],
    train_profiles: tuple[str, ...],
    r0_df: pd.DataFrame,
    cfg: NMCBranchBandsConfig,
) -> dict:
    ocv_cal = estimate_vcorr_ocv_calibration(files, train_profiles, r0_df, cfg)
    r0_lookup = {float(r["temperature_C"]): float(r["r0_ohm"]) for _, r in r0_df.iterrows()}
    fit_stride = max(int(getattr(cfg, "lfpstyle_fit_stride", 5)), 1)
    by_temp: dict[float, list[dict[str, np.ndarray]]] = {}
    for pth in files:
        head = pd.read_csv(pth, nrows=2)
        profile = parse_profile(pth, head)
        if profile not in train_profiles:
            continue
        temp = float(parse_temperature(pth, head))
        df = pd.read_csv(pth)
        time_col = "Step_Time(s)" if "Step_Time(s)" in df.columns else "Test_Time(s)"
        times = pd.to_numeric(df[time_col], errors="coerce").to_numpy(np.float64)
        v_raw = pd.to_numeric(df["Voltage(V)"], errors="coerce").to_numpy(np.float64)
        i_raw = pd.to_numeric(df["Current(A)"], errors="coerce").to_numpy(np.float64)
        soc = _soc01_from_frame(df)
        ocv = _interp_ocv_from_calibration(ocv_cal, temp, soc)
        finite = np.isfinite(times) & np.isfinite(v_raw) & np.isfinite(i_raw) & np.isfinite(ocv)
        if not np.any(finite):
            continue
        by_temp.setdefault(temp, []).append(
            {
                "profile": np.array([profile], dtype=object),
                "times": times[finite],
                "v_raw": v_raw[finite],
                "i_raw": i_raw[finite],
                "soc": soc[finite],
                "ocv": ocv[finite],
            }
        )

    params: dict[float, dict[str, object]] = {}
    param_rows: list[dict] = []
    fit_rows: list[dict] = []
    coef_names = [
        "pol_fast_gain",
        "pol_mid_gain",
        "pol_slow_gain",
        "hys_gain",
        "r0_intercept",
        "r0_absI",
        "r0_absdI",
        "gate_intercept",
        "gate_Vs",
        "gate_absI",
        "gate_absdI",
    ]
    for temp, recs in sorted(by_temp.items()):
        v_all = np.concatenate([r["v_raw"] for r in recs])
        i_all = np.concatenate([r["i_raw"] for r in recs])
        di_all = np.diff(-i_all, prepend=-i_all[0])
        scales = {
            "vmin": float(np.nanpercentile(v_all, 1)),
            "vmax": float(np.nanpercentile(v_all, 99)),
            "i_scale": float(max(np.nanpercentile(np.abs(i_all), 95), 1e-3)),
            "di_scale": float(max(np.nanpercentile(np.abs(di_all), 95), 1e-3)),
        }
        r0_base = float(r0_lookup.get(float(temp), np.nanmedian(r0_df["r0_ohm"].to_numpy(float))))
        r0_frac = np.clip((r0_base - float(cfg.lfpstyle_r0_min_ohm)) / max(float(cfg.lfpstyle_r0_max_ohm) - float(cfg.lfpstyle_r0_min_ohm), 1e-9), 1e-4, 1.0 - 1e-4)
        p0 = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                _logit_np(float(r0_frac)),
                0.0,
                0.0,
                _logit_np(min(1.0 / max(float(cfg.lfpstyle_g_corr_scale), 1e-6), 0.98)),
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        lower = np.array([-4.0, -4.0, -4.0, -4.0, -8.0, -5.0, -5.0, -8.0, -5.0, -5.0, -5.0], dtype=np.float64)
        upper = np.array([4.0, 4.0, 4.0, 4.0, 8.0, 5.0, 5.0, 8.0, 5.0, 5.0, 5.0], dtype=np.float64)

        fit_parts = []
        for rec in recs:
            idx = np.arange(0, len(rec["v_raw"]), fit_stride)
            fit_parts.append(
                {
                    "v_raw": rec["v_raw"][idx],
                    "i_raw": rec["i_raw"][idx],
                    "times": rec["times"][idx],
                    "soc": rec["soc"][idx],
                    "ocv": rec["ocv"][idx],
                }
            )

        def residual(p: np.ndarray) -> np.ndarray:
            chunks = []
            for rec in fit_parts:
                pred = _predict_lfp_style_vcorr(p, rec["v_raw"], rec["i_raw"], rec["times"], scales, cfg).astype(np.float64)
                err = pred - rec["ocv"]
                weight = 1.0 + 0.5 * (rec["soc"] <= 0.20).astype(np.float64)
                chunks.append(err * weight)
            reg = np.array([0.01 * p[0], 0.01 * p[1], 0.01 * p[2], 0.01 * p[3], 0.005 * p[5], 0.005 * p[6], 0.005 * p[8], 0.005 * p[9], 0.005 * p[10]], dtype=np.float64)
            return np.concatenate(chunks + [reg])

        before = residual(p0)
        opt = least_squares(
            residual,
            p0,
            bounds=(lower, upper),
            max_nfev=int(getattr(cfg, "lfpstyle_fit_max_nfev", 250)),
            loss="soft_l1",
            f_scale=0.02,
        )
        p_fit = opt.x.astype(np.float64)
        after = residual(p_fit)
        params[float(temp)] = {"params": p_fit, "scales": scales}
        row = {
            "variant": "lfpstyle_ocvfit_v1",
            "temperature_C": float(temp),
            "n_train_profiles": int(len(recs)),
            "n_fit_points": int(sum(len(r["v_raw"]) for r in fit_parts)),
            "fit_stride": int(fit_stride),
            "r0_base_ohm": float(r0_base),
            "r0_min_ohm": float(cfg.lfpstyle_r0_min_ohm),
            "r0_max_ohm": float(cfg.lfpstyle_r0_max_ohm),
            "train_weighted_mae_before_mV": float(np.nanmean(np.abs(before)) * 1000.0),
            "train_weighted_mae_after_mV": float(np.nanmean(np.abs(after)) * 1000.0),
            "cost": float(opt.cost),
            "status": int(opt.status),
            "message": str(opt.message),
            **{name: float(val) for name, val in zip(coef_names, p_fit)},
            **{f"scale_{k}": float(v) for k, v in scales.items()},
        }
        param_rows.append(row)
        for rec in recs:
            pred = _predict_lfp_style_vcorr(p_fit, rec["v_raw"], rec["i_raw"], rec["times"], scales, cfg).astype(np.float64)
            abs_mv = np.abs(pred - rec["ocv"]) * 1000.0
            fit_rows.append(
                {
                    "variant": "lfpstyle_ocvfit_v1",
                    "temperature_C": float(temp),
                    "profile": str(rec["profile"][0]),
                    "rows": int(len(rec["v_raw"])),
                    "ocv_mae_mV": float(np.nanmean(abs_mv)),
                    "ocv_bias_mV": float(np.nanmean(pred - rec["ocv"]) * 1000.0),
                    "ocv_p95_abs_mV": float(np.nanpercentile(abs_mv, 95)),
                    "low_soc_0_20_mae_mV": float(np.nanmean(abs_mv[rec["soc"] <= 0.20])) if np.any(rec["soc"] <= 0.20) else np.nan,
                }
            )
    if not params:
        raise RuntimeError("Could not fit LFP-style V_corr calibration from train profiles.")
    return {
        "ocv_curve_df": ocv_cal.get("curve_df", pd.DataFrame()),
        "ocv_params_df": ocv_cal.get("params_df", pd.DataFrame()),
        "params": params,
        "params_df": pd.DataFrame(param_rows).sort_values("temperature_C").reset_index(drop=True),
        "fit_df": pd.DataFrame(fit_rows).sort_values(["temperature_C", "profile"]).reset_index(drop=True),
    }


def apply_lfp_style_vcorr(
    v_raw: np.ndarray,
    i_raw: np.ndarray,
    times: np.ndarray,
    temp: float,
    calibration: dict,
    cfg: NMCBranchBandsConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    params = calibration.get("params", {})
    if not params:
        raise RuntimeError("LFP-style V_corr requested, but calibration parameters are empty.")
    key = float(temp)
    if key not in params:
        key = min(params, key=lambda k: abs(float(k) - float(temp)))
    p = np.asarray(params[key]["params"], dtype=np.float64)
    scales = dict(params[key]["scales"])
    vcorr, parts = _predict_lfp_style_vcorr(p, v_raw, i_raw, times, scales, cfg, return_parts=True)
    return vcorr.astype(np.float32), parts


def _tailored_frame_from_path(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    temp = parse_temperature(path, df)
    profile = str(parse_profile(path, df)).upper()
    time_col = "Step_Time(s)" if "Step_Time(s)" in df.columns else "Test_Time(s)"
    soc = pd.to_numeric(df["SOC_CC"], errors="coerce").to_numpy(np.float64)
    if np.nanmax(soc) > 1.5:
        soc = soc / 100.0
    out = pd.DataFrame(
        {
            "file_name": path.name,
            "profile": profile,
            "temperature_C": float(temp),
            "end_index": np.arange(len(df), dtype=np.int64),
            "time_s": pd.to_numeric(df[time_col], errors="coerce").to_numpy(np.float64),
            "SOC_frac": np.clip(soc, 0.0, 1.0),
            "SOC_pct": np.clip(soc, 0.0, 1.0) * 100.0,
            "V_raw": pd.to_numeric(df["Voltage(V)"], errors="coerce").to_numpy(np.float64),
            "I_raw": pd.to_numeric(df["Current(A)"], errors="coerce").to_numpy(np.float64),
        }
    )
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_s", "SOC_frac", "V_raw", "I_raw"]).reset_index(drop=True)


def _add_tailored_baseline_vcorr(frame: pd.DataFrame, r0_ohm: float) -> pd.DataFrame:
    f = frame.copy()
    t = f["time_s"].to_numpy(np.float64)
    ohm_free = f["V_raw"].to_numpy(np.float64) - f["I_raw"].to_numpy(np.float64) * float(r0_ohm)
    f["R0_event_ohm"] = float(r0_ohm)
    f["V_corr_r0_ema120"] = causal_ema(ohm_free, t, 120.0)
    return f


def _fit_tailored_pseudo_ocv_curve(
    train_frames: list[pd.DataFrame],
    r0_ohm: float,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    parts = []
    for frame in train_frames:
        i = frame["I_raw"].to_numpy(np.float64)
        v = frame["V_raw"].to_numpy(np.float64)
        di = np.diff(i, prepend=i[0])
        parts.append(
            pd.DataFrame(
                {
                    "profile": str(frame["profile"].iloc[0]),
                    "soc": frame["SOC_frac"].to_numpy(np.float64),
                    "v_ohm_free": v - i * float(r0_ohm),
                    "abs_i": np.abs(i),
                    "abs_di": np.abs(di),
                }
            )
        )
    allp = pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
    if allp.empty:
        raise RuntimeError("Cannot fit NMC-tailored pseudo OCV curve from empty train frames.")
    abs_i = allp["abs_i"].to_numpy(np.float64)
    abs_di = allp["abs_di"].to_numpy(np.float64)
    low_i = max(0.05, float(np.nanquantile(abs_i, 0.18)))
    low_di = max(0.01, float(np.nanquantile(abs_di, 0.35)))
    source = allp[(allp["abs_i"] <= low_i) & (allp["abs_di"] <= low_di)].copy()
    if len(source) < 500:
        low_i = max(0.10, float(np.nanquantile(abs_i, 0.30)))
        low_di = max(0.02, float(np.nanquantile(abs_di, 0.50)))
        source = allp[(allp["abs_i"] <= low_i) & (allp["abs_di"] <= low_di)].copy()
    if len(source) < 200:
        source = allp.copy()
    bin_width = 0.01
    edges = np.arange(0.0, max(0.91, float(source["soc"].max()) + bin_width), bin_width)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = source[(source["soc"] >= lo) & (source["soc"] < hi if hi < edges[-1] else source["soc"] <= hi)]
        if len(b) >= 5:
            rows.append(
                {
                    "soc": float(b["soc"].median()),
                    "ocv_voltage": float(b["v_ohm_free"].median()),
                    "n": int(len(b)),
                    "soc_lo": float(lo),
                    "soc_hi": float(hi),
                    "low_i_threshold_A": float(low_i),
                    "low_dI_threshold_A": float(low_di),
                    "source_rows": int(len(source)),
                }
            )
    if len(rows) < 8:
        sorted_df = allp.sort_values("soc").reset_index(drop=True)
        step = max(len(sorted_df) // 80, 1)
        rows = []
        for start in range(0, len(sorted_df), step):
            chunk = sorted_df.iloc[start : start + step]
            rows.append(
                {
                    "soc": float(chunk["soc"].median()),
                    "ocv_voltage": float(chunk["v_ohm_free"].median()),
                    "n": int(len(chunk)),
                    "soc_lo": float(chunk["soc"].min()),
                    "soc_hi": float(chunk["soc"].max()),
                    "low_i_threshold_A": float("nan"),
                    "low_dI_threshold_A": float("nan"),
                    "source_rows": int(len(allp)),
                }
            )
    curve = pd.DataFrame(rows).dropna(subset=["soc", "ocv_voltage"]).sort_values("soc")
    curve = curve.drop_duplicates("soc", keep="last")
    x = curve["soc"].to_numpy(np.float64)
    y = np.maximum.accumulate(curve["ocv_voltage"].to_numpy(np.float64))
    curve["ocv_voltage_monotone"] = y
    return x, y, curve.reset_index(drop=True)


def _apply_tailored_ocv_curve(frame: pd.DataFrame, curve_soc: np.ndarray, curve_v: np.ndarray) -> pd.DataFrame:
    f = frame.copy()
    f["OCV_from_SOC"] = np.interp(
        f["SOC_frac"].to_numpy(np.float64),
        np.asarray(curve_soc, dtype=np.float64),
        np.asarray(curve_v, dtype=np.float64),
        left=float(curve_v[0]),
        right=float(curve_v[-1]),
    )
    return f


def _tailored_feature_matrix(frame: pd.DataFrame, scales: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    t = frame["time_s"].to_numpy(np.float64)
    v = frame["V_raw"].to_numpy(np.float64)
    i_dis = -frame["I_raw"].to_numpy(np.float64)
    di = np.diff(i_dis, prepend=i_dis[0])
    abs_i = np.abs(i_dis)
    abs_di = np.abs(di)
    i_scale = max(float(scales["i_scale"]), 1e-6)
    di_scale = max(float(scales["di_scale"]), 1e-6)
    v_s = np.clip((v - float(scales["v_center"])) / max(float(scales["v_scale"]), 1e-6), -5.0, 5.0)
    low_gate = _sigmoid_np((float(scales["low_v_gate"]) - v) / max(float(scales["low_v_slope"]), 1e-6))
    high_gate = _sigmoid_np((v - float(scales["high_v_gate"])) / 0.08)
    i_fast = causal_ema(i_dis, t, 5.0).astype(np.float64)
    i_mid = causal_ema(i_dis, t, 30.0).astype(np.float64)
    i_slow = causal_ema(i_dis, t, 240.0).astype(np.float64)
    i_very_slow = causal_ema(i_dis, t, 900.0).astype(np.float64)
    abs_i_mid = causal_ema(abs_i, t, 30.0).astype(np.float64)
    abs_i_slow = causal_ema(abs_i, t, 240.0).astype(np.float64)
    abs_di_fast = causal_ema(abs_di, t, 10.0).astype(np.float64)
    hys = causal_ema(np.sign(i_dis) * np.sqrt(abs_i + 1e-9), t, 900.0).astype(np.float64)
    names = [
        "bias",
        "i_dis",
        "i_fast",
        "i_mid",
        "i_slow",
        "i_very_slow",
        "abs_i_mid",
        "abs_i_slow",
        "abs_di_fast",
        "hys",
        "v_s_x_i_dis",
        "v_s_x_i_mid",
        "low_gate_x_i_dis",
        "low_gate_x_i_mid",
        "low_gate_x_abs_i_mid",
        "low_gate_x_abs_di_fast",
        "high_gate_x_i_dis",
    ]
    x = np.column_stack(
        [
            np.ones_like(v),
            i_dis / i_scale,
            i_fast / i_scale,
            i_mid / i_scale,
            i_slow / i_scale,
            i_very_slow / i_scale,
            abs_i_mid / i_scale,
            abs_i_slow / i_scale,
            abs_di_fast / di_scale,
            hys / np.sqrt(i_scale),
            v_s * (i_dis / i_scale),
            v_s * (i_mid / i_scale),
            low_gate * (i_dis / i_scale),
            low_gate * (i_mid / i_scale),
            low_gate * (abs_i_mid / i_scale),
            low_gate * (abs_di_fast / di_scale),
            high_gate * (i_dis / i_scale),
        ]
    )
    return x.astype(np.float64), names


def _fit_nmc_tailored_vcorr_model(train: pd.DataFrame) -> dict:
    tr = train.dropna(subset=["V_raw", "I_raw", "OCV_from_SOC", "SOC_pct"]).copy()
    v = tr["V_raw"].to_numpy(np.float64)
    i_dis = -tr["I_raw"].to_numpy(np.float64)
    di = np.diff(i_dis, prepend=i_dis[0])
    y = tr["OCV_from_SOC"].to_numpy(np.float64) - v
    scales = {
        "i_scale": float(max(np.nanpercentile(np.abs(i_dis), 95), 1e-3)),
        "di_scale": float(max(np.nanpercentile(np.abs(di), 95), 1e-3)),
        "v_center": float(np.nanmedian(v)),
        "v_scale": float(max(np.nanpercentile(v, 95) - np.nanpercentile(v, 5), 1e-3)),
        "low_v_gate": float(np.nanpercentile(v, 18)),
        "low_v_slope": float(max((np.nanpercentile(v, 30) - np.nanpercentile(v, 10)) / 3.0, 0.04)),
        "high_v_gate": float(np.nanpercentile(v, 90)),
    }
    x, names = _tailored_feature_matrix(tr, scales)
    finite = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x = x[finite]
    y = y[finite]
    soc = tr["SOC_pct"].to_numpy(np.float64)[finite]
    weights = np.ones_like(y)
    weights += 0.75 * (soc <= 20.0)
    weights += 0.25 * (soc >= 80.0)
    x_mu = np.nanmean(x, axis=0)
    x_sig = np.nanstd(x, axis=0)
    x_mu[0] = 0.0
    x_sig[0] = 1.0
    x_sig = np.where(x_sig > 1e-9, x_sig, 1.0)
    xs = (x - x_mu) / x_sig
    sqrt_w = np.sqrt(weights)
    xw = xs * sqrt_w[:, None]
    yw = y * sqrt_w
    ridge = 2e-3
    beta = np.zeros(xs.shape[1], dtype=np.float64)
    keep = np.ones(len(y), dtype=bool)
    for _ in range(4):
        reg = np.eye(xs.shape[1], dtype=np.float64) * ridge
        reg[0, 0] = ridge * 0.01
        try:
            beta = np.linalg.solve(xw[keep].T @ xw[keep] + reg, xw[keep].T @ yw[keep])
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(xw[keep], yw[keep], rcond=None)[0]
        resid = np.abs((xs @ beta) - y)
        cut = np.nanquantile(resid, 0.92)
        keep = np.isfinite(resid) & (resid <= max(float(cut), 1e-6))
        if int(keep.sum()) < 200:
            break
    raw_delta = xs @ beta
    cap = float(np.clip(np.nanpercentile(np.abs(y), 99.5) * 1.15, 0.08, 0.95))
    train_pred = np.clip(raw_delta, -cap, cap)
    return {
        "scales": scales,
        "names": names,
        "x_mu": x_mu,
        "x_sig": x_sig,
        "beta": beta,
        "cap_V": cap,
        "train_mae_mV": float(np.nanmean(np.abs(train_pred - y)) * 1000.0),
        "train_bias_mV": float(np.nanmean(train_pred - y) * 1000.0),
        "train_rows": int(len(y)),
    }


def _apply_nmc_tailored_vcorr_model(frame: pd.DataFrame, model: dict) -> np.ndarray:
    x, _ = _tailored_feature_matrix(frame, model["scales"])
    xs = (x - model["x_mu"]) / model["x_sig"]
    delta = xs @ model["beta"]
    delta = np.clip(delta, -float(model["cap_V"]), float(model["cap_V"]))
    return np.clip(frame["V_raw"].to_numpy(np.float64) + delta, 2.45, 4.25).astype(np.float32)


def _add_nmc_tailored_minimax_columns(frame: pd.DataFrame, model: dict, alpha: float) -> pd.DataFrame:
    out = frame.copy()
    out["V_corr_nmc_tailored"] = _apply_nmc_tailored_vcorr_model(out, model)
    out["V_corr_nmc_tailored_ema10"] = causal_ema(
        out["V_corr_nmc_tailored"].to_numpy(np.float64),
        out["time_s"].to_numpy(np.float64),
        10.0,
    )
    out["V_corr_nmc_tailored_minimax"] = (
        (1.0 - float(alpha)) * out["V_corr_r0_ema120"].to_numpy(np.float64)
        + float(alpha) * out["V_corr_nmc_tailored_ema10"].to_numpy(np.float64)
    ).astype(np.float32)
    return out


def _alignment_feature_matrix(
    frame: pd.DataFrame,
    scales: dict[str, float],
    base_col: str = "V_corr_nmc_tailored_minimax",
) -> tuple[np.ndarray, list[str]]:
    t = frame["time_s"].to_numpy(np.float64)
    v_raw = frame["V_raw"].to_numpy(np.float64)
    v_base = frame[base_col].to_numpy(np.float64)
    i_dis = -frame["I_raw"].to_numpy(np.float64)
    di = np.diff(i_dis, prepend=i_dis[0])
    dv = np.diff(v_base, prepend=v_base[0])
    dyn = v_raw - v_base
    i_scale = max(float(scales["i_scale"]), 1e-6)
    di_scale = max(float(scales["di_scale"]), 1e-6)
    dyn_scale = max(float(scales["dyn_scale"]), 1e-6)
    dv_scale = max(float(scales["dv_scale"]), 1e-6)
    v_scale = max(float(scales["v_scale"]), 1e-6)
    v_s = np.clip((v_base - float(scales["v_center"])) / v_scale, -5.0, 5.0)
    dyn_s = np.clip(dyn / dyn_scale, -5.0, 5.0)
    low_gate = _sigmoid_np((float(scales["low_v_gate"]) - v_base) / max(float(scales["low_v_slope"]), 1e-6))
    high_gate = _sigmoid_np((v_base - float(scales["high_v_gate"])) / max(float(scales["high_v_slope"]), 1e-6))
    i_fast = causal_ema(i_dis, t, 5.0).astype(np.float64)
    i_mid = causal_ema(i_dis, t, 30.0).astype(np.float64)
    i_slow = causal_ema(i_dis, t, 240.0).astype(np.float64)
    abs_i_mid = causal_ema(np.abs(i_dis), t, 30.0).astype(np.float64)
    abs_di_fast = causal_ema(np.abs(di), t, 10.0).astype(np.float64)
    dyn_fast = causal_ema(dyn, t, 10.0).astype(np.float64)
    dyn_slow = causal_ema(dyn, t, 240.0).astype(np.float64)
    hys = causal_ema(np.sign(i_dis) * np.sqrt(np.abs(i_dis) + 1e-9), t, 900.0).astype(np.float64)
    names = [
        "bias",
        "v_base_s",
        "low_gate",
        "high_gate",
        "dyn_s",
        "dyn_fast_s",
        "dyn_slow_s",
        "dV_base_s",
        "i_dis",
        "i_fast",
        "i_mid",
        "i_slow",
        "abs_i_mid",
        "abs_di_fast",
        "hys",
        "low_gate_x_dyn",
        "low_gate_x_dyn_fast",
        "low_gate_x_i_mid",
        "low_gate_x_abs_di_fast",
        "low_gate_x_dV_base",
        "v_s_x_dyn",
        "high_gate_x_dyn",
    ]
    x = np.column_stack(
        [
            np.ones_like(v_base),
            v_s,
            low_gate,
            high_gate,
            dyn_s,
            dyn_fast / dyn_scale,
            dyn_slow / dyn_scale,
            dv / dv_scale,
            i_dis / i_scale,
            i_fast / i_scale,
            i_mid / i_scale,
            i_slow / i_scale,
            abs_i_mid / i_scale,
            abs_di_fast / di_scale,
            hys / np.sqrt(i_scale),
            low_gate * dyn_s,
            low_gate * (dyn_fast / dyn_scale),
            low_gate * (i_mid / i_scale),
            low_gate * (abs_di_fast / di_scale),
            low_gate * (dv / dv_scale),
            v_s * dyn_s,
            high_gate * dyn_s,
        ]
    )
    return x.astype(np.float64), names


def _fit_alignment_vcorr_model(
    train: pd.DataFrame,
    base_col: str = "V_corr_nmc_tailored_minimax",
) -> dict:
    tr = train.dropna(subset=["V_raw", "I_raw", "OCV_from_SOC", base_col]).copy()
    v_base = tr[base_col].to_numpy(np.float64)
    v_raw = tr["V_raw"].to_numpy(np.float64)
    i_dis = -tr["I_raw"].to_numpy(np.float64)
    di = np.diff(i_dis, prepend=i_dis[0])
    dv = np.diff(v_base, prepend=v_base[0])
    dyn = v_raw - v_base
    y = tr["OCV_from_SOC"].to_numpy(np.float64) - v_base
    scales = {
        "i_scale": float(max(np.nanpercentile(np.abs(i_dis), 95), 1e-3)),
        "di_scale": float(max(np.nanpercentile(np.abs(di), 95), 1e-3)),
        "dyn_scale": float(max(np.nanpercentile(np.abs(dyn), 95), 1e-3)),
        "dv_scale": float(max(np.nanpercentile(np.abs(dv), 95), 1e-4)),
        "v_center": float(np.nanmedian(v_base)),
        "v_scale": float(max(np.nanpercentile(v_base, 95) - np.nanpercentile(v_base, 5), 1e-3)),
        "low_v_gate": float(np.nanpercentile(v_base, 22)),
        "low_v_slope": float(max((np.nanpercentile(v_base, 35) - np.nanpercentile(v_base, 12)) / 3.0, 0.035)),
        "high_v_gate": float(np.nanpercentile(v_base, 88)),
        "high_v_slope": 0.08,
    }
    x, names = _alignment_feature_matrix(tr, scales, base_col)
    finite = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x = x[finite]
    y = y[finite]
    v_fit = v_base[finite]
    low_gate = _sigmoid_np((float(scales["low_v_gate"]) - v_fit) / max(float(scales["low_v_slope"]), 1e-6))
    weights = 1.0 + 1.5 * low_gate
    x_mu = np.nanmean(x, axis=0)
    x_sig = np.nanstd(x, axis=0)
    x_mu[0] = 0.0
    x_sig[0] = 1.0
    x_sig = np.where(x_sig > 1e-9, x_sig, 1.0)
    xs = (x - x_mu) / x_sig
    sqrt_w = np.sqrt(weights)
    xw = xs * sqrt_w[:, None]
    yw = y * sqrt_w
    ridge = 1.5e-2
    beta = np.zeros(xs.shape[1], dtype=np.float64)
    keep = np.ones(len(y), dtype=bool)
    for _ in range(4):
        reg = np.eye(xs.shape[1], dtype=np.float64) * ridge
        reg[0, 0] = ridge * 0.01
        try:
            beta = np.linalg.solve(xw[keep].T @ xw[keep] + reg, xw[keep].T @ yw[keep])
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(xw[keep], yw[keep], rcond=None)[0]
        resid = np.abs((xs @ beta) - y)
        cut = np.nanquantile(resid, 0.90)
        keep = np.isfinite(resid) & (resid <= max(float(cut), 1e-6))
        if int(keep.sum()) < 200:
            break
    train_pred = xs @ beta
    cap = float(np.clip(np.nanpercentile(np.abs(y), 98.0) * 1.25, 0.015, 0.12))
    train_pred = np.clip(train_pred, -cap, cap)
    return {
        "scales": scales,
        "names": names,
        "x_mu": x_mu,
        "x_sig": x_sig,
        "beta": beta,
        "cap_V": cap,
        "train_mae_mV": float(np.nanmean(np.abs(train_pred - y)) * 1000.0),
        "train_bias_mV": float(np.nanmean(train_pred - y) * 1000.0),
        "train_rows": int(len(y)),
    }


def _apply_alignment_vcorr_model(
    frame: pd.DataFrame,
    model: dict,
    gamma: float,
    base_col: str = "V_corr_nmc_tailored_minimax",
) -> np.ndarray:
    x, _ = _alignment_feature_matrix(frame, model["scales"], base_col)
    xs = (x - model["x_mu"]) / model["x_sig"]
    correction = xs @ model["beta"]
    correction = np.clip(correction, -float(model["cap_V"]), float(model["cap_V"]))
    out = frame[base_col].to_numpy(np.float64) + float(gamma) * correction
    return np.clip(out, 2.45, 4.25).astype(np.float32)


def _select_alignment_gamma(
    raw_frames_by_profile: dict[str, pd.DataFrame],
    train_profiles: tuple[str, ...],
    temperature: float,
) -> tuple[float, pd.DataFrame]:
    candidates = [0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.00]
    rows = []
    for valid_profile in train_profiles:
        inner_train_profiles = tuple(p for p in train_profiles if p != valid_profile)
        inner_raw = [raw_frames_by_profile[p] for p in inner_train_profiles]
        valid_raw = raw_frames_by_profile[valid_profile]
        r0_inner = _estimate_tailored_r0_event(inner_raw)[float(temperature)]
        curve_soc, curve_v, _curve_df = _fit_tailored_pseudo_ocv_curve(inner_raw, r0_inner)
        inner_base = [
            _apply_tailored_ocv_curve(_add_tailored_baseline_vcorr(frame, r0_inner), curve_soc, curve_v)
            for frame in inner_raw
        ]
        valid_base = _apply_tailored_ocv_curve(_add_tailored_baseline_vcorr(valid_raw, r0_inner), curve_soc, curve_v)
        model = _fit_nmc_tailored_vcorr_model(pd.concat(inner_base, ignore_index=True))
        inner_alpha_scores = _select_nmc_tailored_minimax_alpha(
            {str(frame["profile"].iloc[0]).upper(): frame for frame in inner_raw},
            inner_train_profiles,
            float(temperature),
        )[1]
        alpha = float(inner_alpha_scores["selected_minimax_alpha"].iloc[0])
        inner_ready = [_add_nmc_tailored_minimax_columns(frame, model, alpha) for frame in inner_base]
        valid_ready = _add_nmc_tailored_minimax_columns(valid_base, model, alpha)
        align_model = _fit_alignment_vcorr_model(pd.concat(inner_ready, ignore_index=True))
        target = valid_ready["OCV_from_SOC"].to_numpy(np.float64)
        for gamma in candidates:
            pred = _apply_alignment_vcorr_model(valid_ready, align_model, float(gamma))
            diff = (pred - target) * 1000.0
            rows.append(
                {
                    "temperature_C": float(temperature),
                    "valid_profile": valid_profile,
                    "inner_train_profiles": "+".join(inner_train_profiles),
                    "gamma": float(gamma),
                    "mae_mV": float(np.nanmean(np.abs(diff))),
                    "bias_mV": float(np.nanmean(diff)),
                }
            )
    scores = pd.DataFrame(rows)
    agg = (
        scores.groupby("gamma", as_index=False)
        .agg(inner_mean_mae_mV=("mae_mV", "mean"), inner_max_mae_mV=("mae_mV", "max"))
        .sort_values(["inner_max_mae_mV", "inner_mean_mae_mV", "gamma"])
    )
    best = agg.iloc[0]
    gamma = float(best["gamma"])
    scores["selected_gamma"] = gamma
    scores["selected_inner_mean_mae_mV"] = float(best["inner_mean_mae_mV"])
    scores["selected_inner_max_mae_mV"] = float(best["inner_max_mae_mV"])
    return gamma, scores


def _select_nmc_tailored_minimax_alpha(
    raw_frames_by_profile: dict[str, pd.DataFrame],
    train_profiles: tuple[str, ...],
    temperature: float,
) -> tuple[float, pd.DataFrame]:
    rows = []
    candidates = np.linspace(0.0, 1.0, 21)
    for valid_profile in train_profiles:
        inner_train_profiles = tuple(p for p in train_profiles if p != valid_profile)
        inner_raw = [raw_frames_by_profile[p] for p in inner_train_profiles]
        valid_raw = raw_frames_by_profile[valid_profile]
        r0_inner = _estimate_tailored_r0_event(inner_raw)[float(temperature)]
        curve_soc, curve_v, _curve_df = _fit_tailored_pseudo_ocv_curve(inner_raw, r0_inner)
        inner_train = [
            _apply_tailored_ocv_curve(_add_tailored_baseline_vcorr(frame, r0_inner), curve_soc, curve_v)
            for frame in inner_raw
        ]
        valid_frame = _apply_tailored_ocv_curve(_add_tailored_baseline_vcorr(valid_raw, r0_inner), curve_soc, curve_v)
        model = _fit_nmc_tailored_vcorr_model(pd.concat(inner_train, ignore_index=True))
        tailored = _apply_nmc_tailored_vcorr_model(valid_frame, model)
        tailored_ema10 = causal_ema(tailored, valid_frame["time_s"].to_numpy(np.float64), 10.0)
        old = valid_frame["V_corr_r0_ema120"].to_numpy(np.float64)
        target = valid_frame["OCV_from_SOC"].to_numpy(np.float64)
        for alpha in candidates:
            pred = (1.0 - float(alpha)) * old + float(alpha) * tailored_ema10
            diff = (pred - target) * 1000.0
            rows.append(
                {
                    "temperature_C": float(temperature),
                    "valid_profile": valid_profile,
                    "inner_train_profiles": "+".join(inner_train_profiles),
                    "alpha": float(alpha),
                    "mae_mV": float(np.nanmean(np.abs(diff))),
                    "bias_mV": float(np.nanmean(diff)),
                }
            )
    scores = pd.DataFrame(rows)
    agg = (
        scores.groupby("alpha", as_index=False)
        .agg(inner_mean_mae_mV=("mae_mV", "mean"), inner_max_mae_mV=("mae_mV", "max"))
        .sort_values(["inner_max_mae_mV", "inner_mean_mae_mV", "alpha"])
    )
    best = agg.iloc[0]
    alpha = float(best["alpha"])
    scores["selected_minimax_alpha"] = alpha
    scores["selected_minimax_inner_mean_mae_mV"] = float(best["inner_mean_mae_mV"])
    scores["selected_minimax_inner_max_mae_mV"] = float(best["inner_max_mae_mV"])
    return alpha, scores


def _estimate_tailored_r0_event(frames: list[pd.DataFrame], quantile: float = 0.5) -> dict[float, float]:
    rows = []
    all_ratio = []
    temps = sorted({float(frame["temperature_C"].iloc[0]) for frame in frames})
    for frame in frames:
        temp = float(frame["temperature_C"].iloc[0])
        i = frame["I_raw"].to_numpy(np.float64)
        v = frame["V_raw"].to_numpy(np.float64)
        di = np.diff(i, prepend=i[0])
        dv = np.diff(v, prepend=v[0])
        ratio = np.full_like(dv, np.nan, dtype=np.float64)
        np.divide(dv, di, out=ratio, where=np.abs(di) > 1e-12)
        mask = np.isfinite(ratio) & (np.abs(di) > 0.05) & (np.abs(dv) > 1e-5) & (ratio > 0.001) & (ratio < 0.5)
        vals = ratio[mask]
        all_ratio.extend([float(r) for r in vals])
        for r in vals:
            rows.append({"temperature_C": temp, "r0_event_ohm": float(r)})
    if not all_ratio:
        raise RuntimeError("No current-step R0 events found for NMC-tailored V_corr.")
    fallback = float(np.quantile(all_ratio, quantile))
    out = {}
    for temp in temps:
        vals = [r["r0_event_ohm"] for r in rows if float(r["temperature_C"]) == float(temp)]
        out[float(temp)] = float(np.quantile(vals, quantile)) if vals else fallback
    return out


def estimate_nmc_tailored_minimax_calibration(
    files: list[Path],
    train_profiles: tuple[str, ...],
    r0_df: pd.DataFrame,
    cfg: NMCBranchBandsConfig,
) -> dict:
    raw_frames = [_tailored_frame_from_path(path) for path in files]
    train_profiles = tuple(str(p).upper() for p in train_profiles)
    r0_lookup = {float(r["temperature_C"]): float(r["r0_ohm"]) for _, r in r0_df.iterrows()}
    params: dict[float, dict] = {}
    param_rows = []
    curve_rows = []
    alpha_rows = []
    alignment_rows = []
    aligned = _nmc_tailored_aligned_variant(str(getattr(cfg, "v_corr_variant", "")))
    train_raw = [f for f in raw_frames if str(f["profile"].iloc[0]).upper() in train_profiles]
    for temp in sorted({float(frame["temperature_C"].iloc[0]) for frame in train_raw}):
        train_temp_raw = [f for f in train_raw if float(f["temperature_C"].iloc[0]) == float(temp)]
        if not train_temp_raw:
            continue
        if float(temp) in r0_lookup:
            r0 = float(r0_lookup[float(temp)])
        else:
            r0 = float(_estimate_tailored_r0_event(train_temp_raw)[float(temp)])
        curve_soc, curve_v, curve_df = _fit_tailored_pseudo_ocv_curve(train_temp_raw, r0)
        train_temp = [
            _apply_tailored_ocv_curve(_add_tailored_baseline_vcorr(frame, r0), curve_soc, curve_v)
            for frame in train_temp_raw
        ]
        model = _fit_nmc_tailored_vcorr_model(pd.concat(train_temp, ignore_index=True))
        raw_by_profile = {str(frame["profile"].iloc[0]).upper(): frame for frame in train_temp_raw}
        alpha, alpha_scores = _select_nmc_tailored_minimax_alpha(raw_by_profile, train_profiles, float(temp))
        align_model = None
        alignment_gamma = 0.0
        if aligned:
            alignment_gamma, alignment_scores = _select_alignment_gamma(raw_by_profile, train_profiles, float(temp))
            train_ready = [_add_nmc_tailored_minimax_columns(frame, model, alpha) for frame in train_temp]
            align_model = _fit_alignment_vcorr_model(pd.concat(train_ready, ignore_index=True))
            alignment_rows.append(
                alignment_scores.assign(
                    temperature_C=float(temp),
                    final_train_profiles="+".join(train_profiles),
                )
            )
        params[float(temp)] = {
            "r0_ohm": r0,
            "curve_soc": curve_soc,
            "curve_v": curve_v,
            "model": model,
            "alpha": float(alpha),
            "alignment_model": align_model,
            "alignment_gamma": float(alignment_gamma),
        }
        curve_rows.append(curve_df.assign(temperature_C=float(temp), train_profiles="+".join(train_profiles)))
        alpha_rows.append(alpha_scores.assign(temperature_C=float(temp), final_train_profiles="+".join(train_profiles)))
        for name, beta, mu, sig in zip(model["names"], model["beta"], model["x_mu"], model["x_sig"]):
            param_rows.append(
                {
                    "temperature_C": float(temp),
                    "train_profiles": "+".join(train_profiles),
                    "feature": name,
                    "beta": float(beta),
                    "x_mu": float(mu),
                    "x_sig": float(sig),
                    "cap_V": float(model["cap_V"]),
                    "train_rows": int(model["train_rows"]),
                    "train_mae_mV": float(model["train_mae_mV"]),
                    "train_bias_mV": float(model["train_bias_mV"]),
                    "r0_ohm": float(r0),
                    "alpha": float(alpha),
                    "alignment_gamma": float(alignment_gamma),
                    "param_group": "tailored",
                }
            )
        if align_model is not None:
            for name, beta, mu, sig in zip(align_model["names"], align_model["beta"], align_model["x_mu"], align_model["x_sig"]):
                param_rows.append(
                    {
                        "temperature_C": float(temp),
                        "train_profiles": "+".join(train_profiles),
                        "feature": name,
                        "beta": float(beta),
                        "x_mu": float(mu),
                        "x_sig": float(sig),
                        "cap_V": float(align_model["cap_V"]),
                        "train_rows": int(align_model["train_rows"]),
                        "train_mae_mV": float(align_model["train_mae_mV"]),
                        "train_bias_mV": float(align_model["train_bias_mV"]),
                        "r0_ohm": float(r0),
                        "alpha": float(alpha),
                        "alignment_gamma": float(alignment_gamma),
                        "param_group": "alignment",
                    }
                )
    if not params:
        raise RuntimeError("Could not fit NMC-tailored minimax V_corr calibration from train profiles.")
    return {
        "params": params,
        "params_df": pd.DataFrame(param_rows),
        "curve_df": pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame(),
        "alpha_scores_df": pd.concat(alpha_rows, ignore_index=True) if alpha_rows else pd.DataFrame(),
        "alignment_scores_df": pd.concat(alignment_rows, ignore_index=True) if alignment_rows else pd.DataFrame(),
    }


def apply_nmc_tailored_minimax_vcorr(
    v_raw: np.ndarray,
    i_raw: np.ndarray,
    times: np.ndarray,
    temp: float,
    calibration: dict,
) -> tuple[np.ndarray, float]:
    params = calibration.get("params", {})
    if not params:
        raise RuntimeError("NMC-tailored minimax V_corr requested, but calibration parameters are empty.")
    key = float(temp)
    if key not in params:
        key = min(params, key=lambda k: abs(float(k) - float(temp)))
    p = params[key]
    frame = pd.DataFrame(
        {
            "time_s": np.asarray(times, dtype=np.float64),
            "V_raw": np.asarray(v_raw, dtype=np.float64),
            "I_raw": np.asarray(i_raw, dtype=np.float64),
        }
    )
    old = causal_ema(
        frame["V_raw"].to_numpy(np.float64) - frame["I_raw"].to_numpy(np.float64) * float(p["r0_ohm"]),
        frame["time_s"].to_numpy(np.float64),
        120.0,
    ).astype(np.float64)
    tailored = _apply_nmc_tailored_vcorr_model(frame, p["model"])
    tailored_ema10 = causal_ema(tailored, frame["time_s"].to_numpy(np.float64), 10.0).astype(np.float64)
    alpha = float(p["alpha"])
    vcorr = (1.0 - alpha) * old + alpha * tailored_ema10
    frame["V_corr_r0_ema120"] = old.astype(np.float32)
    frame["V_corr_nmc_tailored"] = tailored.astype(np.float32)
    frame["V_corr_nmc_tailored_ema10"] = tailored_ema10.astype(np.float32)
    frame["V_corr_nmc_tailored_minimax"] = vcorr.astype(np.float32)
    align_model = p.get("alignment_model")
    alignment_gamma = float(p.get("alignment_gamma", 0.0))
    if align_model is not None and alignment_gamma != 0.0:
        vcorr = _apply_alignment_vcorr_model(frame, align_model, alignment_gamma)
    return vcorr.astype(np.float32), float(p["r0_ohm"])


def build_decomposed_frame(
    path: Path,
    r0_lookup: dict[float, float],
    cfg: NMCBranchBandsConfig,
    ocv_calibration: dict | None = None,
    dynamic_r_calibration: dict | None = None,
    lfp_style_calibration: dict | None = None,
    nmc_tailored_calibration: dict | None = None,
    rtvar_lowv_calibration: dict | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    temp = parse_temperature(path, df)
    profile = parse_profile(path, df)
    r0 = float(r0_lookup[float(temp)])

    time_col = "Step_Time(s)" if "Step_Time(s)" in df.columns else "Test_Time(s)"
    times = pd.to_numeric(df[time_col], errors="coerce").to_numpy(np.float64)
    v_raw = pd.to_numeric(df["Voltage(V)"], errors="coerce").to_numpy(np.float64)
    i_raw = pd.to_numeric(df["Current(A)"], errors="coerce").to_numpy(np.float64)
    d_i = np.diff(i_raw, prepend=i_raw[0]).astype(np.float32)
    abs_i = np.abs(i_raw).astype(np.float32)

    v_ohm = (i_raw * r0).astype(np.float32)
    v_ohm_removed = v_raw - v_ohm
    r_output = np.full(len(df), r0, dtype=np.float32)
    v_corr_variant = str(getattr(cfg, "v_corr_variant", "orig_ohm_ema120"))
    if v_corr_variant in {"orig", "orig_ohm_ema120", "ohm_ema120"}:
        v_corr = causal_ema(v_ohm_removed, times, cfg.v_corr_tau_s)
    elif v_corr_variant in {"ohm_asym_d20_u160", "ohm_asym"}:
        down_tau = float(getattr(cfg, "v_corr_asym_down_tau_s", 20.0))
        up_tau = float(getattr(cfg, "v_corr_asym_up_tau_s", 160.0))
        v_corr = causal_asymmetric_ema(v_ohm_removed, times, down_tau, up_tau)
    elif v_corr_variant in {"low_voltage_blend_orig_asym", "lvblend_orig_asym_v3p35_s0p15"}:
        down_tau = float(getattr(cfg, "v_corr_asym_down_tau_s", 20.0))
        up_tau = float(getattr(cfg, "v_corr_asym_up_tau_s", 160.0))
        gate_v = float(getattr(cfg, "v_corr_lv_gate_v", 3.35))
        gate_s = max(float(getattr(cfg, "v_corr_lv_gate_s", 0.15)), 1e-6)
        v_corr_base = causal_ema(v_ohm_removed, times, cfg.v_corr_tau_s).astype(np.float64)
        v_corr_tail = causal_asymmetric_ema(v_ohm_removed, times, down_tau, up_tau).astype(np.float64)
        tail_gate = 1.0 / (1.0 + np.exp(-np.clip((gate_v - v_corr_base) / gate_s, -60.0, 60.0)))
        v_corr = ((1.0 - tail_gate) * v_corr_base + tail_gate * v_corr_tail).astype(np.float32)
    elif _ocv_calibrated_variant(v_corr_variant):
        if not ocv_calibration:
            raise RuntimeError("OCV-calibrated V_corr requested, but no train-only calibration was supplied.")
        params = ocv_calibration.get("params", {})
        key = float(temp)
        if key not in params:
            key = min(params, key=lambda k: abs(float(k) - float(temp)))
        p = params[key]
        e_fast = causal_ema(i_raw, times, float(cfg.v_corr_ocv_tau_fast_s)).astype(np.float64)
        e_slow = causal_ema(i_raw, times, float(cfg.v_corr_ocv_tau_slow_s)).astype(np.float64)
        dynamic_fit = float(p.get("intercept_v", 0.0)) + float(p.get("a_fast_ohm", 0.0)) * e_fast + float(p.get("a_slow_ohm", 0.0)) * e_slow
        v_corr = (v_ohm_removed - dynamic_fit).astype(np.float32)
    elif _dynamic_r_variant(v_corr_variant):
        if not dynamic_r_calibration:
            raise RuntimeError("Dynamic-R V_corr requested, but no train-only dynamic-R calibration was supplied.")
        r_eff = apply_dynamic_r(v_raw, i_raw, d_i.astype(np.float64), times, float(temp), r0, dynamic_r_calibration, cfg).astype(np.float64)
        r_output = r_eff.astype(np.float32)
        v_ohm = (i_raw * r_eff).astype(np.float32)
        v_ohm_removed = v_raw - v_ohm.astype(np.float64)
        v_corr = causal_ema(v_ohm_removed, times, float(getattr(cfg, "v_corr_tau_s", 10.0)))
    elif _rtvar_lowv_variant(v_corr_variant):
        if not rtvar_lowv_calibration:
            raise RuntimeError("RT-variance V_corr requested, but no train-only RT-variance calibration was supplied.")
        v_corr, r_eff = apply_rtvar_lowv_vcorr(v_raw, i_raw, times, float(temp), r0, rtvar_lowv_calibration, cfg)
        r_eff = np.asarray(r_eff, dtype=np.float64)
        r_output = r_eff.astype(np.float32)
        v_ohm = (i_raw * r_eff).astype(np.float32)
        v_ohm_removed = v_raw - v_ohm.astype(np.float64)
    elif _lfp_style_variant(v_corr_variant):
        if not lfp_style_calibration:
            raise RuntimeError("LFP-style V_corr requested, but no train-only LFP-style calibration was supplied.")
        v_corr, lfp_parts = apply_lfp_style_vcorr(v_raw, i_raw, times, float(temp), lfp_style_calibration, cfg)
        r_eff = np.asarray(lfp_parts["R0"], dtype=np.float64)
        r_output = r_eff.astype(np.float32)
        v_ohm = (i_raw * r_eff).astype(np.float32)
        v_ohm_removed = v_raw - v_ohm.astype(np.float64)
    elif _nmc_tailored_minimax_variant(v_corr_variant):
        if not nmc_tailored_calibration:
            raise RuntimeError("NMC-tailored minimax V_corr requested, but no train-only calibration was supplied.")
        v_corr, r0_tailored = apply_nmc_tailored_minimax_vcorr(
            v_raw,
            i_raw,
            times,
            float(temp),
            nmc_tailored_calibration,
        )
        r_output = np.full(len(df), float(r0_tailored), dtype=np.float32)
        v_ohm = (i_raw * float(r0_tailored)).astype(np.float32)
        v_ohm_removed = v_raw - v_ohm.astype(np.float64)
    elif v_corr_variant == "gated_r0_ema40_by_v_3p30_s0p15":
        gate_v = float(getattr(cfg, "v_corr_tail_gate_v", 3.30))
        gate_s = max(float(getattr(cfg, "v_corr_tail_gate_s", 0.15)), 1e-6)
        tail_tau = float(getattr(cfg, "v_corr_tail_tau_s", 40.0))
        tail_gate = 1.0 / (1.0 + np.exp(-np.clip((gate_v - v_raw) / gate_s, -60.0, 60.0)))
        v_tail_ohm_removed = v_raw - i_raw * r0 * (1.0 - tail_gate)
        v_corr = causal_ema(v_tail_ohm_removed, times, tail_tau)
    else:
        raise ValueError(f"Unknown v_corr_variant={v_corr_variant!r}")
    v_eq_slow = causal_ema(v_ohm_removed, times, cfg.v_pol_slow_tau_s)
    v_dyn_slow = (v_ohm_removed - v_eq_slow).astype(np.float32)
    dynamic = (v_raw - v_corr - v_ohm).astype(np.float32)
    dyn_mid_lp = causal_ema(dynamic, times, cfg.v_pol_mid_tau_s)
    dyn_slow = causal_ema(dynamic, times, cfg.v_pol_slow_tau_s)
    v_pol_fast = (dynamic - dyn_mid_lp).astype(np.float32)
    v_pol_mid = (dyn_mid_lp - dyn_slow).astype(np.float32)
    v_pol_slow = dyn_slow.astype(np.float32)
    v_hys = causal_ema(dynamic, times, cfg.v_hys_tau_s)

    soc = pd.to_numeric(df["SOC_CC"], errors="coerce").to_numpy(np.float32)
    if np.nanmax(soc) > 1.5:
        soc = soc / 100.0
    soc = np.clip(soc, 0.0, 1.0).astype(np.float32)
    trajectory_id = path.stem

    out = pd.DataFrame(
        {
            "file_name": path.name,
            "trajectory_id": trajectory_id,
            "temperature": float(temp),
            "drive_cycle": profile,
            "end_index": np.arange(len(df), dtype=np.int64),
            "SOC_physical": soc,
            "SOC_usable_cutoff": soc,
            "V_raw": v_raw.astype(np.float32),
            "V_corr_raw": v_corr.astype(np.float32),
            "V_ohm_free_raw": v_ohm_removed.astype(np.float32),
            "V_eq_slow_raw": v_eq_slow.astype(np.float32),
            "V_dyn_slow_raw": v_dyn_slow,
            "I_raw": i_raw.astype(np.float32),
            "T": np.full(len(df), float(temp), dtype=np.float32),
            "dI": d_i,
            "absI": abs_i,
            "V_pol_raw": dynamic.astype(np.float32),
            "V_hys_raw": v_hys.astype(np.float32),
            "V_ohm_raw": v_ohm.astype(np.float32),
            "R0": r_output,
            "V_pol_fast_raw": v_pol_fast,
            "V_pol_mid_raw": v_pol_mid,
            "V_pol_slow_raw": v_pol_slow,
        }
    )
    return out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def build_feature_frames(
    cfg: NMCBranchBandsConfig,
    files: list[Path],
    r0_df: pd.DataFrame,
    lfp_style_calibration: dict | None = None,
    nmc_tailored_calibration: dict | None = None,
    rtvar_lowv_calibration: dict | None = None,
) -> dict[str, list[pd.DataFrame]]:
    use_profile_r0 = "profile" in r0_df.columns and str(getattr(cfg, "r0_mode", "train_temperature")) == "profile_observed"
    r0_lookup = {float(r["temperature_C"]): float(r["r0_ohm"]) for _, r in r0_df.iterrows()}
    r0_profile_lookup = {
        (float(r["temperature_C"]), str(r["profile"])): float(r["r0_ohm"])
        for _, r in r0_df.iterrows()
        if "profile" in r
    }
    ocv_calibration = None
    if _ocv_calibrated_variant(str(getattr(cfg, "v_corr_variant", "orig_ohm_ema120"))):
        if use_profile_r0:
            raise RuntimeError("OCV-calibrated V_corr currently expects train-temperature R0, not profile_observed R0.")
        ocv_calibration = estimate_vcorr_ocv_calibration(files, tuple(cfg.train_profiles), r0_df, cfg)
    dynamic_r_calibration = None
    if _dynamic_r_variant(str(getattr(cfg, "v_corr_variant", "orig_ohm_ema120"))):
        if use_profile_r0:
            raise RuntimeError("Dynamic-R V_corr currently expects train-temperature R0, not profile_observed R0.")
        dynamic_r_calibration = estimate_dynamic_r_calibration(files, tuple(cfg.train_profiles), r0_df, cfg)
    if _rtvar_lowv_variant(str(getattr(cfg, "v_corr_variant", "orig_ohm_ema120"))):
        if use_profile_r0:
            raise RuntimeError("RT-variance V_corr currently expects train-temperature R0, not profile_observed R0.")
        if rtvar_lowv_calibration is None:
            rtvar_lowv_calibration = estimate_rtvar_lowv_calibration(files, tuple(cfg.train_profiles), r0_df, cfg)
    if _lfp_style_variant(str(getattr(cfg, "v_corr_variant", "orig_ohm_ema120"))):
        if use_profile_r0:
            raise RuntimeError("LFP-style V_corr currently expects train-temperature R0, not profile_observed R0.")
        if lfp_style_calibration is None:
            lfp_style_calibration = estimate_lfp_style_vcorr_calibration(files, tuple(cfg.train_profiles), r0_df, cfg)
    if _nmc_tailored_minimax_variant(str(getattr(cfg, "v_corr_variant", "orig_ohm_ema120"))):
        if use_profile_r0:
            raise RuntimeError("NMC-tailored minimax V_corr currently expects train-temperature R0, not profile_observed R0.")
        if nmc_tailored_calibration is None:
            nmc_tailored_calibration = estimate_nmc_tailored_minimax_calibration(files, tuple(cfg.train_profiles), r0_df, cfg)
    frames = {"train": [], "valid": [], "test": []}
    valid_profiles = tuple(getattr(cfg, "valid_profiles", ()))
    for p in files:
        head = pd.read_csv(p, nrows=2)
        profile = parse_profile(p, head)
        if use_profile_r0:
            temp = parse_temperature(p, head)
            key = (float(temp), str(profile))
            if key not in r0_profile_lookup:
                raise RuntimeError(f"Missing profile-local R0 for temperature/profile {key}.")
            frame = build_decomposed_frame(
                p,
                {float(temp): float(r0_profile_lookup[key])},
                cfg,
                ocv_calibration,
                dynamic_r_calibration,
                lfp_style_calibration,
                nmc_tailored_calibration,
                rtvar_lowv_calibration,
            )
        else:
            frame = build_decomposed_frame(
                p,
                r0_lookup,
                cfg,
                ocv_calibration,
                dynamic_r_calibration,
                lfp_style_calibration,
                nmc_tailored_calibration,
                rtvar_lowv_calibration,
            )
        if profile in cfg.train_profiles:
            frames["train"].append(frame)
        elif profile in valid_profiles:
            frames["valid"].append(frame)
        elif profile in cfg.test_profiles:
            frames["test"].append(frame)
    frames = add_derived_features(frames)
    return frames


def write_input_schema(feature_cols: list[str], cfg: NMCBranchBandsConfig, out_path: Path) -> pd.DataFrame:
    rows = []
    for idx, col in enumerate(feature_cols, start=1):
        if col.startswith("V_residual"):
            source = "causal voltage residual band"
        elif col.startswith("V_pol_") or col == "V_pol_raw":
            source = "causal dynamic voltage proxy"
        elif col in {"V_raw", "I_raw", "T", "dI", "absI"}:
            source = "instantaneous measured excitation/temperature"
        elif col in {"V_corr_raw", "V_ohm_raw", "R0", "V_hys_raw"}:
            source = "causal voltage decomposition proxy"
        else:
            source = "derived interaction of selected NoCC inputs"
        rows.append(
            {
                "index_1based": idx,
                "feature_name": col,
                "source": source,
                "uses_soc_input": False,
                "uses_cumulative_input": False,
                "uses_explicit_current_integration": False,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def write_leakage_audit(
    feature_cols: list[str],
    source_columns: list[str],
    cfg: NMCBranchBandsConfig,
    out_path: Path,
) -> pd.DataFrame:
    selected_bad = [c for c in feature_cols if any(tok in c for tok in FORBIDDEN_INPUT_PATTERNS)]
    source_forbidden = [c for c in source_columns if any(tok in c for tok in FORBIDDEN_INPUT_PATTERNS)]
    rows = [
        {
            "audit_item": "selected_input_columns_forbidden_name_scan",
            "status": "PASS" if not selected_bad else "FAIL",
            "detail": ",".join(selected_bad) if selected_bad else "No SOC/cumulative/time/progress/capacity columns selected.",
        },
        {
            "audit_item": "source_has_forbidden_columns_but_not_selected",
            "status": "PASS",
            "detail": ",".join(source_forbidden),
        },
        {
            "audit_item": "explicit_soc_state_update",
            "status": "PASS",
            "detail": "No SOC_{t+1}=SOC_t-I*dt/Q update exists in this model; SOC_CC is label only.",
        },
        {
            "audit_item": "current_usage",
            "status": "PASS",
            "detail": "Current is used as instantaneous excitation and in causal voltage decomposition proxies, not as cumulative Ah or SOC state integration.",
        },
        {
            "audit_item": "window_relative_time",
            "status": "PASS" if cfg.window_feature_mode == "delta_start_time" else "WARN",
            "detail": "The appended time feature is linspace(0,1) inside each window; absolute Test_Time/Step_Time is not an input.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    if selected_bad:
        raise RuntimeError(f"CUMULATIVE_FEATURE_LEAK: forbidden selected input columns: {selected_bad}")
    return out


def make_endpoint_feature_lookup(frames: dict[str, list[pd.DataFrame]], cols: list[str]) -> pd.DataFrame:
    keep = ["trajectory_id", "end_index", "temperature", "drive_cycle", "SOC_physical", "SOC_usable_cutoff"] + list(cols)
    rows = []
    for split, split_frames in frames.items():
        for frame in split_frames:
            have = [c for c in keep if c in frame.columns]
            rows.append(frame[have].assign(feature_split=split))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=keep + ["feature_split"])


def add_extra_prediction_features(pred: pd.DataFrame, lookup: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if pred.empty or lookup.empty:
        return pred
    keep = ["trajectory_id", "end_index"] + [c for c in cols if c in lookup.columns]
    # Keep extra inputs that the generic attachment helper does not preserve.
    keep += [c for c in ["I_raw", "dI", "absI", "T", "V_residual_low", "V_residual_mid", "V_residual_high"] if c in lookup.columns and c not in keep]
    extra = lookup[keep].drop_duplicates(["trajectory_id", "end_index"])
    overlap = [c for c in extra.columns if c not in {"trajectory_id", "end_index"} and c in pred.columns]
    extra = extra.drop(columns=overlap)
    return pred.merge(extra, on=["trajectory_id", "end_index"], how="left", validate="many_to_one")


def focus_metrics(pred: pd.DataFrame, cfg: NMCBranchBandsConfig, model_name: str) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    rows = []
    scopes = {
        "overall": np.ones(len(pred), dtype=bool),
        "plateau_20_80": (pred["y_true"] >= 0.2).to_numpy() & (pred["y_true"] <= 0.8).to_numpy(),
        "low_current": pred["absI"].abs().to_numpy(np.float64) < float(cfg.low_current_threshold_A)
        if "absI" in pred.columns
        else np.zeros(len(pred), dtype=bool),
        "catastrophic_gt5_denominator": np.ones(len(pred), dtype=bool),
    }
    for scope, mask in scopes.items():
        g = pred.loc[mask]
        if g.empty:
            continue
        err = g["error"].to_numpy(np.float32)
        row = {
            "model_name": model_name,
            "scope": scope,
            "n_windows": int(len(g)),
            "MAE_pct": float(g["abs_error"].mean() * 100.0),
            "RMSE_pct": float(np.sqrt(np.mean(err**2)) * 100.0),
            "catastrophic_gt5_pct": float((g["abs_error"] > 0.05).mean() * 100.0),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_by_trajectory(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, temp, drive, tid), g in pred.groupby(["model_name", "temperature_C", "drive_cycle", "trajectory_id"]):
        err = g["error"].to_numpy(np.float32)
        rows.append(
            {
                "model_name": model,
                "temperature_C": float(temp),
                "drive_cycle": drive,
                "trajectory_id": tid,
                "n_windows": int(len(g)),
                "MAE_pct": float(g["abs_error"].mean() * 100.0),
                "RMSE_pct": float(np.sqrt(np.mean(err**2)) * 100.0),
                "catastrophic_gt5_pct": float((g["abs_error"] > 0.05).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows).sort_values(["temperature_C", "drive_cycle"]).reset_index(drop=True)


def table_md(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "(empty)"
    sub = df[columns].copy()
    for col in sub.columns:
        if pd.api.types.is_float_dtype(sub[col]):
            sub[col] = sub[col].map(lambda x: "" if not np.isfinite(x) else f"{x:.4f}")
    header = "| " + " | ".join(sub.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(sub.columns)) + " |"
    lines = [header, sep]
    for _, row in sub.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in sub.columns) + " |")
    return "\n".join(lines)


def write_report(
    cfg: NMCBranchBandsConfig,
    out_dir: Path,
    start_audit: pd.DataFrame,
    r0_df: pd.DataFrame,
    schema: pd.DataFrame,
    leakage: pd.DataFrame,
    overall: pd.DataFrame,
    by_temp: pd.DataFrame,
    by_traj: pd.DataFrame,
    focus: pd.DataFrame,
) -> None:
    lines = [
        "# NMC BranchBands TCN NoCC 결과",
        "",
        "## 설정",
        f"- Raw data: `{cfg.raw_root}`",
        f"- Train profiles: {', '.join(cfg.train_profiles)}",
        f"- Test profiles: {', '.join(cfg.test_profiles)}",
        "- Train temperatures: 0, 25, 45 C all included",
        f"- Model: BranchBands TCN, window={cfg.window_len}, stride={cfg.stride}, hidden={cfg.hidden_size}, layers={cfg.layers}, kernel={cfg.kernel_size}",
        f"- Objective: sequence Huber beta={cfg.huber_beta}, REx={cfg.lambda_rex} grouped by `{cfg.rex_group}`",
        "- Strict NoCC: SOC input 없음, cumulative Ah/progress/time 입력 없음, explicit SOC current-integration state update 없음",
        "- Current usage: instantaneous excitation and causal voltage-decomposition proxy only",
        "",
        "## SOC 80% 시작점 감사",
        "CSV는 파일별 원래 시작 `Data_Point/Test_Time`이 다르지만, 각 파일의 첫 row가 이미 drive step 시작이며 `SOC_CC=0.8`로 정렬된 상태다.",
        "",
        table_md(
            start_audit,
            [
                "file_name",
                "temperature_C",
                "profile",
                "first_data_point",
                "first_test_time_s",
                "first_step_index",
                "first_soc_cc",
                "qnet_denom_Ah",
                "starts_at_80pct",
            ],
        ),
        "",
        "## 전압 분해",
        "- R0는 test profile(FUDS)을 제외하고 train profiles에서 온도별 전류 step의 robust dV/dI median으로 추정했다.",
        "- `V_ohm_raw = I_raw * R0`.",
        "- `V_corr_raw`는 ohmic 제거 전압의 causal EMA proxy다.",
        "- `V_pol_raw`, `V_pol_fast_raw`, `V_pol_mid_raw`, `V_pol_slow_raw`, `V_hys_raw`는 현재/과거 전압-전류만으로 만든 causal dynamic-response proxy다.",
        "- 이 분해는 label-free preprocessing이며 physical ECM parameter fitting 결과라고 주장하지 않는다.",
        "",
        table_md(r0_df, ["temperature_C", "r0_ohm", "n_events", "r0_p20_ohm", "r0_p80_ohm"]),
        "",
        "## 입력 스키마",
        f"- Base selected features: {len(schema)}",
        f"- TCN actual input dimension: {len(schema)} raw + {len(schema)} delta-from-window-start + 1 window-local relative position = {len(schema) * 2 + 1}",
        "",
        table_md(schema, ["index_1based", "feature_name", "source"]),
        "",
        "## 누수 감사",
        table_md(leakage, ["audit_item", "status", "detail"]),
        "",
        "## 성능",
        "SOC 값은 0-1 scale에서 학습했고 아래 MAE/RMSE는 %-point로 표시했다.",
        "",
        "### Overall",
        table_md(overall, ["model_name", "n_windows", "MAE_pct", "RMSE_pct", "error_std_pct"]),
        "",
        "### By Temperature",
        table_md(by_temp, ["temperature_C", "n_windows", "MAE_pct", "RMSE_pct", "jitter_ratio"]),
        "",
        "### By Trajectory",
        table_md(by_traj, ["temperature_C", "drive_cycle", "trajectory_id", "n_windows", "MAE_pct", "RMSE_pct", "catastrophic_gt5_pct"]),
        "",
        "### Focus",
        table_md(focus, ["scope", "n_windows", "MAE_pct", "RMSE_pct", "catastrophic_gt5_pct"]),
        "",
        "## 해석 주의",
        "- 이 결과는 NMC에서 온도 0/25/45 C를 모두 학습에 포함하고 FUDS profile을 holdout한 결과다.",
        "- NoCC가 current integration 없이도 가능한지 보는 ablation이지, current integration이 불필요하다는 증명은 아니다.",
        "- current는 사용하지 않은 것이 아니라 instantaneous excitation으로만 사용했고 SOC state로 적산하지 않았다.",
    ]
    (out_dir / f"{cfg.output_prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(cfg: NMCBranchBandsConfig) -> dict[str, pd.DataFrame]:
    cfg.base_dir = Path(cfg.base_dir).resolve()
    if not Path(cfg.raw_root).is_absolute():
        cfg.raw_root = cfg.base_dir / cfg.raw_root
    out_dir = cfg.base_dir / "nmc_branchbands_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_torch_runtime()
    set_seed(cfg.seed)

    files = find_csv_files(cfg.raw_root)
    start_audit = write_start_audit(files, out_dir / f"{cfg.output_prefix}_file_start_audit.csv")
    raw_source_columns = list(pd.read_csv(files[0], nrows=1).columns)

    r0_df = estimate_r0_by_temperature(files, cfg.train_profiles)
    r0_df.to_csv(out_dir / f"{cfg.output_prefix}_decomposition_params.csv", index=False)

    frames = build_feature_frames(cfg, files, r0_df)
    feature_cols = feature_columns("branch_bands")
    available = set().union(*(set(f.columns) for split in frames.values() for f in split))
    missing = [c for c in feature_cols if c not in available]
    if missing:
        raise KeyError(f"Missing NMC BranchBands feature columns: {missing}")
    schema = write_input_schema(feature_cols, cfg, out_dir / f"{cfg.output_prefix}_input_schema.csv")
    leakage = write_leakage_audit(feature_cols, raw_source_columns, cfg, out_dir / f"{cfg.output_prefix}_leakage_audit.csv")

    scaled, _scaler = make_scaled_frames_for_ablation(frames, feature_cols)
    base_cfg = make_cfg()
    base_cfg.output_dir = out_dir
    base_cfg.batch_size = int(cfg.batch_size)
    base_cfg.dataloader_num_workers = int(cfg.num_workers)
    base_cfg.dataloader_prefetch_factor = int(cfg.prefetch_factor)
    base_cfg.dataloader_pin_memory = True
    base_cfg.dataloader_persistent_workers = int(cfg.num_workers) > 0

    train_ds = AugmentedSequenceWindowDataset(
        scaled["train"],
        feature_cols,
        cfg.window_len,
        cfg.stride,
        target_label="physical",
        window_feature_mode=cfg.window_feature_mode,
    )
    test_ds = AugmentedWindowDataset(
        scaled["test"],
        feature_cols,
        cfg.window_len,
        1,
        target_label="physical",
        window_feature_mode=cfg.window_feature_mode,
    )
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(f"Empty dataset: train_windows={len(train_ds)} test_windows={len(test_ds)}")

    input_dim = augmented_input_dim(len(feature_cols), cfg.window_feature_mode)
    model_name = cfg.output_prefix
    model = DeepNoLeakTCN(
        input_dim=input_dim,
        hidden_size=cfg.hidden_size,
        layers=cfg.layers,
        kernel_size=cfg.kernel_size,
        norm_kind=cfg.norm_kind,
        dropout=cfg.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    train_loader = temperature_balanced_loader(train_ds, base_cfg, shuffle=True)
    test_loader = make_eval_loader(test_ds, cfg)

    history = []
    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        losses = []
        mean_losses = []
        rex_losses = []
        smooth_losses = []
        by_group_all: dict[str, list[float]] = {}
        for x, y, meta in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            y = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            pred = model.forward_sequence(x)
            loss, mean_loss, rex_loss, smooth_loss, by_group = temp_balanced_rex_loss(
                pred,
                y,
                meta,
                cfg.lambda_rex,
                cfg.rex_group,
                cfg.loss_kind,
                cfg.huber_beta,
                cfg.lambda_smooth,
                cfg.endpoint_loss_weight,
                cfg.lambda_worst,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            mean_losses.append(float(mean_loss.detach().cpu()))
            rex_losses.append(float(rex_loss.detach().cpu()))
            smooth_losses.append(float(smooth_loss.detach().cpu()))
            for key, val in by_group.items():
                by_group_all.setdefault(key, []).append(float(val))
        row = {
            "model_name": model_name,
            "epoch": ep,
            "loss": float(np.mean(losses)),
            "mean_loss": float(np.mean(mean_losses)),
            "rex_var": float(np.mean(rex_losses)),
            "smooth_delta_loss": float(np.mean(smooth_losses)),
        }
        for key, vals in by_group_all.items():
            safe = str(key).replace(".", "p").replace("-", "N")
            row[f"train_loss_group_{safe}"] = float(np.mean(vals))
        history.append(row)
        if ep == 1 or ep == int(cfg.epochs) or ep % max(1, int(cfg.print_every)) == 0:
            print(
                f"{model_name} epoch={ep} loss={row['loss']:.5f} mean={row['mean_loss']:.5f} rex={row['rex_var']:.6f}",
                flush=True,
            )

    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / f"{cfg.output_prefix}_history.csv", index=False)

    pred = predict(model, test_loader)
    generic_lookup = build_prediction_feature_lookup(frames)
    pred = attach_prediction_features(
        pred.assign(split="test", ablation=model_name),
        generic_lookup,
        ablation_name=model_name,
        target_label="physical",
    )
    endpoint_lookup = make_endpoint_feature_lookup(frames, feature_cols)
    pred = add_extra_prediction_features(pred, endpoint_lookup, feature_cols)
    pred["seed"] = int(cfg.seed)
    pred["train_profiles"] = ",".join(cfg.train_profiles)
    pred["test_profiles"] = ",".join(cfg.test_profiles)
    pred.to_csv(out_dir / f"{cfg.output_prefix}_prediction_rows.csv.gz", index=False, compression="gzip")

    overall = _overall_metrics(pred)
    by_temp = variance_by_temperature(pred)
    by_traj = metrics_by_trajectory(pred)
    focus = focus_metrics(pred, cfg, model_name)
    overall.to_csv(out_dir / f"{cfg.output_prefix}_overall.csv", index=False)
    by_temp.to_csv(out_dir / f"{cfg.output_prefix}_by_temperature.csv", index=False)
    by_traj.to_csv(out_dir / f"{cfg.output_prefix}_by_trajectory.csv", index=False)
    focus.to_csv(out_dir / f"{cfg.output_prefix}_focus.csv", index=False)

    metadata = {
        **asdict(cfg),
        "base_dir": str(cfg.base_dir),
        "raw_root": str(cfg.raw_root),
        "model_name": model_name,
        "feature_set": "branch_bands",
        "feature_columns": feature_cols,
        "base_feature_dim": len(feature_cols),
        "input_feature_dim": int(input_dim),
        "input_dim_explanation": f"{len(feature_cols)} raw + {len(feature_cols)} delta-from-window-start + 1 window-local relative position",
        "uses_soc_input": False,
        "uses_cumulative_input": False,
        "uses_explicit_current_integration": False,
        "label_column": "SOC_CC",
        "train_windows": int(len(train_ds)),
        "test_windows": int(len(test_ds)),
        "train_trajectories": int(len(frames["train"])),
        "test_trajectories": int(len(frames["test"])),
    }
    (out_dir / f"{cfg.output_prefix}_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    write_report(cfg, out_dir, start_audit, r0_df, schema, leakage, overall, by_temp, by_traj, focus)

    print("Overall:")
    print(overall.to_string(index=False), flush=True)
    print("By temperature:")
    print(by_temp.to_string(index=False), flush=True)
    print(f"Report: {out_dir / (cfg.output_prefix + '_report.md')}", flush=True)
    return {
        "history": history_df,
        "pred": pred,
        "overall": overall,
        "by_temperature": by_temp,
        "by_trajectory": by_traj,
        "focus": focus,
        "start_audit": start_audit,
        "r0": r0_df,
        "schema": schema,
        "leakage": leakage,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NMC Samsung INR 18650 2Ah BranchBands TCN NoCC experiment.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--raw-root", default="nmc_samsung_inr_18650_2ah_raw/NMC SAMSUNG INR 18650 2Ah")
    p.add_argument("--output-prefix", default=NMCBranchBandsConfig.output_prefix)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-profiles", default="DST,US06")
    p.add_argument("--test-profiles", default="FUDS")
    p.add_argument("--window-len", type=int, default=150)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.04)
    p.add_argument("--lambda-rex", type=float, default=2.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--print-every", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = NMCBranchBandsConfig(
        base_dir=Path(args.base_dir),
        raw_root=Path(args.raw_root),
        output_prefix=str(args.output_prefix),
        seed=int(args.seed),
        train_profiles=tuple(s.strip() for s in str(args.train_profiles).split(",") if s.strip()),
        test_profiles=tuple(s.strip() for s in str(args.test_profiles).split(",") if s.strip()),
        window_len=int(args.window_len),
        stride=int(args.stride),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        hidden_size=int(args.hidden_size),
        layers=int(args.layers),
        kernel_size=int(args.kernel_size),
        dropout=float(args.dropout),
        lambda_rex=float(args.lambda_rex),
        num_workers=int(args.num_workers),
        print_every=int(args.print_every),
    )
    run(cfg)


if __name__ == "__main__":
    main()
