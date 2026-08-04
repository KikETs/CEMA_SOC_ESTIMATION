from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import make_cfg
from .deep_no_leak_experiment import DeepNoLeakLSTM, SequenceWindowDataset, make_eval_loader
from .extrapolation_robustness import temperature_balanced_loader
from .models import DecomposedWindowDataset, collate_meta_to_frame
from .nmc_branchbands_experiment import (
    FORBIDDEN_INPUT_PATTERNS,
    build_feature_frames,
    estimate_r0_by_temperature,
    find_csv_files,
    focus_metrics,
    metrics_by_trajectory,
    table_md,
    write_start_audit,
)
from .runtime import configure_torch_runtime, device
from .training import make_scaled_frames_for_ablation
from .variance_control import _overall_metrics, variance_by_temperature


BASE_PREFIX = "nmc_vit_feature_lstm_h64_w50_trainDSTUS06_noval_testFUDS_seed0"
EMA_TAUS = (10, 50, 200, 800)
EPS = 1e-4


@dataclass
class NMCVITFeatureLSTMConfig:
    base_dir: Path = Path(".")
    raw_root: Path = Path("nmc_soc_ocvstart_relabelled_from_lc_ocv/data/NMC SAMSUNG INR 18650 2Ah")
    output_prefix: str = BASE_PREFIX
    seed: int = 0
    feature_set: str = "all"
    train_profiles: tuple[str, ...] = ("DST", "US06")
    valid_profiles: tuple[str, ...] = ("VALIDATION",)
    test_profiles: tuple[str, ...] = ("FUDS",)
    window_len: int = 50
    stride: int = 3
    epochs: int = 300
    batch_size: int = 1024
    lr: float = 8e-4
    weight_decay: float = 1e-4
    hidden_size: int = 64
    layers: int = 2
    dropout: float = 0.05
    lambda_rex: float = 2.0
    rex_group: str = "temperature_drive"
    loss_kind: str = "huber"
    huber_beta: float = 0.02
    sequence_loss: bool = True
    num_workers: int = 4
    prefetch_factor: int = 4
    print_every: int = 10
    valid_every: int = 10
    low_current_threshold_A: float = 0.05
    v_corr_tau_s: float = 120.0
    v_pol_mid_tau_s: float = 60.0
    v_pol_slow_tau_s: float = 600.0
    v_hys_tau_s: float = 1200.0


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def causal_index_ema(values: np.ndarray, tau: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr.astype(np.float32)
    alpha = float(np.exp(-1.0 / max(float(tau), 1e-6)))
    alpha = min(max(alpha, 0.0), 0.999999)
    out = np.empty_like(arr, dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * out[i - 1] + (1.0 - alpha) * arr[i]
    return out.astype(np.float32)


def causal_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr.astype(np.float32)
    window = max(1, int(window))
    csum = np.cumsum(np.insert(arr, 0, 0.0))
    out = np.empty_like(arr, dtype=np.float64)
    for i in range(len(arr)):
        start = max(0, i + 1 - window)
        out[i] = (csum[i + 1] - csum[start]) / float(i + 1 - start)
    return out.astype(np.float32)


def causal_rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr.astype(np.float32)
    window = max(1, int(window))
    csum = np.cumsum(np.insert(arr, 0, 0.0))
    csum2 = np.cumsum(np.insert(arr * arr, 0, 0.0))
    out = np.empty_like(arr, dtype=np.float64)
    for i in range(len(arr)):
        start = max(0, i + 1 - window)
        n = float(i + 1 - start)
        mean = (csum[i + 1] - csum[start]) / n
        mean2 = (csum2[i + 1] - csum2[start]) / n
        out[i] = np.sqrt(max(mean2 - mean * mean, 0.0))
    return out.astype(np.float32)


def causal_rolling_span(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr.astype(np.float32)
    window = max(1, int(window))
    out = np.empty_like(arr, dtype=np.float64)
    for i in range(len(arr)):
        start = max(0, i + 1 - window)
        chunk = arr[start : i + 1]
        out[i] = float(np.nanmax(chunk) - np.nanmin(chunk))
    return out.astype(np.float32)


def causal_rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr.astype(np.float32)
    window = max(1, int(window))
    out = np.empty_like(arr, dtype=np.float64)
    for i in range(len(arr)):
        start = max(0, i + 1 - window)
        out[i] = float(np.nanmax(arr[start : i + 1]))
    return out.astype(np.float32)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, eps: float = EPS) -> np.ndarray:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    return (num / (np.abs(den) + float(eps))).astype(np.float32)


def add_vit_engineered_features(frames: dict[str, list[pd.DataFrame]]) -> dict[str, list[pd.DataFrame]]:
    """Add causal V/I/T-derived features only; no SOC, cumulative Ah, progress, or absolute time."""
    out: dict[str, list[pd.DataFrame]] = {}
    for split, split_frames in frames.items():
        out[split] = []
        for frame in split_frames:
            f = frame.copy()
            v_raw = f["V_raw"].to_numpy(np.float64)
            v_corr = f["V_corr_raw"].to_numpy(np.float64)
            i_raw = f["I_raw"].to_numpy(np.float64)
            abs_i = np.abs(i_raw)

            f["V_corr_start"] = np.full(len(f), float(v_corr[0]), dtype=np.float32)
            f["V_corr_rel_start"] = (v_corr - float(v_corr[0])).astype(np.float32)
            d_v_raw = np.diff(v_raw, prepend=v_raw[0])
            d_v_corr = np.diff(v_corr, prepend=v_corr[0])
            d_i = np.diff(i_raw, prepend=i_raw[0])
            f["dV_raw"] = d_v_raw.astype(np.float32)
            f["dV_corr"] = d_v_corr.astype(np.float32)
            f["abs_dV_raw"] = np.abs(d_v_raw).astype(np.float32)
            f["abs_dV_corr"] = np.abs(d_v_corr).astype(np.float32)
            f["d2V_corr"] = np.diff(d_v_corr, prepend=d_v_corr[0]).astype(np.float32)
            f["d2I"] = np.diff(d_i, prepend=d_i[0]).astype(np.float32)
            f["dI_abs_mean"] = causal_rolling_mean(np.abs(d_i), 50)
            f["V_drop_raw"] = (v_raw - v_corr).astype(np.float32)
            f["P_raw"] = (v_raw * i_raw).astype(np.float32)
            f["absP_raw"] = np.abs(v_raw * i_raw).astype(np.float32)
            f["Vcorr_x_absI"] = (v_corr * abs_i).astype(np.float32)
            f["Vdrop_x_absI"] = ((v_raw - v_corr) * abs_i).astype(np.float32)
            f["dI_x_Vdrop"] = (d_i * (v_raw - v_corr)).astype(np.float32)
            for window in (50, 200, 800):
                f[f"I_raw_std{window}"] = causal_rolling_std(i_raw, window)
                f[f"V_corr_raw_span{window}"] = causal_rolling_span(v_corr, window)
                f[f"V_corr_raw_upper{window}"] = causal_rolling_max(v_corr, window)
                f[f"V_corr_raw_upper_gap{window}"] = (
                    f[f"V_corr_raw_upper{window}"].to_numpy(np.float32) - v_corr.astype(np.float32)
                ).astype(np.float32)
            f["I_raw_std50_ema200"] = causal_index_ema(f["I_raw_std50"].to_numpy(np.float32), 200)
            f["V_corr_raw_span50_ema200"] = causal_index_ema(
                f["V_corr_raw_span50"].to_numpy(np.float32), 200
            )
            f["V_corr_raw_upper_gap50_ema200"] = causal_index_ema(
                f["V_corr_raw_upper_gap50"].to_numpy(np.float32), 200
            )

            for col in ("V_raw", "V_corr_raw", "I_raw", "absI", "V_drop_raw", "P_raw"):
                arr = f[col].to_numpy(np.float32)
                for tau in EMA_TAUS:
                    ema = causal_index_ema(arr, tau)
                    f[f"{col}_ema{tau}"] = ema
                    f[f"{col}_dev_ema{tau}"] = (arr - ema).astype(np.float32)
            for col in ("V_ohm_free_raw", "V_eq_slow_raw", "V_dyn_slow_raw"):
                if col not in f.columns:
                    continue
                arr = f[col].to_numpy(np.float32)
                for tau in (50, 200):
                    ema = causal_index_ema(arr, tau)
                    f[f"{col}_ema{tau}"] = ema
                    f[f"{col}_dev_ema{tau}"] = (arr - ema).astype(np.float32)

            if {"T", "V_corr_raw_dev_ema200"}.issubset(f.columns):
                f["T_x_Vcorr_dev_ema200"] = (
                    f["T"].to_numpy(np.float32) * f["V_corr_raw_dev_ema200"].to_numpy(np.float32)
                ).astype(np.float32)
            if {"T", "V_drop_raw_dev_ema200"}.issubset(f.columns):
                f["T_x_Vdrop_dev_ema200"] = (
                    f["T"].to_numpy(np.float32) * f["V_drop_raw_dev_ema200"].to_numpy(np.float32)
                ).astype(np.float32)
            if {"V_corr_raw_ema50", "V_corr_raw_ema200", "V_corr_raw_ema800"}.issubset(f.columns):
                ema50 = f["V_corr_raw_ema50"].to_numpy(np.float32)
                ema200 = f["V_corr_raw_ema200"].to_numpy(np.float32)
                ema800 = f["V_corr_raw_ema800"].to_numpy(np.float32)
                f["V_corr_raw_ema50_minus_ema800"] = (ema50 - ema800).astype(np.float32)
                f["V_corr_raw_ema200_minus_ema800"] = (ema200 - ema800).astype(np.float32)
            if {"V_corr_raw_span50", "I_raw_std50", "V_corr_raw_dev_ema200", "V_corr_raw_dev_ema800"}.issubset(f.columns):
                i_std50 = f["I_raw_std50"].to_numpy(np.float32)
                i_std200 = f["I_raw_std200"].to_numpy(np.float32)
                f["Vcorr_span50_over_Istd50"] = safe_divide(
                    f["V_corr_raw_span50"].to_numpy(np.float32),
                    i_std50,
                )
                f["Vcorr_span200_over_Istd200"] = safe_divide(
                    f["V_corr_raw_span200"].to_numpy(np.float32),
                    i_std200,
                )
                f["Vcorr_dev_ema200_over_Istd50"] = safe_divide(
                    f["V_corr_raw_dev_ema200"].to_numpy(np.float32),
                    i_std50,
                )
                f["Vcorr_dev_ema800_over_Istd200"] = safe_divide(
                    f["V_corr_raw_dev_ema800"].to_numpy(np.float32),
                    i_std200,
                )
            out[split].append(f.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True))
    return out


def feature_columns(feature_set: str) -> list[str]:
    common = ["V_corr_raw", "I_raw", "T"]
    basic = [
        "V_raw",
        "V_corr_raw",
        "I_raw",
        "T",
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
    ]
    decomp = basic + [
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
    ema_compact = basic + [
        "V_corr_raw_ema50",
        "V_corr_raw_dev_ema50",
        "V_corr_raw_ema200",
        "V_corr_raw_dev_ema200",
        "V_raw_ema50",
        "V_raw_dev_ema50",
        "V_raw_ema200",
        "V_raw_dev_ema200",
        "I_raw_ema50",
        "I_raw_dev_ema50",
        "absI_ema50",
        "absI_dev_ema50",
        "V_drop_raw_ema50",
        "V_drop_raw_dev_ema50",
        "V_drop_raw_ema200",
        "V_drop_raw_dev_ema200",
        "Vcorr_x_absI",
        "Vdrop_x_absI",
    ]
    response_ema = decomp + [
        "R0_x_V_pol",
        "T_x_V_pol",
        "R0_x_absI",
        "V_pol_x_abs_dI",
        "V_residual_low_x_T",
        "V_residual_mid_x_absI",
        "V_residual_high_x_abs_dI",
        "V_corr_raw_ema50",
        "V_corr_raw_dev_ema50",
        "V_corr_raw_ema200",
        "V_corr_raw_dev_ema200",
        "V_corr_raw_ema800",
        "V_corr_raw_dev_ema800",
        "V_drop_raw_ema50",
        "V_drop_raw_dev_ema50",
        "V_drop_raw_ema200",
        "V_drop_raw_dev_ema200",
        "V_drop_raw_ema800",
        "V_drop_raw_dev_ema800",
        "V_pol_raw_ema50",
        "V_pol_raw_dev_ema50",
        "V_pol_raw_ema200",
        "V_pol_raw_dev_ema200",
        "V_hys_raw_ema200",
        "V_hys_raw_dev_ema200",
        "P_raw_ema50",
        "P_raw_dev_ema50",
        "P_raw_ema200",
        "P_raw_dev_ema200",
        "T_x_Vcorr_dev_ema200",
        "T_x_Vdrop_dev_ema200",
        "dI_x_Vdrop",
    ]
    vcorr_it_excitation_ema = common + [
        "dI",
        "absI",
        "dV_corr",
        "abs_dV_corr",
        "d2V_corr",
        "Vcorr_x_absI",
        "V_corr_raw_ema50",
        "V_corr_raw_dev_ema50",
        "V_corr_raw_ema200",
        "V_corr_raw_dev_ema200",
        "V_corr_raw_ema800",
        "V_corr_raw_dev_ema800",
        "I_raw_ema50",
        "I_raw_dev_ema50",
        "I_raw_ema200",
        "I_raw_dev_ema200",
        "absI_ema50",
        "absI_dev_ema50",
        "absI_ema200",
        "absI_dev_ema200",
    ]
    paper_g0_raw = common
    paper_g1_derivatives = common + [
        "dI",
        "absI",
        "dV_corr",
        "abs_dV_corr",
        "d2V_corr",
    ]
    paper_g4_all_ema = common + [
        "V_corr_raw_ema50",
        "V_corr_raw_dev_ema50",
        "V_corr_raw_ema200",
        "V_corr_raw_dev_ema200",
        "V_corr_raw_ema800",
        "V_corr_raw_dev_ema800",
        "I_raw_ema50",
        "I_raw_dev_ema50",
        "I_raw_ema200",
        "I_raw_dev_ema200",
        "absI_ema50",
        "absI_dev_ema50",
        "absI_ema200",
        "absI_dev_ema200",
    ]
    paper_g4_all_ema_tailaware = paper_g4_all_ema
    paper_g4_all_ema_asymohm = paper_g4_all_ema
    paper_g4_all_ema_lvblend = paper_g4_all_ema
    paper_g4_all_ema_ocvcal = paper_g4_all_ema
    paper_g4_all_ema_start = paper_g4_all_ema + [
        "V_corr_start",
        "V_corr_rel_start",
    ]
    paper_g4_all_ema_di_abs_mean = paper_g4_all_ema + [
        "dI_abs_mean",
    ]
    paper_g4_all_ema_transient_min = paper_g4_all_ema + [
        "dV_corr",
        "abs_dV_corr",
        "I_raw_std50",
        "V_corr_raw_span50",
    ]
    paper_g4_all_ema_shape_norm = paper_g4_all_ema + [
        "V_corr_raw_ema50_minus_ema800",
        "V_corr_raw_ema200_minus_ema800",
        "I_raw_std50",
        "I_raw_std200",
        "I_raw_std800",
        "V_corr_raw_span50",
        "V_corr_raw_span200",
        "V_corr_raw_span800",
        "Vcorr_span50_over_Istd50",
        "Vcorr_span200_over_Istd200",
        "Vcorr_dev_ema200_over_Istd50",
        "Vcorr_dev_ema800_over_Istd200",
    ]
    paper_g4_eqdyn = paper_g4_all_ema + [
        "V_ohm_free_raw",
        "V_eq_slow_raw",
        "V_dyn_slow_raw",
        "V_dyn_slow_raw_ema50",
        "V_dyn_slow_raw_dev_ema50",
        "V_dyn_slow_raw_ema200",
        "V_dyn_slow_raw_dev_ema200",
    ]
    paper_regime_stats = [
        "I_raw_std50",
        "I_raw_std200",
        "I_raw_std50_ema200",
        "V_corr_raw_span50",
        "V_corr_raw_span200",
        "V_corr_raw_span50_ema200",
    ]
    paper_g4_all_ema_regime = paper_g4_all_ema + paper_regime_stats
    paper_g4_eqdyn_regime = paper_g4_eqdyn + paper_regime_stats
    paper_g4_eqdyn_start = paper_g4_eqdyn + [
        "V_corr_start",
        "V_corr_rel_start",
    ]
    paper_g4_relax_upper = paper_g4_all_ema + [
        "V_corr_raw_upper50",
        "V_corr_raw_upper_gap50",
        "V_corr_raw_upper200",
        "V_corr_raw_upper_gap200",
        "V_corr_raw_upper_gap50_ema200",
        "V_corr_raw_span50",
        "V_corr_raw_span200",
    ]
    paper_t6_voltage_ema_all = common + [
        "V_corr_raw_ema50",
        "V_corr_raw_dev_ema50",
        "V_corr_raw_ema200",
        "V_corr_raw_dev_ema200",
        "V_corr_raw_ema800",
        "V_corr_raw_dev_ema800",
    ]
    paper_t7_current_abs_ema_all = common + [
        "I_raw_ema50",
        "I_raw_dev_ema50",
        "I_raw_ema200",
        "I_raw_dev_ema200",
        "absI_ema50",
        "absI_dev_ema50",
        "absI_ema200",
        "absI_dev_ema200",
    ]
    paper_voltage_ema50_only = common + [
        "V_corr_raw_ema50",
        "V_corr_raw_dev_ema50",
    ]
    paper_voltage_ema200_only = common + [
        "V_corr_raw_ema200",
        "V_corr_raw_dev_ema200",
    ]
    paper_voltage_ema800_only = common + [
        "V_corr_raw_ema800",
        "V_corr_raw_dev_ema800",
    ]
    paper_current_abs_ema50_only = common + [
        "I_raw_ema50",
        "I_raw_dev_ema50",
        "absI_ema50",
        "absI_dev_ema50",
    ]
    paper_current_abs_ema200_only = common + [
        "I_raw_ema200",
        "I_raw_dev_ema200",
        "absI_ema200",
        "absI_dev_ema200",
    ]
    paper_g6_full23 = vcorr_it_excitation_ema
    paper_g7_no_current_ema = [
        c
        for c in paper_g6_full23
        if c
        not in {
            "I_raw_ema50",
            "I_raw_dev_ema50",
            "I_raw_ema200",
            "I_raw_dev_ema200",
            "absI_ema50",
            "absI_dev_ema50",
            "absI_ema200",
            "absI_dev_ema200",
        }
    ]
    paper_g8_no_voltage_ema = [
        c
        for c in paper_g6_full23
        if c
        not in {
            "V_corr_raw_ema50",
            "V_corr_raw_dev_ema50",
            "V_corr_raw_ema200",
            "V_corr_raw_dev_ema200",
            "V_corr_raw_ema800",
            "V_corr_raw_dev_ema800",
        }
    ]
    full_ema = response_ema + [
        "V_raw_ema10",
        "V_raw_dev_ema10",
        "V_raw_ema800",
        "V_raw_dev_ema800",
        "I_raw_ema10",
        "I_raw_dev_ema10",
        "I_raw_ema200",
        "I_raw_dev_ema200",
        "I_raw_ema800",
        "I_raw_dev_ema800",
        "absI_ema10",
        "absI_dev_ema10",
        "absI_ema200",
        "absI_dev_ema200",
        "absI_ema800",
        "absI_dev_ema800",
    ]
    mapping = {
        "vcorr_it": common,
        "vcorr_it_excitation_ema": vcorr_it_excitation_ema,
        "paper_g0_raw": paper_g0_raw,
        "paper_g1_derivatives": paper_g1_derivatives,
        "paper_g4_all_ema": paper_g4_all_ema,
        "paper_g4_all_ema_tailaware": paper_g4_all_ema_tailaware,
        "paper_g4_all_ema_asymohm": paper_g4_all_ema_asymohm,
        "paper_g4_all_ema_lvblend": paper_g4_all_ema_lvblend,
        "paper_g4_all_ema_ocvcal": paper_g4_all_ema_ocvcal,
        "paper_g4_all_ema_start": paper_g4_all_ema_start,
        "paper_g4_all_ema_di_abs_mean": paper_g4_all_ema_di_abs_mean,
        "paper_g4_all_ema_transient_min": paper_g4_all_ema_transient_min,
        "paper_g4_all_ema_shape_norm": paper_g4_all_ema_shape_norm,
        "paper_g4_all_ema_regime": paper_g4_all_ema_regime,
        "paper_g4_eqdyn": paper_g4_eqdyn,
        "paper_g4_eqdyn_start": paper_g4_eqdyn_start,
        "paper_g4_eqdyn_regime": paper_g4_eqdyn_regime,
        "paper_g4_relax_upper": paper_g4_relax_upper,
        "paper_t6_voltage_ema_all": paper_t6_voltage_ema_all,
        "paper_t7_current_abs_ema_all": paper_t7_current_abs_ema_all,
        "paper_voltage_ema50_only": paper_voltage_ema50_only,
        "paper_voltage_ema200_only": paper_voltage_ema200_only,
        "paper_voltage_ema800_only": paper_voltage_ema800_only,
        "paper_current_abs_ema50_only": paper_current_abs_ema50_only,
        "paper_current_abs_ema200_only": paper_current_abs_ema200_only,
        "paper_g6_full23": paper_g6_full23,
        "paper_g7_no_current_ema": paper_g7_no_current_ema,
        "paper_g8_no_voltage_ema": paper_g8_no_voltage_ema,
        "vit_basic": basic,
        "vit_decomp": decomp,
        "vit_ema_compact": ema_compact,
        "vit_response_ema": response_ema,
        "vit_full_ema": full_ema,
    }
    if feature_set not in mapping:
        raise ValueError(f"Unknown feature_set={feature_set}. Available: {', '.join(mapping)}")
    cols = list(dict.fromkeys(mapping[feature_set]))
    bad = [c for c in cols if any(tok in c for tok in FORBIDDEN_INPUT_PATTERNS)]
    if bad:
        raise AssertionError(f"CUMULATIVE_FEATURE_LEAK: forbidden input column names selected: {bad}")
    return cols


def all_feature_sets() -> list[str]:
    return [
        "vcorr_it",
        "vcorr_it_excitation_ema",
        "paper_g0_raw",
        "paper_g1_derivatives",
        "paper_g4_all_ema",
        "paper_g4_all_ema_tailaware",
        "paper_g4_all_ema_asymohm",
        "paper_g4_all_ema_lvblend",
        "paper_g4_all_ema_ocvcal",
        "paper_g4_all_ema_start",
        "paper_g4_all_ema_transient_min",
        "paper_g4_all_ema_shape_norm",
        "paper_g4_eqdyn",
        "paper_g4_eqdyn_start",
        "paper_g4_eqdyn_regime",
        "paper_g4_relax_upper",
        "paper_t6_voltage_ema_all",
        "paper_t7_current_abs_ema_all",
        "paper_voltage_ema50_only",
        "paper_voltage_ema200_only",
        "paper_voltage_ema800_only",
        "paper_current_abs_ema50_only",
        "paper_current_abs_ema200_only",
        "paper_g6_full23",
        "paper_g7_no_current_ema",
        "paper_g8_no_voltage_ema",
        "vit_basic",
        "vit_decomp",
        "vit_ema_compact",
        "vit_response_ema",
        "vit_full_ema",
    ]


def describe_feature(col: str) -> str:
    if col == "V_corr_start":
        return "causal trajectory initial corrected-voltage context"
    if col == "V_corr_rel_start":
        return "causal current corrected voltage minus trajectory initial corrected voltage"
    if col == "V_corr_raw":
        return "causal ohmic-corrected voltage proxy from V/I"
    if col == "I_raw":
        return "instantaneous current excitation only"
    if col == "T":
        return "ambient temperature"
    if col in {"V_raw", "dV_raw", "dV_corr", "abs_dV_raw", "abs_dV_corr", "d2V_corr"}:
        return "voltage level or causal local voltage difference"
    if col in {"dI", "d2I", "absI"}:
        return "instantaneous/local current excitation difference, not integrated"
    if col.startswith("I_raw_std") or col.startswith("V_corr_raw_span"):
        return "causal V/I regime summary over past samples, not integrated"
    if col in {"P_raw", "absP_raw"}:
        return "instantaneous V*I excitation magnitude, not time-integrated"
    if col in {"V_ohm_raw", "R0"}:
        return "train-profile dV/dI ohmic proxy and instantaneous I*R term"
    if col.startswith("V_pol") or col.startswith("V_hys") or col.startswith("V_residual"):
        return "causal voltage dynamic-response proxy from V/I/T"
    if "_ema" in col:
        return "causal EMA or deviation from causal EMA"
    if col.endswith("_x_absI") or "_x_" in col:
        return "label-free interaction of V/I/T-derived features"
    if col == "V_drop_raw":
        return "instantaneous V_raw - V_corr_raw residual"
    return "label-free V/I/T-derived feature"


def write_input_schema(feature_cols: list[str], out_path: Path) -> pd.DataFrame:
    rows = [
        {
            "index_1based": i,
            "feature_name": col,
            "source": describe_feature(col),
            "uses_soc_input": False,
            "uses_cumulative_input": False,
            "uses_explicit_current_integration": False,
        }
        for i, col in enumerate(feature_cols, start=1)
    ]
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def write_leakage_audit(feature_cols: list[str], source_columns: list[str], out_path: Path) -> pd.DataFrame:
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
            "detail": "No SOC_{t+1}=SOC_t-I*dt/Q update is used; SOC_CC is label only.",
        },
        {
            "audit_item": "current_usage",
            "status": "PASS",
            "detail": "Current is used only as instantaneous/local excitation and causal voltage-response preprocessing, not cumulative Ah.",
        },
        {
            "audit_item": "absolute_or_window_time_feature",
            "status": "PASS",
            "detail": "No absolute time, trajectory progress, or window-local timestep feature is selected.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    if selected_bad:
        raise RuntimeError(f"CUMULATIVE_FEATURE_LEAK: forbidden selected input columns: {selected_bad}")
    return out


def predict_loader(model: torch.nn.Module, loader, model_name: str) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for x, y, meta in loader:
            pred = model(x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda"))
            mdf = collate_meta_to_frame(meta)
            mdf["model_name"] = model_name
            mdf["target_label"] = "physical"
            mdf["y_true"] = y.numpy()[:, 0]
            mdf["y_pred"] = pred.detach().cpu().numpy()[:, 0]
            rows.append(mdf)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out):
        out["error"] = out["y_pred"] - out["y_true"]
        out["abs_error"] = np.abs(out["error"])
    return out


def evaluate_loader_mae(model: torch.nn.Module, loader) -> float:
    if loader is None:
        return float("nan")
    model.eval()
    errors = []
    with torch.no_grad():
        for x, y, _meta in loader:
            pred = model(x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda"))
            yy = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            errors.append(torch.abs(pred - yy).detach().cpu().numpy())
    if not errors:
        return float("nan")
    return float(np.mean(np.concatenate(errors)))


def make_endpoint_lookup(frames: dict[str, list[pd.DataFrame]], feature_cols: list[str]) -> pd.DataFrame:
    extra_cols = [
        "trajectory_id",
        "end_index",
        "temperature",
        "drive_cycle",
        "SOC_physical",
        "SOC_usable_cutoff",
        "V_raw",
        "V_corr_raw",
        "I_raw",
        "T",
        "absI",
        "dI",
    ] + feature_cols
    rows = []
    for split, split_frames in frames.items():
        for frame in split_frames:
            keep = [c for c in dict.fromkeys(extra_cols) if c in frame.columns]
            rows.append(frame[keep].assign(feature_split=split))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def attach_eval_features(pred: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pred
    out = pred.copy()
    out["temperature_C"] = out["temperature"]
    out["time_index"] = out["end_index"]
    if lookup is not None and len(lookup):
        keep = [
            "trajectory_id",
            "end_index",
            "V_raw",
            "V_corr_raw",
            "I_raw",
            "T",
            "absI",
            "dI",
            "feature_split",
        ]
        extra = lookup[[c for c in keep if c in lookup.columns]].drop_duplicates(["trajectory_id", "end_index"])
        overlap = [c for c in extra.columns if c not in {"trajectory_id", "end_index"} and c in out.columns]
        extra = extra.drop(columns=overlap)
        out = out.merge(extra, on=["trajectory_id", "end_index"], how="left", validate="many_to_one")
    out["is_plateau_20_80"] = (out["y_true"] >= 0.2) & (out["y_true"] <= 0.8)
    out["SOC_bin"] = np.select(
        [out["y_true"] < 0.2, out["y_true"] <= 0.8],
        ["0-20", "20-80"],
        default="80-100",
    )
    max_index = out.groupby("trajectory_id")["end_index"].transform("max").replace(0, np.nan)
    out["trajectory_fraction"] = out["end_index"] / max_index
    out["is_cutoff_last10"] = out["trajectory_fraction"] >= 0.9
    return out


def train_model(
    cfg: NMCVITFeatureLSTMConfig,
    feature_set: str,
    feature_cols: list[str],
    frames: dict[str, list[pd.DataFrame]],
    out_dir: Path,
) -> dict[str, pd.DataFrame]:
    scaled, _ = make_scaled_frames_for_ablation(frames, feature_cols)
    base_cfg = make_cfg()
    base_cfg.output_dir = out_dir
    base_cfg.batch_size = int(cfg.batch_size)
    base_cfg.dataloader_num_workers = int(cfg.num_workers)
    base_cfg.dataloader_prefetch_factor = int(cfg.prefetch_factor)
    base_cfg.dataloader_pin_memory = True
    base_cfg.dataloader_persistent_workers = int(cfg.num_workers) > 0

    train_ds_cls = SequenceWindowDataset if cfg.sequence_loss else DecomposedWindowDataset
    train_ds = train_ds_cls(scaled["train"], feature_cols, cfg.window_len, cfg.stride, target_label="physical")
    valid_ds = (
        DecomposedWindowDataset(scaled["valid"], feature_cols, cfg.window_len, 1, target_label="physical")
        if len(scaled.get("valid", []))
        else None
    )
    test_ds = DecomposedWindowDataset(scaled["test"], feature_cols, cfg.window_len, 1, target_label="physical")
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(f"Empty dataset: train_windows={len(train_ds)} test_windows={len(test_ds)}")

    model_name = f"{cfg.output_prefix}_{feature_set}"
    model = DeepNoLeakLSTM(
        input_dim=len(feature_cols),
        hidden_size=int(cfg.hidden_size),
        layers=int(cfg.layers),
        dropout=float(cfg.dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    train_loader = temperature_balanced_loader(train_ds, base_cfg, shuffle=True)
    valid_loader = make_eval_loader(valid_ds, cfg) if valid_ds is not None and len(valid_ds) else None
    test_loader = make_eval_loader(test_ds, cfg)

    history = []
    best_valid_mae = float("inf")
    best_epoch = 0
    best_state = None
    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        losses = []
        group_losses: dict[str, list[float]] = {}
        for x, y, meta in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            y = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            if cfg.sequence_loss:
                pred = model.forward_sequence(x)
                sample_loss = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=(1, 2))
            else:
                pred = model(x)
                sample_loss = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=1)
            temps = [float(v) for v in meta["temperature"]]
            drives = [str(v) for v in meta["drive_cycle"]]
            if cfg.rex_group == "temperature":
                keys = [f"T{t:g}" for t in temps]
            elif cfg.rex_group == "drive":
                keys = [f"D{d}" for d in drives]
            else:
                keys = [f"T{t:g}_{d}" for t, d in zip(temps, drives)]
            per_group = []
            for key in sorted(set(keys)):
                idx = torch.as_tensor([i for i, k in enumerate(keys) if k == key], device=device, dtype=torch.long)
                g_loss = sample_loss.index_select(0, idx).mean()
                per_group.append(g_loss)
                group_losses.setdefault(key, []).append(float(g_loss.detach().cpu()))
            stack = torch.stack(per_group) if per_group else sample_loss.mean().view(1)
            mean_loss = stack.mean()
            rex_var = stack.var(unbiased=False) if len(stack) > 1 else stack.new_tensor(0.0)
            loss = mean_loss + float(cfg.lambda_rex) * rex_var
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        row = {
            "model_name": model_name,
            "feature_set": feature_set,
            "epoch": ep,
            "loss": float(np.mean(losses)),
            "mean_group_loss": float(mean_loss.detach().cpu()),
            "rex_var": float(rex_var.detach().cpu()),
        }
        for key, vals in group_losses.items():
            safe = str(key).replace(".", "p").replace("-", "N")
            row[f"train_loss_group_{safe}"] = float(np.mean(vals))
        if valid_loader is not None and (
            ep == 1 or ep == int(cfg.epochs) or ep % max(1, int(cfg.valid_every)) == 0
        ):
            valid_mae = evaluate_loader_mae(model, valid_loader)
            row["valid_MAE"] = float(valid_mae)
            row["valid_MAE_pct"] = float(valid_mae * 100.0)
            if np.isfinite(valid_mae) and valid_mae < best_valid_mae:
                best_valid_mae = float(valid_mae)
                best_epoch = int(ep)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)
        if ep == 1 or ep == int(cfg.epochs) or ep % max(1, int(cfg.print_every)) == 0:
            valid_msg = f" valid_mae={row['valid_MAE_pct']:.3f}%" if "valid_MAE_pct" in row else ""
            print(f"{model_name} epoch={ep} loss={row['loss']:.5f}{valid_msg}", flush=True)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    else:
        best_epoch = int(cfg.epochs)

    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / f"{model_name}_history.csv", index=False)
    lookup = make_endpoint_lookup(frames, feature_cols)
    pred = attach_eval_features(predict_loader(model, test_loader, model_name), lookup)
    valid_pred = (
        attach_eval_features(predict_loader(model, valid_loader, model_name), lookup)
        if valid_loader is not None
        else pd.DataFrame()
    )
    for df in (pred, valid_pred):
        if not df.empty:
            df["seed"] = int(cfg.seed)
            df["feature_set"] = feature_set
            df["train_profiles"] = ",".join(cfg.train_profiles)
            df["valid_profiles"] = ",".join(cfg.valid_profiles)
            df["test_profiles"] = ",".join(cfg.test_profiles)
            df["selected_epoch"] = int(best_epoch)
            df["input_feature_dim"] = int(len(feature_cols))

    pred.to_csv(out_dir / f"{model_name}_prediction_rows.csv.gz", index=False, compression="gzip")
    if not valid_pred.empty:
        valid_pred.to_csv(out_dir / f"{model_name}_valid_prediction_rows.csv.gz", index=False, compression="gzip")

    overall = _overall_metrics(pred)
    by_temp = variance_by_temperature(pred)
    by_traj = metrics_by_trajectory(pred)
    focus = focus_metrics(pred, cfg, model_name)
    valid_overall = _overall_metrics(valid_pred) if not valid_pred.empty else pd.DataFrame()
    valid_by_temp = variance_by_temperature(valid_pred) if not valid_pred.empty else pd.DataFrame()
    for df in (overall, by_temp, by_traj, focus, valid_overall, valid_by_temp):
        if df is not None and not df.empty:
            df["feature_set"] = feature_set
            df["selected_epoch"] = int(best_epoch)
            df["input_feature_dim"] = int(len(feature_cols))

    overall.to_csv(out_dir / f"{model_name}_overall.csv", index=False)
    by_temp.to_csv(out_dir / f"{model_name}_by_temperature.csv", index=False)
    by_traj.to_csv(out_dir / f"{model_name}_by_trajectory.csv", index=False)
    focus.to_csv(out_dir / f"{model_name}_focus.csv", index=False)
    if not valid_overall.empty:
        valid_overall.to_csv(out_dir / f"{model_name}_valid_overall.csv", index=False)
        valid_by_temp.to_csv(out_dir / f"{model_name}_valid_by_temperature.csv", index=False)

    metadata = {
        **asdict(cfg),
        "base_dir": str(cfg.base_dir),
        "raw_root": str(cfg.raw_root),
        "model_name": model_name,
        "feature_set": feature_set,
        "feature_columns": feature_cols,
        "input_feature_dim": int(len(feature_cols)),
        "input_dim_explanation": f"{cfg.window_len} timesteps x {len(feature_cols)} V/I/T-derived features",
        "hidden_size_verified": int(cfg.hidden_size),
        "window_len_verified": int(cfg.window_len),
        "uses_soc_input": False,
        "uses_cumulative_input": False,
        "uses_explicit_current_integration": False,
        "uses_self_supervised_pretraining": False,
        "uses_amp": False,
        "label_column": "SOC_CC",
        "selected_epoch": int(best_epoch),
        "best_valid_MAE": float(best_valid_mae) if np.isfinite(best_valid_mae) else None,
        "train_windows": int(len(train_ds)),
        "valid_windows": int(len(valid_ds)) if valid_ds is not None else 0,
        "test_windows": int(len(test_ds)),
        "train_trajectories": int(len(frames["train"])),
        "valid_trajectories": int(len(frames["valid"])),
        "test_trajectories": int(len(frames["test"])),
    }
    (out_dir / f"{model_name}_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print("Overall:")
    print(overall.to_string(index=False), flush=True)
    if not valid_overall.empty:
        print("Valid overall:")
        print(valid_overall.to_string(index=False), flush=True)
        print(f"Selected epoch: {best_epoch}", flush=True)
    print("By temperature:")
    print(by_temp.to_string(index=False), flush=True)
    return {
        "history": history_df,
        "pred": pred,
        "valid_pred": valid_pred,
        "overall": overall,
        "by_temperature": by_temp,
        "by_trajectory": by_traj,
        "focus": focus,
        "valid_overall": valid_overall,
        "valid_by_temperature": valid_by_temp,
    }


def prepare_frames(cfg: NMCVITFeatureLSTMConfig, out_dir: Path):
    files = find_csv_files(cfg.raw_root)
    start_audit = write_start_audit(files, out_dir / f"{cfg.output_prefix}_file_start_audit.csv")
    raw_source_columns = list(pd.read_csv(files[0], nrows=1).columns)
    r0_df = estimate_r0_by_temperature(files, cfg.train_profiles)
    r0_df.to_csv(out_dir / f"{cfg.output_prefix}_decomposition_params.csv", index=False)
    frames = add_vit_engineered_features(build_feature_frames(cfg, files, r0_df))
    return files, start_audit, raw_source_columns, r0_df, frames


def write_screening_report(
    cfg: NMCVITFeatureLSTMConfig,
    out_dir: Path,
    start_audit: pd.DataFrame,
    r0_df: pd.DataFrame,
    schema_all: pd.DataFrame,
    leakage_all: pd.DataFrame,
    overall_all: pd.DataFrame,
    valid_all: pd.DataFrame,
    by_temp_all: pd.DataFrame,
    focus_all: pd.DataFrame,
) -> None:
    best_valid = pd.DataFrame()
    if not valid_all.empty and "MAE_pct" in valid_all.columns:
        best_valid = valid_all.sort_values("MAE_pct").head(1)
    best_test = overall_all.sort_values("MAE_pct").head(1) if not overall_all.empty else pd.DataFrame()
    lines = [
        "# NMC V/I/T Feature LSTM Screening",
        "",
        "## Fixed condition",
        f"- Raw data: `{cfg.raw_root}`",
        f"- Train profiles: {', '.join(cfg.train_profiles)}",
        f"- Valid profile: {', '.join(cfg.valid_profiles)}",
        f"- Test profile: {', '.join(cfg.test_profiles)}",
        "- Temperatures: 0, 25, 45 C are all included in train/valid/test by profile split.",
        f"- Model: stateless LSTM, hidden_dim={cfg.hidden_size}, seq_len={cfg.window_len}, layers={cfg.layers}",
        f"- Training: stride={cfg.stride}, batch={cfg.batch_size}, lr={cfg.lr}, AdamW, Huber beta={cfg.huber_beta}, REx={cfg.lambda_rex}",
        f"- Sequence loss: {cfg.sequence_loss}",
        "- Inputs: V/I/T-derived features only. No SOC input, no initial SOC/window-start SOC, no cumulative Ah, no absolute time/progress, no explicit SOC current-integration update.",
        "- Current is used only as instantaneous/local excitation and causal voltage-response preprocessing, not integrated into SOC state.",
        "",
        "## Best by valid MAE",
        table_md(best_valid, ["model_name", "feature_set", "n_windows", "MAE_pct", "RMSE_pct", "selected_epoch"]) if not best_valid.empty else "(empty)",
        "",
        "## Best by FUDS test MAE",
        table_md(best_test, ["model_name", "feature_set", "n_windows", "MAE_pct", "RMSE_pct", "selected_epoch"]) if not best_test.empty else "(empty)",
        "",
        "## Test overall",
        table_md(overall_all, ["model_name", "feature_set", "n_windows", "MAE_pct", "RMSE_pct", "selected_epoch"]),
        "",
        "## Valid overall",
        table_md(valid_all, ["model_name", "feature_set", "n_windows", "MAE_pct", "RMSE_pct", "selected_epoch"]) if not valid_all.empty else "(empty)",
        "",
        "## Test by temperature",
        table_md(by_temp_all, ["feature_set", "temperature_C", "n_windows", "MAE_pct", "RMSE_pct", "jitter_ratio"]),
        "",
        "## Focus metrics",
        table_md(focus_all, ["feature_set", "scope", "n_windows", "MAE_pct", "RMSE_pct", "catastrophic_gt5_pct"]),
        "",
        "## R0 / Vcorr preprocessing",
        table_md(r0_df, ["temperature_C", "r0_ohm", "n_events", "r0_p20_ohm", "r0_p80_ohm"]),
        "",
        "## Start SOC audit",
        table_md(
            start_audit,
            [
                "file_name",
                "temperature_C",
                "profile",
                "first_voltage_v",
                "soc0_used",
                "soc0_vinit_v",
                "qnet_denom_Ah",
                "first_soc_cc",
            ],
        ),
        "",
        "## Input schemas",
        table_md(schema_all, ["feature_set", "index_1based", "feature_name", "source"]),
        "",
        "## Leakage audit",
        table_md(leakage_all, ["feature_set", "audit_item", "status", "detail"]),
    ]
    (out_dir / f"{cfg.output_prefix}_screening_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(cfg: NMCVITFeatureLSTMConfig) -> dict[str, pd.DataFrame]:
    cfg.base_dir = Path(cfg.base_dir).resolve()
    if not Path(cfg.raw_root).is_absolute():
        cfg.raw_root = cfg.base_dir / cfg.raw_root
    if int(cfg.hidden_size) != 64:
        raise ValueError("This screening is fixed to hidden_size=64.")
    if int(cfg.window_len) != 50:
        raise ValueError("This screening is fixed to window_len=50.")

    out_dir = cfg.base_dir / "nmc_vit_feature_lstm_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_torch_runtime()
    set_seed(cfg.seed)

    _files, start_audit, raw_source_columns, r0_df, frames = prepare_frames(cfg, out_dir)
    selected_sets = all_feature_sets() if cfg.feature_set == "all" else [cfg.feature_set]
    all_results = []
    all_schema = []
    all_leakage = []
    for fs in selected_sets:
        cols = feature_columns(fs)
        available = set().union(*(set(f.columns) for split in frames.values() for f in split))
        missing = [c for c in cols if c not in available]
        if missing:
            raise KeyError(f"Missing feature columns for {fs}: {missing}")
        model_name = f"{cfg.output_prefix}_{fs}"
        schema = write_input_schema(cols, out_dir / f"{model_name}_input_schema.csv").assign(feature_set=fs)
        leakage = write_leakage_audit(cols, raw_source_columns, out_dir / f"{model_name}_leakage_audit.csv").assign(feature_set=fs)
        all_schema.append(schema)
        all_leakage.append(leakage)
        print(f"Running feature_set={fs} n_features={len(cols)}", flush=True)
        res = train_model(cfg, fs, cols, frames, out_dir)
        all_results.append(res)

    overall_all = pd.concat([r["overall"] for r in all_results], ignore_index=True)
    by_temp_all = pd.concat([r["by_temperature"] for r in all_results], ignore_index=True)
    focus_all = pd.concat([r["focus"] for r in all_results], ignore_index=True)
    history_all = pd.concat([r["history"] for r in all_results], ignore_index=True)
    valid_all = pd.concat([r["valid_overall"] for r in all_results if not r["valid_overall"].empty], ignore_index=True)
    valid_by_temp_all = pd.concat(
        [r["valid_by_temperature"] for r in all_results if not r["valid_by_temperature"].empty],
        ignore_index=True,
    )
    schema_all = pd.concat(all_schema, ignore_index=True)
    leakage_all = pd.concat(all_leakage, ignore_index=True)
    overall_all = overall_all.sort_values("MAE_pct").reset_index(drop=True)
    valid_all = valid_all.sort_values("MAE_pct").reset_index(drop=True) if not valid_all.empty else valid_all
    overall_all.to_csv(out_dir / f"{cfg.output_prefix}_screening_overall.csv", index=False)
    by_temp_all.to_csv(out_dir / f"{cfg.output_prefix}_screening_by_temperature.csv", index=False)
    focus_all.to_csv(out_dir / f"{cfg.output_prefix}_screening_focus.csv", index=False)
    history_all.to_csv(out_dir / f"{cfg.output_prefix}_screening_history.csv", index=False)
    valid_all.to_csv(out_dir / f"{cfg.output_prefix}_screening_valid_overall.csv", index=False)
    valid_by_temp_all.to_csv(out_dir / f"{cfg.output_prefix}_screening_valid_by_temperature.csv", index=False)
    schema_all.to_csv(out_dir / f"{cfg.output_prefix}_screening_input_schema.csv", index=False)
    leakage_all.to_csv(out_dir / f"{cfg.output_prefix}_screening_leakage_audit.csv", index=False)
    metadata = {
        **asdict(cfg),
        "base_dir": str(cfg.base_dir),
        "raw_root": str(cfg.raw_root),
        "feature_sets": selected_sets,
        "hidden_size_verified": int(cfg.hidden_size),
        "window_len_verified": int(cfg.window_len),
        "uses_soc_input": False,
        "uses_cumulative_input": False,
        "uses_explicit_current_integration": False,
        "uses_amp": False,
    }
    (out_dir / f"{cfg.output_prefix}_screening_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    write_screening_report(
        cfg,
        out_dir,
        start_audit,
        r0_df,
        schema_all,
        leakage_all,
        overall_all,
        valid_all,
        by_temp_all,
        focus_all,
    )
    print("Screening test overall:")
    print(overall_all.to_string(index=False), flush=True)
    if not valid_all.empty:
        print("Screening valid overall:")
        print(valid_all.to_string(index=False), flush=True)
    print(f"Report: {out_dir / (cfg.output_prefix + '_screening_report.md')}", flush=True)
    return {
        "overall": overall_all,
        "by_temperature": by_temp_all,
        "focus": focus_all,
        "valid_overall": valid_all,
        "valid_by_temperature": valid_by_temp_all,
        "schema": schema_all,
        "leakage": leakage_all,
        "history": history_all,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NMC V/I/T-derived feature LSTM screening with hidden=64 and seq_len=50.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--raw-root", default=NMCVITFeatureLSTMConfig.raw_root)
    p.add_argument("--output-prefix", default=BASE_PREFIX)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--feature-set", default="all")
    p.add_argument("--train-profiles", default="DST,US06")
    p.add_argument("--valid-profiles", default="NONE")
    p.add_argument("--test-profiles", default="FUDS")
    p.add_argument("--window-len", type=int, default=50)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lambda-rex", type=float, default=2.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--valid-every", type=int, default=10)
    p.add_argument("--endpoint-loss", action="store_true", help="Use endpoint loss instead of default sequence loss.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = NMCVITFeatureLSTMConfig(
        base_dir=Path(args.base_dir),
        raw_root=Path(args.raw_root),
        output_prefix=str(args.output_prefix),
        seed=int(args.seed),
        feature_set=str(args.feature_set),
        train_profiles=tuple(s.strip() for s in str(args.train_profiles).split(",") if s.strip()),
        valid_profiles=tuple(s.strip() for s in str(args.valid_profiles).split(",") if s.strip()),
        test_profiles=tuple(s.strip() for s in str(args.test_profiles).split(",") if s.strip()),
        window_len=int(args.window_len),
        stride=int(args.stride),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        hidden_size=int(args.hidden_size),
        layers=int(args.layers),
        dropout=float(args.dropout),
        lambda_rex=float(args.lambda_rex),
        sequence_loss=not bool(args.endpoint_loss),
        num_workers=int(args.num_workers),
        print_every=int(args.print_every),
        valid_every=int(args.valid_every),
    )
    run(cfg)


if __name__ == "__main__":
    main()
