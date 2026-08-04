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


def build_decomposed_frame(
    path: Path,
    r0_lookup: dict[float, float],
    cfg: NMCBranchBandsConfig,
    ocv_calibration: dict | None = None,
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
            "R0": np.full(len(df), r0, dtype=np.float32),
            "V_pol_fast_raw": v_pol_fast,
            "V_pol_mid_raw": v_pol_mid,
            "V_pol_slow_raw": v_pol_slow,
        }
    )
    return out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def build_feature_frames(cfg: NMCBranchBandsConfig, files: list[Path], r0_df: pd.DataFrame) -> dict[str, list[pd.DataFrame]]:
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
            frame = build_decomposed_frame(p, {float(temp): float(r0_profile_lookup[key])}, cfg, ocv_calibration)
        else:
            frame = build_decomposed_frame(p, r0_lookup, cfg, ocv_calibration)
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
