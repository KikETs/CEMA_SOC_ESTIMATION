#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "nmc_ocvstart_endzero_lopo_clean"
OUT = ROOT / "nmc_goal_vcorr_it_train_dst_selector_results" / "nmc_tailored_vcorr_only"
FIG = OUT / "figures"
OCV_PREP_DIR = Path("/home/user/바탕화면/DL/CEMA-TCN/Data")
OCV_REF_DIR = OCV_PREP_DIR / "raw_reference"
PROFILES = ("VALIDATION", "DST", "FUDS", "US06")
TEMPS = (0.0, 25.0, 45.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def causal_ema(values: np.ndarray, times_s: np.ndarray, tau_s: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    t = np.asarray(times_s, dtype=np.float64)
    if len(x) == 0:
        return x.astype(np.float32)
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    if len(x) == 1:
        return y.astype(np.float32)
    dt_all = np.diff(t)
    valid_dt = dt_all[np.isfinite(dt_all) & (dt_all > 0)]
    dt_default = float(np.nanmedian(valid_dt)) if len(valid_dt) else 1.0
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


def parse_temperature(path: Path) -> float:
    m = re.search(r"(-?\d+(?:\.\d+)?)C", str(path))
    if not m:
        raise ValueError(f"Cannot parse temperature from {path}")
    return float(m.group(1))


def parse_profile(path: Path) -> str:
    m = re.search(r"NMC_[^_]+_(.+)\.csv", path.name)
    if not m:
        raise ValueError(f"Cannot parse profile from {path.name}")
    return m.group(1).upper()


def load_ocv_refs():
    sys.path.insert(0, str(OCV_PREP_DIR))
    import prepare_calce_nmc as prep

    return prep.load_ocv_references(OCV_REF_DIR)


def ocv_from_soc(refs, temp: float, soc_frac: np.ndarray) -> np.ndarray:
    ref = refs.get(float(temp))
    if ref is None or ref.voltage_v is None or ref.soc_fraction is None:
        return np.full_like(np.asarray(soc_frac, dtype=np.float64), np.nan, dtype=np.float64)
    soc = np.asarray(ref.soc_fraction, dtype=np.float64)
    volt = np.asarray(ref.voltage_v, dtype=np.float64)
    order = np.argsort(soc)
    soc = soc[order]
    volt = volt[order]
    keep = np.concatenate([[True], np.diff(soc) > 1e-10])
    return np.interp(np.asarray(soc_frac, dtype=np.float64), soc[keep], volt[keep], left=volt[keep][0], right=volt[keep][-1])


def load_frame(path: Path, refs) -> pd.DataFrame:
    df = pd.read_csv(path)
    temp = parse_temperature(path)
    profile = str(df["Profile"].iloc[0]).upper() if "Profile" in df.columns and len(df) else parse_profile(path)
    time_col = "Step_Time(s)" if "Step_Time(s)" in df.columns else "Test_Time(s)"
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(np.float64)
    v = pd.to_numeric(df["Voltage(V)"], errors="coerce").to_numpy(np.float64)
    i = pd.to_numeric(df["Current(A)"], errors="coerce").to_numpy(np.float64)
    soc = pd.to_numeric(df["SOC_CC"], errors="coerce").to_numpy(np.float64)
    if np.nanmax(soc) > 1.5:
        soc = soc / 100.0
    ocv_ref = ocv_from_soc(refs, temp, soc)
    return pd.DataFrame(
        {
            "file_name": path.name,
            "profile": profile,
            "temperature_C": float(temp),
            "end_index": np.arange(len(df), dtype=np.int64),
            "time_s": t,
            "SOC_frac": soc,
            "SOC_pct": soc * 100.0,
            "V_raw": v,
            "I_raw": i,
            "OCV_ref_from_SOC": ocv_ref,
        }
    ).replace([np.inf, -np.inf], np.nan).dropna(subset=["time_s", "V_raw", "I_raw", "SOC_frac"]).reset_index(drop=True)


def estimate_r0_event(frames: list[pd.DataFrame], quantile: float = 0.5) -> dict[float, float]:
    rows = []
    all_ratio = []
    for frame in frames:
        temp = float(frame["temperature_C"].iloc[0])
        i = frame["I_raw"].to_numpy(np.float64)
        v = frame["V_raw"].to_numpy(np.float64)
        di = np.diff(i, prepend=i[0])
        dv = np.diff(v, prepend=v[0])
        ratio = np.full_like(dv, np.nan, dtype=np.float64)
        np.divide(dv, di, out=ratio, where=np.abs(di) > 1e-12)
        mask = np.isfinite(ratio) & (np.abs(di) > 0.05) & (np.abs(dv) > 1e-5) & (ratio > 0.001) & (ratio < 0.5)
        for r in ratio[mask]:
            rows.append({"temperature_C": temp, "r0_event_ohm": float(r)})
        all_ratio.extend([float(r) for r in ratio[mask]])
    if not all_ratio:
        raise RuntimeError("No current-step R0 events found.")
    fallback = float(np.quantile(all_ratio, quantile))
    out = {}
    for temp in TEMPS:
        vals = [r["r0_event_ohm"] for r in rows if float(r["temperature_C"]) == float(temp)]
        out[float(temp)] = float(np.quantile(vals, quantile)) if vals else fallback
    return out


def add_baseline_vcorr(frame: pd.DataFrame, r0_ohm: float) -> pd.DataFrame:
    f = frame.copy()
    t = f["time_s"].to_numpy(np.float64)
    v = f["V_raw"].to_numpy(np.float64)
    i = f["I_raw"].to_numpy(np.float64)
    ohm_free = v - i * float(r0_ohm)
    f["R0_event_ohm"] = float(r0_ohm)
    f["V_corr_r0_noema"] = ohm_free.astype(np.float32)
    f["V_corr_r0_ema120"] = causal_ema(ohm_free, t, 120.0)
    f["V_corr_r0_ema10"] = causal_ema(ohm_free, t, 10.0)
    return f


def fit_train_pseudo_ocv_curve(train_frames: list[pd.DataFrame], r0_ohm: float) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    parts = []
    for frame in train_frames:
        f = frame.copy()
        v = f["V_raw"].to_numpy(np.float64)
        i = f["I_raw"].to_numpy(np.float64)
        soc = f["SOC_frac"].to_numpy(np.float64)
        di = np.diff(i, prepend=i[0])
        parts.append(
            pd.DataFrame(
                {
                    "profile": str(f["profile"].iloc[0]),
                    "soc": soc,
                    "v_ohm_free": v - i * float(r0_ohm),
                    "abs_i": np.abs(i),
                    "abs_di": np.abs(di),
                }
            )
        )
    allp = pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
    if allp.empty:
        raise RuntimeError("Cannot fit pseudo OCV curve from empty train frames.")
    abs_i = allp["abs_i"].to_numpy(np.float64)
    abs_di = allp["abs_di"].to_numpy(np.float64)
    low_i = max(0.05, float(np.nanquantile(abs_i, 0.18)))
    low_di = max(0.01, float(np.nanquantile(abs_di, 0.35)))
    mask = (allp["abs_i"] <= low_i) & (allp["abs_di"] <= low_di)
    if int(mask.sum()) < 500:
        low_i = max(0.10, float(np.nanquantile(abs_i, 0.30)))
        low_di = max(0.02, float(np.nanquantile(abs_di, 0.50)))
        mask = (allp["abs_i"] <= low_i) & (allp["abs_di"] <= low_di)
    source = allp.loc[mask].copy()
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
        order = np.argsort(allp["soc"].to_numpy(np.float64))
        sorted_df = allp.iloc[order].reset_index(drop=True)
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
    y = curve["ocv_voltage"].to_numpy(np.float64)
    y = np.maximum.accumulate(y)
    curve["ocv_voltage_monotone"] = y
    return x, y, curve.reset_index(drop=True)


def apply_ocv_curve(frame: pd.DataFrame, curve_soc: np.ndarray, curve_v: np.ndarray) -> pd.DataFrame:
    f = frame.copy()
    soc = f["SOC_frac"].to_numpy(np.float64)
    f["OCV_from_SOC"] = np.interp(soc, curve_soc, curve_v, left=float(curve_v[0]), right=float(curve_v[-1]))
    return f


def feature_matrix(frame: pd.DataFrame, scales: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    t = frame["time_s"].to_numpy(np.float64)
    v = frame["V_raw"].to_numpy(np.float64)
    i_raw = frame["I_raw"].to_numpy(np.float64)
    i_dis = -i_raw
    di = np.diff(i_dis, prepend=i_dis[0])
    abs_i = np.abs(i_dis)
    abs_di = np.abs(di)
    i_scale = max(float(scales["i_scale"]), 1e-6)
    di_scale = max(float(scales["di_scale"]), 1e-6)
    v_center = float(scales["v_center"])
    v_scale = max(float(scales["v_scale"]), 1e-6)
    v_s = np.clip((v - v_center) / v_scale, -5.0, 5.0)
    low_gate = sigmoid((float(scales["low_v_gate"]) - v) / max(float(scales["low_v_slope"]), 1e-6))
    high_gate = sigmoid((v - float(scales["high_v_gate"])) / 0.08)
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


def fit_nmc_tailored(train: pd.DataFrame) -> dict:
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
    x, names = feature_matrix(tr, scales)
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


def apply_nmc_tailored(frame: pd.DataFrame, model: dict) -> np.ndarray:
    x, _ = feature_matrix(frame, model["scales"])
    xs = (x - model["x_mu"]) / model["x_sig"]
    delta = xs @ model["beta"]
    delta = np.clip(delta, -float(model["cap_V"]), float(model["cap_V"]))
    vcorr = frame["V_raw"].to_numpy(np.float64) + delta
    return np.clip(vcorr, 2.45, 4.25).astype(np.float32)


def select_innerloo_blend_alpha(
    raw_frames_by_profile: dict[str, pd.DataFrame],
    train_profiles: tuple[str, ...],
    temperature: float,
) -> tuple[float, pd.DataFrame]:
    candidates = np.linspace(0.0, 1.0, 21)
    rows = []
    for valid_profile in train_profiles:
        inner_train_profiles = tuple(p for p in train_profiles if p != valid_profile)
        inner_raw = [raw_frames_by_profile[p] for p in inner_train_profiles]
        valid_raw = raw_frames_by_profile[valid_profile]
        r0_inner = estimate_r0_event(inner_raw)[float(temperature)]
        curve_soc, curve_v, _curve_df = fit_train_pseudo_ocv_curve(inner_raw, r0_inner)
        inner_train = [
            apply_ocv_curve(add_baseline_vcorr(frame, r0_inner), curve_soc, curve_v)
            for frame in inner_raw
        ]
        valid_frame = apply_ocv_curve(add_baseline_vcorr(valid_raw, r0_inner), curve_soc, curve_v)
        model = fit_nmc_tailored(pd.concat(inner_train, ignore_index=True))
        tailored = apply_nmc_tailored(valid_frame, model)
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
    avg = scores.groupby("alpha", as_index=False)["mae_mV"].mean().sort_values(["mae_mV", "alpha"])
    best = float(avg.iloc[0]["alpha"])
    best_score = float(avg.iloc[0]["mae_mV"])
    # If several alphas are effectively tied, prefer the smaller intervention.
    tied = avg[avg["mae_mV"] <= best_score + 0.25].sort_values("alpha")
    if not tied.empty:
        best = float(tied.iloc[0]["alpha"])
    scores["selected_alpha"] = best
    scores["selected_inner_mean_mae_mV"] = best_score
    return best, scores


def select_innerloo_minimax_alpha(scores: pd.DataFrame) -> tuple[float, dict]:
    agg = (
        scores.groupby("alpha", as_index=False)
        .agg(inner_mean_mae_mV=("mae_mV", "mean"), inner_max_mae_mV=("mae_mV", "max"))
        .sort_values(["inner_max_mae_mV", "inner_mean_mae_mV", "alpha"])
    )
    best = agg.iloc[0].to_dict()
    return float(best["alpha"]), best


def alignment_feature_matrix(
    frame: pd.DataFrame,
    scales: dict[str, float],
    base_col: str = "V_corr_nmc_tailored_innerloo_minimax_blend",
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
    low_gate = sigmoid((float(scales["low_v_gate"]) - v_base) / max(float(scales["low_v_slope"]), 1e-6))
    high_gate = sigmoid((v_base - float(scales["high_v_gate"])) / max(float(scales["high_v_slope"]), 1e-6))
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


def fit_alignment_corrector(
    train: pd.DataFrame,
    base_col: str = "V_corr_nmc_tailored_innerloo_minimax_blend",
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
    x, names = alignment_feature_matrix(tr, scales, base_col)
    finite = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x = x[finite]
    y = y[finite]
    v_fit = v_base[finite]
    low_gate = sigmoid((float(scales["low_v_gate"]) - v_fit) / max(float(scales["low_v_slope"]), 1e-6))
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


def apply_alignment_corrector(
    frame: pd.DataFrame,
    model: dict,
    gamma: float,
    base_col: str = "V_corr_nmc_tailored_innerloo_minimax_blend",
) -> np.ndarray:
    x, _ = alignment_feature_matrix(frame, model["scales"], base_col)
    xs = (x - model["x_mu"]) / model["x_sig"]
    correction = xs @ model["beta"]
    correction = np.clip(correction, -float(model["cap_V"]), float(model["cap_V"]))
    out = frame[base_col].to_numpy(np.float64) + float(gamma) * correction
    return np.clip(out, 2.45, 4.25).astype(np.float32)


def add_minimax_vcorr_columns(frame: pd.DataFrame, model: dict, alpha: float) -> pd.DataFrame:
    out = frame.copy()
    out["V_corr_nmc_tailored"] = apply_nmc_tailored(out, model)
    out["V_corr_nmc_tailored_ema10"] = causal_ema(
        out["V_corr_nmc_tailored"].to_numpy(np.float64),
        out["time_s"].to_numpy(np.float64),
        10.0,
    )
    out["V_corr_nmc_tailored_innerloo_minimax_blend"] = (
        (1.0 - float(alpha)) * out["V_corr_r0_ema120"].to_numpy(np.float64)
        + float(alpha) * out["V_corr_nmc_tailored_ema10"].to_numpy(np.float64)
    ).astype(np.float32)
    out["innerloo_minimax_alpha"] = float(alpha)
    return out


def select_alignment_gamma(
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
        r0_inner = estimate_r0_event(inner_raw)[float(temperature)]
        curve_soc, curve_v, _curve_df = fit_train_pseudo_ocv_curve(inner_raw, r0_inner)
        inner_base = [
            apply_ocv_curve(add_baseline_vcorr(frame, r0_inner), curve_soc, curve_v)
            for frame in inner_raw
        ]
        valid_base = apply_ocv_curve(add_baseline_vcorr(valid_raw, r0_inner), curve_soc, curve_v)
        model = fit_nmc_tailored(pd.concat(inner_base, ignore_index=True))
        alpha_scores = select_innerloo_blend_alpha(
            {str(frame["profile"].iloc[0]).upper(): frame for frame in inner_raw},
            inner_train_profiles,
            float(temperature),
        )[1]
        minimax_alpha, _meta = select_innerloo_minimax_alpha(alpha_scores)
        inner_ready = [add_minimax_vcorr_columns(frame, model, minimax_alpha) for frame in inner_base]
        valid_ready = add_minimax_vcorr_columns(valid_base, model, minimax_alpha)
        align_model = fit_alignment_corrector(pd.concat(inner_ready, ignore_index=True))
        target = valid_ready["OCV_from_SOC"].to_numpy(np.float64)
        for gamma in candidates:
            pred = apply_alignment_corrector(valid_ready, align_model, gamma)
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
    best = agg.iloc[0].to_dict()
    gamma = float(best["gamma"])
    scores["selected_gamma"] = gamma
    scores["selected_inner_mean_mae_mV"] = float(best["inner_mean_mae_mV"])
    scores["selected_inner_max_mae_mV"] = float(best["inner_max_mae_mV"])
    return gamma, scores


def voltage_blend_weight(frame: pd.DataFrame, scales: dict, base: float, low_gain: float, high_gain: float) -> np.ndarray:
    v = frame["V_raw"].to_numpy(np.float64)
    slope = max(float(scales["low_v_slope"]), 1e-3)
    low_gate = 1.0 / (1.0 + np.exp((v - float(scales["low_v_gate"])) / slope))
    high_gate = 1.0 / (1.0 + np.exp(-(v - float(scales["high_v_gate"])) / slope))
    return np.clip(float(base) + float(low_gain) * low_gate + float(high_gain) * high_gate, 0.0, 1.0)


def select_innerloo_vgate_blend(
    raw_frames_by_profile: dict[str, pd.DataFrame],
    train_profiles: tuple[str, ...],
    temperature: float,
) -> tuple[dict, pd.DataFrame]:
    bases = [0.0, 0.05, 0.10, 0.20, 0.40]
    low_gains = [0.0, 0.25, 0.50, 0.75, 1.00]
    high_gains = [0.0, 0.25, 0.50]
    rows = []
    for valid_profile in train_profiles:
        inner_train_profiles = tuple(p for p in train_profiles if p != valid_profile)
        inner_raw = [raw_frames_by_profile[p] for p in inner_train_profiles]
        valid_raw = raw_frames_by_profile[valid_profile]
        r0_inner = estimate_r0_event(inner_raw)[float(temperature)]
        curve_soc, curve_v, _curve_df = fit_train_pseudo_ocv_curve(inner_raw, r0_inner)
        inner_train = [
            apply_ocv_curve(add_baseline_vcorr(frame, r0_inner), curve_soc, curve_v)
            for frame in inner_raw
        ]
        valid_frame = apply_ocv_curve(add_baseline_vcorr(valid_raw, r0_inner), curve_soc, curve_v)
        model = fit_nmc_tailored(pd.concat(inner_train, ignore_index=True))
        tailored = apply_nmc_tailored(valid_frame, model)
        tailored_ema10 = causal_ema(tailored, valid_frame["time_s"].to_numpy(np.float64), 10.0)
        old = valid_frame["V_corr_r0_ema120"].to_numpy(np.float64)
        target = valid_frame["OCV_from_SOC"].to_numpy(np.float64)
        for base in bases:
            for low_gain in low_gains:
                for high_gain in high_gains:
                    w = voltage_blend_weight(valid_frame, model["scales"], base, low_gain, high_gain)
                    pred = old + w * (tailored_ema10 - old)
                    diff = (pred - target) * 1000.0
                    rows.append(
                        {
                            "temperature_C": float(temperature),
                            "valid_profile": valid_profile,
                            "inner_train_profiles": "+".join(inner_train_profiles),
                            "base": float(base),
                            "low_gain": float(low_gain),
                            "high_gain": float(high_gain),
                            "mean_weight": float(np.nanmean(w)),
                            "mae_mV": float(np.nanmean(np.abs(diff))),
                            "bias_mV": float(np.nanmean(diff)),
                        }
                    )
    scores = pd.DataFrame(rows)
    avg = (
        scores.groupby(["base", "low_gain", "high_gain"], as_index=False)
        .agg(mae_mV=("mae_mV", "mean"), mean_weight=("mean_weight", "mean"))
        .sort_values(["mae_mV", "mean_weight", "base", "low_gain", "high_gain"])
    )
    best = avg.iloc[0].to_dict()
    best_score = float(best["mae_mV"])
    tied = avg[avg["mae_mV"] <= best_score + 0.25].sort_values(["mean_weight", "base", "low_gain", "high_gain"])
    if not tied.empty:
        best = tied.iloc[0].to_dict()
    selected = {
        "base": float(best["base"]),
        "low_gain": float(best["low_gain"]),
        "high_gain": float(best["high_gain"]),
        "selected_inner_mean_mae_mV": best_score,
        "selected_inner_mean_weight": float(best["mean_weight"]),
    }
    scores["selected_base"] = selected["base"]
    scores["selected_low_gain"] = selected["low_gain"]
    scores["selected_high_gain"] = selected["high_gain"]
    scores["selected_inner_mean_mae_mV"] = selected["selected_inner_mean_mae_mV"]
    scores["selected_inner_mean_weight"] = selected["selected_inner_mean_weight"]
    return selected, scores


def summarize(frame: pd.DataFrame, value_col: str, holdout: str, method: str) -> list[dict]:
    rows = []
    finite = frame.dropna(subset=[value_col, "OCV_from_SOC", "SOC_pct"]).copy()
    finite = finite[np.isfinite(finite["OCV_from_SOC"])]
    for temp, g in finite.groupby("temperature_C"):
        diff = (g[value_col].to_numpy(np.float64) - g["OCV_from_SOC"].to_numpy(np.float64)) * 1000.0
        rows.append(
            {
                "holdout": holdout,
                "temperature_C": float(temp),
                "method": method,
                "scope": "all",
                "n": int(len(g)),
                "bias_mV": float(np.nanmean(diff)),
                "mae_mV": float(np.nanmean(np.abs(diff))),
                "rmse_mV": float(np.sqrt(np.nanmean(diff * diff))),
                "p95_abs_mV": float(np.nanpercentile(np.abs(diff), 95)),
            }
        )
        cats = pd.cut(g["SOC_pct"], bins=[0, 10, 20, 40, 60, 80, 100], include_lowest=True, right=False)
        tmp = pd.DataFrame({"soc_bin": cats, "diff": diff})
        for b, bg in tmp.groupby("soc_bin", observed=True):
            rows.append(
                {
                    "holdout": holdout,
                    "temperature_C": float(temp),
                    "method": method,
                    "scope": str(b),
                    "n": int(len(bg)),
                    "bias_mV": float(bg["diff"].mean()),
                    "mae_mV": float(bg["diff"].abs().mean()),
                    "rmse_mV": float(np.sqrt(np.mean(bg["diff"].to_numpy() ** 2))),
                    "p95_abs_mV": float(bg["diff"].abs().quantile(0.95)),
                }
            )
    return rows


def plot_holdout(frame: pd.DataFrame, holdout: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for temp, g in frame.groupby("temperature_C"):
        g = g.sort_values("end_index")
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        axes[0].plot(g["SOC_pct"], g["OCV_from_SOC"], color="black", lw=1.5, label="OCV(SOC) target")
        axes[0].plot(g["SOC_pct"], g["V_raw"], color="#777777", lw=0.8, alpha=0.8, label="V_raw")
        axes[0].plot(g["SOC_pct"], g["V_corr_r0_ema120"], color="#d55e00", lw=0.9, alpha=0.9, label="R0+EMA120")
        axes[0].plot(g["SOC_pct"], g["V_corr_nmc_tailored"], color="#0072b2", lw=0.9, alpha=0.9, label="NMC tailored")
        if "V_corr_nmc_tailored_innerloo_blend" in g:
            axes[0].plot(
                g["SOC_pct"],
                g["V_corr_nmc_tailored_innerloo_blend"],
                color="#009e73",
                lw=0.9,
                alpha=0.9,
                label="Inner-LOO blend",
            )
        if "V_corr_nmc_tailored_innerloo_minimax_blend" in g:
            axes[0].plot(
                g["SOC_pct"],
                g["V_corr_nmc_tailored_innerloo_minimax_blend"],
                color="#f0e442",
                lw=0.9,
                alpha=0.9,
                label="Inner-LOO minimax",
            )
        if "V_corr_nmc_tailored_minimax_aligned" in g:
            axes[0].plot(
                g["SOC_pct"],
                g["V_corr_nmc_tailored_minimax_aligned"],
                color="#56b4e9",
                lw=0.9,
                alpha=0.9,
                label="Aligned minimax",
            )
        if "V_corr_nmc_tailored_innerloo_vgate_blend" in g:
            axes[0].plot(
                g["SOC_pct"],
                g["V_corr_nmc_tailored_innerloo_vgate_blend"],
                color="#cc79a7",
                lw=0.9,
                alpha=0.9,
                label="V-gated blend",
            )
        axes[0].invert_xaxis()
        axes[0].grid(True, alpha=0.25)
        axes[0].set_ylabel("Voltage (V)")
        axes[0].set_title(f"{holdout} holdout {temp:g}C corrected voltage only")
        axes[0].legend(loc="best", fontsize=8)
        for col, color, label in [
            ("V_corr_r0_ema120", "#d55e00", "R0+EMA120"),
            ("V_corr_nmc_tailored", "#0072b2", "NMC tailored"),
            ("V_corr_nmc_tailored_innerloo_blend", "#009e73", "Inner-LOO blend"),
            ("V_corr_nmc_tailored_innerloo_minimax_blend", "#f0e442", "Inner-LOO minimax"),
            ("V_corr_nmc_tailored_minimax_aligned", "#56b4e9", "Aligned minimax"),
            ("V_corr_nmc_tailored_innerloo_vgate_blend", "#cc79a7", "V-gated blend"),
        ]:
            if col not in g:
                continue
            diff = (g[col].to_numpy(np.float64) - g["OCV_from_SOC"].to_numpy(np.float64)) * 1000.0
            axes[1].plot(g["SOC_pct"], diff, color=color, lw=0.8, alpha=0.8, label=label)
        axes[1].axhline(0.0, color="black", lw=0.8)
        axes[1].set_ylim(-350, 350)
        axes[1].invert_xaxis()
        axes[1].grid(True, alpha=0.25)
        axes[1].set_xlabel("SOC label (%)")
        axes[1].set_ylabel("V_corr - OCV(SOC) (mV)")
        axes[1].legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / f"endzero_nmc_tailored_vcorr_only_holdout{holdout.lower()}_{int(temp)}C.png", dpi=180)
        plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    refs = load_ocv_refs()
    frames = [load_frame(path, refs) for path in sorted(DATA_ROOT.glob("*C/*.csv"))]
    summary_rows = []
    param_rows = []
    full_rows = []
    ocv_curve_rows = []
    alpha_score_rows = []
    vgate_score_rows = []
    alignment_score_rows = []

    for holdout in PROFILES:
        train_profiles = tuple(p for p in PROFILES if p != holdout)
        train_base = [f for f in frames if len(f) and str(f["profile"].iloc[0]).upper() in train_profiles]
        r0_lookup = estimate_r0_event(train_base)
        holdout_parts = []
        for temp in TEMPS:
            train_temp_raw = []
            holdout_raw = None
            for frame in frames:
                if frame.empty:
                    continue
                profile = str(frame["profile"].iloc[0]).upper()
                frame_temp = float(frame["temperature_C"].iloc[0])
                if frame_temp != float(temp):
                    continue
                if profile in train_profiles:
                    train_temp_raw.append(frame)
                elif profile == holdout:
                    holdout_raw = frame
            if holdout_raw is None:
                raise RuntimeError(f"Missing holdout frame for holdout={holdout} temp={temp}")
            curve_soc, curve_v, curve_df = fit_train_pseudo_ocv_curve(train_temp_raw, r0_lookup[float(temp)])
            curve_df = curve_df.assign(holdout=holdout, temperature_C=float(temp), train_profiles="+".join(train_profiles))
            ocv_curve_rows.append(curve_df)
            train_temp = [
                apply_ocv_curve(add_baseline_vcorr(frame, r0_lookup[float(temp)]), curve_soc, curve_v)
                for frame in train_temp_raw
            ]
            holdout_part = apply_ocv_curve(add_baseline_vcorr(holdout_raw, r0_lookup[float(temp)]), curve_soc, curve_v)
            train_df = pd.concat(train_temp, ignore_index=True)
            model = fit_nmc_tailored(train_df)
            raw_frames_by_profile = {
                str(frame["profile"].iloc[0]).upper(): frame
                for frame in train_temp_raw
                if not frame.empty
            }
            blend_alpha, alpha_scores = select_innerloo_blend_alpha(raw_frames_by_profile, train_profiles, float(temp))
            minimax_alpha, minimax_meta = select_innerloo_minimax_alpha(alpha_scores)
            alpha_scores = alpha_scores.assign(
                holdout=holdout,
                final_train_profiles="+".join(train_profiles),
                selected_alpha=blend_alpha,
                selected_minimax_alpha=minimax_alpha,
                selected_minimax_inner_mean_mae_mV=float(minimax_meta["inner_mean_mae_mV"]),
                selected_minimax_inner_max_mae_mV=float(minimax_meta["inner_max_mae_mV"]),
            )
            alpha_score_rows.append(alpha_scores)
            vgate_params, vgate_scores = select_innerloo_vgate_blend(raw_frames_by_profile, train_profiles, float(temp))
            vgate_scores = vgate_scores.assign(
                holdout=holdout,
                final_train_profiles="+".join(train_profiles),
            )
            vgate_score_rows.append(vgate_scores)
            alignment_gamma, alignment_scores = select_alignment_gamma(raw_frames_by_profile, train_profiles, float(temp))
            alignment_scores = alignment_scores.assign(
                holdout=holdout,
                final_train_profiles="+".join(train_profiles),
            )
            alignment_score_rows.append(alignment_scores)
            holdout_part = holdout_part.copy()
            holdout_part["V_corr_nmc_tailored"] = apply_nmc_tailored(holdout_part, model)
            holdout_part["V_corr_nmc_tailored_ema10"] = causal_ema(
                holdout_part["V_corr_nmc_tailored"].to_numpy(np.float64),
                holdout_part["time_s"].to_numpy(np.float64),
                10.0,
            )
            holdout_part["V_corr_nmc_tailored_innerloo_blend"] = (
                (1.0 - blend_alpha) * holdout_part["V_corr_r0_ema120"].to_numpy(np.float64)
                + blend_alpha * holdout_part["V_corr_nmc_tailored_ema10"].to_numpy(np.float64)
            ).astype(np.float32)
            holdout_part["innerloo_blend_alpha"] = float(blend_alpha)
            holdout_part["V_corr_nmc_tailored_innerloo_minimax_blend"] = (
                (1.0 - minimax_alpha) * holdout_part["V_corr_r0_ema120"].to_numpy(np.float64)
                + minimax_alpha * holdout_part["V_corr_nmc_tailored_ema10"].to_numpy(np.float64)
            ).astype(np.float32)
            holdout_part["innerloo_minimax_alpha"] = float(minimax_alpha)
            train_ready = [
                add_minimax_vcorr_columns(frame, model, minimax_alpha)
                for frame in train_temp
            ]
            alignment_model = fit_alignment_corrector(pd.concat(train_ready, ignore_index=True))
            holdout_part["V_corr_nmc_tailored_minimax_aligned"] = apply_alignment_corrector(
                holdout_part,
                alignment_model,
                alignment_gamma,
            )
            holdout_part["alignment_gamma"] = float(alignment_gamma)
            vgate_w = voltage_blend_weight(
                holdout_part,
                model["scales"],
                vgate_params["base"],
                vgate_params["low_gain"],
                vgate_params["high_gain"],
            )
            holdout_part["V_corr_nmc_tailored_innerloo_vgate_blend"] = (
                holdout_part["V_corr_r0_ema120"].to_numpy(np.float64)
                + vgate_w * (
                    holdout_part["V_corr_nmc_tailored_ema10"].to_numpy(np.float64)
                    - holdout_part["V_corr_r0_ema120"].to_numpy(np.float64)
                )
            ).astype(np.float32)
            holdout_part["innerloo_vgate_weight"] = vgate_w.astype(np.float32)
            holdout_part["innerloo_vgate_base"] = float(vgate_params["base"])
            holdout_part["innerloo_vgate_low_gain"] = float(vgate_params["low_gain"])
            holdout_part["innerloo_vgate_high_gain"] = float(vgate_params["high_gain"])
            holdout_parts.append(holdout_part)
            for name, beta, mu, sig in zip(model["names"], model["beta"], model["x_mu"], model["x_sig"]):
                param_rows.append(
                    {
                        "holdout": holdout,
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
                        "r0_event_ohm": float(r0_lookup[float(temp)]),
                    }
                )
        holdout_df = pd.concat(holdout_parts, ignore_index=True)
        full_rows.append(holdout_df)
        plot_holdout(holdout_df, holdout)
        for method, col in [
            ("raw_terminal", "V_raw"),
            ("r0_noema", "V_corr_r0_noema"),
            ("r0_ema10", "V_corr_r0_ema10"),
            ("r0_ema120_old", "V_corr_r0_ema120"),
            ("nmc_tailored_no_final_ema", "V_corr_nmc_tailored"),
            ("nmc_tailored_ema10", "V_corr_nmc_tailored_ema10"),
            ("nmc_tailored_innerloo_blend", "V_corr_nmc_tailored_innerloo_blend"),
            ("nmc_tailored_innerloo_minimax_blend", "V_corr_nmc_tailored_innerloo_minimax_blend"),
            ("nmc_tailored_minimax_aligned", "V_corr_nmc_tailored_minimax_aligned"),
            ("nmc_tailored_innerloo_vgate_blend", "V_corr_nmc_tailored_innerloo_vgate_blend"),
        ]:
            summary_rows.extend(summarize(holdout_df, col, holdout, method))

    full = pd.concat(full_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(["holdout", "temperature_C", "scope", "mae_mV", "method"])
    params = pd.DataFrame(param_rows).sort_values(["holdout", "temperature_C", "feature"])
    full.to_csv(OUT / "endzero_nmc_tailored_vcorr_only_full.csv.gz", index=False, compression="gzip")
    summary.to_csv(OUT / "endzero_nmc_tailored_vcorr_only_summary.csv", index=False)
    params.to_csv(OUT / "endzero_nmc_tailored_vcorr_only_params.csv", index=False)
    if ocv_curve_rows:
        pd.concat(ocv_curve_rows, ignore_index=True).to_csv(OUT / "endzero_nmc_tailored_vcorr_only_train_pseudo_ocv_curves.csv", index=False)
    if alpha_score_rows:
        pd.concat(alpha_score_rows, ignore_index=True).to_csv(OUT / "endzero_nmc_tailored_vcorr_only_innerloo_alpha_scores.csv", index=False)
    if vgate_score_rows:
        pd.concat(vgate_score_rows, ignore_index=True).to_csv(OUT / "endzero_nmc_tailored_vcorr_only_innerloo_vgate_scores.csv", index=False)
    if alignment_score_rows:
        pd.concat(alignment_score_rows, ignore_index=True).to_csv(OUT / "endzero_nmc_tailored_vcorr_only_alignment_gamma_scores.csv", index=False)
    overall = (
        summary[summary["scope"].eq("all")]
        .groupby("method", as_index=False)
        .agg(mean_mae_mV=("mae_mV", "mean"), max_mae_mV=("mae_mV", "max"), mean_bias_mV=("bias_mV", "mean"))
        .sort_values("mean_mae_mV")
    )
    overall.to_csv(OUT / "endzero_nmc_tailored_vcorr_only_overall_rank.csv", index=False)
    by_holdout = (
        summary[summary["scope"].eq("all")]
        .pivot_table(index=["holdout", "temperature_C"], columns="method", values="mae_mV", aggfunc="mean")
        .reset_index()
    )
    by_holdout.to_csv(OUT / "endzero_nmc_tailored_vcorr_only_holdout_temp_pivot.csv", index=False)
    print("overall rank")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("outputs")
    print(OUT)


if __name__ == "__main__":
    main()
