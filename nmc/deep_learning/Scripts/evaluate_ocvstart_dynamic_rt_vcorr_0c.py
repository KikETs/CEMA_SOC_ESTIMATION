#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "nmc_ocvstart_lopo_clean"
OUT_DIR = ROOT / "nmc_goal_vcorr_it_train_dst_selector_results" / "ocvstart_dynamic_rt_vcorr_only"
PREV_DIR = ROOT / "nmc_goal_vcorr_it_train_dst_selector_results" / "ocvstart_lfpstyle_trueocv_vcorr_only"
OCV_PREP_DIR = Path("/home/user/바탕화면/DL/CEMA-TCN/Data")
OCV_REF_DIR = OCV_PREP_DIR / "raw_reference"

sys.path.insert(0, str(OCV_PREP_DIR))
import prepare_calce_nmc as prep  # noqa: E402


PROFILES = ("VALIDATION", "DST", "FUDS", "US06")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def causal_ema(values: np.ndarray, times_s: np.ndarray, tau_s: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    t = np.asarray(times_s, dtype=np.float64)
    if len(x) == 0:
        return x.astype(np.float64)
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    if len(x) == 1:
        return y
    d = np.diff(t)
    good = d[np.isfinite(d) & (d > 0)]
    dt_default = float(np.nanmedian(good)) if len(good) else 1.0
    if not np.isfinite(dt_default) or dt_default <= 0:
        dt_default = 1.0
    for k in range(1, len(x)):
        dt = t[k] - t[k - 1]
        if not np.isfinite(dt) or dt <= 0:
            dt = dt_default
        alpha = float(np.exp(-dt / max(float(tau_s), 1e-6)))
        alpha = min(max(alpha, 0.0), 0.999999)
        y[k] = alpha * y[k - 1] + (1.0 - alpha) * x[k]
    return y


def parse_temp(path: Path, df: pd.DataFrame | None = None) -> float:
    if df is not None and "TempLabel" in df.columns and len(df):
        return float(str(df["TempLabel"].iloc[0]).replace("C", ""))
    return float(path.parent.name.replace("C", ""))


def parse_profile(path: Path, df: pd.DataFrame | None = None) -> str:
    if df is not None and "Profile" in df.columns and len(df):
        return str(df["Profile"].iloc[0]).upper()
    return path.stem.split("_")[-1].upper()


def soc01(df: pd.DataFrame) -> np.ndarray:
    soc = pd.to_numeric(df["SOC_CC"], errors="coerce").to_numpy(np.float64)
    if np.nanmax(soc) > 1.5:
        soc = soc / 100.0
    return np.clip(soc, 0.0, 1.0)


def ocv_from_soc(refs: dict[float, object], temp: float, soc: np.ndarray) -> np.ndarray:
    ref = refs.get(float(temp))
    if ref is None or ref.voltage_v is None or ref.soc_fraction is None:
        return np.full(len(soc), np.nan, dtype=np.float64)
    ref_soc = np.asarray(ref.soc_fraction, dtype=np.float64)
    ref_v = np.asarray(ref.voltage_v, dtype=np.float64)
    order = np.argsort(ref_soc)
    ref_soc = ref_soc[order]
    ref_v = ref_v[order]
    keep = np.isfinite(ref_soc) & np.isfinite(ref_v)
    return np.interp(np.asarray(soc, dtype=np.float64), ref_soc[keep], ref_v[keep]).astype(np.float64)


def load_frame(path: Path, refs: dict[float, object]) -> pd.DataFrame:
    df = pd.read_csv(path)
    temp = parse_temp(path, df)
    time_col = "Step_Time(s)" if "Step_Time(s)" in df.columns else "Test_Time(s)"
    soc = soc01(df)
    out = pd.DataFrame(
        {
            "file_name": path.name,
            "profile": parse_profile(path, df),
            "temperature_C": float(temp),
            "timestep": np.arange(len(df), dtype=np.int64),
            "time_s": pd.to_numeric(df[time_col], errors="coerce").to_numpy(np.float64),
            "SOC_frac": soc,
            "SOC_pct": soc * 100.0,
            "V_raw": pd.to_numeric(df["Voltage(V)"], errors="coerce").to_numpy(np.float64),
            "I_raw": pd.to_numeric(df["Current(A)"], errors="coerce").to_numpy(np.float64),
        }
    )
    out["OCV_from_SOC"] = ocv_from_soc(refs, float(temp), out["SOC_frac"].to_numpy(np.float64))
    return out.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["time_s", "SOC_frac", "V_raw", "I_raw", "OCV_from_SOC"]
    ).reset_index(drop=True)


def estimate_r0(files: list[Path], train_profiles: tuple[str, ...], temp: float = 0.0, quantile: float = 0.5) -> float:
    vals: list[float] = []
    for path in files:
        head = pd.read_csv(path, nrows=2)
        if parse_temp(path, head) != float(temp) or parse_profile(path, head) not in train_profiles:
            continue
        df = pd.read_csv(path, usecols=["Current(A)", "Voltage(V)"])
        i = pd.to_numeric(df["Current(A)"], errors="coerce").to_numpy(np.float64)
        v = pd.to_numeric(df["Voltage(V)"], errors="coerce").to_numpy(np.float64)
        di = np.diff(i, prepend=i[0])
        dv = np.diff(v, prepend=v[0])
        mask = np.isfinite(di) & np.isfinite(dv) & (np.abs(di) > 0.05) & (np.abs(dv) > 1e-5)
        ratio = dv[mask] / di[mask]
        ratio = ratio[np.isfinite(ratio) & (ratio > 0.001) & (ratio < 0.5)]
        vals.extend(float(x) for x in ratio)
    if not vals:
        raise RuntimeError(f"Could not estimate R0 for train profiles {train_profiles}")
    return float(np.quantile(vals, quantile))


def build_feature_matrix(frame: pd.DataFrame, scales: dict[str, float]) -> np.ndarray:
    v = frame["V_raw"].to_numpy(np.float64)
    i = frame["I_raw"].to_numpy(np.float64)
    t = frame["time_s"].to_numpy(np.float64)
    abs_i = np.abs(i)
    abs_di = np.abs(np.diff(i, prepend=i[0]))
    i_dis = np.maximum(-i, 0.0)
    i_ema30 = causal_ema(abs_i, t, 30.0)
    i_ema240 = causal_ema(abs_i, t, 240.0)
    hys = causal_ema(np.sign(i_dis) * np.sqrt(abs_i + 1e-9), t, 900.0)
    vmin = float(scales["vmin"])
    vmax = float(scales["vmax"])
    vrange = max(vmax - vmin, 1e-6)
    i_scale = max(float(scales["i_scale"]), 1e-6)
    di_scale = max(float(scales["di_scale"]), 1e-6)
    sqrt_i_scale = max(np.sqrt(i_scale), 1e-6)
    v_s = np.clip(2.0 * (v - vmin) / vrange - 1.0, -3.0, 3.0)
    abs_i_s = np.clip(abs_i / i_scale, 0.0, 5.0)
    abs_di_s = np.clip(abs_di / di_scale, 0.0, 5.0)
    i30_s = np.clip(i_ema30 / i_scale, 0.0, 5.0)
    i240_s = np.clip(i_ema240 / i_scale, 0.0, 5.0)
    hys_s = np.clip(hys / sqrt_i_scale, -5.0, 5.0)
    low_v = sigmoid((float(scales["lowv_center"]) - v) / max(float(scales["lowv_width"]), 1e-6))
    return np.column_stack(
        [
            np.ones(len(frame), dtype=np.float64),
            v_s,
            abs_i_s,
            abs_di_s,
            i30_s,
            i240_s,
            hys_s,
            low_v,
            low_v * abs_i_s,
            low_v * i240_s,
        ]
    ).astype(np.float64)


def predict_dynamic_rt(frame: pd.DataFrame, p: np.ndarray, scales: dict[str, float], r0: float, log_bound: float = np.log(2.0)) -> tuple[np.ndarray, np.ndarray]:
    x = build_feature_matrix(frame, scales)
    eta = float(log_bound) * np.tanh(x @ np.asarray(p, dtype=np.float64))
    r_t = float(r0) * np.exp(eta)
    v_ohm_free = frame["V_raw"].to_numpy(np.float64) - r_t * frame["I_raw"].to_numpy(np.float64)
    v_corr = causal_ema(v_ohm_free, frame["time_s"].to_numpy(np.float64), 120.0)
    return v_corr.astype(np.float64), r_t.astype(np.float64)


def fit_dynamic_rt(train_frames: list[pd.DataFrame], r0: float, fit_stride: int = 5) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    v_all = np.concatenate([f["V_raw"].to_numpy(np.float64) for f in train_frames])
    i_all = np.concatenate([f["I_raw"].to_numpy(np.float64) for f in train_frames])
    di_all = np.diff(i_all, prepend=i_all[0])
    scales = {
        "vmin": float(np.nanpercentile(v_all, 1)),
        "vmax": float(np.nanpercentile(v_all, 99)),
        "i_scale": float(max(np.nanpercentile(np.abs(i_all), 95), 1e-3)),
        "di_scale": float(max(np.nanpercentile(np.abs(di_all), 95), 1e-3)),
        "lowv_center": float(np.nanpercentile(v_all, 35)),
        "lowv_width": 0.10,
    }
    n_params = build_feature_matrix(train_frames[0].iloc[:4].copy(), scales).shape[1]
    p0 = np.zeros(n_params, dtype=np.float64)
    lower = np.full(n_params, -3.0, dtype=np.float64)
    upper = np.full(n_params, 3.0, dtype=np.float64)
    idx_by_frame = [np.arange(0, len(f), max(int(fit_stride), 1)) for f in train_frames]

    def residual(p: np.ndarray) -> np.ndarray:
        chunks = []
        smooth_chunks = []
        for frame, idx in zip(train_frames, idx_by_frame):
            pred, rt = predict_dynamic_rt(frame, p, scales, r0)
            ocv = frame["OCV_from_SOC"].to_numpy(np.float64)
            soc = frame["SOC_pct"].to_numpy(np.float64)
            v = frame["V_raw"].to_numpy(np.float64)
            err = pred[idx] - ocv[idx]
            weight = 1.0 + 0.5 * (soc[idx] <= 20.0).astype(np.float64) + 0.25 * (v[idx] <= scales["lowv_center"]).astype(np.float64)
            chunks.append(err * weight)
            rt_idx = rt[idx]
            if len(rt_idx) > 1:
                smooth_chunks.append(0.002 * np.diff(rt_idx) / max(float(r0), 1e-9))
        reg = 0.01 * np.asarray(p, dtype=np.float64)
        return np.concatenate(chunks + smooth_chunks + [reg])

    before = residual(p0)
    opt = least_squares(
        residual,
        p0,
        bounds=(lower, upper),
        max_nfev=220,
        loss="soft_l1",
        f_scale=0.02,
    )
    after = residual(opt.x)
    info = {
        "n_fit_points": int(sum(len(idx) for idx in idx_by_frame)),
        "train_weighted_mae_before_mV": float(np.nanmean(np.abs(before)) * 1000.0),
        "train_weighted_mae_after_mV": float(np.nanmean(np.abs(after)) * 1000.0),
        "cost": float(opt.cost),
        "status": int(opt.status),
    }
    return opt.x.astype(np.float64), scales, info


def metric_row(profile: str, frame: pd.DataFrame, col: str, prefix: str) -> dict[str, float | int | str]:
    err = (frame[col].to_numpy(np.float64) - frame["OCV_from_SOC"].to_numpy(np.float64)) * 1000.0
    soc = frame["SOC_pct"].to_numpy(np.float64)
    low20 = soc <= 20.0
    tail10 = soc <= 10.0
    return {
        "profile": profile,
        f"{prefix}_mae_mV": float(np.nanmean(np.abs(err))),
        f"{prefix}_bias_mV": float(np.nanmean(err)),
        f"{prefix}_lowSOC_le20_mae_mV": float(np.nanmean(np.abs(err[low20]))) if np.any(low20) else np.nan,
        f"{prefix}_tail_le10_mae_mV": float(np.nanmean(np.abs(err[tail10]))) if np.any(tail10) else np.nan,
    }


def plot_comparison(full: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 9), sharey=True)
    for ax, profile in zip(axes.ravel(), PROFILES):
        g = full[full["profile"].eq(profile)].copy()
        x = np.arange(len(g), dtype=np.int64)
        ax.plot(x, g["OCV_from_SOC"], color="#111111", lw=1.05, label="OCV(SOC label)")
        ax.plot(x, g["V_raw"], color="#9a9a9a", lw=0.55, alpha=0.7, label="raw")
        ax.plot(x, g["V_corr_r0_ema120"], color="#d55e00", lw=0.95, label="R0+EMA120")
        if "V_corr_lfpstyle_trueocv" in g.columns:
            ax.plot(x, g["V_corr_lfpstyle_trueocv"], color="#1f77b4", lw=0.85, alpha=0.85, label="LFP-style")
        ax.plot(x, g["V_corr_dynRT_ema120"], color="#009e73", lw=1.05, label="dynamic R_T+EMA120")
        ax.set_title(f"{profile} 0C, final SOC={g['SOC_pct'].iloc[-1]:.2f}%")
        ax.set_xlabel("timestep")
        ax.set_ylabel("Voltage (V)")
        ax.grid(alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.suptitle("0C OCV-start label V_corr comparison", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_error(full: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 9), sharey=True)
    for ax, profile in zip(axes.ravel(), PROFILES):
        g = full[full["profile"].eq(profile)].copy()
        x = np.arange(len(g), dtype=np.int64)
        ax.axhline(0.0, color="#111111", lw=0.8)
        ax.plot(x, (g["V_corr_r0_ema120"] - g["OCV_from_SOC"]) * 1000.0, color="#d55e00", lw=0.85, label="R0+EMA120")
        if "V_corr_lfpstyle_trueocv" in g.columns:
            ax.plot(x, (g["V_corr_lfpstyle_trueocv"] - g["OCV_from_SOC"]) * 1000.0, color="#1f77b4", lw=0.75, alpha=0.85, label="LFP-style")
        ax.plot(x, (g["V_corr_dynRT_ema120"] - g["OCV_from_SOC"]) * 1000.0, color="#009e73", lw=0.95, label="dynamic R_T+EMA120")
        ax.set_title(profile)
        ax.set_xlabel("timestep")
        ax.set_ylabel("V_corr - OCV (mV)")
        ax.grid(alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("0C OCV-start label V_corr error comparison", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    refs = prep.load_ocv_references(OCV_REF_DIR)
    files = sorted(RAW_ROOT.rglob("*.csv"))
    files0 = sorted((RAW_ROOT / "0C").glob("NMC_0C_*.csv"))
    frames = [load_frame(path, refs) for path in files0]

    lfp_full_path = PREV_DIR / "ocvstart_lfpstyle_trueocv_vcorr_0C_full.csv.gz"
    lfp_full = pd.read_csv(lfp_full_path) if lfp_full_path.exists() else pd.DataFrame()
    ema_full_path = PREV_DIR / "ocvstart_r0_ema120_trueocv_0C_full.csv.gz"
    ema_full = pd.read_csv(ema_full_path) if ema_full_path.exists() else pd.DataFrame()

    out_frames = []
    fit_rows = []
    for holdout in PROFILES:
        train_profiles = tuple(p for p in PROFILES if p != holdout)
        r0 = estimate_r0(files, train_profiles, temp=0.0)
        train_frames = [f for f in frames if str(f["profile"].iloc[0]).upper() in train_profiles]
        p, scales, info = fit_dynamic_rt(train_frames, r0)
        frame = next(f.copy() for f in frames if str(f["profile"].iloc[0]).upper() == holdout)
        dyn, rt = predict_dynamic_rt(frame, p, scales, r0)
        frame["V_corr_dynRT_ema120"] = dyn
        frame["R_t_dynRT_ohm"] = rt
        frame["R0_train_ohm"] = r0
        if not ema_full.empty:
            e = ema_full[ema_full["profile"].eq(holdout)].reset_index(drop=True)
            if len(e) == len(frame):
                frame["V_corr_r0_ema120"] = e["V_corr_r0_ema120"].to_numpy(np.float64)
        if "V_corr_r0_ema120" not in frame.columns:
            frame["V_corr_r0_ema120"] = causal_ema(
                frame["V_raw"].to_numpy(np.float64) - r0 * frame["I_raw"].to_numpy(np.float64),
                frame["time_s"].to_numpy(np.float64),
                120.0,
            )
        if not lfp_full.empty:
            l = lfp_full[lfp_full["profile"].eq(holdout)].reset_index(drop=True)
            if len(l) == len(frame) and "V_corr_lfpstyle_trueocv" in l.columns:
                frame["V_corr_lfpstyle_trueocv"] = l["V_corr_lfpstyle_trueocv"].to_numpy(np.float64)
        out_frames.append(frame)
        fit_rows.append({"holdout": holdout, "train_profiles": "+".join(train_profiles), "r0_ohm": r0, **info, **{f"scale_{k}": v for k, v in scales.items()}})

    full = pd.concat(out_frames, ignore_index=True)
    full.to_csv(OUT_DIR / "ocvstart_dynamic_rt_trueocv_0C_full.csv.gz", index=False)
    pd.DataFrame(fit_rows).to_csv(OUT_DIR / "ocvstart_dynamic_rt_trueocv_0C_train_fit.csv", index=False)

    rows = []
    for profile, g in full.groupby("profile", sort=True):
        row = {
            "profile": profile,
            "rows": int(len(g)),
            "soc0_pct": float(g["SOC_pct"].iloc[0]),
            "final_soc_pct": float(g["SOC_pct"].iloc[-1]),
        }
        row.update(metric_row(profile, g, "V_raw", "raw"))
        row.update(metric_row(profile, g, "V_corr_r0_ema120", "ema120"))
        if "V_corr_lfpstyle_trueocv" in g.columns:
            row.update(metric_row(profile, g, "V_corr_lfpstyle_trueocv", "lfpstyle"))
        row.update(metric_row(profile, g, "V_corr_dynRT_ema120", "dynRT"))
        row["dynRT_R_t_mean_ohm"] = float(np.nanmean(g["R_t_dynRT_ohm"]))
        row["dynRT_R_t_p05_ohm"] = float(np.nanpercentile(g["R_t_dynRT_ohm"], 5))
        row["dynRT_R_t_p95_ohm"] = float(np.nanpercentile(g["R_t_dynRT_ohm"], 95))
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("profile")
    summary.to_csv(OUT_DIR / "ocvstart_dynamic_rt_trueocv_0C_summary.csv", index=False)

    rank_cols = ["profile", "raw_mae_mV", "ema120_mae_mV", "lfpstyle_mae_mV", "dynRT_mae_mV", "raw_lowSOC_le20_mae_mV", "ema120_lowSOC_le20_mae_mV", "lfpstyle_lowSOC_le20_mae_mV", "dynRT_lowSOC_le20_mae_mV", "raw_tail_le10_mae_mV", "ema120_tail_le10_mae_mV", "lfpstyle_tail_le10_mae_mV", "dynRT_tail_le10_mae_mV"]
    existing = [c for c in rank_cols if c in summary.columns]
    summary[existing].to_csv(OUT_DIR / "ocvstart_dynamic_rt_trueocv_0C_compare_table.csv", index=False)

    plot_comparison(full, fig_dir / "ocvstart_dynamic_rt_all_profiles_0C_vcorr_compare.png")
    plot_error(full, fig_dir / "ocvstart_dynamic_rt_all_profiles_0C_error_compare.png")

    print("summary", OUT_DIR / "ocvstart_dynamic_rt_trueocv_0C_summary.csv")
    print("plot", fig_dir / "ocvstart_dynamic_rt_all_profiles_0C_vcorr_compare.png")
    print("error_plot", fig_dir / "ocvstart_dynamic_rt_all_profiles_0C_error_compare.png")
    print(summary[existing].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
