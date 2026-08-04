#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "nmc_ocvstart_endzero_lopo_clean"
OUT_DIR = ROOT / "nmc_goal_vcorr_it_train_dst_selector_results" / "endzero_dynamic_rt_variance_only"
FIG_DIR = OUT_DIR / "figures"
PROFILES = ("VALIDATION", "DST", "FUDS", "US06")


def parse_temperature(path: Path) -> float:
    m = re.search(r"(-?\d+(?:\.\d+)?)C", str(path))
    if not m:
        raise ValueError(f"Cannot parse temperature from {path}")
    return float(m.group(1))


def parse_profile(path: Path, df: pd.DataFrame | None = None) -> str:
    if df is not None and "Profile" in df.columns and len(df):
        return str(df["Profile"].iloc[0]).upper()
    m = re.search(r"NMC_[^_]+_(.+)\.csv", path.name)
    if not m:
        raise ValueError(f"Cannot parse profile from {path.name}")
    return m.group(1).upper()


def causal_ema(values: np.ndarray, times_s: np.ndarray, tau_s: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    t = np.asarray(times_s, dtype=np.float64)
    if len(x) == 0:
        return x
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    if len(x) == 1:
        return out
    dt_all = np.diff(t)
    good = dt_all[np.isfinite(dt_all) & (dt_all > 0)]
    dt_default = float(np.nanmedian(good)) if len(good) else 1.0
    if not np.isfinite(dt_default) or dt_default <= 0:
        dt_default = 1.0
    for k in range(1, len(x)):
        dt = t[k] - t[k - 1]
        if not np.isfinite(dt) or dt <= 0:
            dt = dt_default
        alpha = float(np.exp(-dt / max(float(tau_s), 1e-6)))
        alpha = min(max(alpha, 0.0), 0.999999)
        out[k] = alpha * out[k - 1] + (1.0 - alpha) * x[k]
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def load_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    time_col = "Step_Time(s)" if "Step_Time(s)" in df.columns else "Test_Time(s)"
    soc = pd.to_numeric(df["SOC_CC"], errors="coerce").to_numpy(np.float64)
    if np.nanmax(soc) > 1.5:
        soc = soc / 100.0
    out = pd.DataFrame(
        {
            "file_name": path.name,
            "profile": parse_profile(path, df),
            "temperature_C": parse_temperature(path),
            "end_index": np.arange(len(df), dtype=np.int64),
            "time_s": pd.to_numeric(df[time_col], errors="coerce").to_numpy(np.float64),
            "SOC_frac": soc,
            "SOC_pct": soc * 100.0,
            "V_raw": pd.to_numeric(df["Voltage(V)"], errors="coerce").to_numpy(np.float64),
            "I_raw": pd.to_numeric(df["Current(A)"], errors="coerce").to_numpy(np.float64),
        }
    )
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_s", "SOC_frac", "V_raw", "I_raw"]).reset_index(drop=True)


def estimate_r0(frames: list[pd.DataFrame], quantile: float = 0.5) -> float:
    vals: list[float] = []
    for frame in frames:
        i = frame["I_raw"].to_numpy(np.float64)
        v = frame["V_raw"].to_numpy(np.float64)
        di = np.diff(i, prepend=i[0])
        dv = np.diff(v, prepend=v[0])
        ratio = np.full_like(dv, np.nan, dtype=np.float64)
        np.divide(dv, di, out=ratio, where=np.abs(di) > 1e-12)
        mask = np.isfinite(ratio) & (np.abs(di) > 0.05) & (np.abs(dv) > 1e-5) & (ratio > 0.001) & (ratio < 0.5)
        vals.extend(float(x) for x in ratio[mask])
    if not vals:
        raise RuntimeError("No R0 event points.")
    return float(np.nanquantile(vals, quantile))


def local_variance_r(frame: pd.DataFrame, r0: float, tau_s: float, rmax: float) -> np.ndarray:
    t = frame["time_s"].to_numpy(np.float64)
    v = frame["V_raw"].to_numpy(np.float64)
    i = frame["I_raw"].to_numpy(np.float64)
    ev = causal_ema(v, t, tau_s)
    ei = causal_ema(i, t, tau_s)
    evi = causal_ema(v * i, t, tau_s)
    ei2 = causal_ema(i * i, t, tau_s)
    cov = evi - ev * ei
    var_i = ei2 - ei * ei
    r = np.full(len(frame), float(r0), dtype=np.float64)
    np.divide(cov, var_i, out=r, where=np.isfinite(var_i) & (var_i > 1e-5))
    r = np.clip(r, 0.0, float(rmax))
    return causal_ema(r, t, max(float(tau_s) * 0.5, 10.0))


def add_base_columns(frame: pd.DataFrame, r0: float, tau_corr: float = 120.0) -> pd.DataFrame:
    out = frame.copy()
    t = out["time_s"].to_numpy(np.float64)
    v = out["V_raw"].to_numpy(np.float64)
    i = out["I_raw"].to_numpy(np.float64)
    out["R0_base"] = float(r0)
    out["V_ohm_free_base"] = v - i * float(r0)
    out["V_corr_base_r0_ema120"] = causal_ema(out["V_ohm_free_base"].to_numpy(np.float64), t, tau_corr)
    return out


def apply_dynamic_rt(
    frame: pd.DataFrame,
    r0: float,
    tau_r: float,
    rmax: float,
    gate_v: float,
    gate_s: float,
    alpha: float,
    tau_corr: float = 120.0,
) -> pd.DataFrame:
    out = add_base_columns(frame, r0, tau_corr=tau_corr)
    t = out["time_s"].to_numpy(np.float64)
    v = out["V_raw"].to_numpy(np.float64)
    i = out["I_raw"].to_numpy(np.float64)
    r_local = local_variance_r(out, r0, tau_s=tau_r, rmax=rmax)
    gate = sigmoid((float(gate_v) - out["V_corr_base_r0_ema120"].to_numpy(np.float64)) / max(float(gate_s), 1e-6))
    gate_state = causal_ema(gate, t, 60.0)
    r_eff = float(r0) + float(alpha) * gate_state * (r_local - float(r0))
    r_eff = np.clip(r_eff, 0.0, float(rmax))
    out["R_local_var"] = r_local
    out["low_v_gate"] = gate_state
    out["R_eff_dynamic"] = r_eff
    out["V_ohm_free_dynamic"] = v - i * r_eff
    out["V_corr_dynamic_rt_ema120"] = causal_ema(out["V_ohm_free_dynamic"].to_numpy(np.float64), t, tau_corr)
    return out


def hf_residual(v: np.ndarray, t: np.ndarray, tau_s: float = 600.0) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    return arr - causal_ema(arr, np.asarray(t, dtype=np.float64), tau_s)


def metrics(frame: pd.DataFrame, col: str, method: str, holdout: str) -> list[dict[str, float | int | str]]:
    rows = []
    for scope, mask in [
        ("all", np.ones(len(frame), dtype=bool)),
        ("SOC<=10", frame["SOC_pct"].to_numpy(np.float64) <= 10.0),
        ("SOC<=20", frame["SOC_pct"].to_numpy(np.float64) <= 20.0),
        ("20<SOC<=80", (frame["SOC_pct"].to_numpy(np.float64) > 20.0) & (frame["SOC_pct"].to_numpy(np.float64) <= 80.0)),
    ]:
        if not np.any(mask):
            continue
        g = frame.loc[mask]
        v = g[col].to_numpy(np.float64)
        t = g["time_s"].to_numpy(np.float64)
        hf = hf_residual(v, t, tau_s=600.0) * 1000.0
        dv = np.diff(v, prepend=v[0]) * 1000.0
        rows.append(
            {
                "holdout": holdout,
                "profile": str(frame["profile"].iloc[0]),
                "method": method,
                "scope": scope,
                "n": int(len(g)),
                "v_mean_mV": float(np.nanmean(v) * 1000.0),
                "v_std_mV": float(np.nanstd(v) * 1000.0),
                "hf_std_mV": float(np.nanstd(hf)),
                "hf_p95_abs_mV": float(np.nanpercentile(np.abs(hf), 95)),
                "dV_p95_abs_mV": float(np.nanpercentile(np.abs(dv), 95)),
                "R_eff_mean_ohm": float(np.nanmean(g["R_eff_dynamic"])) if "R_eff_dynamic" in g else np.nan,
                "R_eff_p95_ohm": float(np.nanpercentile(g["R_eff_dynamic"], 95)) if "R_eff_dynamic" in g else np.nan,
            }
        )
    return rows


def score_train_frame(frame: pd.DataFrame) -> float:
    low = frame["SOC_pct"].to_numpy(np.float64) <= 10.0
    mid = (frame["SOC_pct"].to_numpy(np.float64) > 20.0) & (frame["SOC_pct"].to_numpy(np.float64) <= 80.0)
    if not np.any(low):
        return np.inf
    t_low = frame.loc[low, "time_s"].to_numpy(np.float64)
    base_low = hf_residual(frame.loc[low, "V_corr_base_r0_ema120"].to_numpy(np.float64), t_low)
    dyn_low = hf_residual(frame.loc[low, "V_corr_dynamic_rt_ema120"].to_numpy(np.float64), t_low)
    low_score = float(np.nanstd(dyn_low) * 1000.0)
    penalty = 0.0
    if np.any(mid):
        t_mid = frame.loc[mid, "time_s"].to_numpy(np.float64)
        base_mid = float(np.nanstd(hf_residual(frame.loc[mid, "V_corr_base_r0_ema120"].to_numpy(np.float64), t_mid)) * 1000.0)
        dyn_mid = float(np.nanstd(hf_residual(frame.loc[mid, "V_corr_dynamic_rt_ema120"].to_numpy(np.float64), t_mid)) * 1000.0)
        penalty += max(0.0, dyn_mid - base_mid) * 0.5
    penalty += max(0.0, float(np.nanpercentile(frame["R_eff_dynamic"], 95)) - 0.25) * 50.0
    return low_score + penalty


def select_params(train_frames: list[pd.DataFrame], r0: float) -> tuple[dict[str, float], pd.DataFrame]:
    base_train = [add_base_columns(f, r0) for f in train_frames]
    base_v = np.concatenate([f["V_corr_base_r0_ema120"].to_numpy(np.float64) for f in base_train])
    low_v = np.concatenate([f.loc[f["SOC_pct"] <= 10.0, "V_corr_base_r0_ema120"].to_numpy(np.float64) for f in base_train])
    if len(low_v) == 0:
        low_v = base_v
    gate_candidates = sorted(set(float(x) for x in np.nanpercentile(low_v, [70, 90])))
    tau_candidates = (30.0, 60.0, 120.0)
    rmax_candidates = (0.18, 0.25)
    gate_s_candidates = (0.06, 0.10)
    alpha_candidates = (0.50, 1.00)
    r_cache: dict[tuple[int, float, float], np.ndarray] = {}
    for idx, frame in enumerate(base_train):
        for tau_r in tau_candidates:
            for rmax in rmax_candidates:
                r_cache[(idx, tau_r, rmax)] = local_variance_r(frame, r0, tau_s=tau_r, rmax=rmax)
    rows = []
    for tau_r in tau_candidates:
        for rmax in rmax_candidates:
            for gate_v in gate_candidates:
                for gate_s in gate_s_candidates:
                    for alpha in alpha_candidates:
                        scores = []
                        for idx, frame in enumerate(base_train):
                            pred = frame.copy()
                            t = pred["time_s"].to_numpy(np.float64)
                            v = pred["V_raw"].to_numpy(np.float64)
                            i = pred["I_raw"].to_numpy(np.float64)
                            r_local = r_cache[(idx, tau_r, rmax)]
                            gate = sigmoid(
                                (float(gate_v) - pred["V_corr_base_r0_ema120"].to_numpy(np.float64))
                                / max(float(gate_s), 1e-6)
                            )
                            gate_state = causal_ema(gate, t, 60.0)
                            r_eff = float(r0) + float(alpha) * gate_state * (r_local - float(r0))
                            r_eff = np.clip(r_eff, 0.0, float(rmax))
                            pred["R_local_var"] = r_local
                            pred["low_v_gate"] = gate_state
                            pred["R_eff_dynamic"] = r_eff
                            pred["V_ohm_free_dynamic"] = v - i * r_eff
                            pred["V_corr_dynamic_rt_ema120"] = causal_ema(
                                pred["V_ohm_free_dynamic"].to_numpy(np.float64),
                                t,
                                120.0,
                            )
                            scores.append(score_train_frame(pred))
                        rows.append(
                            {
                                "tau_r": tau_r,
                                "rmax": rmax,
                                "gate_v": gate_v,
                                "gate_s": gate_s,
                                "alpha": alpha,
                                "train_score_mean": float(np.nanmean(scores)),
                                "train_score_max": float(np.nanmax(scores)),
                            }
                        )
    table = pd.DataFrame(rows).sort_values(["train_score_mean", "train_score_max", "rmax", "alpha"])
    best = table.iloc[0].to_dict()
    return {k: float(best[k]) for k in ("tau_r", "rmax", "gate_v", "gate_s", "alpha")}, table


def plot_holdout(frame: pd.DataFrame, holdout: str, out_path: Path) -> None:
    g = frame.copy()
    x = g["end_index"].to_numpy(np.int64)
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    axes[0].plot(x, g["V_raw"], color="#8a8a8a", lw=0.6, label="raw")
    axes[0].plot(x, g["V_corr_base_r0_ema120"], color="#0072B2", lw=0.85, label="R0+EMA120")
    axes[0].plot(x, g["V_corr_dynamic_rt_ema120"], color="#D55E00", lw=0.85, label="R(T;t)+EMA120")
    axes[0].set_title(f"{holdout} 0C end-zero SOC, no OCV target")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=3)

    base_hf = hf_residual(g["V_corr_base_r0_ema120"].to_numpy(np.float64), g["time_s"].to_numpy(np.float64)) * 1000.0
    dyn_hf = hf_residual(g["V_corr_dynamic_rt_ema120"].to_numpy(np.float64), g["time_s"].to_numpy(np.float64)) * 1000.0
    axes[1].axhline(0.0, color="#111111", lw=0.7)
    axes[1].plot(x, base_hf, color="#0072B2", lw=0.65, label="base high-pass")
    axes[1].plot(x, dyn_hf, color="#D55E00", lw=0.65, label="dynamic high-pass")
    axes[1].set_ylabel("HF residual (mV)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    axes[2].plot(x, g["R_eff_dynamic"], color="#D55E00", lw=0.8, label="R(T;t)")
    axes[2].plot(x, g["R_local_var"], color="#555555", lw=0.55, alpha=0.7, label="local variance R")
    axes[2].set_ylabel("Ohm")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False)

    axes[3].plot(x, g["SOC_pct"], color="#009E73", lw=0.8, label="SOC 80->0")
    axes[3].plot(x, g["low_v_gate"] * 100.0, color="#CC79A7", lw=0.8, label="low-V gate x100")
    axes[3].set_ylabel("SOC / gate")
    axes[3].set_xlabel("timestep")
    axes[3].grid(alpha=0.25)
    axes[3].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_tail(frame: pd.DataFrame, holdout: str, out_path: Path, n_tail: int = 2000) -> None:
    g = frame.tail(int(n_tail)).copy()
    x = g["end_index"].to_numpy(np.int64)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(x, g["V_raw"], color="#8a8a8a", lw=0.6, label="raw")
    axes[0].plot(x, g["V_corr_base_r0_ema120"], color="#0072B2", lw=0.9, label="R0+EMA120")
    axes[0].plot(x, g["V_corr_dynamic_rt_ema120"], color="#D55E00", lw=0.9, label="R(T;t)+EMA120")
    axes[0].set_title(f"{holdout} 0C tail zoom, no OCV target")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].plot(x, g["R_eff_dynamic"], color="#D55E00", lw=0.8, label="R(T;t)")
    axes[1].plot(x, g["SOC_pct"], color="#009E73", lw=0.8, label="SOC")
    axes[1].set_ylabel("Ohm / SOC%")
    axes[1].set_xlabel("timestep")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    files0 = sorted((RAW_ROOT / "0C").glob("*.csv"))
    frames = [load_frame(p) for p in files0]
    by_profile = {str(f["profile"].iloc[0]): f for f in frames}
    all_frames = []
    summary_rows = []
    param_rows = []
    selection_tables = []
    for holdout in PROFILES:
        train_profiles = tuple(p for p in PROFILES if p != holdout)
        train_frames = [by_profile[p] for p in train_profiles]
        r0 = estimate_r0(train_frames)
        params, selection = select_params(train_frames, r0)
        selection["holdout"] = holdout
        selection_tables.append(selection)
        pred = apply_dynamic_rt(by_profile[holdout], r0, **params)
        pred["holdout"] = holdout
        pred["train_profiles"] = "+".join(train_profiles)
        for key, val in params.items():
            pred[f"param_{key}"] = val
        all_frames.append(pred)
        param_rows.append({"holdout": holdout, "train_profiles": "+".join(train_profiles), "r0": r0, **params})
        summary_rows.extend(metrics(pred, "V_corr_base_r0_ema120", "base_R0_EMA120", holdout))
        summary_rows.extend(metrics(pred, "V_corr_dynamic_rt_ema120", "dynamic_RT_variance", holdout))
        plot_holdout(pred, holdout, FIG_DIR / f"endzero_dynamic_rt_variance_{holdout}_0C_full.png")
        plot_tail(pred, holdout, FIG_DIR / f"endzero_dynamic_rt_variance_{holdout}_0C_tail.png")

    full = pd.concat(all_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(["scope", "holdout", "method"])
    params = pd.DataFrame(param_rows)
    selections = pd.concat(selection_tables, ignore_index=True)
    full_path = OUT_DIR / "endzero_dynamic_rt_variance_0C_full.csv.gz"
    summary_path = OUT_DIR / "endzero_dynamic_rt_variance_0C_summary.csv"
    params_path = OUT_DIR / "endzero_dynamic_rt_variance_0C_params.csv"
    selection_path = OUT_DIR / "endzero_dynamic_rt_variance_0C_inner_selection.csv"
    full.to_csv(full_path, index=False)
    summary.to_csv(summary_path, index=False)
    params.to_csv(params_path, index=False)
    selections.to_csv(selection_path, index=False)
    print("full", full_path)
    print("summary", summary_path)
    print("params", params_path)
    print("selection", selection_path)
    show = summary[summary["scope"].isin(["SOC<=10", "SOC<=20", "all"])].copy()
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
