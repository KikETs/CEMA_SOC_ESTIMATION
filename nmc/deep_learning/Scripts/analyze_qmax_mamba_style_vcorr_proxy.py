#!/usr/bin/env python3
"""Build a MAMBA-style corrected-voltage proxy for 0C NMC SOC80 qmax data.

This script does not reproduce the unavailable trained LFP MAMBA corrector.
It keeps the same structural pieces described by the user:
fast/mid/slow polarization, hysteresis, R0*I ohmic drop, feature gate, and
soft voltage floor. Coefficients are fitted by LOPO over the four 0C profiles.
SOC is used only through the SOC-OCV voltage target for fitting/evaluation,
not as an input feature.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


BASE = Path("/home/user/바탕화면/DL/CEMA_MLP")
DATA = BASE / "nmc_soc80_train_nofloor_qmax" / "0C"
OUT = BASE / "nmc_voltage_diagnostics"
PREV_FULL = OUT / "qmax_0c_vcorr_raw_vs_soc_ocv_inv_voltage_full.csv"
PREV_SUM = OUT / "qmax_0c_vcorr_raw_vs_soc_ocv_inv_voltage_summary.csv"
PROFILES = ["VALIDATION", "DST", "FUDS", "US06"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def soft_floor(v: np.ndarray, floor: float = 2.02, beta: float = 20.0) -> np.ndarray:
    z = (v - floor) * beta
    out = np.empty_like(v, dtype=float)
    hi = z > 40.0
    lo = z < -40.0
    mid = ~(hi | lo)
    out[hi] = v[hi]
    out[lo] = floor + np.exp(z[lo]) / beta
    out[mid] = floor + np.log1p(np.exp(z[mid])) / beta
    return out


def causal_ema(values: np.ndarray, times: np.ndarray, tau_s: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    out = np.empty_like(values)
    if len(values) == 0:
        return out
    out[0] = values[0]
    prev_t = times[0]
    for i in range(1, len(values)):
        dt = times[i] - prev_t
        if not np.isfinite(dt) or dt < 0:
            dt = 1.0
        alpha = 1.0 - np.exp(-dt / tau_s)
        out[i] = out[i - 1] + alpha * (values[i] - out[i - 1])
        prev_t = times[i]
    return out


def logit(p: float) -> float:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return float(np.log(p / (1.0 - p)))


def build_records() -> dict[str, dict[str, np.ndarray]]:
    prev = pd.read_csv(PREV_FULL)
    prev_idx = {
        (str(r.profile).upper(), int(r["index"])): (
            float(r.SOC_OCV_inv_voltage),
            float(r.V_corr_raw),
        )
        for _, r in prev.iterrows()
    }

    records: dict[str, dict[str, np.ndarray]] = {}
    for profile in PROFILES:
        df = pd.read_csv(DATA / f"NMC_0C_{profile}.csv")
        idx = np.arange(len(df))
        target = np.array([prev_idx[(profile, int(i))][0] for i in idx], dtype=float)
        old_vcorr = np.array([prev_idx[(profile, int(i))][1] for i in idx], dtype=float)
        t = df["Test_Time(s)"].to_numpy(float)
        if np.any(np.diff(t) < 0):
            t = df["Step_Time(s)"].to_numpy(float)
        records[profile] = {
            "index": idx,
            "t": t,
            "v": df["Voltage(V)"].to_numpy(float),
            "i": df["Current(A)"].to_numpy(float),
            "soc": df["SOC_CC(%)"].to_numpy(float),
            "target": target,
            "old": old_vcorr,
        }
    return records


def make_features(
    rec: dict[str, np.ndarray],
    convention: str,
    vmin: float,
    vmax: float,
    i_scale: float,
    di_scale: float,
) -> dict[str, np.ndarray]:
    i_raw = rec["i"]
    if convention == "literal_signed":
        i_used = i_raw.copy()
    elif convention == "discharge_positive":
        i_used = -i_raw.copy()
    else:
        raise ValueError(convention)

    di_used = np.r_[0.0, np.diff(i_used)]
    fast = causal_ema(i_used, rec["t"], 20.0)
    mid = causal_ema(i_used, rec["t"], 120.0)
    slow = causal_ema(i_used, rec["t"], 600.0)
    hys_drive = np.sign(i_used) * np.sqrt(np.abs(i_used) + 1e-9)
    hys = causal_ema(hys_drive, rec["t"], 1200.0)
    v_s = 2.0 * (rec["v"] - vmin) / (vmax - vmin) - 1.0
    return {
        "i_used": i_used,
        "absI_s": np.clip(np.abs(i_used) / i_scale, 0.0, 3.0),
        "absdI_s": np.clip(np.abs(di_used) / di_scale, 0.0, 3.0),
        "V_s": np.clip(v_s, -2.0, 2.0),
        "fast_s": np.clip(fast / i_scale, -3.0, 3.0),
        "mid_s": np.clip(mid / i_scale, -3.0, 3.0),
        "slow_s": np.clip(slow / i_scale, -3.0, 3.0),
        "hys_s": np.clip(hys / np.sqrt(i_scale), -3.0, 3.0),
    }


def predict(
    p: np.ndarray,
    rec: dict[str, np.ndarray],
    feat: dict[str, np.ndarray],
    return_parts: bool = False,
):
    v_pf = 0.16 * np.tanh(p[0] * feat["fast_s"])
    v_pm = 0.12 * np.tanh(p[1] * feat["mid_s"])
    v_ps = 0.10 * np.tanh(p[2] * feat["slow_s"])
    v_pol = np.clip(v_pf + v_pm + v_ps, -0.25, 0.25)
    v_hys = 0.20 * np.tanh(p[3] * feat["hys_s"])
    r0 = 0.16 * sigmoid(p[4] + p[5] * feat["absI_s"] + p[6] * feat["absdI_s"])
    v_ohm = r0 * feat["i_used"]
    gate = 1.35 * sigmoid(
        p[7] + p[8] * feat["V_s"] + p[9] * feat["absI_s"] + p[10] * feat["absdI_s"]
    )
    v_drop = gate * (v_pol + v_hys + v_ohm)
    vcorr = soft_floor(rec["v"] + v_drop, floor=2.02, beta=20.0)
    if not return_parts:
        return vcorr
    return vcorr, {
        "v_pol": v_pol,
        "v_hys": v_hys,
        "v_ohm": v_ohm,
        "g_corr": gate,
        "R0_ohm": r0,
        "v_drop": v_drop,
    }


def metrics(diff_v: np.ndarray, soc_pct: np.ndarray) -> dict[str, float]:
    abs_mv = np.abs(diff_v) * 1000.0
    out = {
        "overall_mae_mV": float(np.nanmean(abs_mv)),
        "overall_bias_mV": float(np.nanmean(diff_v) * 1000.0),
        "rmse_mV": float(np.sqrt(np.nanmean((diff_v * 1000.0) ** 2))),
        "p95_abs_mV": float(np.nanpercentile(abs_mv, 95)),
        "max_abs_mV": float(np.nanmax(abs_mv)),
    }
    for lo, hi, name in [(0, 10, "soc0_10"), (0, 20, "soc0_20"), (20, 80, "soc20_80")]:
        mask = (soc_pct >= lo) & (soc_pct <= hi)
        out[f"{name}_mae_mV"] = float(np.nanmean(abs_mv[mask]))
        out[f"{name}_bias_mV"] = float(np.nanmean(diff_v[mask]) * 1000.0)
        out[f"{name}_rows"] = int(mask.sum())
    return out


def plot_variant(full: pd.DataFrame, summary: pd.DataFrame, variant: str, low_zoom: bool = False) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex="col")
    title = f"0C NMC SOC80 qmax: MAMBA-style V_corr proxy ({variant})"
    if low_zoom:
        title = f"0C low-SOC zoom: MAMBA-style V_corr proxy ({variant})"
    fig.suptitle(title, fontsize=14)
    for j, profile in enumerate(PROFILES):
        data = full[(full["variant"] == variant) & (full["profile"] == profile)].sort_values("index")
        if low_zoom:
            data = data[data["SOC_pct"] <= 25.0]
        ax = axes[0, j]
        ax.plot(data["SOC_pct"], data["SOC_OCV_inv_voltage"], color="black", lw=1.7, label="SOC-OCV voltage")
        ax.plot(data["SOC_pct"], data["V_corr_old"], color="#d55e00", lw=1.1, alpha=0.85, label="old V_corr")
        ax.plot(
            data["SOC_pct"],
            data["V_corr_mamba_style_proxy"],
            color="#0072b2",
            lw=1.1,
            alpha=0.85,
            label="MAMBA-style proxy",
        )
        ax.invert_xaxis()
        ax.grid(True, alpha=0.25)
        row = summary[(summary["variant"] == variant) & (summary["heldout_profile"] == profile)].iloc[0]
        if low_zoom:
            title = f"{profile} low0-10: old {row.old_soc0_10_mae_mV:.1f} -> proxy {row.proxy_soc0_10_mae_mV:.1f} mV"
        else:
            title = f"{profile}: old {row.old_overall_mae_mV:.1f} -> proxy {row.proxy_overall_mae_mV:.1f} mV"
        ax.set_title(title)
        if j == 0:
            ax.set_ylabel("Voltage (V)")
        if j == 3:
            ax.legend(loc="best", fontsize=8)

        ax2 = axes[1, j]
        ax2.axhline(0, color="black", lw=0.8)
        ax2.plot(data["SOC_pct"], data["old_diff_mV"], color="#d55e00", lw=0.9, alpha=0.75)
        ax2.plot(data["SOC_pct"], data["proxy_diff_mV"], color="#0072b2", lw=0.9, alpha=0.75)
        ax2.set_ylim(-450, 450)
        ax2.invert_xaxis()
        ax2.grid(True, alpha=0.25)
        ax2.set_xlabel("SOC_CC (%)")
        if j == 0:
            ax2.set_ylabel("V_corr - SOC-OCV (mV)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    suffix = f"qmax_0c_mamba_style_vcorr_proxy_{variant}"
    if low_zoom:
        suffix += "_lowSOC_zoom"
    fig.savefig(OUT / f"{suffix}.png", dpi=180)
    fig.savefig(OUT / f"{suffix}.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = build_records()
    all_v = np.concatenate([records[p]["v"] for p in PROFILES])
    all_abs_i = np.concatenate([np.abs(records[p]["i"]) for p in PROFILES])
    all_abs_di = np.concatenate([np.abs(np.r_[0.0, np.diff(records[p]["i"])]) for p in PROFILES])
    vmin, vmax = float(np.nanmin(all_v)), float(np.nanmax(all_v))
    i_scale = float(np.nanpercentile(all_abs_i, 99)) or 1.0
    di_scale = float(np.nanpercentile(all_abs_di, 99)) or 1.0

    variants = ["literal_signed", "discharge_positive"]
    features = {
        variant: {
            p: make_features(records[p], variant, vmin, vmax, i_scale, di_scale)
            for p in PROFILES
        }
        for variant in variants
    }

    x0 = np.array(
        [0.8, 0.8, 0.8, 0.5, logit(0.10 / 0.16), 0.0, 0.0, logit(1.0 / 1.35), 0.0, 0.0, 0.0],
        dtype=float,
    )
    lower = np.array([-8.0] * len(x0), dtype=float)
    upper = np.array([8.0] * len(x0), dtype=float)

    summary_rows = []
    full_frames = []
    param_rows = []
    for variant in variants:
        for holdout in PROFILES:
            train_profiles = [p for p in PROFILES if p != holdout]
            train_records = [records[p] for p in train_profiles]
            train_features = [features[variant][p] for p in train_profiles]

            def objective(params: np.ndarray) -> np.ndarray:
                return np.concatenate(
                    [
                        predict(params, rec, feat) - rec["target"]
                        for rec, feat in zip(train_records, train_features)
                    ]
                )

            fit = least_squares(
                objective,
                x0,
                bounds=(lower, upper),
                loss="soft_l1",
                f_scale=0.03,
                max_nfev=250,
            )
            params = fit.x
            rec = records[holdout]
            proxy, parts = predict(params, rec, features[variant][holdout], return_parts=True)
            proxy_diff = proxy - rec["target"]
            old_diff = rec["old"] - rec["target"]

            row = {
                "variant": variant,
                "heldout_profile": holdout,
                "train_profiles": "+".join(train_profiles),
                "nfev": int(fit.nfev),
                "success": bool(fit.success),
                "cost": float(fit.cost),
                "mean_vdrop_mV": float(np.nanmean(parts["v_drop"]) * 1000.0),
                "mean_gate": float(np.nanmean(parts["g_corr"])),
                "mean_r0_ohm": float(np.nanmean(parts["R0_ohm"])),
            }
            for key, value in metrics(proxy_diff, rec["soc"]).items():
                row[f"proxy_{key}"] = value
            for key, value in metrics(old_diff, rec["soc"]).items():
                row[f"old_{key}"] = value
            row["delta_overall_mae_mV"] = row["proxy_overall_mae_mV"] - row["old_overall_mae_mV"]
            row["delta_soc0_10_mae_mV"] = row["proxy_soc0_10_mae_mV"] - row["old_soc0_10_mae_mV"]
            row["delta_soc0_20_mae_mV"] = row["proxy_soc0_20_mae_mV"] - row["old_soc0_20_mae_mV"]
            summary_rows.append(row)

            param_rows.append(
                {
                    "variant": variant,
                    "heldout_profile": holdout,
                    **{f"p{i}": float(v) for i, v in enumerate(params)},
                }
            )
            full_frames.append(
                pd.DataFrame(
                    {
                        "variant": variant,
                        "profile": holdout,
                        "index": rec["index"],
                        "SOC_pct": rec["soc"],
                        "V_raw": rec["v"],
                        "V_corr_old": rec["old"],
                        "V_corr_mamba_style_proxy": proxy,
                        "SOC_OCV_inv_voltage": rec["target"],
                        "old_diff_mV": old_diff * 1000.0,
                        "proxy_diff_mV": proxy_diff * 1000.0,
                        "v_drop_mV": parts["v_drop"] * 1000.0,
                        "v_pol_mV": parts["v_pol"] * 1000.0,
                        "v_hys_mV": parts["v_hys"] * 1000.0,
                        "v_ohm_mV": parts["v_ohm"] * 1000.0,
                        "g_corr": parts["g_corr"],
                        "R0_ohm": parts["R0_ohm"],
                    }
                )
            )

    summary = pd.DataFrame(summary_rows)
    full = pd.concat(full_frames, ignore_index=True)
    params = pd.DataFrame(param_rows)
    summary.to_csv(OUT / "qmax_0c_mamba_style_vcorr_proxy_summary.csv", index=False)
    full.to_csv(OUT / "qmax_0c_mamba_style_vcorr_proxy_full.csv", index=False)
    params.to_csv(OUT / "qmax_0c_mamba_style_vcorr_proxy_params.csv", index=False)

    rank = (
        summary.groupby("variant")
        .agg(
            proxy_mae=("proxy_overall_mae_mV", "mean"),
            old_mae=("old_overall_mae_mV", "mean"),
            proxy_low=("proxy_soc0_10_mae_mV", "mean"),
            old_low=("old_soc0_10_mae_mV", "mean"),
        )
        .reset_index()
    )
    rank["delta"] = rank["proxy_mae"] - rank["old_mae"]
    rank.to_csv(OUT / "qmax_0c_mamba_style_vcorr_proxy_variant_rank.csv", index=False)

    for variant in variants:
        plot_variant(full, summary, variant, low_zoom=False)
    best_variant = str(rank.sort_values("proxy_mae").iloc[0]["variant"])
    plot_variant(full, summary, best_variant, low_zoom=True)

    print(rank.to_string(index=False))
    cols = [
        "variant",
        "heldout_profile",
        "old_overall_mae_mV",
        "proxy_overall_mae_mV",
        "delta_overall_mae_mV",
        "old_soc0_10_mae_mV",
        "proxy_soc0_10_mae_mV",
        "delta_soc0_10_mae_mV",
        "mean_vdrop_mV",
        "mean_gate",
        "mean_r0_ohm",
    ]
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
