#!/usr/bin/env python3
"""NMC-calibrated corrected-voltage diagnostic for 0C SOC80 qmax data.

This is not a direct copy of the LFP MAMBA corrected-voltage constants.
For each heldout profile, the correction scale is derived from the remaining
0C NMC training profiles:

- soft voltage floor/ceiling from the train SOC-OCV voltage range
- correction cap from train (SOC-OCV voltage - V_raw) residuals
- voltage-knee gates from train SOC-OCV quantiles
- current dynamics from causal EMAs of I, |I|, dI, and signed hysteresis proxy

SOC is used only to create/evaluate the SOC-OCV voltage target that already
exists in the diagnostic CSV. SOC is not an input feature to the corrector.
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
PROFILES = ["VALIDATION", "DST", "FUDS", "US06"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


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


def soft_clamp(v: np.ndarray, floor: float, ceiling: float, beta: float = 25.0) -> np.ndarray:
    z_floor = (v - floor) * beta
    floored = np.where(
        z_floor > 40.0,
        v,
        np.where(z_floor < -40.0, floor + np.exp(z_floor) / beta, floor + np.log1p(np.exp(z_floor)) / beta),
    )
    z_ceil = (ceiling - floored) * beta
    return np.where(
        z_ceil > 40.0,
        floored,
        np.where(z_ceil < -40.0, ceiling - np.exp(z_ceil) / beta, ceiling - np.log1p(np.exp(z_ceil)) / beta),
    )


def build_records() -> dict[str, dict[str, np.ndarray]]:
    prev = pd.read_csv(PREV_FULL)
    target_by_index = {
        (str(row.profile).upper(), int(row["index"])): (
            float(row.SOC_OCV_inv_voltage),
            float(row.V_corr_raw),
        )
        for _, row in prev.iterrows()
    }
    records: dict[str, dict[str, np.ndarray]] = {}
    for profile in PROFILES:
        df = pd.read_csv(DATA / f"NMC_0C_{profile}.csv")
        index = np.arange(len(df))
        target = np.array([target_by_index[(profile, int(i))][0] for i in index], dtype=float)
        old_vcorr = np.array([target_by_index[(profile, int(i))][1] for i in index], dtype=float)
        times = df["Test_Time(s)"].to_numpy(float)
        if np.any(np.diff(times) < 0):
            times = df["Step_Time(s)"].to_numpy(float)
        records[profile] = {
            "index": index,
            "t": times,
            "v": df["Voltage(V)"].to_numpy(float),
            "i": df["Current(A)"].to_numpy(float),
            "soc": df["SOC_CC(%)"].to_numpy(float),
            "target": target,
            "old": old_vcorr,
        }
    return records


def make_scaler(records: dict[str, dict[str, np.ndarray]], train_profiles: list[str]) -> dict[str, float]:
    train_target = np.concatenate([records[p]["target"] for p in train_profiles])
    train_i = np.concatenate([records[p]["i"] for p in train_profiles])
    train_di = np.concatenate([np.r_[0.0, np.diff(records[p]["i"])] for p in train_profiles])
    scaler = {
        "vmin": float(np.percentile(train_target, 0.1)),
        "vmax": float(np.percentile(train_target, 99.9)),
        "iscale": float(np.percentile(np.abs(train_i), 99)) or 1.0,
        "discale": float(np.percentile(np.abs(train_di), 99)) or 1.0,
    }
    scaler["knee10"] = float(np.percentile(train_target, 12.5))
    scaler["knee20"] = float(np.percentile(train_target, 25.0))
    scaler["knee80"] = float(np.percentile(train_target, 95.0))
    scaler["slope10"] = max(0.04, float((np.percentile(train_target, 25.0) - np.percentile(train_target, 10.0)) / 2.0))
    scaler["slope20"] = max(0.06, float((np.percentile(train_target, 50.0) - np.percentile(train_target, 25.0)) / 3.0))
    return scaler


def build_features(rec: dict[str, np.ndarray], scaler: dict[str, float]) -> np.ndarray:
    v = rec["v"]
    i = rec["i"]
    t = rec["t"]
    di = np.r_[0.0, np.diff(i)]

    v_s = 2.0 * (v - scaler["vmin"]) / (scaler["vmax"] - scaler["vmin"]) - 1.0
    i_s = i / scaler["iscale"]
    abs_i_s = np.abs(i) / scaler["iscale"]
    di_s = di / scaler["discale"]
    abs_di_s = np.abs(di) / scaler["discale"]

    i_fast = causal_ema(i, t, 5.0) / scaler["iscale"]
    i_mid = causal_ema(i, t, 30.0) / scaler["iscale"]
    i_slow = causal_ema(i, t, 180.0) / scaler["iscale"]
    i_very_slow = causal_ema(i, t, 900.0) / scaler["iscale"]
    abs_i_mid = causal_ema(np.abs(i), t, 30.0) / scaler["iscale"]
    abs_i_slow = causal_ema(np.abs(i), t, 180.0) / scaler["iscale"]
    hys = causal_ema(np.sign(i) * np.sqrt(np.abs(i) + 1e-9), t, 900.0) / np.sqrt(scaler["iscale"])

    low20 = sigmoid((scaler["knee20"] - v) / scaler["slope20"])
    low10 = sigmoid((scaler["knee10"] - v) / scaler["slope10"])
    high80 = sigmoid((v - scaler["knee80"]) / 0.06)

    # NMC adjustment: low-voltage correction is allowed through current-amplitude
    # interactions only. A pure low-voltage branch overcorrects low-current VALIDATION.
    return np.column_stack(
        [
            np.ones_like(v),
            i_s,
            abs_i_s,
            di_s,
            abs_di_s,
            i_fast,
            i_mid,
            i_slow,
            i_very_slow,
            abs_i_mid,
            abs_i_slow,
            hys,
            v_s * i_s,
            v_s * i_mid,
            high80 * abs_i_s,
            low20 * abs_i_s,
            low10 * abs_i_s,
            low20 * abs_i_mid,
            low10 * abs_i_mid,
            low20 * i_s,
            low10 * i_s,
            low20 * i_mid,
            low10 * i_mid,
        ]
    )


def bounded_delta(x_scaled: np.ndarray, beta: np.ndarray, cap: float) -> np.ndarray:
    return cap * np.tanh((x_scaled @ beta) / cap)


def metrics(diff_v: np.ndarray, soc_pct: np.ndarray) -> dict[str, float]:
    abs_mv = np.abs(diff_v) * 1000.0
    signed_mv = diff_v * 1000.0
    out = {
        "overall_mae_mV": float(np.nanmean(abs_mv)),
        "overall_bias_mV": float(np.nanmean(signed_mv)),
        "rmse_mV": float(np.sqrt(np.nanmean(signed_mv**2))),
        "p95_abs_mV": float(np.nanpercentile(abs_mv, 95.0)),
        "max_abs_mV": float(np.nanmax(abs_mv)),
    }
    for lo, hi, name in [(0, 10, "soc0_10"), (0, 20, "soc0_20"), (20, 80, "soc20_80")]:
        mask = (soc_pct >= lo) & (soc_pct <= hi)
        out[f"{name}_mae_mV"] = float(np.nanmean(abs_mv[mask]))
        out[f"{name}_bias_mV"] = float(np.nanmean(signed_mv[mask]))
        out[f"{name}_rows"] = int(mask.sum())
    return out


def plot_full(full: pd.DataFrame, summary: pd.DataFrame, low_zoom: bool = False) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex="col")
    title = "0C NMC-calibrated V_corr vs SOC-OCV inverse voltage"
    if low_zoom:
        title = "0C NMC-calibrated V_corr low-SOC zoom"
    fig.suptitle(title, fontsize=14)
    for j, profile in enumerate(PROFILES):
        data = full[full["profile"] == profile].sort_values("index")
        if low_zoom:
            data = data[data["SOC_pct"] <= 25.0]
        row = summary[summary["heldout_profile"] == profile].iloc[0]

        ax = axes[0, j]
        ax.plot(data["SOC_pct"], data["SOC_OCV_inv_voltage"], color="black", lw=1.7, label="SOC-OCV voltage")
        ax.plot(data["SOC_pct"], data["V_corr_old"], color="#d55e00", lw=1.1, alpha=0.85, label="old V_corr")
        ax.plot(data["SOC_pct"], data["V_corr_nmc_calibrated"], color="#0072b2", lw=1.1, alpha=0.85, label="NMC-calibrated")
        ax.invert_xaxis()
        ax.grid(True, alpha=0.25)
        if low_zoom:
            ax.set_title(
                f"{profile} low0-10: old {row.old_soc0_10_mae_mV:.1f} -> NMC {row.nmc_soc0_10_mae_mV:.1f} mV"
            )
        else:
            ax.set_title(f"{profile}: old {row.old_overall_mae_mV:.1f} -> NMC {row.nmc_overall_mae_mV:.1f} mV")
        if j == 0:
            ax.set_ylabel("Voltage (V)")
        if j == 3:
            ax.legend(loc="best", fontsize=8)

        ax2 = axes[1, j]
        ax2.axhline(0.0, color="black", lw=0.8)
        ax2.plot(data["SOC_pct"], data["old_diff_mV"], color="#d55e00", lw=0.9, alpha=0.75)
        ax2.plot(data["SOC_pct"], data["nmc_diff_mV"], color="#0072b2", lw=0.9, alpha=0.75)
        ax2.set_ylim(-450.0, 450.0)
        ax2.invert_xaxis()
        ax2.grid(True, alpha=0.25)
        ax2.set_xlabel("SOC_CC (%)")
        if j == 0:
            ax2.set_ylabel("V_corr - SOC-OCV (mV)")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    suffix = "qmax_0c_nmc_calibrated_vcorr"
    if low_zoom:
        suffix += "_lowSOC_zoom"
    fig.savefig(OUT / f"{suffix}.png", dpi=180)
    fig.savefig(OUT / f"{suffix}.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = build_records()
    summary_rows = []
    full_rows = []
    param_rows = []

    for holdout in PROFILES:
        train_profiles = [p for p in PROFILES if p != holdout]
        scaler = make_scaler(records, train_profiles)
        x_train_parts = []
        y_train_parts = []
        for profile in train_profiles:
            rec = records[profile]
            x_train_parts.append(build_features(rec, scaler))
            y_train_parts.append(rec["target"] - rec["v"])
        x_train = np.vstack(x_train_parts)
        y_train = np.concatenate(y_train_parts)

        cap = max(0.12, min(0.95, float(np.percentile(np.abs(y_train), 99.2) * 1.15)))
        x_mean = x_train.mean(axis=0)
        x_std = x_train.std(axis=0) + 1e-8
        x_mean[0] = 0.0
        x_std[0] = 1.0
        x_train_scaled = (x_train - x_mean) / x_std

        lam = 0.15

        def objective(beta: np.ndarray) -> np.ndarray:
            residual = bounded_delta(x_train_scaled, beta, cap) - y_train
            return np.r_[residual, np.sqrt(lam) * beta]

        fit = least_squares(
            objective,
            np.zeros(x_train_scaled.shape[1], dtype=float),
            loss="soft_l1",
            f_scale=0.04,
            max_nfev=180,
        )

        rec = records[holdout]
        x_holdout = (build_features(rec, scaler) - x_mean) / x_std
        delta = bounded_delta(x_holdout, fit.x, cap)
        floor = max(2.45, scaler["vmin"] - 0.05)
        ceiling = min(4.25, scaler["vmax"] + 0.08)
        nmc_vcorr = soft_clamp(rec["v"] + delta, floor=floor, ceiling=ceiling, beta=25.0)
        nmc_diff = nmc_vcorr - rec["target"]
        old_diff = rec["old"] - rec["target"]

        row = {
            "heldout_profile": holdout,
            "train_profiles": "+".join(train_profiles),
            "nfev": int(fit.nfev),
            "success": bool(fit.success),
            "cost": float(fit.cost),
            "delta_cap_V": cap,
            "soft_floor_V": floor,
            "soft_ceiling_V": ceiling,
            "train_vmin_target_q001_V": scaler["vmin"],
            "train_vmax_target_q999_V": scaler["vmax"],
            "knee10_V": scaler["knee10"],
            "knee20_V": scaler["knee20"],
            "knee80_V": scaler["knee80"],
        }
        for key, value in metrics(nmc_diff, rec["soc"]).items():
            row[f"nmc_{key}"] = value
        for key, value in metrics(old_diff, rec["soc"]).items():
            row[f"old_{key}"] = value
        row["delta_overall_mae_mV"] = row["nmc_overall_mae_mV"] - row["old_overall_mae_mV"]
        row["delta_soc0_10_mae_mV"] = row["nmc_soc0_10_mae_mV"] - row["old_soc0_10_mae_mV"]
        row["delta_soc0_20_mae_mV"] = row["nmc_soc0_20_mae_mV"] - row["old_soc0_20_mae_mV"]
        summary_rows.append(row)

        param_rows.append(
            {
                "heldout_profile": holdout,
                **{f"beta{i}": float(v) for i, v in enumerate(fit.x)},
            }
        )

        full_rows.append(
            pd.DataFrame(
                {
                    "profile": holdout,
                    "index": rec["index"],
                    "SOC_pct": rec["soc"],
                    "V_raw": rec["v"],
                    "V_corr_old": rec["old"],
                    "V_corr_nmc_calibrated": nmc_vcorr,
                    "SOC_OCV_inv_voltage": rec["target"],
                    "old_diff_mV": old_diff * 1000.0,
                    "nmc_diff_mV": nmc_diff * 1000.0,
                    "nmc_delta_mV": delta * 1000.0,
                }
            )
        )

    summary = pd.DataFrame(summary_rows)
    full = pd.concat(full_rows, ignore_index=True)
    params = pd.DataFrame(param_rows)
    summary.to_csv(OUT / "qmax_0c_nmc_calibrated_vcorr_summary.csv", index=False)
    full.to_csv(OUT / "qmax_0c_nmc_calibrated_vcorr_full.csv", index=False)
    params.to_csv(OUT / "qmax_0c_nmc_calibrated_vcorr_params.csv", index=False)

    aggregate = pd.DataFrame(
        [
            {
                "old_overall_mae_mV": float(summary["old_overall_mae_mV"].mean()),
                "nmc_overall_mae_mV": float(summary["nmc_overall_mae_mV"].mean()),
                "delta_overall_mae_mV": float(summary["delta_overall_mae_mV"].mean()),
                "old_soc0_10_mae_mV": float(summary["old_soc0_10_mae_mV"].mean()),
                "nmc_soc0_10_mae_mV": float(summary["nmc_soc0_10_mae_mV"].mean()),
                "delta_soc0_10_mae_mV": float(summary["delta_soc0_10_mae_mV"].mean()),
                "old_soc0_20_mae_mV": float(summary["old_soc0_20_mae_mV"].mean()),
                "nmc_soc0_20_mae_mV": float(summary["nmc_soc0_20_mae_mV"].mean()),
                "delta_soc0_20_mae_mV": float(summary["delta_soc0_20_mae_mV"].mean()),
            }
        ]
    )
    aggregate.to_csv(OUT / "qmax_0c_nmc_calibrated_vcorr_aggregate.csv", index=False)

    plot_full(full, summary, low_zoom=False)
    plot_full(full, summary, low_zoom=True)

    cols = [
        "heldout_profile",
        "old_overall_mae_mV",
        "nmc_overall_mae_mV",
        "delta_overall_mae_mV",
        "old_soc0_10_mae_mV",
        "nmc_soc0_10_mae_mV",
        "delta_soc0_10_mae_mV",
        "old_soc0_20_mae_mV",
        "nmc_soc0_20_mae_mV",
        "delta_soc0_20_mae_mV",
        "delta_cap_V",
        "soft_floor_V",
        "soft_ceiling_V",
    ]
    print(summary[cols].to_string(index=False))
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
