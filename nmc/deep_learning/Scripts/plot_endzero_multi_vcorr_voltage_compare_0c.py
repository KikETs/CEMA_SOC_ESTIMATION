#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "nmc_goal_vcorr_it_train_dst_selector_results"
OUT = RESULTS / "endzero_multi_vcorr_voltage_compare_0C"
FIG = OUT / "figures"

BASE_FULL = RESULTS / "nmc_tailored_vcorr_only" / "endzero_nmc_tailored_vcorr_only_full.csv.gz"
LFP_FULL = RESULTS / "endzero_lfpstyle_trueocv_vcorr_only" / "endzero_lfpstyle_trueocv_vcorr_0C_full.csv.gz"
DYN_FULL = RESULTS / "endzero_dynamic_rt_variance_only" / "endzero_dynamic_rt_variance_0C_full.csv.gz"
RTVAR_FULL = RESULTS / "endzero_rtvar_lowv_voltage_compare_0C" / "endzero_rtvar_lowv_ema120_lopo_0C_voltage_full.csv.gz"

PROFILES = ("VALIDATION", "DST", "FUDS", "US06")
VARIANTS = [
    ("V_raw", "raw", "#9a9a9a", "-", 0.75),
    ("V_corr_r0_noema", "R0 no EMA", "#d55e00", "-", 0.9),
    ("V_corr_r0_ema10", "R0 EMA10", "#e69f00", "-", 0.9),
    ("V_corr_r0_ema120", "R0 EMA120", "#0072b2", "-", 1.0),
    ("V_corr_dynamic_rt_ema120", "dynamic R(T,V,I)", "#009e73", "-", 0.95),
    ("V_corr_rtvar_lowv_ema120", "rtvar low-V", "#56b4e9", "-", 0.95),
    ("V_corr_lfpstyle_trueocv", "LFP-style", "#cc79a7", "-", 0.95),
    ("V_corr_nmc_tailored_innerloo_minimax_blend", "NMC minimax", "#f0e442", "-", 0.95),
    ("V_corr_nmc_tailored_minimax_aligned", "NMC aligned", "#7a3db8", "-", 0.95),
    ("V_corr_nmc_tailored_innerloo_vgate_blend", "NMC V-gated", "#000000", ":", 1.0),
]


def load_joined() -> pd.DataFrame:
    if not BASE_FULL.exists():
        raise FileNotFoundError(BASE_FULL)
    base = pd.read_csv(BASE_FULL)
    base = base[base["temperature_C"].astype(float).eq(0.0)].copy()
    key = ["profile", "temperature_C", "end_index"]
    keep = [
        *key,
        "SOC_pct",
        "SOC_frac",
        "time_s",
        "V_raw",
        "OCV_ref_from_SOC",
        "V_corr_r0_noema",
        "V_corr_r0_ema10",
        "V_corr_r0_ema120",
        "V_corr_nmc_tailored_innerloo_minimax_blend",
        "V_corr_nmc_tailored_minimax_aligned",
        "V_corr_nmc_tailored_innerloo_vgate_blend",
    ]
    full = base[keep].copy()
    full = full.rename(columns={"OCV_ref_from_SOC": "OCV_from_SOC"})

    joins = [
        (LFP_FULL, {"V_corr_lfpstyle_trueocv": "V_corr_lfpstyle_trueocv"}),
        (DYN_FULL, {"V_corr_dynamic_rt_ema120": "V_corr_dynamic_rt_ema120"}),
        (RTVAR_FULL, {"V_corr_raw": "V_corr_rtvar_lowv_ema120"}),
    ]
    for path, mapping in joins:
        if not path.exists():
            raise FileNotFoundError(path)
        other = pd.read_csv(path)
        other = other[other["temperature_C"].astype(float).eq(0.0)].copy()
        cols = [*key, *mapping.keys()]
        other = other[cols].rename(columns=mapping)
        full = full.merge(other, on=key, how="left", validate="one_to_one")

    required = ["OCV_from_SOC"] + [col for col, *_ in VARIANTS]
    missing = [col for col in required if col not in full.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")
    return full.replace([np.inf, -np.inf], np.nan).dropna(subset=required).reset_index(drop=True)


def summarize(full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for profile, g in full.groupby("profile", sort=True):
        ocv = g["OCV_from_SOC"].to_numpy(np.float64)
        low20 = g["SOC_pct"].to_numpy(np.float64) <= 20.0
        low10 = g["SOC_pct"].to_numpy(np.float64) <= 10.0
        for col, label, *_ in VARIANTS:
            err = (g[col].to_numpy(np.float64) - ocv) * 1000.0
            rows.append(
                {
                    "profile": profile,
                    "variant": label,
                    "column": col,
                    "mae_mV": float(np.nanmean(np.abs(err))),
                    "bias_mV": float(np.nanmean(err)),
                    "low20_mae_mV": float(np.nanmean(np.abs(err[low20]))) if np.any(low20) else np.nan,
                    "low10_mae_mV": float(np.nanmean(np.abs(err[low10]))) if np.any(low10) else np.nan,
                    "p95_abs_mV": float(np.nanpercentile(np.abs(err), 95)),
                    "rows": int(len(g)),
                }
            )
    return pd.DataFrame(rows).sort_values(["profile", "mae_mV"]).reset_index(drop=True)


def plot_panel(full: pd.DataFrame, out_path: Path, tail: bool = False) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 9.5), sharex=False, sharey=True)
    for ax, profile in zip(axes.ravel(), PROFILES):
        g = full[full["profile"].eq(profile)].copy()
        if tail:
            g = g[g["SOC_pct"].le(20.0)].copy()
        x = g["end_index"].to_numpy(np.int64)
        ax.plot(x, g["OCV_from_SOC"].to_numpy(np.float64), color="#111111", lw=1.4, ls="--", label="OCV(SOC)")
        for col, label, color, style, lw in VARIANTS:
            alpha = 0.62 if col == "V_raw" else 0.9
            ax.plot(x, g[col].to_numpy(np.float64), color=color, lw=lw, ls=style, alpha=alpha, label=label)
        title_tail = "0-20% SOC" if tail else "full trajectory"
        ax.set_title(f"{profile} 0C, {title_tail}")
        ax.set_xlabel("timestep")
        ax.set_ylabel("Voltage (V)")
        ax.grid(alpha=0.22)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=9)
    fig.suptitle("0C endzero label: multiple corrected-voltage variants vs OCV(SOC)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_error_panel(full: pd.DataFrame, out_path: Path, tail: bool = True) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 9.5), sharex=False, sharey=True)
    for ax, profile in zip(axes.ravel(), PROFILES):
        g = full[full["profile"].eq(profile)].copy()
        if tail:
            g = g[g["SOC_pct"].le(20.0)].copy()
        x = g["end_index"].to_numpy(np.int64)
        ocv = g["OCV_from_SOC"].to_numpy(np.float64)
        ax.axhline(0.0, color="#111111", lw=0.8)
        for col, label, color, style, lw in VARIANTS:
            err = (g[col].to_numpy(np.float64) - ocv) * 1000.0
            alpha = 0.55 if col == "V_raw" else 0.88
            ax.plot(x, err, color=color, lw=lw, ls=style, alpha=alpha, label=label)
        title_tail = "0-20% SOC" if tail else "full trajectory"
        ax.set_title(f"{profile} 0C error, {title_tail}")
        ax.set_xlabel("timestep")
        ax.set_ylabel("V - OCV (mV)")
        ax.grid(alpha=0.22)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=9)
    fig.suptitle("0C endzero label: corrected-voltage error vs OCV(SOC)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    full = load_joined()
    summary = summarize(full)
    full.to_csv(OUT / "endzero_multi_vcorr_0C_full.csv.gz", index=False)
    summary.to_csv(OUT / "endzero_multi_vcorr_0C_summary.csv", index=False)

    plot_panel(full, FIG / "endzero_multi_vcorr_0C_full_trajectory.png", tail=False)
    plot_panel(full, FIG / "endzero_multi_vcorr_0C_low20_tail.png", tail=True)
    plot_error_panel(full, FIG / "endzero_multi_vcorr_0C_low20_tail_error.png", tail=True)

    print(f"full_csv={OUT / 'endzero_multi_vcorr_0C_full.csv.gz'}")
    print(f"summary_csv={OUT / 'endzero_multi_vcorr_0C_summary.csv'}")
    print(f"figure_full={FIG / 'endzero_multi_vcorr_0C_full_trajectory.png'}")
    print(f"figure_tail={FIG / 'endzero_multi_vcorr_0C_low20_tail.png'}")
    print(f"figure_tail_error={FIG / 'endzero_multi_vcorr_0C_low20_tail_error.png'}")
    print(summary.groupby("variant", as_index=False)["mae_mV"].mean().sort_values("mae_mV").to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
