#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "nmc_ocvstart_endzero_lopo_clean"
OUT_DIR = ROOT / "nmc_goal_vcorr_it_train_dst_selector_results" / "endzero_rtvar_lowv_voltage_compare_0C"
OCV_PREP_DIR = Path("/home/user/바탕화면/DL/CEMA-TCN/Data")
OCV_REF_DIR = OCV_PREP_DIR / "raw_reference"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OCV_PREP_DIR))

import prepare_calce_nmc as prep  # noqa: E402
from soc_decomp.nmc_branchbands_experiment import (  # noqa: E402
    NMCBranchBandsConfig,
    build_feature_frames,
    estimate_r0_by_temperature,
    estimate_rtvar_lowv_calibration,
    find_csv_files,
)


PROFILES = ("VALIDATION", "DST", "FUDS", "US06")


def ocv_from_soc(refs: dict[float, object], temp: float, soc_frac: np.ndarray) -> np.ndarray:
    ref = refs.get(float(temp))
    if ref is None or ref.voltage_v is None or ref.soc_fraction is None:
        return np.full(len(soc_frac), np.nan, dtype=np.float64)
    ref_soc = np.asarray(ref.soc_fraction, dtype=np.float64)
    ref_v = np.asarray(ref.voltage_v, dtype=np.float64)
    order = np.argsort(ref_soc)
    ref_soc = ref_soc[order]
    ref_v = ref_v[order]
    keep = np.isfinite(ref_soc) & np.isfinite(ref_v)
    ref_soc = ref_soc[keep]
    ref_v = ref_v[keep]
    return np.interp(np.asarray(soc_frac, dtype=np.float64), ref_soc, ref_v).astype(np.float64)


def cfg_for_holdout(holdout: str) -> NMCBranchBandsConfig:
    return NMCBranchBandsConfig(
        base_dir=ROOT,
        raw_root=RAW_ROOT,
        train_profiles=tuple(p for p in PROFILES if p != holdout),
        test_profiles=(holdout,),
        v_corr_variant="rtvar_lowv_ema120",
        v_corr_tau_s=120.0,
    )


def plot_voltage(full: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=False, sharey=True)
    colors = {
        "V_raw": "#777777",
        "V_corr_raw": "#1f77b4",
        "OCV_from_SOC": "#111111",
    }
    labels = {
        "V_raw": "raw voltage",
        "V_corr_raw": "corrected voltage",
        "OCV_from_SOC": "OCV(SOC label)",
    }
    for ax, profile in zip(axes.ravel(), PROFILES):
        g = full[full["profile"].eq(profile)].copy()
        x = np.arange(len(g), dtype=np.int64)
        for col in ("V_raw", "V_corr_raw", "OCV_from_SOC"):
            ax.plot(x, g[col].to_numpy(np.float64), lw=0.9 if col != "V_raw" else 0.75, color=colors[col], label=labels[col])
        ax.set_title(f"{profile} 0C, final SOC={g['SOC_pct'].iloc[-1]:.2f}%")
        ax.set_xlabel("timestep")
        ax.set_ylabel("Voltage (V)")
        ax.grid(alpha=0.25)
    handles, labels_out = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels_out, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("0C endzero label: raw vs rtvar_lowv_ema120 corrected vs OCV(SOC)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_error(full: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=False, sharey=True)
    for ax, profile in zip(axes.ravel(), PROFILES):
        g = full[full["profile"].eq(profile)].copy()
        x = np.arange(len(g), dtype=np.int64)
        raw_err = (g["V_raw"].to_numpy(np.float64) - g["OCV_from_SOC"].to_numpy(np.float64)) * 1000.0
        corr_err = (g["V_corr_raw"].to_numpy(np.float64) - g["OCV_from_SOC"].to_numpy(np.float64)) * 1000.0
        ax.axhline(0.0, color="#111111", lw=0.8)
        ax.plot(x, raw_err, lw=0.75, color="#777777", label="raw - OCV")
        ax.plot(x, corr_err, lw=0.9, color="#1f77b4", label="corrected - OCV")
        ax.set_title(profile)
        ax.set_xlabel("timestep")
        ax.set_ylabel("Error (mV)")
        ax.grid(alpha=0.25)
    handles, labels_out = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels_out, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("0C endzero label voltage error", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_single(g: pd.DataFrame, out_path: Path) -> None:
    profile = str(g["profile"].iloc[0])
    x = np.arange(len(g), dtype=np.int64)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x, g["V_raw"].to_numpy(np.float64), lw=0.75, color="#777777", label="raw voltage")
    ax.plot(x, g["V_corr_raw"].to_numpy(np.float64), lw=0.9, color="#1f77b4", label="corrected voltage")
    ax.plot(x, g["OCV_from_SOC"].to_numpy(np.float64), lw=0.9, color="#111111", label="OCV(SOC label)")
    ax.set_title(f"{profile} 0C, final SOC={g['SOC_pct'].iloc[-1]:.2f}%")
    ax.set_xlabel("timestep")
    ax.set_ylabel("Voltage (V)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    refs = prep.load_ocv_references(OCV_REF_DIR)
    files = find_csv_files(RAW_ROOT)
    out_frames = []
    param_rows = []
    selection_rows = []

    for holdout in PROFILES:
        cfg = cfg_for_holdout(holdout)
        r0_df = estimate_r0_by_temperature(files, tuple(cfg.train_profiles))
        rtvar_cal = estimate_rtvar_lowv_calibration(files, tuple(cfg.train_profiles), r0_df, cfg)
        params = rtvar_cal["params_df"].copy()
        params.insert(0, "holdout", holdout)
        params.insert(1, "train_profiles", "+".join(cfg.train_profiles))
        param_rows.append(params)
        sel = rtvar_cal["selection_df"].copy()
        sel.insert(0, "holdout", holdout)
        sel.insert(1, "train_profiles", "+".join(cfg.train_profiles))
        selection_rows.append(sel)

        frames = build_feature_frames(cfg, files, r0_df, rtvar_lowv_calibration=rtvar_cal)
        test_0c = [
            frame.copy()
            for frame in frames["test"]
            if float(frame["temperature"].iloc[0]) == 0.0 and str(frame["drive_cycle"].iloc[0]).upper() == holdout
        ]
        if len(test_0c) != 1:
            raise RuntimeError(f"Expected one 0C frame for holdout {holdout}, got {len(test_0c)}")
        frame = test_0c[0]
        soc = np.clip(frame["SOC_physical"].to_numpy(np.float64), 0.0, 1.0)
        frame["profile"] = holdout
        frame["temperature_C"] = 0.0
        frame["SOC_frac"] = soc
        frame["SOC_pct"] = soc * 100.0
        frame["OCV_from_SOC"] = ocv_from_soc(refs, 0.0, soc)
        frame["calibration_train_profiles"] = "+".join(cfg.train_profiles)
        keep_cols = [
            "profile",
            "temperature_C",
            "calibration_train_profiles",
            "trajectory_id",
            "end_index",
            "SOC_frac",
            "SOC_pct",
            "V_raw",
            "V_corr_raw",
            "V_ohm_free_raw",
            "V_eq_slow_raw",
            "I_raw",
            "R0",
            "OCV_from_SOC",
        ]
        out_frames.append(frame[keep_cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True))

    full = pd.concat(out_frames, ignore_index=True)
    full_path = OUT_DIR / "endzero_rtvar_lowv_ema120_lopo_0C_voltage_full.csv.gz"
    full.to_csv(full_path, index=False)
    pd.concat(param_rows, ignore_index=True).to_csv(OUT_DIR / "endzero_rtvar_lowv_ema120_lopo_params.csv", index=False)
    pd.concat(selection_rows, ignore_index=True).to_csv(OUT_DIR / "endzero_rtvar_lowv_ema120_lopo_selection.csv", index=False)

    summary_rows = []
    for profile, g in full.groupby("profile", sort=True):
        raw_err = (g["V_raw"].to_numpy(np.float64) - g["OCV_from_SOC"].to_numpy(np.float64)) * 1000.0
        corr_err = (g["V_corr_raw"].to_numpy(np.float64) - g["OCV_from_SOC"].to_numpy(np.float64)) * 1000.0
        soc_pct = g["SOC_pct"].to_numpy(np.float64)
        low10 = soc_pct <= 10.0
        low20 = soc_pct <= 20.0
        summary_rows.append(
            {
                "profile": profile,
                "rows": int(len(g)),
                "soc_start_pct": float(soc_pct[0]),
                "soc_end_pct": float(soc_pct[-1]),
                "raw_mae_mV": float(np.nanmean(np.abs(raw_err))),
                "corr_mae_mV": float(np.nanmean(np.abs(corr_err))),
                "corr_bias_mV": float(np.nanmean(corr_err)),
                "raw_lowSOC_le20_mae_mV": float(np.nanmean(np.abs(raw_err[low20]))) if np.any(low20) else np.nan,
                "corr_lowSOC_le20_mae_mV": float(np.nanmean(np.abs(corr_err[low20]))) if np.any(low20) else np.nan,
                "raw_lowSOC_le10_mae_mV": float(np.nanmean(np.abs(raw_err[low10]))) if np.any(low10) else np.nan,
                "corr_lowSOC_le10_mae_mV": float(np.nanmean(np.abs(corr_err[low10]))) if np.any(low10) else np.nan,
                "raw_bias_mV": float(np.nanmean(raw_err)),
                "corr_p95_abs_mV": float(np.nanpercentile(np.abs(corr_err), 95)),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("profile").reset_index(drop=True)
    summary_path = OUT_DIR / "endzero_rtvar_lowv_ema120_lopo_0C_voltage_summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_voltage(full, fig_dir / "endzero_rtvar_lowv_ema120_lopo_all_profiles_0C_voltage_timestep.png")
    plot_error(full, fig_dir / "endzero_rtvar_lowv_ema120_lopo_all_profiles_0C_voltage_error_timestep.png")
    for profile, g in full.groupby("profile", sort=True):
        plot_single(g, fig_dir / f"endzero_rtvar_lowv_ema120_lopo_{profile}_0C_voltage_timestep.png")

    print(f"full_csv={full_path}")
    print(f"summary_csv={summary_path}")
    print(f"figure={fig_dir / 'endzero_rtvar_lowv_ema120_lopo_all_profiles_0C_voltage_timestep.png'}")
    print(f"error_figure={fig_dir / 'endzero_rtvar_lowv_ema120_lopo_all_profiles_0C_voltage_error_timestep.png'}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
