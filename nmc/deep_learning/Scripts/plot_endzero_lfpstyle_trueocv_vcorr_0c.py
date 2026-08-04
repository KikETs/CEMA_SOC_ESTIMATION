#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "nmc_ocvstart_endzero_lopo_clean"
OUT_DIR = ROOT / "nmc_goal_vcorr_it_train_dst_selector_results" / "endzero_lfpstyle_trueocv_vcorr_only"
FIG_DIR = OUT_DIR / "figures"
OCV_PREP_DIR = Path("/home/user/바탕화면/DL/CEMA-TCN/Data")
OCV_REF_DIR = OCV_PREP_DIR / "raw_reference"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OCV_PREP_DIR))

import prepare_calce_nmc as prep  # noqa: E402
from soc_decomp.nmc_branchbands_experiment import (  # noqa: E402
    NMCBranchBandsConfig,
    _predict_lfp_style_vcorr,
    find_csv_files,
    parse_temperature,
)

from plot_ocvstart_lfpstyle_trueocv_vcorr_0c import (  # noqa: E402
    PROFILES,
    fit_lfpstyle_trueocv,
    load_frame,
)


def plot_voltage(full: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharey=True)
    for ax, profile in zip(axes.ravel(), PROFILES):
        g = full[full["profile"].eq(profile)].copy()
        x = np.arange(len(g), dtype=np.int64)
        ax.plot(x, g["V_raw"], color="#8a8a8a", lw=0.65, label="raw voltage")
        ax.plot(x, g["V_corr_lfpstyle_trueocv"], color="#0072B2", lw=0.9, label="corrected voltage")
        ax.plot(x, g["OCV_from_SOC"], color="#111111", lw=1.0, label="OCV(SOC 80->0)")
        ax.set_title(f"{profile} 0C, final SOC={g['SOC_pct'].iloc[-1]:.2f}%")
        ax.set_xlabel("timestep")
        ax.set_ylabel("Voltage (V)")
        ax.grid(alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("0C end-zero SOC label: raw vs corrected vs OCV(SOC)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_error(full: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharey=True)
    for ax, profile in zip(axes.ravel(), PROFILES):
        g = full[full["profile"].eq(profile)].copy()
        x = np.arange(len(g), dtype=np.int64)
        ax.axhline(0.0, color="#111111", lw=0.75)
        ax.plot(x, (g["V_raw"] - g["OCV_from_SOC"]) * 1000.0, color="#8a8a8a", lw=0.65, alpha=0.75, label="raw - OCV")
        ax.plot(x, (g["V_corr_lfpstyle_trueocv"] - g["OCV_from_SOC"]) * 1000.0, color="#0072B2", lw=0.9, label="corrected - OCV")
        ax.set_title(profile)
        ax.set_xlabel("timestep")
        ax.set_ylabel("Error (mV)")
        ax.grid(alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("0C end-zero SOC label voltage error", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_tail(full: pd.DataFrame, out_path: Path, n_tail: int = 2000) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharey=True)
    for ax, profile in zip(axes.ravel(), PROFILES):
        g = full[full["profile"].eq(profile)].copy().tail(n_tail)
        x = g["end_index"].to_numpy(np.int64)
        ax.plot(x, g["V_raw"], color="#8a8a8a", lw=0.65, label="raw voltage")
        ax.plot(x, g["V_corr_lfpstyle_trueocv"], color="#0072B2", lw=0.9, label="corrected voltage")
        ax.plot(x, g["OCV_from_SOC"], color="#111111", lw=1.0, label="OCV(SOC 80->0)")
        ax.set_title(f"{profile} tail, final SOC={g['SOC_pct'].iloc[-1]:.2f}%")
        ax.set_xlabel("timestep")
        ax.set_ylabel("Voltage (V)")
        ax.grid(alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(f"0C end-zero SOC label tail zoom, last {n_tail} samples", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    refs = prep.load_ocv_references(OCV_REF_DIR)
    files = [p for p in find_csv_files(RAW_ROOT) if parse_temperature(p, pd.read_csv(p, nrows=2)) in (0.0, 25.0)]
    files0 = [p for p in files if parse_temperature(p, pd.read_csv(p, nrows=2)) == 0.0]
    cfg = NMCBranchBandsConfig(
        base_dir=ROOT,
        raw_root=RAW_ROOT,
        lfpstyle_fit_stride=5,
        lfpstyle_fit_max_nfev=250,
        lfpstyle_v_floor_raw=2.45,
        lfpstyle_r0_max_ohm=0.22,
    )
    frames = [load_frame(p, refs) for p in files0]

    out_frames = []
    fit_rows = []
    for holdout in PROFILES:
        train_profiles = tuple(p for p in PROFILES if p != holdout)
        calibration, info = fit_lfpstyle_trueocv(frames, files, train_profiles, cfg)
        fit_rows.append({"holdout": holdout, **info})
        frame = next(f for f in frames if str(f["profile"].iloc[0]).upper() == holdout).copy()
        v_corr, parts = _predict_lfp_style_vcorr(
            np.asarray(calibration["params"], dtype=np.float64),
            frame["V_raw"].to_numpy(np.float64),
            frame["I_raw"].to_numpy(np.float64),
            frame["time_s"].to_numpy(np.float64),
            dict(calibration["scales"]),
            cfg,
            return_parts=True,
        )
        frame["V_corr_lfpstyle_trueocv"] = v_corr.astype(np.float64)
        frame["calibration_train_profiles"] = "+".join(train_profiles)
        for key, val in parts.items():
            frame[key] = np.asarray(val).astype(np.float64)
        out_frames.append(frame)

    full = pd.concat(out_frames, ignore_index=True)
    full_path = OUT_DIR / "endzero_lfpstyle_trueocv_vcorr_0C_full.csv.gz"
    summary_path = OUT_DIR / "endzero_lfpstyle_trueocv_vcorr_0C_summary.csv"
    fit_path = OUT_DIR / "endzero_lfpstyle_trueocv_vcorr_0C_train_fit.csv"
    full.to_csv(full_path, index=False)
    rows = []
    for profile, g in full.groupby("profile", sort=True):
        corr_err = (g["V_corr_lfpstyle_trueocv"].to_numpy(np.float64) - g["OCV_from_SOC"].to_numpy(np.float64)) * 1000.0
        raw_err = (g["V_raw"].to_numpy(np.float64) - g["OCV_from_SOC"].to_numpy(np.float64)) * 1000.0
        soc = g["SOC_pct"].to_numpy(np.float64)
        low20 = soc <= 20.0
        tail10 = soc <= 10.0
        rows.append(
            {
                "profile": profile,
                "rows": int(len(g)),
                "soc0_pct": float(g["SOC_pct"].iloc[0]),
                "final_soc_pct": float(g["SOC_pct"].iloc[-1]),
                "ocv_final_v": float(g["OCV_from_SOC"].iloc[-1]),
                "raw_mae_mV": float(np.nanmean(np.abs(raw_err))),
                "corr_mae_mV": float(np.nanmean(np.abs(corr_err))),
                "corr_bias_mV": float(np.nanmean(corr_err)),
                "raw_lowSOC_le20_mae_mV": float(np.nanmean(np.abs(raw_err[low20]))) if np.any(low20) else np.nan,
                "corr_lowSOC_le20_mae_mV": float(np.nanmean(np.abs(corr_err[low20]))) if np.any(low20) else np.nan,
                "raw_tail_le10_mae_mV": float(np.nanmean(np.abs(raw_err[tail10]))) if np.any(tail10) else np.nan,
                "corr_tail_le10_mae_mV": float(np.nanmean(np.abs(corr_err[tail10]))) if np.any(tail10) else np.nan,
            }
        )
    summary = pd.DataFrame(rows).sort_values("profile")
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(fit_rows).to_csv(fit_path, index=False)
    plot_voltage(full, FIG_DIR / "endzero_lfpstyle_trueocv_all_profiles_0C_full_traj_timestep.png")
    plot_error(full, FIG_DIR / "endzero_lfpstyle_trueocv_all_profiles_0C_full_traj_timestep_error.png")
    plot_tail(full, FIG_DIR / "endzero_lfpstyle_trueocv_all_profiles_0C_tail_zoom.png")
    print("full", full_path)
    print("summary", summary_path)
    print("fit", fit_path)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
