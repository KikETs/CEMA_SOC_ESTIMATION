#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "nmc_ocvstart_lopo_clean"
OUT_DIR = ROOT / "nmc_goal_vcorr_it_train_dst_selector_results" / "ocvstart_lfpstyle_trueocv_vcorr_only"
FIG_DIR = OUT_DIR / "figures"
OCV_PREP_DIR = Path("/home/user/바탕화면/DL/CEMA-TCN/Data")
OCV_REF_DIR = OCV_PREP_DIR / "raw_reference"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OCV_PREP_DIR))

import prepare_calce_nmc as prep  # noqa: E402
from soc_decomp.nmc_branchbands_experiment import (  # noqa: E402
    NMCBranchBandsConfig,
    _lfp_style_features,
    _sigmoid_np,
    _soft_floor_np,
    find_csv_files,
    parse_temperature,
)

from plot_ocvstart_lfpstyle_trueocv_vcorr_0c import (  # noqa: E402
    PROFILES,
    fit_lfpstyle_trueocv,
    load_frame,
)


def predict_with_split_components(
    p: np.ndarray,
    frame: pd.DataFrame,
    scales: dict[str, float],
    cfg: NMCBranchBandsConfig,
) -> pd.DataFrame:
    v_raw = frame["V_raw"].to_numpy(np.float64)
    i_raw = frame["I_raw"].to_numpy(np.float64)
    times = frame["time_s"].to_numpy(np.float64)
    feat = _lfp_style_features(v_raw, i_raw, times, scales, cfg)
    p = np.asarray(p, dtype=np.float64)

    v_pol_fast = float(cfg.lfpstyle_pol_fast_limit_v) * np.tanh(p[0] * feat["fast_s"])
    v_pol_mid = float(cfg.lfpstyle_pol_mid_limit_v) * np.tanh(p[1] * feat["mid_s"])
    v_pol_slow = float(cfg.lfpstyle_pol_slow_limit_v) * np.tanh(p[2] * feat["slow_s"])
    v_pol_raw = np.clip(
        v_pol_fast + v_pol_mid + v_pol_slow,
        -float(cfg.lfpstyle_pol_limit_v),
        float(cfg.lfpstyle_pol_limit_v),
    )
    v_hys_raw = float(cfg.lfpstyle_hys_limit_v) * np.tanh(p[3] * feat["hys_s"])

    r0_min = float(getattr(cfg, "lfpstyle_r0_min_ohm", 0.0))
    r0_max = float(getattr(cfg, "lfpstyle_r0_max_ohm", 0.22))
    r0 = r0_min + (r0_max - r0_min) * _sigmoid_np(
        p[4] + p[5] * feat["absI_s"] + p[6] * feat["absdI_s"]
    )
    v_ohm_drop = r0 * feat["i_used"]

    g_corr = float(cfg.lfpstyle_g_corr_scale) * _sigmoid_np(
        p[7] + p[8] * feat["V_s"] + p[9] * feat["absI_s"] + p[10] * feat["absdI_s"]
    )
    v_drop_raw = g_corr * (v_pol_raw + v_hys_raw + v_ohm_drop)
    v_corr_pre = v_raw + v_drop_raw
    v_corr = _soft_floor_np(
        v_corr_pre,
        floor=float(cfg.lfpstyle_v_floor_raw),
        beta=float(cfg.lfpstyle_vfloor_beta),
    )

    out = frame.copy()
    out["V_corr_lfpstyle_trueocv"] = v_corr.astype(np.float64)
    out["V_corr_pre_floor"] = v_corr_pre.astype(np.float64)
    out["v_floor_lift"] = (v_corr - v_corr_pre).astype(np.float64)
    out["R0"] = r0.astype(np.float64)
    out["g_corr"] = g_corr.astype(np.float64)
    out["v_ohm_drop"] = v_ohm_drop.astype(np.float64)
    out["v_pol_fast_raw"] = v_pol_fast.astype(np.float64)
    out["v_pol_mid_raw"] = v_pol_mid.astype(np.float64)
    out["v_pol_slow_raw"] = v_pol_slow.astype(np.float64)
    out["v_pol_raw"] = v_pol_raw.astype(np.float64)
    out["v_hys_raw"] = v_hys_raw.astype(np.float64)
    out["v_drop_raw"] = v_drop_raw.astype(np.float64)
    out["g_v_ohm"] = (g_corr * v_ohm_drop).astype(np.float64)
    out["g_v_pol_fast"] = (g_corr * v_pol_fast).astype(np.float64)
    out["g_v_pol_mid"] = (g_corr * v_pol_mid).astype(np.float64)
    out["g_v_pol_slow"] = (g_corr * v_pol_slow).astype(np.float64)
    out["g_v_pol"] = (g_corr * v_pol_raw).astype(np.float64)
    out["g_v_hys"] = (g_corr * v_hys_raw).astype(np.float64)
    out["raw_err_mV"] = (out["V_raw"] - out["OCV_from_SOC"]) * 1000.0
    out["corr_err_mV"] = (out["V_corr_lfpstyle_trueocv"] - out["OCV_from_SOC"]) * 1000.0
    out["drop_mV"] = out["v_drop_raw"] * 1000.0
    return out


def metric_row(profile: str, g: pd.DataFrame) -> dict[str, float | int | str]:
    low20 = g["SOC_pct"].to_numpy(np.float64) <= 20.0
    tail10 = g["SOC_pct"].to_numpy(np.float64) <= 10.0
    row: dict[str, float | int | str] = {
        "profile": profile,
        "rows": int(len(g)),
        "soc0_pct": float(g["SOC_pct"].iloc[0]),
        "final_soc_pct": float(g["SOC_pct"].iloc[-1]),
        "raw_mae_mV": float(np.nanmean(np.abs(g["raw_err_mV"]))),
        "corr_mae_mV": float(np.nanmean(np.abs(g["corr_err_mV"]))),
        "corr_bias_mV": float(np.nanmean(g["corr_err_mV"])),
        "corr_lowSOC_le20_mae_mV": float(np.nanmean(np.abs(g.loc[low20, "corr_err_mV"]))) if np.any(low20) else np.nan,
        "corr_tail_le10_mae_mV": float(np.nanmean(np.abs(g.loc[tail10, "corr_err_mV"]))) if np.any(tail10) else np.nan,
        "R0_mean_ohm": float(np.nanmean(g["R0"])),
        "R0_p05_ohm": float(np.nanpercentile(g["R0"], 5)),
        "R0_p95_ohm": float(np.nanpercentile(g["R0"], 95)),
        "g_corr_mean": float(np.nanmean(g["g_corr"])),
        "drop_mean_mV": float(np.nanmean(g["drop_mV"])),
        "drop_abs_mean_mV": float(np.nanmean(np.abs(g["drop_mV"]))),
    }
    for col in [
        "g_v_ohm",
        "g_v_pol_fast",
        "g_v_pol_mid",
        "g_v_pol_slow",
        "g_v_pol",
        "g_v_hys",
        "v_floor_lift",
    ]:
        row[f"{col}_mean_mV"] = float(np.nanmean(g[col]) * 1000.0)
        row[f"{col}_abs_mean_mV"] = float(np.nanmean(np.abs(g[col])) * 1000.0)
    return row


def plot_components(full: pd.DataFrame, out_path: Path) -> None:
    colors = {
        "g_v_ohm": "#0072B2",
        "g_v_pol_fast": "#D55E00",
        "g_v_pol_mid": "#E69F00",
        "g_v_pol_slow": "#CC79A7",
        "g_v_hys": "#009E73",
        "v_drop_raw": "#111111",
    }
    fig, axes = plt.subplots(2, 2, figsize=(17, 9), sharey=True)
    for ax, profile in zip(axes.ravel(), PROFILES):
        g = full[full["profile"].eq(profile)].copy()
        x = np.arange(len(g), dtype=np.int64)
        for col, label in [
            ("g_v_ohm", "ohmic"),
            ("g_v_pol_fast", "pol fast"),
            ("g_v_pol_mid", "pol mid"),
            ("g_v_pol_slow", "pol slow"),
            ("g_v_hys", "hysteresis"),
        ]:
            ax.plot(x, g[col].to_numpy(np.float64) * 1000.0, lw=0.75, color=colors[col], label=label)
        ax.plot(x, g["v_drop_raw"].to_numpy(np.float64) * 1000.0, lw=1.0, color=colors["v_drop_raw"], label="total drop")
        ax.set_title(f"{profile} 0C")
        ax.set_xlabel("timestep")
        ax.set_ylabel("effective voltage contribution (mV)")
        ax.grid(alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False)
    fig.suptitle("LFP-reference-style NMC Vcorr decomposition, OCV-start label, 0C", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_voltage_and_error(full: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=False)
    for row, profile in enumerate(PROFILES):
        g = full[full["profile"].eq(profile)].copy()
        x = np.arange(len(g), dtype=np.int64)
        ax_v = axes[row, 0]
        ax_e = axes[row, 1]
        ax_v.plot(x, g["V_raw"], lw=0.65, color="#8a8a8a", label="raw")
        ax_v.plot(x, g["V_corr_lfpstyle_trueocv"], lw=0.9, color="#0072B2", label="corrected")
        ax_v.plot(x, g["OCV_from_SOC"], lw=0.9, color="#111111", label="OCV(SOC)")
        ax_v.set_title(f"{profile} voltage")
        ax_v.set_ylabel("V")
        ax_v.grid(alpha=0.25)
        ax_e.axhline(0.0, color="#111111", lw=0.65)
        ax_e.plot(x, g["raw_err_mV"], lw=0.55, color="#8a8a8a", alpha=0.75, label="raw-OCV")
        ax_e.plot(x, g["corr_err_mV"], lw=0.85, color="#0072B2", label="corrected-OCV")
        ax_e.set_title(f"{profile} error")
        ax_e.set_ylabel("mV")
        ax_e.grid(alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("timestep")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_tail_zoom(full: pd.DataFrame, out_path: Path, n_tail: int = 1800) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 9), sharey=False)
    for ax, profile in zip(axes.ravel(), PROFILES):
        g = full[full["profile"].eq(profile)].copy().tail(int(n_tail))
        x = g["end_index"].to_numpy(np.int64)
        ax2 = ax.twinx()
        ax.plot(x, g["corr_err_mV"], lw=0.9, color="#0072B2", label="corrected error")
        ax.plot(x, g["raw_err_mV"], lw=0.65, color="#8a8a8a", alpha=0.65, label="raw error")
        ax2.plot(x, g["SOC_pct"], lw=0.8, color="#D55E00", label="SOC")
        ax.axhline(0.0, color="#111111", lw=0.65)
        ax.set_title(f"{profile} tail")
        ax.set_xlabel("timestep")
        ax.set_ylabel("voltage error (mV)")
        ax2.set_ylabel("SOC (%)")
        ax.grid(alpha=0.25)
    fig.suptitle(f"0C tail zoom, last {n_tail} samples", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
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
    param_rows = []
    for holdout in PROFILES:
        train_profiles = tuple(p for p in PROFILES if p != holdout)
        calibration, info = fit_lfpstyle_trueocv(frames, files, train_profiles, cfg)
        fit_rows.append({"holdout": holdout, **info})
        p = np.asarray(calibration["params"], dtype=np.float64)
        row = {"holdout": holdout, "train_profiles": "+".join(train_profiles)}
        for name, val in zip(
            [
                "pol_fast_gain",
                "pol_mid_gain",
                "pol_slow_gain",
                "hys_gain",
                "r0_intercept",
                "r0_absI",
                "r0_absdI",
                "gate_intercept",
                "gate_Vs",
                "gate_absI",
                "gate_absdI",
            ],
            p,
        ):
            row[name] = float(val)
        row.update({f"scale_{k}": float(v) for k, v in dict(calibration["scales"]).items()})
        param_rows.append(row)
        frame = next(f for f in frames if str(f["profile"].iloc[0]).upper() == holdout).copy()
        decomp = predict_with_split_components(p, frame, dict(calibration["scales"]), cfg)
        decomp["calibration_train_profiles"] = "+".join(train_profiles)
        out_frames.append(decomp)

    full = pd.concat(out_frames, ignore_index=True)
    full_path = OUT_DIR / "ocvstart_lfpstyle_trueocv_decomposition_0C_full.csv.gz"
    summary_path = OUT_DIR / "ocvstart_lfpstyle_trueocv_decomposition_0C_summary.csv"
    params_path = OUT_DIR / "ocvstart_lfpstyle_trueocv_decomposition_0C_params.csv"
    fit_path = OUT_DIR / "ocvstart_lfpstyle_trueocv_decomposition_0C_train_fit.csv"
    full.to_csv(full_path, index=False)
    pd.DataFrame([metric_row(profile, g) for profile, g in full.groupby("profile", sort=True)]).to_csv(summary_path, index=False)
    pd.DataFrame(param_rows).to_csv(params_path, index=False)
    pd.DataFrame(fit_rows).to_csv(fit_path, index=False)

    plot_components(full, FIG_DIR / "ocvstart_lfpstyle_trueocv_decomposition_0C_components.png")
    plot_voltage_and_error(full, FIG_DIR / "ocvstart_lfpstyle_trueocv_decomposition_0C_voltage_error.png")
    plot_tail_zoom(full, FIG_DIR / "ocvstart_lfpstyle_trueocv_decomposition_0C_tail_zoom.png")

    summary = pd.read_csv(summary_path)
    print("full", full_path)
    print("summary", summary_path)
    print("params", params_path)
    print("fit", fit_path)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
