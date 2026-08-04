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
OUT_DIR = ROOT / "nmc_goal_vcorr_it_train_dst_selector_results" / "ocvstart_lfpstyle_trueocv_vcorr_only"
OCV_PREP_DIR = Path("/home/user/바탕화면/DL/CEMA-TCN/Data")
OCV_REF_DIR = OCV_PREP_DIR / "raw_reference"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OCV_PREP_DIR))

import prepare_calce_nmc as prep  # noqa: E402
from soc_decomp.nmc_branchbands_experiment import (  # noqa: E402
    NMCBranchBandsConfig,
    _predict_lfp_style_vcorr,
    estimate_r0_by_temperature,
    find_csv_files,
    parse_profile,
    parse_temperature,
)


PROFILES = ("VALIDATION", "DST", "FUDS", "US06")


def logit(x: float) -> float:
    x = float(np.clip(x, 1e-6, 1.0 - 1e-6))
    return float(np.log(x / (1.0 - x)))


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
    ref_soc = ref_soc[keep]
    ref_v = ref_v[keep]
    return np.interp(np.asarray(soc, dtype=np.float64), ref_soc, ref_v).astype(np.float64)


def load_frame(path: Path, refs: dict[float, object]) -> pd.DataFrame:
    df = pd.read_csv(path)
    temp = float(parse_temperature(path, df))
    profile = str(parse_profile(path, df)).upper()
    time_col = "Step_Time(s)" if "Step_Time(s)" in df.columns else "Test_Time(s)"
    soc = soc01(df)
    out = pd.DataFrame(
        {
            "file_name": path.name,
            "profile": profile,
            "temperature_C": temp,
            "end_index": np.arange(len(df), dtype=np.int64),
            "time_s": pd.to_numeric(df[time_col], errors="coerce").to_numpy(np.float64),
            "SOC_frac": soc,
            "SOC_pct": soc * 100.0,
            "V_raw": pd.to_numeric(df["Voltage(V)"], errors="coerce").to_numpy(np.float64),
            "I_raw": pd.to_numeric(df["Current(A)"], errors="coerce").to_numpy(np.float64),
        }
    )
    out["OCV_from_SOC"] = ocv_from_soc(refs, temp, out["SOC_frac"].to_numpy(np.float64))
    return out.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["time_s", "SOC_frac", "V_raw", "I_raw", "OCV_from_SOC"]
    ).reset_index(drop=True)


def fit_lfpstyle_trueocv(
    frames: list[pd.DataFrame],
    files: list[Path],
    train_profiles: tuple[str, ...],
    cfg: NMCBranchBandsConfig,
) -> tuple[dict[str, object], dict[str, float]]:
    r0_df = estimate_r0_by_temperature(files, train_profiles)
    r0_rows = r0_df[r0_df["temperature_C"].astype(float).eq(0.0)]
    r0_base = float(r0_rows["r0_ohm"].iloc[0]) if len(r0_rows) else 0.08

    train = [f for f in frames if str(f["profile"].iloc[0]).upper() in train_profiles and float(f["temperature_C"].iloc[0]) == 0.0]
    if not train:
        raise RuntimeError(f"No 0C train frames for {train_profiles}")

    v_all = np.concatenate([f["V_raw"].to_numpy(np.float64) for f in train])
    i_all = np.concatenate([f["I_raw"].to_numpy(np.float64) for f in train])
    di_all = np.diff(-i_all, prepend=-i_all[0])
    scales = {
        "vmin": float(np.nanpercentile(v_all, 1)),
        "vmax": float(np.nanpercentile(v_all, 99)),
        "i_scale": float(max(np.nanpercentile(np.abs(i_all), 95), 1e-3)),
        "di_scale": float(max(np.nanpercentile(np.abs(di_all), 95), 1e-3)),
    }
    r0_frac = np.clip(
        (r0_base - float(cfg.lfpstyle_r0_min_ohm))
        / max(float(cfg.lfpstyle_r0_max_ohm) - float(cfg.lfpstyle_r0_min_ohm), 1e-9),
        1e-4,
        1.0 - 1e-4,
    )
    p0 = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            logit(float(r0_frac)),
            0.0,
            0.0,
            logit(min(1.0 / max(float(cfg.lfpstyle_g_corr_scale), 1e-6), 0.98)),
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    lower = np.array([-4.0, -4.0, -4.0, -4.0, -8.0, -5.0, -5.0, -8.0, -5.0, -5.0, -5.0], dtype=np.float64)
    upper = np.array([4.0, 4.0, 4.0, 4.0, 8.0, 5.0, 5.0, 8.0, 5.0, 5.0, 5.0], dtype=np.float64)
    stride = max(int(cfg.lfpstyle_fit_stride), 1)
    fit_parts = []
    for f in train:
        idx = np.arange(0, len(f), stride)
        fit_parts.append(
            {
                "v_raw": f["V_raw"].to_numpy(np.float64)[idx],
                "i_raw": f["I_raw"].to_numpy(np.float64)[idx],
                "times": f["time_s"].to_numpy(np.float64)[idx],
                "soc": f["SOC_frac"].to_numpy(np.float64)[idx],
                "ocv": f["OCV_from_SOC"].to_numpy(np.float64)[idx],
            }
        )

    def residual(p: np.ndarray) -> np.ndarray:
        chunks = []
        for rec in fit_parts:
            pred = _predict_lfp_style_vcorr(p, rec["v_raw"], rec["i_raw"], rec["times"], scales, cfg).astype(np.float64)
            err = pred - rec["ocv"]
            weight = 1.0 + 0.5 * (rec["soc"] <= 0.20).astype(np.float64)
            chunks.append(err * weight)
        reg = np.array(
            [0.01 * p[0], 0.01 * p[1], 0.01 * p[2], 0.01 * p[3], 0.005 * p[5], 0.005 * p[6], 0.005 * p[8], 0.005 * p[9], 0.005 * p[10]],
            dtype=np.float64,
        )
        return np.concatenate(chunks + [reg])

    before = residual(p0)
    opt = least_squares(
        residual,
        p0,
        bounds=(lower, upper),
        max_nfev=int(cfg.lfpstyle_fit_max_nfev),
        loss="soft_l1",
        f_scale=0.02,
    )
    after = residual(opt.x)
    info = {
        "train_profiles": "+".join(train_profiles),
        "r0_base_ohm": r0_base,
        "n_fit_points": int(sum(len(r["v_raw"]) for r in fit_parts)),
        "train_weighted_mae_before_mV": float(np.nanmean(np.abs(before)) * 1000.0),
        "train_weighted_mae_after_mV": float(np.nanmean(np.abs(after)) * 1000.0),
        "cost": float(opt.cost),
        "status": int(opt.status),
    }
    return {"params": opt.x.astype(np.float64), "scales": scales}, info


def plot_combined(full: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=False, sharey=True)
    for ax, profile in zip(axes.ravel(), PROFILES):
        g = full[full["profile"].eq(profile)].copy()
        x = np.arange(len(g), dtype=np.int64)
        ax.plot(x, g["V_raw"], lw=0.8, color="#7c7c7c", label="raw voltage")
        ax.plot(x, g["V_corr_lfpstyle_trueocv"], lw=1.0, color="#1f77b4", label="corrected voltage")
        ax.plot(x, g["OCV_from_SOC"], lw=1.0, color="#111111", label="OCV(SOC label)")
        ax.set_title(f"{profile} 0C, final SOC={g['SOC_pct'].iloc[-1]:.2f}%")
        ax.set_xlabel("timestep")
        ax.set_ylabel("Voltage (V)")
        ax.grid(alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("0C OCV-start label: raw vs LFP-reference-style corrected vs OCV(SOC)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_error(full: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=False, sharey=True)
    for ax, profile in zip(axes.ravel(), PROFILES):
        g = full[full["profile"].eq(profile)].copy()
        x = np.arange(len(g), dtype=np.int64)
        ax.axhline(0.0, color="#111111", lw=0.8)
        ax.plot(x, (g["V_raw"] - g["OCV_from_SOC"]) * 1000.0, lw=0.8, color="#7c7c7c", label="raw - OCV")
        ax.plot(x, (g["V_corr_lfpstyle_trueocv"] - g["OCV_from_SOC"]) * 1000.0, lw=1.0, color="#1f77b4", label="corrected - OCV")
        ax.set_title(profile)
        ax.set_xlabel("timestep")
        ax.set_ylabel("Error (mV)")
        ax.grid(alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("0C OCV-start label voltage error", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_single(g: pd.DataFrame, out_path: Path) -> None:
    g = g.copy()
    profile = str(g["profile"].iloc[0])
    x = np.arange(len(g), dtype=np.int64)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x, g["V_raw"], lw=0.8, color="#7c7c7c", label="raw voltage")
    ax.plot(x, g["V_corr_lfpstyle_trueocv"], lw=1.0, color="#1f77b4", label="corrected voltage")
    ax.plot(x, g["OCV_from_SOC"], lw=1.0, color="#111111", label="OCV(SOC label)")
    ax.set_title(f"{profile} 0C, OCV-start label, final SOC={g['SOC_pct'].iloc[-1]:.2f}%")
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
    files = [p for p in find_csv_files(RAW_ROOT) if parse_temperature(p, pd.read_csv(p, nrows=2)) in (0.0, 25.0)]
    files0 = [p for p in files if parse_temperature(p, pd.read_csv(p, nrows=2)) == 0.0]
    cfg = NMCBranchBandsConfig(base_dir=ROOT, raw_root=RAW_ROOT, lfpstyle_fit_stride=5, lfpstyle_fit_max_nfev=250)
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
    full_path = OUT_DIR / "ocvstart_lfpstyle_trueocv_vcorr_0C_full.csv.gz"
    full.to_csv(full_path, index=False)

    summary_rows = []
    for profile, g in full.groupby("profile", sort=True):
        corr_err = (g["V_corr_lfpstyle_trueocv"].to_numpy(np.float64) - g["OCV_from_SOC"].to_numpy(np.float64)) * 1000.0
        raw_err = (g["V_raw"].to_numpy(np.float64) - g["OCV_from_SOC"].to_numpy(np.float64)) * 1000.0
        low = g["SOC_pct"].to_numpy(np.float64) <= 20.0
        tail = g["SOC_pct"].to_numpy(np.float64) <= 10.0
        summary_rows.append(
            {
                "profile": profile,
                "rows": int(len(g)),
                "soc0_pct": float(g["SOC_pct"].iloc[0]),
                "final_soc_pct": float(g["SOC_pct"].iloc[-1]),
                "raw_mae_mV": float(np.nanmean(np.abs(raw_err))),
                "corr_mae_mV": float(np.nanmean(np.abs(corr_err))),
                "corr_bias_mV": float(np.nanmean(corr_err)),
                "raw_lowSOC_le20_mae_mV": float(np.nanmean(np.abs(raw_err[low]))) if np.any(low) else np.nan,
                "corr_lowSOC_le20_mae_mV": float(np.nanmean(np.abs(corr_err[low]))) if np.any(low) else np.nan,
                "raw_tail_le10_mae_mV": float(np.nanmean(np.abs(raw_err[tail]))) if np.any(tail) else np.nan,
                "corr_tail_le10_mae_mV": float(np.nanmean(np.abs(corr_err[tail]))) if np.any(tail) else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("profile")
    summary.to_csv(OUT_DIR / "ocvstart_lfpstyle_trueocv_vcorr_0C_summary.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(OUT_DIR / "ocvstart_lfpstyle_trueocv_vcorr_0C_train_fit.csv", index=False)

    plot_combined(full, fig_dir / "ocvstart_lfpstyle_trueocv_all_profiles_0C_full_traj_timestep.png")
    plot_error(full, fig_dir / "ocvstart_lfpstyle_trueocv_all_profiles_0C_full_traj_timestep_error.png")
    for profile, g in full.groupby("profile", sort=True):
        plot_single(g, fig_dir / f"ocvstart_lfpstyle_trueocv_{profile}_0C_full_traj_timestep.png")

    print("full", full_path)
    print("summary", OUT_DIR / "ocvstart_lfpstyle_trueocv_vcorr_0C_summary.csv")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
