#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "nmc_ocvstart_endzero_lopo_clean"
OUT = ROOT / "nmc_goal_vcorr_it_train_dst_selector_results" / "endzero_3lopo_three_profile_vcorr_diagnostic"
FIG = OUT / "figures"
OCV_PREP_DIR = Path("/home/user/바탕화면/DL/CEMA-TCN/Data")
OCV_REF_DIR = OCV_PREP_DIR / "raw_reference"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OCV_PREP_DIR))

import prepare_calce_nmc as prep  # noqa: E402
from soc_decomp.nmc_branchbands_experiment import (  # noqa: E402
    NMCBranchBandsConfig,
    build_feature_frames,
    estimate_r0_by_temperature,
    find_csv_files,
)


PROFILE_SET = ("DST", "FUDS", "US06")
HOLDOUTS = ("US06", "DST", "FUDS")
VARIANTS = [
    ("orig_ohm_ema120", "R0 EMA120", "r0_ema120", "#0072b2"),
    ("rtvar_lowv_ema120", "rtvar low-V", "rtvar_lowv", "#009e73"),
    ("lfpstyle_ocvfit_v1", "LFP-style", "lfpstyle", "#cc79a7"),
]


def ocv_from_soc(refs: dict[float, object], temp: float, soc_frac: np.ndarray) -> np.ndarray:
    ref = refs.get(float(temp))
    if ref is None or ref.voltage_v is None or ref.soc_fraction is None:
        return np.full(len(soc_frac), np.nan, dtype=np.float64)
    soc = np.asarray(ref.soc_fraction, dtype=np.float64)
    volt = np.asarray(ref.voltage_v, dtype=np.float64)
    order = np.argsort(soc)
    soc = soc[order]
    volt = volt[order]
    keep = np.isfinite(soc) & np.isfinite(volt)
    soc = soc[keep]
    volt = volt[keep]
    unique_soc, unique_idx = np.unique(soc, return_index=True)
    unique_volt = volt[unique_idx]
    return np.interp(np.asarray(soc_frac, dtype=np.float64), unique_soc, unique_volt).astype(np.float64)


def train_profiles_for_holdout(holdout: str) -> tuple[str, ...]:
    return tuple(profile for profile in PROFILE_SET if profile != holdout)


def cfg_for(holdout: str, variant: str) -> NMCBranchBandsConfig:
    cfg = NMCBranchBandsConfig(
        base_dir=ROOT,
        raw_root=RAW_ROOT,
        train_profiles=train_profiles_for_holdout(holdout),
        test_profiles=(holdout,),
        v_corr_variant=variant,
        v_corr_tau_s=120.0,
    )
    if variant == "lfpstyle_ocvfit_v1":
        cfg.v_corr_tau_s = 10.0
        cfg.lfpstyle_fit_stride = 5
        cfg.lfpstyle_fit_max_nfev = 250
        cfg.lfpstyle_v_floor_raw = 2.45
        cfg.lfpstyle_r0_max_ohm = 0.22
    return cfg


def build_wide() -> tuple[pd.DataFrame, pd.DataFrame]:
    refs = prep.load_ocv_references(OCV_REF_DIR)
    files = find_csv_files(RAW_ROOT)
    long_parts = []
    manifest_rows = []
    for holdout in HOLDOUTS:
        train_profiles = train_profiles_for_holdout(holdout)
        for variant, label, slug, _color in VARIANTS:
            cfg = cfg_for(holdout, variant)
            r0_df = estimate_r0_by_temperature(files, train_profiles)
            frames = build_feature_frames(cfg, files, r0_df)
            test0 = [
                f.copy()
                for f in frames["test"]
                if float(f["temperature"].iloc[0]) == 0.0 and str(f["drive_cycle"].iloc[0]).upper() == holdout
            ]
            if len(test0) != 1:
                raise RuntimeError(f"Expected one 0C test frame for {holdout}/{variant}, got {len(test0)}")
            f = test0[0]
            soc = np.clip(f["SOC_physical"].to_numpy(np.float64), 0.0, 1.0)
            current = pd.DataFrame(
                {
                    "profile": holdout,
                    "temperature_C": 0.0,
                    "end_index": f["end_index"].to_numpy(np.int64),
                    "SOC_frac": soc,
                    "SOC_pct": soc * 100.0,
                    "V_raw": f["V_raw"].to_numpy(np.float64),
                    "I_raw": f["I_raw"].to_numpy(np.float64),
                    "OCV_ref_from_SOC": ocv_from_soc(refs, 0.0, soc),
                    "variant": label,
                    "slug": slug,
                    "V_corr": f["V_corr_raw"].to_numpy(np.float64),
                }
            )
            long_parts.append(current)
            manifest_rows.append(
                {
                    "holdout": holdout,
                    "train_profiles": "+".join(train_profiles),
                    "v_corr_variant": variant,
                    "label": label,
                    "slug": slug,
                }
            )
    if not long_parts:
        raise RuntimeError("No frames were built.")
    long = pd.concat(long_parts, ignore_index=True)
    key_cols = ["profile", "temperature_C", "end_index"]
    base_cols = ["profile", "temperature_C", "end_index", "SOC_frac", "SOC_pct", "V_raw", "I_raw", "OCV_ref_from_SOC"]
    base = long[base_cols].drop_duplicates(subset=key_cols).copy()
    wide_values = (
        long.pivot_table(index=key_cols, columns="slug", values="V_corr", aggfunc="first")
        .reset_index()
        .rename(columns={slug: f"V_corr_{slug}" for _variant, _label, slug, _color in VARIANTS})
    )
    full = base.merge(wide_values, on=key_cols, how="left", validate="one_to_one")
    return full.sort_values(["profile", "end_index"]).reset_index(drop=True), pd.DataFrame(manifest_rows)


def summarize(full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for profile, g in full.groupby("profile", sort=True):
        ocv = g["OCV_ref_from_SOC"].to_numpy(np.float64)
        low20 = g["SOC_pct"].to_numpy(np.float64) <= 20.0
        low10 = g["SOC_pct"].to_numpy(np.float64) <= 10.0
        for col, label in [("V_raw", "raw")] + [(f"V_corr_{slug}", label) for _variant, label, slug, _color in VARIANTS]:
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


def plot_tail(full: pd.DataFrame, out_path: Path, error: bool = False) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True)
    lines = [("V_raw", "raw", "#9a9a9a")] + [(f"V_corr_{slug}", label, color) for _variant, label, slug, color in VARIANTS]
    for ax, profile in zip(axes, HOLDOUTS):
        g = full[(full["profile"].eq(profile)) & (full["SOC_pct"].le(20.0))].copy()
        x = g["end_index"].to_numpy(np.int64)
        ocv = g["OCV_ref_from_SOC"].to_numpy(np.float64)
        if error:
            ax.axhline(0.0, color="#111111", lw=0.8)
            for col, label, color in lines:
                ax.plot(x, (g[col].to_numpy(np.float64) - ocv) * 1000.0, label=label, color=color, lw=0.9, alpha=0.9)
            ax.set_ylabel("V - OCV_ref (mV)")
            title_suffix = "error"
        else:
            ax.plot(x, ocv, color="#111111", lw=1.3, ls="--", label="OCV_ref(SOC)")
            for col, label, color in lines:
                ax.plot(x, g[col].to_numpy(np.float64), label=label, color=color, lw=0.9, alpha=0.9)
            ax.set_ylabel("Voltage (V)")
            title_suffix = "voltage"
        ax.set_title(f"{profile} holdout 0C, 0-20% SOC {title_suffix}")
        ax.set_xlabel("timestep")
        ax.grid(alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    full, manifest = build_wide()
    summary = summarize(full)
    full.to_csv(OUT / "endzero_3lopo_three_profile_vcorr_0C_full.csv.gz", index=False)
    summary.to_csv(OUT / "endzero_3lopo_three_profile_vcorr_0C_summary.csv", index=False)
    manifest.to_csv(OUT / "endzero_3lopo_three_profile_vcorr_0C_manifest.csv", index=False)
    plot_tail(full, FIG / "endzero_3lopo_three_profile_vcorr_0C_low20_voltage.png", error=False)
    plot_tail(full, FIG / "endzero_3lopo_three_profile_vcorr_0C_low20_error.png", error=True)
    print(f"full={OUT / 'endzero_3lopo_three_profile_vcorr_0C_full.csv.gz'}")
    print(f"summary={OUT / 'endzero_3lopo_three_profile_vcorr_0C_summary.csv'}")
    print(summary.groupby("variant", as_index=False)["mae_mV"].mean().sort_values("mae_mV").to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
