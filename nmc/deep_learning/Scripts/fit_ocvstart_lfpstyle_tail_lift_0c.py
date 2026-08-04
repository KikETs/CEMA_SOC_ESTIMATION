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
    causal_ema,
    find_csv_files,
    parse_temperature,
)

from decompose_ocvstart_lfpstyle_vcorr_0c import predict_with_split_components  # noqa: E402
from plot_ocvstart_lfpstyle_trueocv_vcorr_0c import (  # noqa: E402
    PROFILES,
    fit_lfpstyle_trueocv,
    load_frame,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def add_tail_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    t = out["time_s"].to_numpy(np.float64)
    i_dis = -out["I_raw"].to_numpy(np.float64)
    abs_i = np.abs(i_dis)
    d_i = np.diff(i_dis, prepend=i_dis[0])
    out["i_dis"] = i_dis
    out["abs_i_ema60"] = causal_ema(abs_i, t, 60.0).astype(np.float64)
    out["abs_di_ema20"] = causal_ema(np.abs(d_i), t, 20.0).astype(np.float64)
    out["v_base_slope_ema20"] = causal_ema(
        np.diff(out["V_corr_lfpstyle_trueocv"].to_numpy(np.float64), prepend=out["V_corr_lfpstyle_trueocv"].iloc[0]),
        t,
        20.0,
    ).astype(np.float64)
    return out


def tail_feature_matrix(frame: pd.DataFrame, scales: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    v_base = frame["V_corr_lfpstyle_trueocv"].to_numpy(np.float64)
    v_raw = frame["V_raw"].to_numpy(np.float64)
    i_dis = frame["i_dis"].to_numpy(np.float64)
    abs_i_ema = frame["abs_i_ema60"].to_numpy(np.float64)
    abs_di_ema = frame["abs_di_ema20"].to_numpy(np.float64)
    drop = frame["v_drop_raw"].to_numpy(np.float64)
    g_corr = frame["g_corr"].to_numpy(np.float64)
    r0 = frame["R0"].to_numpy(np.float64)
    slope = frame["v_base_slope_ema20"].to_numpy(np.float64)

    low_gate = sigmoid((float(scales["low_gate_v"]) - v_base) / max(float(scales["low_gate_s"]), 1e-6))
    sag = np.maximum(float(scales["low_gate_v"]) - v_base, 0.0) / max(float(scales["v_scale"]), 1e-6)
    vraw_low = np.maximum(float(scales["raw_low_gate_v"]) - v_raw, 0.0) / max(float(scales["v_scale"]), 1e-6)
    i_scale = max(float(scales["i_scale"]), 1e-6)
    di_scale = max(float(scales["di_scale"]), 1e-6)
    drop_scale = max(float(scales["drop_scale"]), 1e-6)
    slope_scale = max(float(scales["slope_scale"]), 1e-6)
    names = [
        "low_gate",
        "low_gate_x_sag",
        "low_gate_x_rawlow",
        "low_gate_x_i_dis",
        "low_gate_x_abs_i_ema60",
        "low_gate_x_abs_di_ema20",
        "low_gate_x_drop",
        "low_gate_x_g_corr",
        "low_gate_x_R0",
        "low_gate_x_slope",
    ]
    x = np.column_stack(
        [
            low_gate,
            low_gate * sag,
            low_gate * vraw_low,
            low_gate * (i_dis / i_scale),
            low_gate * (abs_i_ema / i_scale),
            low_gate * (abs_di_ema / di_scale),
            low_gate * (drop / drop_scale),
            low_gate * g_corr,
            low_gate * (r0 / max(float(scales["r0_scale"]), 1e-6)),
            low_gate * (slope / slope_scale),
        ]
    ).astype(np.float64)
    return x, names


def fit_tail_model(train: pd.DataFrame, cap_v: float, ridge: float = 2e-3) -> dict:
    tr = train.dropna(
        subset=[
            "V_corr_lfpstyle_trueocv",
            "OCV_from_SOC",
            "V_raw",
            "I_raw",
            "v_drop_raw",
            "R0",
            "g_corr",
        ]
    ).copy()
    tr = tr[np.isfinite(tr["OCV_from_SOC"])]
    v_base = tr["V_corr_lfpstyle_trueocv"].to_numpy(np.float64)
    v_raw = tr["V_raw"].to_numpy(np.float64)
    i_dis = -tr["I_raw"].to_numpy(np.float64)
    d_i = np.diff(i_dis, prepend=i_dis[0])
    drop = tr["v_drop_raw"].to_numpy(np.float64)
    dv = np.diff(v_base, prepend=v_base[0])
    scales = {
        "low_gate_v": float(np.nanpercentile(v_base, 22)),
        "low_gate_s": float(max((np.nanpercentile(v_base, 35) - np.nanpercentile(v_base, 12)) / 3.0, 0.035)),
        "raw_low_gate_v": float(np.nanpercentile(v_raw, 18)),
        "v_scale": float(max(np.nanpercentile(v_base, 90) - np.nanpercentile(v_base, 10), 1e-3)),
        "i_scale": float(max(np.nanpercentile(np.abs(i_dis), 95), 1e-3)),
        "di_scale": float(max(np.nanpercentile(np.abs(d_i), 95), 1e-3)),
        "drop_scale": float(max(np.nanpercentile(np.abs(drop), 95), 1e-3)),
        "r0_scale": float(max(np.nanmedian(tr["R0"].to_numpy(np.float64)), 1e-3)),
        "slope_scale": float(max(np.nanpercentile(np.abs(dv), 95), 1e-5)),
    }
    tr = add_tail_inputs(tr)
    x, names = tail_feature_matrix(tr, scales)
    y = tr["OCV_from_SOC"].to_numpy(np.float64) - tr["V_corr_lfpstyle_trueocv"].to_numpy(np.float64)
    y = np.clip(y, 0.0, float(cap_v))
    low_gate = x[:, 0]
    weights = 1.0 + 5.0 * low_gate + 2.0 * (y > 0.05)
    finite = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x = x[finite]
    y = y[finite]
    weights = weights[finite]
    x_mu = np.nanmean(x, axis=0)
    x_sig = np.nanstd(x, axis=0)
    x_sig = np.where(x_sig > 1e-9, x_sig, 1.0)
    xs = (x - x_mu) / x_sig
    sqrt_w = np.sqrt(weights)
    xw = xs * sqrt_w[:, None]
    yw = y * sqrt_w
    reg = np.eye(xs.shape[1], dtype=np.float64) * float(ridge)
    try:
        beta = np.linalg.solve(xw.T @ xw + reg, xw.T @ yw)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(xw, yw, rcond=None)[0]
    pred = np.clip((xs @ beta), 0.0, float(cap_v))
    return {
        "cap_v": float(cap_v),
        "ridge": float(ridge),
        "scales": scales,
        "names": names,
        "x_mu": x_mu,
        "x_sig": x_sig,
        "beta": beta,
        "train_rows": int(len(y)),
        "train_lift_mae_mV": float(np.nanmean(np.abs(pred - y)) * 1000.0),
        "train_lift_mean_mV": float(np.nanmean(pred) * 1000.0),
    }


def apply_tail_model(frame: pd.DataFrame, model: dict, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    work = add_tail_inputs(frame)
    x, _ = tail_feature_matrix(work, model["scales"])
    xs = (x - model["x_mu"]) / model["x_sig"]
    lift = np.clip(xs @ model["beta"], 0.0, float(model["cap_v"]))
    out = work["V_corr_lfpstyle_trueocv"].to_numpy(np.float64) + float(alpha) * lift
    return out.astype(np.float64), (float(alpha) * lift).astype(np.float64)


def metric(frame: pd.DataFrame, col: str, method: str, holdout: str) -> list[dict[str, float | int | str]]:
    rows = []
    g = frame.dropna(subset=[col, "OCV_from_SOC", "SOC_pct"]).copy()
    err = (g[col].to_numpy(np.float64) - g["OCV_from_SOC"].to_numpy(np.float64)) * 1000.0
    soc = g["SOC_pct"].to_numpy(np.float64)
    bins = [
        ("all", np.ones(len(g), dtype=bool)),
        ("SOC<=20", soc <= 20.0),
        ("SOC<=10", soc <= 10.0),
        ("10<SOC<=20", (soc > 10.0) & (soc <= 20.0)),
        ("SOC>20", soc > 20.0),
    ]
    for scope, mask in bins:
        if not np.any(mask):
            continue
        e = err[mask]
        rows.append(
            {
                "holdout": holdout,
                "method": method,
                "scope": scope,
                "n": int(mask.sum()),
                "mae_mV": float(np.nanmean(np.abs(e))),
                "bias_mV": float(np.nanmean(e)),
                "p95_abs_mV": float(np.nanpercentile(np.abs(e), 95)),
                "max_abs_mV": float(np.nanmax(np.abs(e))),
                "positive_frac": float(np.nanmean(e > 0.0)),
            }
        )
    return rows


def score_for_selection(frame: pd.DataFrame, col: str) -> float:
    rows = metric(frame, col, "tmp", "tmp")
    by_scope = {r["scope"]: r for r in rows}
    all_mae = float(by_scope.get("all", {}).get("mae_mV", np.inf))
    low_mae = float(by_scope.get("SOC<=20", {}).get("mae_mV", all_mae))
    return low_mae + 0.25 * all_mae


def select_cap_alpha(train_frames: dict[str, pd.DataFrame]) -> tuple[float, float, pd.DataFrame]:
    caps = [0.08, 0.12, 0.18, 0.25, 0.35, 0.45]
    alphas = [0.25, 0.50, 0.75, 1.00]
    rows = []
    profiles = tuple(train_frames)
    for cap in caps:
        for alpha in alphas:
            scores = []
            for valid_profile in profiles:
                fit_profiles = [p for p in profiles if p != valid_profile]
                fit_df = pd.concat([train_frames[p] for p in fit_profiles], ignore_index=True)
                model = fit_tail_model(fit_df, cap_v=cap)
                valid = train_frames[valid_profile].copy()
                valid["V_corr_tail_lift"], valid["tail_lift_V"] = apply_tail_model(valid, model, alpha=alpha)
                scores.append(score_for_selection(valid, "V_corr_tail_lift"))
            rows.append(
                {
                    "cap_v": float(cap),
                    "alpha": float(alpha),
                    "inner_score_mV": float(np.mean(scores)),
                    "inner_score_max_mV": float(np.max(scores)),
                }
            )
    scores_df = pd.DataFrame(rows).sort_values(["inner_score_mV", "inner_score_max_mV", "cap_v", "alpha"])
    best = scores_df.iloc[0]
    return float(best["cap_v"]), float(best["alpha"]), scores_df


def plot_holdout(frame: pd.DataFrame, holdout: str, out_path: Path) -> None:
    g = frame.copy()
    x = np.arange(len(g), dtype=np.int64)
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(x, g["V_raw"], color="#999999", lw=0.65, label="raw")
    axes[0].plot(x, g["OCV_from_SOC"], color="#111111", lw=1.0, label="OCV(SOC)")
    axes[0].plot(x, g["V_corr_lfpstyle_trueocv"], color="#0072B2", lw=0.9, label="base")
    axes[0].plot(x, g["V_corr_tail_lift"], color="#D55E00", lw=0.9, label="tail lift")
    axes[0].set_ylabel("V")
    axes[0].set_title(f"{holdout} 0C tail-lift corrected voltage")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=4)

    base_err = (g["V_corr_lfpstyle_trueocv"] - g["OCV_from_SOC"]) * 1000.0
    lift_err = (g["V_corr_tail_lift"] - g["OCV_from_SOC"]) * 1000.0
    axes[1].axhline(0.0, color="#111111", lw=0.7)
    axes[1].plot(x, base_err, color="#0072B2", lw=0.75, label="base error")
    axes[1].plot(x, lift_err, color="#D55E00", lw=0.75, label="tail-lift error")
    axes[1].set_ylabel("mV")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    axes[2].plot(x, g["tail_lift_V"] * 1000.0, color="#D55E00", lw=0.8, label="tail lift")
    axes[2].plot(x, g["SOC_pct"], color="#009E73", lw=0.8, label="SOC (%)")
    axes[2].set_ylabel("mV / SOC%")
    axes[2].set_xlabel("timestep")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False)
    fig.tight_layout()
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
    frames_by_profile = {str(f["profile"].iloc[0]).upper(): f for f in frames}

    all_holdout = []
    summary_rows = []
    model_rows = []
    selection_rows = []
    for holdout in PROFILES:
        train_profiles = tuple(p for p in PROFILES if p != holdout)
        calibration, info = fit_lfpstyle_trueocv(frames, files, train_profiles, cfg)
        p = np.asarray(calibration["params"], dtype=np.float64)
        scales = dict(calibration["scales"])
        train_decomp = {
            prof: predict_with_split_components(p, frames_by_profile[prof], scales, cfg)
            for prof in train_profiles
        }
        train_decomp = {prof: add_tail_inputs(df) for prof, df in train_decomp.items()}
        cap_v, alpha, inner_scores = select_cap_alpha(train_decomp)
        inner_scores["holdout"] = holdout
        selection_rows.append(inner_scores)
        model = fit_tail_model(pd.concat(list(train_decomp.values()), ignore_index=True), cap_v=cap_v)
        holdout_df = predict_with_split_components(p, frames_by_profile[holdout], scales, cfg)
        holdout_df = add_tail_inputs(holdout_df)
        holdout_df["V_corr_tail_lift"], holdout_df["tail_lift_V"] = apply_tail_model(holdout_df, model, alpha=alpha)
        holdout_df["holdout"] = holdout
        holdout_df["tail_cap_v"] = cap_v
        holdout_df["tail_alpha"] = alpha
        holdout_df["calibration_train_profiles"] = "+".join(train_profiles)
        all_holdout.append(holdout_df)
        summary_rows.extend(metric(holdout_df, "V_corr_lfpstyle_trueocv", "base_lfpstyle", holdout))
        summary_rows.extend(metric(holdout_df, "V_corr_tail_lift", "tail_lift_train_only", holdout))
        model_rows.append(
            {
                "holdout": holdout,
                "train_profiles": "+".join(train_profiles),
                "cap_v": cap_v,
                "alpha": alpha,
                **info,
                "tail_train_rows": model["train_rows"],
                "tail_train_lift_mae_mV": model["train_lift_mae_mV"],
                "tail_train_lift_mean_mV": model["train_lift_mean_mV"],
                **{f"scale_{k}": float(v) for k, v in model["scales"].items()},
            }
        )
        plot_holdout(holdout_df, holdout, FIG_DIR / f"ocvstart_lfpstyle_tail_lift_{holdout}_0C.png")

    full = pd.concat(all_holdout, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(["scope", "holdout", "mae_mV", "method"])
    models = pd.DataFrame(model_rows)
    selections = pd.concat(selection_rows, ignore_index=True)
    full_path = OUT_DIR / "ocvstart_lfpstyle_tail_lift_0C_full.csv.gz"
    summary_path = OUT_DIR / "ocvstart_lfpstyle_tail_lift_0C_summary.csv"
    models_path = OUT_DIR / "ocvstart_lfpstyle_tail_lift_0C_models.csv"
    selection_path = OUT_DIR / "ocvstart_lfpstyle_tail_lift_0C_inner_selection.csv"
    full.to_csv(full_path, index=False)
    summary.to_csv(summary_path, index=False)
    models.to_csv(models_path, index=False)
    selections.to_csv(selection_path, index=False)

    print("full", full_path)
    print("summary", summary_path)
    print("models", models_path)
    print("selection", selection_path)
    show = summary[summary["scope"].isin(["all", "SOC<=20", "SOC<=10"])].copy()
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
