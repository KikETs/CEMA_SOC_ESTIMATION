#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "nmc_ocvstart_lopo_clean"
OUT = ROOT / "nmc_goal_vcorr_it_train_dst_selector_results"
FIG = OUT / "vcorr_tail_method_figures"
OCV_PREP_DIR = Path("/home/user/바탕화면/DL/CEMA-TCN/Data")
OCV_REF_DIR = OCV_PREP_DIR / "raw_reference"
PROFILE_SET = ("VALIDATION", "DST", "FUDS", "US06")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def causal_ema(values: np.ndarray, times_s: np.ndarray, tau_s: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    t = np.asarray(times_s, dtype=np.float64)
    if len(x) == 0:
        return x.astype(np.float32)
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    if len(x) == 1:
        return y.astype(np.float32)
    dt_all = np.diff(t)
    valid_dt = dt_all[np.isfinite(dt_all) & (dt_all > 0)]
    dt_default = float(np.nanmedian(valid_dt)) if len(valid_dt) else 1.0
    if not np.isfinite(dt_default) or dt_default <= 0:
        dt_default = 1.0
    for i in range(1, len(x)):
        dt = t[i] - t[i - 1]
        if not np.isfinite(dt) or dt <= 0:
            dt = dt_default
        alpha = float(np.exp(-dt / max(float(tau_s), 1e-6)))
        alpha = min(max(alpha, 0.0), 0.999999)
        y[i] = alpha * y[i - 1] + (1.0 - alpha) * x[i]
    return y.astype(np.float32)


def causal_asymmetric_ema(values: np.ndarray, times_s: np.ndarray, tau_down_s: float, tau_up_s: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    t = np.asarray(times_s, dtype=np.float64)
    if len(x) == 0:
        return x.astype(np.float32)
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    if len(x) == 1:
        return y.astype(np.float32)
    dt_all = np.diff(t)
    valid_dt = dt_all[np.isfinite(dt_all) & (dt_all > 0)]
    dt_default = float(np.nanmedian(valid_dt)) if len(valid_dt) else 1.0
    if not np.isfinite(dt_default) or dt_default <= 0:
        dt_default = 1.0
    for i in range(1, len(x)):
        dt = t[i] - t[i - 1]
        if not np.isfinite(dt) or dt <= 0:
            dt = dt_default
        tau = tau_down_s if x[i] < y[i - 1] else tau_up_s
        alpha = float(np.exp(-dt / max(float(tau), 1e-6)))
        alpha = min(max(alpha, 0.0), 0.999999)
        y[i] = alpha * y[i - 1] + (1.0 - alpha) * x[i]
    return y.astype(np.float32)


def parse_temperature(path: Path) -> float:
    m = re.search(r"(-?\d+(?:\.\d+)?)C", str(path))
    if not m:
        raise ValueError(f"Cannot parse temperature from {path}")
    return float(m.group(1))


def profile_from_path(path: Path) -> str:
    m = re.search(r"NMC_[^_]+_(.+)\.csv", path.name)
    if not m:
        raise ValueError(f"Cannot parse profile from {path.name}")
    return m.group(1)


def load_ocv_refs():
    sys.path.insert(0, str(OCV_PREP_DIR))
    import prepare_calce_nmc as prep

    return prep.load_ocv_references(OCV_REF_DIR)


def ocv_from_soc(refs, temp: float, soc_frac: np.ndarray) -> np.ndarray:
    ref = refs.get(float(temp))
    if ref is None or ref.voltage_v is None or ref.soc_fraction is None:
        return np.full_like(np.asarray(soc_frac, dtype=np.float64), np.nan, dtype=np.float64)
    soc = np.asarray(ref.soc_fraction, dtype=np.float64)
    volt = np.asarray(ref.voltage_v, dtype=np.float64)
    order = np.argsort(soc)
    soc = soc[order]
    volt = volt[order]
    keep = np.concatenate([[True], np.diff(soc) > 1e-10])
    return np.interp(np.asarray(soc_frac, dtype=np.float64), soc[keep], volt[keep], left=volt[keep][0], right=volt[keep][-1])


def find_r0_for_holdout(holdout: str) -> dict[float, float]:
    patterns = [
        f"ocvstart_lopo4_baseline_model_*_head_paper_g4_all_ema_holdout{holdout.lower()}*_decomposition_params.csv",
        f"ocvstart_lopo4_feature_ablation_*_paper_g4_all_ema_*_head_paper_g4_all_ema_holdout{holdout.lower()}*_decomposition_params.csv",
        f"ocvstart_lopo4_*holdout{holdout.lower()}*_decomposition_params.csv",
    ]
    for pat in patterns:
        files = sorted(OUT.glob(pat))
        if files:
            df = pd.read_csv(files[0])
            return {float(r["temperature_C"]): float(r["r0_ohm"]) for _, r in df.iterrows()}
    raise FileNotFoundError(f"No decomposition_params for holdout={holdout}")


def load_series(path: Path, r0: float, refs) -> pd.DataFrame:
    df = pd.read_csv(path)
    temp = parse_temperature(path)
    profile = str(df["Profile"].iloc[0]) if "Profile" in df.columns else profile_from_path(path)
    time_col = "Step_Time(s)" if "Step_Time(s)" in df.columns else "Test_Time(s)"
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(np.float64)
    v = pd.to_numeric(df["Voltage(V)"], errors="coerce").to_numpy(np.float64)
    i = pd.to_numeric(df["Current(A)"], errors="coerce").to_numpy(np.float64)
    soc = pd.to_numeric(df["SOC_CC"], errors="coerce").to_numpy(np.float64)
    if np.nanmax(soc) > 1.5:
        soc = soc / 100.0
    d_i = np.diff(i, prepend=i[0])
    abs_i_ema = causal_ema(np.abs(i), t, 60.0).astype(np.float64)
    di_abs_ema = causal_ema(np.abs(d_i), t, 20.0).astype(np.float64)
    v_ohm_free = v - i * float(r0)
    base = causal_ema(v_ohm_free, t, 120.0).astype(np.float64)
    asym = causal_asymmetric_ema(v_ohm_free, t, 20.0, 160.0).astype(np.float64)
    ocv = ocv_from_soc(refs, temp, soc)
    return pd.DataFrame(
        {
            "profile": profile,
            "temperature_C": temp,
            "time_s": t,
            "SOC_pct": soc * 100.0,
            "V_raw": v,
            "I_raw": i,
            "absI_ema60": abs_i_ema,
            "dI_abs_ema20": di_abs_ema,
            "V_ohm_free": v_ohm_free,
            "V_corr_base": base,
            "V_corr_asym": asym,
            "OCV_from_SOC": ocv,
        }
    ).replace([np.inf, -np.inf], np.nan)


def variant_tail_fixed(df: pd.DataFrame) -> np.ndarray:
    base = df["V_corr_base"].to_numpy(np.float64)
    d = df["dI_abs_ema20"].to_numpy(np.float64)
    # Label-free high-voltage, low-dynamic gate. Correction is intentionally bounded to 12 mV.
    g_high = sigmoid((base - 3.70) / 0.035)
    g_lowdyn = sigmoid((0.05 - d) / 0.025)
    return base - 0.012 * g_high * g_lowdyn


def variant_tail_asym(df: pd.DataFrame) -> np.ndarray:
    base = df["V_corr_base"].to_numpy(np.float64)
    asym = df["V_corr_asym"].to_numpy(np.float64)
    g_high = sigmoid((base - 3.70) / 0.035)
    return (1.0 - g_high) * base + g_high * asym


def fit_linear_calibrator(train: pd.DataFrame) -> dict:
    tr = train.dropna(subset=["V_corr_base", "OCV_from_SOC", "absI_ema60", "dI_abs_ema20"]).copy()
    tr = tr[np.isfinite(tr["OCV_from_SOC"])]
    if len(tr) < 100:
        return {"ok": False}
    y = (tr["V_corr_base"].to_numpy(np.float64) - tr["OCV_from_SOC"].to_numpy(np.float64))
    x0 = tr["V_corr_base"].to_numpy(np.float64)
    x1 = tr["absI_ema60"].to_numpy(np.float64)
    x2 = tr["dI_abs_ema20"].to_numpy(np.float64)
    x3 = tr["I_raw"].to_numpy(np.float64)
    Xraw = np.column_stack([x0, x0 * x0, x1, x2, x3])
    mu = np.nanmean(Xraw, axis=0)
    sig = np.nanstd(Xraw, axis=0)
    sig = np.where(sig > 1e-9, sig, 1.0)
    X = np.column_stack([np.ones(len(Xraw)), (Xraw - mu) / sig])
    ridge = 1e-3
    reg = np.eye(X.shape[1]) * ridge
    reg[0, 0] = 0.0
    beta = np.linalg.solve(X.T @ X + reg, X.T @ y)
    return {"ok": True, "mu": mu, "sig": sig, "beta": beta}


def apply_linear_calibrator(df: pd.DataFrame, model: dict) -> np.ndarray:
    base = df["V_corr_base"].to_numpy(np.float64)
    if not model.get("ok"):
        return base
    x0 = base
    x1 = df["absI_ema60"].to_numpy(np.float64)
    x2 = df["dI_abs_ema20"].to_numpy(np.float64)
    x3 = df["I_raw"].to_numpy(np.float64)
    Xraw = np.column_stack([x0, x0 * x0, x1, x2, x3])
    X = np.column_stack([np.ones(len(Xraw)), (Xraw - model["mu"]) / model["sig"]])
    delta = X @ model["beta"]
    delta = np.clip(delta, -0.020, 0.020)
    return base - delta


def summarize(df: pd.DataFrame, value_col: str, holdout: str, method: str) -> list[dict]:
    rows = []
    finite = df.dropna(subset=[value_col, "OCV_from_SOC", "SOC_pct"]).copy()
    finite = finite[np.isfinite(finite["OCV_from_SOC"])]
    for temp, g in finite.groupby("temperature_C"):
        diff = (g[value_col].to_numpy(np.float64) - g["OCV_from_SOC"].to_numpy(np.float64)) * 1000.0
        rows.append(
            {
                "holdout": holdout,
                "temperature_C": float(temp),
                "method": method,
                "scope": "all",
                "n": int(len(g)),
                "mean_mV": float(np.mean(diff)),
                "mae_mV": float(np.mean(np.abs(diff))),
                "p05_mV": float(np.quantile(diff, 0.05)),
                "p95_mV": float(np.quantile(diff, 0.95)),
                "positive_frac": float(np.mean(diff > 0)),
            }
        )
        cats = pd.cut(g["SOC_pct"], bins=[0, 10, 20, 40, 60, 80, 100], include_lowest=True, right=False)
        tmp = pd.DataFrame({"soc_bin": cats, "diff": diff})
        for b, bg in tmp.groupby("soc_bin", observed=True):
            rows.append(
                {
                    "holdout": holdout,
                    "temperature_C": float(temp),
                    "method": method,
                    "scope": str(b),
                    "n": int(len(bg)),
                    "mean_mV": float(bg["diff"].mean()),
                    "mae_mV": float(bg["diff"].abs().mean()),
                    "p05_mV": float(bg["diff"].quantile(0.05)),
                    "p95_mV": float(bg["diff"].quantile(0.95)),
                    "positive_frac": float((bg["diff"] > 0).mean()),
                }
            )
    return rows


def main() -> None:
    refs = load_ocv_refs()
    r0_by_holdout = {h: find_r0_for_holdout(h) for h in PROFILE_SET}
    all_rows: list[dict] = []
    all_frames: dict[tuple[str, str], pd.DataFrame] = {}
    calibrators: dict[tuple[str, float], dict] = {}

    # Build base frames once per holdout protocol because train-only R0 differs by holdout.
    for holdout in PROFILE_SET:
        r0_lookup = r0_by_holdout[holdout]
        train_profiles = tuple(p for p in PROFILE_SET if p != holdout)
        train_by_temp: dict[float, list[pd.DataFrame]] = {}
        holdout_frames: list[pd.DataFrame] = []
        for path in sorted(DATA_ROOT.glob("*C/*.csv")):
            temp = parse_temperature(path)
            profile = profile_from_path(path)
            if temp not in r0_lookup:
                continue
            frame = load_series(path, r0_lookup[temp], refs)
            if profile in train_profiles:
                train_by_temp.setdefault(temp, []).append(frame)
            if profile == holdout:
                holdout_frames.append(frame)
        for temp, frames in train_by_temp.items():
            calibrators[(holdout, float(temp))] = fit_linear_calibrator(pd.concat(frames, ignore_index=True))
        holdout_df = pd.concat(holdout_frames, ignore_index=True)
        holdout_df["V_corr_tail_fixed"] = variant_tail_fixed(holdout_df)
        holdout_df["V_corr_tail_asym"] = variant_tail_asym(holdout_df)
        calibrated = []
        for temp, g in holdout_df.groupby("temperature_C", sort=False):
            calibrated.append(pd.Series(apply_linear_calibrator(g, calibrators.get((holdout, float(temp)), {"ok": False})), index=g.index))
        holdout_df["V_corr_train_linear_cal"] = pd.concat(calibrated).sort_index()
        all_frames[(holdout, "holdout")] = holdout_df
        for method, col in [
            ("0_base_ema120", "V_corr_base"),
            ("1_tail_fixed_hv_lowdyn_minus12mV", "V_corr_tail_fixed"),
            ("2_train_only_linear_residual_clip20mV", "V_corr_train_linear_cal"),
            ("3_tail_gated_asym_down20_up160", "V_corr_tail_asym"),
        ]:
            all_rows.extend(summarize(holdout_df, col, holdout, method))

    summary = pd.DataFrame(all_rows).sort_values(["holdout", "temperature_C", "scope", "mae_mV", "method"])
    summary_path = OUT / "ocvstart_vcorr_tail_methods_summary.csv"
    summary.to_csv(summary_path, index=False)

    focused = summary[
        (summary["temperature_C"].eq(25.0))
        & (summary["holdout"].isin(["DST", "US06"]))
        & (
            summary["scope"].isin(["all", "[60, 80)", "[80, 100)", "[40, 60)", "[10, 20)"])
        )
    ].copy()
    focused_path = OUT / "ocvstart_vcorr_tail_methods_dst_us06_25c_focused.csv"
    focused.to_csv(focused_path, index=False)

    FIG.mkdir(exist_ok=True)
    for holdout in ("DST", "US06"):
        df = all_frames[(holdout, "holdout")]
        df = df[df["temperature_C"].eq(25.0)].copy()
        df = df[np.isfinite(df["OCV_from_SOC"])]
        order = np.argsort(df["SOC_pct"].to_numpy())
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for method, col in [
            ("base", "V_corr_base"),
            ("tail fixed", "V_corr_tail_fixed"),
            ("train cal", "V_corr_train_linear_cal"),
            ("tail asym", "V_corr_tail_asym"),
        ]:
            diff = (df[col].to_numpy(np.float64) - df["OCV_from_SOC"].to_numpy(np.float64)) * 1000.0
            ax.plot(df["SOC_pct"].to_numpy()[order], diff[order], lw=0.8, label=method)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_xlabel("SOC_CC (%)")
        ax.set_ylabel("V_corr variant - OCV(SOC_CC) (mV)")
        ax.set_title(f"{holdout} 25C V_corr tail methods")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(FIG / f"{holdout}_25C_tail_methods.png", dpi=180)
        plt.close(fig)

    print("summary_path", summary_path)
    print("focused_path", focused_path)
    print("fig_dir", FIG)
    show = focused.sort_values(["holdout", "scope", "mae_mV", "method"])
    print(show[["holdout", "scope", "method", "n", "mean_mV", "mae_mV", "positive_frac"]].to_string(index=False))


if __name__ == "__main__":
    main()
