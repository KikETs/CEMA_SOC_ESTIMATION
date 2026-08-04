from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import gzip

import numpy as np
import pandas as pd


PROFILE_ORDER = ["DST", "US06", "VALIDATION", "FUDS"]
TRAIN_PROFILES = ("DST", "US06")
VALID_PROFILE = "VALIDATION"
TEST_PROFILE = "FUDS"


@dataclass
class AnalysisConfig:
    base_dir: Path = Path(".")
    raw_root: Path = Path("nmc_soc_ocvstart_relabelled_from_lc_ocv/data/NMC SAMSUNG INR 18650 2Ah")
    out_dir: Path = Path("paper_results")
    window_len: int = 50
    stride: int = 1
    temperature_c: float = 25.0
    old_best_prediction_rows: Path = Path(
        "remote_result_summaries/"
        "nmc_vcorrit_h128_l6_seed012_sel7_16_stage2vitdecomp_coldhot_corr1p35_focus16_fixed35_testblind_remote_"
        "seed0_sel7_condinv_trainDST25_selected_seed0_ep7_stage2_0_45_corr_stage2_test_prediction_rows.csv.gz"
    )


def _normalise_soc(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    if np.nanmax(arr) > 1.5:
        arr = arr / 100.0
    return np.clip(arr, 0.0, 1.0)


def _load_profile_csv(path: Path) -> pd.DataFrame:
    use_cols = [
        "Step_Time(s)",
        "Test_Time(s)",
        "Current(A)",
        "Voltage(V)",
        "SOC_CC",
        "SOC_CC(%)",
        "Profile",
        "TempLabel",
        "Qnet_denom(Ah)",
        "SOC0_used",
        "SOC0_OCV_inferred",
    ]
    df = pd.read_csv(path, usecols=lambda col: col in use_cols)
    out = pd.DataFrame(index=df.index)
    out["file_name"] = path.name
    out["trajectory_id"] = path.stem
    out["profile"] = str(df["Profile"].iloc[0]) if "Profile" in df.columns else path.stem.split("_")[-1]
    out["temperature_C"] = float(str(df["TempLabel"].iloc[0]).replace("C", "")) if "TempLabel" in df.columns else 25.0
    time_col = "Step_Time(s)" if "Step_Time(s)" in df.columns else "Test_Time(s)"
    out["time_s"] = pd.to_numeric(df[time_col], errors="coerce").astype(float)
    out["I"] = pd.to_numeric(df["Current(A)"], errors="coerce").astype(float)
    out["V"] = pd.to_numeric(df["Voltage(V)"], errors="coerce").astype(float)
    soc_source = "SOC_CC" if "SOC_CC" in df.columns else "SOC_CC(%)"
    out["SOC"] = _normalise_soc(df[soc_source])
    out["end_index"] = np.arange(len(out), dtype=np.int64)
    out["dt_s"] = out["time_s"].diff().fillna(out["time_s"].diff().median()).clip(lower=1e-6)
    out["dI"] = out["I"].diff().fillna(0.0)
    out["dV"] = out["V"].diff().fillna(0.0)
    out["absI"] = out["I"].abs()
    out["absdI"] = out["dI"].abs()
    out["absdV"] = out["dV"].abs()
    out["P"] = out["V"] * out["I"]
    out["rest_flag"] = out["I"].abs() < 0.05
    for col in ["Qnet_denom(Ah)", "SOC0_used", "SOC0_OCV_inferred"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_s", "I", "V", "SOC"]).reset_index(drop=True)


def _rolling_window_rows(frame: pd.DataFrame, window_len: int, stride: int) -> pd.DataFrame:
    rows: list[dict] = []
    profile = str(frame["profile"].iloc[0])
    temperature = float(frame["temperature_C"].iloc[0])
    for end in range(window_len - 1, len(frame), stride):
        start = end - window_len + 1
        w = frame.iloc[start : end + 1]
        duration = float(w["time_s"].iloc[-1] - w["time_s"].iloc[0])
        duration = max(duration, 1e-6)
        soc = float(w["SOC"].iloc[-1])
        rows.append(
            {
                "profile": profile,
                "temperature_C": temperature,
                "trajectory_id": str(w["trajectory_id"].iloc[-1]),
                "end_index": int(w["end_index"].iloc[-1]),
                "window_start": int(w["end_index"].iloc[0]),
                "window_duration_s": duration,
                "soc_endpoint": soc,
                "soc_bin": _soc_bin(soc),
                "V_end": float(w["V"].iloc[-1]),
                "V_start": float(w["V"].iloc[0]),
                "V_mean": float(w["V"].mean()),
                "V_std": float(w["V"].std(ddof=0)),
                "V_range": float(w["V"].max() - w["V"].min()),
                "V_delta": float(w["V"].iloc[-1] - w["V"].iloc[0]),
                "V_slope": float((w["V"].iloc[-1] - w["V"].iloc[0]) / duration),
                "I_end": float(w["I"].iloc[-1]),
                "I_mean": float(w["I"].mean()),
                "I_std": float(w["I"].std(ddof=0)),
                "absI_mean": float(w["absI"].mean()),
                "absI_p95": float(w["absI"].quantile(0.95)),
                "dI_abs_mean": float(w["absdI"].mean()),
                "dI_energy": float(np.mean(np.square(w["dI"].to_numpy(np.float64)))),
                "dV_abs_mean": float(w["absdV"].mean()),
                "P_mean": float(w["P"].mean()),
                "P_abs_mean": float(w["P"].abs().mean()),
                "rest_frac": float(w["rest_flag"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _soc_bin(soc: float) -> str:
    if soc < 0.2:
        return "00-20"
    if soc < 0.4:
        return "20-40"
    if soc < 0.6:
        return "40-60"
    if soc < 0.8:
        return "60-80"
    return "80-100"


def _window_features() -> list[str]:
    return [
        "V_end",
        "V_mean",
        "V_std",
        "V_range",
        "V_delta",
        "V_slope",
        "I_end",
        "I_mean",
        "I_std",
        "absI_mean",
        "absI_p95",
        "dI_abs_mean",
        "dI_energy",
        "dV_abs_mean",
        "P_mean",
        "P_abs_mean",
        "rest_frac",
    ]


def _standardise(train: pd.DataFrame, target: pd.DataFrame, cols: list[str]) -> np.ndarray:
    mu = train[cols].mean(axis=0).to_numpy(np.float64)
    sigma = train[cols].std(axis=0, ddof=0).replace(0.0, 1.0).to_numpy(np.float64)
    return (target[cols].to_numpy(np.float64) - mu) / sigma


def _pairwise_centroid_distances(windows: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    train = windows[windows["profile"].isin(TRAIN_PROFILES)]
    z_all = _standardise(train, windows, feature_cols)
    z = windows[["profile"]].copy()
    z[feature_cols] = z_all
    centroids = z.groupby("profile")[feature_cols].mean()
    rows = []
    for a in PROFILE_ORDER:
        for b in PROFILE_ORDER:
            if a not in centroids.index or b not in centroids.index:
                continue
            rows.append(
                {
                    "profile_a": a,
                    "profile_b": b,
                    "z_centroid_l2": float(np.linalg.norm(centroids.loc[a].to_numpy() - centroids.loc[b].to_numpy())),
                }
            )
    return pd.DataFrame(rows)


def _nearest_train_distances(windows: pd.DataFrame, feature_cols: list[str], chunk: int = 512) -> pd.DataFrame:
    train = windows[windows["profile"].isin(TRAIN_PROFILES)].reset_index(drop=True)
    train_z = _standardise(train, train, feature_cols)
    rows = []
    for profile in PROFILE_ORDER:
        target = windows[windows["profile"].eq(profile)].reset_index(drop=True)
        if target.empty:
            continue
        target_z = _standardise(train, target, feature_cols)
        nearest = np.empty(len(target_z), dtype=np.float64)
        for start in range(0, len(target_z), chunk):
            q = target_z[start : start + chunk]
            best = np.full(len(q), np.inf, dtype=np.float64)
            for t_start in range(0, len(train_z), 4096):
                t = train_z[t_start : t_start + 4096]
                dist2 = ((q[:, None, :] - t[None, :, :]) ** 2).sum(axis=2)
                best = np.minimum(best, np.sqrt(dist2.min(axis=1)))
            nearest[start : start + chunk] = best
        target = target.assign(nearest_train_z_l2=nearest)
        rows.append(target[["profile", "soc_bin", "end_index", "soc_endpoint", "nearest_train_z_l2"]])
    per_window = pd.concat(rows, ignore_index=True)
    summary = (
        per_window.groupby(["profile", "soc_bin"], as_index=False)
        .agg(
            n_windows=("nearest_train_z_l2", "size"),
            nearest_train_z_l2_mean=("nearest_train_z_l2", "mean"),
            nearest_train_z_l2_p50=("nearest_train_z_l2", "median"),
            nearest_train_z_l2_p90=("nearest_train_z_l2", lambda x: float(np.quantile(x, 0.90))),
            nearest_train_z_l2_p95=("nearest_train_z_l2", lambda x: float(np.quantile(x, 0.95))),
        )
        .sort_values(["profile", "soc_bin"])
    )
    return per_window, summary


def _profile_summary(raw_frames: dict[str, pd.DataFrame], windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for profile in PROFILE_ORDER:
        frame = raw_frames[profile]
        w = windows[windows["profile"].eq(profile)]
        rows.append(
            {
                "profile": profile,
                "n_samples": int(len(frame)),
                "duration_s": float(frame["time_s"].iloc[-1] - frame["time_s"].iloc[0]),
                "soc_start": float(frame["SOC"].iloc[0]),
                "soc_end": float(frame["SOC"].iloc[-1]),
                "soc_min": float(frame["SOC"].min()),
                "soc_max": float(frame["SOC"].max()),
                "n_windows": int(len(w)),
                "mean_absI": float(frame["absI"].mean()),
                "p95_absI": float(frame["absI"].quantile(0.95)),
                "mean_absdI": float(frame["absdI"].mean()),
                "dI_energy": float(np.mean(np.square(frame["dI"].to_numpy(np.float64)))),
                "rest_frac": float(frame["rest_flag"].mean()),
                "V_range": float(frame["V"].max() - frame["V"].min()),
                "V_start": float(frame["V"].iloc[0]),
                "V_end": float(frame["V"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def _socbin_coverage(windows: pd.DataFrame) -> pd.DataFrame:
    counts = windows.groupby(["profile", "soc_bin"], as_index=False).size().rename(columns={"size": "n_windows"})
    totals = counts.groupby("profile")["n_windows"].transform("sum")
    counts["window_frac"] = counts["n_windows"] / totals
    return counts.sort_values(["profile", "soc_bin"]).reset_index(drop=True)


def _old_best_error_summary(cfg: AnalysisConfig) -> pd.DataFrame:
    p = cfg.base_dir / cfg.old_best_prediction_rows
    if not p.exists():
        return pd.DataFrame()
    pred = pd.read_csv(p)
    pred = pred[np.isclose(pd.to_numeric(pred["temperature"], errors="coerce"), float(cfg.temperature_c))]
    pred = pred[pred["drive_cycle"].astype(str).eq(TEST_PROFILE)].copy()
    if pred.empty:
        return pd.DataFrame()
    pred["soc_bin"] = pred["y_true"].map(_soc_bin)
    pred["abs_error_pct"] = pred["abs_error"] * 100.0
    pred["error_pct"] = pred["error"] * 100.0
    return (
        pred.groupby(["temperature", "drive_cycle", "soc_bin"], as_index=False)
        .agg(
            n_windows=("abs_error_pct", "size"),
            MAE_pct=("abs_error_pct", "mean"),
            RMSE_pct=("error_pct", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            bias_pct=("error_pct", "mean"),
            y_true_min=("y_true", "min"),
            y_true_max=("y_true", "max"),
        )
        .sort_values(["soc_bin"])
    )


def _make_plots(cfg: AnalysisConfig, windows: pd.DataFrame, feature_cols: list[str]) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    out_dir = cfg.base_dir / cfg.out_dir / "nmc_25c_profile_shift_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    train = windows[windows["profile"].isin(TRAIN_PROFILES)]
    z = _standardise(train, windows, feature_cols)
    z = np.nan_to_num(z)
    z_center = z - z.mean(axis=0, keepdims=True)
    _u, _s, vh = np.linalg.svd(z_center, full_matrices=False)
    pcs = z_center @ vh[:2].T
    plot_df = windows[["profile", "soc_endpoint"]].copy()
    plot_df["pc1"] = pcs[:, 0]
    plot_df["pc2"] = pcs[:, 1]
    if len(plot_df) > 6000:
        plot_df = plot_df.groupby("profile", group_keys=False).apply(
            lambda g: g.sample(min(len(g), 1500), random_state=7)
        )
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for profile in PROFILE_ORDER:
        g = plot_df[plot_df["profile"].eq(profile)]
        ax.scatter(g["pc1"], g["pc2"], s=5, alpha=0.45, label=profile)
    ax.set_title("25C Window Feature PCA (label-free features)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    p = out_dir / "window_feature_pca.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for profile in PROFILE_ORDER:
        g = windows[windows["profile"].eq(profile)]
        ax.plot(np.arange(len(g)), g["soc_endpoint"] * 100.0, label=profile, linewidth=1.2)
    ax.set_title("25C Endpoint SOC Coverage by Profile")
    ax.set_xlabel("window index")
    ax.set_ylabel("SOC (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "soc_coverage_by_profile.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(p)
    return paths


def _markdown_table(df: pd.DataFrame, max_rows: int = 20, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "_empty_"
    show = df.head(max_rows).copy()
    return show.to_markdown(index=False, floatfmt=floatfmt)


def run(cfg: AnalysisConfig) -> None:
    cfg.base_dir = cfg.base_dir.resolve()
    if not cfg.raw_root.is_absolute():
        cfg.raw_root = cfg.base_dir / cfg.raw_root
    if not cfg.out_dir.is_absolute():
        cfg.out_dir = cfg.base_dir / cfg.out_dir
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    raw_frames: dict[str, pd.DataFrame] = {}
    window_frames: list[pd.DataFrame] = []
    for profile in PROFILE_ORDER:
        p = cfg.raw_root / "25C" / f"NMC_25C_{profile}.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        frame = _load_profile_csv(p)
        raw_frames[profile] = frame
        window_frames.append(_rolling_window_rows(frame, int(cfg.window_len), int(cfg.stride)))
    windows = pd.concat(window_frames, ignore_index=True)
    feature_cols = _window_features()

    profile_summary = _profile_summary(raw_frames, windows)
    coverage = _socbin_coverage(windows)
    pairwise = _pairwise_centroid_distances(windows, feature_cols)
    nearest_windows, nearest_summary = _nearest_train_distances(windows, feature_cols)
    old_best_errors = _old_best_error_summary(cfg)
    plot_paths = _make_plots(cfg, windows, feature_cols)

    profile_summary.to_csv(cfg.out_dir / "nmc_25c_profile_shift_profile_summary.csv", index=False)
    coverage.to_csv(cfg.out_dir / "nmc_25c_profile_shift_socbin_coverage.csv", index=False)
    windows.to_csv(cfg.out_dir / "nmc_25c_profile_shift_window_summary.csv", index=False)
    pairwise.to_csv(cfg.out_dir / "nmc_25c_profile_shift_pairwise_distance.csv", index=False)
    nearest_windows.to_csv(cfg.out_dir / "nmc_25c_profile_shift_nearest_train_distance_by_window.csv", index=False)
    nearest_summary.to_csv(cfg.out_dir / "nmc_25c_profile_shift_nearest_train_distance_summary.csv", index=False)
    if not old_best_errors.empty:
        old_best_errors.to_csv(cfg.out_dir / "nmc_25c_old_best_error_by_socbin.csv", index=False)

    fuds_vs = pairwise[pairwise["profile_a"].eq(TEST_PROFILE)].copy()
    nearest_fuds = nearest_summary[nearest_summary["profile"].eq(TEST_PROFILE)].copy()
    report = [
        "# NMC 25C Profile Shift Analysis",
        "",
        "## Purpose",
        "",
        "This analysis checks why the clean/base-first protocols keep failing at 25C FUDS. It uses raw 25C profiles and label-free window features. SOC labels are used only for coverage/error analysis, not as model inputs.",
        "",
        "## Profile Summary",
        "",
        _markdown_table(profile_summary),
        "",
        "## SOC-Bin Coverage",
        "",
        _markdown_table(coverage, max_rows=40),
        "",
        "## Label-Free Centroid Distances",
        "",
        "Distances are computed after z-scoring window features on train profiles DST+US06.",
        "",
        _markdown_table(pairwise[pairwise["profile_a"].isin([TEST_PROFILE, VALID_PROFILE])].sort_values(["profile_a", "z_centroid_l2"]), max_rows=20),
        "",
        "## FUDS Nearest-Train Distance by SOC Bin",
        "",
        _markdown_table(nearest_fuds, max_rows=20),
        "",
    ]
    if not old_best_errors.empty:
        report += [
            "## Old Best Stage2 Error on 25C FUDS",
            "",
            "This is included only as context. It is the exploratory train25_dst-selected model, not a clean adoption result.",
            "",
            _markdown_table(old_best_errors, max_rows=20),
            "",
        ]
    if plot_paths:
        report += ["## Plots", ""]
        for p in plot_paths:
            report.append(f"- `{p}`")
        report.append("")
    report += [
        "## Interpretation",
        "",
        "- The base-first failures are consistent with a profile-shift problem rather than an epoch-selection problem.",
        "- If FUDS has high nearest-train distance in specific SOC bins, the next model should target those bins with sequence/shape supervision or pretraining, not another cold/hot correction.",
        "- Passing Stage2 correction results should not be used as a universal-model claim unless the base-only 25C gate is passed first.",
        "",
    ]
    (cfg.out_dir / "nmc_25c_profile_shift_report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"Wrote {cfg.out_dir / 'nmc_25c_profile_shift_report.md'}")
    print(profile_summary.to_string(index=False))
    print(nearest_fuds.to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze NMC 25C profile shift for Strict NoCC SOC experiments.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--raw-root", default=str(AnalysisConfig.raw_root))
    p.add_argument("--out-dir", default=str(AnalysisConfig.out_dir))
    p.add_argument("--window-len", type=int, default=50)
    p.add_argument("--stride", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        AnalysisConfig(
            base_dir=Path(args.base_dir),
            raw_root=Path(args.raw_root),
            out_dir=Path(args.out_dir),
            window_len=int(args.window_len),
            stride=int(args.stride),
        )
    )


if __name__ == "__main__":
    main()
