from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROFILE_FEATURES = [
    "absI_mean",
    "I_std",
    "I_span",
    "dI_abs_mean",
    "dI_energy",
    "low_current_frac",
    "longest_low_current_frac",
    "V_corr_span",
    "V_corr_delta",
    "V_corr_abs_delta",
    "dV_corr_abs_mean",
    "dV_corr_energy",
    "V_pol_mean",
    "V_pol_std",
    "V_pol_span",
    "V_pol_slow_mean",
    "V_pol_slow_span",
    "V_corr_dev_ema200_abs_mean",
    "V_corr_dev_ema800_abs_mean",
    "voltage_response_gain",
]


@dataclass
class Config:
    base_dir: Path = Path(".")
    raw_root: Path = Path("nmc_soc_ocvstart_relabelled_from_lc_ocv/data/NMC SAMSUNG INR 18650 2Ah")
    output_dir: Path = Path("paper_results")
    output_prefix: str = "nmc_profile_regime_shift"
    train_profiles: tuple[str, ...] = ("DST", "US06")
    valid_profiles: tuple[str, ...] = ("VALIDATION",)
    test_profiles: tuple[str, ...] = ("FUDS",)
    window_len: int = 50
    stride: int = 3
    low_current_threshold_A: float = 0.05
    max_knn_reference: int = 6000
    max_knn_query: int = 6000
    random_seed: int = 0


def _table_md(df: pd.DataFrame, cols: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    view = df.loc[:, [c for c in cols if c in df.columns]].head(max_rows)
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"

    def fmt(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in view.to_numpy()]
    return "\n".join([header, sep, *rows])


def _parse_temp_label(value: object) -> float:
    return float(str(value).strip().upper().replace("C", ""))


def _parse_profile(path: Path, df: pd.DataFrame | None = None) -> str:
    if df is not None and "Profile" in df.columns and len(df):
        return str(df["Profile"].iloc[0]).upper()
    return path.stem.split("_")[-1].upper()


def _parse_temperature(path: Path, df: pd.DataFrame | None = None) -> float:
    if df is not None and "TempLabel" in df.columns and len(df):
        return _parse_temp_label(df["TempLabel"].iloc[0])
    for part in path.parts:
        if part.upper().endswith("C"):
            try:
                return _parse_temp_label(part)
            except ValueError:
                pass
    return _parse_temp_label(path.stem.split("_")[1])


def _find_csv_files(root: Path) -> list[Path]:
    files = sorted(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No NMC CSV files found under {root}")
    return files


def _causal_ema(values: np.ndarray, times_s: np.ndarray, tau_s: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    t = np.asarray(times_s, dtype=np.float64)
    if len(x) == 0:
        return x.astype(np.float32)
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    if len(x) == 1:
        return out.astype(np.float32)
    dt_all = np.diff(t)
    dt_default = float(np.nanmedian(dt_all[np.isfinite(dt_all) & (dt_all > 0)]))
    if not np.isfinite(dt_default) or dt_default <= 0:
        dt_default = 1.0
    for idx in range(1, len(x)):
        dt = t[idx] - t[idx - 1]
        if not np.isfinite(dt) or dt <= 0:
            dt = dt_default
        alpha = float(np.exp(-dt / max(float(tau_s), 1e-6)))
        alpha = min(max(alpha, 0.0), 0.999999)
        out[idx] = alpha * out[idx - 1] + (1.0 - alpha) * x[idx]
    return out.astype(np.float32)


def _causal_index_ema(values: np.ndarray, tau: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr.astype(np.float32)
    alpha = float(np.exp(-1.0 / max(float(tau), 1e-6)))
    alpha = min(max(alpha, 0.0), 0.999999)
    out = np.empty_like(arr, dtype=np.float64)
    out[0] = arr[0]
    for idx in range(1, len(arr)):
        out[idx] = alpha * out[idx - 1] + (1.0 - alpha) * arr[idx]
    return out.astype(np.float32)


def _estimate_r0_by_temperature(files: list[Path], train_profiles: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    all_ratios: list[float] = []
    train_set = {str(x).upper() for x in train_profiles}
    for path in files:
        head = pd.read_csv(path, nrows=2)
        profile = _parse_profile(path, head)
        if profile not in train_set:
            continue
        temp = _parse_temperature(path, head)
        df = pd.read_csv(path, usecols=["Current(A)", "Voltage(V)"])
        i_raw = pd.to_numeric(df["Current(A)"], errors="coerce").to_numpy(np.float64)
        v_raw = pd.to_numeric(df["Voltage(V)"], errors="coerce").to_numpy(np.float64)
        d_i = np.diff(i_raw, prepend=i_raw[0])
        d_v = np.diff(v_raw, prepend=v_raw[0])
        mask = np.isfinite(d_i) & np.isfinite(d_v) & (np.abs(d_i) > 0.05) & (np.abs(d_v) > 1e-5)
        ratio = d_v[mask] / d_i[mask]
        ratio = ratio[np.isfinite(ratio) & (ratio > 0.001) & (ratio < 0.5)]
        rows.extend({"temperature_C": float(temp), "r0_event_ohm": float(value)} for value in ratio)
        all_ratios.extend(float(value) for value in ratio)
    event_df = pd.DataFrame(rows)
    if event_df.empty:
        raise RuntimeError("Could not estimate R0 from train profiles.")
    fallback = float(np.median(all_ratios))
    summary = event_df.groupby("temperature_C")["r0_event_ohm"].agg(r0_ohm="median", n_events="count").reset_index()
    present = set(summary["temperature_C"].astype(float))
    all_temps = sorted({_parse_temperature(path, pd.read_csv(path, nrows=2)) for path in files})
    for temp in all_temps:
        if float(temp) not in present:
            summary = pd.concat(
                [summary, pd.DataFrame([{"temperature_C": float(temp), "r0_ohm": fallback, "n_events": 0}])],
                ignore_index=True,
            )
    return summary.sort_values("temperature_C").reset_index(drop=True)


def _build_decomposed_frame(path: Path, r0_lookup: dict[float, float]) -> pd.DataFrame:
    df = pd.read_csv(path)
    temp = _parse_temperature(path, df)
    profile = _parse_profile(path, df)
    r0 = float(r0_lookup[float(temp)])
    time_col = "Step_Time(s)" if "Step_Time(s)" in df.columns else "Test_Time(s)"
    times = pd.to_numeric(df[time_col], errors="coerce").to_numpy(np.float64)
    v_raw = pd.to_numeric(df["Voltage(V)"], errors="coerce").to_numpy(np.float64)
    i_raw = pd.to_numeric(df["Current(A)"], errors="coerce").to_numpy(np.float64)
    v_ohm = i_raw * r0
    v_corr = _causal_ema(v_raw - v_ohm, times, 120.0)
    dynamic = v_raw - v_corr - v_ohm
    dyn_slow = _causal_ema(dynamic, times, 600.0)
    soc = pd.to_numeric(df["SOC_CC"], errors="coerce").to_numpy(np.float64)
    if np.nanmax(soc) > 1.5:
        soc = soc / 100.0
    frame = pd.DataFrame(
        {
            "file_name": path.name,
            "trajectory_id": path.stem,
            "temperature": float(temp),
            "drive_cycle": profile,
            "SOC_physical": np.clip(soc, 0.0, 1.0),
            "V_raw": v_raw.astype(np.float32),
            "V_corr_raw": v_corr.astype(np.float32),
            "I_raw": i_raw.astype(np.float32),
            "V_pol_raw": dynamic.astype(np.float32),
            "V_pol_slow_raw": dyn_slow.astype(np.float32),
            "R0": np.full(len(df), r0, dtype=np.float32),
        }
    )
    for tau in (200, 800):
        ema = _causal_index_ema(frame["V_corr_raw"].to_numpy(np.float32), tau)
        frame[f"V_corr_raw_ema{tau}"] = ema
        frame[f"V_corr_raw_dev_ema{tau}"] = frame["V_corr_raw"].to_numpy(np.float32) - ema
    return frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def _longest_true_frac(mask: np.ndarray) -> float:
    best = 0
    cur = 0
    for value in mask.astype(bool):
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return float(best / max(len(mask), 1))


def _window_rows(frames: list[pd.DataFrame], split: str, cfg: Config) -> list[dict]:
    rows: list[dict] = []
    window_len = int(cfg.window_len)
    stride = int(cfg.stride)
    for frame in frames:
        if len(frame) < window_len:
            continue
        temp = float(frame["temperature"].iloc[0])
        profile = str(frame["drive_cycle"].iloc[0]).upper()
        trajectory_id = str(frame["trajectory_id"].iloc[0])
        file_name = str(frame["file_name"].iloc[0])
        v_corr = frame["V_corr_raw"].to_numpy(np.float64)
        i_raw = frame["I_raw"].to_numpy(np.float64)
        v_pol = frame["V_pol_raw"].to_numpy(np.float64) if "V_pol_raw" in frame else np.zeros(len(frame), dtype=np.float64)
        v_pol_slow = (
            frame["V_pol_slow_raw"].to_numpy(np.float64) if "V_pol_slow_raw" in frame else np.zeros(len(frame), dtype=np.float64)
        )
        v_dev200 = (
            frame["V_corr_raw_dev_ema200"].to_numpy(np.float64)
            if "V_corr_raw_dev_ema200" in frame
            else np.zeros(len(frame), dtype=np.float64)
        )
        v_dev800 = (
            frame["V_corr_raw_dev_ema800"].to_numpy(np.float64)
            if "V_corr_raw_dev_ema800" in frame
            else np.zeros(len(frame), dtype=np.float64)
        )
        soc = frame["SOC_physical"].to_numpy(np.float64) if "SOC_physical" in frame else np.full(len(frame), np.nan)

        for start in range(0, len(frame) - window_len + 1, stride):
            end = start + window_len
            vc = v_corr[start:end]
            ii = i_raw[start:end]
            vp = v_pol[start:end]
            vps = v_pol_slow[start:end]
            d_i = np.diff(ii, prepend=ii[0])
            d_vc = np.diff(vc, prepend=vc[0])
            low = np.abs(ii) < float(cfg.low_current_threshold_A)
            i_span = float(np.nanmax(ii) - np.nanmin(ii))
            v_span = float(np.nanmax(vc) - np.nanmin(vc))
            rows.append(
                {
                    "split": split,
                    "temperature_C": temp,
                    "profile": profile,
                    "trajectory_id": trajectory_id,
                    "file_name": file_name,
                    "start_index": int(start),
                    "end_index": int(end - 1),
                    "soc_end_label_diagnostic_only": float(soc[end - 1]),
                    "absI_mean": float(np.nanmean(np.abs(ii))),
                    "I_std": float(np.nanstd(ii)),
                    "I_span": i_span,
                    "dI_abs_mean": float(np.nanmean(np.abs(d_i))),
                    "dI_energy": float(np.nanmean(np.square(d_i))),
                    "low_current_frac": float(np.nanmean(low)),
                    "longest_low_current_frac": _longest_true_frac(low),
                    "V_corr_span": v_span,
                    "V_corr_delta": float(vc[-1] - vc[0]),
                    "V_corr_abs_delta": float(abs(vc[-1] - vc[0])),
                    "dV_corr_abs_mean": float(np.nanmean(np.abs(d_vc))),
                    "dV_corr_energy": float(np.nanmean(np.square(d_vc))),
                    "V_pol_mean": float(np.nanmean(vp)),
                    "V_pol_std": float(np.nanstd(vp)),
                    "V_pol_span": float(np.nanmax(vp) - np.nanmin(vp)),
                    "V_pol_slow_mean": float(np.nanmean(vps)),
                    "V_pol_slow_span": float(np.nanmax(vps) - np.nanmin(vps)),
                    "V_corr_dev_ema200_abs_mean": float(np.nanmean(np.abs(v_dev200[start:end]))),
                    "V_corr_dev_ema800_abs_mean": float(np.nanmean(np.abs(v_dev800[start:end]))),
                    "voltage_response_gain": float(v_span / max(i_span, 1e-6)),
                }
            )
    return rows


def _build_windows(cfg: Config) -> pd.DataFrame:
    root = cfg.base_dir / cfg.raw_root if not cfg.raw_root.is_absolute() else cfg.raw_root
    files = _find_csv_files(root)
    r0_df = _estimate_r0_by_temperature(files, cfg.train_profiles)
    r0_lookup = {float(row["temperature_C"]): float(row["r0_ohm"]) for _, row in r0_df.iterrows()}
    train_set = {str(x).upper() for x in cfg.train_profiles}
    valid_set = {str(x).upper() for x in cfg.valid_profiles}
    test_set = {str(x).upper() for x in cfg.test_profiles}
    frames = {"train": [], "valid": [], "test": []}
    for path in files:
        head = pd.read_csv(path, nrows=2)
        profile = _parse_profile(path, head)
        frame = _build_decomposed_frame(path, r0_lookup)
        if profile in train_set:
            frames["train"].append(frame)
        elif profile in valid_set:
            frames["valid"].append(frame)
        elif profile in test_set:
            frames["test"].append(frame)
    rows = []
    for split, split_frames in frames.items():
        rows.extend(_window_rows(split_frames, split, cfg))
    out = pd.DataFrame(rows)
    for col in PROFILE_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=PROFILE_FEATURES).reset_index(drop=True)


def _standardize_by_train(windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    z_parts = []
    for temp, temp_df in windows.groupby("temperature_C"):
        train = temp_df[temp_df["profile"].isin(["DST", "US06"])]
        if train.empty:
            continue
        means = train[PROFILE_FEATURES].mean()
        stds = train[PROFILE_FEATURES].std(ddof=0).replace(0.0, 1.0)
        ref_q05 = train[PROFILE_FEATURES].quantile(0.05)
        ref_q95 = train[PROFILE_FEATURES].quantile(0.95)
        z = temp_df.copy()
        z_features = (z[PROFILE_FEATURES] - means) / stds
        for col in PROFILE_FEATURES:
            z[f"z_{col}"] = z_features[col]
        z["outside_train_5_95_frac"] = ((z[PROFILE_FEATURES] < ref_q05) | (z[PROFILE_FEATURES] > ref_q95)).mean(axis=1)
        z_parts.append(z)
        for col in PROFILE_FEATURES:
            rows.append(
                {
                    "temperature_C": float(temp),
                    "feature": col,
                    "train_mean": float(means[col]),
                    "train_std": float(stds[col]),
                    "train_q05": float(ref_q05[col]),
                    "train_q95": float(ref_q95[col]),
                }
            )
    return pd.concat(z_parts, ignore_index=True), pd.DataFrame(rows)


def _knn_distance(query: np.ndarray, reference: np.ndarray, max_query: int, max_ref: int, seed: int) -> float:
    if len(query) == 0 or len(reference) == 0:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    if len(query) > max_query:
        query = query[rng.choice(len(query), size=max_query, replace=False)]
    if len(reference) > max_ref:
        reference = reference[rng.choice(len(reference), size=max_ref, replace=False)]
    mins = []
    chunk = 512
    for start in range(0, len(query), chunk):
        q = query[start : start + chunk]
        dist2 = ((q[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
        mins.append(np.sqrt(np.min(dist2, axis=1)))
    return float(np.mean(np.concatenate(mins)))


def _profile_summary(z: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    z_cols = [f"z_{c}" for c in PROFILE_FEATURES]
    for temp, temp_df in z.groupby("temperature_C"):
        train = temp_df[temp_df["profile"].isin(["DST", "US06"])]
        train_mat = train[z_cols].to_numpy(np.float64)
        for profile, g in temp_df.groupby("profile"):
            centroid = g[z_cols].mean().to_numpy(np.float64)
            rows.append(
                {
                    "temperature_C": float(temp),
                    "profile": str(profile),
                    "split": str(g["split"].iloc[0]),
                    "n_windows": int(len(g)),
                    "centroid_distance_to_train": float(np.linalg.norm(centroid)),
                    "mean_outside_train_5_95_frac": float(g["outside_train_5_95_frac"].mean()),
                    "knn_distance_to_train": _knn_distance(
                        g[z_cols].to_numpy(np.float64),
                        train_mat,
                        int(cfg.max_knn_query),
                        int(cfg.max_knn_reference),
                        int(cfg.random_seed) + int(float(temp) * 10) + len(str(profile)),
                    ),
                    "mean_absI": float(g["absI_mean"].mean()),
                    "mean_low_current_frac": float(g["low_current_frac"].mean()),
                    "mean_V_corr_span": float(g["V_corr_span"].mean()),
                    "mean_dI_energy": float(g["dI_energy"].mean()),
                    "mean_soc_end_label_diagnostic_only": float(g["soc_end_label_diagnostic_only"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["temperature_C", "centroid_distance_to_train"], ascending=[True, False])


def _pairwise_profile_distance(z: pd.DataFrame) -> pd.DataFrame:
    rows = []
    z_cols = [f"z_{c}" for c in PROFILE_FEATURES]
    for temp, temp_df in z.groupby("temperature_C"):
        centroids = {profile: g[z_cols].mean().to_numpy(np.float64) for profile, g in temp_df.groupby("profile")}
        profiles = sorted(centroids)
        for i, p1 in enumerate(profiles):
            for p2 in profiles[i + 1 :]:
                rows.append(
                    {
                        "temperature_C": float(temp),
                        "profile_a": p1,
                        "profile_b": p2,
                        "centroid_distance": float(np.linalg.norm(centroids[p1] - centroids[p2])),
                    }
                )
    return pd.DataFrame(rows).sort_values(["temperature_C", "centroid_distance"], ascending=[True, False])


def _feature_shift(z: pd.DataFrame) -> pd.DataFrame:
    rows = []
    z_cols = [f"z_{c}" for c in PROFILE_FEATURES]
    for temp, temp_df in z.groupby("temperature_C"):
        means = {profile: g[z_cols].mean() for profile, g in temp_df.groupby("profile")}
        for profile, mean in means.items():
            for feat in PROFILE_FEATURES:
                rows.append(
                    {
                        "temperature_C": float(temp),
                        "profile": str(profile),
                        "feature": feat,
                        "z_mean_vs_train": float(mean[f"z_{feat}"]),
                        "abs_z_mean_vs_train": float(abs(mean[f"z_{feat}"])),
                    }
                )
        if "FUDS" in means and "VALIDATION" in means:
            diff = means["FUDS"] - means["VALIDATION"]
            for feat in PROFILE_FEATURES:
                rows.append(
                    {
                        "temperature_C": float(temp),
                        "profile": "FUDS_minus_VALIDATION",
                        "feature": feat,
                        "z_mean_vs_train": float(diff[f"z_{feat}"]),
                        "abs_z_mean_vs_train": float(abs(diff[f"z_{feat}"])),
                    }
                )
    return pd.DataFrame(rows).sort_values(["temperature_C", "profile", "abs_z_mean_vs_train"], ascending=[True, True, False])


def _pair_distance(pairwise: pd.DataFrame, temp: float, a: str, b: str) -> float:
    aa = str(a).upper()
    bb = str(b).upper()
    sub = pairwise[
        np.isclose(pairwise["temperature_C"], temp)
        & (
            ((pairwise["profile_a"] == aa) & (pairwise["profile_b"] == bb))
            | ((pairwise["profile_a"] == bb) & (pairwise["profile_b"] == aa))
        )
    ]
    return float(sub["centroid_distance"].iloc[0]) if len(sub) else float("nan")


def _validation_coverage_gate(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for temp in sorted(pairwise["temperature_C"].dropna().unique()):
        train_spread = _pair_distance(pairwise, float(temp), "DST", "US06")
        valid_test = _pair_distance(pairwise, float(temp), "VALIDATION", "FUDS")
        fuds_to_dst = _pair_distance(pairwise, float(temp), "FUDS", "DST")
        fuds_to_us06 = _pair_distance(pairwise, float(temp), "FUDS", "US06")
        nearest_train = float(np.nanmin([fuds_to_dst, fuds_to_us06]))
        rows.append(
            {
                "temperature_C": float(temp),
                "train_internal_DST_US06_distance": train_spread,
                "valid_VALIDATION_to_test_FUDS_distance": valid_test,
                "FUDS_nearest_train_distance": nearest_train,
                "valid_test_over_train_spread": float(valid_test / train_spread) if train_spread else np.nan,
                "valid_test_over_nearest_train": float(valid_test / nearest_train) if nearest_train else np.nan,
                "validation_covers_test_regime": bool(valid_test <= train_spread),
                "adoption_implication": "VALIDATION_only_selector_not_representative" if valid_test > train_spread else "VALIDATION_selector_coverage_ok",
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    cfg: Config,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    feature_shift: pd.DataFrame,
    coverage_gate: pd.DataFrame,
) -> None:
    focus25 = summary[np.isclose(summary["temperature_C"], 25.0)].copy()
    pair25 = pairwise[np.isclose(pairwise["temperature_C"], 25.0)].copy()
    fuds_vs_VALIDATION25 = feature_shift[
        np.isclose(feature_shift["temperature_C"], 25.0) & (feature_shift["profile"] == "FUDS_minus_VALIDATION")
    ].head(12)
    fuds25 = feature_shift[np.isclose(feature_shift["temperature_C"], 25.0) & (feature_shift["profile"] == "FUDS")].head(12)

    lines = [
        "# NMC Profile-Regime Shift Diagnostic",
        "",
        "## Scope",
        "",
        "This diagnostic is NMC strict NoCC compatible. It uses only label-free voltage/current/temperature window-response features for the distribution-shift calculations.",
        "",
        "`soc_end_label_diagnostic_only` is included only as an optional coverage diagnostic column in the CSV. It is not used for distances, ranking, or model selection.",
        "",
        "Train reference profiles: DST + US06. Validation profile: VALIDATION. Test profile: FUDS.",
        "",
        "## 25C Profile Distances",
        "",
        _table_md(
            focus25,
            [
                "profile",
                "split",
                "n_windows",
                "centroid_distance_to_train",
                "mean_outside_train_5_95_frac",
                "knn_distance_to_train",
                "mean_absI",
                "mean_low_current_frac",
                "mean_V_corr_span",
                "mean_dI_energy",
            ],
        ),
        "",
        "## 25C Pairwise Distances",
        "",
        _table_md(pair25, ["profile_a", "profile_b", "centroid_distance"]),
        "",
        "## Validation Coverage Gate",
        "",
        _table_md(
            coverage_gate,
            [
                "temperature_C",
                "train_internal_DST_US06_distance",
                "valid_VALIDATION_to_test_FUDS_distance",
                "valid_test_over_train_spread",
                "validation_covers_test_regime",
                "adoption_implication",
            ],
        ),
        "",
        "## 25C FUDS vs VALIDATION Feature Shifts",
        "",
        _table_md(fuds_vs_VALIDATION25, ["feature", "z_mean_vs_train", "abs_z_mean_vs_train"]),
        "",
        "## 25C FUDS Shift vs Train Reference",
        "",
        _table_md(fuds25, ["feature", "z_mean_vs_train", "abs_z_mean_vs_train"]),
        "",
        "## Interpretation",
        "",
        "This diagnostic checks whether VALIDATION is a reliable validation proxy for FUDS in label-free response-feature space.",
        "",
        "If FUDS is far from VALIDATION or has different dominant shifted features, then a VALIDATION-only selector can be clean but still non-representative. That supports the current adoption rule: use ProfileLOO/profile rotation before claiming a universal strict NoCC SOC model.",
        "",
        "It does not justify adding Stage2 correction. It only identifies which base-model regime shift must be handled before correction can be discussed.",
        "",
        "## Outputs",
        "",
        f"- `{cfg.output_dir / (cfg.output_prefix + '_window_features.csv.gz')}`",
        f"- `{cfg.output_dir / (cfg.output_prefix + '_profile_summary.csv')}`",
        f"- `{cfg.output_dir / (cfg.output_prefix + '_pairwise_distances.csv')}`",
        f"- `{cfg.output_dir / (cfg.output_prefix + '_validation_coverage_gate.csv')}`",
        f"- `{cfg.output_dir / (cfg.output_prefix + '_feature_shifts.csv')}`",
        f"- `{cfg.output_dir / (cfg.output_prefix + '_feature_reference.csv')}`",
    ]
    (cfg.output_dir / f"{cfg.output_prefix}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cfg: Config = Config()) -> dict[str, pd.DataFrame]:
    cfg.base_dir = Path(cfg.base_dir).resolve()
    cfg.output_dir = cfg.base_dir / cfg.output_dir if not cfg.output_dir.is_absolute() else cfg.output_dir
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    windows = _build_windows(cfg)
    z, reference = _standardize_by_train(windows)
    summary = _profile_summary(z, cfg)
    pairwise = _pairwise_profile_distance(z)
    feature_shift = _feature_shift(z)
    coverage_gate = _validation_coverage_gate(pairwise)

    z.to_csv(cfg.output_dir / f"{cfg.output_prefix}_window_features.csv.gz", index=False, compression="gzip")
    reference.to_csv(cfg.output_dir / f"{cfg.output_prefix}_feature_reference.csv", index=False)
    summary.to_csv(cfg.output_dir / f"{cfg.output_prefix}_profile_summary.csv", index=False)
    pairwise.to_csv(cfg.output_dir / f"{cfg.output_prefix}_pairwise_distances.csv", index=False)
    coverage_gate.to_csv(cfg.output_dir / f"{cfg.output_prefix}_validation_coverage_gate.csv", index=False)
    feature_shift.to_csv(cfg.output_dir / f"{cfg.output_prefix}_feature_shifts.csv", index=False)
    _write_report(cfg, summary, pairwise, feature_shift, coverage_gate)
    return {
        "windows": z,
        "reference": reference,
        "summary": summary,
        "pairwise": pairwise,
        "coverage_gate": coverage_gate,
        "feature_shift": feature_shift,
    }


def main() -> None:
    run()


if __name__ == "__main__":
    main()
