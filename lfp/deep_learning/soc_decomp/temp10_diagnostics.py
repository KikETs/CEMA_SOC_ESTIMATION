import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import CFG

try:
    from IPython.display import display
except Exception:
    display = print

try:
    from scipy.stats import ks_2samp as _scipy_ks_2samp
    from scipy.stats import wasserstein_distance as _scipy_wasserstein
except Exception:
    _scipy_ks_2samp = None
    _scipy_wasserstein = None


TEMP10_FEATURE_SETS = {
    "S1_V_pol": ["V_pol_raw"],
    "S2_V_hys": ["V_hys_raw"],
    "S3_R0": ["R0"],
    "S4_pol_hys_R0": ["V_pol_raw", "V_hys_raw", "R0"],
    "S5_full_components": ["V_corr_raw", "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0"],
}


def _as_cfg(cfg):
    if cfg is not None:
        return cfg
    from .config import make_cfg

    return make_cfg()


def _collect_feature_frames_from_artifacts(artifacts, splits=("test",)):
    rows = []
    for split in splits:
        for frame in artifacts["feature_frames"].get(split, []):
            rows.append(frame.assign(split=split))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_feature_frames_from_csv(cfg: CFG, splits=("test",)):
    # File-level split is encoded by drive_cycle under the current DST/US06 train, FUDS eval setup.
    wanted_drive = {"test": {str(getattr(cfg, "eval_drive", "FUDS")).upper()}, "train": set(), "valid": set()}
    train_drives = {str(d).upper() for d in getattr(cfg, "train_drives", ("DST", "US06"))}
    wanted_drive["train"] = train_drives
    rows = []
    for path in sorted(Path(cfg.decomposed_dir).glob("*_features.csv")):
        df = pd.read_csv(path)
        if df.empty or "drive_cycle" not in df.columns:
            continue
        drive = str(df["drive_cycle"].iloc[0]).upper()
        for split in splits:
            if drive in wanted_drive.get(split, set()):
                rows.append(df.assign(split=split))
                break
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _feature_df(artifacts=None, cfg=None, splits=("test",)):
    cfg = _as_cfg(cfg if cfg is not None else (artifacts.get("cfg") if artifacts else None))
    if artifacts is not None and "feature_frames" in artifacts:
        df = _collect_feature_frames_from_artifacts(artifacts, splits=splits)
        if len(df):
            return df, cfg
    return load_feature_frames_from_csv(cfg, splits=splits), cfg


def _clean_vector(x):
    arr = np.asarray(x, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _ks_distance(x, y):
    x = _clean_vector(x)
    y = _clean_vector(y)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    if _scipy_ks_2samp is not None:
        return float(_scipy_ks_2samp(x, y).statistic)
    vals = np.sort(np.unique(np.concatenate([x, y])))
    fx = np.searchsorted(np.sort(x), vals, side="right") / len(x)
    fy = np.searchsorted(np.sort(y), vals, side="right") / len(y)
    return float(np.max(np.abs(fx - fy)))


def _wasserstein_distance(x, y):
    x = _clean_vector(x)
    y = _clean_vector(y)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    if _scipy_wasserstein is not None:
        return float(_scipy_wasserstein(x, y))
    q = np.linspace(0.0, 1.0, min(2000, max(len(x), len(y))))
    return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))


def _regularized_mahalanobis(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x).all(axis=1)]
    y = y[np.isfinite(y).all(axis=1)]
    if len(x) < 3 or len(y) < 3:
        return float("nan")
    dx = x.mean(axis=0) - y.mean(axis=0)
    cov = np.cov(np.vstack([x, y]).T)
    if cov.ndim == 0:
        cov = np.asarray([[float(cov)]])
    ridge = 1e-6 * np.eye(cov.shape[0])
    inv_cov = np.linalg.pinv(cov + ridge)
    return float(np.sqrt(max(0.0, dx @ inv_cov @ dx.T)))


def _deterministic_subsample(x, max_n=1500):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x).all(axis=1)]
    if len(x) <= max_n:
        return x
    idx = np.linspace(0, len(x) - 1, max_n).round().astype(int)
    return x[idx]


def _rbf_mmd(x, y):
    x = _deterministic_subsample(x)
    y = _deterministic_subsample(y)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    xy = np.vstack([x, y])
    sample = xy[_deterministic_subsample(np.arange(len(xy)).reshape(-1, 1), max_n=min(600, len(xy))).reshape(-1).astype(int)]
    d = np.sum((sample[:, None, :] - sample[None, :, :]) ** 2, axis=-1)
    sigma2 = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    sigma2 = max(float(sigma2), 1e-12)

    def kernel(a, b):
        dist2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-dist2 / (2.0 * sigma2))

    kxx = kernel(x, x)
    kyy = kernel(y, y)
    kxy = kernel(x, y)
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def component_temperature_distance(
    feature_df,
    cfg: CFG,
    make_plots=True,
    csv_name="component_temperature_distance.csv",
    heatmap_name="component_temperature_distance_heatmap.png",
):
    if feature_df.empty:
        raise ValueError("No decomposed feature rows available for component temperature distance.")
    comparisons = [0.0, 20.0, 25.0, -10.0, 50.0]
    temp10 = feature_df[np.isclose(feature_df["temperature"].astype(float), 10.0)]
    if temp10.empty:
        raise ValueError("No 10C rows found in feature frames.")
    rows = []
    for set_name, cols in TEMP10_FEATURE_SETS.items():
        missing = [c for c in cols if c not in feature_df.columns]
        if missing:
            warnings.warn(f"Skipping {set_name}; missing columns: {missing}")
            continue
        x_joint = temp10[cols].to_numpy(np.float64)
        for other_temp in comparisons:
            other = feature_df[np.isclose(feature_df["temperature"].astype(float), other_temp)]
            if other.empty:
                continue
            y_joint = other[cols].to_numpy(np.float64)
            rows.append({
                "feature_set": set_name,
                "feature_cols": ",".join(cols),
                "comparison": f"10C_vs_{int(other_temp)}C",
                "temperature_a": 10.0,
                "temperature_b": other_temp,
                "metric": "mahalanobis",
                "feature": "joint",
                "value": _regularized_mahalanobis(x_joint, y_joint),
            })
            rows.append({
                "feature_set": set_name,
                "feature_cols": ",".join(cols),
                "comparison": f"10C_vs_{int(other_temp)}C",
                "temperature_a": 10.0,
                "temperature_b": other_temp,
                "metric": "rbf_mmd",
                "feature": "joint",
                "value": _rbf_mmd(x_joint, y_joint),
            })
            for col in cols:
                rows.append({
                    "feature_set": set_name,
                    "feature_cols": ",".join(cols),
                    "comparison": f"10C_vs_{int(other_temp)}C",
                    "temperature_a": 10.0,
                    "temperature_b": other_temp,
                    "metric": "ks_distance",
                    "feature": col,
                    "value": _ks_distance(temp10[col], other[col]),
                })
                rows.append({
                    "feature_set": set_name,
                    "feature_cols": ",".join(cols),
                    "comparison": f"10C_vs_{int(other_temp)}C",
                    "temperature_a": 10.0,
                    "temperature_b": other_temp,
                    "metric": "wasserstein",
                    "feature": col,
                    "value": _wasserstein_distance(temp10[col], other[col]),
                })
    out = pd.DataFrame(rows)
    out.to_csv(cfg.output_dir / csv_name, index=False)

    if make_plots and len(out):
        heat = out[(out["metric"] == "mahalanobis") & (out["feature"] == "joint")].pivot(
            index="feature_set",
            columns="comparison",
            values="value",
        )
        plt.figure(figsize=(8, 3.6))
        im = plt.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        plt.colorbar(im, label="Mahalanobis distance")
        plt.xticks(np.arange(len(heat.columns)), heat.columns, rotation=30, ha="right")
        plt.yticks(np.arange(len(heat.index)), heat.index)
        plt.title("10C component separability by learned voltage feature set")
        plt.tight_layout()
        plt.savefig(cfg.output_dir / heatmap_name, dpi=180)
        plt.close()
    return out


def event_aligned_component_response(feature_df, cfg: CFG, make_plots=True, percentile=95.0, offsets=(1, 5, 10, 30)):
    if feature_df.empty:
        raise ValueError("No decomposed feature rows available for event-aligned diagnostic.")
    event_rows = []
    for tid, g in feature_df.sort_values(["trajectory_id", "end_index"]).groupby("trajectory_id"):
        g = g.reset_index(drop=True)
        if len(g) <= max(offsets) + 1:
            continue
        threshold = np.nanpercentile(np.abs(g["dI"].to_numpy(dtype=float)), percentile)
        event_idx = np.flatnonzero(np.abs(g["dI"].to_numpy(dtype=float)) > threshold)
        for idx in event_idx:
            if idx + max(offsets) >= len(g):
                continue
            row = {
                "trajectory_id": tid,
                "temperature": float(g.loc[idx, "temperature"]),
                "drive_cycle": g.loc[idx, "drive_cycle"],
                "event_index": int(g.loc[idx, "end_index"]),
                "abs_dI": float(abs(g.loc[idx, "dI"])),
                "R0_near_event": float(g.loc[idx, "R0"]),
            }
            for col in ["V_pol_raw", "V_hys_raw"]:
                base = float(g.loc[idx, col])
                for off in offsets:
                    row[f"{col}_response_tplus_{off}"] = float(g.loc[idx + off, col] - base)
            for col in ["V_raw", "V_corr_raw"]:
                row[f"{col}_recovery_slope_tplus_{max(offsets)}"] = float(
                    (g.loc[idx + max(offsets), col] - g.loc[idx, col]) / max(offsets)
                )
            event_rows.append(row)
    events = pd.DataFrame(event_rows)
    if events.empty:
        warnings.warn("No current-change events found for event-aligned diagnostic.")
        events.to_csv(cfg.output_dir / "event_aligned_component_response_by_temperature.csv", index=False)
        return events
    value_cols = [c for c in events.columns if c not in {"trajectory_id", "temperature", "drive_cycle", "event_index"}]
    summary = (
        events.groupby("temperature")[value_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(c) for c in col if c != ""]).rstrip("_")
        if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    summary.to_csv(cfg.output_dir / "event_aligned_component_response_by_temperature.csv", index=False)

    def plot_response(component, out_name):
        plt.figure(figsize=(7.5, 3.5))
        for temp, g in events.groupby("temperature"):
            means = [g[f"{component}_response_tplus_{off}"].mean() for off in offsets]
            plt.plot(offsets, means, marker="o", label=f"{int(temp)}C")
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel("samples after current-change event")
        plt.ylabel("raw-voltage response (V)")
        title_name = "polarization-like" if component == "V_pol_raw" else "hysteresis-like"
        plt.title(f"Event-aligned {title_name} component response")
        plt.legend(ncol=4, fontsize=8)
        plt.tight_layout()
        plt.savefig(cfg.output_dir / out_name, dpi=180)
        plt.close()

    if make_plots:
        plot_response("V_pol_raw", "event_aligned_pol_response_plot.png")
        plot_response("V_hys_raw", "event_aligned_hys_response_plot.png")
    return summary


def _prediction_df_from_artifacts(artifacts=None, cfg=None):
    if artifacts is not None and "ablation" in artifacts:
        rows = []
        for (target, ablation), split_frames in artifacts["ablation"]["all_predictions"].items():
            df = split_frames.get("test", pd.DataFrame())
            if len(df):
                rows.append(df.assign(target_label=target, ablation=ablation, split="test"))
        if rows:
            return pd.concat(rows, ignore_index=True)
    cfg = _as_cfg(cfg if cfg is not None else (artifacts.get("cfg") if artifacts else None))
    path = cfg.output_dir / "all_predictions_fixed_labels.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _normalize_prediction_columns(pred_df):
    out = pred_df.copy()
    if "target_label" not in out.columns and "label_type" in out.columns:
        out["target_label"] = out["label_type"]
    if "ablation" not in out.columns and "model_name" in out.columns:
        out["ablation"] = out["model_name"]
    if "temperature" not in out.columns and "temperature_C" in out.columns:
        out["temperature"] = out["temperature_C"]
    if "end_index" not in out.columns and "time_index" in out.columns:
        out["end_index"] = out["time_index"]
    if "abs_error" not in out.columns and "error" in out.columns:
        out["abs_error"] = out["error"].abs()
    return out


def _safe_corr(df, a, b):
    if a not in df.columns or b not in df.columns:
        return float("nan")
    sub = df[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 3 or sub[a].std() <= 1e-12 or sub[b].std() <= 1e-12:
        return float("nan")
    return float(sub[a].corr(sub[b]))


def classify_temp10_error(detail, ablation):
    if detail.empty:
        return pd.DataFrame(), "No 10C prediction rows are available for classification."
    overall_mae = float(detail["abs_error"].mean())
    mean_error = float(detail["error"].mean())
    error_std = float(detail["error"].std())
    rows = [{
        "ablation": ablation,
        "scope": "all_10C",
        "n_windows": int(len(detail)),
        "mean_error": mean_error,
        "median_error": float(detail["error"].median()),
        "error_std": error_std,
        "MAE": overall_mae,
        "MAE_pct": overall_mae * 100.0,
    }]
    for phase, g in detail.groupby("phase", observed=True):
        rows.append({
            "ablation": ablation,
            "scope": f"{phase}_trajectory",
            "n_windows": int(len(g)),
            "mean_error": float(g["error"].mean()),
            "median_error": float(g["error"].median()),
            "error_std": float(g["error"].std()),
            "MAE": float(g["abs_error"].mean()),
            "MAE_pct": float(g["abs_error"].mean() * 100.0),
        })
    for scope, mask in [
        ("plateau_20_80", detail["is_plateau_20_80"]),
        ("cutoff_last10", detail["is_cutoff_last10"]),
    ]:
        g = detail[mask]
        if len(g):
            rows.append({
                "ablation": ablation,
                "scope": scope,
                "n_windows": int(len(g)),
                "mean_error": float(g["error"].mean()),
                "median_error": float(g["error"].median()),
                "error_std": float(g["error"].std()),
                "MAE": float(g["abs_error"].mean()),
                "MAE_pct": float(g["abs_error"].mean() * 100.0),
            })
    summary = pd.DataFrame(rows)
    corr_inputs = {
        "corr_error_time_index": "end_index",
        "corr_error_cumulative_Ah": "cumulative_Ah",
        "corr_error_V_raw": "V_raw",
        "corr_error_V_pol_raw": "V_pol_raw",
        "corr_error_V_hys_raw": "V_hys_raw",
        "corr_error_R0": "R0",
    }
    for out_col, in_col in corr_inputs.items():
        summary.loc[summary["scope"] == "all_10C", out_col] = _safe_corr(detail, "error", in_col)

    phase_mae = summary[summary["scope"].str.endswith("_trajectory")].set_index("scope")["MAE"]
    all_row = summary[summary["scope"] == "all_10C"].iloc[0]
    corr_values = {
        col: float(all_row[col])
        for col in corr_inputs
        if col in all_row.index and pd.notna(all_row[col])
    }
    strongest = max(corr_values.items(), key=lambda kv: abs(kv[1])) if corr_values else (None, float("nan"))
    verdicts = []
    bias_ratio = abs(mean_error) / max(error_std, 1e-12)
    if bias_ratio >= 0.75:
        verdicts.append(f"bias/calibration-like: abs(mean_error)/error_std={bias_ratio:.2f}")
    else:
        verdicts.append(f"bias not dominant: abs(mean_error)/error_std={bias_ratio:.2f}")
    if len(phase_mae) and phase_mae.max() > max(overall_mae * 1.35, overall_mae + 0.01):
        verdicts.append(f"shape/dynamics-like: localized max at {phase_mae.idxmax()}")
    else:
        verdicts.append("no single early/mid/late segment dominates strongly")
    if strongest[0] is not None and abs(strongest[1]) >= 0.5:
        if strongest[0] in {"corr_error_time_index", "corr_error_cumulative_Ah"}:
            verdicts.append(f"progress/capacity shortcut risk: {strongest[0]}={strongest[1]:.3f}")
        elif strongest[0] in {"corr_error_V_pol_raw", "corr_error_V_hys_raw", "corr_error_R0"}:
            verdicts.append(f"component dynamics risk: {strongest[0]}={strongest[1]:.3f}")
        else:
            verdicts.append(f"voltage-linked correlation: {strongest[0]}={strongest[1]:.3f}")
    elif strongest[0] is not None:
        verdicts.append(f"no very strong correlation; largest is {strongest[0]}={strongest[1]:.3f}")
    return summary, "; ".join(verdicts)


def temp10_error_diagnostic(feature_df, pred_df, cfg: CFG, ablation_for_error=None, make_plots=True):
    empty_detail_cols = [
        "ablation", "trajectory_id", "end_index", "temperature", "drive_cycle",
        "y_true", "y_pred", "error", "abs_error",
        "V_raw", "V_corr_raw", "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0",
        "cumulative_discharge_Ah",
    ]
    empty_summary_cols = [
        "ablation", "scope", "n_windows", "mean_error", "median_error",
        "error_std", "MAE", "MAE_pct",
    ]
    pred_df = _normalize_prediction_columns(pred_df)
    if pred_df.empty:
        warnings.warn("No prediction rows available. temp10_error_diagnostic.csv will be empty until the notebook is rerun.")
        empty = pd.DataFrame(columns=empty_detail_cols)
        empty.to_csv(cfg.output_dir / "temp10_error_diagnostic.csv", index=False)
        pd.DataFrame(columns=empty_summary_cols).to_csv(cfg.output_dir / "temp10_error_summary.csv", index=False)
        return empty, empty
    physical = pred_df[pred_df["target_label"] == "physical"].copy()
    if physical.empty:
        warnings.warn("No physical-target prediction rows available for 10C error diagnostic.")
        empty = pd.DataFrame(columns=empty_detail_cols)
        empty.to_csv(cfg.output_dir / "temp10_error_diagnostic.csv", index=False)
        pd.DataFrame(columns=empty_summary_cols).to_csv(cfg.output_dir / "temp10_error_summary.csv", index=False)
        return empty, empty
    preferences = [
        ablation_for_error,
        "R5_raw_I_T_all_components",
        "F2_raw_plus_full_decomposed",
        "F1_full_decomposed",
        "A3_V_raw_I_T",
    ]
    ablation = next((a for a in preferences if a and a in set(physical["ablation"])), None)
    if ablation is None:
        ablation = sorted(physical["ablation"].dropna().unique())[0]
    pred = physical[physical["ablation"] == ablation].copy()
    keep = [
        "trajectory_id", "end_index", "V_raw", "V_corr_raw", "V_pol_raw", "V_hys_raw",
        "V_ohm_raw", "R0", "cumulative_discharge_Ah", "SOC_physical", "SOC_usable_cutoff",
    ]
    feat = feature_df[keep].copy()
    existing_feature_cols = [c for c in keep if c in pred.columns and c not in {"trajectory_id", "end_index"}]
    if existing_feature_cols:
        pred = pred.drop(columns=existing_feature_cols)
    detail = pred.merge(feat, on=["trajectory_id", "end_index"], how="left", validate="many_to_one")
    detail = detail[np.isclose(detail["temperature"].astype(float), 10.0)].copy()
    if detail.empty:
        warnings.warn("No 10C prediction rows available for selected ablation.")
        detail.to_csv(cfg.output_dir / "temp10_error_diagnostic.csv", index=False)
        empty_summary = pd.DataFrame(columns=empty_summary_cols)
        empty_summary.to_csv(cfg.output_dir / "temp10_error_summary.csv", index=False)
        return detail, empty_summary
    detail["trajectory_fraction"] = detail.groupby("trajectory_id")["end_index"].transform(
        lambda s: s / max(float(s.max()), 1.0)
    )
    detail["phase"] = pd.cut(
        detail["trajectory_fraction"],
        bins=[-1e-9, 1 / 3, 2 / 3, 1.000001],
        labels=["early", "mid", "late"],
        include_lowest=True,
    )
    detail["model_name"] = ablation
    detail["label_type"] = "physical"
    detail["temperature_C"] = detail["temperature"]
    detail["time_index"] = detail["end_index"]
    detail["cumulative_Ah"] = detail["cumulative_discharge_Ah"]
    detail["is_plateau_20_80"] = (detail["y_true"] >= 0.2) & (detail["y_true"] <= 0.8)
    detail["is_cutoff_last10"] = detail["trajectory_fraction"] >= 0.9
    detail["SOC_bin"] = np.select(
        [detail["y_true"] < 0.2, detail["y_true"] <= 0.8],
        ["0-20", "20-80"],
        default="80-100",
    )
    detail.to_csv(cfg.output_dir / "temp10_error_diagnostic.csv", index=False)

    summary, verdict = classify_temp10_error(detail, ablation)
    summary.to_csv(cfg.output_dir / "temp10_error_summary.csv", index=False)
    classification_text = (
        "# 10C Error Classification\n\n"
        f"- model: {ablation}\n"
        f"- verdict: {verdict}\n"
        "- caution: TEMP_HEAD gains should be interpreted as mapping/calibration changes, not component dynamics fixes.\n"
        "- caution: do not claim 10C polarization-like magnitude explosion unless component diagnostics support it.\n"
    )
    (cfg.output_dir / "temp10_error_classification.txt").write_text(classification_text)

    if make_plots:
        n_traj = detail["trajectory_id"].nunique()
        fig, axes = plt.subplots(n_traj, 2, figsize=(11, max(3.2, 2.6 * n_traj)), squeeze=False)
        for row_idx, (tid, g) in enumerate(detail.sort_values("end_index").groupby("trajectory_id")):
            ax = axes[row_idx, 0]
            ax.plot(g["end_index"], g["y_true"] * 100.0, label="true physical SOC")
            ax.plot(g["end_index"], g["y_pred"] * 100.0, label="predicted physical SOC")
            ax.set_title(f"10C physical SOC trajectory | {tid} | {ablation}")
            ax.set_ylabel("SOC (%SOC)")
            ax.legend()
            ax = axes[row_idx, 1]
            ax.plot(g["end_index"], g["error"] * 100.0)
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_title("prediction error over time")
            ax.set_ylabel("error (%SOC)")
        for ax in axes[-1, :]:
            ax.set_xlabel("time index")
        plt.tight_layout()
        plt.savefig(cfg.output_dir / "temp10_error_over_time.png", dpi=180)
        plt.close()

        xcols = ["V_raw", "V_corr_raw", "V_pol_raw", "V_hys_raw", "R0", "cumulative_Ah", "end_index"]
        fig, axes = plt.subplots(2, 4, figsize=(13, 6))
        axes = axes.reshape(-1)
        for ax, col in zip(axes, xcols):
            ax.scatter(detail[col], detail["error"] * 100.0, s=8, alpha=0.45)
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_xlabel(col)
            ax.set_ylabel("error (%SOC)")
        axes[-1].axis("off")
        plt.suptitle(f"10C error vs learned voltage features | {ablation}")
        plt.tight_layout()
        plt.savefig(cfg.output_dir / "temp10_error_vs_components.png", dpi=180)
        plt.close()
    return detail, summary


def run_temp10_diagnostics(artifacts=None, cfg=None, ablation_for_error=None, make_plots=True):
    feature_df, cfg = _feature_df(artifacts=artifacts, cfg=cfg, splits=("test",))
    if feature_df.empty:
        raise ValueError("No test decomposed feature rows found. Run the pipeline or check cfg.decomposed_dir.")
    distance = component_temperature_distance(feature_df, cfg, make_plots=make_plots)
    event_response = event_aligned_component_response(feature_df, cfg, make_plots=make_plots)
    pred_df = _prediction_df_from_artifacts(artifacts=artifacts, cfg=cfg)
    temp10_detail, temp10_summary = temp10_error_diagnostic(
        feature_df,
        pred_df,
        cfg,
        ablation_for_error=ablation_for_error,
        make_plots=make_plots,
    )
    print("10C component separability distances:")
    display(distance.head(40))
    print("Event-aligned component response by temperature:")
    display(event_response)
    if len(temp10_summary):
        print("10C error decomposition summary:")
        display(temp10_summary)
        mean_error = float(temp10_summary.loc[temp10_summary["scope"] == "all_10C", "mean_error"].iloc[0])
        error_std = float(temp10_summary.loc[temp10_summary["scope"] == "all_10C", "error_std"].iloc[0])
        if abs(mean_error) > 0.5 * max(error_std, 1e-12):
            print("Interpretation hint: 10C error has a notable signed bias component; check calibration/Q_ref/temperature mapping.")
        else:
            print("Interpretation hint: 10C error is not dominated by mean bias; inspect trajectory shape and component dynamics.")
    close_pairs = distance[
        (distance["metric"] == "mahalanobis")
        & (distance["feature"] == "joint")
        & (distance["temperature_b"].isin([0.0, 20.0, 25.0]))
    ]
    if len(close_pairs):
        median_close = float(close_pairs["value"].median())
        far = distance[
            (distance["metric"] == "mahalanobis")
            & (distance["feature"] == "joint")
            & (distance["temperature_b"].isin([-10.0, 50.0]))
        ]
        median_far = float(far["value"].median()) if len(far) else float("nan")
        if np.isfinite(median_far) and median_close < median_far:
            print(
                "Interpretation hint: 10C is closer to 0/20/25C than to endpoint temperatures in learned feature space; "
                "10C degradation may be caused by insufficient feature separability rather than component magnitude explosion."
            )
    return {
        "component_temperature_distance": distance,
        "event_aligned_component_response": event_response,
        "temp10_error_detail": temp10_detail,
        "temp10_error_summary": temp10_summary,
    }
