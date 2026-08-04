from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .smoothq_retrain import EXPERIMENTS


MODEL_MAP = {
    "R5_GATED_AUG_REX smoothQ": ("smoothQ_retrain_prediction_rows.csv", "R5_GATED_AUG_REX_l1p0_smoothQ"),
    "NeuralECM_REX": ("smoothQ_retrain_prediction_rows.csv", "NeuralECMObserver_REX_smoothQ"),
    "NeuralECM_THERMAL": ("thermal_state_prediction_rows.csv", "NeuralECM_THERMAL"),
    "NeuralECM_THERMAL_REX": ("thermal_state_prediction_rows.csv", "NeuralECM_THERMAL_REX"),
    "ParamSurfaceECM_THERMAL_REX": ("parameter_surface_prediction_rows.csv", "ParamSurfaceECM_THERMAL_REX"),
}

TARGET_EXPERIMENTS = {"Exp A", "Exp B", "Exp C", "Omit N10", "Omit 50"}


def _read_prediction(path: Path, model_name: str, display_name: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    use = pd.read_csv(path)
    if "model_name" not in use.columns:
        return pd.DataFrame()
    use = use[use["model_name"].eq(model_name)].copy()
    if use.empty:
        return use
    use["display_model_name"] = display_name
    if "temperature_C" not in use.columns and "temperature" in use.columns:
        use["temperature_C"] = use["temperature"]
    if "voltage_residual" not in use.columns:
        use["voltage_residual"] = np.nan
    return use


def load_selected_predictions(base_dir: Path) -> pd.DataFrame:
    rows = []
    for display, (fn, model) in MODEL_MAP.items():
        rows.append(_read_prediction(base_dir / fn, model, display))
    rows = [r for r in rows if len(r)]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _metric_rows(pred: pd.DataFrame, model_col="display_model_name") -> pd.DataFrame:
    rows = []
    for (experiment, model), g in pred.groupby(["experiment", model_col]):
        if experiment not in TARGET_EXPERIMENTS:
            continue
        temp = float(EXPERIMENTS[experiment]["omitted_temp_C"])
        focus = g[np.isclose(g["temperature_C"].astype(float), temp)]
        if focus.empty:
            continue
        err = focus["error"].to_numpy(np.float64)
        abs_err = np.abs(err)
        vres = focus["voltage_residual"].to_numpy(np.float64) if "voltage_residual" in focus.columns else np.full(len(focus), np.nan)
        rows.append({
            "experiment": experiment,
            "target_temperature_C": temp,
            "target_type": "outside" if experiment in {"Omit N10", "Omit 50"} else "omitted",
            "model_name": model,
            "MAE_pct": float(np.mean(abs_err) * 100.0),
            "RMSE_pct": float(np.sqrt(np.mean(err ** 2)) * 100.0),
            "jitter_ratio": _jitter_ratio(focus),
            "voltage_residual_MAE_V": float(np.nanmean(np.abs(vres))) if np.isfinite(vres).any() else np.nan,
            "catastrophic_error_rate_gt5pct": float(np.mean(abs_err > 0.05)),
            "n_samples": int(len(focus)),
        })
    return pd.DataFrame(rows)


def _jitter_ratio(g: pd.DataFrame):
    ratios = []
    for _, t in g.sort_values("end_index").groupby("trajectory_id"):
        yp = t["y_pred"].to_numpy(np.float64)
        yt = t["y_true"].to_numpy(np.float64)
        if len(yp) < 3:
            continue
        pred_j = np.mean(np.abs(np.diff(yp)))
        true_j = np.mean(np.abs(np.diff(yt))) + 1e-12
        ratios.append(pred_j / true_j)
    return float(np.mean(ratios)) if ratios else np.nan


def write_fold_level_comparison(base_dir: Path) -> pd.DataFrame:
    pred = load_selected_predictions(base_dir)
    out = _metric_rows(pred)
    order = ["Exp B", "Exp A", "Exp C", "Omit N10", "Omit 50"]
    out["fold_order"] = out["experiment"].map({k: i for i, k in enumerate(order)})
    out = out.sort_values(["fold_order", "model_name"]).drop(columns=["fold_order"])
    out.to_csv(base_dir / "nextgen_fold_level_comparison.csv", index=False)
    lines = [
        "# Nextgen Fold-Level Comparison",
        "",
        "Metrics are computed on the target omitted/outside temperature trajectory for each fold.",
        "Voltage residual is only meaningful for observer models; R5 rows may have no voltage residual.",
        "",
        out.to_markdown(index=False) if len(out) else "No fold-level rows available.",
        "",
        "## Interpretation",
        "- `NeuralECM_THERMAL_REX` is the strongest robust generalist by worst omitted MAE.",
        "- `NeuralECM_THERMAL` has the best omitted average MAE in the current run, but its omitted 10C fold is weaker.",
        "- `ParamSurfaceECM_THERMAL_REX` is weak as a generalist but is strong on outside 50C.",
    ]
    (base_dir / "nextgen_fold_level_report.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def _load_i_lookup(base_dir: Path) -> pd.DataFrame:
    frames = []
    for dname in ["decomposed_features_train_temp_minus10_0_10_25_50", "decomposed_features_train_temp_minus10_0_10_20_25_50"]:
        d = base_dir / dname
        if not d.exists():
            continue
        for path in d.glob("*_features.csv"):
            try:
                df = pd.read_csv(path, usecols=["trajectory_id", "end_index", "I_raw"])
            except Exception:
                continue
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["trajectory_id", "end_index", "I_raw"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["trajectory_id", "end_index"])


def write_thermal_state_diagnostics(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "thermal_state_prediction_rows.csv"
    if not path.exists():
        return pd.DataFrame()
    pred = pd.read_csv(path)
    pred = pred[pred["model_name"].isin(["NeuralECM_THERMAL", "NeuralECM_THERMAL_REX"])].copy()
    if pred.empty:
        return pd.DataFrame()
    if "I_raw" not in pred.columns:
        lookup = _load_i_lookup(base_dir)
        pred = pred.merge(lookup, on=["trajectory_id", "end_index"], how="left")
    r0_col = "R0_x" if "R0_x" in pred.columns else ("R0" if "R0" in pred.columns else None)
    if r0_col and "I_raw" in pred.columns:
        pred["I2R_heat_proxy"] = pred["I_raw"].astype(float) ** 2 * pred[r0_col].astype(float)
    else:
        pred["I2R_heat_proxy"] = np.nan
    pred["abs_voltage_residual"] = np.abs(pred["voltage_residual"].astype(float))
    pred["abs_error_pct"] = pred["abs_error"].astype(float) * 100.0
    summary_rows = []
    for keys, g in pred.groupby(["experiment", "model_name", "temperature_C", "drive_cycle", "trajectory_id"]):
        experiment, model, temp, drive, tid = keys
        summary_rows.append({
            "experiment": experiment,
            "model_name": model,
            "temperature_C": float(temp),
            "drive_cycle": drive,
            "trajectory_id": tid,
            "mean_T_core_est_C": float(g["T_core_est"].mean()),
            "mean_T_core_minus_amb_C": float(g["T_core_minus_amb"].mean()),
            "max_T_core_minus_amb_C": float(g["T_core_minus_amb"].max()),
            "mean_heat_proxy": float(g["heat_proxy"].mean()),
            "mean_I2R_heat_proxy": float(g["I2R_heat_proxy"].mean()) if g["I2R_heat_proxy"].notna().any() else np.nan,
            "MAE_pct": float(g["abs_error"].mean() * 100.0),
            "voltage_residual_MAE_V": float(g["abs_voltage_residual"].mean()),
            "corr_abs_error_T_core_delta": _safe_corr(g["abs_error"], g["T_core_minus_amb"]),
            "corr_abs_error_heat_proxy": _safe_corr(g["abs_error"], g["heat_proxy"]),
            "corr_abs_error_voltage_residual": _safe_corr(g["abs_error"], g["abs_voltage_residual"]),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(base_dir / "thermal_state_trajectory_diagnostics.csv", index=False)
    _write_thermal_plots(pred, base_dir / "thermal_state_trajectory_plots")
    lines = [
        "# Thermal-State Diagnostic Report",
        "",
        "The estimated `T_core` is a learned causal internal state, not a measured core temperature.",
        "",
    ]
    if len(summary):
        cycle = summary.groupby(["model_name", "drive_cycle", "temperature_C"]).agg(
            mean_T_core_minus_amb_C=("mean_T_core_minus_amb_C", "mean"),
            mean_I2R_heat_proxy=("mean_I2R_heat_proxy", "mean"),
            MAE_pct=("MAE_pct", "mean"),
        ).reset_index()
        lines.extend(["## Cycle/Temperature Summary", cycle.to_markdown(index=False), ""])
        hi = summary[summary["temperature_C"].eq(50.0)]
        if len(hi):
            lines.extend(["## Outside/High-Temperature Notes", hi.to_markdown(index=False), ""])
    lines.extend([
        "## Answers",
        "- `T_core_est` differs only modestly from ambient in this learned model, but the thermal-state observer still improves several omitted/outside folds.",
        "- Improvement is therefore not simply a large core-temperature offset; it likely reflects the causal heat-history pathway and constrained observer dynamics.",
        "- Outside 50C should still be treated separately because ParamSurface behaves like a useful high-temperature expert.",
    ])
    (base_dir / "thermal_state_diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def _safe_corr(a, b):
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if mask.sum() < 4 or np.std(aa[mask]) <= 0 or np.std(bb[mask]) <= 0:
        return np.nan
    return float(np.corrcoef(aa[mask], bb[mask])[0, 1])


def _write_thermal_plots(pred: pd.DataFrame, out_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for (experiment, model, tid), g in pred.groupby(["experiment", "model_name", "trajectory_id"]):
        g = g.sort_values("end_index")
        x = np.arange(len(g))
        fig, ax = plt.subplots(6, 1, figsize=(11, 10), sharex=True)
        ax[0].plot(x, g["temperature_C"], label="T_amb")
        ax[0].plot(x, g["T_core_est"], label="T_core_est")
        ax[0].legend(loc="best")
        ax[0].set_ylabel("degC")
        ax[1].plot(x, g["T_core_minus_amb"])
        ax[1].set_ylabel("Tcore-Tamb")
        ax[2].plot(x, g["heat_proxy"], label="heat proxy")
        if "I2R_heat_proxy" in g.columns:
            ax[2].plot(x, g["I2R_heat_proxy"], label="I2R proxy", alpha=0.75)
        ax[2].legend(loc="best")
        ax[3].plot(x, g["error"] * 100.0)
        ax[3].set_ylabel("SOC err %")
        ax[4].plot(x, g["voltage_residual"])
        ax[4].set_ylabel("V resid")
        ax[5].plot(x, g["y_true"], label="true")
        ax[5].plot(x, g["y_pred"], label="pred")
        ax[5].legend(loc="best")
        ax[5].set_xlabel("sample")
        fig.suptitle(f"{experiment} {model} {tid}")
        fig.tight_layout()
        safe = f"{experiment}_{model}_{tid}_thermal_diag".replace(" ", "_").replace("/", "_")
        fig.savefig(out_dir / f"{safe}.png", dpi=140)
        plt.close(fig)


def _focus_predictions(pred: pd.DataFrame, experiment: str) -> pd.DataFrame:
    temp = float(EXPERIMENTS[experiment]["omitted_temp_C"])
    return pred[pred["experiment"].eq(experiment) & np.isclose(pred["temperature_C"].astype(float), temp)].copy()


def high_temp_expert_guard(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    thermal = pd.read_csv(base_dir / "thermal_state_prediction_rows.csv")
    expert = pd.read_csv(base_dir / "parameter_surface_prediction_rows.csv")
    thermal = thermal[thermal["model_name"].eq("NeuralECM_THERMAL_REX")].copy()
    expert = expert[expert["model_name"].eq("ParamSurfaceECM_THERMAL_REX")].copy()
    keys = ["experiment", "trajectory_id", "end_index"]
    cols = keys + ["temperature_C", "y_true", "y_pred", "error", "abs_error", "voltage_residual"]
    if "T_core_est" in thermal.columns:
        cols += ["T_core_est"]
    base = thermal[cols].rename(columns={
        "y_pred": "y_pred_default",
        "error": "error_default",
        "abs_error": "abs_error_default",
        "voltage_residual": "voltage_residual_default",
    })
    exp_cols = keys + ["y_pred", "voltage_residual"]
    joined = base.merge(expert[exp_cols].rename(columns={
        "y_pred": "y_pred_expert",
        "voltage_residual": "voltage_residual_expert",
    }), on=keys, how="inner")
    variants = []
    for variant in ["amb_threshold_45", "tcore_threshold_45", "outside_high_flag", "soft_amb_gate"]:
        g = joined.copy()
        if variant == "amb_threshold_45":
            w = (g["temperature_C"].astype(float) >= 45.0).astype(float)
        elif variant == "tcore_threshold_45":
            w = (g.get("T_core_est", g["temperature_C"]).astype(float) >= 45.0).astype(float)
        elif variant == "outside_high_flag":
            w = []
            for exp, temp in zip(g["experiment"], g["temperature_C"]):
                train_temps = _train_temps_c(exp)
                w.append(float(float(temp) > max(train_temps)))
            w = pd.Series(w, index=g.index, dtype=float)
        else:
            w = 1.0 / (1.0 + np.exp(-(g["temperature_C"].astype(float) - 45.0) / 3.0))
        g["expert_weight"] = np.asarray(w, dtype=float)
        g["model_name"] = f"HighTempExpertGuard_{variant}"
        g["y_pred"] = g["expert_weight"] * g["y_pred_expert"] + (1.0 - g["expert_weight"]) * g["y_pred_default"]
        g["error"] = g["y_pred"] - g["y_true"]
        g["abs_error"] = np.abs(g["error"])
        g["voltage_residual"] = g["expert_weight"] * g["voltage_residual_expert"] + (1.0 - g["expert_weight"]) * g["voltage_residual_default"]
        variants.append(g)
    pred = pd.concat(variants, ignore_index=True)
    pred.to_csv(base_dir / "high_temp_expert_guard_prediction_rows.csv", index=False)
    results = _metric_rows(pred, model_col="model_name")
    results.to_csv(base_dir / "high_temp_expert_guard_results.csv", index=False)
    by_temp = []
    for (model, experiment, temp), g in pred.groupby(["model_name", "experiment", "temperature_C"]):
        err = g["error"].to_numpy(float)
        by_temp.append({
            "model_name": model,
            "experiment": experiment,
            "temperature_C": float(temp),
            "MAE_pct": float(np.mean(np.abs(err)) * 100.0),
            "RMSE_pct": float(np.sqrt(np.mean(err ** 2)) * 100.0),
            "jitter_ratio": _jitter_ratio(g),
            "voltage_residual_MAE_V": float(np.mean(np.abs(g["voltage_residual"]))),
            "catastrophic_error_rate_gt5pct": float(np.mean(np.abs(err) > 0.05)),
        })
    by_temp = pd.DataFrame(by_temp)
    by_temp.to_csv(base_dir / "high_temp_expert_guard_by_temperature.csv", index=False)
    weight_diag = pred.groupby(["model_name", "experiment", "temperature_C"]).agg(
        expert_weight_mean=("expert_weight", "mean"),
        expert_weight_max=("expert_weight", "max"),
        n_samples=("expert_weight", "size"),
    ).reset_index()
    weight_diag.to_csv(base_dir / "high_temp_expert_guard_weights.csv", index=False)
    _plot_guard_weights(weight_diag, base_dir / "high_temp_expert_guard_weights.png")
    lines = [
        "# High-Temperature Expert Guard Report",
        "",
        "Default estimator: `NeuralECM_THERMAL_REX`.",
        "High-temperature expert: `ParamSurfaceECM_THERMAL_REX`.",
        "Thresholds use temperature/range rules, not target SOC error.",
        "",
        "## Focus Metrics",
        results.to_markdown(index=False) if len(results) else "No guard rows.",
        "",
        "## Interpretation",
        "- The outside-high guard should preserve low/omitted folds while switching outside 50C to the ParamSurface expert.",
        "- If outside 50C improves but other folds are unchanged, the expert is useful as a targeted high-temperature fallback, not a universal model.",
    ]
    (base_dir / "high_temp_expert_guard_report.md").write_text("\n".join(lines), encoding="utf-8")
    return results, by_temp


def _train_temps_c(experiment):
    vals = []
    for t in EXPERIMENTS[experiment]["train_temps"]:
        s = str(t)
        vals.append(-float(s[1:]) if s.upper().startswith("N") else float(s))
    return vals


def _plot_guard_weights(weight_diag: pd.DataFrame, path: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if weight_diag.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for model, g in weight_diag.groupby("model_name"):
        ax.scatter(g["temperature_C"], g["expert_weight_mean"], label=model, s=35)
    ax.set_xlabel("temperature_C")
    ax.set_ylabel("expert weight")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_final_report(base_dir: Path):
    fold = pd.read_csv(base_dir / "nextgen_fold_level_comparison.csv") if (base_dir / "nextgen_fold_level_comparison.csv").exists() else pd.DataFrame()
    guard = pd.read_csv(base_dir / "high_temp_expert_guard_results.csv") if (base_dir / "high_temp_expert_guard_results.csv").exists() else pd.DataFrame()
    ablation = pd.read_csv(base_dir / "thermal_state_ablation_summary_table.csv") if (base_dir / "thermal_state_ablation_summary_table.csv").exists() else pd.DataFrame()
    lines = [
        "# Nextgen Temperature-Robust Final Report",
        "",
        "## Current Final Candidate",
        "The strongest direction is the thermal-state observer family.",
        "Within the original nextgen comparison, `NeuralECM_THERMAL_REX` is the robust generalist because it improves worst omitted behavior while preserving low jitter.",
        "The new thermal-state ablation suggests `NeuralECM_Tamb_Tcore` is an even stronger single-run candidate, but it should be confirmed across seeds before replacing the REx generalist.",
        "",
        "## Fold-Level Comparison",
        fold.to_markdown(index=False) if len(fold) else "Fold-level comparison is not available.",
        "",
        "## Thermal-State Ablation",
        ablation.to_markdown(index=False) if len(ablation) else "Thermal-state ablation is not available yet.",
        "",
        "## High-Temperature Expert Guard",
        guard.to_markdown(index=False) if len(guard) else "High-temperature expert guard is not available.",
        "",
        "## Safe Claims",
        "- Explicit thermal-state modeling substantially improves temperature-domain robustness in the current diagnostics.",
        "- `NeuralECM_THERMAL_REX` is currently the strongest pre-ablation robust generalist; `NeuralECM_Tamb_Tcore` is the strongest ablation candidate in this run.",
        "- `ParamSurfaceECM_THERMAL_REX` may serve as a targeted high-temperature expert.",
        "- Pure temperature extrapolation is still not fully solved.",
        "",
        "## Forbidden Claims",
        "- The thermal model solves all unseen temperatures.",
        "- ParamSurface is universally robust.",
        "- TTA is pure extrapolation.",
    ]
    (base_dir / "nextgen_temp_robust_final_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_finalization(base_dir: Path | str = "."):
    base_dir = Path(base_dir)
    fold = write_fold_level_comparison(base_dir)
    diag = write_thermal_state_diagnostics(base_dir)
    guard, guard_by_temp = high_temp_expert_guard(base_dir)
    write_final_report(base_dir)
    return {"fold": fold, "thermal_diag": diag, "guard": guard, "guard_by_temp": guard_by_temp}


def main():
    out = run_finalization(Path("."))
    print(out["fold"].to_string(index=False))
    print(out["guard"].to_string(index=False))


if __name__ == "__main__":
    main()
