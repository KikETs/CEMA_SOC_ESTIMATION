from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from .runtime import configure_torch_runtime, device
from .smoothq_retrain import (
    EXPERIMENTS,
    clone_cfg,
    configure_strict_training,
    experiment_cfg,
    load_relabelled_frames,
    load_smoothq_lookup,
)
from .thermal_state_model import (
    ObserverSpec,
    ThermalECMObserver,
    attach_and_summarize_observer,
    predict_observer_trajectories,
    train_observer_model,
)
from .parameter_surface_model import ParamSurfaceECMObserver


TARGET_EXPERIMENTS = ("Exp A", "Exp B", "Exp C", "Omit N10", "Omit 50")


def _train_temps_c(experiment):
    vals = []
    for t in EXPERIMENTS[experiment]["train_temps"]:
        s = str(t)
        vals.append(-float(s[1:]) if s.upper().startswith("N") else float(s))
    return vals


def _jitter_ratio(g: pd.DataFrame):
    vals = []
    for _, t in g.sort_values("end_index").groupby("trajectory_id"):
        yp = t["y_pred"].to_numpy(float)
        yt = t["y_true"].to_numpy(float)
        if len(yp) < 3:
            continue
        vals.append(np.mean(np.abs(np.diff(yp))) / (np.mean(np.abs(np.diff(yt))) + 1e-12))
    return float(np.mean(vals)) if vals else np.nan


def _focus_metrics(pred: pd.DataFrame, model_col="model_name") -> pd.DataFrame:
    rows = []
    for (experiment, model), g in pred.groupby(["experiment", model_col]):
        if experiment not in TARGET_EXPERIMENTS:
            continue
        temp = float(EXPERIMENTS[experiment]["omitted_temp_C"])
        f = g[np.isclose(g["temperature_C"].astype(float), temp)]
        if f.empty:
            continue
        err = f["error"].to_numpy(float)
        abs_err = np.abs(err)
        vres = f["voltage_residual"].to_numpy(float) if "voltage_residual" in f.columns else np.full(len(f), np.nan)
        rows.append({
            "experiment": experiment,
            "target_temperature_C": temp,
            "target_type": "outside" if experiment in {"Omit N10", "Omit 50"} else "omitted",
            "model_name": model,
            "MAE_pct": float(np.mean(abs_err) * 100.0),
            "RMSE_pct": float(np.sqrt(np.mean(err ** 2)) * 100.0),
            "jitter_ratio": _jitter_ratio(f),
            "voltage_residual_MAE_V": float(np.nanmean(np.abs(vres))) if np.isfinite(vres).any() else np.nan,
            "catastrophic_error_rate_gt5pct": float(np.mean(abs_err > 0.05)),
            "n_samples": int(len(f)),
        })
    return pd.DataFrame(rows)


def _aggregate_focus(focus: pd.DataFrame, model_col="model_name") -> pd.DataFrame:
    rows = []
    for scope, df in [
        ("omitted_A_B_C", focus[focus["experiment"].isin(["Exp A", "Exp B", "Exp C"])]),
        ("outside_range", focus[focus["experiment"].isin(["Omit N10", "Omit 50"])]),
    ]:
        for model, g in df.groupby(model_col):
            row = {
                "model_name": model,
                "scope": scope,
                "average_MAE_pct": float(g["MAE_pct"].mean()),
                "worst_MAE_pct": float(g["MAE_pct"].max()),
                "average_RMSE_pct": float(g["RMSE_pct"].mean()),
                "average_jitter_ratio": float(g["jitter_ratio"].mean()),
                "catastrophic_error_rate_gt5pct": float(g["catastrophic_error_rate_gt5pct"].mean()),
                "n_folds": int(g["experiment"].nunique()),
            }
            for exp, name in [("Omit N10", "outside_minus10_MAE_pct"), ("Omit 50", "outside_50_MAE_pct")]:
                sub = g[g["experiment"].eq(exp)]
                row[name] = float(sub["MAE_pct"].iloc[0]) if len(sub) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _plot_weights(weight_diag: pd.DataFrame, path: Path):
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
    ax.set_ylabel("ParamSurface expert weight")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def tamb_tcore_expert_guard(base_dir: Path | str = "."):
    base_dir = Path(base_dir)
    default = pd.read_csv(base_dir / "thermal_state_ablation_prediction_rows.csv")
    expert = pd.read_csv(base_dir / "parameter_surface_prediction_rows.csv")
    default = default[default["model_name"].eq("NeuralECM_Tamb_Tcore")].copy()
    expert = expert[expert["model_name"].eq("ParamSurfaceECM_THERMAL_REX")].copy()
    keys = ["experiment", "trajectory_id", "end_index"]
    cols = keys + ["temperature_C", "y_true", "y_pred", "error", "abs_error", "voltage_residual"]
    if "T_core_est" in default.columns:
        cols.append("T_core_est")
    base = default[cols].rename(columns={
        "y_pred": "y_pred_default",
        "error": "error_default",
        "abs_error": "abs_error_default",
        "voltage_residual": "voltage_residual_default",
    })
    exp = expert[keys + ["y_pred", "voltage_residual"]].rename(columns={
        "y_pred": "y_pred_expert",
        "voltage_residual": "voltage_residual_expert",
    })
    joined = base.merge(exp, on=keys, how="inner")
    variants = []
    for variant in ["amb_threshold_45", "soft_amb_gate", "tcore_threshold_45", "outside_high_flag"]:
        g = joined.copy()
        if variant == "amb_threshold_45":
            w = (g["temperature_C"].astype(float) >= 45.0).astype(float)
        elif variant == "soft_amb_gate":
            w = 1.0 / (1.0 + np.exp(-(g["temperature_C"].astype(float) - 45.0) / 3.0))
        elif variant == "tcore_threshold_45":
            w = (g.get("T_core_est", g["temperature_C"]).astype(float) >= 45.0).astype(float)
        else:
            vals = []
            for experiment, temp in zip(g["experiment"], g["temperature_C"]):
                vals.append(float(float(temp) > max(_train_temps_c(experiment))))
            w = pd.Series(vals, index=g.index, dtype=float)
        g["expert_weight"] = np.asarray(w, dtype=float)
        g["model_name"] = f"TambTcoreExpertGuard_{variant}"
        g["y_pred"] = g["expert_weight"] * g["y_pred_expert"] + (1.0 - g["expert_weight"]) * g["y_pred_default"]
        g["error"] = g["y_pred"] - g["y_true"]
        g["abs_error"] = np.abs(g["error"])
        g["voltage_residual"] = g["expert_weight"] * g["voltage_residual_expert"] + (1.0 - g["expert_weight"]) * g["voltage_residual_default"]
        variants.append(g)
    pred = pd.concat(variants, ignore_index=True)
    pred.to_csv(base_dir / "tamb_tcore_expert_guard_prediction_rows.csv", index=False)
    focus = _focus_metrics(pred)
    focus.to_csv(base_dir / "tamb_tcore_expert_guard_focus.csv", index=False)
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
    by_temp.to_csv(base_dir / "tamb_tcore_expert_guard_by_temperature.csv", index=False)
    summary = _aggregate_focus(focus)
    summary.to_csv(base_dir / "tamb_tcore_expert_guard_results.csv", index=False)
    weights = pred.groupby(["model_name", "experiment", "temperature_C"]).agg(
        expert_weight_mean=("expert_weight", "mean"),
        expert_weight_max=("expert_weight", "max"),
        n_samples=("expert_weight", "size"),
    ).reset_index()
    weights.to_csv(base_dir / "tamb_tcore_expert_guard_weights.csv", index=False)
    _plot_weights(weights, base_dir / "tamb_tcore_expert_guard_weights.png")
    lines = [
        "# Tamb_Tcore Expert Guard Report",
        "",
        "Default estimator: `NeuralECM_Tamb_Tcore`.",
        "High-temperature expert: `ParamSurfaceECM_THERMAL_REX`.",
        "Gates use ambient/core temperature or outside-high flags only; thresholds are fixed from train/LOTO logic and are not tuned from outside-50 SOC error.",
        "",
        "## Aggregate",
        summary.to_markdown(index=False) if len(summary) else "No summary rows.",
        "",
        "## Fold Focus",
        focus.to_markdown(index=False) if len(focus) else "No focus rows.",
        "",
        "## Interpretation",
        "- Hard high-temperature guards preserve omitted/low-temperature folds and switch outside 50C to the ParamSurface expert.",
        "- The soft ambient gate can outperform either single estimator on outside 50C by blending two imperfect predictions, but it should be confirmed across seeds before being treated as final.",
    ]
    (base_dir / "tamb_tcore_expert_guard_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary, by_temp, focus


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_seed_model(feature_frames, cfg, experiment, model_key: str, seed: int):
    if model_key == "NeuralECM_Tamb_Tcore":
        spec = ObserverSpec(name=f"NeuralECM_Tamb_Tcore_seed{seed}", use_rex=False, lambda_v=0.20, correction_limit=0.05)
        factory = lambda q, dt, s: ThermalECMObserver(q_ref_ah=q, dt_sec=dt, correction_limit=s.correction_limit, thermal_mode="tamb_tcore")
    elif model_key == "NeuralECM_THERMAL_REX":
        spec = ObserverSpec(name=f"NeuralECM_THERMAL_REX_seed{seed}", use_rex=True, lambda_v=0.20, lambda_rex=0.5, lambda_worst=0.10, correction_limit=0.05)
        factory = lambda q, dt, s: ThermalECMObserver(q_ref_ah=q, dt_sec=dt, correction_limit=s.correction_limit, thermal_mode="tamb_tcore_heatproxy")
    elif model_key == "ParamSurfaceECM_THERMAL_REX":
        spec = ObserverSpec(name=f"ParamSurfaceECM_THERMAL_REX_seed{seed}", use_rex=True, lambda_v=0.20, lambda_rex=0.5, lambda_worst=0.10, correction_limit=0.05)
        factory = lambda q, dt, s: ParamSurfaceECMObserver(q_ref_ah=q, dt_sec=dt, correction_limit=s.correction_limit, residual_limit=0.15, use_thermal=True)
    else:
        raise ValueError(model_key)
    model, hist, _ = train_observer_model(feature_frames, cfg, spec, experiment, factory)
    pred = predict_observer_trajectories(model, feature_frames["test"], spec.name)
    return model, hist, pred


def _guard_from_seed_predictions(default_pred: pd.DataFrame, expert_pred: pd.DataFrame, seed: int):
    candidate_keys = ["experiment", "trajectory_id", "end_index"]
    keys = [k for k in candidate_keys if k in default_pred.columns and k in expert_pred.columns]
    if "trajectory_id" not in keys or "end_index" not in keys:
        raise ValueError("Seed guard fusion requires trajectory_id and end_index in both prediction frames.")
    d = default_pred.rename(columns={
        "y_pred": "y_pred_default",
        "voltage_residual": "voltage_residual_default",
    })
    e = expert_pred[keys + ["y_pred", "voltage_residual"]].rename(columns={
        "y_pred": "y_pred_expert",
        "voltage_residual": "voltage_residual_expert",
    })
    g = d.merge(e, on=keys, how="inner")
    w = 1.0 / (1.0 + np.exp(-(g["temperature_C"].astype(float) - 45.0) / 3.0))
    g["expert_weight"] = np.asarray(w, dtype=float)
    g["model_name"] = f"TambTcoreExpertGuard_soft_amb_seed{seed}"
    g["y_pred"] = g["expert_weight"] * g["y_pred_expert"] + (1.0 - g["expert_weight"]) * g["y_pred_default"]
    g["error"] = g["y_pred"] - g["y_true"]
    g["abs_error"] = np.abs(g["error"])
    g["voltage_residual"] = g["expert_weight"] * g["voltage_residual_expert"] + (1.0 - g["expert_weight"]) * g["voltage_residual_default"]
    return g


def run_seed_sensitivity(
    cfg=None,
    *,
    seeds=(0, 1, 2),
    experiments=TARGET_EXPERIMENTS,
    include_param_surface=True,
):
    configure_torch_runtime()
    base_cfg = configure_strict_training(clone_cfg(cfg))
    output_dir = base_cfg.output_dir
    lookup = load_smoothq_lookup(output_dir)
    all_focus, all_history = [], []
    for seed in seeds:
        _set_seed(int(seed))
        for experiment in experiments:
            print(f"=== final seed sensitivity seed={seed} {experiment} ===")
            ecfg = experiment_cfg(base_cfg, experiment)
            configure_strict_training(ecfg)
            feature_frames = load_relabelled_frames(ecfg, experiment, lookup)
            pred_rows = []
            _, hist_default, pred_default = _train_seed_model(feature_frames, ecfg, experiment, "NeuralECM_Tamb_Tcore", int(seed))
            hist_default["seed"] = int(seed)
            all_history.append(hist_default)
            pred_rows.append(pred_default.assign(seed=int(seed), seed_model_family="NeuralECM_Tamb_Tcore"))
            _, hist_rex, pred_rex = _train_seed_model(feature_frames, ecfg, experiment, "NeuralECM_THERMAL_REX", int(seed))
            hist_rex["seed"] = int(seed)
            all_history.append(hist_rex)
            pred_rows.append(pred_rex.assign(seed=int(seed), seed_model_family="NeuralECM_THERMAL_REX"))
            if include_param_surface:
                _, hist_expert, pred_expert = _train_seed_model(feature_frames, ecfg, experiment, "ParamSurfaceECM_THERMAL_REX", int(seed))
                hist_expert["seed"] = int(seed)
                all_history.append(hist_expert)
                pred_rows.append(pred_expert.assign(seed=int(seed), seed_model_family="ParamSurfaceECM_THERMAL_REX"))
                guard_pred = _guard_from_seed_predictions(pred_default, pred_expert, int(seed))
                pred_rows.append(guard_pred.assign(seed=int(seed), seed_model_family="TambTcoreExpertGuard_soft_amb"))
            pred = pd.concat(pred_rows, ignore_index=True)
            pred["experiment"] = experiment
            focus = _focus_metrics(pred)
            focus["seed"] = int(seed)
            all_focus.append(focus)
            pd.concat(all_focus, ignore_index=True).to_csv(output_dir / "final_seed_sensitivity_results.csv", index=False)
            pd.concat(all_history, ignore_index=True).to_csv(output_dir / "final_seed_sensitivity_history.csv", index=False)
    results = pd.concat(all_focus, ignore_index=True) if all_focus else pd.DataFrame()
    summary = _seed_summary(results)
    summary.to_csv(output_dir / "final_seed_sensitivity_summary.csv", index=False)
    lines = [
        "# Final Seed Sensitivity Summary",
        "",
        f"Seeds run: {', '.join(str(s) for s in seeds)}.",
        "",
        summary.to_markdown(index=False) if len(summary) else "No summary rows.",
        "",
        "The high-temp guard uses the fixed soft ambient gate. It does not use target SOC error for threshold selection.",
    ]
    (output_dir / "final_seed_sensitivity_report.md").write_text("\n".join(lines), encoding="utf-8")
    return results, summary


def _seed_summary(results: pd.DataFrame):
    if results.empty:
        return pd.DataFrame()
    normalized = results.copy()
    normalized["model_family"] = normalized["model_name"].astype(str).str.replace(r"_seed\d+$", "", regex=True)
    seed_aggs = []
    for seed, g in normalized.groupby("seed"):
        tmp = g.copy()
        tmp["model_name"] = tmp["model_family"]
        agg_seed = _aggregate_focus(tmp)
        agg_seed["seed"] = int(seed)
        seed_aggs.append(agg_seed)
    agg = pd.concat(seed_aggs, ignore_index=True) if seed_aggs else pd.DataFrame()
    if agg.empty:
        return pd.DataFrame()
    rows = []
    for (model, scope), g in agg.groupby(["model_name", "scope"]):
        rows.append({
            "model_name": model,
            "scope": scope,
            "n_seeds": int(g["seed"].nunique()),
            "average_MAE_pct_mean": float(g["average_MAE_pct"].mean()),
            "average_MAE_pct_std": float(g["average_MAE_pct"].std(ddof=0)),
            "worst_MAE_pct_mean": float(g["worst_MAE_pct"].mean()),
            "worst_MAE_pct_std": float(g["worst_MAE_pct"].std(ddof=0)),
            "average_RMSE_pct_mean": float(g["average_RMSE_pct"].mean()),
            "average_RMSE_pct_std": float(g["average_RMSE_pct"].std(ddof=0)),
            "average_jitter_ratio_mean": float(g["average_jitter_ratio"].mean()),
            "average_jitter_ratio_std": float(g["average_jitter_ratio"].std(ddof=0)),
            "outside_50_MAE_pct_mean": float(g["outside_50_MAE_pct"].mean()),
            "outside_50_MAE_pct_std": float(g["outside_50_MAE_pct"].std(ddof=0)),
        })
    return pd.DataFrame(rows)


def write_final_thermal_interpretation(base_dir: Path | str = "."):
    base_dir = Path(base_dir)
    ab = pd.read_csv(base_dir / "thermal_state_ablation_summary_table.csv") if (base_dir / "thermal_state_ablation_summary_table.csv").exists() else pd.DataFrame()
    diag = pd.read_csv(base_dir / "thermal_state_trajectory_diagnostics.csv") if (base_dir / "thermal_state_trajectory_diagnostics.csv").exists() else pd.DataFrame()
    lines = [
        "# Final Thermal-State Interpretation",
        "",
        "Use the term **estimated/effective thermal state** or **internal thermal-state proxy**.",
        "Do not call `T_core_est` a measured core temperature unless a measured core-temperature sensor is added.",
        "",
    ]
    if len(ab):
        lines.extend(["## Ablation Evidence", ab.to_markdown(index=False), ""])
    if len(diag):
        cycle = diag.groupby(["model_name", "drive_cycle", "temperature_C"]).agg(
            mean_T_core_minus_amb_C=("mean_T_core_minus_amb_C", "mean"),
            max_T_core_minus_amb_C=("max_T_core_minus_amb_C", "mean"),
            MAE_pct=("MAE_pct", "mean"),
        ).reset_index()
        lines.extend(["## T_core By Drive Cycle", cycle.to_markdown(index=False), ""])
    lines.extend([
        "## Interpretation",
        "- `Tamb_only` is weaker than several variants that include `T_core` or heat history, so ambient temperature alone is insufficient in the current diagnostics.",
        "- The learned `T_core_est - T_amb` offset is modest; the benefit should be interpreted as a causal thermal-history/state pathway, not as proof of a large measured core-temperature rise.",
        "- `T_core` variants improve omitted/outside robustness in aggregate, but individual folds still differ.",
    ])
    (base_dir / "final_thermal_state_interpretation.md").write_text("\n".join(lines), encoding="utf-8")


def write_final_model_selection_report(base_dir: Path | str = "."):
    base_dir = Path(base_dir)
    guard = pd.read_csv(base_dir / "tamb_tcore_expert_guard_results.csv") if (base_dir / "tamb_tcore_expert_guard_results.csv").exists() else pd.DataFrame()
    seeds = pd.read_csv(base_dir / "final_seed_sensitivity_summary.csv") if (base_dir / "final_seed_sensitivity_summary.csv").exists() else pd.DataFrame()
    lines = [
        "# Final Model Selection Report",
        "",
        "## Selected Candidates",
        "- Performance-oriented model: `NeuralECM_Tamb_Tcore`.",
        "- Robustness-oriented model: `NeuralECM_THERMAL_REX`.",
        "- High-temperature guarded model: `NeuralECM_Tamb_Tcore` plus `ParamSurfaceECM_THERMAL_REX` expert.",
        "",
        "## Why R5 Is No Longer Main Temp-Robust Candidate",
        "R5 remains useful under representative temperature coverage, but its omitted/outside folds show large jitter and catastrophic failure rates compared with the observer family.",
        "",
        "## Why TTA Is Not Main Path",
        "Voltage-only TTA was not consistently beneficial and sometimes worsened SOC. It should remain a diagnostic/adaptation option, not the pure extrapolation claim.",
        "",
    ]
    if len(guard):
        lines.extend(["## Tamb_Tcore Expert Guard", guard.to_markdown(index=False), ""])
    if len(seeds):
        lines.extend(["## Seed Sensitivity", seeds.to_markdown(index=False), ""])
    else:
        lines.append("Seed sensitivity has not been completed yet.")
    lines.extend([
        "## Remaining Limitation",
        "Pure unseen-temperature extrapolation is not fully solved. The current best path is a thermal-state observer generalist with a targeted high-temperature expert guard.",
        "",
        "## Forbidden Claims",
        "- The thermal model solves all unseen temperatures.",
        "- ParamSurface is universally robust.",
        "- TTA is pure extrapolation.",
        "- `T_core_est` is measured internal temperature.",
    ])
    (base_dir / "final_model_selection_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_reports(base_dir: Path | str = "."):
    base_dir = Path(base_dir)
    tamb_tcore_expert_guard(base_dir)
    write_final_thermal_interpretation(base_dir)
    write_final_model_selection_report(base_dir)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Finalize thermal model selection and seed sensitivity.")
    parser.add_argument("--guard-only", action="store_true")
    parser.add_argument("--reports-only", action="store_true")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--experiments", default="Exp A,Exp B,Exp C,Omit N10,Omit 50")
    parser.add_argument("--no-param-surface-seed", action="store_true")
    args = parser.parse_args()
    if args.guard_only:
        tamb_tcore_expert_guard(Path("."))
        write_final_model_selection_report(Path("."))
        return
    if args.reports_only:
        run_reports(Path("."))
        return
    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    exps = tuple(s.strip() for s in args.experiments.split(",") if s.strip())
    run_reports(Path("."))
    run_seed_sensitivity(seeds=seeds, experiments=exps, include_param_surface=not args.no_param_surface_seed)
    write_final_model_selection_report(Path("."))


if __name__ == "__main__":
    main()
