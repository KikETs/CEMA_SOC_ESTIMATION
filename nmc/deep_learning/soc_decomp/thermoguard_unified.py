from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pandas as pd

from .smoothq_retrain import EXPERIMENTS


KEYS = ["experiment", "trajectory_id", "end_index"]
TARGET_EXPERIMENTS = ("Exp A", "Exp B", "Exp C", "Omit N10", "Omit 50")


def _train_temps_c(experiment: str) -> list[float]:
    vals = []
    for t in EXPERIMENTS[experiment]["train_temps"]:
        s = str(t)
        vals.append(-float(s[1:]) if s.upper().startswith("N") else float(s))
    return vals


def _target_temp_c(experiment: str) -> float:
    return float(EXPERIMENTS[experiment]["omitted_temp_C"])


def _read_model_rows(path: Path, model_name: str, extra_cols=()) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    base_cols = [
        "model_name",
        "label_type",
        "trajectory_id",
        "temperature_C",
        "drive_cycle",
        "time_index",
        "end_index",
        "y_true",
        "y_pred",
        "error",
        "abs_error",
        "voltage_residual",
        "T_core_est",
        "T_core_minus_amb",
        "heat_proxy",
        "Q_eff",
        "R0_x",
        "R0",
        "trajectory_fraction",
        "is_plateau_20_80",
        "is_cutoff_last10",
        "experiment",
    ]
    usecols = [c for c in base_cols + list(extra_cols) if c in header.columns]
    df = pd.read_csv(path, usecols=usecols)
    df = df[df["model_name"].eq(model_name)].copy()
    if "voltage_residual" not in df.columns:
        df["voltage_residual"] = np.nan
    if "trajectory_fraction" not in df.columns:
        df["trajectory_fraction"] = np.nan
    if "experiment" in df.columns:
        df = df[df["experiment"].isin(TARGET_EXPERIMENTS)].copy()
    return df


def _prefix_expert(df: pd.DataFrame, prefix: str, keep_meta=False) -> pd.DataFrame:
    cols = KEYS + ["y_pred", "voltage_residual"]
    optional = ["T_core_est", "T_core_minus_amb", "heat_proxy", "Q_eff", "R0_x", "R0"]
    cols += [c for c in optional if c in df.columns]
    if keep_meta:
        meta = [
            "temperature_C",
            "drive_cycle",
            "time_index",
            "y_true",
            "label_type",
            "trajectory_fraction",
            "is_plateau_20_80",
            "is_cutoff_last10",
        ]
        cols += [c for c in meta if c in df.columns]
    out = df[cols].copy()
    rename = {
        "y_pred": f"soc_{prefix}",
        "voltage_residual": f"voltage_residual_{prefix}",
        "Q_eff": f"Q_eff_{prefix}",
        "R0_x": f"R0_{prefix}",
        "R0": f"R0_{prefix}",
    }
    for c in optional:
        if c in out.columns and c not in rename:
            rename[c] = f"{c}_{prefix}"
    return out.rename(columns=rename)


def _recent_jitter(df: pd.DataFrame, pred_col: str, out_col: str, window=25) -> pd.Series:
    result = pd.Series(index=df.index, dtype=float)
    for _, idx in df.sort_values("end_index").groupby("trajectory_id").groups.items():
        g = df.loc[idx].sort_values("end_index")
        d = g[pred_col].astype(float).diff().abs()
        local = d.rolling(window=window, min_periods=3).mean().bfill().fillna(d.mean())
        result.loc[g.index] = local.to_numpy(float)
    return result.fillna(result.median() if result.notna().any() else 0.0)


def _zscore_from_median(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    if not np.isfinite(x.to_numpy(float)).any():
        return pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med))) + 1e-9
    return (x - med) / (1.4826 * mad)


def _jitter_ratio(g: pd.DataFrame, pred_col="y_pred", true_col="y_true") -> float:
    vals = []
    for _, t in g.sort_values("end_index").groupby("trajectory_id"):
        yp = t[pred_col].to_numpy(float)
        yt = t[true_col].to_numpy(float)
        if len(yp) < 3:
            continue
        vals.append(np.mean(np.abs(np.diff(yp))) / (np.mean(np.abs(np.diff(yt))) + 1e-12))
    return float(np.mean(vals)) if vals else np.nan


def _hf_error_energy(g: pd.DataFrame, pred_col="y_pred", true_col="y_true") -> float:
    vals = []
    for _, t in g.sort_values("end_index").groupby("trajectory_id"):
        e = (t[pred_col].to_numpy(float) - t[true_col].to_numpy(float))
        if len(e) < 3:
            continue
        vals.append(np.mean(np.diff(e) ** 2))
    return float(np.mean(vals)) if vals else np.nan


def _metrics(g: pd.DataFrame, pred_col="y_pred", true_col="y_true") -> dict:
    err = g[pred_col].to_numpy(float) - g[true_col].to_numpy(float)
    abs_err = np.abs(err)
    return {
        "MAE_pct": float(np.mean(abs_err) * 100.0),
        "RMSE_pct": float(np.sqrt(np.mean(err ** 2)) * 100.0),
        "jitter_ratio": _jitter_ratio(g, pred_col, true_col),
        "high_frequency_error_energy": _hf_error_energy(g, pred_col, true_col),
        "catastrophic_error_rate_gt5pct": float(np.mean(abs_err > 0.05)),
        "n_samples": int(len(g)),
    }


def _focus_metrics(pred: pd.DataFrame, model_col="model_name") -> pd.DataFrame:
    rows = []
    for (experiment, model), g in pred.groupby(["experiment", model_col]):
        if experiment not in TARGET_EXPERIMENTS:
            continue
        temp = _target_temp_c(experiment)
        f = g[np.isclose(g["temperature_C"].astype(float), temp)]
        if f.empty:
            continue
        row = {
            "experiment": experiment,
            "target_temperature_C": temp,
            "target_type": "outside" if experiment in {"Omit N10", "Omit 50"} else "omitted",
            "model_name": model,
        }
        row.update(_metrics(f))
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_focus(focus: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, df in [
        ("omitted_A_B_C", focus[focus["experiment"].isin(["Exp A", "Exp B", "Exp C"])]),
        ("outside_range", focus[focus["experiment"].isin(["Omit N10", "Omit 50"])]),
    ]:
        for model, g in df.groupby("model_name"):
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
            for exp, col in [("Omit N10", "outside_minus10_MAE_pct"), ("Omit 50", "outside_50_MAE_pct")]:
                sub = g[g["experiment"].eq(exp)]
                row[col] = float(sub["MAE_pct"].iloc[0]) if len(sub) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class ThermoGuardConfig:
    high_temp_threshold_C: float = 45.0
    soft_gate_temperature_C: float = 3.0
    r5_max_weight: float = 0.20
    risk_sharpness: float = 1.0
    robust_prior: float = 1.05
    high_temp_prior: float = 3.0
    non_high_temp_suppression: float = 0.02


class ThermoGuardSOC:
    """Artifact-level mixture of frozen SOC experts.

    The class combines frozen expert prediction rows.  It does not train or
    update the expert dynamics, and the default gates are label-free.
    """

    def __init__(self, config: ThermoGuardConfig | None = None):
        self.config = config or ThermoGuardConfig()

    def add_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for name in ["perf", "robust", "highT", "r5"]:
            out[f"jitter_{name}"] = _recent_jitter(out, f"soc_{name}", f"jitter_{name}")
            out[f"residual_abs_{name}"] = np.abs(out.get(f"voltage_residual_{name}", np.nan))
        out["prediction_disagreement"] = np.nanmean(
            np.abs(np.vstack([
                out["soc_perf"].to_numpy(float) - out["soc_robust"].to_numpy(float),
                out["soc_perf"].to_numpy(float) - out["soc_highT"].to_numpy(float),
                out["soc_perf"].to_numpy(float) - out["soc_r5"].to_numpy(float),
            ])),
            axis=0,
        )
        outside_high, outside_low = [], []
        for exp, temp in zip(out["experiment"], out["temperature_C"]):
            train_t = _train_temps_c(str(exp))
            outside_high.append(float(float(temp) > max(train_t)))
            outside_low.append(float(float(temp) < min(train_t)))
        out["outside_high_temp_flag"] = outside_high
        out["outside_low_temp_flag"] = outside_low
        return out

    def rule_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        high = (df["outside_high_temp_flag"].astype(float) > 0.5) | (
            df["temperature_C"].astype(float) >= self.config.high_temp_threshold_C
        )
        low = df["outside_low_temp_flag"].astype(float) > 0.5
        out["w_perf"] = np.where(high, 0.0, np.where(low, 0.35, 0.65))
        out["w_robust"] = np.where(high, 0.0, np.where(low, 0.65, 0.35))
        out["w_highT"] = np.where(high, 1.0, 0.0)
        out["w_r5"] = 0.0
        return out

    def rule_soft_high_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.rule_weights(df)
        high = (df["outside_high_temp_flag"].astype(float) > 0.5) | (
            df["temperature_C"].astype(float) >= self.config.high_temp_threshold_C
        )
        temp = df["temperature_C"].astype(float)
        soft_high = 1.0 / (1.0 + np.exp(-(temp - self.config.high_temp_threshold_C) / self.config.soft_gate_temperature_C))
        w_high = np.where(high, np.maximum(soft_high, 0.80 * df["outside_high_temp_flag"].astype(float)), out["w_highT"])
        out.loc[high, "w_highT"] = w_high[high]
        out.loc[high, "w_perf"] = 1.0 - out.loc[high, "w_highT"]
        out.loc[high, "w_robust"] = 0.0
        out.loc[high, "w_r5"] = 0.0
        return out

    def proxy_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        risks = {}
        for name in ["perf", "robust", "highT", "r5"]:
            terms = []
            if f"residual_abs_{name}" in df:
                terms.append(_zscore_from_median(df[f"residual_abs_{name}"]).clip(-2, 6))
            terms.append(_zscore_from_median(df[f"jitter_{name}"]).clip(-2, 6))
            if name == "r5":
                terms.append(_zscore_from_median((df["soc_r5"] - df["soc_perf"]).abs()).clip(-2, 6))
                terms.append(_zscore_from_median((df["soc_r5"] - df["soc_robust"]).abs()).clip(-2, 6))
            risks[name] = np.nanmean(np.vstack([t.to_numpy(float) for t in terms]), axis=0)

        logits = {
            "perf": -cfg.risk_sharpness * risks["perf"],
            "robust": math.log(cfg.robust_prior) - cfg.risk_sharpness * risks["robust"],
            "highT": math.log(cfg.high_temp_prior) - cfg.risk_sharpness * risks["highT"],
            "r5": -cfg.risk_sharpness * risks["r5"],
        }
        temp = df["temperature_C"].astype(float)
        soft_high = 1.0 / (1.0 + np.exp(-(temp - cfg.high_temp_threshold_C) / cfg.soft_gate_temperature_C))
        outside_high = df["outside_high_temp_flag"].astype(float)
        high_prior = np.maximum(soft_high, outside_high)
        low_or_normal = 1.0 - high_prior
        multipliers = {
            "perf": 0.25 + 0.75 * low_or_normal,
            "robust": 0.35 + 0.65 * low_or_normal,
            "highT": cfg.non_high_temp_suppression + (1.0 - cfg.non_high_temp_suppression) * high_prior,
            "r5": 0.2 * low_or_normal,
        }
        raw = {}
        for name in ["perf", "robust", "highT", "r5"]:
            raw[name] = np.exp(np.clip(logits[name], -8, 8)) * np.asarray(multipliers[name], dtype=float)
        total = raw["perf"] + raw["robust"] + raw["highT"] + raw["r5"] + 1e-12
        w = pd.DataFrame({f"w_{k}": raw[k] / total for k in raw}, index=df.index)
        w["w_r5"] = np.minimum(w["w_r5"], cfg.r5_max_weight)
        total2 = w[["w_perf", "w_robust", "w_highT", "w_r5"]].sum(axis=1) + 1e-12
        for c in ["w_perf", "w_robust", "w_highT", "w_r5"]:
            w[c] = w[c] / total2
        return w

    def apply(self, df: pd.DataFrame, variant: str) -> pd.DataFrame:
        if variant == "ThermoGuardSOC_rule_gate":
            w = self.rule_weights(df)
        elif variant == "ThermoGuardSOC_rule_soft_high":
            w = self.rule_soft_high_weights(df)
        elif variant == "ThermoGuardSOC_proxy_gate":
            w = self.proxy_weights(df)
        elif variant == "ThermoGuardSOC_proxy_no_r5":
            w = self.proxy_weights(df)
            w["w_r5"] = 0.0
            s = w[["w_perf", "w_robust", "w_highT", "w_r5"]].sum(axis=1) + 1e-12
            for c in ["w_perf", "w_robust", "w_highT", "w_r5"]:
                w[c] = w[c] / s
        else:
            raise ValueError(variant)
        out = df.copy()
        for c in w.columns:
            out[c] = w[c]
        out["y_pred"] = (
            out["w_perf"] * out["soc_perf"]
            + out["w_robust"] * out["soc_robust"]
            + out["w_highT"] * out["soc_highT"]
            + out["w_r5"] * out["soc_r5"]
        )
        out["error"] = out["y_pred"] - out["y_true"]
        out["abs_error"] = np.abs(out["error"])
        out["model_name"] = variant
        out["uncertainty_score"] = (
            out["prediction_disagreement"]
            + out[["jitter_perf", "jitter_robust", "jitter_highT", "jitter_r5"]].mean(axis=1)
        )
        out["risk_flag"] = pd.cut(
            out["uncertainty_score"],
            bins=[-np.inf, out["uncertainty_score"].quantile(0.70), out["uncertainty_score"].quantile(0.90), np.inf],
            labels=["low", "medium", "high"],
            duplicates="drop",
        ).astype(str)
        return out


def load_expert_table(base_dir: Path | str = ".") -> pd.DataFrame:
    base_dir = Path(base_dir)
    perf = _read_model_rows(base_dir / "thermal_state_ablation_prediction_rows.csv", "NeuralECM_Tamb_Tcore")
    robust = _read_model_rows(base_dir / "thermal_state_prediction_rows.csv", "NeuralECM_THERMAL_REX")
    high = _read_model_rows(base_dir / "parameter_surface_prediction_rows.csv", "ParamSurfaceECM_THERMAL_REX")
    r5 = _read_model_rows(base_dir / "smoothQ_retrain_prediction_rows.csv", "R5_GATED_AUG_REX_l1p0_smoothQ")

    base = _prefix_expert(perf, "perf", keep_meta=True)
    joined = base.merge(_prefix_expert(robust, "robust"), on=KEYS, how="inner")
    joined = joined.merge(_prefix_expert(high, "highT"), on=KEYS, how="inner")
    joined = joined.merge(_prefix_expert(r5, "r5"), on=KEYS, how="inner")
    if joined.empty:
        raise RuntimeError("No common prediction rows across experts.")
    return ThermoGuardSOC().add_risk_features(joined)


def _baseline_predictions(expert: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    rows = []
    mapping = {
        "NeuralECM_Tamb_Tcore": "soc_perf",
        "NeuralECM_THERMAL_REX": "soc_robust",
        "ParamSurfaceECM_THERMAL_REX": "soc_highT",
        "R5_GATED_AUG_REX_l1p0_smoothQ": "soc_r5",
    }
    for name, col in mapping.items():
        g = expert[[
            "experiment",
            "trajectory_id",
            "end_index",
            "temperature_C",
            "drive_cycle",
            "time_index",
            "y_true",
            col,
            "trajectory_fraction",
            "is_plateau_20_80",
            "is_cutoff_last10",
        ]].copy()
        g = g.rename(columns={col: "y_pred"})
        g["model_name"] = name
        rows.append(g)
    prev_path = base_dir / "tamb_tcore_expert_guard_prediction_rows.csv"
    if prev_path.exists():
        prev = pd.read_csv(prev_path)
        prev = prev[prev["model_name"].eq("TambTcoreExpertGuard_soft_amb_gate")].copy()
        keep = [
            "experiment",
            "trajectory_id",
            "end_index",
            "temperature_C",
            "drive_cycle",
            "time_index",
            "y_true",
            "y_pred",
            "trajectory_fraction",
        ]
        prev = prev[[c for c in keep if c in prev.columns] + ["model_name"]]
        rows.append(prev)
    return pd.concat(rows, ignore_index=True)


def _by_temperature(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, experiment, temp), g in pred.groupby(["model_name", "experiment", "temperature_C"]):
        row = {"model_name": model, "experiment": experiment, "temperature_C": float(temp)}
        row.update(_metrics(g))
        rows.append(row)
    return pd.DataFrame(rows)


def _expert_weights(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    weight_cols = ["w_perf", "w_robust", "w_highT", "w_r5"]
    for (model, experiment, temp), g in pred.groupby(["model_name", "experiment", "temperature_C"]):
        row = {
            "model_name": model,
            "experiment": experiment,
            "temperature_C": float(temp),
            "n_samples": int(len(g)),
            "selected_perf_frac": float((g[weight_cols].idxmax(axis=1) == "w_perf").mean()),
            "selected_robust_frac": float((g[weight_cols].idxmax(axis=1) == "w_robust").mean()),
            "selected_highT_frac": float((g[weight_cols].idxmax(axis=1) == "w_highT").mean()),
            "selected_r5_frac": float((g[weight_cols].idxmax(axis=1) == "w_r5").mean()),
        }
        for c in weight_cols:
            row[f"{c}_mean"] = float(g[c].mean())
            row[f"{c}_std"] = float(g[c].std(ddof=0))
            row[f"{c}_max"] = float(g[c].max())
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_weight_summary(weights: pd.DataFrame, base_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    for group_col, out_name in [("temperature_C", "thermoguard_weight_by_temperature.png"), ("experiment", "thermoguard_weight_by_fold.png")]:
        models = [
            "ThermoGuardSOC_rule_gate",
            "ThermoGuardSOC_rule_soft_high",
            "ThermoGuardSOC_proxy_gate",
            "ThermoGuardSOC_proxy_no_r5",
        ]
        fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4), sharey=True)
        if len(models) == 1:
            axes = [axes]
        for ax, model in zip(axes, models):
            g = weights[weights["model_name"].eq(model)]
            if g.empty:
                continue
            x = g[group_col].astype(str) if group_col == "experiment" else g[group_col]
            ax.plot(x, g["w_perf_mean"], marker="o", label="perf")
            ax.plot(x, g["w_robust_mean"], marker="o", label="robust")
            ax.plot(x, g["w_highT_mean"], marker="o", label="highT")
            ax.plot(x, g["w_r5_mean"], marker="o", label="r5")
            ax.set_title(model)
            ax.set_xlabel(group_col)
            ax.tick_params(axis="x", rotation=45)
        axes[0].set_ylabel("mean expert weight")
        axes[-1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(base_dir / out_name, dpi=150)
        plt.close(fig)


def _plot_pareto(results: pd.DataFrame, base_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    focus = results[results["scope"].eq("omitted_A_B_C")].copy()
    if focus.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in focus.iterrows():
        ax.scatter(r["average_MAE_pct"], r["average_jitter_ratio"], s=50)
        ax.text(r["average_MAE_pct"], r["average_jitter_ratio"], r["model_name"], fontsize=7)
    ax.set_xlabel("average omitted MAE (%)")
    ax.set_ylabel("average jitter ratio")
    fig.tight_layout()
    fig.savefig(base_dir / "thermoguard_pareto_mae_jitter.png", dpi=150)
    plt.close(fig)


def gate_sensitivity(expert: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in [40.0, 45.0, 50.0]:
        for soft_temp in [2.0, 3.0, 5.0]:
            for r5_max in [0.0, 0.1, 0.2]:
                cfg = ThermoGuardConfig(
                    high_temp_threshold_C=threshold,
                    soft_gate_temperature_C=soft_temp,
                    r5_max_weight=r5_max,
                )
                model = ThermoGuardSOC(cfg)
                pred = model.apply(expert, "ThermoGuardSOC_proxy_gate")
                focus = _focus_metrics(pred)
                agg = _aggregate_focus(focus)
                for _, r in agg.iterrows():
                    rows.append({
                        "high_temp_threshold_C": threshold,
                        "soft_gate_temperature_C": soft_temp,
                        "r5_max_weight": r5_max,
                        **r.to_dict(),
                    })
    return pd.DataFrame(rows)


def run_thermoguard(base_dir: Path | str = ".") -> dict[str, pd.DataFrame]:
    base_dir = Path(base_dir)
    expert = load_expert_table(base_dir)
    guard = ThermoGuardSOC()
    guard_rows = [guard.apply(expert, v) for v in [
        "ThermoGuardSOC_rule_gate",
        "ThermoGuardSOC_rule_soft_high",
        "ThermoGuardSOC_proxy_gate",
        "ThermoGuardSOC_proxy_no_r5",
    ]]
    guard_pred = pd.concat(guard_rows, ignore_index=True)
    guard_pred.to_csv(base_dir / "thermoguard_prediction_rows.csv", index=False)

    baselines = _baseline_predictions(expert, base_dir)
    all_pred = pd.concat([baselines, guard_pred], ignore_index=True, sort=False)
    focus = _focus_metrics(all_pred)
    focus.to_csv(base_dir / "thermoguard_focus.csv", index=False)
    by_temp = _by_temperature(all_pred)
    by_temp.to_csv(base_dir / "thermoguard_by_temperature.csv", index=False)
    results = _aggregate_focus(focus)
    results.to_csv(base_dir / "thermoguard_results.csv", index=False)
    weights = _expert_weights(guard_pred)
    weights.to_csv(base_dir / "thermoguard_expert_weights.csv", index=False)
    sens = gate_sensitivity(expert)
    sens.to_csv(base_dir / "thermoguard_gate_sensitivity.csv", index=False)
    _plot_weight_summary(weights, base_dir)
    _plot_pareto(results, base_dir)
    write_report(base_dir, results, focus, weights, sens)
    return {
        "results": results,
        "focus": focus,
        "by_temperature": by_temp,
        "weights": weights,
        "sensitivity": sens,
    }


def write_report(base_dir: Path, results: pd.DataFrame, focus: pd.DataFrame, weights: pd.DataFrame, sens: pd.DataFrame):
    guard_rows = results[results["model_name"].str.startswith("ThermoGuardSOC", na=False)]
    best_outside = guard_rows[guard_rows["scope"].eq("outside_range")].sort_values("average_MAE_pct").head(1)
    lines = [
        "# ThermoGuard-SOC Report",
        "",
        "ThermoGuard-SOC combines frozen expert predictions with label-free thermal/risk gates.",
        "No final test SOC labels are used for gate fitting. The supervised learned gate is skipped because a clean, separate LOTO validation artifact distinct from the final evaluation rows is not available.",
        "",
        "## Aggregate Results",
        results.to_markdown(index=False) if len(results) else "No results.",
        "",
        "## Focus Metrics",
        focus.to_markdown(index=False) if len(focus) else "No focus rows.",
        "",
        "## Expert Weight Sanity",
        "- High-temperature outside rows should assign high weight to the ParamSurface high-temperature expert.",
        "- Normal and low-temperature rows should mostly use NeuralECM_Tamb_Tcore and/or NeuralECM_THERMAL_REX.",
        "- R5 is capped and suppressed in high-risk/outside conditions.",
        "",
        weights.to_markdown(index=False) if len(weights) else "No weight rows.",
        "",
    ]
    if len(best_outside):
        r = best_outside.iloc[0]
        lines.extend([
            "## Best Outside-Range Guard",
            f"`{r['model_name']}` has outside average MAE {r['average_MAE_pct']:.3f}% and worst outside MAE {r['worst_MAE_pct']:.3f}%.",
            "",
        ])
    lines.extend([
        "## Gate Sensitivity",
        "The sensitivity table varies high-temperature threshold, soft gate temperature, and R5 max weight. It is diagnostic only and should not be tuned from final test SOC error.",
        "",
        "## Interpretation",
        "- The unified model combines complementary estimators through label-free thermal/risk gating.",
        "- ParamSurface is useful as a high-temperature expert, not as a generalist.",
        "- R5 remains an auxiliary/risk signal and is not safe under outside-range conditions without guard.",
        "- Estimated/effective thermal state should not be described as measured core temperature.",
        "",
        "## Forbidden Claims",
        "- Pure unseen-temperature extrapolation is solved.",
        "- The gate guarantees robustness for all temperatures.",
        "- ParamSurface is universally superior.",
        "- `T_core_est` is measured internal temperature.",
        "- R5 is safe under outside-range conditions without guard.",
        "",
        "## Leakage Check",
        "- Expert dynamics are frozen prediction artifacts.",
        "- Rule/proxy gates use temperature, residual, jitter, disagreement, and train-temperature range flags only.",
        "- Final test SOC labels are used only for evaluation metrics.",
        "- No oracle or post-hoc best-expert selection is used in the reported ThermoGuardSOC rule/proxy rows.",
    ])
    (base_dir / "thermoguard_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ThermoGuard-SOC unified expert gates.")
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()
    out = run_thermoguard(Path(args.base_dir))
    print(out["results"].to_string(index=False))


if __name__ == "__main__":
    main()
