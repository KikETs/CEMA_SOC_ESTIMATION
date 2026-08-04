from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .nmc_branchbands_experiment import build_feature_frames, estimate_r0_by_temperature, find_csv_files, table_md, write_start_audit
from .nmc_vcorr_it_lstm_singlehead_bytemp import FEATURE_COLS
from .nmc_vit_feature_lstm_experiment import add_vit_engineered_features, write_input_schema, write_leakage_audit


BASE_PREFIX = "nmc_vcorr_it_data_model_analysis"


@dataclass
class DataModelAnalysisConfig:
    base_dir: Path = Path(".")
    raw_root: Path = Path("nmc_soc_ocvstart_relabelled_from_lc_ocv/data/NMC SAMSUNG INR 18650 2Ah")
    output_prefix: str = BASE_PREFIX
    seed: int = 0
    train_profiles: tuple[str, ...] = ("DST", "US06")
    valid_profiles: tuple[str, ...] = ("VALIDATION",)
    test_profiles: tuple[str, ...] = ("FUDS",)
    v_corr_tau_s: float = 120.0
    v_pol_mid_tau_s: float = 60.0
    v_pol_slow_tau_s: float = 600.0
    v_hys_tau_s: float = 1200.0
    rolling_window: int = 50
    max_model_rows_per_split: int = 240000


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def concat_frames(frames: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    parts = []
    for split, flist in frames.items():
        for df in flist:
            tmp = df.copy()
            tmp["split"] = split
            parts.append(tmp)
    out = pd.concat(parts, ignore_index=True)
    if "temperature_C" not in out.columns:
        out["temperature_C"] = out["temperature"].astype(float)
    return out


def add_causal_window_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = []
    for (_split, traj), g in df.groupby(["split", "trajectory_id"], sort=False):
        g = g.sort_values("end_index").copy()
        v = g["V_corr_raw"]
        i = g["I_raw"]
        di = g["dI"] if "dI" in g.columns else g["I_raw"].diff().fillna(0.0)
        g["V_corr_span_w50"] = v.rolling(window, min_periods=2).max() - v.rolling(window, min_periods=2).min()
        g["V_corr_delta_w50"] = v - v.shift(window - 1)
        g["absI_mean_w50"] = i.abs().rolling(window, min_periods=2).mean()
        g["I_std_w50"] = i.rolling(window, min_periods=2).std(ddof=0)
        g["dI_abs_mean_w50"] = di.abs().rolling(window, min_periods=2).mean()
        g["dI_energy_w50"] = (di.astype(float) ** 2).rolling(window, min_periods=2).mean()
        g["low_current_frac_w50"] = (i.abs() < 0.05).astype(float).rolling(window, min_periods=2).mean()
        out.append(g)
    out = pd.concat(out, ignore_index=True)
    for col in [
        "V_corr_span_w50",
        "V_corr_delta_w50",
        "absI_mean_w50",
        "I_std_w50",
        "dI_abs_mean_w50",
        "dI_energy_w50",
        "low_current_frac_w50",
    ]:
        out[col] = out[col].fillna(0.0)
    return out


def soc_bin(soc: pd.Series) -> pd.Series:
    bins = pd.cut(
        soc,
        bins=[-np.inf, 0.2, 0.5, 0.8, np.inf],
        labels=["low_<20", "midlow_20_50", "plateau_50_80", "high_>80"],
    )
    return bins.astype(str)


def group_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["soc_bin"] = soc_bin(df["SOC_physical"])
    rows = []
    group_cols = ["split", "temperature_C", "drive_cycle"]
    for key, g in df.groupby(group_cols):
        rows.append(
            {
                "split": key[0],
                "temperature_C": float(key[1]),
                "drive_cycle": key[2],
                "n_rows": int(len(g)),
                "n_trajectories": int(g["trajectory_id"].nunique()),
                "soc_min": float(g["SOC_physical"].min()),
                "soc_max": float(g["SOC_physical"].max()),
                "soc_mean": float(g["SOC_physical"].mean()),
                "V_corr_min": float(g["V_corr_raw"].min()),
                "V_corr_max": float(g["V_corr_raw"].max()),
                "V_corr_range": float(g["V_corr_raw"].max() - g["V_corr_raw"].min()),
                "absI_mean": float(g["I_raw"].abs().mean()),
                "absI_p95": float(g["I_raw"].abs().quantile(0.95)),
                "dI_abs_mean": float(g["dI"].abs().mean()) if "dI" in g.columns else float("nan"),
                "V_corr_span_w50_mean": float(g["V_corr_span_w50"].mean()),
                "V_corr_span_w50_p90": float(g["V_corr_span_w50"].quantile(0.90)),
                "low_current_frac_w50_mean": float(g["low_current_frac_w50"].mean()),
            }
        )
    return pd.DataFrame(rows)


def sample_train_rows(df: pd.DataFrame, split: str, max_rows: int, seed: int) -> pd.DataFrame:
    sub = df[df["split"].eq(split)].copy()
    if len(sub) > max_rows:
        sub = sub.sample(max_rows, random_state=int(seed))
    return sub


def eval_predictions(df: pd.DataFrame, pred: np.ndarray, model_name: str) -> pd.DataFrame:
    tmp = df[["split", "temperature_C", "drive_cycle", "SOC_physical"]].copy()
    tmp["model_name"] = model_name
    tmp["y_pred"] = np.clip(pred, 0.0, 1.0)
    tmp["error"] = tmp["y_pred"] - tmp["SOC_physical"]
    tmp["abs_error"] = tmp["error"].abs()
    rows = []
    for key, g in tmp.groupby(["model_name", "split", "temperature_C", "drive_cycle"]):
        err = g["error"].to_numpy()
        rows.append(
            {
                "model_name": key[0],
                "split": key[1],
                "temperature_C": float(key[2]),
                "drive_cycle": key[3],
                "n_rows": int(len(g)),
                "MAE_pct": float(np.mean(np.abs(err)) * 100.0),
                "RMSE_pct": float(np.sqrt(np.mean(err**2)) * 100.0),
                "bias_pct": float(np.mean(err) * 100.0),
            }
        )
    return pd.DataFrame(rows), tmp


def fit_static_and_dynamic_models(df: pd.DataFrame, cfg: DataModelAnalysisConfig):
    train = sample_train_rows(df, "train", cfg.max_model_rows_per_split, cfg.seed)
    eval_df = pd.concat(
        [
            sample_train_rows(df, "train", cfg.max_model_rows_per_split, cfg.seed + 1),
            df[df["split"].eq("valid")],
            df[df["split"].eq("test")],
        ],
        ignore_index=True,
    )
    static_cols = ["V_corr_raw", "temperature_C"]
    dynamic_cols = static_cols + [
        "I_raw",
        "absI",
        "dI",
        "V_corr_span_w50",
        "V_corr_delta_w50",
        "absI_mean_w50",
        "I_std_w50",
        "dI_abs_mean_w50",
        "dI_energy_w50",
        "low_current_frac_w50",
    ]
    dynamic_cols = [c for c in dynamic_cols if c in df.columns]
    y = train["SOC_physical"].to_numpy()
    static_model = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), StandardScaler(), Ridge(alpha=1e-3))
    static_model.fit(train[static_cols], y)
    dynamic_model = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=1e-4,
        random_state=int(cfg.seed),
    )
    dynamic_model.fit(train[dynamic_cols], y)
    static_metrics, static_pred = eval_predictions(eval_df, static_model.predict(eval_df[static_cols]), "static_Vcorr_T_poly3")
    dynamic_metrics, dynamic_pred = eval_predictions(eval_df, dynamic_model.predict(eval_df[dynamic_cols]), "dynamic_window_features_hgb")
    static_pred = static_pred.rename(columns={"y_pred": "static_pred", "error": "static_error", "abs_error": "static_abs_error"})
    dynamic_pred = dynamic_pred.rename(columns={"y_pred": "dynamic_pred", "error": "dynamic_error", "abs_error": "dynamic_abs_error"})
    pred_cols = list(dict.fromkeys(["split", "temperature_C", "drive_cycle", "trajectory_id", "end_index", "SOC_physical"] + dynamic_cols))
    pred = eval_df[pred_cols].copy()
    pred["static_pred"] = static_pred["static_pred"].to_numpy()
    pred["static_error"] = static_pred["static_error"].to_numpy()
    pred["static_abs_error"] = static_pred["static_abs_error"].to_numpy()
    pred["dynamic_pred"] = dynamic_pred["dynamic_pred"].to_numpy()
    pred["dynamic_error"] = dynamic_pred["dynamic_error"].to_numpy()
    pred["dynamic_abs_error"] = dynamic_pred["dynamic_abs_error"].to_numpy()
    metrics = pd.concat([static_metrics, dynamic_metrics], ignore_index=True)
    return metrics, pred, static_cols, dynamic_cols


def residual_correlations(pred: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for (split, temp, drive), g in pred.groupby(["split", "temperature_C", "drive_cycle"]):
        for target in ["static_error", "dynamic_error"]:
            for col in feature_cols:
                x = g[col].to_numpy(dtype=float)
                y = g[target].to_numpy(dtype=float)
                if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
                    corr = np.nan
                else:
                    corr = float(np.corrcoef(x, y)[0, 1])
                rows.append(
                    {
                        "split": split,
                        "temperature_C": float(temp),
                        "drive_cycle": drive,
                        "residual": target,
                        "feature": col,
                        "pearson_corr": corr,
                        "abs_corr": abs(corr) if np.isfinite(corr) else np.nan,
                    }
                )
    return pd.DataFrame(rows).sort_values(["split", "temperature_C", "drive_cycle", "residual", "abs_corr"], ascending=[True, True, True, True, False])


def distribution_shift(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "V_corr_raw",
        "I_raw",
        "absI",
        "dI",
        "V_corr_span_w50",
        "V_corr_delta_w50",
        "absI_mean_w50",
        "I_std_w50",
        "dI_abs_mean_w50",
        "dI_energy_w50",
        "low_current_frac_w50",
    ]
    features = [c for c in features if c in df.columns]
    train = df[df["split"].eq("train")]
    rows = []
    for (temp, drive), g in df[~df["split"].eq("train")].groupby(["temperature_C", "drive_cycle"]):
        tr = train[np.isclose(train["temperature_C"], temp)]
        for col in features:
            mu = float(tr[col].mean())
            sd = float(tr[col].std(ddof=0)) or 1.0
            rows.append(
                {
                    "temperature_C": float(temp),
                    "eval_drive": drive,
                    "feature": col,
                    "train_temp_mean": mu,
                    "eval_mean": float(g[col].mean()),
                    "standardized_shift": float((g[col].mean() - mu) / sd),
                    "train_temp_p10": float(tr[col].quantile(0.10)),
                    "eval_p10": float(g[col].quantile(0.10)),
                    "train_temp_p90": float(tr[col].quantile(0.90)),
                    "eval_p90": float(g[col].quantile(0.90)),
                }
            )
    return pd.DataFrame(rows).sort_values(["temperature_C", "eval_drive", "standardized_shift"], key=lambda s: s.abs() if s.name == "standardized_shift" else s, ascending=False)


def observability_bins(pred: pd.DataFrame) -> pd.DataFrame:
    p = pred.copy()
    terms = []
    for col in ["V_corr_span_w50", "dI_abs_mean_w50", "I_std_w50"]:
        scale = p[col].quantile(0.90) - p[col].quantile(0.10)
        scale = float(scale) if float(scale) > 1e-12 else 1.0
        terms.append((p[col] - p[col].quantile(0.10)) / scale)
    p["observability_score"] = np.clip(sum(terms) / len(terms), 0.0, 2.0)
    p["observability_bin"] = pd.qcut(p["observability_score"].rank(method="first"), 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    rows = []
    for key, g in p.groupby(["split", "temperature_C", "drive_cycle", "observability_bin"], observed=False):
        rows.append(
            {
                "split": key[0],
                "temperature_C": float(key[1]),
                "drive_cycle": key[2],
                "observability_bin": str(key[3]),
                "n_rows": int(len(g)),
                "observability_mean": float(g["observability_score"].mean()),
                "static_MAE_pct": float(g["static_abs_error"].mean() * 100.0),
                "dynamic_MAE_pct": float(g["dynamic_abs_error"].mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def write_report(out_dir: Path, cfg: DataModelAnalysisConfig, summary, baseline, corr, shift, obs, schema, leakage):
    lines = [
        "# NMC Vcorr/I/T Data-Driven Model Analysis",
        "",
        "## Purpose",
        "This analysis is used before model design. It checks static voltage observability, dynamic-response residuals, and validation/test drive-cycle shift under the strict V_corr_raw/I_raw/T-only condition.",
        "",
        "## Dataset Summary",
        table_md(summary, ["split", "temperature_C", "drive_cycle", "n_rows", "soc_min", "soc_max", "V_corr_range", "absI_mean", "dI_abs_mean", "V_corr_span_w50_mean", "low_current_frac_w50_mean"]),
        "",
        "## Static vs Dynamic Baseline",
        table_md(baseline, ["model_name", "split", "temperature_C", "drive_cycle", "n_rows", "MAE_pct", "RMSE_pct", "bias_pct"]),
        "",
        "## Strongest Residual Correlations",
        table_md(corr.head(80), ["split", "temperature_C", "drive_cycle", "residual", "feature", "pearson_corr", "abs_corr"]),
        "",
        "## Largest Drive-Cycle Shifts",
        table_md(shift.head(80), ["temperature_C", "eval_drive", "feature", "standardized_shift", "train_temp_mean", "eval_mean", "train_temp_p10", "eval_p10", "train_temp_p90", "eval_p90"]),
        "",
        "## Observability Bins",
        table_md(obs, ["split", "temperature_C", "drive_cycle", "observability_bin", "n_rows", "observability_mean", "static_MAE_pct", "dynamic_MAE_pct"]),
        "",
        "## Model Implications",
        "- Use a static V_corr/T anchor because voltage alone carries a large part of SOC information.",
        "- Add a dynamic branch because static residuals correlate with instantaneous current and local voltage-response features.",
        "- Treat VALIDATION validation and FUDS test shift explicitly; validation selection alone can prefer different residual regimes.",
        "- A single checkpoint can still use temperature-conditioned adapters, but the design should be justified by measured temperature/drive-cycle response differences rather than arbitrary expert assignment.",
        "",
        "## Input Schema",
        table_md(schema, ["index_1based", "feature_name", "source"]),
        "",
        "## Leakage Audit",
        table_md(leakage, ["audit_item", "status", "detail"]),
    ]
    (out_dir / f"{cfg.output_prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(cfg: DataModelAnalysisConfig) -> dict[str, pd.DataFrame]:
    cfg.base_dir = Path(cfg.base_dir).resolve()
    if not Path(cfg.raw_root).is_absolute():
        cfg.raw_root = cfg.base_dir / cfg.raw_root
    out_dir = cfg.base_dir / "nmc_vcorr_it_data_model_analysis_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    files = find_csv_files(cfg.raw_root)
    raw_source_columns = list(pd.read_csv(files[0], nrows=1).columns)
    write_start_audit(files, out_dir / f"{cfg.output_prefix}_file_start_audit.csv")
    r0_df = estimate_r0_by_temperature(files, cfg.train_profiles)
    r0_df.to_csv(out_dir / f"{cfg.output_prefix}_decomposition_params.csv", index=False)
    frames = add_vit_engineered_features(build_feature_frames(cfg, files, r0_df))
    schema = write_input_schema(FEATURE_COLS, out_dir / f"{cfg.output_prefix}_input_schema.csv")
    leakage = write_leakage_audit(FEATURE_COLS, raw_source_columns, out_dir / f"{cfg.output_prefix}_leakage_audit.csv")
    df = add_causal_window_features(concat_frames(frames), int(cfg.rolling_window))
    summary = group_summary(df)
    baseline, pred, static_cols, dynamic_cols = fit_static_and_dynamic_models(df, cfg)
    corr = residual_correlations(pred, dynamic_cols)
    shift = distribution_shift(df)
    obs = observability_bins(pred)
    summary.to_csv(out_dir / f"{cfg.output_prefix}_dataset_summary.csv", index=False)
    baseline.to_csv(out_dir / f"{cfg.output_prefix}_baseline_metrics.csv", index=False)
    pred.to_csv(out_dir / f"{cfg.output_prefix}_baseline_predictions.csv.gz", index=False, compression="gzip")
    corr.to_csv(out_dir / f"{cfg.output_prefix}_residual_correlations.csv", index=False)
    shift.to_csv(out_dir / f"{cfg.output_prefix}_drive_cycle_shift.csv", index=False)
    obs.to_csv(out_dir / f"{cfg.output_prefix}_observability_bins.csv", index=False)
    metadata = {
        **asdict(cfg),
        "base_dir": str(cfg.base_dir),
        "raw_root": str(cfg.raw_root),
        "feature_columns": FEATURE_COLS,
        "static_baseline_columns": static_cols,
        "dynamic_baseline_columns": dynamic_cols,
        "uses_soc_input": False,
        "uses_cumulative_input": False,
        "uses_explicit_current_integration": False,
        "uses_absolute_time_or_progress": False,
    }
    (out_dir / f"{cfg.output_prefix}_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    write_report(out_dir, cfg, summary, baseline, corr, shift, obs, schema, leakage)
    print(f"Report: {out_dir / (cfg.output_prefix + '_report.md')}", flush=True)
    print("Baseline metrics:")
    print(baseline.to_string(index=False), flush=True)
    print("Top residual correlations:")
    print(corr.head(30).to_string(index=False), flush=True)
    return {
        "dataset_summary": summary,
        "baseline_metrics": baseline,
        "residual_correlations": corr,
        "drive_cycle_shift": shift,
        "observability_bins": obs,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NMC Vcorr/I/T data-driven model design analysis.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--raw-root", default=DataModelAnalysisConfig.raw_root)
    p.add_argument("--output-prefix", default=BASE_PREFIX)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-model-rows-per-split", type=int, default=240000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DataModelAnalysisConfig(
        base_dir=Path(args.base_dir),
        raw_root=Path(args.raw_root),
        output_prefix=str(args.output_prefix),
        seed=int(args.seed),
        max_model_rows_per_split=int(args.max_model_rows_per_split),
    )
    run(cfg)


if __name__ == "__main__":
    main()
