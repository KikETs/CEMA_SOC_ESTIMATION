from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "absI_mean",
    "I_std",
    "dI_energy",
    "dI_abs_mean",
    "V_corr_span",
    "V_corr_delta_abs",
    "low_current_frac",
    "V_corr_endpoint",
]


@dataclass
class Config:
    valid_predictions: Path = Path("paper_results/best_clean_regime_diagnostics_valid/best_clean_prediction_rows_with_regime.csv.gz")
    test_predictions: Path = Path("paper_results/best_clean_regime_diagnostics/best_clean_prediction_rows_with_regime.csv.gz")
    output_dir: Path = Path("paper_results")
    output_prefix: str = "nmc_strict_nocc_failure_risk"
    ridge_lambda: float = 1.0


def _prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "V_corr_delta" in df.columns:
        df["V_corr_delta_abs"] = pd.to_numeric(df["V_corr_delta"], errors="coerce").abs()
    df["abs_error_pct"] = pd.to_numeric(df["abs_error"], errors="coerce") * 100.0
    df["temperature_C"] = pd.to_numeric(df["temperature_C"], errors="coerce")
    keep = ["file_name", "trajectory_id", "end_index", "temperature_C", "drive_cycle", "soc_bin", "abs_error_pct", *FEATURES]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = df[keep].copy()
    for col in FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["abs_error_pct", "temperature_C", *FEATURES])
    return out


def _design(df: pd.DataFrame, *, means: pd.Series | None = None, stds: pd.Series | None = None) -> tuple[np.ndarray, pd.Series, pd.Series]:
    x = df[FEATURES].astype(float).copy()
    if means is None:
        means = x.mean()
    if stds is None:
        stds = x.std(ddof=0).replace(0.0, 1.0)
    z = (x - means) / stds
    temps = pd.get_dummies(df["temperature_C"].astype(int).astype(str), prefix="T", dtype=float)
    for name in ["T_0", "T_25", "T_45"]:
        if name not in temps.columns:
            temps[name] = 0.0
    mat = np.column_stack([np.ones(len(df)), z.to_numpy(np.float64), temps[["T_0", "T_25", "T_45"]].to_numpy(np.float64)])
    return mat, means, stds


def _fit_ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    penalty = np.eye(x.shape[1], dtype=np.float64) * float(lam)
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def _corr(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    if len(x) < 3:
        return np.nan, np.nan
    return float(x.corr(y, method="pearson")), float(x.corr(y, method="spearman"))


def _risk_deciles(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for temp, g in df.groupby("temperature_C"):
        q = pd.qcut(g["risk_score"], q=10, labels=False, duplicates="drop")
        tmp = g.assign(risk_decile=q)
        for decile, d in tmp.groupby("risk_decile"):
            rows.append(
                {
                    "temperature_C": float(temp),
                    "risk_decile": int(decile),
                    "n_windows": int(len(d)),
                    "risk_min": float(d["risk_score"].min()),
                    "risk_max": float(d["risk_score"].max()),
                    "MAE_pct": float(d["abs_error_pct"].mean()),
                    "RMSE_pct": float(np.sqrt(np.mean(np.square(d["abs_error_pct"])))),
                    "gt_1pct_frac": float(np.mean(d["abs_error_pct"] > 1.0)),
                    "gt_2pct_frac": float(np.mean(d["abs_error_pct"] > 2.0)),
                }
            )
    return pd.DataFrame(rows).sort_values(["temperature_C", "risk_decile"])


def _threshold_transfer(valid: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q in [0.50, 0.70, 0.80, 0.90]:
        thr = float(valid["risk_score"].quantile(q))
        for split, frame in [("valid_VALIDATION", valid), ("test_FUDS", test)]:
            for temp, g in frame.groupby("temperature_C"):
                accepted = g[g["risk_score"] <= thr]
                rejected = g[g["risk_score"] > thr]
                rows.append(
                    {
                        "threshold_source": "valid_risk_quantile",
                        "risk_quantile": float(q),
                        "risk_threshold": thr,
                        "split": split,
                        "temperature_C": float(temp),
                        "coverage_frac": float(len(accepted) / max(len(g), 1)),
                        "accepted_MAE_pct": float(accepted["abs_error_pct"].mean()) if len(accepted) else np.nan,
                        "rejected_MAE_pct": float(rejected["abs_error_pct"].mean()) if len(rejected) else np.nan,
                        "overall_MAE_pct": float(g["abs_error_pct"].mean()),
                        "accepted_gt_1pct_frac": float(np.mean(accepted["abs_error_pct"] > 1.0)) if len(accepted) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def run(cfg: Config) -> dict[str, pd.DataFrame]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    valid = _prepare(cfg.valid_predictions)
    test = _prepare(cfg.test_predictions)

    x_valid, means, stds = _design(valid)
    y_valid = np.log1p(valid["abs_error_pct"].to_numpy(np.float64))
    coef = _fit_ridge(x_valid, y_valid, cfg.ridge_lambda)
    valid["risk_score"] = np.expm1(x_valid @ coef)
    x_test, _means, _stds = _design(test, means=means, stds=stds)
    test["risk_score"] = np.expm1(x_test @ coef)

    summary_rows = []
    for split, frame in [("valid_VALIDATION", valid), ("test_FUDS", test)]:
        for temp, g in frame.groupby("temperature_C"):
            pearson, spearman = _corr(g["risk_score"], g["abs_error_pct"])
            summary_rows.append(
                {
                    "split": split,
                    "temperature_C": float(temp),
                    "n_windows": int(len(g)),
                    "MAE_pct": float(g["abs_error_pct"].mean()),
                    "risk_error_pearson": pearson,
                    "risk_error_spearman": spearman,
                    "top20_risk_MAE_pct": float(g[g["risk_score"] >= g["risk_score"].quantile(0.8)]["abs_error_pct"].mean()),
                    "bottom80_risk_MAE_pct": float(g[g["risk_score"] < g["risk_score"].quantile(0.8)]["abs_error_pct"].mean()),
                }
            )
    summary = pd.DataFrame(summary_rows)
    deciles = _risk_deciles(test)
    transfer = _threshold_transfer(valid, test)

    coef_rows = [{"term": "intercept", "coef": float(coef[0])}]
    for name, value in zip([*FEATURES, "T_0", "T_25", "T_45"], coef[1:]):
        coef_rows.append({"term": name, "coef": float(value)})
    coef_df = pd.DataFrame(coef_rows)

    summary.to_csv(cfg.output_dir / f"{cfg.output_prefix}_summary.csv", index=False)
    deciles.to_csv(cfg.output_dir / f"{cfg.output_prefix}_test_deciles.csv", index=False)
    transfer.to_csv(cfg.output_dir / f"{cfg.output_prefix}_threshold_transfer.csv", index=False)
    coef_df.to_csv(cfg.output_dir / f"{cfg.output_prefix}_ridge_coefficients.csv", index=False)
    _write_report(cfg, summary, deciles, transfer, coef_df)
    return {"summary": summary, "deciles": deciles, "threshold_transfer": transfer, "coefficients": coef_df}


def _write_report(cfg: Config, summary: pd.DataFrame, deciles: pd.DataFrame, transfer: pd.DataFrame, coef_df: pd.DataFrame) -> None:
    test_summary = summary[summary["split"] == "test_FUDS"].copy()
    transfer_test = transfer[(transfer["split"] == "test_FUDS") & (transfer["risk_quantile"].isin([0.8, 0.9]))].copy()
    lines = [
        "# NMC Strict NoCC Failure-Risk Analysis",
        "",
        "## Scope",
        "",
        "- Dataset/scope: NMC strict NoCC only",
        "- Base predictions: best clean NoCC base-only prediction rows",
        "- Risk model training: VALIDATION validation prediction errors only",
        "- Test use: apply the validation-fitted risk score to FUDS",
        "- Inputs to risk score: label-free window response features only",
        "- No SOC input, SOC_CC, cumulative Ah, current integration, LFP, or capacity-anchor result is used.",
        "",
        "## Test Summary",
        "",
        test_summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## FUDS Risk Deciles",
        "",
        deciles.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Validation-Threshold Transfer",
        "",
        transfer_test.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Ridge Coefficients",
        "",
        coef_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "This is not a new SOC model and not a model-selection rule. It tests whether validation-fitted, label-free observability features can identify where the strict NoCC base is likely to fail on FUDS.",
        "",
        "If the risk score separates high-error windows, the next NoCC candidate should expose an uncertainty/observability head and use ProfileLOO validation for adoption. If it does not transfer, the safer conclusion is that VALIDATION validation is not representative enough for FUDS-like regimes.",
    ]
    (cfg.output_dir / f"{cfg.output_prefix}_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-predictions", type=Path, default=Config.valid_predictions)
    parser.add_argument("--test-predictions", type=Path, default=Config.test_predictions)
    parser.add_argument("--output-dir", type=Path, default=Config.output_dir)
    parser.add_argument("--output-prefix", default=Config.output_prefix)
    parser.add_argument("--ridge-lambda", type=float, default=Config.ridge_lambda)
    args = parser.parse_args()
    run(
        Config(
            valid_predictions=args.valid_predictions,
            test_predictions=args.test_predictions,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            ridge_lambda=args.ridge_lambda,
        )
    )


if __name__ == "__main__":
    main()
