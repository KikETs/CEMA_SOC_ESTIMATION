from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .nmc_validation_selected_capacity_gate import _load, _predict_q


@dataclass
class VoltageResidualOODConfig:
    raw_root: Path = Path("nmc_samsung_inr_18650_2ah_raw/NMC SAMSUNG INR 18650 2Ah")
    output_dir: Path = Path("paper_results")
    output_prefix: str = "nmc_voltage_residual_ood_audit"
    feature_name: str = "v_range"
    prefix: int = 256
    shrink: float = 0.75
    margin: float = 0.5
    gate: str = "none"
    ridge_lambda: float = 1e-4


KEYS = ["test_profile", "valid_profile", "train_profiles"]
TARGETS = {0.0: 1.0, 25.0: 0.7, 45.0: 0.3}
PREFIX_WINDOWS = [256, 512, 1024]


def _soc_twophase(rec: dict[str, np.ndarray | float], q_base: float, q_adapt: float, prefix: int) -> np.ndarray:
    y = rec["y"]
    ah = rec["ah"]
    assert isinstance(y, np.ndarray) and isinstance(ah, np.ndarray)
    sw = min(max(int(prefix), 1), len(y))
    pred = np.empty_like(y)
    pred[:sw] = np.clip(float(y[0]) - ah[:sw] / max(float(q_base), 1e-9), 0.0, 1.0)
    if sw < len(y):
        pred[sw:] = np.clip(pred[sw - 1] - (ah[sw:] - ah[sw - 1]) / max(float(q_adapt), 1e-9), 0.0, 1.0)
    return pred


def _features(soc: np.ndarray, i_dis: np.ndarray) -> np.ndarray:
    d_i = np.diff(i_dis, prepend=i_dis[0])
    abs_i = np.abs(i_dis)
    return np.column_stack(
        [
            np.ones_like(soc),
            soc,
            soc**2,
            i_dis,
            abs_i,
            d_i,
            soc * i_dis,
            soc * abs_i,
        ]
    ).astype(np.float64)


def _fit_voltage_model(
    data: dict[tuple[str, float], dict[str, np.ndarray | float]],
    train_profiles: list[str],
    temp: float,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    xs = []
    ys = []
    for profile in train_profiles:
        rec = data[(profile, temp)]
        soc = rec["y"]
        i_dis = rec["i_dis"]
        v = rec["v"]
        assert isinstance(soc, np.ndarray) and isinstance(i_dis, np.ndarray) and isinstance(v, np.ndarray)
        xs.append(_features(soc.astype(np.float64), i_dis.astype(np.float64)))
        ys.append(v.astype(np.float64))
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma < 1e-9] = 1.0
    mu[0] = 0.0
    sigma[0] = 1.0
    z = (x - mu) / sigma
    reg = float(ridge_lambda) * np.eye(z.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    beta = np.linalg.solve(z.T @ z + reg, z.T @ y)
    pred = z @ beta
    abs_resid_mv = np.abs(pred - y) * 1000.0
    stats = {
        "train_resid_mean_mV": float(np.mean(abs_resid_mv)),
        "train_resid_p90_mV": float(np.percentile(abs_resid_mv, 90)),
        "train_resid_p95_mV": float(np.percentile(abs_resid_mv, 95)),
        "train_resid_std_mV": float(np.std(abs_resid_mv)),
    }
    return beta, mu, sigma, stats


def _voltage_residual_stats(
    rec: dict[str, np.ndarray | float],
    soc_pred: np.ndarray,
    model: tuple[np.ndarray, np.ndarray, np.ndarray],
    train_stats: dict[str, float],
    prefix_len: int,
) -> dict[str, float]:
    beta, mu, sigma = model
    i_dis = rec["i_dis"]
    v = rec["v"]
    assert isinstance(i_dis, np.ndarray) and isinstance(v, np.ndarray)
    n = min(max(int(prefix_len), 2), len(v))
    x = _features(soc_pred[:n].astype(np.float64), i_dis[:n].astype(np.float64))
    pred = ((x - mu) / sigma) @ beta
    abs_resid_mv = np.abs(pred - v[:n].astype(np.float64)) * 1000.0
    train_mean = max(float(train_stats["train_resid_mean_mV"]), 1e-9)
    train_p95 = max(float(train_stats["train_resid_p95_mV"]), 1e-9)
    return {
        "resid_mean_mV": float(np.mean(abs_resid_mv)),
        "resid_p95_mV": float(np.percentile(abs_resid_mv, 95)),
        "resid_mean_over_train_mean": float(np.mean(abs_resid_mv) / train_mean),
        "resid_p95_over_train_p95": float(np.percentile(abs_resid_mv, 95) / train_p95),
    }


def _fixed_capacity_metrics(
    data: dict[tuple[str, float], dict[str, np.ndarray | float]],
    train_profiles: list[str],
    test_profile: str,
    cfg: VoltageResidualOODConfig,
) -> tuple[dict[str, float | bool | str], dict[float, np.ndarray]]:
    rows: dict[str, float | bool | str] = {}
    soc_preds: dict[float, np.ndarray] = {}
    failures = []
    worst = 0.0
    for temp in sorted({temp for _profile, temp in data}):
        q_base, q_adapt, _x, _confidence = _predict_q(
            data,
            train_profiles=train_profiles,
            target_profile=test_profile,
            temp=temp,
            feature_name=str(cfg.feature_name),
            prefix=int(cfg.prefix),
            shrink=float(cfg.shrink),
            margin=float(cfg.margin),
            gate=str(cfg.gate),
        )
        rec = data[(test_profile, temp)]
        pred = _soc_twophase(rec, q_base, q_adapt, int(cfg.prefix))
        soc_preds[temp] = pred
        y = rec["y"]
        assert isinstance(y, np.ndarray)
        mae = float(np.mean(np.abs(pred - y)) * 100.0)
        rows[f"{temp:g}C_Qbase_Ah"] = q_base
        rows[f"{temp:g}C_Qadapt_Ah"] = q_adapt
        rows[f"{temp:g}C_MAE_pct"] = mae
        if temp in TARGETS:
            worst = max(worst, mae / TARGETS[temp])
            if mae >= TARGETS[temp]:
                failures.append(f"{temp:g}C {mae:.3f}>={TARGETS[temp]:.3f}")
    rows["target_met"] = not failures
    rows["target_norm_worst"] = float(worst)
    rows["failure_detail"] = "; ".join(failures) if failures else "pass"
    return rows, soc_preds


def _rotation_rows(cfg: VoltageResidualOODConfig) -> pd.DataFrame:
    data = _load(cfg.raw_root)
    profiles = sorted({profile for profile, _temp in data})
    temps = sorted({temp for _profile, temp in data})
    rows = []
    for test_profile in profiles:
        remaining = [p for p in profiles if p != test_profile]
        for valid_profile in remaining:
            train_profiles = [p for p in remaining if p != valid_profile]
            cap_metrics, soc_preds = _fixed_capacity_metrics(data, train_profiles, test_profile, cfg)
            row: dict[str, float | bool | str] = {
                "test_profile": test_profile,
                "valid_profile": valid_profile,
                "train_profiles": ",".join(train_profiles),
                "capacity_feature": str(cfg.feature_name),
                "capacity_prefix": int(cfg.prefix),
                "capacity_shrink": float(cfg.shrink),
                "capacity_margin": float(cfg.margin),
                "capacity_gate": str(cfg.gate),
                **cap_metrics,
            }
            for prefix_len in PREFIX_WINDOWS:
                mean_ratios = []
                p95_ratios = []
                raw_mean = []
                raw_p95 = []
                for temp in temps:
                    beta, mu, sigma, train_stats = _fit_voltage_model(
                        data,
                        train_profiles,
                        temp,
                        float(cfg.ridge_lambda),
                    )
                    stats = _voltage_residual_stats(
                        data[(test_profile, temp)],
                        soc_preds[temp],
                        (beta, mu, sigma),
                        train_stats,
                        prefix_len,
                    )
                    for name, value in train_stats.items():
                        row[f"{temp:g}C_train_{name}"] = value
                    for name, value in stats.items():
                        row[f"{temp:g}C_prefix{prefix_len}_{name}"] = value
                    mean_ratios.append(float(stats["resid_mean_over_train_mean"]))
                    p95_ratios.append(float(stats["resid_p95_over_train_p95"]))
                    raw_mean.append(float(stats["resid_mean_mV"]))
                    raw_p95.append(float(stats["resid_p95_mV"]))
                row[f"prefix{prefix_len}_max_mean_resid_ratio"] = float(np.max(mean_ratios))
                row[f"prefix{prefix_len}_mean_mean_resid_ratio"] = float(np.mean(mean_ratios))
                row[f"prefix{prefix_len}_max_p95_resid_ratio"] = float(np.max(p95_ratios))
                row[f"prefix{prefix_len}_mean_p95_resid_ratio"] = float(np.mean(p95_ratios))
                row[f"prefix{prefix_len}_max_resid_mean_mV"] = float(np.max(raw_mean))
                row[f"prefix{prefix_len}_max_resid_p95_mV"] = float(np.max(raw_p95))
            rows.append(row)
    return pd.DataFrame(rows).sort_values("target_norm_worst").reset_index(drop=True)


def _correlations(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = []
    for prefix_len in PREFIX_WINDOWS:
        metrics.extend(
            [
                f"prefix{prefix_len}_max_mean_resid_ratio",
                f"prefix{prefix_len}_mean_mean_resid_ratio",
                f"prefix{prefix_len}_max_p95_resid_ratio",
                f"prefix{prefix_len}_mean_p95_resid_ratio",
                f"prefix{prefix_len}_max_resid_mean_mV",
                f"prefix{prefix_len}_max_resid_p95_mV",
            ]
        )
    out = []
    for metric in metrics:
        out.append(
            {
                "metric": metric,
                "pearson_corr_with_target_norm_worst": float(rows[metric].corr(rows["target_norm_worst"])),
                "mean_pass": float(rows[rows["target_met"].astype(bool)][metric].mean()),
                "mean_fail": float(rows[~rows["target_met"].astype(bool)][metric].mean()),
            }
        )
    return pd.DataFrame(out).sort_values("pearson_corr_with_target_norm_worst", ascending=False)


def _thresholds(rows: pd.DataFrame, corrs: pd.DataFrame) -> pd.DataFrame:
    total_fail = int((~rows["target_met"].astype(bool)).sum())
    selected_metrics = corrs["metric"].head(8).tolist()
    out = []
    for metric in selected_metrics:
        values = sorted(set(float(x) for x in rows[metric].round(6)))
        values.append(float("inf"))
        for threshold in values:
            accepted = rows[metric].astype(float) <= float(threshold)
            accepted_rows = rows[accepted]
            rejected_rows = rows[~accepted]
            if accepted_rows.empty:
                continue
            accepted_pass = int(accepted_rows["target_met"].astype(bool).sum())
            accepted_fail = int(len(accepted_rows) - accepted_pass)
            rejected_fail = int((~rejected_rows["target_met"].astype(bool)).sum()) if len(rejected_rows) else 0
            out.append(
                {
                    "metric": metric,
                    "threshold": "inf" if np.isinf(threshold) else float(threshold),
                    "accepted_count": int(len(accepted_rows)),
                    "accepted_pass": accepted_pass,
                    "accepted_fail": accepted_fail,
                    "accepted_precision": float(accepted_pass / len(accepted_rows)),
                    "rejected_count": int(len(rejected_rows)),
                    "rejected_fail": rejected_fail,
                    "failure_recall_by_rejection": float(rejected_fail / total_fail) if total_fail else np.nan,
                    "accepted_mean_target_norm_worst": float(accepted_rows["target_norm_worst"].mean()),
                    "accepted_max_target_norm_worst": float(accepted_rows["target_norm_worst"].max()),
                }
            )
    return pd.DataFrame(out).sort_values(
        ["accepted_precision", "accepted_count", "failure_recall_by_rejection"],
        ascending=[False, False, False],
    )


def run(cfg: VoltageResidualOODConfig) -> dict[str, pd.DataFrame]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _rotation_rows(cfg)
    corrs = _correlations(rows)
    thresholds = _thresholds(rows, corrs)
    rows.to_csv(cfg.output_dir / f"{cfg.output_prefix}_rows.csv", index=False)
    corrs.to_csv(cfg.output_dir / f"{cfg.output_prefix}_correlations.csv", index=False)
    thresholds.to_csv(cfg.output_dir / f"{cfg.output_prefix}_thresholds.csv", index=False)
    write_report(cfg, rows, corrs, thresholds)
    print(corrs.head(12).to_string(index=False))
    print("\nBest thresholds")
    print(thresholds.head(20).to_string(index=False))
    return {"rows": rows, "correlations": corrs, "thresholds": thresholds}


def write_report(
    cfg: VoltageResidualOODConfig,
    rows: pd.DataFrame,
    corrs: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> None:
    detail_cols = [
        *KEYS,
        "0C_MAE_pct",
        "25C_MAE_pct",
        "45C_MAE_pct",
        "target_met",
        "target_norm_worst",
        "prefix512_max_mean_resid_ratio",
        "prefix512_max_p95_resid_ratio",
        "prefix1024_max_mean_resid_ratio",
        "prefix1024_max_p95_resid_ratio",
        "failure_detail",
    ]
    lines = [
        "# NMC Voltage Residual OOD Audit",
        "",
        "## Purpose",
        "",
        "This checks whether a train-profile voltage-response model can identify profile rotations where the fixed bounded capacity anchor will fail.",
        "",
        "This is not strict NoCC: SOC is propagated by explicit current integration from the known initial SOC. The OOD score itself uses only measured voltage/current and the causal SOC estimate, not test SOC labels.",
        "",
        "## Fixed Capacity Anchor",
        "",
        f"- feature: `{cfg.feature_name}`",
        f"- prefix: `{cfg.prefix}`",
        f"- shrink: `{cfg.shrink}`",
        f"- margin: `{cfg.margin}`",
        f"- gate: `{cfg.gate}`",
        "",
        "## Correlations",
        "",
        corrs.head(20).to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Best Threshold Rows",
        "",
        thresholds.head(30).to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Rotation Details",
        "",
        rows[detail_cols].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Interpretation",
        "",
        "- A useful residual OOD score should be larger for failing rotations and smaller for passing rotations.",
        "- Threshold rows are diagnostic; a final threshold would need profile-rotation validation before being claimed.",
        "- If residual scores separate failures better than simple feature coverage, the next observer should use voltage residual behavior as a calibrated uncertainty input.",
        "",
    ]
    (cfg.output_dir / f"{cfg.output_prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voltage residual OOD audit for causal capacity anchor.")
    parser.add_argument("--raw-root", default=VoltageResidualOODConfig.raw_root)
    parser.add_argument("--output-dir", default=VoltageResidualOODConfig.output_dir)
    parser.add_argument("--output-prefix", default=VoltageResidualOODConfig.output_prefix)
    parser.add_argument("--feature-name", default=VoltageResidualOODConfig.feature_name)
    parser.add_argument("--prefix", type=int, default=VoltageResidualOODConfig.prefix)
    parser.add_argument("--shrink", type=float, default=VoltageResidualOODConfig.shrink)
    parser.add_argument("--margin", type=float, default=VoltageResidualOODConfig.margin)
    parser.add_argument("--gate", default=VoltageResidualOODConfig.gate)
    parser.add_argument("--ridge-lambda", type=float, default=VoltageResidualOODConfig.ridge_lambda)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        VoltageResidualOODConfig(
            raw_root=Path(args.raw_root),
            output_dir=Path(args.output_dir),
            output_prefix=str(args.output_prefix),
            feature_name=str(args.feature_name),
            prefix=int(args.prefix),
            shrink=float(args.shrink),
            margin=float(args.margin),
            gate=str(args.gate),
            ridge_lambda=float(args.ridge_lambda),
        )
    )


if __name__ == "__main__":
    main()
