#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from soc_decomp.nmc_branchbands_experiment import (  # noqa: E402
    NMCBranchBandsConfig,
    build_decomposed_frame,
)
from soc_decomp.nmc_vit_feature_lstm_experiment import (  # noqa: E402
    add_vit_engineered_features,
    causal_index_ema,
    causal_rolling_mean,
    causal_rolling_span,
    causal_rolling_std,
)


ROOT = PROJECT_ROOT
RAW_ROOT = ROOT / "nmc_soc80_train_nofloor_qmax"
OUT_DIR = ROOT / "nmc_goal_vcorr_it_train_dst_selector_results"

SUMMARY_OUT = OUT_DIR / "soc80_w50_lopo_w1p4_lam1_input_gate_feature_audit_v1.csv"
SCREEN_OUT = OUT_DIR / "soc80_w50_lopo_w1p4_lam1_input_gate_threshold_screen_v1.csv"
REPORT_OUT = OUT_DIR / "soc80_w50_lopo_w1p4_lam1_input_gate_audit_v1.md"

PRED_PATTERNS = (
    "soc80_goal_w50_deterministic_lopo_w1p4_lam1_seed012_v1_*_prediction_rows.csv.gz",
    "soc80_goal_w50_deterministic_VALIDATION_w1p4_lam1_seed02_v1_*_prediction_rows.csv.gz",
    "soc80_goal_w50_deterministic_VALIDATION_seed1_nearbest_grid_v1_grid_nearbest_w1p4_w25x2p2_under0p5_lam1p0_*_prediction_rows.csv.gz",
)

FEATURE_COLS = (
    "V_raw",
    "V_corr_raw",
    "V_corr_rel_start",
    "V_corr_raw_ema50",
    "V_corr_raw_ema200",
    "V_corr_raw_ema800",
    "V_corr_raw_dev_ema50",
    "V_corr_raw_dev_ema200",
    "V_corr_raw_ema50_minus_ema800",
    "V_corr_raw_ema200_minus_ema800",
    "V_corr_raw_span50",
    "V_corr_raw_span200",
    "V_corr_raw_span800",
    "V_corr_raw_span50_ema200",
    "I_raw",
    "absI",
    "I_raw_std50",
    "I_raw_std200",
    "dI",
    "dI_abs_mean",
    "V_ohm_free_raw",
    "V_eq_slow_raw",
    "V_dyn_slow_raw",
    "V_dyn_slow_raw_ema50",
    "V_dyn_slow_raw_ema200",
)

GATE_FEATURES = (
    "V_corr_raw",
    "V_corr_rel_start",
    "V_corr_raw_ema200",
    "V_corr_raw_dev_ema200",
    "V_corr_raw_ema50_minus_ema800",
    "V_corr_raw_span50",
    "V_corr_raw_span50_ema200",
    "I_raw_std50",
    "absI_mean50",
    "dI_energy50",
    "voltage_sag50",
    "V_dyn_slow_abs_mean50",
)


def _prediction_files() -> list[Path]:
    files: list[Path] = []
    for pat in PRED_PATTERNS:
        files.extend(sorted(OUT_DIR.glob(pat)))
    unique = {p.resolve(): p for p in files}
    return [unique[k] for k in sorted(unique)]


def _decomp_prefixes() -> list[tuple[str, Path]]:
    rows = []
    for p in OUT_DIR.glob("*_decomposition_params.csv"):
        stem = p.name[: -len("_decomposition_params.csv")]
        rows.append((stem, p))
    return sorted(rows, key=lambda x: len(x[0]), reverse=True)


def _decomp_for_prediction(path: Path, prefixes: list[tuple[str, Path]]) -> Path:
    name = path.name
    for prefix, p in prefixes:
        if name.startswith(prefix + "_seed"):
            return p
    raise FileNotFoundError(f"Could not match decomposition params for {path.name}")


def _seed_from_name(name: str) -> int:
    m = re.search(r"_seed(\d+)_sel", name)
    if not m:
        m = re.search(r"_seeds?(\d+)", name)
    if not m:
        raise ValueError(f"Could not parse seed from {name}")
    return int(m.group(1))


def _temp_dir(temp: float) -> str:
    if abs(float(temp)) < 1e-6:
        return "0C"
    return f"{int(round(float(temp)))}C"


def _build_feature_lookup(raw_path: Path, r0_lookup: dict[float, float], cfg: NMCBranchBandsConfig) -> pd.DataFrame:
    frame = build_decomposed_frame(raw_path, r0_lookup, cfg)
    frame = add_vit_engineered_features({"test": [frame]})["test"][0].copy()
    v = frame["V_corr_raw"].to_numpy(np.float64)
    i = frame["I_raw"].to_numpy(np.float64)
    di = frame["dI"].to_numpy(np.float64)
    dyn = frame["V_dyn_slow_raw"].to_numpy(np.float64)
    frame["absI_mean50"] = causal_rolling_mean(np.abs(i), 50)
    frame["I_raw_std50_recalc"] = causal_rolling_std(i, 50)
    frame["dI_energy50"] = causal_rolling_mean(di * di, 50)
    frame["V_corr_span50_recalc"] = causal_rolling_span(v, 50)
    frame["voltage_sag50"] = causal_rolling_mean(v, 50) - v
    frame["V_dyn_slow_abs_mean50"] = causal_rolling_mean(np.abs(dyn), 50)
    keep = ["file_name", "end_index", "temperature", "drive_cycle", *FEATURE_COLS, *GATE_FEATURES]
    keep = list(dict.fromkeys(c for c in keep if c in frame.columns))
    return frame[keep].copy()


def _attach_features(pred_path: Path, decomp_path: Path, cache: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    pred = pd.read_csv(pred_path)
    pred = pred[pred["temperature"].astype(float).round(6) == 0.0].copy()
    pred["seed"] = _seed_from_name(pred_path.name)
    pred["pred_path"] = pred_path.name
    pred["decomposition_path"] = decomp_path.name

    r0_df = pd.read_csv(decomp_path)
    r0_lookup = {float(r["temperature_C"]): float(r["r0_ohm"]) for _, r in r0_df.iterrows()}
    cfg = NMCBranchBandsConfig(raw_root=RAW_ROOT)

    parts = []
    for (file_name, temp), g in pred.groupby(["file_name", "temperature"], sort=False):
        raw_path = RAW_ROOT / _temp_dir(float(temp)) / str(file_name)
        key = (decomp_path.name, str(raw_path))
        if key not in cache:
            cache[key] = _build_feature_lookup(raw_path, r0_lookup, cfg)
        feat = cache[key]
        merged = g.merge(feat, on=["file_name", "end_index"], how="left", suffixes=("", "_feat"))
        parts.append(merged)
    out = pd.concat(parts, ignore_index=True)
    missing = int(out["V_corr_raw"].isna().sum()) if "V_corr_raw" in out.columns else len(out)
    if missing:
        raise RuntimeError(f"Missing feature joins for {pred_path.name}: {missing}")
    return out


def _add_lag_predictions(df: pd.DataFrame, taus: tuple[int, ...] = (20, 50, 75, 100)) -> pd.DataFrame:
    out = df.copy()
    for tau in taus:
        out[f"y_lag{tau}"] = np.nan
    for _, idx in out.groupby(["pred_path", "trajectory_id"], sort=False).groups.items():
        order = out.loc[idx].sort_values("end_index").index
        raw = out.loc[order, "y_pred"].to_numpy(np.float64)
        for tau in taus:
            out.loc[order, f"y_lag{tau}"] = causal_index_ema(raw, tau)
    for tau in taus:
        out[f"abs_error_lag{tau}"] = (out[f"y_lag{tau}"] - out["y_true"]).abs()
        out[f"error_lag{tau}"] = out[f"y_lag{tau}"] - out["y_true"]
    return out


def _group_metrics(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    d = df.copy()
    d["err_tmp"] = d[pred_col] - d["y_true"]
    d["abs_tmp"] = d["err_tmp"].abs()
    rows = []
    for (holdout, seed), g in d.groupby(["drive_cycle", "seed"], sort=True):
        row = {
            "holdout": holdout,
            "seed": int(seed),
            "n_windows": int(len(g)),
            "mae0_pct": float(g["abs_tmp"].mean() * 100.0),
            "bias0_pct": float(g["err_tmp"].mean() * 100.0),
            "pred_col": pred_col,
        }
        for col in GATE_FEATURES:
            if col in g.columns:
                row[f"{col}_mean"] = float(g[col].mean())
                row[f"{col}_p10"] = float(g[col].quantile(0.10))
                row[f"{col}_p50"] = float(g[col].quantile(0.50))
                row[f"{col}_p90"] = float(g[col].quantile(0.90))
        rows.append(row)
    return pd.DataFrame(rows)


def _screen_threshold_gates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = df[["drive_cycle", "seed", "y_true", "y_pred", "y_lag20", "y_lag50", "y_lag75", "y_lag100", *GATE_FEATURES]].copy()
    for feat in GATE_FEATURES:
        if feat not in base.columns:
            continue
        values = base[feat].replace([np.inf, -np.inf], np.nan).dropna()
        if values.nunique() < 4:
            continue
        qs = np.unique(values.quantile(np.linspace(0.05, 0.95, 19)).to_numpy())
        for tau in (20, 50, 75, 100):
            lag_col = f"y_lag{tau}"
            for direction in ("<=", ">="):
                for thr in qs:
                    gate = base[feat] <= float(thr) if direction == "<=" else base[feat] >= float(thr)
                    pred = np.where(gate.to_numpy(), base[lag_col].to_numpy(), base["y_pred"].to_numpy())
                    err = pred - base["y_true"].to_numpy()
                    tmp = pd.DataFrame(
                        {
                            "drive_cycle": base["drive_cycle"].to_numpy(),
                            "seed": base["seed"].to_numpy(),
                            "abs_error": np.abs(err),
                            "error": err,
                        }
                    )
                    agg = tmp.groupby(["drive_cycle", "seed"], sort=True).agg(
                        mae0=("abs_error", lambda x: float(np.mean(x) * 100.0)),
                        bias0=("error", lambda x: float(np.mean(x) * 100.0)),
                    )
                    rows.append(
                        {
                            "gate_type": "raw_or_lag",
                            "feature": feat,
                            "direction": direction,
                            "threshold": float(thr),
                            "tau": int(tau),
                            "offset_pct": 0.0,
                            "mean_mae0_pct": float(agg["mae0"].mean()),
                            "max_seed_mae0_pct": float(agg["mae0"].max()),
                            "n_groups_over_0p8": int((agg["mae0"] > 0.8).sum()),
                            "worst_group": f"{agg['mae0'].idxmax()[0]}:seed{agg['mae0'].idxmax()[1]}",
                        }
                    )
        for direction in ("<=", ">="):
            for thr in qs:
                gate = base[feat] <= float(thr) if direction == "<=" else base[feat] >= float(thr)
                for offset_pct in (-0.50, -0.35, -0.25, -0.15, 0.15, 0.25, 0.35, 0.50):
                    pred = base["y_pred"].to_numpy() + gate.to_numpy(dtype=float) * (offset_pct / 100.0)
                    err = pred - base["y_true"].to_numpy()
                    tmp = pd.DataFrame(
                        {
                            "drive_cycle": base["drive_cycle"].to_numpy(),
                            "seed": base["seed"].to_numpy(),
                            "abs_error": np.abs(err),
                            "error": err,
                        }
                    )
                    agg = tmp.groupby(["drive_cycle", "seed"], sort=True).agg(
                        mae0=("abs_error", lambda x: float(np.mean(x) * 100.0)),
                        bias0=("error", lambda x: float(np.mean(x) * 100.0)),
                    )
                    rows.append(
                        {
                            "gate_type": "raw_or_offset",
                            "feature": feat,
                            "direction": direction,
                            "threshold": float(thr),
                            "tau": 0,
                            "offset_pct": float(offset_pct),
                            "mean_mae0_pct": float(agg["mae0"].mean()),
                            "max_seed_mae0_pct": float(agg["mae0"].max()),
                            "n_groups_over_0p8": int((agg["mae0"] > 0.8).sum()),
                            "worst_group": f"{agg['mae0'].idxmax()[0]}:seed{agg['mae0'].idxmax()[1]}",
                        }
                    )
    out = pd.DataFrame(rows)
    return out.sort_values(["n_groups_over_0p8", "max_seed_mae0_pct", "mean_mae0_pct"]).reset_index(drop=True)


def main() -> None:
    pred_files = _prediction_files()
    if not pred_files:
        raise FileNotFoundError("No prediction files matched current w50 w1p4_lam1 patterns.")
    prefixes = _decomp_prefixes()
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    parts = []
    for pred_path in pred_files:
        decomp_path = _decomp_for_prediction(pred_path, prefixes)
        parts.append(_attach_features(pred_path, decomp_path, cache))
    rows = _add_lag_predictions(pd.concat(parts, ignore_index=True))

    summary_parts = [_group_metrics(rows, "y_pred")]
    for tau in (20, 50, 75, 100):
        summary_parts.append(_group_metrics(rows, f"y_lag{tau}"))
    summary = pd.concat(summary_parts, ignore_index=True)
    summary.to_csv(SUMMARY_OUT, index=False)

    screen = _screen_threshold_gates(rows)
    screen.to_csv(SCREEN_OUT, index=False)

    raw = summary[summary["pred_col"] == "y_pred"].copy()
    lag75 = summary[summary["pred_col"] == "y_lag75"].copy()
    best = screen.head(20)
    report = [
        "# SOC80 w50 input-only gate audit v1",
        "",
        "Scope: saved w50 w1p4_lam1 predictions only; 0C rows only; raw V/I/T-derived features only.",
        "This is a diagnostic screen, not a paper-valid final protocol by itself, because thresholds are ranked using held-out errors.",
        "",
        "## Raw 0C MAE by holdout/seed",
        raw[["holdout", "seed", "mae0_pct", "bias0_pct", "n_windows"]].sort_values(["holdout", "seed"]).to_markdown(index=False),
        "",
        "## Lag75 0C MAE by holdout/seed",
        lag75[["holdout", "seed", "mae0_pct", "bias0_pct", "n_windows"]].sort_values(["holdout", "seed"]).to_markdown(index=False),
        "",
        "## Best diagnostic threshold gates",
        best.to_markdown(index=False),
        "",
        "## Outputs",
        f"- `{SUMMARY_OUT.name}`",
        f"- `{SCREEN_OUT.name}`",
    ]
    REPORT_OUT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {SUMMARY_OUT}")
    print(f"wrote {SCREEN_OUT}")
    print(f"wrote {REPORT_OUT}")


if __name__ == "__main__":
    main()
