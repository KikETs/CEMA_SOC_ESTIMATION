from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd

from .nmc_branchbands_experiment import build_feature_frames, estimate_r0_by_temperature, find_csv_files
from .nmc_vit_feature_lstm_experiment import add_vit_engineered_features


METRIC_COLS = ["absI_mean", "I_std", "dI_energy", "V_corr_span"]


def _read_prediction(path: Path) -> pd.DataFrame:
    pred = pd.read_csv(path)
    if "temperature_C" not in pred.columns and "temperature" in pred.columns:
        pred["temperature_C"] = pred["temperature"].astype(float)
    if "model_name" not in pred.columns:
        pred["model_name"] = pred["variant"].astype(str) if "variant" in pred.columns else path.stem
    pred["drive_cycle"] = pred["drive_cycle"].astype(str).str.upper()
    pred["trajectory_id"] = pred["trajectory_id"].astype(str)
    pred["end_index"] = pred["end_index"].astype(int)
    pred["error"] = pred["y_pred"].astype(float) - pred["y_true"].astype(float)
    pred["abs_error"] = pred["error"].abs()
    return pred


def _cfg_from_metadata(metadata: dict, base_dir: Path, raw_root: Path | None) -> SimpleNamespace:
    raw = raw_root
    if raw is None:
        raw_meta = metadata.get("raw_root")
        raw = Path(raw_meta) if raw_meta else base_dir / "nmc_soc_ocvstart_relabelled_from_lc_ocv/data/NMC SAMSUNG INR 18650 2Ah"
    if not raw.is_absolute():
        raw = base_dir / raw
    return SimpleNamespace(
        raw_root=raw,
        train_profiles=tuple(metadata.get("train_profiles", ["DST", "US06"])),
        valid_profiles=tuple(metadata.get("valid_profiles", ["VALIDATION"])),
        test_profiles=tuple(metadata.get("test_profiles", ["FUDS"])),
        v_corr_tau_s=float(metadata.get("v_corr_tau_s", 120.0)),
        v_pol_mid_tau_s=float(metadata.get("v_pol_mid_tau_s", 60.0)),
        v_pol_slow_tau_s=float(metadata.get("v_pol_slow_tau_s", 600.0)),
        v_hys_tau_s=float(metadata.get("v_hys_tau_s", 1200.0)),
    )


def _load_frames(base_dir: Path, metadata_path: Path | None, raw_root: Path | None) -> tuple[dict[str, list[pd.DataFrame]], dict]:
    metadata = {}
    if metadata_path is not None and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cfg = _cfg_from_metadata(metadata, base_dir, raw_root)
    files = find_csv_files(cfg.raw_root)
    r0_df = estimate_r0_by_temperature(files, cfg.train_profiles)
    frames = add_vit_engineered_features(build_feature_frames(cfg, files, r0_df))
    return frames, metadata


def _window_metrics(frame: pd.DataFrame, pos: int, window_len: int) -> dict[str, float]:
    start = max(0, int(pos) - int(window_len) + 1)
    end = int(pos)
    win = frame.iloc[start : end + 1]
    i = win["I_raw"].to_numpy(np.float64)
    abs_i = win["absI"].to_numpy(np.float64) if "absI" in win else np.abs(i)
    di = win["dI"].to_numpy(np.float64) if "dI" in win else np.diff(i, prepend=i[0] if len(i) else 0.0)
    v = win["V_corr_raw"].to_numpy(np.float64)
    return {
        "absI_mean": float(np.nanmean(np.abs(abs_i))),
        "I_std": float(np.nanstd(i)),
        "dI_energy": float(np.nanmean(np.square(di))),
        "dI_abs_mean": float(np.nanmean(np.abs(di))),
        "V_corr_span": float(np.nanmax(v) - np.nanmin(v)) if len(v) else float("nan"),
        "V_corr_delta": float(v[-1] - v[0]) if len(v) else float("nan"),
        "low_current_frac": float(np.nanmean(np.abs(abs_i) < 0.05)) if len(abs_i) else float("nan"),
        "V_corr_endpoint": float(v[-1]) if len(v) else float("nan"),
    }


def _make_feature_lookup(frames: dict[str, list[pd.DataFrame]], pred: pd.DataFrame, window_len: int) -> pd.DataFrame:
    frame_by_tid: dict[str, pd.DataFrame] = {}
    pos_by_tid: dict[str, dict[int, int]] = {}
    for split_frames in frames.values():
        for frame in split_frames:
            if frame.empty:
                continue
            tid = str(frame["trajectory_id"].iloc[0])
            frame_by_tid[tid] = frame
            pos_by_tid[tid] = {int(v): int(i) for i, v in enumerate(frame["end_index"].to_numpy())}

    rows = []
    needed = pred[["trajectory_id", "end_index"]].drop_duplicates()
    for _, row in needed.iterrows():
        tid = str(row["trajectory_id"])
        end_idx = int(row["end_index"])
        frame = frame_by_tid.get(tid)
        pos_lookup = pos_by_tid.get(tid, {})
        if frame is None or end_idx not in pos_lookup:
            continue
        pos = pos_lookup[end_idx]
        item = {
            "trajectory_id": tid,
            "end_index": end_idx,
            "SOC_physical_lookup": float(frame["SOC_physical"].iloc[pos]),
        }
        item.update(_window_metrics(frame, pos, int(window_len)))
        rows.append(item)
    return pd.DataFrame(rows)


def _edges_from_file(path: Path | None, train_lookup: pd.DataFrame | None = None) -> dict[str, tuple[float, float]]:
    edges: dict[str, tuple[float, float]] = {}
    if path is not None and path.exists():
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            edges[str(row["metric"])] = (float(row["low_mid_edge"]), float(row["mid_high_edge"]))
    if train_lookup is not None and len(train_lookup):
        for col in METRIC_COLS:
            if col not in edges and col in train_lookup:
                vals = train_lookup[col].replace([np.inf, -np.inf], np.nan).dropna()
                if len(vals):
                    lo, hi = np.nanquantile(vals.to_numpy(float), [1 / 3, 2 / 3])
                    edges[col] = (float(lo), float(hi if hi > lo else lo + 1e-6))
    return edges


def _metric_bin(value: float, edge: tuple[float, float]) -> str:
    if not np.isfinite(value):
        return "nan"
    if value <= edge[0]:
        return "low"
    if value <= edge[1]:
        return "mid"
    return "high"


def _mae_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        err = g["error"].to_numpy(float)
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(
            {
                "n_windows": int(len(g)),
                "MAE_pct": float(np.mean(np.abs(err)) * 100.0),
                "RMSE_pct": float(np.sqrt(np.mean(np.square(err))) * 100.0),
                "bias_pct": float(np.mean(err) * 100.0),
                "catastrophic_gt5_pct": float((np.abs(err) > 0.05).mean() * 100.0),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _correlations(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for (temp, drive), g in df.groupby(["temperature_C", "drive_cycle"], observed=False):
        y = g["abs_error"].astype(float)
        for col in feature_cols:
            x = g[col].astype(float)
            valid = x.replace([np.inf, -np.inf], np.nan).notna() & y.notna()
            if int(valid.sum()) < 5:
                continue
            rows.append(
                {
                    "temperature_C": float(temp),
                    "drive_cycle": str(drive),
                    "feature": col,
                    "pearson_abs_error": float(x[valid].corr(y[valid], method="pearson")),
                    "spearman_abs_error": float(x[valid].corr(y[valid], method="spearman")),
                    "n_windows": int(valid.sum()),
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out["abs_spearman"] = out["spearman_abs_error"].abs()
        out = out.sort_values(["temperature_C", "drive_cycle", "abs_spearman"], ascending=[True, True, False])
    return out


def _table_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    sub = df.copy()
    for col in sub.columns:
        if pd.api.types.is_float_dtype(sub[col]):
            sub[col] = sub[col].map(lambda x: "" if not np.isfinite(x) else f"{x:.4f}")
    header = "| " + " | ".join(str(c) for c in sub.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(sub.columns)) + " |"
    rows = ["| " + " | ".join(str(row[c]) for c in sub.columns) + " |" for _, row in sub.iterrows()]
    return "\n".join([header, sep, *rows])


def _write_report(out_dir: Path, pred_path: Path, pred: pd.DataFrame, regime: pd.DataFrame, corr: pd.DataFrame) -> None:
    selected = pred["diagnostic_test_peek"].any() if "diagnostic_test_peek" in pred.columns else False
    by_temp = _mae_table(pred, ["temperature_C", "drive_cycle"])
    worst_regime = regime.sort_values("MAE_pct", ascending=False).head(12) if len(regime) else pd.DataFrame()
    top_corr = corr.head(12) if len(corr) else pd.DataFrame()
    lines = [
        "# Best Clean Base Regime Diagnostics",
        "",
        "This is a diagnostic analysis of stored prediction rows, not a paper-blind model-selection rule.",
        "",
        f"- prediction file: `{pred_path}`",
        f"- rows: {len(pred)}",
        f"- diagnostic test peek present: {bool(selected)}",
        "",
        "## Temperature Summary",
        "",
        _table_md(by_temp),
        "",
        "## Worst Label-Free Regime Cells",
        "",
        _table_md(worst_regime),
        "",
        "## Strongest Error Correlations",
        "",
        _table_md(top_corr),
        "",
        "## Modeling Implication",
        "",
        "- If high-error cells align with excitation-regime features, the next candidate should change the Stage 1 representation or objective, not add a temperature-only correction.",
        "- Current remains an instantaneous excitation signal; these diagnostics do not introduce current integration or cumulative Ah.",
        "- Any future gate must be based on validation-only or predeclared final/last-K rules, not on this test diagnostic.",
        "",
    ]
    (out_dir / "best_clean_regime_diagnostics_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = ArgumentParser()
    p.add_argument("--base-dir", type=Path, default=Path("."))
    p.add_argument("--prediction", type=Path, required=True)
    p.add_argument("--metadata", type=Path, default=None)
    p.add_argument("--raw-root", type=Path, default=None)
    p.add_argument("--regime-edges", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("paper_results/best_clean_regime_diagnostics"))
    p.add_argument("--window-len", type=int, default=0)
    args = p.parse_args()

    base_dir = args.base_dir.resolve()
    out_dir = args.output_dir if args.output_dir.is_absolute() else base_dir / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, metadata = _load_frames(base_dir, args.metadata, args.raw_root)
    window_len = int(args.window_len or metadata.get("window_len", 50))
    pred = _read_prediction(args.prediction)
    lookup = _make_feature_lookup(frames, pred, window_len)
    merged = pred.merge(lookup, on=["trajectory_id", "end_index"], how="left", validate="many_to_one")
    missing_lookup = int(merged["absI_mean"].isna().sum()) if "absI_mean" in merged else len(merged)
    if missing_lookup:
        print(f"Warning: {missing_lookup} prediction rows did not match feature lookup.")

    edges = _edges_from_file(args.regime_edges)
    for col in METRIC_COLS:
        if col in merged and col in edges:
            merged[f"{col}_bin"] = [_metric_bin(v, edges[col]) for v in merged[col].to_numpy(float)]

    merged["soc_bin"] = pd.cut(
        merged["y_true"].astype(float),
        bins=[-np.inf, 0.2, 0.5, 0.8, np.inf],
        labels=["soc0_0_20", "soc1_20_50", "soc2_50_80", "soc3_80_100"],
    )
    feature_cols = [
        "absI_mean",
        "I_std",
        "dI_energy",
        "dI_abs_mean",
        "V_corr_span",
        "V_corr_delta",
        "low_current_frac",
        "V_corr_endpoint",
    ]
    bin_cols = [f"{col}_bin" for col in METRIC_COLS if f"{col}_bin" in merged.columns]
    regime_cols = ["temperature_C", "drive_cycle"] + bin_cols
    regime = _mae_table(merged, regime_cols) if bin_cols else pd.DataFrame()
    soc_bins = _mae_table(merged, ["temperature_C", "drive_cycle", "soc_bin"])
    by_traj = _mae_table(merged, ["temperature_C", "drive_cycle", "trajectory_id"])
    corr = _correlations(merged, [col for col in feature_cols if col in merged.columns])
    worst = merged.sort_values("abs_error", ascending=False).head(200)

    merged.to_csv(out_dir / "best_clean_prediction_rows_with_regime.csv.gz", index=False, compression="gzip")
    regime.to_csv(out_dir / "best_clean_regime_error_summary.csv", index=False)
    soc_bins.to_csv(out_dir / "best_clean_soc_bin_error_summary.csv", index=False)
    by_traj.to_csv(out_dir / "best_clean_by_trajectory.csv", index=False)
    corr.to_csv(out_dir / "best_clean_error_feature_correlations.csv", index=False)
    worst.to_csv(out_dir / "best_clean_worst_windows.csv", index=False)
    _write_report(out_dir, args.prediction, merged, regime, corr)
    print(f"Wrote diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
