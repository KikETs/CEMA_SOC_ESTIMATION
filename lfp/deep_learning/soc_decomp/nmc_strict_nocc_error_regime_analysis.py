from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .models import DecomposedWindowDataset
from .nmc_branchbands_experiment import build_feature_frames, estimate_r0_by_temperature, find_csv_files
from .nmc_vcorr_it_train_dst_selector_run import (
    REGIME_SELECTOR_METRICS,
    TrainDSTSelectorConfig,
    _filter_scaled_frames_by_temperatures,
    _metric_bin,
    _regime_edges_from_train_ds,
    _selected_feature_columns,
)
from .nmc_vit_feature_lstm_experiment import add_vit_engineered_features
from .training import make_scaled_frames_for_ablation


PREFIX = "nmc_strict_nocc_regime_selector_fixed250_predrows_fast_vcorr_it_excitation_ema_E1_DST_US06_to_FUDS_e250_seed0"
VARIANT = "condinv_trainDST25_selected_seed0_ep250_base"


def _window_rows(ds: DecomposedWindowDataset, metric_prefix: str = "") -> pd.DataFrame:
    feature_to_idx = {name: idx for idx, name in enumerate(ds.feature_cols)}
    rows = []
    for row_id, (fi, start, end) in enumerate(ds.index):
        frame = ds.frames[fi]
        sl = slice(int(start), int(end) + 1)
        x = frame["x"][sl]
        i = x[:, feature_to_idx["I_raw"]].astype(np.float64)
        v = x[:, feature_to_idx["V_corr_raw"]].astype(np.float64)
        if "absI" in feature_to_idx:
            abs_i = np.abs(x[:, feature_to_idx["absI"]].astype(np.float64))
        else:
            abs_i = np.abs(i)
        if "dI" in feature_to_idx:
            d_i = x[:, feature_to_idx["dI"]].astype(np.float64)
        else:
            d_i = np.diff(i, prepend=i[0] if len(i) else 0.0)
        rows.append(
            {
                "row_id": int(row_id),
                "file_name": str(frame["file_name"][end]),
                "trajectory_id": str(frame["trajectory_id"][end]),
                "end_index": int(frame["end_index"][end]),
                "temperature": float(frame["temperature"][end]),
                "drive_cycle": str(frame["drive_cycle"][end]).upper(),
                f"{metric_prefix}absI_mean": float(np.nanmean(np.abs(abs_i))),
                f"{metric_prefix}I_std": float(np.nanstd(i)),
                f"{metric_prefix}dI_energy": float(np.nanmean(np.square(d_i))),
                f"{metric_prefix}V_corr_span": float(np.nanmax(v) - np.nanmin(v)) if len(v) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _add_bins(rows: pd.DataFrame, edges: dict[str, tuple[float, float]]) -> pd.DataFrame:
    out = rows.copy()
    for col in REGIME_SELECTOR_METRICS:
        out[f"{col}_bin"] = [_metric_bin(float(v), edges[col]) for v in out[col]]
    out["regime_key"] = [
        "|".join(str(row[f"{col}_bin"]) for col in REGIME_SELECTOR_METRICS)
        for _, row in out.iterrows()
    ]
    return out


def _make_datasets(base_dir: Path):
    cfg = TrainDSTSelectorConfig(
        base_dir=base_dir,
        raw_root=Path("nmc_soc_ocvstart_relabelled_from_lc_ocv/data/NMC SAMSUNG INR 18650 2Ah"),
        output_prefix=PREFIX,
        seeds=(0,),
        train_profiles=("DST", "US06"),
        valid_profiles=("VALIDATION",),
        test_profiles=("FUDS",),
        feature_set="vcorr_it_excitation_ema",
        stage2_feature_set="vcorr_it_excitation_ema",
        window_len=50,
        stride=3,
    )
    cfg.raw_root = cfg.base_dir / cfg.raw_root
    files = find_csv_files(cfg.raw_root)
    r0_df = estimate_r0_by_temperature(files, cfg.train_profiles)
    frames = add_vit_engineered_features(build_feature_frames(cfg, files, r0_df))
    feature_cols = _selected_feature_columns(cfg.feature_set)
    scaled, _ = make_scaled_frames_for_ablation(frames, feature_cols)
    scaled = _filter_scaled_frames_by_temperatures(
        scaled,
        train_temperatures=(),
        valid_temperatures=(),
        test_temperatures=(),
        name="stage1",
    )
    raw = _filter_scaled_frames_by_temperatures(
        frames,
        train_temperatures=(),
        valid_temperatures=(),
        test_temperatures=(),
        name="raw",
    )
    train_ds = DecomposedWindowDataset(scaled["train"], feature_cols, cfg.window_len, cfg.stride, target_label="physical")
    valid_scaled_ds = DecomposedWindowDataset(scaled["valid"], feature_cols, cfg.window_len, 1, target_label="physical")
    test_scaled_ds = DecomposedWindowDataset(scaled["test"], feature_cols, cfg.window_len, 1, target_label="physical")
    valid_raw_ds = DecomposedWindowDataset(raw["valid"], feature_cols, cfg.window_len, 1, target_label="physical")
    test_raw_ds = DecomposedWindowDataset(raw["test"], feature_cols, cfg.window_len, 1, target_label="physical")
    return train_ds, {"valid": (valid_scaled_ds, valid_raw_ds), "test": (test_scaled_ds, test_raw_ds)}


def _attach_regime(pred: pd.DataFrame, scaled_ds: DecomposedWindowDataset, raw_ds: DecomposedWindowDataset, edges) -> pd.DataFrame:
    scaled = _add_bins(_window_rows(scaled_ds), edges)
    raw = _window_rows(raw_ds, metric_prefix="raw_")
    keep_scaled = [
        "row_id",
        "file_name",
        "trajectory_id",
        "end_index",
        "temperature",
        "drive_cycle",
        *REGIME_SELECTOR_METRICS,
        *(f"{col}_bin" for col in REGIME_SELECTOR_METRICS),
        "regime_key",
    ]
    keep_raw = [
        "row_id",
        "raw_absI_mean",
        "raw_I_std",
        "raw_dI_energy",
        "raw_V_corr_span",
    ]
    merged = pred.merge(scaled[keep_scaled], on=["row_id", "file_name", "trajectory_id", "end_index", "temperature", "drive_cycle"], how="left")
    merged = merged.merge(raw[keep_raw], on="row_id", how="left")
    merged["abs_error_pct"] = merged["abs_error"] * 100.0
    merged["error_pct"] = merged["error"] * 100.0
    merged["soc_pct"] = merged["y_true"] * 100.0
    return merged


def _metric_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n=("abs_error_pct", "size"),
            MAE_pct=("abs_error_pct", "mean"),
            RMSE_pct=("error_pct", lambda x: float(np.sqrt(np.nanmean(np.square(x))))),
            bias_pct=("error_pct", "mean"),
            frac_windows=("abs_error_pct", lambda x: float(len(x) / max(len(df), 1))),
            raw_absI_mean_A=("raw_absI_mean", "mean"),
            raw_I_std_A=("raw_I_std", "mean"),
            raw_dI_energy_A2=("raw_dI_energy", "mean"),
            raw_V_corr_span_V=("raw_V_corr_span", "mean"),
            soc_mean_pct=("soc_pct", "mean"),
        )
        .reset_index()
    )
    out["error_contribution"] = out["n"] * out["MAE_pct"]
    total = float(out["error_contribution"].sum())
    out["error_contribution_frac"] = out["error_contribution"] / total if total > 0 else np.nan
    return out.sort_values(["error_contribution_frac", "MAE_pct"], ascending=[False, False]).reset_index(drop=True)


def _coverage_shift(valid25: pd.DataFrame, test25: pd.DataFrame) -> pd.DataFrame:
    v = valid25["regime_key"].value_counts(normalize=True).rename("valid_frac")
    t = test25["regime_key"].value_counts(normalize=True).rename("test_frac")
    out = pd.concat([v, t], axis=1).fillna(0.0).reset_index().rename(columns={"index": "regime_key"})
    out["test_minus_valid_frac"] = out["test_frac"] - out["valid_frac"]
    test_mae = test25.groupby("regime_key")["abs_error_pct"].mean().rename("test_MAE_pct")
    valid_mae = valid25.groupby("regime_key")["abs_error_pct"].mean().rename("valid_MAE_pct")
    out = out.merge(valid_mae, on="regime_key", how="left").merge(test_mae, on="regime_key", how="left")
    return out.sort_values(["test_minus_valid_frac", "test_MAE_pct"], ascending=[False, False]).reset_index(drop=True)


def _write_report(out_dir: Path, test_summary: pd.DataFrame, by_regime_25: pd.DataFrame, coverage: pd.DataFrame, soc_bins: pd.DataFrame) -> None:
    selected = test_summary[test_summary["variant"].eq(VARIANT)].copy()
    if selected.empty:
        selected = test_summary.tail(1).copy()
    row = selected.iloc[0]
    top_regimes = by_regime_25.head(8)
    top_cov = coverage.head(8)
    worst_soc = soc_bins.sort_values("MAE_pct", ascending=False).head(6)

    def table(df: pd.DataFrame, cols: list[str]) -> str:
        sub = df[cols].copy()
        for col in sub.columns:
            if pd.api.types.is_float_dtype(sub[col]):
                sub[col] = sub[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row_item in sub.iterrows():
            values = [str(row_item[col]).replace("|", "\\|") for col in cols]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    text = f"""# NMC Strict NoCC 25C FUDS Error-Regime Analysis

## Scope

This is not a new model-selection run. The selected epoch is fixed at 250 and FUDS is used only for post-hoc failure analysis.

Strict NoCC status remains unchanged:

- no SOC input;
- no initial/window-start SOC input;
- no SOC_CC or cumulative Ah input;
- no explicit current-integration SOC state update;
- current is used only as instantaneous/local excitation.

## Fixed-Epoch Test Result

| temp | MAE |
|---:|---:|
| 0C | {float(row.get('0.0', np.nan)):.3f} |
| 25C | {float(row.get('25.0', np.nan)):.3f} |
| 45C | {float(row.get('45.0', np.nan)):.3f} |

The model still fails the 25C target of 0.7% MAE.

## Dominant 25C FUDS Error Regimes

Regime key order:

```text
absI_mean_bin | I_std_bin | dI_energy_bin | V_corr_span_bin
```

{table(top_regimes, ['regime_key', 'n', 'frac_windows', 'MAE_pct', 'bias_pct', 'error_contribution_frac', 'raw_absI_mean_A', 'raw_I_std_A', 'raw_dI_energy_A2', 'raw_V_corr_span_V', 'soc_mean_pct'])}

## 25C Validation/Test Coverage Shift

Positive `test_minus_valid_frac` means the regime is more common in FUDS than in VALIDATION.

{table(top_cov, ['regime_key', 'valid_frac', 'test_frac', 'test_minus_valid_frac', 'valid_MAE_pct', 'test_MAE_pct'])}

## 25C Error By SOC Region

This SOC binning is for analysis only, not model selection.

{table(worst_soc, ['soc_bin_pct', 'n', 'MAE_pct', 'bias_pct', 'frac_windows'])}

## Interpretation

The 25C failure is concentrated in label-free excitation/voltage-response regimes, not in a temperature-only failure. The same fixed Stage1 model passes 0C and 45C targets but misses 25C FUDS, which is consistent with a profile-regime transfer problem.

Paper-safe wording:

> In the strict NoCC ablation, the model used current only as instantaneous excitation. A validation-regime-selected Stage1 model transferred to 0C and 45C FUDS within target, but failed 25C FUDS. Post-hoc regime analysis indicates that the failure is associated with profile-specific excitation/voltage-response regimes rather than temperature alone.

Do not claim that strict NoCC is a main estimator unless a Stage1 base passes the fixed promotion gate without correction.
"""
    (out_dir / "nmc_strict_nocc_fuds25_error_regime_analysis.md").write_text(text, encoding="utf-8")


def main() -> None:
    base_dir = Path(".").resolve()
    out_dir = base_dir / "paper_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = base_dir / "remote_result_summaries"
    test_pred_path = pred_dir / f"{PREFIX}_seed0_sel250_{VARIANT}_test_prediction_rows.csv.gz"
    valid_pred_path = pred_dir / f"{PREFIX}_seed0_sel250_{VARIANT}_valid_prediction_rows.csv.gz"
    test_summary_path = pred_dir / f"{PREFIX}_test_summary.csv"

    train_ds, datasets = _make_datasets(base_dir)
    edges = _regime_edges_from_train_ds(train_ds)
    edge_df = pd.DataFrame(
        [{"metric": k, "low_mid_edge": v[0], "mid_high_edge": v[1]} for k, v in edges.items()]
    )

    test_pred = pd.read_csv(test_pred_path)
    valid_pred = pd.read_csv(valid_pred_path)
    test = _attach_regime(test_pred, *datasets["test"], edges)
    valid = _attach_regime(valid_pred, *datasets["valid"], edges)
    all_rows = pd.concat([valid, test], ignore_index=True)

    test25 = test[np.isclose(test["temperature"], 25.0)].copy()
    valid25 = valid[np.isclose(valid["temperature"], 25.0)].copy()

    by_temp_regime = _metric_summary(test, ["temperature", "regime_key"])
    by_regime_25 = _metric_summary(test25, ["regime_key"])
    by_metric_bin_25 = pd.concat(
        [
            _metric_summary(test25, [f"{metric}_bin"]).assign(metric=metric).rename(columns={f"{metric}_bin": "bin"})
            for metric in REGIME_SELECTOR_METRICS
        ],
        ignore_index=True,
    )
    coverage = _coverage_shift(valid25, test25)
    soc_edges = [0, 20, 40, 60, 80, 100]
    test25["soc_bin_pct"] = pd.cut(test25["soc_pct"], bins=soc_edges, include_lowest=True).astype(str)
    soc_bins = _metric_summary(test25, ["soc_bin_pct"])
    top_windows = test25.sort_values("abs_error_pct", ascending=False).head(200)

    all_rows.to_csv(out_dir / "nmc_strict_nocc_fuds25_error_window_rows.csv.gz", index=False, compression="gzip")
    edge_df.to_csv(out_dir / "nmc_strict_nocc_fuds25_error_regime_edges.csv", index=False)
    by_temp_regime.to_csv(out_dir / "nmc_strict_nocc_fuds25_error_by_temp_regime.csv", index=False)
    by_regime_25.to_csv(out_dir / "nmc_strict_nocc_fuds25_error_by_regime.csv", index=False)
    by_metric_bin_25.to_csv(out_dir / "nmc_strict_nocc_fuds25_error_by_metric_bin.csv", index=False)
    coverage.to_csv(out_dir / "nmc_strict_nocc_fuds25_error_coverage_shift.csv", index=False)
    soc_bins.to_csv(out_dir / "nmc_strict_nocc_fuds25_error_by_soc_bin.csv", index=False)
    top_windows.to_csv(out_dir / "nmc_strict_nocc_fuds25_error_top_windows.csv", index=False)
    _write_report(out_dir, pd.read_csv(test_summary_path), by_regime_25, coverage, soc_bins)
    print(out_dir / "nmc_strict_nocc_fuds25_error_regime_analysis.md")


if __name__ == "__main__":
    main()
