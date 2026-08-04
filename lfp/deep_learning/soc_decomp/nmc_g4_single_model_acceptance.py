from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SingleModelFold:
    protocol: str
    holdout: str
    model_family: str
    feature_set: str
    prediction_file: str


ROOT = Path("/home/user/Desktop/CEMA_LOPO_LSTM_anchorL_isolated")
OUT_DIR = ROOT / "nmc_goal_vcorr_it_train_dst_selector_results"
PREFIX = "nmc_g4_single_model_acceptance"


FOLDS = [
    SingleModelFold(
        "single_g4_lstm_start_context_raw",
        "DST",
        "LSTM_h128_l1_seed0_e200",
        "paper_g4_all_ema_start",
        "lopo_g4_lstm_single_g4start_holdoutdst_seed0_e200_seed0_sel200_condinv_trainDST25_selected_seed0_ep200_base_test_prediction_rows.csv.gz",
    ),
    SingleModelFold(
        "single_g4_lstm_start_context_raw",
        "US06",
        "LSTM_h128_l1_seed0_e200",
        "paper_g4_all_ema_start",
        "lopo_g4_lstm_single_g4start_holdoutus06_seed0_e200_seed0_sel200_condinv_trainDST25_selected_seed0_ep200_base_test_prediction_rows.csv.gz",
    ),
    SingleModelFold(
        "single_g4_lstm_start_context_raw",
        "FUDS",
        "LSTM_h128_l1_seed0_e200",
        "paper_g4_all_ema_start",
        "lopo_g4_lstm_single_g4start_holdoutfuds_seed0_e200_seed0_sel200_condinv_trainDST25_selected_seed0_ep200_base_test_prediction_rows.csv.gz",
    ),
]


def load_prediction(fold: SingleModelFold) -> pd.DataFrame:
    path = OUT_DIR / fold.prediction_file
    if not path.exists():
        raise FileNotFoundError(path)
    pred = pd.read_csv(path)
    if "split" in pred.columns:
        pred = pred[pred["split"].eq("test")].copy()
    if "temperature_C" not in pred.columns:
        pred["temperature_C"] = pred["temperature"]
    required = {"temperature_C", "y_true", "y_pred"}
    missing = required - set(pred.columns)
    if missing:
        raise RuntimeError(f"{path.name} missing columns: {sorted(missing)}")
    return pred


def metric_rows(fold: SingleModelFold, pred: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for temp, group in pred.groupby("temperature_C", sort=True):
        err = group["y_pred"].to_numpy(np.float64) - group["y_true"].to_numpy(np.float64)
        rows.append(
            {
                "protocol": fold.protocol,
                "holdout": fold.holdout,
                "model_family": fold.model_family,
                "feature_set": fold.feature_set,
                "temperature_C": float(temp),
                "n_rows": int(len(group)),
                "MAE_pct": float(np.mean(np.abs(err)) * 100.0),
                "RMSE_pct": float(np.sqrt(np.mean(err**2)) * 100.0),
                "bias_pct": float(np.mean(err) * 100.0),
                "prediction_file": fold.prediction_file,
            }
        )
    return rows


def write_report(summary: pd.DataFrame, detail: pd.DataFrame) -> None:
    lines = [
        "# G4 Single-Model Acceptance Check",
        "",
        "This audit reports raw single-model LOPO performance only. It excludes post-hoc OCV offset correction, model ensembling, Transformer fallback, and per-holdout architecture selection.",
        "",
        "`paper_g4_all_ema_start` is `paper_g4_all_ema` plus two causal start-voltage context features: `V_corr_start` and `V_corr_rel_start`. It does not add SOC labels, cumulative Ah, explicit current integration, or future trajectory information.",
        "",
        "## 25C Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Detail",
        "",
        detail.to_markdown(index=False, floatfmt=".3f"),
        "",
    ]
    (OUT_DIR / f"{PREFIX}_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    details = []
    for fold in FOLDS:
        details.extend(metric_rows(fold, load_prediction(fold)))
    detail = pd.DataFrame(details)
    summary = (
        detail[np.isclose(detail["temperature_C"], 25.0)]
        .groupby("protocol", sort=False)
        .agg(
            mean_25C_MAE_pct=("MAE_pct", "mean"),
            max_25C_MAE_pct=("MAE_pct", "max"),
            holdouts=("holdout", lambda s: ",".join(s.astype(str))),
            model_family=("model_family", "first"),
            feature_set=("feature_set", "first"),
            target_met=("MAE_pct", lambda s: bool(float(np.mean(s)) <= 0.6)),
        )
        .reset_index()
    )
    detail.to_csv(OUT_DIR / f"{PREFIX}_detail.csv", index=False)
    summary.to_csv(OUT_DIR / f"{PREFIX}_summary.csv", index=False)
    (OUT_DIR / f"{PREFIX}_metadata.json").write_text(
        json.dumps(
            {
                "single_model_only": True,
                "raw_y_pred_only": True,
                "uses_posthoc_ocv_offset": False,
                "uses_model_ensemble": False,
                "uses_transformer_fallback": False,
                "uses_per_holdout_architecture_selection": False,
                "feature_policy": "paper_g4_all_ema_start = paper_g4_all_ema + V_corr_start + V_corr_rel_start",
                "uses_soc_input": False,
                "uses_cumulative_input": False,
                "uses_explicit_current_integration": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(summary, detail)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
