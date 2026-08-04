from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import error_metrics


def _root_for_model(model_name: str, protocol: dict) -> Path:
    key = "tcn_root" if "TCN" in model_name else "ema_root"
    return Path(protocol["neural_results"][key])


def load_neural_predictions(model_name: str, fold: str, protocol: dict) -> pd.DataFrame:
    pattern = protocol["neural_results"]["models"][model_name].format(holdout=fold.lower())
    paths = sorted(_root_for_model(model_name, protocol).glob(pattern))
    if len(paths) != 3:
        raise FileNotFoundError(f"Expected three seed prediction files for {model_name}/{fold}, got {len(paths)}: {pattern}")
    frames = []
    for seed, path in enumerate(paths):
        frame = pd.read_csv(path)
        keep = ["file_name", "trajectory_id", "end_index", "temperature", "y_true", "y_pred"]
        missing = [column for column in keep if column not in frame.columns]
        if missing:
            raise ValueError(f"{path} missing {missing}")
        frame = frame[keep].copy()
        frame["seed"] = seed
        frame["source_file"] = str(path.resolve())
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def neural_metric_rows(model_name: str, fold: str, predictions: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    rows = []
    for temperature, temperature_frame in predictions.groupby("temperature"):
        seed_metrics = []
        for seed, seed_frame in temperature_frame.groupby("seed"):
            metrics = error_metrics(seed_frame["y_true"].to_numpy(), seed_frame["y_pred"].to_numpy())
            seed_metrics.append(metrics)
        aggregate = {
            key: float(np.mean([metric[key] for metric in seed_metrics]))
            for key in seed_metrics[0]
            if key != "n"
        }
        aggregate["n"] = int(seed_metrics[0]["n"])
        rows.append({"method": model_name, "fold": fold, "temperature_C": float(temperature), "condition": "native", **aggregate})
    pointwise = (
        predictions.assign(
            error_pct=lambda frame: (frame["y_pred"] - frame["y_true"]) * 100.0,
            abs_error_pct=lambda frame: np.abs(frame["y_pred"] - frame["y_true"]) * 100.0,
        )
        .groupby(["file_name", "trajectory_id", "end_index", "temperature"], as_index=False)
        .agg(
            reference_soc=("y_true", "first"),
            neural_prediction_mean=("y_pred", "mean"),
            neural_error_mean_pct=("error_pct", "mean"),
            neural_abs_error_pct=("abs_error_pct", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    return rows, pointwise
