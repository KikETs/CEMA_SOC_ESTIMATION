from __future__ import annotations

import numpy as np


def error_metrics(reference_soc: np.ndarray, predicted_soc: np.ndarray, mask: np.ndarray | None = None) -> dict:
    reference = np.asarray(reference_soc, dtype=np.float64)
    prediction = np.asarray(predicted_soc, dtype=np.float64)
    selected = np.isfinite(reference) & np.isfinite(prediction)
    if mask is not None:
        selected &= np.asarray(mask, dtype=bool)
    error_pct = (prediction[selected] - reference[selected]) * 100.0
    absolute = np.abs(error_pct)
    if len(absolute) == 0:
        return {
            "n": 0,
            "mae_pct": np.nan,
            "rmse_pct": np.nan,
            "p95_ae_pct": np.nan,
            "max_ae_pct": np.nan,
            "bias_pct": np.nan,
            "ae_gt_2pct_fraction": np.nan,
            "ae_gt_3pct_fraction": np.nan,
        }
    return {
        "n": int(len(absolute)),
        "mae_pct": float(np.mean(absolute)),
        "rmse_pct": float(np.sqrt(np.mean(np.square(error_pct)))),
        "p95_ae_pct": float(np.percentile(absolute, 95)),
        "max_ae_pct": float(np.max(absolute)),
        "bias_pct": float(np.mean(error_pct)),
        "ae_gt_2pct_fraction": float(np.mean(absolute > 2.0)),
        "ae_gt_3pct_fraction": float(np.mean(absolute > 3.0)),
    }


def convergence_metrics(
    elapsed_s: np.ndarray,
    reference_soc: np.ndarray,
    predicted_soc: np.ndarray,
    start_index: int,
    threshold_pct: float = 2.0,
    sustained_duration_s: float = 60.0,
) -> dict:
    elapsed = np.asarray(elapsed_s, dtype=np.float64)
    absolute = np.abs(np.asarray(predicted_soc) - np.asarray(reference_soc)) * 100.0
    valid = np.isfinite(absolute)
    valid[: int(start_index)] = False
    below = valid & (absolute < float(threshold_pct))
    indices = np.flatnonzero(below)
    first_entry = float(elapsed[indices[0]] - elapsed[start_index]) if len(indices) else np.nan
    sustained = np.nan
    for begin in indices:
        end = begin
        while end + 1 < len(below) and below[end + 1]:
            end += 1
            if elapsed[end] - elapsed[begin] >= float(sustained_duration_s):
                sustained = float(elapsed[begin] - elapsed[start_index])
                break
        if np.isfinite(sustained):
            break
    return {
        "time_to_ae_lt_2pct_s": first_entry,
        "time_to_sustained_ae_lt_2pct_60s_s": sustained,
    }


def bootstrap_paired_mean_difference(
    differences: dict[tuple[str, float], np.ndarray],
    seed: int,
    replicates: int,
    block_length: int = 60,
) -> dict:
    rng = np.random.default_rng(int(seed))
    keys = sorted(differences)
    observed = float(np.mean(np.concatenate([differences[key] for key in keys])))
    samples = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        chunks = []
        for key in keys:
            values = np.asarray(differences[key], dtype=np.float64)
            block = min(int(block_length), len(values))
            starts = rng.integers(0, len(values), int(np.ceil(len(values) / block)))
            indices = np.concatenate([(np.arange(block) + start) % len(values) for start in starts])[: len(values)]
            chunks.append(values[indices])
        samples[replicate] = float(np.mean(np.concatenate(chunks)))
    return {
        "mean_difference_pct_point": observed,
        "ci95_low_pct_point": float(np.percentile(samples, 2.5)),
        "ci95_high_pct_point": float(np.percentile(samples, 97.5)),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "bootstrap_method": "fold_temperature_stratified_circular_moving_block",
        "bootstrap_block_length_samples": int(block_length),
        "positive_means_kf_worse": True,
    }
