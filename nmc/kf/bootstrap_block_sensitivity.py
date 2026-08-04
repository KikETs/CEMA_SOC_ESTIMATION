#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from run_all_lopo import ROOT, load_yaml
from src.data_io import discover_dynamic_files, load_trajectory
from src.neural_results import load_neural_predictions


BOOTSTRAP_SEED = 20260711
BOOTSTRAP_REPLICATES = 10_000
BLOCK_LENGTHS = [60, 300, 900]
KF_METHODS = {
    "1RC-EKF": "1rc_ekf",
    "2RC-EKF": "2rc_ekf",
    "adaptive 2RC-EKF": "adaptive_2rc_ekf",
    "2RC-UKF": "2rc_ukf",
}
OUTPUT = ROOT / "results" / "bootstrap_sensitivity.csv"
REPORT = ROOT / "report.md"
START_MARKER = "<!-- BOOTSTRAP_SENSITIVITY_START -->"
END_MARKER = "<!-- BOOTSTRAP_SENSITIVITY_END -->"


@dataclass(frozen=True)
class RecordDifference:
    record_id: str
    fold: str
    temperature_C: float
    values: np.ndarray
    acf_lag01: int


def _eval_mask(frame: pd.DataFrame) -> np.ndarray:
    values = frame["eval_mask"]
    if values.dtype == bool:
        return values.to_numpy()
    normalized = values.astype(str).str.lower()
    if not normalized.isin(["true", "false"]).all():
        raise RuntimeError("KF eval_mask contains non-boolean values")
    return normalized.eq("true").to_numpy()


def assert_record_timestamp_alignment(
    proposed: pd.DataFrame,
    kf: pd.DataFrame,
    record_id: str,
    method: str,
    seed: int,
) -> None:
    key = ["record_id", "timestamp_s"]
    if proposed.duplicated(key).any() or kf.duplicated(key).any():
        raise RuntimeError(
            f"Duplicate record_id+timestamp before bootstrap: pair={method}, record={record_id}, seed={seed}"
        )
    left = proposed.sort_values(key).reset_index(drop=True)
    right = kf.sort_values(key).reset_index(drop=True)
    if len(left) != len(right):
        raise RuntimeError(
            f"Per-sample length mismatch: pair={method}, record={record_id}, seed={seed}, "
            f"proposed={len(left)}, kf={len(right)}"
        )
    if not np.array_equal(left["record_id"].to_numpy(), right["record_id"].to_numpy()):
        raise RuntimeError(f"Record-id mismatch: pair={method}, record={record_id}, seed={seed}")
    left_time = left["timestamp_s"].to_numpy(np.float64)
    right_time = right["timestamp_s"].to_numpy(np.float64)
    if not np.allclose(left_time, right_time, rtol=0.0, atol=1e-9):
        maximum = float(np.max(np.abs(left_time - right_time)))
        raise RuntimeError(
            f"Timestamp mismatch: pair={method}, record={record_id}, seed={seed}, max_abs={maximum:.3e}s"
        )
    if not np.array_equal(left["end_index"].to_numpy(np.int64), right["end_index"].to_numpy(np.int64)):
        raise RuntimeError(f"Endpoint-index mismatch: pair={method}, record={record_id}, seed={seed}")
    if not np.allclose(
        left["reference_soc"].to_numpy(np.float64),
        right["reference_soc"].to_numpy(np.float64),
        rtol=0.0,
        atol=1e-7,
    ):
        raise RuntimeError(f"Reference-SOC mismatch: pair={method}, record={record_id}, seed={seed}")


def acf_lag_below(values: np.ndarray, threshold: float = 0.1) -> int:
    series = np.asarray(values, dtype=np.float64)
    if len(series) < 2 or not np.isfinite(series).all():
        raise ValueError("ACF requires at least two finite samples")
    centered = series - float(np.mean(series))
    variance = float(np.dot(centered, centered) / len(centered))
    if variance <= np.finfo(np.float64).eps:
        return 1
    fft_length = 1 << (2 * len(centered) - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_length)
    raw = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_length)[: len(centered)].real
    autocovariance = raw / np.arange(len(centered), 0, -1, dtype=np.float64)
    acf = autocovariance / autocovariance[0]
    hits = np.flatnonzero(np.abs(acf[1:]) < float(threshold))
    return int(hits[0] + 1) if len(hits) else int(len(series) - 1)


def circular_block_bootstrap_sums(
    values: np.ndarray,
    block_length: int,
    replicates: int,
    rng: np.random.Generator,
    chunk_size: int = 512,
) -> np.ndarray:
    series = np.asarray(values, dtype=np.float64)
    count = len(series)
    block = int(block_length)
    if count == 0 or block <= 0 or block > count:
        raise ValueError(f"Invalid circular block request: n={count}, block={block}")
    doubled = np.concatenate([series, series])
    prefix = np.concatenate([[0.0], np.cumsum(doubled, dtype=np.float64)])
    starts = np.arange(count, dtype=np.int64)
    full_block_sums = prefix[starts + block] - prefix[starts]
    full_count, remainder = divmod(count, block)
    output = np.empty(int(replicates), dtype=np.float64)
    for begin in range(0, int(replicates), int(chunk_size)):
        end = min(begin + int(chunk_size), int(replicates))
        batch = end - begin
        draw_count = full_count + int(remainder > 0)
        sampled_starts = rng.integers(0, count, size=(batch, draw_count))
        totals = full_block_sums[sampled_starts[:, :full_count]].sum(axis=1)
        if remainder:
            partial = prefix[starts + remainder] - prefix[starts]
            totals += partial[sampled_starts[:, -1]]
        output[begin:end] = totals
    return output


def load_record_differences(protocol: dict) -> dict[str, list[RecordDifference]]:
    trajectories = {}
    for path in discover_dynamic_files(
        Path(protocol["data_root"]), protocol["profiles"], protocol["temperatures_C"]
    ):
        trajectory = load_trajectory(path, float(protocol["current_convention"]["multiplier"]))
        trajectories[(trajectory.profile, trajectory.temperature_C)] = trajectory

    output = {method: [] for method in KF_METHODS}
    for fold in protocol["folds"]:
        proposed_all = load_neural_predictions("Proposed_EMA_MLP_G4_residual", fold, protocol)
        if proposed_all["seed"].nunique() != 3:
            raise RuntimeError(f"Expected three proposed seeds for fold={fold}")
        for temperature in map(float, protocol["temperatures_C"]):
            trajectory = trajectories[(fold, temperature)]
            file_name = trajectory.file_name
            record_id = f"{fold}|{temperature:g}C|{file_name}"
            proposed_temperature = proposed_all[
                np.isclose(proposed_all["temperature"].to_numpy(np.float64), temperature)
            ].copy()
            if proposed_temperature["file_name"].nunique() != 1 or proposed_temperature["file_name"].iloc[0] != file_name:
                raise RuntimeError(f"Proposed record mismatch for {record_id}")
            if not proposed_temperature["trajectory_id"].eq(Path(file_name).stem).all():
                raise RuntimeError(f"Proposed trajectory_id mismatch for {record_id}")
            endpoint = proposed_temperature["end_index"].to_numpy(np.int64)
            if endpoint.min() < 0 or endpoint.max() >= len(trajectory.time_s):
                raise RuntimeError(f"Proposed endpoint out of source bounds for {record_id}")
            proposed_temperature["record_id"] = record_id
            proposed_temperature["timestamp_s"] = trajectory.elapsed_s[endpoint]
            proposed_temperature["reference_soc"] = proposed_temperature["y_true"].to_numpy(np.float64)
            proposed_temperature["proposed_abs_error_pct"] = (
                np.abs(proposed_temperature["y_pred"] - proposed_temperature["y_true"]) * 100.0
            )

            for method, file_token in KF_METHODS.items():
                prediction_path = (
                    ROOT
                    / "results"
                    / "predictions"
                    / f"{fold.lower()}_{temperature:g}C_{file_token}_oracle.csv.gz"
                )
                if not prediction_path.exists():
                    raise FileNotFoundError(prediction_path)
                kf = pd.read_csv(prediction_path)
                kf = kf.loc[_eval_mask(kf)].copy()
                kf["record_id"] = record_id
                kf["timestamp_s"] = pd.to_numeric(kf["elapsed_s"], errors="raise")
                kf["reference_soc"] = pd.to_numeric(kf["reference_soc"], errors="raise")
                kf = kf.sort_values("end_index").reset_index(drop=True)

                seed_errors = []
                for seed, proposed_seed in proposed_temperature.groupby("seed", sort=True):
                    proposed_seed = proposed_seed.sort_values("end_index").reset_index(drop=True)
                    assert_record_timestamp_alignment(
                        proposed_seed,
                        kf,
                        record_id,
                        method,
                        int(seed),
                    )
                    seed_errors.append(proposed_seed["proposed_abs_error_pct"].to_numpy(np.float64))
                proposed_mean_abs_error = np.mean(np.stack(seed_errors, axis=0), axis=0)
                difference = kf["abs_error_pct"].to_numpy(np.float64) - proposed_mean_abs_error
                if not np.isfinite(difference).all():
                    raise RuntimeError(f"Non-finite paired difference for {method}/{record_id}")
                output[method].append(
                    RecordDifference(
                        record_id=record_id,
                        fold=fold,
                        temperature_C=temperature,
                        values=difference,
                        acf_lag01=acf_lag_below(difference),
                    )
                )
    for method, records in output.items():
        if len(records) != 9:
            raise RuntimeError(f"Expected nine aligned records for {method}, got {len(records)}")
    return output


def run_sensitivity(records_by_method: dict[str, list[RecordDifference]]) -> pd.DataFrame:
    rows = []
    for pair_index, (method, records) in enumerate(records_by_method.items()):
        pair = f"proposed_vs_{method}"
        scope_records = {"pooled": records}
        for fold in ["US06", "DST", "FUDS"]:
            scope_records[f"fold:{fold}"] = [record for record in records if record.fold == fold]
        for block_length in BLOCK_LENGTHS:
            bootstrap_sums = {}
            for record_index, record in enumerate(records):
                seed = np.random.SeedSequence(
                    [BOOTSTRAP_SEED, pair_index, int(block_length), record_index]
                )
                bootstrap_sums[record.record_id] = circular_block_bootstrap_sums(
                    record.values,
                    int(block_length),
                    BOOTSTRAP_REPLICATES,
                    np.random.default_rng(seed),
                )
            for scope, selected in scope_records.items():
                sample_count = sum(len(record.values) for record in selected)
                bootstrap_mean = (
                    sum(bootstrap_sums[record.record_id] for record in selected) / sample_count
                )
                observed = float(
                    sum(float(np.sum(record.values)) for record in selected) / sample_count
                )
                rows.append(
                    {
                        "pair": pair,
                        "scope": scope,
                        "block_len": int(block_length),
                        "mean_diff": observed,
                        "ci_lo": float(np.percentile(bootstrap_mean, 2.5)),
                        "ci_hi": float(np.percentile(bootstrap_mean, 97.5)),
                        "acf_lag01_median": float(
                            np.median([record.acf_lag01 for record in selected])
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["pair", "scope", "block_len"]).reset_index(drop=True)


def update_report(frame: pd.DataFrame) -> None:
    summarized = []
    for (pair, scope), group in frame.groupby(["pair", "scope"], sort=True):
        excludes = (group["ci_lo"] > 0.0) | (group["ci_hi"] < 0.0)
        summarized.append(
            {
                "pair": pair,
                "scope": scope,
                "all_60_300_900_exclude_0": bool(excludes.all()),
                "lowest_ci_lo": float(group["ci_lo"].min()),
                "highest_ci_hi": float(group["ci_hi"].max()),
                "acf_lag01_median": float(group["acf_lag01_median"].iloc[0]),
            }
        )
    summary = pd.DataFrame(summarized)
    every = bool(summary["all_60_300_900_exclude_0"].all())
    display = summary.copy()
    for column in ["lowest_ci_lo", "highest_ci_hi", "acf_lag01_median"]:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    section = "\n".join(
        [
            START_MARKER,
            "## Paired block-bootstrap sensitivity",
            "",
            "Paired differences are defined as `KF AE - proposed 3-seed mean AE` in percentage points; positive values favor proposed. Before resampling, every proposed seed and KF file was required to match exactly by record id and endpoint and within 1e-9 s by timestamp.",
            "",
            f"Circular moving blocks were sampled independently within each record only, using block lengths 60/300/900, B={BOOTSTRAP_REPLICATES:,}, and deterministic seed {BOOTSTRAP_SEED}. Blocks never cross temperature, profile, or fold record boundaries.",
            "",
            f"Every pair/scope CI excludes zero at all three block lengths: **{every}**.",
            "",
            display.to_markdown(index=False),
            "",
            "`acf_lag01_median` is the median, over records in the scope, of the smallest positive lag where the de-meaned paired AE-difference series has `|ACF| < 0.1` using an unbiased FFT autocovariance estimate. These medians are often greater than 900 samples, so stability across the tested 60/300/900 blocks is evidence of sensitivity robustness over this range, not proof that 900 fully spans the record-level dependence scale.",
            END_MARKER,
        ]
    )
    report = REPORT.read_text(encoding="utf-8")
    if START_MARKER in report:
        prefix, remainder = report.split(START_MARKER, 1)
        _, suffix = remainder.split(END_MARKER, 1)
        updated = prefix.rstrip() + "\n\n" + section + suffix
    else:
        updated = report.rstrip() + "\n\n" + section + "\n"
    REPORT.write_text(updated, encoding="utf-8")


def main() -> None:
    protocol = load_yaml(ROOT / "configs" / "protocol.yaml")
    records = load_record_differences(protocol)
    frame = run_sensitivity(records)
    frame.to_csv(OUTPUT, index=False)
    update_report(frame)
    excludes = (frame["ci_lo"] > 0.0) | (frame["ci_hi"] < 0.0)
    print(
        frame.groupby(["pair", "scope"], as_index=False)
        .agg(
            min_ci_lo=("ci_lo", "min"),
            max_ci_hi=("ci_hi", "max"),
            acf_lag01_median=("acf_lag01_median", "first"),
        )
        .to_string(index=False)
    )
    print(f"all_ci_exclude_zero={bool(excludes.all())} rows={len(frame)}")


if __name__ == "__main__":
    main()
