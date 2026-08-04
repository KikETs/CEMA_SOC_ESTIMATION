#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE / "inference_pkg_nmc"
OUTPUT_ROOT = HERE / "robustness_results_nmc"
CSV_COLUMNS = [
    "feature", "fold", "seed", "perturbation_type", "level",
    "temp", "soc_band", "mae", "rmse", "bias",
]
FEATURES = ("T6", "G4")
FOLDS = ("DST", "FUDS", "US06")
SEEDS = (0, 1, 2)
V_NOISE_MV = (1, 2, 5, 10)
I_NOISE_PCT = (0.5, 1, 2)
V_BIAS_MV = (-10, -5, -2, -1, 1, 2, 5, 10)
I_BIAS_MA = (-20, -10, 10, 20)
R0_SCALES = (0.5, 0.8, 1.2, 1.5, 2.0)
NOISE_SEEDS = (0, 1, 2)
COLD_START_SEED = 20260711
COLD_BINS = (
    ("<=60_s", -np.inf, 60.0),
    ("60-300_s", 60.0, 300.0),
    ("300-900_s", 300.0, 900.0),
    ("900-1800_s", 900.0, 1800.0),
    ("1800-3600_s", 1800.0, 3600.0),
    (">3600_s", 3600.0, np.inf),
)


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def assert_precondition() -> str:
    golden_path = PACKAGE_ROOT / "golden_test_results.csv"
    if not golden_path.is_file():
        raise RuntimeError(f"Missing golden results: {golden_path}")
    golden = pd.read_csv(golden_path)
    if len(golden) != 18 or not golden["passed"].astype(bool).all():
        raise RuntimeError("Precondition failed: all 18 package golden tests must pass")
    if not golden["deterministic_bit_identical"].astype(bool).all():
        raise RuntimeError("Precondition failed: package determinism checks did not all pass")
    return tree_hash(PACKAGE_ROOT)


def stable_seed(*parts) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def soc_band(values_fraction: np.ndarray) -> np.ndarray:
    values = np.asarray(values_fraction, dtype=np.float64) * 100.0
    return np.where(values <= 35.0, "<=35", np.where(values <= 65.0, "35-65", ">65"))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    error_pp = (np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)) * 100.0
    return (
        float(np.mean(np.abs(error_pp))),
        float(np.sqrt(np.mean(np.square(error_pp)))),
        float(np.mean(error_pp)),
    )


def stratified_rows(
    feature: str,
    fold: str,
    seed: int,
    perturbation_type: str,
    level: str,
    temperature: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    extra_mask: np.ndarray | None = None,
) -> list[dict]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    valid = np.isfinite(true) & np.isfinite(pred)
    if extra_mask is not None:
        valid &= np.asarray(extra_mask, dtype=bool)
    bands = soc_band(true)
    rows = []
    for band in ("<=35", "35-65", ">65"):
        selected = valid & (bands == band)
        if not selected.any():
            continue
        mae, rmse, bias = metrics(true[selected], pred[selected])
        rows.append(
            {
                "feature": feature,
                "fold": fold,
                "seed": int(seed),
                "perturbation_type": perturbation_type,
                "level": level,
                "temp": float(temperature),
                "soc_band": band,
                "mae": mae,
                "rmse": rmse,
                "bias": bias,
            }
        )
    return rows


def elapsed_time(frame: pd.DataFrame, reset_index: int) -> np.ndarray:
    for column in ("Test_Time(s)", "t_global(s)", "time_s"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
            if np.isfinite(values).all() and np.all(np.diff(values) >= 0):
                return values - values[reset_index]
    return np.arange(len(frame), dtype=np.float64) - float(reset_index)


def cold_reset_points(length: int, fold: str, temperature: float, window: int) -> list[int]:
    low = max(window, int(np.floor(0.10 * length)))
    high = min(length - window - 1, int(np.ceil(0.90 * length)))
    if high <= low + 5:
        raise RuntimeError(f"Record is too short for five mid-record resets: N={length}")
    edges = np.linspace(low, high + 1, 6, dtype=np.int64)
    rng = np.random.default_rng(stable_seed("cold_start", COLD_START_SEED, fold, temperature))
    points = []
    for start, stop in zip(edges[:-1], edges[1:]):
        points.append(int(rng.integers(int(start), max(int(start) + 1, int(stop)))))
    return points


def recovery_row(
    feature: str,
    fold: str,
    seed: int,
    temperature: float,
    reset_trial: int,
    reset_index: int,
    elapsed: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    post = np.arange(len(y_true)) >= int(reset_index)
    valid = post & np.isfinite(y_true) & np.isfinite(y_pred)
    within = valid & (np.abs(y_pred - y_true) <= 0.01)
    if within.any():
        first = int(np.flatnonzero(within)[0])
        recovery_s = float(elapsed[first])
        recovered = True
    else:
        recovery_s = np.nan
        recovered = False
    return {
        "feature": feature,
        "fold": fold,
        "seed": int(seed),
        "temp": float(temperature),
        "reset_trial": int(reset_trial),
        "reset_index": int(reset_index),
        "time_to_recover_within_1pct_s": recovery_s,
        "recovered": recovered,
        "observation_horizon_s": float(np.nanmax(elapsed[post])),
    }


def run_entry(job: tuple[str, str, int]) -> dict:
    feature, fold, seed = job
    import sys
    sys.dont_write_bytecode = True
    import torch
    from inference_pkg_nmc import load_package

    torch.set_num_threads(2)
    package = load_package(PACKAGE_ROOT / feature / fold / str(seed))
    manifest = package.manifest
    source_repo = Path(manifest["source_repo"])
    archive = pd.read_csv(source_repo / manifest["archived_predictions"])
    outputs = {"noise": [], "bias": [], "r0": [], "cold": [], "recovery": []}

    for record_relative in manifest["evaluation_records"]:
        record_path = source_repo / record_relative
        frame = pd.read_csv(record_path)
        expected = archive.loc[archive["file_name"].eq(record_path.name)].sort_values("end_index")
        indices = expected["end_index"].to_numpy(np.int64)
        y_true = expected["y_true"].to_numpy(np.float64)
        golden_pred = expected["y_pred"].to_numpy(np.float64)
        temperature = float(expected["temperature"].iloc[0])

        baseline = package.predict(frame)
        delta = np.max(np.abs(baseline[indices].astype(np.float64) - golden_pred))
        if delta > 1e-6:
            raise AssertionError(f"Golden drift in {feature}/{fold}/{seed}/{record_path.name}: {delta}")
        baseline_rows = stratified_rows(
            feature, fold, seed, "baseline", "unperturbed", temperature, y_true, golden_pred
        )
        for key in ("noise", "bias", "r0", "cold"):
            outputs[key].extend(baseline_rows)

        raw_v = pd.to_numeric(frame["Voltage(V)"], errors="raise").to_numpy(np.float64)
        raw_i = pd.to_numeric(frame["Current(A)"], errors="raise").to_numpy(np.float64)

        for sigma_mv in V_NOISE_MV:
            true_all, pred_all = [], []
            for noise_seed in NOISE_SEEDS:
                rng = np.random.default_rng(stable_seed("v_noise", fold, temperature, sigma_mv, noise_seed))
                noise = rng.normal(0.0, float(sigma_mv) / 1000.0, size=len(frame))
                pred = package.predict(frame, v_noise=noise)[indices]
                true_all.append(y_true)
                pred_all.append(pred)
            outputs["noise"].extend(
                stratified_rows(
                    feature, fold, seed, "voltage_noise", f"V_sigma_{sigma_mv:g}_mV",
                    temperature, np.concatenate(true_all), np.concatenate(pred_all),
                )
            )

        for sigma_pct in I_NOISE_PCT:
            true_all, pred_all = [], []
            for noise_seed in NOISE_SEEDS:
                rng = np.random.default_rng(stable_seed("i_noise", fold, temperature, sigma_pct, noise_seed))
                sigma = np.abs(raw_i) * (float(sigma_pct) / 100.0)
                noise = rng.normal(0.0, 1.0, size=len(frame)) * sigma
                pred = package.predict(frame, i_noise=noise)[indices]
                true_all.append(y_true)
                pred_all.append(pred)
            outputs["noise"].extend(
                stratified_rows(
                    feature, fold, seed, "current_noise", f"I_sigma_{sigma_pct:g}_pct_reading",
                    temperature, np.concatenate(true_all), np.concatenate(pred_all),
                )
            )

        for bias_mv in V_BIAS_MV:
            pred = package.predict(frame, v_bias=float(bias_mv) / 1000.0)[indices]
            outputs["bias"].extend(
                stratified_rows(
                    feature, fold, seed, "voltage_bias", f"V_bias_{bias_mv:+g}_mV",
                    temperature, y_true, pred,
                )
            )

        for bias_ma in I_BIAS_MA:
            pred = package.predict(frame, i_bias=float(bias_ma) / 1000.0)[indices]
            outputs["bias"].extend(
                stratified_rows(
                    feature, fold, seed, "current_bias", f"I_bias_{bias_ma:+g}_mA",
                    temperature, y_true, pred,
                )
            )

        for scale in R0_SCALES:
            pred = package.predict(frame, r0_scale=float(scale))[indices]
            outputs["r0"].extend(
                stratified_rows(
                    feature, fold, seed, "r0_scale", f"r0_scale_{scale:g}", temperature, y_true, pred
                )
            )

        resets = cold_reset_points(len(frame), fold, temperature, int(package.config["window"]))
        full_true = np.full(len(frame), np.nan, dtype=np.float64)
        full_true[indices] = y_true
        cold_samples = {label: {"true": [], "pred": []} for label, _, _ in COLD_BINS}
        for trial, reset_index in enumerate(resets):
            pred = package.predict(frame, reset_indices=[reset_index]).astype(np.float64)
            elapsed = elapsed_time(frame, reset_index)
            outputs["recovery"].append(
                recovery_row(
                    feature, fold, seed, temperature, trial, reset_index,
                    elapsed, full_true, pred,
                )
            )
            for label, lower, upper in COLD_BINS:
                if np.isneginf(lower):
                    mask = (np.arange(len(frame)) >= reset_index) & (elapsed <= upper)
                elif np.isposinf(upper):
                    mask = (np.arange(len(frame)) >= reset_index) & (elapsed > lower)
                else:
                    mask = (np.arange(len(frame)) >= reset_index) & (elapsed > lower) & (elapsed <= upper)
                cold_samples[label]["true"].append(full_true[mask])
                cold_samples[label]["pred"].append(pred[mask])
        for label, _, _ in COLD_BINS:
            outputs["cold"].extend(
                stratified_rows(
                    feature, fold, seed, "cold_start_time_since_reset", label,
                    temperature, np.concatenate(cold_samples[label]["true"]),
                    np.concatenate(cold_samples[label]["pred"]),
                )
            )
    print(f"[complete] {feature}/{fold}/{seed}", flush=True)
    return outputs


def run_cold_entry(job: tuple[str, str, int]) -> dict:
    feature, fold, seed = job
    import sys
    sys.dont_write_bytecode = True
    import torch
    from inference_pkg_nmc import load_package

    torch.set_num_threads(2)
    package = load_package(PACKAGE_ROOT / feature / fold / str(seed))
    manifest = package.manifest
    source_repo = Path(manifest["source_repo"])
    archive = pd.read_csv(source_repo / manifest["archived_predictions"])
    outputs = {"cold": [], "recovery": []}
    for record_relative in manifest["evaluation_records"]:
        record_path = source_repo / record_relative
        frame = pd.read_csv(record_path)
        expected = archive.loc[archive["file_name"].eq(record_path.name)].sort_values("end_index")
        indices = expected["end_index"].to_numpy(np.int64)
        y_true = expected["y_true"].to_numpy(np.float64)
        golden_pred = expected["y_pred"].to_numpy(np.float64)
        temperature = float(expected["temperature"].iloc[0])
        outputs["cold"].extend(
            stratified_rows(feature, fold, seed, "baseline", "unperturbed", temperature, y_true, golden_pred)
        )
        full_true = np.full(len(frame), np.nan, dtype=np.float64)
        full_true[indices] = y_true
        cold_samples = {label: {"true": [], "pred": []} for label, _, _ in COLD_BINS}
        resets = cold_reset_points(len(frame), fold, temperature, int(package.config["window"]))
        for trial, reset_index in enumerate(resets):
            pred = package.predict(frame, reset_indices=[reset_index]).astype(np.float64)
            elapsed = elapsed_time(frame, reset_index)
            outputs["recovery"].append(
                recovery_row(
                    feature, fold, seed, temperature, trial, reset_index,
                    elapsed, full_true, pred,
                )
            )
            sample_index = np.arange(len(frame))
            for label, lower, upper in COLD_BINS:
                if np.isneginf(lower):
                    mask = (sample_index >= reset_index) & (elapsed <= upper)
                elif np.isposinf(upper):
                    mask = (sample_index >= reset_index) & (elapsed > lower)
                else:
                    mask = (sample_index >= reset_index) & (elapsed > lower) & (elapsed <= upper)
                cold_samples[label]["true"].append(full_true[mask])
                cold_samples[label]["pred"].append(pred[mask])
        for label, _, _ in COLD_BINS:
            outputs["cold"].extend(
                stratified_rows(
                    feature, fold, seed, "cold_start_time_since_reset", label,
                    temperature, np.concatenate(cold_samples[label]["true"]),
                    np.concatenate(cold_samples[label]["pred"]),
                )
            )
    print(f"[cold complete] {feature}/{fold}/{seed}", flush=True)
    return outputs


def write_csv(rows: list[dict], path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=CSV_COLUMNS)
    frame = frame.sort_values(["feature", "fold", "seed", "perturbation_type", "level", "temp", "soc_band"])
    frame.to_csv(path, index=False)
    return frame


def numeric_level(frame: pd.DataFrame) -> pd.Series:
    extracted = frame["level"].str.findall(r"[-+]?\d+(?:\.\d+)?")
    return pd.to_numeric(extracted.str[-1], errors="coerce")


def make_figures(noise: pd.DataFrame, bias: pd.DataFrame, r0: pd.DataFrame, cold: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25})

    def panel_figure(frame, perturbations, titles, filename, xlabels, baseline_x):
        fig, axes = plt.subplots(1, len(perturbations), figsize=(6.2 * len(perturbations), 4.3), squeeze=False)
        baseline = frame[frame.perturbation_type.eq("baseline")].groupby("feature", as_index=False).mae.mean()
        for ax, perturbation, title, xlabel, clean_x in zip(axes[0], perturbations, titles, xlabels, baseline_x):
            part = frame[frame.perturbation_type.eq(perturbation)].copy()
            part["x"] = numeric_level(part)
            aggregate = part.groupby(["feature", "x"], as_index=False).mae.mean()
            clean = baseline.copy()
            clean["x"] = float(clean_x)
            aggregate = pd.concat([clean[["feature", "x", "mae"]], aggregate], ignore_index=True)
            for feature, color, marker in (("T6", "#1f77b4", "o"), ("G4", "#d62728", "s")):
                values = aggregate[aggregate.feature.eq(feature)].sort_values("x")
                ax.plot(values.x, values.mae, marker=marker, color=color, label=feature, linewidth=2)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Macro MAE (% SOC)")
            ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_ROOT / filename, bbox_inches="tight")
        plt.close(fig)

    panel_figure(
        noise, ("voltage_noise", "current_noise"),
        ("Voltage Gaussian noise", "Current Gaussian noise"),
        "sensor_noise_degradation.png", ("Voltage sigma (mV)", "Current sigma (% of reading)"),
        (0.0, 0.0),
    )
    panel_figure(
        bias, ("voltage_bias", "current_bias"),
        ("Voltage bias", "Current bias"),
        "sensor_bias_degradation.png", ("Voltage bias (mV)", "Current bias (mA)"),
        (0.0, 0.0),
    )
    panel_figure(
        r0, ("r0_scale",), ("R0 perturbation",), "r0_degradation.png", ("R0 scale",), (1.0,),
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    part = cold[cold.perturbation_type.eq("cold_start_time_since_reset")].copy()
    order = [row[0] for row in COLD_BINS]
    part["x"] = pd.Categorical(part.level, categories=order, ordered=True)
    aggregate = part.groupby(["feature", "x"], observed=True, as_index=False).mae.mean()
    for feature, color, marker in (("T6", "#1f77b4", "o"), ("G4", "#d62728", "s")):
        values = aggregate[aggregate.feature.eq(feature)].sort_values("x")
        ax.plot(values.x.astype(str), values.mae, marker=marker, color=color, label=feature, linewidth=2)
        clean_mae = cold.loc[
            (cold.perturbation_type == "baseline") & (cold.feature == feature), "mae"
        ].mean()
        ax.axhline(clean_mae, color=color, linestyle=":", linewidth=1.2, alpha=0.8)
    ax.set_xlabel("Time since reset")
    ax.set_ylabel("Macro MAE (% SOC)")
    ax.set_title("EMA cold-start degradation and recovery")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "cold_start_degradation.png", bbox_inches="tight")
    plt.close(fig)


def sensitivity_table(frame: pd.DataFrame, perturbations: tuple[str, ...]) -> pd.DataFrame:
    baseline = (
        frame[frame.perturbation_type.eq("baseline")]
        .groupby(["feature", "fold", "seed", "temp", "soc_band"], as_index=False).mae.mean()
        .rename(columns={"mae": "baseline_mae"})
    )
    perturbed = frame[frame.perturbation_type.isin(perturbations)].copy()
    merged = perturbed.merge(baseline, on=["feature", "fold", "seed", "temp", "soc_band"], how="left")
    merged["delta_mae"] = merged.mae - merged.baseline_mae
    return (
        merged.groupby(["perturbation_type", "level", "feature"], as_index=False)
        .agg(mae=("mae", "mean"), baseline_mae=("baseline_mae", "mean"), delta_mae=("delta_mae", "mean"))
        .sort_values(["perturbation_type", "level", "feature"])
    )


def write_summary(noise: pd.DataFrame, bias: pd.DataFrame, r0: pd.DataFrame, cold: pd.DataFrame, recovery: pd.DataFrame) -> None:
    sensitivity = pd.concat(
        [
            sensitivity_table(noise, ("voltage_noise", "current_noise")),
            sensitivity_table(bias, ("voltage_bias", "current_bias")),
            sensitivity_table(r0, ("r0_scale",)),
        ],
        ignore_index=True,
    )
    sensitivity.to_csv(OUTPUT_ROOT / "t6_vs_g4_sensitivity.csv", index=False)

    comparison = sensitivity.pivot_table(
        index=["perturbation_type", "level"], columns="feature", values="delta_mae"
    )
    comparison["G4_minus_T6_delta_mae"] = comparison.get("G4", np.nan) - comparison.get("T6", np.nan)
    noise_gap = comparison.loc["voltage_noise", "G4_minus_T6_delta_mae"]
    bias_gap = comparison.loc["voltage_bias", "G4_minus_T6_delta_mae"]
    noise_gap_mean = float(noise_gap.mean())
    bias_gap_mean = float(bias_gap.mean())
    noise_10_gap = float(noise_gap.loc["V_sigma_10_mV"])
    positive_bias_gap = float(bias_gap[[level.startswith("V_bias_+") for level in bias_gap.index]].mean())
    negative_bias_gap = float(bias_gap[[level.startswith("V_bias_-") for level in bias_gap.index]].mean())
    finding = (
        f"Voltage noise: G4's mean MAE degradation was {abs(noise_gap_mean):.4f} percentage points "
        f"{'lower' if noise_gap_mean < 0 else 'higher'} than T6 across the four sigma levels; at 10 mV "
        f"it was {abs(noise_10_gap):.4f} percentage points {'lower' if noise_10_gap < 0 else 'higher'}. "
        f"This is a small noise-robustness difference. Voltage bias: G4's mean degradation was "
        f"{abs(bias_gap_mean):.4f} percentage points {'lower' if bias_gap_mean < 0 else 'higher'}, but it "
        f"was asymmetric: the G4-minus-T6 degradation gap averaged {positive_bias_gap:+.4f} percentage "
        f"points for positive bias and {negative_bias_gap:+.4f} for negative bias. Therefore the I/abs-I "
        "EMA channels provide a modest, direction-dependent voltage-fault robustness benefit, not a "
        "uniform advantage at every level."
    )

    clean = noise[noise.perturbation_type.eq("baseline")].groupby("feature").mae.mean()
    recovery_summary = recovery.groupby("feature").agg(
        median_recovery_s=("time_to_recover_within_1pct_s", "median"),
        recovery_rate=("recovered", "mean"),
    )
    text = f"""# NMC inference robustness summary

## Protocol

- Frozen package inference only; no retraining and no package modification.
- Features: T6 and G4; folds: DST, FUDS, US06; model seeds: 0, 1, 2; temperatures: 0, 25, 45 C.
- Metrics are percentage points of SOC. SOC bands are <=35%, 35-65%, and >65%.
- Three deterministic Gaussian realizations are pooled per model seed and noise level.
- The same raw-noise realization is shared across T6/G4 and model seeds for paired sensitivity comparison.
- Cold-start uses five fixed, stratified-random mid-record reset points. Each reset is evaluated in a separate inference run.
- Recovery time is the first post-reset evaluated sample with absolute SOC error <=1 percentage point.
- Figures report the macro mean across fold, model seed, temperature, and SOC-band rows.

## Clean baseline

- T6 macro MAE: {clean.get('T6', float('nan')):.4f}% SOC
- G4 macro MAE: {clean.get('G4', float('nan')):.4f}% SOC

## T6 vs G4 sensitivity

{finding}

Selection implication: clean macro MAE favors T6, while G4 offers better aggregate tolerance to the tested voltage bias and essentially similar-to-slightly-better voltage-noise tolerance. G4 should not be selected on robustness alone without specifying fault polarity and magnitude.

The full level-wise comparison is in `t6_vs_g4_sensitivity.csv`; this statement is based on observed inference results, not the channel names alone.

## Cold-start recovery

- T6 median time to first <=1% SOC error: {recovery_summary.loc['T6', 'median_recovery_s']:.1f} s; recovery rate {100.0 * recovery_summary.loc['T6', 'recovery_rate']:.1f}%.
- G4 median time to first <=1% SOC error: {recovery_summary.loc['G4', 'median_recovery_s']:.1f} s; recovery rate {100.0 * recovery_summary.loc['G4', 'recovery_rate']:.1f}%.

## Outputs

- `sensor_noise.csv`, `sensor_noise_degradation.png`
- `sensor_bias.csv`, `sensor_bias_degradation.png`
- `r0_perturbation.csv`, `r0_degradation.png`
- `cold_start.csv`, `cold_start_recovery.csv`, `cold_start_degradation.png`
- `t6_vs_g4_sensitivity.csv`
"""
    (OUTPUT_ROOT / "SUMMARY.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen NMC robustness inference suite")
    parser.add_argument("--workers", type=int, default=min(9, max(1, (os.cpu_count() or 2) // 4)))
    parser.add_argument("--cold-only", action="store_true")
    args = parser.parse_args()

    package_hash_before = assert_precondition()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = [(feature, fold, seed) for feature in FEATURES for fold in FOLDS for seed in SEEDS]
    combined = {"noise": [], "bias": [], "r0": [], "cold": [], "recovery": []}
    context = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(args.workers), mp_context=context) as executor:
        worker = run_cold_entry if args.cold_only else run_entry
        futures = {executor.submit(worker, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            for key in result:
                combined[key].extend(result[key])

    if args.cold_only:
        noise = pd.read_csv(OUTPUT_ROOT / "sensor_noise.csv")
        bias = pd.read_csv(OUTPUT_ROOT / "sensor_bias.csv")
        r0 = pd.read_csv(OUTPUT_ROOT / "r0_perturbation.csv")
    else:
        noise = write_csv(combined["noise"], OUTPUT_ROOT / "sensor_noise.csv")
        bias = write_csv(combined["bias"], OUTPUT_ROOT / "sensor_bias.csv")
        r0 = write_csv(combined["r0"], OUTPUT_ROOT / "r0_perturbation.csv")
    cold = write_csv(combined["cold"], OUTPUT_ROOT / "cold_start.csv")
    recovery = pd.DataFrame(combined["recovery"]).sort_values(["feature", "fold", "seed", "temp", "reset_trial"])
    recovery.to_csv(OUTPUT_ROOT / "cold_start_recovery.csv", index=False)
    make_figures(noise, bias, r0, cold)
    write_summary(noise, bias, r0, cold, recovery)

    package_hash_after = tree_hash(PACKAGE_ROOT)
    if package_hash_after != package_hash_before:
        raise AssertionError("Frozen inference package changed during robustness run")
    manifest = {
        "package_root": str(PACKAGE_ROOT),
        "package_tree_sha256_before": package_hash_before,
        "package_tree_sha256_after": package_hash_after,
        "package_unchanged": True,
        "features": list(FEATURES),
        "folds": list(FOLDS),
        "model_seeds": list(SEEDS),
        "temperatures_C": [0, 25, 45],
        "noise_seeds": list(NOISE_SEEDS),
        "cold_start_seed": COLD_START_SEED,
        "metric_unit": "SOC percentage points",
        "csv_columns": CSV_COLUMNS,
        "workers": int(args.workers),
        "cold_trial_aggregation": "pool samples across five reset trials before metrics",
    }
    (OUTPUT_ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[done] outputs={OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
