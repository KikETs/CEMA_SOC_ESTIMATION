#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from run_all_lopo import (
    KF_METHODS,
    ROOT,
    add_fold_summary,
    aggregate_neural_results,
    benchmark_proposed_mlp,
    bootstrap_paired_mean_difference,
    build_complexity_table,
    load_yaml,
    make_figures,
    run_method,
    write_readme,
    write_report,
)
from src.data_io import discover_dynamic_files, load_trajectory
from src.ecm_models import ECMParameters
from src.ocv import build_ocv_map


def selected_noise(frame, fold, temperature, method):
    rows = frame[
        (frame["fold"] == fold)
        & np.isclose(frame["temperature_C"], temperature)
        & (frame["method"] == method)
        & (frame["validation_profile"] == "CV_MEAN")
        & frame["selected"].astype(bool)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Selected noise mismatch: {fold}/{temperature}/{method}: {len(rows)}")
    row = rows.iloc[0]
    config = load_yaml(ROOT / "configs" / "ecm.yaml")
    return next(candidate for candidate in config["filter"]["noise_candidates"] if candidate["name"] == row["noise_name"])


def selected_beta(frame, fold, temperature):
    rows = frame[
        (frame["fold"] == fold)
        & np.isclose(frame["temperature_C"], temperature)
        & (frame["method"] == "Adaptive_2RC_EKF")
        & (frame["validation_profile"] == "CV_MEAN")
        & frame["selected"].astype(bool)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Selected adaptive beta mismatch: {fold}/{temperature}: {len(rows)}")
    return float(rows.iloc[0]["adaptive_beta"])


def parameter_map(frame, fold, temperature):
    output = {}
    for order in [1, 2]:
        row = frame[(frame["fold"] == fold) & np.isclose(frame["temperature_C"], temperature) & (frame["order"] == order)].iloc[0]
        output[order] = ECMParameters(
            order=order,
            temperature_C=float(row["temperature_C"]),
            R0_ohm=float(row["R0_ohm"]),
            R1_ohm=float(row["R1_ohm"]),
            tau1_s=float(row["tau1_s"]),
            R2_ohm=float(row["R2_ohm"]),
            tau2_s=float(row["tau2_s"]),
            fit_bounds_name=str(row["fit_bounds_name"]),
            fit_voltage_rmse_V=float(row["fit_voltage_rmse_V"]),
        )
    return output


def main():
    started = perf_counter()
    protocol = load_yaml(ROOT / "configs" / "protocol.yaml")
    ecm_config = load_yaml(ROOT / "configs" / "ecm.yaml")
    ocv_map, _, _ = build_ocv_map(protocol, ecm_config)
    trajectories = {}
    for path in discover_dynamic_files(Path(protocol["data_root"]), protocol["profiles"], protocol["temperatures_C"]):
        trajectory = load_trajectory(path, float(protocol["current_convention"]["multiplier"]))
        trajectories[(trajectory.profile, trajectory.temperature_C)] = trajectory

    prediction_paths = sorted((ROOT / "results" / "predictions").glob("*_oracle.csv.gz"))
    if len(prediction_paths) != 45:
        raise RuntimeError(f"Expected 45 oracle prediction files, found {len(prediction_paths)}")
    kf_predictions = pd.concat([pd.read_csv(path) for path in prediction_paths], ignore_index=True)
    temperature_metrics = pd.read_csv(ROOT / "results" / "temperature_metrics.csv")
    fold_metrics = pd.read_csv(ROOT / "results" / "fold_metrics.csv")
    robustness = pd.read_csv(ROOT / "results" / "initial_soc_robustness.csv")
    params_frame = pd.read_csv(ROOT / "results" / "parameters" / "ecm_parameters.csv")
    noise_frame = pd.read_csv(ROOT / "results" / "parameters" / "filter_noise_selection.csv")
    neural_temperature_rows, _, neural_pointwise, _ = aggregate_neural_results(protocol)
    if len(temperature_metrics) != 45 + len(neural_temperature_rows):
        raise RuntimeError("Existing temperature metrics are incomplete")

    fold_summary = add_fold_summary(fold_metrics)
    fold_summary.to_csv(ROOT / "results" / "fold_summary.csv", index=False)
    best_kf = str(
        fold_summary[fold_summary["method"].isin(["1RC_EKF", "2RC_EKF", "2RC_UKF", "Adaptive_2RC_EKF"])]
        .sort_values("fold_mean_mae_pct")
        .iloc[0]["method"]
    )
    differences = {}
    for fold in protocol["folds"]:
        proposed = neural_pointwise[("Proposed_EMA_MLP_G4_residual", fold)]
        kf = kf_predictions[(kf_predictions["method"] == best_kf) & (kf_predictions["fold"] == fold) & kf_predictions["eval_mask"].astype(bool)]
        for temperature in map(float, protocol["temperatures_C"]):
            left = kf[np.isclose(kf["temperature_C"], temperature)][["file_name", "end_index", "abs_error_pct"]]
            right = proposed[np.isclose(proposed["temperature"], temperature)][["file_name", "end_index", "neural_abs_error_pct"]]
            aligned = left.merge(right, on=["file_name", "end_index"], validate="one_to_one")
            if len(aligned) != len(left) or len(aligned) != len(right):
                raise RuntimeError(f"Endpoint mismatch: {fold}/{temperature:g}C")
            differences[(fold, temperature)] = aligned["abs_error_pct"].to_numpy() - aligned["neural_abs_error_pct"].to_numpy()
    statistic = bootstrap_paired_mean_difference(
        differences,
        int(ecm_config["statistics"]["bootstrap_seed"]),
        int(ecm_config["statistics"]["bootstrap_replicates"]),
        int(ecm_config["statistics"]["bootstrap_block_length_samples"]),
    )
    paired_rows = [
        {
            "fold": fold,
            "temperature_C": temperature,
            "comparison": f"{best_kf} minus Proposed_EMA_MLP_G4_residual absolute error",
            "n_endpoints": len(values),
            "mean_difference_pct_point": float(np.mean(values)),
            "positive_means_kf_worse": True,
        }
        for (fold, temperature), values in sorted(differences.items())
    ]
    pd.DataFrame(paired_rows).to_csv(ROOT / "results" / "paired_fold_temperature.csv", index=False)
    statistics = pd.DataFrame([{
        "comparison": f"{best_kf} minus Proposed_EMA_MLP_G4_residual absolute error",
        "unit": "percentage point",
        **statistic,
    }])
    statistics.to_csv(ROOT / "results" / "paired_bootstrap_statistics.csv", index=False)
    fold_metrics[fold_metrics["method"].isin(["Proposed_EMA_MLP_G4_residual", best_kf])].sort_values(["fold", "method"]).to_csv(
        ROOT / "results" / "direct_comparison.csv", index=False
    )

    latency_samples = {method: [] for method in KF_METHODS}
    for fold in protocol["folds"]:
        for temperature in map(float, protocol["temperatures_C"]):
            trajectory = trajectories[(fold, temperature)]
            parameters = parameter_map(params_frame, fold, temperature)
            noises = {
                method: selected_noise(noise_frame, fold, temperature, method)
                for method in ["1RC_EKF", "2RC_EKF", "2RC_UKF"]
            }
            beta = selected_beta(noise_frame, fold, temperature)
            for method in KF_METHODS:
                result = run_method(
                    method,
                    trajectory,
                    float(trajectory.reference_soc[0]),
                    parameters,
                    noises,
                    ocv_map,
                    float(ocv_map.q_ref_by_temperature[temperature]),
                    ecm_config,
                    beta,
                )
                latency_samples[method].extend(result.latency_ns[result.latency_ns > 0].tolist())

    source_outputs = [ROOT / "results" / "data_inventory.csv", ROOT / "results" / "temperature_metrics.csv"]
    original_runtime_s = max(path.stat().st_mtime for path in source_outputs) - min(path.stat().st_mtime for path in source_outputs)
    finalize_runtime_s = perf_counter() - started
    total_runtime_s = float(original_runtime_s + finalize_runtime_s)
    complexity = build_complexity_table(latency_samples, ocv_map, total_runtime_s, benchmark_proposed_mlp())
    complexity.to_csv(ROOT / "results" / "complexity.csv", index=False)
    failures = robustness[robustness["diverged"].astype(bool)][["method", "fold", "temperature_C", "initial_perturbation_pct_point"]].to_dict("records")
    adaptive_stable = not any(record["method"] == "Adaptive_2RC_EKF" for record in failures)
    make_figures(kf_predictions, neural_pointwise, temperature_metrics, robustness, complexity, fold_summary)
    write_report(fold_summary, temperature_metrics, robustness, complexity, statistics, adaptive_stable, failures, total_runtime_s)
    write_readme()
    leakage = json.loads((ROOT / "leakage_audit.json").read_text())
    manifest = {
        "completed": True,
        "full_protocol_run": True,
        "folds": list(protocol["folds"]),
        "temperatures_C": list(map(float, protocol["temperatures_C"])),
        "runtime_s_estimated_from_artifact_timestamps_plus_finalize": total_runtime_s,
        "best_kf_by_fold_mean_mae": best_kf,
        "leakage_status": leakage["status"],
        "failed_or_diverged_runs": len(failures),
        "finalize_runtime_s": finalize_runtime_s,
    }
    (ROOT / "results" / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
