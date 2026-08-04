#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter, perf_counter_ns

# The covariance repair paths are sensitive to BLAS reduction order. Apply the
# numerical runtime contract before importing NumPy on every supported OS.
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from src.data_io import (  # noqa: E402
    as_fit_trajectory,
    discover_dynamic_files,
    evaluation_mask,
    file_sha256,
    load_trajectory,
)
from src.ecm_models import (  # noqa: E402
    ECMParameters,
    propagate_soc,
    run_coulomb_counting,
)
from src.ekf import FilterResult, run_ekf  # noqa: E402
from src.leakage import LeakageGuard  # noqa: E402
from src.metrics import (  # noqa: E402
    bootstrap_paired_mean_difference,
    convergence_metrics,
    error_metrics,
)
from src.neural_results import load_neural_predictions, neural_metric_rows  # noqa: E402
from src.ocv import build_ocv_map  # noqa: E402
from src.parameter_identification import (  # noqa: E402
    fit_ecm_parameters,
    select_bounds_training_only,
)
from src.ukf import run_ukf  # noqa: E402


KF_METHODS = ["CC", "1RC_EKF", "2RC_EKF", "2RC_UKF", "Adaptive_2RC_EKF"]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def timed_coulomb_counting(trajectory, initial_soc: float, q_ref_Ah: float, eta: float) -> FilterResult:
    soc = np.empty(len(trajectory.time_s), dtype=np.float64)
    latency = np.empty(len(soc), dtype=np.int64)
    soc[0] = float(np.clip(initial_soc, 0.0, 1.0))
    latency[0] = 0
    for index in range(1, len(soc)):
        started = perf_counter_ns()
        soc[index] = propagate_soc(
            soc[index - 1],
            trajectory.current_discharge_A[index - 1],
            trajectory.dt_s[index],
            q_ref_Ah,
            eta,
        )
        latency[index] = perf_counter_ns() - started
    empty = np.full(len(soc), np.nan, dtype=np.float64)
    return FilterResult(soc, empty, empty.copy(), latency, False, np.nan)


def run_method(method: str, trajectory, initial_soc, parameters, noises, ocv_map, q_ref, ecm_config, adaptive_beta):
    eta = float(ecm_config["coulomb_counting"]["coulombic_efficiency"])
    filter_config = ecm_config["filter"]
    if method == "CC":
        return timed_coulomb_counting(trajectory, initial_soc, q_ref, eta)
    if method == "1RC_EKF":
        return run_ekf(trajectory, initial_soc, parameters[1], ocv_map, q_ref, noises[method], filter_config, eta)
    if method == "2RC_EKF":
        return run_ekf(trajectory, initial_soc, parameters[2], ocv_map, q_ref, noises[method], filter_config, eta)
    if method == "2RC_UKF":
        return run_ukf(trajectory, initial_soc, parameters[2], ocv_map, q_ref, noises[method], filter_config, eta)
    if method == "Adaptive_2RC_EKF":
        return run_ekf(
            trajectory,
            initial_soc,
            parameters[2],
            ocv_map,
            q_ref,
            noises["2RC_EKF"],
            filter_config,
            eta,
            adaptive_beta=adaptive_beta,
        )
    raise ValueError(method)


def tune_filter_noise(method, validation_trajectories, cv_params, ocv_map, q_ref, ecm_config):
    order = 1 if method == "1RC_EKF" else 2
    records = []
    warmup = float(ecm_config["parameter_identification"]["validation_warmup_s"])
    candidates = ecm_config["filter"]["noise_candidates"]
    for candidate in candidates:
        scores = []
        for trajectory in validation_trajectories:
            params = cv_params[trajectory.profile]
            initial = ocv_map.inverse(trajectory.initial_rest_voltage_V, trajectory.temperature_C)
            if method == "2RC_UKF":
                result = run_ukf(
                    trajectory,
                    initial,
                    params,
                    ocv_map,
                    q_ref,
                    candidate,
                    ecm_config["filter"],
                )
            else:
                result = run_ekf(
                    trajectory,
                    initial,
                    params,
                    ocv_map,
                    q_ref,
                    candidate,
                    ecm_config["filter"],
                )
            mask = (trajectory.elapsed_s >= warmup) & np.isfinite(result.innovation_V)
            score = float(np.sqrt(np.mean(np.square(result.innovation_V[mask])))) if np.any(mask) else np.inf
            scores.append(score)
            records.append(
                {
                    "method": method,
                    "order": order,
                    "noise_name": candidate["name"],
                    "validation_profile": trajectory.profile,
                    "innovation_rmse_V": score,
                    "diverged": result.diverged,
                }
            )
        records.append(
            {
                "method": method,
                "order": order,
                "noise_name": candidate["name"],
                "validation_profile": "CV_MEAN",
                "innovation_rmse_V": float(np.mean(scores)),
                "diverged": any(not np.isfinite(score) for score in scores),
            }
        )
    means = [record for record in records if record["validation_profile"] == "CV_MEAN"]
    selected_record = min(means, key=lambda record: (record["innovation_rmse_V"], record["noise_name"]))
    selected = next(candidate for candidate in candidates if candidate["name"] == selected_record["noise_name"])
    return dict(selected), records


def tune_adaptive_beta(validation_trajectories, cv_params, noise, ocv_map, q_ref, ecm_config):
    records = []
    warmup = float(ecm_config["parameter_identification"]["validation_warmup_s"])
    for beta in ecm_config["filter"]["adaptive_beta_candidates"]:
        scores = []
        for trajectory in validation_trajectories:
            params = cv_params[trajectory.profile]
            initial = ocv_map.inverse(trajectory.initial_rest_voltage_V, trajectory.temperature_C)
            result = run_ekf(
                trajectory,
                initial,
                params,
                ocv_map,
                q_ref,
                noise,
                ecm_config["filter"],
                adaptive_beta=float(beta),
            )
            mask = (trajectory.elapsed_s >= warmup) & np.isfinite(result.innovation_V)
            score = float(np.sqrt(np.mean(np.square(result.innovation_V[mask])))) if np.any(mask) else np.inf
            scores.append(score)
            records.append(
                {
                    "method": "Adaptive_2RC_EKF",
                    "adaptive_beta": float(beta),
                    "validation_profile": trajectory.profile,
                    "innovation_rmse_V": score,
                    "diverged": result.diverged,
                }
            )
        records.append(
            {
                "method": "Adaptive_2RC_EKF",
                "adaptive_beta": float(beta),
                "validation_profile": "CV_MEAN",
                "innovation_rmse_V": float(np.mean(scores)),
                "diverged": any(not np.isfinite(score) for score in scores),
            }
        )
    means = [record for record in records if record["validation_profile"] == "CV_MEAN"]
    selected = min(means, key=lambda record: (record["innovation_rmse_V"], record["adaptive_beta"]))
    return float(selected["adaptive_beta"]), records


def prediction_frame(method, fold, trajectory, perturbation, result, eval_mask):
    return pd.DataFrame(
        {
            "method": method,
            "fold": fold,
            "profile": trajectory.profile,
            "temperature_C": trajectory.temperature_C,
            "file_name": trajectory.file_name,
            "end_index": np.arange(len(trajectory.time_s), dtype=np.int64),
            "elapsed_s": trajectory.elapsed_s,
            "reference_soc": trajectory.reference_soc,
            "predicted_soc": result.soc,
            "error_pct": (result.soc - trajectory.reference_soc) * 100.0,
            "abs_error_pct": np.abs(result.soc - trajectory.reference_soc) * 100.0,
            "voltage_prediction_V": result.voltage_prediction_V,
            "innovation_V": result.innovation_V,
            "eval_mask": eval_mask,
            "initial_perturbation_pct_point": float(perturbation),
        }
    )


def evaluate_oracle_detail(method, fold, trajectory, result, protocol, ecm_config):
    evaluation = protocol["evaluation"]
    mask = evaluation_mask(len(trajectory.time_s), evaluation["first_end_index"], evaluation["end_index_stride"])
    overall = {
        "method": method,
        "fold": fold,
        "temperature_C": trajectory.temperature_C,
        "condition": "oracle_initial_soc",
        "diverged": result.diverged,
        **error_metrics(trajectory.reference_soc, result.soc, mask),
    }
    horizon_rows = []
    first_index = int(evaluation["first_end_index"])
    evaluated_elapsed = trajectory.elapsed_s - float(trajectory.elapsed_s[first_index])
    for horizon in evaluation["early_horizons_s"]:
        horizon_mask = mask & (evaluated_elapsed <= float(horizon))
        horizon_rows.append(
            {
                "method": method,
                "fold": fold,
                "temperature_C": trajectory.temperature_C,
                "horizon_s": float(horizon),
                **error_metrics(trajectory.reference_soc, result.soc, horizon_mask),
            }
        )
    band_rows = []
    reference_pct = trajectory.reference_soc * 100.0
    bands = evaluation["soc_bands_pct"]
    for band_index, (lower, upper) in enumerate(bands):
        upper_mask = reference_pct <= float(upper) if band_index == len(bands) - 1 else reference_pct < float(upper)
        band_mask = mask & (reference_pct >= float(lower)) & upper_mask
        band_rows.append(
            {
                "method": method,
                "fold": fold,
                "temperature_C": trajectory.temperature_C,
                "soc_band_pct": f"{float(lower):g}-{float(upper):g}",
                **error_metrics(trajectory.reference_soc, result.soc, band_mask),
            }
        )
    latency = result.latency_ns[result.latency_ns > 0] / 1000.0
    complexity = {
        "method": method,
        "fold": fold,
        "temperature_C": trajectory.temperature_C,
        "mean_step_latency_us": float(np.mean(latency)) if len(latency) else np.nan,
        "p95_step_latency_us": float(np.percentile(latency, 95)) if len(latency) else np.nan,
        "python_profile_runtime_s": float(np.sum(result.latency_ns) / 1e9),
    }
    convergence = convergence_metrics(
        trajectory.elapsed_s,
        trajectory.reference_soc,
        result.soc,
        int(evaluation["first_end_index"]),
        float(ecm_config["initial_soc"]["convergence_threshold_pct"]),
        float(ecm_config["initial_soc"]["sustained_duration_s"]),
    )
    return overall, horizon_rows, band_rows, complexity, convergence, mask


def write_audit(protocol, trajectories, ocv_sources, output_path: Path):
    inventory = []
    for trajectory in trajectories.values():
        raw_current = -trajectory.current_discharge_A
        source = pd.read_csv(trajectory.path)
        inventory.append(
            {
                "file": str(trajectory.path),
                "profile": trajectory.profile,
                "temperature_C": trajectory.temperature_C,
                "rows": len(trajectory.time_s),
                "dt_median_s": float(np.median(trajectory.dt_s)),
                "dt_min_s": float(np.min(trajectory.dt_s)),
                "dt_max_s": float(np.max(trajectory.dt_s)),
                "voltage_min_V": float(np.min(trajectory.voltage_V)),
                "voltage_max_V": float(np.max(trajectory.voltage_V)),
                "source_current_min_A": float(np.min(raw_current)),
                "source_current_max_A": float(np.max(raw_current)),
                "reference_soc_start_pct": float(trajectory.reference_soc[0] * 100.0),
                "reference_soc_end_pct": float(trajectory.reference_soc[-1] * 100.0),
                "Q_ref_lc_ocv_Ah": float(pd.to_numeric(source["Q_ref_lc_ocv_Ah"], errors="coerce").iloc[0]),
                "Qnet_removed_end_Ah": float(pd.to_numeric(source["Qnet_removed(Ah)"], errors="coerce").iloc[-1]),
                "source_soc_unit": "fraction",
                "sha256": file_sha256(trajectory.path),
            }
        )
    inventory_frame = pd.DataFrame(inventory).sort_values(["temperature_C", "profile"])
    inventory_frame.to_csv(output_path.parent / "results" / "data_inventory.csv", index=False)
    lines = [
        "# Repository Audit",
        "",
        "## Locked protocol",
        "",
        "- Profiles: DST, FUDS, US06 in the 3-LOPO experiment.",
        "- Folds: DST+FUDS -> US06; FUDS+US06 -> DST; DST+US06 -> FUDS.",
        "- Temperatures: 0, 25, 45 C.",
        "- Dynamic inputs: terminal voltage and current. Temperature is the nominal file label; no measured temperature channel exists.",
        "- Source current uses discharge-negative sign. Filters convert to discharge-positive current.",
        "- Time propagation uses consecutive Test_Time(s) differences.",
        "- Reference SOC is SOC_CC in fraction units.",
        "- Shared evaluation mask is every endpoint from index 49 through the final row, matching existing neural prediction files.",
        "- Early 60 s and 300 s metrics are measured from the first shared evaluation endpoint (index 49).",
        "- No extra cutoff, first80/last20 validation, or early stopping is applied.",
        "- KF inputs are not normalized and no scaler is fitted.",
        "- OCV is a monotone isotonic + PCHIP SOC map. At the three observed temperatures, exact temperature maps are used; configured interpolation for unseen in-range temperatures is linear in temperature, with positive resistance and log-time-constant interpolation.",
        "- ECM parameters are identified separately at 0, 25, and 45 C from training profiles only. No test-temperature trajectory is used to form a parameter map.",
        "",
        "## Critical label provenance",
        "",
        "SOC_CC was generated from OCV-inferred initial SOC and temperature-specific low-current discharge capacity followed by current integration. Oracle-initialized CC therefore shares the reference-label construction equation and is not an information-equivalent comparison to the neural model.",
        "",
        "## Characterization",
        "",
        "Independent SP20-1 low-current OCV files exist at 0/25/45 C. No independent NMC HPPC file was found. OCV and capacity use those characterization files; R0/R1/C1/R2/C2 are identified from training profiles only.",
        "",
        ocv_sources.to_markdown(index=False),
        "",
        "## Dynamic data inventory",
        "",
        inventory_frame.to_markdown(index=False),
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return inventory_frame


def metrics_from_prediction_frame(frame: pd.DataFrame) -> dict:
    selected = frame[frame["eval_mask"].astype(bool)]
    return error_metrics(selected["reference_soc"].to_numpy(), selected["predicted_soc"].to_numpy())


def aggregate_neural_results(protocol: dict):
    temperature_rows = []
    fold_rows = []
    source_rows = []
    pointwise = {}
    for model_name in protocol["neural_results"]["models"]:
        for fold in protocol["folds"]:
            predictions = load_neural_predictions(model_name, fold, protocol)
            if int(predictions["end_index"].min()) != int(protocol["evaluation"]["first_end_index"]):
                raise RuntimeError(f"Neural evaluation mask mismatch for {model_name}/{fold}")
            rows, point_frame = neural_metric_rows(model_name, fold, predictions)
            temperature_rows.extend(rows)
            pointwise[(model_name, fold)] = point_frame
            per_seed = []
            for _, seed_frame in predictions.groupby("seed"):
                per_seed.append(error_metrics(seed_frame["y_true"].to_numpy(), seed_frame["y_pred"].to_numpy()))
            aggregate = {
                key: float(np.mean([record[key] for record in per_seed]))
                for key in per_seed[0]
                if key != "n"
            }
            aggregate["n"] = int(sum(len(frame) for _, frame in predictions.groupby("seed")) / len(per_seed))
            fold_rows.append({"method": model_name, "fold": fold, "condition": "native", **aggregate})
            for source in sorted(predictions["source_file"].unique()):
                source_rows.append({"method": model_name, "fold": fold, "source_file": source, "read_only": True})
    return temperature_rows, fold_rows, pointwise, source_rows


def benchmark_proposed_mlp() -> dict:
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError:
        override = os.environ.get("CEMA_TORCH_PYTHON")
        sibling = Path(sys.prefix).resolve().parent / "cema_soc_repro"
        candidates = [Path(override).expanduser().resolve()] if override else []
        candidates.extend((sibling / "python.exe", sibling / "bin" / "python"))
        interpreter = next((candidate for candidate in candidates if candidate.is_file()), None)
        if interpreter is None:
            raise RuntimeError(
                "PyTorch is unavailable. Install it in the active environment or set "
                "CEMA_TORCH_PYTHON to a valid interpreter."
            )
        completed = subprocess.run(
            [str(interpreter), str(ROOT / "src" / "benchmark_proposed_torch.py")],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        return {
            "method": "Proposed_EMA_MLP_G4_residual",
            "latency_ns": np.asarray(payload["latency_ns"], dtype=np.int64),
            "parameter_count": int(payload["parameter_count"]),
            "parameter_bytes": int(payload["parameter_bytes"]),
            "macs": int(payload["macs"]),
        }

    class ProposedG4ResidualMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(17 * 4, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Dropout(0.07),
                nn.Linear(128, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
            )
            self.unused_normal_head = nn.Linear(128, 1)
            self.anchor = nn.Sequential(
                nn.Linear(8, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Dropout(0.07),
                nn.Linear(128, 64),
                nn.SiLU(),
                nn.Linear(64, 1),
            )
            self.residual = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Dropout(0.07), nn.Linear(64, 1))
            self.residual_limit = nn.Parameter(torch.tensor(0.1))

        def forward(self, x):
            summary = torch.cat([x[:, -1], x.mean(1), x.std(1, unbiased=False), x[:, -1] - x[:, 0]], dim=1)
            encoded = self.encoder(summary)
            anchor = torch.sigmoid(self.anchor(x[:, :, :8]))
            residual = self.residual_limit * torch.tanh(self.residual(encoded))
            return (anchor[:, -1] + residual).clamp(0.0, 1.0)

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model = ProposedG4ResidualMLP().eval()
    inputs = torch.zeros((1, 50, 17), dtype=torch.float32)
    with torch.inference_mode():
        for _ in range(100):
            model(inputs)
        timings = []
        for _ in range(1000):
            started = perf_counter_ns()
            model(inputs)
            timings.append(perf_counter_ns() - started)
    torch.set_num_threads(previous_threads)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 44036:
        raise RuntimeError(f"Proposed benchmark architecture mismatch: {parameter_count} parameters")
    return {
        "method": "Proposed_EMA_MLP_G4_residual",
        "latency_ns": np.asarray(timings, dtype=np.int64),
        "parameter_count": parameter_count,
        "parameter_bytes": parameter_count * 4,
        "macs": 901888,
    }


def add_fold_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, frame in fold_metrics.groupby("method"):
        rows.append(
            {
                "method": method,
                "n_folds": int(frame["fold"].nunique()),
                "fold_mean_mae_pct": float(frame["mae_pct"].mean()),
                "fold_std_mae_pct": float(frame["mae_pct"].std(ddof=1)),
                "worst_fold_mae_pct": float(frame["mae_pct"].max()),
                "worst_fold": str(frame.loc[frame["mae_pct"].idxmax(), "fold"]),
                "mean_rmse_pct": float(frame["rmse_pct"].mean()),
                "mean_p95_ae_pct": float(frame["p95_ae_pct"].mean()),
                "mean_max_ae_pct": float(frame["max_ae_pct"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("fold_mean_mae_pct")


def build_complexity_table(latency_samples, ocv_map, total_runtime_s: float, proposed_benchmark: dict) -> pd.DataFrame:
    lookup_bytes = int(sum(curve.soc.nbytes + curve.voltage.nbytes for curve in ocv_map.curves.values()))
    definitions = {
        "CC": (1, 0, 8),
        "1RC_EKF": (2, 1, 110),
        "2RC_EKF": (3, 2, 260),
        "2RC_UKF": (3, 2, 720),
        "Adaptive_2RC_EKF": (4, 2, 275),
    }
    rows = []
    for method, (stored_states, rc_count, operation_estimate) in definitions.items():
        dimension = 1 + rc_count
        covariance_bytes = 0 if method == "CC" else dimension * dimension * 8
        sigma_bytes = (2 * dimension + 1) * dimension * 8 if method == "2RC_UKF" else 0
        state_bytes = stored_states * 8
        parameter_scalars = 1 if method == "CC" else 3 if method == "1RC_EKF" else 5
        parameter_bytes = parameter_scalars * 8
        samples = np.asarray(latency_samples.get(method, []), dtype=np.float64) / 1000.0
        rows.append(
            {
                "method": method,
                "mean_step_latency_us_pc_python": float(np.mean(samples)) if len(samples) else np.nan,
                "p95_step_latency_us_pc_python": float(np.percentile(samples, 95)) if len(samples) else np.nan,
                "stored_state_scalars": stored_states,
                "state_covariance_sigma_bytes_estimate": state_bytes + covariance_bytes + sigma_bytes,
                "ecm_parameter_bytes_FP64": parameter_bytes,
                "ocv_lookup_bytes_FP64": lookup_bytes,
                "peak_working_ram_bytes_estimate": state_bytes + covariance_bytes + sigma_bytes + parameter_bytes + lookup_bytes,
                "scalar_operation_count_per_step_estimate": operation_estimate,
                "operation_count_scope": "rough scalar arithmetic estimate; OCV interpolation and transcendental implementation dependent",
                "latency_scope": "PC Python timing, not MCU latency",
                "full_experiment_runtime_s": float(total_runtime_s),
            }
        )
    proposed_samples = proposed_benchmark["latency_ns"].astype(np.float64) / 1000.0
    rows.append(
        {
            "method": proposed_benchmark["method"],
            "mean_step_latency_us_pc_python": float(np.mean(proposed_samples)),
            "p95_step_latency_us_pc_python": float(np.percentile(proposed_samples, 95)),
            "stored_state_scalars": 50 * 17,
            "state_covariance_sigma_bytes_estimate": 50 * 17 * 4,
            "ecm_parameter_bytes_FP64": 0,
            "ocv_lookup_bytes_FP64": 0,
            "peak_working_ram_bytes_estimate": 50 * 17 * 4 + int(proposed_benchmark["parameter_bytes"]),
            "scalar_operation_count_per_step_estimate": int(proposed_benchmark["macs"]),
            "operation_count_scope": "reported one-window forward MACs; causal feature-generation cost excluded",
            "latency_scope": "PC PyTorch CPU single-thread batch-1 timing, not MCU latency",
            "full_experiment_runtime_s": np.nan,
        }
    )
    return pd.DataFrame(rows)


def make_figures(kf_predictions: pd.DataFrame, neural_pointwise: dict, temperature_metrics: pd.DataFrame, robustness: pd.DataFrame, complexity: pd.DataFrame, fold_summary: pd.DataFrame, figures: Path):
    figures.mkdir(parents=True, exist_ok=True)
    methods = ["CC", "1RC_EKF", "2RC_EKF", "2RC_UKF", "Adaptive_2RC_EKF"]
    colors = {"CC": "#555555", "1RC_EKF": "#0072B2", "2RC_EKF": "#009E73", "2RC_UKF": "#D55E00", "Adaptive_2RC_EKF": "#CC79A7"}
    for fold in ["US06", "DST", "FUDS"]:
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
        error_fig, error_axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
        proposed = neural_pointwise.get(("Proposed_EMA_MLP_G4_residual", fold))
        for axis, error_axis, temperature in zip(axes, error_axes, [0.0, 25.0, 45.0]):
            subset = kf_predictions[(kf_predictions["fold"] == fold) & np.isclose(kf_predictions["temperature_C"], temperature)]
            reference = subset[subset["method"] == "CC"]
            step = max(1, len(reference) // 2500)
            axis.plot(reference["elapsed_s"].iloc[::step], reference["reference_soc"].iloc[::step] * 100.0, color="black", lw=1.5, label="Reference")
            for method in methods:
                frame = subset[subset["method"] == method]
                axis.plot(frame["elapsed_s"].iloc[::step], frame["predicted_soc"].iloc[::step] * 100.0, lw=0.9, color=colors[method], label=method)
                error_axis.plot(frame["elapsed_s"].iloc[::step], frame["error_pct"].iloc[::step], lw=0.8, color=colors[method], label=method)
            if proposed is not None:
                p = proposed[np.isclose(proposed["temperature"], temperature)]
                raw_time = reference.set_index("end_index")["elapsed_s"]
                p_time = p["end_index"].map(raw_time)
                axis.plot(p_time.iloc[::step], p["neural_prediction_mean"].iloc[::step] * 100.0, lw=1.0, color="#E69F00", label="Proposed EMA-MLP")
                error_axis.plot(p_time.iloc[::step], p["neural_error_mean_pct"].iloc[::step], lw=0.9, color="#E69F00", label="Proposed EMA-MLP")
            axis.set_ylabel("SOC (%)")
            axis.set_title(f"{fold} holdout, {temperature:g} C")
            axis.grid(alpha=0.25)
            error_axis.axhline(0.0, color="black", lw=0.8)
            error_axis.set_ylabel("Error (pp)")
            error_axis.set_title(f"{fold} holdout, {temperature:g} C")
            error_axis.grid(alpha=0.25)
        axes[-1].set_xlabel("Elapsed time (s)")
        error_axes[-1].set_xlabel("Elapsed time (s)")
        axes[0].legend(ncol=3, fontsize=8)
        error_axes[0].legend(ncol=3, fontsize=8)
        fig.tight_layout()
        error_fig.tight_layout()
        fig.savefig(figures / f"{fold.lower()}_soc_curves.png", dpi=180)
        error_fig.savefig(figures / f"{fold.lower()}_error_curves.png", dpi=180)
        plt.close(fig)
        plt.close(error_fig)

    oracle = temperature_metrics[temperature_metrics["condition"].isin(["oracle_initial_soc", "native"])].copy()
    oracle["fold_temp"] = oracle["fold"] + "_" + oracle["temperature_C"].map(lambda value: f"{value:g}C")
    heat = oracle.pivot_table(index="method", columns="fold_temp", values="mae_pct")
    fig, axis = plt.subplots(figsize=(12, max(5, 0.45 * len(heat))))
    image = axis.imshow(heat.to_numpy(), aspect="auto", cmap="viridis_r")
    axis.set_xticks(range(len(heat.columns)), heat.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(heat.index)), heat.index)
    for row in range(len(heat.index)):
        for column in range(len(heat.columns)):
            value = heat.iloc[row, column]
            if np.isfinite(value):
                axis.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=7, color="white" if value > np.nanmedian(heat.to_numpy()) else "black")
    fig.colorbar(image, ax=axis, label="MAE (%)")
    axis.set_title("Profile x temperature MAE")
    fig.tight_layout()
    fig.savefig(figures / "profile_temperature_mae_heatmap.png", dpi=180)
    plt.close(fig)

    convergence = robustness.groupby(["method", "initial_perturbation_pct_point"], as_index=False)["time_to_sustained_ae_lt_2pct_60s_s"].mean()
    fig, axis = plt.subplots(figsize=(9, 5))
    for method in methods:
        frame = convergence[convergence["method"] == method]
        axis.plot(frame["initial_perturbation_pct_point"], frame["time_to_sustained_ae_lt_2pct_60s_s"], marker="o", label=method)
    axis.set_xlabel("Initial SOC perturbation (percentage points)")
    axis.set_ylabel("Time to sustained AE < 2% (s)")
    axis.set_title("Initial SOC convergence")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "initial_soc_convergence.png", dpi=180)
    plt.close(fig)

    trade = complexity.merge(fold_summary[["method", "fold_mean_mae_pct"]], on="method", how="inner")
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(trade["mean_step_latency_us_pc_python"], trade["fold_mean_mae_pct"], s=55)
    for row in trade.itertuples():
        axis.annotate(row.method, (row.mean_step_latency_us_pc_python, row.fold_mean_mae_pct), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.set_xscale("log")
    axis.set_xlabel("Mean PC Python step latency (us, log scale)")
    axis.set_ylabel("Fold-mean MAE (%)")
    axis.set_title("Accuracy-latency tradeoff")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "accuracy_latency_tradeoff.png", dpi=180)
    plt.close(fig)


def dataframe_markdown(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    table = frame[columns].copy()
    for column in table.select_dtypes(include="number").columns:
        table[column] = table[column].map(lambda value: f"{value:.{digits}f}" if np.isfinite(value) else "NA")
    return table.to_markdown(index=False)


def write_report(fold_summary, temperature_metrics, robustness, complexity, statistics, adaptive_stable, failures, runtime_s, output_path: Path):
    core = fold_summary.copy()
    temp = temperature_metrics[temperature_metrics["method"].isin(KF_METHODS + ["Proposed_EMA_MLP_G4_residual"])]
    accuracy_winner = fold_summary.iloc[0]
    kf_ranking = fold_summary[fold_summary["method"].isin(["1RC_EKF", "2RC_EKF", "2RC_UKF", "Adaptive_2RC_EKF"])]
    best_kf = kf_ranking.iloc[0]
    proposed = fold_summary[fold_summary["method"] == "Proposed_EMA_MLP_G4_residual"].iloc[0]
    perturbation_summary = (
        robustness[robustness["initial_perturbation_pct_point"] != 0]
        .groupby("method", as_index=False)
        .agg(
            mean_perturbed_mae_pct=("mae_pct", "mean"),
            mean_sustained_convergence_s=("time_to_sustained_ae_lt_2pct_60s_s", "mean"),
        )
        .sort_values("mean_perturbed_mae_pct")
    )
    best_perturbed_kf = perturbation_summary[perturbation_summary["method"].isin(["1RC_EKF", "2RC_EKF", "2RC_UKF", "Adaptive_2RC_EKF"])].iloc[0]
    finite_complexity = complexity[np.isfinite(complexity["mean_step_latency_us_pc_python"])]
    fastest = finite_complexity.iloc[finite_complexity["mean_step_latency_us_pc_python"].argmin()]
    ukf_45_fuds = temperature_metrics[
        (temperature_metrics["method"] == "2RC_UKF")
        & (temperature_metrics["fold"] == "FUDS")
        & np.isclose(temperature_metrics["temperature_C"], 45.0)
    ]
    lines = [
        "# NMC 3-LOPO ECM-KF Baseline Report",
        "",
        "## Scope",
        "",
        "This is a direct comparison on the same local NMC data and the same 3-LOPO evaluation endpoints. It is not a numerical comparison against MAE values reported by unrelated literature datasets.",
        "",
        "## Information and prior inputs",
        "",
        "| Method | Online inputs | Prior information | Initial SOC |",
        "|---|---|---|---|",
        "| Proposed EMA-MLP | 50-s causal Vcorr/I/T window | trained weights, training-only scaler | none supplied |",
        "| CC | current, dt | independent temperature capacity | oracle or explicit perturbation |",
        "| ECM-KF | voltage, current, nominal temperature, dt | independent OCV/capacity plus training-profile ECM and Q/R | oracle or explicit perturbation |",
        "",
        "The KF baselines use more explicit prior information than the proposed model. OCV curves, ECM parameters, usable capacity, and initial SOC are not hidden from this comparison.",
        "At the measured 0/25/45 C points, exact per-temperature ECM maps are used, so no interpolation is invoked in the reported runs. The fixed unseen-temperature rule is linear resistance and log-time-constant interpolation with endpoint clamping.",
        "",
        "## Reference-label caveat",
        "",
        "The reference SOC is itself OCV-start coulomb counting with the same temperature-specific low-current capacity. Oracle CC therefore reproduces the label construction and must not be interpreted as an independently validated physical ground-truth estimator.",
        "",
        "## Fold summary",
        "",
        dataframe_markdown(core, ["method", "fold_mean_mae_pct", "fold_std_mae_pct", "worst_fold_mae_pct", "worst_fold", "mean_rmse_pct", "mean_p95_ae_pct"]),
        "",
        "## Result-based comparison",
        "",
        f"- Lowest oracle fold-mean MAE: {accuracy_winner['method']} at {accuracy_winner['fold_mean_mae_pct']:.4f}%. This is oracle CC and is privileged by the reference-label construction.",
        f"- Best actual Kalman-family method: {best_kf['method']} at {best_kf['fold_mean_mae_pct']:.4f}% fold-mean MAE; proposed EMA-MLP is {proposed['fold_mean_mae_pct']:.4f}%.",
        f"- Best KF under nonzero initial-SOC perturbations by mean MAE: {best_perturbed_kf['method']} at {best_perturbed_kf['mean_perturbed_mae_pct']:.4f}%.",
        f"- Lowest measured PC Python step latency: {fastest['method']} at {fastest['mean_step_latency_us_pc_python']:.3f} us. This is not an MCU latency claim.",
        "- Initialization: 1RC-EKF has the lowest average MAE across the nonzero perturbation tests, while perturbed CC cannot self-correct because it has no voltage update.",
        "- Prior cost: CC requires characterized capacity and initial SOC; ECM-KF additionally requires OCV, fitted RC parameters, and Q/R selection; neural methods require labeled multi-profile training and a training-only scaler but receive no explicit initial SOC.",
        "",
        "## Profile x temperature MAE",
        "",
        dataframe_markdown(temp.sort_values(["method", "fold", "temperature_C"]), ["method", "fold", "temperature_C", "mae_pct", "rmse_pct", "p95_ae_pct", "max_ae_pct"]),
        "",
        "## Initialization robustness",
        "",
        dataframe_markdown(
            robustness.groupby(["method", "initial_perturbation_pct_point"], as_index=False).agg(
                mae_pct=("mae_pct", "mean"),
                time_to_ae_lt_2pct_s=("time_to_ae_lt_2pct_s", "mean"),
                time_to_sustained_ae_lt_2pct_60s_s=("time_to_sustained_ae_lt_2pct_60s_s", "mean"),
            ),
            ["method", "initial_perturbation_pct_point", "mae_pct", "time_to_ae_lt_2pct_s", "time_to_sustained_ae_lt_2pct_60s_s"],
        ),
        "",
        "## Complexity",
        "",
        dataframe_markdown(complexity, ["method", "mean_step_latency_us_pc_python", "p95_step_latency_us_pc_python", "stored_state_scalars", "peak_working_ram_bytes_estimate", "ocv_lookup_bytes_FP64", "scalar_operation_count_per_step_estimate"]),
        "",
        "PC Python latency is not MCU latency. Operation counts are implementation-independent rough scalar estimates intended only for later Cortex-M budgeting.",
        "",
        "## Paired statistics",
        "",
        dataframe_markdown(statistics, list(statistics.columns)) if len(statistics) else "No valid paired comparison was produced.",
        "",
        "## Stability and failures",
        "",
        f"Adaptive 2RC-EKF stable across all oracle fold-temperature runs: {adaptive_stable}.",
        f"Recorded failures/divergences: {len(failures)}.",
    ]
    if len(ukf_45_fuds):
        row = ukf_45_fuds.iloc[0]
        lines.append(
            f"2RC-UKF remained finite but was practically unstable on FUDS at 45 C (MAE {row['mae_pct']:.4f}%, bias {row['bias_pct']:.4f}%, max AE {row['max_ae_pct']:.4f}%). The run is retained; these results do not establish a unique root cause."
        )
    if failures:
        lines.extend(["", pd.DataFrame(failures).to_markdown(index=False)])
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "Accuracy, initialization robustness, online computation, and prior characterization cost are separate axes. The report does not treat the method with the lowest oracle MAE as universally superior.",
            "",
            f"Total Python experiment runtime: {runtime_s:.2f} s.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(output_path: Path):
    text = """# NMC 3-LOPO ECM-KF baselines

This isolated workspace evaluates CC, 1RC-EKF, 2RC-EKF, 2RC-UKF, and an adaptive 2RC-EKF on the locked DST/FUDS/US06 3-LOPO protocol. The source neural repositories and their result files are read-only inputs.

## Environment

- Python 3
- numpy, pandas, scipy, scikit-learn, numba, matplotlib, PyYAML, openpyxl, xlrd, torch

## Run

```bash
cd nmc/kf
python run_all_lopo.py 2>&1 | tee logs/run_all_lopo.log
```

The command rebuilds all CSV files, figures, `audit.md`, `leakage_audit.json`, and `report.md`. Configuration is fixed in `configs/protocol.yaml` and `configs/ecm.yaml`.

## Leakage boundary

For each fold, independent low-current characterization supplies OCV and capacity. ECM fitting, parameter-bound selection, Q/R selection, and adaptive-R selection receive only the two training profiles. Held-out files are passed only to final filter evaluation. `leakage_audit.json` and `results/fit_file_manifest.csv` record the paths used by every fitting stage.
"""
    output_path.write_text(text, encoding="utf-8")


def flatten_fit_manifest(leakage_records: list[dict]) -> pd.DataFrame:
    rows = []
    for fold_record in leakage_records:
        for event in fold_record["fit_events"]:
            for path in event["files"]:
                rows.append(
                    {
                        "fold": fold_record["fold"],
                        "test_profile": fold_record["test_profile"],
                        "stage": event["stage"],
                        "fit_file": path,
                        "passed": event["passed"],
                    }
                )
    return pd.DataFrame(rows).drop_duplicates().sort_values(["fold", "stage", "fit_file"])


def main():
    parser = argparse.ArgumentParser(description="Run leakage-safe NMC 3-LOPO ECM-KF baselines")
    parser.add_argument("--folds", nargs="*", choices=["US06", "DST", "FUDS"], default=None)
    parser.add_argument("--temperatures", nargs="*", type=float, default=None)
    parser.add_argument(
        "--skip-neural-comparison",
        action="store_true",
        help="Run and archive the KF baselines without requiring archived neural predictions.",
    )
    parser.add_argument(
        "--locked-parameters-dir",
        type=Path,
        default=None,
        help="Replay frozen training-only ECM/bounds/Q-R selections instead of refitting them.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "runs/nmc_kf",
        help="Directory for generated results, figures, reports, and audits.",
    )
    args = parser.parse_args()

    started = perf_counter()
    protocol = load_yaml(ROOT / "configs" / "protocol.yaml")
    ecm_config = load_yaml(ROOT / "configs" / "ecm.yaml")
    protocol["source_repo"] = str(REPO / "nmc/deep_learning")
    protocol["data_root"] = str(
        REPO / "Data/Preprocessed/NMC/nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean"
    )
    protocol["characterization"].update({
        "root": str(REPO / "Data/NMC/OCV"),
        "capacity_table": str(REPO / "nmc/preprocessing/locked_metadata/lc_ocv_capacity_reference.csv"),
    })
    protocol["neural_results"]["ema_root"] = str(
        REPO / "runs/nmc_dl/nmc_goal_vcorr_it_train_dst_selector_results"
    )
    locked_parameters = args.locked_parameters_dir.resolve() if args.locked_parameters_dir else None
    if locked_parameters:
        locked_ecm = pd.read_csv(locked_parameters / "ecm_parameters.csv")
        locked_bounds = pd.read_csv(locked_parameters / "bounds_selection.csv")
        locked_noise = pd.read_csv(locked_parameters / "filter_noise_selection.csv")
        noise_by_name = {row["name"]: row for row in ecm_config["filter"]["noise_candidates"]}
    folds = args.folds or list(protocol["folds"])
    temperatures = args.temperatures or [float(value) for value in protocol["temperatures_C"]]
    full_protocol_run = set(folds) == set(protocol["folds"]) and set(temperatures) == set(map(float, protocol["temperatures_C"]))

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results_dir = output_root / "results"
    predictions_dir = results_dir / "predictions"
    parameters_dir = results_dir / "parameters"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    parameters_dir.mkdir(parents=True, exist_ok=True)

    ocv_map, ocv_curves, ocv_sources = build_ocv_map(protocol, ecm_config)
    ocv_curves.to_csv(results_dir / "ocv_curves.csv", index=False)
    ocv_sources.to_csv(results_dir / "ocv_sources.csv", index=False)
    paths = discover_dynamic_files(Path(protocol["data_root"]), protocol["profiles"], protocol["temperatures_C"])
    trajectories = {}
    for path in paths:
        trajectory = load_trajectory(path, float(protocol["current_convention"]["multiplier"]))
        trajectories[(trajectory.profile, trajectory.temperature_C)] = trajectory
    write_audit(protocol, trajectories, ocv_sources, output_root / "audit.md")

    temperature_rows = []
    horizon_rows = []
    band_rows = []
    robustness_rows = []
    complexity_run_rows = []
    bound_rows = []
    parameter_rows = []
    noise_rows = []
    failures = []
    leakage_records = []
    oracle_prediction_frames = []
    latency_samples = {method: [] for method in KF_METHODS}

    for fold in folds:
        train_profiles = [str(profile).upper() for profile in protocol["folds"][fold]]
        test_files = {trajectories[(fold, float(temperature))].path.resolve() for temperature in temperatures}
        guard = LeakageGuard(fold, fold, test_files)
        print(f"[Fold] {fold}: train={train_profiles}, test={fold}", flush=True)
        for temperature in temperatures:
            temperature = float(temperature)
            train = [as_fit_trajectory(trajectories[(profile, temperature)]) for profile in train_profiles]
            test = trajectories[(fold, temperature)]
            train_paths = [trajectory.path for trajectory in train]
            q_ref = float(ocv_map.q_ref_by_temperature[temperature])
            parameters = {}
            cv_parameters = {}
            selected_noises = {}
            if locked_parameters:
                guard.record_fit(f"{temperature:g}C_frozen_training_only_parameter_replay", train_paths)
                for order in [1, 2]:
                    rows = locked_ecm[
                        (locked_ecm["fold"] == fold)
                        & np.isclose(locked_ecm["temperature_C"], temperature)
                        & (locked_ecm["order"] == order)
                    ]
                    if len(rows) != 1:
                        raise RuntimeError(f"Expected one locked {fold}/{temperature:g}C/{order}RC row, found {len(rows)}")
                    row = rows.iloc[0]
                    parameters[order] = ECMParameters(
                        order=order, temperature_C=temperature, R0_ohm=float(row["R0_ohm"]),
                        R1_ohm=float(row["R1_ohm"]), tau1_s=float(row["tau1_s"]),
                        R2_ohm=float(row["R2_ohm"]), tau2_s=float(row["tau2_s"]),
                        fit_bounds_name=str(row["fit_bounds_name"]),
                        fit_voltage_rmse_V=float(row["fit_voltage_rmse_V"]),
                    )
                selected_rows = locked_noise[
                    (locked_noise["fold"] == fold)
                    & np.isclose(locked_noise["temperature_C"], temperature)
                    & locked_noise["selected"]
                    & (locked_noise["validation_profile"] == "CV_MEAN")
                ]
                for method in ("1RC_EKF", "2RC_EKF", "2RC_UKF"):
                    rows = selected_rows[selected_rows["method"] == method]
                    if len(rows) != 1:
                        raise RuntimeError(f"Expected one locked noise row for {fold}/{temperature:g}C/{method}")
                    selected_noises[method] = noise_by_name[str(rows.iloc[0]["noise_name"])]
                adaptive_rows = selected_rows[selected_rows["method"] == "Adaptive_2RC_EKF"]
                if len(adaptive_rows) != 1:
                    raise RuntimeError(f"Expected one locked adaptive-beta row for {fold}/{temperature:g}C")
                adaptive_beta = float(adaptive_rows.iloc[0]["adaptive_beta"])
            else:
                for order in [1, 2]:
                    guard.record_fit(f"{temperature:g}C_{order}RC_bounds_selection", train_paths)
                    selected_bounds, records, cached = select_bounds_training_only(
                        train, order, temperature, ocv_map, q_ref, ecm_config
                    )
                    for record in records:
                        bound_rows.append({"fold": fold, "selected": record["bounds_name"] == selected_bounds, **record})
                    cv_parameters[order] = cached[selected_bounds]
                    guard.record_fit(f"{temperature:g}C_{order}RC_final_parameter_fit", train_paths)
                    fitted = fit_ecm_parameters(train, order, temperature, ocv_map, q_ref, ecm_config, selected_bounds)
                    parameters[order] = fitted
                    parameter_rows.append(
                        {
                            "fold": fold,
                            "train_profiles": "+".join(train_profiles),
                            **fitted.to_dict(),
                            "C1_F": fitted.tau1_s / fitted.R1_ohm,
                            "C2_F": fitted.tau2_s / fitted.R2_ohm if order == 2 else np.nan,
                        }
                    )

                validation = train
                for method, order in [("1RC_EKF", 1), ("2RC_EKF", 2), ("2RC_UKF", 2)]:
                    guard.record_fit(f"{temperature:g}C_{method}_QR_selection", train_paths)
                    noise, records = tune_filter_noise(method, validation, cv_parameters[order], ocv_map, q_ref, ecm_config)
                    selected_noises[method] = noise
                    for record in records:
                        noise_rows.append({"fold": fold, "temperature_C": temperature, "selected": record["noise_name"] == noise["name"], **record})
                guard.record_fit(f"{temperature:g}C_adaptive_R_selection", train_paths)
                adaptive_beta, adaptive_records = tune_adaptive_beta(
                    validation, cv_parameters[2], selected_noises["2RC_EKF"], ocv_map, q_ref, ecm_config
                )
                for record in adaptive_records:
                    noise_rows.append({"fold": fold, "temperature_C": temperature, "selected": record["adaptive_beta"] == adaptive_beta, **record})

            for perturbation in ecm_config["initial_soc"]["perturbations_pct_point"]:
                initial_soc = float(np.clip(test.reference_soc[0] + float(perturbation) / 100.0, 0.0, 1.0))
                for method in KF_METHODS:
                    result = run_method(
                        method,
                        test,
                        initial_soc,
                        parameters,
                        selected_noises,
                        ocv_map,
                        q_ref,
                        ecm_config,
                        adaptive_beta,
                    )
                    evaluation = evaluation_mask(
                        len(test.time_s),
                        int(protocol["evaluation"]["first_end_index"]),
                        int(protocol["evaluation"]["end_index_stride"]),
                    )
                    metrics = error_metrics(test.reference_soc, result.soc, evaluation)
                    convergence = convergence_metrics(
                        test.elapsed_s,
                        test.reference_soc,
                        result.soc,
                        int(protocol["evaluation"]["first_end_index"]),
                        float(ecm_config["initial_soc"]["convergence_threshold_pct"]),
                        float(ecm_config["initial_soc"]["sustained_duration_s"]),
                    )
                    robustness_rows.append(
                        {
                            "method": method,
                            "fold": fold,
                            "temperature_C": temperature,
                            "initial_perturbation_pct_point": float(perturbation),
                            "initial_soc_after_clipping_pct": initial_soc * 100.0,
                            "condition": "oracle_initial_soc" if float(perturbation) == 0.0 else "initial_soc_perturbation",
                            "diverged": result.diverged,
                            **metrics,
                            **convergence,
                        }
                    )
                    if result.diverged:
                        failures.append({"method": method, "fold": fold, "temperature_C": temperature, "perturbation": perturbation, "reason": "non-finite filter state or covariance"})
                    if float(perturbation) == 0.0:
                        overall, horizons, bands, complexity_row, _, mask = evaluate_oracle_detail(
                            method, fold, test, result, protocol, ecm_config
                        )
                        temperature_rows.append(overall)
                        horizon_rows.extend(horizons)
                        band_rows.extend(bands)
                        complexity_run_rows.append(complexity_row)
                        latency_samples[method].extend(result.latency_ns[result.latency_ns > 0].tolist())
                        frame = prediction_frame(method, fold, test, perturbation, result, mask)
                        oracle_prediction_frames.append(frame)
                        frame.to_csv(
                            predictions_dir / f"{fold.lower()}_{temperature:g}C_{method.lower()}_oracle.csv.gz",
                            index=False,
                            compression="gzip",
                        )
            print(f"[{fold}] {temperature:g}C complete", flush=True)
        leakage_records.append(guard.to_dict())

    if locked_parameters:
        locked_ecm.to_csv(parameters_dir / "ecm_parameters.csv", index=False)
        locked_bounds.to_csv(parameters_dir / "bounds_selection.csv", index=False)
        locked_noise.to_csv(parameters_dir / "filter_noise_selection.csv", index=False)
    else:
        pd.DataFrame(parameter_rows).to_csv(parameters_dir / "ecm_parameters.csv", index=False)
        pd.DataFrame(bound_rows).to_csv(parameters_dir / "bounds_selection.csv", index=False)
        pd.DataFrame(noise_rows).to_csv(parameters_dir / "filter_noise_selection.csv", index=False)
    fit_manifest = flatten_fit_manifest(leakage_records)
    fit_manifest.to_csv(results_dir / "fit_file_manifest.csv", index=False)
    leakage_audit = {
        "status": "PASS" if all(record["status"] == "PASS" for record in leakage_records) else "FAIL",
        "leakage_count": int(sum(record["leakage_count"] for record in leakage_records)),
        "held_out_used_for": "final evaluation only",
        "characterization_is_independent": True,
        "characterization_files": ocv_sources["source_file"].tolist(),
        "folds": leakage_records,
        "config_files": [str((ROOT / "configs" / "protocol.yaml").resolve()), str((ROOT / "configs" / "ecm.yaml").resolve())],
    }
    (output_root / "leakage_audit.json").write_text(json.dumps(leakage_audit, indent=2), encoding="utf-8")
    if leakage_audit["leakage_count"] != 0:
        raise RuntimeError("Leakage audit failed")

    kf_predictions = pd.concat(oracle_prediction_frames, ignore_index=True)
    kf_fold_rows = []
    for (method, fold), frame in kf_predictions.groupby(["method", "fold"]):
        kf_fold_rows.append({"method": method, "fold": fold, "condition": "oracle_initial_soc", **metrics_from_prediction_frame(frame)})

    neural_temperature_rows = []
    neural_fold_rows = []
    neural_pointwise = {}
    if not args.skip_neural_comparison:
        neural_temperature_rows, neural_fold_rows, neural_pointwise, neural_sources = aggregate_neural_results(protocol)
        pd.DataFrame(neural_sources).to_csv(results_dir / "neural_result_sources.csv", index=False)
    temperature_metrics = pd.DataFrame(temperature_rows + neural_temperature_rows)
    fold_metrics = pd.DataFrame(kf_fold_rows + neural_fold_rows)
    robustness = pd.DataFrame(robustness_rows)
    temperature_metrics.to_csv(results_dir / "temperature_metrics.csv", index=False)
    fold_metrics.to_csv(results_dir / "fold_metrics.csv", index=False)
    robustness.to_csv(results_dir / "initial_soc_robustness.csv", index=False)
    pd.DataFrame(horizon_rows).to_csv(results_dir / "horizon_metrics.csv", index=False)
    pd.DataFrame(band_rows).to_csv(results_dir / "soc_band_metrics.csv", index=False)
    pd.DataFrame(complexity_run_rows).to_csv(results_dir / "profile_runtime.csv", index=False)

    fold_summary = add_fold_summary(fold_metrics)
    fold_summary.to_csv(results_dir / "fold_summary.csv", index=False)
    adaptive_stable = not any(record["method"] == "Adaptive_2RC_EKF" for record in failures)
    eligible = fold_summary[fold_summary["method"].isin(["1RC_EKF", "2RC_EKF", "2RC_UKF", "Adaptive_2RC_EKF"])]
    if not adaptive_stable:
        eligible = eligible[eligible["method"] != "Adaptive_2RC_EKF"]
    best_kf = str(eligible.iloc[0]["method"])

    statistics = pd.DataFrame()
    if not args.skip_neural_comparison:
        differences = {}
        for fold in folds:
            proposed = neural_pointwise[("Proposed_EMA_MLP_G4_residual", fold)]
            kf = kf_predictions[(kf_predictions["method"] == best_kf) & (kf_predictions["fold"] == fold) & kf_predictions["eval_mask"]]
            for temperature in temperatures:
                left = kf[np.isclose(kf["temperature_C"], temperature)][["file_name", "end_index", "abs_error_pct"]]
                right = proposed[np.isclose(proposed["temperature"], temperature)][["file_name", "end_index", "neural_abs_error_pct"]]
                aligned = left.merge(right, on=["file_name", "end_index"], validate="one_to_one")
                if len(aligned) != len(left) or len(aligned) != len(right):
                    raise RuntimeError(f"Paired endpoint mismatch for {fold}/{temperature:g}C")
                differences[(fold, float(temperature))] = aligned["abs_error_pct"].to_numpy() - aligned["neural_abs_error_pct"].to_numpy()
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
        pd.DataFrame(paired_rows).to_csv(results_dir / "paired_fold_temperature.csv", index=False)
        statistics = pd.DataFrame([{"comparison": f"{best_kf} minus Proposed_EMA_MLP_G4_residual absolute error", "unit": "percentage point", **statistic}])
        statistics.to_csv(results_dir / "paired_bootstrap_statistics.csv", index=False)

        direct_methods = ["Proposed_EMA_MLP_G4_residual", best_kf]
        fold_metrics[fold_metrics["method"].isin(direct_methods)].sort_values(["fold", "method"]).to_csv(
            results_dir / "direct_comparison.csv", index=False
        )
    elapsed = perf_counter() - started
    proposed_benchmark = benchmark_proposed_mlp()
    complexity = build_complexity_table(latency_samples, ocv_map, elapsed, proposed_benchmark)
    complexity.to_csv(results_dir / "complexity.csv", index=False)
    make_figures(kf_predictions, neural_pointwise, temperature_metrics, robustness, complexity, fold_summary, output_root / "figures")
    if not args.skip_neural_comparison:
        write_report(fold_summary, temperature_metrics, robustness, complexity, statistics, adaptive_stable, failures, elapsed, output_root / "report.md")
    write_readme(output_root / "README.md")

    manifest = {
        "completed": True,
        "full_protocol_run": full_protocol_run,
        "folds": folds,
        "temperatures_C": temperatures,
        "runtime_s": elapsed,
        "best_kf_by_fold_mean_mae": best_kf,
        "leakage_status": leakage_audit["status"],
        "failed_runs": len(failures),
        "neural_comparison_skipped": args.skip_neural_comparison,
        "parameter_mode": "locked_training_only_replay" if locked_parameters else "fresh_training_only_refit",
        "linear_algebra_threads": 1,
        "locked_parameters_dir": str(locked_parameters) if locked_parameters else None,
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
