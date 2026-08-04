#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from finalize_existing import parameter_map, selected_beta, selected_noise
from run_all_lopo import KF_METHODS, ROOT, load_yaml, run_method
from src.data_io import Trajectory, discover_dynamic_files, load_trajectory
from src.ocv import build_ocv_map


MEASURED_STEPS = 100_000
WARMUP_STEPS = 1_000
TORCH_PYTHON = Path("/home/user/anaconda3/envs/torch_env/bin/python")
OUTPUT = ROOT / "results" / "deployment_assets.csv"
REPORT = ROOT / "report.md"
START_MARKER = "<!-- DEPLOYMENT_ASSETS_START -->"
END_MARKER = "<!-- DEPLOYMENT_ASSETS_END -->"


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def repeat_trajectory(base: Trajectory, length: int) -> Trajectory:
    indices = np.arange(length, dtype=np.int64) % len(base.time_s)
    dt = base.dt_s[indices].copy()
    time_s = np.cumsum(dt) - dt[0]
    return Trajectory(
        path=base.path,
        file_name=f"benchmark_repeat_{base.file_name}",
        profile=base.profile,
        temperature_C=base.temperature_C,
        time_s=time_s,
        dt_s=dt,
        voltage_V=base.voltage_V[indices].copy(),
        current_discharge_A=base.current_discharge_A[indices].copy(),
        reference_soc=base.reference_soc[indices].copy(),
        initial_rest_voltage_V=base.initial_rest_voltage_V,
    )


def benchmark_filters(protocol: dict, ecm_config: dict, ocv_map) -> dict[str, dict]:
    trajectories = {}
    for path in discover_dynamic_files(
        Path(protocol["data_root"]), protocol["profiles"], protocol["temperatures_C"]
    ):
        trajectory = load_trajectory(path, float(protocol["current_convention"]["multiplier"]))
        trajectories[(trajectory.profile, trajectory.temperature_C)] = trajectory
    base = trajectories[("US06", 25.0)]
    warmup = repeat_trajectory(base, WARMUP_STEPS + 1)
    measured = repeat_trajectory(base, MEASURED_STEPS + 1)
    params_frame = pd.read_csv(ROOT / "results" / "parameters" / "ecm_parameters.csv")
    noise_frame = pd.read_csv(ROOT / "results" / "parameters" / "filter_noise_selection.csv")
    parameters = parameter_map(params_frame, "US06", 25.0)
    noises = {
        method: selected_noise(noise_frame, "US06", 25.0, method)
        for method in ["1RC_EKF", "2RC_EKF", "2RC_UKF"]
    }
    adaptive_beta = selected_beta(noise_frame, "US06", 25.0)
    q_ref = float(ocv_map.q_ref_by_temperature[25.0])
    results = {}
    with threadpool_limits(limits=1):
        for method in KF_METHODS:
            run_method(
                method,
                warmup,
                float(base.reference_soc[0]),
                parameters,
                noises,
                ocv_map,
                q_ref,
                ecm_config,
                adaptive_beta,
            )
            result = run_method(
                method,
                measured,
                float(base.reference_soc[0]),
                parameters,
                noises,
                ocv_map,
                q_ref,
                ecm_config,
                adaptive_beta,
            )
            latency_ns = result.latency_ns[1:]
            if result.diverged or len(latency_ns) != MEASURED_STEPS or not np.all(latency_ns > 0):
                raise RuntimeError(f"Invalid deployment benchmark for {method}")
            results[method] = {
                "benchmark_steps": len(latency_ns),
                "median_latency_us": float(np.median(latency_ns) / 1000.0),
                "p95_latency_us": float(np.percentile(latency_ns, 95) / 1000.0),
            }
    return results


def benchmark_proposed() -> dict:
    completed = subprocess.run(
        [str(TORCH_PYTHON), str(ROOT / "src" / "benchmark_deployment_proposed_torch.py")],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
    )
    return json.loads(completed.stdout)


def convergence_consequence(method: str, robustness: pd.DataFrame) -> str:
    selected = robustness[
        (robustness["method"] == method)
        & robustness["initial_perturbation_pct_point"].isin([-5.0, 5.0])
    ]
    sustained = selected["time_to_sustained_ae_lt_2pct_60s_s"]
    finite = sustained.dropna()
    if method == "CC":
        return (
            f"+/-5 pp error persists; 0/{len(selected)} reached sustained AE<2%; "
            f"mean MAE={selected['mae_pct'].mean():.3f}%"
        )
    never = len(selected) - len(finite)
    return (
        f"first sustained AE<2% for 60 s: median={finite.median():.1f} s from shared eval start, "
        f"finite={len(finite)}/{len(selected)}, worst finite={finite.max():.1f} s"
        + (f", never={never}" if never else "")
    )


def asset_definitions(ocv_bytes: int, proposed_weight_bytes: int) -> dict[str, dict]:
    qref_bytes = 3 * 8
    scaler_bytes = 17 * 2 * 4
    r0_hat_bytes = 3 * 4
    assets = {
        "CC-oracle": {
            "asset_bytes": qref_bytes,
            "state_bytes": 8,
            "stored_assets": "Q_ref(T): 3 FP64 capacities=0.023 KB; oracle SOC0 is supplied at evaluation, not stored characterization",
        },
        "CC-realistic(+/-5% init)": {
            "asset_bytes": qref_bytes,
            "state_bytes": 8,
            "stored_assets": "Q_ref(T): 3 FP64 capacities=0.023 KB; external SOC0 estimator is outside this baseline",
        },
        "1RC-EKF": {
            "asset_bytes": ocv_bytes + qref_bytes + 3 * 3 * 8 + 3 * 3 * 8 + 2 * 8,
            "state_bytes": 48,
            "stored_assets": "OCV-SOC-T 606x(SOC,V) FP64=9.469 KB; Q_ref(T)=0.023 KB; per-T [R0,R1,C1]=0.070 KB; per-T Q/R=0.070 KB; P0 diag=0.016 KB",
        },
        "2RC-EKF": {
            "asset_bytes": ocv_bytes + qref_bytes + 3 * 5 * 8 + 3 * 3 * 8 + 3 * 8,
            "state_bytes": 96,
            "stored_assets": "OCV-SOC-T=9.469 KB; Q_ref(T)=0.023 KB; per-T [R0,R1,C1,R2,C2]=0.117 KB; per-T Q/R=0.070 KB; P0 diag=0.023 KB",
        },
        "adaptive 2RC-EKF": {
            "asset_bytes": ocv_bytes + qref_bytes + 3 * 5 * 8 + 3 * 3 * 8 + 3 * 8 + 3 * 8 + 2 * 8,
            "state_bytes": 104,
            "stored_assets": "2RC-EKF assets plus per-T adaptive beta=0.023 KB and R-scale limits=0.016 KB",
        },
        "2RC-UKF": {
            "asset_bytes": ocv_bytes + qref_bytes + 3 * 5 * 8 + 3 * 3 * 8 + 3 * 8 + 3 * 8,
            "state_bytes": 264,
            "stored_assets": "2RC-EKF assets plus UKF alpha/beta/kappa=0.023 KB",
        },
        "proposed (EMA+NN)": {
            "asset_bytes": proposed_weight_bytes + scaler_bytes + r0_hat_bytes,
            "state_bytes": 50 * 17 * 4,
            "stored_assets": (
                f"NN weights: 44,036 FP32={proposed_weight_bytes / 1024:.3f} KB tensor payload; "
                f"normalization mean/std: 34 FP32={scaler_bytes / 1024:.3f} KB; "
                f"per-T R0_hat: 3 FP32={r0_hat_bytes / 1024:.3f} KB"
            ),
        },
    }
    return assets


def build_rows(filter_latency: dict, proposed_latency: dict, ocv_bytes: int, robustness: pd.DataFrame):
    asset_rows = asset_definitions(ocv_bytes, int(proposed_latency["parameter_bytes_FP32"]))
    cpu = cpu_model()
    common_scope = "CPU single-thread; 100,000 measured steps after warm-up; US06 25C repeated input"
    methods = [
        {
            "method": "CC-oracle",
            "kernel": "CC",
            "runtime_signals": "I,T",
            "offline_prerequisites": "independent low-current capacity characterization; reference SOC0 oracle (evaluation only, non-deployable)",
            "needs_initial_SOC": "yes",
            "initial_soc_consequence": "exact reference SOC0 supplied; optimistic tracking-only condition",
            "runtime_state_dim": 1,
            "state_definition": "SOC",
            "analytic_flops_per_step": 8,
            "special_ops_per_step": "clip; temperature capacity lookup",
        },
        {
            "method": "CC-realistic(+/-5% init)",
            "kernel": "CC",
            "runtime_signals": "I,T",
            "offline_prerequisites": "independent low-current capacity characterization plus an external practical SOC initializer",
            "needs_initial_SOC": "yes",
            "initial_soc_consequence": convergence_consequence("CC", robustness),
            "runtime_state_dim": 1,
            "state_definition": "SOC",
            "analytic_flops_per_step": 8,
            "special_ops_per_step": "clip; temperature capacity lookup",
        },
        {
            "method": "1RC-EKF",
            "kernel": "1RC_EKF",
            "runtime_signals": "V,I,T",
            "offline_prerequisites": "independent OCV/capacity test; training-profile-only 1RC fit and voltage-innovation Q/R selection",
            "needs_initial_SOC": "yes",
            "initial_soc_consequence": convergence_consequence("1RC_EKF", robustness),
            "runtime_state_dim": 2,
            "state_definition": "[SOC,Vp1]",
            "analytic_flops_per_step": 110,
            "special_ops_per_step": "OCV+slope lookup; exp; 2x2 covariance eigensafeguard",
        },
        {
            "method": "2RC-EKF",
            "kernel": "2RC_EKF",
            "runtime_signals": "V,I,T",
            "offline_prerequisites": "independent OCV/capacity test; training-profile-only 2RC fit and voltage-innovation Q/R selection",
            "needs_initial_SOC": "yes",
            "initial_soc_consequence": convergence_consequence("2RC_EKF", robustness),
            "runtime_state_dim": 3,
            "state_definition": "[SOC,Vp1,Vp2]",
            "analytic_flops_per_step": 260,
            "special_ops_per_step": "OCV+slope lookup; 2 exp; 3x3 covariance eigensafeguard",
        },
        {
            "method": "adaptive 2RC-EKF",
            "kernel": "Adaptive_2RC_EKF",
            "runtime_signals": "V,I,T",
            "offline_prerequisites": "2RC prerequisites plus training-profile-only adaptive-R beta selection",
            "needs_initial_SOC": "yes",
            "initial_soc_consequence": convergence_consequence("Adaptive_2RC_EKF", robustness),
            "runtime_state_dim": 4,
            "state_definition": "[SOC,Vp1,Vp2] + adaptive R",
            "analytic_flops_per_step": 275,
            "special_ops_per_step": "2RC-EKF special ops plus adaptive-R update and clipping",
        },
        {
            "method": "2RC-UKF",
            "kernel": "2RC_UKF",
            "runtime_signals": "V,I,T",
            "offline_prerequisites": "independent OCV/capacity test; training-profile-only 2RC fit and UKF Q/R selection",
            "needs_initial_SOC": "yes",
            "initial_soc_consequence": convergence_consequence("2RC_UKF", robustness),
            "runtime_state_dim": 3,
            "state_definition": "[SOC,Vp1,Vp2]; 7 sigma points",
            "analytic_flops_per_step": 720,
            "special_ops_per_step": "2 Cholesky factorizations; sigma propagation; OCV lookup per sigma; eigensafeguard",
        },
        {
            "method": "proposed (EMA+NN)",
            "kernel": "proposed",
            "runtime_signals": "V,I,T",
            "offline_prerequisites": "training-profile R0_hat fit; training-only normalization; labeled SOC training (labels are OCV-start CC-derived)",
            "needs_initial_SOC": "no",
            "initial_soc_consequence": "not needed; no SOC0 is supplied to the model",
            "runtime_state_dim": 850,
            "state_definition": "50x17 causal rolling feature window; no recurrent SOC state",
            "analytic_flops_per_step": 1_803_840,
            "special_ops_per_step": "901,888 MAC NN lower bound plus 7 EMA updates/scaling; LayerNorm/SiLU/sigmoid/tanh/clamp not FLOP-counted",
        },
    ]
    rows = []
    for definition in methods:
        assets = asset_rows[definition["method"]]
        if definition["kernel"] == "proposed":
            timing = proposed_latency
            benchmark_scope = proposed_latency["scope"]
            benchmark_cpu = proposed_latency["cpu"]
        else:
            timing = filter_latency[definition["kernel"]]
            benchmark_scope = common_scope
            benchmark_cpu = cpu
        rows.append(
            {
                **{key: value for key, value in definition.items() if key != "kernel"},
                "stored_assets": assets["stored_assets"],
                "stored_assets_size_KB": assets["asset_bytes"] / 1024.0,
                "per_step_cost": (
                    f"{definition['analytic_flops_per_step']:,} FLOPs lower-bound + "
                    f"{timing['median_latency_us']:.3f} us median over {int(timing['benchmark_steps']):,} steps"
                ),
                "median_latency_us": timing["median_latency_us"],
                "p95_latency_us": timing["p95_latency_us"],
                "benchmark_steps": int(timing["benchmark_steps"]),
                "benchmark_scope": benchmark_scope,
                "benchmark_cpu": benchmark_cpu,
                "runtime_memory_KB": (assets["asset_bytes"] + assets["state_bytes"]) / 1024.0,
                "runtime_memory_scope": "resident stored assets plus persistent state/buffer; allocator and transient activations excluded",
                "analytic_flops_scope": "lower-bound scalar FLOPs; special/transcendental/decomposition ops listed separately",
                "asset_scope": "one deployed fold with exact 0/25/45C maps",
            }
        )
    return pd.DataFrame(rows)


def update_report(frame: pd.DataFrame):
    columns = [
        "method",
        "runtime_signals",
        "stored_assets_size_KB",
        "offline_prerequisites",
        "needs_initial_SOC",
        "initial_soc_consequence",
        "runtime_state_dim",
        "analytic_flops_per_step",
        "per_step_cost",
        "median_latency_us",
        "runtime_memory_KB",
    ]
    table = frame[columns].copy()
    for column in ["stored_assets_size_KB", "median_latency_us", "runtime_memory_KB"]:
        table[column] = table[column].map(lambda value: f"{value:.3f}")
    section = "\n".join(
        [
            START_MARKER,
            "## Deployment assets & information asymmetry",
            "",
            "The deployment comparison below uses one fold-specific model with exact 0/25/45 C maps. KF assets are FP64, while proposed weights, scaler statistics, and R0_hat are FP32. Stored-asset and runtime-memory values are logical payload sizes, not filesystem serialization sizes.",
            "",
            table.to_markdown(index=False),
            "",
            "Latency is the median of at least 100,000 CPU single-thread steps. KF/CC timings use the same per-step kernels as `results/complexity.csv` on repeated US06 25 C input. Proposed timing intentionally matches that file's batch-1 precomputed-window NN harness; causal feature-generation latency is excluded, although its EMA arithmetic is included in the analytic lower-bound FLOP column.",
            "",
            "The FLOP counts are lower bounds. OCV interpolation, exponentials, Cholesky/eigendecomposition, activation functions, clipping, and memory movement are listed as special operations and are represented by measured latency rather than forced into an architecture-dependent FLOP equivalence.",
            "",
            "Information is asymmetric: CC-oracle receives the reference initial SOC and shares the reference-label integration structure; realistic CC cannot correct a +/-5 pp initialization error. KF variants require explicit OCV/capacity/ECM/Q-R characterization and an initial SOC estimate. Proposed requires labeled training trajectories, training-only normalization, per-temperature R0_hat, and substantially larger weights, but no runtime initial SOC or OCV table.",
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


def main():
    protocol = load_yaml(ROOT / "configs" / "protocol.yaml")
    ecm_config = load_yaml(ROOT / "configs" / "ecm.yaml")
    ocv_map, ocv_curves, _ = build_ocv_map(protocol, ecm_config)
    ocv_bytes = len(ocv_curves) * 2 * 8
    filter_latency = benchmark_filters(protocol, ecm_config, ocv_map)
    proposed_latency = benchmark_proposed()
    robustness = pd.read_csv(ROOT / "results" / "initial_soc_robustness.csv")
    frame = build_rows(filter_latency, proposed_latency, ocv_bytes, robustness)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    update_report(frame)
    print(frame[["method", "stored_assets_size_KB", "median_latency_us", "runtime_memory_KB"]].to_string(index=False))


if __name__ == "__main__":
    main()
