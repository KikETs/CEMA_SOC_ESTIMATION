from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np

from .data_io import FitTrajectory, Trajectory
from .ecm_models import ECMParameters, measurement_jacobian, terminal_voltage, transition_jacobian, transition_state
from .ocv import OCVMap


@dataclass
class FilterResult:
    soc: np.ndarray
    voltage_prediction_V: np.ndarray
    innovation_V: np.ndarray
    latency_ns: np.ndarray
    diverged: bool
    final_measurement_variance: float


def run_ekf(
    trajectory: Trajectory | FitTrajectory,
    initial_soc: float,
    params: ECMParameters,
    ocv_map: OCVMap,
    q_ref_Ah: float,
    noise: dict,
    filter_config: dict,
    eta: float = 1.0,
    adaptive_beta: float | None = None,
) -> FilterResult:
    state_count = params.state_count
    state = np.zeros(state_count, dtype=np.float64)
    state[0] = float(np.clip(initial_soc, 0.0, 1.0))
    p0 = np.asarray(filter_config["initial_covariance_diag"], dtype=np.float64)[:state_count]
    covariance = np.diag(p0)
    process_diag = np.asarray([float(noise["q_soc"])] + [float(noise["q_vp"])] * int(params.order))
    measurement_variance = float(noise["r_voltage"])
    base_measurement_variance = measurement_variance
    r_scale_limits = filter_config.get("adaptive_r_scale_limits", [0.25, 25.0])
    count = len(trajectory.time_s)
    soc = np.empty(count, dtype=np.float64)
    voltage_prediction = np.empty(count, dtype=np.float64)
    innovation = np.empty(count, dtype=np.float64)
    latency = np.empty(count, dtype=np.int64)
    diverged = False
    identity = np.eye(state_count)
    covariance_floor = float(filter_config["covariance_floor"])
    innovation_floor = float(filter_config["innovation_variance_floor"])
    voltage_state_limit = float(filter_config["voltage_state_limit_V"])

    for index in range(count):
        started = perf_counter_ns()
        if index > 0:
            transition = transition_jacobian(trajectory.dt_s[index], params)
            state = transition_state(
                state,
                trajectory.current_discharge_A[index - 1],
                trajectory.dt_s[index],
                q_ref_Ah,
                params,
                eta,
            )
            covariance = transition @ covariance @ transition.T + np.diag(process_diag * trajectory.dt_s[index])
        predicted_voltage = terminal_voltage(
            state,
            trajectory.current_discharge_A[index],
            trajectory.temperature_C,
            params,
            ocv_map,
        )
        residual = float(trajectory.voltage_V[index] - predicted_voltage)
        if adaptive_beta is not None:
            candidate = (1.0 - float(adaptive_beta)) * measurement_variance + float(adaptive_beta) * residual * residual
            measurement_variance = float(
                np.clip(
                    candidate,
                    base_measurement_variance * float(r_scale_limits[0]),
                    base_measurement_variance * float(r_scale_limits[1]),
                )
            )
        observation = measurement_jacobian(state, trajectory.temperature_C, ocv_map)
        innovation_variance = float((observation @ covariance @ observation.T)[0, 0]) + measurement_variance
        innovation_variance = max(innovation_variance, innovation_floor)
        gain = (covariance @ observation.T)[:, 0] / innovation_variance
        state = state + gain * residual
        state[0] = np.clip(state[0], 0.0, 1.0)
        state[1:] = np.clip(state[1:], -voltage_state_limit, voltage_state_limit)
        update = identity - np.outer(gain, observation[0])
        covariance = update @ covariance @ update.T + np.outer(gain, gain) * measurement_variance
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        covariance = (eigenvectors * np.maximum(eigenvalues, covariance_floor)) @ eigenvectors.T
        latency[index] = perf_counter_ns() - started
        soc[index] = state[0]
        voltage_prediction[index] = predicted_voltage
        innovation[index] = residual
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(covariance)):
            diverged = True
            soc[index:] = np.nan
            voltage_prediction[index:] = np.nan
            innovation[index:] = np.nan
            latency[index:] = 0
            break
    return FilterResult(soc, voltage_prediction, innovation, latency, diverged, measurement_variance)
