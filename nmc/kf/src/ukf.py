from __future__ import annotations

from time import perf_counter_ns

import numpy as np

from .data_io import FitTrajectory, Trajectory
from .ecm_models import ECMParameters, terminal_voltage, transition_state
from .ekf import FilterResult
from .ocv import OCVMap


def _sigma_points(mean: np.ndarray, covariance: np.ndarray, alpha: float, beta: float, kappa: float):
    count = len(mean)
    lam = alpha * alpha * (count + kappa) - count
    scale = count + lam
    jitter = 1e-12
    for _ in range(8):
        try:
            root = np.linalg.cholesky(scale * (covariance + np.eye(count) * jitter))
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        raise np.linalg.LinAlgError("UKF covariance is not positive definite")
    points = np.empty((2 * count + 1, count), dtype=np.float64)
    points[0] = mean
    for index in range(count):
        points[index + 1] = mean + root[:, index]
        points[count + index + 1] = mean - root[:, index]
    mean_weights = np.full(2 * count + 1, 1.0 / (2.0 * scale), dtype=np.float64)
    covariance_weights = mean_weights.copy()
    mean_weights[0] = lam / scale
    covariance_weights[0] = mean_weights[0] + (1.0 - alpha * alpha + beta)
    return points, mean_weights, covariance_weights


def run_ukf(
    trajectory: Trajectory | FitTrajectory,
    initial_soc: float,
    params: ECMParameters,
    ocv_map: OCVMap,
    q_ref_Ah: float,
    noise: dict,
    filter_config: dict,
    eta: float = 1.0,
) -> FilterResult:
    state_count = params.state_count
    state = np.zeros(state_count, dtype=np.float64)
    state[0] = float(np.clip(initial_soc, 0.0, 1.0))
    covariance = np.diag(np.asarray(filter_config["initial_covariance_diag"], dtype=np.float64)[:state_count])
    process_diag = np.asarray([float(noise["q_soc"])] + [float(noise["q_vp"])] * int(params.order))
    measurement_variance = float(noise["r_voltage"])
    ukf = filter_config["ukf"]
    alpha, beta, kappa = float(ukf["alpha"]), float(ukf["beta"]), float(ukf["kappa"])
    voltage_state_limit = float(filter_config["voltage_state_limit_V"])
    covariance_floor = float(filter_config["covariance_floor"])
    count = len(trajectory.time_s)
    soc = np.empty(count, dtype=np.float64)
    voltage_prediction = np.empty(count, dtype=np.float64)
    innovation = np.empty(count, dtype=np.float64)
    latency = np.empty(count, dtype=np.int64)
    diverged = False

    for index in range(count):
        started = perf_counter_ns()
        points, wm, wc = _sigma_points(state, covariance, alpha, beta, kappa)
        if index > 0:
            propagated = np.asarray(
                [
                    transition_state(
                        point,
                        trajectory.current_discharge_A[index - 1],
                        trajectory.dt_s[index],
                        q_ref_Ah,
                        params,
                        eta,
                    )
                    for point in points
                ]
            )
            state = np.sum(wm[:, None] * propagated, axis=0)
            differences = propagated - state
            covariance = np.einsum("i,ij,ik->jk", wc, differences, differences) + np.diag(process_diag * trajectory.dt_s[index])
        points, wm, wc = _sigma_points(state, covariance, alpha, beta, kappa)
        voltages = np.asarray(
            [
                terminal_voltage(point, trajectory.current_discharge_A[index], trajectory.temperature_C, params, ocv_map)
                for point in points
            ]
        )
        predicted_voltage = float(np.dot(wm, voltages))
        voltage_delta = voltages - predicted_voltage
        state_delta = points - state
        innovation_variance = float(np.dot(wc, voltage_delta * voltage_delta) + measurement_variance)
        innovation_variance = max(innovation_variance, float(filter_config["innovation_variance_floor"]))
        cross_covariance = np.sum(wc[:, None] * state_delta * voltage_delta[:, None], axis=0)
        gain = cross_covariance / innovation_variance
        residual = float(trajectory.voltage_V[index] - predicted_voltage)
        state = state + gain * residual
        state[0] = np.clip(state[0], 0.0, 1.0)
        state[1:] = np.clip(state[1:], -voltage_state_limit, voltage_state_limit)
        covariance = covariance - np.outer(gain, gain) * innovation_variance
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
