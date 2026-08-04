from __future__ import annotations

from dataclasses import replace

import numpy as np
from numba import njit
from scipy.optimize import least_squares

from .data_io import FitTrajectory
from .ecm_models import ECMParameters, simulate_open_loop
from .ocv import OCVMap


@njit(cache=True)
def _soc_from_current(initial_soc, current, dt, q_ref_Ah):
    output = np.empty(len(current), dtype=np.float64)
    output[0] = min(max(initial_soc, 0.0), 1.0)
    for index in range(1, len(current)):
        value = output[index - 1] - current[index - 1] * dt[index] / (3600.0 * q_ref_Ah)
        output[index] = min(max(value, 0.0), 1.0)
    return output


@njit(cache=True)
def _voltage_from_components(ocv, current, dt, vector, order):
    output = np.empty(len(current), dtype=np.float64)
    vp1 = 0.0
    vp2 = 0.0
    for index in range(len(current)):
        if index > 0:
            a1 = np.exp(-dt[index] / vector[2])
            vp1 = a1 * vp1 + vector[1] * (1.0 - a1) * current[index - 1]
            if order == 2:
                a2 = np.exp(-dt[index] / vector[4])
                vp2 = a2 * vp2 + vector[3] * (1.0 - a2) * current[index - 1]
        output[index] = ocv[index] - vector[0] * current[index] - vp1 - vp2
    return output


def _parameter_bounds(order: int, bounds: dict) -> tuple[np.ndarray, np.ndarray]:
    lower = [bounds["R0_ohm"][0], bounds["R1_ohm"][0], bounds["tau1_s"][0]]
    upper = [bounds["R0_ohm"][1], bounds["R1_ohm"][1], bounds["tau1_s"][1]]
    if int(order) == 2:
        lower.extend([bounds["R2_ohm"][0], bounds["tau2_s"][0]])
        upper.extend([bounds["R2_ohm"][1], bounds["tau2_s"][1]])
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def vector_to_parameters(vector: np.ndarray, order: int, temperature_C: float, bounds_name: str = "") -> ECMParameters:
    values = np.asarray(vector, dtype=np.float64)
    if int(order) == 1:
        return ECMParameters(1, temperature_C, values[0], values[1], values[2], fit_bounds_name=bounds_name)
    branches = sorted([(values[2], values[1]), (values[4], values[3])], key=lambda item: item[0])
    return ECMParameters(
        2,
        temperature_C,
        values[0],
        branches[0][1],
        branches[0][0],
        branches[1][1],
        branches[1][0],
        fit_bounds_name=bounds_name,
    )


def parameters_to_vector(params: ECMParameters) -> np.ndarray:
    values = [params.R0_ohm, params.R1_ohm, params.tau1_s]
    if int(params.order) == 2:
        values.extend([params.R2_ohm, params.tau2_s])
    return np.asarray(values, dtype=np.float64)


def fit_ecm_parameters(
    trajectories: list[FitTrajectory],
    order: int,
    temperature_C: float,
    ocv_map: OCVMap,
    q_ref_Ah: float,
    config: dict,
    bounds_name: str,
) -> ECMParameters:
    identification = config["parameter_identification"]
    bounds = identification["bounds_candidates"][bounds_name]
    lower, upper = _parameter_bounds(order, bounds)
    initial = np.asarray(identification["initial_guess"][f"{order}rc"], dtype=np.float64)
    initial = np.clip(initial, lower + 1e-9, upper - 1e-9)
    stride = int(identification["fit_stride"])
    warmup = float(identification["fit_warmup_s"])
    prepared = []
    for trajectory in trajectories:
        initial_soc = ocv_map.inverse(trajectory.initial_rest_voltage_V, trajectory.temperature_C)
        soc = _soc_from_current(initial_soc, trajectory.current_discharge_A, trajectory.dt_s, q_ref_Ah)
        ocv = np.asarray(ocv_map.ocv(soc, trajectory.temperature_C), dtype=np.float64)
        selected = np.flatnonzero(trajectory.elapsed_s >= warmup)[::stride]
        prepared.append((trajectory, ocv, selected))

    def residual(vector: np.ndarray) -> np.ndarray:
        pieces = []
        for trajectory, ocv, selected in prepared:
            predicted = _voltage_from_components(
                ocv,
                trajectory.current_discharge_A,
                trajectory.dt_s,
                np.asarray(vector, dtype=np.float64),
                int(order),
            )
            pieces.append(predicted[selected] - trajectory.voltage_V[selected])
        return np.concatenate(pieces)

    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        loss=str(identification["loss"]),
        f_scale=float(identification["f_scale_V"]),
        max_nfev=int(identification["max_nfev"]),
        x_scale="jac",
    )
    params = vector_to_parameters(result.x, order, temperature_C, bounds_name)
    rmse = float(np.sqrt(np.mean(np.square(residual(parameters_to_vector(params))))))
    return replace(params, fit_voltage_rmse_V=rmse)


def validation_voltage_rmse(
    trajectory: FitTrajectory,
    params: ECMParameters,
    ocv_map: OCVMap,
    q_ref_Ah: float,
    warmup_s: float,
) -> float:
    predicted, _ = simulate_open_loop(trajectory, params, ocv_map, q_ref_Ah)
    mask = trajectory.elapsed_s >= float(warmup_s)
    return float(np.sqrt(np.mean(np.square(predicted[mask] - trajectory.voltage_V[mask]))))


def select_bounds_training_only(
    trajectories: list[FitTrajectory],
    order: int,
    temperature_C: float,
    ocv_map: OCVMap,
    q_ref_Ah: float,
    config: dict,
) -> tuple[str, list[dict], dict[str, dict[str, ECMParameters]]]:
    if len(trajectories) < 2:
        raise ValueError("Training-only bounds selection requires at least two profiles")
    records = []
    cached = {}
    warmup = float(config["parameter_identification"]["validation_warmup_s"])
    for bounds_name in config["parameter_identification"]["bounds_candidates"]:
        cached[bounds_name] = {}
        scores = []
        for validation in trajectories:
            fit_set = [trajectory for trajectory in trajectories if trajectory.profile != validation.profile]
            params = fit_ecm_parameters(fit_set, order, temperature_C, ocv_map, q_ref_Ah, config, bounds_name)
            cached[bounds_name][validation.profile] = params
            score = validation_voltage_rmse(validation, params, ocv_map, q_ref_Ah, warmup)
            scores.append(score)
            records.append(
                {
                    "order": order,
                    "temperature_C": temperature_C,
                    "bounds_name": bounds_name,
                    "fit_profiles": "+".join(sorted(item.profile for item in fit_set)),
                    "validation_profile": validation.profile,
                    "validation_voltage_rmse_V": score,
                }
            )
        records.append(
            {
                "order": order,
                "temperature_C": temperature_C,
                "bounds_name": bounds_name,
                "fit_profiles": "CV_MEAN",
                "validation_profile": "CV_MEAN",
                "validation_voltage_rmse_V": float(np.mean(scores)),
            }
        )
    mean_records = [record for record in records if record["validation_profile"] == "CV_MEAN"]
    selected = min(mean_records, key=lambda record: (record["validation_voltage_rmse_V"], record["bounds_name"]))
    return str(selected["bounds_name"]), records, cached
