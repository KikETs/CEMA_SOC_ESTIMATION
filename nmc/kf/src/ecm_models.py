from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .data_io import FitTrajectory, Trajectory
from .ocv import OCVMap


@dataclass(frozen=True)
class ECMParameters:
    order: int
    temperature_C: float
    R0_ohm: float
    R1_ohm: float
    tau1_s: float
    R2_ohm: float = 0.0
    tau2_s: float = 1.0
    fit_bounds_name: str = ""
    fit_voltage_rmse_V: float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def state_count(self) -> int:
        return 1 + int(self.order)


def propagate_soc(soc: float, current_discharge_A: float, dt_s: float, q_ref_Ah: float, eta: float = 1.0) -> float:
    current = float(current_discharge_A)
    efficiency = float(eta) if current >= 0.0 else 1.0
    return float(np.clip(soc - efficiency * current * float(dt_s) / (3600.0 * float(q_ref_Ah)), 0.0, 1.0))


def transition_state(state: np.ndarray, current_discharge_A: float, dt_s: float, q_ref_Ah: float, params: ECMParameters, eta: float = 1.0) -> np.ndarray:
    output = np.asarray(state, dtype=np.float64).copy()
    output[0] = propagate_soc(output[0], current_discharge_A, dt_s, q_ref_Ah, eta)
    a1 = float(np.exp(-float(dt_s) / max(float(params.tau1_s), 1e-9)))
    output[1] = a1 * output[1] + float(params.R1_ohm) * (1.0 - a1) * float(current_discharge_A)
    if int(params.order) == 2:
        a2 = float(np.exp(-float(dt_s) / max(float(params.tau2_s), 1e-9)))
        output[2] = a2 * output[2] + float(params.R2_ohm) * (1.0 - a2) * float(current_discharge_A)
    return output


def transition_jacobian(dt_s: float, params: ECMParameters) -> np.ndarray:
    diagonal = [1.0, float(np.exp(-float(dt_s) / max(float(params.tau1_s), 1e-9)))]
    if int(params.order) == 2:
        diagonal.append(float(np.exp(-float(dt_s) / max(float(params.tau2_s), 1e-9))))
    return np.diag(diagonal)


def terminal_voltage(state: np.ndarray, current_discharge_A: float, temperature_C: float, params: ECMParameters, ocv_map: OCVMap) -> float:
    polarization = float(np.sum(np.asarray(state, dtype=np.float64)[1:]))
    return float(ocv_map.ocv(float(state[0]), float(temperature_C))) - float(current_discharge_A) * float(params.R0_ohm) - polarization


def measurement_jacobian(state: np.ndarray, temperature_C: float, ocv_map: OCVMap) -> np.ndarray:
    slope = float(ocv_map.slope(float(state[0]), float(temperature_C)))
    return np.asarray([slope] + [-1.0] * (len(state) - 1), dtype=np.float64)[None, :]


def simulate_open_loop(
    trajectory: FitTrajectory,
    params: ECMParameters,
    ocv_map: OCVMap,
    q_ref_Ah: float,
    initial_soc: float | None = None,
    eta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(trajectory.time_s)
    soc0 = ocv_map.inverse(trajectory.initial_rest_voltage_V, trajectory.temperature_C) if initial_soc is None else float(initial_soc)
    state = np.zeros(params.state_count, dtype=np.float64)
    state[0] = np.clip(soc0, 0.0, 1.0)
    voltage = np.empty(count, dtype=np.float64)
    soc = np.empty(count, dtype=np.float64)
    for index in range(count):
        if index > 0:
            state = transition_state(
                state,
                trajectory.current_discharge_A[index - 1],
                trajectory.dt_s[index],
                q_ref_Ah,
                params,
                eta,
            )
        voltage[index] = terminal_voltage(
            state,
            trajectory.current_discharge_A[index],
            trajectory.temperature_C,
            params,
            ocv_map,
        )
        soc[index] = state[0]
    return voltage, soc


def run_coulomb_counting(trajectory: Trajectory, initial_soc: float, q_ref_Ah: float, eta: float = 1.0) -> np.ndarray:
    soc = np.empty(len(trajectory.time_s), dtype=np.float64)
    soc[0] = float(np.clip(initial_soc, 0.0, 1.0))
    for index in range(1, len(soc)):
        soc[index] = propagate_soc(
            soc[index - 1],
            trajectory.current_discharge_A[index - 1],
            trajectory.dt_s[index],
            q_ref_Ah,
            eta,
        )
    return soc
