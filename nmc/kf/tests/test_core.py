from pathlib import Path

import numpy as np
import pytest

from src.data_io import FitTrajectory, Trajectory
from src.ecm_models import ECMParameters, run_coulomb_counting, terminal_voltage, transition_state
from src.ekf import run_ekf
from src.leakage import LeakageGuard
from src.ocv import OCVMap, TemperatureCurve
from src.ukf import run_ukf


def synthetic_ocv_map():
    soc = np.linspace(0.0, 1.0, 101)
    voltage = 3.0 + 1.1 * soc
    curve = TemperatureCurve(25.0, soc, voltage, "synthetic", 2.0)
    return OCVMap({25.0: curve}, slope_min=1e-3, slope_max=8.0)


def synthetic_trajectory(length=400):
    time = np.arange(length, dtype=np.float64)
    dt = np.ones(length, dtype=np.float64)
    current = 0.8 + 0.3 * np.sin(np.arange(length) / 30.0)
    params = ECMParameters(2, 25.0, 0.025, 0.012, 12.0, 0.018, 160.0)
    ocv = synthetic_ocv_map()
    state = np.asarray([0.8, 0.0, 0.0])
    voltage = np.empty(length)
    reference = np.empty(length)
    for index in range(length):
        if index:
            state = transition_state(state, current[index - 1], dt[index], 2.0, params)
        reference[index] = state[0]
        voltage[index] = terminal_voltage(state, current[index], 25.0, params, ocv)
    trajectory = Trajectory(
        Path("synthetic.csv"), "synthetic.csv", "SYNTH", 25.0, time, dt, voltage, current, reference, 3.88
    )
    return trajectory, params, ocv


def test_ocv_is_monotone_and_invertible():
    ocv = synthetic_ocv_map()
    soc = np.linspace(0.0, 1.0, 1001)
    voltage = ocv.ocv(soc, 25.0)
    assert np.all(np.diff(voltage) >= 0.0)
    assert ocv.inverse(float(ocv.ocv(0.37, 25.0)), 25.0) == pytest.approx(0.37, abs=1e-3)


def test_coulomb_counting_matches_synthetic_reference():
    trajectory, _, _ = synthetic_trajectory()
    prediction = run_coulomb_counting(trajectory, trajectory.reference_soc[0], 2.0)
    np.testing.assert_allclose(prediction, trajectory.reference_soc, atol=1e-12)


def test_ekf_and_ukf_remain_finite():
    trajectory, params, ocv = synthetic_trajectory()
    noise = {"q_soc": 1e-8, "q_vp": 1e-6, "r_voltage": 4e-5}
    config = {
        "initial_covariance_diag": [0.01, 0.01, 0.01],
        "covariance_floor": 1e-12,
        "innovation_variance_floor": 1e-10,
        "voltage_state_limit_V": 1.0,
        "ukf": {"alpha": 0.1, "beta": 2.0, "kappa": 0.0},
    }
    for result in [
        run_ekf(trajectory, 0.7, params, ocv, 2.0, noise, config),
        run_ukf(trajectory, 0.7, params, ocv, 2.0, noise, config),
    ]:
        assert not result.diverged
        assert np.isfinite(result.soc).all()
        assert np.mean(np.abs(result.soc[-100:] - trajectory.reference_soc[-100:])) < 0.01


def test_fit_trajectory_excludes_reference_soc():
    fields = set(FitTrajectory.__dataclass_fields__)
    assert "reference_soc" not in fields


def test_leakage_guard_rejects_held_out_path(tmp_path):
    test_path = (tmp_path / "NMC_25C_US06.csv").resolve()
    guard = LeakageGuard("US06", "US06", {test_path})
    with pytest.raises(RuntimeError, match="Held-out leakage"):
        guard.record_fit("parameter_fit", [test_path])
