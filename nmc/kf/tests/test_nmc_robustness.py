from __future__ import annotations

import numpy as np

import run_nmc_robustness as robustness


def test_soc_band_boundaries() -> None:
    values = np.array([0.0, 0.35, 0.35001, 0.65, 0.65001, 1.0])
    assert robustness.soc_band(values).tolist() == ["<=35", "<=35", "35-65", "35-65", ">65", ">65"]


def test_metric_units_are_percentage_points() -> None:
    mae, rmse, bias = robustness.metrics(np.array([0.5, 0.5]), np.array([0.51, 0.48]))
    assert np.isclose(mae, 1.5)
    assert np.isclose(rmse, np.sqrt(2.5))
    assert np.isclose(bias, -0.5)


def test_cold_reset_points_are_fixed_mid_record() -> None:
    first = robustness.cold_reset_points(10_000, "US06", 0.0, 50)
    second = robustness.cold_reset_points(10_000, "US06", 0.0, 50)
    assert first == second
    assert len(first) == 5 and len(set(first)) == 5
    assert all(1_000 <= point <= 9_000 for point in first)
