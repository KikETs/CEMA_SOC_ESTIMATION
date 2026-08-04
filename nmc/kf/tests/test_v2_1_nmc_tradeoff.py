from __future__ import annotations

import numpy as np

import run_v2_1_nmc_tradeoff as module


def test_declared_q_soc_sweep() -> None:
    assert len(module.Q_SOC_VALUES) == 8
    assert np.isclose(module.Q_SOC_VALUES[0], 1e-12)
    assert np.isclose(module.Q_SOC_VALUES[-1], 1e-4)
    assert np.all(np.diff(np.log10(module.Q_SOC_VALUES)) > 0)


def test_overlay_schema_matches_v2_1() -> None:
    assert module.OUTPUT_COLUMNS == [
        "chemistry", "fold", "profile", "temperature_C", "source_file",
        "sweep_point", "q_soc", "open_loop", "q_vp", "q_h", "r_voltage",
        "oracle_steady_MAE_pct", "oracle_recovery_time_s", "oracle_residual_error_pp",
        "plus5_steady_MAE_pct", "plus5_recovery_time_s", "plus5_residual_error_pp",
        "minus5_steady_MAE_pct", "minus5_recovery_time_s", "minus5_residual_error_pp",
    ]
