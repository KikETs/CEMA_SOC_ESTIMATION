from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_METHODS = {
    "CC-oracle",
    "CC-realistic(+/-5% init)",
    "1RC-EKF",
    "2RC-EKF",
    "adaptive 2RC-EKF",
    "2RC-UKF",
    "proposed (EMA+NN)",
}
REQUIRED_COLUMNS = {
    "method",
    "runtime_signals",
    "stored_assets",
    "stored_assets_size_KB",
    "offline_prerequisites",
    "needs_initial_SOC",
    "initial_soc_consequence",
    "runtime_state_dim",
    "analytic_flops_per_step",
    "per_step_cost",
    "median_latency_us",
    "benchmark_steps",
    "runtime_memory_KB",
}


def test_every_deployment_method_row_is_complete():
    output = ROOT / "results" / "deployment_assets.csv"
    if not output.is_file():
        pytest.skip("generated deployment asset output is not present in a clean checkout")
    frame = pd.read_csv(output)
    assert REQUIRED_COLUMNS.issubset(frame.columns)
    assert set(frame["method"]) == EXPECTED_METHODS
    assert frame["method"].is_unique
    assert not frame[list(REQUIRED_COLUMNS)].isna().any().any()
    for column in ["runtime_signals", "stored_assets", "offline_prerequisites", "initial_soc_consequence", "per_step_cost"]:
        assert frame[column].astype(str).str.strip().ne("").all()
    for column in [
        "stored_assets_size_KB",
        "runtime_state_dim",
        "analytic_flops_per_step",
        "median_latency_us",
        "runtime_memory_KB",
    ]:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
        assert np.isfinite(values).all()
        assert (values > 0).all()
    assert (frame["benchmark_steps"] >= 100_000).all()
    assert frame["needs_initial_SOC"].isin(["yes", "no"]).all()
