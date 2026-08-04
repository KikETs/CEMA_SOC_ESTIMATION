from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bootstrap_block_sensitivity import circular_block_bootstrap_sums


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PAIRS = {
    "proposed_vs_1RC-EKF",
    "proposed_vs_2RC-EKF",
    "proposed_vs_adaptive 2RC-EKF",
    "proposed_vs_2RC-UKF",
}
EXPECTED_SCOPES = {"pooled", "fold:US06", "fold:DST", "fold:FUDS"}


def test_circular_blocks_preserve_a_constant_record_sum():
    values = np.full(37, 2.5)
    sums = circular_block_bootstrap_sums(
        values,
        block_length=9,
        replicates=200,
        rng=np.random.default_rng(42),
    )
    np.testing.assert_allclose(sums, np.sum(values), atol=1e-12)


def test_bootstrap_sensitivity_output_is_complete():
    output = ROOT / "results" / "bootstrap_sensitivity.csv"
    if not output.is_file():
        pytest.skip("generated bootstrap sensitivity output is not present in a clean checkout")
    frame = pd.read_csv(output)
    required = {
        "pair",
        "scope",
        "block_len",
        "mean_diff",
        "ci_lo",
        "ci_hi",
        "acf_lag01_median",
    }
    assert set(frame.columns) == required
    assert len(frame) == 4 * 4 * 3
    assert set(frame["pair"]) == EXPECTED_PAIRS
    assert set(frame["scope"]) == EXPECTED_SCOPES
    assert set(frame["block_len"]) == {60, 300, 900}
    assert not frame.isna().any().any()
    assert (frame["ci_lo"] <= frame["mean_diff"]).all()
    assert (frame["mean_diff"] <= frame["ci_hi"]).all()
    assert (frame["acf_lag01_median"] >= 1).all()
    for _, group in frame.groupby(["pair", "scope"]):
        assert np.ptp(group["mean_diff"].to_numpy(float)) < 1e-12
        assert np.ptp(group["acf_lag01_median"].to_numpy(float)) < 1e-12


def test_report_contains_one_bootstrap_sensitivity_section():
    report = (ROOT / "report.md").read_text(encoding="utf-8")
    assert report.count("<!-- BOOTSTRAP_SENSITIVITY_START -->") == 1
    assert report.count("<!-- BOOTSTRAP_SENSITIVITY_END -->") == 1
