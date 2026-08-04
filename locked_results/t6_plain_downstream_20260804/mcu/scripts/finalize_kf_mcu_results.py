#!/usr/bin/env python3
"""Validate and summarize the completed STM32 KF benchmark."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def retention_class(row: pd.Series) -> str:
    mean_diff = float(row["mean_abs_mcu_python_diff_pct_point"])
    max_diff = float(row["max_abs_mcu_python_diff_pct_point"])
    if max_diff <= 0.01:
        return "tight"
    if mean_diff <= 0.1 and max_diff <= 1.0:
        return "bounded_backend_sensitivity"
    return "material_backend_sensitivity"


def main() -> None:
    parity = pd.read_csv(ROOT / "kf_mcu_onchip_parity.csv")
    latency = pd.read_csv(ROOT / "kf_mcu_latency.csv")
    latency_raw = pd.read_csv(ROOT / "kf_mcu_latency_raw.csv")
    memory = pd.read_csv(ROOT / "kf_mcu_memory.csv")
    configs = pd.read_csv(ROOT / "kf_config_manifest.csv")
    failures = pd.read_csv(ROOT / "kf_mcu_failures.csv")
    summary = pd.read_csv(ROOT / "kf_mcu_summary.csv")

    assert len(configs) == 30, f"expected 30 configurations, found {len(configs)}"
    assert len(parity) == 165, f"expected 165 slices, found {len(parity)}"
    assert len(latency) == len(memory) == 30
    assert len(latency_raw) == 30 * 1024
    assert latency_raw.groupby("model_id").size().eq(1024).all()
    assert failures.empty, f"benchmark failures present:\n{failures}"
    assert parity["onchip_status"].eq("PASS").all()
    assert configs["build_status"].eq("PASS").all()
    assert configs["flash_status"].eq("PASS").all()

    retention = parity.copy()
    retention.insert(
        retention.columns.get_loc("onchip_status") + 1,
        "retention_class",
        retention.apply(retention_class, axis=1),
    )
    retention.to_csv(ROOT / "kf_mcu_accuracy_retention.csv", index=False)

    retention_summary = (
        retention.groupby(["chemistry", "method", "retention_class"], as_index=False)
        .size()
        .rename(columns={"size": "n_slices"})
    )
    retention_summary.to_csv(ROOT / "kf_mcu_accuracy_retention_summary.csv", index=False)

    nn = pd.read_csv(ROOT / "mcu_summary.csv")
    nn = nn[nn["pipeline"].eq("raw_vit_full_preprocessing_on_mcu")].copy()
    nn_rows = pd.DataFrame(
        {
            "chemistry": nn["chemistry"],
            "method": "proposed_" + nn["feature"],
            "family": "neural_network",
            "precision": nn["precision"].str.upper(),
            "initial_soc": "not_required",
            "input_signals": "V/I/T",
            "n_configurations": nn["n_checkpoints"],
            "median_latency_us": nn["latency_us_checkpoint_median"],
            "p95_latency_us": np.nan,
            "max_latency_us": nn["latency_us_checkpoint_max"],
            "flash_bytes": nn["flash_total_bytes"],
            "static_ram_bytes": nn["ram_total_bytes"],
            "stack_highwater_bytes": nn["stack_highwater_bytes"],
            "stored_asset_or_model_bytes": nn["flash_model_bytes"],
            "mcu_mae_pct": np.nan,
            "python_mae_pct": np.nan,
            "mae_delta_mcu_minus_python_pct": nn["onnx_delta_mae_pct"],
            "mean_abs_mcu_python_diff_pct_point": np.nan,
            "worst_max_abs_mcu_python_diff_pct_point": nn[
                "onchip_max_abs_diff_pct"
            ],
            "execution_status": nn["status"],
        }
    )

    kf_rows = pd.DataFrame(
        {
            "chemistry": summary["chemistry"],
            "method": summary["method"],
            "family": np.where(
                summary["method"].isin(["CC", "coulomb_count"]),
                "coulomb_counting",
                "kalman_filter",
            ),
            "precision": summary["precision"],
            "initial_soc": summary["initial_soc"],
            "input_signals": summary["online_sensor_signals"],
            "n_configurations": 3,
            "median_latency_us": summary["median_latency_us"],
            "p95_latency_us": summary["p95_latency_us"],
            "max_latency_us": np.nan,
            "flash_bytes": summary["max_flash_bytes"],
            "static_ram_bytes": summary["max_static_ram_bytes"],
            "stack_highwater_bytes": summary["max_stack_highwater_bytes"],
            "stored_asset_or_model_bytes": summary["max_stored_asset_bytes"],
            "mcu_mae_pct": summary["mcu_slice_mean_mae_pct"],
            "python_mae_pct": summary["python_slice_mean_mae_pct"],
            "mae_delta_mcu_minus_python_pct": summary[
                "mae_delta_mcu_minus_python_pct"
            ],
            "mean_abs_mcu_python_diff_pct_point": summary[
                "mean_abs_mcu_python_diff_pct_point"
            ],
            "worst_max_abs_mcu_python_diff_pct_point": summary[
                "worst_max_abs_mcu_python_diff_pct_point"
            ],
            "execution_status": "PASS",
        }
    )

    combined = pd.concat([nn_rows, kf_rows], ignore_index=True)
    combined = combined.sort_values(
        ["chemistry", "family", "median_latency_us", "method"]
    )
    combined.to_csv(ROOT / "mcu_all_methods_summary.csv", index=False)

    print(f"configurations={len(configs)}")
    print(f"slices={len(parity)}")
    print(f"failures={len(failures)}")
    print(retention_summary.to_string(index=False))


if __name__ == "__main__":
    main()
