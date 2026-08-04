#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd


DROP = Path(__file__).resolve().parents[1]
REPO = DROP.parents[1]
PACKAGE_ROOTS = {
    "NMC": REPO / "nmc/kf/inference_pkg_nmc",
    "LFP": REPO / "lfp/deep_learning/inference_pkg_lfp",
}
RECORD_ROOTS = {
    "NMC": REPO
    / "Data/Preprocessed/NMC/"
    "nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean",
    "LFP": REPO / "Data/Preprocessed/LFP/prepared_data_ocv_discharge_soc",
}
PARITY_TEMPS = {"NMC": (0, 25, 45), "LFP": (-10, 25, 50)}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    manifest = pd.read_csv(DROP / "stedgeai_generate_manifest.csv")
    detail = pd.read_csv(DROP / "raw_vit_onchip_parity_rows.csv")
    aggregate = pd.read_csv(DROP / "raw_vit_onchip_parity.csv")
    loaders = {
        chemistry: load_module(
            root / "loader.py", f"_raw_onnx_ref_{chemistry.lower()}"
        )
        for chemistry, root in PACKAGE_ROOTS.items()
    }
    expected_map: dict[tuple[str, int, int], np.float32] = {}
    for row in manifest.itertuples(index=False):
        package = loaders[row.chemistry].load_package(
            PACKAGE_ROOTS[row.chemistry] / row.feature / row.fold / str(row.seed)
        )
        inference_module = sys.modules[package.model.__class__.__module__]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(DROP / "onnx" / f"{row.model_id}_static.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name
        for temperature in PARITY_TEMPS[row.chemistry]:
            source = (
                RECORD_ROOTS[row.chemistry]
                / f"{temperature}C"
                / f"{row.chemistry}_{temperature}C_{row.fold}.csv"
            )
            raw = pd.read_csv(source, usecols=["Voltage(V)", "Current(A)"]).iloc[:71]
            frame = pd.DataFrame(
                {
                    "Voltage(V)": raw["Voltage(V)"].to_numpy(np.float32),
                    "Current(A)": raw["Current(A)"].to_numpy(np.float32),
                    "T": np.full(71, temperature, dtype=np.float32),
                }
            )
            matrix, segments = inference_module.build_feature_matrix(
                frame, package.config, package.r0_table
            )
            if segments != [(0, 71)]:
                raise RuntimeError(f"{row.model_id}: {segments}")
            scaled = np.ascontiguousarray(
                (matrix - package.mean) / package.std, dtype=np.float32
            )
            for end_index in range(49, 71):
                value = session.run(
                    None, {input_name: scaled[end_index - 49 : end_index + 1][None]}
                )[0].reshape(-1)[0]
                expected_map[(row.model_id, temperature, end_index)] = np.float32(
                    value
                )

    expected = np.asarray(
        [
            expected_map[(row.model, int(row.temperature_C), int(row.sample_index))]
            for row in detail.itertuples(index=False)
        ],
        dtype=np.float32,
    )
    detail["expected_onnx_boundary_fraction"] = expected
    detail["abs_diff_pct"] = (
        np.abs(
            detail["mcu_fraction"].to_numpy(np.float64)
            - expected.astype(np.float64)
        )
        * 100.0
    )
    for model, group in detail.groupby("model"):
        mask = aggregate["model"] == model
        aggregate.loc[mask, "max_abs_diff_pct"] = group["abs_diff_pct"].max()
        aggregate.loc[mask, "mean_abs_diff_pct"] = group["abs_diff_pct"].mean()
        aggregate.loc[mask, "status"] = (
            "PASS" if group["abs_diff_pct"].max() < 1e-3 else "FAIL"
        )
    detail.to_csv(DROP / "raw_vit_onchip_parity_rows.csv", index=False)
    aggregate.to_csv(DROP / "raw_vit_onchip_parity.csv", index=False)
    if len(expected_map) != 45 * 3 * 22:
        raise SystemExit(f"Expected 2970 reference rows, got {len(expected_map)}")
    if not (aggregate["status"] == "PASS").all():
        raise SystemExit("Raw V/I/T ONNX-reference G3 failed")
    print(
        f"45/45 PASS; max={aggregate['max_abs_diff_pct'].max():.12g} %SOC"
    )


if __name__ == "__main__":
    main()
