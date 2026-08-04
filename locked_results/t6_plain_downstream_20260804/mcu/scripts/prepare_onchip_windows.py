#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


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
TEMPS = {"NMC": (0, 25, 45), "LFP": (-10, 0, 10, 20, 25, 30, 40, 50)}
TAKES = {"NMC": (22, 21, 21), "LFP": (8,) * 8}


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
    output_dir = DROP / "onchip_windows"
    output_dir.mkdir(exist_ok=True)
    loaders = {
        chemistry: load_module(root / "loader.py", f"_mcu_onchip_{chemistry.lower()}")
        for chemistry, root in PACKAGE_ROOTS.items()
    }
    rows = []
    for row in manifest.itertuples(index=False):
        package_path = (
            PACKAGE_ROOTS[row.chemistry] / row.feature / row.fold / str(row.seed)
        )
        package = loaders[row.chemistry].load_package(package_path)
        inference_module = sys.modules[package.model.__class__.__module__]
        selected_windows = []
        selected_indices = []
        selected_files = []
        for temperature, take in zip(TEMPS[row.chemistry], TAKES[row.chemistry]):
            source = (
                RECORD_ROOTS[row.chemistry]
                / f"{temperature}C"
                / f"{row.chemistry}_{temperature}C_{row.fold}.csv"
            )
            frame = pd.read_csv(source)
            matrix, segments = inference_module.build_feature_matrix(
                frame, package.config, package.r0_table
            )
            if segments != [(0, len(frame))]:
                raise AssertionError(f"Unexpected segments for {source}: {segments}")
            scaled = np.ascontiguousarray(
                (matrix - package.mean) / package.std, dtype=np.float32
            )
            window = int(package.config["window"])
            tensor = torch.from_numpy(scaled)
            windows = tensor.unfold(0, window, 1).permute(0, 2, 1)
            chosen = np.linspace(0, len(windows) - 1, num=take, dtype=np.int64)
            selected_windows.append(windows[chosen].contiguous().numpy())
            selected_indices.append(chosen + window - 1)
            selected_files.extend([source.name] * take)
        windows_np = np.ascontiguousarray(
            np.concatenate(selected_windows, axis=0), dtype=np.float32
        )
        if windows_np.shape[0] != 64:
            raise AssertionError(f"{row.model_id}: expected 64 windows, got {windows_np.shape}")
        destination = output_dir / f"{row.model_id}.npz"
        np.savez(
            destination,
            windows=windows_np,
            end_indices=np.concatenate(selected_indices),
            file_names=np.asarray(selected_files),
            selection_rule=np.asarray(
                "NMC: 22/21/21 by ascending temperature; "
                "LFP: 8 per temperature; np.linspace endpoints included"
            ),
        )
        rows.append(
            {
                "model": row.model_id,
                "chemistry": row.chemistry,
                "feature": row.feature,
                "fold": row.fold,
                "seed": row.seed,
                "n_windows": len(windows_np),
                "shape": json.dumps(list(windows_np.shape)),
                "file": str(destination.relative_to(DROP)),
            }
        )
        print(f"[onchip] {len(rows):02d}/45 {row.model_id} {windows_np.shape}", flush=True)
    pd.DataFrame(rows).to_csv(DROP / "onchip_windows_manifest.csv", index=False)


if __name__ == "__main__":
    main()
