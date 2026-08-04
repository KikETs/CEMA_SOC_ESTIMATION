#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE / "inference_pkg_nmc"
SOURCE = HERE.parent / "deep_learning"
RAW_ROOT = HERE.parent / "data/preprocessed/nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean"
ATOL = 1e-6


def load_entry(entry: Path):
    module_name = "frozen_" + "_".join(entry.relative_to(ROOT).parts)
    spec = importlib.util.spec_from_file_location(module_name, entry / "code" / "inference.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {entry}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.load_package(entry)


def validate() -> list[dict]:
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    index = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for item in index["entries"]:
        entry = ROOT / item["path"]
        package = load_entry(entry)
        manifest = package.manifest
        archive = pd.read_csv(SOURCE / manifest["archived_predictions"])
        max_error = 0.0
        compared = 0
        deterministic = True
        for record_relative in manifest["evaluation_records"]:
            record_path = SOURCE / record_relative
            frame = pd.read_csv(record_path)
            first = package.predict(frame)
            second = package.predict(frame)
            deterministic = deterministic and np.array_equal(first, second, equal_nan=True)
            expected = archive.loc[archive["file_name"].eq(record_path.name)].sort_values("end_index")
            indices = expected["end_index"].to_numpy(np.int64)
            if len(expected) != len(frame) - package.config["window"] + 1:
                raise AssertionError(f"Archived row count mismatch for {entry} / {record_path.name}")
            if not np.array_equal(indices, np.arange(package.config["window"] - 1, len(frame))):
                raise AssertionError(f"Archived end_index mask mismatch for {entry} / {record_path.name}")
            delta = np.abs(first[indices].astype(np.float64) - expected["y_pred"].to_numpy(np.float64))
            record_max = float(np.max(delta)) if len(delta) else 0.0
            max_error = max(max_error, record_max)
            compared += len(delta)
        passed = deterministic and max_error <= ATOL
        row = {
            **item,
            "samples_compared": compared,
            "max_abs_diff": max_error,
            "atol": ATOL,
            "deterministic_bit_identical": deterministic,
            "passed": passed,
        }
        rows.append(row)
        print(
            f"[golden] {item['feature']}/{item['fold']}/{item['seed']} "
            f"n={compared} max={max_error:.3e} deterministic={deterministic} pass={passed}",
            flush=True,
        )
        if not passed:
            raise AssertionError(f"Golden validation failed: {row}")
    pd.DataFrame(rows).to_csv(ROOT / "golden_test_results.csv", index=False)
    return rows


if __name__ == "__main__":
    validate()
