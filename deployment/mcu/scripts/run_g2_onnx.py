#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
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
FEATURES = {"NMC": ("G4", "T6"), "LFP": ("G4", "T6", "T7")}
FOLDS = ("DST", "FUDS", "US06")
SEEDS = (0, 1, 2)
TEMPS = {"NMC": (0, 25, 45), "LFP": (-10, 0, 10, 20, 25, 30, 40, 50)}
BATCH = 2048


@dataclass(frozen=True)
class Entry:
    chemistry: str
    feature: str
    fold: str
    seed: int
    path: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_loader(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def all_entries() -> list[Entry]:
    return [
        Entry(chemistry, feature, fold, seed, PACKAGE_ROOTS[chemistry] / feature / fold / str(seed))
        for chemistry, features in FEATURES.items()
        for feature in features
        for fold in FOLDS
        for seed in SEEDS
    ]


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def run_ort(model: ort.InferenceSession, windows: np.ndarray) -> np.ndarray:
    chunks = []
    for start in range(0, len(windows), BATCH):
        value = np.ascontiguousarray(windows[start : start + BATCH], dtype=np.float32)
        chunks.append(model.run(["soc"], {"window": value})[0])
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def run_torch(model: torch.nn.Module, windows: torch.Tensor) -> np.ndarray:
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(windows), BATCH):
            chunks.append(model(windows[start : start + BATCH]).cpu())
    return torch.cat(chunks, dim=0).numpy().astype(np.float32, copy=False)


def graph_ops(path: Path) -> dict[str, int]:
    model = onnx.load(path)
    return dict(sorted(Counter(node.op_type for node in model.graph.node).items()))


def export_models(package, prefix: Path) -> tuple[Path, Path]:
    static_path = prefix.with_name(prefix.name + "_static.onnx")
    dynamic_path = prefix.with_name(prefix.name + "_dynamic.onnx")
    input_dim = len(package.config["channels"])
    example = torch.zeros((1, int(package.config["window"]), input_dim), dtype=torch.float32)
    common = {
        "input_names": ["window"],
        "output_names": ["soc"],
        "opset_version": 17,
        "dynamo": False,
        "do_constant_folding": True,
    }
    torch.onnx.export(package.model, (example,), static_path, **common)
    torch.onnx.export(
        package.model,
        (example,),
        dynamic_path,
        dynamic_axes={"window": {0: "batch"}, "soc": {0: "batch"}},
        **common,
    )
    for path in (static_path, dynamic_path):
        onnx.checker.check_model(onnx.load(path))
    return static_path, dynamic_path


def main() -> None:
    g1 = pd.read_csv(DROP / "golden_gate_results.csv")
    if len(g1) != 45 or set(g1["status"]) != {"PASS"}:
        raise SystemExit("G1 is not a complete 45/45 PASS; refusing ONNX export")

    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
    (DROP / "onnx").mkdir(exist_ok=True)
    loaders = {
        chemistry: load_loader(PACKAGE_ROOTS[chemistry] / "loader.py", f"_g2_{chemistry.lower()}")
        for chemistry in PACKAGE_ROOTS
    }
    manifest_rows = []
    parity_rows = []
    slice_rows = []
    failures = []
    for number, entry in enumerate(all_entries(), start=1):
        print(
            f"[G2] {number:02d}/45 {entry.chemistry} {entry.feature} "
            f"{entry.fold} seed={entry.seed}",
            flush=True,
        )
        package = loaders[entry.chemistry].load_package(entry.path)
        prefix = (
            DROP
            / "onnx"
            / f"{entry.chemistry.lower()}_{entry.feature.lower()}_"
            f"{entry.fold.lower()}_s{entry.seed}"
        )
        static_path, dynamic_path = export_models(package, prefix)
        shape = [1, int(package.config["window"]), len(package.config["channels"])]
        operations = graph_ops(static_path)
        (prefix.with_name(prefix.name + "_ops.json")).write_text(
            json.dumps(operations, indent=2) + "\n", encoding="utf-8"
        )
        for kind, path in (("static", static_path), ("dynamic_batch", dynamic_path)):
            manifest_rows.append(
                {
                    "chemistry": entry.chemistry,
                    "feature": entry.feature,
                    "fold": entry.fold,
                    "seed": entry.seed,
                    "onnx_file": str(path.relative_to(DROP)),
                    "sha256": sha256(path),
                    "opset": 17,
                    "static_shape": json.dumps(shape) if kind == "static" else "batch_dynamic",
                    "rewrites_applied": "none",
                    "graph_kind": kind,
                    "operations": json.dumps(operations, sort_keys=True),
                }
            )

        static_session = session(static_path)
        dynamic_session = session(dynamic_path)
        golden_file = (
            DROP
            / "golden_windows"
            / f"{entry.chemistry.lower()}_{entry.feature.lower()}_"
            f"{entry.fold.lower()}_s{entry.seed}.npz"
        )
        golden = np.load(golden_file)
        golden_windows = np.ascontiguousarray(golden["windows"], dtype=np.float32)
        golden_torch = np.ascontiguousarray(golden["torch_out"], dtype=np.float32)
        static_out = np.concatenate(
            [
                static_session.run(["soc"], {"window": window[None]})[0]
                for window in golden_windows
            ],
            axis=0,
        )
        dynamic_out_1 = run_ort(dynamic_session, golden_windows)
        dynamic_out_2 = run_ort(dynamic_session, golden_windows)
        bit_identical = np.array_equal(dynamic_out_1, dynamic_out_2)
        golden_max_pct = float(
            max(
                np.max(np.abs(static_out - golden_torch)),
                np.max(np.abs(dynamic_out_1 - golden_torch)),
            )
            * 100.0
        )

        all_deltas = []
        all_torch_pred = []
        all_onnx_pred = []
        all_labels = []
        for temperature in TEMPS[entry.chemistry]:
            source = (
                RECORD_ROOTS[entry.chemistry]
                / f"{temperature}C"
                / f"{entry.chemistry}_{temperature}C_{entry.fold}.csv"
            )
            frame = pd.read_csv(source)
            module = sys.modules[package.model.__class__.__module__]
            matrix, segments = module.build_feature_matrix(
                frame, package.config, package.r0_table
            )
            if segments != [(0, len(frame))]:
                raise AssertionError(f"Unexpected segment layout: {source}")
            scaled = np.ascontiguousarray(
                (matrix - package.mean) / package.std, dtype=np.float32
            )
            window = int(package.config["window"])
            windows = (
                torch.from_numpy(scaled)
                .unfold(0, window, 1)
                .permute(0, 2, 1)
                .contiguous()
            )
            torch_pred = run_torch(package.model, windows)[:, 0]
            onnx_pred = run_ort(dynamic_session, windows.numpy())[:, 0]
            labels = pd.to_numeric(frame["SOC_CC"], errors="raise").to_numpy(np.float64)[
                window - 1 :
            ]
            all_deltas.append(np.abs(torch_pred - onnx_pred))
            all_torch_pred.append(torch_pred)
            all_onnx_pred.append(onnx_pred)
            all_labels.append(labels)
            slice_rows.append(
                {
                    "chemistry": entry.chemistry,
                    "feature": entry.feature,
                    "fold": entry.fold,
                    "seed": entry.seed,
                    "temperature_C": temperature,
                    "mae_torch_pct": float(np.mean(np.abs(torch_pred - labels)) * 100.0),
                    "mae_onnx_pct": float(np.mean(np.abs(onnx_pred - labels)) * 100.0),
                }
            )

        delta = np.concatenate(all_deltas)
        torch_pred = np.concatenate(all_torch_pred)
        onnx_pred = np.concatenate(all_onnx_pred)
        labels = np.concatenate(all_labels)
        mae_torch = float(np.mean(np.abs(torch_pred - labels)) * 100.0)
        mae_onnx = float(np.mean(np.abs(onnx_pred - labels)) * 100.0)
        record_max_pct = float(np.max(delta) * 100.0)
        # Section 7 defines 1e-5 %SOC as an expectation, not a gate. The actual
        # G2 gate is deterministic execution plus the Section 6 golden-window
        # parity limit of 1e-6 in fraction units (1e-4 %SOC).
        record_expectation_met = record_max_pct < 1e-5
        status = "PASS" if bit_identical and golden_max_pct < 1e-4 else "FAIL"
        if status != "PASS":
            failures.append(
                f"{entry.chemistry}/{entry.feature}/{entry.fold}/s{entry.seed}: "
                f"golden_max_pct={golden_max_pct}, bit_identical={bit_identical}"
            )
        parity_rows.append(
            {
                "chemistry": entry.chemistry,
                "feature": entry.feature,
                "fold": entry.fold,
                "seed": entry.seed,
                "n_pred_samples": len(labels),
                "golden_max_abs_pct": golden_max_pct,
                "record_max_abs_pct": record_max_pct,
                "record_mean_abs_pct": float(np.mean(delta) * 100.0),
                "mae_torch_pct": mae_torch,
                "mae_onnx_pct": mae_onnx,
                "delta_mae_pct": mae_onnx - mae_torch,
                "bit_identical": bit_identical,
                "status": status,
                "record_expectation_met": record_expectation_met,
            }
        )

    write_csv(
        DROP / "onnx_export_manifest.csv",
        [
            "chemistry",
            "feature",
            "fold",
            "seed",
            "onnx_file",
            "sha256",
            "opset",
            "static_shape",
            "rewrites_applied",
            "graph_kind",
            "operations",
        ],
        manifest_rows,
    )
    write_csv(
        DROP / "onnx_parity_pc.csv",
        [
            "chemistry",
            "feature",
            "fold",
            "seed",
            "n_pred_samples",
            "golden_max_abs_pct",
            "record_max_abs_pct",
            "record_mean_abs_pct",
            "mae_torch_pct",
            "mae_onnx_pct",
            "delta_mae_pct",
            "bit_identical",
            "status",
            "record_expectation_met",
        ],
        parity_rows,
    )
    write_csv(
        DROP / "onnx_parity_slices.csv",
        [
            "chemistry",
            "feature",
            "fold",
            "seed",
            "temperature_C",
            "mae_torch_pct",
            "mae_onnx_pct",
        ],
        slice_rows,
    )

    slices = pd.DataFrame(slice_rows)
    aggregate_rows = []
    for (chemistry, feature), group in slices.groupby(["chemistry", "feature"], sort=True):
        torch_mae = float(group["mae_torch_pct"].mean())
        onnx_mae = float(group["mae_onnx_pct"].mean())
        aggregate_rows.append(
            {
                "chemistry": chemistry,
                "feature": feature,
                "agg_mae_torch_pct": torch_mae,
                "agg_mae_onnx_pct": onnx_mae,
                "agg_delta_pct": onnx_mae - torch_mae,
                "display_3dp_changed": f"{torch_mae:.3f}" != f"{onnx_mae:.3f}",
            }
        )
    write_csv(
        DROP / "onnx_parity_agg.csv",
        [
            "chemistry",
            "feature",
            "agg_mae_torch_pct",
            "agg_mae_onnx_pct",
            "agg_delta_pct",
            "display_3dp_changed",
        ],
        aggregate_rows,
    )
    if failures:
        raise SystemExit("G2 FAILED:\n" + "\n".join(failures))
    print("G2 PASS", flush=True)


if __name__ == "__main__":
    main()
