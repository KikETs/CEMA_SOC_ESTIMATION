#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch

sys.dont_write_bytecode = True


DROP = Path(__file__).resolve().parents[1]
CAMPAIGN = DROP.parent
PACKAGE_ROOTS = {
    "NMC": CAMPAIGN / "inference_pkg_nmc",
    "LFP": CAMPAIGN / "inference_pkg_lfp",
}
FOLDS = ("DST", "FUDS", "US06")
SEEDS = (0, 1, 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ort_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def run_batched(model, windows: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    pieces = []
    for start in range(0, len(windows), batch_size):
        value = np.ascontiguousarray(windows[start : start + batch_size], dtype=np.float32)
        pieces.append(model.run(["soc"], {"window": value})[0].reshape(-1))
    return np.concatenate(pieces)


def source_paths(package_path: Path) -> list[Path]:
    manifest = json.loads((package_path / "manifest.json").read_text(encoding="utf-8"))
    source_root = Path(manifest["source_root"])
    return [source_root / value for value in manifest["evaluation_records"]]


def main() -> None:
    gate = pd.read_csv(CAMPAIGN / "golden_gate_packages.csv")
    gate = gate[
        (gate["feature"] == "T6")
        & gate["seed"].isin(SEEDS)
        & gate["chemistry"].str.upper().isin(PACKAGE_ROOTS)
    ].copy()
    if len(gate) != 18 or not gate["passed"].astype(bool).all():
        raise SystemExit("Frozen-package golden gate is not 18/18 PASS")
    gate["status"] = "PASS"
    gate.to_csv(DROP / "golden_gate.csv", index=False)

    onnx_dir = DROP / "onnx"
    window_dir = DROP / "onchip_windows"
    onnx_dir.mkdir(exist_ok=True)
    window_dir.mkdir(exist_ok=True)
    loaders = {
        chemistry: load_module(root / "loader.py", f"_t6plain_mcu_{chemistry.lower()}")
        for chemistry, root in PACKAGE_ROOTS.items()
    }
    manifest_rows: list[dict] = []
    parity_rows: list[dict] = []
    slice_rows: list[dict] = []
    window_rows: list[dict] = []

    entries = [
        (chemistry, fold, seed)
        for chemistry in PACKAGE_ROOTS
        for fold in FOLDS
        for seed in SEEDS
    ]
    for number, (chemistry, fold, seed) in enumerate(entries, start=1):
        package_path = PACKAGE_ROOTS[chemistry] / "T6" / fold / str(seed)
        package = loaders[chemistry].load_package(package_path)
        package.model.cpu().eval()
        module = sys.modules[package.model.__class__.__module__]
        model_id = f"{chemistry.lower()}_t6plain_{fold.lower()}_s{seed}"
        destination = onnx_dir / f"{model_id}_static.onnx"
        dynamic_destination = onnx_dir / f"{model_id}_dynamic.onnx"
        example = torch.zeros((1, 50, 9), dtype=torch.float32)
        torch.onnx.export(
            package.model,
            (example,),
            destination,
            input_names=["window"],
            output_names=["soc"],
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
        )
        torch.onnx.export(
            package.model,
            (example,),
            dynamic_destination,
            input_names=["window"],
            output_names=["soc"],
            dynamic_axes={"window": {0: "batch"}, "soc": {0: "batch"}},
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
        )
        onnx.checker.check_model(onnx.load(destination))
        onnx.checker.check_model(onnx.load(dynamic_destination))
        session = ort_session(dynamic_destination)

        deltas = []
        torch_predictions = []
        onnx_predictions = []
        labels = []
        selected_windows = []
        selected_indices = []
        selected_files = []
        paths = source_paths(package_path)
        takes = (22, 21, 21) if chemistry == "NMC" else (8,) * len(paths)
        for source, take in zip(paths, takes):
            frame = pd.read_csv(source)
            matrix, segments = module.build_feature_matrix(
                frame, package.config, package.r0_table
            )
            if segments != [(0, len(frame))]:
                raise AssertionError(f"Unexpected segments for {source}: {segments}")
            scaled = np.ascontiguousarray(
                (matrix - package.mean) / package.std, dtype=np.float32
            )
            windows = (
                torch.from_numpy(scaled)
                .unfold(0, 50, 1)
                .permute(0, 2, 1)
                .contiguous()
            )
            with torch.inference_mode():
                torch_out = package.model(windows).numpy().reshape(-1)
            windows_np = windows.numpy()
            onnx_out = run_batched(session, windows_np)
            delta = np.abs(torch_out.astype(np.float64) - onnx_out.astype(np.float64))
            label = pd.to_numeric(frame["SOC_CC"], errors="raise").to_numpy(np.float64)[49:]
            deltas.append(delta)
            torch_predictions.append(torch_out)
            onnx_predictions.append(onnx_out)
            labels.append(label)
            temperature = package.config["temperatures_C"][len(labels) - 1]
            slice_rows.append(
                {
                    "chemistry": chemistry,
                    "feature": "T6PLAIN",
                    "fold": fold,
                    "seed": seed,
                    "temperature_C": temperature,
                    "mae_torch_pct": float(np.mean(np.abs(torch_out - label)) * 100.0),
                    "mae_onnx_pct": float(np.mean(np.abs(onnx_out - label)) * 100.0),
                    "max_abs_diff_pct": float(delta.max() * 100.0),
                }
            )
            chosen = np.linspace(0, len(windows_np) - 1, num=take, dtype=np.int64)
            selected_windows.append(windows_np[chosen])
            selected_indices.append(chosen + 49)
            selected_files.extend([source.name] * take)

        delta = np.concatenate(deltas)
        torch_pred = np.concatenate(torch_predictions)
        onnx_pred = np.concatenate(onnx_predictions)
        target = np.concatenate(labels)
        max_abs_pct = float(delta.max() * 100.0)
        status = "PASS" if max_abs_pct < 1e-3 else "FAIL"
        parity_rows.append(
            {
                "chemistry": chemistry,
                "feature": "T6PLAIN",
                "fold": fold,
                "seed": seed,
                "n_pred_samples": len(target),
                "record_max_abs_pct": max_abs_pct,
                "record_mean_abs_pct": float(delta.mean() * 100.0),
                "mae_torch_pct": float(np.mean(np.abs(torch_pred - target)) * 100.0),
                "mae_onnx_pct": float(np.mean(np.abs(onnx_pred - target)) * 100.0),
                "delta_mae_pct": float(
                    (np.mean(np.abs(onnx_pred - target)) - np.mean(np.abs(torch_pred - target)))
                    * 100.0
                ),
                "status": status,
            }
        )
        selected = np.ascontiguousarray(np.concatenate(selected_windows), dtype=np.float32)
        if selected.shape != (64, 50, 9):
            raise AssertionError(f"{model_id}: {selected.shape}")
        window_file = window_dir / f"{model_id}.npz"
        static_session = ort_session(destination)
        static_out = np.concatenate(
            [
                static_session.run(["soc"], {"window": window[None]})[0].reshape(-1)
                for window in selected
            ]
        )
        dynamic_out = run_batched(session, selected)
        static_max_pct = float(
            np.max(np.abs(static_out.astype(np.float64) - dynamic_out.astype(np.float64)))
            * 100.0
        )
        parity_rows[-1]["static_vs_dynamic_64_max_abs_pct"] = static_max_pct
        if static_max_pct >= 1e-3:
            parity_rows[-1]["status"] = "FAIL"
        np.savez(
            window_file,
            windows=selected,
            end_indices=np.concatenate(selected_indices),
            file_names=np.asarray(selected_files),
        )
        window_rows.append(
            {
                "model": model_id,
                "chemistry": chemistry,
                "feature": "T6PLAIN",
                "fold": fold,
                "seed": seed,
                "n_windows": 64,
                "shape": json.dumps(list(selected.shape)),
                "file": str(window_file.relative_to(DROP)),
            }
        )
        manifest_rows.append(
            {
                "chemistry": chemistry,
                "feature": "T6PLAIN",
                "fold": fold,
                "seed": seed,
                "onnx_file": str(destination.relative_to(DROP)),
                "sha256": sha256(destination),
                "opset": 17,
                "static_shape": json.dumps([1, 50, 9]),
                "rewrites_applied": "none",
                "graph_kind": "static",
            }
        )
        print(f"[ONNX] {number:02d}/18 {model_id} {status} max={max_abs_pct:.9g}%", flush=True)

    pd.DataFrame(manifest_rows).to_csv(DROP / "onnx_export_manifest.csv", index=False)
    pd.DataFrame(parity_rows).to_csv(DROP / "onnx_parity_pc.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(DROP / "onnx_parity_slices.csv", index=False)
    pd.DataFrame(window_rows).to_csv(DROP / "onchip_windows_manifest.csv", index=False)
    if {row["status"] for row in parity_rows} != {"PASS"}:
        raise SystemExit("ONNX parity failed")


if __name__ == "__main__":
    main()
