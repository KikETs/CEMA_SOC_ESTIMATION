#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import TensorProto
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static


REPO = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = REPO / "nmc/kf/inference_pkg_nmc/G4/DST/0"
DEFAULT_DATA = REPO / "nmc/data/preprocessed/nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean"


def load_frozen_module(package: Path):
    source = package / "code/inference.py"
    spec = importlib.util.spec_from_file_location("frozen_nmc_smoke", source)
    if spec is None or spec.loader is None:
        raise ImportError(source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EquivalentUnrolledGRU(nn.Module):
    """Exact eval-mode expression of the frozen one-layer PyTorch GRU model."""

    def __init__(self, original: nn.Module, anchor_indices: list[int], window: int):
        super().__init__()
        self.input_proj = original.dynamic.input_proj
        self.gru = original.dynamic.rnn
        self.output_norm = original.dynamic.norm
        self.anchor_head = original.anchor_head
        self.residual_head = original.residual_head
        self.residual_limit_param = original.residual_limit_param
        self.register_buffer("anchor_indices", torch.tensor(anchor_indices, dtype=torch.long))
        self.window = int(window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.input_proj(x)
        hidden = torch.zeros(x.shape[0], self.gru.hidden_size, dtype=x.dtype, device=x.device)
        weight_ih = self.gru.weight_ih_l0
        weight_hh = self.gru.weight_hh_l0
        bias_ih = self.gru.bias_ih_l0
        bias_hh = self.gru.bias_hh_l0
        for step in range(self.window):
            gi = F.linear(projected[:, step, :], weight_ih, bias_ih)
            gh = F.linear(hidden, weight_hh, bias_hh)
            i_r, i_z, i_n = gi.chunk(3, dim=1)
            h_r, h_z, h_n = gh.chunk(3, dim=1)
            reset = torch.sigmoid(i_r + h_r)
            update = torch.sigmoid(i_z + h_z)
            candidate = torch.tanh(i_n + reset * h_n)
            hidden = candidate + update * (hidden - candidate)
        hidden = self.output_norm(hidden)
        anchor_input = x[:, -1, :].index_select(1, self.anchor_indices)
        anchor = torch.sigmoid(self.anchor_head(anchor_input))
        residual = self.residual_limit_param.to(hidden.dtype) * torch.tanh(self.residual_head(hidden))
        return (anchor + residual).clamp(0.0, 1.0)


class WindowCalibrationReader(CalibrationDataReader):
    def __init__(self, windows: np.ndarray):
        self.rows = iter([{"input": np.ascontiguousarray(row[None], dtype=np.float32)} for row in windows])

    def get_next(self):
        return next(self.rows, None)


def calibration_windows(module, package, data_root: Path, count: int) -> np.ndarray:
    windows = []
    window = int(package.config["window"])
    for profile in package.config["train_profiles"]:
        for temperature in package.config["temperatures_C"]:
            path = data_root / f"{int(temperature)}C/NMC_{int(temperature)}C_{profile}.csv"
            frame = pd.read_csv(path)
            features, segments = module.build_feature_matrix(frame, package.config, package.r0_table)
            scaled = np.ascontiguousarray((features - package.mean) / package.std, dtype=np.float32)
            for start, stop in segments:
                available = stop - start - window + 1
                if available <= 0:
                    continue
                picks = np.linspace(0, available - 1, min(8, available), dtype=int)
                windows.extend(scaled[start + pick : start + pick + window] for pick in picks)
    if not windows:
        raise RuntimeError("No training-only calibration windows found")
    rng = np.random.default_rng(20260713)
    order = rng.permutation(len(windows))[:count]
    return np.stack([windows[index] for index in order]).astype(np.float32)


def graph_summary(path: Path) -> dict:
    model = onnx.load(path)
    operations = Counter(node.op_type for node in model.graph.node)
    initializer_types = Counter(TensorProto.DataType.Name(item.data_type) for item in model.graph.initializer)
    initializer_bytes = sum(len(item.raw_data) for item in model.graph.initializer)
    return {
        "file": str(path),
        "size_bytes": path.stat().st_size,
        "operations": dict(sorted(operations.items())),
        "initializer_types": dict(sorted(initializer_types.items())),
        "initializer_raw_bytes": initializer_bytes,
    }


def run_fixed_batch_one(session: ort.InferenceSession, windows: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [session.run(None, {"input": np.ascontiguousarray(window[None], dtype=np.float32)})[0] for window in windows],
        axis=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "smoke_nmc_g4_dst_seed0")
    parser.add_argument("--calibration-windows", type=int, default=48)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    module = load_frozen_module(args.package)
    package = module.load_package(args.package)
    original = package.model.eval()
    equivalent = EquivalentUnrolledGRU(original, package.config["anchor_indices"], package.config["window"]).eval()
    calibration = calibration_windows(module, package, args.data_root, args.calibration_windows)

    with torch.inference_mode():
        tensor = torch.from_numpy(calibration[:8])
        reference = original(tensor).numpy()
        rewritten = equivalent(tensor).numpy()
    rewrite_max_abs_error = float(np.max(np.abs(reference - rewritten)))
    if rewrite_max_abs_error > 1e-6:
        raise RuntimeError(f"Unrolled GRU is not equivalent: max_abs_error={rewrite_max_abs_error}")

    fp32_path = args.output / "model_fp32.onnx"
    int8_path = args.output / "model_int8_qdq.onnx"
    example = torch.from_numpy(calibration[:1])
    torch.onnx.export(
        equivalent,
        (example,),
        fp32_path,
        input_names=["input"],
        output_names=["soc"],
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(fp32_path))

    fp32_session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    onnx_fp32 = run_fixed_batch_one(fp32_session, calibration[:8])
    fp32_max_abs_error = float(np.max(np.abs(reference - onnx_fp32)))
    if fp32_max_abs_error > 1e-5:
        raise RuntimeError(f"ONNX FP32 mismatch: max_abs_error={fp32_max_abs_error}")

    quantize_static(
        model_input=fp32_path,
        model_output=int8_path,
        calibration_data_reader=WindowCalibrationReader(calibration),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm"],
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    )
    onnx.checker.check_model(onnx.load(int8_path))
    int8_session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    int8_output = run_fixed_batch_one(int8_session, calibration[:8])

    report = {
        "package": str(args.package.resolve()),
        "calibration_scope": "training_profiles_only",
        "calibration_profiles": package.config["train_profiles"],
        "calibration_window_count": int(len(calibration)),
        "rewrite_max_abs_error": rewrite_max_abs_error,
        "onnx_fp32_max_abs_error": fp32_max_abs_error,
        "int8_vs_fp32_max_abs_error": float(np.max(np.abs(int8_output - onnx_fp32))),
        "int8_vs_fp32_mae": float(np.mean(np.abs(int8_output - onnx_fp32))),
        "fp32": graph_summary(fp32_path),
        "int8_qdq": graph_summary(int8_path),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
