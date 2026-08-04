#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


DROP = Path(__file__).resolve().parents[1]
REPO = DROP.parents[1]
PACKAGE_ROOTS = {
    "NMC": REPO / "nmc/kf/inference_pkg_nmc",
    "LFP": REPO / "lfp/deep_learning/inference_pkg_lfp",
}
FEATURES = {"NMC": ("G4", "T6"), "LFP": ("G4", "T6", "T7")}


def load_loader(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class StageProbe(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        index = torch.as_tensor(self.model.anchor_indices, device=x.device)
        anchor = torch.sigmoid(
            self.model.anchor_head(x.index_select(dim=2, index=index))
        )
        projected = self.model.dynamic.input_proj(x)
        gru, _ = self.model.dynamic.rnn(projected)
        normalized = self.model.dynamic.norm(gru)
        residual = self.model.residual_limit_param.to(normalized.dtype) * torch.tanh(
            self.model.residual_head(normalized)
        )
        final = (anchor + residual).clamp(0.0, 1.0)
        return (
            anchor[:, -1, :],
            projected[:, -1, :],
            gru[:, -1, :],
            normalized[:, -1, :],
            residual[:, -1, :],
            final[:, -1, :],
        )


def ort_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def main() -> None:
    out_dir = DROP / "onnx_diagnostics"
    out_dir.mkdir(exist_ok=True)
    rows = []
    output_names = [
        "anchor_last",
        "projected_last",
        "gru_last",
        "normalized_last",
        "residual_last",
        "soc",
    ]
    for chemistry, features in FEATURES.items():
        loader = load_loader(
            PACKAGE_ROOTS[chemistry] / "loader.py", f"_diag_{chemistry.lower()}"
        )
        for feature in features:
            package = loader.load_package(PACKAGE_ROOTS[chemistry] / feature / "DST" / "0")
            probe = StageProbe(package.model).eval()
            input_dim = len(package.config["channels"])
            window = int(package.config["window"])
            example = torch.zeros((1, window, input_dim), dtype=torch.float32)
            model_path = out_dir / f"{chemistry.lower()}_{feature.lower()}_dst_s0_probe.onnx"
            torch.onnx.export(
                probe,
                (example,),
                model_path,
                input_names=["window"],
                output_names=output_names,
                opset_version=17,
                dynamo=False,
                do_constant_folding=True,
            )
            onnx.checker.check_model(onnx.load(model_path))
            session = ort_session(model_path)
            golden_path = (
                DROP
                / "golden_windows"
                / f"{chemistry.lower()}_{feature.lower()}_dst_s0.npz"
            )
            windows = np.load(golden_path)["windows"].astype(np.float32, copy=False)
            torch_outputs = [[] for _ in output_names]
            ort_outputs = [[] for _ in output_names]
            with torch.inference_mode():
                for window_value in windows:
                    values = probe(torch.from_numpy(window_value[None]))
                    for index, value in enumerate(values):
                        torch_outputs[index].append(value.numpy())
                    values_ort = session.run(
                        output_names, {"window": np.ascontiguousarray(window_value[None])}
                    )
                    for index, value in enumerate(values_ort):
                        ort_outputs[index].append(value)
            for stage, torch_values, ort_values in zip(
                output_names, torch_outputs, ort_outputs
            ):
                torch_array = np.concatenate(torch_values, axis=0)
                ort_array = np.concatenate(ort_values, axis=0)
                delta = np.abs(torch_array - ort_array)
                scale = 100.0 if stage in {"anchor_last", "residual_last", "soc"} else 1.0
                rows.append(
                    {
                        "chemistry": chemistry,
                        "feature": feature,
                        "fold": "DST",
                        "seed": 0,
                        "stage": stage,
                        "unit": "%SOC" if scale == 100.0 else "activation",
                        "n_windows": len(windows),
                        "max_abs_delta": float(delta.max() * scale),
                        "mean_abs_delta": float(delta.mean() * scale),
                    }
                )
    with (DROP / "onnx_bisection_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "chemistry",
            "feature",
            "fold",
            "seed",
            "stage",
            "unit",
            "n_windows",
            "max_abs_delta",
            "mean_abs_delta",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote onnx_bisection_diagnostics.csv")


if __name__ == "__main__":
    main()
