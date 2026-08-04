# Validated FP32 ONNX models

This directory contains the frozen ONNX exports used by the MCU deployment
benchmark.

- 45 `*_static.onnx` files: fixed input shape `[1, 50, channels]`, used for
  STM32 deployment.
- 45 `*_dynamic.onnx` files: dynamic batch dimension, retained for portable
  host inference.
- Opset: 17.
- Scope: NMC G4/T6 and LFP G4/T6/T7, each with DST/FUDS/US06 holdouts and
  seeds 0, 1, and 2.

The authoritative filename, input-shape, graph-operation, and SHA-256 inventory
is
[`../results/evidence/onnx_export_manifest.csv`](../results/evidence/onnx_export_manifest.csv).
PC parity against the corresponding PyTorch packages is recorded in
[`../results/evidence/onnx_parity_slices.csv`](../results/evidence/onnx_parity_slices.csv).

These files contain model graphs and weights only. No raw or preprocessed
battery records are included.
