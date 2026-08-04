#!/usr/bin/env python3
"""Verify the numerical runtime required by the locked reproduction."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


NMC_EXPECTED = {
    "numpy": "2.1.3",
    "pandas": "2.2.3",
    "scipy": "1.15.3",
    "sklearn": "1.6.1",
}
LFP_KF_EXPECTED = {
    "numpy": "2.3.4",
    "pandas": "2.3.3",
    "scipy": "1.17.0",
    "numba": "0.66.0",
}


def blas_identity(numpy_module) -> tuple[str, str]:
    """Normalize NumPy BLAS metadata across conda-forge Linux and Windows builds."""
    blas = numpy_module.__config__.CONFIG.get("Build Dependencies", {}).get("blas", {})
    name, version = str(blas.get("name", "")), str(blas.get("version", ""))
    if name in {"blas", "lapack"}:
        records = Path(sys.prefix, "conda-meta").glob("libopenblas-*.json")
        record = next(records, None)
        if record:
            name = "openblas"
            version = record.name.removeprefix("libopenblas-").split("-", 1)[0]
    return name, version


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("nmc", "lfp-kf"), default="nmc")
    parser.add_argument("--require-mkl", action="store_true")
    parser.add_argument("--require-torch", action="store_true")
    parser.add_argument(
        "--torch-backend", choices=("auto", "cuda", "mps", "cpu"), default="auto"
    )
    args = parser.parse_args()

    failures: list[str] = []
    expected_python = (3, 13, 5) if args.profile == "nmc" else (3, 12, 0)
    if sys.version_info[:3] != expected_python:
        required = ".".join(map(str, expected_python))
        failures.append(f"Python {required} required, found {sys.version.split()[0]}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        failures.append("PYTHONNOUSERSITE=1 is required to isolate ~/.local packages")

    loaded = {}
    expected_packages = NMC_EXPECTED if args.profile == "nmc" else LFP_KF_EXPECTED
    for name, expected in expected_packages.items():
        module = importlib.import_module(name)
        loaded[name] = module
        actual = str(module.__version__)
        if actual != expected:
            failures.append(f"{name} {expected} required, found {actual}")

    blas_name, blas_version = blas_identity(loaded["numpy"])
    if args.require_mkl:
        if blas_name != "mkl-sdl" or blas_version != "2023.1":
            failures.append(
                "NumPy MKL 2023.1 build required for the paper UKF; "
                f"found {blas_name} {blas_version}"
            )
    if args.profile == "lfp-kf":
        if blas_name != "openblas" or blas_version != "0.3.30":
            failures.append(
                "OpenBLAS 0.3.30 LP64 build required for the historical LFP KF runtime; "
                f"found {blas_name} {blas_version}"
            )

    torch_status = "not checked"
    if args.require_torch:
        torch = importlib.import_module("torch")
        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available()
        preferred_backend = "cuda" if cuda_available else "mps" if mps_available else "cpu"
        torch_status = (
            f"{torch.__version__}, preferred backend={preferred_backend}, CUDA={torch.version.cuda}, "
            f"CUDA available={cuda_available}, MPS available={mps_available}"
        )
        if str(torch.__version__).split("+", 1)[0] != "2.9.1":
            failures.append(f"torch 2.9.1 required, found {torch.__version__}")
        if args.torch_backend == "cuda" and not cuda_available:
            failures.append("Torch CUDA backend required, but CUDA is unavailable")
        if args.torch_backend == "mps" and not mps_available:
            failures.append("Torch MPS backend required, but MPS is unavailable")

    print(f"Python: {sys.version.split()[0]}")
    print("Packages: " + ", ".join(f"{name}={module.__version__}" for name, module in loaded.items()))
    print(f"BLAS: {blas_name} {blas_version}")
    print(f"Torch: {torch_status}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("PASS: reproduction environment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
