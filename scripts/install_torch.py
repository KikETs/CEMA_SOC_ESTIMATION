#!/usr/bin/env python3
"""Install the locked PyTorch version for the current host and accelerator."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys


TORCH_VERSION = "2.9.1"
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def nvidia_driver_available() -> bool:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return False
    result = subprocess.run(
        [executable, "--query-gpu=name", "--format=csv,noheader"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def resolve_backend(
    requested: str,
    *,
    system: str | None = None,
    machine: str | None = None,
    has_nvidia: bool | None = None,
) -> str:
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    if requested != "auto":
        backend = requested
    elif system == "Darwin":
        backend = "mps" if machine in {"arm64", "aarch64"} else "cpu"
    else:
        backend = "cu128" if (nvidia_driver_available() if has_nvidia is None else has_nvidia) else "cpu"

    if backend == "mps" and not (system == "Darwin" and machine in {"arm64", "aarch64"}):
        raise RuntimeError("MPS requires Apple Silicon macOS")
    if backend == "cu128" and system not in {"Linux", "Windows"}:
        raise RuntimeError("The cu128 wheel is supported here only on Linux and Windows")
    return backend


def install_command(backend: str, *, system: str | None = None) -> list[str]:
    system = system or platform.system()
    command = [sys.executable, "-m", "pip", "install", f"torch=={TORCH_VERSION}"]
    if backend == "cu128":
        command.extend(["--index-url", CUDA_INDEX])
    elif backend == "cpu" and system != "Darwin":
        command.extend(["--index-url", CPU_INDEX])
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("auto", "cu128", "cpu", "mps"),
        default=os.environ.get("CEMA_TORCH_BACKEND", "auto"),
        help="Wheel/backend to install; auto detects NVIDIA or Apple Silicon MPS",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    backend = resolve_backend(args.backend)
    command = install_command(backend)
    print(f"Host: {platform.system()} {platform.machine()}")
    print(f"Selected backend: {backend}")
    print("Command: " + " ".join(command))
    if args.dry_run:
        return 0

    subprocess.run(command, check=True)
    check = (
        "import torch; "
        "print(f'Installed torch={torch.__version__}, CUDA={torch.version.cuda}, '"
        "f'CUDA available={torch.cuda.is_available()}, '"
        "f'MPS available={torch.backends.mps.is_available()}')"
    )
    subprocess.run([sys.executable, "-c", check], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
