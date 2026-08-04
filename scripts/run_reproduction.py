#!/usr/bin/env python3
"""Single command surface for preprocessing, paper DL runs, and KF runs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
COMMANDS = {
    "verify-env": [
        sys.executable,
        str(REPO / "scripts/verify_environment.py"),
        "--require-mkl",
        "--require-torch",
    ],
    "preprocess": [sys.executable, str(REPO / "scripts/prepare_data.py")],
    "verify-data": [sys.executable, str(REPO / "scripts/verify_preprocessing.py")],
    "nmc-dl": [sys.executable, str(REPO / "nmc/deep_learning/run_paper_t6_plain_10seed.py")],
    "lfp-dl": [sys.executable, str(REPO / "lfp/deep_learning/run_paper_t6_plain_10seed.py")],
    "nmc-kf": [
        sys.executable,
        str(REPO / "nmc/kf/run_all_lopo.py"),
        "--skip-neural-comparison",
        "--locked-parameters-dir",
        str(REPO / "nmc/kf/locked_parameters"),
        "--output-root",
        str(REPO / "runs/nmc_kf"),
    ],
    "nmc-kf-refit": [
        sys.executable,
        str(REPO / "nmc/kf/run_all_lopo.py"),
        "--skip-neural-comparison",
        "--output-root",
        str(REPO / "runs/nmc_kf_refit"),
    ],
    "lfp-kf": [sys.executable, str(REPO / "lfp/kf/run_paper_filters.py")],
    "verify-results": [sys.executable, str(REPO / "scripts/verify_paper_results.py")],
}
PREFLIGHTS = {
    "nmc-dl": ["--profile", "nmc", "--require-torch"],
    "lfp-dl": ["--profile", "nmc", "--require-torch"],
    "nmc-kf": ["--profile", "nmc", "--require-mkl"],
    "nmc-kf-refit": ["--profile", "nmc", "--require-mkl"],
    "lfp-kf": ["--profile", "lfp-kf"],
}


def conda_sibling_interpreter(env_name: str, current_prefix: Path | None = None) -> Path | None:
    """Return a sibling Conda environment interpreter on POSIX or Windows."""
    prefix = (current_prefix or Path(sys.prefix)).resolve()
    sibling = prefix.parent / env_name
    candidates = (sibling / "python.exe", sibling / "bin" / "python")
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def interpreter_for(stage: str) -> str:
    if stage != "lfp-kf":
        return sys.executable
    override = os.environ.get("CEMA_LFP_KF_PYTHON")
    if override:
        candidate = Path(override).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"CEMA_LFP_KF_PYTHON does not exist: {candidate}")
        return str(candidate)
    sibling = conda_sibling_interpreter("cema_soc_lfp_kf")
    return str(sibling) if sibling else sys.executable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stages", nargs="+", choices=tuple(COMMANDS) + ("all",))
    args, remainder = parser.parse_known_args()
    stages = list(COMMANDS) if "all" in args.stages else args.stages
    if remainder and len(stages) != 1:
        parser.error("extra stage arguments are supported only when running one stage")
    for stage in stages:
        interpreter = interpreter_for(stage)
        if stage in PREFLIGHTS:
            preflight = [
                interpreter,
                str(REPO / "scripts/verify_environment.py"),
                *PREFLIGHTS[stage],
            ]
            print(f"=== {stage} preflight: {' '.join(preflight)} ===", flush=True)
            subprocess.run(preflight, cwd=REPO, check=True)
        command = [interpreter, *COMMANDS[stage][1:]] + remainder
        print(f"=== {stage}: {' '.join(command)} ===", flush=True)
        subprocess.run(command, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
