#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


DROP = Path(__file__).resolve().parents[1]
REPO = DROP.parents[1]
INSTRUCTION = DROP / "BENCHMARK_PROTOCOL.md"
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
ARCHIVE_ROOTS = {
    "NMC": REPO / "Data/ArchivedPredictions/NMC",
    "LFP": REPO / "Data/ArchivedPredictions/LFP",
}
FEATURES = {"NMC": ("G4", "T6"), "LFP": ("G4", "T6", "T7")}
FOLDS = ("DST", "FUDS", "US06")
SEEDS = (0, 1, 2)
TEMPS = {"NMC": (0, 25, 45), "LFP": (-10, 0, 10, 20, 25, 30, 40, 50)}
EXPECTED_PARAMS = {"G4": 120068, "T6": 119044, "T7": 118532}
EXPECTED_MAE = {
    ("NMC", "G4"): 0.328,
    ("LFP", "G4"): 0.535,
    ("LFP", "T6"): 0.620,
    ("LFP", "T7"): 0.796,
}


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


def command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        return result.stdout.strip()
    except OSError as error:
        return f"unavailable: {error}"


def load_loader(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def entries() -> list[Entry]:
    output = []
    for chemistry, features in FEATURES.items():
        for feature in features:
            for fold in FOLDS:
                for seed in SEEDS:
                    path = PACKAGE_ROOTS[chemistry] / feature / fold / str(seed)
                    if not path.is_dir():
                        raise FileNotFoundError(path)
                    output.append(Entry(chemistry, feature, fold, seed, path))
    if len(output) != 45:
        raise AssertionError(f"Expected 45 entries, found {len(output)}")
    return output


def record_path(chemistry: str, temperature: int, fold: str) -> Path:
    return RECORD_ROOTS[chemistry] / f"{temperature}C" / f"{chemistry}_{temperature}C_{fold}.csv"


def archive_path(entry: Entry) -> Path:
    seed_token = f"seed{entry.seed}_"
    if entry.chemistry == "NMC":
        stem = "bm_gru_g4" if entry.feature == "G4" else "rg_gru_t6"
        prefix = f"s3_3l_{stem}_r_{entry.fold.lower()}_"
    else:
        prefix = (
            "lfpconfirm_all8_train_all8_test_tier1_gru_residual_"
            f"{entry.feature.lower()}_holdout{entry.fold.lower()}_"
        )
    candidates = sorted(
        path
        for path in ARCHIVE_ROOTS[entry.chemistry].glob(f"{prefix}*prediction_rows.csv.gz")
        if seed_token in path.name
    )
    if len(candidates) != 1:
        raise RuntimeError(f"Archive match for {entry}: {candidates}")
    return candidates[0]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_environments() -> None:
    packages = []
    for name in ("numpy", "pandas", "torch", "onnx", "onnxruntime"):
        try:
            module = importlib.import_module(name)
            packages.append(f"{name}=={getattr(module, '__version__', 'unknown')}")
        except Exception as error:
            packages.append(f"{name}=unavailable ({error})")
    host = [
        f"execution_date={datetime.now().astimezone().isoformat()}",
        f"os={platform.platform()}",
        f"cpu={platform.processor() or command_output(['lscpu'])}",
        f"python={sys.version.replace(os.linesep, ' ')}",
        f"packages={';'.join(packages)}",
        f"instruction_file={INSTRUCTION}",
        f"instruction_sha256={sha256(INSTRUCTION)}",
        f"torch_threads={torch.get_num_threads()}",
        "deterministic_sampling=per-record np.linspace(..., dtype=int), 8 windows",
        f"stedgeai={command_output([os.environ.get('STEDGEAI', 'stedgeai'), '--version'])}",
    ]
    (DROP / "environment_host.txt").write_text("\n".join(host) + "\n", encoding="utf-8")

    compiler = command_output(
        [
            os.environ.get("ARM_GCC", "arm-none-eabi-gcc"),
            "--version",
        ]
    )
    programmer = command_output(
        [
            os.environ.get("STM32_PROGRAMMER_CLI", "STM32_Programmer_CLI"),
            "-c",
            "port=SWD",
            "freq=8000",
            "mode=UR",
            "-q",
        ]
    )
    mcu = [
        "board=NUCLEO-H563ZI",
        "mcu=STM32H563ZIT6",
        "core=Cortex-M33",
        "configured_clock_mhz=250",
        "measured_SystemCoreClock_mhz=PENDING_G3_UART_CONFIRMATION",
        "flash_total_bytes=2097152",
        "ram_total_bytes=655360",
        "fpu=FPv5-SP-D16 hard-float",
        "toolchain=STM32CubeIDE 2.2.0 GNU Arm",
        "compile_flags=-mcpu=cortex-m33 -Os -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb",
        "inference_runtime=ST Edge AI Core 4.0.1; STM32CubeAI 12.0.1-RC2",
        "firmware_package=STM32CubeH5 v1.7.0",
        "board_revision=NUCLEO-H563ZI revision not exposed by ST-LINK CLI",
        "power=USB; ST-LINK measured target voltage 3.28 V before G1",
        "programmer_probe_output_begin",
        programmer,
        "programmer_probe_output_end",
        "compiler_output_begin",
        compiler,
        "compiler_output_end",
    ]
    (DROP / "environment_mcu.txt").write_text("\n".join(mcu) + "\n", encoding="utf-8")


def checkpoint_manifest(all_entries: list[Entry]) -> None:
    rows = []
    for entry in all_entries:
        config = json.loads((entry.path / "config.json").read_text(encoding="utf-8"))
        weights = entry.path / config["weights_file"]
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        count = sum(int(value.numel()) for value in checkpoint["model_state_dict"].values())
        if count != EXPECTED_PARAMS[entry.feature]:
            raise AssertionError(
                f"{entry.chemistry}/{entry.feature}/{entry.fold}/{entry.seed}: "
                f"{count} != {EXPECTED_PARAMS[entry.feature]}"
            )
        rows.append(
            {
                "chemistry": entry.chemistry,
                "feature": entry.feature,
                "fold": entry.fold,
                "seed": entry.seed,
                "weights_file": str(weights),
                "sha256": sha256(weights),
                "param_count": count,
                "weights_provenance": "frozen_package",
            }
        )
    write_csv(
        DROP / "checkpoints_manifest.csv",
        [
            "chemistry",
            "feature",
            "fold",
            "seed",
            "weights_file",
            "sha256",
            "param_count",
            "weights_provenance",
        ],
        rows,
    )


def equally_spaced_indices(count: int, take: int = 8) -> np.ndarray:
    if count < take:
        raise ValueError(f"Need at least {take} windows, found {count}")
    return np.linspace(0, count - 1, num=take, dtype=np.int64)


def run_g1(all_entries: list[Entry]) -> None:
    loaders = {
        chemistry: load_loader(PACKAGE_ROOTS[chemistry] / "loader.py", f"_mcu_{chemistry.lower()}")
        for chemistry in PACKAGE_ROOTS
    }
    results = []
    window_manifest = []
    for number, entry in enumerate(all_entries, start=1):
        print(
            f"[G1] {number:02d}/45 {entry.chemistry} {entry.feature} "
            f"{entry.fold} seed={entry.seed}",
            flush=True,
        )
        package = loaders[entry.chemistry].load_package(entry.path)
        archive_file = archive_path(entry)
        archived = pd.read_csv(archive_file)
        archived = archived.loc[archived["file_name"].astype(str).str.upper().str.contains(entry.fold)]

        checkpoint_maes = []
        all_windows = []
        all_torch = []
        all_end_indices = []
        all_file_names = []
        max_archived_delta = 0.0
        max_label_delta = 0.0
        n_pred_samples = 0
        deterministic = True
        for temperature in TEMPS[entry.chemistry]:
            source = record_path(entry.chemistry, temperature, entry.fold)
            if not source.is_file():
                raise FileNotFoundError(source)
            frame = pd.read_csv(source)
            first = package.predict(frame)
            second = package.predict(frame)
            deterministic = deterministic and np.array_equal(first, second, equal_nan=True)
            valid = np.flatnonzero(np.isfinite(first))
            expected_valid = np.arange(int(package.config["window"]) - 1, len(frame))
            if not np.array_equal(valid, expected_valid):
                raise AssertionError(f"Eval mask mismatch: {source}")

            label = pd.to_numeric(frame["SOC_CC"], errors="raise").to_numpy(np.float64)
            checkpoint_maes.append(float(np.mean(np.abs(first[valid] - label[valid])) * 100.0))
            n_pred_samples += len(valid)

            file_rows = archived.loc[archived["file_name"] == source.name].copy()
            if len(file_rows) != len(valid):
                raise AssertionError(
                    f"Archived rows mismatch {archive_file.name}/{source.name}: "
                    f"{len(file_rows)} != {len(valid)}"
                )
            file_rows = file_rows.sort_values("end_index")
            archived_indices = file_rows["end_index"].to_numpy(np.int64)
            if not np.array_equal(archived_indices, valid):
                raise AssertionError(f"Archived index mismatch: {archive_file.name}/{source.name}")
            archived_pred = file_rows["y_pred"].to_numpy(np.float64)
            archived_true = file_rows["y_true"].to_numpy(np.float64)
            max_archived_delta = max(
                max_archived_delta, float(np.max(np.abs(first[valid] - archived_pred)))
            )
            max_label_delta = max(
                max_label_delta, float(np.max(np.abs(label[valid] - archived_true)))
            )

            features, segments = package.model.__class__.__module__, None
            module = sys.modules[package.model.__class__.__module__]
            matrix, segments = module.build_feature_matrix(
                frame, package.config, package.r0_table
            )
            if segments != [(0, len(frame))]:
                raise AssertionError(f"Unexpected segments for {source}: {segments}")
            scaled = np.ascontiguousarray(
                (matrix - package.mean) / package.std, dtype=np.float32
            )
            tensor = torch.from_numpy(scaled)
            windows = tensor.unfold(0, int(package.config["window"]), 1).permute(0, 2, 1)
            chosen = equally_spaced_indices(len(windows))
            selected = windows[chosen].contiguous()
            with torch.inference_mode():
                torch_out = package.model(selected).cpu().numpy().astype(np.float32)
            all_windows.append(selected.numpy().astype(np.float32, copy=False))
            all_torch.append(torch_out)
            all_end_indices.append(chosen + int(package.config["window"]) - 1)
            all_file_names.extend([source.name] * len(chosen))

        window_path = (
            DROP
            / "golden_windows"
            / f"{entry.chemistry.lower()}_{entry.feature.lower()}_"
            f"{entry.fold.lower()}_s{entry.seed}.npz"
        )
        window_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            window_path,
            windows=np.concatenate(all_windows, axis=0),
            torch_out=np.concatenate(all_torch, axis=0),
            end_indices=np.concatenate(all_end_indices, axis=0),
            file_names=np.asarray(all_file_names),
        )
        window_manifest.append(
            {
                "chemistry": entry.chemistry,
                "feature": entry.feature,
                "fold": entry.fold,
                "seed": entry.seed,
                "file": str(window_path.relative_to(DROP)),
                "sha256": sha256(window_path),
                "n_windows": sum(len(value) for value in all_windows),
                "shape": json.dumps(list(np.concatenate(all_windows, axis=0).shape)),
            }
        )
        status = (
            "PASS"
            if deterministic and max_archived_delta < 1e-6 and max_label_delta < 1e-6
            else "FAIL"
        )
        results.append(
            {
                "chemistry": entry.chemistry,
                "feature": entry.feature,
                "fold": entry.fold,
                "seed": entry.seed,
                "n_pred_samples": n_pred_samples,
                "max_abs_delta_vs_archived": max_archived_delta,
                "bit_identical": deterministic,
                "mae_anchor_check": float(np.mean(checkpoint_maes)),
                "status": status,
                "max_abs_label_delta": max_label_delta,
                "archive_file": str(archive_file),
            }
        )

    aggregates = {}
    for key, group in pd.DataFrame(results).groupby(["chemistry", "feature"], sort=True):
        aggregates[key] = float(group["mae_anchor_check"].mean())
    failures = []
    for row in results:
        key = (row["chemistry"], row["feature"])
        expected = EXPECTED_MAE.get(key)
        if expected is not None and abs(aggregates[key] - expected) > 0.002:
            row["status"] = "FAIL"
            failures.append(
                f"{key} aggregate MAE {aggregates[key]:.9f} outside {expected} +/- 0.002"
            )
        if row["status"] != "PASS":
            failures.append(
                f"{row['chemistry']}/{row['feature']}/{row['fold']}/s{row['seed']}"
            )

    write_csv(
        DROP / "golden_gate_results.csv",
        [
            "chemistry",
            "feature",
            "fold",
            "seed",
            "n_pred_samples",
            "max_abs_delta_vs_archived",
            "bit_identical",
            "mae_anchor_check",
            "status",
            "max_abs_label_delta",
            "archive_file",
        ],
        results,
    )
    write_csv(
        DROP / "golden_windows_manifest.csv",
        ["chemistry", "feature", "fold", "seed", "file", "sha256", "n_windows", "shape"],
        window_manifest,
    )
    aggregate_rows = [
        {
            "chemistry": chemistry,
            "feature": feature,
            "slice_unweighted_mae_pct": value,
            "expected_mae_pct": EXPECTED_MAE.get((chemistry, feature), ""),
            "within_declared_tolerance": (
                ""
                if (chemistry, feature) not in EXPECTED_MAE
                else abs(value - EXPECTED_MAE[(chemistry, feature)]) <= 0.002
            ),
        }
        for (chemistry, feature), value in sorted(aggregates.items())
    ]
    write_csv(
        DROP / "golden_gate_aggregate.csv",
        [
            "chemistry",
            "feature",
            "slice_unweighted_mae_pct",
            "expected_mae_pct",
            "within_declared_tolerance",
        ],
        aggregate_rows,
    )
    if failures:
        raise SystemExit("G1 FAILED:\n" + "\n".join(sorted(set(failures))))


def main() -> None:
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
    DROP.mkdir(parents=True, exist_ok=True)
    (DROP / "golden_windows").mkdir(exist_ok=True)
    write_environments()
    all_entries = entries()
    checkpoint_manifest(all_entries)
    run_g1(all_entries)
    print("G0/G1 PASS", flush=True)


if __name__ == "__main__":
    main()
