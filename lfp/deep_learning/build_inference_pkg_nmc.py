#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nmc_frozen_inference import build_feature_matrix


HERE = Path(__file__).resolve().parent
SOURCE = Path("/home/user/바탕화면/DL/CEMA_MLP_OCVSTART_FULLGRID").resolve()
RAW_ROOT = SOURCE / "nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean"
RESULTS = SOURCE / "nmc_goal_vcorr_it_train_dst_selector_results"
OUTPUT = HERE / "inference_pkg_nmc"
TEMPLATE = HERE / "nmc_frozen_inference.py"
TEMPERATURES = (0, 25, 45)
FOLDS = ("DST", "FUDS", "US06")
PROFILES = ("DST", "FUDS", "US06")
SEEDS = (0, 1, 2)
FEATURES = {
    "T6": {
        "source_name": "paper_t6_voltage_ema_all",
        "prefix": "s3_3l_rg_gru_t6_r_{fold}_s012_b2048_e200",
    },
    "G4": {
        "source_name": "paper_g4_all_ema",
        "prefix": "s3_3l_bm_gru_g4_r_{fold}_s012_b2048_e200",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: Path, root: Path) -> dict:
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def source_git() -> tuple[str | None, str]:
    result = subprocess.run(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return result.stdout.strip(), "available"
    return None, "unavailable_unborn_git_branch"


def json_write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_prefix(feature: str, fold: str) -> Path:
    name = FEATURES[feature]["prefix"].format(fold=fold.lower())
    return RESULTS / name


def paths_for_fold(fold: str) -> tuple[list[Path], list[Path]]:
    train = tuple(profile for profile in PROFILES if profile != fold)
    train_paths = [RAW_ROOT / f"{temp}C" / f"NMC_{temp}C_{profile}.csv" for temp in TEMPERATURES for profile in train]
    test_paths = [RAW_ROOT / f"{temp}C" / f"NMC_{temp}C_{fold}.csv" for temp in TEMPERATURES]
    return train_paths, test_paths


def prediction_path(prefix: Path, seed: int) -> Path:
    matches = sorted(prefix.parent.glob(prefix.name + f"_seed{seed}_*_test_prediction_rows.csv.gz"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one archived prediction file for {prefix.name} seed {seed}, got {matches}")
    return matches[0]


def load_channels(prefix: Path) -> list[str]:
    schema = pd.read_csv(str(prefix) + "_input_schema.csv")
    return schema.sort_values("index_1based")["feature_name"].astype(str).tolist()


def load_r0(prefix: Path, fold: str) -> dict:
    frame = pd.read_csv(str(prefix) + "_decomposition_params.csv").sort_values("temperature_C")
    train_profiles = [profile for profile in PROFILES if profile != fold]
    return {
        "method": "train_temperature_event_median",
        "fit_scope": "training_profiles_only",
        "fit_profiles": train_profiles,
        "source_file": Path(str(prefix) + "_decomposition_params.csv").name,
        "interpolation": "linear_with_endpoint_hold",
        "table": [
            {
                "temperature_C": float(row.temperature_C),
                "r0_ohm": float(row.r0_ohm),
                "n_events": int(row.n_events),
            }
            for row in frame.itertuples(index=False)
        ],
    }


def base_config(feature: str, fold: str, channels: list[str], weights_file: str) -> dict:
    train_profiles = [profile for profile in PROFILES if profile != fold]
    anchor_preferred = [
        "V_corr_raw", "V_eq_slow_raw", "T", "V_corr_raw_ema50",
        "V_corr_raw_dev_ema50", "V_corr_raw_ema200", "V_corr_raw_dev_ema200",
        "V_corr_raw_ema800", "V_corr_raw_dev_ema800", "dV_corr",
        "abs_dV_corr", "Vcorr_x_absI",
    ]
    return {
        "schema_version": 1,
        "feature": feature,
        "source_feature_set": FEATURES[feature]["source_name"],
        "fold": fold,
        "holdout_profile": fold,
        "train_profiles": train_profiles,
        "temperatures_C": list(TEMPERATURES),
        "channels": channels,
        "feature_ema_taus_samples": [50, 200, 800],
        "feature_ema_taus_by_signal": {
            "V_corr_raw": [50, 200, 800],
            "I_raw": [50, 200] if feature == "G4" else [],
            "absI": [50, 200] if feature == "G4" else [],
        },
        "vcorr_tau_s": 120.0,
        "window": 50,
        "strides": {"training": 3, "evaluation": 1},
        "eval_mask": {
            "scope": "each_record_independently",
            "prediction_end_indices": "window-1 through record_length-1 inclusive",
            "warmup_samples": 49,
            "additional_cutoff_or_mask": "none",
            "reset_behavior": "49 NaN outputs after each reset, then stride-1 predictions",
        },
        "sampling_period_s": {
            "nominal": 1.0,
            "time_column_priority": ["Step_Time(s)", "Test_Time(s)", "time_s"],
            "nonpositive_or_invalid_delta": "record median positive delta, else nominal",
        },
        "raw_columns": {"voltage": "Voltage(V)", "current": "Current(A)", "temperature": "TempLabel"},
        "perturbation_order": "raw V/I noise and bias, then R0 correction, then Vcorr EMA and feature EMAs",
        "model": "GRU anchor-residual sequence",
        "head": "residual",
        "hidden_size": 128,
        "layers": 1,
        "dropout": 0.06,
        "anchor_indices": [channels.index(name) for name in anchor_preferred if name in channels],
        "residual_limit_mode": "learnable",
        "inference_batch_size": 2048,
        "weights_file": weights_file,
    }


def scaler_for_fold(config: dict, r0: dict, train_paths: list[Path]) -> dict:
    matrices = []
    for path in train_paths:
        matrix, _ = build_feature_matrix(pd.read_csv(path), config, r0)
        # Source DataFrame.to_numpy() and concatenate yield column-major arrays;
        # float32 reduction order is part of the archived scaler numerics.
        matrices.append(np.asfortranarray(matrix))
    values = np.concatenate(matrices, axis=0)
    if not values.flags.f_contiguous:
        raise RuntimeError("Training scaler matrix must remain Fortran-contiguous")
    mean = np.nanmean(values, axis=0).astype(np.float32)
    std = np.nanstd(values, axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return {
        "method": "numpy_nanmean_nanstd_float32",
        "fit_scope": "training_profiles_only",
        "fit_profiles": config["train_profiles"],
        "fit_files": [path.name for path in train_paths],
        "columns": config["channels"],
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "epsilon": 1e-8,
        "source_numeric_layout": "Fortran-contiguous column-major",
    }


def verify_checkpoint(weights: Path, config: dict, feature: str, fold: str, seed: int) -> None:
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    expected = {
        "seed": seed,
        "recurrent": "gru",
        "model_kind": "anchor_residual_sequence",
        "feature_set": FEATURES[feature]["source_name"],
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"Checkpoint {weights.name}: {key}={checkpoint.get(key)!r}, expected {value!r}")
    if list(checkpoint.get("feature_cols", [])) != config["channels"]:
        raise RuntimeError(f"Checkpoint channel mismatch for {feature}/{fold}/{seed}")


def build() -> None:
    commit, git_status = source_git()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "__init__.py").write_text(
        '"""Frozen NMC inference package."""\n\nfrom .loader import load_package\n\n__all__ = ["load_package"]\n',
        encoding="utf-8",
    )
    (OUTPUT / "loader.py").write_text(
        """from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path


def load_package(path):
    entry = Path(path).resolve()
    source = entry / "code" / "inference.py"
    if not source.is_file():
        raise FileNotFoundError(f"Frozen inference code not found: {source}")
    name = "_nmc_frozen_" + hashlib.sha256(str(entry).encode("utf-8")).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen inference module: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.load_package(entry)
""",
        encoding="utf-8",
    )
    entries = []
    for feature in FEATURES:
        for fold in FOLDS:
            prefix = source_prefix(feature, fold)
            channels = load_channels(prefix)
            r0 = load_r0(prefix, fold)
            train_paths, test_paths = paths_for_fold(fold)
            weights0 = Path(str(prefix) + "_seed0_final_epoch200_weights.pt")
            config0 = base_config(feature, fold, channels, weights0.name)
            scaler = scaler_for_fold(config0, r0, train_paths)
            for seed in SEEDS:
                weights = Path(str(prefix) + f"_seed{seed}_final_epoch200_weights.pt")
                archived = prediction_path(prefix, seed)
                config = base_config(feature, fold, channels, weights.name)
                verify_checkpoint(weights, config, feature, fold, seed)
                entry = OUTPUT / feature / fold / str(seed)
                code_dir = entry / "code"
                code_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(weights, entry / weights.name)
                shutil.copy2(TEMPLATE, code_dir / "inference.py")
                (code_dir / "__init__.py").write_text(
                    "from .inference import FrozenPackage, load_package\n\n__all__ = ['FrozenPackage', 'load_package']\n",
                    encoding="utf-8",
                )
                json_write(entry / "config.json", config)
                json_write(entry / "scaler.json", scaler)
                json_write(entry / "r0_table.json", r0)
                source_files = [
                    weights,
                    archived,
                    Path(str(prefix) + "_metadata.json"),
                    Path(str(prefix) + "_input_schema.csv"),
                    Path(str(prefix) + "_decomposition_params.csv"),
                    *train_paths,
                    *test_paths,
                ]
                package_files = sorted(path for path in entry.rglob("*") if path.is_file() and path.name != "manifest.json")
                manifest = {
                    "schema_version": 1,
                    "entry": {"feature": feature, "fold": fold, "seed": seed},
                    "source_repo": str(SOURCE),
                    "source_git_commit": commit,
                    "source_git_status": git_status,
                    "source_artifacts": [file_info(path, SOURCE) for path in source_files],
                    "archived_predictions": str(archived.relative_to(SOURCE)),
                    "evaluation_records": [str(path.relative_to(SOURCE)) for path in test_paths],
                    "package_files": [file_info(path, entry) for path in package_files],
                }
                json_write(entry / "manifest.json", manifest)
                entries.append({"feature": feature, "fold": fold, "seed": seed, "path": str(entry.relative_to(OUTPUT))})
                print(f"[packaged] {feature}/{fold}/{seed}", flush=True)
    json_write(
        OUTPUT / "manifest.json",
        {
            "schema_version": 1,
            "entries": entries,
            "source_repo": str(SOURCE),
            "source_git_commit": commit,
            "source_git_status": git_status,
        },
    )


if __name__ == "__main__":
    build()
