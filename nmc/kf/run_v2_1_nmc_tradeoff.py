#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.data_io import evaluation_mask, load_trajectory
from src.ecm_models import ECMParameters, run_coulomb_counting
from src.ekf import run_ekf
from src.ocv import OCVMap, TemperatureCurve


ROOT = Path(__file__).resolve().parent
FOLD = "DST"
Q_SOC_VALUES = np.logspace(-12, -4, 8)
OUTPUT_COLUMNS = [
    "chemistry", "fold", "profile", "temperature_C", "source_file",
    "sweep_point", "q_soc", "open_loop", "q_vp", "q_h", "r_voltage",
    "oracle_steady_MAE_pct", "oracle_recovery_time_s", "oracle_residual_error_pp",
    "plus5_steady_MAE_pct", "plus5_recovery_time_s", "plus5_residual_error_pp",
    "minus5_steady_MAE_pct", "minus5_recovery_time_s", "minus5_residual_error_pp",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_list(paths: list[Path]) -> str:
    payload = "\n".join(sorted(str(path.resolve()) for path in paths)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def saved_ocv_map(ecm_config: dict) -> OCVMap:
    curves_frame = pd.read_csv(ROOT / "results/ocv_curves.csv")
    sources = pd.read_csv(ROOT / "results/ocv_sources.csv")
    q_ref = dict(zip(sources.temperature_C.astype(float), sources.q_ref_Ah.astype(float)))
    curves = {}
    for temperature, frame in curves_frame.groupby("temperature_C"):
        temperature = float(temperature)
        ordered = frame.sort_values("soc")
        curves[temperature] = TemperatureCurve(
            temperature_C=temperature,
            soc=ordered.soc.to_numpy(np.float64),
            voltage=ordered.ocv_V.to_numpy(np.float64),
            source_file=str(sources.loc[sources.temperature_C == temperature, "source_file"].iloc[0]),
            q_ref_Ah=float(q_ref[temperature]),
        )
    return OCVMap(
        curves,
        slope_min=float(ecm_config["ocv"]["slope_min_V_per_soc"]),
        slope_max=float(ecm_config["ocv"]["slope_max_V_per_soc"]),
    )


def parameters_by_temperature() -> dict[float, ECMParameters]:
    frame = pd.read_csv(ROOT / "results/parameters/ecm_parameters.csv")
    frame = frame[(frame.fold == FOLD) & (frame.order == 2)].copy()
    if len(frame) != 3:
        raise RuntimeError(f"Expected three DST 2RC parameter rows, found {len(frame)}")
    return {
        float(row.temperature_C): ECMParameters(
            order=2,
            temperature_C=float(row.temperature_C),
            R0_ohm=float(row.R0_ohm),
            R1_ohm=float(row.R1_ohm),
            tau1_s=float(row.tau1_s),
            R2_ohm=float(row.R2_ohm),
            tau2_s=float(row.tau2_s),
            fit_bounds_name=str(row.fit_bounds_name),
            fit_voltage_rmse_V=float(row.fit_voltage_rmse_V),
        )
        for row in frame.itertuples(index=False)
    }


def selected_noise_by_temperature(ecm_config: dict) -> dict[float, dict]:
    selection = pd.read_csv(ROOT / "results/parameters/filter_noise_selection.csv")
    selected = selection[
        (selection.fold == FOLD)
        & (selection.method == "2RC_EKF")
        & selection.selected.astype(bool)
        & (selection.validation_profile == "CV_MEAN")
    ]
    candidates = {row["name"]: dict(row) for row in ecm_config["filter"]["noise_candidates"]}
    if len(selected) != 3:
        raise RuntimeError(f"Expected three selected DST 2RC-EKF noise rows, found {len(selected)}")
    result = {}
    for row in selected.itertuples(index=False):
        candidate = candidates[str(row.noise_name)]
        result[float(row.temperature_C)] = {
            "name": str(candidate["name"]),
            "q_soc_cycle1": float(candidate["q_soc"]),
            "q_vp": float(candidate["q_vp"]),
            "q_h": 0.0,
            "r_voltage": float(candidate["r_voltage"]),
        }
    return result


def recovery_time_s(elapsed_s: np.ndarray, reference: np.ndarray, estimate: np.ndarray) -> float:
    error_pp = 100.0 * np.abs(np.asarray(estimate) - np.asarray(reference))
    hit = np.flatnonzero(np.isfinite(error_pp) & (error_pp < 1.5) & (elapsed_s <= 1800.0))
    return float(elapsed_s[hit[0]]) if len(hit) else np.nan


def evaluate_soc(trajectory, soc: np.ndarray, eval_mask: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(soc)
    late = eval_mask & finite & (trajectory.elapsed_s >= 1800.0)
    if not late.any():
        late = eval_mask & finite
    steady = float(100.0 * np.mean(np.abs(soc[late] - trajectory.reference_soc[late])))
    return {
        "steady_MAE_pct": steady,
        "recovery_time_s": recovery_time_s(trajectory.elapsed_s, trajectory.reference_soc, soc),
        "residual_error_pp": steady,
    }


def verify_fit_scope(protocol: dict, test_paths: list[Path]) -> dict:
    manifest = pd.read_csv(ROOT / "results/fit_file_manifest.csv")
    fold_rows = manifest[manifest.fold == FOLD].copy()
    fit_paths = sorted({Path(path).resolve() for path in fold_rows.fit_file})
    test_resolved = {path.resolve() for path in test_paths}
    intersection = sorted(str(path) for path in set(fit_paths) & test_resolved)
    if intersection or not fold_rows.passed.astype(bool).all():
        raise RuntimeError(f"NMC fit-scope leakage audit failed: intersection={intersection}")
    expected_training = []
    for temperature in protocol["temperatures_C"]:
        for profile in protocol["folds"][FOLD]:
            expected_training.append(
                Path(protocol["data_root"]) / f"{int(float(temperature))}C" / f"NMC_{int(float(temperature))}C_{profile}.csv"
            )
    expected_training = sorted(path.resolve() for path in expected_training)
    if fit_paths != expected_training:
        raise RuntimeError("Fit manifest does not match the declared DST training-profile file list")
    return {
        "training_files": [str(path) for path in expected_training],
        "training_file_list_sha256": sha256_list(expected_training),
        "test_files": [str(path.resolve()) for path in test_paths],
        "test_file_list_sha256": sha256_list(test_paths),
        "train_test_intersection": intersection,
    }


def plot_tradeoff(frame: pd.DataFrame, output: Path) -> None:
    finite = frame[~frame.open_loop].groupby("q_soc", as_index=False).agg(
        oracle_MAE=("oracle_steady_MAE_pct", "mean"),
        plus5_recovery=("plus5_recovery_time_s", lambda x: np.nanmedian(x.fillna(1800.0))),
        minus5_recovery=("minus5_recovery_time_s", lambda x: np.nanmedian(x.fillna(1800.0))),
    )
    finite["worst_sign_recovery"] = finite[["plus5_recovery", "minus5_recovery"]].max(axis=1)
    open_loop = frame[frame.open_loop]
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.plot(finite.q_soc, finite.oracle_MAE, "o-", color="#2f6f9f")
    axis.set_xscale("log")
    axis.set_xlabel("q_soc")
    axis.set_ylabel("Oracle steady MAE (%)", color="#2f6f9f")
    recovery_axis = axis.twinx()
    recovery_axis.plot(finite.q_soc, finite.worst_sign_recovery, "s--", color="#c66b2b")
    recovery_axis.set_ylabel("Worst-sign median recovery (s; failures=1800)", color="#c66b2b")
    recovery_axis.set_ylim(0, 1900)
    open_mae = float(open_loop.oracle_steady_MAE_pct.mean())
    recovered = int(open_loop.minus5_recovery_time_s.notna().sum())
    axis.text(
        0.02, 0.97,
        f"Exact open-loop: oracle MAE={open_mae:.3f}%\n-5 pp recovered={recovered}/{len(open_loop)}",
        transform=axis.transAxes, va="top",
        bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.9},
    )
    axis.set_title("NMC DST: steady accuracy vs initial-error recovery")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    protocol = load_yaml(ROOT / "configs/protocol.yaml")
    ecm_config = load_yaml(ROOT / "configs/ecm.yaml")
    ocv_map = saved_ocv_map(ecm_config)
    parameters = parameters_by_temperature()
    noises = selected_noise_by_temperature(ecm_config)
    test_paths = [
        Path(protocol["data_root"]) / f"{int(float(temp))}C" / f"NMC_{int(float(temp))}C_{FOLD}.csv"
        for temp in protocol["temperatures_C"]
    ]
    scope_audit = verify_fit_scope(protocol, test_paths)

    inventory = pd.read_csv(ROOT / "results/data_inventory.csv")
    rows = []
    test_hashes = {}
    for path in test_paths:
        path = path.resolve()
        expected_hash = str(inventory.loc[inventory.file == str(path), "sha256"].iloc[0])
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Data hash drift: {path}")
        test_hashes[str(path)] = actual_hash
        trajectory = load_trajectory(path, float(protocol["current_convention"]["multiplier"]))
        temperature = float(trajectory.temperature_C)
        q_ref = float(ocv_map.q_ref_by_temperature[temperature])
        noise_base = noises[temperature]
        mask = evaluation_mask(
            len(trajectory.time_s),
            int(protocol["evaluation"]["first_end_index"]),
            int(protocol["evaluation"]["end_index_stride"]),
        )
        points = [(float(value), False) for value in Q_SOC_VALUES] + [(np.nan, True)]
        for q_soc, open_loop in points:
            metrics = {}
            for delta, prefix in ((0.0, "oracle"), (5.0, "plus5"), (-5.0, "minus5")):
                initial_soc = float(np.clip(trajectory.reference_soc[0] + delta / 100.0, 0.0, 1.0))
                if open_loop:
                    soc = run_coulomb_counting(trajectory, initial_soc, q_ref, eta=1.0)
                else:
                    use_noise = {
                        "q_soc": float(q_soc),
                        "q_vp": float(noise_base["q_vp"]),
                        "r_voltage": float(noise_base["r_voltage"]),
                    }
                    result = run_ekf(
                        trajectory, initial_soc, parameters[temperature], ocv_map, q_ref,
                        use_noise, ecm_config["filter"], eta=1.0,
                    )
                    if result.diverged:
                        raise RuntimeError(f"NMC EKF diverged: {path.name}, q_soc={q_soc}, delta={delta}")
                    soc = result.soc
                values = evaluate_soc(trajectory, soc, mask)
                for key, value in values.items():
                    metrics[f"{prefix}_{key}"] = value
            rows.append(
                {
                    "chemistry": "NMC",
                    "fold": FOLD,
                    "profile": trajectory.profile,
                    "temperature_C": temperature,
                    "source_file": str(path),
                    "sweep_point": "open_loop" if open_loop else f"q_soc={q_soc:.9g}",
                    "q_soc": q_soc,
                    "open_loop": bool(open_loop),
                    "q_vp": float(noise_base["q_vp"]),
                    "q_h": 0.0,
                    "r_voltage": float(noise_base["r_voltage"]),
                    **metrics,
                }
            )

    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if len(frame) != 27:
        raise RuntimeError(f"Expected 27 NMC trade-off rows, found {len(frame)}")
    frame.to_csv(output / "tradeoff_nmc.csv", index=False)
    plot_tradeoff(frame, output / "tradeoff_nmc.png")

    source_files = [
        ROOT / "configs/protocol.yaml", ROOT / "configs/ecm.yaml",
        ROOT / "results/ocv_curves.csv", ROOT / "results/ocv_sources.csv",
        ROOT / "results/parameters/ecm_parameters.csv",
        ROOT / "results/parameters/filter_noise_selection.csv",
        ROOT / "results/fit_file_manifest.csv", ROOT / "results/data_inventory.csv",
        ROOT / "src/data_io.py", ROOT / "src/ecm_models.py", ROOT / "src/ekf.py", ROOT / "src/ocv.py",
        Path(__file__).resolve(),
    ]
    audit = {
        "status": "PASS",
        "chemistry": "NMC",
        "representative_fold": FOLD,
        "model": "training-only per-temperature 2RC-EKF",
        "hysteresis_state": "not present in the NMC baseline; q_h fixed to 0.0",
        "q_soc_sweep": [float(value) for value in Q_SOC_VALUES],
        "exact_open_loop": "Coulomb-counting SOC; no SOC voltage update",
        "steady_definition": "existing evaluation mask intersect elapsed>=1800 s",
        "recovery_definition": "first |SOC error|<1.5 percentage points within 1800 s",
        "selected_noise_by_temperature": {str(key): value for key, value in noises.items()},
        "fit_scope": scope_audit,
        "test_file_sha256": test_hashes,
        "source_artifact_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_files},
        "output_rows": len(frame),
        "schema": OUTPUT_COLUMNS,
    }
    (output / "nmc_tradeoff_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    status = {
        "status": "COMPLETE",
        "source_root": str(ROOT),
        "representative_fold": FOLD,
        "output_rows": len(frame),
        "temperatures_C": sorted(frame.temperature_C.unique().tolist()),
        "q_h_note": "NMC model has no hysteresis state; q_h=0.0",
        "schema": OUTPUT_COLUMNS,
    }
    (output / "nmc_source_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
