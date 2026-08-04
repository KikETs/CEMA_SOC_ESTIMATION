#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from numba import njit


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_io import load_all  # noqa: E402
from src.v2_core import FoldParameterMap, OCVGrid, V2FilterResult, _grid_eval, run_v2_ekf  # noqa: E402


METHOD = "vff_hysteresis_2rc_ekf"
REFERENCE = "hysteresis_2rc_ekf"
PROPOSED = "proposed_17ch_g4_gru_residual"
INITIALS = (("oracle", 0.0), ("minus5pp", -5.0), ("minus10pp", -10.0))
BLOCK = 60
BOOTSTRAPS = 10_000
SEED = 20260803
RECOVERY_THRESHOLD_PP = 1.5
RECOVERY_HOLD_S = 60.0

# Fixed before held-out evaluation. Every combination is evaluated only on the
# two training profiles of the current outer fold.
VFF_GRID = {
    "lambda_min": (0.90, 0.95, 0.98, 0.99, 0.995),
    "nis_threshold": (1.0, 2.0, 4.0, 9.0),
    "nis_ema_beta": (0.0, 0.5, 0.9, 0.99),
}

FROZEN_FILES = (
    "src/data_io.py",
    "src/v2_core.py",
    "src/v2_1_core.py",
    "results_v2_1/qr_selection.csv",
    "results_v2_1/ecm_fit_quality.csv",
    "results_v2_1/r0_estimates.csv",
    "results_v2_1/prediction_rows_v2_1.csv.gz",
    "results_v2_2/main_table.csv",
    "artifacts/ocv_table.csv",
    "artifacts/ecm_parameter_map.csv",
    "configs/v2.yaml",
    "configs/v2_1.yaml",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_list_sha256(paths: list[str]) -> str:
    payload = "\n".join(sorted(paths)) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def frozen_hashes() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in FROZEN_FILES}


def load_context():
    v21 = yaml.safe_load((ROOT / "configs/v2_1.yaml").read_text())
    v2 = yaml.safe_load((ROOT / v21["v2_config"]).read_text())
    base = yaml.safe_load((ROOT / v21["base_config"]).read_text())
    base["inputs"]["prepared_root"] = str(ROOT / "inputs/prepared_data_ocv_discharge_soc")
    base["inputs"]["prepared_manifest"] = str(ROOT / "inputs/prepared_dataset_manifest.csv")
    base["inputs"]["proposed_results_root"] = str(ROOT / "inputs/proposed_results")
    trajectories = load_all(base)
    ocv = OCVGrid(pd.read_csv(ROOT / "artifacts/ocv_table.csv"), v2["slope_noise"])
    ecm = pd.read_csv(ROOT / "results_v2_1/ecm_fit_quality.csv")
    unique = ecm.groupby(["fold_holdout", "temperature_C"], as_index=False).first()
    pmap = FoldParameterMap(unique, pd.read_csv(ROOT / "artifacts/ecm_parameter_map.csv"))
    qr = (
        pd.read_csv(ROOT / "results_v2_1/qr_selection.csv")
        .sort_values("range_cycle")
        .groupby("fold_holdout")
        .tail(1)
    )
    selections = {
        row.fold_holdout: {
            key: float(getattr(row, key))
            for key in ("q_soc", "q_vp", "q_h", "r_voltage")
        }
        for row in qr.itertuples()
    }
    return base, v2, trajectories, ocv, ecm, pmap, selections


def gamma_for(ecm: pd.DataFrame, holdout: str, temp: float) -> float:
    row = ecm[(ecm.fold_holdout == holdout) & (ecm.temperature_C == float(temp))]
    if len(row) != 2:
        raise RuntimeError(f"Expected two training-record ECM rows for {holdout}/{temp:g}C")
    if row.gamma.nunique() != 1:
        raise RuntimeError(f"Gamma mismatch within {holdout}/{temp:g}C")
    return float(row.gamma.iloc[0])


@njit(cache=True)
def _vff_ekf_core(
    time_s, dt_s, current, voltage, temp, q_ref,
    temps, soc_grid, base_grid, hmag_grid, dbase_grid, dhmag_grid, discharge_slope_grid,
    r0, r1, r2, tau1, tau2, initial_soc, initial_h, gamma,
    q_soc, q_vp, q_h, r_voltage, slope_aware,
    s_ref, s_min, factor_min, factor_max,
    lambda_min, nis_threshold, nis_ema_beta,
):
    n = len(time_s)
    nstate = 4
    x = np.zeros(4)
    x[0] = min(1.0, max(0.0, initial_soc))
    x[3] = min(1.0, max(-1.0, initial_h))
    P = np.zeros((4, 4))
    P[0, 0] = 2.5e-3
    P[1, 1] = 2.5e-3
    P[2, 2] = 2.5e-3
    P[3, 3] = 0.25
    qdiag = np.array([q_soc, q_vp, q_vp, q_h])
    soc_out = np.empty(n)
    vhat = np.empty(n)
    innovation = np.empty(n)
    states = np.empty((n, 4))
    ptrace = np.empty(n)
    r_used = np.empty(n)
    lambda_used = np.empty(n)
    nis_used = np.empty(n)
    nis_ema = 1.0
    diverged_index = -1

    for k in range(n):
        if k > 0:
            dt = dt_s[k]
            ip = current[k - 1]
            x[0] = min(1.0, max(0.0, x[0] - ip * dt / (3600.0 * q_ref)))
            a1 = np.exp(-dt / max(tau1[k], dt + 1e-9))
            a2 = np.exp(-dt / max(tau2[k], dt + 1e-9))
            x[1] = a1 * x[1] + (1.0 - a1) * r1[k] * ip
            x[2] = a2 * x[2] + (1.0 - a2) * r2[k] * ip
            ah = np.exp(
                -max(gamma, 0.0) * abs(ip) * max(dt, 0.0)
                / (3600.0 * max(q_ref, 1e-12))
            )
            if abs(ip) > 1e-12:
                x[3] = min(1.0, max(-1.0, ah * x[3] + (1.0 - ah) * (-np.sign(ip))))
            fdiag = np.array([1.0, a1, a2, ah])
            for i in range(nstate):
                for j in range(nstate):
                    P[i, j] = fdiag[i] * P[i, j] * fdiag[j]
                P[i, i] += qdiag[i]

        base = _grid_eval(base_grid, temps, soc_grid, x[0], temp[k])
        hmag = _grid_eval(hmag_grid, temps, soc_grid, x[0], temp[k])
        predicted = base + hmag * x[3] - current[k] * r0[k] - x[1] - x[2]
        H = np.zeros(4)
        H[0] = min(10.0, max(-10.0, _grid_eval(dbase_grid, temps, soc_grid, x[0], temp[k])))
        H[0] += min(10.0, max(-10.0, _grid_eval(dhmag_grid, temps, soc_grid, x[0], temp[k]))) * x[3]
        H[1] = -1.0
        H[2] = -1.0
        H[3] = hmag

        factor = 1.0
        if slope_aware:
            slope = abs(_grid_eval(discharge_slope_grid, temps, soc_grid, x[0], temp[k]))
            factor = (s_ref / max(slope, s_min)) ** 2
            factor = min(factor_max, max(factor_min, factor))
        R = r_voltage * factor
        r_used[k] = R
        residual = voltage[k] - predicted

        ph = np.zeros(4)
        for i in range(nstate):
            for j in range(nstate):
                ph[i] += P[i, j] * H[j]
        S_nominal = R
        for i in range(nstate):
            S_nominal += H[i] * ph[i]
        if not np.isfinite(S_nominal) or S_nominal <= 1e-16:
            diverged_index = k
            break

        nis = residual * residual / S_nominal
        nis_ema = nis_ema_beta * nis_ema + (1.0 - nis_ema_beta) * nis
        forgetting = nis_threshold / max(nis_threshold, nis_ema)
        forgetting = min(1.0, max(lambda_min, forgetting))
        lambda_used[k] = forgetting
        nis_used[k] = nis_ema

        # Innovation-triggered fading memory: inflate only the prior covariance.
        # The state model, Q/R, ECM, OCV, and Joseph update remain frozen v2.2.
        for i in range(nstate):
            for j in range(nstate):
                P[i, j] /= forgetting

        ph[:] = 0.0
        for i in range(nstate):
            for j in range(nstate):
                ph[i] += P[i, j] * H[j]
        S = R
        for i in range(nstate):
            S += H[i] * ph[i]
        if not np.isfinite(S) or S <= 1e-16:
            diverged_index = k
            break
        K = ph / S
        for i in range(nstate):
            x[i] += K[i] * residual
        x[0] = min(1.0, max(0.0, x[0]))
        x[3] = min(1.0, max(-1.0, x[3]))

        A = np.eye(4)
        for i in range(nstate):
            for j in range(nstate):
                A[i, j] -= K[i] * H[j]
        AP = np.zeros((4, 4))
        newP = np.zeros((4, 4))
        for i in range(nstate):
            for j in range(nstate):
                for z in range(nstate):
                    AP[i, j] += A[i, z] * P[z, j]
        for i in range(nstate):
            for j in range(nstate):
                for z in range(nstate):
                    newP[i, j] += AP[i, z] * A[j, z]
                newP[i, j] += K[i] * K[j] * R
        for i in range(nstate):
            for j in range(nstate):
                P[i, j] = 0.5 * (newP[i, j] + newP[j, i])
            P[i, i] = min(1.0, max(1e-12, P[i, i]))

        states[k] = x
        soc_out[k] = x[0]
        vhat[k] = predicted
        innovation[k] = residual
        ptrace[k] = P[0, 0] + P[1, 1] + P[2, 2] + P[3, 3]
        finite = True
        for i in range(nstate):
            if not np.isfinite(x[i]) or not np.isfinite(P[i, i]):
                finite = False
        if not finite:
            diverged_index = k
            break

    if diverged_index >= 0:
        for k in range(diverged_index, n):
            soc_out[k] = np.nan
            vhat[k] = np.nan
            innovation[k] = np.nan
            ptrace[k] = np.nan
            r_used[k] = np.nan
            lambda_used[k] = np.nan
            nis_used[k] = np.nan
            for j in range(4):
                states[k, j] = np.nan
    return soc_out, vhat, innovation, states, ptrace, r_used, lambda_used, nis_used, diverged_index


@dataclass
class VFFRun:
    result: V2FilterResult
    forgetting_factor: np.ndarray
    nis_ema: np.ndarray


def run_vff(tr, holdout, ocv, pmap, noise, gamma, v2, delta_pp, params) -> VFFRun:
    soc0 = float(np.clip(tr.soc_ref[0] + delta_pp / 100.0, 0.0, 1.0))
    h0 = ocv.initial_h(tr, soc0)
    arrays = pmap.arrays(holdout, tr.temperature_series_C, bool(v2["fixes"]["f1_fold_training_ecm"]))
    slope = v2["slope_noise"]
    out = _vff_ekf_core(
        tr.time_s, tr.dt_s, tr.current_A, tr.voltage_V, tr.temperature_series_C, tr.q_ref_Ah,
        ocv.temperatures, ocv.soc_grid, ocv.base, ocv.hmag, ocv.dbase, ocv.dhmag,
        ocv.discharge_slope, *arrays, soc0, h0, gamma,
        noise["q_soc"], noise["q_vp"], noise["q_h"], noise["r_voltage"],
        bool(v2["fixes"]["f2_slope_aware_measurement_noise"]),
        float(slope["s_ref_mV_per_pctSOC"]) * 0.1,
        float(slope["s_min_mV_per_pctSOC"]) * 0.1,
        float(slope["factor_min"]), float(slope["factor_max"]),
        params["lambda_min"], params["nis_threshold"], params["nis_ema_beta"],
    )
    failed = int(out[-1])
    result = V2FilterResult(
        *out[:6], diverged=failed >= 0,
        divergence_reason="" if failed < 0 else f"non-finite covariance at index {failed}",
    )
    return VFFRun(result=result, forgetting_factor=out[6], nis_ema=out[7])


def recovery_time_s(soc_pred: np.ndarray, tr) -> float:
    error = 100.0 * np.abs(soc_pred - tr.soc_ref)
    good = np.isfinite(error) & (error < RECOVERY_THRESHOLD_PP)
    start = 0
    while start < len(good):
        while start < len(good) and not good[start]:
            start += 1
        end = start
        while end < len(good) and good[end]:
            end += 1
        if end > start and tr.time_s[end - 1] - tr.time_s[start] >= RECOVERY_HOLD_S:
            return float(tr.time_s[start] - tr.time_s[0])
        start = end + 1
    return float("nan")


def result_metrics(run: VFFRun, tr) -> dict[str, float]:
    mask = tr.evaluation_mask & np.isfinite(run.result.soc)
    if run.result.diverged or not mask.any():
        return {
            "mae_pct": 100.0, "rmse_pct": 100.0, "recovery_time_s": np.nan,
            "residual_error_1800s_pp": np.nan, "mean_lambda": np.nan,
            "inflated_fraction": np.nan, "diverged": True,
        }
    error = 100.0 * (run.result.soc[mask] - tr.soc_ref[mask])
    at1800 = int(np.argmin(np.abs((tr.time_s - tr.time_s[0]) - 1800.0)))
    return {
        "mae_pct": float(np.mean(np.abs(error))),
        "rmse_pct": float(np.sqrt(np.mean(error * error))),
        "recovery_time_s": recovery_time_s(run.result.soc, tr),
        "residual_error_1800s_pp": float(100.0 * (run.result.soc[at1800] - tr.soc_ref[at1800])),
        "mean_lambda": float(np.nanmean(run.forgetting_factor)),
        "inflated_fraction": float(np.mean(run.forgetting_factor < 1.0 - 1e-12)),
        "diverged": False,
    }


def rows_from_result(method: str, initial: str, tr, result: V2FilterResult, ocv: OCVGrid) -> pd.DataFrame:
    mask = tr.evaluation_mask
    idx = np.flatnonzero(mask)
    true = tr.soc_ref[mask]
    pred = result.soc[mask]
    slope = np.array([
        ocv.evaluate("dbase", float(soc), float(temp))
        for soc, temp in zip(true, tr.temperature_series_C[mask])
    ])
    plateau = np.where(np.abs(slope) < 0.10, "plateau", "edge")
    elapsed = tr.time_s[mask] - tr.time_s[0]
    time_bin = np.select(
        [elapsed <= 60, elapsed <= 300, elapsed <= 900, elapsed <= 1800, elapsed <= 3600],
        ["<=60", "60-300", "300-900", "900-1800", "1800-3600"],
        default=">3600",
    )
    return pd.DataFrame({
        "method": method, "initial_condition": initial, "profile": tr.profile,
        "temperature_C": tr.temperature_C, "end_index": idx, "elapsed_s": elapsed,
        "soc_true": true, "soc_pred": pred, "abs_error": np.abs(pred - true),
        "plateau_edge_region": plateau, "time_since_start_bin_s": time_bin,
        "diverged": result.diverged,
    })


def golden_gate(trajectories, ocv, ecm, pmap, selections, v2) -> pd.DataFrame:
    locked = pd.read_csv(ROOT / "results_v2_1/prediction_rows_v2_1.csv.gz")
    locked = locked[(locked.method == REFERENCE) & (locked.initial_condition == "oracle")]
    rows = []
    for tr in trajectories:
        gamma = gamma_for(ecm, tr.profile, tr.temperature_C)
        soc0 = float(tr.soc_ref[0])
        result = run_v2_ekf(
            tr, tr.profile, ocv, pmap, soc0, selections[tr.profile], gamma,
            v2["fixes"], v2["slope_noise"], True, initial_h=ocv.initial_h(tr, soc0),
        )
        idx = np.flatnonzero(tr.evaluation_mask)
        current_mae = float(100.0 * np.mean(np.abs(result.soc[idx] - tr.soc_ref[idx])))
        ref = locked[(locked.profile == tr.profile) & (locked.temperature_C == tr.temperature_C)]
        if not np.array_equal(ref.end_index.to_numpy(int), idx):
            raise AssertionError(f"Golden index mismatch {tr.profile}/{tr.temperature_C:g}C")
        locked_mae = float(100.0 * ref.abs_error.mean())
        rows.append({
            "profile": tr.profile, "temperature_C": tr.temperature_C,
            "locked_mae_pct": locked_mae, "rerun_mae_pct": current_mae,
            "abs_diff_pct": abs(current_mae - locked_mae), "tolerance_pct": 1e-6,
            "pass": abs(current_mae - locked_mae) <= 1e-6,
        })
    frame = pd.DataFrame(rows)
    aggregate = {
        "profile": "AGGREGATE", "temperature_C": np.nan,
        "locked_mae_pct": float(frame.locked_mae_pct.mean()),
        "rerun_mae_pct": float(frame.rerun_mae_pct.mean()),
        "abs_diff_pct": float(frame.abs_diff_pct.max()), "tolerance_pct": 1e-6,
        "pass": bool(frame["pass"].all()),
    }
    frame = pd.concat([frame, pd.DataFrame([aggregate])], ignore_index=True)
    frame.to_csv(OUT / "golden_gate.csv", index=False)
    if not bool(aggregate["pass"]):
        raise RuntimeError(f"Golden gate failed: max difference {aggregate['abs_diff_pct']:.9g}%")
    return frame


def grid_candidates():
    keys = tuple(VFF_GRID)
    for values in itertools.product(*(VFF_GRID[key] for key in keys)):
        yield dict(zip(keys, map(float, values)))


def select_vff(holdout, trajectories, ocv, ecm, pmap, selections, v2):
    training = [tr for tr in trajectories if tr.profile != holdout]
    held_out = [tr for tr in trajectories if tr.profile == holdout]
    training_files = sorted(str(tr.path) for tr in training)
    rows = []
    for trial_id, params in enumerate(grid_candidates()):
        details = []
        for tr in training:
            gamma = gamma_for(ecm, holdout, tr.temperature_C)
            for init_name, delta in INITIALS:
                run = run_vff(tr, holdout, ocv, pmap, selections[holdout], gamma, v2, delta, params)
                metrics = result_metrics(run, tr)
                details.append((init_name, metrics))
        objective = float(np.mean([item[1]["mae_pct"] for item in details]))
        rows.append({
            "fold_holdout": holdout, "trial_id": trial_id, "selected": False,
            "objective_training_slice_unweighted_mae_pct": objective,
            "oracle_training_mae_pct": float(np.mean([m["mae_pct"] for init, m in details if init == "oracle"])),
            "minus5pp_training_mae_pct": float(np.mean([m["mae_pct"] for init, m in details if init == "minus5pp"])),
            "minus10pp_training_mae_pct": float(np.mean([m["mae_pct"] for init, m in details if init == "minus10pp"])),
            "minus5pp_recovered_fraction": float(np.mean([np.isfinite(m["recovery_time_s"]) for init, m in details if init == "minus5pp"])),
            "minus10pp_recovered_fraction": float(np.mean([np.isfinite(m["recovery_time_s"]) for init, m in details if init == "minus10pp"])),
            "diverged_runs": int(sum(bool(m["diverged"]) for _, m in details)),
            "training_profiles": "+".join(sorted({tr.profile for tr in training})),
            "training_files_sha256": file_list_sha256(training_files),
            "n_training_records": len(training), "n_training_runs": len(details),
            "heldout_files_passed_to_selection": False,
            "heldout_file_count": len(held_out),
            **params,
        })
    table = pd.DataFrame(rows)
    best_index = table.sort_values(
        ["objective_training_slice_unweighted_mae_pct", "oracle_training_mae_pct", "trial_id"]
    ).index[0]
    table.loc[best_index, "selected"] = True
    selected = {key: float(table.loc[best_index, key]) for key in VFF_GRID}
    return selected, table


def evaluate_test(trajectories, ocv, ecm, pmap, selections, v2, selected):
    condition_rows = []
    prediction_rows = []
    for tr in trajectories:
        params = selected[tr.profile]
        gamma = gamma_for(ecm, tr.profile, tr.temperature_C)
        for init_name, delta in INITIALS:
            run = run_vff(tr, tr.profile, ocv, pmap, selections[tr.profile], gamma, v2, delta, params)
            metrics = result_metrics(run, tr)
            condition_rows.append({
                "method": METHOD, "init": init_name,
                "condition": f"{tr.profile}_{tr.temperature_C:g}C",
                "profile": tr.profile, "temperature_C": tr.temperature_C,
                "aggregation": "condition", "mae": metrics["mae_pct"],
                "rmse": metrics["rmse_pct"], "recovery_time_s": metrics["recovery_time_s"],
                "recovered": bool(np.isfinite(metrics["recovery_time_s"])),
                "residual_error_1800s_pp": metrics["residual_error_1800s_pp"],
                "mean_lambda": metrics["mean_lambda"],
                "inflated_fraction": metrics["inflated_fraction"],
                "diverged": metrics["diverged"], "n_samples": int(tr.evaluation_mask.sum()),
                **params,
            })
            frame = rows_from_result(METHOD, init_name, tr, run.result, ocv)
            frame["forgetting_factor"] = run.forgetting_factor[frame.end_index.to_numpy(int)]
            frame["nis_ema"] = run.nis_ema[frame.end_index.to_numpy(int)]
            prediction_rows.append(frame)
    conditions = pd.DataFrame(condition_rows)
    reference_rows = []
    for tr in trajectories:
        gamma = gamma_for(ecm, tr.profile, tr.temperature_C)
        for init_name, delta in INITIALS:
            soc0 = float(np.clip(tr.soc_ref[0] + delta / 100.0, 0.0, 1.0))
            result = run_v2_ekf(
                tr, tr.profile, ocv, pmap, soc0, selections[tr.profile], gamma,
                v2["fixes"], v2["slope_noise"], True, initial_h=ocv.initial_h(tr, soc0),
            )
            mask = tr.evaluation_mask & np.isfinite(result.soc)
            error = 100.0 * (result.soc[mask] - tr.soc_ref[mask])
            recovery = recovery_time_s(result.soc, tr)
            at1800 = int(np.argmin(np.abs((tr.time_s - tr.time_s[0]) - 1800.0)))
            reference_rows.append({
                "method": REFERENCE, "init": init_name,
                "condition": f"{tr.profile}_{tr.temperature_C:g}C",
                "profile": tr.profile, "temperature_C": tr.temperature_C,
                "aggregation": "condition", "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error * error))),
                "recovery_time_s": recovery, "recovered": bool(np.isfinite(recovery)),
                "residual_error_1800s_pp": float(100.0 * (result.soc[at1800] - tr.soc_ref[at1800])),
                "mean_lambda": 1.0, "inflated_fraction": 0.0,
                "diverged": result.diverged, "n_samples": int(mask.sum()),
                "lambda_min": np.nan, "nis_threshold": np.nan, "nis_ema_beta": np.nan,
            })
    conditions = pd.concat([conditions, pd.DataFrame(reference_rows)], ignore_index=True)
    aggregate = []
    for (method, init_name), group in conditions.groupby(["method", "init"], sort=False):
        aggregate.append({
            "method": method, "init": init_name, "condition": "ALL_24",
            "profile": "ALL", "temperature_C": np.nan,
            "aggregation": "slice_unweighted_primary", "mae": float(group.mae.mean()),
            "rmse": float(group.rmse.mean()),
            "recovery_time_s": float(group.recovery_time_s.median()) if group.recovery_time_s.notna().any() else np.nan,
            "recovered": float(group.recovered.mean()),
            "residual_error_1800s_pp": float(group.residual_error_1800s_pp.abs().mean()),
            "mean_lambda": float(group.mean_lambda.mean()),
            "inflated_fraction": float(group.inflated_fraction.mean()),
            "diverged": bool(group.diverged.any()), "n_samples": int(group.n_samples.sum()),
            "lambda_min": np.nan, "nis_threshold": np.nan, "nis_ema_beta": np.nan,
        })
    table = pd.concat([conditions, pd.DataFrame(aggregate)], ignore_index=True)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    table.to_csv(OUT / "main_table_rows.csv", index=False)
    predictions.to_csv(OUT / "prediction_rows.csv.gz", index=False, compression="gzip")
    return table, predictions


def circular_block_mean(diff: np.ndarray, starts: np.ndarray, block: int) -> np.ndarray:
    n = len(diff)
    extended = np.r_[diff, diff[: block - 1]]
    prefix = np.r_[0.0, np.cumsum(extended)]
    block_sums = prefix[np.arange(n) + block] - prefix[np.arange(n)]
    full = n // block
    rem = n % block
    total = block_sums[starts[:, :full]].sum(axis=1) if full else np.zeros(len(starts))
    if rem:
        rem_sums = prefix[np.arange(n) + rem] - prefix[np.arange(n)]
        total += rem_sums[starts[:, full]]
    return total / n


def bootstrap_vs_locked(predictions: pd.DataFrame, locked: pd.DataFrame) -> list[dict]:
    rng = np.random.default_rng(SEED)
    rows = []
    for init_name, _ in INITIALS:
        a = predictions[predictions.initial_condition == init_name]
        b = locked[(locked.method == REFERENCE) & (locked.initial_condition == init_name)]
        observed = []
        boot = np.zeros(BOOTSTRAPS)
        used = 0
        for (profile, temp), ag in a.groupby(["profile", "temperature_C"], sort=True):
            bg = b[(b.profile == profile) & (b.temperature_C == temp)].sort_values("end_index")
            ag = ag.sort_values("end_index")
            if not np.array_equal(ag.end_index.to_numpy(), bg.end_index.to_numpy()):
                raise AssertionError(f"Bootstrap alignment failed {init_name}/{profile}/{temp:g}C")
            diff = 100.0 * (ag.abs_error.to_numpy() - bg.abs_error.to_numpy())
            observed.append(float(diff.mean()))
            starts = rng.integers(0, len(diff), size=(BOOTSTRAPS, len(diff) // BLOCK + (len(diff) % BLOCK > 0)))
            boot += circular_block_mean(diff, starts, BLOCK)
            used += 1
        boot /= used
        rows.append({
            "method": METHOD, "reference_method": REFERENCE, "init": init_name,
            "weighting": "slice_unweighted", "mean_diff_mae_pct": float(np.mean(observed)),
            "ci_lo_pct": float(np.quantile(boot, 0.025)), "ci_hi_pct": float(np.quantile(boot, 0.975)),
            "block_samples": BLOCK, "B": BOOTSTRAPS, "seed": SEED,
            "n_records": used, "n_reference_seeds": 1,
        })
    return rows


def load_proposed_five_seeds() -> pd.DataFrame:
    roots = (
        ROOT / "inputs/proposed_results",
        Path("${SOURCE_WORKSPACE}/soc_paper_drop4_trajectories/LFP"),
    )
    frames = []
    for profile in ("DST", "FUDS", "US06"):
        found = []
        for seed in range(5):
            candidates = []
            for root in roots:
                candidates.extend(root.glob(f"*gru_residual_g4_holdout{profile.lower()}*seed{seed}*prediction_rows.csv.gz"))
            unique = sorted({str(path): path for path in candidates}.values(), key=str)
            if not unique:
                raise RuntimeError(f"Missing proposed seed {seed} for {profile}")
            # Prefer the locked all8 confirmatory naming if multiple copies exist.
            ranked = sorted(unique, key=lambda p: ("all8_train_all8_test" not in p.name, len(str(p))))
            path = ranked[0]
            frame = pd.read_csv(path, usecols=["temperature", "drive_cycle", "end_index", "y_true", "y_pred"])
            frame["seed"] = seed
            frame["source_path"] = str(path)
            frames.append(frame)
            found.append(seed)
        if found != list(range(5)):
            raise RuntimeError(f"Proposed seed set mismatch for {profile}: {found}")
    output = pd.concat(frames, ignore_index=True)
    output["abs_error"] = (output.y_pred - output.y_true).abs()
    slices = output.groupby(["seed", "drive_cycle", "temperature"], as_index=False).abs_error.mean()
    headline = float(100.0 * slices.groupby("seed").abs_error.mean().mean())
    if abs(headline - 0.541648071) > 1e-6:
        raise RuntimeError(f"Proposed five-seed headline mismatch: {headline:.9f}%")
    return output


def bootstrap_vs_proposed(predictions: pd.DataFrame, proposed: pd.DataFrame) -> list[dict]:
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for init_name, _ in INITIALS:
        kf = predictions[predictions.initial_condition == init_name]
        observed = []
        boot = np.zeros(BOOTSTRAPS)
        used = 0
        for (profile, temp), kg in kf.groupby(["profile", "temperature_C"], sort=True):
            pg = proposed[(proposed.drive_cycle == profile) & (proposed.temperature == temp)]
            kg = kg.sort_values("end_index")
            pivot = pg.pivot(index="end_index", columns="seed", values="y_pred")
            pivot = pivot.loc[kg.end_index.to_numpy()]
            truth = pg.groupby("end_index").y_true.first().loc[kg.end_index.to_numpy()].to_numpy()
            if not np.allclose(truth, kg.soc_true.to_numpy(), atol=2e-6):
                raise AssertionError(f"Proposed truth alignment failed {profile}/{temp:g}C")
            prop_abs = np.abs(pivot.to_numpy() - truth[:, None])
            diff = 100.0 * (kg.abs_error.to_numpy()[:, None] - prop_abs)
            observed.append(float(diff.mean()))
            starts = rng.integers(0, len(diff), size=(BOOTSTRAPS, len(diff) // BLOCK + (len(diff) % BLOCK > 0)))
            per_seed = np.zeros((BOOTSTRAPS, 5))
            for seed_col in range(5):
                per_seed[:, seed_col] = circular_block_mean(diff[:, seed_col], starts, BLOCK)
            boot += per_seed.mean(axis=1)
            used += 1
        boot /= used
        rows.append({
            "method": METHOD, "reference_method": PROPOSED, "init": init_name,
            "weighting": "per-seed_slice-unweighted", "mean_diff_mae_pct": float(np.mean(observed)),
            "ci_lo_pct": float(np.quantile(boot, 0.025)), "ci_hi_pct": float(np.quantile(boot, 0.975)),
            "block_samples": BLOCK, "B": BOOTSTRAPS, "seed": SEED + 1,
            "n_records": used, "n_reference_seeds": 5,
        })
    return rows


def write_manifest(before, after, base, selections, selection_table, proposed):
    training = {}
    for holdout, group in selection_table[selection_table.selected].groupby("fold_holdout"):
        row = group.iloc[0]
        training[holdout] = {
            "training_profiles": row.training_profiles.split("+"),
            "training_files_sha256": row.training_files_sha256,
            "heldout_files_passed_to_selection": False,
            "selected": selections[holdout],
            "objective": "equal mean of record MAE across oracle, minus5pp, minus10pp on 16 training records",
        }
    manifest = {
        "status": "PASS" if before == after else "FAIL_FROZEN_ASSET_MUTATION",
        "method": METHOD,
        "mechanism": {
            "name": "innovation_triggered_variable_forgetting_factor",
            "equation": "nis_ema=beta*previous+(1-beta)*innovation^2/S; lambda=clip(threshold/max(threshold,nis_ema),lambda_min,1); P_prior=P_prior/lambda",
            "scope": "prediction covariance only; frozen state equations, OCV, ECM, Q/R, slope-aware R, and Joseph update",
            "literature_claim": "same mechanism family only; not claimed as an exact reproduction of a named paper",
        },
        "protocol": {
            "profiles": base["protocol"]["profiles"],
            "temperatures_C": base["protocol"]["temperatures_C"],
            "initial_conditions": [name for name, _ in INITIALS],
            "aggregation": "24 fold-temperature slices equally weighted",
            "recovery": f"first |SOC error|<{RECOVERY_THRESHOLD_PP} pp interval sustained for {RECOVERY_HOLD_S:g} s",
            "bootstrap": {"circular_within_record": True, "block_samples": BLOCK, "B": BOOTSTRAPS},
        },
        "selection": {"grid": VFF_GRID, "per_fold_training_only": training},
        "proposed_prediction_sources": sorted(proposed.source_path.unique().tolist()),
        "frozen_hashes_before": before,
        "frozen_hashes_after": after,
        "inference_only_after_selection": True,
        "no_ecm_refit": True,
        "no_qr_retuning": True,
        "seed": SEED,
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if before != after:
        raise RuntimeError("Frozen v2.2 inputs changed during the run")


def write_hashes():
    names = [
        "run_adaptive_filter_baseline.py", "main_table_rows.csv", "prediction_rows.csv.gz",
        "paired_bootstrap.csv", "qr_or_adaptation_selection.csv", "golden_gate.csv",
        "run_manifest.json",
    ]
    lines = [f"{sha256(OUT / name)}  {name}" for name in names]
    (OUT / "sha256sums.txt").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = frozen_hashes()
    base, v2, trajectories, ocv, ecm, pmap, qr = load_context()
    gate = golden_gate(trajectories, ocv, ecm, pmap, qr, v2)
    print(f"golden PASS: max_diff={gate.abs_diff_pct.max():.3g}%", flush=True)

    selections = {}
    selection_frames = []
    for holdout in base["protocol"]["profiles"]:
        selected, frame = select_vff(holdout, trajectories, ocv, ecm, pmap, qr, v2)
        selections[holdout] = selected
        selection_frames.append(frame)
        print(f"selected {holdout}: {selected}", flush=True)
    selection_table = pd.concat(selection_frames, ignore_index=True)
    selection_table.to_csv(OUT / "qr_or_adaptation_selection.csv", index=False)

    main_table, predictions = evaluate_test(trajectories, ocv, ecm, pmap, qr, v2, selections)
    locked = pd.concat([
        pd.read_csv(ROOT / "results_v2_1/prediction_rows_v2_1.csv.gz"),
        pd.read_csv(ROOT / "results_v2_2/prediction_rows_minus10pp.csv.gz"),
    ], ignore_index=True)
    proposed = load_proposed_five_seeds()
    bootstrap = bootstrap_vs_locked(predictions, locked) + bootstrap_vs_proposed(predictions, proposed)
    pd.DataFrame(bootstrap).to_csv(OUT / "paired_bootstrap.csv", index=False)

    after = frozen_hashes()
    write_manifest(before, after, base, selections, selection_table, proposed)
    write_hashes()
    aggregates = main_table[main_table.aggregation == "slice_unweighted_primary"]
    print(aggregates[["init", "mae", "rmse", "recovery_time_s"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
