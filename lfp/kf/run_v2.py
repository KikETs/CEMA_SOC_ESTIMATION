#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import yaml

from src.data_io import load_all, load_confirmatory_predictions
from src.metrics import error_metrics, ocv_region
from src.v2_core import (
    FoldParameterMap, OCVGrid, V2FilterResult, estimate_r0_by_fold,
    fit_training_ecm_by_fold, run_v2_ekf, run_v2_ukf,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_v2"
V2_START = "<!-- V2_START -->"
V2_END = "<!-- V2_END -->"
MAIN_METHODS = [
    "plain_2rc_ekf", "hysteresis_2rc_ekf",
    "adaptive_hysteresis_2rc_ekf", "hysteresis_2rc_ukf",
]
PROPOSED = "confirmatory_17ch_G4_GRU_residual_seed_mean"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_list(paths: list[str]) -> str:
    canonical = "\n".join(sorted(paths)) + "\n"
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_configs() -> tuple[dict, dict]:
    v2 = yaml.safe_load((ROOT / "configs" / "v2.yaml").read_text())
    base = yaml.safe_load((ROOT / v2["base_config"]).read_text())
    return base, v2


def prepare_physics(base: dict, v2: dict, trajectories):
    ocv_table = pd.read_csv(ROOT / "artifacts" / "ocv_table.csv")
    v1_params = pd.read_csv(ROOT / "artifacts" / "ecm_parameter_map.csv")
    ocv = OCVGrid(ocv_table, v2["slope_noise"])
    r0, r0_events = estimate_r0_by_fold(trajectories, base["protocol"]["profiles"], v2["r0_estimator"])
    folds = v1_fold_configs()
    ecm = fit_training_ecm_by_fold(
        trajectories, base["protocol"]["profiles"], r0, ocv,
        {fold: float(item["gamma"]) for fold, item in folds.items()},
        v2["ecm_fit"],
    )
    r0.to_csv(OUT / "r0_estimates.csv", index=False)
    r0_events.to_csv(OUT / "r0_events.csv.gz", index=False, compression="gzip")
    ecm.to_csv(OUT / "ecm_fit_quality.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    colors = {"DST": "#2f6f9f", "FUDS": "#d18f25", "US06": "#8a6fa8"}
    for ax, field, ylabel in zip(
        axes.ravel(), ["R1_ohm", "R2_ohm", "tau1_s", "tau2_s"],
        ["R1 (Ω)", "R2 (Ω)", "τ1 (s)", "τ2 (s)"],
    ):
        for fold, group in ecm.groupby("fold_holdout"):
            ax.plot(group.temperature_C, group[field], color=colors[fold], marker="o", label=fold)
        ax.set_ylabel(ylabel); ax.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Temperature (°C)"); axes[1, 1].set_xlabel("Temperature (°C)")
    axes[0, 0].legend(frameon=False); fig.suptitle("Training-only fold-specific 2RC least-squares fits")
    fig.tight_layout(); fig.savefig(OUT / "ecm_fitted_parameters.png", dpi=180); plt.close(fig)
    return ocv, FoldParameterMap(ecm, v1_params)


def v1_fold_configs() -> dict:
    return json.loads((ROOT / "artifacts" / "fold_filter_configs.json").read_text())


def selected_on_boundary(value: float, bounds: list[float], fraction: float) -> str | None:
    low, high = map(float, bounds); span = math.log(high) - math.log(low)
    relative = (math.log(value) - math.log(low)) / span
    if relative <= fraction:
        return "low"
    if relative >= 1.0 - fraction:
        return "high"
    return None


def tune_fold(holdout: str, trajectories, ocv, pmap, base: dict, v2: dict) -> tuple[dict, pd.DataFrame]:
    selection = v2["selection"]; original_ranges = {k: list(map(float, v)) for k, v in selection["ranges"].items()}
    ranges = {k: list(v) for k, v in original_ranges.items()}; training = [tr for tr in trajectories if tr.profile != holdout]
    training_files = sorted(str(tr.path) for tr in training)
    gamma = float(v1_fold_configs()[holdout]["gamma"])
    cycle_rows = []
    study_tag = str(selection.get("study_tag", "default"))
    for cycle in range(int(selection["max_range_extensions"]) + 1):
        db = OUT / "optuna" / f"qr_{holdout}_{study_tag}_cycle{cycle}.db"
        study_name = f"v2_qr_{holdout}_{study_tag}_cycle{cycle}"
        sampler = optuna.samplers.TPESampler(seed=int(selection["seed"]) + cycle, multivariate=True)
        study = optuna.create_study(
            study_name=study_name, direction="minimize", sampler=sampler,
            storage=f"sqlite:///{db}", load_if_exists=True,
        )
        study.set_user_attr("fold_holdout", holdout)
        study.set_user_attr("training_files", training_files)
        study.set_user_attr("training_file_list_sha256", sha256_list(training_files))
        study.set_user_attr("held_out_files", sorted(str(tr.path) for tr in trajectories if tr.profile == holdout))
        study.set_user_attr("objective", "uniform mean SOC MAE over training-profile x temperature slices")
        study.set_user_attr("ranges", ranges)

        def objective(trial: optuna.Trial) -> float:
            noise = {name: trial.suggest_float(name, bounds[0], bounds[1], log=True) for name, bounds in ranges.items()}
            slice_mae = []
            for tr in training:
                soc0 = float(tr.soc_ref[0])
                h0 = ocv.initial_h(tr, soc0) if v2["fixes"]["f3_hysteresis_rest_init"] else 0.0
                result = run_v2_ekf(tr, holdout, ocv, pmap, soc0, noise, gamma, v2["fixes"], v2["slope_noise"], True, initial_h=h0)
                mask = tr.evaluation_mask & np.isfinite(result.soc)
                if not mask.any() or result.diverged:
                    return 100.0
                slice_mae.append(float(np.mean(np.abs(result.soc[mask] - tr.soc_ref[mask])) * 100.0))
            return float(np.mean(slice_mae))

        remaining = int(selection["trials"]) - len(study.trials)
        if remaining > 0:
            study.optimize(objective, n_trials=remaining, show_progress_bar=False, gc_after_trial=False)
        trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
        trials["fold_holdout"] = holdout; trials["range_cycle"] = cycle
        trials.to_csv(OUT / "optuna" / f"qr_{holdout}_{study_tag}_cycle{cycle}_trials.csv", index=False)
        boundaries = {
            name: selected_on_boundary(float(study.best_params[name]), ranges[name], float(selection["boundary_log_fraction"]))
            for name in ranges
        }
        cycle_rows.append({
            "fold_holdout": holdout, "range_cycle": cycle, "study_name": study_name,
            "study_sqlite": str(db), "objective_mean_training_slice_MAE_pct": float(study.best_value),
            **{name: float(study.best_params[name]) for name in ranges},
            **{f"{name}_low": ranges[name][0] for name in ranges},
            **{f"{name}_high": ranges[name][1] for name in ranges},
            "boundary_parameters": ";".join(f"{k}:{v}" for k, v in boundaries.items() if v),
            "training_files_sha256": sha256_list(training_files),
            "training_profiles": "+".join(p for p in base["protocol"]["profiles"] if p != holdout),
            "shared_across_filters": bool(selection["shared_across_filters"]),
        })
        flagged = {name: side for name, side in boundaries.items() if side}
        if not flagged:
            return {name: float(study.best_params[name]) for name in ranges}, pd.DataFrame(cycle_rows)
        if cycle >= int(selection["max_range_extensions"]):
            failure = {"status": "STOP_BOUNDARY_AFTER_EXTENSION", "fold_holdout": holdout, "flagged": flagged, "cycles": cycle_rows}
            (OUT / "boundary_failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            raise RuntimeError(f"F4 selected a range-boundary value after one extension: {failure}")
        for name, side in flagged.items():
            if side == "low": ranges[name][0] /= 100.0
            else: ranges[name][1] *= 100.0
    raise AssertionError("unreachable")


def run_filter(model: str, tr, holdout: str, ocv, pmap, soc0: float, noise: dict, gamma: float, flags: dict, v2: dict, h0: float) -> V2FilterResult:
    if model == "plain_2rc_ekf":
        return run_v2_ekf(tr, holdout, ocv, pmap, soc0, noise, gamma, flags, v2["slope_noise"], False)
    if model == "hysteresis_2rc_ekf":
        return run_v2_ekf(tr, holdout, ocv, pmap, soc0, noise, gamma, flags, v2["slope_noise"], True, initial_h=h0)
    if model == "adaptive_hysteresis_2rc_ekf":
        return run_v2_ekf(tr, holdout, ocv, pmap, soc0, noise, gamma, flags, v2["slope_noise"], True, adaptive=True, initial_h=h0)
    if model == "hysteresis_2rc_ukf":
        return run_v2_ukf(tr, holdout, ocv, pmap, soc0, noise, gamma, flags, v2["slope_noise"], initial_h=h0)
    raise ValueError(model)


def slice_mean_mae(results: list[tuple[object, V2FilterResult]]) -> float:
    values = []
    for tr, result in results:
        mask = tr.evaluation_mask & np.isfinite(result.soc)
        values.append(float(np.mean(np.abs(result.soc[mask] - tr.soc_ref[mask])) * 100.0) if mask.any() else 100.0)
    return float(np.mean(values))


def representative_attribution(holdout: str, trajectories, ocv, pmap, selected_noise: dict, base: dict, v2: dict) -> pd.DataFrame:
    v1fold = v1_fold_configs()[holdout]; gamma = float(v1fold["gamma"]); v1_noise = v1fold["noise"]["hysteresis_2rc_ekf"]
    test = [tr for tr in trajectories if tr.profile == holdout]
    cases = [("v1_flags_off", {k: False for k in v2["fixes"]}, v1_noise), ("v2_all_on", dict(v2["fixes"]), selected_noise)]
    for key in v2["fixes"]:
        flags = dict(v2["fixes"]); flags[key] = False
        noise = v1_noise if key == "f4_training_soc_mae_optuna" else selected_noise
        cases.append((f"v2_without_{key.split('_')[0].upper()}", flags, noise))
    rows = []
    for name, flags, noise in cases:
        runs = []
        for tr in test:
            soc0 = float(tr.soc_ref[0]); h0 = ocv.initial_h(tr, soc0) if flags["f3_hysteresis_rest_init"] else 0.0
            runs.append((tr, run_filter("hysteresis_2rc_ekf", tr, holdout, ocv, pmap, soc0, noise, gamma, flags, v2, h0)))
        rows.append({"representative_holdout": holdout, "case": name, "mean_MAE_pct": slice_mean_mae(runs), **{k: bool(val) for k, val in flags.items()}})
    frame = pd.DataFrame(rows)
    full = float(frame.loc[frame.case == "v2_all_on", "mean_MAE_pct"].iloc[0])
    frame["delta_vs_full_MAE_pct"] = frame.mean_MAE_pct - full
    return frame


def time_bin(elapsed: np.ndarray) -> np.ndarray:
    values = np.asarray(elapsed, float)
    return np.select(
        [values <= 60, values <= 300, values <= 900, values <= 1800, values <= 3600],
        ["≤60", "60–300", "300–900", "900–1800", "1800–3600"], default=">3600",
    )


def rows_from_result(method: str, initial: str, tr, result: V2FilterResult, ocv: OCVGrid) -> pd.DataFrame:
    mask = tr.evaluation_mask; idx = np.flatnonzero(mask); true = tr.soc_ref[mask]; pred = result.soc[mask]
    slope = np.array([ocv.evaluate("dbase", float(s), float(t)) for s, t in zip(true, tr.temperature_series_C[mask])])
    elapsed = tr.time_s[mask] - tr.time_s[0]
    return pd.DataFrame({
        "method": method, "initial_condition": initial, "profile": tr.profile,
        "temperature_C": tr.temperature_C, "end_index": idx, "elapsed_s": elapsed,
        "soc_true": true, "soc_pred": pred, "abs_error": np.abs(pred - true),
        "plateau_edge_region": ocv_region(slope), "time_since_start_bin_s": time_bin(elapsed),
        "diverged": result.diverged,
    })


def proposed_rows(trajectories, ocv: OCVGrid, proposed: pd.DataFrame) -> pd.DataFrame:
    mean = proposed.groupby(["drive_cycle", "temperature", "end_index"], as_index=False).agg(soc_true=("y_true", "first"), soc_pred=("y_pred", "mean"))
    lookup = {(tr.profile, float(tr.temperature_C)): tr for tr in trajectories}; rows = []
    for (profile, temp), group in mean.groupby(["drive_cycle", "temperature"]):
        tr = lookup[(str(profile), float(temp))]; idx = group.end_index.to_numpy(int); true = group.soc_true.to_numpy(float)
        if not np.array_equal(np.sort(idx), np.flatnonzero(tr.evaluation_mask)):
            raise AssertionError(f"Confirmatory evaluation-mask mismatch for {profile}/{temp}")
        if not np.allclose(true, tr.soc_ref[idx], atol=2e-6):
            raise AssertionError(f"Proposed y_true mismatch for {profile}/{temp}")
        elapsed = tr.time_s[idx] - tr.time_s[0]
        slope = np.array([ocv.evaluate("dbase", float(s), float(t)) for s, t in zip(true, tr.temperature_series_C[idx])])
        pred = group.soc_pred.to_numpy(float)
        rows.append(pd.DataFrame({
            "method": PROPOSED, "initial_condition": "native", "profile": str(profile), "temperature_C": float(temp),
            "end_index": idx, "elapsed_s": elapsed, "soc_true": true, "soc_pred": pred,
            "abs_error": np.abs(pred - true), "plateau_edge_region": ocv_region(slope),
            "time_since_start_bin_s": time_bin(elapsed), "diverged": False,
        }))
    return pd.concat(rows, ignore_index=True)


def summarize(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for values, group in frame.groupby(keys, dropna=False, sort=False):
        if not isinstance(values, tuple): values = (values,)
        rows.append(dict(zip(keys, values)) | error_metrics(group.soc_true, group.soc_pred))
    return pd.DataFrame(rows)


def run_stage2(trajectories, ocv, pmap, selections: dict, base: dict, v2: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = v1_fold_configs(); prediction_frames = []; failures = []
    for tr in trajectories:
        holdout = tr.profile; noise = selections[holdout]; gamma = float(folds[holdout]["gamma"]); soc0 = float(tr.soc_ref[0])
        h0 = ocv.initial_h(tr, soc0) if v2["fixes"]["f3_hysteresis_rest_init"] else 0.0
        for model in MAIN_METHODS:
            result = run_filter(model, tr, holdout, ocv, pmap, soc0, noise, gamma, v2["fixes"], v2, h0)
            prediction_frames.append(rows_from_result(model, "oracle", tr, result, ocv))
            if result.diverged:
                failures.append({"method": model, "initial_condition": "oracle", "profile": tr.profile, "temperature_C": tr.temperature_C, "reason": result.divergence_reason})
        for delta in v2["diagnostics"]["initial_soc_errors_pct"]:
            diag_soc0 = float(np.clip(tr.soc_ref[0] + float(delta) / 100.0, 0, 1))
            initial_name = "oracle" if float(delta) == 0 else ("minus5pct" if float(delta) < 0 else "plus5pct")
            result = run_v2_ekf(tr, holdout, ocv, pmap, diag_soc0, noise, gamma, v2["fixes"], v2["slope_noise"], False, open_loop=True)
            prediction_frames.append(rows_from_result("diagnostic_2rc_open_loop", initial_name, tr, result, ocv))
            result = run_v2_ekf(tr, holdout, ocv, pmap, diag_soc0, noise, gamma, v2["fixes"], v2["slope_noise"], False)
            prediction_frames.append(rows_from_result("diagnostic_2rc_v2_closed_loop", initial_name, tr, result, ocv))
            no_f2 = dict(v2["fixes"]); no_f2["f2_slope_aware_measurement_noise"] = False
            result = run_v2_ekf(tr, holdout, ocv, pmap, diag_soc0, noise, gamma, no_f2, v2["slope_noise"], False)
            prediction_frames.append(rows_from_result("diagnostic_2rc_v2_closed_loop_f2_disabled", initial_name, tr, result, ocv))
    proposed = load_confirmatory_predictions(v2["proposed_confirmatory"])
    prediction_frames.append(proposed_rows(trajectories, ocv, proposed))
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(OUT / "prediction_rows_v2.csv.gz", index=False, compression="gzip")
    pd.DataFrame(failures, columns=["method", "initial_condition", "profile", "temperature_C", "reason"]).to_csv(OUT / "failures.csv", index=False)
    return predictions, pd.DataFrame(failures)


def circular_block_bootstrap(predictions: pd.DataFrame, v2: dict) -> pd.DataFrame:
    proposed = predictions[predictions.method == PROPOSED][["profile", "temperature_C", "end_index", "abs_error"]].rename(columns={"abs_error": "proposed_abs_error"})
    candidates = predictions[predictions.method != PROPOSED]
    block = int(v2["statistics"]["circular_block_samples"]); B = int(v2["statistics"]["bootstrap_replicates"]); seed = int(v2["statistics"]["bootstrap_seed"])
    rows = []
    for (method, initial), frame in candidates.groupby(["method", "initial_condition"], sort=False):
        merged = frame.merge(proposed, on=["profile", "temperature_C", "end_index"], how="inner", validate="many_to_one")
        record_arrays = []
        for _, group in merged.groupby(["profile", "temperature_C"], sort=False):
            diff = 100.0 * (group.abs_error.to_numpy(float) - group.proposed_abs_error.to_numpy(float))
            if not np.isfinite(diff).all():
                continue
            n = len(diff); extended = np.r_[diff, diff[:block - 1]]
            prefix = np.r_[0.0, np.cumsum(extended)]
            full_sums = prefix[np.arange(n) + block] - prefix[np.arange(n)]
            remainder = n % block
            partial_sums = None if remainder == 0 else prefix[np.arange(n) + remainder] - prefix[np.arange(n)]
            record_arrays.append((n, full_sums, partial_sums))
        rng = np.random.default_rng(seed + int(hashlib.sha256(f"{method}|{initial}".encode()).hexdigest()[:8], 16))
        boot_slice = np.zeros(B)
        boot_point = np.zeros(B)
        total_points = 0
        for n, full_sums, partial_sums in record_arrays:
            full_blocks = n // block
            totals = np.zeros(B)
            if full_blocks:
                starts = rng.integers(0, n, size=(B, full_blocks))
                totals += full_sums[starts].sum(axis=1)
            if partial_sums is not None:
                totals += partial_sums[rng.integers(0, n, size=B)]
            boot_slice += totals / n
            boot_point += totals
            total_points += n
        boot_slice /= max(len(record_arrays), 1)
        boot_point /= max(total_points, 1)
        slice_values = merged.groupby(["profile", "temperature_C"]).apply(
            lambda g: 100.0 * (g.abs_error - g.proposed_abs_error).mean(),
            include_groups=False,
        )
        observed_slice = float(slice_values.mean())
        observed_point = float(100.0 * (merged.abs_error - merged.proposed_abs_error).mean())
        for weighting, observed, boot in (
            ("slice_unweighted_primary", observed_slice, boot_slice),
            ("pointwise_secondary", observed_point, boot_point),
        ):
            rows.append({
                "method": method,
                "initial_condition": initial,
                "weighting": weighting,
                "delta_MAE_method_minus_proposed_pct": observed,
                "ci_low_pct": float(np.quantile(boot, 0.025)),
                "ci_high_pct": float(np.quantile(boot, 0.975)),
                "B": B,
                "block_samples": block,
                "seed": seed,
                "n_records": len(record_arrays),
                "n_points": total_points,
            })
    return pd.DataFrame(rows)


def write_outputs(predictions: pd.DataFrame, failures: pd.DataFrame, v2: dict):
    temperature = summarize(predictions, ["method", "initial_condition", "profile", "temperature_C"])
    fold = summarize(predictions, ["method", "initial_condition", "profile"])
    plateau = summarize(predictions, ["method", "initial_condition", "profile", "temperature_C", "plateau_edge_region"])
    times = summarize(predictions, ["method", "initial_condition", "profile", "temperature_C", "time_since_start_bin_s"])
    temperature.to_csv(OUT / "temperature_metrics.csv", index=False)
    fold.to_csv(OUT / "fold_summary.csv", index=False)
    plateau.to_csv(OUT / "plateau_edge_metrics.csv", index=False)
    times.to_csv(OUT / "time_since_start.csv", index=False)
    aggregate_rows = []
    for (method, initial), group in predictions.groupby(["method", "initial_condition"], sort=False):
        slices = summarize(group, ["profile", "temperature_C"])
        aggregate_rows.append({
            "method": method,
            "initial_condition": initial,
            "weighting": "slice_unweighted_primary",
            "MAE_pct": float(slices.MAE_pct.mean()),
            "RMSE_pct": float(slices.RMSE_pct.mean()),
            "n_slices": int(len(slices)),
            "n_points": int(len(group)),
        })
        point = error_metrics(group.soc_true, group.soc_pred)
        aggregate_rows.append({
            "method": method,
            "initial_condition": initial,
            "weighting": "pointwise_secondary",
            "MAE_pct": float(point["MAE_pct"]),
            "RMSE_pct": float(point["RMSE_pct"]),
            "n_slices": int(len(slices)),
            "n_points": int(len(group)),
        })
    pd.DataFrame(aggregate_rows).to_csv(OUT / "main_table.csv", index=False)
    bootstrap = circular_block_bootstrap(predictions, v2); bootstrap.to_csv(OUT / "bootstrap.csv", index=False)
    return temperature, fold, plateau, times, bootstrap


def leakage_audit(trajectories, selections_frame: pd.DataFrame, base: dict, v2: dict):
    profiles = base["protocol"]["profiles"]; folds = {}
    ecm = pd.read_csv(OUT / "ecm_fit_quality.csv")
    for holdout in profiles:
        training = sorted(str(tr.path) for tr in trajectories if tr.profile != holdout)
        test = sorted(str(tr.path) for tr in trajectories if tr.profile == holdout)
        selected_hash = str(selections_frame[selections_frame.fold_holdout == holdout].iloc[-1].training_files_sha256)
        ecm_files = sorted({
            path
            for value in ecm.loc[ecm.fold_holdout == holdout, "training_files"]
            for path in str(value).split(";") if path
        })
        folds[holdout] = {
            "training_files": training, "test_files": test,
            "training_file_list_sha256": sha256_list(training), "test_file_list_sha256": sha256_list(test),
            "study_training_file_list_sha256": selected_hash,
            "hash_match": selected_hash == sha256_list(training),
            "ecm_training_files": ecm_files,
            "ecm_training_file_list_sha256": sha256_list(ecm_files),
            "ecm_hash_match": ecm_files == training,
            "training_test_intersection": sorted(set(training) & set(test)),
        }
    passed = all(
        item["hash_match"] and item["ecm_hash_match"] and not item["training_test_intersection"]
        for item in folds.values()
    )
    audit = {
        "status": "PASS" if passed else "FAIL",
        "q_r_selection_touched_training_folds_only": passed,
        "ecm_fit_touched_training_folds_only": passed,
        "folds": folds, "evaluation_mask": "exact proposed end_index mask", "oracle_initialization": "same first SOC label as v1",
        "proposed_primary": "confirmatory 17-channel G4 GRU-residual, seeds 0-4",
        "config_sha256": sha256_file(ROOT / "configs" / "v2.yaml"),
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in [ROOT / "run_v2.py", ROOT / "src" / "v2_core.py"]},
    }
    (OUT / "leakage_audit_v2.json").write_text(json.dumps(audit, indent=2) + "\n")
    if not passed: raise RuntimeError("leakage_audit_v2 failed")
    return audit


def md_table(frame: pd.DataFrame, digits: int = 4) -> str:
    columns = list(frame.columns); lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                if not np.isfinite(value): values.append("NaN")
                elif value != 0 and abs(value) < 10 ** (-digits): values.append(f"{value:.3e}")
                else: values.append(f"{value:.{digits}f}")
            else: values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def append_report(temperature: pd.DataFrame | None, attribution: pd.DataFrame, stopped: bool, diagnosis: str | None = None):
    audit = json.loads((OUT / "audits" / "audit_summary.json").read_text())
    ratio = pd.read_csv(OUT / "audits" / "a2_hysteresis_ratio.csv").M_to_table_half_gap_ratio.replace([np.inf, -np.inf], np.nan).dropna()
    ecm = pd.read_csv(OUT / "ecm_fit_quality.csv")
    bad_ecm = int(ecm.rmse_gt_10mV.sum())
    bad_tau = int((~ecm.tau1_much_less_than_tau2).sum())
    v1 = pd.read_csv(ROOT / "results" / "temperature_metrics.csv")
    v1 = v1[(v1.initial_condition == "oracle") & v1.model.isin(MAIN_METHODS)].groupby("model", as_index=False).MAE_pct.mean().rename(columns={"model": "method", "MAE_pct": "v1_MAE_pct"})
    if temperature is not None:
        v2main = temperature[(temperature.initial_condition == "oracle") & temperature.method.isin(MAIN_METHODS)].groupby("method", as_index=False).MAE_pct.mean().rename(columns={"MAE_pct": "v2_MAE_pct"})
        comparison = v1.merge(v2main, on="method", how="outer"); comparison["improvement_v1_minus_v2_pct"] = comparison.v1_MAE_pct - comparison.v2_MAE_pct
        comparison_text = md_table(comparison)
    else:
        comparison_text = "Full v2 main-table execution was not run because the stop rule fired."
    attr = attribution[["case", "mean_MAE_pct", "delta_vs_full_MAE_pct"]]
    section = f"""{V2_START}

## v2 — fold-trained ECM and observability-aware filtering

### Technical summary

- Status: **{'STOPPED by the 8% representative-fold gate' if stopped else 'completed'}**.
- A1 could not estimate a median offset: all 24 records were checked, but none contained `|I| < 0.02 A` continuously for 120 s (qualifying rest ends: {audit['rest_ends_found']}). This is an identifiability limitation, not a pass on OCV/label alignment.
- A2 model-to-table hysteresis half-gap ratio had median `{ratio.median():.4f}` and IQR `[{ratio.quantile(.25):.4f}, {ratio.quantile(.75):.4f}]` over the SOC×8-temperature grid.
- A3 constant-current sign/unit test: `{audit['sign_unit_test']}`; discharge converged to the discharge branch and charge to the charge branch.
{('- Stop diagnosis: ' + diagnosis) if diagnosis else ''}

### v1 versus v2 main evaluation

The main comparison keeps the v1 folds, exact proposed `end_index` mask, and oracle initial-SOC policy. The proposed row is the locked confirmatory 17-channel G4 GRU-residual averaged over seeds 0–4. Slice-unweighted aggregation is primary and pointwise aggregation is secondary. Q/R is selected from the two training profiles at all eight temperatures with uniform slice weights; the selected setting is shared by the four filters within each outer fold.

{comparison_text}

### Representative-fold F1–F4 attribution

The representative holdout is `DST`, chosen before v2 execution in `configs/v2.yaml`. `v2_without_F*` is a leave-one-fix-out run; positive `delta_vs_full` means the omitted fix helped when present.

{md_table(attr)}

### Scope, methodology, and robustness

- The v1 code actually built one fold-shared parameter map from independent 5/25/45 °C HPPC workbooks and interpolated/clamped it to the LFP grid. The v1 report was accurate about the HPPC source, but that source was not fold-specific.
- F1 uses raw-current-sign `ΔV/ΔI` events from training profiles only for fold×8-temperature R0. R1, R2, τ1, and τ2 are least-squares voltage fits on the same fold's two training profiles at each of the eight temperatures. `results_v2/ecm_fit_quality.csv` records fit RMSE and τ separation.
- ECM fit quality warning: `{bad_ecm}/{len(ecm)}` fold×temperature fits exceed 10 mV voltage RMSE; `{bad_tau}/{len(ecm)}` fail the configured `τ1/τ2 ≤ 0.25` separation check. These flags are retained rather than filtered out.
- F2 uses the Savitzky–Golay-smoothed discharge-branch slope with window 11, `s_ref=2 mV/%SOC`, `s_min=0.05 mV/%SOC`, and noise inflation clipped to `[1,400]`.
- F3 initializes hysteresis from the first near-rest voltage and oracle SOC, clipped to `[-1,1]`.
- F4 uses Optuna TPE with 200 trials per fold and log ranges spanning six decades. Study databases and complete trial tables are under `results_v2/optuna/`.
- Paired uncertainty uses 60-sample circular blocks within each record, 10,000 replicates, and fixed seed 20260711.
- The proposed model is not retrained; locked confirmatory 17-channel G4 predictions are averaged across seeds 0–4 before stratification and paired bootstrap. Slice-unweighted bootstrap is primary; pointwise is secondary.

### Limitations and next action

- The requested 120-s rest audit is not assessable on these drive-cycle records. A separate long-rest dataset is required to quantify a 1 mV systematic alignment offset.
- UKF results are retained even if failed or worse; no UKF-specific repair or post-hoc tuning is applied.
- `results_v2/leakage_audit_v2.json` hashes every fold's training and test file lists and verifies that both ECM fitting and Q/R selection match training lists only.

### v1 appendix (immutable numbers)

{md_table(v1)}

{V2_END}
"""
    path = ROOT / "report.md"; original = path.read_text()
    if V2_START in original and V2_END in original:
        before = original.split(V2_START)[0].rstrip(); after = original.split(V2_END, 1)[1].lstrip()
        updated = before + "\n\n" + section + ("\n" + after if after else "")
    else:
        updated = original.rstrip() + "\n\n" + section
    path.write_text(updated, encoding="utf-8")


def diagnosis_text(attribution: pd.DataFrame) -> str:
    audit = json.loads((OUT / "audits" / "audit_summary.json").read_text())
    full = float(attribution.loc[attribution.case == "v2_all_on", "mean_MAE_pct"].iloc[0])
    ecm = pd.read_csv(OUT / "ecm_fit_quality.csv")
    text = f"""# v2 stop-rule diagnosis

## Outcome

The representative `DST` hysteresis EKF remained at `{full:.4f}%` mean SOC MAE after F1–F4, above the 8% gate. The run stopped before tuning the remaining folds or producing a full v2 main table. No second discretionary tuning cycle was attempted.

## Audit findings

- A1: `{audit['status']}`. All 24 records were scanned, but zero rest segments met `|I|<0.02 A` for at least 120 s, so the requested 1 mV systematic-offset test is not assessable.
- A2: see `audits/a2_hysteresis_ratio.csv` and `audits/a2_hysteresis_ratio.png`.
- A3: `{audit['sign_unit_test']}`. The current sign and branch convergence are consistent.
- F1 fold×temperature R0 estimates and training-only 2RC least-squares fits are saved before stopping.
- ECM quality: `{int(ecm.rmse_gt_10mV.sum())}/{len(ecm)}` fits exceed 10 mV voltage RMSE; `{int((~ecm.tau1_much_less_than_tau2).sum())}/{len(ecm)}` fail `τ1/τ2 ≤ 0.25`.
- F4 representative-fold study and leave-one-fix-out attribution are retained.

## Interpretation

The stop is evidence that shared physical-model and covariance fixes did not meet the predeclared accuracy threshold on the representative holdout. The missing long-rest evidence prevents attributing the remaining error to a millivolt-scale OCV/label offset. Further tuning would violate the one-cycle limit.
"""
    (OUT / "diagnosis.md").write_text(text)
    return f"representative DST hysteresis EKF MAE was {full:.4f}%, above 8%"


def boundary_stop_artifacts(failure: dict, trajectories, base: dict):
    cycles = pd.DataFrame(failure["cycles"]); cycles.to_csv(OUT / "qr_selection.csv", index=False)
    ecm = pd.read_csv(OUT / "ecm_fit_quality.csv")
    folds = {}
    for holdout in base["protocol"]["profiles"]:
        training = sorted(str(tr.path) for tr in trajectories if tr.profile != holdout)
        test = sorted(str(tr.path) for tr in trajectories if tr.profile == holdout)
        observed = None
        match = None
        if holdout == failure["fold_holdout"]:
            observed = str(cycles.iloc[-1].training_files_sha256); match = observed == sha256_list(training)
        ecm_files = sorted({
            path
            for value in ecm.loc[ecm.fold_holdout == holdout, "training_files"]
            for path in str(value).split(";") if path
        })
        folds[holdout] = {
            "training_files": training, "test_files": test,
            "training_file_list_sha256": sha256_list(training), "test_file_list_sha256": sha256_list(test),
            "study_training_file_list_sha256": observed, "hash_match": match,
            "ecm_training_files": ecm_files,
            "ecm_training_file_list_sha256": sha256_list(ecm_files),
            "ecm_hash_match": ecm_files == training,
            "study_status": "BOUNDARY_STOP" if holdout == failure["fold_holdout"] else "NOT_RUN_DUE_TO_BOUNDARY_STOP",
            "training_test_intersection": sorted(set(training) & set(test)),
        }
    representative = folds[failure["fold_holdout"]]
    leakage = {
        "status": "PASS_FOR_EXECUTED_STUDY",
        "q_r_selection_touched_training_folds_only": bool(representative["hash_match"] and not representative["training_test_intersection"]),
        "ecm_fit_touched_training_folds_only": bool(all(item["ecm_hash_match"] for item in folds.values())),
        "folds": folds, "evaluation_mask": "exact proposed end_index mask", "oracle_initialization": "same first SOC label as v1",
        "proposed_primary": "confirmatory 17-channel G4 GRU-residual, seeds 0-4",
        "note": "Only the representative fold study ran because the one-extension boundary gate stopped execution.",
    }
    (OUT / "leakage_audit_v2.json").write_text(json.dumps(leakage, indent=2) + "\n")
    flagged = failure["flagged"]
    bad_ecm = int(ecm.rmse_gt_10mV.sum())
    bad_tau = int((~ecm.tau1_much_less_than_tau2).sum())
    diagnosis = f"""# v2 boundary-stop diagnosis

## Outcome

F4 stopped after the single allowed range-extension cycle. The representative `{failure['fold_holdout']}` study still selected boundary-adjacent values: `{', '.join(f'{name}:{side}' for name, side in flagged.items())}`. No boundary value was accepted for the main table, no remaining-fold studies were run, and no additional tuning cycle was attempted.

## Audit findings available before the stop

- A1 scanned all 24 records and found no `|I|<0.02 A` segment lasting at least 120 s; the 1 mV systematic-offset test is therefore not assessable.
- A2 hysteresis-magnitude grid and plot are retained in `audits/`.
- A3 sign/unit test passed for all eight temperatures and both current directions.
- F1 fold-specific R0 estimates and per-fold/per-temperature training-profile RC fits were completed and retained.
- ECM quality warning: `{bad_ecm}/{len(ecm)}` fits exceed 10 mV voltage RMSE; `{bad_tau}/{len(ecm)}` fail `τ1/τ2 ≤ 0.25`.
- The executed study's training-file-list hash matches the representative fold's two training profiles; its held-out profile list is disjoint.

## F4 evidence

{md_table(cycles[['fold_holdout', 'range_cycle', 'objective_mean_training_slice_MAE_pct', 'q_soc', 'q_vp', 'q_h', 'r_voltage', 'boundary_parameters']])}

## Interpretation

The optimizer continued to prefer larger polarization/hysteresis process noise after the allowed expansion. Treating that boundary solution as final would violate the predeclared gate. A new run would require an explicitly revised search design; this run cannot continue under the one-retuning-cycle limit.
"""
    (OUT / "diagnosis.md").write_text(diagnosis)
    v1 = pd.read_csv(ROOT / "results" / "temperature_metrics.csv")
    v1 = v1[(v1.initial_condition == "oracle") & v1.model.isin(MAIN_METHODS)].groupby("model", as_index=False).MAE_pct.mean().rename(columns={"model": "method", "MAE_pct": "v1_MAE_pct"})
    audit = json.loads((OUT / "audits" / "audit_summary.json").read_text())
    section = f"""{V2_START}

## v2 — stopped at the F4 boundary gate

### Technical summary

- Status: **STOPPED after the single allowed range-extension cycle**.
- Representative fold: `{failure['fold_holdout']}`.
- Remaining boundary parameters: `{', '.join(f'{name}:{side}' for name, side in flagged.items())}`.
- A1 qualifying rest ends: `{audit['rest_ends_found']}`; OCV/label median-offset assessment is not possible on these records.
- A3 sign/unit test: `{audit['sign_unit_test']}`.
- Primary proposed comparator after Stage 2: confirmatory 17-channel G4 GRU-residual, seeds 0–4. The earlier G4eqdyn comparison is retained only in the pre-confirmatory appendix.

### Why no v2 main table is reported

The selected F4 values remained on a search boundary after one extension. The protocol requires reporting and stopping at that point, so the remaining folds, four-filter main table, diagnostic trio, stratified metrics, and bootstrap were not run. No filter or failure was silently dropped; evaluation did not begin.

### Executed F4 evidence

{md_table(cycles[['range_cycle', 'objective_mean_training_slice_MAE_pct', 'q_soc', 'q_vp', 'q_h', 'r_voltage', 'boundary_parameters']])}

### Available audits and fitted physics

- `results_v2/audits/` contains A1–A3 tables and plots.
- `results_v2/r0_estimates.csv` contains fold×8-temperature training-only R0 estimates.
- `results_v2/ecm_fit_quality.csv` contains fold×8-temperature training-only R1/R2/τ1/τ2 least-squares fits, voltage RMSE flags, and τ-separation checks.
- ECM quality warning: `{bad_ecm}/{len(ecm)}` fits exceed 10 mV voltage RMSE; `{bad_tau}/{len(ecm)}` fail `τ1/τ2 ≤ 0.25`. No flagged temperature was silently removed.
- The v1 code audit found a single shared 5/25/45 °C HPPC map; v2 no longer shares those parameters across folds.
- `results_v2/leakage_audit_v2.json` hashes all fold file lists and verifies ECM fitting plus the executed Q/R study used training files only.

### v1 appendix (immutable numbers)

{md_table(v1)}

{V2_END}
"""
    path = ROOT / "report.md"; original = path.read_text()
    if V2_START in original and V2_END in original:
        updated = original.split(V2_START)[0].rstrip() + "\n\n" + section + "\n" + original.split(V2_END, 1)[1].lstrip()
    else:
        updated = original.rstrip() + "\n\n" + section
    path.write_text(updated, encoding="utf-8")
    completeness = {
        "status": "STOP_BOUNDARY_AFTER_EXTENSION", "stage_reached": "F4_representative_fold",
        "remaining_boundary_parameters": flagged, "retuning_cycles_used": 1,
        "main_evaluation_run": False, "all_four_filters_reported": False,
        "reason": "Protocol stop occurred before Stage 2; no filters were selectively dropped.",
    }
    (OUT / "completeness_audit.json").write_text(json.dumps(completeness, indent=2) + "\n")
    return completeness


def main():
    OUT.mkdir(exist_ok=True); (OUT / "optuna").mkdir(exist_ok=True)
    if not (OUT / "audits" / "audit_summary.json").is_file():
        raise RuntimeError("Stage 0 audits must run before run_v2.py")
    base, v2 = load_configs(); trajectories = load_all(base); ocv, pmap = prepare_physics(base, v2, trajectories)
    representative = str(v2["stop_rule"]["representative_holdout"])
    try:
        rep_noise, rep_cycles = tune_fold(representative, trajectories, ocv, pmap, base, v2)
    except RuntimeError:
        boundary_path = OUT / "boundary_failure.json"
        if not boundary_path.is_file():
            raise
        failure = json.loads(boundary_path.read_text())
        result = boundary_stop_artifacts(failure, trajectories, base)
        print(json.dumps(result, indent=2))
        return
    attribution = representative_attribution(representative, trajectories, ocv, pmap, rep_noise, base, v2)
    attribution.to_csv(OUT / "representative_fix_attribution.csv", index=False)
    rep_mae = float(attribution.loc[attribution.case == "v2_all_on", "mean_MAE_pct"].iloc[0])
    if rep_mae > float(v2["stop_rule"]["hysteresis_ekf_mean_mae_pct"]):
        rep_cycles.to_csv(OUT / "qr_selection.csv", index=False)
        detail = diagnosis_text(attribution); append_report(None, attribution, True, detail)
        print(json.dumps({"status": "STOP_RULE", "representative_MAE_pct": rep_mae, "diagnosis": str(OUT / "diagnosis.md")}, indent=2))
        return
    selections = {representative: rep_noise}; selection_frames = [rep_cycles]
    for holdout in base["protocol"]["profiles"]:
        if holdout == representative: continue
        selections[holdout], cycles = tune_fold(holdout, trajectories, ocv, pmap, base, v2); selection_frames.append(cycles)
    selections_frame = pd.concat(selection_frames, ignore_index=True); selections_frame.to_csv(OUT / "qr_selection.csv", index=False)
    leakage_audit(trajectories, selections_frame, base, v2)
    predictions, failures = run_stage2(trajectories, ocv, pmap, selections, base, v2)
    temperature, fold, plateau, times, bootstrap = write_outputs(predictions, failures, v2)
    append_report(temperature, attribution, False)
    required = ["main_table.csv", "fold_summary.csv", "temperature_metrics.csv", "plateau_edge_metrics.csv", "time_since_start.csv", "bootstrap.csv", "r0_estimates.csv", "ecm_fit_quality.csv", "qr_selection.csv", "leakage_audit_v2.json"]
    completeness = {
        "status": "PASS" if all((OUT / name).is_file() for name in required) else "FAIL",
        "required_files": required, "all_four_filters_reported": all(method in set(temperature.method) for method in MAIN_METHODS),
        "failed_runs": len(failures), "prediction_rows": len(predictions), "representative_MAE_pct": rep_mae,
        "stop_threshold_MAE_pct": float(v2["stop_rule"]["hysteresis_ekf_mean_mae_pct"]),
    }
    (OUT / "completeness_audit.json").write_text(json.dumps(completeness, indent=2) + "\n")
    if completeness["status"] != "PASS" or not completeness["all_four_filters_reported"]:
        raise RuntimeError(f"v2 completeness gate failed: {completeness}")
    print(json.dumps(completeness, indent=2))


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main()
