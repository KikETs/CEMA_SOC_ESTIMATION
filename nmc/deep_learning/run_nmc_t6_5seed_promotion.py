#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from Scripts import run_ocvstart_3lopo_fullgrid as grid
from soc_decomp.nmc_vcorr_it_train_dst_selector_run import run


ROOT = Path(__file__).resolve().parent
PROMOTION_ROOT = ROOT / "nmc_5seed_promotion" / "t6_gru_residual_5seed"
RESULTS = PROMOTION_ROOT / "nmc_goal_vcorr_it_train_dst_selector_results"
OLD = ROOT / "nmc_goal_vcorr_it_train_dst_selector_results"
RAW = ROOT / "nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean"
G4 = ROOT / "nmc_5seed_promotion" / "g4_gru_residual_5seed_aggregate.csv"
HOLDOUTS = ("DST", "FUDS", "US06")
ANALYSIS_STAGE = "posthoc_5seed"


def job(holdout: str) -> grid.Job:
    return grid.Job(
        "remaining_grid", "gru", "residual", "anchor_residual_sequence",
        "gru", "paper_t6_voltage_ema_all", holdout,
    )


def selected(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected_rows = frame[frame.variant.astype(str).str.contains("selected")] if "variant" in frame else frame
    return selected_rows if len(selected_rows) else frame


def rows_for(prefix_pattern: str, seeds_source: str) -> list[dict]:
    rows = []
    for holdout in HOLDOUTS:
        path = next((OLD if seeds_source == "existing_seeds012" else RESULTS).glob(prefix_pattern.format(holdout=holdout.lower())))
        for _, record in selected(path).iterrows():
            for temperature in (0.0, 25.0, 45.0):
                rows.append(
                    {
                        "seed": int(record.seed),
                        "holdout": holdout,
                        "temperature_C": temperature,
                        "MAE_pct": float(record[str(temperature)]),
                        "source": seeds_source,
                        "analysis_stage": ANALYSIS_STAGE,
                    }
                )
    return rows


def grid_three_seed_mean(feature_code: str) -> float:
    rows = []
    for holdout in HOLDOUTS:
        path = next(OLD.glob(f"s3_3l_*_gru_{feature_code}_r_{holdout.lower()}_s012_b2048_e200_test_summary.csv"))
        for _, record in selected(path).iterrows():
            rows.extend(float(record[str(temp)]) for temp in (0.0, 25.0, 45.0))
    if len(rows) != 27:
        raise RuntimeError(f"Expected 27 {feature_code} grid slices, found {len(rows)}")
    return float(pd.Series(rows).mean())


def aggregate() -> dict:
    rows = rows_for(
        "s3_3l_rg_gru_t6_r_{holdout}_s012_b2048_e200_test_summary.csv",
        "existing_seeds012",
    )
    rows += rows_for(
        "nmc5seedpromo_t6_3l_rg_gru_t6_r_{holdout}_s34_b2048_e200_test_summary.csv",
        "new_seeds34",
    )
    slices = pd.DataFrame(rows).drop_duplicates(["seed", "holdout", "temperature_C"])
    if len(slices) != 45 or set(slices.seed) != {0, 1, 2, 3, 4}:
        raise RuntimeError(f"Bad T6 promotion coverage: shape={slices.shape}, seeds={set(slices.seed)}")
    slices.to_csv(PROMOTION_ROOT / "t6_gru_residual_5seed_slice_rows.csv", index=False)

    by_seed = slices.groupby("seed", as_index=False).agg(MAE_pct=("MAE_pct", "mean"), n_slices=("MAE_pct", "size"))
    by_seed["analysis_stage"] = ANALYSIS_STAGE
    by_seed.to_csv(PROMOTION_ROOT / "t6_gru_residual_5seed_by_seed.csv", index=False)
    aggregate_frame = pd.DataFrame(
        [
            {
                "model": "T6_GRU_residual",
                "feature_set": "paper_t6_voltage_ema_all",
                "seeds": "0,1,2,3,4",
                "n_seeds": 5,
                "n_slices_per_seed": 9,
                "slice_unweighted_MAE_pct": float(by_seed.MAE_pct.mean()),
                "seed_SD_pct": float(by_seed.MAE_pct.std(ddof=1)),
                "analysis_stage": ANALYSIS_STAGE,
            }
        ]
    )
    aggregate_frame.to_csv(PROMOTION_ROOT / "t6_gru_residual_5seed_aggregate.csv", index=False)

    subset_mean = float(slices[slices.seed.isin([0, 1, 2])].MAE_pct.mean())
    if abs(subset_mean - 0.3052) > 0.001:
        raise RuntimeError(f"T6 seed012 self-check failed: {subset_mean:.9f}")
    g4_mean = float(pd.read_csv(G4).slice_unweighted_MAE_pct.iloc[0])
    if abs(g4_mean - 0.32457) > 0.001:
        raise RuntimeError(f"G4 5-seed reference self-check failed: {g4_mean:.9f}")
    t6_mean = float(aggregate_frame.slice_unweighted_MAE_pct.iloc[0])
    g0_mean = grid_three_seed_mean("g0")
    carrier = pd.DataFrame(
        [
            {
                "T6_5seed_MAE_pct": t6_mean,
                "G4_5seed_MAE_pct": g4_mean,
                "G0_3seed_grid_MAE_pct": g0_mean,
                "G4_minus_T6_over_T6_percent": 100.0 * (g4_mean - t6_mean) / t6_mean,
                "T6_minus_G0_over_G0_percent": 100.0 * (t6_mean - g0_mean) / g0_mean,
                "T6_seed_basis": "seeds_0_1_2_3_4",
                "G4_seed_basis": "seeds_0_1_2_3_4",
                "G0_seed_basis": "seeds_0_1_2_grid_only",
                "aggregation": "slice_unweighted_fold_x_temperature_then_seed_mean",
                "analysis_stage": ANALYSIS_STAGE,
            }
        ]
    )
    carrier.to_csv(PROMOTION_ROOT / "carrier_5seed.csv", index=False)
    return {
        "t6_seed012_selfcheck": subset_mean,
        "t6_5seed": t6_mean,
        "g4_5seed_reference": g4_mean,
        "g0_3seed_grid": g0_mean,
        "carrier": carrier.iloc[0].to_dict(),
    }


def main() -> None:
    if PROMOTION_ROOT.exists():
        raise SystemExit(f"Refusing to overwrite existing promotion directory: {PROMOTION_ROOT}")
    RESULTS.mkdir(parents=True)
    args = argparse.Namespace(
        base_dir=str(PROMOTION_ROOT), raw_root=str(RAW), seeds="3,4",
        epochs=200, batch_size=2048, include_train_final_eval=False,
    )
    started = time.time()
    jobs = []
    for holdout in HOLDOUTS:
        current = job(holdout)
        prefix = grid.prefix_for(current, (3, 4), 200, 2048, "nmc5seedpromo_t6")
        config = grid.cfg_for_job(current, args, prefix)
        if not (
            config.feature_set == "paper_t6_voltage_ema_all"
            and config.model_kind == "anchor_residual_sequence"
            and config.recurrent == "gru"
            and config.epochs == 200
            and config.batch_size == 2048
            and config.stage1_selector == "last_epoch"
        ):
            raise RuntimeError(f"Frozen T6 promotion config mismatch: {config}")
        job_start = time.time()
        run(config)
        jobs.append(
            {
                "holdout": holdout,
                "seeds": [3, 4],
                "feature_set": config.feature_set,
                "model_kind": config.model_kind,
                "recurrent": config.recurrent,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "selector": config.stage1_selector,
                "elapsed_seconds": time.time() - job_start,
                "analysis_stage": ANALYSIS_STAGE,
            }
        )
    checks = aggregate()
    total_seconds = time.time() - started
    full_grid_runs = 396
    manifest = {
        "jobs": jobs,
        "checks": checks,
        "total_elapsed_seconds": total_seconds,
        "excluded_full_grid_5seed_runs": full_grid_runs,
        "excluded_full_grid_estimated_serial_gpu_hours": total_seconds * (full_grid_runs / 6.0) / 3600.0,
        "estimate_basis": "396/6 times the measured six-run T6 promotion wall time; no full-grid runs executed",
        "analysis_stage": ANALYSIS_STAGE,
    }
    (PROMOTION_ROOT / "promotion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
