#!/usr/bin/env python3
"""Run the frozen NMC T6-plain headline configuration."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from Scripts import run_ocvstart_3lopo_fullgrid as grid  # noqa: E402
from soc_decomp.nmc_vcorr_it_train_dst_selector_run import run  # noqa: E402


HOLDOUTS = ("DST", "FUDS", "US06")
FEATURE_SET = "paper_t6_voltage_ema_all"
LOCKED_RECIPE = {
    "lr": 8e-4,
    "weight_decay": 2e-4,
    "huber_beta": 0.02,
    "batch_size": 2048,
    "epochs": 200,
    "window_len": 50,
    "stride": 3,
    "stage1_selector": "last_epoch",
    "recurrent": "gru",
    "model_kind": "single",
    "head_kind": "linear",
    "weight_0": 1.0,
    "weight_25": 1.0,
    "weight_45": 1.0,
    "lambda_rex": 0.0,
    "lambda_condinv": 0.0,
    "lambda_anchor_loss": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO / "Data/Preprocessed/NMC/nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean",
    )
    parser.add_argument("--output-root", type=Path, default=REPO / "runs/nmc_dl")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--holdouts", default=",".join(HOLDOUTS))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def assert_recipe(cfg, seeds: tuple[int, ...]) -> None:
    for name, expected in LOCKED_RECIPE.items():
        actual = getattr(cfg, name)
        if actual != expected:
            raise RuntimeError(f"Locked recipe mismatch: {name}={actual!r}, expected {expected!r}")
    if tuple(cfg.seeds) != seeds:
        raise RuntimeError(f"Locked seed mismatch: {cfg.seeds!r}, expected {seeds!r}")


def main() -> None:
    args = parse_args()
    if args.epochs != 200 or args.batch_size != 2048:
        raise RuntimeError("The headline protocol requires epochs=200 and batch_size=2048")
    seeds = grid.parse_seeds(args.seeds)
    if seeds != tuple(range(10)):
        raise RuntimeError(f"The headline protocol requires seeds 0-9, got {seeds!r}")
    holdouts = tuple(value.strip().upper() for value in args.holdouts.split(",") if value.strip())
    run_args = argparse.Namespace(
        base_dir=str(args.output_root.resolve()),
        raw_root=str(args.data_root.resolve()),
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        include_train_final_eval=False,
    )
    result_dir = args.output_root / "nmc_goal_vcorr_it_train_dst_selector_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    markers: list[Path] = []
    for holdout in holdouts:
        job = grid.Job("headline", "gru", "normal", "single", "gru", FEATURE_SET, holdout)
        prefix = (
            f"paper_t6_plain_nmc_gru_normal_t6_holdout{holdout.lower()}_"
            f"s0123456789_b{args.batch_size}_e{args.epochs}"
        )
        cfg = grid.cfg_for_job(job, run_args, prefix)
        cfg = replace(cfg, **{name: value for name, value in LOCKED_RECIPE.items() if hasattr(cfg, name)})
        assert_recipe(cfg, seeds)
        marker = result_dir / f"{prefix}_by_temperature.csv"
        markers.append(marker)
        if marker.is_file():
            print(f"SKIP {holdout}: {marker.name}", flush=True)
        else:
            print(f"START {holdout}: seeds={args.seeds}", flush=True)
            run(cfg)
            if not marker.is_file():
                raise RuntimeError(f"Missing completion marker: {marker}")

    frames = []
    for marker in markers:
        frame = pd.read_csv(marker)
        frames.append(frame[(frame["split"] == "test") & frame["variant"].astype(str).str.endswith("_selector_base")])
    slices = pd.concat(frames, ignore_index=True)
    expected = len(holdouts) * len(seeds) * 3
    if len(slices) != expected:
        raise RuntimeError(f"Expected {expected} NMC slices, found {len(slices)}")
    slices.to_csv(args.output_root / "paper_t6_plain_gru_linear_auxoff_slice_rows.csv", index=False)
    by_seed = slices.groupby("seed", as_index=False).agg(
        MAE_pct=("MAE_pct", "mean"), RMSE_pct=("RMSE_pct", "mean"), n_slices=("MAE_pct", "size")
    )
    by_seed.to_csv(args.output_root / "paper_t6_plain_gru_linear_auxoff_by_seed.csv", index=False)
    pd.DataFrame([{
        "chemistry": "NMC", "feature_set": FEATURE_SET, "model": "GRU",
        "head": "plain_linear", "auxiliary_losses": "all_off", "seeds": args.seeds,
        "epochs": args.epochs, "slice_unweighted_MAE_pct": float(by_seed["MAE_pct"].mean()),
        "slice_unweighted_RMSE_pct": float(by_seed["RMSE_pct"].mean()),
    }]).to_csv(args.output_root / "paper_t6_plain_gru_linear_auxoff_summary.csv", index=False)
    print(by_seed.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
