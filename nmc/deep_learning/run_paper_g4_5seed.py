#!/usr/bin/env python3
"""Run the frozen NMC paper G4+GRU anchor-residual configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from Scripts import run_ocvstart_3lopo_fullgrid as grid  # noqa: E402
from soc_decomp.nmc_vcorr_it_train_dst_selector_run import run  # noqa: E402


HOLDOUTS = ("DST", "FUDS", "US06")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO / "Data/Preprocessed/NMC/nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean",
    )
    parser.add_argument("--output-root", type=Path, default=REPO / "runs/nmc_dl")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--holdouts", default=",".join(HOLDOUTS))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = grid.parse_seeds(args.seeds)
    holdouts = tuple(value.strip().upper() for value in args.holdouts.split(",") if value.strip())
    run_args = argparse.Namespace(
        base_dir=str(args.output_root.resolve()), raw_root=str(args.data_root.resolve()),
        seeds=args.seeds, epochs=args.epochs, batch_size=args.batch_size,
        include_train_final_eval=False,
    )
    result_dir = args.output_root / "nmc_goal_vcorr_it_train_dst_selector_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    markers: list[Path] = []
    for holdout in holdouts:
        job = grid.Job(
            "baseline_model", "gru", "residual", "anchor_residual_sequence",
            "gru", "paper_g4_all_ema", holdout,
        )
        prefix = grid.prefix_for(job, seeds, args.epochs, args.batch_size, "paper_repro")
        cfg = grid.cfg_for_job(job, run_args, prefix)
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
        frame = frame[(frame["split"] == "test") & frame["variant"].astype(str).str.endswith("_selector_base")]
        frames.append(frame)
    slices = pd.concat(frames, ignore_index=True)
    expected = len(holdouts) * len(seeds) * 3
    if len(slices) != expected:
        raise RuntimeError(f"Expected {expected} NMC slices, found {len(slices)}")
    slices.to_csv(args.output_root / "paper_g4_gru_residual_slice_rows.csv", index=False)
    by_seed = slices.groupby("seed", as_index=False).agg(MAE_pct=("MAE_pct", "mean"), RMSE_pct=("RMSE_pct", "mean"), n_slices=("MAE_pct", "size"))
    by_seed.to_csv(args.output_root / "paper_g4_gru_residual_by_seed.csv", index=False)
    pd.DataFrame([{
        "chemistry": "NMC", "feature_set": "paper_g4_all_ema", "model": "GRU",
        "head": "anchor_residual", "seeds": args.seeds, "epochs": args.epochs,
        "slice_unweighted_MAE_pct": float(by_seed["MAE_pct"].mean()),
        "slice_unweighted_RMSE_pct": float(by_seed["RMSE_pct"].mean()),
    }]).to_csv(args.output_root / "paper_g4_gru_residual_summary.csv", index=False)
    print(by_seed.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
