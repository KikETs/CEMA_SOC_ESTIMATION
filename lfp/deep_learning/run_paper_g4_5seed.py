#!/usr/bin/env python3
"""Run the frozen all-eight-temperature LFP paper configuration."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from Scripts import run_confirmatory_minipanel as panel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path,
        default=REPO / "Data/Preprocessed/LFP/prepared_data_ocv_discharge_soc",
    )
    parser.add_argument("--output-root", type=Path, default=REPO / "runs/lfp_dl")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--holdouts", default=",".join(panel.PROFILES))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    holdouts = tuple(value.strip().upper() for value in args.holdouts.split(",") if value.strip())
    condition = panel.CONDITIONS[0]
    result_dir = args.output_root / "nmc_goal_vcorr_it_train_dst_selector_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    cfg_args = argparse.Namespace(epochs=args.epochs, batch_size=args.batch_size)
    markers: list[Path] = []
    for holdout in holdouts:
        cfg = panel.make_cfg(condition, "tier1", holdout, "G4", "gru", "residual", seeds, cfg_args)
        prefix = (
            f"paper_repro_all8_gru_residual_g4_holdout{holdout.lower()}_"
            f"s{''.join(map(str, seeds))}_b{args.batch_size}_e{args.epochs}"
        )
        cfg = replace(cfg, base_dir=args.output_root.resolve(), raw_root=args.data_root.resolve(), output_prefix=prefix)
        marker = result_dir / f"{prefix}_by_temperature.csv"
        markers.append(marker)
        if marker.is_file():
            print(f"SKIP {holdout}: {marker.name}", flush=True)
        else:
            print(f"START {holdout}: seeds={args.seeds}", flush=True)
            panel.clear_caches()
            panel.selector.run(cfg)
            if not marker.is_file():
                raise RuntimeError(f"Missing completion marker: {marker}")

    frames = []
    for marker in markers:
        frame = pd.read_csv(marker)
        frame = frame[(frame["split"] == "test") & frame["variant"].astype(str).str.endswith("_selector_base")]
        frames.append(frame)
    slices = pd.concat(frames, ignore_index=True)
    expected = len(holdouts) * len(seeds) * len(panel.ALL_TEMPS)
    if len(slices) != expected:
        raise RuntimeError(f"Expected {expected} LFP slices, found {len(slices)}")
    slices.to_csv(args.output_root / "paper_g4_gru_residual_slice_rows.csv", index=False)
    by_seed = slices.groupby("seed", as_index=False).agg(MAE_pct=("MAE_pct", "mean"), RMSE_pct=("RMSE_pct", "mean"), n_slices=("MAE_pct", "size"))
    by_seed.to_csv(args.output_root / "paper_g4_gru_residual_by_seed.csv", index=False)
    pd.DataFrame([{
        "chemistry": "LFP", "feature_set": "paper_g4_all_ema", "model": "GRU",
        "head": "anchor_residual", "seeds": args.seeds, "epochs": args.epochs,
        "slice_unweighted_MAE_pct": float(by_seed["MAE_pct"].mean()),
        "slice_unweighted_RMSE_pct": float(by_seed["RMSE_pct"].mean()),
    }]).to_csv(args.output_root / "paper_g4_gru_residual_summary.csv", index=False)
    print(by_seed.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
