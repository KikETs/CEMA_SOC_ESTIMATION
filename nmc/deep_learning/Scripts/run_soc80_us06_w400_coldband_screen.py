#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from soc_decomp.nmc_vcorr_it_train_dst_selector_run import TrainDSTSelectorConfig, run


@dataclass(frozen=True)
class Candidate:
    name: str
    band_lower: float
    band_upper: float
    loss_weight: float
    dynamic_weight: float = 0.0


CANDIDATES = (
    Candidate("w400_coldband10_25_w2_dyn0", 0.10, 0.25, 2.0, 0.0),
    Candidate("w400_coldband10_25_w4_dyn0", 0.10, 0.25, 4.0, 0.0),
    Candidate("w400_coldband10_25_w3_dyn05", 0.10, 0.25, 3.0, 0.5),
    Candidate("w400_coldband05_25_w3_dyn05", 0.05, 0.25, 3.0, 0.5),
)


def parse_seeds(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


def parse_names(raw: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in str(raw).split(",") if x.strip())


def candidate_map() -> dict[str, Candidate]:
    return {candidate.name: candidate for candidate in CANDIDATES}


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def prefix_for(candidate: Candidate, seeds: tuple[int, ...], epochs: int, batch_size: int) -> str:
    seed_tag = "seeds" + "".join(str(seed) for seed in seeds)
    return f"soc80_us06_w400_coldband_screen_{candidate.name}_{seed_tag}_b{int(batch_size)}_e{int(epochs)}"


def cfg_for_candidate(candidate: Candidate, args: argparse.Namespace, prefix: str) -> TrainDSTSelectorConfig:
    epochs = int(args.epochs)
    return TrainDSTSelectorConfig(
        base_dir=Path(args.base_dir),
        raw_root=Path(args.raw_root),
        output_prefix=prefix,
        seeds=parse_seeds(args.seeds),
        train_profiles=("VALIDATION", "DST", "FUDS"),
        valid_profiles=("NONE",),
        test_profiles=("US06",),
        train_temperatures=(0.0, 25.0, 45.0),
        valid_temperatures=(),
        test_temperatures=(0.0, 25.0, 45.0),
        window_len=400,
        stride=3,
        epochs=epochs,
        stage1_eval_every=epochs,
        selector_min_epoch=epochs,
        selector_max_epoch=0,
        fixed_stage1_epoch=epochs,
        batch_size=int(args.batch_size),
        num_workers=0,
        prefetch_factor=2,
        hidden_size=128,
        layers=1,
        kernel_size=5,
        model_kind="single",
        recurrent="gru",
        head_kind="linear",
        dropout=float(args.dropout),
        feature_set="paper_g4_all_ema",
        stage2_feature_set="paper_g4_all_ema",
        temp_mode="none",
        train_sampler="temperature_profile_regime_balanced",
        stage1_selector="last_epoch",
        selector_regime_min_windows=30,
        lambda_regime_cvar=0.0,
        lambda_zbin_cvar=0.0,
        zbin_cvar_top_frac=0.34,
        lambda_condinv=0.02,
        lambda_rex=2.0,
        weight_0=0.8,
        weight_25=2.2,
        weight_45=1.0,
        hard_region_loss_weight=float(candidate.loss_weight),
        hard_region_soc_lower=float(candidate.band_lower),
        hard_region_soc_upper=float(candidate.band_upper),
        hard_region_dynamic_weight=float(candidate.dynamic_weight),
        hard_region_cold_only=True,
        stage2_select_rule="none",
        skip_stage2=True,
        save_predictions=True,
        save_final_weights=True,
        final_only_eval=True,
        skip_train_final_eval=True,
        valid_split_mode="profile",
        sequence_training=True,
        cache_dataset_cuda=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Screen w400 GRU cold band-weighted loss on US06 holdout.")
    p.add_argument("--base-dir", default=str(PROJECT_ROOT))
    p.add_argument("--raw-root", default="nmc_soc80_train_nofloor_qmax")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.06)
    p.add_argument("--candidates", default=",".join(candidate.name for candidate in CANDIDATES))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    out_dir = base_dir / "nmc_goal_vcorr_it_train_dst_selector_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    cmap = candidate_map()
    candidates = []
    for name in parse_names(args.candidates):
        if name not in cmap:
            raise ValueError(f"Unknown candidate={name!r}; choices={','.join(cmap)}")
        candidates.append(cmap[name])
    candidates = tuple(candidates)

    manifest_rows = []
    for idx, candidate in enumerate(candidates):
        manifest_rows.append(
            {
                **candidate.__dict__,
                "idx": idx,
                "window_len": 400,
                "sequence_training": True,
                "feature_set": "paper_g4_all_ema",
                "prefix": prefix_for(candidate, seeds, int(args.epochs), int(args.batch_size)),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(out_dir / "soc80_us06_w400_coldband_screen_manifest.csv", index=False)

    print(
        f"us06_w400_coldband_screen jobs={len(CANDIDATES)} seeds={args.seeds} "
        f"epochs={args.epochs} batch_size={args.batch_size}",
        flush=True,
    )
    durations: list[float] = []
    completed = skipped = 0
    for idx, candidate in enumerate(candidates):
        prefix = prefix_for(candidate, seeds, int(args.epochs), int(args.batch_size))
        summary_path = out_dir / f"{prefix}_test_summary.csv"
        if summary_path.exists():
            skipped += 1
            print(f"=== skip existing idx={idx} candidate={candidate.name} prefix={prefix} ===", flush=True)
            continue
        started = time.time()
        print(f"=== start idx={idx} candidate={candidate.name} prefix={prefix} {time.strftime('%Y-%m-%d %H:%M:%S %Z')} ===", flush=True)
        run(cfg_for_candidate(candidate, args, prefix))
        elapsed = time.time() - started
        durations.append(elapsed)
        completed += 1
        remaining = len(CANDIDATES) - idx - 1
        mean_duration = sum(durations) / max(1, len(durations))
        print(
            f"=== done idx={idx} candidate={candidate.name} elapsed={format_eta(elapsed)} "
            f"completed={completed} skipped={skipped} eta={format_eta(mean_duration * remaining)} "
            f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} ===",
            flush=True,
        )

    rows = []
    for candidate in candidates:
        prefix = prefix_for(candidate, seeds, int(args.epochs), int(args.batch_size))
        summary_path = out_dir / f"{prefix}_test_summary.csv"
        if not summary_path.exists():
            continue
        df = pd.read_csv(summary_path)
        if "variant" in df.columns:
            selected = df[df["variant"].astype(str).str.contains("selected", case=False, na=False)]
            if len(selected):
                df = selected
        rows.append(
            {
                "candidate": candidate.name,
                "mae_0C": float(df["0.0"].mean()),
                "mae_25C": float(df["25.0"].mean()),
                "mae_45C": float(df["45.0"].mean()),
                "mae_mean_temp": float(df[["0.0", "25.0", "45.0"]].mean(axis=1).mean()),
                "n_seeds": int(df["seed"].nunique()),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["mae_0C", "mae_mean_temp"]) if rows else pd.DataFrame()
    summary_path = out_dir / "soc80_us06_w400_coldband_screen_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"us06_w400_coldband_screen finished completed={completed} skipped={skipped} summary={summary_path}", flush=True)
    if len(summary):
        print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
