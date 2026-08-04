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
    window_len: int
    sequence_training: bool = False


CANDIDATES = (
    Candidate("gru_normal_g4_w100_endpoint", 100, False),
    Candidate("gru_normal_g4_w200_endpoint", 200, False),
    Candidate("gru_normal_g4_w100_sequence", 100, True),
    Candidate("gru_normal_g4_w200_sequence", 200, True),
)


def parse_seeds(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def prefix_for(candidate: Candidate, seeds: tuple[int, ...], epochs: int, batch_size: int) -> str:
    seed_tag = "seeds" + "".join(str(seed) for seed in seeds)
    return f"soc80_us06_context_screen_{candidate.name}_{seed_tag}_b{int(batch_size)}_e{int(epochs)}"


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
        window_len=int(candidate.window_len),
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
        dropout=0.06,
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
        stage2_select_rule="none",
        skip_stage2=True,
        save_predictions=True,
        save_final_weights=True,
        final_only_eval=True,
        skip_train_final_eval=True,
        valid_split_mode="profile",
        sequence_training=bool(candidate.sequence_training),
        cache_dataset_cuda=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Screen longer-context GRU G4 candidates on US06.")
    p.add_argument("--base-dir", default=str(PROJECT_ROOT))
    p.add_argument("--raw-root", default="nmc_soc80_train_nofloor_qmax")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2048)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    out_dir = base_dir / "nmc_goal_vcorr_it_train_dst_selector_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)

    manifest_rows = []
    for idx, candidate in enumerate(CANDIDATES):
        manifest_rows.append({**candidate.__dict__, "idx": idx, "prefix": prefix_for(candidate, seeds, int(args.epochs), int(args.batch_size))})
    pd.DataFrame(manifest_rows).to_csv(out_dir / "soc80_us06_context_screen_manifest.csv", index=False)

    print(
        f"us06_context_screen jobs={len(CANDIDATES)} seeds={args.seeds} epochs={args.epochs} "
        f"batch_size={args.batch_size}",
        flush=True,
    )
    durations: list[float] = []
    completed = skipped = 0
    for idx, candidate in enumerate(CANDIDATES):
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
    print(f"us06_context_screen finished completed={completed} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
