#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from soc_decomp.nmc_vcorr_it_train_dst_selector_run import TrainDSTSelectorConfig, run


@dataclass(frozen=True)
class Candidate:
    name: str
    model_kind: str
    recurrent: str
    feature_set: str
    weight_0: float
    hard_weight: float = 0.0
    hard_soc: float = 0.4
    hard_dynamic: float = 0.5
    hard_cold_only: bool = False


CANDIDATES = (
    Candidate("gru_normal_eqdyn_w0p8", "single", "gru", "paper_g4_eqdyn", 0.8),
    Candidate("gru_normal_eqdyn_w2", "single", "gru", "paper_g4_eqdyn", 2.0),
    Candidate("gru_normal_eqdyn_w4", "single", "gru", "paper_g4_eqdyn", 4.0),
    Candidate("gru_normal_eqdyn_hardcold3", "single", "gru", "paper_g4_eqdyn", 0.8, hard_weight=3.0, hard_cold_only=True),
    Candidate("gru_residual_eqdyn_w2_hardcold3", "anchor_residual_sequence", "gru", "paper_g4_eqdyn", 2.0, hard_weight=3.0, hard_cold_only=True),
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
    return f"soc80_us06_0c_screen_{candidate.name}_{seed_tag}_b{int(batch_size)}_e{int(epochs)}"


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
        model_kind=candidate.model_kind,
        recurrent=candidate.recurrent,
        head_kind="linear",
        dropout=0.06,
        feature_set=candidate.feature_set,
        stage2_feature_set=candidate.feature_set,
        temp_mode="none",
        train_sampler="temperature_profile_regime_balanced",
        stage1_selector="last_epoch",
        selector_regime_min_windows=30,
        anchor_residual_limit=0.12,
        anchor_residual_limit_init="rand01",
        anchor_residual_limit_mode="learnable",
        anchor_residual_limit_lower=0.0,
        anchor_residual_limit_upper=0.2,
        lambda_anchor_loss=0.1,
        lambda_regime_cvar=0.0,
        lambda_residual_aux_loss=0.0,
        lambda_mid_delta_loss=0.0,
        lambda_zbin_cvar=0.0,
        zbin_cvar_top_frac=0.34,
        lambda_condinv=0.02,
        lambda_rex=2.0,
        weight_0=float(candidate.weight_0),
        weight_25=2.2,
        weight_45=1.0,
        hard_region_loss_weight=float(candidate.hard_weight),
        hard_region_soc_threshold=float(candidate.hard_soc),
        hard_region_dynamic_weight=float(candidate.hard_dynamic),
        hard_region_cold_only=bool(candidate.hard_cold_only),
        stage2_select_rule="none",
        skip_stage2=True,
        save_predictions=True,
        save_final_weights=True,
        final_only_eval=True,
        skip_train_final_eval=True,
        valid_split_mode="profile",
        cache_dataset_cuda=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Screen paper-defensible US06 0C candidates.")
    p.add_argument("--base-dir", default=str(Path(__file__).resolve().parents[1]))
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
        manifest_rows.append(
            {
                "idx": idx,
                "name": candidate.name,
                "model_kind": candidate.model_kind,
                "recurrent": candidate.recurrent,
                "feature_set": candidate.feature_set,
                "weight_0": candidate.weight_0,
                "hard_weight": candidate.hard_weight,
                "hard_cold_only": candidate.hard_cold_only,
                "prefix": prefix_for(candidate, seeds, int(args.epochs), int(args.batch_size)),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(out_dir / "soc80_us06_0c_screen_manifest.csv", index=False)

    print(
        f"us06_0c_screen jobs={len(CANDIDATES)} seeds={args.seeds} epochs={args.epochs} batch_size={args.batch_size}",
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
            f"=== done idx={idx} candidate={candidate.name} elapsed={format_eta(elapsed)} completed={completed} "
            f"skipped={skipped} eta={format_eta(mean_duration * remaining)} {time.strftime('%Y-%m-%d %H:%M:%S %Z')} ===",
            flush=True,
        )
    print(f"us06_0c_screen finished completed={completed} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
