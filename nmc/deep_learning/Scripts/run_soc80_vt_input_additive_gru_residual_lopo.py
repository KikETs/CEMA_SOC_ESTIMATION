#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from soc_decomp.nmc_vcorr_it_train_dst_selector_run import TrainDSTSelectorConfig, run


PROFILE_SET = ("VALIDATION", "DST", "FUDS", "US06")
HOLDOUTS = ("US06", "DST", "FUDS", "VALIDATION")


def parse_seeds(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


def train_profiles_for_holdout(holdout: str) -> tuple[str, ...]:
    return tuple(profile for profile in PROFILE_SET if profile != holdout)


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def prefix_for(holdout: str, seeds: tuple[int, ...], epochs: int, batch_size: int) -> str:
    seed_tag = "seeds" + "".join(str(seed) for seed in seeds)
    return (
        "soc80_nofloor_lopo4_gru_residual_head_"
        "paper_g4_all_ema_vt_input_additive_"
        f"holdout{holdout.lower()}_{seed_tag}_b{int(batch_size)}_e{int(epochs)}"
    )


def cfg_for_holdout(holdout: str, args: argparse.Namespace, prefix: str) -> TrainDSTSelectorConfig:
    epochs = int(args.epochs)
    return TrainDSTSelectorConfig(
        base_dir=Path(args.base_dir),
        raw_root=Path(args.raw_root),
        output_prefix=prefix,
        seeds=parse_seeds(args.seeds),
        train_profiles=train_profiles_for_holdout(holdout),
        valid_profiles=("NONE",),
        test_profiles=(holdout,),
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
        model_kind="anchor_residual_vt_input_additive_mlp_sequence",
        recurrent="gru",
        head_kind="linear",
        dropout=float(args.dropout),
        feature_set="paper_g4_all_ema",
        stage2_feature_set="paper_g4_all_ema",
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
        weight_0=0.8,
        weight_25=2.2,
        weight_45=1.0,
        cold_corrector_limit=float(args.input_correction_limit),
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
    p = argparse.ArgumentParser(description="Run LOPO V/T input-additive GRU residual ablation.")
    p.add_argument("--base-dir", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--raw-root", default="nmc_soc80_train_nofloor_qmax")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--holdouts", default=",".join(HOLDOUTS))
    p.add_argument("--input-correction-limit", type=float, default=0.12)
    p.add_argument("--dropout", type=float, default=0.06)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    out_dir = base_dir / "nmc_goal_vcorr_it_train_dst_selector_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    holdouts = tuple(x.strip().upper() for x in str(args.holdouts).split(",") if x.strip())

    manifest_rows = []
    for idx, holdout in enumerate(holdouts):
        manifest_rows.append(
            {
                "idx": idx,
                "holdout": holdout,
                "train_profiles": ",".join(train_profiles_for_holdout(holdout)),
                "feature_set": "paper_g4_all_ema",
                "model_kind": "anchor_residual_vt_input_additive_mlp_sequence",
                "recurrent": "gru",
                "head": "residual",
                "input_correction_limit": float(args.input_correction_limit),
                "prefix": prefix_for(holdout, seeds, int(args.epochs), int(args.batch_size)),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(out_dir / "soc80_vt_input_additive_gru_residual_lopo_manifest.csv", index=False)

    print(
        f"vt_input_additive_gru_residual_lopo jobs={len(holdouts)} seeds={args.seeds} "
        f"epochs={args.epochs} batch_size={args.batch_size} input_limit={args.input_correction_limit}",
        flush=True,
    )
    durations: list[float] = []
    completed = skipped = 0
    for idx, holdout in enumerate(holdouts):
        prefix = prefix_for(holdout, seeds, int(args.epochs), int(args.batch_size))
        summary_path = out_dir / f"{prefix}_test_summary.csv"
        if summary_path.exists():
            skipped += 1
            print(f"=== skip existing idx={idx} holdout={holdout} prefix={prefix} ===", flush=True)
            continue
        started = time.time()
        print(
            f"=== start idx={idx} holdout={holdout} train={','.join(train_profiles_for_holdout(holdout))} "
            f"prefix={prefix} {time.strftime('%Y-%m-%d %H:%M:%S %Z')} ===",
            flush=True,
        )
        run(cfg_for_holdout(holdout, args, prefix))
        elapsed = time.time() - started
        durations.append(elapsed)
        completed += 1
        remaining = len(holdouts) - idx - 1
        mean_duration = sum(durations) / max(1, len(durations))
        print(
            f"=== done idx={idx} holdout={holdout} elapsed={format_eta(elapsed)} "
            f"completed={completed} skipped={skipped} eta={format_eta(mean_duration * remaining)} "
            f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} ===",
            flush=True,
        )
    print(f"vt_input_additive_gru_residual_lopo finished completed={completed} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
