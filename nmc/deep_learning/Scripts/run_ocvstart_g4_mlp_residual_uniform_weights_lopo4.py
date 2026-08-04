#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soc_decomp.nmc_vcorr_it_train_dst_selector_run import TrainDSTSelectorConfig, run

DATA_ROOT = ROOT / "nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean"
RESULT_DIR = ROOT / "nmc_goal_vcorr_it_train_dst_selector_results"
PROFILES = ("VALIDATION", "DST", "FUDS", "US06")
HOLDOUTS = ("US06", "DST", "FUDS", "VALIDATION")


def parse_seeds(raw: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


def prefix_for(holdout: str, seeds: tuple[int, ...], epochs: int, batch_size: int) -> str:
    seed_tag = "s" + "".join(str(seed) for seed in seeds)
    return f"uniformw_lopo4_mlp_residual_g4_holdout{holdout.lower()}_{seed_tag}_b{batch_size}_e{epochs}"


def cfg_for(holdout: str, args: argparse.Namespace, prefix: str) -> TrainDSTSelectorConfig:
    epochs = int(args.epochs)
    return TrainDSTSelectorConfig(
        base_dir=ROOT,
        raw_root=DATA_ROOT,
        output_prefix=prefix,
        seeds=parse_seeds(args.seeds),
        train_profiles=tuple(profile for profile in PROFILES if profile != holdout),
        valid_profiles=("NONE",),
        test_profiles=(holdout,),
        train_temperatures=(0.0, 25.0, 45.0),
        valid_temperatures=(),
        test_temperatures=(0.0, 25.0, 45.0),
        window_len=50,
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
        model_kind="anchor_residual_window_summary_mlp",
        recurrent="lstm",
        head_kind="linear",
        dropout=0.06,
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
        weight_0=1.0,
        weight_25=1.0,
        weight_45=1.0,
        stage2_select_rule="none",
        skip_stage2=True,
        save_predictions=True,
        save_final_weights=True,
        final_only_eval=True,
        skip_train_final_eval=True,
        valid_split_mode="profile",
        cache_dataset_cuda=True,
    )


def selected_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame[frame["variant"].astype(str).str.contains("_selected_seed")].copy()
    if selected.empty:
        selected = frame.sort_values(["seed", "epoch"]).groupby("seed", as_index=False).tail(1)
    return selected.sort_values("seed").drop_duplicates("seed", keep="last")


def summarize(seeds: tuple[int, ...], epochs: int, batch_size: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for holdout in HOLDOUTS:
        prefix = prefix_for(holdout, seeds, epochs, batch_size)
        path = RESULT_DIR / f"{prefix}_test_summary.csv"
        if not path.exists():
            continue
        for row in selected_rows(path).to_dict("records"):
            rows.append(
                {
                    "holdout": holdout,
                    "seed": int(row["seed"]),
                    "mae_0C": float(row["0.0"]),
                    "mae_25C": float(row["25.0"]),
                    "mae_45C": float(row["45.0"]),
                    "mean_mae": float((row["0.0"] + row["25.0"] + row["45.0"]) / 3.0),
                    "summary_file": path.name,
                }
            )
    seed_df = pd.DataFrame(rows)
    if seed_df.empty:
        return seed_df
    seed_df.to_csv(RESULT_DIR / "uniformw_lopo4_mlp_residual_g4_seed_results.csv", index=False)
    aggregate = (
        seed_df.groupby("holdout", as_index=False)
        .agg(
            mae_0C=("mae_0C", "mean"),
            std_0C=("mae_0C", "std"),
            mae_25C=("mae_25C", "mean"),
            std_25C=("mae_25C", "std"),
            mae_45C=("mae_45C", "mean"),
            std_45C=("mae_45C", "std"),
            mean_mae=("mean_mae", "mean"),
            std_mean=("mean_mae", "std"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values("mean_mae")
    )
    aggregate.to_csv(RESULT_DIR / "uniformw_lopo4_mlp_residual_g4_aggregate.csv", index=False)
    return aggregate


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run G4 residual MLP 4-LOPO with uniform temperature loss weights.")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(
        [
            {
                "idx": idx,
                "holdout": holdout,
                "train_profiles": ",".join(profile for profile in PROFILES if profile != holdout),
                "model": "MLP",
                "head": "residual",
                "feature": "paper_g4_all_ema",
                "weight_0": 1.0,
                "weight_25": 1.0,
                "weight_45": 1.0,
                "seeds": ",".join(str(seed) for seed in seeds),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "prefix": prefix_for(holdout, seeds, int(args.epochs), int(args.batch_size)),
            }
            for idx, holdout in enumerate(HOLDOUTS)
        ]
    )
    manifest.to_csv(RESULT_DIR / "uniformw_lopo4_mlp_residual_g4_manifest.csv", index=False)

    started = time.time()
    durations: list[float] = []
    completed = skipped = failed = 0
    for idx, holdout in enumerate(HOLDOUTS):
        prefix = prefix_for(holdout, seeds, int(args.epochs), int(args.batch_size))
        summary = RESULT_DIR / f"{prefix}_test_summary.csv"
        if summary.exists() and not args.force:
            skipped += 1
            print(f"=== skip idx={idx} holdout={holdout} ===", flush=True)
            continue
        print(f"=== start idx={idx} holdout={holdout} weights=1,1,1 ===", flush=True)
        t0 = time.time()
        try:
            run(cfg_for(holdout, args, prefix))
            completed += 1
            duration = time.time() - t0
            durations.append(duration)
            eta = sum(durations) / len(durations) * (len(HOLDOUTS) - idx - 1)
            print(
                f"=== done idx={idx} elapsed={format_eta(duration)} completed={completed} "
                f"skipped={skipped} failed={failed} eta={format_eta(eta)} ===",
                flush=True,
            )
        except Exception:
            failed += 1
            raise
        finally:
            summarize(seeds, int(args.epochs), int(args.batch_size))

    aggregate = summarize(seeds, int(args.epochs), int(args.batch_size))
    print(
        f"all_done completed={completed} skipped={skipped} failed={failed} elapsed={format_eta(time.time() - started)}",
        flush=True,
    )
    if not aggregate.empty:
        print(aggregate.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
