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


PROFILE_SET = ("VALIDATION", "DST", "FUDS", "US06")


@dataclass(frozen=True)
class Candidate:
    name: str
    feature_set: str
    sequence_training: bool
    hidden_size: int
    layers: int
    dropout: float
    weight_decay: float
    weight_0: float
    weight_25: float
    lambda_regime_cvar: float = 0.0
    regime_cvar_top_frac: float = 0.34
    lambda_rex: float = 2.0
    lambda_condinv: float = 0.02
    lr: float = 8e-4


CANDIDATES = (
    Candidate("all_ema_h128_endpoint_w2p0", "paper_g4_all_ema", False, 128, 1, 0.06, 2e-4, 2.0, 2.2),
    Candidate("all_ema_h128_endpoint_w4p0", "paper_g4_all_ema", False, 128, 1, 0.06, 2e-4, 4.0, 2.2),
    Candidate("all_ema_h128_seq_w0p8", "paper_g4_all_ema", True, 128, 1, 0.06, 2e-4, 0.8, 2.2),
    Candidate("all_ema_h128_seq_w2p0", "paper_g4_all_ema", True, 128, 1, 0.06, 2e-4, 2.0, 2.2),
    Candidate("all_ema_h128_seq_w4p0", "paper_g4_all_ema", True, 128, 1, 0.06, 2e-4, 4.0, 2.2),
    Candidate("all_ema_h64_l2_seq_cvar0p1_wd2e3_w0p8_w25p15", "paper_g4_all_ema", True, 64, 2, 0.07, 2e-3, 0.8, 1.5, 0.1),
    Candidate("all_ema_h64_l2_seq_cvar0p1_wd2e3_w1p5_w25p15", "paper_g4_all_ema", True, 64, 2, 0.07, 2e-3, 1.5, 1.5, 0.1),
    Candidate("all_ema_h64_l2_seq_cvar0p1_wd2e3_w2p0_w25p15", "paper_g4_all_ema", True, 64, 2, 0.07, 2e-3, 2.0, 1.5, 0.1),
    Candidate("eqdyn_h128_endpoint_w2p0_w25p15", "paper_g4_eqdyn", False, 128, 1, 0.06, 2e-4, 2.0, 1.5),
    Candidate("eqdyn_h128_endpoint_w4p0_w25p15", "paper_g4_eqdyn", False, 128, 1, 0.06, 2e-4, 4.0, 1.5),
)


def parse_seeds(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


def parse_list(raw: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in str(raw).split(",") if x.strip())


def train_profiles_for_holdout(holdout: str) -> tuple[str, ...]:
    holdout = holdout.upper()
    return tuple(profile for profile in PROFILE_SET if profile != holdout)


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def candidate_map() -> dict[str, Candidate]:
    return {c.name: c for c in CANDIDATES}


def seed_tag(seeds: tuple[int, ...]) -> str:
    return "seeds" + "".join(str(seed) for seed in seeds)


def prefix_for(candidate: Candidate, holdout: str, seeds: tuple[int, ...], epochs: int, batch_size: int, tag: str) -> str:
    return (
        f"soc80_goal_w50_{tag}_{candidate.name}_gru_normal_"
        f"holdout{holdout.lower()}_{seed_tag(seeds)}_b{int(batch_size)}_e{int(epochs)}"
    )


def cfg_for_job(candidate: Candidate, holdout: str, args: argparse.Namespace, prefix: str) -> TrainDSTSelectorConfig:
    epochs = int(args.epochs)
    return TrainDSTSelectorConfig(
        base_dir=Path(args.base_dir),
        raw_root=Path(args.raw_root),
        output_prefix=prefix,
        seeds=parse_seeds(args.seeds),
        train_profiles=train_profiles_for_holdout(holdout),
        valid_profiles=("NONE",),
        test_profiles=(holdout.upper(),),
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
        lr=float(candidate.lr),
        weight_decay=float(candidate.weight_decay),
        hidden_size=int(candidate.hidden_size),
        layers=int(candidate.layers),
        kernel_size=5,
        model_kind="single",
        recurrent="gru",
        head_kind="linear",
        dropout=float(candidate.dropout),
        feature_set=str(candidate.feature_set),
        stage2_feature_set=str(candidate.feature_set),
        temp_mode="none",
        train_sampler="temperature_profile_regime_balanced",
        stage1_selector="last_epoch",
        selector_regime_min_windows=30,
        lambda_regime_cvar=float(candidate.lambda_regime_cvar),
        regime_cvar_top_frac=float(candidate.regime_cvar_top_frac),
        lambda_zbin_cvar=0.0,
        zbin_cvar_top_frac=0.34,
        lambda_condinv=float(candidate.lambda_condinv),
        lambda_rex=float(candidate.lambda_rex),
        weight_0=float(candidate.weight_0),
        weight_25=float(candidate.weight_25),
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


def read_summary(path: Path, candidate: Candidate, holdout: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "variant" in df.columns:
        selected = df[df["variant"].astype(str).str.contains("selected", case=False, na=False)]
        if len(selected):
            df = selected
    rows = []
    for _, row in df.drop_duplicates(subset=["seed", "0.0", "25.0", "45.0"]).iterrows():
        vals = [float(row[str(temp)]) for temp in (0.0, 25.0, 45.0) if str(temp) in row]
        rows.append(
            {
                "candidate": candidate.name,
                "holdout": holdout.upper(),
                "seed": int(row["seed"]),
                "window_len": 50,
                "sequence_training": bool(candidate.sequence_training),
                "feature_set": candidate.feature_set,
                "hidden_size": int(candidate.hidden_size),
                "layers": int(candidate.layers),
                "dropout": float(candidate.dropout),
                "weight_decay": float(candidate.weight_decay),
                "weight_0": float(candidate.weight_0),
                "weight_25": float(candidate.weight_25),
                "lambda_regime_cvar": float(candidate.lambda_regime_cvar),
                "mae_0C": float(row["0.0"]),
                "mae_25C": float(row["25.0"]),
                "mae_45C": float(row["45.0"]),
                "mae_mean_temp": sum(vals) / len(vals),
                "source": path.name,
            }
        )
    return pd.DataFrame(rows)


def write_outputs(out_dir: Path, manifest: pd.DataFrame, tag: str) -> tuple[Path, Path]:
    frames = []
    cmap = candidate_map()
    for _, job in manifest.iterrows():
        summary = out_dir / str(job["summary_path"])
        if not summary.exists():
            continue
        frames.append(read_summary(summary, cmap[str(job["candidate"])], str(job["holdout"])))
    seed_path = out_dir / f"soc80_goal_w50_{tag}_seed_summary.csv"
    agg_path = out_dir / f"soc80_goal_w50_{tag}_agg_summary.csv"
    if not frames:
        pd.DataFrame().to_csv(seed_path, index=False)
        pd.DataFrame().to_csv(agg_path, index=False)
        return seed_path, agg_path
    seed_df = pd.concat(frames, ignore_index=True).sort_values(["holdout", "mae_0C", "candidate", "seed"])
    seed_df.to_csv(seed_path, index=False)
    agg = (
        seed_df.groupby(
            [
                "candidate",
                "holdout",
                "window_len",
                "sequence_training",
                "feature_set",
                "hidden_size",
                "layers",
                "dropout",
                "weight_decay",
                "weight_0",
                "weight_25",
                "lambda_regime_cvar",
            ],
            as_index=False,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            mae_0C_mean=("mae_0C", "mean"),
            mae_0C_std=("mae_0C", "std"),
            mae_0C_min=("mae_0C", "min"),
            mae_25C_mean=("mae_25C", "mean"),
            mae_45C_mean=("mae_45C", "mean"),
            mae_mean_temp=("mae_mean_temp", "mean"),
        )
        .sort_values(["holdout", "mae_0C_mean", "mae_mean_temp", "candidate"])
    )
    agg.to_csv(agg_path, index=False)
    return seed_path, agg_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run seq_len=50 goal screen without intermediate eval.")
    p.add_argument("--base-dir", default=str(PROJECT_ROOT))
    p.add_argument("--raw-root", default="nmc_soc80_train_nofloor_qmax")
    p.add_argument("--seeds", default="0")
    p.add_argument("--epochs", type=int, default=220)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--holdouts", default="VALIDATION")
    p.add_argument("--tag", default="screen")
    p.add_argument("--candidates", default=",".join(c.name for c in CANDIDATES))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    out_dir = base_dir / "nmc_goal_vcorr_it_train_dst_selector_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    holdouts = tuple(x.upper() for x in parse_list(args.holdouts))
    cmap = candidate_map()
    candidates = tuple(cmap[name] for name in parse_list(args.candidates))

    manifest_rows = []
    for candidate in candidates:
        for holdout in holdouts:
            prefix = prefix_for(candidate, holdout, seeds, int(args.epochs), int(args.batch_size), str(args.tag))
            manifest_rows.append(
                {
                    "candidate": candidate.name,
                    "holdout": holdout,
                    "train_profiles": ",".join(train_profiles_for_holdout(holdout)),
                    "seeds": ",".join(str(seed) for seed in seeds),
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "window_len": 50,
                    "sequence_training": bool(candidate.sequence_training),
                    "feature_set": candidate.feature_set,
                    "hidden_size": int(candidate.hidden_size),
                    "layers": int(candidate.layers),
                    "dropout": float(candidate.dropout),
                    "weight_decay": float(candidate.weight_decay),
                    "weight_0": float(candidate.weight_0),
                    "weight_25": float(candidate.weight_25),
                    "lambda_regime_cvar": float(candidate.lambda_regime_cvar),
                    "prefix": prefix,
                    "summary_path": f"{prefix}_test_summary.csv",
                }
            )
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / f"soc80_goal_w50_{args.tag}_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(
        f"w50_goal_screen jobs={len(manifest)} holdouts={','.join(holdouts)} "
        f"seeds={args.seeds} epochs={args.epochs} batch_size={args.batch_size}",
        flush=True,
    )
    print(f"manifest={manifest_path}", flush=True)

    durations: list[float] = []
    completed = skipped = 0
    for idx, job in manifest.iterrows():
        candidate = cmap[str(job["candidate"])]
        holdout = str(job["holdout"])
        prefix = str(job["prefix"])
        summary_path = out_dir / f"{prefix}_test_summary.csv"
        if summary_path.exists():
            skipped += 1
            print(f"=== skip existing idx={idx} candidate={candidate.name} holdout={holdout} ===", flush=True)
            continue
        started = time.time()
        print(
            f"=== start idx={idx} candidate={candidate.name} holdout={holdout} "
            f"train={','.join(train_profiles_for_holdout(holdout))} ===",
            flush=True,
        )
        run(cfg_for_job(candidate, holdout, args, prefix))
        elapsed = time.time() - started
        durations.append(elapsed)
        completed += 1
        remaining = len(manifest) - int(idx) - 1
        mean_duration = sum(durations) / max(1, len(durations))
        print(
            f"=== done idx={idx} candidate={candidate.name} holdout={holdout} "
            f"elapsed={format_eta(elapsed)} completed={completed} skipped={skipped} "
            f"eta={format_eta(mean_duration * remaining)} ===",
            flush=True,
        )
        _, agg_path = write_outputs(out_dir, manifest, str(args.tag))
        print(f"aggregate={agg_path}", flush=True)

    seed_path, agg_path = write_outputs(out_dir, manifest, str(args.tag))
    print(f"w50_goal_screen finished seed_summary={seed_path} agg_summary={agg_path}", flush=True)
    if agg_path.exists() and agg_path.stat().st_size:
        agg = pd.read_csv(agg_path)
        if len(agg):
            print(agg.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
