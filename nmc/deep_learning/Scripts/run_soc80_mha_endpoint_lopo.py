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
HOLDOUTS = ("US06", "DST", "FUDS", "VALIDATION")


@dataclass(frozen=True)
class Candidate:
    name: str
    window_len: int


CANDIDATES = (
    Candidate("w400_mha_endpoint", 400),
)


def parse_seeds(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


def parse_list(raw: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in str(raw).split(",") if x.strip())


def train_profiles_for_holdout(holdout: str) -> tuple[str, ...]:
    return tuple(profile for profile in PROFILE_SET if profile != holdout)


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def prefix_for(
    candidate: Candidate,
    holdout: str,
    recurrent: str,
    seeds: tuple[int, ...],
    epochs: int,
    batch_size: int,
    tag: str,
) -> str:
    seed_tag = "seeds" + "".join(str(seed) for seed in seeds)
    return (
        f"soc80_nofloor_lopo4_{tag}_{recurrent}_mha_endpoint_normal_head_paper_g4_all_ema_"
        f"{candidate.name}_holdout{holdout.lower()}_{seed_tag}_b{int(batch_size)}_e{int(epochs)}"
    )


def cfg_for_job(
    candidate: Candidate,
    holdout: str,
    args: argparse.Namespace,
    prefix: str,
) -> TrainDSTSelectorConfig:
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
        hidden_size=int(args.hidden_size),
        layers=int(args.layers),
        kernel_size=5,
        model_kind="single_mha_endpoint",
        recurrent=str(args.recurrent),
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
        stage2_select_rule="none",
        skip_stage2=True,
        save_predictions=True,
        save_final_weights=True,
        final_only_eval=True,
        skip_train_final_eval=True,
        valid_split_mode="profile",
        sequence_training=False,
        cache_dataset_cuda=True,
    )


def read_selected_summary(path: Path, candidate: Candidate, holdout: str, recurrent: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "variant" in df.columns:
        selected = df[df["variant"].astype(str).str.contains("selected", case=False, na=False)]
        if len(selected):
            df = selected
    rows = []
    for _, row in df.iterrows():
        vals = [float(row[str(temp)]) for temp in (0.0, 25.0, 45.0) if str(temp) in row]
        rows.append(
            {
                "candidate": candidate.name,
                "window_len": int(candidate.window_len),
                "recurrent": str(recurrent),
                "holdout": holdout,
                "seed": int(row["seed"]),
                "mae_0C": float(row["0.0"]),
                "mae_25C": float(row["25.0"]),
                "mae_45C": float(row["45.0"]),
                "mae_mean_temp": sum(vals) / len(vals),
                "source": path.name,
            }
        )
    return pd.DataFrame(rows)


def write_aggregate(out_dir: Path, manifest: pd.DataFrame, tag: str) -> tuple[Path, Path]:
    seed_rows = []
    for _, job in manifest.iterrows():
        summary_path = out_dir / str(job["summary_path"])
        if not summary_path.exists():
            continue
        seed_rows.append(
            read_selected_summary(
                summary_path,
                Candidate(str(job["candidate"]), int(job["window_len"])),
                str(job["holdout"]),
                str(job["recurrent"]),
            )
        )
    seed_path = out_dir / f"soc80_nofloor_lopo4_{tag}_mha_endpoint_seed_summary.csv"
    agg_path = out_dir / f"soc80_nofloor_lopo4_{tag}_mha_endpoint_agg_summary.csv"
    if not seed_rows:
        pd.DataFrame().to_csv(seed_path, index=False)
        pd.DataFrame().to_csv(agg_path, index=False)
        return seed_path, agg_path

    seed_df = pd.concat(seed_rows, ignore_index=True)
    seed_df.to_csv(seed_path, index=False)
    agg = (
        seed_df.groupby(["candidate", "window_len", "recurrent", "holdout"], as_index=False)
        .agg(
            mae_0C_mean=("mae_0C", "mean"),
            mae_0C_std=("mae_0C", "std"),
            mae_25C_mean=("mae_25C", "mean"),
            mae_25C_std=("mae_25C", "std"),
            mae_45C_mean=("mae_45C", "mean"),
            mae_45C_std=("mae_45C", "std"),
            mae_mean_temp=("mae_mean_temp", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    overall = (
        seed_df.groupby(["candidate", "window_len", "recurrent"], as_index=False)
        .agg(
            holdout=("holdout", lambda _: "ALL"),
            mae_0C_mean=("mae_0C", "mean"),
            mae_0C_std=("mae_0C", "std"),
            mae_25C_mean=("mae_25C", "mean"),
            mae_25C_std=("mae_25C", "std"),
            mae_45C_mean=("mae_45C", "mean"),
            mae_45C_std=("mae_45C", "std"),
            mae_mean_temp=("mae_mean_temp", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    agg = pd.concat([agg, overall[agg.columns]], ignore_index=True)
    agg = agg.sort_values(["holdout", "mae_0C_mean", "mae_mean_temp", "candidate"])
    agg.to_csv(agg_path, index=False)
    return seed_path, agg_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SOC80 no-floor last-query MHA endpoint LOPO screen.")
    p.add_argument("--base-dir", default=str(PROJECT_ROOT))
    p.add_argument("--raw-root", default="nmc_soc80_train_nofloor_qmax")
    p.add_argument("--seeds", default="0")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.06)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--recurrent", choices=["lstm", "gru", "rnn"], default="lstm")
    p.add_argument("--holdouts", default=",".join(HOLDOUTS))
    p.add_argument("--candidates", default=",".join(candidate.name for candidate in CANDIDATES))
    p.add_argument("--tag", default="mha_endpoint")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    out_dir = base_dir / "nmc_goal_vcorr_it_train_dst_selector_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    holdouts = tuple(x.upper() for x in parse_list(args.holdouts))
    candidate_map = {candidate.name: candidate for candidate in CANDIDATES}
    candidates = tuple(candidate_map[name] for name in parse_list(args.candidates))

    manifest_rows = []
    for candidate in candidates:
        for holdout in holdouts:
            prefix = prefix_for(
                candidate,
                holdout,
                str(args.recurrent),
                seeds,
                int(args.epochs),
                int(args.batch_size),
                str(args.tag),
            )
            manifest_rows.append(
                {
                    "candidate": candidate.name,
                    "window_len": int(candidate.window_len),
                    "holdout": holdout,
                    "train_profiles": ",".join(train_profiles_for_holdout(holdout)),
                    "feature_set": "paper_g4_all_ema",
                    "model_kind": "single_mha_endpoint",
                    "recurrent": str(args.recurrent),
                    "head": "normal",
                    "seeds": ",".join(str(seed) for seed in seeds),
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "hidden_size": int(args.hidden_size),
                    "layers": int(args.layers),
                    "prefix": prefix,
                    "summary_path": f"{prefix}_test_summary.csv",
                }
            )
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / f"soc80_nofloor_lopo4_{args.tag}_{args.recurrent}_mha_endpoint_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print(
        f"mha_endpoint_lopo jobs={len(manifest)} recurrent={args.recurrent} "
        f"holdouts={','.join(holdouts)} seeds={args.seeds} epochs={args.epochs} batch_size={args.batch_size}",
        flush=True,
    )
    print(f"manifest={manifest_path}", flush=True)

    durations: list[float] = []
    completed = skipped = 0
    for idx, job in manifest.iterrows():
        candidate = candidate_map[str(job["candidate"])]
        holdout = str(job["holdout"])
        prefix = str(job["prefix"])
        summary_path = out_dir / f"{prefix}_test_summary.csv"
        if summary_path.exists():
            skipped += 1
            print(f"=== skip existing idx={idx} candidate={candidate.name} holdout={holdout} prefix={prefix} ===", flush=True)
            continue
        started = time.time()
        print(
            f"=== start idx={idx} candidate={candidate.name} holdout={holdout} "
            f"train={','.join(train_profiles_for_holdout(holdout))} prefix={prefix} "
            f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} ===",
            flush=True,
        )
        run(cfg_for_job(candidate, holdout, args, prefix))
        elapsed = time.time() - started
        durations.append(elapsed)
        completed += 1
        remaining = len(manifest) - idx - 1
        mean_duration = sum(durations) / max(1, len(durations))
        print(
            f"=== done idx={idx} candidate={candidate.name} holdout={holdout} elapsed={format_eta(elapsed)} "
            f"completed={completed} skipped={skipped} eta={format_eta(mean_duration * remaining)} "
            f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} ===",
            flush=True,
        )

    seed_path, agg_path = write_aggregate(out_dir, manifest, str(args.tag))
    print(f"aggregate={agg_path}", flush=True)
    print(
        f"mha_endpoint_lopo finished completed={completed} skipped={skipped} "
        f"seed_summary={seed_path} agg_summary={agg_path}",
        flush=True,
    )
    if agg_path.exists() and agg_path.stat().st_size:
        print("Aggregate summary:", flush=True)
        print(pd.read_csv(agg_path).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
