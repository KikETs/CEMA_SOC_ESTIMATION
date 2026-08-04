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


def parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in str(raw).split(",") if x.strip())


def parse_seeds(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


def train_profiles_for_holdout(holdout: str) -> tuple[str, ...]:
    holdout = holdout.upper()
    return tuple(profile for profile in PROFILE_SET if profile != holdout)


def seed_tag(seeds: tuple[int, ...]) -> str:
    return "seeds" + "".join(str(seed) for seed in seeds)


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def prefix_for(recurrent: str, holdout: str, seeds: tuple[int, ...], epochs: int, batch_size: int) -> str:
    return (
        "ocvstart_g4eqdyn_w50_"
        f"{recurrent}_normal_h64_l2_drop0p07_cvar0p1_wd2e3_"
        f"holdout{holdout.lower()}_{seed_tag(seeds)}_b{int(batch_size)}_e{int(epochs)}"
    )


def cfg_for_job(recurrent: str, holdout: str, args: argparse.Namespace, prefix: str) -> TrainDSTSelectorConfig:
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
        lr=float(args.lr),
        weight_decay=2e-3,
        hidden_size=64,
        layers=2,
        kernel_size=5,
        model_kind="single",
        recurrent=str(recurrent),
        head_kind="linear",
        dropout=0.07,
        feature_set="paper_g4_eqdyn",
        stage2_feature_set="paper_g4_eqdyn",
        temp_mode="none",
        train_sampler="temperature_profile_regime_balanced",
        stage1_selector="last_epoch",
        selector_regime_min_windows=30,
        lambda_regime_cvar=0.1,
        regime_cvar_top_frac=0.34,
        lambda_zbin_cvar=0.0,
        zbin_cvar_top_frac=0.34,
        lambda_condinv=0.02,
        lambda_rex=2.0,
        weight_0=0.8,
        weight_25=1.5,
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


def read_summary(path: Path, recurrent: str, holdout: str) -> pd.DataFrame:
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
                "label_protocol": "ocvstart",
                "feature_set": "paper_g4_eqdyn",
                "model_kind": "single",
                "recurrent": recurrent,
                "head_kind": "linear",
                "holdout": holdout.upper(),
                "seed": int(row["seed"]),
                "window_len": 50,
                "sequence_training": False,
                "hidden_size": 64,
                "layers": 2,
                "dropout": 0.07,
                "weight_decay": 2e-3,
                "weight_0": 0.8,
                "weight_25": 1.5,
                "lambda_regime_cvar": 0.1,
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
    for _, job in manifest.iterrows():
        summary_path = out_dir / str(job["summary_path"])
        if not summary_path.exists():
            continue
        frames.append(read_summary(summary_path, str(job["recurrent"]), str(job["holdout"])))
    seed_path = out_dir / f"{tag}_seed_summary.csv"
    agg_path = out_dir / f"{tag}_agg_summary.csv"
    if not frames:
        pd.DataFrame().to_csv(seed_path, index=False)
        pd.DataFrame().to_csv(agg_path, index=False)
        return seed_path, agg_path
    seed_df = pd.concat(frames, ignore_index=True).sort_values(
        ["holdout", "recurrent", "seed"], kind="stable"
    )
    seed_df.to_csv(seed_path, index=False)
    group_cols = [
        "label_protocol",
        "feature_set",
        "model_kind",
        "recurrent",
        "head_kind",
        "holdout",
        "window_len",
        "sequence_training",
        "hidden_size",
        "layers",
        "dropout",
        "weight_decay",
        "weight_0",
        "weight_25",
        "lambda_regime_cvar",
    ]
    agg = (
        seed_df.groupby(group_cols, as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mae_0C_mean=("mae_0C", "mean"),
            mae_0C_std=("mae_0C", "std"),
            mae_25C_mean=("mae_25C", "mean"),
            mae_25C_std=("mae_25C", "std"),
            mae_45C_mean=("mae_45C", "mean"),
            mae_45C_std=("mae_45C", "std"),
            mae_mean_temp=("mae_mean_temp", "mean"),
            mae_mean_temp_std=("mae_mean_temp", "std"),
        )
        .sort_values(["holdout", "mae_mean_temp", "mae_0C_mean", "recurrent"], kind="stable")
    )
    overall = (
        seed_df.assign(holdout="ALL")
        .groupby(group_cols, as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mae_0C_mean=("mae_0C", "mean"),
            mae_0C_std=("mae_0C", "std"),
            mae_25C_mean=("mae_25C", "mean"),
            mae_25C_std=("mae_25C", "std"),
            mae_45C_mean=("mae_45C", "mean"),
            mae_45C_std=("mae_45C", "std"),
            mae_mean_temp=("mae_mean_temp", "mean"),
            mae_mean_temp_std=("mae_mean_temp", "std"),
        )
    )
    agg = pd.concat([agg, overall[agg.columns]], ignore_index=True)
    agg = agg.sort_values(["holdout", "mae_mean_temp", "mae_0C_mean", "recurrent"], kind="stable")
    agg.to_csv(agg_path, index=False)
    return seed_path, agg_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run OCV-start G4 eqdyn GRU/LSTM LOPO.")
    p.add_argument("--base-dir", default=str(PROJECT_ROOT))
    p.add_argument("--raw-root", default=str(PROJECT_ROOT / "nmc_ocvstart_lopo_clean"))
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--holdouts", default=",".join(HOLDOUTS))
    p.add_argument("--recurrents", default="gru,lstm")
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--tag", default="ocvstart_g4eqdyn_w50_gru_lstm_lopo")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    out_dir = base_dir / "nmc_goal_vcorr_it_train_dst_selector_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    holdouts = tuple(x.upper() for x in parse_csv(args.holdouts))
    recurrents = tuple(x.lower() for x in parse_csv(args.recurrents))

    manifest_rows = []
    for recurrent in recurrents:
        for holdout in holdouts:
            prefix = prefix_for(recurrent, holdout, seeds, int(args.epochs), int(args.batch_size))
            manifest_rows.append(
                {
                    "label_protocol": "ocvstart",
                    "raw_root": str(Path(args.raw_root)),
                    "recurrent": recurrent,
                    "holdout": holdout,
                    "train_profiles": ",".join(train_profiles_for_holdout(holdout)),
                    "seeds": ",".join(str(seed) for seed in seeds),
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "window_len": 50,
                    "feature_set": "paper_g4_eqdyn",
                    "model_kind": "single",
                    "head_kind": "linear",
                    "hidden_size": 64,
                    "layers": 2,
                    "dropout": 0.07,
                    "weight_decay": 2e-3,
                    "weight_0": 0.8,
                    "weight_25": 1.5,
                    "lambda_regime_cvar": 0.1,
                    "prefix": prefix,
                    "summary_path": f"{prefix}_test_summary.csv",
                }
            )
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / f"{args.tag}_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print(
        f"ocvstart_g4_gru_lstm jobs={len(manifest)} recurrents={','.join(recurrents)} "
        f"holdouts={','.join(holdouts)} seeds={args.seeds} epochs={args.epochs} "
        f"batch_size={args.batch_size}",
        flush=True,
    )
    print(f"manifest={manifest_path}", flush=True)

    durations: list[float] = []
    completed = skipped = 0
    for idx, job in manifest.iterrows():
        recurrent = str(job["recurrent"])
        holdout = str(job["holdout"])
        prefix = str(job["prefix"])
        summary_path = out_dir / f"{prefix}_test_summary.csv"
        if summary_path.exists():
            skipped += 1
            print(f"=== skip existing idx={idx} recurrent={recurrent} holdout={holdout} prefix={prefix} ===", flush=True)
            write_outputs(out_dir, manifest, str(args.tag))
            continue
        started = time.time()
        print(
            f"=== start idx={idx} recurrent={recurrent} holdout={holdout} "
            f"train={','.join(train_profiles_for_holdout(holdout))} prefix={prefix} "
            f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} ===",
            flush=True,
        )
        run(cfg_for_job(recurrent, holdout, args, prefix))
        elapsed = time.time() - started
        durations.append(elapsed)
        completed += 1
        remaining = len(manifest) - int(idx) - 1
        mean_duration = sum(durations) / max(1, len(durations))
        print(
            f"=== done idx={idx} recurrent={recurrent} holdout={holdout} elapsed={format_eta(elapsed)} "
            f"completed={completed} skipped={skipped} eta={format_eta(mean_duration * remaining)} "
            f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} ===",
            flush=True,
        )
        seed_path, agg_path = write_outputs(out_dir, manifest, str(args.tag))
        print(f"summary={seed_path} aggregate={agg_path}", flush=True)

    seed_path, agg_path = write_outputs(out_dir, manifest, str(args.tag))
    print(f"finished completed={completed} skipped={skipped}", flush=True)
    print(f"summary={seed_path}", flush=True)
    print(f"aggregate={agg_path}", flush=True)


if __name__ == "__main__":
    main()
