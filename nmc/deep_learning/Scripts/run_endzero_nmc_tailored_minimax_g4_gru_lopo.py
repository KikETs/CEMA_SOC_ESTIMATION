#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soc_decomp.nmc_vcorr_it_train_dst_selector_run import TrainDSTSelectorConfig, run


PROFILE_SET = ("VALIDATION", "DST", "FUDS", "US06")
HOLDOUTS = ("US06", "DST", "FUDS", "VALIDATION")
HEADS = ("normal", "residual")
VARIANT_SLUGS = {
    "nmc_tailored_minimax_v1": "nmc_tailored_minimax",
    "nmc_tailored_minimax": "nmc_tailored_minimax",
    "nmc_tailored_minimax_aligned_v1": "nmc_tailored_minimax_aligned",
    "nmc_tailored_minimax_aligned": "nmc_tailored_minimax_aligned",
}


def parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in str(raw).split(",") if x.strip())


def parse_seeds(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


def seed_tag(seeds: tuple[int, ...]) -> str:
    return "seed" + str(seeds[0]) if len(seeds) == 1 else "seeds" + "".join(str(s) for s in seeds)


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def train_profiles_for_holdout(holdout: str) -> tuple[str, ...]:
    holdout = str(holdout).upper()
    return tuple(profile for profile in PROFILE_SET if profile != holdout)


def variant_slug(v_corr_variant: str) -> str:
    v_corr_variant = str(v_corr_variant)
    if v_corr_variant not in VARIANT_SLUGS:
        raise ValueError(f"Unsupported v_corr_variant={v_corr_variant!r}")
    return VARIANT_SLUGS[v_corr_variant]


def prefix_for(head: str, holdout: str, seeds: tuple[int, ...], epochs: int, batch_size: int, v_corr_variant: str) -> str:
    return (
        f"endzero_lopo4_{variant_slug(v_corr_variant)}_g4_gru_"
        f"{head}_head_holdout{holdout.lower()}_{seed_tag(seeds)}_b{int(batch_size)}_e{int(epochs)}"
    )


def cfg_for_job(head: str, holdout: str, args: argparse.Namespace, prefix: str) -> TrainDSTSelectorConfig:
    epochs = int(args.epochs)
    is_residual = str(head) == "residual"
    return TrainDSTSelectorConfig(
        base_dir=ROOT,
        raw_root=Path("nmc_ocvstart_endzero_lopo_clean"),
        output_prefix=prefix,
        seeds=parse_seeds(args.seeds),
        train_profiles=train_profiles_for_holdout(holdout),
        valid_profiles=("NONE",),
        test_profiles=(holdout.upper(),),
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
        model_kind="anchor_residual_sequence" if is_residual else "single",
        recurrent="gru",
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
        lambda_anchor_loss=0.1 if is_residual else 0.0,
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
        stage2_select_rule="none",
        skip_stage2=True,
        save_predictions=True,
        save_final_weights=True,
        final_only_eval=True,
        skip_train_final_eval=True,
        valid_split_mode="profile",
        cache_dataset_cuda=True,
        v_corr_variant=str(args.v_corr_variant),
    )


def read_summary(path: Path, head: str, holdout: str, v_corr_variant: str) -> pd.DataFrame:
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
                "label_protocol": "ocvstart_endzero",
                "v_corr_variant": str(v_corr_variant),
                "feature_set": "paper_g4_all_ema",
                "model": "gru",
                "head": head,
                "holdout": holdout.upper(),
                "seed": int(row["seed"]),
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
        if summary_path.exists():
            frames.append(read_summary(summary_path, str(job["head"]), str(job["holdout"]), str(job["v_corr_variant"])))
    seed_path = out_dir / f"{tag}_seed_summary.csv"
    agg_path = out_dir / f"{tag}_agg_summary.csv"
    if not frames:
        pd.DataFrame().to_csv(seed_path, index=False)
        pd.DataFrame().to_csv(agg_path, index=False)
        return seed_path, agg_path
    seed_df = pd.concat(frames, ignore_index=True).sort_values(["holdout", "head", "seed"], kind="stable")
    seed_df.to_csv(seed_path, index=False)
    group_cols = ["label_protocol", "v_corr_variant", "feature_set", "model", "head", "holdout"]
    agg = (
        seed_df.groupby(group_cols, as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mae_0C_mean=("mae_0C", "mean"),
            mae_25C_mean=("mae_25C", "mean"),
            mae_45C_mean=("mae_45C", "mean"),
            mae_mean_temp=("mae_mean_temp", "mean"),
        )
        .sort_values(["holdout", "mae_mean_temp", "head"], kind="stable")
    )
    overall = (
        seed_df.assign(holdout="ALL")
        .groupby(group_cols, as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mae_0C_mean=("mae_0C", "mean"),
            mae_25C_mean=("mae_25C", "mean"),
            mae_45C_mean=("mae_45C", "mean"),
            mae_mean_temp=("mae_mean_temp", "mean"),
        )
    )
    agg = pd.concat([agg, overall[agg.columns]], ignore_index=True)
    agg = agg.sort_values(["holdout", "mae_mean_temp", "head"], kind="stable")
    agg.to_csv(agg_path, index=False)
    return seed_path, agg_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run end-zero NMC-tailored minimax V_corr G4 GRU LOPO heads.")
    p.add_argument("--seeds", default="0")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--holdouts", default=",".join(HOLDOUTS))
    p.add_argument("--heads", default=",".join(HEADS))
    p.add_argument("--tag", default="endzero_nmc_tailored_minimax_g4_gru_lopo_heads")
    p.add_argument("--v-corr-variant", default="nmc_tailored_minimax_v1", choices=tuple(VARIANT_SLUGS))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / "nmc_goal_vcorr_it_train_dst_selector_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    v_corr_variant = str(args.v_corr_variant)
    holdouts = tuple(x.upper() for x in parse_csv(args.holdouts))
    heads = tuple(x.lower() for x in parse_csv(args.heads))
    manifest_rows = []
    for holdout in holdouts:
        for head in heads:
            prefix = prefix_for(head, holdout, seeds, int(args.epochs), int(args.batch_size), v_corr_variant)
            manifest_rows.append(
                {
                    "holdout": holdout,
                    "head": head,
                    "v_corr_variant": v_corr_variant,
                    "train_profiles": ",".join(train_profiles_for_holdout(holdout)),
                    "seeds": ",".join(str(seed) for seed in seeds),
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "prefix": prefix,
                    "summary_path": f"{prefix}_test_summary.csv",
                }
            )
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / f"{args.tag}_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(
        f"endzero_nmc_tailored_minimax_g4_gru_lopo jobs={len(manifest)} "
        f"holdouts={','.join(holdouts)} heads={','.join(heads)} seeds={args.seeds} "
        f"epochs={args.epochs} batch_size={args.batch_size} v_corr_variant={v_corr_variant}",
        flush=True,
    )
    print(f"manifest={manifest_path}", flush=True)
    durations: list[float] = []
    completed = skipped = 0
    for idx, job in manifest.iterrows():
        head = str(job["head"])
        holdout = str(job["holdout"])
        prefix = str(job["prefix"])
        summary_path = out_dir / f"{prefix}_test_summary.csv"
        if summary_path.exists() and not bool(args.force):
            skipped += 1
            print(f"=== skip existing idx={idx} holdout={holdout} head={head} prefix={prefix} ===", flush=True)
            write_outputs(out_dir, manifest, str(args.tag))
            continue
        started = time.time()
        print(
            f"=== start idx={idx} holdout={holdout} head={head} "
            f"train={','.join(train_profiles_for_holdout(holdout))} prefix={prefix} "
            f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} ===",
            flush=True,
        )
        run(cfg_for_job(head, holdout, args, prefix))
        elapsed = time.time() - started
        durations.append(elapsed)
        completed += 1
        remaining = len(manifest) - int(idx) - 1
        mean_duration = sum(durations) / max(1, len(durations))
        print(
            f"=== done idx={idx} holdout={holdout} head={head} elapsed={format_eta(elapsed)} "
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
