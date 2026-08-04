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
    sequence_training: bool


CANDIDATES = (
    Candidate("w100_sequence", 100, True),
    Candidate("w150_sequence", 150, True),
    Candidate("w200_sequence", 200, True),
    Candidate("w300_sequence", 300, True),
    Candidate("w400_sequence", 400, True),
    Candidate("w500_sequence", 500, True),
)


def parse_seeds(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


def parse_list(raw: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in str(raw).split(",") if x.strip())


def train_profiles_for_holdout(holdout: str) -> tuple[str, ...]:
    return tuple(profile for profile in PROFILE_SET if profile != holdout)


def candidate_by_name() -> dict[str, Candidate]:
    return {candidate.name: candidate for candidate in CANDIDATES}


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def prefix_for(
    candidate: Candidate,
    holdout: str,
    seeds: tuple[int, ...],
    epochs: int,
    batch_size: int,
    tag: str,
    feature_set: str,
    recurrent: str,
    model_kind: str,
) -> str:
    seed_tag = "seeds" + "".join(str(seed) for seed in seeds)
    if str(recurrent) == "gru" and str(model_kind) == "single":
        model_label = "gru_normal_head"
    else:
        model_label = f"{str(recurrent)}_{str(model_kind)}"
    return (
        f"soc80_nofloor_lopo4_{tag}_{model_label}_{feature_set}_"
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
        lr=float(args.lr),
        hidden_size=int(args.hidden_size),
        layers=1,
        kernel_size=5,
        model_kind=str(args.model_kind),
        recurrent=str(args.recurrent),
        head_kind="linear",
        dropout=float(args.dropout),
        anchor_residual_limit=float(args.anchor_residual_limit),
        anchor_residual_limit_init=str(args.anchor_residual_limit_init),
        anchor_residual_limit_mode=str(args.anchor_residual_limit_mode),
        anchor_residual_limit_lower=float(args.anchor_residual_limit_lower),
        anchor_residual_limit_upper=float(args.anchor_residual_limit_upper),
        lambda_anchor_loss=float(args.lambda_anchor_loss),
        feature_set=str(args.feature_set),
        stage2_feature_set=str(args.feature_set),
        temp_mode="none",
        train_sampler="temperature_profile_regime_balanced",
        stage1_selector="last_epoch",
        selector_regime_min_windows=30,
        lambda_regime_cvar=0.0,
        lambda_zbin_cvar=0.0,
        zbin_cvar_top_frac=0.34,
        hard_region_loss_weight=float(args.hard_region_loss_weight),
        hard_region_soc_lower=float(args.hard_region_soc_lower),
        hard_region_soc_upper=float(args.hard_region_soc_upper),
        hard_region_dynamic_weight=float(args.hard_region_dynamic_weight),
        hard_region_cold_only=bool(args.hard_region_cold_only),
        cold_corrector_limit=float(args.cold_corrector_limit),
        cold_corrector_soc_threshold=float(args.cold_corrector_soc_threshold),
        cold_corrector_temp_threshold=float(args.cold_corrector_temp_threshold),
        cold_corrector_gate_sharpness=float(args.cold_corrector_gate_sharpness),
        cold_corrector_sag_gate=bool(args.cold_corrector_sag_gate),
        ssl_pretrain_epochs=int(args.ssl_pretrain_epochs),
        ssl_lr=float(args.ssl_lr),
        ssl_recon_weight=float(args.ssl_recon_weight),
        ssl_next_vcorr_weight=float(args.ssl_next_vcorr_weight),
        ssl_slope_weight=float(args.ssl_slope_weight),
        lambda_condinv=float(args.lambda_condinv),
        lambda_rex=float(args.lambda_rex),
        weight_0=float(args.weight_0),
        weight_25=float(args.weight_25),
        weight_45=float(args.weight_45),
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


def read_selected_summary(path: Path, candidate: Candidate, holdout: str) -> pd.DataFrame:
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
                "sequence_training": bool(candidate.sequence_training),
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
                Candidate(str(job["candidate"]), int(job["window_len"]), bool(job["sequence_training"])),
                str(job["holdout"]),
            )
        )
    seed_path = out_dir / f"soc80_nofloor_lopo4_{tag}_gru_normal_longcontext_seed_summary.csv"
    agg_path = out_dir / f"soc80_nofloor_lopo4_{tag}_gru_normal_longcontext_agg_summary.csv"
    if not seed_rows:
        pd.DataFrame().to_csv(seed_path, index=False)
        pd.DataFrame().to_csv(agg_path, index=False)
        return seed_path, agg_path

    seed_df = pd.concat(seed_rows, ignore_index=True)
    seed_df.to_csv(seed_path, index=False)
    agg = (
        seed_df.groupby(["candidate", "window_len", "sequence_training", "holdout"], as_index=False)
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
        seed_df.groupby(["candidate", "window_len", "sequence_training"], as_index=False)
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
    agg = agg.sort_values(["holdout", "mae_mean_temp", "mae_0C_mean", "candidate"])
    agg.to_csv(agg_path, index=False)
    return seed_path, agg_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SOC80 no-floor long-context GRU normal-head LOPO screen.")
    p.add_argument("--base-dir", default=str(PROJECT_ROOT))
    p.add_argument("--raw-root", default="nmc_soc80_train_nofloor_qmax")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.06)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-0", type=float, default=0.8)
    p.add_argument("--weight-25", type=float, default=2.2)
    p.add_argument("--weight-45", type=float, default=1.0)
    p.add_argument("--lambda-condinv", type=float, default=0.02)
    p.add_argument("--lambda-rex", type=float, default=2.0)
    p.add_argument("--ssl-pretrain-epochs", type=int, default=0)
    p.add_argument("--ssl-lr", type=float, default=8e-4)
    p.add_argument("--ssl-recon-weight", type=float, default=1.0)
    p.add_argument("--ssl-next-vcorr-weight", type=float, default=0.5)
    p.add_argument("--ssl-slope-weight", type=float, default=0.25)
    p.add_argument("--hard-region-loss-weight", type=float, default=0.0)
    p.add_argument("--hard-region-soc-lower", type=float, default=-1.0)
    p.add_argument("--hard-region-soc-upper", type=float, default=-1.0)
    p.add_argument("--hard-region-dynamic-weight", type=float, default=0.5)
    p.add_argument("--hard-region-cold-only", action="store_true")
    p.add_argument("--cold-corrector-limit", type=float, default=0.08)
    p.add_argument("--cold-corrector-soc-threshold", type=float, default=0.35)
    p.add_argument("--cold-corrector-temp-threshold", type=float, default=-0.45)
    p.add_argument("--cold-corrector-gate-sharpness", type=float, default=8.0)
    p.add_argument("--cold-corrector-sag-gate", action="store_true")
    p.add_argument("--anchor-residual-limit", type=float, default=0.12)
    p.add_argument("--anchor-residual-limit-init", choices=["value", "rand01"], default="rand01")
    p.add_argument("--anchor-residual-limit-mode", choices=["fixed", "learnable", "bounded_learnable"], default="learnable")
    p.add_argument("--anchor-residual-limit-lower", type=float, default=0.0)
    p.add_argument("--anchor-residual-limit-upper", type=float, default=0.2)
    p.add_argument("--lambda-anchor-loss", type=float, default=0.1)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--feature-set", default="paper_g4_all_ema")
    p.add_argument("--recurrent", choices=["gru", "lstm", "rnn", "transformer"], default="gru")
    p.add_argument(
        "--model-kind",
        choices=[
            "single",
            "anchor_residual_cold_low_soc_signed_sequence",
            "normal_cold_low_soc_signed_sequence",
            "normal_cold_low_soc_positive_sequence",
            "regime_gated_normal_anchor_signed_sequence",
        ],
        default="single",
    )
    p.add_argument("--holdouts", default=",".join(HOLDOUTS))
    p.add_argument("--candidates", default=",".join(candidate.name for candidate in CANDIDATES))
    p.add_argument("--tag", default="longctx")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    out_dir = base_dir / "nmc_goal_vcorr_it_train_dst_selector_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    holdouts = tuple(x.upper() for x in parse_list(args.holdouts))
    candidates = []
    candidate_map = candidate_by_name()
    for name in parse_list(args.candidates):
        if name not in candidate_map:
            raise ValueError(f"Unknown candidate={name!r}; choices={','.join(candidate_map)}")
        candidates.append(candidate_map[name])
    candidates = tuple(candidates)

    manifest_rows = []
    for candidate in candidates:
        for holdout in holdouts:
            prefix = prefix_for(
                candidate,
                holdout,
                seeds,
                int(args.epochs),
                int(args.batch_size),
                str(args.tag),
                str(args.feature_set),
                str(args.recurrent),
                str(args.model_kind),
            )
            manifest_rows.append(
                {
                    "candidate": candidate.name,
                    "window_len": int(candidate.window_len),
                    "sequence_training": bool(candidate.sequence_training),
                    "holdout": holdout,
                    "train_profiles": ",".join(train_profiles_for_holdout(holdout)),
                    "feature_set": str(args.feature_set),
                    "model_kind": str(args.model_kind),
                    "recurrent": str(args.recurrent),
                    "head": "normal",
                    "seeds": ",".join(str(seed) for seed in seeds),
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "lr": float(args.lr),
                    "weight_0": float(args.weight_0),
                    "weight_25": float(args.weight_25),
                    "weight_45": float(args.weight_45),
                    "lambda_condinv": float(args.lambda_condinv),
                    "lambda_rex": float(args.lambda_rex),
                    "ssl_pretrain_epochs": int(args.ssl_pretrain_epochs),
                    "ssl_lr": float(args.ssl_lr),
                    "ssl_recon_weight": float(args.ssl_recon_weight),
                    "ssl_next_vcorr_weight": float(args.ssl_next_vcorr_weight),
                    "ssl_slope_weight": float(args.ssl_slope_weight),
                    "hard_region_loss_weight": float(args.hard_region_loss_weight),
                    "hard_region_soc_lower": float(args.hard_region_soc_lower),
                    "hard_region_soc_upper": float(args.hard_region_soc_upper),
                    "hard_region_dynamic_weight": float(args.hard_region_dynamic_weight),
                    "hard_region_cold_only": bool(args.hard_region_cold_only),
                    "cold_corrector_limit": float(args.cold_corrector_limit),
                    "cold_corrector_soc_threshold": float(args.cold_corrector_soc_threshold),
                    "cold_corrector_temp_threshold": float(args.cold_corrector_temp_threshold),
                    "cold_corrector_gate_sharpness": float(args.cold_corrector_gate_sharpness),
                    "cold_corrector_sag_gate": bool(args.cold_corrector_sag_gate),
                    "anchor_residual_limit": float(args.anchor_residual_limit),
                    "anchor_residual_limit_init": str(args.anchor_residual_limit_init),
                    "anchor_residual_limit_mode": str(args.anchor_residual_limit_mode),
                    "anchor_residual_limit_lower": float(args.anchor_residual_limit_lower),
                    "anchor_residual_limit_upper": float(args.anchor_residual_limit_upper),
                    "lambda_anchor_loss": float(args.lambda_anchor_loss),
                    "hidden_size": int(args.hidden_size),
                    "prefix": prefix,
                    "summary_path": f"{prefix}_test_summary.csv",
                }
            )
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / f"soc80_nofloor_lopo4_{args.tag}_gru_normal_longcontext_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print(
        f"longcontext_gru_normal_lopo jobs={len(manifest)} candidates={','.join(c.name for c in candidates)} "
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
        remaining = len(manifest) - int(idx) - 1
        mean_duration = sum(durations) / max(1, len(durations))
        print(
            f"=== done idx={idx} candidate={candidate.name} holdout={holdout} elapsed={format_eta(elapsed)} "
            f"completed={completed} skipped={skipped} eta={format_eta(mean_duration * remaining)} "
            f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} ===",
            flush=True,
        )
        _, agg_path = write_aggregate(out_dir, manifest, str(args.tag))
        print(f"aggregate={agg_path}", flush=True)

    seed_path, agg_path = write_aggregate(out_dir, manifest, str(args.tag))
    print(
        f"longcontext_gru_normal_lopo finished completed={completed} skipped={skipped} "
        f"seed_summary={seed_path} agg_summary={agg_path}",
        flush=True,
    )
    if agg_path.exists() and agg_path.stat().st_size:
        agg = pd.read_csv(agg_path)
        if len(agg):
            print("Aggregate summary:")
            print(agg.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
