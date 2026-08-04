#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from soc_decomp.nmc_vcorr_it_train_dst_selector_run import TrainDSTSelectorConfig, run


FEATURE_SETS = (
    "paper_g0_raw",
    "paper_g1_derivatives",
    "paper_t6_voltage_ema_all",
    "paper_t7_current_abs_ema_all",
    "paper_g7_no_current_ema",
    "paper_g8_no_voltage_ema",
    "paper_g4_all_ema",
)
PRIORITY_FEATURES = ("paper_g0_raw",)
REMAINING_FEATURES = (
    "paper_g1_derivatives",
    "paper_t6_voltage_ema_all",
    "paper_t7_current_abs_ema_all",
    "paper_g7_no_current_ema",
    "paper_g8_no_voltage_ema",
)
BASELINE_MODELS = ("gru", "rnn", "transformer", "mlp")
BACKBONES = ("gru", "rnn", "transformer", "mlp")
PROFILE_SET = ("DST", "FUDS", "US06")
HOLDOUTS = ("US06", "DST", "FUDS")
FEATURE_CODES = {
    "paper_g0_raw": "g0",
    "paper_g1_derivatives": "g1",
    "paper_t6_voltage_ema_all": "t6",
    "paper_t7_current_abs_ema_all": "t7",
    "paper_g7_no_current_ema": "g7",
    "paper_g8_no_voltage_ema": "g8",
    "paper_g4_all_ema": "g4",
}
EXP_CODES = {
    "window_summary_mlp": "mlp",
    "transformer": "tr",
}


@dataclass(frozen=True)
class Job:
    suite: str
    exp: str
    head: str
    model_kind: str
    recurrent: str
    feature_set: str
    holdout: str


def parse_seeds(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


def train_profiles_for_holdout(holdout: str) -> tuple[str, ...]:
    holdout = holdout.upper()
    return tuple(profile for profile in PROFILE_SET if profile != holdout)


def add_lstm_feature_jobs(jobs: list[Job]) -> None:
    for holdout in HOLDOUTS:
        for feature_set in FEATURE_SETS:
            jobs.append(Job("feature_ablation", "lstm", "normal", "single", "lstm", feature_set, holdout))
            jobs.append(Job("feature_ablation", "lstm", "residual", "anchor_residual_sequence", "lstm", feature_set, holdout))


def add_g4_baseline_jobs(jobs: list[Job]) -> None:
    for holdout in HOLDOUTS:
        for baseline in BASELINE_MODELS:
            if baseline == "mlp":
                jobs.append(
                    Job(
                        "baseline_model",
                        "window_summary_mlp",
                        "normal",
                        "window_summary_mlp",
                        "lstm",
                        "paper_g4_all_ema",
                        holdout,
                    )
                )
                jobs.append(
                    Job(
                        "baseline_model",
                        "window_summary_mlp",
                        "residual",
                        "anchor_residual_window_summary_mlp",
                        "lstm",
                        "paper_g4_all_ema",
                        holdout,
                    )
                )
            else:
                jobs.append(Job("baseline_model", baseline, "normal", "single", baseline, "paper_g4_all_ema", holdout))
                jobs.append(
                    Job(
                        "baseline_model",
                        baseline,
                        "residual",
                        "anchor_residual_sequence",
                        baseline,
                        "paper_g4_all_ema",
                        holdout,
                    )
                )


def add_non_lstm_feature_grid_jobs(jobs: list[Job]) -> None:
    for suite, features in (("priority_raw", PRIORITY_FEATURES), ("remaining_grid", REMAINING_FEATURES)):
        for holdout in HOLDOUTS:
            for feature_set in features:
                for backbone in BACKBONES:
                    if backbone == "mlp":
                        jobs.append(
                            Job(
                                suite,
                                "window_summary_mlp",
                                "normal",
                                "window_summary_mlp",
                                "lstm",
                                feature_set,
                                holdout,
                            )
                        )
                        jobs.append(
                            Job(
                                suite,
                                "window_summary_mlp",
                                "residual",
                                "anchor_residual_window_summary_mlp",
                                "lstm",
                                feature_set,
                                holdout,
                            )
                        )
                    else:
                        jobs.append(Job(suite, backbone, "normal", "single", backbone, feature_set, holdout))
                        jobs.append(Job(suite, backbone, "residual", "anchor_residual_sequence", backbone, feature_set, holdout))


def build_jobs() -> list[Job]:
    jobs: list[Job] = []
    add_lstm_feature_jobs(jobs)
    add_g4_baseline_jobs(jobs)
    add_non_lstm_feature_grid_jobs(jobs)
    return jobs


def seed_tag(seeds: tuple[int, ...]) -> str:
    return "s" + "".join(str(seed) for seed in seeds)


def prefix_for(job: Job, seeds: tuple[int, ...], epochs: int, batch_size: int, prefix_tag: str) -> str:
    suite_code = {"feature_ablation": "fa", "baseline_model": "bm", "priority_raw": "pr", "remaining_grid": "rg"}[job.suite]
    exp_code = EXP_CODES.get(job.exp, job.exp)
    feature_code = FEATURE_CODES[job.feature_set]
    head_code = "r" if job.head == "residual" else "n"
    return (
        f"{prefix_tag}_3l_{suite_code}_{exp_code}_{feature_code}_{head_code}_"
        f"{job.holdout.lower()}_{seed_tag(seeds)}_b{int(batch_size)}_e{int(epochs)}"
    )


def cfg_for_job(job: Job, args: argparse.Namespace, prefix: str) -> TrainDSTSelectorConfig:
    epochs = int(args.epochs)
    return TrainDSTSelectorConfig(
        base_dir=Path(args.base_dir),
        raw_root=Path(args.raw_root),
        output_prefix=prefix,
        seeds=parse_seeds(args.seeds),
        train_profiles=train_profiles_for_holdout(job.holdout),
        valid_profiles=("NONE",),
        test_profiles=(job.holdout,),
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
        model_kind=job.model_kind,
        recurrent=job.recurrent,
        head_kind="linear",
        dropout=0.06,
        feature_set=job.feature_set,
        stage2_feature_set=job.feature_set,
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
        stage2_select_rule="none",
        skip_stage2=True,
        save_predictions=True,
        save_final_weights=True,
        final_only_eval=True,
        skip_train_final_eval=not bool(args.include_train_final_eval),
        valid_split_mode="profile",
        sequence_training=False,
        cache_dataset_cuda=True,
    )


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the OCV-start three-profile LOPO full grid.")
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--base-dir", default=str(root))
    p.add_argument("--raw-root", default=str(root / "nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean"))
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--prefix-tag", default="s3")
    p.add_argument("--include-train-final-eval", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    out_dir = base_dir / "nmc_goal_vcorr_it_train_dst_selector_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    jobs = build_jobs()
    selected = jobs[int(args.start_index) :]
    if args.limit is not None:
        selected = selected[: int(args.limit)]

    manifest_rows = []
    for offset, job in enumerate(selected, start=int(args.start_index)):
        prefix = prefix_for(job, seeds, int(args.epochs), int(args.batch_size), str(args.prefix_tag))
        manifest_rows.append(
            {
                "idx": offset,
                "profile_protocol": "three_profile_3lopo",
                "profiles": ",".join(PROFILE_SET),
                "suite": job.suite,
                "exp": job.exp,
                "head": job.head,
                "model_kind": job.model_kind,
                "recurrent": job.recurrent,
                "feature_set": job.feature_set,
                "holdout": job.holdout,
                "train_profiles": ",".join(train_profiles_for_holdout(job.holdout)),
                "seeds": ",".join(str(seed) for seed in seeds),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "prefix": prefix,
                "summary_path": f"{prefix}_test_summary.csv",
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / f"{args.prefix_tag}_three_profile_3lopo_fullgrid_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print(
        f"three_profile_3lopo_fullgrid total_jobs={len(jobs)} selected_jobs={len(selected)} "
        f"profiles={','.join(PROFILE_SET)} seeds={args.seeds} epochs={args.epochs} "
        f"batch_size={args.batch_size} prefix_tag={args.prefix_tag}",
        flush=True,
    )
    print(f"manifest={manifest_path}", flush=True)

    completed = skipped = failed = 0
    durations: list[float] = []
    total_start = time.time()
    for i, job in enumerate(selected):
        idx = int(args.start_index) + i
        prefix = prefix_for(job, seeds, int(args.epochs), int(args.batch_size), str(args.prefix_tag))
        summary_path = out_dir / f"{prefix}_test_summary.csv"
        if summary_path.exists():
            skipped += 1
            print(f"=== skip existing idx={idx} prefix={prefix} ===", flush=True)
            continue
        print(
            f"=== start idx={idx} suite={job.suite} exp={job.exp} head={job.head} "
            f"holdout={job.holdout} train={','.join(train_profiles_for_holdout(job.holdout))} "
            f"feature={job.feature_set} model={job.model_kind} recurrent={job.recurrent} "
            f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} ===",
            flush=True,
        )
        start = time.time()
        try:
            run(cfg_for_job(job, args, prefix))
            elapsed = time.time() - start
            durations.append(elapsed)
            completed += 1
            mean_duration = sum(durations) / len(durations)
            remaining = len(selected) - i - 1 - skipped
            print(
                f"=== done idx={idx} prefix={prefix} elapsed={format_eta(elapsed)} "
                f"completed={completed} skipped={skipped} failed={failed} "
                f"eta={format_eta(mean_duration * max(0, remaining))} "
                f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} ===",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            print(f"=== failed idx={idx} prefix={prefix} error={type(exc).__name__}: {exc} ===", flush=True)
            if args.stop_on_error:
                raise

    total_elapsed = time.time() - total_start
    print(
        f"all_done completed={completed} skipped={skipped} failed={failed} "
        f"elapsed={format_eta(total_elapsed)} manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
