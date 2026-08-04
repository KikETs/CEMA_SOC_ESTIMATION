#!/usr/bin/env python3
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
import gc
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soc_decomp.nmc_vcorr_it_train_dst_selector_run import TrainDSTSelectorConfig, run


FEATURE_SET = "paper_g4_eqdyn_no_vcorr"
PROFILE_SET = ("VALIDATION", "DST", "FUDS", "US06")
HOLDOUTS = ("US06", "DST", "FUDS", "VALIDATION")
BASELINE_MODELS = ("lstm", "gru", "rnn", "transformer", "mlp")
HEADS = ("normal", "residual")


@dataclass(frozen=True)
class Job:
    model: str
    head: str
    model_kind: str
    recurrent: str
    holdout: str


def train_profiles_for_holdout(holdout: str) -> tuple[str, ...]:
    return tuple(profile for profile in PROFILE_SET if profile != holdout)


def build_jobs() -> list[Job]:
    jobs: list[Job] = []
    for holdout in HOLDOUTS:
        for model in BASELINE_MODELS:
            for head in HEADS:
                if model == "mlp":
                    model_kind = "window_summary_mlp" if head == "normal" else "anchor_residual_window_summary_mlp"
                    recurrent = "lstm"
                else:
                    model_kind = "single" if head == "normal" else "anchor_residual_sequence"
                    recurrent = model
                jobs.append(Job(model=model, head=head, model_kind=model_kind, recurrent=recurrent, holdout=holdout))
    return jobs


def prefix_for(job: Job) -> str:
    if job.model == "gru" and job.head == "residual":
        return (
            "endzero_lopo4_rtvar_lowv_ema120_gru_residual_eqdyn_no_vcorr_"
            f"holdout{job.holdout.lower()}_seed0_b2048_e200"
        )
    return (
        "endzero_lopo4_rtvar_lowv_ema120_baseline_"
        f"{job.model}_{job.head}_head_eqdyn_no_vcorr_holdout{job.holdout.lower()}_seed0_b2048_e200"
    )


def cfg_for_job(job: Job, prefix: str) -> TrainDSTSelectorConfig:
    return TrainDSTSelectorConfig(
        base_dir=ROOT,
        raw_root=Path("nmc_ocvstart_endzero_lopo_clean"),
        output_prefix=prefix,
        seeds=(0,),
        train_profiles=train_profiles_for_holdout(job.holdout),
        valid_profiles=("NONE",),
        test_profiles=(job.holdout,),
        train_temperatures=(0.0, 25.0, 45.0),
        valid_temperatures=(),
        test_temperatures=(0.0, 25.0, 45.0),
        epochs=200,
        stage1_eval_every=200,
        selector_min_epoch=200,
        selector_max_epoch=0,
        fixed_stage1_epoch=200,
        batch_size=2048,
        num_workers=0,
        prefetch_factor=2,
        hidden_size=128,
        layers=1,
        kernel_size=5,
        model_kind=job.model_kind,
        recurrent=job.recurrent,
        head_kind="linear",
        dropout=0.06,
        feature_set=FEATURE_SET,
        stage2_feature_set=FEATURE_SET,
        temp_mode="none",
        train_sampler="temperature_profile_regime_balanced",
        stage1_selector="last_epoch",
        selector_regime_min_windows=30,
        anchor_residual_limit=0.12,
        anchor_residual_limit_init="rand01",
        anchor_residual_limit_mode="learnable",
        anchor_residual_limit_lower=0.0,
        anchor_residual_limit_upper=0.2,
        lambda_anchor_loss=0.1 if job.head == "residual" else 0.0,
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
        v_corr_variant="rtvar_lowv_ema120",
        v_corr_tau_s=120.0,
    )


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def selected_summary(summary_path: Path) -> dict[str, float]:
    df = pd.read_csv(summary_path)
    selected = df[df["variant"].astype(str).str.contains("selected_seed0_ep200")]
    if selected.empty:
        selected = df.tail(1)
    row = selected.iloc[0]
    vals = {
        "mae_0C": float(row["0.0"]) if "0.0" in row else float("nan"),
        "mae_25C": float(row["25.0"]) if "25.0" in row else float("nan"),
        "mae_45C": float(row["45.0"]) if "45.0" in row else float("nan"),
    }
    vals["mean_mae"] = float(pd.Series([vals["mae_0C"], vals["mae_25C"], vals["mae_45C"]]).mean())
    vals["max_mae"] = float(pd.Series([vals["mae_0C"], vals["mae_25C"], vals["mae_45C"]]).max())
    return vals


def main() -> None:
    out_dir = ROOT / "nmc_goal_vcorr_it_train_dst_selector_results"
    log_dir = out_dir / "logs" / "eqdyn_no_vcorr_baseline_heads"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs()
    manifest_rows = []
    started = time.time()
    durations: list[float] = []
    completed = skipped = failed = 0
    print(
        f"start eqdyn_no_vcorr baseline heads jobs={len(jobs)} models={','.join(BASELINE_MODELS)} "
        f"heads={','.join(HEADS)} holdouts={','.join(HOLDOUTS)} feature={FEATURE_SET}",
        flush=True,
    )

    for idx, job in enumerate(jobs):
        prefix = prefix_for(job)
        summary_path = out_dir / f"{prefix}_test_summary.csv"
        by_temp_path = out_dir / f"{prefix}_by_temperature.csv"
        log_path = log_dir / f"{prefix}.log"
        row = {
            "idx": idx,
            "model": job.model,
            "head": job.head,
            "model_kind": job.model_kind,
            "recurrent": job.recurrent,
            "feature_set": FEATURE_SET,
            "holdout": job.holdout,
            "train_profiles": ",".join(train_profiles_for_holdout(job.holdout)),
            "prefix": prefix,
            "summary_path": str(summary_path),
            "by_temperature_path": str(by_temp_path),
            "log_path": str(log_path),
        }
        if summary_path.exists():
            skipped += 1
            row.update({"status": "skipped_existing", **selected_summary(summary_path), "elapsed_s": 0.0})
            manifest_rows.append(row)
            print(f"skip {idx + 1}/{len(jobs)} {job.model}/{job.head}/{job.holdout}", flush=True)
            continue

        t0 = time.time()
        print(f"run {idx + 1}/{len(jobs)} {job.model}/{job.head}/{job.holdout}", flush=True)
        try:
            with log_path.open("w") as log_fh, redirect_stdout(log_fh), redirect_stderr(log_fh):
                print(
                    f"job idx={idx} model={job.model} head={job.head} holdout={job.holdout} "
                    f"model_kind={job.model_kind} recurrent={job.recurrent}",
                    flush=True,
                )
                run(cfg_for_job(job, prefix))
            elapsed = time.time() - t0
            durations.append(elapsed)
            completed += 1
            row.update({"status": "completed", **selected_summary(summary_path), "elapsed_s": elapsed})
            remaining = len(jobs) - idx - 1
            mean_duration = sum(durations) / max(1, len(durations))
            print(
                f"done {idx + 1}/{len(jobs)} {job.model}/{job.head}/{job.holdout} "
                f"elapsed={format_eta(elapsed)} eta={format_eta(mean_duration * remaining)}",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "elapsed_s": time.time() - t0})
            print(f"failed {idx + 1}/{len(jobs)} {job.model}/{job.head}/{job.holdout}: {type(exc).__name__}: {exc}", flush=True)
        manifest_rows.append(row)
        pd.DataFrame(manifest_rows).to_csv(out_dir / "endzero_rtvar_lowv_ema120_eqdyn_no_vcorr_baseline_heads_lopo_manifest.csv", index=False)
        gc.collect()

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "endzero_rtvar_lowv_ema120_eqdyn_no_vcorr_baseline_heads_lopo_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    if "status" in manifest.columns and (manifest["status"] == "completed").any():
        done = manifest[manifest["status"].isin(["completed", "skipped_existing"])].copy()
        done.sort_values(["mean_mae", "max_mae"]).to_csv(
            out_dir / "endzero_rtvar_lowv_ema120_eqdyn_no_vcorr_baseline_heads_lopo_ranked.csv",
            index=False,
        )
    print(
        f"all_done completed={completed} skipped={skipped} failed={failed} "
        f"elapsed={format_eta(time.time() - started)} manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
