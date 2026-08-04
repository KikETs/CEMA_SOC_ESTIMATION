#!/usr/bin/env python3
"""LOPO CEMA-LSTM runner with a learnable anchor-residual scale L.

This script is intentionally isolated from the manuscript/release repository.
It calls the copied experimental training module under this folder only.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


ISOLATED_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODULE = "soc_decomp.nmc_vcorr_it_train_dst_selector_run"
RAW_ROOT = "nmc_soc_ocvstart_relabelled_from_lc_ocv/data/NMC SAMSUNG INR 18650 2Ah"
RESULT_DIR = ISOLATED_ROOT / "nmc_goal_vcorr_it_train_dst_selector_results"
LOG_DIR = ISOLATED_ROOT / "lopo_cema_lstm_anchorL_logs"
REPORT_DIR = ISOLATED_ROOT / "reports"
PROFILES = ("VALIDATION", "DST", "FUDS", "US06")


def _default_python() -> Path:
    override = os.environ.get("CEMA_PYTHON")
    if override:
        return Path(override).expanduser().resolve()

    for relative in ("anaconda3/envs/torch_env/bin/python", "miniconda3/envs/torch_env/bin/python"):
        candidate = Path.home() / relative
        if candidate.exists():
            return candidate.resolve()

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = Path(conda_prefix) / "bin" / "python"
        if candidate.exists():
            return candidate.resolve()

    return Path(sys.executable).resolve()


@dataclass(frozen=True)
class RunSpec:
    run_group: str
    model_id: str
    feature_set: str
    model_kind: str
    recurrent: str
    layers: int
    hidden_size: int = 128
    notes: str = ""


FEATURE_ABLATION = (
    RunSpec("feature_ablation", "CEMA-LSTM_G0", "paper_g0_raw", "anchor_residual_sequence", "lstm", 1, notes="G0 corrected voltage/current/temperature"),
    RunSpec("feature_ablation", "CEMA-LSTM_G1", "paper_g1_derivatives", "anchor_residual_sequence", "lstm", 1, notes="G1 local derivative/excitation descriptors"),
    RunSpec("feature_ablation", "CEMA-LSTM_G4", "paper_g4_all_ema", "anchor_residual_sequence", "lstm", 1, notes="G4 proposed causal EMA input"),
    RunSpec("feature_ablation", "CEMA-LSTM_G6", "paper_g6_full23", "anchor_residual_sequence", "lstm", 1, notes="G6 full derivative/excitation descriptors"),
    RunSpec("feature_ablation", "CEMA-LSTM_G7", "paper_g7_no_current_ema", "anchor_residual_sequence", "lstm", 1, notes="G7 voltage-memory-only ablation"),
    RunSpec("feature_ablation", "CEMA-LSTM_G8", "paper_g8_no_voltage_ema", "anchor_residual_sequence", "lstm", 1, notes="G8 current-memory-only ablation"),
)

MODEL_COMPARISON = (
    RunSpec("model_comparison", "CEMA-LSTM_proposed", "paper_g4_all_ema", "anchor_residual_sequence", "lstm", 1, notes="Proposed LSTM encoder with anchor-residual head and learnable L"),
    RunSpec("model_comparison", "Vanilla_LSTM_G4", "paper_g4_all_ema", "single", "lstm", 1, notes="Simple LSTM regression head"),
    RunSpec("model_comparison", "GRU_G4", "paper_g4_all_ema", "single", "gru", 1, notes="Simple GRU regression head"),
    RunSpec("model_comparison", "Transformer_G4_L2", "paper_g4_all_ema", "single", "transformer", 2, notes="Two-layer transformer encoder, past-window attention only"),
    RunSpec("model_comparison", "Endpoint_MLP_G4", "paper_g4_all_ema", "endpoint_mlp", "lstm", 1, notes="Endpoint G4 vector only; recurrent argument ignored"),
    RunSpec("model_comparison", "Vanilla_RNN_G4", "paper_g4_all_ema", "single", "rnn", 1, notes="Simple tanh RNN regression head"),
)

ANCHOR_HEAD_CONTROLS = (
    RunSpec("anchor_head_control", "CEMA-LSTM_anchorL_G4", "paper_g4_all_ema", "anchor_residual_sequence", "lstm", 1, notes="Anchor-residual LSTM control"),
    RunSpec("anchor_head_control", "GRU_anchorL_G4", "paper_g4_all_ema", "anchor_residual_sequence", "gru", 1, notes="Anchor-residual GRU control"),
    RunSpec("anchor_head_control", "RNN_anchorL_G4", "paper_g4_all_ema", "anchor_residual_sequence", "rnn", 1, notes="Anchor-residual tanh RNN control"),
    RunSpec("anchor_head_control", "Transformer_anchorL_G4_L2", "paper_g4_all_ema", "anchor_residual_sequence", "transformer", 2, notes="Anchor-residual transformer control"),
    RunSpec("anchor_head_control", "Endpoint_MLP_anchorL_G4", "paper_g4_all_ema", "anchor_residual_endpoint_mlp", "lstm", 1, notes="Anchor-residual endpoint MLP control"),
)


def _profile_set_for_holdout(holdout: str) -> tuple[str, ...]:
    return tuple(p for p in PROFILES if p != holdout)


def _sanitize(text: str) -> str:
    return text.lower().replace("-", "").replace("_", "")


def _prefix(spec: RunSpec, holdout: str, seeds: str, epochs: int) -> str:
    seed_tag = "seed" + "".join(s.strip() for s in seeds.split(",") if s.strip())
    return f"lopo_anchorL_{_sanitize(spec.model_id)}_holdout{holdout.lower()}_{seed_tag}_e{int(epochs)}"


def _experiment_key(spec: RunSpec) -> tuple[str, str, str, int, int]:
    return (spec.feature_set, spec.model_kind, spec.recurrent, int(spec.layers), int(spec.hidden_size))


def _local_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _specs_for_mode(mode: str) -> list[RunSpec]:
    if mode == "smoke":
        return [MODEL_COMPARISON[0]]
    if mode == "feature-ablation":
        return list(FEATURE_ABLATION)
    if mode == "model-comparison":
        return list(MODEL_COMPARISON)
    if mode == "anchor-head-controls":
        return list(ANCHOR_HEAD_CONTROLS)
    if mode == "all":
        return list((*FEATURE_ABLATION, *MODEL_COMPARISON, *ANCHOR_HEAD_CONTROLS))
    raise ValueError(f"Unknown mode={mode!r}")


def _base_args(args: argparse.Namespace, spec: RunSpec, holdout: str, prefix: str) -> list[str]:
    train_profiles = ",".join(_profile_set_for_holdout(holdout))
    cmd = [
        "-m",
        SOURCE_MODULE,
        "--base-dir",
        str(ISOLATED_ROOT),
        "--raw-root",
        RAW_ROOT,
        "--output-prefix",
        prefix,
        "--seeds",
        str(args.seeds),
        "--epochs",
        str(args.epochs),
        "--stage1-eval-every",
        str(args.epochs),
        "--eval-every",
        str(args.epochs),
        "--diagnostic-test-every",
        str(args.epochs),
        "--selector-min-epoch",
        str(args.epochs),
        "--selector-max-epoch",
        "0",
        "--fixed-stage1-epoch",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--hidden-size",
        str(spec.hidden_size),
        "--layers",
        str(spec.layers),
        "--kernel-size",
        "5",
        "--model-kind",
        spec.model_kind,
        "--recurrent",
        spec.recurrent,
        "--head-kind",
        "linear",
        "--dropout",
        "0.06",
        "--feature-set",
        spec.feature_set,
        "--stage2-feature-set",
        spec.feature_set,
        "--temp-mode",
        "none",
        "--train-sampler",
        "temperature_profile_regime_balanced",
        "--stage1-selector",
        "last_epoch",
        "--selector-regime-min-windows",
        "30",
        "--anchor-residual-limit",
        "0.12",
        "--anchor-residual-limit-init",
        "rand01",
        "--anchor-residual-limit-mode",
        "learnable",
        "--lambda-anchor-loss",
        "0.1",
        "--lambda-condinv",
        "0.02",
        "--lambda-rex",
        "2.0",
        "--weight-0",
        "0.8",
        "--weight-25",
        "2.2",
        "--weight-45",
        "1.0",
        "--stage2-select-rule",
        "none",
        "--skip-stage2",
        "--test-blind",
        "--train-profiles",
        train_profiles,
        "--valid-profiles",
        "NONE",
        "--test-profiles",
        holdout,
        "--train-temperatures",
        "0,25,45",
        "--valid-temperatures",
        "all",
        "--test-temperatures",
        "0,25,45",
        "--valid-split-mode",
        "train_profile_blocks",
        "--valid-block-rows",
        "800",
        "--valid-block-mod",
        "5",
        "--valid-block-index",
        "4",
    ]
    if args.save_predictions:
        cmd.append("--save-predictions")
    if args.cache_dataset_cuda:
        cmd.append("--cache-dataset-cuda")
    return cmd


def build_jobs(args: argparse.Namespace) -> list[dict[str, object]]:
    holds = [h.strip().upper() for h in args.holdouts.split(",") if h.strip()]
    bad = sorted(set(holds) - set(PROFILES))
    if bad:
        raise ValueError(f"Unknown holdout profile(s): {bad}")
    specs = _specs_for_mode(args.mode)
    jobs: list[dict[str, object]] = []
    aliases: list[dict[str, object]] = []
    canonical_by_key: dict[tuple[str, tuple[str, str, str, int, int]], dict[str, object]] = {}
    for holdout in holds:
        for spec in specs:
            prefix = _prefix(spec, holdout, args.seeds, args.epochs)
            train_profiles = ",".join(_profile_set_for_holdout(holdout))
            key = (holdout, _experiment_key(spec))
            canonical = canonical_by_key.get(key)
            if canonical is not None:
                aliases.append(
                    {
                        "prefix": prefix,
                        "source_prefix": canonical["prefix"],
                        "holdout_profile": holdout,
                        "train_profiles": train_profiles,
                        "spec": asdict(spec),
                        "source_spec": canonical["spec"],
                        "expected_summary": str(RESULT_DIR / f"{canonical['prefix']}_test_summary.csv"),
                        "alias_reason": "duplicate experimental config; source_prefix contains the trained artifacts",
                    }
                )
                continue
            cmd = [str(args.python), *_base_args(args, spec, holdout, prefix)]
            job = {
                "prefix": prefix,
                "holdout_profile": holdout,
                "train_profiles": train_profiles,
                "spec": asdict(spec),
                "command": cmd,
                "expected_summary": str(RESULT_DIR / f"{prefix}_test_summary.csv"),
            }
            canonical_by_key[key] = job
            jobs.append(job)
    if args.max_jobs > 0:
        jobs = jobs[: int(args.max_jobs)]
        prefixes = {str(job["prefix"]) for job in jobs}
        aliases = [alias for alias in aliases if str(alias.get("source_prefix")) in prefixes]
    args.dedupe_aliases = aliases
    return jobs


def write_manifest(jobs: list[dict[str, object]], args: argparse.Namespace) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "original_repository_path": "/home/user/바탕화면/DL/LSTM_STATELESS_DECOMP_SOC",
        "isolated_working_path": str(ISOLATED_ROOT),
        "writes_only_under_isolated_path": True,
        "source_module": SOURCE_MODULE,
        "mode": args.mode,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "cache_dataset_cuda": bool(args.cache_dataset_cuda),
        "residual_L": {
            "mode": "learnable",
            "init": "rand01",
            "initial_reference_value": 0.12,
            "note": "rand01 follows the earlier seeded random-L request; L is optimized as an unbounded scalar parameter.",
        },
        "jobs": jobs,
        "aliases": list(getattr(args, "dedupe_aliases", [])),
    }
    (REPORT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    with (REPORT_DIR / "reproduction_commands.md").open("w", encoding="utf-8") as f:
        f.write("# Reproduction commands\n\n")
        f.write("All commands are intended to run from the isolated folder only.\n\n")
        for job in jobs:
            f.write(f"## {job['prefix']}\n\n")
            f.write("```bash\n")
            f.write(" ".join(str(x) for x in job["command"]))
            f.write("\n```\n\n")
        aliases = list(getattr(args, "dedupe_aliases", []))
        if aliases:
            f.write("# Aliases not rerun\n\n")
            f.write("These names share the same experimental config as `source_prefix`; no duplicate training command is scheduled.\n\n")
            for alias in aliases:
                f.write(f"- `{alias['prefix']}` -> `{alias['source_prefix']}`\n")


def _run_one_job(idx: int, total: int, job: dict[str, object], *, force: bool, env: dict[str, str]) -> None:
    prefix = str(job["prefix"])
    summary = Path(str(job["expected_summary"]))
    if summary.exists() and not force:
        print(f"[skip {idx}/{total}] {prefix}: existing summary", flush=True)
        return
    log_path = LOG_DIR / f"{prefix}.log"
    print(f"[run {idx}/{total}] {prefix}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        start_unix = time.time()
        start_perf = time.perf_counter()
        log.write(f"prefix={prefix}\n")
        log.write(f"started_at_local={_local_timestamp()}\n")
        log.write(f"started_at_unix={start_unix:.6f}\n")
        log.write("command=" + " ".join(str(x) for x in job["command"]) + "\n\n")
        log.flush()
        try:
            subprocess.run(
                [str(x) for x in job["command"]],
                cwd=ISOLATED_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            elapsed = time.perf_counter() - start_perf
            log.write("\n")
            log.write("job_status=failed\n")
            log.write(f"returncode={int(exc.returncode)}\n")
            log.write(f"finished_at_local={_local_timestamp()}\n")
            log.write(f"finished_at_unix={time.time():.6f}\n")
            log.write(f"elapsed_seconds={elapsed:.3f}\n")
            log.write(f"elapsed_hms={_format_duration(elapsed)}\n")
            log.flush()
            print(f"[fail {idx}/{total}] {prefix}: {_format_duration(elapsed)}", flush=True)
            raise
        elapsed = time.perf_counter() - start_perf
        log.write("\n")
        log.write("job_status=complete\n")
        log.write("returncode=0\n")
        log.write(f"finished_at_local={_local_timestamp()}\n")
        log.write(f"finished_at_unix={time.time():.6f}\n")
        log.write(f"elapsed_seconds={elapsed:.3f}\n")
        log.write(f"elapsed_hms={_format_duration(elapsed)}\n")
        log.flush()
    print(f"[done {idx}/{total}] {prefix}: {_format_duration(elapsed)}", flush=True)


def run_jobs(jobs: list[dict[str, object]], *, force: bool, env: dict[str, str], parallel_jobs: int = 1) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(parallel_jobs))
    if workers == 1:
        for idx, job in enumerate(jobs, start=1):
            _run_one_job(idx, len(jobs), job, force=force, env=env)
        return
    print(f"parallel jobs: {workers}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_run_one_job, idx, len(jobs), job, force=force, env=env)
            for idx, job in enumerate(jobs, start=1)
        ]
        for future in as_completed(futures):
            future.result()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run isolated LOPO CEMA-LSTM experiments with learnable L.")
    p.add_argument("--mode", choices=["smoke", "feature-ablation", "model-comparison", "anchor-head-controls", "all"], default="smoke")
    p.add_argument("--holdouts", default="VALIDATION,DST,FUDS,US06")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--python", type=Path, default=_default_python())
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-jobs", type=int, default=0)
    p.add_argument("--parallel-jobs", type=int, default=1)
    p.add_argument(
        "--cache-dataset-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preload each training job's window datasets onto CUDA. Use --no-cache-dataset-cuda to disable.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"original repository path: /home/user/바탕화면/DL/LSTM_STATELESS_DECOMP_SOC", flush=True)
    print(f"isolated working path: {ISOLATED_ROOT}", flush=True)
    print("write policy: all new outputs/logs/reports are written only under the isolated working path", flush=True)
    jobs = build_jobs(args)
    write_manifest(jobs, args)
    print(f"scheduled jobs: {len(jobs)}", flush=True)
    if args.dry_run:
        for job in jobs:
            print(str(job["prefix"]), flush=True)
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ISOLATED_ROOT)
    run_jobs(jobs, force=bool(args.force), env=env, parallel_jobs=int(args.parallel_jobs))


if __name__ == "__main__":
    main()
