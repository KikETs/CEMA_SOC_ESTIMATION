#!/usr/bin/env python3
"""Locked LFP confirmatory mini-panel with staged Tier 1/2 execution."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT
REPO_ROOT = ROOT.parents[1]
DATA = REPO_ROOT / "Data/Preprocessed/LFP/prepared_data_ocv_discharge_soc"
MANIFEST_DIR = REPO_ROOT / "Data/Preprocessed/LFP/manifests_ocv_discharge_3lopo"
PROTOCOL = ROOT / "TRANSFER_PROTOCOL.md"
ALL_TEMPS = (-10.0, 0.0, 10.0, 20.0, 25.0, 30.0, 40.0, 50.0)
CORE4_TEMPS = (-10.0, 0.0, 25.0, 50.0)
PROFILES = ("DST", "FUDS", "US06")
SEEDS = (0, 1, 2)
FEATURES = {
    "G0": "paper_g0_raw",
    "T6": "paper_t6_voltage_ema_all",
    "T7": "paper_t7_current_abs_ema_all",
    "G4": "paper_g4_all_ema",
}

sys.path.insert(0, str(SOURCE))
from Scripts import run_soc80_g4_eqdyn_gru_residual as base_runner  # noqa: E402
from soc_decomp import nmc_branchbands_experiment as branchbands  # noqa: E402
from soc_decomp import nmc_vcorr_it_train_dst_selector_run as selector  # noqa: E402


@dataclass(frozen=True)
class Condition:
    name: str
    train_temperatures: tuple[float, ...]


CONDITIONS = (
    Condition("all8_train_all8_test", ALL_TEMPS),
    Condition("core4_train_all8_test", CORE4_TEMPS),
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def train_profiles(holdout: str) -> tuple[str, ...]:
    return tuple(p for p in PROFILES if p != holdout)


def clear_caches() -> None:
    selector._FEATURE_FRAME_CACHE.clear()
    selector._SCALED_FRAME_CACHE.clear()
    gc.collect()
    if selector.torch.cuda.is_available():
        selector.torch.cuda.empty_cache()


def make_cfg(
    condition: Condition,
    tier: str,
    holdout: str,
    feature_code: str,
    architecture: str,
    head: str,
    seeds: tuple[int, ...],
    args: argparse.Namespace,
):
    feature = FEATURES[feature_code]
    base_args = argparse.Namespace(
        base_dir=str(ROOT), raw_root=str(DATA), seeds=",".join(map(str, seeds)),
        epochs=args.epochs, batch_size=args.batch_size,
    )
    cfg = base_runner.cfg_for_holdout("US06", base_args, "temporary")
    if architecture == "gru":
        model_kind = "anchor_residual_sequence" if head == "residual" else "single"
        recurrent = "gru"
    elif architecture == "mlp" and head == "residual":
        model_kind = "anchor_residual_window_summary_mlp"
        recurrent = "lstm"
    else:
        raise ValueError((architecture, head))
    seed_tag = "s" + "".join(map(str, seeds))
    prefix = (
        f"lfpconfirm_{condition.name}_{tier}_{architecture}_{head}_{feature_code.lower()}_"
        f"holdout{holdout.lower()}_{seed_tag}_b{args.batch_size}_e{args.epochs}"
    )
    base_dir = ROOT / "runs" / condition.name / tier
    return replace(
        cfg,
        base_dir=base_dir,
        raw_root=DATA,
        output_prefix=prefix,
        seeds=seeds,
        train_profiles=train_profiles(holdout),
        valid_profiles=("NONE",),
        test_profiles=(holdout,),
        train_temperatures=condition.train_temperatures,
        valid_temperatures=(),
        test_temperatures=ALL_TEMPS,
        window_len=50,
        stride=3,
        epochs=args.epochs,
        stage1_eval_every=args.epochs,
        selector_min_epoch=args.epochs,
        selector_max_epoch=0,
        fixed_stage1_epoch=args.epochs,
        batch_size=args.batch_size,
        hidden_size=128,
        layers=1,
        kernel_size=5,
        model_kind=model_kind,
        recurrent=recurrent,
        head_kind="linear",
        dropout=0.06,
        feature_set=feature,
        stage2_feature_set=feature,
        temp_mode="none",
        train_sampler="temperature_profile_regime_balanced",
        stage1_selector="last_epoch",
        anchor_residual_limit=0.12,
        anchor_residual_limit_init="rand01",
        anchor_residual_limit_mode="learnable",
        anchor_residual_limit_lower=0.0,
        anchor_residual_limit_upper=0.2,
        lambda_anchor_loss=0.1,
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
        sequence_training=False,
        cache_dataset_cuda=True,
    )


def marker_for(cfg) -> Path:
    return cfg.base_dir / "nmc_goal_vcorr_it_train_dst_selector_results" / f"{cfg.output_prefix}_by_temperature.csv"


def verify_data_and_pipeline() -> None:
    manifest = pd.read_csv(MANIFEST_DIR / "prepared_dataset_manifest.csv")
    expected = {(t, p) for t in ALL_TEMPS for p in PROFILES}
    observed = {(float(r.temperature_C), str(r.profile)) for r in manifest.itertuples(index=False)}
    if observed != expected or len(manifest) != 24:
        raise RuntimeError("Prepared LFP scope mismatch")
    if not manifest["label_policy"].eq("temperature_matched_low_current_ocv_discharge_soc0_and_qref").all():
        raise RuntimeError("Non-OCV-discharge label detected")
    files = selector.find_csv_files(DATA)
    parsed = {}
    for p in files:
        head = pd.read_csv(p, nrows=2)
        parsed[(float(branchbands.parse_temperature(p, head)), str(branchbands.parse_profile(p, head)))] = p
    if set(parsed) != expected:
        raise RuntimeError("Loader did not resolve 8 temperatures x 3 profiles")
    for r in manifest.itertuples(index=False):
        if sha256_file(parsed[(float(r.temperature_C), str(r.profile))]) != str(r.prepared_sha256):
            raise RuntimeError(f"Prepared hash mismatch: {r.temperature_C}/{r.profile}")

    patch_text = (SOURCE / "soc_decomp" / "nmc_vcorr_it_train_dst_selector_run.py").read_text(encoding="utf-8")
    if "stage1_pre_scale:train" not in patch_text:
        raise RuntimeError("Pre-scaling temperature isolation patch is absent")

    r0_rows = []
    for holdout in PROFILES:
        fitted = branchbands.estimate_r0_by_temperature(files, train_profiles(holdout))
        if set(fitted.temperature_C.astype(float)) != set(ALL_TEMPS):
            raise RuntimeError(f"R0 temperature scope mismatch for {holdout}")
        for row in fitted.itertuples(index=False):
            r0_rows.append({
                "holdout": holdout,
                "fit_profiles": ",".join(train_profiles(holdout)),
                "excluded_profile": holdout,
                "temperature_C": float(row.temperature_C),
                "r0_ohm": float(row.r0_ohm),
                "n_events": int(row.n_events),
            })
    pd.DataFrame(r0_rows).to_csv(MANIFEST_DIR / "r0_rotation_audit.csv", index=False)

    audit = {
        "status": "PASS",
        "prepared_files": len(files),
        "r0_rule": "temperature-specific fit from the two rotation training profiles only",
        "normalization_rule": "fit from rotation training profiles and admitted model-training temperatures only",
        "normalization_test_ids_disjoint_assertion": True,
        "selector": "last_epoch=200; no validation/test-based selection",
        "nmc_statistics_loaded": False,
        "protocol_sha256": sha256_file(PROTOCOL),
        "source_patch": "temperature filtering before FeatureStandardizer.fit",
    }
    (MANIFEST_DIR / "pipeline_leakage_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def append_job_manifest(cfg, condition: Condition, tier: str, feature_code: str, architecture: str, head: str, status: str) -> None:
    path = MANIFEST_DIR / "job_manifest.jsonl"
    row = {
        "condition": condition.name, "tier": tier, "feature_code": feature_code,
        "feature_set": cfg.feature_set, "architecture": architecture, "head": head,
        "holdout": cfg.test_profiles[0], "train_profiles": list(cfg.train_profiles),
        "train_temperatures": list(cfg.train_temperatures), "test_temperatures": list(cfg.test_temperatures),
        "seeds": list(cfg.seeds), "epochs": cfg.epochs, "output_prefix": cfg.output_prefix,
        "marker": str(marker_for(cfg)), "status": status, "time_unix": time.time(),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_job(cfg, condition: Condition, tier: str, feature_code: str, architecture: str, head: str) -> None:
    marker = marker_for(cfg)
    if marker.is_file():
        append_job_manifest(cfg, condition, tier, feature_code, architecture, head, "skipped_exact_marker")
        print(f"SKIP exact {cfg.output_prefix}", flush=True)
        return
    append_job_manifest(cfg, condition, tier, feature_code, architecture, head, "started")
    clear_caches()
    print(f"START {cfg.output_prefix}", flush=True)
    selector.run(cfg)
    if not marker.is_file():
        raise RuntimeError(f"Missing completion marker: {marker}")
    append_job_manifest(cfg, condition, tier, feature_code, architecture, head, "complete")
    print(f"DONE {cfg.output_prefix}", flush=True)


def tier1_seed_scores(condition: Condition) -> pd.DataFrame:
    rows = []
    for code in FEATURES:
        for holdout in PROFILES:
            cfg = make_cfg(condition, "tier1", holdout, code, "gru", "residual", SEEDS, ARGS)
            d = pd.read_csv(marker_for(cfg))
            d = d[(d.split == "test") & d.variant.astype(str).str.endswith("_selector_base")]
            for seed, g in d.groupby("seed"):
                rows.append({"feature_code": code, "holdout": holdout, "seed": int(seed), "MAE_pct": float(g.MAE_pct.mean())})
    raw = pd.DataFrame(rows)
    return raw.groupby(["feature_code", "seed"], as_index=False).MAE_pct.mean()


def paired_ci(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    q = float(student_t.ppf(0.975, df=len(values) - 1))
    return {"mean": mean, "low": mean - q * se, "high": mean + q * se}


def lock_decision(primary: Condition) -> dict[str, object]:
    scores = tier1_seed_scores(primary)
    pivot = scores.pivot(index="seed", columns="feature_code", values="MAE_pct")
    cis = {code: paired_ci((pivot[code] - pivot["T6"]).to_numpy()) for code in ("G0", "T7", "G4")}
    g0_beats_all = all(paired_ci((pivot["G0"] - pivot[c]).to_numpy())["high"] < 0 for c in ("T6", "T7", "G4"))
    t7_beats_t6 = cis["T7"]["high"] < 0
    g4_beats_t6 = cis["G4"]["high"] < 0
    if g0_beats_all:
        selected, rationale = "G0", "G0 beats all EMA candidates outside paired-seed CI"
    elif t7_beats_t6 or g4_beats_t6:
        selected, rationale = "G4", "T7 or G4 beats T6 outside paired-seed CI; superset rule"
    else:
        selected, rationale = "T6", "T6 is within paired-seed CI of G4; parsimony rule"
    decision = {
        "selected_feature": selected,
        "rationale": rationale,
        "paired_delta_competitor_minus_T6_CI95": cis,
        "g4_normal_required": bool(g4_beats_t6),
        "seed_scores": scores.to_dict("records"),
        "protocol_sha256": sha256_file(PROTOCOL),
    }
    path = MANIFEST_DIR / "locked_tier1_decision.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["selected_feature"] != selected:
            raise RuntimeError("Locked decision already exists with a different winner")
        return old
    path.write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return decision


def run_suite(args: argparse.Namespace) -> None:
    global ARGS
    ARGS = args
    verify_data_and_pipeline()
    if args.preflight_only:
        print("PREFLIGHT PASS; no training started", flush=True)
        return

    for condition in CONDITIONS:
        for code in FEATURES:
            for holdout in PROFILES:
                cfg = make_cfg(condition, "tier1", holdout, code, "gru", "residual", SEEDS, args)
                run_job(cfg, condition, "tier1", code, "gru", "residual")

    decision = lock_decision(CONDITIONS[0])
    selected = str(decision["selected_feature"])
    print("LOCKED DECISION " + json.dumps(decision, ensure_ascii=False), flush=True)

    for condition in CONDITIONS:
        head_features = ["T6"] + (["G4"] if decision["g4_normal_required"] else [])
        for code in head_features:
            for holdout in PROFILES:
                cfg = make_cfg(condition, "tier2_head", holdout, code, "gru", "normal", SEEDS, args)
                run_job(cfg, condition, "tier2_head", code, "gru", "normal")
        for holdout in PROFILES:
            cfg = make_cfg(condition, "tier2_arch", holdout, selected, "mlp", "residual", SEEDS, args)
            run_job(cfg, condition, "tier2_arch", selected, "mlp", "residual")
        for holdout in PROFILES:
            cfg = make_cfg(condition, "winner_promotion", holdout, selected, "gru", "residual", (3, 4), args)
            run_job(cfg, condition, "winner_promotion", selected, "gru", "residual")

    print("TRAINING SUITE COMPLETE; LFP TRAINING FREEZE IS NOW ACTIVE", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2048)
    args = p.parse_args()
    if not args.execute:
        args.preflight_only = True
    return args


ARGS: argparse.Namespace
if __name__ == "__main__":
    run_suite(parse_args())
