#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soc_decomp.nmc_vcorr_it_train_dst_selector_run import TrainDSTSelectorConfig, run


PROFILE_SET = ("VALIDATION", "DST", "FUDS", "US06")
HOLDOUT = "US06"
OUT = ROOT / "nmc_goal_vcorr_it_train_dst_selector_results"


@dataclass(frozen=True)
class HeadSpec:
    name: str
    model_kind: str
    residual_limit: float
    residual_limit_init: str
    residual_limit_mode: str
    residual_limit_lower: float = 0.0
    residual_limit_upper: float = 0.12
    lambda_residual_zeromean_loss: float = 0.0
    cold_corrector_limit: float = 0.08
    cold_corrector_soc_threshold: float = 0.35


SPECS = (
    HeadSpec(
        name="fixed_l012",
        model_kind="anchor_residual_sequence",
        residual_limit=0.12,
        residual_limit_init="value",
        residual_limit_mode="fixed",
    ),
    HeadSpec(
        name="bounded_l012",
        model_kind="anchor_residual_sequence",
        residual_limit=0.06,
        residual_limit_init="value",
        residual_limit_mode="bounded_learnable",
    ),
    HeadSpec(
        name="zeromean_bounded_l012_lam5",
        model_kind="anchor_residual_sequence",
        residual_limit=0.06,
        residual_limit_init="value",
        residual_limit_mode="bounded_learnable",
        lambda_residual_zeromean_loss=5.0,
    ),
    HeadSpec(
        name="dynamic_only_fixed_l012",
        model_kind="anchor_residual_dynamic_only_sequence",
        residual_limit=0.12,
        residual_limit_init="value",
        residual_limit_mode="fixed",
    ),
    HeadSpec(
        name="dynamic_gated_fixed_l012",
        model_kind="anchor_residual_dynamic_gated_sequence",
        residual_limit=0.12,
        residual_limit_init="value",
        residual_limit_mode="fixed",
    ),
    HeadSpec(
        name="normal_dynamic_corr_l006",
        model_kind="normal_dynamic_correction_sequence",
        residual_limit=0.06,
        residual_limit_init="value",
        residual_limit_mode="fixed",
    ),
    HeadSpec(
        name="temp_budget_l004_010_004",
        model_kind="anchor_residual_temp_budget_sequence",
        residual_limit=0.10,
        residual_limit_init="value",
        residual_limit_mode="fixed",
    ),
    HeadSpec(
        name="dynamic_gated_zeromean_l012_lam5",
        model_kind="anchor_residual_dynamic_gated_sequence",
        residual_limit=0.12,
        residual_limit_init="value",
        residual_limit_mode="fixed",
        lambda_residual_zeromean_loss=5.0,
    ),
    HeadSpec(
        name="low_voltage_tail_l006",
        model_kind="normal_low_voltage_tail_sequence",
        residual_limit=0.0,
        residual_limit_init="value",
        residual_limit_mode="fixed",
        cold_corrector_limit=0.06,
        cold_corrector_soc_threshold=0.35,
    ),
    HeadSpec(
        name="affine_calibration_l006",
        model_kind="normal_affine_calibration_sequence",
        residual_limit=0.06,
        residual_limit_init="value",
        residual_limit_mode="fixed",
    ),
)


def train_profiles_for_holdout(holdout: str) -> tuple[str, ...]:
    return tuple(profile for profile in PROFILE_SET if profile != holdout.upper())


def cfg_for_spec(spec: HeadSpec) -> TrainDSTSelectorConfig:
    prefix = f"ocvstart_lopo4_headscreen_{spec.name}_gru_g4_r0ema120_holdoutus06_seed0_b2048_e200"
    return TrainDSTSelectorConfig(
        base_dir=ROOT,
        raw_root=ROOT / "nmc_ocvstart_lopo_clean",
        output_prefix=prefix,
        seeds=(0,),
        train_profiles=train_profiles_for_holdout(HOLDOUT),
        valid_profiles=("NONE",),
        test_profiles=(HOLDOUT,),
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
        model_kind=spec.model_kind,
        recurrent="gru",
        head_kind="linear",
        dropout=0.06,
        feature_set="paper_g4_all_ema",
        stage2_feature_set="paper_g4_all_ema",
        temp_mode="none",
        train_sampler="temperature_profile_regime_balanced",
        stage1_selector="last_epoch",
        selector_regime_min_windows=30,
        anchor_residual_limit=spec.residual_limit,
        anchor_residual_limit_init=spec.residual_limit_init,
        anchor_residual_limit_mode=spec.residual_limit_mode,
        anchor_residual_limit_lower=spec.residual_limit_lower,
        anchor_residual_limit_upper=spec.residual_limit_upper,
        lambda_anchor_loss=0.1,
        lambda_residual_zeromean_loss=spec.lambda_residual_zeromean_loss,
        lambda_regime_cvar=0.0,
        lambda_residual_aux_loss=0.0,
        lambda_mid_delta_loss=0.0,
        lambda_zbin_cvar=0.0,
        zbin_cvar_top_frac=0.34,
        lambda_condinv=0.02,
        cold_corrector_limit=spec.cold_corrector_limit,
        cold_corrector_soc_threshold=spec.cold_corrector_soc_threshold,
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
        v_corr_variant="orig_ohm_ema120",
        v_corr_tau_s=120.0,
    )


def collect_summary(manifest_rows: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    baseline_files = {
        "baseline_normal": OUT
        / "ocvstart_lopo4_baseline_model_gru_normal_head_paper_g4_all_ema_holdoutus06_seeds012_b2048_opt123preidx_e200_by_temperature.csv",
        "baseline_residual_unbounded": OUT
        / "ocvstart_lopo4_baseline_model_gru_residual_head_paper_g4_all_ema_holdoutus06_seeds012_b2048_opt123preidx_e200_by_temperature.csv",
    }
    for label, path in baseline_files.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["seed"].eq(0)].copy()
        df = df.drop_duplicates(subset=["seed", "temperature_C"], keep="last")
        for _, r in df.iterrows():
            rows.append(
                {
                    "head_design": label,
                    "seed": int(r["seed"]),
                    "temperature_C": float(r["temperature_C"]),
                    "MAE_pct": float(r["MAE_pct"]),
                    "bias_pct": float(r["bias_pct"]),
                    "source": str(path),
                }
            )
    for item in manifest_rows:
        path = Path(item["by_temperature_path"])
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df.drop_duplicates(subset=["seed", "temperature_C"], keep="last")
        for _, r in df.iterrows():
            rows.append(
                {
                    "head_design": item["head_design"],
                    "seed": int(r["seed"]),
                    "temperature_C": float(r["temperature_C"]),
                    "MAE_pct": float(r["MAE_pct"]),
                    "bias_pct": float(r["bias_pct"]),
                    "source": str(path),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    mean_rows = []
    for (design, seed), g in out.groupby(["head_design", "seed"]):
        mean_rows.append(
            {
                "head_design": design,
                "seed": int(seed),
                "temperature_C": "mean",
                "MAE_pct": float(g["MAE_pct"].mean()),
                "bias_pct": float(g["bias_pct"].mean()),
                "source": "temperature_mean",
            }
        )
    out = pd.concat([out, pd.DataFrame(mean_rows)], ignore_index=True)
    order = out[out["temperature_C"].eq("mean")][["head_design", "MAE_pct"]].sort_values("MAE_pct")
    rank = {name: idx for idx, name in enumerate(order["head_design"].tolist())}
    out["_rank"] = out["head_design"].map(rank).fillna(999).astype(int)
    out["_temp_order"] = out["temperature_C"].map({0.0: 0, 25.0: 1, 45.0: 2, "mean": 3}).fillna(9).astype(int)
    return out.sort_values(["_rank", "head_design", "seed", "_temp_order"]).drop(columns=["_rank", "_temp_order"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []
    t_all = time.time()
    for idx, spec in enumerate(SPECS, start=1):
        cfg = cfg_for_spec(spec)
        summary_path = OUT / f"{cfg.output_prefix}_test_summary.csv"
        by_temp_path = OUT / f"{cfg.output_prefix}_by_temperature.csv"
        t0 = time.time()
        print(
            f"=== [{idx}/{len(SPECS)}] start head={spec.name} model_kind={spec.model_kind} "
            f"L={spec.residual_limit_mode}:{spec.residual_limit} zero_mean={spec.lambda_residual_zeromean_loss} ===",
            flush=True,
        )
        if summary_path.exists() and by_temp_path.exists():
            print(f"=== skip existing head={spec.name} ===", flush=True)
        else:
            run(cfg)
        elapsed = time.time() - t0
        manifest_rows.append(
            {
                "head_design": spec.name,
                "holdout": HOLDOUT,
                "train_profiles": ",".join(cfg.train_profiles),
                "feature_set": cfg.feature_set,
                "v_corr_variant": cfg.v_corr_variant,
                "model_kind": cfg.model_kind,
                "residual_limit": spec.residual_limit,
                "residual_limit_init": spec.residual_limit_init,
                "residual_limit_mode": spec.residual_limit_mode,
                "residual_limit_lower": spec.residual_limit_lower,
                "residual_limit_upper": spec.residual_limit_upper,
                "lambda_residual_zeromean_loss": spec.lambda_residual_zeromean_loss,
                "cold_corrector_limit": spec.cold_corrector_limit,
                "cold_corrector_soc_threshold": spec.cold_corrector_soc_threshold,
                "seed": "0",
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "summary_path": str(summary_path),
                "by_temperature_path": str(by_temp_path),
                "elapsed_s": elapsed,
            }
        )
        done = idx
        avg = (time.time() - t_all) / max(done, 1)
        eta = avg * (len(SPECS) - done)
        print(f"=== done head={spec.name} elapsed_s={elapsed:.1f} eta_s={eta:.1f} ===", flush=True)
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = OUT / "ocvstart_us06_head_design_screen_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    summary = collect_summary(manifest_rows)
    summary_path = OUT / "ocvstart_us06_head_design_screen_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"manifest {manifest_path}", flush=True)
    print(f"summary {summary_path}", flush=True)
    if not summary.empty:
        mean = summary[summary["temperature_C"].eq("mean")].sort_values("MAE_pct")
        print(mean[["head_design", "seed", "MAE_pct", "bias_pct"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
