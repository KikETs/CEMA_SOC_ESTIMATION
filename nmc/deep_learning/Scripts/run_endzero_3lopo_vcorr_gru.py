#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soc_decomp.nmc_vcorr_it_train_dst_selector_run import TrainDSTSelectorConfig, run  # noqa: E402


PROFILE_SET = ("DST", "FUDS", "US06")
HOLDOUTS = ("US06", "DST", "FUDS")
VARIANTS = {
    "orig_ohm_ema120": "r0_ema120",
    "rtvar_lowv_ema120": "rtvar_lowv",
    "lfpstyle_ocvfit_v1": "lfpstyle",
}
EPOCHS = 200
BATCH_SIZE = 2048
SEEDS = (0,)
RUN_TAG = "fixedcache"


def train_profiles_for_holdout(holdout: str) -> tuple[str, ...]:
    return tuple(profile for profile in PROFILE_SET if profile != holdout)


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def cfg_for(holdout: str, variant: str) -> TrainDSTSelectorConfig:
    slug = VARIANTS[variant]
    prefix = f"endzero_3lopo_three_profile_{RUN_TAG}_{slug}_g4_gru_residual_holdout{holdout.lower()}_seed0_b{BATCH_SIZE}_e{EPOCHS}"
    cfg = TrainDSTSelectorConfig(
        base_dir=ROOT,
        raw_root=Path("nmc_ocvstart_endzero_lopo_clean"),
        output_prefix=prefix,
        seeds=SEEDS,
        train_profiles=train_profiles_for_holdout(holdout),
        valid_profiles=("NONE",),
        test_profiles=(holdout,),
        train_temperatures=(0.0, 25.0, 45.0),
        valid_temperatures=(),
        test_temperatures=(0.0, 25.0, 45.0),
        epochs=EPOCHS,
        stage1_eval_every=EPOCHS,
        selector_min_epoch=EPOCHS,
        selector_max_epoch=0,
        fixed_stage1_epoch=EPOCHS,
        batch_size=BATCH_SIZE,
        num_workers=0,
        prefetch_factor=2,
        hidden_size=128,
        layers=1,
        kernel_size=5,
        model_kind="anchor_residual_sequence",
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
        skip_train_final_eval=True,
        valid_split_mode="profile",
        cache_dataset_cuda=True,
        v_corr_variant=variant,
        v_corr_tau_s=120.0,
    )
    if variant == "lfpstyle_ocvfit_v1":
        cfg.v_corr_tau_s = 10.0
        cfg.lfpstyle_fit_stride = 5
        cfg.lfpstyle_fit_max_nfev = 250
        cfg.lfpstyle_v_floor_raw = 2.45
        cfg.lfpstyle_r0_max_ohm = 0.22
    return cfg


def read_summary(path: Path, holdout: str, variant: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "variant" in df.columns:
        selected = df[df["variant"].astype(str).str.contains("selected", case=False, na=False)]
        if len(selected):
            df = selected
    rows = []
    for _, row in df.drop_duplicates(subset=["seed", "0.0", "25.0", "45.0"]).iterrows():
        vals = [float(row[str(temp)]) for temp in (0.0, 25.0, 45.0)]
        rows.append(
            {
                "label_protocol": "ocvstart_endzero",
                "profile_protocol": "3lopo_three_profile",
                "v_corr_variant": variant,
                "v_corr_slug": VARIANTS[variant],
                "feature_set": "paper_g4_all_ema",
                "model": "gru",
                "head": "residual",
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


def write_summaries(out_dir: Path, manifest: pd.DataFrame) -> None:
    frames = []
    for _, row in manifest.iterrows():
        summary_path = out_dir / str(row["summary_path"])
        if summary_path.exists():
            frames.append(read_summary(summary_path, str(row["holdout"]), str(row["v_corr_variant"])))
    tag = f"endzero_3lopo_three_profile_{RUN_TAG}_vcorr_g4_gru_residual"
    seed_path = out_dir / f"{tag}_seed_summary.csv"
    agg_path = out_dir / f"{tag}_agg_summary.csv"
    if not frames:
        pd.DataFrame().to_csv(seed_path, index=False)
        pd.DataFrame().to_csv(agg_path, index=False)
        return
    seed_df = pd.concat(frames, ignore_index=True).sort_values(["v_corr_slug", "holdout", "seed"])
    seed_df.to_csv(seed_path, index=False)
    group_cols = ["label_protocol", "profile_protocol", "v_corr_variant", "v_corr_slug", "feature_set", "model", "head"]
    agg = (
        seed_df.groupby(group_cols, as_index=False)
        .agg(
            n_holdouts=("holdout", "nunique"),
            n_seeds=("seed", "nunique"),
            mae_0C_mean=("mae_0C", "mean"),
            mae_25C_mean=("mae_25C", "mean"),
            mae_45C_mean=("mae_45C", "mean"),
            mae_mean_temp=("mae_mean_temp", "mean"),
            mae_mean_temp_max_holdout=("mae_mean_temp", "max"),
        )
        .sort_values(["mae_mean_temp", "mae_0C_mean"])
    )
    agg.to_csv(agg_path, index=False)
    print(f"seed_summary={seed_path}", flush=True)
    print(f"agg_summary={agg_path}", flush=True)
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"), flush=True)


def main() -> None:
    out_dir = ROOT / "nmc_goal_vcorr_it_train_dst_selector_results"
    jobs = [(variant, holdout) for variant in VARIANTS for holdout in HOLDOUTS]
    manifest_rows = []
    started = time.time()
    durations = []
    for idx, (variant, holdout) in enumerate(jobs, start=1):
        cfg = cfg_for(holdout, variant)
        summary_path = out_dir / f"{cfg.output_prefix}_test_summary.csv"
        if summary_path.exists():
            elapsed = 0.0
            print(f"=== skip existing [{idx}/{len(jobs)}] variant={variant} holdout={holdout} summary={summary_path.name} ===", flush=True)
        else:
            t0 = time.time()
            print(
                f"=== start [{idx}/{len(jobs)}] profile_protocol=3lopo_three_profile "
                f"v_corr={variant} model=GRU_G4_residual holdout={holdout} "
                f"train={','.join(cfg.train_profiles)} seed=0 epochs={EPOCHS} ===",
                flush=True,
            )
            run(cfg)
            elapsed = time.time() - t0
            durations.append(elapsed)
            remaining = len(jobs) - idx
            mean_job = sum(durations) / max(len(durations), 1)
            print(
                f"=== done [{idx}/{len(jobs)}] variant={variant} holdout={holdout} "
                f"elapsed_s={elapsed:.1f} eta={format_eta(mean_job * remaining)} ===",
                flush=True,
            )
        manifest_rows.append(
            {
                "job_index": idx,
                "v_corr_variant": variant,
                "v_corr_slug": VARIANTS[variant],
                "holdout": holdout,
                "train_profiles": "+".join(train_profiles_for_holdout(holdout)),
                "prefix": cfg.output_prefix,
                "summary_path": f"{cfg.output_prefix}_test_summary.csv",
                "by_temperature_path": f"{cfg.output_prefix}_by_temperature.csv",
                "elapsed_s": elapsed,
            }
        )
        manifest = pd.DataFrame(manifest_rows)
        manifest_path = out_dir / f"endzero_3lopo_three_profile_{RUN_TAG}_vcorr_g4_gru_residual_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        write_summaries(out_dir, manifest)
    print(f"manifest={out_dir / f'endzero_3lopo_three_profile_{RUN_TAG}_vcorr_g4_gru_residual_manifest.csv'}", flush=True)
    print(f"total_elapsed_s={time.time() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
