#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soc_decomp.nmc_vcorr_it_train_dst_selector_run import TrainDSTSelectorConfig, run


PROFILE_SET = ("VALIDATION", "DST", "FUDS", "US06")
HOLDOUTS = ("US06", "DST", "FUDS", "VALIDATION")


def train_profiles_for_holdout(holdout: str) -> tuple[str, ...]:
    return tuple(profile for profile in PROFILE_SET if profile != holdout)


def cfg_for_holdout(holdout: str) -> TrainDSTSelectorConfig:
    prefix = f"endzero_lopo4_lfpstyle_ocvfit_g4_gru_residual_holdout{holdout.lower()}_seed0_b2048_e200"
    return TrainDSTSelectorConfig(
        base_dir=ROOT,
        raw_root=Path("nmc_ocvstart_endzero_lopo_clean"),
        output_prefix=prefix,
        seeds=(0,),
        train_profiles=train_profiles_for_holdout(holdout),
        valid_profiles=("NONE",),
        test_profiles=(holdout,),
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
        v_corr_variant="lfpstyle_ocvfit_v1",
        v_corr_tau_s=10.0,
        lfpstyle_fit_stride=5,
        lfpstyle_fit_max_nfev=250,
        lfpstyle_v_floor_raw=2.45,
        lfpstyle_r0_max_ohm=0.22,
    )


def main() -> None:
    out_dir = ROOT / "nmc_goal_vcorr_it_train_dst_selector_results"
    rows = []
    started = time.time()
    for holdout in HOLDOUTS:
        cfg = cfg_for_holdout(holdout)
        t0 = time.time()
        print(
            f"=== start label=endzero v_corr=lfpstyle_ocvfit_v1 model=GRU_G4_residual "
            f"holdout={holdout} train={','.join(cfg.train_profiles)} seed=0 epochs=200 ===",
            flush=True,
        )
        run(cfg)
        elapsed = time.time() - t0
        rows.append(
            {
                "holdout": holdout,
                "prefix": cfg.output_prefix,
                "summary_path": str(out_dir / f"{cfg.output_prefix}_test_summary.csv"),
                "by_temperature_path": str(out_dir / f"{cfg.output_prefix}_by_temperature.csv"),
                "lfpstyle_params_path": str(out_dir / f"{cfg.output_prefix}_lfpstyle_params.csv"),
                "lfpstyle_train_fit_path": str(out_dir / f"{cfg.output_prefix}_lfpstyle_train_fit.csv"),
                "elapsed_s": elapsed,
            }
        )
        print(f"=== done holdout={holdout} elapsed_s={elapsed:.1f} ===", flush=True)
    manifest = pd.DataFrame(rows)
    manifest_path = out_dir / "endzero_lfpstyle_ocvfit_g4_gru_lopo_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"manifest {manifest_path}", flush=True)
    print(f"total_elapsed_s={time.time() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
