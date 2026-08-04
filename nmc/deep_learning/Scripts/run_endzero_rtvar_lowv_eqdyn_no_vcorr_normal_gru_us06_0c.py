#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soc_decomp.nmc_vcorr_it_train_dst_selector_run import TrainDSTSelectorConfig, run


def main() -> None:
    cfg = TrainDSTSelectorConfig(
        base_dir=ROOT,
        raw_root=Path("nmc_ocvstart_endzero_lopo_clean"),
        output_prefix="endzero_rtvar_lowv_ema120_normal_gru_linear_eqdyn_no_vcorr_holdoutus06_test0c_seed0_b2048_e200",
        seeds=(0,),
        train_profiles=("VALIDATION", "DST", "FUDS"),
        valid_profiles=("NONE",),
        test_profiles=("US06",),
        train_temperatures=(0.0, 25.0, 45.0),
        valid_temperatures=(),
        test_temperatures=(0.0,),
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
        model_kind="single",
        recurrent="gru",
        head_kind="linear",
        dropout=0.06,
        feature_set="paper_g4_eqdyn_no_vcorr",
        stage2_feature_set="paper_g4_eqdyn_no_vcorr",
        temp_mode="none",
        train_sampler="temperature_profile_regime_balanced",
        stage1_selector="last_epoch",
        selector_regime_min_windows=30,
        lambda_anchor_loss=0.0,
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
    started = time.time()
    print(
        "=== start label=endzero v_corr=rtvar_lowv_ema120 model=normal_GRU_linear "
        "feature=paper_g4_eqdyn_no_vcorr holdout=US06 test_temp=0C seed=0 epochs=200 ===",
        flush=True,
    )
    run(cfg)
    print(f"=== done elapsed_s={time.time() - started:.1f} ===", flush=True)


if __name__ == "__main__":
    main()
