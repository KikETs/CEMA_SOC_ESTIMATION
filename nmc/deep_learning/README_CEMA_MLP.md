# CEMA_MLP workspace

This folder is a clean MLP/CEMA experiment workspace copied out of the older
`CEMA_LOPO_LSTM_anchorL_isolated_3090` tree.

## Included

- `soc_decomp/`: model, feature, baseline, and train runner code.
- `Scripts/`: helper scripts, including the no-floor SOC relabel script.
- `nmc_soc80_train_nofloor_qmax/`: SOC80 labels recomputed without long 0-floor tails.
- `run_soc80_feature_ablation_baselines_all_holdouts_e200.sh`: 88-job feature/baseline launcher.
- `run_soc80_full_feature_grid_missing_nonlstm_e200.sh`: remaining backbone-feature grid launcher.

## Current Label Formula

`SOC_CC = clip(0.8 * (1 - Qnet_removed(Ah) / max(Qnet_removed(Ah))), 0, 1)`

The previous floor-clipped labels are preserved in each CSV as
`SOC_CC_floorclip_prev` and `SOC_CC_floorclip_prev(%)`.

## Run

From this folder:

```bash
CEMA_GPU=0 CEMA_SEEDS=0,1,2 ./run_soc80_feature_ablation_baselines_all_holdouts_e200.sh
```

The scripts write outputs under `nmc_goal_vcorr_it_train_dst_selector_results/`.
