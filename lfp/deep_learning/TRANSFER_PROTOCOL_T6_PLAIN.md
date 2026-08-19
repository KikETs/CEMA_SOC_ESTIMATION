# LFP T6-plain Transfer Protocol

## Current configuration

This is the current proposed-model protocol for the LFP 3-LOPO evaluation.

- Feature set: `paper_t6_voltage_ema_all` (T6), with nine channels in this exact order:
  1. `V_corr_raw`
  2. `I_raw`
  3. `T`
  4. `V_corr_raw_ema50`
  5. `V_corr_raw_dev_ema50`
  6. `V_corr_raw_ema200`
  7. `V_corr_raw_dev_ema200`
  8. `V_corr_raw_ema800`
  9. `V_corr_raw_dev_ema800`
- Network: linear input embedding, LayerNorm, SiLU, one-layer GRU with hidden size 128, output LayerNorm, and a plain linear output followed by sigmoid.
- Head: `plain_linear`; no anchor or residual output branch.
- Objective: Huber loss only (`beta=0.02`). The auxiliary weights are fixed to zero: `lambda_rex=0`, `lambda_condinv=0`, and `lambda_anchor_loss=0`.
- Seeds: `0` through `9`.
- Selection: 200 epochs and final-epoch weights; no validation- or test-based checkpoint selection.
- Windowing: 50 samples, training stride 3, evaluation stride 1.
- LOPO folds: held-out profile in `{DST, FUDS, US06}`; the other two profiles are the training profiles.

## Fold-specific fitting

The following assets are refit independently for each LOPO fold using only that fold's two training profiles:

- `R0(T)`, stored in each package entry as `r0_table.json`;
- per-channel normalization mean and standard deviation, stored as `scaler.json`.

The held-out profile is not used to fit either asset.

## Frozen package and integrity pointers

The frozen T6-plain package is archived at:

- [`locked_results/t6_plain_downstream_20260804/inference_pkg_lfp/`](../../locked_results/t6_plain_downstream_20260804/inference_pkg_lfp/)

It contains all three folds and seeds 0-9. Each `T6/<fold>/<seed>/` entry includes its exact configuration, fold scaler, fold `R0(T)` table, final-epoch weights, frozen inference code, and entry manifest.

SHA-256 verification pointers:

- Package-local index: [`inference_pkg_lfp/sha256sums.txt`](../../locked_results/t6_plain_downstream_20260804/inference_pkg_lfp/sha256sums.txt)
- Repository lock index: [`locks/t6_plain_locked_results_sha256.txt`](../../locks/t6_plain_locked_results_sha256.txt)

The historical pre-pivot protocol remains available in [`TRANSFER_PROTOCOL.md`](TRANSFER_PROTOCOL.md) for audit provenance; it does not define the current proposed configuration.
