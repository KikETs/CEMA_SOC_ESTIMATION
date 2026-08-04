# T6-plain frozen inference packages

## Scope

The package roots are `inference_pkg_nmc/` and `inference_pkg_lfp/`. Each root
contains T6 packages for holdouts DST, FUDS, and US06 and model seeds 0 through
9. The package weights are copied byte-for-byte from `normalhead_auxoff`; no
checkpoint was retrained or reselected.

Each entry is addressed as `T6/<fold>/<seed>/` and contains:

- `config.json`: exact nine-channel order, GRU and plain-head definition,
  sampling, window, stride, reset, and training-recipe metadata.
- `scaler.json`: training-profile-only feature means and standard deviations.
- `r0_table.json`: fold-specific, training-profile-only temperature/R0 table.
- the original `*_final_epoch200_weights.pt` filename.
- `code/inference.py`: the frozen feature builder, model definition, loader,
  perturbation path, and reset-aware prediction implementation.
- `manifest.json`: source provenance and SHA-256 hashes.

The root `loader.py` is the only required import. The package has no imports
from either locked training repository.

## Exact feature and perturbation order

The channel order is `V_corr_raw`, `I_raw`, `T`, followed by Vcorr EMA state and
deviation pairs for time constants 50, 200, and 800 samples. Raw V/I noise and
bias are applied first. The perturbed current therefore propagates through R0
correction as well as the model current channel. Vcorr and its EMA channels are
then computed, followed by fold scaling and window construction.

`reset_indices` resets the Vcorr/EMA state and the 50-sample window buffer. A
reset therefore produces 49 unavailable outputs before the first new estimate.

## Golden tolerance and execution device

All 60 entries reproduce their archived per-sample predictions within `1e-6`
SOC fraction and are deterministic across repeated calls. The maximum observed
CUDA FP32 difference was `9.176e-8`. TF32 is explicitly disabled.

The archived predictions were generated on CUDA with TF32 disabled. CPU FP32
uses a different GRU kernel and reached a maximum difference of approximately
`1.58e-6`, slightly outside the declared gate. Consequently, the loader uses
CUDA FP32 when CUDA is available and records CPU as a portability fallback, not
as the golden-validation backend.
