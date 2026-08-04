# Frozen NMC inference package

## Scope

This package freezes the archived 3-LOPO proposed-model inference path without
retraining and without importing any module from the source repository.

- Features: `T6` and `G4`
- Holdouts: `DST`, `FUDS`, and `US06`
- Seeds: `0`, `1`, and `2`
- Architecture: GRU, one layer, hidden size 128, anchor-residual sequence head
- Temperatures: 0, 25, and 45 degrees C
- Entries: 18 under `{feature}/{fold}/{seed}/`

The returned SOC is a fraction in `[0, 1]`, matching archived `y_pred`, not a
percentage.

## Usage

Run with Python packages `numpy`, `pandas`, and `torch` available:

```python
from inference_pkg_nmc import load_package

pkg = load_package("inference_pkg_nmc/G4/US06/0")
soc = pkg.predict(
    df_record,
    reset_indices=None,
    r0_scale=1.0,
    v_noise=None,
    i_noise=None,
    v_bias=0.0,
    i_bias=0.0,
)
```

`df_record` must be one complete record with `Voltage(V)`, `Current(A)`, a
single temperature identifiable from `T`, `temperature_C`, `temperature`, or
`TempLabel`, and preferably `Step_Time(s)` or `Test_Time(s)`.

`v_noise` and `i_noise` accept `None`, a scalar, or a length-N array. Noise and
bias are applied to raw voltage/current before R0 correction, Vcorr EMA, and
all feature EMAs. Random noise generation is intentionally external so its
seed and realization remain explicit.

`reset_indices` are zero-based sample indices. Index 0 is always a reset. At a
reset, Vcorr EMA state, all feature EMA states, and the 50-sample model window
are discarded. The output is `NaN` for the first 49 samples of each segment.

## Frozen assumptions

1. The source artifacts are the `s3` 3-LOPO runs in
   `nmc_goal_vcorr_it_train_dst_selector_results`. No checkpoint was selected
   or modified during packaging; the stored final epoch-200 weights are used.
2. Training stride is 3. Evaluation is record-local stride 1, with prediction
   end indices 49 through N-1 and no additional evaluation mask.
3. Nominal sampling is 1 second. Vcorr uses a 120-second causal time EMA.
   Feature EMAs use 50, 200, and 800 sample constants as listed per channel in
   each `config.json`.
4. R0 is the archived training-temperature event median for each fold. Linear
   interpolation is used only for temperatures not present in the table.
5. Scalers are refit only from the two training profiles using the exact
   archived algorithm. The source pipeline reduced a Fortran-contiguous
   float32 matrix; preserving this numeric layout is required because float32
   reduction order changes the last digits.
6. Each entry contains the exact original weight filename. `code/inference.py`
   is a minimal frozen implementation and contains zero source-repository
   imports.
7. The source repository has an unborn `master` branch and no resolvable Git
   commit. Therefore `source_git_commit` is `null`; manifests report
   `unavailable_unborn_git_branch` and pin every relevant source artifact by
   SHA-256 instead.

## Golden validation

All 18 entries were evaluated on all three archived holdout-temperature
records. Every archived end index was compared, for 30,899 to 32,278 samples
per entry. Two consecutive public `predict()` calls were bit-identical for
every record.

The largest observed absolute difference was below `3e-7`, so the requested
`atol=1e-6` is retained without relaxation. Full entry-level results are in
`golden_test_results.csv`.

Golden validation needs the source evaluation CSVs and archived predictions;
ordinary inference does not. The caller only supplies a record DataFrame.

## Reproduction

From `/home/user/바탕화면/DL/NMC_KF_3LOPO_BASELINES`:

```bash
/home/user/anaconda3/envs/torch_env/bin/python build_inference_pkg_nmc.py
/home/user/anaconda3/envs/torch_env/bin/python validate_inference_pkg_nmc.py
/home/user/anaconda3/envs/torch_env/bin/python -m pytest tests/test_inference_pkg_nmc.py
```

The builder reads the source repository but does not write to it. Rebuilding
overwrites only this package's generated entry files.
