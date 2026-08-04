# LFP T6-plain core4 temperature interpolation

## Protocol

- Network-training temperatures: -10, 0, 25, and 50 degC.
- Test temperatures: -10, 0, 10, 20, 25, 30, 40, and 50 degC.
- Three-profile LOPO: DST, FUDS, and US06; seeds 0, 1, and 2.
- Model: GRU, plain linear head, all auxiliary-loss coefficients set to zero.
- Frozen recipe: AdamW 8e-4, weight decay 2e-4, Huber beta 0.02, batch 2048,
  200 epochs, final-epoch selection, window 50, and train/test strides 3/1.
- Fold-specific R0(T) was fitted at every test temperature from the two training
  profiles, matching the prior core4 convention. Normalization used only the
  admitted core4 network-training temperatures.
- The all8 condition is a reference reprint from the locked `normalhead_auxoff`
  runs. It was not retrained.

The all8 T6-plain golden re-evaluation passed 72/72 fold x seed x temperature
slices. The largest prediction difference from the locked per-sample archive was
9.176e-08 in SOC fraction, below the required 1e-6 tolerance.

## Aggregate results

Values are the unweighted mean over 3 folds x 3 seeds x 8 temperatures.

| Feature | all8 MAE (%SOC) | core4 MAE (%SOC) | core4 - all8 | Relative change |
|---|---:|---:|---:|---:|
| G0 | 2.421706 | 2.460614 | +0.038908 | +1.61% |
| T6 | 0.640037 | 0.781143 | +0.141105 | +22.05% |
| T7 | 0.906284 | 1.151310 | +0.245026 | +27.04% |
| G4 | 0.726453 | 0.838137 | +0.111684 | +15.37% |

## T6 temperature results

Values are averaged over folds and seeds.

| Temperature (degC) | all8 MAE | core4 MAE | core4 - all8 |
|---:|---:|---:|---:|
| -10 | 1.519479 | 1.595526 | +0.076047 |
| 0 | 0.763784 | 0.827424 | +0.063640 |
| 10 | 0.580019 | 0.909394 | +0.329375 |
| 20 | 0.433291 | 0.787450 | +0.354159 |
| 25 | 0.396988 | 0.432980 | +0.035992 |
| 30 | 0.418954 | 0.579992 | +0.161038 |
| 40 | 0.465333 | 0.549261 | +0.083928 |
| 50 | 0.542450 | 0.567114 | +0.024664 |

## Pre-declared questions

1. **Seen/unseen behavior.** T6 core4 seen MAE is 0.855761 and unseen MAE is
   0.706524, giving an unseen/seen ratio of 0.825609. The corresponding all8
   values are 0.805675, 0.474399, and 0.588822. Absolute unseen error remains
   lower because the seen set contains the difficult -10 degC condition.
   However, the matched core4 penalty is +0.050086 at seen temperatures and
   +0.232125 at unseen temperatures. Therefore, for T6-plain the result does
   not support a pure training-set-reduction explanation; a distinct
   interpolation penalty is present.
2. **Gap localization.** The mean matched T6 penalty is +0.341767 in the
   0-to-25 degC gap (10 and 20 degC) and +0.122483 in the 25-to-50 degC gap
   (30 and 40 degC). The interpolation penalty is concentrated primarily in
   the lower temperature gap.
3. **T7 behavior.** T7 has the largest overall degradation (+0.245026,
   +27.04%) and the largest unseen penalty (+0.318679). It is not the only
   carrier that degrades at unseen temperatures: T6 and G4 have unseen
   penalties of +0.232125 and +0.175886, respectively. T7 core4 unseen MAE
   (1.161387) is only slightly above its seen MAE (1.141233), so these runs do
   not show a unique catastrophic unseen-temperature divergence.

## Artifact integrity

- Aggregate rows: 576, with no duplicate condition/feature/fold/seed/temperature keys.
- Per-sample prediction archives: 72, comprising 36 locked all8 references and
  36 newly trained core4 runs.
- Newly trained final-epoch weights: 36.
- All 12 feature x fold jobs completed, and the locked source/result hashes were
  unchanged after execution.

Primary machine-readable outputs are `seed_holdout_temperature_mae.csv`,
`seen_unseen_summary.csv`, `seen_unseen_ratio.csv`,
`matched_core4_penalty_rows.csv`, `temperature_gap_penalty.csv`, and
`prediction_rows_manifest.csv`.
