# T6-plain inference robustness summary

## Protocol

- Frozen package inference only; no retraining and no package modification.
- Feature: T6-plain; folds: DST, FUDS, US06; model seeds: 0, 1, 2.
- Metrics are percentage points of SOC. SOC bands are <=35%, 35-65%, and >65%.
- Three deterministic Gaussian realizations are pooled per model seed and noise level.
- The same deterministic raw-noise realization is shared across model seeds.
- Cold-start uses five fixed, stratified-random mid-record reset points. Each reset is evaluated in a separate inference run.
- Recovery time is the first post-reset evaluated sample with absolute SOC error <=1 percentage point.
- Figures report the macro mean across fold, model seed, temperature, and SOC-band rows.

## Clean baseline

- T6-plain macro MAE: 0.6497% SOC

## Cold-start recovery

- T6-plain median time to first <=1% SOC error: 1513.7 s; recovery rate 95.8%.

## Outputs

- `sensor_noise.csv`, `sensor_noise_degradation.png`
- `sensor_bias.csv`, `sensor_bias_degradation.png`
- `r0_perturbation.csv`, `r0_degradation.png`
- `cold_start.csv`, `cold_start_recovery.csv`, `cold_start_degradation.png`
- `t6_vs_g4_sensitivity.csv`
