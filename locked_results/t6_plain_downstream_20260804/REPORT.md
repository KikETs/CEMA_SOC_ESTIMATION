# T6-plain downstream regeneration report

## Frozen-package gate

- 60/60 package entries passed; maximum archived-prediction difference 9.17638588e-08 SOC fraction.
- No T6-plain checkpoint was retrained or reselected.

## Tier-3 robustness

Values after the clean column are macro-MAE changes in percentage points. Sensor faults were injected into raw V/I before R0 correction, EMA, normalization, and windowing.

| chemistry   |   clean_mae_pct |   delta_v_noise_10mV |   delta_i_noise_2pct |   worst_delta_v_bias_10mV |   worst_delta_i_bias_20mA |   delta_r0_scale_2 |
|:------------|----------------:|---------------------:|---------------------:|--------------------------:|--------------------------:|-------------------:|
| NMC         |        0.329158 |            0.0136523 |         -0.000156517 |                   1.23771 |                 0.0778843 |            6.13342 |
| LFP         |        0.649734 |            0.0208586 |          0.00352206  |                   3.31927 |                 0.837293  |           27.4927  |

## Cold-start recovery

Fixed deterministic mid-record reset points:

| chemistry   |   reset_trials |   recovery_rate |   median_time_to_1pct_s |
|:------------|---------------:|----------------:|------------------------:|
| NMC         |            135 |        0.977778 |                 953.435 |
| LFP         |            360 |        0.958333 |                1513.73  |

SOC-band-stratified reset points:

| chemistry   | reset_band   |   n_reset_trials |   recovery_rate |   median_time_to_1pct_s |
|:------------|:-------------|-----------------:|----------------:|------------------------:|
| LFP         | 35to65       |              216 |        0.925926 |               1171.36   |
| LFP         | gt65         |              216 |        0.958333 |               2838.88   |
| LFP         | le35         |              216 |        0.824074 |                486.444  |
| NMC         | 35to65       |               81 |        1        |               1297.62   |
| NMC         | gt65         |               81 |        1        |                 49.5465 |
| NMC         | le35         |               81 |        0.864198 |               1059.26   |

## C1 fair-input comparison (LFP, seeds 0-2)

| feature   |   mean_mae_pct |   sd_mae_pct |   n_seeds |
|:----------|---------------:|-------------:|----------:|
| RAW_W200  |       2.158    |    0.119208  |         3 |
| RAW_W800  |       1.21337  |    0.100583  |         3 |
| SMA9      |       1.1772   |    0.0571345 |         3 |
| T6_plain  |       0.640037 |    0.0172588 |         3 |

| comparison        |   mean_delta_mae_pct |   ci_lo_pct |   ci_hi_pct |   n_matched_seeds | method                        |
|:------------------|---------------------:|------------:|------------:|------------------:|:------------------------------|
| RAW_W200-T6_plain |             1.51796  |    1.26405  |    1.77187  |                 3 | matched-seed Student-t 95% CI |
| RAW_W800-T6_plain |             0.573335 |    0.295598 |    0.851072 |                 3 | matched-seed Student-t 95% CI |
| SMA9-T6_plain     |             0.537164 |    0.419366 |    0.654962 |                 3 | matched-seed Student-t 95% CI |

RAW-W800 used microbatch 1024 with two-step gradient accumulation; effective batch remained 2048.

## C4 NMC-to-LFP zero-shot

| variant             |   condition_equal_MAE_pct |   seed_SD_pct |   n_seeds |   n_conditions |   clamped_sample_fraction |
|:--------------------|--------------------------:|--------------:|----------:|---------------:|--------------------------:|
| measurement_adapted |                   12.8083 |      0.187522 |         3 |             72 |                      0    |
| strict              |                   47.479  |      0.386559 |         3 |             72 |                      0.25 |

## Hysteresis-EKF paired bootstrap

Positive differences mean the EKF has larger MAE than T6-plain.

| method             | reference_method           | initial_condition   | weighting                 |   delta_MAE_method_minus_proposed_pct |   ci_low_pct |   ci_high_pct |     B |   block_samples |     seed |   n_records |   n_proposed_seeds |
|:-------------------|:---------------------------|:--------------------|:--------------------------|--------------------------------------:|-------------:|--------------:|------:|----------------:|---------:|------------:|-------------------:|
| hysteresis_2rc_ekf | T6_plain_GRU_linear_auxoff | minus10pp           | per-seed slice-unweighted |                              6.06338  |     6.02498  |      6.10178  | 10000 |              60 | 20260804 |          24 |                 10 |
| hysteresis_2rc_ekf | T6_plain_GRU_linear_auxoff | minus5pp            | per-seed slice-unweighted |                              0.766348 |     0.745692 |      0.78627  | 10000 |              60 | 20260804 |          24 |                 10 |
| hysteresis_2rc_ekf | T6_plain_GRU_linear_auxoff | oracle              | per-seed slice-unweighted |                             -0.437571 |    -0.45749  |     -0.418698 | 10000 |              60 | 20260804 |          24 |                 10 |

## MCU

| chemistry   |   network_us_median |   network_us_p90 |   flash_total_bytes |   ram_total_bytes |   flash_model_bytes |   ram_arena_bytes |   max_abs_diff_pct |   preprocessing_us_median |   end_to_end_us_median |
|:------------|--------------------:|-----------------:|--------------------:|------------------:|--------------------:|------------------:|-------------------:|--------------------------:|-----------------------:|
| LFP         |              190935 |           191079 |              448820 |             81880 |              403476 |             54784 |        1.78814e-05 |                     92.26 |                 191789 |
| NMC         |              190702 |           190879 |              448820 |             81880 |              403476 |             54784 |        1.49012e-05 |                     92.3  |                 191839 |

All MCU neural-network measurements are FP32, batch-1, one 50x9 window per estimate.
