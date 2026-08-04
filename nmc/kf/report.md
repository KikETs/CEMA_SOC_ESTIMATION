# NMC 3-LOPO ECM-KF Baseline Report

## Scope

This is a direct comparison on the same local NMC data and the same 3-LOPO evaluation endpoints. It is not a numerical comparison against MAE values reported by unrelated literature datasets.

## Information and prior inputs

| Method | Online inputs | Prior information | Initial SOC |
|---|---|---|---|
| Proposed EMA-MLP | 50-s causal Vcorr/I/T window | trained weights, training-only scaler | none supplied |
| CC | current, dt | independent temperature capacity | oracle or explicit perturbation |
| ECM-KF | voltage, current, nominal temperature, dt | independent OCV/capacity plus training-profile ECM and Q/R | oracle or explicit perturbation |

The KF baselines use more explicit prior information than the proposed model. OCV curves, ECM parameters, usable capacity, and initial SOC are not hidden from this comparison.
At the measured 0/25/45 C points, exact per-temperature ECM maps are used, so no interpolation is invoked in the reported runs. The fixed unseen-temperature rule is linear resistance and log-time-constant interpolation with endpoint clamping.

## Reference-label caveat

The reference SOC is itself OCV-start coulomb counting with the same temperature-specific low-current capacity. Oracle CC therefore reproduces the label construction and must not be interpreted as an independently validated physical ground-truth estimator.

## Fold summary

| method                       |   fold_mean_mae_pct |   fold_std_mae_pct |   worst_fold_mae_pct | worst_fold   |   mean_rmse_pct |   mean_p95_ae_pct |
|:-----------------------------|--------------------:|-------------------:|---------------------:|:-------------|----------------:|------------------:|
| CC                           |              0.1022 |             0.035  |               0.1425 | DST          |          0.1346 |            0.2946 |
| GRU_T6_residual              |              0.2969 |             0.0368 |               0.3393 | FUDS         |          0.4069 |            0.8332 |
| RNN_T6_residual              |              0.2983 |             0.0245 |               0.326  | FUDS         |          0.4101 |            0.8238 |
| LSTM_T6_residual             |              0.3006 |             0.0242 |               0.3286 | FUDS         |          0.4254 |            0.8573 |
| Transformer_T6_residual      |              0.3207 |             0.0268 |               0.344  | FUDS         |          0.439  |            0.8932 |
| EMA_MLP_T6_residual          |              0.3218 |             0.0257 |               0.3515 | FUDS         |          0.4435 |            0.9163 |
| CEMA_TCN_G4_residual         |              0.3227 |             0.0306 |               0.3503 | FUDS         |          0.4435 |            0.9192 |
| Proposed_EMA_MLP_G4_residual |              0.3365 |             0.0431 |               0.3835 | FUDS         |          0.4605 |            0.9388 |
| Adaptive_2RC_EKF             |              1.6954 |             0.5452 |               2.2697 | FUDS         |          2.0478 |            3.5087 |
| 2RC_EKF                      |              1.7192 |             0.5249 |               2.2913 | FUDS         |          2.0833 |            3.5313 |
| 1RC_EKF                      |              1.9635 |             0.0864 |               2.0328 | DST          |          2.4066 |            4.8191 |
| 2RC_UKF                      |              5.9406 |             7.0742 |              14.1    | FUDS         |          9.5045 |           21.6148 |

## Result-based comparison

- Lowest oracle fold-mean MAE: CC at 0.1022%. This is oracle CC and is privileged by the reference-label construction.
- Best actual Kalman-family method: Adaptive_2RC_EKF at 1.6954% fold-mean MAE; proposed EMA-MLP is 0.3365%.
- Best KF under nonzero initial-SOC perturbations by mean MAE: 1RC_EKF at 2.0254%.
- Lowest measured PC Python step latency: CC at 1.783 us. This is not an MCU latency claim.
- Initialization: 1RC-EKF has the lowest average MAE across the nonzero perturbation tests, while perturbed CC cannot self-correct because it has no voltage update.
- Prior cost: CC requires characterized capacity and initial SOC; ECM-KF additionally requires OCV, fitted RC parameters, and Q/R selection; neural methods require labeled multi-profile training and a training-only scaler but receive no explicit initial SOC.

## Profile x temperature MAE

| method                       | fold   |   temperature_C |   mae_pct |   rmse_pct |   p95_ae_pct |   max_ae_pct |
|:-----------------------------|:-------|----------------:|----------:|-----------:|-------------:|-------------:|
| 1RC_EKF                      | DST    |               0 |    2.0148 |     2.3939 |       4.4233 |       5.5636 |
| 1RC_EKF                      | DST    |              25 |    1.809  |     2.1897 |       2.931  |       9.6908 |
| 1RC_EKF                      | DST    |              45 |    2.2582 |     2.7872 |       5.3607 |       5.7826 |
| 1RC_EKF                      | FUDS   |               0 |    2.0626 |     2.3666 |       3.9519 |       4.5678 |
| 1RC_EKF                      | FUDS   |              25 |    1.6454 |     1.8419 |       2.576  |       9.2094 |
| 1RC_EKF                      | FUDS   |              45 |    2.2611 |     2.7801 |       5.1911 |       5.9764 |
| 1RC_EKF                      | US06   |               0 |    2.0582 |     2.5003 |       5.4233 |       8.8142 |
| 1RC_EKF                      | US06   |              25 |    1.4592 |     1.9795 |       2.8794 |       8.1844 |
| 1RC_EKF                      | US06   |              45 |    2.0998 |     2.6003 |       5.0332 |       5.5254 |
| 2RC_EKF                      | DST    |               0 |    1.9022 |     2.2718 |       4.3987 |       4.6036 |
| 2RC_EKF                      | DST    |              25 |    1.709  |     2.124  |       2.8269 |       9.5058 |
| 2RC_EKF                      | DST    |              45 |    1.2609 |     1.4108 |       2.4701 |       2.5625 |
| 2RC_EKF                      | FUDS   |               0 |    2.1933 |     2.6113 |       4.3773 |       9.5293 |
| 2RC_EKF                      | FUDS   |              25 |    1.5392 |     1.7526 |       2.671  |       9.0018 |
| 2RC_EKF                      | FUDS   |              45 |    3.0905 |     3.1469 |       3.538  |       3.6632 |
| 2RC_EKF                      | US06   |               0 |    1.8658 |     2.2563 |       4.3836 |       8.2176 |
| 2RC_EKF                      | US06   |              25 |    0.8235 |     1.5526 |       3.6711 |       7.6067 |
| 2RC_EKF                      | US06   |              45 |    1.1605 |     1.2905 |       2.217  |       2.2658 |
| 2RC_UKF                      | DST    |               0 |    1.9947 |     2.5192 |       5.1769 |       6.3274 |
| 2RC_UKF                      | DST    |              25 |    2.1323 |     2.4343 |       3.558  |       9.4967 |
| 2RC_UKF                      | DST    |              45 |    2.4251 |     3.0957 |       6.3775 |       6.9337 |
| 2RC_UKF                      | FUDS   |               0 |    2.586  |     3.0473 |       5.3924 |       6.3532 |
| 2RC_UKF                      | FUDS   |              25 |    2.1169 |     2.4043 |       3.904  |       9.2769 |
| 2RC_UKF                      | FUDS   |              45 |   35.1359 |    39.2312 |      56.2861 |      56.7363 |
| 2RC_UKF                      | US06   |               0 |    1.7763 |     2.0458 |       3.5722 |       6.628  |
| 2RC_UKF                      | US06   |              25 |    0.4674 |     1.0275 |       1.5685 |       6.8776 |
| 2RC_UKF                      | US06   |              45 |    2.3465 |     3.0266 |       6.2367 |       6.9076 |
| Adaptive_2RC_EKF             | DST    |               0 |    1.9815 |     2.3419 |       4.4481 |       4.6127 |
| Adaptive_2RC_EKF             | DST    |              25 |    1.7127 |     2.1299 |       2.8364 |       9.6908 |
| Adaptive_2RC_EKF             | DST    |              45 |    1.2605 |     1.4094 |       2.4648 |       2.5518 |
| Adaptive_2RC_EKF             | FUDS   |               0 |    2.0799 |     2.4176 |       4.0714 |       6.3125 |
| Adaptive_2RC_EKF             | FUDS   |              25 |    1.5567 |     1.7694 |       2.6539 |       8.9809 |
| Adaptive_2RC_EKF             | FUDS   |              45 |    3.1082 |     3.1644 |       3.5753 |       3.716  |
| Adaptive_2RC_EKF             | US06   |               0 |    1.9611 |     2.4404 |       5.6444 |       6.7002 |
| Adaptive_2RC_EKF             | US06   |              25 |    0.5228 |     0.8893 |       1.1834 |       6.5528 |
| Adaptive_2RC_EKF             | US06   |              45 |    1.1588 |     1.2885 |       2.2142 |       2.2642 |
| CC                           | DST    |               0 |    0.1039 |     0.129  |       0.2176 |       0.2469 |
| CC                           | DST    |              25 |    0.0579 |     0.0696 |       0.1128 |       0.1404 |
| CC                           | DST    |              45 |    0.2545 |     0.2966 |       0.4861 |       0.5128 |
| CC                           | FUDS   |               0 |    0.0257 |     0.0317 |       0.0596 |       0.0956 |
| CC                           | FUDS   |              25 |    0.0906 |     0.1028 |       0.1668 |       0.2127 |
| CC                           | FUDS   |              45 |    0.1144 |     0.1206 |       0.1681 |       0.2035 |
| CC                           | US06   |               0 |    0.0368 |     0.0466 |       0.0904 |       0.1338 |
| CC                           | US06   |              25 |    0.1644 |     0.1785 |       0.2907 |       0.3226 |
| CC                           | US06   |              45 |    0.0468 |     0.0547 |       0.0935 |       0.1205 |
| Proposed_EMA_MLP_G4_residual | DST    |               0 |    0.3696 |     0.4581 |       0.8744 |       1.9881 |
| Proposed_EMA_MLP_G4_residual | DST    |              25 |    0.2908 |     0.3569 |       0.6475 |       1.5193 |
| Proposed_EMA_MLP_G4_residual | DST    |              45 |    0.247  |     0.3313 |       0.6515 |       1.415  |
| Proposed_EMA_MLP_G4_residual | FUDS   |               0 |    0.5174 |     0.6593 |       1.3352 |       1.9547 |
| Proposed_EMA_MLP_G4_residual | FUDS   |              25 |    0.3061 |     0.3949 |       0.8018 |       1.529  |
| Proposed_EMA_MLP_G4_residual | FUDS   |              45 |    0.3456 |     0.4687 |       0.9627 |       2.1798 |
| Proposed_EMA_MLP_G4_residual | US06   |               0 |    0.4728 |     0.6054 |       1.3291 |       1.5995 |
| Proposed_EMA_MLP_G4_residual | US06   |              25 |    0.3872 |     0.5708 |       0.9136 |       3.1244 |
| Proposed_EMA_MLP_G4_residual | US06   |              45 |    0.1411 |     0.1856 |       0.3996 |       0.9463 |

## Initialization robustness

| method           |   initial_perturbation_pct_point |   mae_pct | time_to_ae_lt_2pct_s   | time_to_sustained_ae_lt_2pct_60s_s   |
|:-----------------|---------------------------------:|----------:|:-----------------------|:-------------------------------------|
| 1RC_EKF          |                              -20 |    2.073  | 301.6809               | 326.5053                             |
| 1RC_EKF          |                              -10 |    1.986  | 38.0412                | 39.0499                              |
| 1RC_EKF          |                               -5 |    1.9766 | 24.8070                | 24.8070                              |
| 1RC_EKF          |                                0 |    1.9631 | 3.0295                 | 5.3888                               |
| 1RC_EKF          |                                5 |    1.9524 | 0.0000                 | 0.0000                               |
| 1RC_EKF          |                               10 |    1.9618 | 24.9233                | 25.7133                              |
| 1RC_EKF          |                               20 |    2.2029 | 440.5423               | 441.5510                             |
| 2RC_EKF          |                              -20 |    2.7209 | 823.1695               | 841.9857                             |
| 2RC_EKF          |                              -10 |    1.9685 | 310.2839               | 311.1687                             |
| 2RC_EKF          |                               -5 |    1.8913 | 98.9892                | 100.1220                             |
| 2RC_EKF          |                                0 |    1.7272 | 0.0000                 | 0.0000                               |
| 2RC_EKF          |                                5 |    1.7816 | 19.4426                | 1277.4007                            |
| 2RC_EKF          |                               10 |    2.2507 | 201.1269               | 202.3866                             |
| 2RC_EKF          |                               20 |    3.8453 | 2064.8317              | 2064.8317                            |
| 2RC_UKF          |                              -20 |    2.7054 | 536.2025               | 536.8761                             |
| 2RC_UKF          |                              -10 |    2.5365 | 183.7781               | 203.5400                             |
| 2RC_UKF          |                               -5 |    3.022  | 37.2965                | 129.9382                             |
| 2RC_UKF          |                                0 |    5.6646 | 0.0000                 | 0.0000                               |
| 2RC_UKF          |                                5 |    4.6338 | 2.0191                 | 8.9930                               |
| 2RC_UKF          |                               10 |    4.8495 | 37.2457                | 54.7904                              |
| 2RC_UKF          |                               20 |    2.9414 | 0.0000                 | 23.6057                              |
| Adaptive_2RC_EKF |                              -20 |    2.8737 | 719.3737               | 735.6724                             |
| Adaptive_2RC_EKF |                              -10 |    2.0341 | 281.9815               | 283.4971                             |
| Adaptive_2RC_EKF |                               -5 |    1.7543 | 359.0926               | 381.5491                             |
| Adaptive_2RC_EKF |                                0 |    1.7047 | 0.0000                 | 0.0000                               |
| Adaptive_2RC_EKF |                                5 |    1.8995 | 19.4426                | 1272.3504                            |
| Adaptive_2RC_EKF |                               10 |    2.0101 | 211.1053               | 212.8729                             |
| Adaptive_2RC_EKF |                               20 |    3.8484 | 2072.6413              | 2075.4089                            |
| CC               |                              -20 |   18.848  | NA                     | NA                                   |
| CC               |                              -10 |    9.9096 | NA                     | NA                                   |
| CC               |                               -5 |    5.0291 | NA                     | NA                                   |
| CC               |                                0 |    0.0995 | 0.0000                 | 0.0000                               |
| CC               |                                5 |    4.9545 | NA                     | NA                                   |
| CC               |                               10 |    9.9545 | NA                     | NA                                   |
| CC               |                               20 |   15.1738 | NA                     | NA                                   |

## Complexity

| method                       |   mean_step_latency_us_pc_python |   p95_step_latency_us_pc_python |   stored_state_scalars |   peak_working_ram_bytes_estimate |   ocv_lookup_bytes_FP64 |   scalar_operation_count_per_step_estimate |
|:-----------------------------|---------------------------------:|--------------------------------:|-----------------------:|----------------------------------:|------------------------:|-------------------------------------------:|
| CC                           |                           1.7827 |                           1.814 |                      1 |                              9712 |                    9696 |                                          8 |
| 1RC_EKF                      |                          47.0234 |                          48.602 |                      2 |                              9768 |                    9696 |                                        110 |
| 2RC_EKF                      |                          48.8376 |                          50.314 |                      3 |                              9832 |                    9696 |                                        260 |
| 2RC_UKF                      |                         152.44   |                         155.694 |                      3 |                             10000 |                    9696 |                                        720 |
| Adaptive_2RC_EKF             |                          50.594  |                          52.238 |                      4 |                              9840 |                    9696 |                                        275 |
| Proposed_EMA_MLP_G4_residual |                          69.569  |                          71.847 |                    850 |                            179544 |                       0 |                                     901888 |

PC Python latency is not MCU latency. Operation counts are implementation-independent rough scalar estimates intended only for later Cortex-M budgeting.

## Paired statistics

| comparison                                                         | unit             |   mean_difference_pct_point |   ci95_low_pct_point |   ci95_high_pct_point |   bootstrap_replicates |   bootstrap_seed | bootstrap_method                                  |   bootstrap_block_length_samples | positive_means_kf_worse   |
|:-------------------------------------------------------------------|:-----------------|----------------------------:|---------------------:|----------------------:|-----------------------:|-----------------:|:--------------------------------------------------|---------------------------------:|:--------------------------|
| Adaptive_2RC_EKF minus Proposed_EMA_MLP_G4_residual absolute error | percentage point |                      1.3665 |               1.3238 |                1.4104 |                   5000 |      2.02607e+07 | fold_temperature_stratified_circular_moving_block |                               60 | True                      |

## Stability and failures

Adaptive 2RC-EKF stable across all oracle fold-temperature runs: True.
Recorded failures/divergences: 0.
2RC-UKF remained finite but was practically unstable on FUDS at 45 C (MAE 35.1359%, bias 35.1302%, max AE 56.7363%). The run is retained; these results do not establish a unique root cause.

## Interpretation boundaries

Accuracy, initialization robustness, online computation, and prior characterization cost are separate axes. The report does not treat the method with the lowest oracle MAE as universally superior.

Total Python experiment runtime: 606.92 s.

<!-- DEPLOYMENT_ASSETS_START -->
## Deployment assets & information asymmetry

The deployment comparison below uses one fold-specific model with exact 0/25/45 C maps. KF assets are FP64, while proposed weights, scaler statistics, and R0_hat are FP32. Stored-asset and runtime-memory values are logical payload sizes, not filesystem serialization sizes.

| method                   | runtime_signals   |   stored_assets_size_KB | offline_prerequisites                                                                                            | needs_initial_SOC   | initial_soc_consequence                                                                                            |   runtime_state_dim |   analytic_flops_per_step | per_step_cost                                                     |   median_latency_us |   runtime_memory_KB |
|:-------------------------|:------------------|------------------------:|:-----------------------------------------------------------------------------------------------------------------|:--------------------|:-------------------------------------------------------------------------------------------------------------------|--------------------:|--------------------------:|:------------------------------------------------------------------|--------------------:|--------------------:|
| CC-oracle                | I,T               |                   0.023 | independent low-current capacity characterization; reference SOC0 oracle (evaluation only, non-deployable)       | yes                 | exact reference SOC0 supplied; optimistic tracking-only condition                                                  |                   1 |                         8 | 8 FLOPs lower-bound + 1.693 us median over 100,000 steps          |               1.693 |               0.031 |
| CC-realistic(+/-5% init) | I,T               |                   0.023 | independent low-current capacity characterization plus an external practical SOC initializer                     | yes                 | +/-5 pp error persists; 0/18 reached sustained AE<2%; mean MAE=4.992%                                              |                   1 |                         8 | 8 FLOPs lower-bound + 1.693 us median over 100,000 steps          |               1.693 |               0.031 |
| 1RC-EKF                  | V,I,T             |                   9.648 | independent OCV/capacity test; training-profile-only 1RC fit and voltage-innovation Q/R selection                | yes                 | first sustained AE<2% for 60 s: median=0.0 s from shared eval start, finite=18/18, worst finite=223.3 s            |                   2 |                       110 | 110 FLOPs lower-bound + 46.137 us median over 100,000 steps       |              46.137 |               9.695 |
| 2RC-EKF                  | V,I,T             |                   9.703 | independent OCV/capacity test; training-profile-only 2RC fit and voltage-innovation Q/R selection                | yes                 | first sustained AE<2% for 60 s: median=0.0 s from shared eval start, finite=17/18, worst finite=11496.6 s, never=1 |                   3 |                       260 | 260 FLOPs lower-bound + 48.481 us median over 100,000 steps       |              48.481 |               9.797 |
| adaptive 2RC-EKF         | V,I,T             |                   9.742 | 2RC prerequisites plus training-profile-only adaptive-R beta selection                                           | yes                 | first sustained AE<2% for 60 s: median=0.0 s from shared eval start, finite=18/18, worst finite=11451.2 s          |                   4 |                       275 | 275 FLOPs lower-bound + 50.375 us median over 100,000 steps       |              50.375 |               9.844 |
| 2RC-UKF                  | V,I,T             |                   9.727 | independent OCV/capacity test; training-profile-only 2RC fit and UKF Q/R selection                               | yes                 | first sustained AE<2% for 60 s: median=0.0 s from shared eval start, finite=18/18, worst finite=890.5 s            |                   3 |                       720 | 720 FLOPs lower-bound + 151.545 us median over 100,000 steps      |             151.545 |               9.984 |
| proposed (EMA+NN)        | V,I,T             |                 172.16  | training-profile R0_hat fit; training-only normalization; labeled SOC training (labels are OCV-start CC-derived) | no                  | not needed; no SOC0 is supplied to the model                                                                       |                 850 |                   1803840 | 1,803,840 FLOPs lower-bound + 68.409 us median over 100,000 steps |              68.409 |             175.48  |

Latency is the median of at least 100,000 CPU single-thread steps. KF/CC timings use the same per-step kernels as `results/complexity.csv` on repeated US06 25 C input. Proposed timing intentionally matches that file's batch-1 precomputed-window NN harness; causal feature-generation latency is excluded, although its EMA arithmetic is included in the analytic lower-bound FLOP column.

The FLOP counts are lower bounds. OCV interpolation, exponentials, Cholesky/eigendecomposition, activation functions, clipping, and memory movement are listed as special operations and are represented by measured latency rather than forced into an architecture-dependent FLOP equivalence.

Information is asymmetric: CC-oracle receives the reference initial SOC and shares the reference-label integration structure; realistic CC cannot correct a +/-5 pp initialization error. KF variants require explicit OCV/capacity/ECM/Q-R characterization and an initial SOC estimate. Proposed requires labeled training trajectories, training-only normalization, per-temperature R0_hat, and substantially larger weights, but no runtime initial SOC or OCV table.
<!-- DEPLOYMENT_ASSETS_END -->

<!-- BOOTSTRAP_SENSITIVITY_START -->
## Paired block-bootstrap sensitivity

Paired differences are defined as `KF AE - proposed 3-seed mean AE` in percentage points; positive values favor proposed. Before resampling, every proposed seed and KF file was required to match exactly by record id and endpoint and within 1e-9 s by timestamp.

Circular moving blocks were sampled independently within each record only, using block lengths 60/300/900, B=10,000, and deterministic seed 20260711. Blocks never cross temperature, profile, or fold record boundaries.

Every pair/scope CI excludes zero at all three block lengths: **True**.

| pair                         | scope     | all_60_300_900_exclude_0   |   lowest_ci_lo |   highest_ci_hi |   acf_lag01_median |
|:-----------------------------|:----------|:---------------------------|---------------:|----------------:|-------------------:|
| proposed_vs_1RC-EKF          | fold:DST  | True                       |         1.3646 |          2.13   |               1579 |
| proposed_vs_1RC-EKF          | fold:FUDS | True                       |         1.2618 |          1.9682 |               1539 |
| proposed_vs_1RC-EKF          | fold:US06 | True                       |         1.1298 |          1.9814 |               1872 |
| proposed_vs_1RC-EKF          | pooled    | True                       |         1.4127 |          1.8557 |               1579 |
| proposed_vs_2RC-EKF          | fold:DST  | True                       |         1.0377 |          1.5857 |               1339 |
| proposed_vs_2RC-EKF          | fold:FUDS | True                       |         1.6862 |          2.1232 |               1554 |
| proposed_vs_2RC-EKF          | fold:US06 | True                       |         0.6679 |          1.2363 |                900 |
| proposed_vs_2RC-EKF          | pooled    | True                       |         1.2432 |          1.543  |               1339 |
| proposed_vs_2RC-UKF          | fold:DST  | True                       |         1.4737 |          2.367  |               1397 |
| proposed_vs_2RC-UKF          | fold:FUDS | True                       |        10.4235 |         16.7732 |               1785 |
| proposed_vs_2RC-UKF          | fold:US06 | True                       |         0.8108 |          1.6331 |                955 |
| proposed_vs_2RC-UKF          | pooled    | True                       |         4.5745 |          6.7724 |               1397 |
| proposed_vs_adaptive 2RC-EKF | fold:DST  | True                       |         1.0778 |          1.6068 |               1323 |
| proposed_vs_adaptive 2RC-EKF | fold:FUDS | True                       |         1.6683 |          2.1019 |                850 |
| proposed_vs_adaptive 2RC-EKF | fold:US06 | True                       |         0.6526 |          1.082  |               1020 |
| proposed_vs_adaptive 2RC-EKF | pooled    | True                       |         1.2344 |          1.5016 |               1123 |

`acf_lag01_median` is the median, over records in the scope, of the smallest positive lag where the de-meaned paired AE-difference series has `|ACF| < 0.1` using an unbiased FFT autocovariance estimate. These medians are often greater than 900 samples, so stability across the tested 60/300/900 blocks is evidence of sensitivity robustness over this range, not proof that 900 fully spans the record-level dependence scale.
<!-- BOOTSTRAP_SENSITIVITY_END -->
