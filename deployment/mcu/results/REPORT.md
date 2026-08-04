# MCU Results Intake Report

## Q1. Board and precision

The measured board was NUCLEO-H563ZI with STM32H563ZIT6 Cortex-M33 at a
measured SystemCoreClock of 250 MHz. Neural inference used FP32 through
ST Edge AI Core 4.0.1 / STM32CubeAI 12.0.1-RC2. The compiler was GNU Arm
14.3.1 with `-Os -mfpu=fpv5-sp-d16 -mfloat-abi=hard`.

CC used FP32. All EKF/UKF builds used software FP64 for state, covariance,
assets, transport and output. Therefore the requested EKF/UKF FP32 rows are
`MISSING_not_measured`, not relabeled FP64 results. Evidence:
`environment_mcu.txt`, `kf_mcu_results.csv:precision`.

## Q2. On-board preprocessing equivalence

For the benchmarked fixed-1 s path, the MCU uses double EMA recurrence and
float32 feature storage, matching the frozen Python arithmetic convention.
The PC transcription proxy reports zero divergence before the network, while
the measured network-output maximum relative difference is recorded in
`nn_stage_divergence.csv`. Intermediate MCU values were not exposed, so all
pre-network stage rows are explicitly marked as PC proxy rather than on-chip
capture.

## Q3. Time and temperature source

The temperature source is identical: the host transmitted the chamber
`TempLabel` scalar. The MCU did not use `Temperature(C)`.

The time source is not identical. Python prefers recorded timestamps, whereas
the MCU protocol transmits no timestamp and uses exactly 1.0 s. The resulting
PC-only proxy has mean absolute prediction difference 0.011469887004986948 %SOC,
maximum difference 0.1646503806114196 %SOC, and worst signed slice MAE change
0.0131059341871008 %SOC. It is not an on-chip full-record measurement. See
`nn_dt_proxy.csv` and `nn_input_convention.csv`.

## Q4. Neural accuracy display precision

Full-record PyTorch-to-ONNX accuracy is available, but full-record
MCU-to-Python accuracy was not measured. Consequently MCU delta-MAE and the
three-decimal display-change decision remain `MISSING_not_measured_full_record`
in `nn_accuracy_delta.csv`. The partial-record end-to-end maximum difference
across 45 checkpoints is 1.921325684106634e-05 %SOC.

## Q5. Latency and duty

For raw V/I/T end-to-end G4 GRU inference, median latency is
233941.62 us for NMC and 234398.562 us for LFP. At 1 Hz these occupy
23.394161999999998% and 23.4398562% of the one-second
budget. Median G4 preprocessing cost is 164.382 us; the remainder is
network inference. Each row uses 1024 DWT measurements after ten warmups.
UART is outside the cycle interval. Evidence: `nn_latency.csv`.

## Q6. Flash and RAM

The exact per-checkpoint values are in `nn_memory.csv`. G4 uses a 3400-byte
window, T6 uses 1800 bytes, and T7 uses 2200 bytes. Activation arenas, not EMA
state, dominate RAM. Feature-count changes explain the window and scaler-table
differences, while the recurrent network dominates Flash and latency.

## Q7. KF precision and initial-condition changes

Only oracle initialization was run on the MCU. CC FP32 can be compared with
the locked reference. EKF/UKF results are actual software FP64 and are retained
in appended `mae_mcu_actual_pct` columns, but they do not populate
`mae_mcu_fp32_pct`. Minus-5 and minus-10 pp MCU rows are unmeasured. Evidence:
`kf_precision_delta.csv`, `coverage_matrix.csv`.

Actual-precision slice-unweighted MCU values:

| chemistry   | method                      | precision   |   mae_pct |
|:------------|:----------------------------|:------------|----------:|
| LFP         | 2rc_ukf                     | fp64        | 7.7955    |
| LFP         | adaptive_hysteresis_2rc_ekf | fp64        | 0.216237  |
| LFP         | cc                          | fp32        | 0.112684  |
| LFP         | hysteresis_2rc_ekf          | fp64        | 0.207108  |
| LFP         | plain_2rc_ekf               | fp64        | 0.241791  |
| NMC         | 1RC_EKF                     | fp64        | 1.96315   |
| NMC         | 2RC_EKF                     | fp64        | 1.72721   |
| NMC         | 2RC_UKF                     | fp64        | 6.25504   |
| NMC         | Adaptive_2RC_EKF            | fp64        | 1.69537   |
| NMC         | CC                          | fp32        | 0.0994736 |

## Q8. Four qualitative FP32 claims

The four-claim FP32 filter comparison cannot be determined because EKF/UKF
FP32 and minus-5/minus-10 pp MCU runs were not performed. The CC-only FP32
result remains close to its locked reference, but it is insufficient to claim
preservation of the complete method ordering. This is a missing measurement,
not a negative result.

## Q9. FP32 degradation attribution

The mandatory FP32 diagnostic counters were not instrumented. For actual
software-FP64 EKF/UKF builds, FP32 Q-addition and cancellation counters are
not applicable. Existing adaptive-EKF/UKF discrepancies are classified as
numerical-backend sensitivity between NumPy LAPACK and the MCU Jacobi
projection/local Cholesky, not FP32 truncation. See
`kf_fp32_diagnostics.csv` and `KF_ALIGNMENT_AUDIT.md`.

## Q10. Mitigations

The firmware applies Joseph-form EKF covariance updates, explicit
symmetrization, covariance flooring/projection, and FP64 OCV/ECM assets.
It does not use Kahan CC accumulation, `expm1`, or innovation gating.
No controlled before/after MAE ablation was run; those cells remain missing.
See `kf_mitigations.csv`.

## Q11. MCU failures

All 30 KF/CC firmware configurations and 165 oracle-init slices completed
with finite outputs. `kf_failures.csv` has zero data rows. Numerical retention
is separate: NMC adaptive EKF and both chemistries' UKF have backend-sensitive
slices documented in `KF_ALIGNMENT_AUDIT.md`.

## Q12. OCV table reduction

Compiled table footprints are measured and listed in
`kf_tables_footprint.csv`. The accuracy cost of table representation or
reduction was not independently measured, so it remains
`MISSING_not_measured`. No claim is made that the unreduced source table
would or would not fit solely from these measurements.

## Q13. Runtime assets and initial SOC

`runtime_asset_asymmetry.csv` lists each compiled asset in bytes. Neural
estimators store weights, fold-specific R0 and normalization constants,
double EMA state, a float32 window and an activation arena; they do not
require initial SOC. KF/CC stores fold-specific OCV/ECM/Q/R assets and
state/covariance and requires oracle initial SOC at reset.

## Q14. Precision-sensitivity asymmetry

It cannot be concluded which estimator family is more sensitive to an
FP64-to-FP32 reduction: the neural FP32 path was measured, but EKF/UKF were
measured in software FP64. The observed UKF/adaptive discrepancies are
backend-algorithm sensitivity and must not be presented as FP32 degradation.

## Q15. Missing, failed and anomalous items

- Missing: full-record neural MCU parity and MCU delta-MAE.
- Missing: on-chip intermediate-stage captures; PC proxies are provided.
- Missing: EKF/UKF FP32 builds, FP32 diagnostic counters and mitigation
  before/after ablations.
- Missing: MCU minus-5/minus-10 pp initial-SOC runs.
- Missing: KF predict/update/OCV-lookup latency decomposition and RMSE.
- Missing: OCV-table representation accuracy cost.
- Excluded: one-off NMC G4 MLP result, because the user explicitly requested
  that no result artifact be saved.
- Failed MCU runs: none.
- Anomaly: adaptive EKF/UKF backend-sensitive slices are retained and listed
  in `KF_ALIGNMENT_AUDIT.md`.

## Coverage summary

| track                   | chemistry   | measured   |   count |
|:------------------------|:------------|:-----------|--------:|
| A_nn_full_record_e2e    | LFP         | False      |      27 |
| A_nn_full_record_e2e    | NMC         | False      |      18 |
| A_nn_latency            | LFP         | True       |      27 |
| A_nn_latency            | NMC         | True       |      18 |
| A_nn_memory             | LFP         | True       |      27 |
| A_nn_memory             | NMC         | True       |      18 |
| A_nn_oneoff_G4_MLP      | NMC         | False      |       1 |
| A_nn_partial_record_e2e | LFP         | True       |      27 |
| A_nn_partial_record_e2e | NMC         | True       |      18 |
| A_nn_stage_proxy        | LFP         | True       |      27 |
| A_nn_stage_proxy        | NMC         | True       |      18 |
| B_kf_fp32_filter        | LFP         | False      |      36 |
| B_kf_fp32_filter        | NMC         | False      |      36 |
| B_kf_onchip_result      | LFP         | False      |      30 |
| B_kf_onchip_result      | LFP         | True       |      15 |
| B_kf_onchip_result      | NMC         | False      |      30 |
| B_kf_onchip_result      | NMC         | True       |      15 |
