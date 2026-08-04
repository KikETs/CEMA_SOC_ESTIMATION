# KF Python-C alignment audit

## Scope

This audit compares the STM32 implementation with the frozen Python paths used
to create the archived NMC and LFP results. The NMC reference is
`nmc/kf/src/{ekf,ukf,ocv}.py`. The LFP paper reference is the frozen v2.2 replay
in `lfp/kf/run_paper_filters.py`, which calls `lfp/kf/src/v2_core.py`; the
generic LFP EKF module is not the paper-result reference.

## Corrections

1. NMC OCV evaluation now packages the exact SciPy PCHIP knots and
   coefficients. Firmware evaluates the cubic and its analytic derivative
   instead of using a dense float32 linear approximation.
2. LFP usable capacity and hysteresis gamma are latched from nominal record
   temperature at reset, matching the record-constant Python inputs.
3. NMC and LFP UKF use their reference-specific jitter placement, covariance
   projection order, clipping order, and innovation floors.
4. All non-CC filters now use FP64 state, covariance, fitted assets, raw V/I/T
   transport, and SOC output. CC remains FP32.
5. Protocol v5 carries both actual initial temperature and nominal record
   temperature so online temperature and record-level characterization are not
   conflated.

## Measured outcome

The corrected run completed 30/30 firmware configurations and 165/165
fold-temperature slices with zero failures.

| Group | Mean absolute MCU-Python difference (%SOC) | Worst maximum difference (%SOC) |
|---|---:|---:|
| NMC 1RC-EKF | 0.0000010 | 0.0000030 |
| NMC 2RC-EKF | 0.0000014 | 0.0000307 |
| LFP plain 2RC-EKF | 0.0000011 | 0.0000030 |
| LFP hysteresis 2RC-EKF | 0.0000011 | 0.0000030 |
| LFP adaptive hysteresis 2RC-EKF | 0.0000011 | 0.0000030 |
| NMC adaptive 2RC-EKF | 0.0157142 | 7.9797563 |
| NMC 2RC-UKF | 0.6266384 | 37.5804259 |
| LFP hysteresis 2RC-UKF | 0.0238780 | 3.9195204 |

The ordinary EKFs and all LFP EKFs now reproduce the frozen Python trajectories
to numerical noise. The remaining large deviations are confined to NMC
adaptive US06 25 C and unstable UKF slices. These filters repeatedly project a
near-singular covariance: NumPy uses LAPACK eigendecomposition while firmware
uses an FP64 Jacobi eigensolver and local Cholesky. Tiny ordering differences
can select different later trajectories. The MCU rows are retained, not
replaced or hidden, and are labeled numerical-backend sensitivity.

## Interpretation

`onchip_status=PASS` means the full replay completed with finite output. It does
not assert trajectory equivalence. Use `retention_class` and the explicit
MCU-Python difference columns for that question. Do not use the sensitive UKF
or NMC adaptive MCU MAE as a bit-equivalent reproduction of the NumPy result.
