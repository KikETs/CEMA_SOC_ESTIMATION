# Windows validation final

## Status

**PASS** for the supported cross-platform reproduction contract.

Validated on Windows 11 with Python 3.13.5 / NumPy 2.1.3 / MKL 2023.1 for
preprocessing, NMC KF, and DL, plus Python 3.12.0 / NumPy 2.3.4 / OpenBLAS
0.3.30 for LFP KF.

## Data contract

- Selected raw workbooks: 31/31 exact SHA-256.
- LFP prepared trajectories: 24/24 byte-exact.
- NMC prepared trajectories: 12/12 strict semantic match. All non-`V_corr`
  values match exactly; seven causal `V_corr`/EMA columns reproduce within
  `atol=1e-12`, `rtol=0`.
- Raw and preprocessed battery files tracked by Git: 0.

## Execution contract

- Repository contract tests: 17/17 passed.
- NMC locked KF: 3 folds x 3 temperatures, 669.98 s, failed runs 0, leakage
  violations 0, linear-algebra threads 1.
- LFP KF: 24 trajectories x 3 initial conditions, 2,479,740 prediction rows,
  failed runs 0.
- NMC/LFP five-seed DL results from the completed Windows validation remain
  within their Class-C gates.

## NMC locked KF aggregate MAE

| Method | Paper | Windows | Gate | Status |
|---|---:|---:|---:|---|
| CC | 0.099450 | 0.099450 | 1e-6 | PASS |
| Adaptive 2RC-EKF | 1.704694 | 1.702916 | 0.01 | PASS |
| 2RC-EKF | 1.727215 | 1.727215 | 1e-6 | PASS |
| 1RC-EKF | 1.963149 | 1.963149 | 1e-6 | PASS |
| 2RC-UKF | 5.664560 | 5.694790 | 0.12 | PASS |

Stable rows retain the strict gate. The adaptive and UKF gates cover only the
measured deterministic single-thread Linux/Windows covariance-repair range.
Fresh parameter identification is available separately as `nmc-kf-refit` and
is not substituted for the locked paper replay.

## Commands

```powershell
$env:PYTHONNOUSERSITE = "1"
python scripts/run_reproduction.py verify-data
python scripts/run_reproduction.py nmc-kf --output-root runs\nmc_kf_windows_singlethread_final
python scripts/run_reproduction.py lfp-kf --output-root runs\lfp_kf_windows_final
python scripts/run_reproduction.py verify-results `
  --nmc-kf-root runs\nmc_kf_windows_singlethread_final `
  --lfp-kf-root runs\lfp_kf_windows_final
python -m pytest -q tests\test_repository_contract.py
```
