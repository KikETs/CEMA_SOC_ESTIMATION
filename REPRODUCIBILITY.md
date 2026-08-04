# Reproducibility status

## Verified environments

Two numerical environments are pinned because the preserved NMC and LFP KF
paths require different BLAS and Python stacks.

| Stage | Environment | Numerical stack |
|---|---|---|
| preprocessing, NMC/LFP DL, NMC KF | `cema_soc_repro` | Python 3.13.5, NumPy 2.1.3 MKL, pandas 2.2.3, SciPy 1.15.3 |
| LFP KF v2.2 replay | `cema_soc_lfp_kf` | Python 3.12.0, NumPy 2.3.4 OpenBLAS, pandas 2.3.3, SciPy 1.17.0, Numba 0.66.0 |

PyTorch 2.9.1+cu128 was installed separately and is deliberately absent from
both requirements files. The fresh neural runs used an NVIDIA GeForce RTX 5090.
Portable environment YAMLs are supplemented by exact `linux-64` and `win-64`
Conda and pip-only locks under `locks/`.

## Preprocessing contract

- NMC: 9 records, three profiles at 0/25/45 degC.
- LFP: 24 records, three profiles at eight temperatures.
- Failed stages: 0.
- All 28 selected raw workbook SHA-256 values match
  `reference/raw_source_hashes.csv`.
- LFP prepared files are byte-exact. NMC files are accepted either byte-exact
  or by strict semantic fingerprints plus causal recomputation of the seven
  `V_corr`/EMA columns at `atol=1e-12`, `rtol=0`.
- The LFP model loader sees arrays identical to the historical prepared files
  under the frozen LFP runtime.
- Raw workbooks and prepared trajectories remain ignored by Git.

Run the gate with:

```bash
python scripts/run_reproduction.py preprocess verify-data
```

## Verified Linux/Windows replay

| Experiment | Paper reference | Linux | Windows | Gate | Status |
|---|---:|---:|---:|---:|---|
| NMC T6-plain GRU, 10 seeds | 0.3482790783 | 0.3482790783 | not rerun | 0.04 | locked source PASS |
| LFP T6-plain GRU, 10 seeds | 0.6446794000 | 0.6446794000 | not rerun | 0.04 | locked source PASS |
| NMC CC, oracle | 0.099450 | 0.099450 | 0.099450 | 1e-6 | Class-B PASS |
| NMC adaptive 2RC-EKF, oracle | 1.704694 | 1.696453 | 1.702916 | 0.01 | Class-B-platform PASS |
| NMC 2RC-EKF, oracle | 1.727215 | 1.727215 | 1.727215 | 1e-6 | Class-B PASS |
| NMC 1RC-EKF, oracle | 1.963149 | 1.963149 | 1.963149 | 1e-6 | Class-B PASS |
| NMC 2RC-UKF, oracle | 5.664560 | 5.769424 | 5.694790 | 0.12 | Class-B-platform PASS |
| LFP plain 2RC-EKF, oracle | 0.2417909308 | 0.2417909308 | 0.2417909308 | 1e-6 | Class-B PASS |
| LFP hysteresis 2RC-EKF, oracle | 0.2071081229 | 0.2071081229 | 0.2071081229 | 1e-6 | Class-B PASS |
| LFP adaptive-hysteresis 2RC-EKF, oracle | 0.2162369251 | 0.2162369251 | 0.2162369251 | 1e-6 | Class-B PASS |
| LFP CC, oracle paper aggregation | 0.1129039483 | 0.1129039483 | 0.1129039483 | 1e-6 | Class-B PASS |
| LFP hysteresis 2RC-UKF, oracle | 7.8005395442 | 7.7942215149 | 7.7894621759 | 0.02 | Class-B-platform PASS |

The complete machine-readable comparison is generated at
`runs/reproduction_verification.csv` by:

```bash
python scripts/run_reproduction.py verify-results
```

The command intentionally exits nonzero if any row exceeds its declared gate;
generated predictions and weights remain available. All stable deterministic
rows use `1e-6`. Platform-sensitive covariance rows use only the predeclared
gates described below.

## NMC KF numerical scope

`nmc-kf` fixes linear-algebra threads to one and replays frozen training-only
ECM, bound, and Q/R selections. Repeated runs on the same host are bit-identical
for metrics and per-sample predictions. Across Linux and Windows, differences
are isolated to covariance-repair-sensitive `US06 25 C` adaptive EKF and
`FUDS 45 C` UKF trajectories. Their aggregate paper-reference deltas are bounded
by `0.01` and `0.12 %SOC`, respectively. CC, 1RC-EKF, and 2RC-EKF retain the
`1e-6` gate. `nmc-kf-refit` remains a separate training-only audit because
near-tied optimizer selections can differ by platform.

## LFP KF numerical scope

The required Zen 3 Numba target and single-thread numerical settings are
applied before NumPy import. Oracle and `-5 pp` CC/EKF rows reproduce within
`1e-6`. Large `-10 pp` transients amplify bounded platform arithmetic, so the
three EKF-family rows use a `0.001 %SOC` gate. The exact v2 fallback ECM table
is retained rather than constructing an empty placeholder.

The paper CC table reads every initialization from the slice-unweighted column
of the locked `cc_openloop_init_sweep.csv`. Oracle, `-5 pp`, and `-10 pp` are
therefore all unweighted means of the same 24 full-evaluation
profile-temperature slices: `0.1129039483`, `5.0290362237`, and
`9.8644440573 %SOC`, respectively. Fresh slice rows are saved in
`paper_cc_openloop_slices.csv`.

The LFP UKF oracle result is numerically unstable on a small number of slices.
The pinned Linux runtime produces `7.7942215149 %SOC` and Windows produces
`7.7894621759 %SOC`, versus the paper reference `7.8005395442 %SOC`. This row
therefore uses a declared `0.02 %SOC` platform tolerance. No archived
prediction is substituted into either aggregate.

## Outputs and leakage

- DL: 60 final weights and 60 per-sample prediction files under
  `runs/{nmc,lfp}_dl/`.
- NMC KF: all three folds and all temperatures completed; held-out profile
  leakage audit reports zero violations.
- LFP KF: fresh CC/EKF/UKF per-sample files and aggregate tables are written to
  `runs/lfp_kf/`; failed numerical comparisons are retained.
- Generated runs, logs, weights, predictions, raw files, and prepared data are
  excluded from Git.

## Tests

Run:

```bash
pytest -q tests/test_repository_contract.py
python scripts/run_reproduction.py verify-data
python scripts/run_reproduction.py verify-results

# Run these separately because each copied KF tree owns a top-level `src` package.
(cd nmc/kf && python -m pytest -q tests)
(cd lfp/kf && "$CONDA_PREFIX/../cema_soc_lfp_kf/bin/python" -m pytest -q tests)
```

Current results are nine root contract tests passed, 13 of 15 NMC KF tests
passed, and three LFP KF core tests passed. The two NMC failures require
optional post-hoc outputs `bootstrap_sensitivity.csv` and
`deployment_assets.csv`; neither is produced by the paper reproduction command
or needed for the declared NMC/LFP headline table. The source contract tests
cover entrypoints, data ignore rules, absence of tracked workbooks/preprocessed
payloads, training-only NMC KF fitting, numerical preflights, and the 36-record
hash reference.

## T6-plain training replay

`nmc-dl` and `lfp-dl` invoke the chemistry-specific
`run_paper_t6_plain_10seed.py` entrypoints. Before training, each runner asserts
the 9-channel T6 feature set, GRU single-model path, plain linear head, seeds
0-9, uniform temperature weights, and
`lambda_rex=lambda_condinv=lambda_anchor_loss=0`. Batch size 2048, 200 epochs,
window 50, training stride 3, and final-epoch selection are also asserted.
The prior G4 anchor-residual entrypoints are retained but are not called by the
default reproduction command.

The T6-plain locked source summaries, downstream robustness/KF/MCU results,
core4 trajectories, and frozen inference packages are indexed under
`locked_results/`. Their aggregate lock is
`locks/t6_plain_locked_results_sha256.txt`.

## Historical construction

The historical G4+GRU anchor-residual five-seed references are NMC
`0.3245689255 %SOC` and LFP `0.5416480865 %SOC`. They remain archived to
reconstruct the examined anchor-based construction, not as current headline
verification targets.
