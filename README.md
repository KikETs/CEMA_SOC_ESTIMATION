# CEMA SOC Estimation

Reproduction repository for the NMC and LFP 3-LOPO SOC experiments. The model,
feature, training, EKF, and UKF implementations are preserved from the research
workspaces. Repository changes are limited to data paths, raw-workbook adapters,
failure manifests, and thin orchestration entrypoints.

## Environment

```bash
conda env create -f environment.yml
conda env create -f environment_lfp_kf.yml
conda activate cema_soc_repro
export PYTHONNOUSERSITE=1
```

`PYTHONNOUSERSITE=1` prevents user-site packages from leaking into the clean
Conda environment. Keep it set for installation and every reproduction
command. The YAML files are portable Linux/Windows version/backend contracts.
Exact Linux and Windows Conda builds plus pip-only package locks are under
[`locks/`](locks/README.md). A same-version PyPI NumPy wheel uses a different
BLAS stack and is not equivalent for numerically sensitive UKF trajectories.

The LFP KF paper artifacts came from a different numerical stack. The second
environment, `cema_soc_lfp_kf`, freezes Python 3.12, NumPy 2.3.4/OpenBLAS,
pandas 2.3.3, SciPy 1.17.0, and Numba 0.66. `run_reproduction.py lfp-kf`
automatically uses this sibling environment. All other stages use
`cema_soc_repro`.

Torch is intentionally not in `requirements.txt` because its wheel depends on
the host platform and accelerator. Install the locked PyTorch 2.9.1 build with:

```bash
python scripts/install_torch.py --backend auto
python scripts/verify_environment.py --profile nmc --require-torch
```

`auto` selects the cu128 wheel on Linux/Windows when an NVIDIA driver is
available, the standard macOS wheel with MPS on Apple Silicon, and a CPU wheel
otherwise. Use `--backend cu128`, `--backend mps`, or `--backend cpu` to make
the choice explicit; `--dry-run` prints the command without installing it.

DL execution selects CUDA, then MPS, then CPU. Override this with
`CEMA_TORCH_DEVICE=cuda|mps|cpu`; an unavailable explicitly requested device
is an error rather than a silent fallback. The paper retraining check used an
NVIDIA GPU. Cross-backend results are checked with declared numerical
tolerances and are not expected to be bit-identical. KF and preprocessing do
not require Torch.

On Apple Silicon macOS, create a Python 3.13 environment with the package
versions in `environment.yml`, omitting its Intel MKL-only packages, and then
run the installer above. macOS MPS is supported for DL execution, but it is not
an exact paper numerical environment and has no platform lock in this repository.
`python scripts/run_reproduction.py verify-env` remains the stricter verified
Linux/Windows MKL + Torch check.

## Data

Populate the four raw directories documented in [DATA.md](DATA.md). Do not add
ZIP files to the repository and do not force-add anything under `Data/`.

```bash
python scripts/run_reproduction.py preprocess verify-data
```

Expected prepared scope:

- NMC: 9 records (DST/FUDS/US06 x 0/25/45 C), followed by the declared 25 C
  US06 SOC0 correction.
- LFP: 24 A1-007 records (DST/FUDS/US06 x 8 temperatures).

## Paper runs

Run each stage independently or use `all`:

```bash
python scripts/run_reproduction.py nmc-dl
python scripts/run_reproduction.py lfp-dl
python scripts/run_reproduction.py nmc-kf
python scripts/run_reproduction.py nmc-kf-refit
python scripts/run_reproduction.py lfp-kf
python scripts/run_reproduction.py verify-results
```

DL defaults are the frozen T6-plain configuration: base `V_corr`/current/
temperature plus Vcorr EMA state/deviation pairs at 50, 200, and 800 samples
(9 channels), one-layer GRU (`h=128`, dropout `0.06`), and a plain linear head.
All auxiliary losses are zero. The protocol uses seeds 0-9, batch 2048, 200
epochs, uniform temperature weights, and final-epoch selection. Every run saves
final weights and per-sample test predictions below `runs/{nmc,lfp}_dl/`.

The former G4+GRU anchor-residual configuration is an examined construction
with an anchor-based head and anchor supervision. It was not adopted as the
headline estimator; its runners, exports, and audit records remain available
for historical reconstruction.

`nmc-kf` is the cross-platform paper replay. It uses the frozen selections in
`nmc/kf/locked_parameters/`, all fitted using only each fold's two training
profiles. `nmc-kf-refit` separately repeats ECM, bounds, and Q/R selection from
training profiles and writes to `runs/nmc_kf_refit/`; numerical ties can change
its selected trial, so it is an audit mode and is not used by the paper gate.
LFP KF replays the paper v2.2 per-fold
training-only ECM/Q-R settings and writes fresh CC/EKF/UKF predictions plus
copied parameter tables below `runs/lfp_kf/`. It also writes
`paper_reference_metrics.csv`. Every paper CC initialization is aggregated as
the unweighted mean of the 24 full-evaluation profile-temperature slices; the
slice rows are retained in `paper_cc_openloop_slices.csv`.

Generated `runs/`, `results/`, logs, checkpoints, raw workbooks, and prepared
records are all ignored by Git.

## Frozen inference

Self-contained T6-plain inference packages are tracked under
`locked_results/t6_plain_downstream_20260804/inference_pkg_{nmc,lfp}/`. They
cover all three folds and seeds 0-9. Each entry includes final weights,
training-only normalization and R0 assets, exact channels, frozen inference
code, hashes, and golden-test results. These packages do not require the
training code at inference time. The former seeds 0-2 G4/T6/T7 packages remain
under `nmc/kf/inference_pkg_nmc/` and `lfp/deep_learning/inference_pkg_lfp/` as
historical construction assets.

## Reproduction classes

- Preprocessing and locked KF replay are numerical gates.
- All 28 selected raw workbooks use exact SHA-256 checks. LFP prepared files
  remain byte-exact. NMC cross-platform CSVs must match schema, row count, and
  every non-`V_corr` value exactly; the seven `V_corr`/EMA columns are rebuilt
  causally and compared with `atol=1e-12`, `rtol=0`.
- Full DL retraining is seed/device dependent. Compare the ten-seed aggregate
  and per-seed spread, not bitwise weights across hardware.
- The locked T6-plain ten-seed, condition-unweighted MAEs are NMC
  `0.348279078267482%SOC` and LFP `0.644679400016474%SOC`.

Fresh-run numbers and their numerical scope are recorded in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). `verify-results` exits nonzero when a
declared row is outside its published gate. Stable deterministic rows retain a
`1e-6` gate. Only declared adaptive/UKF and large initial-error rows use measured
Linux/Windows platform gates; generated predictions are always used and no
archived prediction is substituted.

## MCU deployment

The STM32H563ZI neural-network and KF benchmark source, portable path settings,
measured summary tables, and rerun instructions are under
[`deployment/mcu/`](deployment/mcu/README.md). The historical 90-model ONNX
panel remains there. The 18 T6-plain fold/seed builds, their static and dynamic
ONNX exports, ST Edge AI manifest, and measured tables are locked under
`locked_results/t6_plain_downstream_20260804/mcu/`. Raw battery records,
vendor-generated runtime/model code, UART traces, and firmware build products
are not committed.
