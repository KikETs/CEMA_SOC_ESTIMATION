# Platform environment locks

`environment.yml` and `environment_lfp_kf.yml` are portable version/backend
contracts. The files below capture the exact Conda builds and pip-installed
packages used for the verified Linux and Windows runs:

- `linux-64/{cema_soc_repro,cema_soc_lfp_kf}-conda-explicit.txt`
- `win-64/{cema_soc_repro,cema_soc_lfp_kf}-conda-explicit.txt`
- matching `*-pip-freeze.txt` files containing only packages whose Conda
  channel is `pypi`

Create an exact environment with the lock for the host platform, then install
the matching pip-only file. The recorded Linux/Windows NMC locks contain the
verified cu128 wheel and therefore retain the official PyTorch CUDA index:

```bash
conda create -n cema_soc_repro --file locks/linux-64/cema_soc_repro-conda-explicit.txt
conda run -n cema_soc_repro python -m pip install \
  -r locks/linux-64/cema_soc_repro-pip-freeze.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128

conda create -n cema_soc_lfp_kf --file locks/linux-64/cema_soc_lfp_kf-conda-explicit.txt
conda run -n cema_soc_lfp_kf python -m pip install \
  -r locks/linux-64/cema_soc_lfp_kf-pip-freeze.txt
```

Replace `linux-64` with `win-64` on Windows. The lock files contain package
metadata only. They contain no raw or preprocessed battery records.

For a portable fresh environment, install the non-Torch dependencies first and
then run `python scripts/install_torch.py --backend auto`. macOS has no exact
Conda build lock in this repository; use the YAML's package versions without
its Intel MKL-only packages, followed by the locked PyTorch version. macOS is a
compatible DL execution target rather than the reference paper environment.

## Publication-locked T6-plain outputs

`t6_plain_locked_results_sha256.txt` covers every file under
`locked_results/`, including the 10-seed source rows, frozen T6-plain inference
packages, robustness and KF comparisons, core4 per-sample trajectories, ONNX
exports, and STM32 measured summaries. Subdirectories also carry local
`sha256sums.txt` files for independent transfer checks.
