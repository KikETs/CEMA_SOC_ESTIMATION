# STM32H563ZI deployment benchmark

The current T6-plain FP32 batch-1 results and 18-model ONNX panel are locked
under `../../locked_results/t6_plain_downstream_20260804/mcu/`. The material
below preserves the broader historical construction panel and shared
firmware/KF benchmark source.

This directory preserves the source and compact evidence for the measured
CEMA SOC deployment benchmark. The measured target was a NUCLEO-H563ZI with
an STM32H563ZIT6 Cortex-M33 at 250 MHz.

## Contents

- `BENCHMARK_PROTOCOL.md`: original pre-declared benchmark protocol.
- `scripts/`: frozen export, parity, firmware-install, board-runner, and
  summarization scripts. Only repository/tool paths were made portable.
- `firmware/h563_bench/`: normalized-window neural inference firmware.
- `firmware/h563_raw_vit/`: raw V/I/T input, causal preprocessing, and neural
  inference firmware.
- `firmware/h563_kf/`: CC, EKF, adaptive EKF, and UKF firmware.
- `onnx/`: validated FP32 ONNX exports for all 45 frozen neural configurations,
  in static-batch and dynamic-batch forms.
- `results/`: final report, numerical-alignment audit, compact measured tables,
  and the non-raw source summaries used to produce them.

The algorithm implementations and measured CSV values are unchanged from the
completed benchmark drop. The paths were changed from the original workstation
layout to repository-relative paths or environment-variable overrides.

## Required local assets

The repository tracks the frozen NMC and LFP inference packages. Battery
records remain local under the paths defined by `Data/README.md`:

```text
Data/Preprocessed/NMC/nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean/
Data/Preprocessed/LFP/prepared_data_ocv_discharge_soc/
```

The G1 archived-prediction check additionally expects untracked files under:

```text
Data/ArchivedPredictions/NMC/
Data/ArchivedPredictions/LFP/
```

No raw or preprocessed battery trajectory is included in Git.

## Toolchain

The measured environment used:

- ST Edge AI Core 4.0.1 / STM32CubeAI 12.0.1-RC2
- STM32CubeIDE 2.2.0 and GNU Arm 14.3.1
- STM32CubeH5 v1.7.0
- STM32CubeProgrammer
- Python packages from the repository environment plus `onnx`,
  `onnxruntime`, and `pyserial`

Set tool locations before running when they are not on `PATH`:

```bash
export STEDGEAI=/path/to/stedgeai
export ARM_GCC_BIN=/path/to/gnu-arm/bin
export ARM_GCC="$ARM_GCC_BIN/arm-none-eabi-gcc"
export ARM_SIZE="$ARM_GCC_BIN/arm-none-eabi-size"
export STM32_PROGRAMMER_CLI=/path/to/STM32_Programmer_CLI
export STM32CUBEIDE_HEADLESS_BUILD=/path/to/headless-build.sh
export CEMA_SERIAL_PORT=/dev/ttyACM0
export PYTHON=python
```

For the Makefiles, pass the same toolchain and firmware roots:

```bash
make -C deployment/mcu/firmware/h563_raw_vit \
  TOOLCHAIN="$ARM_GCC_BIN" \
  FW_ROOT="$HOME/STM32Cube/Repository/STM32Cube_FW_H5_V1.7.0"
```

## Generated vendor assets

The validated ONNX source models are tracked, but ST-generated neural assets
are intentionally excluded. After
`scripts/run_stedgeai_generate.py`, install the selected model with
`scripts/install_firmware_model.py` or `scripts/install_raw_vit_model.py`.
For the first neural build, copy the generated `Inc/` and `Lib/` directories
from one ST Edge AI model output to the corresponding firmware
`AI/Runtime/Inc` and `AI/Runtime/Lib` directories. The model-specific
`cema*.c` and `cema*.h` files are installed by the scripts.

KF parameter headers are generated from the frozen training-only assets by
`scripts/install_kf_model.py` and written under `firmware/h563_kf/Generated/`.

## Reproduction order

Run from `deployment/mcu/`:

```bash
python scripts/run_g0_g1.py
python scripts/run_g2_onnx.py
python scripts/diagnose_g2_fp32.py
python scripts/run_stedgeai_generate.py
python scripts/prepare_onchip_windows.py
python scripts/run_mcu_benchmark.py
python scripts/run_raw_vit_mcu_benchmark.py
python scripts/recompute_raw_vit_onnx_reference.py
python scripts/finalize_drop.py
python scripts/run_kf_mcu_benchmark.py
python scripts/finalize_kf_mcu_results.py
```

The board runners flash one configuration at a time. They accept `--start-at`
and `--stop-after` for resumable execution. UART transfer is excluded from the
DWT timing interval.

## Tracked versus excluded evidence

Tracked evidence includes aggregate latency, memory, parity, accuracy-retention,
coverage, environment, and alignment tables. The 90 validated ONNX source
models are also tracked. Excluded local artifacts include:

- ST-generated model/runtime files
- frozen-weight duplicates
- raw 1024-cycle latency rows
- full per-sample MCU/Python parity trajectories
- UART/build logs and Debug/Release/Build directories

The exclusion keeps the repository free of battery records, generated vendor
code, and raw traces while retaining the implementation, portable ONNX models,
and numerical tables used in the deployment report.
