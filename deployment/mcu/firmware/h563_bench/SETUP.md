# STM32H563ZI AI/KF benchmark base

## Target

- Board: NUCLEO-H563ZI
- MCU: STM32H563ZIT6, Cortex-M33 with FPU
- TrustZone: disabled
- SYSCLK/HCLK: 250 MHz from the 8 MHz ST-LINK MCO clock
- Instruction cache: direct-mapped ICACHE enabled
- Debug: SWD through on-board ST-LINK V3
- Console: USART3 on PD8/PD9, 115200 baud, 8 data bits, no parity, 1 stop bit
- User LED: PB0
- User button: PC13
- HAL time base: SysTick
- Heap: 4 KiB
- Stack: 8 KiB
- Toolchain: STM32CubeIDE GNU Arm
- FPU: FPv5 single precision, hard-float ABI
- Build profiles: Debug (`-O0`) and Release (`-Os`)

## Installed firmware

- Package: STM32CubeH5 v1.7.0
- Default package path: `$HOME/STM32Cube/Repository/STM32Cube_FW_H5_V1.7.0`
- Upstream tag commit: `4e039436d8561003a1d88b78a1520f3f81eadd24`

The application source is tracked. ST-generated runtime/model files, model
weights, and build products are intentionally excluded and regenerated locally.

## Regenerate

CubeMX 6.18 cannot generate directly into the Korean desktop path in command-line
mode. `regenerate.sh` uses an ASCII staging directory and copies the generated
project back without changing the final project location.

```bash
./regenerate.sh
```

## Build

```bash
./build.sh Debug
./build.sh Release
```

The generated ELF, map, and listing files are written under `Debug/` or
`Release/`. Flashing is intentionally not part of these scripts.
