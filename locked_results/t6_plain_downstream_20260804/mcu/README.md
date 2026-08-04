# T6-plain STM32H563 benchmark

## Scope

- Board: NUCLEO-H563ZI, STM32H563ZIT6 Cortex-M33 at 250 MHz.
- Runtime: ST Edge AI Core 4.0.1-20581, STM32CubeAI 12.0.1-RC2.
- Build: GNU Arm, `-Os`, hard-float FP32.
- Models: NMC/LFP x DST/FUDS/US06 x seeds 0/1/2 = 18.
- Runtime input shape: batch 1 x 50 samples x 9 T6 channels.

All 18 networks were exported to ONNX, generated with ST Edge AI, rebuilt,
flashed, and measured separately. Network-only latency uses 64 deterministic
windows and 16 repeated inferences per window (1,024 cycle samples per model).

The raw-sensor firmware sends only one V/I/T sample at a time. Vcorr, R0 lookup,
EMA state, normalization, and the 50-sample window buffer run on the MCU. Since
those operations are identical across folds and seeds, this end-to-end path was
measured for one representative model per chemistry. The full 18-model set was
still built and measured for network latency, memory, and on-chip parity.

## Gate results

- PC ONNX parity: 18/18 PASS.
- ST Edge AI generation: 18/18 PASS.
- On-chip network parity: 18/18 PASS.
- Raw V/I/T end-to-end parity: 2/2 PASS.
- Median network latency: approximately 190.7-191.0 ms per estimate.
- Representative preprocessing latency: approximately 92.3 us per sample.
- Model weights: 403,476 bytes; activation arena: 54,784 bytes.
- Total firmware flash: 448,820 bytes; total measured static RAM: 81,880 bytes.

PC runtime is not presented as MCU latency. The MCU tables are derived from the
DWT cycle counter and retain the raw 1,024-cycle samples for every model.
