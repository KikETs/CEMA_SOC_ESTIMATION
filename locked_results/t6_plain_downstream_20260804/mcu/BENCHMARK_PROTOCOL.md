# MCU benchmark protocol

1. Use frozen T6-plain package entries for NMC/LFP, three folds, seeds 0-2.
2. Export an FP32 static ONNX graph with shape `(1, 50, 9)` without graph
   rewrites and verify full-record Torch/ONNX parity.
3. Generate C for `stm32h5` with ST Edge AI Core 4.0.1, optimization=`time`.
4. Build each generated network independently using `-Os` and hard-float FP32.
5. Flash each ELF to NUCLEO-H563ZI and confirm a 250 MHz DWT timebase.
6. For network-only parity, send 64 deterministic normalized windows and compare
   MCU outputs against the corresponding static ONNX graph.
7. For latency, repeat every window 16 times, yielding 1,024 cycle samples per
   model. Report median and p90; no UART time is included.
8. Derive flash/RAM from the linked ELF, map, ST activation report, input buffer,
   and observed stack high-water mark.
9. For end-to-end validation, send raw V/I/T samples to representative NMC and
   LFP builds. R0 lookup, Vcorr, EMA, scaling, buffer update, and inference run
   on the MCU. Report preprocessing and network cycles separately.

Gates:

- all 18 PC ONNX and on-chip network entries must pass;
- absolute MCU/ONNX output difference must remain below `1e-3 %SOC`;
- the two raw V/I/T representative paths must also pass the same parity gate;
- failed models remain in output tables and stop finalization.
