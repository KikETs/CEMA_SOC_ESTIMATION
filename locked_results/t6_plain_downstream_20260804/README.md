# T6-plain downstream regeneration

This workspace is isolated from the locked LFP and NMC training repositories.
Source checkpoints and prepared records are read-only; generated files remain
under this directory.

## Frozen proposal

- T6: Vcorr, current, temperature, and Vcorr EMA state/deviation pairs at
  50/200/800 samples (9 channels).
- one-layer GRU, hidden size 128, dropout 0.06.
- plain linear output head with sigmoid SOC output.
- all auxiliary losses disabled.
- batch 2048, 200 epochs, final-epoch weights, uniform temperature weighting.

## Reproduction order

Use `/home/user/anaconda3/envs/torch_env/bin/python`.

```bash
python scripts/build_t6_plain_packages.py
python scripts/validate_t6_plain_packages.py
python scripts/run_t6_plain_robustness.py
python scripts/run_t6_plain_c2_coldstart.py
python scripts/run_kf_bootstrap_t6_plain.py
python scripts/run_c1_t6_plain_baselines.py --execute
python scripts/finalize_c1_t6_plain.py
python scripts/run_c4_t6_plain_zeroshot.py --workers 3
```

MCU export and measurement use the NUCLEO-H563ZI and ST Edge AI Core 4.0.1:

```bash
python mcu/scripts/export_t6_plain_onnx.py
STEDGEAI=/opt/ST/STEdgeAI/4.0/Utilities/linux/stedgeai \
  python mcu/scripts/run_stedgeai_generate.py
STM32_PROGRAMMER_CLI=/home/user/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/STM32_Programmer_CLI \
  python mcu/scripts/run_mcu_benchmark.py
```

The first three commands after packaging are inference-only. C1 is the only
new training in this downstream workspace; C4 is frozen NMC-weight inference on
LFP. The MCU path exports and benchmarks frozen T6-plain weights only.
