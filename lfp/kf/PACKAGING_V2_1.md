# LFP KF v2.1 + NMC trade-off package

## Scope

- LFP v2.1 feasible-branch Q/R selection, four-filter D3 evaluation, confirmatory 17-channel G4 comparison, block bootstrap, ECM/A1 diagnostics, and D4 trade-off sweep.
- NMC D4 overlay from the pre-declared representative DST fold at 0/25/45 C, with eight finite q_soc values plus exact open-loop.
- Frozen v1/v2 numerical artifacts are not duplicated or modified; their immutability result is recorded in `results_v2_1/leakage_audit_v2_1.json`.

## NMC provenance

- Remote source repo: `/home/user/바탕화면/DL/NMC_KF_3LOPO_BASELINES`
- Training/test file hashes and source-artifact hashes: `results_v2_1/nmc_tradeoff_audit.json`
- Packaged remote runner snapshot: `results_v2_1/nmc_runner_source.py`
- NMC output: `results_v2_1/tradeoff_nmc.csv` (27 rows, schema-identical to the LFP trade-off CSV)

## Entry points

- Narrative: `report.md`, section `v2.1`
- Completion gate: `results_v2_1/completeness_audit.json`
- LFP leakage/immutability audit: `results_v2_1/leakage_audit_v2_1.json`
- NMC leakage/provenance audit: `results_v2_1/nmc_tradeoff_audit.json`
- Main comparison: `results_v2_1/main_table.csv`
- LFP/NMC trade-off: `results_v2_1/tradeoff.csv`, `results_v2_1/tradeoff_nmc.csv`
- Reproduction config/code: `configs/v2_1.yaml`, `run_v2_1.py`, `src/v2_1_core.py`

`MANIFEST.sha256` contains a SHA256 checksum for every packaged file except the manifest itself.
