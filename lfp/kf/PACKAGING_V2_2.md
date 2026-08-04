# LFP KF v2.2 reporting-finalization package

## Scope

- Additive v2.2 reporting finalization with no Q/R re-selection, ECM refit, or filter-equation change.
- Official five-seed, slice-unweighted confirmatory G4 headline and matched circular-block bootstrap.
- Frozen-parameter minus-10 pp inference, UKF 120 s predict-only diagnostic, and current-bias stress.
- Locally generated LFP/NMC trade-off overlay and exact plotted CSV.
- v2.1 frozen inputs are included for traceability; v1/v2/v2.1 numerical artifacts were not edited.

## Entry points

- Narrative: `report.md`, section `v2.2 — reporting finalization (no re-tuning)`
- Main table: `results_v2_2/main_table.csv`
- Bootstrap: `results_v2_2/paired_bootstrap.csv`
- UKF SI: `results_v2_2/si_ukf_table.csv`
- Current-bias stress: `results_v2_2/current_bias_stress.csv`
- Trade-off overlay: `results_v2_2/fig_tradeoff_overlay.png`, `.pdf`, and `_plotted.csv`
- Immutability/leakage gate: `results_v2_2/leakage_audit_v2_2.json`
- Completeness gate: `results_v2_2/completeness_audit.json`
- Reproduction code: `run_v2_2.py`

`MANIFEST.sha256` contains a SHA-256 checksum for every packaged file except the manifest itself.
