# Locked T6-plain results

This tree contains the data-free locked outputs used by the T6-plain paper
configuration. Raw workbooks and prepared battery trajectories are not present.

- `CEMA_CARRIER_FAIRNESS_ABLATIONS_20260803/`: normal-head/auxiliary-loss
  comparisons, including the 10-seed T6-plain headline source rows.
- `t6_plain_downstream_20260804/`: 10-seed frozen inference packages, tier-3
  robustness, cold-start, fair-input, zero-shot, KF bootstrap, core4, ONNX, and
  STM32 measured outputs.
- `adaptive_filter_baseline/`: the additional frozen-ECM adaptive filter result.
- `lfp_kf_v2_2/cc_openloop_init_sweep.csv`: the corrected CC aggregation source.

The inference package weights are copied byte-for-byte. Public-package
manifests use repository-relative asset names instead of workstation-specific
absolute paths. This path rebasing does not alter model, scaler, R0, code, or
prediction payloads. Each logical directory has a regenerated `sha256sums.txt`,
and the complete tree is covered by
`locks/t6_plain_locked_results_sha256.txt`.

## Not included

The manuscript-side 500-key `numbers.yaml` and its `make_si_tables.py`,
`make_result_figures.py`, `make_trajectory_figures.py`,
`make_threshold_analysis.py`, and `make_graphical_abstract.py` sources were not
present in the workstation, attachments, or public remote at packaging time.
They are not reconstructed here. The repository's executable headline/KF gate
is `scripts/verify_paper_results.py`; the locked CSVs needed to ingest the new
T6-plain values are present in this tree.
