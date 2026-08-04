# Frozen LFP v2.2 parameters

Copied from:

`/home/user/바탕화면/LFP_REMOTE_EXACT_COPY_20260713/LFP_KF_LOPO_BASELINES_ISOLATED`

Included files are the compact outputs required by the paper replay:

- `artifacts/ocv_table.csv`
- `artifacts/ecm_parameter_map.csv`
- `v2_1/ecm_fit_quality.csv`
- `v2_1/r0_estimates.csv`
- `v2_1/qr_selection.csv`

The v2.1 ECM/R0 tables are fold-specific training-profile fits.
`ecm_parameter_map.csv` is the exact v2 fallback/anchor table consumed by the
preserved interpolation class and does not contain a held-out-profile fit.
