# NMC 3-LOPO ECM-KF baselines

This isolated workspace evaluates CC, 1RC-EKF, 2RC-EKF, 2RC-UKF, and an adaptive 2RC-EKF on the locked DST/FUDS/US06 3-LOPO protocol. The source neural repositories and their result files are read-only inputs.

## Environment

- Python 3
- numpy, pandas, scipy, scikit-learn, numba, matplotlib, PyYAML, openpyxl, xlrd, torch

## Run

```bash
cd /home/user/바탕화면/DL/NMC_KF_3LOPO_BASELINES
python run_all_lopo.py 2>&1 | tee logs/run_all_lopo.log
```

The command rebuilds all CSV files, figures, `audit.md`, `leakage_audit.json`, and `report.md`. Configuration is fixed in `configs/protocol.yaml` and `configs/ecm.yaml`.

## Leakage boundary

For each fold, independent low-current characterization supplies OCV and capacity. ECM fitting, parameter-bound selection, Q/R selection, and adaptive-R selection receive only the two training profiles. Held-out files are passed only to final filter evaluation. `leakage_audit.json` and `results/fit_file_manifest.csv` record the paths used by every fitting stage.
