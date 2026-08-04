# NMC data preprocessing

`prepare_calce_nmc.py` is the CALCE NMC raw-file converter and base OCV-start
SOC label builder used by the CEMA code lineage.

Inputs:

- 9 raw dynamic Excel files: DST, FUDS, and US06 at 0, 25, and 45 degC;
- low-current OCV/reference files under `nmc/data/characterization/raw_reference/`;
- compact SOC0 and capacity provenance tables under
  `nmc/data/characterization/source_tables/`.

Label rule:

```text
SOC0 = inverse_OCV_T(voltage at the end of the preceding rest step)
Q_removed = Q_discharge - Q_charge within the drive segment
SOC = clip(SOC0 - Q_removed / Q_ref_low_current_OCV_T, 0, 1)
```

Run:

```bash
python nmc/preprocessing/prepare_calce_nmc.py \
  --raw-dir /path/to/CALCE_NMC_dynamic_files \
  --reference-dir nmc/data/characterization/raw_reference \
  --out-dir nmc/data/preprocessed/nmc_ocvstart_lopo_clean
```

Raw dynamic and generated preprocessed files are gitignored. The original
source was `/home/user/바탕화면/DL/CEMA-TCN/Data/prepare_calce_nmc.py`; only
repository filesystem paths were changed in this copy.

The paper 3-LOPO dataset applies the later 25 degC US06 SOC0 correction
as a separate, deterministic post-processing step implemented in
`apply_soc0fix25.py`. It replaces that 25 degC initial-SOC value with the
DST/FUDS mean and writes the final dataset consumed by the NMC paper
runner. `scripts/prepare_data.py` executes both stages in order.
