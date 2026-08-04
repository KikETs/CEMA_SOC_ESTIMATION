# Required local battery data

Download the files from the [University of Maryland CALCE Battery Data
page](https://calce.umd.edu/battery-data). The repository does not redistribute
the workbooks. ZIP archives, extracted workbooks, and generated records are
ignored by Git.

The complete download set is:

- NMC: 12 archives (3 low-current OCV + 9 SP2 dynamic-profile archives)
- LFP: 16 archives (8 A123 low-current OCV + 8 A123 dynamic-profile archives)

Do not rename the selected extracted workbooks. The preprocessing code uses the
exact names listed below. SHA-256 values for all 28 selected workbooks are in
[`../reference/raw_source_hashes.csv`](../reference/raw_source_hashes.csv).
Access and local-acquisition dates are recorded in
[`PROVENANCE.md`](PROVENANCE.md).

## NMC: CALCE SP1/SP2

Download these three SP1 low-current OCV archives:

| Temperature | Official archive | Required extracted workbook |
|---:|---|---|
| 0 C | [SP1_0C_LC_OCV_02_24_2016.zip](https://web.calce.umd.edu/batteries/data/SP1_0C_LC_OCV_02_24_2016.zip) | `02_24_2016_SP20-1_0C_lowcurrentOCV.xls` |
| 25 C | [SP1_25C_LC_OCV_11_5_2015.zip](https://web.calce.umd.edu/batteries/data/SP1_25C_LC_OCV_11_5_2015.zip) | `11_5_2015_low current OCV test_SP20-1.xlsx` |
| 45 C | [SP1_45C_LC_OCV_11_21_2015.zip](https://web.calce.umd.edu/batteries/data/SP1_45C_LC_OCV_11_21_2015.zip) | `11_21_2015_low current OCV test_SP20-1.xlsx` |

Download all nine SP2 dynamic-profile archives:

| Temperature | DST | FUDS | US06 |
|---:|---|---|---|
| 0 C | [SP2_0C_DST.zip](https://web.calce.umd.edu/batteries/data/SP2_0C_DST.zip) | [SP2_0C_FUDS.zip](https://web.calce.umd.edu/batteries/data/SP2_0C_FUDS.zip) | [SP2_0C_US06.zip](https://web.calce.umd.edu/batteries/data/SP2_0C_US06.zip) |
| 25 C | [SP2_25C_DST.zip](https://web.calce.umd.edu/batteries/data/SP2_25C_DST.zip) | [SP2_25C_FUDS.zip](https://web.calce.umd.edu/batteries/data/SP2_25C_FUDS.zip) | [SP2_25C_US06.zip](https://web.calce.umd.edu/batteries/data/SP2_25C_US06.zip) |
| 45 C | [SP2_45C_DST.zip](https://web.calce.umd.edu/batteries/data/SP2_45C_DST.zip) | [SP2_45C_FUDS.zip](https://web.calce.umd.edu/batteries/data/SP2_45C_FUDS.zip) | [SP2_45C_US06.zip](https://web.calce.umd.edu/batteries/data/SP2_45C_US06.zip) |

Each dynamic archive may contain both `50SOC` and `80SOC` workbooks. The paper
pipeline requires and selects only these nine `80SOC` workbooks:

```text
02_24_2016_SP20-2_0C_DST_80SOC.xls
02_25_2016_SP20-2_0C_FUDS_80SOC.xls
02_26_2016_SP20-2_0C_US06_80SOC.xls
11_05_2015_SP20-2_DST_80SOC.xls
11_06_2015_SP20-2_FUDS_80SOC.xls
11_11_2015_SP20-2_US06_80SOC.xls
12_11_2015_SP20-2_45C_DST_80SOC.xls
12_15_2015_SP20-2_45C_FUDS_80SOC.xls
12_16_2015_SP20-2_45C_US06_80SOC.xls
```

Place the three OCV workbooks in `Data/NMC/OCV/` and the nine required profile
workbooks in `Data/NMC/Profiles/`. Extra `50SOC` files may remain in
`Data/NMC/Profiles/`; they are recorded as excluded.

## LFP: CALCE A123 A1-007

Download all eight A123 low-current OCV archives and all eight dynamic-profile
archives:

| Temperature | Official OCV archive | Official profile archive |
|---:|---|---|
| -10 C | [A123_OCV-10-20120629.zip](https://web.calce.umd.edu/batteries/data/A123_OCV-10-20120629.zip) | [A123_DST-US06-FUDS-N10.zip](https://web.calce.umd.edu/batteries/data/A123_DST-US06-FUDS-N10.zip) |
| 0 C | [A123_OCV0-20120618.zip](https://web.calce.umd.edu/batteries/data/A123_OCV0-20120618.zip) | [A123_DST-US06-FUDS-0.zip](https://web.calce.umd.edu/batteries/data/A123_DST-US06-FUDS-0.zip) |
| 10 C | [A123_OCV10-20120611.zip](https://web.calce.umd.edu/batteries/data/A123_OCV10-20120611.zip) | [A123_DST-US06-FUDS-10.zip](https://web.calce.umd.edu/batteries/data/A123_DST-US06-FUDS-10.zip) |
| 20 C | [A123_OCV20-20120614.zip](https://web.calce.umd.edu/batteries/data/A123_OCV20-20120614.zip) | [A123_DST-US06-FUDS-20.zip](https://web.calce.umd.edu/batteries/data/A123_DST-US06-FUDS-20.zip) |
| 25 C | [A123_OCV25-20120905.zip](https://web.calce.umd.edu/batteries/data/A123_OCV25-20120905.zip) | [A123_DST-US06-FUDS-25.zip](https://web.calce.umd.edu/batteries/data/A123_DST-US06-FUDS-25.zip) |
| 30 C | [A123_OCV30-20120625.zip](https://web.calce.umd.edu/batteries/data/A123_OCV30-20120625.zip) | [A123_DST-US06-FUDS-30.zip](https://web.calce.umd.edu/batteries/data/A123_DST-US06-FUDS-30.zip) |
| 40 C | [A123_OCV40-20120627.zip](https://web.calce.umd.edu/batteries/data/A123_OCV40-20120627.zip) | [A123_DST-US06-FUDS-40.zip](https://web.calce.umd.edu/batteries/data/A123_DST-US06-FUDS-40.zip) |
| 50 C | [A123_OCV50-20120702.zip](https://web.calce.umd.edu/batteries/data/A123_OCV50-20120702.zip) | [A123_DST-US06-FUDS-50.zip](https://web.calce.umd.edu/batteries/data/A123_DST-US06-FUDS-50.zip) |

The archives can also contain A1-008 and alternate 20 C exports. This project
uses only these exact 16 A1-007 workbooks:

```text
# Data/LFP/OCV/
A1-007-OCV-10-20120629.xlsx
A1-007-OCV0-20120618.xlsx
A1-007-OCV10-20120611.xlsx
A1-007-OCV20-20120614.xlsx
A1-007-OCV-25-20120905.xlsx
A1-007-OCV30-20120625.xlsx
A1-007-OCV40-20120627.xlsx
A1-007-OCV50-20120702.xlsx

# Data/LFP/Profiles/
A1-007-DST-US06-FUDS-N10-20120829.xlsx
A1-007-DST-US06-FUDS-0-20120813.xlsx
A1-007-DST-US06-FUDS-10-20120815.xlsx
A1-007-DST-US06-FUDS-20-20120817.xlsx
A1-007-DST-US06-FUDS-25-20120827.xlsx
A1-007-DST-US06-FUDS-30-20120820.xlsx
A1-007-DST-US06-FUDS-40-20120822.xlsx
A1-007-DST-US06-FUDS-50-20120824.xlsx
```

Do not substitute `A1-007-DST-US06-FUDS-20-20120817-newprofile.xlsx` or the
older `.xls` export for the required 20 C workbook. A1-008 files may remain
locally but are not read by the paper preprocessing pipeline.

## Final layout and verification

```text
Data/
  NMC/
    OCV/       # 3 required workbooks
    Profiles/  # 12 required 80SOC workbooks; 50SOC extras allowed
  LFP/
    OCV/       # 8 required A1-007 workbooks
    Profiles/  # 8 required A1-007 workbooks
  Preprocessed/  # generated; never commit
```

Run preprocessing and strict source/preprocessed verification:

```bash
python scripts/run_reproduction.py preprocess verify-data
```

The command attempts both chemistries, writes logs under `Data/logs/`, and
writes failures to `Data/Preprocessed/preprocessing_status.csv`. Verification
checks the selected raw-workbook SHA-256 values, expected schemas, record
counts, and derived NMC corrected-voltage channels.
