# LFP data preprocessing

## Headline dataset builder

`prepare_lfp_ocv_discharge_soc.py` is the preprocessing implementation used to
build the 24 LFP records consumed by the confirmatory 3-LOPO experiment.

It uses:

- raw dynamic records for DST, FUDS, and US06 at -10, 0, 10, 20, 25, 30, 40,
  and 50 degC;
- the temperature-matched Step-5 `discharge_ocv` branch;
- `Q_removed=(Discharge-Discharge0)-(Charge-Charge0)`;
- `Q_ref=max(Discharge_Capacity)-min(Discharge_Capacity)` from the Step-5
  discharge branch; and
- `SOC=clip(SOC0_OCV_discharge-Q_removed/Q_ref,0,1)`.

Existing SOC, Q-net, Q-effective, and drive-profile endpoint fields are not
label inputs. Raw dynamic data and generated files remain gitignored.

`prepare_lfp_raw_excel.py` is the raw-workbook adapter invoked before this
builder. Its output formatting preserves the historical mixed-precision CSV
contract: 10-significant-digit time fields, fixed-nine voltage/energy fields,
and the temperature-dependent decimal representation used by the archived
pipeline. `scripts/verify_preprocessing.py` checks all 24 final files against
their locked SHA-256 values, so a parser or serialization drift fails loudly.

Example:

```bash
python lfp/preprocessing/prepare_lfp_ocv_discharge_soc.py \
  --dynamic-root /path/to/LFP_NEW \
  --ocv-root lfp/data/ocv \
  --prepared-root lfp/data/preprocessed/prepared_data_ocv_discharge_soc \
  --manifest-dir lfp/data/preprocessed/manifests_ocv_discharge_3lopo \
  --force
```

The original source was
`/home/user/바탕화면/DL/CEMA_LFP/lfp_userlabel_validation/prepare_lfp_ocv_discharge_soc.py`.
Only filesystem-path configuration was changed in this repository copy.

## Legacy code

`legacy/prepare_lfp_user_soc.py` uses each profile endpoint as its capacity and
therefore forces each trajectory to end at 0% SOC. It is retained only for
provenance and is not the paper dataset builder. The invalidation notes explain
the two discarded runs.
