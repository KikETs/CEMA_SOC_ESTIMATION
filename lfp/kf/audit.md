# Repository and data audit

Generated before test-result interpretation. Original repositories are read-only inputs copied into this isolated remote workspace.

## Exact 3-LOPO scope

- Profiles: DST, FUDS, US06.
- Folds: train FUDS+US06/test DST; train DST+US06/test FUDS; train DST+FUDS/test US06.
- Temperatures: -10, 0, 10, 20, 25, 30, 40, 50 degC.
- Prepared trajectories: 24 (3 profiles x 8 temperatures).
- Evaluation mask: exact proposed prediction indices, beginning at end_index=49; mask rows=165316.
- Columns: Test_Time(s), Current(A), Voltage(V), Temperature(C), SOC_CC, Q_ref_lc_ocv_discharge_Ah.
- Reference SOC unit: fraction [0,1]. Capacity unit: Ah.
- Raw current is negative on discharge; internal ECM current is discharge-positive via I=-Current(A).
- Median positive dt: 1.003004000 s.

## Independent characterization

- OCV files: 8 temperatures. Step 5 is low-current discharge (-0.05 A raw); Step 7 is low-current charge (+0.05 A raw).
- OCV base is the charge/discharge center; hysteresis magnitude is their non-negative half-gap.
- No held-out drive profile is used for OCV, ECM, or hysteresis-magnitude identification.

## Rest and hysteresis identifiability

- Dynamic rows with |I|<0.01 A: 33407.
- Dynamic current-sign transitions: 15146.
- Both charge/discharge OCV branches exist, so a constrained one-state hysteresis model is identifiable without inventing a gap.

## Proposed artifacts

- Available prediction rows: 495948 from 3 folds x 3 seeds.
- The available completed model is G4eqdyn GRU-residual, not an EMA-MLP. It is reported under its exact name; the report must not relabel it as MLP.

## Cutoff and mask

No additional KF-only cutoff is applied. Every quantitative comparison uses the exact per-trajectory end_index values from the existing proposed prediction files. Diverged rows remain present as missing predictions and are reported, not deleted.
