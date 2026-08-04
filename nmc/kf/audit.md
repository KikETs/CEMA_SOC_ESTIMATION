# Repository Audit

## Locked protocol

- Profiles: DST, FUDS, US06 in the 3-LOPO experiment.
- Folds: DST+FUDS -> US06; FUDS+US06 -> DST; DST+US06 -> FUDS.
- Temperatures: 0, 25, 45 C.
- Dynamic inputs: terminal voltage and current. Temperature is the nominal file label; no measured temperature channel exists.
- Source current uses discharge-negative sign. Filters convert to discharge-positive current.
- Time propagation uses consecutive Test_Time(s) differences.
- Reference SOC is SOC_CC in fraction units.
- Shared evaluation mask is every endpoint from index 49 through the final row, matching existing neural prediction files.
- Early 60 s and 300 s metrics are measured from the first shared evaluation endpoint (index 49).
- No extra cutoff, first80/last20 validation, or early stopping is applied.
- KF inputs are not normalized and no scaler is fitted.
- OCV is a monotone isotonic + PCHIP SOC map. At the three observed temperatures, exact temperature maps are used; configured interpolation for unseen in-range temperatures is linear in temperature, with positive resistance and log-time-constant interpolation.
- ECM parameters are identified separately at 0, 25, and 45 C from training profiles only. No test-temperature trajectory is used to form a parameter map.

## Critical label provenance

SOC_CC was generated from OCV-inferred initial SOC and temperature-specific low-current discharge capacity followed by current integration. Oracle-initialized CC therefore shares the reference-label construction equation and is not an information-equivalent comparison to the neural model.

## Characterization

Independent SP20-1 low-current OCV files exist at 0/25/45 C. OCV and capacity use those characterization files; R0/R1/C1/R2/C2 are identified from training profiles only.

|   temperature_C | source_file                                                                                    |   q_ref_Ah | selection                                                |   raw_points |   processed_points |
|----------------:|:-----------------------------------------------------------------------------------------------|-----------:|:---------------------------------------------------------|-------------:|-------------------:|
|               0 | /home/user/바탕화면/DL/CEMA-TCN/Data/raw_reference/02_24_2016_SP20-1_0C_lowcurrentOCV.xls      |    1.88167 | largest negative-current step; capacity span=1.881640 Ah |        67153 |                202 |
|              25 | /home/user/바탕화면/DL/CEMA-TCN/Data/raw_reference/11_5_2015_low current OCV test_SP20-1.xlsx  |    2.15687 | longest negative-current segment; rows=77976             |        77976 |                202 |
|              45 | /home/user/바탕화면/DL/CEMA-TCN/Data/raw_reference/11_21_2015_low current OCV test_SP20-1.xlsx |    2.02217 | longest negative-current segment; rows=73146             |        73146 |                202 |

## Dynamic data inventory

| file                                                                                                                           | profile   |   temperature_C |   rows |   dt_median_s |   dt_min_s |   dt_max_s |   voltage_min_V |   voltage_max_V |   source_current_min_A |   source_current_max_A |   reference_soc_start_pct |   reference_soc_end_pct |   Q_ref_lc_ocv_Ah |   Qnet_removed_end_Ah | source_soc_unit   | sha256                                                           |
|:-------------------------------------------------------------------------------------------------------------------------------|:----------|----------------:|-------:|--------------:|-----------:|-----------:|----------------:|----------------:|-----------------------:|-----------------------:|--------------------------:|------------------------:|------------------:|----------------------:|:------------------|:-----------------------------------------------------------------|
| /home/user/바탕화면/DL/CEMA_MLP_OCVSTART_FULLGRID/nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean/0C/NMC_0C_DST.csv    | DST       |               0 |   9527 |       1.01551 |   0.124903 |    1.01756 |         2.49899 |         4.11729 |               -4.00048 |               1.99968  |                   85.1973 |                 9.65021 |           1.88167 |               1.42155 | fraction          | 4b777cce3ec960281180f656e4f66c070355f1d47ca4deb282afeb5623afd842 |
| /home/user/바탕화면/DL/CEMA_MLP_OCVSTART_FULLGRID/nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean/0C/NMC_0C_FUDS.csv   | FUDS      |               0 |   9707 |       1.01549 |   0.14064  |    1.01828 |         2.49948 |         4.15163 |               -4.00048 |               2.1413   |                   85.0562 |                11.1061  |           1.88167 |               1.3915  | fraction          | b858aa2800df0c7395a90856d13e2d52a2eaba362eebf26de8d221ec1b02848a |
| /home/user/바탕화면/DL/CEMA_MLP_OCVSTART_FULLGRID/nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean/0C/NMC_0C_US06.csv   | US06      |               0 |   9482 |       1.01554 |   0.0937   |    1.01831 |         2.49948 |         4.01928 |               -3.99634 |               0.858814 |                   85.0714 |                 7.14417 |           1.88167 |               1.46633 | fraction          | 46ebea30d8185b20962f16c748f397fd391ac9ad5dd096d1d27e9c850360d612 |
| /home/user/바탕화면/DL/CEMA_MLP_OCVSTART_FULLGRID/nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean/25C/NMC_25C_DST.csv  | DST       |              25 |  10621 |       1.01554 |   0.15613  |    1.03218 |         2.40337 |         4.05493 |               -4.00196 |               2.00113  |                   83.5712 |                 9.56001 |           2.15687 |               1.59633 | fraction          | 28db046b6f290450ad3e0e45d6b51d01c5826d2196758680adff5b55655e380a |
| /home/user/바탕화면/DL/CEMA_MLP_OCVSTART_FULLGRID/nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean/25C/NMC_25C_FUDS.csv | FUDS      |              25 |  11092 |       1.01554 |   0.07813  |    1.03488 |         2.49678 |         4.07694 |               -4.00033 |               2.14219  |                   83.6132 |                 9.42323 |           2.15687 |               1.60018 | fraction          | ef4d47e12c45a29e3e3f39a7d21efcce2e0bdbe1b4947e3d32a6cb6e8c944071 |
| /home/user/바탕화면/DL/CEMA_MLP_OCVSTART_FULLGRID/nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean/25C/NMC_25C_US06.csv | US06      |              25 |  10680 |       1.01541 |   0.0623   |    1.03221 |         2.49823 |         3.98694 |               -3.99654 |               0.85932  |                   83.5922 |                 7.15585 |           2.15687 |               1.64863 | fraction          | 0b46a9f471beda42ae8e94520078d166695c23e681bd4a7fffd80d10789b5bc2 |
| /home/user/바탕화면/DL/CEMA_MLP_OCVSTART_FULLGRID/nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean/45C/NMC_45C_DST.csv  | DST       |              45 |  11304 |       1.01554 |   0.03129  |    1.61561 |         2.49553 |         4.08583 |               -3.99981 |               2.00024  |                   85.491  |                 2.45899 |           2.02217 |               1.67905 | fraction          | 6ec8f99d3bee331efd6eebcfb9328d6bbae8f1b0fd3fc4b1ae73c708fda5ee45 |
| /home/user/바탕화면/DL/CEMA_MLP_OCVSTART_FULLGRID/nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean/45C/NMC_45C_FUDS.csv | FUDS      |              45 |  11626 |       1.01556 |   0.01545  |    1.03344 |         2.49844 |         4.10317 |               -3.99981 |               2.14176  |                   85.7022 |                 2.55625 |           2.02217 |               1.68135 | fraction          | 53105b6f99f2ab0a5644929d5ce28ceffc3d2d872fe78f1cb8cf1f9d915be239 |
| /home/user/바탕화면/DL/CEMA_MLP_OCVSTART_FULLGRID/nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean/45C/NMC_45C_US06.csv | US06      |              45 |  10884 |       1.01557 |   0.04684  |    1.03061 |         2.49925 |         4.0022  |               -3.99567 |               0.859105 |                   85.7315 |                 2.61531 |           2.02217 |               1.68075 | fraction          | ba6ca20ca6a375cab32810418ce2dac00253d56ff5e7b7b51f178c110c20f0d7 |
