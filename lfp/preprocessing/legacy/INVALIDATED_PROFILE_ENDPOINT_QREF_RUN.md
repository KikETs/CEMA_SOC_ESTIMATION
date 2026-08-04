# Invalidated LFP run

The completed prefix below must not be used as an OCV-discharge SOC result:

`lfp_reversesoc_capacitycounter_train_dst_us06_test_fuds_all8temp_matched1s_g4eqdyn_gruresidual_seeds012_b2048_e200`

Reason: it used each drive profile's final `Q_removed` as its own denominator,
which forced every trajectory to 100% -> 0%. The requested definition instead
uses the temperature-matched low-current OCV Step-5 discharge capacity and the
same discharge curve for initial-SOC inversion.

The invalidated files are preserved only for provenance and are not overwritten.
