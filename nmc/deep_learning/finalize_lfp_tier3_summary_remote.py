#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd
ROOT=Path('/home/lab/바탕화면/DL/CEMA_LFP/lfp_confirmatory_minipanel'); O=ROOT/'tier3_inference_only'
noise=pd.read_csv(O/'sensor_noise.csv'); bias=pd.read_csv(O/'sensor_bias.csv'); r0=pd.read_csv(O/'r0_perturbation.csv'); cold=pd.read_csv(O/'cold_start.csv')
def clean(d): return d[d.perturbation_type.eq('baseline')].groupby('feature').mae.mean()
c=clean(noise)
sens=pd.read_csv(O/'t6_vs_g4_sensitivity.csv')
v10=sens[(sens.perturbation_type=='voltage_bias') & (sens.level.isin(['V_bias_+10_mV','V_bias_-10_mV']))]
r05=sens[(sens.perturbation_type=='r0_scale') & (sens.level.isin(['r0_scale_0.5','r0_scale_1.5']))]
text=f'''# LFP Tier-3 inference-only robustness summary

## Protocol and clean baseline

- Locked `TRANSFER_PROTOCOL.md`; no training, fitting, R0 estimation, normalization fitting, or checkpoint selection was performed.
- Frozen all8 Tier-1 T6/T7/G4 checkpoints, seeds 0-2, all three held-out profiles and eight temperatures.
- Frozen-package golden check: 27/27 entries passed, maximum archived-prediction deviation below 1e-6.
- Cold-start time uses recorded `Test_Time(s)`. Five deterministic mid-record reset points per record use the NMC suite rule and regenerate causal features.
- Clean macro MAE (SOC-band-stratified rows): T6 {c.get('T6',float('nan')):.4f}, G4 {c.get('G4',float('nan')):.4f} %SOC. T7 is retained only in the predeclared V-I ambiguity local comparison.

## Diagnostics

- Voltage noise and bias include 1, 2, 5, and 10 mV magnitudes; full level-wise results are in `sensor_noise.csv`, `sensor_bias.csv`, and `t6_vs_g4_sensitivity.csv`.
- R0 test-time perturbations include -50%, -20%, +20%, and +50%; full results are in `r0_perturbation.csv`.
- Existing-start and forced-reset cold-start results are separated in `cold_start_existing_record.csv` and `cold_start.csv`/`cold_start_recovery.csv`.
- -10 C SOC-bin by profile decomposition is in `minus10_socbin_profile_bias.csv`.
- Data-only V-I ambiguity statistics and local T6/T7/G4 comparison are in `vi_ambiguity_data_only.csv` and `vi_ambiguity_model_comparison.csv`.
- OCV-discharge SOC0/Qref fields are exported descriptively in `ocv_discharge_label_uncertainty_audit.csv`; no unsupported uncertainty interval was invented.

## Interpretation limits

These are inference perturbations of frozen models, not refits. The label audit quantifies available record-level inputs and ranges; it does not establish independent ground-truth uncertainty where no replicate OCV/capacity measurement exists.
'''
(O/'SUMMARY.md').write_text(text,encoding='utf-8')
manifest=json.loads((O/'run_manifest.json').read_text())
manifest['temperatures_C']=[-10,0,10,20,25,30,40,50]
manifest['executed_perturbations']={'voltage_noise_mV':[1,2,5,10],'voltage_bias_mV':[-10,-5,-2,-1,1,2,5,10],'r0_scale':[0.5,0.8,1.2,1.5],'mid_record_resets_per_record':5}
manifest['excluded_as_out_of_scope']=['current_noise','current_bias']
(O/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
