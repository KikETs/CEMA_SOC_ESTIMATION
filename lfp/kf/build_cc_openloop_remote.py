#!/usr/bin/env python3
from pathlib import Path
import hashlib, json
import pandas as pd

ROOT=Path('/home/lab/바탕화면/LFP_KF_LOPO_BASELINES_ISOLATED')
OUT=ROOT/'results_v2_2/cc_openloop_full'; OUT.mkdir(parents=True,exist_ok=True)
src=ROOT/'results/initial_soc_robustness.csv'; d=pd.read_csv(src)
d=d[d.model.eq('coulomb_count')].copy()
labels={-20:'minus20pp',-10:'minus10pp',-5:'minus5pp',0:'oracle',5:'plus5pp_clipped',10:'plus10pp_clipped',20:'plus20pp_clipped'}
rows=[]
for init,g in d.groupby('initial_soc_perturbation_pct'):
 if int(init) not in labels: continue
 rows.append({'analysis_stage':'inference_only','method':'coulomb_count_open_loop','initial_condition':labels[int(init)],'initial_offset_pp':int(init),
  'slice_unweighted_MAE_pct':g.MAE_pct.mean(),'plotted_median_MAE_pct':g.MAE_pct.median(),'IQR_low_pct':g.MAE_pct.quantile(.25),'IQR_high_pct':g.MAE_pct.quantile(.75),'n_slices':len(g),
  'status':'n/a_clipped_to_100pct' if init>0 else 'complete','source_file':str(src)})
o=pd.DataFrame(rows).sort_values('initial_offset_pp'); o.to_csv(OUT/'cc_openloop_init_sweep.csv',index=False)
oracle=o[o.initial_condition.eq('oracle')].iloc[0]; minus5=o[o.initial_condition.eq('minus5pp')].iloc[0]
checks={'analysis_stage':'inference_only','training_or_fitting_performed':False,'oracle_slice_unweighted_MAE_pct':float(oracle.slice_unweighted_MAE_pct),
 'oracle_expected_approx_0p112':bool(abs(oracle.slice_unweighted_MAE_pct-.112)<.01),'minus5_plotted_median_MAE_pct':float(minus5.plotted_median_MAE_pct),
 'minus5_expected_approx_5p04':bool(abs(minus5.plotted_median_MAE_pct-5.04)<.03),'positive_offsets':'clipped because records start at SOC=100%'}
(OUT/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
(OUT/'README.md').write_text('# CC open-loop initial-condition sweep\n\nRe-aggregation of the frozen open-loop coulomb-counting rows in `results/initial_soc_robustness.csv`; no training or fitting. Positive offsets are retained explicitly as clipped n/a conditions because all records start at 100% SOC.\n')
print(o.to_string(index=False)); print(checks)
