#!/usr/bin/env python3
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT=Path('/home/lab/바탕화면/DL/CEMA_LFP/lfp_confirmatory_minipanel')
POST=ROOT/'posthoc_5seed_precision'
OUT=POST/'validation_exports_5seed'; OUT.mkdir(parents=True,exist_ok=True)
CONDS=('all8_train_all8_test','core4_train_all8_test'); FEATURES=('G4','T6','T7','G0'); HOLDS=('DST','FUDS','US06')

def summary_path(cond,feat,hold,seed):
 if seed<=2:
  d=ROOT/f'runs/{cond}/tier1/nmc_goal_vcorr_it_train_dst_selector_results'
  name=f'lfpconfirm_{cond}_tier1_gru_residual_{feat.lower()}_holdout{hold.lower()}_s012_b2048_e200_test_summary.csv'
 elif feat=='G4':
  d=ROOT/f'runs/{cond}/winner_promotion/nmc_goal_vcorr_it_train_dst_selector_results'
  name=f'lfpconfirm_{cond}_winner_promotion_gru_residual_g4_holdout{hold.lower()}_s34_b2048_e200_test_summary.csv'
 else:
  d=POST/f'runs/{cond}/tier1_posthoc_5seed/nmc_goal_vcorr_it_train_dst_selector_results'
  name=f'lfpconfirm_{cond}_posthoc_5seed_gru_residual_{feat.lower()}_holdout{hold.lower()}_s34_b2048_e200_test_summary.csv'
 return d/name

rows=[]
for cond in CONDS:
 for feat in FEATURES:
  for hold in HOLDS:
   for seed in range(5):
    p=summary_path(cond,feat,hold,seed)
    if not p.exists(): raise RuntimeError(f'missing {p}')
    d=pd.read_csv(p); d=d[d.variant.astype(str).str.endswith('_selector_base') & d.seed.eq(seed)]
    if len(d)!=1: raise RuntimeError(f'expected one selector row: {p}, seed={seed}, got={len(d)}')
    for temp in (-10,0,10,20,25,30,40,50):
     rows.append({'analysis_stage':'posthoc_5seed','condition':cond,'tier':'tier1_plus_posthoc_precision','architecture':'gru','head':'residual','feature':feat,'holdout':hold,'seed':seed,'temperature_C':float(temp),'MAE_pct':float(d[str(float(temp))].iloc[0]),'source_file':str(p)})
slice_rows=pd.DataFrame(rows).sort_values(['condition','feature','holdout','seed','temperature_C'])
slice_rows.to_csv(OUT/'seed_holdout_temperature_mae_5seed.csv',index=False)
scores=(slice_rows.groupby(['condition','feature','seed'],as_index=False).MAE_pct.mean().rename(columns={'MAE_pct':'slice_unweighted_MAE_pct'}))
scores.insert(0,'analysis_stage','posthoc_5seed'); scores.to_csv(OUT/'tier1_seed_slice_unweighted_mae_5seed.csv',index=False)

ci=[]
for cond in CONDS:
 p=scores[scores.condition.eq(cond)].pivot(index='seed',columns='feature',values='slice_unweighted_MAE_pct')
 for comp in ('G4','T7','G0'):
  x=(p[comp]-p.T6).to_numpy(float); se=x.std(ddof=1)/math.sqrt(len(x)); q=student_t.ppf(.975,len(x)-1)
  ci.append({'analysis_stage':'posthoc_5seed','condition':cond,'contrast':f'{comp}-T6','n_pairs':len(x),'mean_delta_pct':x.mean(),'ci95_low_pct':x.mean()-q*se,'ci95_high_pct':x.mean()+q*se})
pd.DataFrame(ci).to_csv(OUT/'tier1_paired_seed_ci_5seed.csv',index=False)

expected={'G4':.535,'T6':.620,'T7':.796,'G0':2.411}
checks={}
for feat,v in expected.items():
 got=float(scores[(scores.condition==CONDS[0])&(scores.feature==feat)&(scores.seed<=2)].slice_unweighted_MAE_pct.mean())
 checks[f'all8_seed012_{feat}']={'observed':got,'expected':v,'pass':abs(got-v)<=.001}
g4=float(scores[(scores.condition==CONDS[0])&(scores.feature=='G4')].slice_unweighted_MAE_pct.mean())
checks['all8_G4_5seed']={'observed':g4,'expected':.5416,'pass':abs(g4-.5416)<=.001}
if not all(x['pass'] for x in checks.values()): raise RuntimeError(checks)
guard=json.loads((POST/'preregistration_guard.json').read_text())
(OUT/'validation.json').write_text(json.dumps({'analysis_stage':'posthoc_5seed','checks':checks,'preregistration_guard':guard},indent=2)+'\n')
print(scores.groupby(['condition','feature']).slice_unweighted_MAE_pct.mean())
print(pd.DataFrame(ci).to_string(index=False))
