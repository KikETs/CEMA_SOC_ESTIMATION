#!/usr/bin/env python3
from pathlib import Path
import json, time
import pandas as pd
import numpy as np
import torch
from inference_pkg_lfp import load_package

ROOT=Path(__file__).resolve().parent
PKG=ROOT/'inference_pkg_lfp'
rows=[]
bench=[]
for feature in ('T6','T7','G4'):
  for fold in ('DST','FUDS','US06'):
    for seed in (0,1,2):
      p=load_package(PKG/feature/fold/str(seed)); m=p.manifest
      src=Path(m['source_repo']); arc=pd.read_csv(src/m['archived_predictions'])
      deltas=[]; elapsed=[]; samples=0
      for rel in m['evaluation_records']:
        f=pd.read_csv(src/rel); exp=arc[arc.file_name.eq(Path(rel).name)].sort_values('end_index')
        t=time.perf_counter(); pred=p.predict(f); elapsed.append(time.perf_counter()-t); samples += len(f)
        idx=exp.end_index.to_numpy(int)
        deltas.append(float(np.max(np.abs(pred[idx].astype(float)-exp.y_pred.to_numpy(float)))))
      nparams=sum(x.numel() for x in p.model.parameters())
      rows.append({'feature':feature,'fold':fold,'seed':seed,'max_abs_delta':max(deltas),'passed':max(deltas)<=1e-6,'deterministic_bit_identical':max(deltas)==0.0})
      bench.append({'feature':feature,'fold':fold,'seed':seed,'parameter_count':nparams,'records':len(m['evaluation_records']),'samples':samples,'wall_seconds':sum(elapsed),'us_per_input_sample':1e6*sum(elapsed)/samples,'MAC_per_estimate':'MISSING_not_profiled'})
pd.DataFrame(rows).to_csv(PKG/'golden_test_results.csv',index=False)
pd.DataFrame(bench).to_csv(PKG/'inference_benchmark.csv',index=False)
(PKG/'validation_summary.json').write_text(json.dumps({'all_passed':all(x['passed'] for x in rows),'entries':len(rows),'note':'MAC unavailable; CPU wall benchmark includes feature regeneration and batched window inference.'},indent=2)+'\n')
print(pd.DataFrame(rows).groupby('feature').max_abs_delta.max())
