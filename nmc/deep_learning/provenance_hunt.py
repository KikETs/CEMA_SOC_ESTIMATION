#!/usr/bin/env python3
from pathlib import Path
import shutil, hashlib
import pandas as pd
import numpy as np

ROOTS=[Path('/home/user/바탕화면/DL/CEMA_MLP_OCVSTART_FULLGRID'),Path('/home/user/바탕화면/DL/CEMA_LFP')]
OUT=Path('/home/user/바탕화면/DL/CEMA_MLP_OCVSTART_FULLGRID/provenance_hunt'); OUT.mkdir(exist_ok=True)
targets={'NMC_GRU_G4_overall':.324,'NMC_DST':.299,'NMC_FUDS':.326,'NMC_US06':.346,'NMC_T6':.303,
'NMC_marg_T6':.313,'NMC_marg_G4':.341,'NMC_marg_G8':.413,'NMC_marg_G1':.515,'NMC_marg_G0':.588,'NMC_head_normal':.527,'NMC_MLP':.327,
'LFP_US06_0_T6':.910,'LFP_US06_0_G4':.626,'LFP_FUDS_n10_T6':1.584,'LFP_FUDS_n10_G4':1.000,'LFP_US06_n10_G4':1.729,'LFP_US06_n10_T6':2.065,
'LFP_T6_seed0':2.117,'LFP_T6_seed1':1.875,'LFP_T6_seed2':2.201}
cands=[]
for root in ROOTS:
 for p in root.rglob('*.csv'):
  if not any(x in p.name.lower() or x in str(p.parent).lower() for x in ('nmc_goal','lfpconfirm','rank','summary','holdout_temperature')): continue
  try:
   d=pd.read_csv(p); nums=d.select_dtypes(include='number').to_numpy().ravel(); nums=nums[np.isfinite(nums)]
  except Exception: continue
  hits=[k for k,v in targets.items() if np.any(np.abs(nums-v)<=.001)]
  if hits: cands.append((p,p.stat().st_mtime,hits))
cands.sort(key=lambda x:x[1],reverse=True)
selected=[]
for p,mt,hits in cands:
 if len(hits)<2 and not any(k.startswith('LFP_') for k in hits): continue
 key=hashlib.sha256(str(p).encode()).hexdigest()[:10]
 dest=OUT/f'{key}_{p.name}'; shutil.copy2(p,dest); selected.append((p,mt,hits,dest))
lines=['# Provenance hunt report','','Search-only audit; no experiment was run or modified.','', '| candidate file | mtime_unix | reproduced keys within ±0.001 | copied as |','|---|---:|---|---|']
for p,mt,hits,dest in selected: lines.append(f'| `{p}` | {mt:.3f} | {", ".join(hits)} | `{dest.name}` |')
found={k for _,_,h,_ in selected for k in h}; missing=[k for k in targets if k not in found]
lines += ['','## Not reproduced from a qualifying extant export','',*(f'- {k}: target {targets[k]}' for k in missing)]
if missing: lines += ['','Conclusion: at least some requested snapshot keys are not jointly recoverable from extant qualifying exports; overwrite or rerun loss is possible but not proven.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('candidates',len(cands),'copied',len(selected),'missing',len(missing))
