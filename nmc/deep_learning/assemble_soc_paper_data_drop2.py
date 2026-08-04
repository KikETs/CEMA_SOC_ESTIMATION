#!/usr/bin/env python3
from pathlib import Path
import shutil, subprocess, hashlib, datetime, os

HOME=Path.home(); DROP=HOME/'soc_paper_data_drop2'; LOCK=DROP/'data/locked'
if DROP.exists(): raise SystemExit(f'refuse to overwrite existing {DROP}')
for rel in ['robustness_results_lfp','inference_pkg_lfp','NMC_KF_3LOPO_BASELINES/results_v2_sliceunweighted','CEMA_MLP_OCVSTART_FULLGRID/nmc_5seed_promotion','LFP_KF_LOPO_BASELINES_ISOLATED/source_exports','provenance_hunt']:
 (LOCK/rel).mkdir(parents=True,exist_ok=True)
remote='lab@100.121.61.51:/home/lab/바탕화면/DL/CEMA_LFP/lfp_confirmatory_minipanel/'
subprocess.run(['rsync','-a',remote+'tier3_inference_only/',str(LOCK/'robustness_results_lfp')+'/'],check=True)
subprocess.run(['rsync','-a',remote+'inference_pkg_lfp/',str(LOCK/'inference_pkg_lfp')+'/'],check=True)
copies=[
 (Path('/home/user/바탕화면/DL/NMC_KF_3LOPO_BASELINES/results_v2_sliceunweighted'),LOCK/'NMC_KF_3LOPO_BASELINES/results_v2_sliceunweighted'),
 (Path('/home/user/바탕화면/DL/CEMA_MLP_OCVSTART_FULLGRID/nmc_5seed_promotion'),LOCK/'CEMA_MLP_OCVSTART_FULLGRID/nmc_5seed_promotion'),
 (Path('/home/user/바탕화면/LFP_KF_LOPO_BASELINES_ISOLATED_v2_2_WORK/source_exports'),LOCK/'LFP_KF_LOPO_BASELINES_ISOLATED/source_exports'),
 (Path('/home/user/바탕화면/DL/CEMA_MLP_OCVSTART_FULLGRID/provenance_hunt_clean'),LOCK/'provenance_hunt')]
for src,dst in copies:
 for p in src.rglob('*'):
  if p.is_file(): q=dst/p.relative_to(src); q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,q)

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
files=sorted(p for p in DROP.rglob('*') if p.is_file())
headers=[]; sums=[]; rows=[]
def source_for(rel):
 s=str(rel)
 maps={
 'data/locked/robustness_results_lfp/':'/home/lab/바탕화면/DL/CEMA_LFP/lfp_confirmatory_minipanel/tier3_inference_only/',
 'data/locked/inference_pkg_lfp/':'/home/lab/바탕화면/DL/CEMA_LFP/lfp_confirmatory_minipanel/inference_pkg_lfp/',
 'data/locked/NMC_KF_3LOPO_BASELINES/results_v2_sliceunweighted/':'/home/user/바탕화면/DL/NMC_KF_3LOPO_BASELINES/results_v2_sliceunweighted/',
 'data/locked/CEMA_MLP_OCVSTART_FULLGRID/nmc_5seed_promotion/':'/home/user/바탕화면/DL/CEMA_MLP_OCVSTART_FULLGRID/nmc_5seed_promotion/',
 'data/locked/LFP_KF_LOPO_BASELINES_ISOLATED/source_exports/':'/home/user/바탕화면/LFP_KF_LOPO_BASELINES_ISOLATED_v2_2_WORK/source_exports/',
 'data/locked/provenance_hunt/':'/home/user/바탕화면/DL/CEMA_MLP_OCVSTART_FULLGRID/provenance_hunt_clean/'}
 for prefix,base in maps.items():
  if s.startswith(prefix): return base+s[len(prefix):]
 return 'generated in drop root'
for p in files:
 rel=p.relative_to(DROP); st=p.stat(); digest=sha(p); sums.append(f'{digest}  {rel}')
 origin=source_for(rel)
 rows.append(f'| `{rel}` | generated/copy | `{origin}` | {st.st_size} | {datetime.datetime.fromtimestamp(st.st_mtime).isoformat()} | `{digest}` | additive drop2 artifact |')
 if p.suffix.lower()=='.csv':
  with p.open(errors='replace') as f: headers += [f'===== {rel} =====']+[next(f,'').rstrip('\n') for _ in range(3)]+['']
(DROP/'headers.txt').write_text('\n'.join(headers)+'\n')
(DROP/'sha256sums.txt').write_text('\n'.join(sums)+'\n')
missing=[]
for rel in ['data/locked/robustness_results_lfp/SUMMARY.md','data/locked/inference_pkg_lfp/manifest.json','data/locked/NMC_KF_3LOPO_BASELINES/results_v2_sliceunweighted/main_table_sliceunweighted.csv','data/locked/CEMA_MLP_OCVSTART_FULLGRID/nmc_5seed_promotion/g4_gru_residual_5seed_aggregate.csv']:
 if not (DROP/rel).exists(): missing.append(rel)
missing += ['LFP inference MAC count: unavailable (latency/parameter benchmark exported)',
            'US06 held-out minus5 pp residual 1.86 pp: exact numeric source export unavailable']
manifest=['# SOC paper data drop 2 manifest','', '| relative path | source/generated | absolute source | size | mtime | sha256 | note |','|---|---|---|---:|---|---|---|',*rows,'','## MISSING',*(f'- `{x}`' for x in missing)]
if not missing: manifest.append('- None.')
manifest += ['','## AMBIGUOUS','- US06 held-out minus5 pp residual 1.86 pp: numeric source export not found; see source_exports audit.','- Expected plateau slope 0.156 mV/%SOC has a factor-10 unit mismatch; correct and alternate conventions are both exported without forcing the expected value.']
(DROP/'MANIFEST.md').write_text('\n'.join(manifest)+'\n')
print({'files':len(files),'bytes':sum(p.stat().st_size for p in files),'missing':missing})
