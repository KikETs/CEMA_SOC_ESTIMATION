#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from dataclasses import replace
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from Scripts import run_confirmatory_minipanel as panel

OUT=ROOT/'posthoc_5seed_precision'
RUNS=OUT/'runs'
FEATURES=('T6','T7','G0')
SEEDS=(3,4)

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def marker(cfg):
 return cfg.base_dir/'nmc_goal_vcorr_it_train_dst_selector_results'/f'{cfg.output_prefix}_by_temperature.csv'

def main():
 args=argparse.Namespace(epochs=200,batch_size=2048)
 panel.ARGS=args
 OUT.mkdir(exist_ok=True)
 protected=[ROOT/'TRANSFER_PROTOCOL.md',ROOT/'manifests/locked_tier1_decision.json',ROOT/'validation_exports/tier1_paired_seed_ci.csv',ROOT/'validation_exports/tier1_seed_slice_unweighted_mae.csv']
 before={str(p):sha(p) for p in protected}
 jobs=[]
 requested=os.environ.get('LFP_POSTHOC_CONDITIONS','all8,core4').split(',')
 conditions=[c for c in panel.CONDITIONS if ('all8' in requested and c.name.startswith('all8')) or ('core4' in requested and c.name.startswith('core4'))]
 for condition in conditions:
  for feature in FEATURES:
   for holdout in panel.PROFILES:
    cfg=panel.make_cfg(condition,'tier1',holdout,feature,'gru','residual',SEEDS,args)
    base=RUNS/condition.name/'tier1_posthoc_5seed'
    prefix=f'lfpconfirm_{condition.name}_posthoc_5seed_gru_residual_{feature.lower()}_holdout{holdout.lower()}_s34_b2048_e200'
    cfg=replace(cfg,base_dir=base,output_prefix=prefix)
    m=marker(cfg)
    row={'analysis_stage':'posthoc_5seed','condition':condition.name,'feature':feature,'holdout':holdout,'seeds':[3,4],'marker':str(m),'started_unix':time.time()}
    if m.exists(): row['status']='skipped_exact_marker'
    else:
     panel.clear_caches(); panel.selector.run(cfg)
     if not m.exists(): raise RuntimeError(f'missing marker {m}')
     row['status']='complete'
    row['completed_unix']=time.time(); jobs.append(row)
    manifest_name='job_manifest_core4.json' if requested==['core4'] else 'job_manifest.json'
    (OUT/manifest_name).write_text(json.dumps({'analysis_stage':'posthoc_5seed','jobs':jobs},indent=2)+'\n')
 after={str(p):sha(p) for p in protected}
 if before!=after: raise RuntimeError('preregistered artifact hash changed')
 guard_name='preregistration_guard_core4.json' if requested==['core4'] else 'preregistration_guard.json'
 (OUT/guard_name).write_text(json.dumps({'analysis_stage':'posthoc_5seed','protected_hashes_before':before,'protected_hashes_after':after,'unchanged':True,'new_training_only':{'features':list(FEATURES),'seeds':list(SEEDS),'conditions':[c.name for c in conditions],'holdouts':list(panel.PROFILES),'individual_runs':18 if requested==['core4'] else 36}},indent=2)+'\n')

if __name__=='__main__': main()
