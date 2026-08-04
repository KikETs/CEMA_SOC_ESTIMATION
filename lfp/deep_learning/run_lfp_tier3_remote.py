#!/usr/bin/env python3
from pathlib import Path
import run_nmc_robustness as r

ROOT=Path(__file__).resolve().parent
r.HERE=ROOT
r.PACKAGE_ROOT=ROOT/'inference_pkg_lfp'
r.OUTPUT_ROOT=ROOT/'tier3_inference_only'
r.FEATURES=('T6','G4')
r.FOLDS=('DST','FUDS','US06')
r.SEEDS=(0,1,2)
r.R0_SCALES=(0.5,0.8,1.2,1.5)
r.V_NOISE_MV=(1,2,5,10)
r.V_BIAS_MV=(-10,-5,-2,-1,1,2,5,10)
r.I_NOISE_PCT=()
r.I_BIAS_MA=()
def assert_precondition():
  import pandas as pd
  g=pd.read_csv(r.PACKAGE_ROOT/'golden_test_results.csv')
  if len(g)!=27 or not g.passed.astype(bool).all(): raise RuntimeError('all 27 LFP golden tests must pass')
  return r.tree_hash(r.PACKAGE_ROOT)
r.assert_precondition=assert_precondition
if __name__=='__main__':
  r.main()
