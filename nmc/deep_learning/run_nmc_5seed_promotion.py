#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import pandas as pd

from Scripts import run_ocvstart_3lopo_fullgrid as grid
from soc_decomp.nmc_vcorr_it_train_dst_selector_run import run

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"nmc_5seed_promotion"
OLD=ROOT/"nmc_goal_vcorr_it_train_dst_selector_results"
RAW=ROOT/"nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean"
HOLDS=("DST","FUDS","US06")

def job(h): return grid.Job("baseline_model","gru","residual","anchor_residual_sequence","gru","paper_g4_all_ema",h)

def main():
    OUT.mkdir(exist_ok=True); (OUT/"nmc_goal_vcorr_it_train_dst_selector_results").mkdir(exist_ok=True)
    args=argparse.Namespace(base_dir=str(OUT),raw_root=str(RAW),seeds="3,4",epochs=200,batch_size=2048,include_train_final_eval=False)
    manifest=[]
    for h in HOLDS:
        j=job(h); prefix=grid.prefix_for(j,(3,4),200,2048,"nmc5seedpromo")
        cfg=grid.cfg_for_job(j,args,prefix); marker=OUT/"nmc_goal_vcorr_it_train_dst_selector_results"/f"{prefix}_test_summary.csv"
        manifest.append({"holdout":h,"seeds":[3,4],"feature_set":cfg.feature_set,"model_kind":cfg.model_kind,"recurrent":cfg.recurrent,"epochs":cfg.epochs,"batch_size":cfg.batch_size,"marker":str(marker)})
        if marker.is_file(): print("SKIP",prefix,flush=True)
        else: print("START",prefix,flush=True); run(cfg); print("DONE",prefix,flush=True)
    (OUT/"promotion_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    aggregate()

def selected(path):
    d=pd.read_csv(path); z=d[d.variant.astype(str).str.contains("selected")] if "variant" in d else d
    return z if len(z) else d

def aggregate():
    rows=[]
    for h in HOLDS:
        old=selected(next(OLD.glob(f"s3_3l_bm_gru_g4_r_{h.lower()}_s012_b2048_e200_test_summary.csv")))
        new=selected(next((OUT/"nmc_goal_vcorr_it_train_dst_selector_results").glob(f"nmc5seedpromo_3l_bm_gru_g4_r_{h.lower()}_s34_b2048_e200_test_summary.csv")))
        for d,source in ((old,"existing_seeds012"),(new,"new_seeds34")):
            for _,r in d.iterrows():
                for temp in (0.,25.,45.): rows.append({"seed":int(r.seed),"holdout":h,"temperature_C":temp,"MAE_pct":float(r[str(temp)]),"source":source})
    out=pd.DataFrame(rows).drop_duplicates(["seed","holdout","temperature_C"])
    if len(out)!=45 or set(out.seed)!={0,1,2,3,4}: raise RuntimeError(f"bad promotion coverage {out.shape} seeds={set(out.seed)}")
    out.to_csv(OUT/"g4_gru_residual_5seed_slice_rows.csv",index=False)
    seed=out.groupby("seed",as_index=False).agg(MAE_pct=("MAE_pct","mean"),n_slices=("MAE_pct","size"))
    agg=pd.DataFrame([{"model":"G4_GRU_residual","feature_set":"paper_g4_all_ema","seeds":"0,1,2,3,4","n_seeds":5,"n_slices_per_seed":9,"slice_unweighted_MAE_pct":seed.MAE_pct.mean(),"seed_SD_pct":seed.MAE_pct.std(ddof=1)}])
    agg.to_csv(OUT/"g4_gru_residual_5seed_aggregate.csv",index=False); seed.to_csv(OUT/"g4_gru_residual_5seed_by_seed.csv",index=False)
    print(agg.to_string(index=False),flush=True)

if __name__=="__main__": main()
