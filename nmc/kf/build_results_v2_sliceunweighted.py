#!/usr/bin/env python3
from pathlib import Path
import json, re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"results_v2_sliceunweighted"; OUT.mkdir(exist_ok=True)
PRED=ROOT/"results/predictions"; GRID=Path("/home/user/바탕화면/DL/CEMA_MLP_OCVSTART_FULLGRID/nmc_goal_vcorr_it_train_dst_selector_results")
rows=[]
for f in sorted(PRED.glob("*.csv.gz")):
    d=pd.read_csv(f); d=d[d.eval_mask.astype(bool)]
    if d.empty: continue
    rows.append({"method":str(d.method.iloc[0]),"initial_condition":"oracle","fold":str(d.fold.iloc[0]),"temperature_C":float(d.temperature_C.iloc[0]),"MAE_pct":float(d.abs_error_pct.mean()),"n":len(d),"source_file":str(f)})
slices=pd.DataFrame(rows); slices.to_csv(OUT/"kf_slice_rows.csv",index=False)
wide=slices.pivot_table(index=["method","initial_condition"],columns=["fold","temperature_C"],values="MAE_pct").reset_index()
wide.columns=[c if isinstance(c,str) else (str(c[0]) if c[1]=="" else f"{c[0]}_{float(c[1]):g}C") for c in wide.columns]
slice_cols=[c for c in wide.columns if c not in ("method","initial_condition")]; wide["MAE_pct"]=wide[slice_cols].mean(axis=1); wide["n_slices"]=wide[slice_cols].notna().sum(axis=1)
# Non-oracle initial conditions exist only as frozen per-slice summaries, not prediction rows.
rob=pd.read_csv(ROOT/"results/initial_soc_robustness.csv")
for (method,delta),g in rob.groupby(["method","initial_perturbation_pct_point"]):
    if float(delta)==0: continue
    vals={(str(r.fold),float(r.temperature_C)):float(r.mae_pct) for r in g.itertuples()}
    row={"method":method,"initial_condition":f"{delta:+g}pp","MAE_pct":np.mean(list(vals.values())),"n_slices":len(vals)}
    for (fold,temp),value in vals.items(): row[f"{fold}_{temp:g}C"]=value
    wide=pd.concat([wide,pd.DataFrame([row])],ignore_index=True)
# Official proposed row from exact G4+GRU-residual test summaries, seeds 0..2.
prop=[]
for hold in ("DST","FUDS","US06"):
    f=GRID/f"s3_3l_bm_gru_g4_r_{hold.lower()}_s012_b2048_e200_test_summary.csv"; d=pd.read_csv(f); z=d[d.variant.astype(str).str.contains("selected")]; d=z if len(z) else d
    for _,r in d.iterrows():
        for temp in (0.,25.,45.): prop.append({"seed":int(r.seed),"fold":hold,"temperature_C":temp,"MAE_pct":float(r[str(temp)]),"source_file":str(f)})
prop=pd.DataFrame(prop).drop_duplicates(["seed","fold","temperature_C"]); prop.to_csv(OUT/"proposed_G4_GRU_residual_slice_rows.csv",index=False)
seed=prop.groupby("seed").MAE_pct.mean(); prow={"method":"proposed_G4_GRU_residual","initial_condition":"native","MAE_pct":seed.mean(),"n_slices":9}
for (fold,temp),g in prop.groupby(["fold","temperature_C"]): prow[f"{fold}_{temp:g}C"]=g.MAE_pct.mean()
wide=pd.concat([wide,pd.DataFrame([prow])],ignore_index=True)
front=["method","initial_condition","MAE_pct","n_slices"]; wide=wide[front+sorted(c for c in wide.columns if c not in front)]
wide.to_csv(OUT/"main_table_sliceunweighted.csv",index=False)
cc=float(wide[(wide.method=="CC")&(wide.initial_condition=="oracle")].MAE_pct.iloc[0]); ukf=slices[slices.method=="2RC_UKF"].sort_values("MAE_pct",ascending=False).iloc[0]
audit={"CC_oracle_slice_unweighted_MAE_pct":cc,"CC_expected_fold_mean_pct":0.102,"CC_warning":abs(cc-.102)>.02,"UKF_worst_fold":ukf.fold,"UKF_worst_temperature_C":ukf.temperature_C,"UKF_worst_MAE_pct":ukf.MAE_pct,"UKF_expected_45C_approx_pct":35.14,"UKF_warning":not (ukf.temperature_C==45 and abs(ukf.MAE_pct-35.14)<1.0),"proposed_slice_unweighted_MAE_pct":float(seed.mean()),"nonoracle_source":"results/initial_soc_robustness.csv because prediction row files contain oracle only"}
(OUT/"validation.json").write_text(json.dumps(audit,indent=2)+"\n"); print(json.dumps(audit,indent=2))
