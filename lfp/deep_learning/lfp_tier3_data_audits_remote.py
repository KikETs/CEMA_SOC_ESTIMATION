#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent; DATA=ROOT.parents[1]/'data/preprocessed/prepared_data_ocv_discharge_soc'; OUT=ROOT/'tier3_inference_only'; OUT.mkdir(exist_ok=True)
RES=ROOT/'runs/all8_train_all8_test/tier1/nmc_goal_vcorr_it_train_dst_selector_results'
temps=(-10,0,10,20,25,30,40,50); profiles=('DST','FUDS','US06'); features=('T6','T7','G4')

def prefix(f,p): return RES/f'lfpconfirm_all8_train_all8_test_tier1_gru_residual_{f.lower()}_holdout{p.lower()}_s012_b2048_e200'
def predfile(f,p,s):
 m=list(RES.glob(prefix(f,p).name+f'_seed{s}_*_test_prediction_rows.csv.gz'))
 if len(m)!=1: raise RuntimeError((f,p,s,m))
 return m[0]

# Existing record-start cold-start uses recorded Test_Time(s), never row number.
cold=[]; bias=[]; local=[]; data_bins=[]
time_bins=[(-np.inf,60,'<=60_s'),(60,300,'60-300_s'),(300,900,'300-900_s'),(900,1800,'900-1800_s'),(1800,3600,'1800-3600_s'),(3600,np.inf,'>3600_s')]
for p in profiles:
 raw_by_temp={t:pd.read_csv(DATA/f'{t}C/LFP_{t}C_{p}.csv') for t in temps}
 for f in features:
  for s in (0,1,2):
   a=pd.read_csv(predfile(f,p,s))
   for t,g in a.groupby('temperature'):
    raw=raw_by_temp[int(t)]; idx=g.end_index.to_numpy(int); tt=pd.to_numeric(raw['Test_Time(s)']).to_numpy(float); elapsed=tt[idx]-tt[0]
    err=(g.y_pred.to_numpy(float)-g.y_true.to_numpy(float))*100
    for lo,hi,label in time_bins:
     m=(elapsed>lo)&(elapsed<=hi)
     if m.any(): cold.append({'feature':f,'fold':p,'seed':s,'temp':t,'elapsed_bin':label,'n':m.sum(),'mae':np.mean(abs(err[m])),'bias':np.mean(err[m]),'time_source':'recorded Test_Time(s)'})
    if float(t)==-10:
     sb=pd.cut(g.y_true*100,[-.001,20,40,60,80,100.001],labels=['0-20','20-40','40-60','60-80','80-100'])
     for b in sb.cat.categories:
      m=sb.eq(b).to_numpy()
      if m.any(): bias.append({'feature':f,'profile':p,'seed':s,'soc_bin':str(b),'n':m.sum(),'mae':np.mean(abs(err[m])),'bias':np.mean(err[m])})
   
 # data-only V-I bins by temperature; local comparison uses common end indices.
 for t,raw in raw_by_temp.items():
  v=pd.to_numeric(raw['Voltage(V)']); i=pd.to_numeric(raw['Current(A)']); soc=pd.to_numeric(raw['SOC_CC'])
  vb=pd.qcut(v,20,duplicates='drop'); ib=pd.qcut(i,10,duplicates='drop'); tmp=pd.DataFrame({'vb':vb.astype(str),'ib':ib.astype(str),'soc':soc})
  z=tmp.groupby(['vb','ib'],observed=True).agg(n=('soc','size'),soc_iqr=('soc',lambda x:np.percentile(x,75)-np.percentile(x,25))).reset_index(); valid=z[z.n>=20]
  q=float(valid.soc_iqr.quantile(.9)) if len(valid) else np.nan
  data_bins.append({'temperature_C':t,'profile':p,'valid_bins':len(valid),'median_SOC_IQR_pct':100*valid.soc_iqr.median(),'P90_SOC_IQR_pct':100*valid.soc_iqr.quantile(.9),'max_SOC_IQR_pct':100*valid.soc_iqr.max(),'ambiguous_fraction':float((valid.soc_iqr>=q).mean()) if len(valid) else np.nan,'ambiguity_rule':'top decile SOC IQR among VxI quantile bins with n>=20'})
  # compare each model only at predictions whose raw sample lies in ambiguous bins
  key=set(map(tuple,valid.loc[valid.soc_iqr>=q,['vb','ib']].to_numpy())) if len(valid) else set()
  tags=list(zip(vb.astype(str),ib.astype(str)))
  for f in features:
   for s in (0,1,2):
    a=pd.read_csv(predfile(f,p,s)); g=a[a.temperature.eq(t)]; idx=g.end_index.to_numpy(int); m=np.array([tags[x] in key for x in idx]); e=(g.y_pred.to_numpy()-g.y_true.to_numpy())*100
    if m.any(): local.append({'temperature_C':t,'profile':p,'feature':f,'seed':s,'n':m.sum(),'MAE_pct':np.mean(abs(e[m])),'bias_pct':np.mean(e[m])})

pd.DataFrame(cold).to_csv(OUT/'cold_start_existing_record.csv',index=False)
pd.DataFrame(bias).to_csv(OUT/'minus10_socbin_profile_bias.csv',index=False)
pd.DataFrame(data_bins).to_csv(OUT/'vi_ambiguity_data_only.csv',index=False)
pd.DataFrame(local).to_csv(OUT/'vi_ambiguity_model_comparison.csv',index=False)

# Label audit: preserve every explicit OCV/Qref/SOC0 field and summarize by record.
lab=[]
for t in temps:
 for p in profiles:
  path=DATA/f'{t}C/LFP_{t}C_{p}.csv'; d=pd.read_csv(path)
  cols=[c for c in d.columns if any(k in c.lower() for k in ('soc0','qref','capacity','ocv','soc_initial'))]
  row={'temperature_C':t,'profile':p,'file':str(path),'n':len(d),'audit_columns':';'.join(cols)}
  for c in cols:
   x=pd.to_numeric(d[c],errors='coerce').dropna()
   if len(x): row[c+'_first']=x.iloc[0]; row[c+'_last']=x.iloc[-1]; row[c+'_min']=x.min(); row[c+'_max']=x.max()
  lab.append(row)
pd.DataFrame(lab).to_csv(OUT/'ocv_discharge_label_uncertainty_audit.csv',index=False)
(OUT/'data_audit_manifest.json').write_text(json.dumps({'training_or_fitting_performed':False,'predictions':'frozen archived rows','time_axis':'recorded Test_Time(s)','ambiguity_bins':'data-only V-I quantile bins','label_audit':'descriptive export; no uncertainty invented'},indent=2)+'\n')
