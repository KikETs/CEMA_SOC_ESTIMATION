from __future__ import annotations

import numpy as np
import pandas as pd


def error_metrics(y_true, y_pred) -> dict[str,float]:
    y=np.asarray(y_true,float); p=np.asarray(y_pred,float); mask=np.isfinite(y)&np.isfinite(p)
    if not mask.any():
        return {k:np.nan for k in ("MAE_pct","RMSE_pct","p95_AE_pct","max_AE_pct","AE_gt_2pct_frac","AE_gt_3pct_frac","bias_pct")}|{"n":0}
    e=(p[mask]-y[mask])*100; ae=np.abs(e)
    return {
        "n":int(len(e)),"MAE_pct":float(ae.mean()),"RMSE_pct":float(np.sqrt(np.mean(e**2))),
        "p95_AE_pct":float(np.percentile(ae,95)),"max_AE_pct":float(ae.max()),
        "AE_gt_2pct_frac":float(np.mean(ae>2)),"AE_gt_3pct_frac":float(np.mean(ae>3)),"bias_pct":float(e.mean()),
    }


def soc_band(soc: np.ndarray) -> np.ndarray:
    bins=np.array([0,.1,.3,.7,.9,1.000001]); labels=np.array(["0-10","10-30","30-70","70-90","90-100"],object)
    return labels[np.clip(np.digitize(np.asarray(soc,float),bins)-1,0,len(labels)-1)]


def ocv_region(slope: np.ndarray, plateau: float=.10, transition: float=.50) -> np.ndarray:
    a=np.abs(np.asarray(slope,float)); return np.where(a<plateau,"plateau",np.where(a<transition,"transition","edge"))


def bootstrap_paired_ci(frame: pd.DataFrame, delta_column: str, replicates: int=10000, seed: int=20260711) -> dict[str,float]:
    values=frame[delta_column].dropna().to_numpy(float); rng=np.random.default_rng(seed)
    if len(values)==0: return {"mean":np.nan,"ci_low":np.nan,"ci_high":np.nan,"n":0}
    samples=rng.choice(values,size=(int(replicates),len(values)),replace=True).mean(axis=1)
    return {"mean":float(values.mean()),"ci_low":float(np.percentile(samples,2.5)),"ci_high":float(np.percentile(samples,97.5)),"n":int(len(values))}
