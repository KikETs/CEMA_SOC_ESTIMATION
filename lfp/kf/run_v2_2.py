#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from run_v2 import MAIN_METHODS, PROPOSED, rows_from_result, summarize
from src.data_io import load_all, load_confirmatory_predictions
from src.ekf import project_psd
from src.lfp_ecm import ECMParameters, propagate_state
from src.v2_core import FoldParameterMap, OCVGrid, V2FilterResult, run_v2_ekf


ROOT = Path(__file__).resolve().parent
V21 = ROOT / "results_v2_1"
OUT = ROOT / "results_v2_2"
SEED = 20260712
BLOCK = 60
B = 10_000
KF_MAIN = ("plain_2rc_ekf", "hysteresis_2rc_ekf", "adaptive_hysteresis_2rc_ekf")
FROZEN_FILES = (
    "results_v2_1/qr_selection.csv",
    "results_v2_1/ecm_fit_quality.csv",
    "results_v2_1/r0_estimates.csv",
    "artifacts/ocv_table.csv",
    "artifacts/ecm_parameter_map.csv",
    "configs/v2.yaml",
    "configs/v2_1.yaml",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_hashes() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in FROZEN_FILES}


def configs():
    v21 = yaml.safe_load((ROOT / "configs/v2_1.yaml").read_text())
    v2 = yaml.safe_load((ROOT / v21["v2_config"]).read_text())
    base = yaml.safe_load((ROOT / v21["base_config"]).read_text())
    return base, v2, v21


def frozen_context():
    base, v2, v21 = configs()
    trajectories = load_all(base)
    ocv = OCVGrid(pd.read_csv(ROOT / "artifacts/ocv_table.csv"), v2["slope_noise"])
    ecm = pd.read_csv(V21 / "ecm_fit_quality.csv")
    unique = ecm.groupby(["fold_holdout", "temperature_C"], as_index=False).first()
    pmap = FoldParameterMap(unique, pd.read_csv(ROOT / "artifacts/ecm_parameter_map.csv"))
    q = pd.read_csv(V21 / "qr_selection.csv").sort_values("range_cycle").groupby("fold_holdout").tail(1)
    selections = {r.fold_holdout: {k: float(getattr(r, k)) for k in ("q_soc", "q_vp", "q_h", "r_voltage")} for r in q.itertuples()}
    return base, v2, v21, trajectories, ocv, ecm, pmap, selections


def gamma_for(ecm: pd.DataFrame, holdout: str, temp: float) -> float:
    return float(ecm.loc[(ecm.fold_holdout == holdout) & (ecm.temperature_C == float(temp)), "gamma"].iloc[0])


def run_filter(method, tr, ocv, pmap, ecm, selections, v2, delta_pp=0.0):
    soc0 = float(np.clip(tr.soc_ref[0] + delta_pp / 100.0, 0, 1))
    gamma = gamma_for(ecm, tr.profile, tr.temperature_C)
    h0 = ocv.initial_h(tr, soc0)
    kwargs = dict(trajectory=tr, holdout=tr.profile, ocv=ocv, pmap=pmap, initial_soc=soc0,
                  noise=selections[tr.profile], gamma=gamma, flags=v2["fixes"], slope_cfg=v2["slope_noise"])
    if method == "plain_2rc_ekf":
        return run_v2_ekf(**kwargs, with_hysteresis=False)
    if method == "hysteresis_2rc_ekf":
        return run_v2_ekf(**kwargs, with_hysteresis=True, initial_h=h0)
    if method == "adaptive_hysteresis_2rc_ekf":
        return run_v2_ekf(**kwargs, with_hysteresis=True, adaptive=True, initial_h=h0)
    raise ValueError(method)


def proposed_seed_rows(trajectories, ocv, config) -> pd.DataFrame:
    proposed = load_confirmatory_predictions(config)
    lookup = {(tr.profile, float(tr.temperature_C)): tr for tr in trajectories}
    rows = []
    for (seed, profile, temp), g in proposed.groupby(["seed", "drive_cycle", "temperature"]):
        tr = lookup[(str(profile), float(temp))]
        g = g.sort_values("end_index")
        idx = g.end_index.to_numpy(int)
        true = g.y_true.to_numpy(float)
        pred = g.y_pred.to_numpy(float)
        if not np.array_equal(idx, np.flatnonzero(tr.evaluation_mask)):
            raise AssertionError(f"proposed mask mismatch {seed}/{profile}/{temp}")
        if not np.allclose(true, tr.soc_ref[idx], atol=2e-6):
            raise AssertionError(f"proposed truth mismatch {seed}/{profile}/{temp}")
        elapsed = tr.time_s[idx] - tr.time_s[0]
        rows.append(pd.DataFrame({"seed": int(seed), "profile": str(profile), "temperature_C": float(temp),
                                  "end_index": idx, "elapsed_s": elapsed, "soc_true": true,
                                  "soc_pred": pred, "abs_error": np.abs(pred - true)}))
    out = pd.concat(rows, ignore_index=True)
    if set(out.seed.unique()) != {0, 1, 2, 3, 4}:
        raise AssertionError("confirmatory proposed seeds are not 0..4")
    return out


def official_proposed_tables(prop: pd.DataFrame):
    slices = prop.groupby(["seed", "profile", "temperature_C"], as_index=False).agg(
        MAE_pct=("abs_error", lambda x: 100 * x.mean()), n=("abs_error", "size"))
    seeds = slices.groupby("seed", as_index=False).agg(MAE_pct=("MAE_pct", "mean"), n_slices=("MAE_pct", "size"))
    headline = float(seeds.MAE_pct.mean())
    if abs(headline - 0.541) > 1e-3:
        raise AssertionError(f"locked proposed headline mismatch: {headline:.9f} vs 0.541")
    slices.to_csv(OUT / "proposed_seed_slice_mae.csv", index=False)
    seeds.to_csv(OUT / "proposed_seed_headline.csv", index=False)
    ensemble = prop.groupby(["profile", "temperature_C", "end_index"], as_index=False).agg(
        soc_true=("soc_true", "first"), soc_pred=("soc_pred", "mean"))
    ensemble["abs_error"] = (ensemble.soc_pred - ensemble.soc_true).abs()
    ens_slices = ensemble.groupby(["profile", "temperature_C"], as_index=False).agg(MAE_pct=("abs_error", lambda x: 100*x.mean()))
    pd.DataFrame([{"method": PROPOSED, "aggregation": "seed-ensemble (secondary)",
                   "MAE_pct": ens_slices.MAE_pct.mean(), "n_slices": len(ens_slices)}]).to_csv(
        OUT / "appendix_seed_ensemble_secondary.csv", index=False)
    return headline


def ukf_delayed_updates(tr, holdout, ocv, pmap, initial_soc, noise, gamma, flags, slope_cfg, initial_h, delay_s=120.0):
    r0, r1, r2, tau1, tau2 = pmap.arrays(holdout, tr.temperature_series_C, bool(flags["f1_fold_training_ecm"]))
    n = len(tr.time_s); ns = 4; alpha = .1; beta = 2.; lam = alpha**2*ns-ns; scale = ns+lam
    wm = np.full(2*ns+1, 1/(2*scale)); wc = wm.copy(); wm[0] = lam/scale; wc[0] = wm[0]+1-alpha**2+beta
    x = np.array([np.clip(initial_soc,0,1),0.,0.,np.clip(initial_h,-1,1)]); P=np.diag([2.5e-3]*3+[.25])
    Q=np.diag([noise["q_soc"],noise["q_vp"],noise["q_vp"],noise["q_h"]])
    soc=np.empty(n); vhat=np.empty(n); innov=np.empty(n); states=np.empty((n,ns)); ptrace=np.empty(n); rused=np.empty(n)
    diverged=False; reason=""; sref=float(slope_cfg["s_ref_mV_per_pctSOC"])*.1; smin=float(slope_cfg["s_min_mV_per_pctSOC"])*.1
    def sigma(mean,cov):
        jitter=1e-12
        for _ in range(8):
            try:
                root=np.linalg.cholesky(project_psd(cov,1e-12,1.)*scale+jitter*np.eye(ns)); return np.vstack([mean,mean+root.T,mean-root.T])
            except np.linalg.LinAlgError: jitter*=10
        raise np.linalg.LinAlgError("sigma point Cholesky failed")
    for k in range(n):
        p=ECMParameters(float(r0[k]),float(r1[k]),float(tau1[k]/r1[k]),float(r2[k]),float(tau2[k]/r2[k]),float(gamma)); temp=float(tr.temperature_series_C[k])
        try:
            sig=sigma(x,P)
            if k>0:
                sig=np.array([propagate_state(s,float(tr.current_A[k-1]),float(tr.dt_s[k]),tr.q_ref_Ah,p,True) for s in sig])
                x=np.sum(wm[:,None]*sig,axis=0); x[0]=np.clip(x[0],0,1); x[3]=np.clip(x[3],-1,1)
                dev=sig-x; P=project_psd(np.einsum("i,ij,ik->jk",wc,dev,dev)+Q,1e-12,1.); sig=sigma(x,P)
            zsig=np.array([ocv.evaluate("base",float(s[0]),temp)+ocv.evaluate("hmag",float(s[0]),temp)*float(s[3])-float(tr.current_A[k])*p.R0-float(s[1])-float(s[2]) for s in sig])
            zmean=float(np.sum(wm*zsig)); residual=float(tr.voltage_V[k]-zmean); factor=1.
            if flags["f2_slope_aware_measurement_noise"]:
                slope=abs(ocv.evaluate("discharge_slope",float(x[0]),temp)); factor=float(np.clip((sref/max(slope,smin))**2,slope_cfg["factor_min"],slope_cfg["factor_max"]))
            R=float(noise["r_voltage"])*factor; rused[k]=R
            if float(tr.time_s[k]-tr.time_s[0]) >= delay_s:
                dz=zsig-zmean; dx=sig-x; S=float(np.sum(wc*dz*dz)+R); cross=np.sum(wc[:,None]*dx*dz[:,None],axis=0); K=cross/max(S,1e-12)
                x=x+K*residual; x[0]=np.clip(x[0],0,1); x[3]=np.clip(x[3],-1,1); P=project_psd(P-np.outer(K,K)*S,1e-12,1.)
            states[k]=x; soc[k]=x[0]; vhat[k]=zmean; innov[k]=residual; ptrace[k]=np.trace(P)
        except Exception as exc:
            diverged=True; reason=f"{type(exc).__name__} at index {k}: {exc}"; soc[k:]=np.nan; vhat[k:]=np.nan; innov[k:]=np.nan; states[k:]=np.nan; ptrace[k:]=np.nan; rused[k:]=np.nan; break
    return V2FilterResult(soc,vhat,innov,states,ptrace,rused,diverged,reason)


def run_inference(trajectories, ocv, ecm, pmap, selections, v2):
    cached_new=OUT/"prediction_rows_minus10pp.csv.gz"; cached_ukf=OUT/"prediction_rows_ukf_predict120s.csv.gz"; cached_bias=OUT/"current_bias_stress.csv"
    if cached_new.is_file() and cached_ukf.is_file() and cached_bias.is_file():
        old_bias=pd.read_csv(cached_bias); bias_rows=old_bias[old_bias.method=="hysteresis_2rc_ekf"].to_dict("records"); cc_rows=[]
        for tr in trajectories:
            idx=np.flatnonzero(tr.evaluation_mask); last=int(idx[-1]); elapsed_h=(tr.time_s[idx]-tr.time_s[0])/3600.; duration_h=float(elapsed_h[-1])
            for init,bias_A,label in ((-.05,0.,"minus5pp"),(0.,-.01,"minus10mA"),(0.,.01,"plus10mA")):
                err=100*init-100*bias_A*elapsed_h/tr.q_ref_Ah
                cc_rows.append({"method":"realistic_coulomb_count","initial_condition":label,"profile":tr.profile,"temperature_C":tr.temperature_C,"current_bias_mA":bias_A*1000,"MAE_pct_analytic":float(np.mean(np.abs(err))),"end_drift_pp":float(err[-1]),"analytic_CC_drift_pp":-100*bias_A*duration_h/tr.q_ref_Ah,"duration_h":duration_h,"Qref_Ah":tr.q_ref_Ah})
        pd.concat([pd.DataFrame(bias_rows),pd.DataFrame(cc_rows)],ignore_index=True,sort=False).to_csv(cached_bias,index=False)
        return pd.read_csv(cached_new),pd.read_csv(cached_ukf)
    new=[]; ukf=[]; bias_rows=[]; cc_rows=[]
    for tr in trajectories:
        for method in KF_MAIN:
            result=run_filter(method,tr,ocv,pmap,ecm,selections,v2,-10)
            new.append(rows_from_result(method,"minus10pp",tr,result,ocv))
        gamma=gamma_for(ecm,tr.profile,tr.temperature_C); soc0=float(tr.soc_ref[0]); h0=ocv.initial_h(tr,soc0)
        u=ukf_delayed_updates(tr,tr.profile,ocv,pmap,soc0,selections[tr.profile],gamma,v2["fixes"],v2["slope_noise"],h0,120.)
        ukf.append(rows_from_result("hysteresis_2rc_ukf_predict120s", "oracle", tr, u, ocv))
        for bias_A in (-.02,-.01,.01,.02):
            biased=replace(tr,current_A=tr.current_A+bias_A)
            r=run_filter("hysteresis_2rc_ekf",biased,ocv,pmap,ecm,selections,v2,0)
            mask=tr.evaluation_mask; idx=np.flatnonzero(mask); last=int(idx[-1]); duration_h=float((tr.time_s[last]-tr.time_s[0])/3600.)
            mae=float(100*np.mean(np.abs(r.soc[mask]-tr.soc_ref[mask]))); drift=float(100*(r.soc[last]-tr.soc_ref[last])); analytic=-100*bias_A*duration_h/tr.q_ref_Ah
            bias_rows.append({"method":"hysteresis_2rc_ekf","profile":tr.profile,"temperature_C":tr.temperature_C,"current_bias_mA":bias_A*1000,"MAE_pct":mae,"end_drift_pp":drift,"analytic_CC_drift_pp":analytic,"duration_h":duration_h,"Qref_Ah":tr.q_ref_Ah})
        idx=np.flatnonzero(tr.evaluation_mask); last=int(idx[-1]); duration_h=float((tr.time_s[last]-tr.time_s[0])/3600.)
        for init,bias_A,label in ((-.05,0.,"minus5pp"),(0.,-.01,"minus10mA"),(0.,.01,"plus10mA")):
            elapsed_h=(tr.time_s[idx]-tr.time_s[0])/3600.
            err=100*init-100*bias_A*elapsed_h/tr.q_ref_Ah; drift=float(err[-1])
            cc_rows.append({"method":"realistic_coulomb_count","initial_condition":label,"profile":tr.profile,"temperature_C":tr.temperature_C,"current_bias_mA":bias_A*1000,"MAE_pct_analytic":float(np.mean(np.abs(err))),"end_drift_pp":drift,"analytic_CC_drift_pp":-100*bias_A*duration_h/tr.q_ref_Ah,"duration_h":duration_h,"Qref_Ah":tr.q_ref_Ah})
    new=pd.concat(new,ignore_index=True); ukf=pd.concat(ukf,ignore_index=True)
    new.to_csv(OUT/"prediction_rows_minus10pp.csv.gz",index=False,compression="gzip")
    ukf.to_csv(OUT/"prediction_rows_ukf_predict120s.csv.gz",index=False,compression="gzip")
    pd.concat([pd.DataFrame(bias_rows),pd.DataFrame(cc_rows)],ignore_index=True,sort=False).to_csv(OUT/"current_bias_stress.csv",index=False)
    return new,ukf


def build_tables(oldpred,newpred,ukf,pheadline):
    keep=oldpred[oldpred.method.isin(KF_MAIN)&oldpred.initial_condition.isin(["oracle","minus5pp"])]
    combined=pd.concat([keep,newpred],ignore_index=True)
    summarize(combined,["method","initial_condition","profile","temperature_C"]).to_csv(OUT/"temperature_metrics.csv",index=False)
    summarize(combined,["method","initial_condition","profile"]).to_csv(OUT/"fold_summary.csv",index=False)
    summarize(combined,["method","initial_condition","profile","temperature_C","plateau_edge_region"]).to_csv(OUT/"plateau_edge_metrics.csv",index=False)
    summarize(combined,["method","initial_condition","profile","temperature_C","time_since_start_bin_s"]).to_csv(OUT/"time_since_start.csv",index=False)
    rows=[]
    for method in KF_MAIN:
        for initial in ("oracle","minus5pp","minus10pp"):
            g=combined[(combined.method==method)&(combined.initial_condition==initial)]; s=summarize(g,["profile","temperature_C"])
            rows.append({"method":method,"initial_condition":initial,"weighting":"slice_unweighted_primary","MAE_pct":s.MAE_pct.mean(),"RMSE_pct":s.RMSE_pct.mean(),"n_slices":len(s),"n_points":len(g),"status":"complete","reason":""})
        rows.append({"method":method,"initial_condition":"plus5pp","weighting":"slice_unweighted_primary","MAE_pct":np.nan,"RMSE_pct":np.nan,"n_slices":0,"n_points":0,"status":"n/a","reason":"clipped: all records start at SOC = 100%"})
    rows.append({"method":PROPOSED,"initial_condition":"native","weighting":"per-seed slice-unweighted primary","MAE_pct":pheadline,"RMSE_pct":np.nan,"n_slices":24,"n_points":np.nan,"status":"complete","reason":"mean over seed 0..4 headline MAE"})
    pd.DataFrame(rows).to_csv(OUT/"main_table.csv",index=False)
    base_ukf=oldpred[(oldpred.method=="hysteresis_2rc_ukf")&(oldpred.initial_condition=="oracle")]
    ur=[]
    for name,g in (("v2.1_standard",base_ukf),("predict_only_first_120s",ukf)):
        s=summarize(g,["profile","temperature_C"]); ur.append({"condition":name,"MAE_pct":s.MAE_pct.mean(),"RMSE_pct":s.RMSE_pct.mean(),"n_slices":len(s),"n_points":len(g)})
    pd.DataFrame(ur).to_csv(OUT/"si_ukf_table.csv",index=False)
    return combined,pd.DataFrame(ur)


def block_bootstrap(combined, prop):
    prop=prop[["seed","profile","temperature_C","end_index","abs_error"]].rename(columns={"abs_error":"prop_abs"})
    rows=[]; rng=np.random.default_rng(SEED)
    records=sorted(combined[["profile","temperature_C"]].drop_duplicates().itertuples(index=False,name=None))
    starts={}
    for profile,temp in records:
        n=combined[(combined.profile==profile)&(combined.temperature_C==temp)].end_index.nunique()
        nb=n//BLOCK; rem=n%BLOCK
        starts[(profile,temp)]=(rng.integers(0,n,size=(B,nb)),rng.integers(0,n,size=B) if rem else None,n,rem)
    for (method,initial),kf in combined.groupby(["method","initial_condition"]):
        boot=np.zeros(B); observed=[]; used=0
        for profile,temp in records:
            kg=kf[(kf.profile==profile)&(kf.temperature_C==temp)].sort_values("end_index")
            pg=prop[(prop.profile==profile)&(prop.temperature_C==temp)].sort_values(["seed","end_index"])
            if kg.empty or pg.empty: continue
            piv=pg.pivot(index="end_index",columns="seed",values="prop_abs").loc[kg.end_index]
            diff=100*(kg.abs_error.to_numpy()[:,None]-piv.to_numpy())
            observed.append(float(diff.mean(axis=0).mean())); n=len(diff); full,partial,_,rem=starts[(profile,temp)]
            ext=np.vstack([diff,diff[:BLOCK-1]]); pref=np.vstack([np.zeros((1,5)),np.cumsum(ext,axis=0)])
            sums=pref[np.arange(n)+BLOCK]-pref[np.arange(n)]
            totals=sums[full].sum(axis=1) if full.shape[1] else np.zeros((B,5))
            if rem:
                ps=pref[np.arange(n)+rem]-pref[np.arange(n)]; totals+=ps[partial]
            boot+=totals.mean(axis=1)/n; used+=1
        boot/=used
        rows.append({"method":method,"initial_condition":initial,"weighting":"per-seed slice-unweighted","delta_MAE_method_minus_proposed_pct":np.mean(observed),"ci_low_pct":np.quantile(boot,.025),"ci_high_pct":np.quantile(boot,.975),"B":B,"block_samples":BLOCK,"seed":SEED,"n_records":used,"n_proposed_seeds":5})
    out=pd.DataFrame(rows); out.to_csv(OUT/"paired_bootstrap.csv",index=False); return out


def tradeoff_plot():
    lfp=pd.read_csv(V21/"tradeoff.csv"); nmc=pd.read_csv(V21/"tradeoff_nmc.csv")
    plotted=[]
    for chem,d,band in (("LFP",lfp,"IQR"),("NMC",nmc,"minmax")):
        for open_loop,q,g in d.groupby(["open_loop","q_soc"],dropna=False):
            for panel,col in (("oracle","oracle_steady_MAE_pct"),("minus5","minus5_residual_error_pp")):
                x=g[col].to_numpy(float); plotted.append({"chemistry":chem,"panel":panel,"open_loop":bool(open_loop),"q_soc":q,"median":np.nanmedian(x),"low":np.nanquantile(x,.25) if band=="IQR" else np.nanmin(x),"high":np.nanquantile(x,.75) if band=="IQR" else np.nanmax(x),"band":band,"n":np.isfinite(x).sum()})
    p=pd.DataFrame(plotted); p.to_csv(OUT/"fig_tradeoff_overlay_plotted.csv",index=False)
    finite=p[~p.open_loop].q_soc.dropna(); xmin,xmax=finite.min(),finite.max(); openx=xmin/25
    fig,axs=plt.subplots(1,2,figsize=(13,5.2))
    for ax,panel,ylabel in zip(axs,("oracle","minus5"),("Oracle steady MAE (%)","|minus5 residual| (pp)")):
        for chem,color in (("LFP","#2166ac"),("NMC","#b2182b")):
            g=p[(p.panel==panel)&(p.chemistry==chem)&(~p.open_loop)].sort_values("q_soc")
            ax.plot(g.q_soc,g["median"],"o-",color=color,label=chem); ax.fill_between(g.q_soc,g.low,g.high,color=color,alpha=.18)
            og=p[(p.panel==panel)&(p.chemistry==chem)&p.open_loop]
            if len(og): ax.scatter([openx],[og.iloc[0]["median"]],facecolors="none",edgecolors=color,s=70)
        ax.axvline(xmin/5,color="0.4",ls=":"); ax.text(openx,ax.get_ylim()[0] if ax.get_ylim()[0]>0 else .01,"open-loop",rotation=90,va="bottom",ha="center",fontsize=8)
        ax.axhline(.541,color="#2b8c4b",ls="--",label="proposed G4 (no init required)")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("q_soc"); ax.set_ylabel(ylabel); ax.grid(alpha=.2,which="both")
    axs[1].axhline(1.5,color="black",ls=":",label="1.5 pp threshold")
    lfp2=p[(p.chemistry=="LFP")&(p.panel=="minus5")&(~p.open_loop)].sort_values("q_soc")
    scar=np.median(lfp2["median"].to_numpy()-p[(p.chemistry=="LFP")&(p.panel=="oracle")&(~p.open_loop)].sort_values("q_soc")["median"].to_numpy())
    axs[1].annotate("≈0.78 pp persists across the viable gain range",xy=(lfp2.q_soc.median(),lfp2["median"].median()),xytext=(xmin*20,lfp2["median"].max()*1.4),arrowprops={"arrowstyle":"->"})
    for fold,star in pd.read_csv(V21/"qr_selection.csv").sort_values("range_cycle").groupby("fold_holdout").tail(1).groupby("fold_holdout"):
        q=float(star.q_soc.iloc[0]); y=np.interp(np.log(q),np.log(lfp2.q_soc),lfp2["median"]); axs[1].scatter(q,y,marker="*",s=130,color="#2166ac")
    axs[0].annotate("v1 regime",xy=(xmax,float(p[(p.chemistry=="LFP")&(p.panel=="oracle")&(~p.open_loop)].sort_values("q_soc")["median"].iloc[-1])),xytext=(xmax/80,3),arrowprops={"arrowstyle":"->"})
    axs[0].legend(frameon=False); axs[1].legend(frameon=False); fig.tight_layout()
    fig.savefig(OUT/"fig_tradeoff_overlay.png",dpi=300); fig.savefig(OUT/"fig_tradeoff_overlay.pdf"); plt.close(fig)
    return scar


def audits_and_report(before, ukf_table, bootstrap, headline, scar, ecm):
    after=frozen_hashes(); same=before==after
    gamma_cap=int(np.isclose(ecm.gamma,1e-2,rtol=0,atol=1e-12).sum()); tau2_cap=int(np.isclose(ecm.tau2_s,600.,rtol=0,atol=1e-6).sum())
    audit={"status":"PASS" if same else "FAIL","inference_aggregation_only":True,"optimization_attempted":False,"frozen_files":{k:{"before":before[k],"after":after[k],"identical":before[k]==after[k]} for k in before},"q_r_ecm_byte_identical":same,"gamma_at_1e-2_cap":gamma_cap,"gamma_rows":len(ecm),"tau2_at_600s_cap":tau2_cap,"tau2_rows":len(ecm)}
    (OUT/"leakage_audit_v2_2.json").write_text(json.dumps(audit,indent=2)+"\n")
    main=pd.read_csv(OUT/"main_table.csv",keep_default_na=False); expected=len(KF_MAIN)*4+1
    plus5_na=int(((main.initial_condition=="plus5pp")&(main.status=="n/a")&main.reason.str.contains("clipped")).sum())
    complete={"status":"PASS" if len(main)==expected and same and plus5_na==len(KF_MAIN) else "FAIL","main_table_rows":len(main),"expected_rows":expected,"plus5_na_rows":plus5_na,"expected_plus5_na_rows":len(KF_MAIN),"ukf_moved_to_si":True,"proposed_primary_aggregation":"mean of per-seed slice-unweighted MAE, seeds 0..4"}
    (OUT/"completeness_audit.json").write_text(json.dumps(complete,indent=2)+"\n")
    u0=float(ukf_table.loc[ukf_table.condition=="v2.1_standard","MAE_pct"].iloc[0]); u1=float(ukf_table.loc[ukf_table.condition=="predict_only_first_120s","MAE_pct"].iloc[0])
    verdict="collapsed toward EKF level" if u1 < u0*.6 else "did not collapse toward EKF level"
    us=bootstrap[(bootstrap.method=="hysteresis_2rc_ekf")&(bootstrap.initial_condition=="minus10pp")].iloc[0]
    boot_lines=["| Method | Init | Delta KF−G4 (pp) | 95% CI |","|---|---:|---:|---:|"]
    for r in bootstrap.sort_values(["method","initial_condition"]).itertuples():
        boot_lines.append(f"| {r.method} | {r.initial_condition} | {r.delta_MAE_method_minus_proposed_pct:.3f} | [{r.ci_low_pct:.3f}, {r.ci_high_pct:.3f}] |")
    bias=pd.read_csv(OUT/"current_bias_stress.csv"); hb=bias[bias.method=="hysteresis_2rc_ekf"].groupby("current_bias_mA")[["MAE_pct","end_drift_pp","analytic_CC_drift_pp"]].mean()
    bias_text="; ".join(f"{b:+.0f} mA: MAE {r.MAE_pct:.3f}%, end drift {r.end_drift_pp:+.3f} pp (CC analytic {r.analytic_CC_drift_pp:+.3f} pp)" for b,r in hb.iterrows())
    block=f"""

<!-- V2_2_START -->
## v2.2 — reporting finalization (no re-tuning)

This pass is additive and inference/aggregation-only. No Q/R re-selection, ECM refit, or filter-equation change was performed. The frozen Q/R and ECM inputs are byte-identical to v2.1 (`leakage_audit_v2_2.json`: {audit['status']}).

### Headline aggregation and initial-condition axis

- Proposed G4 is reported as the mean of per-seed, 24-slice-unweighted MAE over seeds 0–4: **{headline:.3f}%**. The seed-ensemble result is retained only in `appendix_seed_ensemble_secondary.csv`.
- The +5 pp condition is **n/a (clipped: all records start at SOC = 100%)**; the original clipping evidence remains in v2.1.
- Frozen-parameter −10 pp inference is included for plain, hysteresis, and adaptive EKF. For hysteresis EKF, the paired block-bootstrap delta versus proposed is {us.delta_MAE_method_minus_proposed_pct:.3f} pp (95% CI {us.ci_low_pct:.3f}, {us.ci_high_pct:.3f}).

{chr(10).join(boot_lines)}

### UKF disposition

UKF rows are moved to `si_ukf_table.csv`. With measurement updates disabled for the first 120 s, UKF MAE changed from {u0:.3f}% to {u1:.3f}%; it **{verdict}**. This diagnostic therefore {'supports' if verdict.startswith('collapsed') else 'does not support'} the start-at-OCV-cliff sigma-point hypothesis as the dominant explanation.

### Current-bias stress

`current_bias_stress.csv` applies ±10/±20 mA only to the frozen hysteresis-EKF input and retains untouched labels. The analytic CC reference is 0.909%/h per 10 mA at 1.1 Ah; realistic CC rows include −5 pp initialization and ±10 mA. Across the 24 records: {bias_text}.

### Manuscript-ready interpretation

**Label-circularity footnote.** With q_soc ≈ 1e-12, the plateau filter is effectively the label-generating integrator; the oracle row reads as label consistency, not independent physical validation.

**US06 recovery generalization gap.** The training gate passed at 0.906, but the held-out −5 pp residual was 1.86 pp, exceeding the 1.5 pp threshold.

**Fit-bound rails.** Gamma was at the 1e-2 cap in {gamma_cap}/{len(ecm)} fits; tau2 was at the 600 s cap in {tau2_cap}/{len(ecm)} fits.

**Scar definition.** Scar = |residual after −5 pp init| − oracle steady MAE, per sweep point. The plotted median scar is {scar:.3f} pp across the finite LFP sweep.
<!-- V2_2_END -->
"""
    report=ROOT/"report.md"; text=report.read_text();
    if "<!-- V2_2_START -->" in text: text=text.split("<!-- V2_2_START -->")[0].rstrip()+"\n"
    report.write_text(text+block)
    if not same: raise RuntimeError("immutable v2.1 Q/R or ECM artifact changed")


def main():
    OUT.mkdir(exist_ok=True)
    source=Path(__file__).read_text()
    forbidden_tokens = ["study." + "optimize(", "least_" + "squares(",
                        "fit_hysteresis_ecm_by_" + "fold(", "tune_" + "fold("]
    for forbidden in forbidden_tokens:
        if forbidden in source: raise RuntimeError(f"optimization token detected: {forbidden}")
    before=frozen_hashes(); base,v2,v21,trajectories,ocv,ecm,pmap,selections=frozen_context()
    (OUT/"initial_condition_clipping.csv").write_bytes((V21/"initial_condition_clipping.csv").read_bytes())
    prop=proposed_seed_rows(trajectories,ocv,v2["proposed_confirmatory"]); headline=official_proposed_tables(prop)
    oldpred=pd.read_csv(V21/"prediction_rows_v2_1.csv.gz")
    newpred,ukf=run_inference(trajectories,ocv,ecm,pmap,selections,v2)
    combined,ukf_table=build_tables(oldpred,newpred,ukf,headline)
    boot=block_bootstrap(combined,prop)
    plotted_path=OUT/"fig_tradeoff_overlay_plotted.csv"
    for required in (plotted_path,OUT/"fig_tradeoff_overlay.png",OUT/"fig_tradeoff_overlay.pdf"):
        if not required.is_file(): raise RuntimeError(f"missing locally generated NMC/LFP overlay artifact: {required}")
    plotted=pd.read_csv(plotted_path)
    a=plotted[(plotted.chemistry=="LFP")&(plotted.panel=="oracle")&(~plotted.open_loop)].sort_values("q_soc")
    m=plotted[(plotted.chemistry=="LFP")&(plotted.panel=="minus5")&(~plotted.open_loop)].sort_values("q_soc")
    scar=float(np.median(m["median"].to_numpy()-a["median"].to_numpy()))
    audits_and_report(before,ukf_table,boot,headline,scar,ecm)
    print(json.dumps({"status":"PASS","headline":headline,"outputs":str(OUT)},indent=2))


if __name__ == "__main__": main()
