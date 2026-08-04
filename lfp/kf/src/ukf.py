from __future__ import annotations

import time
import numpy as np

from .data_io import Trajectory
from .ekf import FilterResult, project_psd
from .lfp_ecm import OCVMap, TemperatureParameterMap, propagate_state, terminal_voltage


def run_ukf(
    trajectory: Trajectory, ocv: OCVMap, parameter_map: TemperatureParameterMap,
    initial_soc: float, noise: dict[str,float], gamma: float,
    initial_polarization: tuple[float,float]=(0.0,0.0), initial_h: float=0.0,
    voltage_bias_V: float=0.0, current_offset_A: float=0.0, temperature_bias_C: float=0.0,
    psd_floor: float=1e-12, psd_ceiling: float=1.0,
) -> FilterResult:
    started=time.perf_counter(); nstate=4; n=len(trajectory.time_s)
    alpha=0.1; beta=2.0; kappa=0.0; lam=alpha**2*(nstate+kappa)-nstate; scale=nstate+lam
    wm=np.full(2*nstate+1,1/(2*scale)); wc=wm.copy(); wm[0]=lam/scale; wc[0]=wm[0]+(1-alpha**2+beta)
    x=np.array([np.clip(initial_soc,0,1),initial_polarization[0],initial_polarization[1],np.clip(initial_h,-1,1)],float)
    P=np.diag([2.5e-3,2.5e-3,2.5e-3,0.25]); Q=np.diag([noise['q_soc'],noise['q_vp'],noise['q_vp'],noise['q_h']]); R=float(noise['r_voltage'])
    states=np.empty((n,nstate)); soc=np.empty(n); vhat=np.empty(n); innov=np.empty(n); ptrace=np.empty(n); diverged=False; reason=''
    def sigma_points(mean,cov):
        jitter=psd_floor
        for _ in range(6):
            try: root=np.linalg.cholesky(project_psd(cov,psd_floor,psd_ceiling)*scale+jitter*np.eye(nstate)); break
            except np.linalg.LinAlgError: jitter*=10
        else: raise np.linalg.LinAlgError('sigma point Cholesky failed')
        return np.vstack([mean,mean+root.T,mean-root.T])
    for k in range(n):
        temp=float(trajectory.temperature_series_C[k]+temperature_bias_C); p=parameter_map.lookup(temp,gamma)
        try:
            sig=sigma_points(x,P)
            if k>0:
                i_prev=float(trajectory.current_A[k-1]+current_offset_A); dt=float(trajectory.dt_s[k])
                sig=np.array([propagate_state(s,i_prev,dt,trajectory.q_ref_Ah,p,True) for s in sig])
                x=np.sum(wm[:,None]*sig,axis=0); x[0]=np.clip(x[0],0,1); x[3]=np.clip(x[3],-1,1)
                dev=sig-x; P=project_psd(np.einsum('i,ij,ik->jk',wc,dev,dev)+Q,psd_floor,psd_ceiling); sig=sigma_points(x,P)
            current=float(trajectory.current_A[k]+current_offset_A)
            zsig=np.array([terminal_voltage(s,current,temp,p,ocv,True) for s in sig]); zmean=float(np.sum(wm*zsig))
            dz=zsig-zmean; dx=sig-x; S=float(np.sum(wc*dz*dz)+R); cross=np.sum(wc[:,None]*dx*dz[:,None],axis=0)
            K=cross/max(S,1e-12); residual=float(trajectory.voltage_V[k]+voltage_bias_V-zmean); x=x+K*residual; x[0]=np.clip(x[0],0,1); x[3]=np.clip(x[3],-1,1)
            P=project_psd(P-np.outer(K,K)*S,psd_floor,psd_ceiling)
            states[k]=x; soc[k]=x[0]; vhat[k]=zmean; innov[k]=residual; ptrace[k]=np.trace(P)
        except Exception as exc:
            diverged=True; reason=f'{type(exc).__name__} at index {k}: {exc}'; states[k:]=np.nan; soc[k:]=np.nan; vhat[k:]=np.nan; innov[k:]=np.nan; ptrace[k:]=np.nan; break
    return FilterResult(soc,vhat,innov,states,ptrace,diverged,reason,time.perf_counter()-started)
