from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .data_io import Trajectory
from .hysteresis_model import hysteresis_jacobian_alpha
from .lfp_ecm import OCVMap, TemperatureParameterMap, propagate_state, terminal_voltage


@dataclass
class FilterResult:
    soc: np.ndarray
    voltage_prediction: np.ndarray
    innovation: np.ndarray
    states: np.ndarray
    covariance_trace: np.ndarray
    diverged: bool
    divergence_reason: str
    runtime_s: float


def project_psd(matrix: np.ndarray, floor: float = 1e-12, ceiling: float = 1.0) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(sym)
    values = np.clip(values, floor, ceiling)
    return (vectors * values) @ vectors.T


def coulomb_count(trajectory: Trajectory, initial_soc: float, current_offset_A: float = 0.0) -> FilterResult:
    started = time.perf_counter(); n = len(trajectory.time_s)
    soc = np.empty(n); soc[0] = np.clip(initial_soc, 0.0, 1.0)
    for k in range(1, n):
        soc[k] = np.clip(
            soc[k-1] - (trajectory.current_A[k-1] + current_offset_A) * trajectory.dt_s[k] / (3600*trajectory.q_ref_Ah),
            0.0, 1.0,
        )
    return FilterResult(soc, np.full(n, np.nan), np.full(n, np.nan), soc[:,None], np.zeros(n), False, "", time.perf_counter()-started)


def run_ekf(
    trajectory: Trajectory,
    ocv: OCVMap,
    parameter_map: TemperatureParameterMap,
    initial_soc: float,
    noise: dict[str, float],
    gamma: float,
    with_hysteresis: bool,
    adaptive: bool = False,
    initial_polarization: tuple[float, float] = (0.0, 0.0),
    initial_h: float = 0.0,
    voltage_bias_V: float = 0.0,
    current_offset_A: float = 0.0,
    temperature_bias_C: float = 0.0,
    psd_floor: float = 1e-12,
    psd_ceiling: float = 1.0,
    adaptive_alpha: float = 0.01,
    adaptive_bounds: tuple[float,float] = (1e-7,1e-3),
) -> FilterResult:
    started = time.perf_counter(); nstate = 4 if with_hysteresis else 3; n = len(trajectory.time_s)
    x = np.zeros(nstate); x[0] = np.clip(initial_soc,0,1); x[1:3] = initial_polarization
    if with_hysteresis: x[3] = np.clip(initial_h,-1,1)
    P = np.diag([2.5e-3, 2.5e-3, 2.5e-3] + ([0.25] if with_hysteresis else []))
    Q = np.diag([noise["q_soc"], noise["q_vp"], noise["q_vp"]] + ([noise["q_h"]] if with_hysteresis else []))
    R = float(noise["r_voltage"])
    states=np.empty((n,nstate)); soc=np.empty(n); vhat=np.empty(n); innov=np.empty(n); ptrace=np.empty(n)
    diverged=False; reason=""
    for k in range(n):
        temp = float(trajectory.temperature_series_C[k] + temperature_bias_C)
        p = parameter_map.lookup(temp, gamma)
        if k > 0:
            i_prev = float(trajectory.current_A[k-1] + current_offset_A); dt = float(trajectory.dt_s[k])
            a1=np.exp(-dt/max(p.tau1,dt+1e-9)); a2=np.exp(-dt/max(p.tau2,dt+1e-9))
            F=np.eye(nstate); F[1,1]=a1; F[2,2]=a2
            if with_hysteresis: F[3,3]=hysteresis_jacobian_alpha(i_prev,dt,trajectory.q_ref_Ah,gamma)
            x=propagate_state(x,i_prev,dt,trajectory.q_ref_Ah,p,with_hysteresis)
            P=project_psd(F@P@F.T+Q,psd_floor,psd_ceiling)
        current=float(trajectory.current_A[k]+current_offset_A)
        predicted=terminal_voltage(x,current,temp,p,ocv,with_hysteresis)
        d_ocv=float(np.clip(ocv.docv_dsoc(x[0],temp),-10.0,10.0))
        H=np.zeros((1,nstate)); H[0,0]=d_ocv; H[0,1:3]=-1.0
        if with_hysteresis:
            H[0,0] += float(np.clip(ocv.dhmag_dsoc(x[0],temp),-10,10))*x[3]
            H[0,3] = float(ocv.hmag(x[0],temp))
        measurement=float(trajectory.voltage_V[k]+voltage_bias_V)
        residual=measurement-predicted; S=float((H@P@H.T).item()+R)
        if not np.isfinite(S) or S <= 0:
            diverged=True; reason=f"non-positive innovation covariance at index {k}"; S=max(abs(S),1e-12)
        K=(P@H.T/S).reshape(-1); x=x+K*residual; x[0]=np.clip(x[0],0,1)
        if with_hysteresis: x[3]=np.clip(x[3],-1,1)
        I=np.eye(nstate); KH=np.outer(K,H.reshape(-1)); P=project_psd((I-KH)@P@(I-KH).T+np.outer(K,K)*R,psd_floor,psd_ceiling)
        if adaptive:
            R=float(np.clip((1-adaptive_alpha)*R+adaptive_alpha*max(residual**2-float((H@P@H.T).item()),adaptive_bounds[0]),*adaptive_bounds))
        states[k]=x; soc[k]=x[0]; vhat[k]=predicted; innov[k]=residual; ptrace[k]=np.trace(P)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(P)):
            diverged=True; reason=f"non-finite state/covariance at index {k}"; states[k:]=np.nan; soc[k:]=np.nan; vhat[k:]=np.nan; innov[k:]=np.nan; ptrace[k:]=np.nan; break
    return FilterResult(soc,vhat,innov,states,ptrace,diverged,reason,time.perf_counter()-started)
