from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import least_squares
from scipy.signal import lfilter
from scipy.signal import savgol_filter

from .data_io import Trajectory
from .ekf import project_psd
from .hysteresis_model import propagate_hysteresis
from .lfp_ecm import ECMParameters, propagate_state


@dataclass
class V2FilterResult:
    soc: np.ndarray
    voltage_prediction: np.ndarray
    innovation: np.ndarray
    states: np.ndarray
    covariance_trace: np.ndarray
    measurement_variance: np.ndarray
    diverged: bool
    divergence_reason: str


class OCVGrid:
    def __init__(self, table: pd.DataFrame, slope_cfg: dict):
        table = table.sort_values(["temperature_C", "soc"])
        self.temperatures = np.sort(table.temperature_C.unique().astype(float))
        first = table[table.temperature_C == self.temperatures[0]].sort_values("soc")
        self.soc_grid = first.soc.to_numpy(float)
        arrays: dict[str, list[np.ndarray]] = {
            "base": [], "hmag": [], "raw_mid": [], "discharge_slope": [],
        }
        window = int(slope_cfg["savgol_window"]); poly = int(slope_cfg["savgol_polyorder"])
        for temp in self.temperatures:
            d = table[table.temperature_C == temp].sort_values("soc")
            discharge = d.ocv_discharge_raw_V.to_numpy(float)
            smoothed_discharge = savgol_filter(discharge, window_length=window, polyorder=poly, mode="interp")
            arrays["base"].append(d.ocv_base_monotonic_V.to_numpy(float))
            arrays["hmag"].append(d.hysteresis_half_monotonic_V.to_numpy(float))
            arrays["raw_mid"].append(0.5 * (discharge + d.ocv_charge_raw_V.to_numpy(float)))
            arrays["discharge_slope"].append(np.gradient(smoothed_discharge, self.soc_grid))
        self.base = np.asarray(arrays["base"], dtype=float)
        self.hmag = np.asarray(arrays["hmag"], dtype=float)
        self.raw_mid = np.asarray(arrays["raw_mid"], dtype=float)
        self.discharge_slope = np.asarray(arrays["discharge_slope"], dtype=float)
        self.dbase = np.gradient(self.base, self.soc_grid, axis=1)
        self.dhmag = np.gradient(self.hmag, self.soc_grid, axis=1)

    def evaluate(self, field: str, soc: float, temperature_C: float) -> float:
        array = getattr(self, field)
        values = np.array([np.interp(soc, self.soc_grid, row) for row in array])
        return float(np.interp(temperature_C, self.temperatures, values))

    def initial_h(self, trajectory: Trajectory, soc0: float) -> float:
        temp = float(trajectory.temperature_series_C[0])
        mid = self.evaluate("raw_mid", soc0, temp)
        magnitude = self.evaluate("hmag", soc0, temp)
        if magnitude <= 1e-12:
            return 0.0
        return float(np.clip((float(trajectory.voltage_V[0]) - mid) / magnitude, -1.0, 1.0))


def estimate_r0_by_fold(trajectories: list[Trajectory], profiles: list[str], cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []; events = []
    min_di = float(cfg["min_abs_delta_i_A"]); low = float(cfg["min_ratio_ohm"]); high = float(cfg["max_ratio_ohm"])
    temperatures = sorted({float(tr.temperature_C) for tr in trajectories})
    for holdout in profiles:
        for temp in temperatures:
            values = []
            for tr in trajectories:
                if tr.profile == holdout or float(tr.temperature_C) != temp:
                    continue
                raw_current = -tr.current_A
                di = np.diff(raw_current, prepend=raw_current[0])
                dv = np.diff(tr.voltage_V, prepend=tr.voltage_V[0])
                ratio = np.divide(dv, di, out=np.full_like(dv, np.nan), where=np.abs(di) > 1e-12)
                mask = np.isfinite(di) & np.isfinite(dv) & (np.abs(di) > min_di) & (ratio > low) & (ratio < high)
                for idx in np.flatnonzero(mask):
                    value = float(ratio[idx]); values.append(value)
                    events.append({
                        "fold_holdout": holdout, "profile": tr.profile, "temperature_C": temp,
                        "source_file": str(tr.path), "end_index": int(idx),
                        "delta_I_raw_A": float(di[idx]), "delta_V_V": float(dv[idx]), "R0_ohm": value,
                    })
            if not values:
                raise RuntimeError(f"F1 found no R0 events for fold={holdout}, temperature={temp}")
            summaries.append({
                "fold_holdout": holdout, "temperature_C": temp,
                "R0_ohm": float(np.median(values)), "n_events": len(values),
                "R0_p20_ohm": float(np.quantile(values, 0.20)),
                "R0_p80_ohm": float(np.quantile(values, 0.80)),
                "training_profiles": "+".join(p for p in profiles if p != holdout),
            })
    return pd.DataFrame(summaries), pd.DataFrame(events)


def _unit_rc_state(current: np.ndarray, dt_s: float, tau_s: float) -> np.ndarray:
    """Polarization state for R=1, aligned to predict voltage at sample k."""
    a = float(np.exp(-dt_s / tau_s))
    previous_current = np.r_[0.0, np.asarray(current[:-1], dtype=float)]
    return lfilter([1.0 - a], [1.0, -a], previous_current)


def fit_training_ecm_by_fold(
    trajectories: list[Trajectory],
    profiles: list[str],
    r0_table: pd.DataFrame,
    ocv: OCVGrid,
    gamma_by_fold: dict[str, float],
    cfg: dict,
) -> pd.DataFrame:
    """Least-squares 2RC fit on outer-fold training records only.

    R0 is fixed to the fold/temperature dV/dI median. R1, R2, tau1 and tau2
    minimize terminal-voltage residuals jointly over the two training profiles
    at that temperature. Per-record constant offsets prevent a static OCV
    branch mismatch from being misidentified as polarization resistance.
    """
    rows = []
    temperatures = sorted({float(tr.temperature_C) for tr in trajectories})
    r1_bounds = tuple(map(float, cfg["r1_bounds_ohm"]))
    r2_bounds = tuple(map(float, cfg["r2_bounds_ohm"]))
    tau1_bounds = tuple(map(float, cfg["tau1_bounds_s"]))
    tau2_bounds = tuple(map(float, cfg["tau2_bounds_s"]))
    offset_bounds = tuple(map(float, cfg["record_offset_bounds_V"]))
    for holdout in profiles:
        gamma = float(gamma_by_fold[holdout])
        for temp in temperatures:
            records = [
                tr for tr in trajectories
                if tr.profile != holdout and float(tr.temperature_C) == temp
            ]
            if len(records) != len(profiles) - 1:
                raise RuntimeError(f"Expected two ECM training records for {holdout}/{temp:g}C")
            r0 = float(r0_table.loc[
                (r0_table.fold_holdout == holdout) & (r0_table.temperature_C == temp), "R0_ohm"
            ].iloc[0])
            targets = []
            currents = []
            med_dt = []
            for tr in records:
                h = np.empty(len(tr.time_s), dtype=float)
                h[0] = ocv.initial_h(tr, float(tr.soc_ref[0]))
                for k in range(1, len(h)):
                    h[k], _ = propagate_hysteresis(
                        h[k - 1], float(tr.current_A[k - 1]), float(tr.dt_s[k]),
                        float(tr.q_ref_Ah), gamma,
                    )
                base = np.array([
                    ocv.evaluate("base", float(s), float(t))
                    for s, t in zip(tr.soc_ref, tr.temperature_series_C)
                ])
                hmag = np.array([
                    ocv.evaluate("hmag", float(s), float(t))
                    for s, t in zip(tr.soc_ref, tr.temperature_series_C)
                ])
                target = base + hmag * h - tr.current_A * r0 - tr.voltage_V
                targets.append(target)
                currents.append(np.asarray(tr.current_A, dtype=float))
                med_dt.append(float(np.median(tr.dt_s)))

            n_records = len(records)
            lower = np.log([r1_bounds[0], r2_bounds[0], tau1_bounds[0], tau2_bounds[0]])
            upper = np.log([r1_bounds[1], r2_bounds[1], tau1_bounds[1], tau2_bounds[1]])
            lower = np.r_[lower, np.repeat(offset_bounds[0], n_records)]
            upper = np.r_[upper, np.repeat(offset_bounds[1], n_records)]
            x0 = np.r_[
                np.log([
                    np.sqrt(r1_bounds[0] * r1_bounds[1]),
                    np.sqrt(r2_bounds[0] * r2_bounds[1]),
                    np.sqrt(tau1_bounds[0] * tau1_bounds[1]),
                    np.sqrt(tau2_bounds[0] * tau2_bounds[1]),
                ]),
                np.zeros(n_records),
            ]

            def residual(theta: np.ndarray) -> np.ndarray:
                r1, r2, tau1, tau2 = np.exp(theta[:4])
                values = []
                for idx, (current, target, dt) in enumerate(zip(currents, targets, med_dt)):
                    z1 = _unit_rc_state(current, dt, tau1)
                    z2 = _unit_rc_state(current, dt, tau2)
                    values.append(r1 * z1 + r2 * z2 + theta[4 + idx] - target)
                return np.concatenate(values)

            fit = least_squares(
                residual, x0=x0, bounds=(lower, upper),
                max_nfev=int(cfg["max_nfev"]), method="trf",
            )
            r1, r2, tau1, tau2 = map(float, np.exp(fit.x[:4]))
            if tau1 > tau2:
                r1, r2, tau1, tau2 = r2, r1, tau2, tau1
            fit_residual = residual(fit.x)
            rmse_mV = float(1000.0 * np.sqrt(np.mean(fit_residual ** 2)))
            separation_ratio = float(tau1 / tau2)
            rows.append({
                "fold_holdout": holdout,
                "temperature_C": temp,
                "R0_ohm": r0,
                "R1_ohm": r1,
                "R2_ohm": r2,
                "tau1_s": tau1,
                "tau2_s": tau2,
                "C1_F": tau1 / r1,
                "C2_F": tau2 / r2,
                "voltage_fit_RMSE_mV": rmse_mV,
                "rmse_gt_10mV": bool(rmse_mV > float(cfg["rmse_flag_mV"])),
                "tau1_tau2_ratio": separation_ratio,
                "tau1_much_less_than_tau2": bool(
                    separation_ratio <= float(cfg["tau_separation_ratio_max"])
                ),
                "fit_success": bool(fit.success),
                "fit_nfev": int(fit.nfev),
                "training_profiles": "+".join(p for p in profiles if p != holdout),
                "training_files": ";".join(sorted(str(tr.path) for tr in records)),
            })
    result = pd.DataFrame(rows)
    if len(result) != len(profiles) * len(temperatures):
        raise RuntimeError(f"Expected 24 ECM fit rows, found {len(result)}")
    return result


class FoldParameterMap:
    def __init__(self, ecm: pd.DataFrame, v1_anchor: pd.DataFrame):
        self.ecm = ecm.copy()
        self.v1 = v1_anchor.sort_values("temperature_C").copy()

    def arrays(self, holdout: str, temperature_series_C: np.ndarray, use_f1: bool) -> tuple[np.ndarray, ...]:
        t = np.asarray(temperature_series_C, float)
        if use_f1:
            ecm = self.ecm[self.ecm.fold_holdout == holdout].sort_values("temperature_C")
            if len(ecm) != 8:
                raise RuntimeError(f"Expected 8 ECM rows for fold {holdout}, found {len(ecm)}")
            r0_values = np.interp(t, ecm.temperature_C, ecm.R0_ohm)
            r1 = np.interp(t, ecm.temperature_C, ecm.R1_ohm)
            r2 = np.interp(t, ecm.temperature_C, ecm.R2_ohm)
            tau1 = np.interp(t, ecm.temperature_C, ecm.tau1_s)
            tau2 = np.interp(t, ecm.temperature_C, ecm.tau2_s)
        else:
            anchor = self.v1
            r0_values = np.interp(t, anchor.temperature_C, anchor.R0)
            r1 = np.interp(t, anchor.temperature_C, anchor.R1)
            r2 = np.interp(t, anchor.temperature_C, anchor.R2)
            tau1 = np.interp(t, anchor.temperature_C, anchor.R1 * anchor.C1)
            tau2 = np.interp(t, anchor.temperature_C, anchor.R2 * anchor.C2)
        return tuple(np.asarray(a, dtype=float) for a in (r0_values, r1, r2, tau1, tau2))


@njit(cache=True)
def _grid_eval(array, temps, soc_grid, soc, temp):
    s = min(1.0, max(0.0, soc))
    ti = np.searchsorted(temps, temp)
    if ti <= 0:
        t0 = t1 = 0; tw = 0.0
    elif ti >= len(temps):
        t0 = t1 = len(temps) - 1; tw = 0.0
    else:
        t0 = ti - 1; t1 = ti; tw = (temp - temps[t0]) / (temps[t1] - temps[t0])
    si = np.searchsorted(soc_grid, s)
    if si <= 0:
        s0 = s1 = 0; sw = 0.0
    elif si >= len(soc_grid):
        s0 = s1 = len(soc_grid) - 1; sw = 0.0
    else:
        s0 = si - 1; s1 = si; sw = (s - soc_grid[s0]) / (soc_grid[s1] - soc_grid[s0])
    v0 = array[t0, s0] * (1.0 - sw) + array[t0, s1] * sw
    v1 = array[t1, s0] * (1.0 - sw) + array[t1, s1] * sw
    return v0 * (1.0 - tw) + v1 * tw


@njit(cache=True)
def _ekf_core(time_s, dt_s, current, voltage, temp, q_ref, evaluation_mask,
              temps, soc_grid, base_grid, hmag_grid, dbase_grid, dhmag_grid, discharge_slope_grid,
              r0, r1, r2, tau1, tau2, initial_soc, initial_h, gamma,
              q_soc, q_vp, q_h, r_voltage, with_hysteresis, adaptive, open_loop, slope_aware,
              s_ref, s_min, factor_min, factor_max, adaptive_alpha, adaptive_r_min, adaptive_r_max):
    n = len(time_s); nstate = 4 if with_hysteresis else 3
    x = np.zeros(4); x[0] = min(1.0, max(0.0, initial_soc)); x[3] = min(1.0, max(-1.0, initial_h))
    P = np.zeros((4, 4))
    P[0, 0] = 2.5e-3; P[1, 1] = 2.5e-3; P[2, 2] = 2.5e-3; P[3, 3] = 0.25
    qdiag = np.array([q_soc, q_vp, q_vp, q_h])
    soc_out = np.empty(n); vhat = np.empty(n); innovation = np.empty(n)
    states = np.empty((n, 4)); ptrace = np.empty(n); r_used = np.empty(n)
    adaptive_base_r = r_voltage; diverged_index = -1
    for k in range(n):
        if k > 0:
            dt = dt_s[k]; ip = current[k - 1]
            x[0] = min(1.0, max(0.0, x[0] - ip * dt / (3600.0 * q_ref)))
            a1 = np.exp(-dt / max(tau1[k], dt + 1e-9)); a2 = np.exp(-dt / max(tau2[k], dt + 1e-9))
            x[1] = a1 * x[1] + (1.0 - a1) * r1[k] * ip
            x[2] = a2 * x[2] + (1.0 - a2) * r2[k] * ip
            ah = 1.0
            if with_hysteresis:
                ah = np.exp(-max(gamma, 0.0) * abs(ip) * max(dt, 0.0) / (3600.0 * max(q_ref, 1e-12)))
                if abs(ip) > 1e-12:
                    x[3] = min(1.0, max(-1.0, ah * x[3] + (1.0 - ah) * (-np.sign(ip))))
            fdiag = np.array([1.0, a1, a2, ah])
            for i in range(nstate):
                for j in range(nstate):
                    P[i, j] = fdiag[i] * P[i, j] * fdiag[j]
                P[i, i] += qdiag[i]
        base = _grid_eval(base_grid, temps, soc_grid, x[0], temp[k])
        hmag = _grid_eval(hmag_grid, temps, soc_grid, x[0], temp[k])
        predicted = base - current[k] * r0[k] - x[1] - x[2]
        if with_hysteresis:
            predicted += hmag * x[3]
        H = np.zeros(4)
        H[0] = min(10.0, max(-10.0, _grid_eval(dbase_grid, temps, soc_grid, x[0], temp[k])))
        H[1] = -1.0; H[2] = -1.0
        if with_hysteresis:
            H[0] += min(10.0, max(-10.0, _grid_eval(dhmag_grid, temps, soc_grid, x[0], temp[k]))) * x[3]
            H[3] = hmag
        factor = 1.0
        if slope_aware:
            slope = abs(_grid_eval(discharge_slope_grid, temps, soc_grid, x[0], temp[k]))
            factor = (s_ref / max(slope, s_min)) ** 2
            factor = min(factor_max, max(factor_min, factor))
        R = adaptive_base_r * factor; r_used[k] = R
        residual = voltage[k] - predicted
        if not open_loop:
            ph = np.zeros(4)
            for i in range(nstate):
                for j in range(nstate): ph[i] += P[i, j] * H[j]
            S = R
            for i in range(nstate): S += H[i] * ph[i]
            if not np.isfinite(S) or S <= 1e-16:
                diverged_index = k; break
            K = ph / S
            for i in range(nstate): x[i] += K[i] * residual
            x[0] = min(1.0, max(0.0, x[0]))
            if with_hysteresis: x[3] = min(1.0, max(-1.0, x[3]))
            A = np.eye(4)
            for i in range(nstate):
                for j in range(nstate): A[i, j] -= K[i] * H[j]
            AP = np.zeros((4, 4)); newP = np.zeros((4, 4))
            for i in range(nstate):
                for j in range(nstate):
                    for z in range(nstate): AP[i, j] += A[i, z] * P[z, j]
            for i in range(nstate):
                for j in range(nstate):
                    for z in range(nstate): newP[i, j] += AP[i, z] * A[j, z]
                    newP[i, j] += K[i] * K[j] * R
            for i in range(nstate):
                for j in range(nstate): P[i, j] = 0.5 * (newP[i, j] + newP[j, i])
                P[i, i] = min(1.0, max(1e-12, P[i, i]))
            if adaptive:
                predicted_var = 0.0
                for i in range(nstate):
                    for j in range(nstate): predicted_var += H[i] * P[i, j] * H[j]
                target = max(residual * residual - predicted_var, adaptive_r_min)
                adaptive_base_r = min(adaptive_r_max, max(adaptive_r_min, (1.0 - adaptive_alpha) * adaptive_base_r + adaptive_alpha * target))
        states[k] = x; soc_out[k] = x[0]; vhat[k] = predicted; innovation[k] = residual
        trace = 0.0
        for i in range(nstate): trace += P[i, i]
        ptrace[k] = trace
        finite = True
        for i in range(nstate):
            if not np.isfinite(x[i]) or not np.isfinite(P[i, i]): finite = False
        if not finite:
            diverged_index = k; break
    if diverged_index >= 0:
        for k in range(diverged_index, n):
            soc_out[k] = np.nan; vhat[k] = np.nan; innovation[k] = np.nan; ptrace[k] = np.nan; r_used[k] = np.nan
            for j in range(4): states[k, j] = np.nan
    return soc_out, vhat, innovation, states, ptrace, r_used, diverged_index


def run_v2_ekf(trajectory: Trajectory, holdout: str, ocv: OCVGrid, pmap: FoldParameterMap,
               initial_soc: float, noise: dict[str, float], gamma: float, flags: dict,
               slope_cfg: dict, with_hysteresis: bool, adaptive: bool = False,
               open_loop: bool = False, initial_h: float = 0.0) -> V2FilterResult:
    arrays = pmap.arrays(holdout, trajectory.temperature_series_C, bool(flags["f1_fold_training_ecm"]))
    s_ref = float(slope_cfg["s_ref_mV_per_pctSOC"]) * 0.1
    s_min = float(slope_cfg["s_min_mV_per_pctSOC"]) * 0.1
    out = _ekf_core(
        trajectory.time_s, trajectory.dt_s, trajectory.current_A, trajectory.voltage_V,
        trajectory.temperature_series_C, trajectory.q_ref_Ah, trajectory.evaluation_mask,
        ocv.temperatures, ocv.soc_grid, ocv.base, ocv.hmag, ocv.dbase, ocv.dhmag, ocv.discharge_slope,
        *arrays, float(initial_soc), float(initial_h), float(gamma),
        float(noise["q_soc"]), float(noise["q_vp"]), float(noise["q_h"]), float(noise["r_voltage"]),
        bool(with_hysteresis), bool(adaptive), bool(open_loop), bool(flags["f2_slope_aware_measurement_noise"]),
        s_ref, s_min, float(slope_cfg["factor_min"]), float(slope_cfg["factor_max"]), 0.01, 1e-7, 1e-3,
    )
    idx = int(out[-1]); reason = "" if idx < 0 else f"non-finite or non-positive covariance at index {idx}"
    return V2FilterResult(*out[:-1], diverged=idx >= 0, divergence_reason=reason)


@njit(cache=True)
def _chol4(matrix, nstate):
    for attempt in range(8):
        jitter = 10.0 ** (attempt - 12)
        L = np.zeros((4, 4)); ok = True
        for i in range(nstate):
            for j in range(i + 1):
                value = matrix[i, j]
                if i == j: value += jitter
                for k in range(j): value -= L[i, k] * L[j, k]
                if i == j:
                    if value <= 0.0 or not np.isfinite(value): ok = False; break
                    L[i, j] = np.sqrt(value)
                else:
                    if L[j, j] <= 0.0: ok = False; break
                    L[i, j] = value / L[j, j]
            if not ok: break
        if ok: return L, True
    return np.zeros((4, 4)), False


@njit(cache=True)
def _ukf_core(time_s, dt_s, current, voltage, temp, q_ref,
              temps, soc_grid, base_grid, hmag_grid, discharge_slope_grid,
              r0, r1, r2, tau1, tau2, initial_soc, initial_h, gamma,
              q_soc, q_vp, q_h, r_voltage, slope_aware, s_ref, s_min, factor_min, factor_max):
    n = len(time_s); ns = 4; npnt = 9; alpha = 0.1; beta = 2.0
    lam = alpha * alpha * ns - ns; scale = ns + lam
    wm = np.full(npnt, 1.0 / (2.0 * scale)); wc = wm.copy()
    wm[0] = lam / scale; wc[0] = wm[0] + 1.0 - alpha * alpha + beta
    x = np.array([min(1.0, max(0.0, initial_soc)), 0.0, 0.0, min(1.0, max(-1.0, initial_h))])
    P = np.diag(np.array([2.5e-3, 2.5e-3, 2.5e-3, 0.25])); qdiag = np.array([q_soc, q_vp, q_vp, q_h])
    soc_out = np.empty(n); vhat = np.empty(n); innov = np.empty(n); states = np.empty((n, 4)); ptrace = np.empty(n); r_used = np.empty(n)
    failed = -1
    for k in range(n):
        root, ok = _chol4(P * scale, ns)
        if not ok: failed = k; break
        sig = np.empty((npnt, ns)); sig[0] = x
        for j in range(ns):
            for i in range(ns):
                sig[1 + j, i] = x[i] + root[i, j]
                sig[1 + ns + j, i] = x[i] - root[i, j]
        if k > 0:
            dt = dt_s[k]; ip = current[k - 1]
            a1 = np.exp(-dt / max(tau1[k], dt + 1e-9)); a2 = np.exp(-dt / max(tau2[k], dt + 1e-9))
            ah = np.exp(-max(gamma, 0.0) * abs(ip) * max(dt, 0.0) / (3600.0 * max(q_ref, 1e-12)))
            for q in range(npnt):
                sig[q, 0] = min(1.0, max(0.0, sig[q, 0] - ip * dt / (3600.0 * q_ref)))
                sig[q, 1] = a1 * sig[q, 1] + (1.0 - a1) * r1[k] * ip
                sig[q, 2] = a2 * sig[q, 2] + (1.0 - a2) * r2[k] * ip
                if abs(ip) > 1e-12: sig[q, 3] = min(1.0, max(-1.0, ah * sig[q, 3] + (1.0 - ah) * (-np.sign(ip))))
            x[:] = 0.0
            for q in range(npnt): x += wm[q] * sig[q]
            x[0] = min(1.0, max(0.0, x[0])); x[3] = min(1.0, max(-1.0, x[3]))
            P[:] = 0.0
            for q in range(npnt):
                dev = sig[q] - x
                for i in range(ns):
                    for j in range(ns): P[i, j] += wc[q] * dev[i] * dev[j]
            for i in range(ns): P[i, i] += qdiag[i]
            root, ok = _chol4(P * scale, ns)
            if not ok: failed = k; break
            sig[0] = x
            for j in range(ns):
                for i in range(ns):
                    sig[1 + j, i] = x[i] + root[i, j]
                    sig[1 + ns + j, i] = x[i] - root[i, j]
        zsig = np.empty(npnt)
        for q in range(npnt):
            base = _grid_eval(base_grid, temps, soc_grid, sig[q, 0], temp[k])
            mag = _grid_eval(hmag_grid, temps, soc_grid, sig[q, 0], temp[k])
            zsig[q] = base + mag * sig[q, 3] - current[k] * r0[k] - sig[q, 1] - sig[q, 2]
        zmean = 0.0
        for q in range(npnt): zmean += wm[q] * zsig[q]
        factor = 1.0
        if slope_aware:
            slope = abs(_grid_eval(discharge_slope_grid, temps, soc_grid, x[0], temp[k]))
            factor = min(factor_max, max(factor_min, (s_ref / max(slope, s_min)) ** 2))
        R = r_voltage * factor; r_used[k] = R
        S = R; cross = np.zeros(ns)
        for q in range(npnt):
            dz = zsig[q] - zmean
            S += wc[q] * dz * dz
            dev = sig[q] - x
            for i in range(ns): cross[i] += wc[q] * dev[i] * dz
        if S <= 1e-16 or not np.isfinite(S): failed = k; break
        K = cross / S; residual = voltage[k] - zmean; x += K * residual
        x[0] = min(1.0, max(0.0, x[0])); x[3] = min(1.0, max(-1.0, x[3]))
        for i in range(ns):
            for j in range(ns): P[i, j] -= K[i] * K[j] * S
        for i in range(ns):
            for j in range(ns): P[i, j] = 0.5 * (P[i, j] + P[j, i])
            P[i, i] = min(1.0, max(1e-12, P[i, i]))
        soc_out[k] = x[0]; vhat[k] = zmean; innov[k] = residual; states[k] = x; ptrace[k] = np.trace(P)
    if failed >= 0:
        for k in range(failed, n):
            soc_out[k] = np.nan; vhat[k] = np.nan; innov[k] = np.nan; ptrace[k] = np.nan; r_used[k] = np.nan
            for j in range(4): states[k, j] = np.nan
    return soc_out, vhat, innov, states, ptrace, r_used, failed


def run_v2_ukf(trajectory: Trajectory, holdout: str, ocv: OCVGrid, pmap: FoldParameterMap,
               initial_soc: float, noise: dict[str, float], gamma: float, flags: dict,
               slope_cfg: dict, initial_h: float = 0.0) -> V2FilterResult:
    arrays = pmap.arrays(holdout, trajectory.temperature_series_C, bool(flags["f1_fold_training_ecm"]))
    r0, r1, r2, tau1, tau2 = arrays
    n = len(trajectory.time_s); ns = 4; alpha = 0.1; beta = 2.0; lam = alpha**2 * ns - ns; scale = ns + lam
    wm = np.full(2 * ns + 1, 1 / (2 * scale)); wc = wm.copy(); wm[0] = lam / scale; wc[0] = wm[0] + 1 - alpha**2 + beta
    x = np.array([np.clip(initial_soc, 0, 1), 0.0, 0.0, np.clip(initial_h, -1, 1)], float)
    P = np.diag([2.5e-3, 2.5e-3, 2.5e-3, 0.25])
    Q = np.diag([noise["q_soc"], noise["q_vp"], noise["q_vp"], noise["q_h"]])
    soc_out = np.empty(n); vhat = np.empty(n); innov = np.empty(n); states = np.empty((n, ns)); ptrace = np.empty(n); r_used = np.empty(n)
    diverged = False; reason = ""

    def sigma_points(mean, covariance):
        jitter = 1e-12
        for _ in range(8):
            try:
                root = np.linalg.cholesky(project_psd(covariance, 1e-12, 1.0) * scale + jitter * np.eye(ns))
                return np.vstack([mean, mean + root.T, mean - root.T])
            except np.linalg.LinAlgError:
                jitter *= 10
        raise np.linalg.LinAlgError("sigma point Cholesky failed")

    s_ref = float(slope_cfg["s_ref_mV_per_pctSOC"]) * 0.1
    s_min = float(slope_cfg["s_min_mV_per_pctSOC"]) * 0.1
    for k in range(n):
        p = ECMParameters(float(r0[k]), float(r1[k]), float(tau1[k] / r1[k]), float(r2[k]), float(tau2[k] / r2[k]), float(gamma))
        temp = float(trajectory.temperature_series_C[k])
        try:
            sig = sigma_points(x, P)
            if k > 0:
                sig = np.array([propagate_state(s, float(trajectory.current_A[k - 1]), float(trajectory.dt_s[k]), trajectory.q_ref_Ah, p, True) for s in sig])
                x = np.sum(wm[:, None] * sig, axis=0); x[0] = np.clip(x[0], 0, 1); x[3] = np.clip(x[3], -1, 1)
                dev = sig - x; P = project_psd(np.einsum("i,ij,ik->jk", wc, dev, dev) + Q, 1e-12, 1.0); sig = sigma_points(x, P)
            zsig = np.array([
                ocv.evaluate("base", float(s[0]), temp) + ocv.evaluate("hmag", float(s[0]), temp) * float(s[3])
                - float(trajectory.current_A[k]) * p.R0 - float(s[1]) - float(s[2])
                for s in sig
            ])
            zmean = float(np.sum(wm * zsig)); dz = zsig - zmean; dx = sig - x
            factor = 1.0
            if flags["f2_slope_aware_measurement_noise"]:
                slope = abs(ocv.evaluate("discharge_slope", float(x[0]), temp))
                factor = float(np.clip((s_ref / max(slope, s_min)) ** 2, slope_cfg["factor_min"], slope_cfg["factor_max"]))
            R = float(noise["r_voltage"]) * factor; r_used[k] = R
            S = float(np.sum(wc * dz * dz) + R); cross = np.sum(wc[:, None] * dx * dz[:, None], axis=0)
            K = cross / max(S, 1e-12); residual = float(trajectory.voltage_V[k] - zmean)
            x = x + K * residual; x[0] = np.clip(x[0], 0, 1); x[3] = np.clip(x[3], -1, 1)
            P = project_psd(P - np.outer(K, K) * S, 1e-12, 1.0)
            states[k] = x; soc_out[k] = x[0]; vhat[k] = zmean; innov[k] = residual; ptrace[k] = np.trace(P)
        except Exception as exc:
            diverged = True; reason = f"{type(exc).__name__} at index {k}: {exc}"
            soc_out[k:] = np.nan; vhat[k:] = np.nan; innov[k:] = np.nan; states[k:] = np.nan; ptrace[k:] = np.nan; r_used[k:] = np.nan
            break
    return V2FilterResult(soc_out, vhat, innov, states, ptrace, r_used, diverged, reason)
