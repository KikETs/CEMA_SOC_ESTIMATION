from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import least_squares

from .hysteresis_model import propagate_hysteresis
from .v2_core import OCVGrid, _unit_rc_state


@njit(cache=True)
def _h_state(current, dt_s, q_ref, gamma, h0):
    out = np.empty(len(current)); out[0] = min(1.0, max(-1.0, h0))
    for k in range(1, len(out)):
        ip = current[k - 1]
        alpha = np.exp(-max(gamma, 0.0) * abs(ip) * max(dt_s[k], 0.0) / (3600.0 * max(q_ref, 1e-12)))
        target = out[k - 1] if abs(ip) < 1e-12 else -np.sign(ip)
        out[k] = min(1.0, max(-1.0, alpha * out[k - 1] + (1.0 - alpha) * target))
    return out


def fit_hysteresis_ecm_by_fold(trajectories, profiles, r0_table, ocv: OCVGrid, cfg: dict) -> pd.DataFrame:
    """Training-only fold x temperature 2RC fit with jointly fitted hysteresis gamma."""
    rows = []
    temps = sorted({float(tr.temperature_C) for tr in trajectories})
    rb1, rb2 = cfg["r1_bounds_ohm"], cfg["r2_bounds_ohm"]
    tb1, tb2 = cfg["tau1_bounds_s"], cfg["tau2_bounds_s"]
    gb = cfg["gamma_bounds"]
    for holdout in profiles:
        for temp in temps:
            records = [tr for tr in trajectories if tr.profile != holdout and float(tr.temperature_C) == temp]
            if len(records) != len(profiles) - 1:
                raise RuntimeError(f"Expected two training records for {holdout}/{temp:g}C")
            r0 = float(r0_table.loc[(r0_table.fold_holdout == holdout) & (r0_table.temperature_C == temp), "R0_ohm"].iloc[0])
            prepared = []
            for tr in records:
                base = np.array([ocv.evaluate("base", s, t) for s, t in zip(tr.soc_ref, tr.temperature_series_C)])
                mag = np.array([ocv.evaluate("hmag", s, t) for s, t in zip(tr.soc_ref, tr.temperature_series_C)])
                prepared.append((tr, base, mag, float(np.median(tr.dt_s))))

            lower = np.log([rb1[0], rb2[0], tb1[0], tb2[0], gb[0]])
            upper = np.log([rb1[1], rb2[1], tb1[1], tb2[1], gb[1]])
            x0 = 0.5 * (lower + upper)

            def record_residual(theta, item):
                tr, base, mag, dt = item
                r1, r2, tau1, tau2, gamma = np.exp(theta)
                h = _h_state(tr.current_A, tr.dt_s, tr.q_ref_Ah, gamma, ocv.initial_h(tr, float(tr.soc_ref[0])))
                vp1 = r1 * _unit_rc_state(tr.current_A, dt, tau1)
                vp2 = r2 * _unit_rc_state(tr.current_A, dt, tau2)
                predicted = base + mag * h - tr.current_A * r0 - vp1 - vp2
                return predicted - tr.voltage_V

            fit = least_squares(lambda x: np.concatenate([record_residual(x, item) for item in prepared]), x0, bounds=(lower, upper), max_nfev=int(cfg["max_nfev"]), method="trf")
            r1, r2, tau1, tau2, gamma = map(float, np.exp(fit.x))
            if tau1 > tau2:
                r1, r2, tau1, tau2 = r2, r1, tau2, tau1
            ratio = tau1 / tau2
            for item in prepared:
                tr = item[0]; residual = record_residual(fit.x, item)
                mean_mv = float(1000 * np.mean(residual)); rmse_mv = float(1000 * np.sqrt(np.mean(residual ** 2)))
                rows.append({
                    "fold_holdout": holdout, "temperature_C": temp, "training_profile": tr.profile,
                    "training_file": str(tr.path), "R0_ohm": r0, "R1_ohm": r1, "R2_ohm": r2,
                    "tau1_s": tau1, "tau2_s": tau2, "C1_F": tau1 / r1, "C2_F": tau2 / r2,
                    "gamma": gamma, "residual_mean_mV": mean_mv, "voltage_fit_RMSE_mV": rmse_mv,
                    "rmse_gt_10mV": rmse_mv > float(cfg["rmse_flag_mV"]),
                    "tau1_tau2_ratio": ratio,
                    "tau1_much_less_than_tau2": ratio <= float(cfg["tau_separation_ratio_max"]),
                    "fit_success": bool(fit.success), "fit_nfev": int(fit.nfev),
                    "training_profiles": "+".join(p for p in profiles if p != holdout),
                    "training_files": ";".join(sorted(str(x[0].path) for x in prepared)),
                })
    result = pd.DataFrame(rows)
    if len(result) != 2 * len(profiles) * len(temps):
        raise RuntimeError(f"Expected 48 per-record ECM quality rows, found {len(result)}")
    return result


def a1_post_rest_start(trajectories, ocv: OCVGrid) -> pd.DataFrame:
    rows = []
    for tr in trajectories:
        load = np.flatnonzero(np.abs(tr.current_A) >= 0.05)
        first_load = int(load[0]) if len(load) else len(tr.current_A)
        rests = np.flatnonzero((np.arange(len(tr.current_A)) < first_load) & (np.abs(tr.current_A) < 0.05))
        if abs(float(tr.current_A[0])) < 0.05:
            idx, rule = 0, "V[0]_post_rest_start"
        elif len(rests):
            idx, rule = int(rests[0]), "first_preload_rest_sample"
        else:
            idx, rule = 0, "fallback_first_sample_no_preload_rest"
        soc = float(tr.soc_ref[idx]); temp = float(tr.temperature_series_C[idx])
        discharge = ocv.evaluate("base", soc, temp) - ocv.evaluate("hmag", soc, temp)
        rows.append({
            "profile": tr.profile, "temperature_C": tr.temperature_C, "source_file": str(tr.path),
            "sample_index": idx, "selection_rule": rule, "time_s": float(tr.time_s[idx]),
            "current_A": float(tr.current_A[idx]), "SOC_label": soc, "V_start_V": float(tr.voltage_V[idx]),
            "OCV_discharge_V": discharge, "residual_mV": 1000 * (float(tr.voltage_V[idx]) - discharge),
        })
    return pd.DataFrame(rows)
