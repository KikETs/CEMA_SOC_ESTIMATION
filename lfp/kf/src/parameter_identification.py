from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def discover_hppc(root: str | Path) -> list[tuple[float, Path]]:
    rows = []
    for path in Path(root).rglob("*HPPC*.xlsx"):
        text = str(path)
        match = re.search(r"/(5|25|45)(?: Degree| DEGREE)/", text)
        if match and "DST FUDS UDDS HPPC WLTS US06" in path.name:
            rows.append((float(match.group(1)), path))
    rows.sort()
    if [t for t, _ in rows] != [5.0, 25.0, 45.0]:
        raise RuntimeError(f"Expected independent HPPC at 5/25/45 C, got {rows}")
    return rows


def load_hppc(path: str | Path, current_scale: float = 0.001) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="HPPC")
    cols = list(raw.columns)
    out = pd.DataFrame({
        "time_s": pd.to_numeric(raw[cols[0]], errors="coerce"),
        # Raw HPPC current is negative on discharge; convert mA to discharge-positive A.
        "current_A": -pd.to_numeric(raw[cols[1]], errors="coerce") * current_scale,
        "voltage_V": pd.to_numeric(raw[cols[2]], errors="coerce"),
        "capacity_mAh": pd.to_numeric(raw[cols[3]], errors="coerce"),
    }).dropna(subset=["time_s", "current_A", "voltage_V"]).reset_index(drop=True)
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False]) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def _two_exp(theta: np.ndarray, t: np.ndarray) -> np.ndarray:
    a1, a2, tau1, tau2, offset = theta
    return offset + a1*np.exp(-t/tau1) + a2*np.exp(-t/tau2)


def identify_hppc_temperature(frame: pd.DataFrame, temperature_C: float, cfg: dict) -> tuple[dict, pd.DataFrame]:
    time = frame.time_s.to_numpy(float); current = frame.current_A.to_numpy(float); voltage = frame.voltage_V.to_numpy(float)
    active = np.abs(current) > float(cfg["active_threshold_raw"]) * float(cfg["current_scale_A_per_raw"])
    events = []
    for start, end in _runs(active):
        duration = float(time[end-1] - time[start] + np.median(np.diff(time)))
        if duration < 5 or duration > 45 or start < 10 or end >= len(frame)-10:
            continue
        next_active = np.flatnonzero(active[end:])
        rest_end = len(frame) if len(next_active) == 0 else end + int(next_active[0])
        rest_duration = float(time[rest_end-1] - time[end]) if rest_end > end else 0.0
        if rest_duration < float(cfg["minimum_rest_s"]):
            continue
        i_pulse = float(np.median(current[start:end]))
        if abs(i_pulse) < 0.2:
            continue
        v_before = float(np.median(voltage[max(0,start-10):start]))
        r0 = -(float(voltage[start]) - v_before) / i_pulse
        stop = min(rest_end, end + 181)
        tr = time[end:stop] - time[end]
        vr = voltage[end:stop]
        if len(tr) < 40:
            continue
        offset0 = float(np.median(vr[-20:])); y0 = float(vr[0] - offset0)
        x0 = np.array([0.6*y0, 0.4*y0, 5.0, 80.0, offset0])
        lo = np.array([-1.0, -1.0, cfg["tau1_bounds_s"][0], cfg["tau2_bounds_s"][0], 1.5])
        hi = np.array([1.0, 1.0, cfg["tau1_bounds_s"][1], cfg["tau2_bounds_s"][1], 4.2])
        fit = least_squares(lambda z: _two_exp(z, tr) - vr, x0=x0, bounds=(lo, hi), max_nfev=300)
        a1, a2, tau1, tau2, offset = fit.x
        if tau1 > tau2:
            a1, a2, tau1, tau2 = a2, a1, tau2, tau1
        r1 = abs(a1) / (abs(i_pulse) * max(1-np.exp(-duration/tau1), 1e-4))
        r2 = abs(a2) / (abs(i_pulse) * max(1-np.exp(-duration/tau2), 1e-4))
        if not (cfg["resistance_bounds_ohm"][0] <= r0 <= cfg["resistance_bounds_ohm"][1]):
            continue
        if not (cfg["resistance_bounds_ohm"][0] <= r1 <= cfg["resistance_bounds_ohm"][1]):
            continue
        if not (cfg["resistance_bounds_ohm"][0] <= r2 <= cfg["resistance_bounds_ohm"][1]):
            continue
        events.append({
            "temperature_C": temperature_C, "start_index": start, "pulse_duration_s": duration,
            "rest_duration_s": rest_duration, "current_A": i_pulse, "R0": r0, "R1": r1,
            "tau1": tau1, "R2": r2, "tau2": tau2, "fit_rmse_V": float(np.sqrt(np.mean(fit.fun**2))),
            "source": "independent_HPPC",
        })
    event_frame = pd.DataFrame(events)
    if len(event_frame) < 5:
        raise RuntimeError(f"Too few usable HPPC pulses at {temperature_C} C: {len(event_frame)}")
    # Robust aggregation limits a few imperfect pulse/rest fits without using drive-profile data.
    med = event_frame[["R0","R1","tau1","R2","tau2"]].median()
    if med.tau1 > med.tau2:
        med.R1, med.R2, med.tau1, med.tau2 = med.R2, med.R1, med.tau2, med.tau1
    summary = {
        "temperature_C": temperature_C, "R0": float(med.R0), "R1": float(med.R1),
        "C1": float(med.tau1/med.R1), "R2": float(med.R2), "C2": float(med.tau2/med.R2),
        "tau1": float(med.tau1), "tau2": float(med.tau2), "n_pulses": int(len(event_frame)),
        "source": "independent_HPPC",
    }
    return summary, event_frame


def identify_parameter_map(root: str | Path, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, events = [], []
    for temp, path in discover_hppc(root):
        summary, event = identify_hppc_temperature(load_hppc(path, config["current_scale_A_per_raw"]), temp, config)
        summary["source_file"] = str(path); event["source_file"] = str(path)
        summaries.append(summary); events.append(event)
    return pd.DataFrame(summaries), pd.concat(events, ignore_index=True)

