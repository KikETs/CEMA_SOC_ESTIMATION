from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


@dataclass(frozen=True)
class ECMParameters:
    R0: float
    R1: float
    C1: float
    R2: float
    C2: float
    gamma: float

    @property
    def tau1(self) -> float:
        return self.R1 * self.C1

    @property
    def tau2(self) -> float:
        return self.R2 * self.C2


class TemperatureParameterMap:
    def __init__(self, table: pd.DataFrame):
        required = {"temperature_C", "R0", "R1", "C1", "R2", "C2"}
        if required - set(table.columns):
            raise ValueError(f"Parameter table missing {sorted(required-set(table.columns))}")
        self.table = table.sort_values("temperature_C").reset_index(drop=True)

    def lookup(self, temperature_C: float, gamma: float) -> ECMParameters:
        t = self.table.temperature_C.to_numpy(float)
        x = float(np.clip(temperature_C, t.min(), t.max()))
        values = {name: float(np.interp(x, t, self.table[name].to_numpy(float))) for name in ("R0", "R1", "C1", "R2", "C2")}
        return ECMParameters(**values, gamma=float(gamma))


class OCVMap:
    def __init__(self, table: pd.DataFrame, variant: str = "monotonic"):
        self.table = table.copy()
        self.variant = variant
        self.temperatures = np.sort(table.temperature_C.unique().astype(float))
        self._curves = {}
        for temp in self.temperatures:
            d = table[table.temperature_C == temp].sort_values("soc")
            self._curves[float(temp)] = {
                name: PchipInterpolator(d.soc, d[name], extrapolate=False)
                for name in (f"ocv_base_{variant}_V", f"hysteresis_half_{variant}_V")
            }

    def _at_temp(self, soc: np.ndarray, temp: float, name: str, derivative: bool = False) -> np.ndarray:
        curve = self._curves[float(temp)][f"{name}_{self.variant}_V"]
        return np.asarray(curve.derivative()(soc) if derivative else curve(soc), dtype=float)

    def evaluate(self, soc, temperature_C, field: str = "ocv_base", derivative: bool = False):
        s = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        original_shape = s.shape
        s = np.atleast_1d(s)
        temp = np.broadcast_to(np.asarray(temperature_C, dtype=float), original_shape or ()).reshape(-1)
        sflat = s.reshape(-1)
        out = np.empty_like(sflat)
        for idx, (sv, tv) in enumerate(zip(sflat, temp)):
            tv = float(np.clip(tv, self.temperatures[0], self.temperatures[-1]))
            hi = int(np.searchsorted(self.temperatures, tv))
            if hi == 0:
                lo_t = hi_t = float(self.temperatures[0]); w = 0.0
            elif hi == len(self.temperatures):
                lo_t = hi_t = float(self.temperatures[-1]); w = 0.0
            else:
                lo_t, hi_t = float(self.temperatures[hi-1]), float(self.temperatures[hi]); w = (tv-lo_t)/(hi_t-lo_t)
            lo = self._at_temp(np.array([sv]), lo_t, field, derivative)[0]
            hi_v = lo if hi_t == lo_t else self._at_temp(np.array([sv]), hi_t, field, derivative)[0]
            out[idx] = (1-w)*lo + w*hi_v
        out = out.reshape(original_shape)
        return float(out) if out.ndim == 0 else out

    def ocv(self, soc, temperature_C):
        return self.evaluate(soc, temperature_C, "ocv_base")

    def docv_dsoc(self, soc, temperature_C):
        return self.evaluate(soc, temperature_C, "ocv_base", True)

    def hmag(self, soc, temperature_C):
        return np.maximum(self.evaluate(soc, temperature_C, "hysteresis_half"), 0.0)

    def dhmag_dsoc(self, soc, temperature_C):
        return self.evaluate(soc, temperature_C, "hysteresis_half", True)


def _branch_soc(frame: pd.DataFrame, step: int) -> tuple[np.ndarray, np.ndarray]:
    d = frame[frame.Step_Index == step].copy()
    if step == 5:
        q = d["Discharge_Capacity(Ah)"].to_numpy(float)
        q = q - np.nanmin(q); soc = 1.0 - q / max(np.nanmax(q), 1e-12)
    elif step == 7:
        q = d["Charge_Capacity(Ah)"].to_numpy(float)
        q = q - np.nanmin(q); soc = q / max(np.nanmax(q), 1e-12)
    else:
        raise ValueError(step)
    return np.clip(soc, 0.0, 1.0), d["Voltage(V)"].to_numpy(float)


def build_ocv_table(ocv_root: str | Path, grid_points: int = 1001) -> pd.DataFrame:
    rows = []
    grid = np.linspace(0.0, 1.0, int(grid_points))
    for path in sorted(Path(ocv_root).glob("LFP_OCV_*.csv"), key=lambda p: float(p.stem.split("_")[-1])):
        temp = float(path.stem.split("_")[-1]); raw = pd.read_csv(path)
        sd, vd = _branch_soc(raw, 5); sc, vc = _branch_soc(raw, 7)
        order_d, order_c = np.argsort(sd), np.argsort(sc)
        vd_raw = np.interp(grid, sd[order_d], vd[order_d]); vc_raw = np.interp(grid, sc[order_c], vc[order_c])
        base_raw = 0.5 * (vd_raw + vc_raw); h_raw = 0.5 * np.abs(vc_raw - vd_raw)
        vd_mon = np.maximum.accumulate(vd_raw); vc_mon = np.maximum.accumulate(vc_raw)
        base_mon = np.maximum.accumulate(0.5 * (vd_mon + vc_mon))
        h_mon = pd.Series(0.5 * np.abs(vc_mon - vd_mon)).rolling(21, center=True, min_periods=1).median().to_numpy()
        for idx, soc in enumerate(grid):
            rows.append({
                "temperature_C": temp, "soc": soc,
                "ocv_discharge_raw_V": vd_raw[idx], "ocv_charge_raw_V": vc_raw[idx],
                "ocv_base_raw_V": base_raw[idx], "hysteresis_half_raw_V": h_raw[idx],
                "ocv_base_monotonic_V": base_mon[idx], "hysteresis_half_monotonic_V": max(h_mon[idx], 0.0),
                "source_file": str(path),
            })
    return pd.DataFrame(rows)


def propagate_state(x: np.ndarray, current_A: float, dt_s: float, q_ref_Ah: float, p: ECMParameters, with_hysteresis: bool) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    x[0] = np.clip(x[0] - current_A * dt_s / (3600.0 * q_ref_Ah), 0.0, 1.0)
    a1 = np.exp(-dt_s / max(p.tau1, dt_s + 1e-9)); a2 = np.exp(-dt_s / max(p.tau2, dt_s + 1e-9))
    x[1] = a1*x[1] + (1-a1)*p.R1*current_A
    x[2] = a2*x[2] + (1-a2)*p.R2*current_A
    if with_hysteresis:
        from .hysteresis_model import propagate_hysteresis
        x[3], _ = propagate_hysteresis(x[3], current_A, dt_s, q_ref_Ah, p.gamma)
    return x


def terminal_voltage(x: np.ndarray, current_A: float, temperature_C: float, p: ECMParameters, ocv: OCVMap, with_hysteresis: bool) -> float:
    value = float(ocv.ocv(x[0], temperature_C)) - current_A*p.R0 - x[1] - x[2]
    if with_hysteresis:
        value += float(ocv.hmag(x[0], temperature_C)) * x[3]
    return value
