from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import IsotonicRegression


def _contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _read_measurement_sheets(path: Path) -> pd.DataFrame:
    excel = pd.ExcelFile(path)
    frames = []
    for sheet in excel.sheet_names:
        try:
            frame = pd.read_excel(path, sheet_name=sheet)
        except Exception:
            continue
        normalized = {str(column).strip().lower() for column in frame.columns}
        if normalized & {"voltage(v)", "mv"} and normalized & {"current(a)", "ma"}:
            frames.append(frame)
    if not frames:
        raise ValueError(f"No measurement sheet found in {path}")
    return pd.concat(frames, ignore_index=True)


def _find_column(columns, candidates: list[str]) -> str | None:
    lookup = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def extract_low_current_curve(path: Path, temperature_C: float, q_ref_Ah: float) -> tuple[np.ndarray, np.ndarray, dict]:
    frame = _read_measurement_sheets(path)
    voltage_col = _find_column(frame.columns, ["Voltage(V)", "mV"])
    current_col = _find_column(frame.columns, ["Current(A)", "mA"])
    time_col = _find_column(frame.columns, ["Test_Time(s)", "Duration (sec)"])
    step_col = _find_column(frame.columns, ["Step_Index", "Pgm step"])
    discharge_col = _find_column(frame.columns, ["Discharge_Capacity(Ah)"])
    if voltage_col is None or current_col is None or time_col is None:
        raise ValueError(f"Missing low-current OCV columns in {path}")
    voltage = pd.to_numeric(frame[voltage_col], errors="coerce").to_numpy(np.float64)
    current = pd.to_numeric(frame[current_col], errors="coerce").to_numpy(np.float64)
    time_s = pd.to_numeric(frame[time_col], errors="coerce").to_numpy(np.float64)
    if str(voltage_col).strip().lower() == "mv":
        voltage = voltage / 1000.0
    if str(current_col).strip().lower() == "ma":
        current = current / 1000.0
    valid = np.isfinite(voltage) & np.isfinite(current) & np.isfinite(time_s)
    frame, voltage, current, time_s = frame.loc[valid].copy(), voltage[valid], current[valid], time_s[valid]

    if step_col is not None and discharge_col is not None:
        frame["_voltage"] = voltage
        frame["_current"] = current
        best = None
        best_capacity = -np.inf
        for _, group in frame.groupby(pd.to_numeric(frame[step_col], errors="coerce")):
            group_current = pd.to_numeric(group["_current"], errors="coerce")
            if len(group) < 100 or float(group_current.mean()) >= -0.01:
                continue
            capacity = pd.to_numeric(group[discharge_col], errors="coerce")
            span = float(capacity.max() - capacity.min())
            if span > best_capacity:
                best, best_capacity = group, span
        if best is None:
            raise ValueError(f"No low-current discharge step in {path}")
        q_removed = pd.to_numeric(best[discharge_col], errors="coerce").to_numpy(np.float64)
        q_removed = q_removed - float(np.nanmin(q_removed))
        curve_voltage = pd.to_numeric(best["_voltage"], errors="coerce").to_numpy(np.float64)
        selection = f"largest negative-current step; capacity span={best_capacity:.6f} Ah"
    else:
        segments = _contiguous_segments(current < -0.01)
        if not segments:
            raise ValueError(f"No low-current discharge segment in {path}")
        start, end = max(segments, key=lambda segment: segment[1] - segment[0])
        curve_voltage = voltage[start:end]
        segment_time = time_s[start:end]
        segment_current = current[start:end]
        delta = np.diff(segment_time, prepend=segment_time[0])
        positive = delta[np.isfinite(delta) & (delta > 0)]
        fallback = float(np.median(positive)) if len(positive) else 1.0
        delta = np.where(np.isfinite(delta) & (delta > 0), delta, fallback)
        q_removed = np.cumsum(np.maximum(-segment_current, 0.0) * delta / 3600.0)
        q_removed = q_removed - float(q_removed[0])
        selection = f"longest negative-current segment; rows={end - start}"

    soc = np.clip(1.0 - q_removed / float(q_ref_Ah), 0.0, 1.0)
    keep = np.isfinite(soc) & np.isfinite(curve_voltage) & (curve_voltage >= 2.40) & (curve_voltage <= 4.30)
    meta = {
        "temperature_C": float(temperature_C),
        "source_file": str(path.resolve()),
        "q_ref_Ah": float(q_ref_Ah),
        "selection": selection,
        "raw_points": int(np.count_nonzero(keep)),
    }
    return soc[keep], curve_voltage[keep], meta


@dataclass
class TemperatureCurve:
    temperature_C: float
    soc: np.ndarray
    voltage: np.ndarray
    source_file: str
    q_ref_Ah: float

    def __post_init__(self):
        self._spline = PchipInterpolator(self.soc, self.voltage, extrapolate=False)
        self._derivative = self._spline.derivative()

    def ocv(self, soc) -> np.ndarray:
        value = self._spline(np.clip(np.asarray(soc, dtype=np.float64), 0.0, 1.0))
        return np.asarray(value, dtype=np.float64)

    def slope(self, soc) -> np.ndarray:
        value = self._derivative(np.clip(np.asarray(soc, dtype=np.float64), 0.0, 1.0))
        return np.asarray(value, dtype=np.float64)

    def inverse(self, voltage_V: float) -> float:
        unique_voltage, unique_index = np.unique(self.voltage, return_index=True)
        unique_soc = self.soc[unique_index]
        return float(np.interp(float(voltage_V), unique_voltage, unique_soc))


class OCVMap:
    def __init__(self, curves: dict[float, TemperatureCurve], slope_min: float, slope_max: float):
        self.curves = dict(sorted(curves.items()))
        self.slope_min = float(slope_min)
        self.slope_max = float(slope_max)

    def _bracket(self, temperature_C: float) -> tuple[TemperatureCurve, TemperatureCurve, float]:
        temperatures = np.asarray(sorted(self.curves), dtype=np.float64)
        value = float(np.clip(temperature_C, temperatures[0], temperatures[-1]))
        upper_index = int(np.searchsorted(temperatures, value, side="left"))
        if upper_index == 0:
            curve = self.curves[float(temperatures[0])]
            return curve, curve, 0.0
        if upper_index >= len(temperatures):
            curve = self.curves[float(temperatures[-1])]
            return curve, curve, 0.0
        lower_temp, upper_temp = float(temperatures[upper_index - 1]), float(temperatures[upper_index])
        weight = (value - lower_temp) / (upper_temp - lower_temp)
        return self.curves[lower_temp], self.curves[upper_temp], float(weight)

    def ocv(self, soc, temperature_C: float) -> np.ndarray:
        lower, upper, weight = self._bracket(temperature_C)
        return (1.0 - weight) * lower.ocv(soc) + weight * upper.ocv(soc)

    def slope(self, soc, temperature_C: float) -> np.ndarray:
        lower, upper, weight = self._bracket(temperature_C)
        slope = (1.0 - weight) * lower.slope(soc) + weight * upper.slope(soc)
        return np.clip(slope, self.slope_min, self.slope_max)

    def inverse(self, voltage_V: float, temperature_C: float) -> float:
        lower, upper, weight = self._bracket(temperature_C)
        if weight == 0.0 or lower is upper:
            return lower.inverse(voltage_V)
        soc_grid = np.linspace(0.0, 1.0, 2001)
        voltage_grid = self.ocv(soc_grid, temperature_C)
        return float(np.interp(float(voltage_V), voltage_grid, soc_grid))

    @property
    def q_ref_by_temperature(self) -> dict[float, float]:
        return {temperature: curve.q_ref_Ah for temperature, curve in self.curves.items()}


def build_ocv_map(protocol: dict, ecm_config: dict) -> tuple[OCVMap, pd.DataFrame, pd.DataFrame]:
    characterization = protocol["characterization"]
    root = Path(characterization["root"])
    capacities = pd.read_csv(characterization["capacity_table"])
    q_lookup = {float(row.temperature_C): float(row.Q_ref_lc_ocv_Ah) for row in capacities.itertuples()}
    grid_points = int(ecm_config["ocv"]["grid_points"])
    curves = {}
    curve_rows = []
    source_rows = []
    for temperature_key, file_name in characterization["ocv_files"].items():
        temperature = float(temperature_key)
        source_path = root / file_name
        raw_soc, raw_voltage, metadata = extract_low_current_curve(source_path, temperature, q_lookup[temperature])
        order = np.argsort(raw_soc)
        raw_soc, raw_voltage = raw_soc[order], raw_voltage[order]
        bins = np.linspace(0.0, 1.0, grid_points)
        bin_id = np.clip(np.digitize(raw_soc, bins) - 1, 0, grid_points - 1)
        grouped = pd.DataFrame({"bin": bin_id, "soc": raw_soc, "voltage": raw_voltage}).groupby("bin", as_index=False).median()
        soc_points = grouped["soc"].to_numpy(np.float64)
        voltage_points = grouped["voltage"].to_numpy(np.float64)
        if soc_points[0] > 0.0:
            soc_points = np.r_[0.0, soc_points]
            voltage_points = np.r_[voltage_points[0], voltage_points]
        if soc_points[-1] < 1.0:
            soc_points = np.r_[soc_points, 1.0]
            voltage_points = np.r_[voltage_points, voltage_points[-1]]
        isotonic = IsotonicRegression(increasing=True, out_of_bounds="clip")
        monotonic_voltage = isotonic.fit_transform(soc_points, voltage_points)
        unique_soc, unique_index = np.unique(soc_points, return_index=True)
        monotonic_voltage = monotonic_voltage[unique_index]
        curve = TemperatureCurve(
            temperature,
            unique_soc,
            monotonic_voltage,
            str(source_path.resolve()),
            q_lookup[temperature],
        )
        curves[temperature] = curve
        for soc_value, voltage_value in zip(curve.soc, curve.voltage):
            curve_rows.append({"temperature_C": temperature, "soc": soc_value, "ocv_V": voltage_value})
        source_rows.append(metadata | {"processed_points": len(curve.soc)})
    ocv_map = OCVMap(
        curves,
        slope_min=float(ecm_config["ocv"]["slope_min_V_per_soc"]),
        slope_max=float(ecm_config["ocv"]["slope_max_V_per_soc"]),
    )
    return ocv_map, pd.DataFrame(curve_rows), pd.DataFrame(source_rows)
