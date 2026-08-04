from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Trajectory:
    path: Path
    file_name: str
    profile: str
    temperature_C: float
    time_s: np.ndarray
    dt_s: np.ndarray
    voltage_V: np.ndarray
    current_discharge_A: np.ndarray
    reference_soc: np.ndarray
    initial_rest_voltage_V: float

    @property
    def elapsed_s(self) -> np.ndarray:
        return self.time_s - float(self.time_s[0])


@dataclass(frozen=True)
class FitTrajectory:
    path: Path
    file_name: str
    profile: str
    temperature_C: float
    time_s: np.ndarray
    dt_s: np.ndarray
    voltage_V: np.ndarray
    current_discharge_A: np.ndarray
    initial_rest_voltage_V: float

    @property
    def elapsed_s(self) -> np.ndarray:
        return self.time_s - float(self.time_s[0])


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_dynamic_files(data_root: Path, profiles: list[str], temperatures: list[float]) -> list[Path]:
    expected = {
        f"NMC_{int(temp)}C_{profile}.csv"
        for temp in temperatures
        for profile in profiles
    }
    found = {path.name: path.resolve() for path in data_root.glob("*C/NMC_*C_*.csv") if path.name in expected}
    missing = sorted(expected - set(found))
    if missing:
        raise FileNotFoundError(f"Missing protocol files: {missing}")
    return [found[name] for name in sorted(expected)]


def _temperature_from_path(path: Path) -> float:
    return float(path.parent.name.upper().removesuffix("C"))


def load_trajectory(path: Path, current_multiplier: float = -1.0) -> Trajectory:
    frame = pd.read_csv(path)
    required = ["Test_Time(s)", "Voltage(V)", "Current(A)", "SOC_CC", "Profile", "SOC0_Vinit(V)"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    time_s = pd.to_numeric(frame["Test_Time(s)"], errors="coerce").to_numpy(np.float64)
    voltage = pd.to_numeric(frame["Voltage(V)"], errors="coerce").to_numpy(np.float64)
    current = current_multiplier * pd.to_numeric(frame["Current(A)"], errors="coerce").to_numpy(np.float64)
    reference = pd.to_numeric(frame["SOC_CC"], errors="coerce").to_numpy(np.float64)
    if np.nanmax(reference) > 1.5:
        reference = reference / 100.0
    valid = np.isfinite(time_s) & np.isfinite(voltage) & np.isfinite(current) & np.isfinite(reference)
    if not np.all(valid):
        time_s, voltage, current, reference = time_s[valid], voltage[valid], current[valid], reference[valid]
    delta = np.diff(time_s)
    positive = delta[np.isfinite(delta) & (delta > 0)]
    fallback = float(np.median(positive)) if len(positive) else 1.0
    dt = np.diff(time_s, prepend=time_s[0] - fallback)
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, fallback)
    profile = str(frame["Profile"].iloc[0]).upper()
    initial_rest_voltage = float(pd.to_numeric(frame["SOC0_Vinit(V)"], errors="coerce").iloc[0])
    return Trajectory(
        path=path.resolve(),
        file_name=path.name,
        profile=profile,
        temperature_C=_temperature_from_path(path),
        time_s=time_s,
        dt_s=dt,
        voltage_V=voltage,
        current_discharge_A=current,
        reference_soc=np.clip(reference, 0.0, 1.0),
        initial_rest_voltage_V=initial_rest_voltage,
    )


def as_fit_trajectory(trajectory: Trajectory) -> FitTrajectory:
    return FitTrajectory(
        path=trajectory.path,
        file_name=trajectory.file_name,
        profile=trajectory.profile,
        temperature_C=trajectory.temperature_C,
        time_s=trajectory.time_s,
        dt_s=trajectory.dt_s,
        voltage_V=trajectory.voltage_V,
        current_discharge_A=trajectory.current_discharge_A,
        initial_rest_voltage_V=trajectory.initial_rest_voltage_V,
    )


def evaluation_mask(length: int, first_end_index: int = 49, stride: int = 1) -> np.ndarray:
    mask = np.zeros(int(length), dtype=bool)
    mask[int(first_end_index) :: int(stride)] = True
    return mask
