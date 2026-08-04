#!/usr/bin/env python3
"""Adapt the paper A1-007 raw Excel files to the frozen LFP CSV interface."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


TEMPERATURES = {
    -10.0: ("N10", "A1-007-DST-US06-FUDS-N10-20120829.xlsx", "A1-007-OCV-10-20120629.xlsx"),
    0.0: ("0", "A1-007-DST-US06-FUDS-0-20120813.xlsx", "A1-007-OCV0-20120618.xlsx"),
    10.0: ("10", "A1-007-DST-US06-FUDS-10-20120815.xlsx", "A1-007-OCV10-20120611.xlsx"),
    20.0: ("20", "A1-007-DST-US06-FUDS-20-20120817.xlsx", "A1-007-OCV20-20120614.xlsx"),
    25.0: ("25", "A1-007-DST-US06-FUDS-25-20120827.xlsx", "A1-007-OCV-25-20120905.xlsx"),
    30.0: ("30", "A1-007-DST-US06-FUDS-30-20120820.xlsx", "A1-007-OCV30-20120625.xlsx"),
    40.0: ("40", "A1-007-DST-US06-FUDS-40-20120822.xlsx", "A1-007-OCV40-20120627.xlsx"),
    50.0: ("50", "A1-007-DST-US06-FUDS-50-20120824.xlsx", "A1-007-OCV50-20120702.xlsx"),
}
PROFILE_STEPS = {"DST": 8, "US06": 16, "FUDS": 24}


def round_significant_10_half_up(value: float) -> float:
    if not np.isfinite(value) or value == 0.0:
        return value
    rounded = float(f"{value:.10g}")
    scale = 10.0 ** (9 - np.floor(np.log10(abs(value))))
    scaled = abs(value) * scale
    fraction = scaled - np.floor(scaled)
    if abs(fraction - 0.5) <= 1e-5:
        return float(np.sign(value) * (np.floor(scaled) + 1.0) / scale)
    return rounded


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def measurement_sheet(path: Path) -> str:
    sheets = [name for name in pd.ExcelFile(path).sheet_names if name.startswith("Channel_")]
    if len(sheets) != 1:
        raise RuntimeError(f"Expected one Channel_ sheet in {path.name}, found {sheets}")
    return sheets[0]


def read_measurements(path: Path) -> tuple[pd.DataFrame, str]:
    sheet = measurement_sheet(path)
    frame = pd.read_excel(path, sheet_name=sheet)
    required = {
        "Test_Time(s)", "Step_Time(s)", "Step_Index", "Current(A)", "Voltage(V)",
        "Charge_Capacity(Ah)", "Discharge_Capacity(Ah)", "Temperature (C)_1",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Missing columns {missing} in {path.name}:{sheet}")
    return frame, sheet


def historical_export_precision(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    test_time = pd.to_numeric(out["Test_Time(s)"], errors="coerce")
    out["Test_Time(s)"] = test_time.map(round_significant_10_half_up)
    step_time = pd.to_numeric(out["Step_Time(s)"], errors="coerce")
    out["Step_Time(s)"] = step_time.map(
        lambda value: float(f"{value:.10g}") if np.isfinite(value) else value
    )
    temperature = pd.to_numeric(out["Temperature (C)_1"], errors="coerce").to_numpy(float)
    low_magnitude = np.abs(temperature) < 10.0
    temperature[low_magnitude] = np.round(temperature[low_magnitude], 9)
    temperature[~low_magnitude] = np.asarray(
        [round_significant_10_half_up(value) for value in temperature[~low_magnitude]],
        dtype=float,
    )
    out["Temperature (C)_1"] = temperature
    for column in ("Voltage(V)", "Charge_Energy(Wh)", "Discharge_Energy(Wh)"):
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce").round(9)
    for column in ("Current(A)", "Charge_Capacity(Ah)", "Discharge_Capacity(Ah)"):
        if column not in out:
            continue
        values = pd.to_numeric(out[column], errors="coerce").to_numpy(float)
        if column == "Current(A)":
            rounded = np.round(values, 9)
        else:
            scaled = np.abs(values) * 1e9
            rounded = np.sign(values) * np.floor(scaled + 0.5 + 1e-5) / 1e9
        near_zero = np.abs(values) < 1e-4
        rounded[near_zero] = np.asarray(
            [float(f"{value:.6g}") for value in values[near_zero]], dtype=float
        )
        out[column] = rounded
    return out


def adapt_ocv(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    step = pd.to_numeric(out["Step_Index"], errors="raise").astype(int)
    out["label"] = "charge"
    out.loc[step.isin((5, 6)), "label"] = "discharge_ocv"
    out.loc[step.isin((7, 8)), "label"] = "charge_ocv"
    out["abs_charge_ah"] = np.nan
    out["abs_discharge_ah"] = np.nan
    charge = pd.to_numeric(out["Charge_Capacity(Ah)"], errors="raise")
    discharge = pd.to_numeric(out["Discharge_Capacity(Ah)"], errors="raise")
    initial_charge = out["label"].eq("charge")
    charge_ocv = out["label"].eq("charge_ocv")
    discharge_ocv = out["label"].eq("discharge_ocv")
    out.loc[initial_charge, "abs_charge_ah"] = charge[initial_charge]
    out.loc[charge_ocv, "abs_charge_ah"] = charge[charge_ocv] - float(charge[charge_ocv].min())
    out.loc[discharge_ocv, "abs_discharge_ah"] = (
        discharge[discharge_ocv] - float(discharge[discharge_ocv].min())
    )
    return out


def run(profile_root: Path, ocv_root: Path, output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dynamic_root = output_root / "raw_adapter"
    staged_ocv_root = output_root / "ocv_csv"
    dynamic_root.mkdir(parents=True, exist_ok=True)
    staged_ocv_root.mkdir(parents=True, exist_ok=True)
    success: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for temp, (dirname, profile_name, ocv_name) in TEMPERATURES.items():
        profile_path = profile_root / profile_name
        try:
            frame, sheet = read_measurements(profile_path)
            frame = historical_export_precision(frame)
            for profile, step_index in PROFILE_STEPS.items():
                selected = frame[pd.to_numeric(frame["Step_Index"], errors="coerce").eq(step_index)].copy()
                if len(selected) < 50:
                    raise RuntimeError(f"Step {step_index} ({profile}) has only {len(selected)} rows")
                output_dir = dynamic_root / dirname
                output_dir.mkdir(parents=True, exist_ok=True)
                output = output_dir / f"LFP_{dirname}_{profile}_usableSOC.csv"
                selected.to_csv(output, index=False, float_format="%.10g", lineterminator="\n")
                success.append({
                    "chemistry": "LFP", "stage": "profile_excel_adapter", "temperature_C": temp,
                    "profile": profile, "source_file": str(profile_path.resolve()), "source_sheet": sheet,
                    "source_sha256": sha256_file(profile_path), "output_file": str(output.resolve()),
                    "output_sha256": sha256_file(output), "rows": len(selected), "status": "PASS",
                })
        except Exception as exc:
            failures.append({
                "chemistry": "LFP", "stage": "profile_excel_adapter", "temperature_C": temp,
                "profile": "ALL", "source_file": str(profile_path.resolve()), "status": "FAIL",
                "error_type": type(exc).__name__, "error": str(exc),
            })

        ocv_path = ocv_root / ocv_name
        try:
            frame, sheet = read_measurements(ocv_path)
            frame = historical_export_precision(frame)
            adapted = adapt_ocv(frame)
            output = staged_ocv_root / f"LFP_OCV_{temp:g}.csv"
            adapted.to_csv(output, index=False, float_format="%.10g", lineterminator="\n")
            success.append({
                "chemistry": "LFP", "stage": "ocv_excel_adapter", "temperature_C": temp,
                "profile": "OCV", "source_file": str(ocv_path.resolve()), "source_sheet": sheet,
                "source_sha256": sha256_file(ocv_path), "output_file": str(output.resolve()),
                "output_sha256": sha256_file(output), "rows": len(adapted), "status": "PASS",
            })
        except Exception as exc:
            failures.append({
                "chemistry": "LFP", "stage": "ocv_excel_adapter", "temperature_C": temp,
                "profile": "OCV", "source_file": str(ocv_path.resolve()), "status": "FAIL",
                "error_type": type(exc).__name__, "error": str(exc),
            })

    success_frame = pd.DataFrame(success)
    failure_frame = pd.DataFrame(
        failures,
        columns=[
            "chemistry", "stage", "temperature_C", "profile", "source_file",
            "status", "error_type", "error",
        ],
    )
    success_frame.to_csv(output_root / "raw_adapter_manifest.csv", index=False, lineterminator="\n")
    failure_frame.to_csv(output_root / "raw_adapter_failures.csv", index=False, lineterminator="\n")
    return success_frame, failure_frame


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", type=Path, default=repo / "Data/LFP/Profiles")
    parser.add_argument("--ocv-root", type=Path, default=repo / "Data/LFP/OCV")
    parser.add_argument("--output-root", type=Path, default=repo / "Data/Preprocessed/LFP")
    args = parser.parse_args()
    success, failures = run(args.profile_root.resolve(), args.ocv_root.resolve(), args.output_root.resolve())
    print(f"LFP raw adapter: pass={len(success)} fail={len(failures)}")


if __name__ == "__main__":
    main()
