#!/usr/bin/env python3
"""Reverse-calculate LFP SOC with the user's usable-capacity definition.

Existing SOC/Q_net/Q_eff columns are not inputs to the label calculation.
SOC is rebuilt from the raw charge and discharge capacity counters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


WORK_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = WORK_DIR.parents[1] / "SOC" / "LFP_NEW"
PREPARED_ROOT = WORK_DIR / "prepared_data_user_soc"
MANIFEST_DIR = WORK_DIR / "manifests"

TEMPERATURE_DIRS = {
    -10.0: "N10",
    0.0: "0",
    10.0: "10",
    20.0: "20",
    25.0: "25",
    30.0: "30",
    40.0: "40",
    50.0: "50",
}
PROFILES = ("DST", "FUDS", "US06")
LABEL_FRACTION = "SOC_usable"
LABEL_PERCENT = "SOC_usable(%)"
TEMPERATURE_COLUMN = "Temperature (C)_1"


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def source_path(temp: float, dirname: str, profile: str) -> Path:
    path = SOURCE_ROOT / dirname / f"LFP_{dirname}_{profile}_usableSOC.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def adapter_frame(source: Path, temp: float, profile: str) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_csv(source)
    required = {
        "Test_Time(s)",
        "Step_Time(s)",
        "Step_Index",
        "Current(A)",
        "Voltage(V)",
        TEMPERATURE_COLUMN,
        "Charge_Capacity(Ah)",
        "Discharge_Capacity(Ah)",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(f"Missing required user-label columns {missing}: {source}")

    numeric_cols = sorted(required - {"Step_Index"}) + ["Step_Index"]
    numeric = {column: pd.to_numeric(raw[column], errors="coerce").to_numpy(np.float64) for column in numeric_cols}
    finite = np.column_stack([np.isfinite(numeric[column]) for column in numeric_cols]).all(axis=1)
    if not bool(finite.all()):
        raise RuntimeError(f"Non-finite user source rows were found: {source}")

    time_raw = numeric["Test_Time(s)"]
    if len(time_raw) < 51 or bool(np.any(np.diff(time_raw) <= 0)):
        raise RuntimeError(f"Invalid or too-short time axis: {source}")
    time = time_raw - time_raw[0]
    charge_capacity = numeric["Charge_Capacity(Ah)"]
    discharge_capacity = numeric["Discharge_Capacity(Ah)"]
    q_removed = (discharge_capacity - discharge_capacity[0]) - (charge_capacity - charge_capacity[0])
    q_eff_median = float(q_removed[-1])
    if not np.isfinite(q_eff_median) or q_eff_median <= 0:
        raise RuntimeError(f"Reverse-calculated usable capacity is invalid: {q_eff_median} Ah, {source}")
    soc = np.clip(1.0 - q_removed / q_eff_median, 0.0, 1.0)
    soc_pct = 100.0 * soc

    existing_soc_comparison_diff = float("nan")
    if LABEL_FRACTION in raw.columns:
        existing_soc = pd.to_numeric(raw[LABEL_FRACTION], errors="coerce").to_numpy(np.float64)
        if np.isfinite(existing_soc).all():
            existing_soc_comparison_diff = float(np.max(np.abs(soc - existing_soc)))
    initial_soc = float(soc[0])
    output = pd.DataFrame(
        {
            "Data_Point": np.arange(len(raw), dtype=np.int64),
            "Test_Time(s)": time,
            "Step_Time(s)": numeric["Step_Time(s)"],
            "Step_Index": numeric["Step_Index"].astype(np.int64),
            "DriveStepIndex": numeric["Step_Index"].astype(np.int64),
            "Current(A)": numeric["Current(A)"],
            "Voltage(V)": numeric["Voltage(V)"],
            "Temperature(C)": numeric[TEMPERATURE_COLUMN],
            "TempLabel": np.full(len(raw), f"{temp:g}C", dtype=object),
            "Profile": np.full(len(raw), profile, dtype=object),
            "SOC0_used": np.full(len(raw), initial_soc, dtype=np.float64),
            "SOC0_Vinit(V)": np.full(len(raw), float(numeric["Voltage(V)"][0]), dtype=np.float64),
            "SOC0_restStep": np.zeros(len(raw), dtype=np.int16),
            "Qnet_denom(Ah)": np.full(len(raw), q_eff_median, dtype=np.float64),
            "Qnet_removed(Ah)": q_removed,
            "SOC_CC": soc,
            "SOC_CC(%)": soc_pct,
        }
    )
    audit = {
        "temperature_C": temp,
        "profile": profile,
        "source_file": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "source_rows": len(raw),
        "reverse_charge_capacity_column": "Charge_Capacity(Ah)",
        "reverse_discharge_capacity_column": "Discharge_Capacity(Ah)",
        "label_policy": "reverse_from_delta_discharge_minus_delta_charge_capacity",
        "label_equation": "SOC=clip(1-((Dis-Dis0)-(Chg-Chg0))/Qeff,0,1); Qeff=final_Q_removed",
        "ocv_inverse_used": False,
        "existing_soc_columns_used_for_label": False,
        "existing_q_columns_used_for_label": False,
        "max_abs_SOC_CC_minus_reverse_calculation": float(np.max(np.abs(output["SOC_CC"].to_numpy() - soc))),
        "max_recomputed_minus_existing_SOC_usable": existing_soc_comparison_diff,
        "q_eff_profile_Ah": q_eff_median,
        "initial_soc": initial_soc,
        "final_soc": float(soc[-1]),
        "min_soc": float(np.min(soc)),
        "max_soc": float(np.max(soc)),
        "median_dt_s": float(np.median(np.diff(time))),
        "duration_s": float(time[-1]),
        "initial_current_A": float(numeric["Current(A)"][0]),
        "final_current_A": float(numeric["Current(A)"][-1]),
        "initial_voltage_V": float(numeric["Voltage(V)"][0]),
        "final_voltage_V": float(numeric["Voltage(V)"][-1]),
    }
    return output, audit


def prepare(force: bool) -> None:
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(SOURCE_ROOT)
    PREPARED_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for temp, dirname in TEMPERATURE_DIRS.items():
        for profile in PROFILES:
            source = source_path(temp, dirname, profile)
            frame, audit = adapter_frame(source, temp, profile)
            output_dir = PREPARED_ROOT / f"{temp:g}C"
            output_dir.mkdir(parents=True, exist_ok=True)
            output = output_dir / f"LFP_{temp:g}C_{profile}.csv"
            if output.exists() and not force:
                raise FileExistsError(f"Use --force to rebuild {output}")
            frame.to_csv(output, index=False, float_format="%.10g")
            rows.append(
                {
                    **audit,
                    "prepared_file": str(output.resolve()),
                    "prepared_sha256": sha256_file(output),
                    "prepared_rows": len(frame),
                    "prepared_size_bytes": output.stat().st_size,
                }
            )

    manifest = pd.DataFrame(rows).sort_values(["temperature_C", "profile"]).reset_index(drop=True)
    if len(manifest) != len(TEMPERATURE_DIRS) * len(PROFILES):
        raise RuntimeError(f"Prepared {len(manifest)} files; expected 24.")
    manifest.to_csv(MANIFEST_DIR / "prepared_dataset_manifest.csv", index=False)
    summary = {
        "source_root": str(SOURCE_ROOT.resolve()),
        "prepared_root": str(PREPARED_ROOT.resolve()),
        "profiles": list(PROFILES),
        "temperatures_C": list(TEMPERATURE_DIRS),
        "prepared_files": int(len(manifest)),
        "label_inputs": ["Charge_Capacity(Ah)", "Discharge_Capacity(Ah)"],
        "adapter_label": "SOC_CC(%)",
        "label_policy": "reverse from delta discharge capacity minus delta charge capacity",
        "label_equation": "Q_removed=(Dis-Dis0)-(Chg-Chg0); Q_eff=Q_removed[-1]; SOC=clip(1-Q_removed/Q_eff,0,1)",
        "ocv_inverse_used": False,
        "existing_soc_columns_used_for_label": False,
        "existing_q_columns_used_for_label": False,
        "max_reverse_label_write_error": float(manifest["max_abs_SOC_CC_minus_reverse_calculation"].max()),
        "max_recomputed_minus_existing_SOC_usable": float(
            manifest["max_recomputed_minus_existing_SOC_usable"].max()
        ),
        "training_started": False,
    }
    (MANIFEST_DIR / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(force=bool(parse_args().force))
