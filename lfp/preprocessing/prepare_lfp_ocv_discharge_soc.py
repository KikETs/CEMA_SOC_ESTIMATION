#!/usr/bin/env python3
"""Build LFP SOC labels from temperature-matched low-current OCV discharge data.

Only the raw charge/discharge capacity counters from each drive file and the
temperature-matched Step-5 ``discharge_ocv`` branch are label inputs. Existing
SOC, Q_net, and profile-end capacity fields are audit-only and never inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


WORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORK_DIR.parents[1]
DYNAMIC_ROOT = REPO_ROOT / "lfp" / "data" / "raw_dynamic"
OCV_ROOT = REPO_ROOT / "lfp" / "data" / "ocv"
PREPARED_ROOT = REPO_ROOT / "lfp" / "data" / "preprocessed" / "prepared_data_ocv_discharge_soc"
MANIFEST_DIR = REPO_ROOT / "lfp" / "data" / "preprocessed" / "manifests_ocv_discharge_3lopo"

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
TEMPERATURE_COLUMN = "Temperature (C)_1"
OCV_STEP_INDEX = 5
OCV_LABEL = "discharge_ocv"
LABEL_POLICY = "temperature_matched_low_current_ocv_discharge_soc0_and_qref"


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def dynamic_source_path(dirname: str, profile: str) -> Path:
    path = DYNAMIC_ROOT / dirname / f"LFP_{dirname}_{profile}_usableSOC.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def ocv_source_path(temp: float) -> Path:
    path = OCV_ROOT / f"LFP_OCV_{temp:g}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _monotone_discharge_inverse(
    branch_soc: np.ndarray,
    branch_voltage: np.ndarray,
    initial_voltage: float,
) -> tuple[float, str, float, float]:
    """Invert the discharge curve after SOC-bin aggregation and monotone QA."""
    grid = np.linspace(0.0, 1.0, 4001, dtype=np.float64)
    bins = np.clip(np.rint(branch_soc * 4000.0).astype(np.int64), 0, 4000)
    table = pd.DataFrame({"bin": bins, "voltage": branch_voltage})
    med = table.groupby("bin", sort=True)["voltage"].median().reindex(range(4001))
    med = med.interpolate(method="linear", limit_direction="both")
    voltage_grid = med.to_numpy(np.float64)
    if not np.isfinite(voltage_grid).all():
        raise RuntimeError("The OCV discharge inverse contains non-finite voltage bins.")
    voltage_grid = np.maximum.accumulate(voltage_grid)
    v_low = float(voltage_grid[0])
    v_high = float(voltage_grid[-1])
    if initial_voltage <= v_low:
        return 0.0, "clamped_below_discharge_curve", v_low, v_high
    if initial_voltage >= v_high:
        return 1.0, "clamped_above_discharge_curve", v_low, v_high

    unique_voltage, first_index = np.unique(voltage_grid, return_index=True)
    unique_soc = grid[first_index]
    soc0 = float(np.interp(initial_voltage, unique_voltage, unique_soc))
    return float(np.clip(soc0, 0.0, 1.0)), "interpolated_on_discharge_curve", v_low, v_high


def load_ocv_reference(temp: float) -> tuple[dict[str, object], dict[str, object]]:
    path = ocv_source_path(temp)
    raw = pd.read_csv(path)
    required = {
        "Test_Time(s)",
        "Step_Index",
        "Current(A)",
        "Voltage(V)",
        "Discharge_Capacity(Ah)",
        "label",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(f"Missing OCV columns {missing}: {path}")

    step = pd.to_numeric(raw["Step_Index"], errors="coerce")
    label = raw["label"].astype(str).str.strip().str.lower()
    branch = raw[(step == OCV_STEP_INDEX) & (label == OCV_LABEL)].copy()
    if len(branch) < 1000:
        raise RuntimeError(f"OCV Step-5 discharge branch is too short ({len(branch)} rows): {path}")

    time_s = pd.to_numeric(branch["Test_Time(s)"], errors="coerce").to_numpy(np.float64)
    current_a = pd.to_numeric(branch["Current(A)"], errors="coerce").to_numpy(np.float64)
    voltage_v = pd.to_numeric(branch["Voltage(V)"], errors="coerce").to_numpy(np.float64)
    discharge_ah = pd.to_numeric(branch["Discharge_Capacity(Ah)"], errors="coerce").to_numpy(np.float64)
    if not np.column_stack(
        [np.isfinite(time_s), np.isfinite(current_a), np.isfinite(voltage_v), np.isfinite(discharge_ah)]
    ).all():
        raise RuntimeError(f"Non-finite values in OCV Step-5 discharge branch: {path}")

    current_median = float(np.median(current_a))
    if not (-0.07 < current_median < -0.03):
        raise RuntimeError(f"Step 5 is not the expected low-current discharge branch: I_median={current_median}, {path}")
    q_min = float(np.min(discharge_ah))
    q_max = float(np.max(discharge_ah))
    q_ref = q_max - q_min
    if not (0.8 < q_ref < 1.3):
        raise RuntimeError(f"Invalid OCV discharge capacity span {q_ref} Ah: {path}")

    branch_soc = np.clip(1.0 - (discharge_ah - q_min) / q_ref, 0.0, 1.0)
    reference = {
        "temperature_C": float(temp),
        "path": path,
        "sha256": sha256_file(path),
        "q_ref_Ah": float(q_ref),
        "branch_soc": branch_soc,
        "branch_voltage_V": voltage_v,
    }
    audit = {
        "temperature_C": float(temp),
        "ocv_source_file": str(path.resolve()),
        "ocv_source_sha256": reference["sha256"],
        "ocv_branch_label": OCV_LABEL,
        "ocv_step_index": OCV_STEP_INDEX,
        "ocv_branch_rows": int(len(branch)),
        "ocv_discharge_current_median_A": current_median,
        "ocv_discharge_current_min_A": float(np.min(current_a)),
        "ocv_discharge_current_max_A": float(np.max(current_a)),
        "ocv_discharge_voltage_min_V": float(np.min(voltage_v)),
        "ocv_discharge_voltage_max_V": float(np.max(voltage_v)),
        "q_ref_ocv_discharge_Ah": float(q_ref),
        "q_ref_policy": "Step5 discharge_ocv max(Discharge_Capacity)-min(Discharge_Capacity)",
        "ocv_time_nonpositive_diffs": int(np.sum(np.diff(time_s) <= 0)),
        "ocv_capacity_negative_diffs": int(np.sum(np.diff(discharge_ah) < 0)),
        "ocv_charge_branch_used": False,
        "processed_charge_discharge_mean_table_used": False,
    }
    return reference, audit


def adapter_frame(
    source: Path,
    temp: float,
    profile: str,
    ocv_reference: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
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
        raise RuntimeError(f"Missing dynamic columns {missing}: {source}")

    numeric_cols = sorted(required - {"Step_Index"}) + ["Step_Index"]
    numeric = {column: pd.to_numeric(raw[column], errors="coerce").to_numpy(np.float64) for column in numeric_cols}
    finite = np.column_stack([np.isfinite(numeric[column]) for column in numeric_cols]).all(axis=1)
    if not bool(finite.all()):
        raise RuntimeError(f"Non-finite dynamic rows were found: {source}")

    time_raw = numeric["Test_Time(s)"]
    if len(time_raw) < 51 or bool(np.any(np.diff(time_raw) <= 0)):
        raise RuntimeError(f"Invalid or too-short dynamic time axis: {source}")
    time = time_raw - time_raw[0]

    charge_capacity = numeric["Charge_Capacity(Ah)"]
    discharge_capacity = numeric["Discharge_Capacity(Ah)"]
    q_removed = (discharge_capacity - discharge_capacity[0]) - (charge_capacity - charge_capacity[0])
    q_ref = float(ocv_reference["q_ref_Ah"])
    initial_voltage = float(numeric["Voltage(V)"][0])
    soc0, soc0_status, inverse_v_low, inverse_v_high = _monotone_discharge_inverse(
        np.asarray(ocv_reference["branch_soc"], dtype=np.float64),
        np.asarray(ocv_reference["branch_voltage_V"], dtype=np.float64),
        initial_voltage,
    )
    soc_unclipped = soc0 - q_removed / q_ref
    soc = np.clip(soc_unclipped, 0.0, 1.0)

    existing_soc_diff = float("nan")
    if "SOC_usable" in raw.columns:
        existing_soc = pd.to_numeric(raw["SOC_usable"], errors="coerce").to_numpy(np.float64)
        if np.isfinite(existing_soc).all():
            existing_soc_diff = float(np.max(np.abs(soc - existing_soc)))

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
            "SOC0_used": np.full(len(raw), soc0, dtype=np.float64),
            "SOC0_OCV_inferred": np.full(len(raw), soc0, dtype=np.float64),
            "SOC0_Vinit(V)": np.full(len(raw), initial_voltage, dtype=np.float64),
            "SOC0_restStep": np.zeros(len(raw), dtype=np.int16),
            "Qnet_denom(Ah)": np.full(len(raw), q_ref, dtype=np.float64),
            "Q_ref_lc_ocv_discharge_Ah": np.full(len(raw), q_ref, dtype=np.float64),
            "Qnet_removed(Ah)": q_removed,
            "SOC_CC_unclipped": soc_unclipped,
            "SOC_CC": soc,
            "SOC_CC(%)": 100.0 * soc,
            "SOC_scale_mode": np.full(
                len(raw), "SOC0_ocv_discharge_minus_Qremoved_over_temp_ocv_discharge_capacity", dtype=object
            ),
            "OCV_DischargeStepUsed": np.full(len(raw), OCV_STEP_INDEX, dtype=np.int16),
        }
    )
    formula_soc = np.clip(soc0 - output["Qnet_removed(Ah)"].to_numpy(np.float64) / q_ref, 0.0, 1.0)
    audit = {
        "temperature_C": float(temp),
        "profile": profile,
        "source_file": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "source_rows": int(len(raw)),
        "label_policy": LABEL_POLICY,
        "label_equation": "SOC=clip(SOC0_OCV_discharge-Q_removed/Qref_OCV_discharge,0,1)",
        "q_removed_equation": "(Discharge-Discharge0)-(Charge-Charge0)",
        "ocv_source_file": str(Path(ocv_reference["path"]).resolve()),
        "ocv_source_sha256": str(ocv_reference["sha256"]),
        "ocv_step_index": OCV_STEP_INDEX,
        "ocv_branch": "discharge",
        "ocv_inverse_used": True,
        "ocv_charge_branch_used": False,
        "processed_charge_discharge_mean_table_used": False,
        "existing_soc_columns_used_for_label": False,
        "existing_q_columns_used_for_label": False,
        "profile_endpoint_used_as_capacity": False,
        "q_ref_ocv_discharge_Ah": q_ref,
        "q_removed_end_Ah": float(q_removed[-1]),
        "soc0_ocv_discharge": float(soc0),
        "soc0_inverse_status": soc0_status,
        "soc0_vinit_V": initial_voltage,
        "soc0_inverse_curve_min_V": inverse_v_low,
        "soc0_inverse_curve_max_V": inverse_v_high,
        "initial_soc": float(soc[0]),
        "final_soc_unclipped": float(soc_unclipped[-1]),
        "final_soc": float(soc[-1]),
        "min_soc_unclipped": float(np.min(soc_unclipped)),
        "min_soc": float(np.min(soc)),
        "max_soc_unclipped": float(np.max(soc_unclipped)),
        "max_soc": float(np.max(soc)),
        "upper_clip_rows": int(np.sum(soc_unclipped > 1.0)),
        "lower_clip_rows": int(np.sum(soc_unclipped < 0.0)),
        "max_abs_SOC_CC_minus_formula": float(np.max(np.abs(output["SOC_CC"].to_numpy(np.float64) - formula_soc))),
        "max_recomputed_minus_existing_SOC_usable": existing_soc_diff,
        "median_dt_s": float(np.median(np.diff(time))),
        "duration_s": float(time[-1]),
        "initial_current_A": float(numeric["Current(A)"][0]),
        "final_current_A": float(numeric["Current(A)"][-1]),
        "initial_voltage_V": initial_voltage,
        "final_voltage_V": float(numeric["Voltage(V)"][-1]),
    }
    return output, audit


def prepare(force: bool) -> None:
    if not DYNAMIC_ROOT.is_dir():
        raise FileNotFoundError(DYNAMIC_ROOT)
    if not OCV_ROOT.is_dir():
        raise FileNotFoundError(OCV_ROOT)
    PREPARED_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    references: dict[float, dict[str, object]] = {}
    reference_rows: list[dict[str, object]] = []
    for temp in TEMPERATURE_DIRS:
        reference, audit = load_ocv_reference(temp)
        references[temp] = reference
        reference_rows.append(audit)
    reference_manifest = pd.DataFrame(reference_rows).sort_values("temperature_C").reset_index(drop=True)
    reference_manifest.to_csv(MANIFEST_DIR / "ocv_discharge_reference_manifest.csv", index=False, lineterminator="\n")

    rows: list[dict[str, object]] = []
    for temp, dirname in TEMPERATURE_DIRS.items():
        for profile in PROFILES:
            source = dynamic_source_path(dirname, profile)
            frame, audit = adapter_frame(source, temp, profile, references[temp])
            output_dir = PREPARED_ROOT / f"{temp:g}C"
            output_dir.mkdir(parents=True, exist_ok=True)
            output = output_dir / f"LFP_{temp:g}C_{profile}.csv"
            if output.exists() and not force:
                raise FileExistsError(f"Use --force to rebuild {output}")
            frame.to_csv(output, index=False, float_format="%.17g", lineterminator="\n")
            rows.append(
                {
                    **audit,
                    "prepared_file": str(output.resolve()),
                    "prepared_sha256": sha256_file(output),
                    "prepared_rows": int(len(frame)),
                    "prepared_size_bytes": int(output.stat().st_size),
                }
            )

    manifest = pd.DataFrame(rows).sort_values(["temperature_C", "profile"]).reset_index(drop=True)
    if len(manifest) != len(TEMPERATURE_DIRS) * len(PROFILES):
        raise RuntimeError(f"Prepared {len(manifest)} files; expected 24.")
    if not np.allclose(manifest["initial_soc"].to_numpy(np.float64), 1.0, atol=1e-12):
        raise RuntimeError("Not every OCV-discharge initial SOC is 1.0 as expected from the source voltages.")
    if bool((manifest["final_soc"] <= 0.0).any()):
        raise RuntimeError("At least one corrected OCV-discharge label ends at the lower SOC clamp.")
    if int(manifest["lower_clip_rows"].sum()) != 0:
        raise RuntimeError("Corrected OCV-discharge labels unexpectedly contain lower-clipped rows.")
    manifest.to_csv(MANIFEST_DIR / "prepared_dataset_manifest.csv", index=False, lineterminator="\n")

    summary = {
        "status": "PASS",
        "dynamic_source_root": str(DYNAMIC_ROOT.resolve()),
        "ocv_source_root": str(OCV_ROOT.resolve()),
        "prepared_root": str(PREPARED_ROOT.resolve()),
        "profiles": list(PROFILES),
        "temperatures_C": list(TEMPERATURE_DIRS),
        "prepared_files": int(len(manifest)),
        "label_inputs": [
            "dynamic Charge_Capacity(Ah)",
            "dynamic Discharge_Capacity(Ah)",
            "dynamic initial Voltage(V)",
            "temperature-matched OCV Step-5 discharge curve",
        ],
        "label_policy": LABEL_POLICY,
        "label_equation": "SOC=clip(SOC0_OCV_discharge-Q_removed/Qref_OCV_discharge,0,1)",
        "q_ref_policy": "Step5 discharge_ocv max(Discharge_Capacity)-min(Discharge_Capacity)",
        "ocv_inverse_used": True,
        "ocv_charge_branch_used": False,
        "processed_charge_discharge_mean_table_used": False,
        "existing_soc_columns_used_for_label": False,
        "existing_q_columns_used_for_label": False,
        "profile_endpoint_used_as_capacity": False,
        "all_initial_soc": float(manifest["initial_soc"].iloc[0]),
        "final_soc_min": float(manifest["final_soc"].min()),
        "final_soc_max": float(manifest["final_soc"].max()),
        "total_upper_clip_rows": int(manifest["upper_clip_rows"].sum()),
        "total_lower_clip_rows": int(manifest["lower_clip_rows"].sum()),
        "max_label_write_error": float(manifest["max_abs_SOC_CC_minus_formula"].max()),
        "training_started": False,
    }
    (MANIFEST_DIR / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dynamic-root", type=Path, default=DYNAMIC_ROOT)
    parser.add_argument("--ocv-root", type=Path, default=OCV_ROOT)
    parser.add_argument("--prepared-root", type=Path, default=PREPARED_ROOT)
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DYNAMIC_ROOT = args.dynamic_root.resolve()
    OCV_ROOT = args.ocv_root.resolve()
    PREPARED_ROOT = args.prepared_root.resolve()
    MANIFEST_DIR = args.manifest_dir.resolve()
    prepare(force=bool(args.force))
