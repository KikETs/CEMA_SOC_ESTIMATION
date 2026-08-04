from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROFILE_ORDER = ("DST", "US06", "FUDS")
TEMP_ORDER = (0.0, 25.0, 45.0)
MANUSCRIPT_SOC_TOKEN_PERCENT = 80.0
REQUIRED_OUTPUT_COLUMNS = [
    "time_s",
    "V_raw",
    "I_raw",
    "T",
    "SOC_percent",
    "V_corr_raw",
    "V_corr_raw_ema50",
    "V_corr_raw_dev_ema50",
    "V_corr_raw_ema200",
    "V_corr_raw_dev_ema200",
    "V_corr_raw_ema800",
    "V_corr_raw_dev_ema800",
    "I_raw_ema50",
    "I_raw_dev_ema50",
    "I_raw_ema200",
    "I_raw_dev_ema200",
    "absI_ema50",
    "absI_dev_ema50",
    "absI_ema200",
    "absI_dev_ema200",
    "profile",
    "temperature_C",
    "source_file",
]


@dataclass
class R0Estimate:
    temperature_C: float
    r0_ohm: float
    n_events: int
    source: str


@dataclass
class OcvReference:
    temperature_C: float
    q_ref_ah: float
    voltage_v: np.ndarray | None
    soc_fraction: np.ndarray | None
    source: str


def norm_col(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_col(columns: list[object], aliases: list[str]) -> str | None:
    normalized = {norm_col(c): str(c) for c in columns}
    for alias in aliases:
        hit = normalized.get(norm_col(alias))
        if hit is not None:
            return hit
    return None


def infer_profile(path: Path) -> str | None:
    upper = path.stem.upper()
    for profile in PROFILE_ORDER:
        if profile in upper:
            return profile
    return None


def infer_temperature(path: Path) -> float | None:
    match = re.search(r"(?<!\d)(-?\d+(?:\.\d+)?)\s*C", path.stem, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    stem = path.stem.lower()
    if re.search(r"(?:^|_)02_\d{2}_2016", stem):
        return 0.0
    if re.search(r"(?:^|_)11_\d{1,2}_2015", stem):
        return 25.0
    if re.search(r"(?:^|_)12_\d{2}_2015", stem):
        return 45.0
    return None


def infer_filename_soc_token_percent(path: Path) -> float | None:
    stem = path.stem.upper()
    profile_pattern = "|".join(PROFILE_ORDER)
    patterns = [
        rf"(?:{profile_pattern})[^A-Z0-9]+(?P<soc>\d+(?:\.\d+)?)\s*(?:SOC)?(?:$|[^A-Z0-9])",
        r"(?P<soc>\d+(?:\.\d+)?)\s*SOC(?:$|[^A-Z0-9])",
    ]
    for pattern in patterns:
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if match:
            value = float(match.group("soc"))
            if np.isclose(value, MANUSCRIPT_SOC_TOKEN_PERCENT):
                return MANUSCRIPT_SOC_TOKEN_PERCENT
            raise ValueError(f"Unsupported start-SOC token in {path.name}: {value:g}SOC. Expected 80 or 80SOC.")
    return None


def format_soc_for_filename(soc_percent: float) -> str:
    text = f"{float(soc_percent):g}"
    return text.replace(".", "p")


def expected_manuscript_records() -> set[tuple[float, str, float]]:
    return {
        (float(temp), str(profile), MANUSCRIPT_SOC_TOKEN_PERCENT)
        for temp in TEMP_ORDER
        for profile in PROFILE_ORDER
    }


def observed_records(frames: list[pd.DataFrame]) -> set[tuple[float, str, float]]:
    out = set()
    for frame in frames:
        temp = float(frame["temperature_C"].iloc[0])
        profile = str(frame["profile"].iloc[0])
        soc = round(float(frame.attrs.get("filename_soc_token_percent", MANUSCRIPT_SOC_TOKEN_PERCENT)), 6)
        out.add((temp, profile, soc))
    return out


def validate_manuscript_dynamic_set(frames: list[pd.DataFrame]) -> None:
    missing = sorted(expected_manuscript_records() - observed_records(frames))
    if not missing:
        return
    formatted = "\n".join(f"  *{int(temp)}C_{profile}_{format_soc_for_filename(soc)}*.xls*" for temp, profile, soc in missing)
    raise ValueError(
        "Missing required manuscript dynamic-profile files. "
        "Place all 0/25/45 C DST/US06/FUDS 80 % SOC files in Data/raw_dynamic/ "
        "or pass --allow-incomplete for a partial conversion.\n"
        f"{formatted}"
    )


def read_measurement_sheets(path: Path) -> list[pd.DataFrame]:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        cols = list(frame.columns)
        has_v = find_col(cols, ["Voltage(V)", "Voltage", "V", "V_raw", "mV"]) is not None
        has_i = find_col(cols, ["Current(A)", "Current", "I", "I_raw", "mA"]) is not None
        if has_v and has_i:
            return [frame]
        raise ValueError(f"No measurement-like columns found in {path}")

    excel = pd.ExcelFile(path)
    frames: list[pd.DataFrame] = []
    for sheet in excel.sheet_names:
        header = pd.read_excel(path, sheet_name=sheet, nrows=0)
        cols = list(header.columns)
        has_v = find_col(cols, ["Voltage(V)", "Voltage", "V", "V_raw", "mV"]) is not None
        has_i = find_col(cols, ["Current(A)", "Current", "I", "I_raw", "mA"]) is not None
        if has_v and has_i:
            frame = pd.read_excel(path, sheet_name=sheet)
            if not frame.empty:
                frames.append(frame)
    if not frames:
        raise ValueError(f"No measurement-like sheet found in {path}")
    return frames


def read_first_measurement_sheet(path: Path) -> pd.DataFrame:
    return read_measurement_sheets(path)[0]


def round_calce_export_precision(frame: pd.DataFrame) -> pd.DataFrame:
    """Match the historical CALCE CSV export precision before labeling."""
    out = frame.copy()
    significant_10 = ("Test_Time(s)", "Step_Time(s)")
    fixed_9 = (
        "Charge_Capacity(Ah)",
        "Discharge_Capacity(Ah)",
        "Charge_Energy(Wh)", "Discharge_Energy(Wh)", "dV/dt(V/s)",
    )
    for alias in significant_10:
        column = find_col(list(out.columns), [alias])
        if column is not None:
            values = pd.to_numeric(out[column], errors="coerce")
            out[column] = values.map(lambda value: float(f"{value:.10g}") if np.isfinite(value) else value)
    for alias in fixed_9:
        column = find_col(list(out.columns), [alias])
        if column is not None:
            out[column] = pd.to_numeric(out[column], errors="coerce").round(9)
    current_column = find_col(list(out.columns), ["Current(A)"])
    if current_column is not None:
        current = pd.to_numeric(out[current_column], errors="coerce").to_numpy(float)
        rounded_current = np.round(current, 9)
        near_zero = np.abs(current) < 1e-4
        rounded_current[near_zero] = np.asarray(
            [float(f"{value:.6g}") for value in current[near_zero]], dtype=float
        )
        out[current_column] = rounded_current
    voltage_column = find_col(list(out.columns), ["Voltage(V)"])
    if voltage_column is not None:
        voltage = pd.to_numeric(out[voltage_column], errors="coerce").to_numpy(float)
        out[voltage_column] = np.sign(voltage) * np.floor(np.abs(voltage) * 1e9 + 0.5) / 1e9
    return out


def standardize_frame(path: Path, default_temperature: float | None = None, default_profile: str | None = None) -> pd.DataFrame:
    raw = round_calce_export_precision(read_first_measurement_sheet(path))
    cols = list(raw.columns)
    time_col = find_col(cols, ["Test_Time(s)", "time_s", "Step_Time(s)", "Duration (sec)", "Time"])
    voltage_col = find_col(cols, ["Voltage(V)", "V_raw", "Voltage", "V"])
    current_col = find_col(cols, ["Current(A)", "I_raw", "Current", "I"])
    mv_col = find_col(cols, ["mV"])
    ma_col = find_col(cols, ["mA"])
    temp_col = find_col(cols, ["Temperature(C)", "Temperature", "Temp", "T", "TempLabel"])

    if time_col is None:
        time = np.arange(len(raw), dtype=float)
    else:
        time = pd.to_numeric(raw[time_col], errors="coerce").to_numpy(float)
        if not np.isfinite(time).any():
            time = np.arange(len(raw), dtype=float)
    time = time - np.nanmin(time)

    if voltage_col is not None:
        voltage = pd.to_numeric(raw[voltage_col], errors="coerce").to_numpy(float)
    elif mv_col is not None:
        voltage = pd.to_numeric(raw[mv_col], errors="coerce").to_numpy(float) / 1000.0
    else:
        raise ValueError(f"Voltage column not found in {path}")

    if current_col is not None:
        current = pd.to_numeric(raw[current_col], errors="coerce").to_numpy(float)
    elif ma_col is not None:
        current = pd.to_numeric(raw[ma_col], errors="coerce").to_numpy(float) / 1000.0
    else:
        raise ValueError(f"Current column not found in {path}")

    temp = infer_temperature(path) if default_temperature is None else default_temperature
    if temp is None and temp_col is not None:
        tvals = pd.to_numeric(raw[temp_col], errors="coerce").dropna()
        temp = float(tvals.median()) if not tvals.empty else np.nan
    if temp is None:
        temp = np.nan

    profile = infer_profile(path) if default_profile is None else default_profile
    if profile is None:
        profile = "UNKNOWN"

    out = raw.copy()
    out["time_s"] = time
    out["t_global(s)"] = time
    out["V_raw"] = voltage
    out["I_raw"] = current
    out["T"] = float(temp)
    out["profile"] = profile
    out["temperature_C"] = float(temp) if np.isfinite(temp) else np.nan
    out["source_file"] = path.name
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_s", "V_raw", "I_raw"]).copy()
    point_col = find_col(list(out.columns), ["Data_Point"])
    if point_col is not None:
        out = out.sort_values(point_col)
    else:
        out = out.sort_values("time_s")
    out = out.reset_index(drop=True)
    return out


def read_optional_csv(*paths: Path) -> pd.DataFrame | None:
    for path in paths:
        if path.exists():
            return pd.read_csv(path)
    return None


def load_soc0_anchor_table(reference_dir: Path) -> dict[tuple[float, str], dict[str, float]]:
    table = read_optional_csv(reference_dir / "ocv_inferred_start_soc_by_file.csv")
    if table is None:
        return {}
    anchors: dict[tuple[float, str], dict[str, float]] = {}
    for _, row in table.iterrows():
        temp = float(row["temperature_C"])
        profile = str(row["profile"]).upper()
        anchors[(temp, profile)] = {
            "soc0": float(row["start_SOC_from_SOC0_Vinit"]),
            "soc0_pct": float(row["start_SOC_from_SOC0_Vinit_pct"]),
            "soc0_vinit": float(row["SOC0_Vinit_V"]),
            "q_ref": float(row["q_ref_lc_ocv_Ah"]),
            "source": "ocv_inferred_start_soc_by_file.csv",
        }
    return anchors


def load_qref_table(reference_dir: Path) -> dict[float, float]:
    table = read_optional_csv(reference_dir / "lc_ocv_capacity_reference.csv")
    if table is None:
        return {}
    return {
        float(row["temperature_C"]): float(row["Q_ref_lc_ocv_Ah"])
        for _, row in table.iterrows()
        if np.isfinite(float(row["Q_ref_lc_ocv_Ah"]))
    }


def contiguous_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, len(mask)))
    return segments


def infer_reference_temperature(path: Path) -> float | None:
    stem = path.stem.lower()
    # The CALCE 25 C and 45 C characterization names encode dates, not temperatures.
    if "11_21_2015" in stem:
        return 45.0
    if "11_5_2015" in stem:
        return 25.0
    if "45c" in stem:
        return 45.0
    if "25c" in stem:
        return 25.0
    if "0c" in stem or "_oc_" in stem or "02_24_2016" in stem:
        return 0.0
    return infer_temperature(path)


def build_ocv_reference_from_file(path: Path, qref_by_temp: dict[float, float]) -> OcvReference | None:
    temp = infer_reference_temperature(path)
    if temp is None:
        return None
    try:
        sheets = read_measurement_sheets(path)
    except Exception:
        return None
    frame = pd.concat(sheets, ignore_index=True)
    cols = list(frame.columns)
    voltage_col = find_col(cols, ["Voltage(V)", "Voltage", "V", "mV"])
    current_col = find_col(cols, ["Current(A)", "Current", "I", "mA"])
    time_col = find_col(cols, ["Test_Time(s)", "Duration (sec)", "time_s", "Time"])
    discharge_col = find_col(cols, ["Discharge_Capacity(Ah)", "Discharge Capacity(Ah)", "Discharge_Capacity"])
    step_col = find_col(cols, ["Step_Index", "Step Index", "Pgm step"])
    if voltage_col is None or current_col is None or time_col is None:
        return None

    voltage = pd.to_numeric(frame[voltage_col], errors="coerce").to_numpy(float)
    current = pd.to_numeric(frame[current_col], errors="coerce").to_numpy(float)
    time = pd.to_numeric(frame[time_col], errors="coerce").to_numpy(float)
    if norm_col(voltage_col) == "mv":
        voltage = voltage / 1000.0
    if norm_col(current_col) == "ma":
        current = current / 1000.0
    valid = np.isfinite(voltage) & np.isfinite(current) & np.isfinite(time)
    frame = frame.loc[valid].copy()
    voltage = voltage[valid]
    current = current[valid]
    time = time[valid]
    if len(frame) < 2:
        return None

    if step_col is not None and discharge_col is not None:
        frame["_current_A"] = current
        frame["_voltage_V"] = voltage
        step_values = pd.to_numeric(frame[step_col], errors="coerce")
        best: pd.DataFrame | None = None
        best_score = -np.inf
        for _, group in frame.groupby(step_values):
            cur = pd.to_numeric(group["_current_A"], errors="coerce")
            if len(group) < 10 or cur.mean() >= -0.01:
                continue
            q = pd.to_numeric(group[discharge_col], errors="coerce")
            score = float(q.max(skipna=True) - q.min(skipna=True))
            if score > best_score:
                best_score = score
                best = group.copy()
        if best is None:
            return None
        q_removed = pd.to_numeric(best[discharge_col], errors="coerce").to_numpy(float)
        q_ref = float(qref_by_temp.get(temp, np.nanmax(q_removed)))
        v_curve = pd.to_numeric(best["_voltage_V"], errors="coerce").to_numpy(float)
        source = f"{path.name}: low-current discharge step"
    else:
        neg_segments = contiguous_true_segments(current < -0.01)
        if not neg_segments:
            return None
        start, end = max(neg_segments, key=lambda pair: pair[1] - pair[0])
        v_curve = voltage[start:end]
        t_curve = time[start:end]
        i_curve = current[start:end]
        dt = np.diff(t_curve, prepend=t_curve[0])
        positive_dt = dt[1:][np.isfinite(dt[1:]) & (dt[1:] > 0)]
        fallback_dt = float(np.nanmedian(positive_dt)) if len(positive_dt) else 1.0
        dt[1:] = np.where(np.isfinite(dt[1:]) & (dt[1:] > 0), dt[1:], fallback_dt)
        dt[0] = 0.0
        q_removed = np.cumsum(np.maximum(-i_curve, 0.0) * dt / 3600.0)
        q_ref = float(qref_by_temp.get(temp, np.nanmax(q_removed)))
        source = f"{path.name}: low-current negative-current segment"

    keep = np.isfinite(v_curve) & np.isfinite(q_removed)
    v_curve = np.asarray(v_curve[keep], dtype=float)
    q_removed = np.asarray(q_removed[keep], dtype=float)
    if len(v_curve) < 2 or not np.isfinite(q_ref) or q_ref <= 0:
        return None
    soc = np.clip(1.0 - q_removed / q_ref, 0.0, 1.0)
    order = np.argsort(v_curve)
    return OcvReference(temp, q_ref, v_curve[order], soc[order], source)


def load_ocv_references(reference_dir: Path) -> dict[float, OcvReference]:
    qref_by_temp = load_qref_table(reference_dir)
    refs: dict[float, OcvReference] = {}
    for path in sorted(reference_dir.glob("*.xls*")):
        if "ocv" not in path.name.lower():
            continue
        ref = build_ocv_reference_from_file(path, qref_by_temp)
        if ref is not None:
            refs[ref.temperature_C] = ref
    for temp, qref in qref_by_temp.items():
        refs.setdefault(temp, OcvReference(temp, qref, None, None, "lc_ocv_capacity_reference.csv"))
    return refs


def interpolate_soc0_from_ocv(ref: OcvReference, voltage_v: float) -> float:
    if ref.voltage_v is None or ref.soc_fraction is None:
        raise ValueError(f"No OCV voltage-SOC curve is available for {ref.temperature_C:g} C")
    return float(np.interp(float(voltage_v), ref.voltage_v, ref.soc_fraction))


def integrate_current_removed_ah(frame: pd.DataFrame) -> np.ndarray:
    time = frame["time_s"].to_numpy(float)
    current = frame["I_raw"].to_numpy(float)
    dt = np.diff(time, prepend=time[0])
    positive_dt = dt[np.isfinite(dt) & (dt > 0)]
    fallback_dt = float(np.nanmedian(positive_dt)) if len(positive_dt) else 1.0
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, fallback_dt)
    return np.cumsum((np.maximum(-current, 0.0) - np.maximum(current, 0.0)) * dt / 3600.0)


def compute_q_removed(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, str]:
    cols = list(frame.columns)
    qnet_col = find_col(cols, ["Qnet_removed(Ah)", "Qnet_removed_Ah"])
    charge_col = find_col(cols, ["Charge_Capacity(Ah)", "Charge Capacity(Ah)", "Charge_Capacity", "Charge_Capacity_Ah"])
    discharge_col = find_col(cols, ["Discharge_Capacity(Ah)", "Discharge Capacity(Ah)", "Discharge_Capacity", "Discharge_Capacity_Ah"])
    if qnet_col is not None:
        qnet = pd.to_numeric(frame[qnet_col], errors="coerce").ffill().fillna(0.0)
        qnet = qnet - float(qnet.iloc[0])
        qdis = qnet.clip(lower=0.0)
        qchg = pd.Series(0.0, index=frame.index)
        return qdis, qchg, qnet, "existing_Qnet_removed(Ah)"
    if discharge_col is not None:
        qdis = pd.to_numeric(frame[discharge_col], errors="coerce").ffill().fillna(0.0)
        qdis = qdis - float(qdis.iloc[0])
        if charge_col is not None:
            qchg = pd.to_numeric(frame[charge_col], errors="coerce").ffill().fillna(0.0)
            qchg = qchg - float(qchg.iloc[0])
        else:
            qchg = pd.Series(0.0, index=frame.index)
        qnet = qdis - qchg
        return qdis, qchg, qnet, "capacity_columns_Qdis_minus_Qchg"
    qnet = pd.Series(integrate_current_removed_ah(frame), index=frame.index)
    qdis = qnet.clip(lower=0.0)
    qchg = (-qnet).clip(lower=0.0)
    return qdis, qchg, qnet, "current_integration_fallback"


def causal_ema(values: np.ndarray, span: float) -> np.ndarray:
    alpha = 2.0 / (float(span) + 1.0)
    out = np.empty_like(values, dtype=float)
    prev = float(values[0])
    for idx, value in enumerate(values):
        if not np.isfinite(value):
            value = prev
        prev = alpha * float(value) + (1.0 - alpha) * prev
        out[idx] = prev
    return out


def causal_time_ema(values: np.ndarray, time_s: np.ndarray, tau_s: float) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    prev = float(values[0])
    last_t = float(time_s[0])
    for idx, (t, value) in enumerate(zip(time_s, values, strict=True)):
        dt = max(float(t) - last_t, 0.0)
        alpha = 1.0 - np.exp(-dt / tau_s) if tau_s > 0 else 1.0
        if not np.isfinite(value):
            value = prev
        prev = alpha * float(value) + (1.0 - alpha) * prev
        out[idx] = prev
        last_t = float(t)
    return out


def estimate_r0(frame: pd.DataFrame, min_delta_i: float = 0.5, window: int = 3) -> R0Estimate:
    current = frame["I_raw"].to_numpy(float)
    voltage = frame["V_raw"].to_numpy(float)
    jumps = np.where(np.abs(np.diff(current, prepend=current[0])) >= min_delta_i)[0]
    vals = []
    for idx in jumps:
        lo0 = max(0, idx - window)
        lo1 = max(0, idx)
        hi0 = min(len(frame), idx + 1)
        hi1 = min(len(frame), idx + 1 + window)
        if lo1 <= lo0 or hi1 <= hi0:
            continue
        di = np.nanmean(current[hi0:hi1]) - np.nanmean(current[lo0:lo1])
        dv = np.nanmean(voltage[hi0:hi1]) - np.nanmean(voltage[lo0:lo1])
        if np.isfinite(di) and abs(di) >= min_delta_i and np.isfinite(dv):
            vals.append(abs(dv / di))
    if vals:
        r0 = float(np.nanmedian(vals))
        return R0Estimate(float(frame["temperature_C"].iloc[0]), r0, len(vals), "current-step median |dV/dI|")
    return R0Estimate(float(frame["temperature_C"].iloc[0]), 0.05, 0, "fallback 0.05 ohm")


def add_features(frame: pd.DataFrame, r0_ohm: float) -> pd.DataFrame:
    out = frame.copy()
    v_ohmic = out["V_raw"].to_numpy(float) - out["I_raw"].to_numpy(float) * r0_ohm
    out["V_corr_raw"] = causal_time_ema(v_ohmic, out["time_s"].to_numpy(float), tau_s=120.0)
    for span in (50, 200, 800):
        ema = causal_ema(out["V_corr_raw"].to_numpy(float), span)
        out[f"V_corr_raw_ema{span}"] = ema
        out[f"V_corr_raw_dev_ema{span}"] = out["V_corr_raw"].to_numpy(float) - ema
    for span in (50, 200):
        ema_i = causal_ema(out["I_raw"].to_numpy(float), span)
        out[f"I_raw_ema{span}"] = ema_i
        out[f"I_raw_dev_ema{span}"] = out["I_raw"].to_numpy(float) - ema_i
        ema_abs = causal_ema(np.abs(out["I_raw"].to_numpy(float)), span)
        out[f"absI_ema{span}"] = ema_abs
        out[f"absI_dev_ema{span}"] = np.abs(out["I_raw"].to_numpy(float)) - ema_abs
    return out


def extract_drive_segment(frame: pd.DataFrame) -> pd.DataFrame:
    cols = list(frame.columns)
    step_col = find_col(cols, ["Step_Index", "Step Index"])
    if step_col is None:
        out = frame.copy()
        out["DriveExtractMethod"] = "full_file_no_step_index"
        out["DriveStepIndex"] = np.nan
        out["SOC0_restStep"] = np.nan
        existing_soc0_col = find_col(list(out.columns), ["SOC0_Vinit(V)", "SOC0_Vinit"])
        if existing_soc0_col is not None and out[existing_soc0_col].notna().any():
            out["SOC0_Vinit(V)"] = float(pd.to_numeric(out[existing_soc0_col], errors="coerce").dropna().iloc[0])
        else:
            out["SOC0_Vinit(V)"] = float(out["V_raw"].dropna().iloc[0])
    else:
        step_values = pd.to_numeric(frame[step_col], errors="coerce")
        best_step = None
        best_score = -np.inf
        for step, group in frame.groupby(step_values):
            current = pd.to_numeric(group["I_raw"], errors="coerce")
            if len(group) < 50:
                continue
            if current.abs().max(skipna=True) < 0.2 and current.abs().mean(skipna=True) < 0.05:
                continue
            score = float(len(group))
            if score > best_score:
                best_score = score
                best_step = step
        if best_step is None:
            best_step = step_values.mode(dropna=True).iloc[0]
        out = frame.loc[step_values == best_step].copy()
        previous = frame.loc[step_values == best_step - 1].copy()
        if previous.empty:
            existing_soc0_col = find_col(list(out.columns), ["SOC0_Vinit(V)", "SOC0_Vinit"])
            if existing_soc0_col is not None and out[existing_soc0_col].notna().any():
                soc0_v = float(pd.to_numeric(out[existing_soc0_col], errors="coerce").dropna().iloc[0])
            else:
                soc0_v = float(out["V_raw"].dropna().iloc[0])
            rest_step = np.nan
            status = "fallback_first_loaded_voltage"
        else:
            soc0_v = float(previous["V_raw"].dropna().iloc[-1])
            rest_step = float(best_step - 1)
            status = "ok"
        out["DriveExtractMethod"] = "step"
        out["DriveStepIndex"] = int(best_step) if np.isfinite(best_step) else best_step
        out["SOC0_restStep"] = rest_step
        out["SOC0_Vinit(V)"] = soc0_v
        out["SOC0_vinit_status"] = status

    out = out.reset_index(drop=True)
    out["time_s"] = out["time_s"] - float(out["time_s"].iloc[0])
    if "Step_Time(s)" in out.columns:
        step_time = pd.to_numeric(out["Step_Time(s)"], errors="coerce")
        if step_time.notna().any():
            out["time_s"] = step_time - float(step_time.iloc[0])
    out["T"] = float(out["temperature_C"].iloc[0])
    out["TempLabel"] = f"{int(float(out['temperature_C'].iloc[0]))}C"
    out["Profile"] = str(out["profile"].iloc[0])
    if "SOC0_vinit_status" not in out.columns:
        out["SOC0_vinit_status"] = "ok"
    return out


def add_soc_labels(
    frame: pd.DataFrame,
    ocv_refs: dict[float, OcvReference],
    soc0_anchors: dict[tuple[float, str], dict[str, float]],
) -> pd.DataFrame:
    out = frame.copy()
    temp = float(out["temperature_C"].iloc[0])
    profile = str(out["profile"].iloc[0]).upper()
    ref = ocv_refs.get(temp)
    anchor = soc0_anchors.get((temp, profile))
    actual_soc0_vinit = float(out["SOC0_Vinit(V)"].iloc[0])
    use_anchor = False
    if anchor is not None:
        anchor_v = float(anchor["soc0_vinit"])
        use_anchor = abs(actual_soc0_vinit - anchor_v) <= 5e-3 or ref is None or ref.voltage_v is None
    if anchor is not None and use_anchor:
        soc0 = float(anchor["soc0"])
        soc0_vinit = float(anchor["soc0_vinit"])
        q_ref = float(anchor["q_ref"])
        soc0_mode = "ocv_inferred_from_SOC0_Vinit_and_lc_ocv_curve"
    elif ref is not None:
        soc0_vinit = actual_soc0_vinit
        soc0 = interpolate_soc0_from_ocv(ref, soc0_vinit)
        q_ref = float(ref.q_ref_ah)
        soc0_mode = "ocv_inferred_from_SOC0_Vinit_and_lc_ocv_curve"
    else:
        raise ValueError(
            f"Missing LC-OCV SOC reference for {temp:g} C. "
            "Provide low-current OCV files or ocv_inferred_start_soc_by_file.csv/lc_ocv_capacity_reference.csv."
        )

    qdis, qchg, qnet, q_source = compute_q_removed(out)
    qnet_np = qnet.to_numpy(float)
    qnet_final = float(np.nanmax(qnet_np)) if np.isfinite(qnet_np).any() else np.nan
    if not np.isfinite(q_ref) or q_ref <= 0:
        raise ValueError(f"Invalid Q_ref_lc_ocv_Ah for {temp:g} C: {q_ref}")
    soc_unclipped = soc0 - qnet_np / q_ref
    soc = np.clip(soc_unclipped, 0.0, 1.0)
    q_from_current = integrate_current_removed_ah(out)

    out["Qdis_seg(Ah)"] = qdis.to_numpy(float)
    out["Qchg_seg(Ah)"] = qchg.to_numpy(float)
    out["Qnet_removed(Ah)"] = qnet_np
    out["Qnet_denom(Ah)"] = q_ref
    out["progress_p"] = qnet_np / qnet_final if np.isfinite(qnet_final) and qnet_final > 0 else np.nan
    out["SOC0_used"] = soc0
    out["SOC_CC"] = soc
    out["SOC_CC(%)"] = soc * 100.0
    out["SOC_percent"] = soc * 100.0
    out["CE_used"] = 1.0
    out["SOC_scale_mode"] = "SOC0_ocv_file_minus_Qremoved_over_temp_lc_ocv_capacity"
    out["DENOM_MODE"] = "temperature_lc_ocv_discharge_capacity"
    out["ENFORCE_MONOTONIC"] = True
    out["SOC0_mode"] = soc0_mode
    out["SOC0_Vinit(V)"] = soc0_vinit
    out["Qeff_ocv_Ah(debug)"] = soc0 * q_ref
    out["Qeff_reaches_cutoff(debug)"] = bool(np.nanmin(soc_unclipped) <= 0.0)
    out["SOC0_OCV_inferred"] = soc0
    out["SOC0_OCV_inferred(%)"] = soc0 * 100.0
    out["Q_ref_lc_ocv_Ah"] = q_ref
    out["Q_available_from_SOC0_OCV_Ah"] = soc0 * q_ref
    out["Q_removed_for_OCV_start_label_Ah"] = qnet_np
    out["Q_removed_from_current_recomputed_Ah"] = q_from_current
    out["SOC_CC_unclipped"] = soc_unclipped
    out["Q_removed_primary_source"] = q_source
    out["SOC0_anchor_source"] = "summary_anchor_table" if anchor is not None and use_anchor else (ref.source if ref is not None else "missing")
    return out


def ordered_output_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "Data_Point",
        "Test_Time(s)",
        "Date_Time",
        "Step_Time(s)",
        "Step_Index",
        "Cycle_Index",
        "Current(A)",
        "Voltage(V)",
        "Charge_Capacity(Ah)",
        "Discharge_Capacity(Ah)",
        "Charge_Energy(Wh)",
        "Discharge_Energy(Wh)",
        "dV/dt(V/s)",
        "Internal_Resistance(Ohm)",
        "Is_FC_Data",
        "AC_Impedance(Ohm)",
        "ACI_Phase_Angle(Deg)",
        "time_s",
        "t_global(s)",
        "V_raw",
        "I_raw",
        "T",
        "SOC_percent",
        "Qdis_seg(Ah)",
        "Qchg_seg(Ah)",
        "Qnet_removed(Ah)",
        "Qnet_denom(Ah)",
        "progress_p",
        "SOC0_used",
        "SOC_CC",
        "SOC_CC(%)",
        "CE_used",
        "SOC_scale_mode",
        "DENOM_MODE",
        "ENFORCE_MONOTONIC",
        "TempLabel",
        "Profile",
        "profile",
        "temperature_C",
        "source_file",
        "DriveExtractMethod",
        "DriveStepIndex",
        "SOC0_mode",
        "SOC0_Vinit(V)",
        "SOC0_restStep",
        "SOC0_vinit_status",
        "Qeff_ocv_Ah(debug)",
        "Qeff_reaches_cutoff(debug)",
        "SOC0_OCV_inferred",
        "SOC0_OCV_inferred(%)",
        "Q_ref_lc_ocv_Ah",
        "Q_available_from_SOC0_OCV_Ah",
        "Q_removed_for_OCV_start_label_Ah",
        "Q_removed_from_current_recomputed_Ah",
        "SOC_CC_unclipped",
        "Q_removed_primary_source",
        *[col for col in REQUIRED_OUTPUT_COLUMNS if col not in {"time_s", "V_raw", "I_raw", "T", "SOC_percent", "profile", "temperature_C", "source_file"}],
    ]
    out = [col for col in preferred if col in frame.columns]
    out.extend([col for col in frame.columns if col not in out])
    return out


def prepare_dataset(
    raw_dir: Path,
    reference_dir: Path,
    reference_metadata_dir: Path,
    out_dir: Path,
    expected_soc_token_percent: float,
    allow_incomplete: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    dynamic_paths = sorted([*raw_dir.rglob("*.xls*"), *raw_dir.rglob("*.csv")])
    for path in dynamic_paths:
        frame = standardize_frame(path)
        if frame["profile"].iloc[0] == "UNKNOWN" or not np.isfinite(frame["temperature_C"].iloc[0]):
            print(f"SKIP metadata not inferred: {path.name}")
            continue
        filename_soc = infer_filename_soc_token_percent(path)
        frame.attrs["filename_soc_token_percent"] = filename_soc if filename_soc is not None else expected_soc_token_percent
        frame.attrs["filename_soc_token_source"] = "filename" if filename_soc is not None else "expected_token_default"
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No dynamic profile Excel files found in {raw_dir}")
    if not allow_incomplete:
        validate_manuscript_dynamic_set(frames)

    soc0_anchors = load_soc0_anchor_table(reference_metadata_dir)
    qref_by_temp = load_qref_table(reference_metadata_dir)
    ocv_refs: dict[float, OcvReference] = {}
    for path in sorted(reference_dir.glob("*.xls*")):
        if "ocv" not in path.name.lower():
            continue
        ref = build_ocv_reference_from_file(path, qref_by_temp)
        if ref is not None:
            ocv_refs[ref.temperature_C] = ref
    for temp, qref in qref_by_temp.items():
        ocv_refs.setdefault(temp, OcvReference(temp, qref, None, None, "lc_ocv_capacity_reference.csv"))
    drive_frames = [extract_drive_segment(frame) for frame in frames]

    r0_rows = []
    for frame in drive_frames:
        est = estimate_r0(frame)
        r0_rows.append(est.__dict__)
    r0_table = pd.DataFrame(r0_rows)
    r0_by_temp = r0_table.groupby("temperature_C")["r0_ohm"].median().to_dict()
    key_counts: dict[tuple[float, str], int] = {}
    for frame in drive_frames:
        key = (float(frame["temperature_C"].iloc[0]), str(frame["profile"].iloc[0]))
        key_counts[key] = key_counts.get(key, 0) + 1

    manifest_rows = []
    for frame in drive_frames:
        temp = float(frame["temperature_C"].iloc[0])
        profile = str(frame["profile"].iloc[0])
        r0 = float(r0_by_temp.get(temp, 0.05))
        start_soc = float(frame.attrs.get("filename_soc_token_percent", expected_soc_token_percent))
        start_soc_source = str(frame.attrs.get("filename_soc_token_source", "expected_token_default"))
        frame = add_soc_labels(frame, ocv_refs, soc0_anchors)
        frame = add_features(frame, r0)
        frame = frame[ordered_output_columns(frame)]
        key = (temp, profile)
        soc_suffix = f"_{format_soc_for_filename(start_soc)}SOC" if key_counts.get(key, 0) > 1 else ""
        out_file = out_dir / f"NMC_{int(temp)}C_{profile}{soc_suffix}.csv"
        frame.to_csv(out_file, index=False, lineterminator="\n")
        manifest_rows.append(
            {
                "file": out_file.name,
                "raw_source_file": str(frame["source_file"].iloc[0]),
                "profile": profile,
                "temperature_C": temp,
                "n_samples": len(frame),
                "duration_s": float(frame["time_s"].max() - frame["time_s"].min()),
                "SOC0_OCV_inferred_percent": float(frame["SOC0_OCV_inferred(%)"].iloc[0]),
                "Q_ref_lc_ocv_Ah": float(frame["Q_ref_lc_ocv_Ah"].iloc[0]),
                "filename_soc_token_percent": start_soc,
                "filename_soc_token_source": start_soc_source,
                "r0_ohm_used": r0,
                "drive_step_index": frame["DriveStepIndex"].iloc[0],
                "q_removed_primary_source": frame["Q_removed_primary_source"].iloc[0],
                "SOC_min_percent": float(frame["SOC_percent"].min()),
                "SOC_max_percent": float(frame["SOC_percent"].max()),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(out_dir / "processed_manifest.csv", index=False, lineterminator="\n")
    r0_table.to_csv(out_dir / "r0_estimates.csv", index=False, lineterminator="\n")
    print(f"Wrote processed CSV files to {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare CALCE NMC Excel files for CEMA-TCN training.")
    data_dir = Path(__file__).resolve().parents[1] / "data"
    parser.add_argument("--raw-dir", default=(data_dir / "raw_dynamic").as_posix())
    parser.add_argument("--reference-dir", default=(data_dir / "characterization" / "raw_reference").as_posix())
    parser.add_argument(
        "--reference-metadata-dir",
        default=(Path(__file__).resolve().parent / "locked_metadata").as_posix(),
        help="Tracked compact protocol metadata; raw OCV workbooks remain under --reference-dir.",
    )
    parser.add_argument("--out-dir", default=(data_dir / "preprocessed" / "nmc_ocvstart_lopo_clean").as_posix())
    parser.add_argument("--expected-soc-token-percent", type=float, default=80.0, help="Expected SOC token in manuscript dynamic-profile file names. This is a file-selection check, not the SOC label.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Allow partial dynamic-profile conversion instead of requiring the full 12-file manuscript 80SOC set.")
    args = parser.parse_args()
    prepare_dataset(
        Path(args.raw_dir), Path(args.reference_dir), Path(args.reference_metadata_dir),
        Path(args.out_dir), args.expected_soc_token_percent, args.allow_incomplete,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
