from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _repo_root() -> Path:
    p = Path(__file__).resolve().parents[1]
    if (p / "LFP_ABS_SOC").exists():
        return p
    return Path.cwd()


@dataclass
class CapacityAuditConfig:
    base_dir: Path = _repo_root()
    raw_dir: Path | None = None
    output_dir: Path | None = None
    decomposed_dir: Path | None = None
    existing_label_file: Path | None = None
    voltage_cutoff: float = 2.0
    charge_current_threshold: float = 0.02
    discharge_current_threshold: float = 0.02
    rest_current_threshold: float = 0.02
    ce_ok_threshold: float = 0.05
    low_current_threshold: float = 0.2
    min_valid_duration: float = 30.0
    nominal_capacity_Ah: float | None = None
    dt_sec_default: float = 1.0
    current_sign_convention: str = "auto"
    use_monotonic_q_smooth: bool = False
    label_chunk_size: int = 200_000

    def __post_init__(self):
        self.base_dir = Path(self.base_dir)
        self.raw_dir = Path(self.raw_dir) if self.raw_dir is not None else self.base_dir / "LFP_ABS_SOC"
        self.output_dir = Path(self.output_dir) if self.output_dir is not None else self.base_dir
        self.decomposed_dir = Path(self.decomposed_dir) if self.decomposed_dir is not None else None
        self.output_dir.mkdir(parents=True, exist_ok=True)


def parse_temp_key(temp_key: str) -> float:
    s = str(temp_key).strip().upper()
    if s.startswith("N"):
        return -float(s[1:])
    return float(s)


def parse_metadata(path: Path) -> dict:
    parts = path.stem.split("_")
    if len(parts) >= 3 and parts[0] == "LFP" and parts[1].upper() == "OCV":
        temp_key = parts[2]
        profile = "OCV"
    elif len(parts) >= 3 and parts[0] == "LFP":
        temp_key = parts[1]
        profile = "_".join(parts[2:])
    else:
        temp_key = "unknown"
        profile = "unknown"
    try:
        temp_c = parse_temp_key(temp_key)
    except Exception:
        temp_c = float("nan")
    test_type = {
        "OCV": "low_current_ocv",
        "DST": "DST",
        "US06": "US06",
        "FUDS": "FUDS",
    }.get(profile, "unknown")
    return {
        "file_name": path.name,
        "trajectory_id": path.stem,
        "temperature_key": temp_key,
        "temperature_C": temp_c,
        "drive_cycle": profile,
        "test_type": test_type,
    }


def num(df: pd.DataFrame, col: str, default=np.nan) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(np.float64)


def scalar_median(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    arr = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.nanmedian(arr)) if arr.size else float("nan")


def bool_median(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return np.nan
    s = df[col]
    if s.dtype == bool:
        return bool(s.mode(dropna=True).iloc[0]) if len(s.dropna()) else np.nan
    txt = s.astype(str).str.lower()
    vals = txt.map({"true": True, "false": False, "1": True, "0": False})
    vals = vals.dropna()
    return bool(vals.mode().iloc[0]) if len(vals) else np.nan


def time_vector(df: pd.DataFrame, cfg: CapacityAuditConfig) -> tuple[np.ndarray, str]:
    for col in ["t_rel(s)", "Test_Time(s)", "Step_Time(s)"]:
        if col in df.columns:
            t = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float64)
            if np.isfinite(t).sum() > max(3, 0.5 * len(t)):
                return t, col
    return np.arange(len(df), dtype=np.float64) * float(cfg.dt_sec_default), "row_index_default_dt"


def dt_from_time(t: np.ndarray, cfg: CapacityAuditConfig) -> tuple[np.ndarray, dict]:
    t = np.asarray(t, dtype=np.float64)
    finite = np.isfinite(t)
    missing = int((~finite).sum())
    t2 = t.copy()
    if finite.any():
        fill = np.nanmedian(t2[finite])
        t2[~finite] = fill
    else:
        t2 = np.arange(len(t2), dtype=np.float64) * float(cfg.dt_sec_default)
    raw_dt = np.diff(t2, prepend=t2[:1])
    pos = raw_dt[np.isfinite(raw_dt) & (raw_dt > 0)]
    med = float(np.nanmedian(pos)) if pos.size else float(cfg.dt_sec_default)
    dt = raw_dt.copy()
    bad = ~np.isfinite(dt) | (dt <= 0) | (dt > max(10.0 * med, med + 60.0))
    duplicate = int((raw_dt == 0).sum())
    dt[bad] = med
    if len(dt):
        dt[0] = med
    meta = {
        "dt_mean": float(np.nanmean(dt)) if len(dt) else float("nan"),
        "dt_std": float(np.nanstd(dt)) if len(dt) else float("nan"),
        "dt_min": float(np.nanmin(dt)) if len(dt) else float("nan"),
        "dt_max": float(np.nanmax(dt)) if len(dt) else float("nan"),
        "missing_timestamp_count": missing,
        "duplicate_timestamp_count": duplicate,
        "bad_dt_count": int(bad.sum()),
    }
    return dt, meta


def infer_current_sign(df: pd.DataFrame, I: np.ndarray) -> tuple[int, str, float]:
    """Return +1 if discharge current is positive, -1 if discharge current is negative."""
    I = np.asarray(I, dtype=np.float64)
    if "label" in df.columns:
        label = df["label"].astype(str).str.lower()
        is_dis = label.str.contains("discharge", na=False) & ~label.str.contains("charge_ocv", na=False)
        is_chg = label.str.contains("charge", na=False) & ~label.str.contains("discharge", na=False)
        dis = I[is_dis.to_numpy()]
        chg = I[is_chg.to_numpy()]
        if len(dis) and len(chg):
            dis_med = float(np.nanmedian(dis))
            chg_med = float(np.nanmedian(chg))
            if abs(dis_med) > 1e-5 and np.sign(dis_med) != np.sign(chg_med):
                return (1 if dis_med > 0 else -1), "label_median_current", dis_med
    for qcol in ["Qdis_cum(Ah)", "Discharge_Capacity(Ah)", "abs_discharge_ah"]:
        if qcol in df.columns:
            q = pd.to_numeric(df[qcol], errors="coerce").to_numpy(np.float64)
            dq = np.diff(q, prepend=q[:1])
            m = np.isfinite(dq) & np.isfinite(I) & (np.abs(dq) > 1e-9) & (np.abs(I) > 1e-5)
            if m.sum() >= 10:
                corr = float(np.corrcoef(dq[m], I[m])[0, 1])
                if np.isfinite(corr):
                    return (1 if corr > 0 else -1), f"corr({qcol},Current)", corr
    return -1, "default_discharge_negative", float("nan")


def integrate_current(I: np.ndarray, dt: np.ndarray, sign: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sign not in (-1, 1):
        raise ValueError("sign must be +1 or -1")
    I = np.asarray(I, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)
    dis_current = np.maximum(sign * I, 0.0)
    chg_current = np.maximum(-sign * I, 0.0)
    qdis = np.cumsum(dis_current * dt / 3600.0)
    qchg = np.cumsum(chg_current * dt / 3600.0)
    net = qdis - qchg
    return qdis, qchg, net


def estimate_rest_bias(I: np.ndarray, cfg: CapacityAuditConfig) -> tuple[float, int]:
    I = np.asarray(I, dtype=np.float64)
    m = np.isfinite(I) & (np.abs(I) <= float(cfg.rest_current_threshold))
    if m.sum() < 20:
        finite = I[np.isfinite(I)]
        if finite.size:
            cutoff = np.nanpercentile(np.abs(finite), 10.0)
            m = np.isfinite(I) & (np.abs(I) <= max(cutoff, 1e-5))
    if m.sum() == 0:
        return 0.0, 0
    return float(np.nanmedian(I[m])), int(m.sum())


def first_cutoff_index(V: np.ndarray, cutoff: float) -> tuple[bool, int]:
    m = np.where(np.asarray(V, dtype=np.float64) <= float(cutoff))[0]
    if len(m):
        return True, int(m[0])
    return False, int(np.nanargmin(V)) if len(V) else -1


def normalize_fraction_like(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    finite = out[np.isfinite(out)]
    if finite.size and np.nanmax(finite) > 1.5:
        out = out / 100.0
    return out


def existing_q_ref_from_label(df: pd.DataFrame, net_delivered: np.ndarray) -> float:
    if "SOC_CC" not in df.columns or len(df) < 2:
        return float("nan")
    soc = normalize_fraction_like(pd.to_numeric(df["SOC_CC"], errors="coerce").to_numpy(np.float64))
    dsoc = float(soc[0] - soc[-1])
    q = float(net_delivered[-1] - net_delivered[0]) if len(net_delivered) else float("nan")
    if np.isfinite(dsoc) and abs(dsoc) > 1e-9 and np.isfinite(q):
        return q / dsoc
    return float("nan")


def classify_ce_failure(row: dict, all_q_by_temp: pd.DataFrame, cfg: CapacityAuditConfig) -> str:
    if bool(row.get("CE_ok_bias_corrected", False)):
        return "ok_after_bias_correction"
    if row.get("duplicate_timestamp_count", 0) > 5 or row.get("bad_dt_count", 0) > 5:
        return "timestamp_dt_error"
    if row.get("missing_timestamp_count", 0) > 0:
        return "missing_or_duplicate_timestamp"
    if row.get("CE_ok_raw") is True:
        return "ok"
    ce = row.get("CE_raw")
    ce_corr = row.get("CE_bias_corrected")
    if np.isfinite(ce) and np.isfinite(ce_corr) and abs(ce_corr - 1.0) < abs(ce - 1.0) * 0.4:
        return "current_bias_at_rest"
    if str(row.get("test_type")) == "low_current_ocv":
        if row.get("cutoff_reached") and np.isfinite(row.get("Q_discharge_low_current_Ah", np.nan)):
            temp = row.get("temperature_C")
            measured = row.get("Q_discharge_low_current_Ah")
            peers = all_q_by_temp[
                (all_q_by_temp["temperature_C"].ne(temp))
                & (all_q_by_temp["CE_ok_raw"].eq(True))
                & np.isfinite(all_q_by_temp["Q_discharge_low_current_Ah"])
            ]
            if len(peers):
                lo = float(np.nanpercentile(peers["Q_discharge_low_current_Ah"], 10))
                if measured < 0.9 * lo:
                    return "low_temperature_polarization_cutoff"
        if not row.get("charge_reaches_high_existing", True) or row.get("V_max", np.nan) < 3.5:
            return "incomplete_charge_or_discharge"
        return "capacity_reference_mismatch"
    if row.get("cutoff_reached") and row.get("duration_s", 0) < cfg.min_valid_duration:
        return "premature_voltage_cutoff"
    return "capacity_reference_mismatch" if np.isfinite(ce) else "unknown"


def audit_one_file(path: Path, cfg: CapacityAuditConfig) -> dict:
    df = pd.read_csv(path)
    meta = parse_metadata(path)
    t, time_col = time_vector(df, cfg)
    dt, dt_meta = dt_from_time(t, cfg)
    V = num(df, "Voltage(V)")
    I = num(df, "Current(A)")
    sign, sign_source, sign_score = infer_current_sign(df, I)
    if cfg.current_sign_convention == "discharge_positive":
        sign, sign_source = 1, "config_discharge_positive"
    elif cfg.current_sign_convention == "discharge_negative":
        sign, sign_source = -1, "config_discharge_negative"
    qdis, qchg, net = integrate_current(I, dt, sign)
    bias, n_rest = estimate_rest_bias(I, cfg)
    qdis_corr, qchg_corr, net_corr = integrate_current(I - bias, dt, sign)
    cutoff_reached, cutoff_idx = first_cutoff_index(V, cfg.voltage_cutoff)
    duration = float(np.nansum(dt))
    charge = float(qchg[-1]) if len(qchg) else float("nan")
    discharge = float(qdis[-1]) if len(qdis) else float("nan")
    charge_corr = float(qchg_corr[-1]) if len(qchg_corr) else float("nan")
    discharge_corr = float(qdis_corr[-1]) if len(qdis_corr) else float("nan")
    qdis_ocv = scalar_median(df, "Qdis_ocv_Ah")
    qchg_ocv = scalar_median(df, "Qchg_ocv_Ah")
    if not np.isfinite(qdis_ocv):
        qdis_ocv = float(np.nanmax(num(df, "abs_discharge_ah"))) if "abs_discharge_ah" in df.columns else float("nan")
    if not np.isfinite(qchg_ocv):
        qchg_ocv = float(np.nanmax(num(df, "abs_charge_ah"))) if "abs_charge_ah" in df.columns else float("nan")
    ce_from_ocv = qdis_ocv / qchg_ocv if np.isfinite(qdis_ocv) and np.isfinite(qchg_ocv) and qchg_ocv > 1e-9 else float("nan")
    ce_integrated = discharge / charge if charge > 1e-9 else float("nan")
    ce_corr_integrated = discharge_corr / charge_corr if charge_corr > 1e-9 else float("nan")
    existing_ce_ok = bool_median(df, "CE_ok")
    existing_ce = scalar_median(df, "CE_est")
    ce_raw = ce_from_ocv if np.isfinite(ce_from_ocv) else (existing_ce if np.isfinite(existing_ce) else ce_integrated)
    ce_ok_raw = bool(np.isfinite(ce_raw) and abs(ce_raw - 1.0) <= float(cfg.ce_ok_threshold))
    ce_corr = ce_corr_integrated if meta["test_type"] == "low_current_ocv" else float("nan")
    ce_ok_corr = bool(np.isfinite(ce_corr) and abs(ce_corr - 1.0) <= float(cfg.ce_ok_threshold))
    ce_used_for_label = scalar_median(df, "CE_used")
    if not np.isfinite(ce_used_for_label) or ce_used_for_label <= 0:
        ce_used_for_label = 1.0
    if "Qdis_cum(Ah)" in df.columns and "Qchg_cum(Ah)" in df.columns:
        src_qdis = num(df, "Qdis_cum(Ah)")
        src_qchg = num(df, "Qchg_cum(Ah)")
        source_net = src_qdis - ce_used_for_label * src_qchg
    else:
        source_net = net
    q_drive = float(source_net[-1] - source_net[0]) if len(source_net) else float("nan")
    q_drive_corr = float(net_corr[-1] - net_corr[0]) if len(net_corr) else float("nan")
    existing_soc = normalize_fraction_like(num(df, "SOC_CC"))
    existing_use = normalize_fraction_like(num(df, "SOC_use"))
    q_ref_existing = existing_q_ref_from_label(df, source_net)
    if not np.isfinite(q_ref_existing) and np.isfinite(qdis_ocv):
        q_ref_existing = qdis_ocv
    q_candidate_vals = [x for x in [qdis_ocv, qchg_ocv] if np.isfinite(x) and x > 0]
    q_candidate = float(np.nanmedian(q_candidate_vals)) if q_candidate_vals else float("nan")
    q_candidate_source = "median(Qdis_ocv,Qchg_ocv)" if len(q_candidate_vals) > 1 else "single_ocv_candidate"
    cutoff_physical = 1.0 - q_drive / q_candidate if np.isfinite(q_candidate) and q_candidate > 0 else float("nan")
    cutoff_usable = 1.0 - q_drive / q_drive if np.isfinite(q_drive) and abs(q_drive) > 1e-9 else float("nan")
    label_forced = bool(
        np.isfinite(existing_soc).any()
        and abs(float(existing_soc[-1])) < 0.02
        and cutoff_reached
        and np.isfinite(cutoff_physical)
        and abs(cutoff_physical) > 0.03
    )
    return {
        **meta,
        "time_column_used": time_col,
        "start_time": float(t[0]) if len(t) else float("nan"),
        "end_time": float(t[-1]) if len(t) else float("nan"),
        "duration_s": duration,
        **dt_meta,
        "V_start": float(V[0]) if len(V) else float("nan"),
        "V_end": float(V[-1]) if len(V) else float("nan"),
        "V_min": float(np.nanmin(V)) if len(V) else float("nan"),
        "V_max": float(np.nanmax(V)) if len(V) else float("nan"),
        "cutoff_reached": bool(cutoff_reached),
        "cutoff_voltage": float(cfg.voltage_cutoff),
        "cutoff_index": int(cutoff_idx),
        "premature_cutoff_flag": bool(cutoff_reached and cutoff_idx >= 0 and cutoff_idx < 0.5 * len(df)),
        "I_mean": float(np.nanmean(I)) if len(I) else float("nan"),
        "I_abs_mean": float(np.nanmean(np.abs(I))) if len(I) else float("nan"),
        "I_min": float(np.nanmin(I)) if len(I) else float("nan"),
        "I_max": float(np.nanmax(I)) if len(I) else float("nan"),
        "current_sign_convention": "discharge_positive" if sign == 1 else "discharge_negative",
        "current_sign_source": sign_source,
        "current_sign_score": sign_score,
        "rest_current_bias_estimate": bias,
        "rest_current_bias_n": n_rest,
        "charge_Ah_raw": charge,
        "discharge_Ah_raw": discharge,
        "charge_Ah_bias_corrected": charge_corr,
        "discharge_Ah_bias_corrected": discharge_corr,
        "CE_raw": ce_raw,
        "CE_integrated_from_current": ce_integrated,
        "CE_from_low_current_capacity": ce_from_ocv,
        "CE_bias_corrected": ce_corr,
        "CE_bias_corrected_integrated_from_current": ce_corr_integrated,
        "CE_ok_raw": ce_ok_raw,
        "CE_ok_bias_corrected": ce_ok_corr,
        "CE_ok_existing": existing_ce_ok,
        "CE_est_existing": existing_ce,
        "Q_discharge_low_current_Ah": qdis_ocv if meta["test_type"] == "low_current_ocv" or np.isfinite(qdis_ocv) else float("nan"),
        "Q_charge_low_current_Ah": qchg_ocv if meta["test_type"] == "low_current_ocv" or np.isfinite(qchg_ocv) else float("nan"),
        "Q_drive_delivered_to_cutoff_Ah": q_drive,
        "Q_drive_delivered_to_cutoff_bias_corrected_Ah": q_drive_corr,
        "Q_nominal_Ah": float(cfg.nominal_capacity_Ah) if cfg.nominal_capacity_Ah is not None else float("nan"),
        "Q_ref_existing_label_Ah": q_ref_existing,
        "Q_ref_candidate_Ah": q_candidate,
        "Q_ref_candidate_source": q_candidate_source,
        "existing_SOC_start": float(existing_soc[0]) if len(existing_soc) and np.isfinite(existing_soc[0]) else float("nan"),
        "existing_SOC_end": float(existing_soc[-1]) if len(existing_soc) and np.isfinite(existing_soc[-1]) else float("nan"),
        "physical_SOC_end_existing": float(existing_soc[-1]) if len(existing_soc) and np.isfinite(existing_soc[-1]) else float("nan"),
        "usable_SOC_end_existing": float(existing_use[-1]) if len(existing_use) and np.isfinite(existing_use[-1]) else float("nan"),
        "cutoff_physical_SOC_if_Q_candidate": cutoff_physical,
        "cutoff_usable_SOC_if_Q_candidate": cutoff_usable,
        "physical_minus_usable_at_cutoff_if_Q_candidate": cutoff_physical - cutoff_usable if np.isfinite(cutoff_physical) and np.isfinite(cutoff_usable) else float("nan"),
        "label_forced_100_to_0_flag": label_forced,
        "charge_reaches_high_existing": bool_median(df, "chg_reaches_high"),
        "discharge_reaches_low_existing": bool_median(df, "dis_reaches_low"),
    }


def run_effective_capacity_audit(cfg: CapacityAuditConfig) -> pd.DataFrame:
    paths = sorted(cfg.raw_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {cfg.raw_dir}")
    rows = [audit_one_file(p, cfg) for p in paths]
    audit = pd.DataFrame(rows)
    audit["CE_ok_effective"] = audit["CE_ok_existing"]
    missing_existing = audit["CE_ok_effective"].isna()
    audit.loc[missing_existing, "CE_ok_effective"] = audit.loc[missing_existing, "CE_ok_raw"]
    audit["CE_ok_effective"] = audit["CE_ok_effective"].astype(bool)
    reasons = []
    for _, row in audit.iterrows():
        if row.get("CE_ok_effective") is True:
            reasons.append("ok")
        else:
            reasons.append(classify_ce_failure(row.to_dict(), audit, cfg))
    audit["CE_failure_reason"] = reasons
    audit.to_csv(cfg.output_dir / "effective_capacity_audit.csv", index=False)
    fail = audit[audit["CE_ok_effective"].eq(False)].copy()
    fail.to_csv(cfg.output_dir / "ce_failure_audit.csv", index=False)
    summary = (
        audit.groupby(["temperature_C", "test_type", "drive_cycle"], dropna=False)
        .agg(
            n_files=("file_name", "count"),
            CE_raw_mean=("CE_raw", "mean"),
            CE_ok_existing_rate=("CE_ok_existing", "mean"),
            Q_discharge_low_current_Ah=("Q_discharge_low_current_Ah", "median"),
            Q_charge_low_current_Ah=("Q_charge_low_current_Ah", "median"),
            Q_drive_delivered_to_cutoff_Ah=("Q_drive_delivered_to_cutoff_Ah", "median"),
            Q_ref_existing_label_Ah=("Q_ref_existing_label_Ah", "median"),
            cutoff_physical_SOC_if_Q_candidate=("cutoff_physical_SOC_if_Q_candidate", "median"),
        )
        .reset_index()
        .sort_values(["temperature_C", "test_type", "drive_cycle"])
    )
    summary.to_csv(cfg.output_dir / "capacity_candidate_summary.csv", index=False)
    return audit


def qeff_survey(audit: pd.DataFrame, cfg: CapacityAuditConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    temps = sorted(audit["temperature_C"].dropna().unique())
    rows = []
    for temp in temps:
        g = audit[audit["temperature_C"].eq(temp)]
        low = g[np.isfinite(g["Q_discharge_low_current_Ah"])]
        ce_ok = bool(g["CE_ok_effective"].dropna().astype(bool).mode().iloc[0]) if len(g["CE_ok_effective"].dropna()) else False
        row = {
            "temperature_C": temp,
            "Q_low_current_discharge_Ah": float(np.nanmedian(low["Q_discharge_low_current_Ah"])) if len(low) else float("nan"),
            "Q_low_current_charge_Ah": float(np.nanmedian(low["Q_charge_low_current_Ah"])) if len(low) else float("nan"),
            "Q_drive_FUDS_to_cutoff_Ah": float(np.nanmedian(g[g["drive_cycle"].eq("FUDS")]["Q_drive_delivered_to_cutoff_Ah"])),
            "Q_drive_DST_to_cutoff_Ah": float(np.nanmedian(g[g["drive_cycle"].eq("DST")]["Q_drive_delivered_to_cutoff_Ah"])),
            "Q_drive_US06_to_cutoff_Ah": float(np.nanmedian(g[g["drive_cycle"].eq("US06")]["Q_drive_delivered_to_cutoff_Ah"])),
            "Q_existing_label_Ah": float(np.nanmedian(g["Q_ref_existing_label_Ah"])),
            "CE_ok": ce_ok,
            "CE_failure_reason": ";".join(sorted(set(map(str, g[g["CE_ok_effective"].eq(False)]["CE_failure_reason"].dropna())))),
        }
        rows.append(row)
    survey = pd.DataFrame(rows).sort_values("temperature_C")
    valid = survey[survey["CE_ok"].eq(True) & np.isfinite(survey["Q_low_current_discharge_Ah"])].copy()
    x = valid["temperature_C"].to_numpy(np.float64)
    y = valid["Q_low_current_discharge_Ah"].to_numpy(np.float64)
    allx = survey["temperature_C"].to_numpy(np.float64)
    if len(x) >= 2:
        measured_only = np.interp(allx, x, y, left=np.nan, right=np.nan)
        robust_spline = np.interp(allx, x, y, left=y[0], right=y[-1])
    elif len(x) == 1:
        measured_only = np.full_like(allx, np.nan, dtype=np.float64)
        robust_spline = np.full_like(allx, y[0], dtype=np.float64)
    else:
        measured_only = np.full_like(allx, np.nan, dtype=np.float64)
        robust_spline = survey["Q_low_current_discharge_Ah"].to_numpy(np.float64)
    if cfg.use_monotonic_q_smooth and len(robust_spline):
        robust_spline = np.maximum.accumulate(robust_spline)
    survey["Q_smooth_fit_Ah"] = robust_spline
    survey["Q_measured_only_fit_Ah"] = measured_only
    survey["Q_exclude_CE_fail_Ah"] = robust_spline
    survey["Q_include_CE_fail_as_lower_bound_Ah"] = np.maximum(
        robust_spline,
        np.nan_to_num(survey["Q_low_current_discharge_Ah"].to_numpy(np.float64), nan=-np.inf),
    )
    rec_rows = []
    variant_rows = []
    for _, r in survey.iterrows():
        temp = r["temperature_C"]
        measured = r["Q_low_current_discharge_Ah"]
        smooth = r["Q_smooth_fit_Ah"]
        ce_ok = bool(r["CE_ok"])
        if ce_ok and np.isfinite(measured):
            q_final = measured
            conf = "high"
            reason = "CE_ok low-current discharge capacity"
        elif np.isfinite(smooth):
            q_final = smooth
            conf = "low" if temp <= -5 else "medium"
            reason = "CE_fail excluded from fit; robust smooth Q_eff used"
        else:
            q_final = measured
            conf = "low"
            reason = "no CE_ok smooth fit available; measured value is diagnostic only"
        lower_flag = bool((not ce_ok) and np.isfinite(measured) and np.isfinite(smooth) and measured < 0.98 * smooth)
        rec_rows.append({
            "temperature_C": temp,
            "Q_final_recommended_Ah": q_final,
            "Q_confidence": conf,
            "reason": reason,
            "CE_ok": ce_ok,
            "low_current_capacity_lower_bound_flag": lower_flag,
            "Q_ref_selection_uses_test_error": False,
        })
        for name, val in [
            ("measured_only", measured if ce_ok else np.nan),
            ("robust_spline", smooth),
            ("exclude_CE_fail", smooth),
            ("include_CE_fail_as_lower_bound", max(smooth, measured) if np.isfinite(measured) and np.isfinite(smooth) else smooth),
        ]:
            variant_rows.append({"temperature_C": temp, "fit_variant": name, "Q_eff_Ah": val})
    variants = pd.DataFrame(variant_rows)
    rec = pd.DataFrame(rec_rows)
    survey = survey.merge(rec, on=["temperature_C", "CE_ok"], how="left")
    survey.to_csv(cfg.output_dir / "qeff_temperature_survey.csv", index=False)
    variants.to_csv(cfg.output_dir / "qeff_fit_variants.csv", index=False)
    rec.to_csv(cfg.output_dir / "qeff_final_recommendation.csv", index=False)
    return survey, variants, rec


def build_label_rows_for_file(path: Path, cfg: CapacityAuditConfig, qrec: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)
    meta = parse_metadata(path)
    t, _ = time_vector(df, cfg)
    dt, _ = dt_from_time(t, cfg)
    V = num(df, "Voltage(V)")
    I = num(df, "Current(A)")
    sign, _, _ = infer_current_sign(df, I)
    qdis, qchg, net = integrate_current(I, dt, sign)
    ce_used = scalar_median(df, "CE_used")
    if not np.isfinite(ce_used) or ce_used <= 0:
        ce_used = 1.0
    net_ce = qdis - ce_used * qchg
    net_ce = net_ce - net_ce[0]
    existing = normalize_fraction_like(num(df, "SOC_CC"))
    existing_use = normalize_fraction_like(num(df, "SOC_use"))
    soc0 = existing[0] if len(existing) and np.isfinite(existing[0]) else 1.0
    temp = meta["temperature_C"]
    rr = qrec[qrec["temperature_C"].eq(temp)]
    q_smooth = float(rr["Q_final_recommended_Ah"].iloc[0]) if len(rr) else float("nan")
    q_conf = str(rr["Q_confidence"].iloc[0]) if len(rr) else "low"
    qdis_ocv = scalar_median(df, "Qdis_ocv_Ah")
    qchg_ocv = scalar_median(df, "Qchg_ocv_Ah")
    q_meas = qdis_ocv if np.isfinite(qdis_ocv) else q_smooth
    ce_ok = bool_median(df, "CE_ok")
    ce_reason = ""
    if ce_ok is False:
        ce_reason = "CE_ok_false_in_source"
    phys_meas = soc0 - net_ce / max(q_meas, 1e-9) if np.isfinite(q_meas) else np.full(len(df), np.nan)
    phys_smooth = soc0 - net_ce / max(q_smooth, 1e-9) if np.isfinite(q_smooth) else np.full(len(df), np.nan)
    q_cut = float(net_ce[-1]) if len(net_ce) else float("nan")
    usable = 1.0 - net_ce / q_cut if np.isfinite(q_cut) and abs(q_cut) > 1e-9 else np.full(len(df), np.nan)
    out = pd.DataFrame({
        "trajectory_id": meta["trajectory_id"],
        "file_name": meta["file_name"],
        "temperature_C": temp,
        "drive_cycle": meta["drive_cycle"],
        "test_type": meta["test_type"],
        "time_index": np.arange(len(df), dtype=np.int64),
        "end_index": np.arange(len(df), dtype=np.int64),
        "time_s": t,
        "V_raw": V,
        "I_raw": I,
        "cumulative_discharge_Ah": net_ce,
        "SOC_existing": existing,
        "SOC_usable_existing": existing_use,
        "SOC_physical_measuredQ": phys_meas,
        "SOC_physical_smoothQ": phys_smooth,
        "SOC_usable_cutoff": usable,
        "SOC_start": soc0,
        "SOC_end_physical_measuredQ": phys_meas[-1] if len(df) else np.nan,
        "SOC_end_physical_smoothQ": phys_smooth[-1] if len(df) else np.nan,
        "SOC_end_usable_cutoff": usable[-1] if len(df) else np.nan,
        "cutoff_SOC": phys_smooth[-1] if len(df) else np.nan,
        "label_quality": "diagnostic_only_low" if ce_ok is False and temp <= -5 else q_conf,
        "label_source": "capacity_audit",
        "Q_ref_used_measuredQ_Ah": q_meas,
        "Q_ref_used_smoothQ_Ah": q_smooth,
        "Q_ref_source": "CE_ok measured Qdis_ocv" if ce_ok is True else "smoothQ for physical_smoothQ; measuredQ diagnostic only",
        "CE_ok_used": ce_ok,
        "CE_failure_reason_if_any": ce_reason,
    })
    return out


def generate_labels(cfg: CapacityAuditConfig, qrec: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = sorted(cfg.raw_dir.glob("*.csv"))
    frames = [build_label_rows_for_file(p, cfg, qrec) for p in paths]
    labels = pd.concat(frames, ignore_index=True)
    common = [
        "trajectory_id", "file_name", "temperature_C", "drive_cycle", "test_type", "time_index", "end_index",
        "time_s", "V_raw", "I_raw", "cumulative_discharge_Ah", "label_quality", "label_source",
        "Q_ref_source", "CE_ok_used", "CE_failure_reason_if_any",
    ]
    measured = labels[common + ["SOC_physical_measuredQ", "Q_ref_used_measuredQ_Ah"]].rename(
        columns={"SOC_physical_measuredQ": "SOC", "Q_ref_used_measuredQ_Ah": "Q_ref_used_Ah"}
    )
    smooth = labels[common + ["SOC_physical_smoothQ", "Q_ref_used_smoothQ_Ah"]].rename(
        columns={"SOC_physical_smoothQ": "SOC", "Q_ref_used_smoothQ_Ah": "Q_ref_used_Ah"}
    )
    usable = labels[common + ["SOC_usable_cutoff", "Q_ref_used_smoothQ_Ah"]].rename(
        columns={"SOC_usable_cutoff": "SOC", "Q_ref_used_smoothQ_Ah": "Q_ref_used_Ah"}
    )
    measured["label_variant"] = "physical_measuredQ"
    smooth["label_variant"] = "physical_smoothQ"
    usable["label_variant"] = "usable_cutoff"
    measured.to_csv(cfg.output_dir / "labels_physical_measuredQ.csv", index=False)
    smooth.to_csv(cfg.output_dir / "labels_physical_smoothQ.csv", index=False)
    usable.to_csv(cfg.output_dir / "labels_usable_cutoff.csv", index=False)
    comp_rows = []
    for tid, g in labels.groupby("trajectory_id"):
        last = g.iloc[-1]
        comp_rows.append({
            "trajectory_id": tid,
            "temperature_C": last["temperature_C"],
            "drive_cycle": last["drive_cycle"],
            "existing_SOC_end": last["SOC_existing"],
            "physical_measuredQ_end": last["SOC_physical_measuredQ"],
            "physical_smoothQ_end": last["SOC_physical_smoothQ"],
            "usable_cutoff_end": last["SOC_usable_cutoff"],
            "existing_minus_smoothQ_end": last["SOC_existing"] - last["SOC_physical_smoothQ"],
            "measuredQ_minus_smoothQ_end": last["SOC_physical_measuredQ"] - last["SOC_physical_smoothQ"],
            "existing_minus_usable_end": last["SOC_existing"] - last["SOC_usable_cutoff"],
            "label_quality": last["label_quality"],
        })
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(cfg.output_dir / "label_comparison_summary.csv", index=False)
    by_temp = (
        comp.groupby("temperature_C")
        .agg(
            n=("trajectory_id", "count"),
            existing_minus_smoothQ_end_mean=("existing_minus_smoothQ_end", "mean"),
            existing_minus_smoothQ_end_maxabs=("existing_minus_smoothQ_end", lambda s: float(np.nanmax(np.abs(s)))),
            measuredQ_minus_smoothQ_end_mean=("measuredQ_minus_smoothQ_end", "mean"),
            existing_minus_usable_end_mean=("existing_minus_usable_end", "mean"),
        )
        .reset_index()
    )
    by_temp.to_csv(cfg.output_dir / "label_difference_by_temperature.csv", index=False)
    return labels, measured, smooth, usable


def _metric_rows(df: pd.DataFrame, group_cols: list[str], model_source: str, sensitivity_mode: str) -> list[dict]:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        d = dict(zip(group_cols, keys))
        err = g["error_variant"].to_numpy(np.float64)
        ae = np.abs(err)
        rows.append({
            **d,
            "model_source_file": model_source,
            "sensitivity_mode": sensitivity_mode,
            "n": int(len(g)),
            "MAE": float(np.nanmean(ae)),
            "MAE_pct": float(np.nanmean(ae) * 100.0),
            "RMSE": float(np.sqrt(np.nanmean(err ** 2))),
            "RMSE_pct": float(np.sqrt(np.nanmean(err ** 2)) * 100.0),
            "Max_error_pct": float(np.nanmax(ae) * 100.0),
        })
    return rows


def run_label_sensitivity(cfg: CapacityAuditConfig, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_lookup = labels[[
        "trajectory_id", "end_index", "SOC_existing", "SOC_physical_measuredQ", "SOC_physical_smoothQ", "SOC_usable_cutoff",
        "temperature_C", "drive_cycle",
    ]].copy()
    variants = {
        "existing_label": "SOC_existing",
        "physical_measuredQ": "SOC_physical_measuredQ",
        "physical_smoothQ": "SOC_physical_smoothQ",
        "usable_cutoff": "SOC_usable_cutoff",
    }
    sources = [
        ("train_temp_minus10_0_10_25_50_prediction_rows.csv", ["A3_V_raw_I_T", "R5_GATED", "R5_raw_I_T_all_components"]),
        ("rex_prediction_rows.csv", ["R5_GATED_AUG_REX", "R5_GATED_REX", "R5_GATED"]),
        ("neural_ecm_prediction_rows.csv", ["NeuralECMObserver_REX", "NeuralECMObserver"]),
        ("hard_guard_fusion_prediction_rows.csv", ["hard_guard"]),
        ("outside_range_fusion_prediction_rows.csv", ["hard_guard", "jitter", "fusion"]),
    ]
    all_rows = []
    temp_rows = []
    focus_rows = []
    usecols = None
    for fname, patterns in sources:
        path = cfg.output_dir / fname
        if not path.exists():
            continue
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue
        cols = list(header.columns)
        keep = [c for c in [
            "model_name", "label_type", "trajectory_id", "end_index", "time_index", "y_pred", "y_true",
            "temperature_C", "drive_cycle", "experiment",
        ] if c in cols]
        if not {"model_name", "trajectory_id", "y_pred"}.issubset(set(keep)):
            continue
        for chunk in pd.read_csv(path, usecols=keep, chunksize=cfg.label_chunk_size):
            chunk = chunk[chunk["model_name"].astype(str).apply(lambda s: any(p in s for p in patterns))]
            if "label_type" in chunk.columns:
                chunk = chunk[chunk["label_type"].astype(str).eq("physical")]
            if chunk.empty:
                continue
            if "end_index" not in chunk.columns and "time_index" in chunk.columns:
                chunk["end_index"] = chunk["time_index"]
            merged = chunk.merge(
                label_lookup,
                on=["trajectory_id", "end_index"],
                how="left",
                suffixes=("", "_label"),
            )
            if "temperature_C" not in merged.columns and "temperature_C_label" in merged.columns:
                merged["temperature_C"] = merged["temperature_C_label"]
            if "drive_cycle" not in merged.columns and "drive_cycle_label" in merged.columns:
                merged["drive_cycle"] = merged["drive_cycle_label"]
            for vname, col in variants.items():
                if col not in merged.columns:
                    continue
                tmp = merged[np.isfinite(merged[col]) & np.isfinite(merged["y_pred"])].copy()
                if tmp.empty:
                    continue
                tmp["label_variant"] = vname
                tmp["error_variant"] = tmp["y_pred"] - tmp[col]
                tmp["is_minus10_focus"] = tmp["temperature_C"].eq(-10)
                group = ["model_name", "label_variant"]
                if "experiment" in tmp.columns:
                    group = ["experiment"] + group
                all_rows.extend(_metric_rows(tmp, group, fname, "prediction_relabel_only_no_retraining"))
                tgroup = group + ["temperature_C"]
                temp_rows.extend(_metric_rows(tmp, tgroup, fname, "prediction_relabel_only_no_retraining"))
                focus = tmp[tmp["temperature_C"].eq(-10)]
                if len(focus):
                    focus_rows.extend(_metric_rows(focus, tgroup, fname, "prediction_relabel_only_no_retraining"))
    res = pd.DataFrame(all_rows)
    by_temp = pd.DataFrame(temp_rows)
    focus = pd.DataFrame(focus_rows)
    res.to_csv(cfg.output_dir / "label_sensitivity_results.csv", index=False)
    by_temp.to_csv(cfg.output_dir / "label_sensitivity_by_temperature.csv", index=False)
    focus.to_csv(cfg.output_dir / "label_sensitivity_focus_minus10.csv", index=False)
    return res, by_temp, focus


def make_plots(
    cfg: CapacityAuditConfig,
    audit: pd.DataFrame,
    survey: pd.DataFrame,
    variants: pd.DataFrame,
    labels: pd.DataFrame,
    label_sens_temp: pd.DataFrame,
):
    plot_dir = cfg.output_dir / "capacity_audit_plots"
    diff_dir = cfg.output_dir / "label_difference_plots"
    minus_dir = cfg.output_dir / "minus10_label_sensitivity_plots"
    plot_dir.mkdir(exist_ok=True)
    diff_dir.mkdir(exist_ok=True)
    minus_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.plot(survey["temperature_C"], survey["Q_low_current_discharge_Ah"], marker="o", label="low-current discharge Q")
    plt.plot(survey["temperature_C"], survey["Q_low_current_charge_Ah"], marker="o", label="low-current charge Q")
    plt.plot(survey["temperature_C"], survey["Q_final_recommended_Ah"], marker="s", label="recommended Q_ref")
    plt.xlabel("temperature (C)")
    plt.ylabel("Ah")
    plt.title("Q_eff candidates by temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "qeff_candidates_by_temperature.png", dpi=180)
    plt.savefig(cfg.output_dir / "qeff_temperature_plot.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    for name, g in variants.groupby("fit_variant"):
        plt.plot(g["temperature_C"], g["Q_eff_Ah"], marker="o", label=name)
    plt.xlabel("temperature (C)")
    plt.ylabel("Q_eff Ah")
    plt.title("Q_eff fit variants")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "qeff_fit_variant_plot.png", dpi=180)
    plt.savefig(cfg.output_dir / "qeff_fit_variant_plot.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.scatter(audit["temperature_C"], audit["CE_raw"], label="integrated CE", alpha=0.7)
    if "CE_est_existing" in audit.columns:
        plt.scatter(audit["temperature_C"], audit["CE_est_existing"], label="source CE_est", alpha=0.7)
    plt.axhline(1.0, color="k", linewidth=0.8)
    plt.xlabel("temperature (C)")
    plt.ylabel("CE")
    plt.title("CE by temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "ce_by_temperature.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(audit["charge_Ah_raw"], audit["discharge_Ah_raw"], c=audit["temperature_C"], cmap="coolwarm")
    lim = np.nanmax([audit["charge_Ah_raw"].max(), audit["discharge_Ah_raw"].max()])
    plt.plot([0, lim], [0, lim], color="k", linewidth=0.8)
    plt.xlabel("charge Ah raw")
    plt.ylabel("discharge Ah raw")
    plt.title("charge_Ah vs discharge_Ah")
    plt.colorbar(label="temperature C")
    plt.tight_layout()
    plt.savefig(plot_dir / "charge_vs_discharge_by_temperature.png", dpi=180)
    plt.close()

    n10_ocv = cfg.raw_dir / "LFP_OCV_N10.csv"
    if n10_ocv.exists():
        df = pd.read_csv(n10_ocv)
        t, _ = time_vector(df, cfg)
        fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        axes[0].plot(t, num(df, "Voltage(V)"), linewidth=0.8)
        axes[0].axhline(cfg.voltage_cutoff, color="r", linestyle="--", linewidth=0.8)
        axes[0].set_ylabel("V")
        axes[1].plot(t, num(df, "Current(A)"), linewidth=0.8)
        axes[1].set_ylabel("I raw")
        if "abs_charge_ah" in df.columns:
            axes[2].plot(t, num(df, "abs_charge_ah"), label="abs charge Ah", linewidth=0.8)
        if "abs_discharge_ah" in df.columns:
            axes[2].plot(t, num(df, "abs_discharge_ah"), label="abs discharge Ah", linewidth=0.8)
        axes[2].set_ylabel("Ah")
        axes[2].set_xlabel("time (s)")
        axes[2].legend()
        fig.suptitle("-10C low-current OCV V/I/Ah curve")
        fig.tight_layout()
        fig.savefig(plot_dir / "minus10_low_current_ocv_v_i_ah_curve.png", dpi=180)
        plt.close(fig)

    m10 = labels[labels["trajectory_id"].astype(str).str.contains("N10") & labels["drive_cycle"].isin(["DST", "US06", "FUDS"])]
    if len(m10):
        for tid, g in m10.groupby("trajectory_id"):
            plt.figure(figsize=(8, 4))
            plt.plot(g["time_index"], g["SOC_existing"], label="existing SOC", linewidth=0.9)
            plt.plot(g["time_index"], g["SOC_physical_smoothQ"], label="physical smoothQ", linewidth=0.9)
            plt.plot(g["time_index"], g["SOC_usable_cutoff"], label="usable-to-cutoff", linewidth=0.9)
            plt.xlabel("time index")
            plt.ylabel("fraction")
            plt.title(f"-10C label comparison | {tid}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / f"minus10_existing_vs_smoothQ_vs_usable_{tid}.png", dpi=180)
            plt.savefig(diff_dir / f"minus10_label_difference_{tid}.png", dpi=180)
            plt.close()

    comp = pd.read_csv(cfg.output_dir / "label_comparison_summary.csv")
    plt.figure(figsize=(7, 4))
    plt.scatter(comp["temperature_C"], comp["physical_smoothQ_end"], label="physical smoothQ cutoff/end")
    plt.scatter(comp["temperature_C"], comp["usable_cutoff_end"], label="usable cutoff/end")
    plt.xlabel("temperature (C)")
    plt.ylabel("SOC fraction at end")
    plt.title("cutoff/end SOC by temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "cutoff_physical_soc_by_temperature.png", dpi=180)
    plt.close()

    pivot = comp.pivot_table(index="temperature_C", columns="drive_cycle", values="existing_minus_smoothQ_end", aggfunc="mean")
    plt.figure(figsize=(7, 4))
    plt.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
    plt.yticks(range(len(pivot.index)), [f"{x:g}" for x in pivot.index])
    plt.colorbar(label="existing - physical_smoothQ end")
    plt.title("label difference heatmap")
    plt.tight_layout()
    plt.savefig(plot_dir / "label_difference_heatmap.png", dpi=180)
    plt.savefig(diff_dir / "label_difference_heatmap.png", dpi=180)
    plt.close()

    if len(label_sens_temp):
        sub = label_sens_temp[label_sens_temp["temperature_C"].eq(-10)].copy()
        if len(sub):
            top = sub.sort_values("MAE_pct").head(30)
            plt.figure(figsize=(10, 5))
            labels_txt = (top["model_name"].astype(str).str.slice(0, 24) + "\n" + top["label_variant"].astype(str)).to_list()
            plt.bar(range(len(top)), top["MAE_pct"])
            plt.xticks(range(len(top)), labels_txt, rotation=75, ha="right", fontsize=7)
            plt.ylabel("MAE (%SOC)")
            plt.title("-10C model performance sensitivity to label variant")
            plt.tight_layout()
            plt.savefig(plot_dir / "model_performance_sensitivity_minus10.png", dpi=180)
            plt.savefig(minus_dir / "minus10_label_sensitivity.png", dpi=180)
            plt.close()


def write_reports(
    cfg: CapacityAuditConfig,
    audit: pd.DataFrame,
    survey: pd.DataFrame,
    qrec: pd.DataFrame,
    label_comp: pd.DataFrame,
    label_sens: pd.DataFrame,
    label_sens_focus: pd.DataFrame,
):
    fail = audit[audit["CE_ok_effective"].eq(False)]
    fail_temp = (
        fail.groupby("temperature_C")["file_name"].count().reset_index(name="n_ce_fail_files")
        if len(fail) else pd.DataFrame(columns=["temperature_C", "n_ce_fail_files"])
    )
    fail_type = (
        fail.groupby("test_type")["file_name"].count().reset_index(name="n_ce_fail_files")
        if len(fail) else pd.DataFrame(columns=["test_type", "n_ce_fail_files"])
    )
    fail_temp.to_csv(cfg.output_dir / "ce_failure_by_temperature.csv", index=False)
    fail_type.to_csv(cfg.output_dir / "ce_failure_by_test_type.csv", index=False)

    n10 = audit[audit["temperature_C"].eq(-10)].copy()
    n10_fail = n10[n10["CE_ok_effective"].eq(False)]
    n10_rec = qrec[qrec["temperature_C"].eq(-10)]
    issue_lines = [
        "# -10C Capacity Issue Report",
        "",
        "CE_ok=False is not ignored and is not forced into a ground-truth physical Q_ref.",
        "",
        "## -10C files",
        n10[[
            "file_name", "test_type", "CE_ok_existing", "CE_est_existing", "Q_discharge_low_current_Ah",
            "Q_charge_low_current_Ah", "Q_drive_delivered_to_cutoff_Ah", "CE_failure_reason",
            "Q_ref_existing_label_Ah",
        ]].to_markdown(index=False) if len(n10) else "No -10C files found.",
        "",
        "## Recommended -10C Q_ref handling",
        n10_rec.to_markdown(index=False) if len(n10_rec) else "No -10C recommendation row.",
        "",
        "Interpretation: if the CE failure remains ambiguous, -10C physical labels should carry low confidence. "
        "Usable-to-cutoff labels remain trajectory-delivered-capacity labels and are not physical SOC.",
    ]
    (cfg.output_dir / "minus10_capacity_issue_report.md").write_text("\n".join(issue_lines), encoding="utf-8")

    ce_lines = [
        "# CE Failure Detailed Report",
        "",
        "CE failure reasons are diagnostic labels, not automatic corrections.",
        "",
        "## CE-failed files",
        fail[[
            "file_name", "temperature_C", "test_type", "CE_ok_existing", "CE_raw", "CE_bias_corrected",
            "rest_current_bias_estimate", "V_min", "V_max", "cutoff_reached", "CE_failure_reason",
        ]].to_markdown(index=False) if len(fail) else "No CE failures found.",
        "",
        "## By temperature",
        fail_temp.to_markdown(index=False),
        "",
        "## By test type",
        fail_type.to_markdown(index=False),
    ]
    (cfg.output_dir / "ce_failure_detailed_report.md").write_text("\n".join(ce_lines), encoding="utf-8")

    sens_minus = label_sens_focus.sort_values("MAE_pct").head(30) if len(label_sens_focus) else pd.DataFrame()
    sens_lines = [
        "# Label Sensitivity Summary",
        "",
        "This table uses existing prediction artifacts and relabels y_true variants when possible. "
        "It is a prediction relabel diagnostic, not a full retraining result.",
        "",
        "## Overall label sensitivity",
        label_sens.sort_values("MAE_pct").head(40).to_markdown(index=False) if len(label_sens) else "No prediction artifacts were available.",
        "",
        "## -10C focus",
        sens_minus.to_markdown(index=False) if len(sens_minus) else "No -10C sensitivity rows were available.",
    ]
    (cfg.output_dir / "label_sensitivity_summary.md").write_text("\n".join(sens_lines), encoding="utf-8")

    main_lines = [
        "# Capacity Audit Report",
        "",
        "## Guardrails",
        "- CE_ok=False capacity is not blindly used as physical Q_ref.",
        "- Low-temperature delivered capacity may be cutoff-limited usable capacity rather than true physical capacity.",
        "- Physical SOC and usable-to-cutoff labels are generated and reported separately.",
        "- Q_ref recommendations are derived from capacity/CE diagnostics only; test model error is not used.",
        "",
        "## CE_ok=False files",
        fail[["file_name", "temperature_C", "test_type", "CE_failure_reason", "CE_est_existing", "CE_raw", "CE_bias_corrected"]].to_markdown(index=False) if len(fail) else "No CE failures found.",
        "",
        "## Q_eff(T) recommendation",
        qrec.to_markdown(index=False),
        "",
        "## Physical vs usable-to-cutoff label separation",
        label_comp.sort_values(["temperature_C", "drive_cycle"])[[
            "trajectory_id", "temperature_C", "drive_cycle", "existing_SOC_end", "physical_smoothQ_end",
            "usable_cutoff_end", "existing_minus_smoothQ_end", "existing_minus_usable_end", "label_quality",
        ]].to_markdown(index=False),
        "",
        "## -10C label quality",
        "If -10C remains CE-failed after audit, measuredQ is diagnostic-only/low-confidence; physical_smoothQ uses the robust Q_eff recommendation.",
        "",
        "## Model label sensitivity",
        "See `label_sensitivity_results.csv`; rows are marked `prediction_relabel_only_no_retraining` unless a future run retrains models per label variant.",
        "",
        "## Final recommended treatment",
        "Use `labels_physical_smoothQ.csv` for physical SOC diagnostics when CE-failed Q is ambiguous, and use "
        "`labels_usable_cutoff.csv` only for remaining-to-cutoff experiments. Do not tune Q_ref by model test error.",
        "",
        "## Forbidden conclusions",
        "- Do not force -10C cutoff to SOC=0 for physical SOC.",
        "- Do not use CE-failed low-current capacity as ground-truth Q_ref without the audit flags.",
        "- Do not mix usable-to-cutoff label with physical SOC label.",
        "- Do not tune Q_ref using test error.",
    ]
    (cfg.output_dir / "capacity_audit_report.md").write_text("\n".join(main_lines), encoding="utf-8")


def assert_no_standard_leakage(cfg: CapacityAuditConfig):
    def tid(temp, drive):
        key = "N10" if temp == -10 else str(int(temp))
        return f"LFP_{key}_{drive}"

    experiments = {
        "Exp A": ([-10, 0, 25, 50], [10]),
        "Exp B": ([-10, 10, 25, 50], [0]),
        "Exp C": ([-10, 0, 10, 25, 50], [20]),
        "Exp D": ([-10, 0, 10, 20, 25, 50], [20]),
    }
    rows = []
    for name, (train_temps, eval_temps) in experiments.items():
        train = {tid(t, d) for t in train_temps for d in ["DST", "US06"]}
        test = {tid(t, "FUDS") for t in [-10, 0, 10, 20, 25, 30, 40, 50]}
        overlap = train & test
        rows.append({"experiment": name, "train_n": len(train), "test_n": len(test), "overlap_n": len(overlap), "overlap": ";".join(sorted(overlap))})
        assert not overlap, f"Train/test leakage in {name}: {overlap}"
    pd.DataFrame(rows).to_csv(cfg.output_dir / "capacity_audit_train_test_leakage_check.csv", index=False)


def run_capacity_audit(cfg: CapacityAuditConfig | None = None):
    cfg = cfg or CapacityAuditConfig()
    assert_no_standard_leakage(cfg)
    audit = run_effective_capacity_audit(cfg)
    survey, variants, qrec = qeff_survey(audit, cfg)
    labels, measured, smooth, usable = generate_labels(cfg, qrec)
    label_comp = pd.read_csv(cfg.output_dir / "label_comparison_summary.csv")
    label_sens, label_sens_temp, label_sens_focus = run_label_sensitivity(cfg, labels)
    minus10_sens = label_sens_focus.copy()
    minus10_sens.to_csv(cfg.output_dir / "minus10_label_sensitivity.csv", index=False)
    make_plots(cfg, audit, survey, variants, labels, label_sens_temp)
    write_reports(cfg, audit, survey, qrec, label_comp, label_sens, label_sens_focus)
    return {
        "audit": audit,
        "qeff_survey": survey,
        "qeff_variants": variants,
        "qeff_recommendation": qrec,
        "labels": labels,
        "label_sensitivity": label_sens,
        "label_sensitivity_by_temperature": label_sens_temp,
        "label_sensitivity_minus10": label_sens_focus,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Capacity and SOC-label audit for LFP SOC experiments.")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--voltage-cutoff", type=float, default=2.0)
    parser.add_argument("--ce-ok-threshold", type=float, default=0.05)
    parser.add_argument("--rest-current-threshold", type=float, default=0.02)
    parser.add_argument("--label-chunk-size", type=int, default=200_000)
    args = parser.parse_args()
    cfg = CapacityAuditConfig(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        voltage_cutoff=args.voltage_cutoff,
        ce_ok_threshold=args.ce_ok_threshold,
        rest_current_threshold=args.rest_current_threshold,
        label_chunk_size=args.label_chunk_size,
    )
    out = run_capacity_audit(cfg)
    print("capacity audit complete")
    print("effective_capacity_audit rows:", len(out["audit"]))
    print("qeff recommendations:")
    print(out["qeff_recommendation"].to_string(index=False))


if __name__ == "__main__":
    main()
