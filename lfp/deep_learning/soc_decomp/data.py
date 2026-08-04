import warnings
from pathlib import Path
import numpy as np
import pandas as pd

from .config import CFG

# Scalers, metadata parsing, and label generation
class MinMax1D:
    def __init__(self, feature_range=(-1.0, 1.0), eps=1e-12):
        self.a, self.b = feature_range
        self.eps = eps
        self.min_ = None
        self.max_ = None
        self.fit_ids = None

    def fit(self, values, fit_ids=None):
        x = np.asarray(values, dtype=np.float32)
        self.min_ = float(np.nanmin(x))
        self.max_ = float(np.nanmax(x))
        if abs(self.max_ - self.min_) < self.eps:
            self.max_ = self.min_ + 1.0
        self.fit_ids = set(fit_ids or [])
        return self

    def transform(self, values):
        x = np.asarray(values, dtype=np.float32)
        z = (x - self.min_) / (self.max_ - self.min_ + self.eps)
        return (self.b - self.a) * z + self.a


class FeatureStandardizer:
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.mean_ = None
        self.std_ = None
        self.columns = None
        self.fit_ids = None

    def fit(self, frames, columns, fit_ids):
        self.columns = list(columns)
        x = np.concatenate([f[self.columns].to_numpy(np.float32) for f in frames], axis=0)
        self.mean_ = np.nanmean(x, axis=0).astype(np.float32)
        self.std_ = np.nanstd(x, axis=0).astype(np.float32)
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_).astype(np.float32)
        self.fit_ids = set(fit_ids)
        return self

    def transform_frame(self, frame):
        out = frame.copy()
        out[self.columns] = (out[self.columns].to_numpy(np.float32) - self.mean_) / self.std_
        return out


def normalize_soc_array(values):
    s = np.asarray(values, dtype=np.float32)
    if np.nanmax(s) > 1.5:
        s = s / 100.0
    return s


def parse_temp_key(temp_key: str) -> float:
    s = str(temp_key).strip().upper()
    return -float(s[1:]) if s.startswith("N") else float(s)


def parse_profile_metadata(path: Path) -> dict:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0] == "LFP":
        temp_key = parts[1]
        drive_cycle = "_".join(parts[2:])
    else:
        temp_key = "unknown"
        drive_cycle = "unknown"
    try:
        temp_c = parse_temp_key(temp_key)
    except Exception:
        temp_c = float("nan")
    return {
        "trajectory_id": stem,
        "file_name": path.name,
        "temperature_key": temp_key,
        "temperature": temp_c,
        "drive_cycle": drive_cycle,
    }


def estimate_q_ref_Ah(df: pd.DataFrame, cfg: CFG) -> float:
    if cfg.q_ref_override_Ah is not None:
        return float(cfg.q_ref_override_Ah)
    candidates = ["Q_ref_Ah", "Qdis_ocv_Ah", "Qchg_ocv_Ah", "Capacity_Ah", "capacity_Ah"]
    vals = []
    for c in candidates:
        if c in df.columns:
            arr = pd.to_numeric(df[c], errors="coerce").to_numpy(np.float32)
            arr = arr[np.isfinite(arr) & (arr > 0)]
            if arr.size:
                vals.append(float(np.nanmedian(arr)))
    if vals:
        return float(np.nanmedian(vals))
    if "Qdis_cum(Ah)" in df.columns:
        arr = pd.to_numeric(df["Qdis_cum(Ah)"], errors="coerce").to_numpy(np.float32)
        q = float(np.nanmax(arr) - np.nanmin(arr))
        if np.isfinite(q) and q > 1e-6:
            warnings.warn("Q_ref_Ah inferred from this trajectory cumulative discharge. This is weaker than RPT/nominal capacity.")
            return q
    raise ValueError("Could not infer Q_ref_Ah. Set cfg.q_ref_override_Ah or provide Qdis_ocv_Ah/Qchg_ocv_Ah.")


def cumulative_from_current(I_raw, t_rel=None, cfg: CFG | None = None):
    I = np.asarray(I_raw, dtype=np.float32)
    if t_rel is not None and len(t_rel) == len(I):
        t = np.asarray(t_rel, dtype=np.float32)
        dt = np.diff(t, prepend=t[:1])
        dt[0] = np.nanmedian(dt[1:]) if len(dt) > 1 and np.isfinite(dt[1:]).any() else (cfg.dt_sec_default if cfg else 1.0)
        dt = np.where(np.isfinite(dt) & (dt > 0), dt, (cfg.dt_sec_default if cfg else 1.0))
    else:
        dt = np.full_like(I, (cfg.dt_sec_default if cfg else 1.0), dtype=np.float32)
    qdis = np.cumsum(np.maximum(I, 0.0) * dt / 3600.0)
    qchg = np.cumsum(np.maximum(-I, 0.0) * dt / 3600.0)
    return qdis.astype(np.float32), qchg.astype(np.float32)


def build_soc_labels(df: pd.DataFrame, cfg: CFG) -> dict:
    q_ref = estimate_q_ref_Ah(df, cfg)
    if "Qdis_cum(Ah)" in df.columns and "Qchg_cum(Ah)" in df.columns:
        qdis = pd.to_numeric(df["Qdis_cum(Ah)"], errors="coerce").to_numpy(np.float32)
        qchg = pd.to_numeric(df["Qchg_cum(Ah)"], errors="coerce").to_numpy(np.float32)
    else:
        t_rel = df[cfg.time_col].to_numpy(np.float32) if cfg.time_col in df.columns else None
        qdis, qchg = cumulative_from_current(df[cfg.i_col].to_numpy(np.float32), t_rel=t_rel, cfg=cfg)

    ce = 1.0
    if "CE_used" in df.columns:
        ce_arr = pd.to_numeric(df["CE_used"], errors="coerce").to_numpy(np.float32)
        ce_arr = ce_arr[np.isfinite(ce_arr) & (ce_arr > 0)]
        if ce_arr.size:
            ce = float(np.nanmedian(ce_arr))

    net_discharge = (qdis - ce * qchg).astype(np.float32)
    net_discharge = net_discharge - net_discharge[0]

    if "SOC0_est" in df.columns and np.isfinite(pd.to_numeric(df["SOC0_est"], errors="coerce").iloc[0]):
        soc_start = float(pd.to_numeric(df["SOC0_est"], errors="coerce").iloc[0])
        if soc_start > 1.5:
            soc_start /= 100.0
    elif cfg.physical_soc_col in df.columns:
        soc_start = float(normalize_soc_array(df[cfg.physical_soc_col].to_numpy(np.float32))[0])
    else:
        soc_start = float(cfg.soc_start_default)

    physical_formula = soc_start - net_discharge / max(q_ref, 1e-6)
    physical_source = "cc_formula"
    physical = physical_formula.copy()
    if cfg.use_existing_soc_cc_if_available and cfg.physical_soc_col in df.columns:
        s = normalize_soc_array(pd.to_numeric(df[cfg.physical_soc_col], errors="coerce").to_numpy(np.float32))
        if np.isfinite(s).mean() > 0.95:
            physical = s
            physical_source = cfg.physical_soc_col

    usable_source = "usable_to_cutoff_formula"
    q_dis = net_discharge - net_discharge[0]
    q_cutoff = float(q_dis[-1])
    if abs(q_cutoff) < 1e-9:
        warnings.warn("Usable-to-cutoff denominator is near zero; setting remaining-to-cutoff fraction to NaN.")
        usable = np.full_like(q_dis, np.nan, dtype=np.float32)
    else:
        if q_cutoff < 0:
            warnings.warn("Usable-to-cutoff denominator is negative; check current sign convention.")
        usable = 1.0 - q_dis / q_cutoff

    for c in [cfg.usable_soc_col, "SOC_use(%)", "SOC_usable", "SOC_usable(%)"]:
        if c in df.columns:
            s = normalize_soc_array(pd.to_numeric(df[c], errors="coerce").to_numpy(np.float32))
            if np.isfinite(s).mean() > 0.95 and np.nanmax(np.abs(s - usable)) > 0.02:
                warnings.warn(
                    f"Ignoring existing {c!r}; usable-to-cutoff is rebuilt from cumulative discharge "
                    "so cutoff is near zero."
                )
            break

    if cfg.clip_training_soc_to_0_1:
        physical_train = np.clip(physical, 0.0, 1.0).astype(np.float32)
        usable_train = np.clip(usable, 0.0, 1.0).astype(np.float32)
    else:
        physical_train = physical.astype(np.float32)
        usable_train = usable.astype(np.float32)

    return {
        "SOC_physical_CC": physical_train,
        "SOC_physical_CC_raw": physical.astype(np.float32),
        "SOC_usable_cutoff": usable_train,
        "SOC_usable_cutoff_raw": usable.astype(np.float32),
        "cumulative_discharge_Ah": q_dis.astype(np.float32),
        "q_cutoff_Ah": float(q_cutoff),
        "Q_ref_Ah": q_ref,
        "physical_label_source": physical_source,
        "usable_label_source": usable_source,
    }

# Data discovery, loading, current sign normalization, and trajectory-level split
def profile_csv_path(temp_key, drive, cfg: CFG) -> Path:
    return cfg.data_dir / f"LFP_{temp_key}_{drive}.csv"


def discover_split_paths(cfg: CFG):
    train_temps = cfg.smoke_train_temps if cfg.smoke_mode else cfg.train_temps
    eval_temps = cfg.smoke_eval_temps if cfg.smoke_mode else cfg.eval_temps
    train_paths = [profile_csv_path(t, d, cfg) for t in train_temps for d in cfg.train_drives]
    valid_paths = []
    test_paths = [profile_csv_path(t, cfg.eval_drive, cfg) for t in eval_temps]
    missing = [p for p in train_paths + valid_paths + test_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required CSV files:\n" + "\n".join(map(str, missing)))
    return train_paths, valid_paths, test_paths


def load_profile_csv(path: Path, cfg: CFG) -> dict:
    df = pd.read_csv(path)
    if cfg.smoke_mode and cfg.smoke_max_rows_per_trajectory and len(df) > cfg.smoke_max_rows_per_trajectory:
        df = df.iloc[: cfg.smoke_max_rows_per_trajectory].copy()

    for c in [cfg.v_col, cfg.i_col, cfg.t_col]:
        if c not in df.columns:
            raise KeyError(f"Column {c!r} not found in {path}. Available columns: {list(df.columns)[:40]}")

    labels = build_soc_labels(df, cfg)
    meta = parse_profile_metadata(path)

    v = pd.to_numeric(df[cfg.v_col], errors="coerce").to_numpy(np.float32)
    i = pd.to_numeric(df[cfg.i_col], errors="coerce").to_numpy(np.float32)
    t = pd.to_numeric(df[cfg.t_col], errors="coerce").to_numpy(np.float32)
    phys = labels["SOC_physical_CC"]
    usable = labels["SOC_usable_cutoff"]
    finite = np.isfinite(v) & np.isfinite(i) & np.isfinite(t) & np.isfinite(phys) & np.isfinite(usable)
    if finite.sum() < len(finite):
        warnings.warn(f"{path.name}: dropping {len(finite) - int(finite.sum())} non-finite rows")

    out = {
        **meta,
        "path": str(path),
        "V": v[finite],
        "I": i[finite],
        "T": t[finite],
        "SOC_physical": phys[finite],
        "SOC_physical_raw": labels["SOC_physical_CC_raw"][finite],
        "SOC_usable_cutoff": usable[finite],
        "SOC_usable_cutoff_raw": labels["SOC_usable_cutoff_raw"][finite],
        "cumulative_discharge_Ah": labels["cumulative_discharge_Ah"][finite],
        "q_cutoff_Ah": float(labels["q_cutoff_Ah"]),
        "Q_ref_Ah": float(labels["Q_ref_Ah"]),
        "physical_label_source": labels["physical_label_source"],
        "usable_label_source": labels["usable_label_source"],
    }
    return out


def infer_cc_flip_one(profile, label_key="SOC_physical", thr=0.5):
    I = profile["I"].astype(np.float32)
    soc = profile[label_key].astype(np.float32)
    if len(I) < 5:
        return 1.0, float("nan")
    ds = np.diff(soc)
    I1 = I[1:]
    m = np.abs(I1) > thr
    if m.sum() < 50:
        m = np.abs(I1) > (thr * 0.4)
    if m.sum() < 5:
        return 1.0, float("nan")
    corr = np.corrcoef(ds[m], I1[m])[0, 1]
    if not np.isfinite(corr):
        return 1.0, float("nan")
    # Desired convention: discharge current is positive and SOC decreases, so corr(dsoc, I) < 0.
    flip = -1.0 if corr > 0 else 1.0
    return flip, float(corr)


def apply_cc_flip_inplace(profiles, label_key="SOC_physical", thr=0.5, tag=""):
    print(f"current sign normalization ({tag}):")
    for p in profiles:
        flip, corr_before = infer_cc_flip_one(p, label_key=label_key, thr=thr)
        p["I"] = p["I"] * flip
        _, corr_after = infer_cc_flip_one(p, label_key=label_key, thr=thr)
        print(f"  {p['file_name']}: flip={flip:+.0f}, corr_before={corr_before:.4f}, corr_after={corr_after:.4f}")


def fit_input_scalers(train_profiles):
    ids = [p["trajectory_id"] for p in train_profiles]
    v_scaler = MinMax1D((-1, 1)).fit(np.concatenate([p["V"] for p in train_profiles]), fit_ids=ids)
    i_scaler = MinMax1D((-1, 1)).fit(np.concatenate([p["I"] for p in train_profiles]), fit_ids=ids)
    t_scaler = MinMax1D((-1, 1)).fit(np.concatenate([p["T"] for p in train_profiles]), fit_ids=ids)
    return v_scaler, i_scaler, t_scaler


def apply_input_scalers(profile, v_scaler, i_scaler, t_scaler):
    out = dict(profile)
    out["V_s"] = np.clip(v_scaler.transform(profile["V"]), -1.0, 1.0).astype(np.float32)
    out["I_s"] = np.clip(i_scaler.transform(profile["I"]), -1.0, 1.0).astype(np.float32)
    out["T_s"] = np.clip(t_scaler.transform(profile["T"]), -1.0, 1.0).astype(np.float32)
    return out


def assert_trajectory_split(train_profiles, valid_profiles, test_profiles):
    train_ids = {p["trajectory_id"] for p in train_profiles}
    valid_ids = {p["trajectory_id"] for p in valid_profiles}
    test_ids = {p["trajectory_id"] for p in test_profiles}
    if not train_ids.isdisjoint(valid_ids):
        warnings.warn("Leakage risk: train and valid share trajectory IDs")
    if not train_ids.isdisjoint(test_ids):
        warnings.warn("Leakage risk: train and test share trajectory IDs")
    if not valid_ids.isdisjoint(test_ids):
        warnings.warn("Leakage risk: valid and test share trajectory IDs")
    assert train_ids.isdisjoint(valid_ids), "Leakage: train and valid share trajectory IDs"
    assert train_ids.isdisjoint(test_ids), "Leakage: train and test share trajectory IDs"
    assert valid_ids.isdisjoint(test_ids), "Leakage: valid and test share trajectory IDs"
    print("trajectory split check passed:", len(train_ids), "train,", len(test_ids), "eval")


def build_trajectory_split_verification_table(train_profiles, valid_profiles, test_profiles):
    rows = []
    for split, profiles in [("train", train_profiles), ("valid", valid_profiles), ("test", test_profiles)]:
        for p in profiles:
            rows.append({
                "split": split,
                "trajectory_id": p["trajectory_id"],
                "file_name": p["file_name"],
                "temperature": p["temperature"],
                "temperature_key": p["temperature_key"],
                "drive_cycle": p["drive_cycle"],
                "n_rows": int(len(p["V"])),
                "physical_label_source": p["physical_label_source"],
                "usable_label_source": p["usable_label_source"],
            })
    df = pd.DataFrame(rows)
    dup = df.groupby("trajectory_id")["split"].nunique()
    shared = dup[dup > 1]
    if len(shared):
        warnings.warn(f"Leakage risk: trajectory IDs appear in multiple splits: {shared.index.tolist()}")
    return df



def load_and_prepare_data(cfg: CFG):
    train_paths, valid_paths, test_paths = discover_split_paths(cfg)
    train_profiles_raw = [load_profile_csv(p, cfg) for p in train_paths]
    valid_profiles_raw = [load_profile_csv(p, cfg) for p in valid_paths]
    test_profiles_raw = [load_profile_csv(p, cfg) for p in test_paths]

    all_raw = train_profiles_raw + valid_profiles_raw + test_profiles_raw
    apply_cc_flip_inplace(all_raw, label_key="SOC_physical", tag="all")

    assert_trajectory_split(train_profiles_raw, valid_profiles_raw, test_profiles_raw)
    v_scaler, i_scaler, t_scaler = fit_input_scalers(train_profiles_raw)
    train_profiles = [apply_input_scalers(p, v_scaler, i_scaler, t_scaler) for p in train_profiles_raw]
    valid_profiles = [apply_input_scalers(p, v_scaler, i_scaler, t_scaler) for p in valid_profiles_raw]
    test_profiles = [apply_input_scalers(p, v_scaler, i_scaler, t_scaler) for p in test_profiles_raw]

    trajectory_split_verification = build_trajectory_split_verification_table(
        train_profiles_raw, valid_profiles_raw, test_profiles_raw
    )
    trajectory_split_verification.to_csv(cfg.output_dir / "trajectory_split_verification.csv", index=False)

    label_sources = pd.DataFrame([
        {
            "file": p["file_name"],
            "physical": p["physical_label_source"],
            "usable": p["usable_label_source"],
            "Q_ref_Ah": p["Q_ref_Ah"],
            "q_cutoff_Ah": p["q_cutoff_Ah"],
            "cutoff_physical_SOC": float(p["SOC_physical"][-1]),
            "cutoff_usable_SOC": float(p["SOC_usable_cutoff"][-1]),
            "physical_minus_usable_at_cutoff": float(p["SOC_physical"][-1] - p["SOC_usable_cutoff"][-1]),
        }
        for p in all_raw
    ])

    return {
        "train_profiles_raw": train_profiles_raw,
        "valid_profiles_raw": valid_profiles_raw,
        "test_profiles_raw": test_profiles_raw,
        "train_profiles": train_profiles,
        "valid_profiles": valid_profiles,
        "test_profiles": test_profiles,
        "all_raw": all_raw,
        "v_scaler": v_scaler,
        "i_scaler": i_scaler,
        "t_scaler": t_scaler,
        "trajectory_split_verification": trajectory_split_verification,
        "label_sources": label_sources,
    }
