import warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from .config import CFG
from .runtime import device
from .corrector import tensor_profile

# Causal feature extraction
STAGE2_FEATURE_COLUMNS = [
    "V_raw", "V_corr_raw",
    "V_pol_raw", "V_pol_fast_raw", "V_pol_mid_raw", "V_pol_slow_raw",
    "V_hys_raw", "V_ohm_raw", "R0",
    "I_raw", "T", "dI", "absI", "gate", "g_corr", "temp_gate",
]


@torch.no_grad()
def extract_decomposed_features_for_profile(corrector, profile, cfg: CFG, v_scaler, save=True):
    corrector.eval()
    for p in corrector.parameters():
        p.requires_grad_(False)

    vmin = torch.tensor(v_scaler.min_, device=device, dtype=torch.float32)
    vmax = torch.tensor(v_scaler.max_, device=device, dtype=torch.float32)
    n = len(profile["V"])
    if n == 0:
        return pd.DataFrame()
    batch = tensor_profile(profile)
    _, _, aux = corrector(
        batch["V_s"], batch["I_s"], batch["T_s"],
        batch["V_raw"], batch["I_raw"], vmin, vmax, state=None
    )
    dI_raw = np.diff(profile["I"], prepend=profile["I"][0]).astype(np.float32)
    df = pd.DataFrame({
        "trajectory_id": profile["trajectory_id"],
        "file_name": profile["file_name"],
        "end_index": np.arange(n, dtype=np.int64),
        "temperature": float(profile["temperature"]),
        "temperature_key": profile["temperature_key"],
        "drive_cycle": profile["drive_cycle"],
        "V_raw": profile["V"].astype(np.float32),
        "I_raw": profile["I"].astype(np.float32),
        "T": profile["T"].astype(np.float32),
        "dI": dI_raw,
        "absI": np.abs(profile["I"]).astype(np.float32),
        "SOC_physical": profile["SOC_physical"].astype(np.float32),
        "SOC_usable_cutoff": profile["SOC_usable_cutoff"].astype(np.float32),
        "cumulative_discharge_Ah": profile["cumulative_discharge_Ah"].astype(np.float32),
        "q_cutoff_Ah": float(profile["q_cutoff_Ah"]),
        "Q_ref_Ah": float(profile["Q_ref_Ah"]),
        "V_corr_raw": aux["V_corr_raw"].squeeze().detach().cpu().numpy().reshape(-1).astype(np.float32),
        "V_pol_fast_raw": aux["v_pol_fast_raw"].squeeze().detach().cpu().numpy().reshape(-1).astype(np.float32),
        "V_pol_mid_raw": aux["v_pol_mid_raw"].squeeze().detach().cpu().numpy().reshape(-1).astype(np.float32),
        "V_pol_slow_raw": aux["v_pol_slow_raw"].squeeze().detach().cpu().numpy().reshape(-1).astype(np.float32),
        "V_pol_raw": aux["v_pol_raw"].squeeze().detach().cpu().numpy().reshape(-1).astype(np.float32),
        "V_hys_raw": aux["v_hys_raw"].squeeze().detach().cpu().numpy().reshape(-1).astype(np.float32),
        "V_ohm_raw": aux["v_ohm_raw"].squeeze().detach().cpu().numpy().reshape(-1).astype(np.float32),
        "R0": aux["R0"].squeeze().detach().cpu().numpy().reshape(-1).astype(np.float32),
        "gate": aux["gate"].squeeze().detach().cpu().numpy().reshape(-1).astype(np.float32),
        "g_corr": aux["g_corr"].squeeze().detach().cpu().numpy().reshape(-1).astype(np.float32),
        "temp_gate": aux["corr_temp_gate"].squeeze().detach().cpu().numpy().reshape(-1).astype(np.float32),
    })
    if save and cfg.save_decomposed_features:
        path = cfg.decomposed_dir / f"{profile['trajectory_id']}_features.csv"
        df.to_csv(path, index=False)
    return df


def assert_no_future_feature_processing(cfg: CFG):
    assert cfg.use_future_smoothing is False, "Leakage risk: future smoothing is enabled"
    assert cfg.feature_normalization_scope == "train_only", "Leakage risk: feature normalization must be train_only"
    print("future smoothing / normalization scope check passed")


def causal_feature_extraction_check(corrector, profile, cfg: CFG, v_scaler, n_check=None, atol=1e-5):
    n_check = int(n_check or cfg.causal_check_len)
    n_check = min(n_check, len(profile["V"]))
    full = extract_decomposed_features_for_profile(corrector, profile, cfg, v_scaler, save=False).iloc[:n_check].reset_index(drop=True)
    prefix = dict(profile)
    for key, value in profile.items():
        if isinstance(value, np.ndarray) and len(value) == len(profile["V"]):
            prefix[key] = value[:n_check].copy()
    pref = extract_decomposed_features_for_profile(corrector, prefix, cfg, v_scaler, save=False).reset_index(drop=True)
    check_cols = ["V_corr_raw", "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0", "gate", "g_corr", "temp_gate"]
    max_abs = float(np.max(np.abs(full[check_cols].to_numpy() - pref[check_cols].to_numpy())))
    assert max_abs <= atol, f"Causality check failed: prefix/full mismatch {max_abs}"
    print(f"causal feature extraction check passed: max_abs={max_abs:.3e}")



def extract_all_feature_frames(corrector, train_profiles, valid_profiles, test_profiles, cfg: CFG, v_scaler):
    assert_no_future_feature_processing(cfg)
    if cfg.run_causal_feature_check and train_profiles:
        causal_feature_extraction_check(corrector, train_profiles[0], cfg, v_scaler)

    feature_frames = {
        "train": [extract_decomposed_features_for_profile(corrector, p, cfg, v_scaler, save=True) for p in train_profiles],
        "valid": [extract_decomposed_features_for_profile(corrector, p, cfg, v_scaler, save=True) for p in valid_profiles],
        "test": [extract_decomposed_features_for_profile(corrector, p, cfg, v_scaler, save=True) for p in test_profiles],
    }
    return feature_frames


def load_cached_feature_frames(cfg: CFG, data=None, decomposed_dir=None):
    decomposed_dir = Path(decomposed_dir or cfg.decomposed_dir)
    feature_frames = {"train": [], "valid": [], "test": []}

    expected = None
    if data is not None:
        expected = {
            "train": {p["trajectory_id"] for p in data.get("train_profiles", [])},
            "valid": {p["trajectory_id"] for p in data.get("valid_profiles", [])},
            "test": {p["trajectory_id"] for p in data.get("test_profiles", [])},
        }

    train_drives = {str(d).upper() for d in cfg.train_drives}
    eval_drive = str(cfg.eval_drive).upper()
    train_temps = set(cfg.smoke_train_temps if cfg.smoke_mode else cfg.train_temps)
    eval_temps = set(cfg.smoke_eval_temps if cfg.smoke_mode else cfg.eval_temps)

    for path in sorted(decomposed_dir.glob("*_features.csv")):
        frame = pd.read_csv(path)
        if frame.empty or "trajectory_id" not in frame.columns:
            continue
        tid = str(frame["trajectory_id"].iloc[0])
        if expected is not None:
            split = next((k for k, ids in expected.items() if tid in ids), None)
        else:
            drive = str(frame["drive_cycle"].iloc[0]).upper() if "drive_cycle" in frame.columns else ""
            temp_key = str(frame["temperature_key"].iloc[0]) if "temperature_key" in frame.columns else None
            if drive in train_drives and temp_key in train_temps:
                split = "train"
            elif drive == eval_drive and temp_key in eval_temps:
                split = "test"
            else:
                split = None
        if split is not None:
            feature_frames[split].append(frame)

    missing = [split for split in ("train", "test") if not feature_frames[split]]
    if missing:
        raise ValueError(f"Cached decomposed features missing required splits {missing} in {decomposed_dir}")

    train_ids = {f["trajectory_id"].iloc[0] for f in feature_frames["train"]}
    test_ids = {f["trajectory_id"].iloc[0] for f in feature_frames["test"]}
    overlap = train_ids & test_ids
    if overlap:
        raise AssertionError(f"Cached feature leakage: train/test share trajectories {sorted(overlap)[:5]}")
    return feature_frames
