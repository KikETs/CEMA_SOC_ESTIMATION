from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Trajectory:
    path: Path
    profile: str
    temperature_C: float
    time_s: np.ndarray
    dt_s: np.ndarray
    current_A: np.ndarray
    voltage_V: np.ndarray
    temperature_series_C: np.ndarray
    soc_ref: np.ndarray
    q_ref_Ah: float
    evaluation_mask: np.ndarray


def _dt(time_s: np.ndarray) -> np.ndarray:
    delta = np.diff(time_s, prepend=np.nan)
    valid = delta[np.isfinite(delta) & (delta > 0)]
    fallback = float(np.median(valid)) if len(valid) else 1.0
    delta = np.where(np.isfinite(delta) & (delta > 0), delta, fallback)
    return delta.astype(float)


def discover_prepared(root: str | Path) -> list[Path]:
    paths = sorted(Path(root).rglob("LFP_*C_*.csv"))
    if len(paths) != 24:
        raise RuntimeError(f"Expected 24 prepared trajectories, found {len(paths)}")
    return paths


def parse_identity(path: str | Path) -> tuple[float, str]:
    match = re.fullmatch(r"LFP_(-?\d+(?:\.\d+)?)C_(DST|FUDS|US06)", Path(path).stem, re.I)
    if not match:
        raise ValueError(f"Cannot parse LFP identity: {path}")
    return float(match.group(1)), match.group(2).upper()


def load_evaluation_indices(proposed_root: str | Path) -> dict[tuple[float, str], np.ndarray]:
    root = Path(proposed_root)
    output: dict[tuple[float, str], np.ndarray] = {}
    for profile in ("DST", "FUDS", "US06"):
        candidates = sorted(root.glob(f"*gru_residual_g4_holdout{profile.lower()}*seed0*prediction_rows.csv.gz"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one seed-0 proposed mask for {profile}, found {len(candidates)}")
        frame = pd.read_csv(candidates[0], usecols=["temperature", "drive_cycle", "end_index"])
        for temp, group in frame.groupby("temperature"):
            output[(float(temp), profile)] = np.sort(group["end_index"].astype(int).unique())
    if len(output) != 24:
        raise RuntimeError(f"Expected 24 evaluation masks, found {len(output)}")
    return output


def load_trajectory(
    path: str | Path,
    evaluation_indices: dict[tuple[float, str], np.ndarray],
    evaluation_start_index: int = 49,
) -> Trajectory:
    path = Path(path)
    temp, profile = parse_identity(path)
    frame = pd.read_csv(path)
    required = {
        "Test_Time(s)", "Current(A)", "Voltage(V)", "Temperature(C)",
        "SOC_CC", "Q_ref_lc_ocv_discharge_Ah",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{path} missing {sorted(missing)}")
    time_s = pd.to_numeric(frame["Test_Time(s)"], errors="coerce").to_numpy(float)
    time_s = time_s - time_s[0]
    # Raw A123 files use negative current for discharge. Internal convention is discharge-positive.
    current_A = -pd.to_numeric(frame["Current(A)"], errors="coerce").to_numpy(float)
    voltage = pd.to_numeric(frame["Voltage(V)"], errors="coerce").to_numpy(float)
    temperature = pd.to_numeric(frame["Temperature(C)"], errors="coerce").to_numpy(float)
    soc = pd.to_numeric(frame["SOC_CC"], errors="coerce").to_numpy(float)
    if np.nanmax(soc) > 1.5:
        soc = soc / 100.0
    q_ref = float(pd.to_numeric(frame["Q_ref_lc_ocv_discharge_Ah"], errors="coerce").median())
    finite = np.isfinite(time_s) & np.isfinite(current_A) & np.isfinite(voltage) & np.isfinite(soc)
    if not finite.all():
        raise RuntimeError(f"Non-finite core values in {path}; row removal would break the proposed mask")
    mask = np.zeros(len(frame), dtype=bool)
    indices = evaluation_indices.get((temp, profile), np.arange(int(evaluation_start_index), len(frame)))
    if len(indices) == 0 or indices.min() < 0 or indices.max() >= len(frame):
        raise RuntimeError(f"Invalid evaluation indices for {temp}/{profile}")
    mask[indices] = True
    return Trajectory(
        path=path, profile=profile, temperature_C=temp, time_s=time_s, dt_s=_dt(time_s),
        current_A=current_A, voltage_V=voltage, temperature_series_C=temperature,
        soc_ref=np.clip(soc, 0.0, 1.0), q_ref_Ah=q_ref, evaluation_mask=mask,
    )


def load_all(config: dict) -> list[Trajectory]:
    proposed_root = Path(config["inputs"]["proposed_results_root"])
    prediction_files = list(proposed_root.glob("*prediction_rows.csv.gz")) if proposed_root.is_dir() else []
    masks = load_evaluation_indices(proposed_root) if prediction_files else {}
    start = int(config["protocol"].get("evaluation_start_index", config["protocol"]["window_len"] - 1))
    return [load_trajectory(p, masks, start) for p in discover_prepared(config["inputs"]["prepared_root"])]


def load_proposed_predictions(proposed_root: str | Path) -> pd.DataFrame:
    rows = []
    root = Path(proposed_root)
    for profile in ("DST", "FUDS", "US06"):
        paths = sorted(root.glob(f"lfp_3lopo_ocvdischarge_*holdout{profile.lower()}*seed[0-2]*prediction_rows.csv.gz"))
        if len(paths) != 3:
            raise RuntimeError(f"Expected three proposed seeds for {profile}, found {len(paths)}")
        for path in paths:
            seed = int(re.search(r"_seed(\d)_", path.name).group(1))
            frame = pd.read_csv(path)
            frame["seed"] = seed
            frame["model"] = "existing_G4eqdyn_GRU_residual"
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def load_confirmatory_predictions(config: dict) -> pd.DataFrame:
    """Load the locked 17-channel confirmatory G4 seeds 0..4.

    Seeds 0..2 live in the Tier-1 directory and seeds 3..4 in the locked
    winner-promotion directory. The schema and row keys are checked here so a
    similarly named pre-confirmatory G4eqdyn artifact cannot be substituted.
    """
    import json

    tier1 = Path(config["tier1_root"])
    promotion = Path(config["winner_promotion_root"])
    expected_feature = str(config["feature_set"])
    expected_channels = int(config["channels"])
    expected_seeds = [int(seed) for seed in config["seeds"]]
    rows = []
    reference_keys: dict[str, np.ndarray] = {}
    for profile in ("DST", "FUDS", "US06"):
        paths = sorted(tier1.glob(f"*gru_residual_g4_holdout{profile.lower()}*seed[0-2]*prediction_rows.csv.gz"))
        paths += sorted(promotion.glob(f"*gru_residual_g4_holdout{profile.lower()}*seed[3-4]*prediction_rows.csv.gz"))
        if len(paths) != len(expected_seeds):
            raise RuntimeError(f"Expected five confirmatory G4 seeds for {profile}, found {len(paths)}")
        observed_seeds = []
        for path in paths:
            match = re.search(r"_seed(\d+)_", path.name)
            if not match:
                raise RuntimeError(f"Cannot parse confirmatory seed: {path}")
            seed = int(match.group(1)); observed_seeds.append(seed)
            stem = str(path).split(f"_seed{seed}_", 1)[0]
            metadata_path = Path(stem + "_metadata.json")
            schema_path = Path(stem + "_input_schema.csv")
            if not metadata_path.is_file() or not schema_path.is_file():
                raise RuntimeError(f"Missing confirmatory metadata/schema beside {path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            schema = pd.read_csv(schema_path)
            if metadata.get("feature_set") != expected_feature or len(schema) != expected_channels:
                raise RuntimeError(
                    f"Confirmatory schema mismatch for {path}: "
                    f"feature={metadata.get('feature_set')} channels={len(schema)}"
                )
            frame = pd.read_csv(path)
            keys = frame[["temperature", "drive_cycle", "end_index"]].sort_values(
                ["temperature", "drive_cycle", "end_index"]
            ).to_numpy()
            if profile not in reference_keys:
                reference_keys[profile] = keys
            elif not np.array_equal(reference_keys[profile], keys):
                raise RuntimeError(f"Confirmatory prediction-row mismatch for {profile} seed {seed}")
            frame["seed"] = seed
            frame["model"] = "confirmatory_17ch_G4_GRU_residual"
            frame["source_file"] = str(path)
            rows.append(frame)
        if sorted(observed_seeds) != expected_seeds:
            raise RuntimeError(f"Confirmatory seed mismatch for {profile}: {sorted(observed_seeds)}")
    return pd.concat(rows, ignore_index=True)
