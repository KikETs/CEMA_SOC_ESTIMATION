from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import argparse
import gzip
import json
import math
import random
import shutil
import time
import traceback
from typing import Iterable

import numpy as np
import pandas as pd
import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    class tqdm:  # minimal context-manager fallback
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable if iterable is not None else range(0)

        def __iter__(self):
            return iter(self.iterable)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def set_postfix(self, **kwargs):
            return None

from .config import make_cfg
from .runtime import configure_torch_runtime, device
from .smoothq_retrain import (
    EXPERIMENTS,
    clone_cfg,
    configure_strict_training,
    experiment_cfg,
    load_relabelled_frames,
    load_smoothq_lookup,
    train_a3_smoothq,
)
from .extrapolation_robustness import train_rex_model
from .neural_ecm_observer import ECMSpec, train_ecm_model, predict_full_trajectories
from .thermal_state_model import (
    ObserverSpec,
    ThermalECMObserver,
    attach_and_summarize_observer,
    predict_observer_trajectories,
    train_observer_model,
)
from .parameter_surface_model import ParamSurfaceECMObserver
from .training import attach_prediction_features, build_prediction_feature_lookup
from .variance_control import R5_GATED_FEATURES


DEFAULT_SEEDS = tuple(range(10))
DEFAULT_FOLDS = ("Exp A", "Exp B", "Exp C", "Exp D", "Omit N10", "Omit 50")
DEFAULT_MODELS = (
    "A3_V_raw_I_T",
    "R5_GATED_AUG_REX",
    "NeuralECM_REX",
    "NeuralECM_Tamb_Tcore",
    "NeuralECM_THERMAL_REX",
    "ParamSurfaceECM_THERMAL_REX",
    "ThermoGuardSOC_rule_soft_high",
)
THERMOGUARD_DEPS = (
    "NeuralECM_Tamb_Tcore",
    "NeuralECM_THERMAL_REX",
    "ParamSurfaceECM_THERMAL_REX",
    "R5_GATED_AUG_REX",
)
THERMOGUARD_NO_CURRENT_INTEGRATION_DEPS = (
    "NeuralECM_Tamb_Tcore_NoCurrentInt",
    "NeuralECM_THERMAL_REX_NoCurrentInt",
    "ParamSurfaceECM_THERMAL_REX_NoCurrentInt",
    "R5_GATED_AUG_REX",
)
THERMOGUARD_DEPS_BY_MODEL = {
    "ThermoGuardSOC_rule_soft_high": THERMOGUARD_DEPS,
    "ThermoGuardSOC_rule_soft_high_NoCurrentInt": THERMOGUARD_NO_CURRENT_INTEGRATION_DEPS,
}
NO_CURRENT_INTEGRATION_MODELS = (
    "NeuralECM_REX_NoCurrentInt",
    "NeuralECM_Tamb_Tcore_NoCurrentInt",
    "NeuralECM_THERMAL_REX_NoCurrentInt",
    "ParamSurfaceECM_THERMAL_REX_NoCurrentInt",
    "ThermoGuardSOC_rule_soft_high_NoCurrentInt",
)
MODEL_ORDER = DEFAULT_MODELS + tuple(m for m in NO_CURRENT_INTEGRATION_MODELS if m not in DEFAULT_MODELS)


@dataclass
class Final10SeedConfig:
    base_dir: Path = Path(".")
    run_dir: Path = Path("final_10seed_runs")
    output_prefix: str = "final_10seed"
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    folds: tuple[str, ...] = DEFAULT_FOLDS
    models: tuple[str, ...] = DEFAULT_MODELS
    label_variant: str = "physical_smoothQ"
    split_config: str = "DST_US06_to_FUDS_file_level"
    quick_run: bool = False
    max_epochs_override: int | None = None
    allow_frozen_thermoguard_fallback: bool = False
    train_thermoguard_dependencies: bool = True
    deterministic: bool = True
    save_raw_predictions: bool = True
    compress_predictions: bool = True
    smoke_max_epochs: int = 2
    notes: str = ""


def _pathify(cfg: Final10SeedConfig) -> Final10SeedConfig:
    cfg.base_dir = Path(cfg.base_dir)
    cfg.run_dir = cfg.base_dir / cfg.run_dir if not Path(cfg.run_dir).is_absolute() else Path(cfg.run_dir)
    return cfg


def quick_config(base_dir: Path | str = ".") -> Final10SeedConfig:
    return Final10SeedConfig(
        base_dir=Path(base_dir),
        seeds=(0,),
        folds=("Exp C", "Omit 50"),
        models=("A3_V_raw_I_T", "NeuralECM_Tamb_Tcore", "ThermoGuardSOC_rule_soft_high"),
        quick_run=True,
        max_epochs_override=2,
        allow_frozen_thermoguard_fallback=True,
        train_thermoguard_dependencies=False,
        notes="Smoke test only; not for final comparison.",
    )


def full_config(base_dir: Path | str = ".") -> Final10SeedConfig:
    return Final10SeedConfig(base_dir=Path(base_dir))


def set_global_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def make_run_key(model_name: str, seed: int, fold_name: str, label_variant="physical_smoothQ", split_config="DST_US06_to_FUDS_file_level") -> str:
    safe = (
        f"{model_name}__seed{seed}__{fold_name}__{label_variant}__{split_config}"
        .replace(" ", "_")
        .replace("/", "_")
    )
    return safe


def run_paths(cfg: Final10SeedConfig, model_name: str, seed: int, fold_name: str) -> dict[str, Path]:
    key = make_run_key(model_name, seed, fold_name, cfg.label_variant, cfg.split_config)
    d = cfg.run_dir / key
    pred_name = "predictions.csv.gz" if cfg.compress_predictions else "predictions.csv"
    return {
        "key": d,
        "dir": d,
        "status": d / "status.json",
        "predictions": d / pred_name,
        "history": d / "history.csv",
        "checkpoint": d / "checkpoint.pt",
        "metadata": d / "metadata.json",
        "traceback": d / "traceback.txt",
    }


def status_is_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return data.get("status") == "completed"


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def read_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_predictions(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip" if str(path).endswith(".gz") else None)


def configure_training_cfg(base_cfg, fold_name: str, final_cfg: Final10SeedConfig):
    ecfg = experiment_cfg(base_cfg, fold_name)
    configure_strict_training(ecfg)
    if final_cfg.max_epochs_override is not None:
        ecfg.lstm_epochs = int(final_cfg.max_epochs_override)
        ecfg.ecm_epochs = int(final_cfg.max_epochs_override)
        ecfg.lstm_early_stop = False
        ecfg.ecm_early_stop = False
    if final_cfg.quick_run:
        ecfg.lstm_epochs = int(final_cfg.max_epochs_override or final_cfg.smoke_max_epochs)
        ecfg.ecm_epochs = int(final_cfg.max_epochs_override or final_cfg.smoke_max_epochs)
        ecfg.lstm_early_stop = False
        ecfg.ecm_early_stop = False
        ecfg.batch_size = min(int(ecfg.batch_size), 4096)
        ecfg.lstm_print_every = 1
    return ecfg


def sanity_checks(cfg: Final10SeedConfig) -> dict:
    base = cfg.base_dir
    required = [
        "labels_physical_smoothQ.csv",
        "labels_usable_cutoff.csv",
        "final_label_policy.md",
    ]
    missing = [p for p in required if not (base / p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required label/policy files: {missing}")
    available_feature_dirs = {
        fold: (base / EXPERIMENTS[fold]["feature_dir"]).exists()
        for fold in cfg.folds
    }
    if not all(available_feature_dirs.values()):
        bad = [k for k, v in available_feature_dirs.items() if not v]
        raise FileNotFoundError(f"Missing decomposed feature directories for folds: {bad}")
    policy = (base / "final_label_policy.md").read_text(encoding="utf-8", errors="ignore")[:4000]
    return {
        "label_policy_exists": True,
        "usable_cutoff_kept_separate": True,
        "feature_dirs": available_feature_dirs,
        "policy_excerpt": policy,
        "supervised_train_cycles": ["DST", "US06"],
        "supervised_test_cycle": "FUDS",
        "tta_in_main_result": False,
        "q_ref_selected_by_test_error": False,
    }


def load_frames_for_fold(base_cfg, fold_name: str, lookup: pd.DataFrame, final_cfg: Final10SeedConfig):
    ecfg = configure_training_cfg(base_cfg, fold_name, final_cfg)
    frames = load_relabelled_frames(ecfg, fold_name, lookup)
    train_ids = {f["trajectory_id"].iloc[0] for f in frames["train"]}
    test_ids = {f["trajectory_id"].iloc[0] for f in frames["test"]}
    overlap = sorted(train_ids & test_ids)
    if overlap:
        raise AssertionError(f"{fold_name}: train/test leakage by trajectory_id: {overlap}")
    train_cycles = {str(f["drive_cycle"].iloc[0]) for f in frames["train"]}
    test_cycles = {str(f["drive_cycle"].iloc[0]) for f in frames["test"]}
    if not train_cycles.issubset({"DST", "US06"}):
        raise AssertionError(f"{fold_name}: unexpected train cycles {train_cycles}")
    if test_cycles != {"FUDS"}:
        raise AssertionError(f"{fold_name}: expected FUDS-only test, got {test_cycles}")
    return ecfg, frames


def _attach_lstm_predictions(pred: pd.DataFrame, feature_frames: dict, model_name: str, fold_name: str) -> pd.DataFrame:
    lookup = build_prediction_feature_lookup(feature_frames)
    p = pred.assign(split="test", ablation=model_name)
    out = attach_prediction_features(p, lookup, ablation_name=model_name, target_label="physical")
    out["experiment"] = fold_name
    out["label_policy"] = "physical_smoothQ"
    return out


def _attach_observer_predictions(pred: pd.DataFrame, feature_frames: dict, fold_name: str) -> pd.DataFrame:
    omitted = EXPERIMENTS[fold_name]["omitted_temp_C"]
    out, _, _, _ = attach_and_summarize_observer(pred, feature_frames, fold_name, omitted)
    return out


def train_single_model(model_name: str, seed: int, fold_name: str, final_cfg: Final10SeedConfig, base_cfg, lookup: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, torch.nn.Module | None, dict]:
    set_global_seed(seed, final_cfg.deterministic)
    ecfg, frames = load_frames_for_fold(base_cfg, fold_name, lookup, final_cfg)
    metadata = {
        "model_name": model_name,
        "seed": seed,
        "fold_name": fold_name,
        "label_variant": final_cfg.label_variant,
        "split_config": final_cfg.split_config,
        "window_len": int(ecfg.window_len),
        "stride": int(ecfg.stride),
        "max_epochs_lstm": int(ecfg.lstm_epochs),
        "max_epochs_ecm": int(ecfg.ecm_epochs),
        "train_cycles": list(ecfg.train_drives),
        "test_cycle": ecfg.eval_drive,
        "train_temps": list(ecfg.train_temps),
        "eval_temps": list(ecfg.eval_temps),
        "feature_dir": str(ecfg.decomposed_dir),
        "corrector_feature_cache_reused": True,
        "tta_used": False,
        "test_soc_used_for_training_or_gate": False,
        "conda_env": "torch_env",
    }
    model = None
    if model_name == "A3_V_raw_I_T":
        hist, pred = train_a3_smoothq(frames, ecfg, fold_name)
        pred = _attach_lstm_predictions(pred, frames, "A3_V_raw_I_T", fold_name)
        pred["model_name"] = "A3_V_raw_I_T"
        metadata["input_features"] = ["V_raw", "I_raw", "T_amb"]
        metadata["model_family"] = "stateless_lstm"
    elif model_name == "R5_GATED_AUG_REX":
        internal_name = "R5_GATED_AUG_REX_l1p0_smoothQ"
        model, hist, _, pred_raw = train_rex_model(
            frames,
            ecfg,
            internal_name,
            lambda_rex=1.0,
            use_aug=True,
            experiment=fold_name,
        )
        pred = _attach_lstm_predictions(pred_raw, frames, "R5_GATED_AUG_REX", fold_name)
        pred["model_name"] = "R5_GATED_AUG_REX"
        metadata["input_features"] = list(R5_GATED_FEATURES)
        metadata["model_family"] = "decomposed_feature_lstm_rex_aug"
        metadata["lambda_rex"] = 1.0
        metadata["component_aug"] = True
    elif model_name in {"NeuralECM_REX", "NeuralECM_REX_NoCurrentInt"}:
        use_current_integration = model_name == "NeuralECM_REX"
        spec = ECMSpec(
            name=model_name,
            use_rex=True,
            lambda_v=0.20,
            lambda_rex=0.5,
            lambda_worst=0.10,
            correction_limit=0.05,
            use_current_integration=use_current_integration,
        )
        model, hist, _ = train_ecm_model(frames, ecfg, spec, fold_name)
        pred_raw = predict_full_trajectories(model, frames["test"], spec.name)
        pred = _attach_observer_predictions(pred_raw, frames, fold_name)
        pred["model_name"] = model_name
        metadata["model_family"] = "neural_ecm_observer_rex"
        metadata["inputs"] = ["I_raw", "V_raw", "T_amb"]
        metadata["use_current_integration"] = use_current_integration
        metadata["soc_state_update"] = "current_integrated" if use_current_integration else "voltage_residual_corrected_only"
    elif model_name in {"NeuralECM_Tamb_Tcore", "NeuralECM_Tamb_Tcore_NoCurrentInt"}:
        use_current_integration = model_name == "NeuralECM_Tamb_Tcore"
        spec = ObserverSpec(
            name=model_name,
            use_rex=False,
            lambda_v=0.20,
            correction_limit=0.05,
            use_current_integration=use_current_integration,
        )
        factory = lambda q, dt, s: ThermalECMObserver(
            q_ref_ah=q,
            dt_sec=dt,
            correction_limit=s.correction_limit,
            thermal_mode="tamb_tcore",
            use_current_integration=s.use_current_integration,
        )
        model, hist, _ = train_observer_model(frames, ecfg, spec, fold_name, factory)
        pred_raw = predict_observer_trajectories(model, frames["test"], spec.name)
        pred = _attach_observer_predictions(pred_raw, frames, fold_name)
        pred["model_name"] = model_name
        metadata["model_family"] = "thermal_ecm_observer"
        metadata["thermal_mode"] = "tamb_tcore"
        metadata["T_core_est_interpretation"] = "estimated/effective thermal-state proxy, not measured core temperature"
        metadata["use_current_integration"] = use_current_integration
        metadata["soc_state_update"] = "current_integrated" if use_current_integration else "voltage_residual_corrected_only"
    elif model_name in {"NeuralECM_THERMAL_REX", "NeuralECM_THERMAL_REX_NoCurrentInt"}:
        use_current_integration = model_name == "NeuralECM_THERMAL_REX"
        spec = ObserverSpec(
            name=model_name,
            use_rex=True,
            lambda_v=0.20,
            lambda_rex=0.5,
            lambda_worst=0.10,
            correction_limit=0.05,
            use_current_integration=use_current_integration,
        )
        factory = lambda q, dt, s: ThermalECMObserver(
            q_ref_ah=q,
            dt_sec=dt,
            correction_limit=s.correction_limit,
            thermal_mode="tamb_tcore_heatproxy",
            use_current_integration=s.use_current_integration,
        )
        model, hist, _ = train_observer_model(frames, ecfg, spec, fold_name, factory)
        pred_raw = predict_observer_trajectories(model, frames["test"], spec.name)
        pred = _attach_observer_predictions(pred_raw, frames, fold_name)
        pred["model_name"] = model_name
        metadata["model_family"] = "thermal_ecm_observer_rex"
        metadata["thermal_mode"] = "tamb_tcore_heatproxy"
        metadata["use_current_integration"] = use_current_integration
        metadata["soc_state_update"] = "current_integrated" if use_current_integration else "voltage_residual_corrected_only"
    elif model_name in {"ParamSurfaceECM_THERMAL_REX", "ParamSurfaceECM_THERMAL_REX_NoCurrentInt"}:
        use_current_integration = model_name == "ParamSurfaceECM_THERMAL_REX"
        spec = ObserverSpec(
            name=model_name,
            use_rex=True,
            lambda_v=0.20,
            lambda_rex=0.5,
            lambda_worst=0.10,
            correction_limit=0.05,
            use_current_integration=use_current_integration,
        )
        factory = lambda q, dt, s: ParamSurfaceECMObserver(
            q_ref_ah=q,
            dt_sec=dt,
            correction_limit=s.correction_limit,
            residual_limit=0.15,
            use_thermal=True,
            use_current_integration=s.use_current_integration,
        )
        model, hist, _ = train_observer_model(frames, ecfg, spec, fold_name, factory)
        pred_raw = predict_observer_trajectories(model, frames["test"], spec.name)
        pred = _attach_observer_predictions(pred_raw, frames, fold_name)
        pred["model_name"] = model_name
        metadata["model_family"] = "parameter_surface_ecm_thermal_rex"
        metadata["role"] = "high_temperature_expert_not_generalist"
        metadata["use_current_integration"] = use_current_integration
        metadata["soc_state_update"] = "current_integrated" if use_current_integration else "voltage_residual_corrected_only"
    else:
        raise ValueError(f"Unsupported trainable model: {model_name}")
    pred["seed"] = int(seed)
    pred["fold_name"] = fold_name
    pred["run_model_name"] = model_name
    pred["label_variant"] = final_cfg.label_variant
    if "time_index" not in pred.columns and "end_index" in pred.columns:
        pred["time_index"] = pred["end_index"]
    return pred, hist, model, metadata


def _normalize_for_fusion(df: pd.DataFrame, prefix: str, keep_meta=False) -> pd.DataFrame:
    cols = ["experiment", "trajectory_id", "end_index", "y_pred", "voltage_residual"]
    optional = ["T_core_est", "T_core_minus_amb", "heat_proxy", "Q_eff", "R0", "R0_x"]
    cols += [c for c in optional if c in df.columns]
    if keep_meta:
        meta = [
            "temperature_C",
            "drive_cycle",
            "time_index",
            "y_true",
            "trajectory_fraction",
            "is_plateau_20_80",
            "is_cutoff_last10",
            "label_policy",
        ]
        cols += [c for c in meta if c in df.columns]
    out = df[[c for c in cols if c in df.columns]].copy()
    if "voltage_residual" not in out.columns:
        out["voltage_residual"] = np.nan
    rename = {
        "y_pred": f"soc_{prefix}",
        "voltage_residual": f"voltage_residual_{prefix}",
        "Q_eff": f"Q_eff_{prefix}",
        "R0": f"R0_{prefix}",
        "R0_x": f"R0_{prefix}",
    }
    for c in optional:
        if c in out.columns and c not in rename:
            rename[c] = f"{c}_{prefix}"
    return out.rename(columns=rename)


def _recent_jitter(df: pd.DataFrame, pred_col: str) -> pd.Series:
    result = pd.Series(index=df.index, dtype=float)
    for _, idx in df.sort_values("end_index").groupby("trajectory_id").groups.items():
        g = df.loc[idx].sort_values("end_index")
        d = g[pred_col].astype(float).diff().abs()
        local = d.rolling(window=25, min_periods=3).mean().bfill().fillna(d.mean())
        result.loc[g.index] = local.to_numpy(float)
    return result.fillna(result.median() if result.notna().any() else 0.0)


def fuse_thermoguard_from_predictions(
    perf: pd.DataFrame,
    robust: pd.DataFrame,
    high: pd.DataFrame,
    r5: pd.DataFrame,
    seed: int,
    fold_name: str,
    model_name: str = "ThermoGuardSOC_rule_soft_high",
) -> pd.DataFrame:
    keys = ["experiment", "trajectory_id", "end_index"]
    base = _normalize_for_fusion(perf, "perf", keep_meta=True)
    joined = base.merge(_normalize_for_fusion(robust, "robust"), on=keys, how="inner")
    joined = joined.merge(_normalize_for_fusion(high, "highT"), on=keys, how="inner")
    joined = joined.merge(_normalize_for_fusion(r5, "r5"), on=keys, how="inner")
    if joined.empty:
        raise RuntimeError(f"{fold_name} seed={seed}: no common rows for ThermoGuard fusion")
    for name in ["perf", "robust", "highT", "r5"]:
        joined[f"jitter_{name}"] = _recent_jitter(joined, f"soc_{name}")
    temp = joined["temperature_C"].astype(float)
    outside_high, outside_low = [], []
    for exp, t in zip(joined["experiment"], temp):
        train_t = []
        for raw in EXPERIMENTS[str(exp)]["train_temps"]:
            s = str(raw)
            train_t.append(-float(s[1:]) if s.upper().startswith("N") else float(s))
        outside_high.append(float(float(t) > max(train_t)))
        outside_low.append(float(float(t) < min(train_t)))
    joined["outside_high_temp_flag"] = outside_high
    joined["outside_low_temp_flag"] = outside_low
    # Label-free rule_soft_high, matched to the selected final ThermoGuard.
    high_mask = (joined["outside_high_temp_flag"] > 0.5) | (temp >= 45.0)
    low_mask = joined["outside_low_temp_flag"] > 0.5
    joined["w_perf"] = np.where(low_mask, 0.35, 0.65)
    joined["w_robust"] = np.where(low_mask, 0.65, 0.35)
    joined["w_highT"] = 0.0
    joined["w_r5"] = 0.0
    soft_high = 1.0 / (1.0 + np.exp(-(temp - 45.0) / 3.0))
    joined.loc[high_mask, "w_highT"] = np.maximum(soft_high[high_mask], 0.80 * joined.loc[high_mask, "outside_high_temp_flag"].astype(float))
    joined.loc[high_mask, "w_perf"] = 1.0 - joined.loc[high_mask, "w_highT"]
    joined.loc[high_mask, "w_robust"] = 0.0
    joined["y_pred"] = (
        joined["w_perf"] * joined["soc_perf"]
        + joined["w_robust"] * joined["soc_robust"]
        + joined["w_highT"] * joined["soc_highT"]
        + joined["w_r5"] * joined["soc_r5"]
    )
    joined["error"] = joined["y_pred"] - joined["y_true"]
    joined["abs_error"] = np.abs(joined["error"])
    joined["model_name"] = model_name
    joined["seed"] = int(seed)
    joined["fold_name"] = fold_name
    joined["run_model_name"] = model_name
    joined["label_variant"] = "physical_smoothQ"
    joined["T_core_est_interpretation"] = "estimated/effective thermal-state proxy, not measured core temperature"
    return joined


def frozen_fallback_thermoguard(base_dir: Path, seed: int, fold_name: str) -> pd.DataFrame:
    path = base_dir / "thermoguard_prediction_rows.csv"
    if not path.exists():
        raise FileNotFoundError("No ThermoGuard frozen artifact available for fallback.")
    df = pd.read_csv(path)
    df = df[df["model_name"].eq("ThermoGuardSOC_rule_soft_high") & df["experiment"].eq(fold_name)].copy()
    if df.empty:
        raise RuntimeError(f"No frozen ThermoGuard rows for {fold_name}")
    df["seed"] = int(seed)
    df["fold_name"] = fold_name
    df["run_model_name"] = "ThermoGuardSOC_rule_soft_high"
    df["frozen_artifact_reuse"] = True
    df["label_variant"] = "physical_smoothQ"
    return df


def load_completed_prediction(cfg: Final10SeedConfig, model_name: str, seed: int, fold_name: str) -> pd.DataFrame | None:
    paths = run_paths(cfg, model_name, seed, fold_name)
    if status_is_done(paths["status"]) and paths["predictions"].exists():
        return read_predictions(paths["predictions"])
    return None


def run_thermoguard_model(model_name: str, seed: int, fold_name: str, final_cfg: Final10SeedConfig, base_cfg, lookup: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, None, dict]:
    dep_names = THERMOGUARD_DEPS_BY_MODEL[model_name]
    deps = {}
    missing = []
    for dep in dep_names:
        pred = load_completed_prediction(final_cfg, dep, seed, fold_name)
        if pred is None:
            missing.append(dep)
        else:
            deps[dep] = pred
    if missing and final_cfg.train_thermoguard_dependencies:
        for dep in missing:
            run_one(dep, seed, fold_name, final_cfg, base_cfg, lookup)
            pred = load_completed_prediction(final_cfg, dep, seed, fold_name)
            if pred is None:
                raise RuntimeError(f"Dependency {dep} did not complete for ThermoGuard")
            deps[dep] = pred
        missing = []
    if missing:
        if final_cfg.allow_frozen_thermoguard_fallback:
            if model_name != "ThermoGuardSOC_rule_soft_high":
                raise RuntimeError(f"No frozen fallback is defined for {model_name}")
            pred = frozen_fallback_thermoguard(final_cfg.base_dir, seed, fold_name)
            hist = pd.DataFrame([{
                "model_name": model_name,
                "seed": seed,
                "experiment": fold_name,
                "frozen_artifact_reuse": True,
                "note": "Frozen artifact fallback used because seed-specific expert dependencies were not available.",
            }])
            metadata = {
                "model_name": model_name,
                "seed": seed,
                "fold_name": fold_name,
                "frozen_artifact_reuse": True,
                "missing_seed_specific_dependencies": missing,
                "test_soc_used_for_gate": False,
                "gate": "label_free_rule_soft_high",
            }
            return pred, hist, None, metadata
        raise RuntimeError(f"Missing ThermoGuard dependencies for seed={seed} fold={fold_name}: {missing}")
    perf_key, robust_key, high_key, r5_key = dep_names
    pred = fuse_thermoguard_from_predictions(
        deps[perf_key],
        deps[robust_key],
        deps[high_key],
        deps[r5_key],
        seed,
        fold_name,
        model_name=model_name,
    )
    hist = pd.DataFrame([{
        "model_name": model_name,
        "seed": seed,
        "experiment": fold_name,
        "frozen_artifact_reuse": False,
        "gate": "label_free_rule_soft_high",
    }])
    metadata = {
        "model_name": model_name,
        "seed": seed,
        "fold_name": fold_name,
        "frozen_artifact_reuse": False,
        "dependencies": list(dep_names),
        "gate": "label_free_rule_soft_high",
        "test_soc_used_for_gate": False,
        "r5_weight_max": 0.0,
        "T_core_est_interpretation": "estimated/effective thermal-state proxy, not measured core temperature",
        "use_current_integration": model_name == "ThermoGuardSOC_rule_soft_high",
        "soc_state_update": "current_integrated_experts" if model_name == "ThermoGuardSOC_rule_soft_high" else "no_current_integrated_ecm_experts",
    }
    return pred, hist, None, metadata


def run_one(model_name: str, seed: int, fold_name: str, final_cfg: Final10SeedConfig, base_cfg=None, lookup=None, force=False) -> dict:
    paths = run_paths(final_cfg, model_name, seed, fold_name)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    run_key = paths["dir"].name
    if not force and status_is_done(paths["status"]) and paths["predictions"].exists():
        return {
            "run_key": run_key,
            "model_name": model_name,
            "seed": seed,
            "fold_name": fold_name,
            "status": "skipped_completed",
            "predictions_path": str(paths["predictions"]),
        }
    started = time.time()
    status = {
        "run_key": run_key,
        "model_name": model_name,
        "seed": seed,
        "fold_name": fold_name,
        "label_variant": final_cfg.label_variant,
        "split_config": final_cfg.split_config,
        "status": "running",
        "started_at_unix": started,
    }
    write_json(paths["status"], status)
    try:
        base_cfg = base_cfg or make_cfg()
        lookup = lookup if lookup is not None else load_smoothq_lookup(final_cfg.base_dir)
        if model_name in THERMOGUARD_DEPS_BY_MODEL:
            pred, hist, model, metadata = run_thermoguard_model(model_name, seed, fold_name, final_cfg, base_cfg, lookup)
        else:
            pred, hist, model, metadata = train_single_model(model_name, seed, fold_name, final_cfg, base_cfg, lookup)
        write_predictions(pred, paths["predictions"])
        hist.to_csv(paths["history"], index=False)
        if model is not None:
            try:
                torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, paths["checkpoint"])
            except Exception as exc:
                metadata["checkpoint_warning"] = str(exc)
        metadata.update({
            "run_key": run_key,
            "duration_s": time.time() - started,
            "prediction_rows": int(len(pred)),
            "status": "completed",
        })
        write_json(paths["metadata"], metadata)
        status.update({
            "status": "completed",
            "completed_at_unix": time.time(),
            "duration_s": time.time() - started,
            "predictions_path": str(paths["predictions"]),
        })
        write_json(paths["status"], status)
        return status
    except Exception as exc:
        tb = traceback.format_exc()
        paths["traceback"].write_text(tb, encoding="utf-8")
        status.update({
            "status": "failed",
            "failed_at_unix": time.time(),
            "duration_s": time.time() - started,
            "error": repr(exc),
            "traceback_path": str(paths["traceback"]),
        })
        write_json(paths["status"], status)
        return status


def planned_runs(cfg: Final10SeedConfig) -> pd.DataFrame:
    rows = []
    for seed in cfg.seeds:
        for fold in cfg.folds:
            for model in cfg.models:
                key = make_run_key(model, seed, fold, cfg.label_variant, cfg.split_config)
                paths = run_paths(cfg, model, seed, fold)
                status = "pending"
                if paths["status"].exists():
                    try:
                        status = json.loads(paths["status"].read_text(encoding="utf-8")).get("status", "unknown")
                    except Exception:
                        status = "unknown"
                rows.append({
                    "run_key": key,
                    "model_name": model,
                    "seed": seed,
                    "fold_name": fold,
                    "label_variant": cfg.label_variant,
                    "split_config": cfg.split_config,
                    "status": status,
                    "predictions_path": str(paths["predictions"]),
                    "run_dir": str(paths["dir"]),
                })
    return pd.DataFrame(rows)


def save_registry(cfg: Final10SeedConfig):
    reg = planned_runs(cfg)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    reg.to_csv(cfg.base_dir / f"{cfg.output_prefix}_run_registry.csv", index=False)
    failed = reg[reg["status"].eq("failed")]
    failed.to_csv(cfg.base_dir / f"{cfg.output_prefix}_failed_runs.csv", index=False)
    metadata = {
        **asdict(cfg),
        "base_dir": str(cfg.base_dir),
        "run_dir": str(cfg.run_dir),
        "created_at_unix": time.time(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "principles": {
            "main_label": "labels_physical_smoothQ.csv",
            "usable_cutoff_not_mixed": True,
            "fuds_supervised_training": False,
            "tta_in_main_result": False,
            "test_soc_for_gate_threshold": False,
            "minus10_label_flag": "low_confidence_due_to_capacity_consistency_mismatch",
        },
    }
    write_json(cfg.base_dir / f"{cfg.output_prefix}_metadata.json", metadata)
    return reg


def run_all(cfg: Final10SeedConfig, *, force=False) -> pd.DataFrame:
    cfg = _pathify(cfg)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    configure_torch_runtime()
    sanity = sanity_checks(cfg)
    write_json(cfg.base_dir / f"{cfg.output_prefix}_sanity_checks.json", sanity)
    base_cfg = make_cfg()
    lookup = load_smoothq_lookup(cfg.base_dir)
    statuses = []
    save_registry(cfg)
    expanded = []
    for seed in cfg.seeds:
        for fold in cfg.folds:
            ordered_models = list(cfg.models)
            thermoguard_models = [m for m in ordered_models if m in THERMOGUARD_DEPS_BY_MODEL]
            if thermoguard_models and cfg.train_thermoguard_dependencies:
                for tg_model in thermoguard_models:
                    for dep in THERMOGUARD_DEPS_BY_MODEL[tg_model]:
                        if dep not in ordered_models:
                            ordered_models.insert(0, dep)
                ordered_models = [m for m in MODEL_ORDER if m in ordered_models] + [
                    m for m in ordered_models if m not in MODEL_ORDER
                ]
            for model_name in ordered_models:
                expanded.append((int(seed), str(fold), str(model_name)))
    with tqdm(expanded, total=len(expanded), desc="final 10-seed runs", unit="run") as pbar:
        for seed, fold, model_name in pbar:
            pbar.set_postfix(seed=seed, fold=fold, model=model_name[:24])
            st = run_one(model_name, seed, fold, cfg, base_cfg=base_cfg, lookup=lookup, force=force)
            statuses.append(st)
            pd.DataFrame(statuses).to_csv(cfg.base_dir / f"{cfg.output_prefix}_run_status_log.csv", index=False)
            save_registry(cfg)
    return pd.DataFrame(statuses)


def load_all_completed_predictions(cfg: Final10SeedConfig, models: Iterable[str] | None = None) -> pd.DataFrame:
    cfg = _pathify(cfg)
    rows = []
    reg = planned_runs(cfg)
    if models is not None:
        reg = reg[reg["model_name"].isin(list(models))]
    for _, r in reg.iterrows():
        paths = run_paths(cfg, r["model_name"], int(r["seed"]), r["fold_name"])
        if status_is_done(paths["status"]) and paths["predictions"].exists():
            df = read_predictions(paths["predictions"])
            rows.append(df)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _trajectory_jitter(g: pd.DataFrame, pred_col="y_pred", true_col="y_true") -> tuple[float, float, float]:
    pred_vals, true_vals = [], []
    for _, t in g.sort_values("end_index").groupby("trajectory_id"):
        yp = t[pred_col].to_numpy(float)
        yt = t[true_col].to_numpy(float)
        if len(yp) < 3:
            continue
        pred_vals.append(np.mean(np.abs(np.diff(yp))))
        true_vals.append(np.mean(np.abs(np.diff(yt))))
    pred_j = float(np.mean(pred_vals)) if pred_vals else np.nan
    true_j = float(np.mean(true_vals)) if true_vals else np.nan
    ratio = pred_j / (true_j + 1e-12) if np.isfinite(pred_j) and np.isfinite(true_j) else np.nan
    return pred_j, true_j, float(ratio)


def _hf_error(g: pd.DataFrame, pred_col="y_pred", true_col="y_true") -> float:
    vals = []
    for _, t in g.sort_values("end_index").groupby("trajectory_id"):
        e = t[pred_col].to_numpy(float) - t[true_col].to_numpy(float)
        if len(e) >= 3:
            vals.append(np.mean(np.diff(e) ** 2))
    return float(np.mean(vals)) if vals else np.nan


def _region_mae(g: pd.DataFrame, mask: pd.Series) -> float:
    if mask is None or mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(g.loc[mask, "y_pred"].to_numpy(float) - g.loc[mask, "y_true"].to_numpy(float))) * 100.0)


def compute_metrics(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if pred.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df = pred.copy()
    if "fold_name" not in df.columns:
        df["fold_name"] = df["experiment"]
    if "run_model_name" not in df.columns:
        df["run_model_name"] = df["model_name"]
    if "seed" not in df.columns:
        df["seed"] = -1
    if "trajectory_fraction" not in df.columns:
        df["trajectory_fraction"] = np.nan
    metrics = []
    for keys, g in df.groupby(["run_model_name", "seed", "fold_name"]):
        model, seed, fold = keys
        err = g["y_pred"].to_numpy(float) - g["y_true"].to_numpy(float)
        abs_err = np.abs(err)
        pred_j, true_j, jr = _trajectory_jitter(g)
        frac = g["trajectory_fraction"].astype(float)
        if frac.notna().any():
            early = frac <= 0.33
            mid = (frac > 0.33) & (frac <= 0.67)
            late = frac > 0.67
        else:
            rank = g.groupby("trajectory_id")["end_index"].rank(pct=True)
            early = rank <= 0.33
            mid = (rank > 0.33) & (rank <= 0.67)
            late = rank > 0.67
        plateau = g["is_plateau_20_80"].astype(bool) if "is_plateau_20_80" in g.columns else pd.Series(False, index=g.index)
        cutoff = g["is_cutoff_last10"].astype(bool) if "is_cutoff_last10" in g.columns else pd.Series(False, index=g.index)
        row = {
            "model_name": model,
            "seed": int(seed),
            "fold_name": fold,
            "target_temperature_C": float(EXPERIMENTS[fold]["omitted_temp_C"]) if fold in EXPERIMENTS else np.nan,
            "target_type": "outside" if fold in {"Omit N10", "Omit 50"} else ("included_diagnostic" if fold == "Exp D" else "omitted"),
            "MAE_pct": float(np.mean(abs_err) * 100.0),
            "RMSE_pct": float(np.sqrt(np.mean(err ** 2)) * 100.0),
            "Max_error_pct": float(np.max(abs_err) * 100.0),
            "error_std_pct": float(np.std(err) * 100.0),
            "pred_jitter": pred_j,
            "true_jitter": true_j,
            "jitter_ratio": jr,
            "high_frequency_error_energy": _hf_error(g),
            "catastrophic_error_rate_5pct": float(np.mean(abs_err > 0.05)),
            "catastrophic_error_rate_10pct": float(np.mean(abs_err > 0.10)),
            "plateau_20_80_MAE_pct": _region_mae(g, plateau),
            "early_trajectory_MAE_pct": _region_mae(g, early),
            "mid_trajectory_MAE_pct": _region_mae(g, mid),
            "late_trajectory_MAE_pct": _region_mae(g, late),
            "cutoff_last10_MAE_pct": _region_mae(g, cutoff),
            "n_samples": int(len(g)),
        }
        if "voltage_residual" in g.columns and np.isfinite(g["voltage_residual"].astype(float)).any():
            vr = g["voltage_residual"].astype(float).to_numpy()
            row["voltage_residual_MAE_V"] = float(np.nanmean(np.abs(vr)))
            row["voltage_residual_RMSE_V"] = float(np.sqrt(np.nanmean(vr ** 2)))
        metrics.append(row)
    per_run = pd.DataFrame(metrics)
    by_temp_rows = []
    for keys, g in df.groupby(["run_model_name", "seed", "fold_name", "temperature_C"]):
        model, seed, fold, temp = keys
        err = g["y_pred"].to_numpy(float) - g["y_true"].to_numpy(float)
        abs_err = np.abs(err)
        pred_j, true_j, jr = _trajectory_jitter(g)
        by_temp_rows.append({
            "model_name": model,
            "seed": int(seed),
            "fold_name": fold,
            "temperature_C": float(temp),
            "MAE_pct": float(np.mean(abs_err) * 100.0),
            "RMSE_pct": float(np.sqrt(np.mean(err ** 2)) * 100.0),
            "Max_error_pct": float(np.max(abs_err) * 100.0),
            "error_std_pct": float(np.std(err) * 100.0),
            "pred_jitter": pred_j,
            "true_jitter": true_j,
            "jitter_ratio": jr,
            "high_frequency_error_energy": _hf_error(g),
            "catastrophic_error_rate_5pct": float(np.mean(abs_err > 0.05)),
            "catastrophic_error_rate_10pct": float(np.mean(abs_err > 0.10)),
            "n_samples": int(len(g)),
        })
    by_temp = pd.DataFrame(by_temp_rows)
    by_fold = per_run.copy()
    return per_run, by_temp, by_fold


def _summary_stats(group: pd.DataFrame, value_cols: list[str]) -> dict:
    out = {}
    for col in value_cols:
        vals = group[col].dropna().astype(float)
        if len(vals) == 0:
            out[f"{col}_mean"] = np.nan
            out[f"{col}_std"] = np.nan
            out[f"{col}_median"] = np.nan
            out[f"{col}_min"] = np.nan
            out[f"{col}_max"] = np.nan
            out[f"{col}_ci95"] = np.nan
        else:
            out[f"{col}_mean"] = float(vals.mean())
            out[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            out[f"{col}_median"] = float(vals.median())
            out[f"{col}_min"] = float(vals.min())
            out[f"{col}_max"] = float(vals.max())
            out[f"{col}_ci95"] = float(1.96 * vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else np.nan
    return out


def aggregate_summaries(per_run: pd.DataFrame, by_temp: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if per_run.empty:
        empty = pd.DataFrame()
        return {k: empty for k in ["summary_by_model", "summary_omitted", "summary_outside", "summary_catastrophic"]}
    value_cols = ["MAE_pct", "RMSE_pct", "jitter_ratio", "catastrophic_error_rate_5pct", "catastrophic_error_rate_10pct"]
    rows = []
    for model, g in per_run.groupby("model_name"):
        row = {"model_name": model, "n_runs": int(len(g)), "n_seeds": int(g["seed"].nunique())}
        row.update(_summary_stats(g, value_cols))
        rows.append(row)
    summary_by_model = pd.DataFrame(rows)
    target_rows = []
    if not by_temp.empty:
        for _, r in by_temp.iterrows():
            fold = str(r["fold_name"])
            if fold not in EXPERIMENTS:
                continue
            target_temp = float(EXPERIMENTS[fold]["omitted_temp_C"])
            if np.isclose(float(r["temperature_C"]), target_temp):
                row = r.to_dict()
                row["target_temperature_C"] = target_temp
                row["target_type"] = "outside" if fold in {"Omit N10", "Omit 50"} else ("included_diagnostic" if fold == "Exp D" else "omitted")
                target_rows.append(row)
    target_metrics = pd.DataFrame(target_rows)
    seed_scope_rows = []
    for (model, seed), g in target_metrics.groupby(["model_name", "seed"]):
        omitted = g[g["fold_name"].isin(["Exp A", "Exp B", "Exp C"])]
        outside = g[g["fold_name"].isin(["Omit N10", "Omit 50"])]
        all_main = g[g["fold_name"].isin(DEFAULT_FOLDS)]
        if len(omitted):
            seed_scope_rows.append({
                "model_name": model,
                "seed": int(seed),
                "scope": "omitted_A_B_C",
                "avg_MAE_pct": float(omitted["MAE_pct"].mean()),
                "worst_MAE_pct": float(omitted["MAE_pct"].max()),
                "avg_RMSE_pct": float(omitted["RMSE_pct"].mean()),
                "worst_RMSE_pct": float(omitted["RMSE_pct"].max()),
                "avg_jitter_ratio": float(omitted["jitter_ratio"].mean()),
                "overall_worst_fold_MAE_pct": float(all_main["MAE_pct"].max()) if len(all_main) else np.nan,
            })
        if len(outside):
            seed_scope_rows.append({
                "model_name": model,
                "seed": int(seed),
                "scope": "outside_range",
                "avg_MAE_pct": float(outside["MAE_pct"].mean()),
                "worst_MAE_pct": float(outside["MAE_pct"].max()),
                "avg_RMSE_pct": float(outside["RMSE_pct"].mean()),
                "worst_RMSE_pct": float(outside["RMSE_pct"].max()),
                "avg_jitter_ratio": float(outside["jitter_ratio"].mean()),
                "outside_minus10_MAE_pct": float(outside[outside["fold_name"].eq("Omit N10")]["MAE_pct"].iloc[0]) if len(outside[outside["fold_name"].eq("Omit N10")]) else np.nan,
                "outside_50_MAE_pct": float(outside[outside["fold_name"].eq("Omit 50")]["MAE_pct"].iloc[0]) if len(outside[outside["fold_name"].eq("Omit 50")]) else np.nan,
                "overall_worst_fold_MAE_pct": float(all_main["MAE_pct"].max()) if len(all_main) else np.nan,
            })
    seed_scope = pd.DataFrame(seed_scope_rows)
    scope_summary_rows = []
    for (model, scope), g in seed_scope.groupby(["model_name", "scope"]):
        row = {"model_name": model, "scope": scope, "n_seeds": int(g["seed"].nunique())}
        row.update(_summary_stats(g, ["avg_MAE_pct", "worst_MAE_pct", "avg_RMSE_pct", "worst_RMSE_pct", "avg_jitter_ratio", "outside_minus10_MAE_pct", "outside_50_MAE_pct"]))
        scope_summary_rows.append(row)
    scope_summary = pd.DataFrame(scope_summary_rows)
    summary_omitted = scope_summary[scope_summary["scope"].eq("omitted_A_B_C")].copy()
    summary_outside = scope_summary[scope_summary["scope"].eq("outside_range")].copy()
    cat_rows = []
    for model, g in per_run.groupby("model_name"):
        row = {"model_name": model, "n_runs": int(len(g))}
        row.update(_summary_stats(g, ["catastrophic_error_rate_5pct", "catastrophic_error_rate_10pct"]))
        cat_rows.append(row)
    return {
        "summary_by_model": summary_by_model,
        "target_metrics": target_metrics,
        "seed_scope": seed_scope,
        "summary_omitted": summary_omitted,
        "summary_outside": summary_outside,
        "summary_catastrophic": pd.DataFrame(cat_rows),
    }


def pairwise_comparison(seed_scope: pd.DataFrame, proposed="ThermoGuardSOC_rule_soft_high") -> tuple[pd.DataFrame, pd.DataFrame]:
    if seed_scope.empty or proposed not in set(seed_scope["model_name"]):
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    metrics = ["avg_MAE_pct", "worst_MAE_pct", "avg_jitter_ratio"]
    for scope in seed_scope["scope"].dropna().unique():
        prop = seed_scope[(seed_scope["model_name"].eq(proposed)) & (seed_scope["scope"].eq(scope))]
        for model in sorted(set(seed_scope["model_name"]) - {proposed}):
            base = seed_scope[(seed_scope["model_name"].eq(model)) & (seed_scope["scope"].eq(scope))]
            joined = base.merge(prop, on=["seed", "scope"], suffixes=("_baseline", "_proposed"))
            for metric in metrics:
                if joined.empty:
                    continue
                delta = joined[f"{metric}_baseline"] - joined[f"{metric}_proposed"]
                row = {
                    "baseline_model": model,
                    "proposed_model": proposed,
                    "scope": scope,
                    "metric": metric,
                    "n_paired_seeds": int(len(delta)),
                    "delta_baseline_minus_proposed_mean": float(delta.mean()),
                    "delta_baseline_minus_proposed_std": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                    "delta_baseline_minus_proposed_median": float(delta.median()),
                }
                if len(delta) > 1 and float(delta.std(ddof=1)) > 0:
                    row["cohens_d"] = float(delta.mean() / delta.std(ddof=1))
                else:
                    row["cohens_d"] = np.nan
                try:
                    from scipy import stats

                    if len(delta) > 1:
                        row["paired_ttest_p"] = float(stats.ttest_rel(joined[f"{metric}_baseline"], joined[f"{metric}_proposed"]).pvalue)
                        row["wilcoxon_p"] = float(stats.wilcoxon(delta).pvalue) if np.any(delta != 0) else np.nan
                except Exception:
                    row["paired_ttest_p"] = np.nan
                    row["wilcoxon_p"] = np.nan
                rows.append(row)
    comp = pd.DataFrame(rows)
    if comp.empty:
        return comp, pd.DataFrame()
    sig = comp.pivot_table(
        index=["baseline_model", "scope"],
        columns="metric",
        values="delta_baseline_minus_proposed_mean",
        aggfunc="first",
    ).reset_index()
    return comp, sig


def make_plots(cfg: Final10SeedConfig, per_run: pd.DataFrame, by_temp: pd.DataFrame, summaries: dict[str, pd.DataFrame]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    prefix = cfg.output_prefix
    base = cfg.base_dir
    if per_run.empty:
        return
    seed_scope = summaries.get("seed_scope", pd.DataFrame())

    def save_bar(summary, value_col, title, path):
        if summary.empty or value_col not in summary.columns:
            return
        s = summary.sort_values(value_col)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(s["model_name"], s[value_col], yerr=s.get(value_col.replace("_mean", "_std"), None), color="#2563eb", alpha=0.8)
        ax.set_ylabel(value_col)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(base / path, dpi=150)
        plt.close(fig)

    save_bar(summaries.get("summary_omitted", pd.DataFrame()), "avg_MAE_pct_mean", "Omitted target average MAE", f"{prefix}_bar_omitted_mae.png")
    save_bar(summaries.get("summary_outside", pd.DataFrame()), "avg_MAE_pct_mean", "Outside target average MAE", f"{prefix}_bar_outside_mae.png")
    save_bar(summaries.get("summary_by_model", pd.DataFrame()), "jitter_ratio_mean", "Average jitter ratio across runs", f"{prefix}_bar_jitter.png")

    for scope, out in [("omitted_A_B_C", f"{prefix}_boxplot_omitted_mae.png"), ("outside_range", f"{prefix}_boxplot_outside_mae.png")]:
        data = seed_scope[seed_scope["scope"].eq(scope)]
        if data.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        groups = [g["avg_MAE_pct"].to_numpy(float) for _, g in data.groupby("model_name")]
        labels = [m for m, _ in data.groupby("model_name")]
        ax.boxplot(groups, labels=labels, showmeans=True)
        ax.set_ylabel("MAE (%)")
        ax.set_title(scope)
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(base / out, dpi=150)
        plt.close(fig)

    if not seed_scope.empty:
        omitted = seed_scope[seed_scope["scope"].eq("omitted_A_B_C")]
        if len(omitted):
            fig, ax = plt.subplots(figsize=(8, 5))
            for model, g in omitted.groupby("model_name"):
                ax.scatter(g["avg_MAE_pct"], g["avg_jitter_ratio"], label=model, s=45)
            ax.set_xlabel("omitted avg MAE (%)")
            ax.set_ylabel("omitted avg jitter ratio")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(base / f"{prefix}_pareto_mae_jitter.png", dpi=150)
            plt.close(fig)

    cat = summaries.get("summary_catastrophic", pd.DataFrame())
    if len(cat):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(cat["model_name"], cat["catastrophic_error_rate_5pct_mean"], label=">5%")
        ax.bar(cat["model_name"], cat["catastrophic_error_rate_10pct_mean"], bottom=cat["catastrophic_error_rate_5pct_mean"], label=">10%", alpha=0.7)
        ax.set_ylabel("catastrophic error rate")
        ax.set_title("Catastrophic error rate")
        ax.legend()
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(base / f"{prefix}_catastrophic_error_rate.png", dpi=150)
        plt.close(fig)

    pivot = per_run.pivot_table(index="model_name", columns="fold_name", values="MAE_pct", aggfunc="mean")
    if len(pivot):
        fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(pivot))))
        im = ax.imshow(pivot.to_numpy(float), aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white")
        fig.colorbar(im, ax=ax, label="MAE (%)")
        fig.tight_layout()
        fig.savefig(base / f"{prefix}_fold_heatmap_mae.png", dpi=150)
        plt.close(fig)

    if not by_temp.empty:
        pivot = by_temp.pivot_table(index="model_name", columns="temperature_C", values="MAE_pct", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(pivot))))
        im = ax.imshow(pivot.to_numpy(float), aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)), labels=[f"{c:g}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
        fig.colorbar(im, ax=ax, label="MAE (%)")
        ax.set_xlabel("temperature_C")
        fig.tight_layout()
        fig.savefig(base / f"{prefix}_temperature_heatmap_mae.png", dpi=150)
        plt.close(fig)


def write_reports(cfg: Final10SeedConfig, per_run: pd.DataFrame, by_temp: pd.DataFrame, summaries: dict[str, pd.DataFrame], pairwise: pd.DataFrame, sig: pd.DataFrame):
    prefix = cfg.output_prefix
    lines = [
        "# Final 10-Seed Comparison Report",
        "",
        f"Full-run configuration: `{not cfg.quick_run}`.",
        f"Seeds requested: `{list(cfg.seeds)}`.",
        f"Folds requested: `{list(cfg.folds)}`.",
        "",
        "## Experimental Guardrails",
        "- Main physical SOC label: `labels_physical_smoothQ.csv`.",
        "- Usable-to-cutoff label is not mixed with physical SOC.",
        "- Train cycles are DST + US06; test cycle is FUDS.",
        "- FUDS is not used for supervised training.",
        "- TTA is excluded from main results.",
        "- `T_core_est` is an estimated/effective thermal-state proxy, not measured core temperature.",
        "- -10C labels retain a low-confidence capacity consistency flag.",
        "",
    ]
    for name in ["summary_by_model", "summary_omitted", "summary_outside", "summary_catastrophic"]:
        df = summaries.get(name, pd.DataFrame())
        lines.extend([f"## {name}", df.to_markdown(index=False) if len(df) else "Not available.", ""])
    lines.extend([
        "## Pairwise Baseline Minus Proposed",
        pairwise.to_markdown(index=False) if len(pairwise) else "Pairwise comparison not available. Need matching seeds for proposed and baselines.",
        "",
        "## Safe Claims",
        "- ThermoGuard-SOC combines complementary thermal-state observers using a label-free thermal guard.",
        "- Estimated/effective thermal state improves temperature-domain robustness compared with ambient-only conditioning if supported by the tables above.",
        "- ParamSurface expert is useful as a high-temperature outside-range expert, not a generalist.",
        "- R5 feature-regression remains useful as a baseline but is unstable under omitted/outside temperature shifts if supported by the tables above.",
        "",
        "## Forbidden Claims",
        "- Pure unseen-temperature extrapolation is solved.",
        "- The model is robust to all unseen temperatures.",
        "- `T_core_est` is measured core temperature.",
        "- ParamSurfaceECM is universally superior.",
        "- Voltage-only TTA is pure extrapolation.",
        "- Learned decomposed components are true physical polarization/hysteresis.",
        "- Single-seed result proves robustness.",
    ])
    (cfg.base_dir / f"{prefix}_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")

    tables = []
    for name in ["summary_omitted", "summary_outside", "summary_by_model"]:
        df = summaries.get(name, pd.DataFrame())
        if len(df):
            tables.append(f"% {name}\n" + df.to_latex(index=False, float_format="%.4f"))
    (cfg.base_dir / f"{prefix}_latex_tables.tex").write_text("\n\n".join(tables), encoding="utf-8")

    md = ["# Paper-Ready Tables", ""]
    for name in ["summary_omitted", "summary_outside", "summary_catastrophic"]:
        df = summaries.get(name, pd.DataFrame())
        md.extend([f"## {name}", df.to_markdown(index=False) if len(df) else "Not available.", ""])
    (cfg.base_dir / f"{prefix}_paper_ready_tables.md").write_text("\n".join(md), encoding="utf-8")


def aggregate_outputs(cfg: Final10SeedConfig):
    cfg = _pathify(cfg)
    pred = load_all_completed_predictions(cfg)
    if pred.empty:
        raise RuntimeError("No completed predictions available to aggregate.")
    if cfg.save_raw_predictions:
        raw_path = cfg.base_dir / f"{cfg.output_prefix}_raw_predictions.csv.gz"
        pred.to_csv(raw_path, index=False, compression="gzip")
    per_run, by_temp, by_fold = compute_metrics(pred)
    per_run.to_csv(cfg.base_dir / f"{cfg.output_prefix}_metrics_per_run.csv", index=False)
    by_temp.to_csv(cfg.base_dir / f"{cfg.output_prefix}_metrics_by_temperature.csv", index=False)
    by_fold.to_csv(cfg.base_dir / f"{cfg.output_prefix}_metrics_by_fold.csv", index=False)
    summaries = aggregate_summaries(per_run, by_temp)
    for key, df in summaries.items():
        if key == "seed_scope":
            df.to_csv(cfg.base_dir / f"{cfg.output_prefix}_seed_scope_metrics.csv", index=False)
        elif key == "target_metrics":
            df.to_csv(cfg.base_dir / f"{cfg.output_prefix}_target_metrics.csv", index=False)
        else:
            df.to_csv(cfg.base_dir / f"{cfg.output_prefix}_{key}.csv", index=False)
    pairwise, sig = pairwise_comparison(summaries.get("seed_scope", pd.DataFrame()))
    pairwise.to_csv(cfg.base_dir / f"{cfg.output_prefix}_pairwise_comparison.csv", index=False)
    sig.to_csv(cfg.base_dir / f"{cfg.output_prefix}_significance_summary.csv", index=False)
    make_plots(cfg, per_run, by_temp, summaries)
    write_reports(cfg, per_run, by_temp, summaries, pairwise, sig)
    return {
        "predictions": pred,
        "per_run": per_run,
        "by_temperature": by_temp,
        "by_fold": by_fold,
        **summaries,
        "pairwise": pairwise,
        "significance": sig,
    }


def run_and_aggregate(cfg: Final10SeedConfig, *, force=False):
    cfg = _pathify(cfg)
    statuses = run_all(cfg, force=force)
    outputs = aggregate_outputs(cfg)
    save_registry(cfg)
    return statuses, outputs


def _parse_csv_tuple(text: str, cast=str):
    if text is None or text == "":
        return tuple()
    return tuple(cast(x.strip()) for x in str(text).split(",") if x.strip())


def main():
    parser = argparse.ArgumentParser(description="Run final 10-seed SOC comparison with resume/cache.")
    parser.add_argument("--quick-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--models", default=None)
    parser.add_argument("--folds", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--allow-frozen-thermoguard-fallback", action="store_true")
    parser.add_argument("--no-train-thermoguard-deps", action="store_true")
    args = parser.parse_args()

    cfg = quick_config(".") if args.quick_run else full_config(".")
    if args.seeds is not None:
        cfg.seeds = _parse_csv_tuple(args.seeds, int)
    if args.models is not None:
        cfg.models = _parse_csv_tuple(args.models, str)
    if args.folds is not None:
        cfg.folds = _parse_csv_tuple(args.folds, str)
    if args.max_epochs is not None:
        cfg.max_epochs_override = int(args.max_epochs)
    if args.allow_frozen_thermoguard_fallback:
        cfg.allow_frozen_thermoguard_fallback = True
    if args.no_train_thermoguard_deps:
        cfg.train_thermoguard_dependencies = False
    cfg = _pathify(cfg)
    save_registry(cfg)
    if args.aggregate_only:
        outputs = aggregate_outputs(cfg)
        print(outputs["per_run"].to_string(index=False))
    else:
        statuses, outputs = run_and_aggregate(cfg, force=args.force)
        print(statuses.to_string(index=False))
        print(outputs["per_run"].to_string(index=False))


if __name__ == "__main__":
    main()
