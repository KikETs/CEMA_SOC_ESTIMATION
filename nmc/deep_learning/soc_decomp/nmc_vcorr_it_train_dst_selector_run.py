from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import copy
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler

from .config import make_cfg
from .deep_no_leak_experiment import SequenceWindowDataset, make_eval_loader
from .extrapolation_robustness import temperature_balanced_loader
from .models import DecomposedWindowDataset, collate_meta_to_frame
from .nmc_branchbands_experiment import (
    build_feature_frames,
    estimate_dynamic_r_calibration,
    estimate_lfp_style_vcorr_calibration,
    estimate_nmc_tailored_minimax_calibration,
    estimate_rtvar_lowv_calibration,
    estimate_vcorr_ocv_calibration,
    estimate_r0_by_temperature,
    estimate_r0_by_temperature_profile,
    find_csv_files,
    write_start_audit,
)
from .nmc_vcorr_it_condinv_staged_exact import (
    CondInvConfig,
    CondInvVariant,
    SnapshotEnsembleBase,
    Stage2CorrectedModel,
    _meta_list,
    conditional_profile_mmd,
    eval_by_temp,
    make_model,
    set_seed,
    to_variant,
    train_stage2_correction,
)
from .nmc_vcorr_it_goal_remote_screen import VcorrITGoalModel, group_keys, temp_weights
from .nmc_vcorr_it_lstm_singlehead_bytemp import FEATURE_COLS
from .nmc_vcorr_it_train_selector_audit import eval_by_temp_drive
from .nmc_vit_feature_lstm_experiment import (
    add_vit_engineered_features,
    write_input_schema,
    write_leakage_audit,
)
from .nmc_vit_feature_lstm_experiment import feature_columns as vit_feature_columns
from .runtime import configure_torch_runtime, device
from .training import make_scaled_frames_for_ablation


BASE_PREFIX = "nmc_goal_vcorr_it_train_dst25_selector"


@dataclass
class TrainDSTSelectorConfig:
    base_dir: Path = Path(".")
    raw_root: Path = Path("nmc_soc_ocvstart_relabelled_from_lc_ocv/data/NMC SAMSUNG INR 18650 2Ah")
    output_prefix: str = BASE_PREFIX
    seeds: tuple[int, ...] = (0, 1, 2)
    train_profiles: tuple[str, ...] = ("DST", "US06")
    valid_profiles: tuple[str, ...] = ("VALIDATION",)
    test_profiles: tuple[str, ...] = ("FUDS",)
    train_temperatures: tuple[float, ...] = ()
    valid_temperatures: tuple[float, ...] = ()
    test_temperatures: tuple[float, ...] = ()
    finetune_temperatures: tuple[float, ...] = ()
    finetune_epochs: int = 0
    finetune_lr_scale: float = 0.35
    stage2_train_temperatures: tuple[float, ...] = ()
    stage2_valid_temperatures: tuple[float, ...] = ()
    stage2_test_temperatures: tuple[float, ...] = ()
    window_len: int = 50
    stride: int = 3
    epochs: int = 10
    selector_min_epoch: int = 1
    selector_max_epoch: int = 0
    stage2_epochs: int = 60
    eval_every: int = 5
    stage1_eval_every: int = 1
    batch_size: int = 1024
    lr: float = 8e-4
    lr_stage2: float = 8e-4
    weight_decay: float = 2e-4
    hidden_size: int = 64
    layers: int = 5
    kernel_size: int = 5
    recurrent: str = "tcn"
    head_kind: str = "linear"
    temp_mode: str = "none"
    dropout: float = 0.06
    huber_beta: float = 0.02
    rex_group: str = "temperature_drive"
    lambda_rex: float = 2.0
    lambda_condinv: float = 0.02
    weight_0: float = 4.0
    weight_25: float = 2.2
    weight_45: float = 1.0
    stage1_selector: str = "train25_dst"
    selector_regime_min_windows: int = 30
    fixed_stage1_epoch: int = 0
    model_kind: str = "single"
    fusion_h64_weight: float = 0.5
    multiscale_local_len: int = 200
    dual_context_gate_center: float = 0.15
    dual_context_gate_sharpness: float = 60.0
    dual_context_gate_local_min: float = 0.2
    dual_context_gate_local_max: float = 0.8
    dual_context_fusion_delta_limit: float = 2.0
    focus45_weight: float = 12.0
    keep_lambda: float = 4.0
    corr_mode: str = "cold_hot"
    corr_limit: float = 1.2
    corr_zero_init: bool = True
    stage2_select_rule: str = "none"
    stage2_stage1_threshold: int = 7
    stage2_early_epoch: int = 30
    stage2_late_epoch: int = 35
    test_blind: bool = False
    train_sampler: str = "temperature_balanced"
    stage1_ensemble_epochs: str = ""
    feature_set: str = "vcorr_it"
    stage2_feature_set: str = ""
    sampler_seed_mode: str = "seed"
    test_blind_rng_burn: int = 0
    diagnostic_test_every: int = 0
    skip_stage2: bool = False
    lambda_current_consistency: float = 0.0
    current_noise_std: float = 0.0
    current_dropout_prob: float = 0.0
    lambda_profile_adv: float = 0.0
    profile_adv_grl: float = 1.0
    lambda_profile_supcon: float = 0.0
    supcon_temperature: float = 0.15
    anchor_residual_limit: float = 0.12
    anchor_residual_limit_init: str = "value"
    anchor_residual_limit_mode: str = "fixed"
    anchor_residual_limit_lower: float = 0.0
    anchor_residual_limit_upper: float = 0.2
    lambda_anchor_loss: float = 0.2
    lambda_regime_cvar: float = 0.0
    regime_cvar_top_frac: float = 0.34
    lambda_regime_var: float = 0.0
    lambda_residual_aux_loss: float = 0.0
    lambda_residual_zeromean_loss: float = 0.0
    lambda_dual_context_branch_aux_loss: float = 0.0
    dual_context_branch_pretrain_epochs: int = 0
    dual_context_freeze_branches_after_pretrain: bool = False
    base_pretrain_epochs: int = 0
    freeze_base_after_pretrain: bool = False
    lambda_mid_delta_loss: float = 0.0
    lambda_zbin_cvar: float = 0.0
    zbin_cvar_top_frac: float = 0.34
    lambda_cold_socbin_cvar: float = 0.0
    cold_socbin_cvar_top_frac: float = 0.34
    cold_socbin_cvar_bin_width: float = 0.1
    cold_socbin_cvar_temp: float = 0.0
    lambda_cold_profile_socbin_cvar: float = 0.0
    cold_profile_socbin_cvar_top_frac: float = 0.34
    cold_profile_socbin_cvar_bin_width: float = 0.1
    cold_profile_socbin_cvar_temp: float = 0.0
    lambda_cold_socbin_bias: float = 0.0
    cold_socbin_bias_bin_width: float = 0.1
    cold_socbin_bias_soc_upper: float = 0.5
    cold_socbin_bias_temp: float = 0.0
    lambda_cold_profile_socbin_underbias: float = 0.0
    cold_profile_socbin_underbias_bin_width: float = 0.1
    cold_profile_socbin_underbias_soc_upper: float = 0.5
    cold_profile_socbin_underbias_temp: float = 0.0
    weight_ema_decay: float = 0.0
    weight_ema_start_epoch: int = 1
    lambda_cold_lowsoc_overpred_loss: float = 0.0
    cold_lowsoc_overpred_soc_upper: float = 0.2
    cold_lowsoc_overpred_temp: float = 0.0
    lambda_cold_lowsoc_underpred_loss: float = 0.0
    cold_lowsoc_underpred_soc_upper: float = 0.2
    cold_lowsoc_underpred_temp: float = 0.0
    lambda_regime_asym_loss: float = 0.0
    regime_asym_soc_upper: float = 0.5
    regime_asym_cold_temp: float = 0.0
    regime_asym_under_weight: float = 1.0
    regime_asym_over_weight: float = 1.0
    lambda_lowdyn_highspan_under_loss: float = 0.0
    lowdyn_highspan_soc_upper: float = 0.5
    lowdyn_highspan_cold_temp: float = 0.0
    lowdyn_highspan_dynamic_center: float = 1.4
    lowdyn_highspan_dynamic_sharpness: float = 3.0
    lowdyn_highspan_span_center: float = 0.08
    lowdyn_highspan_span_sharpness: float = 60.0
    voltage_sag_augment_prob: float = 0.0
    voltage_sag_augment_scale: float = 0.0
    voltage_sag_augment_soc_upper: float = 0.5
    voltage_sag_augment_temp: float = 0.0
    lambda_cold_voltage_shift_consistency: float = 0.0
    cold_voltage_shift_consistency_scale: float = 0.05
    cold_voltage_shift_consistency_soc_upper: float = 0.5
    cold_voltage_shift_consistency_temp: float = 0.0
    hard_region_loss_weight: float = 0.0
    hard_region_soc_threshold: float = 0.4
    hard_region_soc_lower: float = -1.0
    hard_region_soc_upper: float = -1.0
    hard_region_dynamic_weight: float = 0.5
    hard_region_cold_only: bool = False
    cold_corrector_limit: float = 0.08
    cold_corrector_soc_threshold: float = 0.35
    cold_corrector_temp_threshold: float = -0.45
    cold_corrector_gate_sharpness: float = 8.0
    cold_corrector_sag_gate: bool = False
    high_dynamic_corrector_center: float = 0.15
    high_dynamic_corrector_sharpness: float = 40.0
    high_dynamic_corrector_istd_center: float = -1.0
    high_dynamic_corrector_istd_sharpness: float = 20.0
    ssl_pretrain_epochs: int = 0
    ssl_lr: float = 8e-4
    ssl_recon_weight: float = 1.0
    ssl_next_vcorr_weight: float = 0.5
    ssl_slope_weight: float = 0.25
    valid_split_mode: str = "profile"
    valid_block_rows: int = 800
    valid_block_mod: int = 5
    valid_block_index: int = 4
    save_predictions: bool = False
    save_train_predictions: bool = False
    save_checkpoints: bool = False
    save_final_weights: bool = False
    final_only_eval: bool = False
    skip_train_final_eval: bool = False
    ema_perturbation_importance: bool = False
    sequence_training: bool = False
    cache_dataset_cuda: bool = False
    tqdm_epochs: bool = False
    num_workers: int = 4
    prefetch_factor: int = 4
    v_corr_tau_s: float = 120.0
    v_pol_mid_tau_s: float = 60.0
    v_pol_slow_tau_s: float = 600.0
    v_hys_tau_s: float = 1200.0
    v_corr_variant: str = "auto"
    v_corr_tail_gate_v: float = 3.30
    v_corr_tail_gate_s: float = 0.15
    v_corr_tail_tau_s: float = 40.0
    v_corr_asym_down_tau_s: float = 20.0
    v_corr_asym_up_tau_s: float = 160.0
    v_corr_lv_gate_v: float = 3.35
    v_corr_lv_gate_s: float = 0.15
    v_corr_ocv_low_current_a: float = 0.05
    v_corr_ocv_bin_width_soc: float = 0.01
    v_corr_ocv_min_bin_points: int = 5
    v_corr_ocv_tau_fast_s: float = 20.0
    v_corr_ocv_tau_slow_s: float = 200.0
    v_corr_ocv_ridge: float = 1e-4
    v_corr_ocv_coef_limit: float = 0.30
    v_corr_ocv_intercept_limit_v: float = 0.20
    r0_mode: str = "train_temperature"
    r0_profile_quantile: float = 0.5
    lfpstyle_fit_stride: int = 5
    lfpstyle_fit_max_nfev: int = 250
    lfpstyle_v_floor_raw: float = 2.45
    lfpstyle_vfloor_beta: float = 20.0
    lfpstyle_r0_max_ohm: float = 0.22
    lfpstyle_r0_min_ohm: float = 0.0
    lfpstyle_g_corr_scale: float = 1.35
    lfpstyle_pol_fast_limit_v: float = 0.16
    lfpstyle_pol_mid_limit_v: float = 0.12
    lfpstyle_pol_slow_limit_v: float = 0.10
    lfpstyle_pol_limit_v: float = 0.25
    lfpstyle_hys_limit_v: float = 0.20


_CSV_FILE_CACHE: dict[Path, tuple[Path, ...]] = {}
_RAW_SOURCE_COLUMNS_CACHE: dict[Path, list[str]] = {}
_R0_CACHE: dict[tuple, pd.DataFrame] = {}
_FEATURE_FRAME_CACHE: dict[tuple, dict[str, list[pd.DataFrame]]] = {}
_SCALED_FRAME_CACHE: dict[tuple, tuple[dict[str, list[pd.DataFrame]], object]] = {}


def _mean_scalar(values) -> float:
    if not values:
        return 0.0
    first = values[0]
    if torch.is_tensor(first):
        return float(torch.stack([v.reshape(()) for v in values]).mean().detach().cpu())
    return float(np.mean(values))


def _clear_per_config_cuda_cache() -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _cached_find_csv_files(raw_root: Path) -> list[Path]:
    root = Path(raw_root).resolve()
    if root not in _CSV_FILE_CACHE:
        _CSV_FILE_CACHE[root] = tuple(find_csv_files(root))
        print(f"[cache miss] csv_files raw_root={root} n={len(_CSV_FILE_CACHE[root])}", flush=True)
    else:
        print(f"[cache hit] csv_files raw_root={root} n={len(_CSV_FILE_CACHE[root])}", flush=True)
    return list(_CSV_FILE_CACHE[root])


def _cached_raw_source_columns(first_file: Path) -> list[str]:
    path = Path(first_file).resolve()
    if path not in _RAW_SOURCE_COLUMNS_CACHE:
        _RAW_SOURCE_COLUMNS_CACHE[path] = list(pd.read_csv(path, nrows=1).columns)
    return list(_RAW_SOURCE_COLUMNS_CACHE[path])


def _r0_cache_key(cfg: TrainDSTSelectorConfig, files: list[Path]) -> tuple:
    file_sig = tuple((str(Path(p).resolve()), Path(p).stat().st_mtime_ns, Path(p).stat().st_size) for p in files)
    return (file_sig, tuple(cfg.train_profiles), str(cfg.r0_mode), float(cfg.r0_profile_quantile))


def _cached_r0_by_temperature(cfg: TrainDSTSelectorConfig, files: list[Path]) -> pd.DataFrame:
    key = _r0_cache_key(cfg, files)
    if key not in _R0_CACHE:
        if str(cfg.r0_mode) == "profile_observed":
            _R0_CACHE[key] = estimate_r0_by_temperature_profile(files, quantile=float(cfg.r0_profile_quantile))
            print(f"[cache miss] r0 mode=profile_observed q={float(cfg.r0_profile_quantile):.3f}", flush=True)
        elif str(cfg.r0_mode) == "train_temperature":
            _R0_CACHE[key] = estimate_r0_by_temperature(files, cfg.train_profiles)
            print(f"[cache miss] r0 train_profiles={','.join(cfg.train_profiles)}", flush=True)
        elif str(cfg.r0_mode) == "train_temperature_quantile":
            _R0_CACHE[key] = estimate_r0_by_temperature(files, cfg.train_profiles, quantile=float(cfg.r0_profile_quantile))
            print(f"[cache miss] r0 train_profiles={','.join(cfg.train_profiles)} q={float(cfg.r0_profile_quantile):.3f}", flush=True)
        else:
            raise ValueError(f"Unknown r0_mode={cfg.r0_mode!r}")
    else:
        print(f"[cache hit] r0 mode={cfg.r0_mode}", flush=True)
    return _R0_CACHE[key].copy()


def _effective_v_corr_variant(cfg: TrainDSTSelectorConfig) -> str:
    explicit = str(getattr(cfg, "v_corr_variant", "auto") or "auto")
    if explicit != "auto":
        return explicit
    feature_sets = {str(cfg.feature_set), str(cfg.stage2_feature_set or cfg.feature_set)}
    if "paper_g4_all_ema_asymohm" in feature_sets:
        return "ohm_asym_d20_u160"
    if "paper_g4_all_ema_lvblend" in feature_sets:
        return "lvblend_orig_asym_v3p35_s0p15"
    if "paper_g4_all_ema_ocvcal" in feature_sets:
        return "ocvcal_dynfit_v1"
    if "paper_g4_all_ema_tailaware" in feature_sets:
        return "gated_r0_ema40_by_v_3p30_s0p15"
    return "orig_ohm_ema120"


def _feature_frame_cache_key(cfg: TrainDSTSelectorConfig, files: list[Path], r0_df: pd.DataFrame) -> tuple:
    file_sig = tuple((str(Path(p).resolve()), Path(p).stat().st_mtime_ns, Path(p).stat().st_size) for p in files)
    sort_cols = ["temperature_C", "profile"] if "profile" in r0_df.columns else ["temperature_C"]
    r0_sig = tuple(
        (
            float(r["temperature_C"]),
            str(r["profile"]) if "profile" in r0_df.columns else "",
            float(r["r0_ohm"]),
        )
        for _, r in r0_df.sort_values(sort_cols).iterrows()
    )
    return (
        file_sig,
        r0_sig,
        tuple(cfg.train_profiles),
        tuple(cfg.valid_profiles),
        tuple(cfg.test_profiles),
        float(cfg.v_corr_tau_s),
        float(cfg.v_pol_mid_tau_s),
        float(cfg.v_pol_slow_tau_s),
        float(cfg.v_hys_tau_s),
        str(_effective_v_corr_variant(cfg)),
        float(cfg.v_corr_tail_gate_v),
        float(cfg.v_corr_tail_gate_s),
        float(cfg.v_corr_tail_tau_s),
        float(cfg.v_corr_asym_down_tau_s),
        float(cfg.v_corr_asym_up_tau_s),
        float(cfg.v_corr_lv_gate_v),
        float(cfg.v_corr_lv_gate_s),
        float(cfg.v_corr_ocv_low_current_a),
        float(cfg.v_corr_ocv_bin_width_soc),
        int(cfg.v_corr_ocv_min_bin_points),
        float(cfg.v_corr_ocv_tau_fast_s),
        float(cfg.v_corr_ocv_tau_slow_s),
        float(cfg.v_corr_ocv_ridge),
        float(cfg.v_corr_ocv_coef_limit),
        float(cfg.v_corr_ocv_intercept_limit_v),
        float(getattr(cfg, "dynr_ridge", 1e-3)),
        float(getattr(cfg, "dynr_log_bound", 0.6931471805599453)),
        int(getattr(cfg, "lfpstyle_fit_stride", 5)),
        int(getattr(cfg, "lfpstyle_fit_max_nfev", 250)),
        float(getattr(cfg, "lfpstyle_v_floor_raw", 2.45)),
        float(getattr(cfg, "lfpstyle_vfloor_beta", 20.0)),
        float(getattr(cfg, "lfpstyle_r0_max_ohm", 0.22)),
        float(getattr(cfg, "lfpstyle_r0_min_ohm", 0.0)),
        float(getattr(cfg, "lfpstyle_g_corr_scale", 1.35)),
        float(getattr(cfg, "lfpstyle_pol_fast_limit_v", 0.16)),
        float(getattr(cfg, "lfpstyle_pol_mid_limit_v", 0.12)),
        float(getattr(cfg, "lfpstyle_pol_slow_limit_v", 0.10)),
        float(getattr(cfg, "lfpstyle_pol_limit_v", 0.25)),
        float(getattr(cfg, "lfpstyle_hys_limit_v", 0.20)),
        str(cfg.r0_mode),
        float(cfg.r0_profile_quantile),
    )


def _cached_feature_frames(
    cfg: TrainDSTSelectorConfig,
    files: list[Path],
    r0_df: pd.DataFrame,
    lfp_style_calibration: dict | None = None,
    nmc_tailored_calibration: dict | None = None,
    rtvar_lowv_calibration: dict | None = None,
) -> dict[str, list[pd.DataFrame]]:
    key = _feature_frame_cache_key(cfg, files, r0_df)
    if key not in _FEATURE_FRAME_CACHE:
        _FEATURE_FRAME_CACHE[key] = add_vit_engineered_features(
            build_feature_frames(
                cfg,
                files,
                r0_df,
                lfp_style_calibration=lfp_style_calibration,
                nmc_tailored_calibration=nmc_tailored_calibration,
                rtvar_lowv_calibration=rtvar_lowv_calibration,
            )
        )
        counts = {split: len(frames) for split, frames in _FEATURE_FRAME_CACHE[key].items()}
        print(f"[cache miss] feature_frames splits={counts}", flush=True)
    else:
        counts = {split: len(frames) for split, frames in _FEATURE_FRAME_CACHE[key].items()}
        print(f"[cache hit] feature_frames splits={counts}", flush=True)
    return _FEATURE_FRAME_CACHE[key]


def _scaled_frame_cache_key(frames: dict[str, list[pd.DataFrame]], feature_cols: list[str]) -> tuple:
    value_sig = []
    selected_cols = list(feature_cols)
    for split, split_frames in sorted(frames.items()):
        split_sig = []
        for frame in split_frames:
            col_sig = []
            for col in selected_cols:
                arr = pd.to_numeric(frame[col], errors="coerce").to_numpy(np.float64)
                if len(arr) == 0:
                    col_sig.append((col, 0, 0.0, 0.0, 0.0, 0.0))
                    continue
                finite = np.isfinite(arr)
                if not finite.any():
                    col_sig.append((col, len(arr), float("nan"), float("nan"), float("nan"), float("nan")))
                    continue
                vals = arr[finite]
                col_sig.append(
                    (
                        col,
                        len(arr),
                        round(float(vals[0]), 8),
                        round(float(vals[-1]), 8),
                        round(float(np.nanmean(vals)), 8),
                        round(float(np.nanstd(vals)), 8),
                    )
                )
            split_sig.append((str(frame["trajectory_id"].iloc[0]), int(len(frame)), tuple(col_sig)))
        value_sig.append((split, tuple(split_sig)))
    frame_sig = tuple(
        (split, tuple((str(frame["trajectory_id"].iloc[0]), int(len(frame))) for frame in split_frames))
        for split, split_frames in sorted(frames.items())
    )
    return (frame_sig, tuple(feature_cols), tuple(value_sig))


def _cached_scaled_frames_for_ablation(frames: dict[str, list[pd.DataFrame]], feature_cols: list[str]):
    key = _scaled_frame_cache_key(frames, list(feature_cols))
    if key not in _SCALED_FRAME_CACHE:
        _SCALED_FRAME_CACHE[key] = make_scaled_frames_for_ablation(frames, feature_cols)
        print(f"[cache miss] scaled_frames features={len(feature_cols)}", flush=True)
    else:
        print(f"[cache hit] scaled_frames features={len(feature_cols)}", flush=True)
    return _SCALED_FRAME_CACHE[key]


class H64H128FusionModel(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.06, fixed_h64_weight: float | None = None):
        super().__init__()
        self.fixed_h64_weight = fixed_h64_weight
        self.branch64 = VcorrITGoalModel(
            input_dim=input_dim,
            hidden_size=64,
            recurrent="tcn",
            layers=5,
            head_kind="linear",
            temp_mode="none",
            dropout=float(dropout),
            kernel_size=5,
            norm_kind="channel",
        )
        self.branch128 = VcorrITGoalModel(
            input_dim=input_dim,
            hidden_size=128,
            recurrent="tcn",
            layers=6,
            head_kind="linear",
            temp_mode="none",
            dropout=float(dropout),
            kernel_size=5,
            norm_kind="channel",
        )
        self.proj64 = nn.Linear(64, 128)
        gate_dim = input_dim * 4 + 3
        self.gate = nn.Sequential(
            nn.Linear(gate_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(64, 2),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def _window_stats(self, x: torch.Tensor) -> torch.Tensor:
        x_end = x[:, -1, :]
        x_mean = x.mean(dim=1)
        x_std = x.std(dim=1, unbiased=False)
        x_delta = x_end - x[:, 0, :]
        return torch.cat([x_end, x_mean, x_std, x_delta], dim=1)

    def _branch_outputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h64 = self.branch64.encode_sequence(x)
        h128 = self.branch128.encode_sequence(x)
        log64 = self._logits_from_hidden(self.branch64, h64, x)
        log128 = self._logits_from_hidden(self.branch128, h128, x)
        return h64, h128, log64, log128

    @staticmethod
    def _logits_from_hidden(model: VcorrITGoalModel, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        temp = x[..., model.temp_idx:model.temp_idx + 1]
        if model.temp_mode == "none":
            return model.base_head(h)
        if model.temp_mode == "bias":
            return model.base_head(h) + model.temp_bias(temp)
        raise ValueError(f"Unknown temp_mode={model.temp_mode}")

    def _weights(self, x: torch.Tensor, log64: torch.Tensor, log128: torch.Tensor) -> torch.Tensor:
        if self.fixed_h64_weight is not None:
            w64 = torch.full((x.shape[0], 1), float(self.fixed_h64_weight), device=x.device, dtype=x.dtype)
            return torch.cat([w64, 1.0 - w64], dim=1)
        end64 = log64[:, -1, :]
        end128 = log128[:, -1, :]
        gate_input = torch.cat([self._window_stats(x), end64, end128, end64 - end128], dim=1)
        return torch.softmax(self.gate(gate_input), dim=1)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h64, h128, log64, log128 = self._branch_outputs(x)
        weights = self._weights(x, log64, log128)
        h64_proj = self.proj64(h64)
        return weights[:, None, 0:1] * h64_proj + weights[:, None, 1:2] * h128

    def logits_sequence(self, x: torch.Tensor) -> torch.Tensor:
        _h64, _h128, log64, log128 = self._branch_outputs(x)
        weights = self._weights(x, log64, log128)
        return weights[:, None, 0:1] * log64 + weights[:, None, 1:2] * log128

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits_sequence(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1, :]


class LastQueryMHAEndpointModel(nn.Module):
    """Endpoint SOC estimator with attention over the observed history window."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        recurrent: str,
        layers: int,
        head_kind: str,
        temp_mode: str,
        dropout: float,
        kernel_size: int,
        num_heads: int | None = None,
    ):
        super().__init__()
        self.backbone = VcorrITGoalModel(
            input_dim=input_dim,
            hidden_size=int(hidden_size),
            recurrent=str(recurrent),
            layers=int(layers),
            head_kind=str(head_kind),
            temp_mode=str(temp_mode),
            dropout=float(dropout),
            kernel_size=int(kernel_size),
            norm_kind="channel",
        )
        heads = int(num_heads or (4 if int(hidden_size) % 4 == 0 else 1))
        self.mha = nn.MultiheadAttention(
            embed_dim=int(hidden_size),
            num_heads=heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm = nn.LayerNorm(int(hidden_size))
        self.dropout = nn.Dropout(float(dropout))

    def _endpoint_context(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone.encode_sequence(x)
        q = h[:, -1:, :]
        attended, _ = self.mha(q, h, h, need_weights=False)
        context = self.norm(q + self.dropout(attended))
        return h, context

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h, context = self._endpoint_context(x)
        if h.shape[1] == 1:
            return context
        return torch.cat([h[:, :-1, :], context], dim=1)

    def logits_sequence(self, x: torch.Tensor) -> torch.Tensor:
        _h, context = self._endpoint_context(x)
        logits = self.backbone.base_head(context)
        return logits.expand(-1, x.shape[1], -1)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits_sequence(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _h, context = self._endpoint_context(x)
        return torch.sigmoid(self.backbone.base_head(context))[:, 0, :]


class CausalMultiScaleEndpointModel(nn.Module):
    """Single-architecture causal head mixing endpoint, local, and global contexts."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        recurrent: str,
        layers: int,
        head_kind: str,
        temp_mode: str,
        dropout: float,
        kernel_size: int,
        feature_cols: list[str],
        local_len: int = 200,
    ):
        super().__init__()
        if str(temp_mode) != "none":
            raise ValueError("CausalMultiScaleEndpointModel currently expects temp_mode='none'.")
        self.feature_cols = list(feature_cols)
        self.local_len = int(local_len)
        self.backbone = VcorrITGoalModel(
            input_dim=input_dim,
            hidden_size=int(hidden_size),
            recurrent=str(recurrent),
            layers=int(layers),
            head_kind=str(head_kind),
            temp_mode=str(temp_mode),
            dropout=float(dropout),
            kernel_size=int(kernel_size),
            norm_kind="channel",
        )
        self.local_norm = nn.LayerNorm(int(hidden_size))
        self.global_norm = nn.LayerNorm(int(hidden_size))
        self.endpoint_head = VcorrITGoalModel._make_head(int(hidden_size), str(head_kind), float(dropout))
        self.local_head = VcorrITGoalModel._make_head(int(hidden_size), str(head_kind), float(dropout))
        self.global_head = VcorrITGoalModel._make_head(int(hidden_size), str(head_kind), float(dropout))
        self.gate = nn.Sequential(
            nn.Linear(6, max(16, int(hidden_size) // 4)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(max(16, int(hidden_size) // 4), 3),
        )

    def _feature_index(self, name: str) -> int | None:
        try:
            return self.feature_cols.index(name)
        except ValueError:
            return None

    @staticmethod
    def _causal_mean(z: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
        b, length, dim = z.shape
        csum = torch.cumsum(z, dim=1)
        if max_len is None or int(max_len) <= 0 or int(max_len) >= length:
            denom = torch.arange(1, length + 1, device=z.device, dtype=z.dtype).view(1, length, 1)
            return csum / denom
        padded = torch.cat([z.new_zeros((b, 1, dim)), csum], dim=1)
        end_idx = torch.arange(1, length + 1, device=z.device)
        start_idx = (end_idx - int(max_len)).clamp_min(0)
        sums = padded.index_select(1, end_idx) - padded.index_select(1, start_idx)
        denom = (end_idx - start_idx).to(dtype=z.dtype).view(1, length, 1).clamp_min(1.0)
        return sums / denom

    def _causal_gate_features(self, x: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        i_idx = self._feature_index("I_raw")
        v_idx = self._feature_index("V_corr_raw")
        t_idx = self._feature_index("T")
        if i_idx is None or v_idx is None or t_idx is None:
            raise RuntimeError("multi_scale_endpoint requires V_corr_raw, I_raw, and T features.")
        i = x[:, :, i_idx:i_idx + 1]
        v = x[:, :, v_idx:v_idx + 1]
        temp = x[:, :, t_idx:t_idx + 1]
        abs_idx = self._feature_index("absI")
        abs_i = x[:, :, abs_idx:abs_idx + 1].abs() if abs_idx is not None else i.abs()
        di_idx = self._feature_index("dI")
        if di_idx is not None:
            d_i = x[:, :, di_idx:di_idx + 1]
        else:
            d_i = torch.cat([i[:, :1, :].new_zeros((b, 1, 1)), i[:, 1:, :] - i[:, :-1, :]], dim=1)
        denom = torch.arange(1, length + 1, device=x.device, dtype=x.dtype).view(1, length, 1)
        abs_i_mean = torch.cumsum(abs_i, dim=1) / denom
        i_mean = torch.cumsum(i, dim=1) / denom
        i_sq_mean = torch.cumsum(i.square(), dim=1) / denom
        i_std = (i_sq_mean - i_mean.square()).clamp_min(0.0).sqrt()
        di_energy = torch.cumsum(d_i.square(), dim=1) / denom
        v_span = torch.cummax(v, dim=1).values - torch.cummin(v, dim=1).values
        temp_mean = torch.cumsum(temp, dim=1) / denom
        return torch.cat([abs_i_mean, i_std, di_energy, v_span, temp, temp_mean], dim=-1)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode_sequence(x)

    def logits_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone.encode_sequence(x)
        local = self.local_norm(self._causal_mean(h, self.local_len))
        global_ctx = self.global_norm(self._causal_mean(h, None))
        logits = torch.stack(
            [
                self.endpoint_head(h),
                self.local_head(local),
                self.global_head(global_ctx),
            ],
            dim=-1,
        )
        weights = torch.softmax(self.gate(self._causal_gate_features(x)), dim=-1).unsqueeze(-2)
        return torch.sum(logits * weights, dim=-1)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits_sequence(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1, :]


class DualContextEndpointModel(nn.Module):
    """Endpoint estimator with local and full-window causal branches."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        recurrent: str,
        layers: int,
        head_kind: str,
        temp_mode: str,
        dropout: float,
        kernel_size: int,
        feature_cols: list[str],
        local_len: int = 200,
        global_feature_cols: list[str] | None = None,
        gate_mode: str = "learned",
        dynamic_gate_center: float = 0.15,
        dynamic_gate_sharpness: float = 60.0,
        dynamic_gate_local_min: float = 0.2,
        dynamic_gate_local_max: float = 0.8,
        fusion_delta_limit: float = 2.0,
    ):
        super().__init__()
        if str(temp_mode) != "none":
            raise ValueError("DualContextEndpointModel currently expects temp_mode='none'.")
        self.feature_cols = list(feature_cols)
        self.local_len = int(local_len)
        self.gate_mode = str(gate_mode)
        if self.gate_mode not in {"learned", "dynamic", "fusion"}:
            raise ValueError(f"Unknown dual-context gate_mode={gate_mode!r}")
        self.dynamic_gate_center = float(dynamic_gate_center)
        self.dynamic_gate_sharpness = float(dynamic_gate_sharpness)
        self.dynamic_gate_local_min = float(dynamic_gate_local_min)
        self.dynamic_gate_local_max = float(dynamic_gate_local_max)
        self.fusion_delta_limit = float(fusion_delta_limit)
        if not 0.0 <= self.dynamic_gate_local_min <= self.dynamic_gate_local_max <= 1.0:
            raise ValueError("dynamic gate local bounds must satisfy 0 <= min <= max <= 1.")
        if global_feature_cols is None:
            self.global_indices: list[int] | None = None
            global_input_dim = int(input_dim)
        else:
            missing = [c for c in global_feature_cols if c not in self.feature_cols]
            if missing:
                raise RuntimeError(f"dual context global branch is missing selected features: {missing}")
            self.global_indices = [self.feature_cols.index(c) for c in global_feature_cols]
            global_input_dim = len(self.global_indices)
        self.global_model = VcorrITGoalModel(
            input_dim=global_input_dim,
            hidden_size=int(hidden_size),
            recurrent=str(recurrent),
            layers=int(layers),
            head_kind=str(head_kind),
            temp_mode=str(temp_mode),
            dropout=float(dropout),
            kernel_size=int(kernel_size),
            norm_kind="channel",
        )
        self.local_model = VcorrITGoalModel(
            input_dim=input_dim,
            hidden_size=int(hidden_size),
            recurrent=str(recurrent),
            layers=int(layers),
            head_kind=str(head_kind),
            temp_mode=str(temp_mode),
            dropout=float(dropout),
            kernel_size=int(kernel_size),
            norm_kind="channel",
        )
        if self.gate_mode == "learned":
            gate_hidden = max(16, int(hidden_size) // 4)
            self.gate = nn.Sequential(
                nn.Linear(7, gate_hidden),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(gate_hidden, 2),
            )
            self.fusion_head = None
        elif self.gate_mode == "fusion":
            self.gate = None
            fusion_hidden = max(16, int(hidden_size) // 4)
            self.fusion_head = nn.Sequential(
                nn.LayerNorm(10),
                nn.Linear(10, fusion_hidden),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(fusion_hidden, 1),
            )
            nn.init.zeros_(self.fusion_head[-1].weight)
            nn.init.zeros_(self.fusion_head[-1].bias)
        else:
            self.gate = None
            self.fusion_head = None

    def _feature_index(self, name: str) -> int | None:
        try:
            return self.feature_cols.index(name)
        except ValueError:
            return None

    def _window_gate_features(self, x: torch.Tensor) -> torch.Tensor:
        i_idx = self._feature_index("I_raw")
        v_idx = self._feature_index("V_corr_raw")
        t_idx = self._feature_index("T")
        if i_idx is None or v_idx is None or t_idx is None:
            raise RuntimeError("dual_context_endpoint requires V_corr_raw, I_raw, and T features.")
        i = x[:, :, i_idx:i_idx + 1]
        v = x[:, :, v_idx:v_idx + 1]
        temp = x[:, :, t_idx:t_idx + 1]
        abs_idx = self._feature_index("absI")
        abs_i = x[:, :, abs_idx:abs_idx + 1].abs() if abs_idx is not None else i.abs()
        di_idx = self._feature_index("dI")
        if di_idx is not None:
            d_i = x[:, :, di_idx:di_idx + 1]
        else:
            d_i = torch.diff(i, dim=1, prepend=i[:, :1, :])
        local = x[:, -min(self.local_len, x.shape[1]):, :]
        local_i = local[:, :, i_idx:i_idx + 1]
        local_v = local[:, :, v_idx:v_idx + 1]
        if abs_idx is not None:
            local_abs_i = local[:, :, abs_idx:abs_idx + 1].abs()
        else:
            local_abs_i = local_i.abs()
        if di_idx is not None:
            local_d_i = local[:, :, di_idx:di_idx + 1]
        else:
            local_d_i = torch.diff(local_i, dim=1, prepend=local_i[:, :1, :])
        return torch.cat(
            [
                abs_i.mean(dim=1),
                i.std(dim=1, unbiased=False),
                d_i.square().mean(dim=1),
                v.max(dim=1).values - v.min(dim=1).values,
                local_abs_i.mean(dim=1),
                local_d_i.square().mean(dim=1),
                local_v.max(dim=1).values - local_v.min(dim=1).values,
            ],
            dim=1,
        )

    def _dynamic_gate(self, x: torch.Tensor) -> torch.Tensor:
        i_idx = self._feature_index("I_raw")
        if i_idx is None:
            raise RuntimeError("dual-context dynamic gate requires I_raw feature.")
        current = x[:, :, i_idx:i_idx + 1]
        d_current = torch.diff(current, dim=1, prepend=current[:, :1, :])
        dynamic_score = d_current.abs().mean(dim=1)
        local_weight = torch.sigmoid(
            (float(self.dynamic_gate_center) - dynamic_score) * float(self.dynamic_gate_sharpness)
        )
        local_weight = float(self.dynamic_gate_local_min) + (
            float(self.dynamic_gate_local_max) - float(self.dynamic_gate_local_min)
        ) * local_weight
        return torch.cat([local_weight, 1.0 - local_weight], dim=1)

    def _branch_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.global_indices is None:
            x_global = x
        else:
            idx = torch.as_tensor(self.global_indices, device=x.device)
            x_global = x.index_select(dim=2, index=idx)
        h_global = self.global_model.encode_sequence(x_global)
        x_local = x[:, -min(self.local_len, x.shape[1]):, :]
        h_local = self.local_model.encode_sequence(x_local)
        global_logits = self.global_model.base_head(h_global[:, -1:, :])[:, 0, :]
        local_logits = self.local_model.base_head(h_local[:, -1:, :])[:, 0, :]
        if self.gate_mode == "learned":
            gate = torch.softmax(self.gate(self._window_gate_features(x)), dim=1)
        else:
            gate = self._dynamic_gate(x)
        return local_logits, global_logits, gate

    def _fusion_logits(self, x: torch.Tensor) -> torch.Tensor:
        local_logits, global_logits, _gate = self._branch_logits(x)
        stats = self._window_gate_features(x)
        fusion_in = torch.cat(
            [
                local_logits,
                global_logits,
                local_logits - global_logits,
                stats,
            ],
            dim=1,
        )
        base = 0.5 * (local_logits + global_logits)
        delta = float(self.fusion_delta_limit) * torch.tanh(self.fusion_head(fusion_in))
        return base + delta

    def branch_auxiliary_loss(self, x: torch.Tensor, y: torch.Tensor, beta: float) -> torch.Tensor:
        local_logits, global_logits, _gate = self._branch_logits(x)
        y_endpoint = y[:, -1, :] if y.ndim == 3 else y
        local_pred = torch.sigmoid(local_logits)
        global_pred = torch.sigmoid(global_logits)
        local_loss = F.smooth_l1_loss(local_pred, y_endpoint, beta=float(beta), reduction="none").mean(dim=1)
        global_loss = F.smooth_l1_loss(global_pred, y_endpoint, beta=float(beta), reduction="none").mean(dim=1)
        return 0.5 * (local_loss + global_loss)

    def dual_context_branch_parameters(self):
        yield from self.local_model.parameters()
        yield from self.global_model.parameters()

    def freeze_dual_context_branches(self) -> None:
        for param in self.dual_context_branch_parameters():
            param.requires_grad_(False)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if self.global_indices is None:
            x_global = x
        else:
            idx = torch.as_tensor(self.global_indices, device=x.device)
            x_global = x.index_select(dim=2, index=idx)
        return self.global_model.encode_sequence(x_global)

    def logits_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_mode == "fusion":
            endpoint = self._fusion_logits(x)
            return endpoint[:, None, :].expand(-1, x.shape[1], -1)
        local_logits, global_logits, gate = self._branch_logits(x)
        endpoint = gate[:, 0:1] * local_logits + gate[:, 1:2] * global_logits
        return endpoint[:, None, :].expand(-1, x.shape[1], -1)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits_sequence(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_mode == "fusion":
            return torch.sigmoid(self._fusion_logits(x))
        local_logits, global_logits, gate = self._branch_logits(x)
        logits = gate[:, 0:1] * local_logits + gate[:, 1:2] * global_logits
        return torch.sigmoid(logits)


class AnchorResidualTCNModel(nn.Module):
    """Voltage-anchor SOC estimator with a bounded dynamic residual.

    The anchor branch sees only causal voltage/temperature-derived channels at
    each time step. The TCN may use the full selected feature set, but it can
    only add a bounded SOC residual around the anchor prediction.
    """

    def __init__(
        self,
        input_dim: int,
        feature_cols: list[str],
        hidden_size: int,
        layers: int,
        kernel_size: int,
        dropout: float,
        residual_limit: float,
        residual_limit_init: str = "value",
        residual_limit_mode: str = "fixed",
        residual_limit_lower: float = 0.0,
        residual_limit_upper: float = 0.2,
        recurrent: str = "tcn",
    ):
        super().__init__()
        self.residual_limit_mode = str(residual_limit_mode)
        residual_init = torch.rand((), dtype=torch.float32).item() if str(residual_limit_init) == "rand01" else float(residual_limit)
        self.residual_limit = float(residual_init)
        self.residual_limit_initial_value = float(residual_init)
        self.residual_limit_init = str(residual_limit_init)
        self.residual_limit_lower = float(residual_limit_lower)
        self.residual_limit_upper = float(residual_limit_upper)
        if self.residual_limit_mode == "learnable":
            self.residual_limit_param = nn.Parameter(torch.tensor(float(residual_init), dtype=torch.float32))
        elif self.residual_limit_mode == "bounded_learnable":
            lower = float(residual_limit_lower)
            upper = float(residual_limit_upper)
            if not lower < upper:
                raise ValueError("anchor residual limit lower bound must be smaller than upper bound.")
            init = min(max(float(residual_init), lower + 1e-6), upper - 1e-6)
            p = (init - lower) / (upper - lower)
            self.residual_limit_param = nn.Parameter(torch.tensor(float(np.log(p / (1.0 - p))), dtype=torch.float32))
        elif self.residual_limit_mode != "fixed":
            raise ValueError(f"Unknown residual_limit_mode={residual_limit_mode!r}")
        self.dynamic = VcorrITGoalModel(
            input_dim=input_dim,
            hidden_size=int(hidden_size),
            recurrent=str(recurrent),
            layers=int(layers),
            head_kind="linear",
            temp_mode="none",
            dropout=float(dropout),
            kernel_size=int(kernel_size),
            norm_kind="channel",
        )
        preferred = [
            "V_corr_raw",
            "V_eq_slow_raw",
            "T",
            "V_corr_raw_ema50",
            "V_corr_raw_dev_ema50",
            "V_corr_raw_ema200",
            "V_corr_raw_dev_ema200",
            "V_corr_raw_ema800",
            "V_corr_raw_dev_ema800",
            "dV_corr",
            "abs_dV_corr",
            "Vcorr_x_absI",
        ]
        idx = [feature_cols.index(col) for col in preferred if col in feature_cols]
        if ("V_corr_raw" not in feature_cols and "V_eq_slow_raw" not in feature_cols) or "T" not in feature_cols:
            raise RuntimeError("AnchorResidualTCNModel requires V_corr_raw or V_eq_slow_raw, plus T features.")
        self.anchor_indices = idx
        anchor_dim = len(idx)
        self.anchor_head = nn.Sequential(
            nn.Linear(anchor_dim, int(hidden_size)),
            nn.LayerNorm(int(hidden_size)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), int(hidden_size) // 2),
            nn.SiLU(),
            nn.Linear(int(hidden_size) // 2, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(int(hidden_size), int(hidden_size) // 2),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size) // 2, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def residual_limit_tensor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.residual_limit_mode == "learnable":
            return self.residual_limit_param.to(device=device, dtype=dtype)
        if self.residual_limit_mode == "bounded_learnable":
            lower = torch.as_tensor(self.residual_limit_lower, device=device, dtype=dtype)
            upper = torch.as_tensor(self.residual_limit_upper, device=device, dtype=dtype)
            return lower + (upper - lower) * torch.sigmoid(self.residual_limit_param.to(device=device, dtype=dtype))
        return torch.as_tensor(self.residual_limit, device=device, dtype=dtype)

    def residual_limit_value(self) -> float:
        if self.residual_limit_mode == "learnable":
            return float(self.residual_limit_param.detach().cpu())
        if self.residual_limit_mode == "bounded_learnable":
            raw = float(self.residual_limit_param.detach().cpu())
            sig = 1.0 / (1.0 + float(np.exp(-raw)))
            return float(self.residual_limit_lower + (self.residual_limit_upper - self.residual_limit_lower) * sig)
        return float(self.residual_limit)

    def residual_limit_initial(self) -> float:
        return float(self.residual_limit_initial_value)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return self.dynamic.encode_sequence(x)

    def anchor_sequence(self, x: torch.Tensor) -> torch.Tensor:
        anchor_x = x.index_select(dim=2, index=torch.as_tensor(self.anchor_indices, device=x.device))
        return torch.sigmoid(self.anchor_head(anchor_x))

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        anchor = self.anchor_sequence(x)
        h = self.encode_sequence(x)
        limit = self.residual_limit_tensor(device=h.device, dtype=h.dtype)
        residual = limit * torch.tanh(self.residual_head(h))
        return (anchor + residual).clamp(0.0, 1.0)

    def residual_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode_sequence(x)
        limit = self.residual_limit_tensor(device=h.device, dtype=h.dtype)
        return limit * torch.tanh(self.residual_head(h))

    def logits_sequence(self, x: torch.Tensor) -> torch.Tensor:
        soc = self.forward_sequence(x).clamp(1e-5, 1.0 - 1e-5)
        return torch.logit(soc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1, :]


class TemperatureLimitAnchorResidualTCNModel(AnchorResidualTCNModel):
    """Anchor-residual sequence model with temperature-conditioned residual capacity."""

    def __init__(
        self,
        input_dim: int,
        feature_cols: list[str],
        hidden_size: int,
        layers: int,
        kernel_size: int,
        dropout: float,
        residual_limit: float,
        residual_limit_init: str = "value",
        residual_limit_mode: str = "fixed",
        residual_limit_lower: float = 0.0,
        residual_limit_upper: float = 0.2,
        recurrent: str = "lstm",
    ):
        super().__init__(
            input_dim=input_dim,
            feature_cols=feature_cols,
            hidden_size=hidden_size,
            layers=layers,
            kernel_size=kernel_size,
            dropout=dropout,
            residual_limit=residual_limit,
            residual_limit_init=residual_limit_init,
            residual_limit_mode=residual_limit_mode,
            residual_limit_lower=residual_limit_lower,
            residual_limit_upper=residual_limit_upper,
            recurrent=recurrent,
        )
        self.feature_cols = list(feature_cols)
        if "T" not in self.feature_cols:
            raise RuntimeError("TemperatureLimitAnchorResidualTCNModel requires T feature.")
        self.temp_idx = int(self.feature_cols.index("T"))
        init = torch.full((3,), float(self.residual_limit_initial_value), dtype=torch.float32)
        if self.residual_limit_mode == "learnable":
            self.residual_limit_param = nn.Parameter(init)
        elif self.residual_limit_mode == "bounded_learnable":
            lower = float(residual_limit_lower)
            upper = float(residual_limit_upper)
            if not lower < upper:
                raise ValueError("anchor residual limit lower bound must be smaller than upper bound.")
            clipped = init.clamp(lower + 1e-6, upper - 1e-6)
            p = (clipped - lower) / (upper - lower)
            self.residual_limit_param = nn.Parameter(torch.log(p / (1.0 - p)))

    def residual_limit_tensor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.residual_limit_mode == "learnable":
            return self.residual_limit_param.to(device=device, dtype=dtype)
        if self.residual_limit_mode == "bounded_learnable":
            lower = torch.as_tensor(self.residual_limit_lower, device=device, dtype=dtype)
            upper = torch.as_tensor(self.residual_limit_upper, device=device, dtype=dtype)
            return lower + (upper - lower) * torch.sigmoid(self.residual_limit_param.to(device=device, dtype=dtype))
        return torch.full((3,), float(self.residual_limit), device=device, dtype=dtype)

    def residual_limit_value(self) -> float:
        vals = self.residual_limit_tensor(device=torch.device("cpu"), dtype=torch.float32).detach().cpu().numpy()
        return float(np.mean(vals))

    def _temperature_gates(self, x: torch.Tensor) -> torch.Tensor:
        temp = x[:, :, self.temp_idx:self.temp_idx + 1]
        sharpness = 8.0
        cold = torch.sigmoid((-0.45 - temp) * sharpness)
        mid = torch.sigmoid((temp + 0.45) * sharpness) * torch.sigmoid((0.65 - temp) * sharpness)
        hot = torch.sigmoid((temp - 0.65) * sharpness)
        gates = torch.cat([cold, mid, hot], dim=2)
        return gates / gates.sum(dim=2, keepdim=True).clamp_min(1e-6)

    def residual_limit_sequence(self, x: torch.Tensor) -> torch.Tensor:
        limits = self.residual_limit_tensor(device=x.device, dtype=x.dtype).view(1, 1, 3)
        return (self._temperature_gates(x) * limits).sum(dim=2, keepdim=True)

    def residual_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode_sequence(x)
        return self.residual_limit_sequence(x) * torch.tanh(self.residual_head(h))

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        anchor = self.anchor_sequence(x)
        h = self.encode_sequence(x)
        residual = self.residual_limit_sequence(x) * torch.tanh(self.residual_head(h))
        return (anchor + residual).clamp(0.0, 1.0)


class ColdLowSOCPositiveCorrectorModel(AnchorResidualTCNModel):
    """Anchor-residual sequence model with a causal cold low-SOC positive corrector.

    The extra branch is deliberately one-sided: it can only add SOC when the
    learned anchor places the sequence in a cold, low-SOC regime. This targets
    an observed 0C low-SOC negative bias without giving the model a
    profile label.
    """

    def __init__(
        self,
        input_dim: int,
        feature_cols: list[str],
        hidden_size: int,
        layers: int,
        kernel_size: int,
        dropout: float,
        residual_limit: float,
        residual_limit_init: str = "value",
        residual_limit_mode: str = "fixed",
        residual_limit_lower: float = 0.0,
        residual_limit_upper: float = 0.2,
        recurrent: str = "gru",
        correction_limit: float = 0.08,
        soc_threshold: float = 0.35,
        temp_threshold: float = -0.45,
        gate_sharpness: float = 8.0,
        sag_gate: bool = False,
    ):
        super().__init__(
            input_dim=input_dim,
            feature_cols=feature_cols,
            hidden_size=hidden_size,
            layers=layers,
            kernel_size=kernel_size,
            dropout=dropout,
            residual_limit=residual_limit,
            residual_limit_init=residual_limit_init,
            residual_limit_mode=residual_limit_mode,
            residual_limit_lower=residual_limit_lower,
            residual_limit_upper=residual_limit_upper,
            recurrent=recurrent,
        )
        self.feature_cols = list(feature_cols)
        if "V_corr_raw" not in self.feature_cols or "I_raw" not in self.feature_cols or "T" not in self.feature_cols:
            raise RuntimeError("ColdLowSOCPositiveCorrectorModel requires V_corr_raw, I_raw, and T features.")
        self.vcorr_idx = int(self.feature_cols.index("V_corr_raw"))
        self.current_idx = int(self.feature_cols.index("I_raw"))
        self.temp_idx = int(self.feature_cols.index("T"))
        self.abs_i_idx = self.feature_cols.index("absI") if "absI" in self.feature_cols else None
        self.di_idx = self.feature_cols.index("dI") if "dI" in self.feature_cols else None
        self.correction_limit = float(correction_limit)
        self.cold_soc_threshold = float(soc_threshold)
        self.cold_temp_threshold = float(temp_threshold)
        self.cold_gate_sharpness = float(gate_sharpness)
        self.sag_gate_enabled = bool(sag_gate)
        regime_dim = 6
        self.corrector_norm = nn.LayerNorm(regime_dim)
        self.positive_corrector = nn.Sequential(
            nn.Linear(int(hidden_size) + regime_dim, int(hidden_size) // 2),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size) // 2, 1),
        )
        nn.init.zeros_(self.positive_corrector[-1].weight)
        nn.init.constant_(self.positive_corrector[-1].bias, -3.5)
        if self.sag_gate_enabled:
            gate_hidden = max(8, int(hidden_size) // 4)
            self.sag_gate_head = nn.Sequential(
                nn.Linear(regime_dim, gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, 1),
            )
            nn.init.zeros_(self.sag_gate_head[-1].weight)
            nn.init.zeros_(self.sag_gate_head[-1].bias)

    def _regime_sequence(self, x: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(1, int(x.shape[1]) + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        current = x[:, :, self.current_idx:self.current_idx + 1]
        abs_current = x[:, :, self.abs_i_idx:self.abs_i_idx + 1].abs() if self.abs_i_idx is not None else current.abs()
        abs_i_mean = abs_current.cumsum(dim=1) / steps
        i_mean = current.cumsum(dim=1) / steps
        i2_mean = current.square().cumsum(dim=1) / steps
        i_std = (i2_mean - i_mean.square()).clamp_min(0.0).sqrt()
        if self.di_idx is not None:
            di = x[:, :, self.di_idx:self.di_idx + 1]
        else:
            di = torch.diff(current, dim=1, prepend=current[:, :1, :])
        di_energy = di.square().cumsum(dim=1) / steps
        vcorr = x[:, :, self.vcorr_idx:self.vcorr_idx + 1]
        vcorr_span = torch.cummax(vcorr, dim=1).values - torch.cummin(vcorr, dim=1).values
        vcorr_end = vcorr
        vcorr_mean = vcorr.cumsum(dim=1) / steps
        voltage_sag = vcorr_mean - vcorr_end
        return torch.cat([abs_i_mean, i_std, di_energy, vcorr_span, voltage_sag, anchor], dim=2)

    def positive_correction_sequence(
        self,
        x: torch.Tensor,
        anchor: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if anchor is None:
            anchor = self.anchor_sequence(x)
        if h is None:
            h = self.encode_sequence(x)
        z = self.corrector_norm(self._regime_sequence(x, anchor))
        temp = x[:, :, self.temp_idx:self.temp_idx + 1]
        cold_gate = torch.sigmoid((float(self.cold_temp_threshold) - temp) * float(self.cold_gate_sharpness))
        low_soc_gate = torch.sigmoid((float(self.cold_soc_threshold) - anchor) * 20.0)
        raw = self.positive_corrector(torch.cat([h, z], dim=2))
        sag_gate = torch.sigmoid(self.sag_gate_head(z)) if self.sag_gate_enabled else 1.0
        correction = float(self.correction_limit) * cold_gate * low_soc_gate * sag_gate * torch.sigmoid(raw)
        return correction

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        anchor = self.anchor_sequence(x)
        h = self.encode_sequence(x)
        limit = self.residual_limit_tensor(device=h.device, dtype=h.dtype)
        residual = limit * torch.tanh(self.residual_head(h))
        base = (anchor + residual).clamp(0.0, 1.0)
        correction = self.positive_correction_sequence(x, anchor=anchor, h=h)
        return (base + correction).clamp(0.0, 1.0)


class ColdLowSOCSignedCorrectorModel(ColdLowSOCPositiveCorrectorModel):
    """Anchor-residual sequence model with a signed cold low-SOC regime corrector."""

    def __init__(self, *args, **kwargs):
        hidden_size = int(kwargs.get("hidden_size", args[2] if len(args) > 2 else 64))
        dropout = float(kwargs.get("dropout", args[5] if len(args) > 5 else 0.0))
        super().__init__(*args, **kwargs)
        regime_dim = int(self.corrector_norm.normalized_shape[0])
        self.signed_corrector = nn.Sequential(
            nn.Linear(hidden_size + regime_dim, max(8, hidden_size // 2)),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, hidden_size // 2), 1),
        )
        nn.init.zeros_(self.signed_corrector[-1].weight)
        nn.init.zeros_(self.signed_corrector[-1].bias)

    def signed_correction_sequence(
        self,
        x: torch.Tensor,
        gate_soc: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        z = self.corrector_norm(self._regime_sequence(x, gate_soc))
        temp = x[:, :, self.temp_idx:self.temp_idx + 1]
        cold_gate = torch.sigmoid((float(self.cold_temp_threshold) - temp) * float(self.cold_gate_sharpness))
        low_soc_gate = torch.sigmoid((float(self.cold_soc_threshold) - gate_soc) * 20.0)
        raw = self.signed_corrector(torch.cat([h, z], dim=2))
        sag_gate = torch.sigmoid(self.sag_gate_head(z)) if self.sag_gate_enabled else 1.0
        return float(self.correction_limit) * cold_gate * low_soc_gate * sag_gate * torch.tanh(raw)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        anchor = self.anchor_sequence(x)
        h = self.encode_sequence(x)
        limit = self.residual_limit_tensor(device=h.device, dtype=h.dtype)
        residual = limit * torch.tanh(self.residual_head(h))
        base = (anchor + residual).clamp(0.0, 1.0)
        correction = self.signed_correction_sequence(x, gate_soc=base, h=h)
        return (base + correction).clamp(0.0, 1.0)


class ColdLowSOCNormalPositiveCorrectorModel(nn.Module):
    """Single sequence regressor with a one-sided cold low-SOC corrector."""

    def __init__(
        self,
        input_dim: int,
        feature_cols: list[str],
        hidden_size: int,
        layers: int,
        kernel_size: int,
        dropout: float,
        recurrent: str = "gru",
        head_kind: str = "linear",
        temp_mode: str = "none",
        correction_limit: float = 0.08,
        soc_threshold: float = 0.35,
        temp_threshold: float = -0.45,
        gate_sharpness: float = 8.0,
        sag_gate: bool = False,
    ):
        super().__init__()
        self.feature_cols = list(feature_cols)
        if "V_corr_raw" not in self.feature_cols or "I_raw" not in self.feature_cols or "T" not in self.feature_cols:
            raise RuntimeError("ColdLowSOCNormalPositiveCorrectorModel requires V_corr_raw, I_raw, and T features.")
        self.vcorr_idx = int(self.feature_cols.index("V_corr_raw"))
        self.current_idx = int(self.feature_cols.index("I_raw"))
        self.temp_idx = int(self.feature_cols.index("T"))
        self.abs_i_idx = self.feature_cols.index("absI") if "absI" in self.feature_cols else None
        self.di_idx = self.feature_cols.index("dI") if "dI" in self.feature_cols else None
        self.correction_limit = float(correction_limit)
        self.cold_soc_threshold = float(soc_threshold)
        self.cold_temp_threshold = float(temp_threshold)
        self.cold_gate_sharpness = float(gate_sharpness)
        self.sag_gate_enabled = bool(sag_gate)
        self.base = VcorrITGoalModel(
            input_dim=input_dim,
            hidden_size=int(hidden_size),
            recurrent=str(recurrent),
            layers=int(layers),
            head_kind=str(head_kind),
            temp_mode=str(temp_mode),
            dropout=float(dropout),
            kernel_size=int(kernel_size),
            norm_kind="channel",
        )
        regime_dim = 6
        self.corrector_norm = nn.LayerNorm(regime_dim)
        self.positive_corrector = nn.Sequential(
            nn.Linear(int(hidden_size) + regime_dim, int(hidden_size) // 2),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size) // 2, 1),
        )
        nn.init.zeros_(self.positive_corrector[-1].weight)
        nn.init.constant_(self.positive_corrector[-1].bias, -3.5)
        if self.sag_gate_enabled:
            gate_hidden = max(8, int(hidden_size) // 4)
            self.sag_gate_head = nn.Sequential(
                nn.Linear(regime_dim, gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, 1),
            )
            nn.init.zeros_(self.sag_gate_head[-1].weight)
            nn.init.zeros_(self.sag_gate_head[-1].bias)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.encode_sequence(x)

    def base_parameters(self):
        yield from self.base.parameters()

    def freeze_base(self) -> None:
        for param in self.base_parameters():
            param.requires_grad_(False)

    def _regime_sequence(self, x: torch.Tensor, base_soc: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(1, int(x.shape[1]) + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        current = x[:, :, self.current_idx:self.current_idx + 1]
        abs_current = x[:, :, self.abs_i_idx:self.abs_i_idx + 1].abs() if self.abs_i_idx is not None else current.abs()
        abs_i_mean = abs_current.cumsum(dim=1) / steps
        i_mean = current.cumsum(dim=1) / steps
        i2_mean = current.square().cumsum(dim=1) / steps
        i_std = (i2_mean - i_mean.square()).clamp_min(0.0).sqrt()
        if self.di_idx is not None:
            di = x[:, :, self.di_idx:self.di_idx + 1]
        else:
            di = torch.diff(current, dim=1, prepend=current[:, :1, :])
        di_energy = di.square().cumsum(dim=1) / steps
        vcorr = x[:, :, self.vcorr_idx:self.vcorr_idx + 1]
        vcorr_span = torch.cummax(vcorr, dim=1).values - torch.cummin(vcorr, dim=1).values
        vcorr_end = vcorr
        vcorr_mean = vcorr.cumsum(dim=1) / steps
        voltage_sag = vcorr_mean - vcorr_end
        return torch.cat([abs_i_mean, i_std, di_energy, vcorr_span, voltage_sag, base_soc], dim=2)

    def _base_soc_from_hidden(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        logits = self.base.base_head(h)
        temp = x[..., self.base.temp_idx:self.base.temp_idx + 1]
        if self.base.temp_mode == "none":
            return torch.sigmoid(logits)
        if self.base.temp_mode == "bias":
            return torch.sigmoid(logits + self.base.temp_bias(temp))
        if self.base.temp_mode == "hard_affine":
            idx = self.base._hard_head_indices(temp)
            flat_idx = idx.reshape(-1)
            scale = torch.exp(self.base.temp_affine_log_scale.clamp(-1.0, 1.0)).to(logits.device, logits.dtype)
            bias = self.base.temp_affine_bias.to(logits.device, logits.dtype)
            selected_scale = scale.index_select(0, flat_idx).reshape_as(logits)
            selected_bias = bias.index_select(0, flat_idx).reshape_as(logits)
            return torch.sigmoid(selected_scale * logits + selected_bias)
        return self.base.forward_sequence(x)

    def positive_correction_sequence(
        self,
        x: torch.Tensor,
        base_soc: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if h is None:
            h = self.encode_sequence(x)
        if base_soc is None:
            base_soc = self._base_soc_from_hidden(x, h)
        z = self.corrector_norm(self._regime_sequence(x, base_soc))
        temp = x[:, :, self.temp_idx:self.temp_idx + 1]
        cold_gate = torch.sigmoid((float(self.cold_temp_threshold) - temp) * float(self.cold_gate_sharpness))
        low_soc_gate = torch.sigmoid((float(self.cold_soc_threshold) - base_soc) * 20.0)
        raw = self.positive_corrector(torch.cat([h, z], dim=2))
        sag_gate = torch.sigmoid(self.sag_gate_head(z)) if self.sag_gate_enabled else 1.0
        return float(self.correction_limit) * cold_gate * low_soc_gate * sag_gate * torch.sigmoid(raw)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode_sequence(x)
        base_soc = self._base_soc_from_hidden(x, h)
        correction = self.positive_correction_sequence(x, base_soc=base_soc, h=h)
        return (base_soc + correction).clamp(0.0, 1.0)

    def base_forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode_sequence(x)
        return self._base_soc_from_hidden(x, h)

    def base_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_forward_sequence(x)[:, -1, :]

    def logits_sequence(self, x: torch.Tensor) -> torch.Tensor:
        soc = self.forward_sequence(x).clamp(1e-5, 1.0 - 1e-5)
        return torch.logit(soc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1, :]


class ColdLowSOCNormalSignedCorrectorModel(ColdLowSOCNormalPositiveCorrectorModel):
    """Single sequence regressor with a signed cold low-SOC regime corrector."""

    def __init__(self, *args, **kwargs):
        hidden_size = int(kwargs.get("hidden_size", args[2] if len(args) > 2 else 64))
        dropout = float(kwargs.get("dropout", args[5] if len(args) > 5 else 0.0))
        super().__init__(*args, **kwargs)
        regime_dim = int(self.corrector_norm.normalized_shape[0])
        self.signed_corrector = nn.Sequential(
            nn.Linear(hidden_size + regime_dim, max(8, hidden_size // 2)),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, hidden_size // 2), 1),
        )
        nn.init.zeros_(self.signed_corrector[-1].weight)
        nn.init.zeros_(self.signed_corrector[-1].bias)

    def signed_correction_sequence(
        self,
        x: torch.Tensor,
        base_soc: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if h is None:
            h = self.encode_sequence(x)
        if base_soc is None:
            base_soc = self._base_soc_from_hidden(x, h)
        z = self.corrector_norm(self._regime_sequence(x, base_soc))
        temp = x[:, :, self.temp_idx:self.temp_idx + 1]
        cold_gate = torch.sigmoid((float(self.cold_temp_threshold) - temp) * float(self.cold_gate_sharpness))
        low_soc_gate = torch.sigmoid((float(self.cold_soc_threshold) - base_soc) * 20.0)
        raw = self.signed_corrector(torch.cat([h, z], dim=2))
        sag_gate = torch.sigmoid(self.sag_gate_head(z)) if self.sag_gate_enabled else 1.0
        return float(self.correction_limit) * cold_gate * low_soc_gate * sag_gate * torch.tanh(raw)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode_sequence(x)
        base_soc = self._base_soc_from_hidden(x, h)
        correction = self.signed_correction_sequence(x, base_soc=base_soc, h=h)
        return (base_soc + correction).clamp(0.0, 1.0)


class ColdHighDynamicNormalSignedCorrectorModel(ColdLowSOCNormalSignedCorrectorModel):
    """Single sequence regressor with a signed cold high-dynamic regime corrector."""

    def __init__(
        self,
        *args,
        dynamic_center: float = 0.15,
        dynamic_sharpness: float = 40.0,
        istd_center: float = -1.0,
        istd_sharpness: float = 20.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dynamic_center = float(dynamic_center)
        self.dynamic_sharpness = float(dynamic_sharpness)
        self.istd_center = float(istd_center)
        self.istd_sharpness = float(istd_sharpness)

    def _high_dynamic_gate(self, x: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(1, int(x.shape[1]) + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        current = x[:, :, self.current_idx:self.current_idx + 1]
        if self.di_idx is not None:
            di = x[:, :, self.di_idx:self.di_idx + 1]
        else:
            di = torch.diff(current, dim=1, prepend=current[:, :1, :])
        dynamic_score = di.abs().cumsum(dim=1) / steps
        dynamic_gate = torch.sigmoid((dynamic_score - float(self.dynamic_center)) * float(self.dynamic_sharpness))
        if float(self.istd_center) <= 0.0:
            return dynamic_gate
        i_mean = current.cumsum(dim=1) / steps
        i2_mean = current.square().cumsum(dim=1) / steps
        i_std = (i2_mean - i_mean.square()).clamp_min(0.0).sqrt()
        istd_gate = torch.sigmoid((i_std - float(self.istd_center)) * float(self.istd_sharpness))
        return torch.maximum(dynamic_gate, istd_gate)

    def signed_correction_sequence(
        self,
        x: torch.Tensor,
        base_soc: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if h is None:
            h = self.encode_sequence(x)
        if base_soc is None:
            base_soc = self._base_soc_from_hidden(x, h)
        z = self.corrector_norm(self._regime_sequence(x, base_soc))
        temp = x[:, :, self.temp_idx:self.temp_idx + 1]
        cold_gate = torch.sigmoid((float(self.cold_temp_threshold) - temp) * float(self.cold_gate_sharpness))
        dynamic_gate = self._high_dynamic_gate(x)
        raw = self.signed_corrector(torch.cat([h, z], dim=2))
        sag_gate = torch.sigmoid(self.sag_gate_head(z)) if self.sag_gate_enabled else 1.0
        return float(self.correction_limit) * cold_gate * dynamic_gate * sag_gate * torch.tanh(raw)


class ColdRegimeSignedBiasModel(nn.Module):
    """Single sequence regressor with a cold-only signed bias head from V/I/T regime statistics."""

    def __init__(
        self,
        input_dim: int,
        feature_cols: list[str],
        hidden_size: int,
        layers: int,
        kernel_size: int,
        dropout: float,
        recurrent: str = "gru",
        head_kind: str = "linear",
        temp_mode: str = "none",
        correction_limit: float = 0.012,
        temp_threshold: float = -0.45,
        gate_sharpness: float = 8.0,
    ):
        super().__init__()
        self.feature_cols = list(feature_cols)
        if "V_corr_raw" not in self.feature_cols or "I_raw" not in self.feature_cols or "T" not in self.feature_cols:
            raise RuntimeError("ColdRegimeSignedBiasModel requires V_corr_raw, I_raw, and T features.")
        self.vcorr_idx = int(self.feature_cols.index("V_corr_raw"))
        self.current_idx = int(self.feature_cols.index("I_raw"))
        self.temp_idx = int(self.feature_cols.index("T"))
        self.abs_i_idx = self.feature_cols.index("absI") if "absI" in self.feature_cols else None
        self.di_idx = self.feature_cols.index("dI") if "dI" in self.feature_cols else None
        self.correction_limit = float(correction_limit)
        self.cold_temp_threshold = float(temp_threshold)
        self.cold_gate_sharpness = float(gate_sharpness)
        self.base = VcorrITGoalModel(
            input_dim=input_dim,
            hidden_size=int(hidden_size),
            recurrent=str(recurrent),
            layers=int(layers),
            head_kind=str(head_kind),
            temp_mode=str(temp_mode),
            dropout=float(dropout),
            kernel_size=int(kernel_size),
            norm_kind="channel",
        )
        regime_dim = 6
        self.regime_norm = nn.LayerNorm(regime_dim)
        self.bias_head = nn.Sequential(
            nn.Linear(int(hidden_size) + regime_dim, max(8, int(hidden_size) // 2)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(max(8, int(hidden_size) // 2), 1),
        )
        nn.init.zeros_(self.bias_head[-1].weight)
        nn.init.zeros_(self.bias_head[-1].bias)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.encode_sequence(x)

    def _base_soc_from_hidden(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        logits = self.base.base_head(h)
        temp = x[..., self.base.temp_idx:self.base.temp_idx + 1]
        if self.base.temp_mode == "none":
            return torch.sigmoid(logits)
        if self.base.temp_mode == "bias":
            return torch.sigmoid(logits + self.base.temp_bias(temp))
        if self.base.temp_mode == "hard_affine":
            idx = self.base._hard_head_indices(temp)
            flat_idx = idx.reshape(-1)
            scale = torch.exp(self.base.temp_affine_log_scale.clamp(-1.0, 1.0)).to(logits.device, logits.dtype)
            bias = self.base.temp_affine_bias.to(logits.device, logits.dtype)
            selected_scale = scale.index_select(0, flat_idx).reshape_as(logits)
            selected_bias = bias.index_select(0, flat_idx).reshape_as(logits)
            return torch.sigmoid(selected_scale * logits + selected_bias)
        return self.base.forward_sequence(x)

    def _regime_sequence(self, x: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(1, int(x.shape[1]) + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        current = x[:, :, self.current_idx:self.current_idx + 1]
        abs_current = x[:, :, self.abs_i_idx:self.abs_i_idx + 1].abs() if self.abs_i_idx is not None else current.abs()
        abs_i_mean = abs_current.cumsum(dim=1) / steps
        i_mean = current.cumsum(dim=1) / steps
        i2_mean = current.square().cumsum(dim=1) / steps
        i_std = (i2_mean - i_mean.square()).clamp_min(0.0).sqrt()
        if self.di_idx is not None:
            di = x[:, :, self.di_idx:self.di_idx + 1]
        else:
            di = torch.diff(current, dim=1, prepend=current[:, :1, :])
        di_energy = di.square().cumsum(dim=1) / steps
        vcorr = x[:, :, self.vcorr_idx:self.vcorr_idx + 1]
        vcorr_span = torch.cummax(vcorr, dim=1).values - torch.cummin(vcorr, dim=1).values
        voltage_sag = vcorr.cumsum(dim=1) / steps - vcorr
        vcorr_end = vcorr
        return torch.cat([abs_i_mean, i_std, di_energy, vcorr_span, voltage_sag, vcorr_end], dim=2)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode_sequence(x)
        base_soc = self._base_soc_from_hidden(x, h)
        z = self.regime_norm(self._regime_sequence(x))
        temp = x[:, :, self.temp_idx:self.temp_idx + 1]
        cold_gate = torch.sigmoid((float(self.cold_temp_threshold) - temp) * float(self.cold_gate_sharpness))
        correction = float(self.correction_limit) * cold_gate * torch.tanh(self.bias_head(torch.cat([h, z], dim=2)))
        return (base_soc + correction).clamp(0.0, 1.0)

    def logits_sequence(self, x: torch.Tensor) -> torch.Tensor:
        soc = self.forward_sequence(x).clamp(1e-5, 1.0 - 1e-5)
        return torch.logit(soc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1, :]


class RegimeGatedNormalAnchorSignedModel(nn.Module):
    """Regime-gated blend of normal and anchor-residual sequence heads with cold signed correction."""

    def __init__(
        self,
        input_dim: int,
        feature_cols: list[str],
        hidden_size: int,
        layers: int,
        kernel_size: int,
        dropout: float,
        residual_limit: float,
        residual_limit_init: str = "value",
        residual_limit_mode: str = "fixed",
        residual_limit_lower: float = 0.0,
        residual_limit_upper: float = 0.2,
        recurrent: str = "gru",
        head_kind: str = "linear",
        temp_mode: str = "none",
        correction_limit: float = 0.08,
        soc_threshold: float = 0.35,
        temp_threshold: float = -0.45,
        gate_sharpness: float = 8.0,
    ):
        super().__init__()
        self.feature_cols = list(feature_cols)
        if "V_corr_raw" not in self.feature_cols or "I_raw" not in self.feature_cols or "T" not in self.feature_cols:
            raise RuntimeError("RegimeGatedNormalAnchorSignedModel requires V_corr_raw, I_raw, and T features.")
        self.vcorr_idx = int(self.feature_cols.index("V_corr_raw"))
        self.current_idx = int(self.feature_cols.index("I_raw"))
        self.temp_idx = int(self.feature_cols.index("T"))
        self.abs_i_idx = self.feature_cols.index("absI") if "absI" in self.feature_cols else None
        self.di_idx = self.feature_cols.index("dI") if "dI" in self.feature_cols else None
        self.correction_limit = float(correction_limit)
        self.cold_soc_threshold = float(soc_threshold)
        self.cold_temp_threshold = float(temp_threshold)
        self.cold_gate_sharpness = float(gate_sharpness)
        self.normal = VcorrITGoalModel(
            input_dim=input_dim,
            hidden_size=int(hidden_size),
            recurrent=str(recurrent),
            layers=int(layers),
            head_kind=str(head_kind),
            temp_mode=str(temp_mode),
            dropout=float(dropout),
            kernel_size=int(kernel_size),
            norm_kind="channel",
        )
        self.anchor_model = AnchorResidualTCNModel(
            input_dim=input_dim,
            feature_cols=list(feature_cols),
            hidden_size=int(hidden_size),
            layers=int(layers),
            kernel_size=int(kernel_size),
            dropout=float(dropout),
            residual_limit=float(residual_limit),
            residual_limit_init=str(residual_limit_init),
            residual_limit_mode=str(residual_limit_mode),
            residual_limit_lower=float(residual_limit_lower),
            residual_limit_upper=float(residual_limit_upper),
            recurrent=str(recurrent),
        )
        regime_dim = 6
        branch_dim = int(hidden_size) + regime_dim
        self.regime_norm = nn.LayerNorm(regime_dim)
        self.mix_gate = nn.Sequential(
            nn.Linear(branch_dim, max(8, int(hidden_size) // 2)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(max(8, int(hidden_size) // 2), 1),
        )
        self.signed_corrector = nn.Sequential(
            nn.Linear(branch_dim, max(8, int(hidden_size) // 2)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(max(8, int(hidden_size) // 2), 1),
        )
        nn.init.zeros_(self.mix_gate[-1].weight)
        nn.init.constant_(self.mix_gate[-1].bias, -2.0)
        nn.init.zeros_(self.signed_corrector[-1].weight)
        nn.init.zeros_(self.signed_corrector[-1].bias)

    def residual_limit_initial(self) -> float:
        return self.anchor_model.residual_limit_initial()

    def residual_limit_value(self) -> float:
        return self.anchor_model.residual_limit_value()

    def anchor_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return self.anchor_model.anchor_sequence(x)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return self.normal.encode_sequence(x)

    def _normal_soc_from_hidden(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        logits = self.normal.base_head(h)
        temp = x[..., self.normal.temp_idx:self.normal.temp_idx + 1]
        if self.normal.temp_mode == "none":
            return torch.sigmoid(logits)
        if self.normal.temp_mode == "bias":
            return torch.sigmoid(logits + self.normal.temp_bias(temp))
        if self.normal.temp_mode == "hard_affine":
            idx = self.normal._hard_head_indices(temp)
            flat_idx = idx.reshape(-1)
            scale = torch.exp(self.normal.temp_affine_log_scale.clamp(-1.0, 1.0)).to(logits.device, logits.dtype)
            bias = self.normal.temp_affine_bias.to(logits.device, logits.dtype)
            selected_scale = scale.index_select(0, flat_idx).reshape_as(logits)
            selected_bias = bias.index_select(0, flat_idx).reshape_as(logits)
            return torch.sigmoid(selected_scale * logits + selected_bias)
        return self.normal.forward_sequence(x)

    def _regime_sequence(self, x: torch.Tensor, base_soc: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(1, int(x.shape[1]) + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        current = x[:, :, self.current_idx:self.current_idx + 1]
        abs_current = x[:, :, self.abs_i_idx:self.abs_i_idx + 1].abs() if self.abs_i_idx is not None else current.abs()
        abs_i_mean = abs_current.cumsum(dim=1) / steps
        i_mean = current.cumsum(dim=1) / steps
        i2_mean = current.square().cumsum(dim=1) / steps
        i_std = (i2_mean - i_mean.square()).clamp_min(0.0).sqrt()
        if self.di_idx is not None:
            di = x[:, :, self.di_idx:self.di_idx + 1]
        else:
            di = torch.diff(current, dim=1, prepend=current[:, :1, :])
        di_energy = di.square().cumsum(dim=1) / steps
        vcorr = x[:, :, self.vcorr_idx:self.vcorr_idx + 1]
        vcorr_span = torch.cummax(vcorr, dim=1).values - torch.cummin(vcorr, dim=1).values
        voltage_sag = vcorr.cumsum(dim=1) / steps - vcorr
        return torch.cat([abs_i_mean, i_std, di_energy, vcorr_span, voltage_sag, base_soc], dim=2)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.normal.encode_sequence(x)
        normal_soc = self._normal_soc_from_hidden(x, h)
        anchor_soc = self.anchor_model.forward_sequence(x)
        z = self.regime_norm(self._regime_sequence(x, normal_soc))
        branch_input = torch.cat([h, z], dim=2)
        mix = torch.sigmoid(self.mix_gate(branch_input))
        base = ((1.0 - mix) * normal_soc + mix * anchor_soc).clamp(0.0, 1.0)
        temp = x[:, :, self.temp_idx:self.temp_idx + 1]
        cold_gate = torch.sigmoid((float(self.cold_temp_threshold) - temp) * float(self.cold_gate_sharpness))
        low_soc_gate = torch.sigmoid((float(self.cold_soc_threshold) - base) * 20.0)
        correction = float(self.correction_limit) * cold_gate * low_soc_gate * torch.tanh(self.signed_corrector(branch_input))
        return (base + correction).clamp(0.0, 1.0)

    def logits_sequence(self, x: torch.Tensor) -> torch.Tensor:
        soc = self.forward_sequence(x).clamp(1e-5, 1.0 - 1e-5)
        return torch.logit(soc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1, :]


class NormalDynamicCorrectionModel(nn.Module):
    """Normal sequence regressor with a small dynamic-only signed correction."""

    def __init__(
        self,
        input_dim: int,
        feature_cols: list[str],
        hidden_size: int,
        layers: int,
        kernel_size: int,
        dropout: float,
        residual_limit: float,
        recurrent: str = "gru",
        head_kind: str = "linear",
        temp_mode: str = "none",
    ):
        super().__init__()
        self.feature_cols = list(feature_cols)
        if "I_raw" not in self.feature_cols or "V_corr_raw" not in self.feature_cols or "T" not in self.feature_cols:
            raise RuntimeError("NormalDynamicCorrectionModel requires V_corr_raw, I_raw, and T features.")
        self.vcorr_idx = int(self.feature_cols.index("V_corr_raw"))
        self.current_idx = int(self.feature_cols.index("I_raw"))
        self.temp_idx = int(self.feature_cols.index("T"))
        self.abs_i_idx = self.feature_cols.index("absI") if "absI" in self.feature_cols else None
        self.di_idx = self.feature_cols.index("dI") if "dI" in self.feature_cols else None
        self.residual_limit = float(residual_limit)
        self.base = VcorrITGoalModel(
            input_dim=input_dim,
            hidden_size=int(hidden_size),
            recurrent=str(recurrent),
            layers=int(layers),
            head_kind=str(head_kind),
            temp_mode=str(temp_mode),
            dropout=float(dropout),
            kernel_size=int(kernel_size),
            norm_kind="channel",
        )
        dynamic_cols = [
            "I_raw",
            "I_raw_ema50",
            "I_raw_dev_ema50",
            "I_raw_ema200",
            "I_raw_dev_ema200",
            "absI",
            "absI_ema50",
            "absI_dev_ema50",
            "absI_ema200",
            "absI_dev_ema200",
            "V_corr_raw_dev_ema50",
            "V_corr_raw_dev_ema200",
            "V_corr_raw_dev_ema800",
            "dV_corr",
            "abs_dV_corr",
            "dI",
            "dI_abs_mean",
        ]
        self.dynamic_indices = [self.feature_cols.index(col) for col in dynamic_cols if col in self.feature_cols]
        if not self.dynamic_indices:
            raise RuntimeError("NormalDynamicCorrectionModel found no dynamic correction input features.")
        self.dynamic = VcorrITGoalModel(
            input_dim=len(self.dynamic_indices),
            hidden_size=int(hidden_size),
            recurrent=str(recurrent),
            layers=int(layers),
            head_kind="linear",
            temp_mode="none",
            dropout=float(dropout),
            kernel_size=int(kernel_size),
            norm_kind="channel",
        )
        regime_dim = 5
        branch_dim = int(hidden_size) + regime_dim
        self.regime_norm = nn.LayerNorm(regime_dim)
        self.correction_head = nn.Sequential(
            nn.Linear(branch_dim, max(8, int(hidden_size) // 2)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(max(8, int(hidden_size) // 2), 1),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(branch_dim, max(8, int(hidden_size) // 4)),
            nn.SiLU(),
            nn.Linear(max(8, int(hidden_size) // 4), 1),
        )
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, -2.0)

    def residual_limit_initial(self) -> float:
        return float(self.residual_limit)

    def residual_limit_value(self) -> float:
        return float(self.residual_limit)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.encode_sequence(x)

    def _base_soc_from_hidden(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        logits = self.base.base_head(h)
        temp = x[..., self.base.temp_idx:self.base.temp_idx + 1]
        if self.base.temp_mode == "none":
            return torch.sigmoid(logits)
        if self.base.temp_mode == "bias":
            return torch.sigmoid(logits + self.base.temp_bias(temp))
        if self.base.temp_mode == "hard_affine":
            idx = self.base._hard_head_indices(temp)
            flat_idx = idx.reshape(-1)
            scale = torch.exp(self.base.temp_affine_log_scale.clamp(-1.0, 1.0)).to(logits.device, logits.dtype)
            bias = self.base.temp_affine_bias.to(logits.device, logits.dtype)
            selected_scale = scale.index_select(0, flat_idx).reshape_as(logits)
            selected_bias = bias.index_select(0, flat_idx).reshape_as(logits)
            return torch.sigmoid(selected_scale * logits + selected_bias)
        return self.base.forward_sequence(x)

    def _dynamic_input(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(dim=2, index=torch.as_tensor(self.dynamic_indices, device=x.device))

    def _regime_sequence(self, x: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(1, int(x.shape[1]) + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        current = x[:, :, self.current_idx:self.current_idx + 1]
        abs_current = x[:, :, self.abs_i_idx:self.abs_i_idx + 1].abs() if self.abs_i_idx is not None else current.abs()
        abs_i_mean = abs_current.cumsum(dim=1) / steps
        i_mean = current.cumsum(dim=1) / steps
        i2_mean = current.square().cumsum(dim=1) / steps
        i_std = (i2_mean - i_mean.square()).clamp_min(0.0).sqrt()
        if self.di_idx is not None:
            di = x[:, :, self.di_idx:self.di_idx + 1]
        else:
            di = torch.diff(current, dim=1, prepend=current[:, :1, :])
        di_energy = di.square().cumsum(dim=1) / steps
        vcorr = x[:, :, self.vcorr_idx:self.vcorr_idx + 1]
        vcorr_span = torch.cummax(vcorr, dim=1).values - torch.cummin(vcorr, dim=1).values
        return torch.cat([abs_i_mean, i_std, di_energy, vcorr_span, vcorr], dim=2)

    def residual_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h_dyn = self.dynamic.encode_sequence(self._dynamic_input(x))
        z = self.regime_norm(self._regime_sequence(x))
        branch_input = torch.cat([h_dyn, z], dim=2)
        gate = torch.sigmoid(self.gate_head(branch_input))
        return float(self.residual_limit) * gate * torch.tanh(self.correction_head(branch_input))

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h_base = self.base.encode_sequence(x)
        base_soc = self._base_soc_from_hidden(x, h_base)
        return (base_soc + self.residual_sequence(x)).clamp(0.0, 1.0)

    def logits_sequence(self, x: torch.Tensor) -> torch.Tensor:
        soc = self.forward_sequence(x).clamp(1e-5, 1.0 - 1e-5)
        return torch.logit(soc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1, :]


class TemperatureBudgetAnchorResidualModel(AnchorResidualTCNModel):
    """Anchor-residual head with fixed cold/mid/hot residual budgets."""

    def __init__(self, *args, cold_limit: float = 0.04, mid_limit: float = 0.10, hot_limit: float = 0.04, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_cols = list(kwargs.get("feature_cols", args[1] if len(args) > 1 else []))
        if "T" not in self.feature_cols:
            raise RuntimeError("TemperatureBudgetAnchorResidualModel requires T feature.")
        self.temp_idx = int(self.feature_cols.index("T"))
        self.temp_budget_limits = (float(cold_limit), float(mid_limit), float(hot_limit))

    def residual_limit_initial(self) -> float:
        return float(np.mean(self.temp_budget_limits))

    def residual_limit_value(self) -> float:
        return float(np.mean(self.temp_budget_limits))

    def _temperature_gates(self, x: torch.Tensor) -> torch.Tensor:
        temp = x[:, :, self.temp_idx:self.temp_idx + 1]
        sharpness = 8.0
        cold = torch.sigmoid((-0.45 - temp) * sharpness)
        mid = torch.sigmoid((temp + 0.45) * sharpness) * torch.sigmoid((0.65 - temp) * sharpness)
        hot = torch.sigmoid((temp - 0.65) * sharpness)
        gates = torch.cat([cold, mid, hot], dim=2)
        return gates / gates.sum(dim=2, keepdim=True).clamp_min(1e-6)

    def residual_limit_sequence(self, x: torch.Tensor) -> torch.Tensor:
        limits = x.new_tensor(self.temp_budget_limits).view(1, 1, 3)
        return (self._temperature_gates(x) * limits).sum(dim=2, keepdim=True)

    def residual_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode_sequence(x)
        return self.residual_limit_sequence(x) * torch.tanh(self.residual_head(h))

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return (self.anchor_sequence(x) + self.residual_sequence(x)).clamp(0.0, 1.0)


class NormalLowVoltageTailCorrectionModel(ColdLowSOCNormalPositiveCorrectorModel):
    """Normal sequence regressor with a negative-only learned tail correction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        regime_dim = int(self.corrector_norm.normalized_shape[0])
        hidden_size = int(kwargs.get("hidden_size", args[2] if len(args) > 2 else 64))
        dropout = float(kwargs.get("dropout", args[5] if len(args) > 5 else 0.0))
        self.tail_gate_head = nn.Sequential(
            nn.Linear(hidden_size + regime_dim, max(8, hidden_size // 2)),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, hidden_size // 2), 1),
        )
        self.tail_magnitude_head = nn.Sequential(
            nn.Linear(hidden_size + regime_dim, max(8, hidden_size // 2)),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, hidden_size // 2), 1),
        )
        nn.init.zeros_(self.tail_gate_head[-1].weight)
        nn.init.constant_(self.tail_gate_head[-1].bias, -2.0)
        nn.init.zeros_(self.tail_magnitude_head[-1].weight)
        nn.init.constant_(self.tail_magnitude_head[-1].bias, -3.5)

    def tail_correction_sequence(
        self,
        x: torch.Tensor,
        base_soc: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if h is None:
            h = self.encode_sequence(x)
        if base_soc is None:
            base_soc = self._base_soc_from_hidden(x, h)
        z = self.corrector_norm(self._regime_sequence(x, base_soc))
        low_soc_gate = torch.sigmoid((float(self.cold_soc_threshold) - base_soc) * 20.0)
        branch_input = torch.cat([h, z], dim=2)
        learned_gate = torch.sigmoid(self.tail_gate_head(branch_input))
        magnitude = torch.sigmoid(self.tail_magnitude_head(branch_input))
        return -float(self.correction_limit) * low_soc_gate * learned_gate * magnitude

    def residual_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode_sequence(x)
        base_soc = self._base_soc_from_hidden(x, h)
        return self.tail_correction_sequence(x, base_soc=base_soc, h=h)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode_sequence(x)
        base_soc = self._base_soc_from_hidden(x, h)
        return (base_soc + self.tail_correction_sequence(x, base_soc=base_soc, h=h)).clamp(0.0, 1.0)


class NormalAffineCalibrationModel(nn.Module):
    """Normal sequence regressor with a bounded regime-conditioned affine output calibration."""

    def __init__(
        self,
        input_dim: int,
        feature_cols: list[str],
        hidden_size: int,
        layers: int,
        kernel_size: int,
        dropout: float,
        residual_limit: float,
        recurrent: str = "gru",
        head_kind: str = "linear",
        temp_mode: str = "none",
    ):
        super().__init__()
        self.feature_cols = list(feature_cols)
        if "I_raw" not in self.feature_cols or "V_corr_raw" not in self.feature_cols or "T" not in self.feature_cols:
            raise RuntimeError("NormalAffineCalibrationModel requires V_corr_raw, I_raw, and T features.")
        self.vcorr_idx = int(self.feature_cols.index("V_corr_raw"))
        self.current_idx = int(self.feature_cols.index("I_raw"))
        self.temp_idx = int(self.feature_cols.index("T"))
        self.abs_i_idx = self.feature_cols.index("absI") if "absI" in self.feature_cols else None
        self.di_idx = self.feature_cols.index("dI") if "dI" in self.feature_cols else None
        self.residual_limit = float(residual_limit)
        self.base = VcorrITGoalModel(
            input_dim=input_dim,
            hidden_size=int(hidden_size),
            recurrent=str(recurrent),
            layers=int(layers),
            head_kind=str(head_kind),
            temp_mode=str(temp_mode),
            dropout=float(dropout),
            kernel_size=int(kernel_size),
            norm_kind="channel",
        )
        regime_dim = 6
        branch_dim = int(hidden_size) + regime_dim
        self.regime_norm = nn.LayerNorm(regime_dim)
        self.scale_head = nn.Sequential(
            nn.Linear(branch_dim, max(8, int(hidden_size) // 2)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(max(8, int(hidden_size) // 2), 1),
        )
        self.bias_head = nn.Sequential(
            nn.Linear(branch_dim, max(8, int(hidden_size) // 2)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(max(8, int(hidden_size) // 2), 1),
        )
        nn.init.zeros_(self.scale_head[-1].weight)
        nn.init.zeros_(self.scale_head[-1].bias)
        nn.init.zeros_(self.bias_head[-1].weight)
        nn.init.zeros_(self.bias_head[-1].bias)

    def residual_limit_initial(self) -> float:
        return float(self.residual_limit)

    def residual_limit_value(self) -> float:
        return float(self.residual_limit)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.encode_sequence(x)

    def _base_soc_from_hidden(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        logits = self.base.base_head(h)
        temp = x[..., self.base.temp_idx:self.base.temp_idx + 1]
        if self.base.temp_mode == "none":
            return torch.sigmoid(logits)
        if self.base.temp_mode == "bias":
            return torch.sigmoid(logits + self.base.temp_bias(temp))
        if self.base.temp_mode == "hard_affine":
            idx = self.base._hard_head_indices(temp)
            flat_idx = idx.reshape(-1)
            scale = torch.exp(self.base.temp_affine_log_scale.clamp(-1.0, 1.0)).to(logits.device, logits.dtype)
            bias = self.base.temp_affine_bias.to(logits.device, logits.dtype)
            selected_scale = scale.index_select(0, flat_idx).reshape_as(logits)
            selected_bias = bias.index_select(0, flat_idx).reshape_as(logits)
            return torch.sigmoid(selected_scale * logits + selected_bias)
        return self.base.forward_sequence(x)

    def _regime_sequence(self, x: torch.Tensor, base_soc: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(1, int(x.shape[1]) + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        current = x[:, :, self.current_idx:self.current_idx + 1]
        abs_current = x[:, :, self.abs_i_idx:self.abs_i_idx + 1].abs() if self.abs_i_idx is not None else current.abs()
        abs_i_mean = abs_current.cumsum(dim=1) / steps
        i_mean = current.cumsum(dim=1) / steps
        i2_mean = current.square().cumsum(dim=1) / steps
        i_std = (i2_mean - i_mean.square()).clamp_min(0.0).sqrt()
        if self.di_idx is not None:
            di = x[:, :, self.di_idx:self.di_idx + 1]
        else:
            di = torch.diff(current, dim=1, prepend=current[:, :1, :])
        di_energy = di.square().cumsum(dim=1) / steps
        vcorr = x[:, :, self.vcorr_idx:self.vcorr_idx + 1]
        vcorr_span = torch.cummax(vcorr, dim=1).values - torch.cummin(vcorr, dim=1).values
        voltage_sag = vcorr.cumsum(dim=1) / steps - vcorr
        return torch.cat([abs_i_mean, i_std, di_energy, vcorr_span, voltage_sag, base_soc], dim=2)

    def residual_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.base.encode_sequence(x)
        base_soc = self._base_soc_from_hidden(x, h)
        z = self.regime_norm(self._regime_sequence(x, base_soc))
        branch_input = torch.cat([h, z], dim=2)
        scale = 1.0 + float(self.residual_limit) * torch.tanh(self.scale_head(branch_input))
        bias = float(self.residual_limit) * torch.tanh(self.bias_head(branch_input))
        calibrated = (scale * base_soc + bias).clamp(0.0, 1.0)
        return calibrated - base_soc

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.base.encode_sequence(x)
        base_soc = self._base_soc_from_hidden(x, h)
        z = self.regime_norm(self._regime_sequence(x, base_soc))
        branch_input = torch.cat([h, z], dim=2)
        scale = 1.0 + float(self.residual_limit) * torch.tanh(self.scale_head(branch_input))
        bias = float(self.residual_limit) * torch.tanh(self.bias_head(branch_input))
        return (scale * base_soc + bias).clamp(0.0, 1.0)

    def logits_sequence(self, x: torch.Tensor) -> torch.Tensor:
        soc = self.forward_sequence(x).clamp(1e-5, 1.0 - 1e-5)
        return torch.logit(soc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1, :]


class VTSequenceAdditiveMLPResidualModel(AnchorResidualTCNModel):
    """Anchor-residual sequence model with an additive V/T-only window MLP."""

    def __init__(
        self,
        input_dim: int,
        feature_cols: list[str],
        hidden_size: int,
        layers: int,
        kernel_size: int,
        dropout: float,
        residual_limit: float,
        residual_limit_init: str = "value",
        residual_limit_mode: str = "fixed",
        residual_limit_lower: float = 0.0,
        residual_limit_upper: float = 0.2,
        recurrent: str = "gru",
        correction_limit: float = 0.12,
    ):
        super().__init__(
            input_dim=input_dim,
            feature_cols=feature_cols,
            hidden_size=hidden_size,
            layers=layers,
            kernel_size=kernel_size,
            dropout=dropout,
            residual_limit=residual_limit,
            residual_limit_init=residual_limit_init,
            residual_limit_mode=residual_limit_mode,
            residual_limit_lower=residual_limit_lower,
            residual_limit_upper=residual_limit_upper,
            recurrent=recurrent,
        )
        self.feature_cols = list(feature_cols)
        vt_preferred = [
            "V_corr_raw",
            "T",
            "V_corr_raw_ema50",
            "V_corr_raw_dev_ema50",
            "V_corr_raw_ema200",
            "V_corr_raw_dev_ema200",
            "V_corr_raw_ema800",
            "V_corr_raw_dev_ema800",
            "dV_corr",
            "abs_dV_corr",
        ]
        self.vt_indices = [self.feature_cols.index(col) for col in vt_preferred if col in self.feature_cols]
        if "V_corr_raw" not in self.feature_cols or "T" not in self.feature_cols or not self.vt_indices:
            raise RuntimeError("VTSequenceAdditiveMLPResidualModel requires V_corr_raw/T voltage features.")
        self.vt_correction_limit = float(correction_limit)
        vt_dim = len(self.vt_indices)
        summary_dim = vt_dim * 4
        self.vt_summary_norm = nn.LayerNorm(summary_dim)
        self.vt_additive_head = nn.Sequential(
            nn.Linear(summary_dim, int(hidden_size)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), int(hidden_size) // 2),
            nn.SiLU(),
            nn.Linear(int(hidden_size) // 2, 1),
        )
        nn.init.zeros_(self.vt_additive_head[-1].weight)
        nn.init.zeros_(self.vt_additive_head[-1].bias)

    def _vt_window_summary(self, x: torch.Tensor) -> torch.Tensor:
        vt = x.index_select(dim=2, index=torch.as_tensor(self.vt_indices, device=x.device))
        vt_end = vt[:, -1, :]
        vt_mean = vt.mean(dim=1)
        vt_std = vt.std(dim=1, unbiased=False)
        vt_delta = vt_end - vt[:, 0, :]
        return self.vt_summary_norm(torch.cat([vt_end, vt_mean, vt_std, vt_delta], dim=1))

    def vt_additive_correction(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.vt_additive_head(self._vt_window_summary(x))
        return float(self.vt_correction_limit) * torch.tanh(raw)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        base = super().forward_sequence(x)
        correction = self.vt_additive_correction(x)[:, None, :]
        return (base + correction).clamp(0.0, 1.0)


class VTInputAdditiveMLPResidualModel(AnchorResidualTCNModel):
    """Anchor-residual model with a V/T MLP that edits voltage input channels."""

    def __init__(
        self,
        input_dim: int,
        feature_cols: list[str],
        hidden_size: int,
        layers: int,
        kernel_size: int,
        dropout: float,
        residual_limit: float,
        residual_limit_init: str = "value",
        residual_limit_mode: str = "fixed",
        residual_limit_lower: float = 0.0,
        residual_limit_upper: float = 0.2,
        recurrent: str = "gru",
        correction_limit: float = 0.15,
    ):
        super().__init__(
            input_dim=input_dim,
            feature_cols=feature_cols,
            hidden_size=hidden_size,
            layers=layers,
            kernel_size=kernel_size,
            dropout=dropout,
            residual_limit=residual_limit,
            residual_limit_init=residual_limit_init,
            residual_limit_mode=residual_limit_mode,
            residual_limit_lower=residual_limit_lower,
            residual_limit_upper=residual_limit_upper,
            recurrent=recurrent,
        )
        self.feature_cols = list(feature_cols)
        vt_input_cols = [
            "V_corr_raw",
            "T",
            "V_corr_raw_ema50",
            "V_corr_raw_dev_ema50",
            "V_corr_raw_ema200",
            "V_corr_raw_dev_ema200",
            "V_corr_raw_ema800",
            "V_corr_raw_dev_ema800",
            "dV_corr",
            "abs_dV_corr",
        ]
        voltage_edit_cols = [
            "V_corr_raw",
            "V_corr_raw_ema50",
            "V_corr_raw_dev_ema50",
            "V_corr_raw_ema200",
            "V_corr_raw_dev_ema200",
            "V_corr_raw_ema800",
            "V_corr_raw_dev_ema800",
        ]
        self.vt_input_indices = [self.feature_cols.index(col) for col in vt_input_cols if col in self.feature_cols]
        self.voltage_edit_indices = [self.feature_cols.index(col) for col in voltage_edit_cols if col in self.feature_cols]
        if "V_corr_raw" not in self.feature_cols or "T" not in self.feature_cols or not self.voltage_edit_indices:
            raise RuntimeError("VTInputAdditiveMLPResidualModel requires V_corr_raw/T voltage features.")
        self.input_correction_limit = float(correction_limit)
        vt_dim = len(self.vt_input_indices)
        self.input_vt_mlp = nn.Sequential(
            nn.Linear(vt_dim, int(hidden_size) // 2),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size) // 2, 1),
        )
        nn.init.zeros_(self.input_vt_mlp[-1].weight)
        nn.init.zeros_(self.input_vt_mlp[-1].bias)

    def corrected_input_sequence(self, x: torch.Tensor) -> torch.Tensor:
        vt = x.index_select(dim=2, index=torch.as_tensor(self.vt_input_indices, device=x.device))
        delta = float(self.input_correction_limit) * torch.tanh(self.input_vt_mlp(vt))
        x_corr = x.clone()
        edit_idx = torch.as_tensor(self.voltage_edit_indices, device=x.device)
        x_corr[:, :, edit_idx] = x_corr.index_select(dim=2, index=edit_idx) + delta
        return x_corr

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return AnchorResidualTCNModel.encode_sequence(self, self.corrected_input_sequence(x))

    def anchor_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return AnchorResidualTCNModel.anchor_sequence(self, self.corrected_input_sequence(x))

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        x_corr = self.corrected_input_sequence(x)
        anchor = AnchorResidualTCNModel.anchor_sequence(self, x_corr)
        h = AnchorResidualTCNModel.encode_sequence(self, x_corr)
        limit = self.residual_limit_tensor(device=h.device, dtype=h.dtype)
        residual = limit * torch.tanh(self.residual_head(h))
        return (anchor + residual).clamp(0.0, 1.0)

    def logits_sequence(self, x: torch.Tensor) -> torch.Tensor:
        soc = self.forward_sequence(x).clamp(1e-5, 1.0 - 1e-5)
        return torch.logit(soc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1, :]


class DynamicOnlyAnchorResidualModel(nn.Module):
    """Voltage anchor with a residual branch restricted to dynamic excitation features."""

    def __init__(
        self,
        input_dim: int,
        feature_cols: list[str],
        hidden_size: int,
        layers: int,
        kernel_size: int,
        dropout: float,
        residual_limit: float,
        residual_limit_init: str = "value",
        residual_limit_mode: str = "fixed",
        residual_limit_lower: float = 0.0,
        residual_limit_upper: float = 0.2,
        recurrent: str = "lstm",
        gated: bool = False,
    ):
        super().__init__()
        self.feature_cols = list(feature_cols)
        self.gated = bool(gated)
        self.residual_limit_mode = str(residual_limit_mode)
        residual_init = torch.rand((), dtype=torch.float32).item() if str(residual_limit_init) == "rand01" else float(residual_limit)
        self.residual_limit = float(residual_init)
        self.residual_limit_initial_value = float(residual_init)
        self.residual_limit_init = str(residual_limit_init)
        self.residual_limit_lower = float(residual_limit_lower)
        self.residual_limit_upper = float(residual_limit_upper)
        if self.residual_limit_mode == "learnable":
            self.residual_limit_param = nn.Parameter(torch.tensor(float(residual_init), dtype=torch.float32))
        elif self.residual_limit_mode == "bounded_learnable":
            lower = float(residual_limit_lower)
            upper = float(residual_limit_upper)
            if not lower < upper:
                raise ValueError("anchor residual limit lower bound must be smaller than upper bound.")
            init = min(max(float(residual_init), lower + 1e-6), upper - 1e-6)
            p = (init - lower) / (upper - lower)
            self.residual_limit_param = nn.Parameter(torch.tensor(float(np.log(p / (1.0 - p))), dtype=torch.float32))
        elif self.residual_limit_mode != "fixed":
            raise ValueError(f"Unknown residual_limit_mode={residual_limit_mode!r}")

        anchor_preferred = [
            "V_corr_raw",
            "V_eq_slow_raw",
            "T",
            "V_corr_raw_ema50",
            "V_corr_raw_ema200",
            "V_corr_raw_ema800",
        ]
        dynamic_preferred = [
            "I_raw",
            "I_raw_ema50",
            "I_raw_dev_ema50",
            "I_raw_ema200",
            "I_raw_dev_ema200",
            "absI",
            "absI_ema50",
            "absI_dev_ema50",
            "absI_ema200",
            "absI_dev_ema200",
            "V_corr_raw_dev_ema50",
            "V_corr_raw_dev_ema200",
            "V_corr_raw_dev_ema800",
            "dV_corr",
            "abs_dV_corr",
            "dI",
            "dI_abs_mean",
        ]
        self.anchor_indices = [self.feature_cols.index(col) for col in anchor_preferred if col in self.feature_cols]
        self.dynamic_indices = [self.feature_cols.index(col) for col in dynamic_preferred if col in self.feature_cols]
        if ("V_corr_raw" not in self.feature_cols and "V_eq_slow_raw" not in self.feature_cols) or "T" not in self.feature_cols:
            raise RuntimeError("DynamicOnlyAnchorResidualModel requires V_corr_raw or V_eq_slow_raw, plus T features.")
        if not self.dynamic_indices:
            raise RuntimeError("DynamicOnlyAnchorResidualModel found no dynamic residual input features.")
        anchor_dim = len(self.anchor_indices)
        self.anchor_head = nn.Sequential(
            nn.Linear(anchor_dim, int(hidden_size)),
            nn.LayerNorm(int(hidden_size)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), int(hidden_size) // 2),
            nn.SiLU(),
            nn.Linear(int(hidden_size) // 2, 1),
        )
        self.dynamic = VcorrITGoalModel(
            input_dim=len(self.dynamic_indices),
            hidden_size=int(hidden_size),
            recurrent=str(recurrent),
            layers=int(layers),
            head_kind="linear",
            temp_mode="none",
            dropout=float(dropout),
            kernel_size=int(kernel_size),
            norm_kind="channel",
        )
        self.residual_head = nn.Sequential(
            nn.Linear(int(hidden_size), int(hidden_size) // 2),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size) // 2, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        if self.gated:
            gate_hidden = max(8, int(hidden_size) // 4)
            self.residual_gate = nn.Sequential(
                nn.Linear(int(hidden_size), gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, 1),
            )
            nn.init.zeros_(self.residual_gate[-1].weight)
            nn.init.constant_(self.residual_gate[-1].bias, -1.0)

    def residual_limit_tensor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.residual_limit_mode == "learnable":
            return self.residual_limit_param.to(device=device, dtype=dtype)
        if self.residual_limit_mode == "bounded_learnable":
            lower = torch.as_tensor(self.residual_limit_lower, device=device, dtype=dtype)
            upper = torch.as_tensor(self.residual_limit_upper, device=device, dtype=dtype)
            return lower + (upper - lower) * torch.sigmoid(self.residual_limit_param.to(device=device, dtype=dtype))
        return torch.as_tensor(self.residual_limit, device=device, dtype=dtype)

    def residual_limit_value(self) -> float:
        if self.residual_limit_mode == "learnable":
            return float(self.residual_limit_param.detach().cpu())
        if self.residual_limit_mode == "bounded_learnable":
            raw = float(self.residual_limit_param.detach().cpu())
            sig = 1.0 / (1.0 + float(np.exp(-raw)))
            return float(self.residual_limit_lower + (self.residual_limit_upper - self.residual_limit_lower) * sig)
        return float(self.residual_limit)

    def residual_limit_initial(self) -> float:
        return float(self.residual_limit_initial_value)

    def _dynamic_input(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(dim=2, index=torch.as_tensor(self.dynamic_indices, device=x.device))

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return self.dynamic.encode_sequence(self._dynamic_input(x))

    def anchor_sequence(self, x: torch.Tensor) -> torch.Tensor:
        anchor_x = x.index_select(dim=2, index=torch.as_tensor(self.anchor_indices, device=x.device))
        return torch.sigmoid(self.anchor_head(anchor_x))

    def residual_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode_sequence(x)
        limit = self.residual_limit_tensor(device=h.device, dtype=h.dtype)
        residual = limit * torch.tanh(self.residual_head(h))
        if self.gated:
            residual = residual * torch.sigmoid(self.residual_gate(h))
        return residual

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return (self.anchor_sequence(x) + self.residual_sequence(x)).clamp(0.0, 1.0)

    def logits_sequence(self, x: torch.Tensor) -> torch.Tensor:
        soc = self.forward_sequence(x).clamp(1e-5, 1.0 - 1e-5)
        return torch.logit(soc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1, :]


class TemperatureBandedAnchorResidualModel(nn.Module):
    """CEMA sequence model with cold/mid/hot residual heads.

    The residual correction is conditioned on causal window-regime features
    rather than profile labels, so it remains valid for unseen drive profiles.
    """

    def __init__(
        self,
        input_dim: int,
        feature_cols: list[str],
        hidden_size: int,
        layers: int,
        kernel_size: int,
        dropout: float,
        residual_limit: float,
        residual_limit_init: str = "value",
        residual_limit_mode: str = "fixed",
        residual_limit_lower: float = 0.0,
        residual_limit_upper: float = 0.2,
        recurrent: str = "lstm",
        mid_regime_dynamic: bool = False,
        mid_interpolation: bool = False,
        soc_basis: bool = False,
        dynamic_residual_scale: bool = False,
        transition_branch: bool = False,
        mha_pooling: bool = False,
        gate_mode: str = "temperature",
    ):
        super().__init__()
        self.feature_cols = list(feature_cols)
        self.mid_regime_dynamic = bool(mid_regime_dynamic)
        self.mid_interpolation = bool(mid_interpolation)
        self.soc_basis = bool(soc_basis)
        self.dynamic_residual_scale = bool(dynamic_residual_scale)
        self.transition_branch = bool(transition_branch)
        self.mha_pooling = bool(mha_pooling)
        if self.mid_regime_dynamic and self.mid_interpolation:
            raise ValueError("mid_regime_dynamic and mid_interpolation are mutually exclusive.")
        self.gate_mode = str(gate_mode)
        if self.gate_mode not in {"temperature", "tz"}:
            raise ValueError(f"Unknown TBR gate_mode={gate_mode!r}")
        self.residual_limit_mode = str(residual_limit_mode)
        residual_init = torch.rand((), dtype=torch.float32).item() if str(residual_limit_init) == "rand01" else float(residual_limit)
        self.residual_limit = float(residual_init)
        self.residual_limit_initial_value = float(residual_init)
        self.residual_limit_init = str(residual_limit_init)
        self.residual_limit_lower = float(residual_limit_lower)
        self.residual_limit_upper = float(residual_limit_upper)
        if self.residual_limit_mode == "learnable":
            self.residual_limit_param = nn.Parameter(torch.tensor(float(residual_init), dtype=torch.float32))
        elif self.residual_limit_mode == "bounded_learnable":
            lower = float(residual_limit_lower)
            upper = float(residual_limit_upper)
            if not lower < upper:
                raise ValueError("anchor residual limit lower bound must be smaller than upper bound.")
            init = min(max(float(residual_init), lower + 1e-6), upper - 1e-6)
            p = (init - lower) / (upper - lower)
            self.residual_limit_param = nn.Parameter(torch.tensor(float(np.log(p / (1.0 - p))), dtype=torch.float32))
        elif self.residual_limit_mode != "fixed":
            raise ValueError(f"Unknown residual_limit_mode={residual_limit_mode!r}")

        self.dynamic = VcorrITGoalModel(
            input_dim=input_dim,
            hidden_size=int(hidden_size),
            recurrent=str(recurrent),
            layers=int(layers),
            head_kind="linear",
            temp_mode="none",
            dropout=float(dropout),
            kernel_size=int(kernel_size),
            norm_kind="channel",
        )
        if self.mha_pooling:
            num_heads = 4 if int(hidden_size) % 4 == 0 else 1
            self.mha = nn.MultiheadAttention(
                embed_dim=int(hidden_size),
                num_heads=num_heads,
                dropout=float(dropout),
                batch_first=True,
            )
            self.mha_norm = nn.LayerNorm(int(hidden_size))
            self.mha_dropout = nn.Dropout(float(dropout))
        preferred = [
            "V_corr_raw",
            "V_eq_slow_raw",
            "T",
            "V_corr_raw_ema50",
            "V_corr_raw_dev_ema50",
            "V_corr_raw_ema200",
            "V_corr_raw_dev_ema200",
            "V_corr_raw_ema800",
            "V_corr_raw_dev_ema800",
            "dV_corr",
            "abs_dV_corr",
            "Vcorr_x_absI",
        ]
        if "V_corr_raw" not in self.feature_cols or "I_raw" not in self.feature_cols or "T" not in self.feature_cols:
            raise RuntimeError("TemperatureBandedAnchorResidualModel requires V_corr_raw, I_raw, and T features.")
        self.vcorr_idx = int(self.feature_cols.index("V_corr_raw"))
        self.current_idx = int(self.feature_cols.index("I_raw"))
        self.temp_idx = int(self.feature_cols.index("T"))
        self.abs_i_idx = self.feature_cols.index("absI") if "absI" in self.feature_cols else None
        self.di_idx = self.feature_cols.index("dI") if "dI" in self.feature_cols else None
        self.anchor_indices = [self.feature_cols.index(col) for col in preferred if col in self.feature_cols]
        anchor_dim = len(self.anchor_indices)
        self.anchor_head = nn.Sequential(
            nn.Linear(anchor_dim, int(hidden_size)),
            nn.LayerNorm(int(hidden_size)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), int(hidden_size) // 2),
            nn.SiLU(),
            nn.Linear(int(hidden_size) // 2, 1),
        )
        regime_dim = 5
        soc_basis_dim = 3 if self.soc_basis else 0
        branch_dim = int(hidden_size) + regime_dim + soc_basis_dim
        self.regime_norm = nn.LayerNorm(regime_dim)
        self.cold_head = self._make_residual_branch(branch_dim, int(hidden_size), float(dropout))
        self.mid_head = self._make_residual_branch(branch_dim, int(hidden_size), float(dropout))
        self.hot_head = self._make_residual_branch(branch_dim, int(hidden_size), float(dropout))
        if self.mid_interpolation:
            gate_hidden = max(8, int(hidden_size) // 4)
            self.mid_interp_gate = nn.Sequential(
                nn.Linear(regime_dim, gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, 1),
            )
            nn.init.zeros_(self.mid_interp_gate[-1].weight)
            nn.init.zeros_(self.mid_interp_gate[-1].bias)
        if self.dynamic_residual_scale:
            scale_hidden = max(8, int(hidden_size) // 4)
            self.residual_scale_head = nn.Sequential(
                nn.Linear(regime_dim, scale_hidden),
                nn.SiLU(),
                nn.Linear(scale_hidden, 1),
            )
            nn.init.zeros_(self.residual_scale_head[-1].weight)
            nn.init.zeros_(self.residual_scale_head[-1].bias)
        if self.transition_branch:
            self.transition_head = self._make_residual_branch(branch_dim, int(hidden_size), float(dropout))
            gate_hidden = max(8, int(hidden_size) // 4)
            self.transition_gate = nn.Sequential(
                nn.Linear(regime_dim, gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, 1),
            )
            nn.init.zeros_(self.transition_gate[-1].weight)
            nn.init.constant_(self.transition_gate[-1].bias, -2.0)
        if self.mid_regime_dynamic:
            self.mid_dynamic_head = self._make_residual_branch(branch_dim, int(hidden_size), float(dropout))
            gate_hidden = max(8, int(hidden_size) // 4)
            self.mid_dynamic_gate = nn.Sequential(
                nn.Linear(regime_dim, gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, 1),
            )
            nn.init.zeros_(self.mid_dynamic_gate[-1].weight)
            nn.init.constant_(self.mid_dynamic_gate[-1].bias, -1.0)
        if self.gate_mode == "tz":
            gate_hidden = max(16, int(hidden_size) // 4)
            self.tz_gate_delta = nn.Sequential(
                nn.Linear(regime_dim + 1, gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, 3),
            )
            nn.init.zeros_(self.tz_gate_delta[-1].weight)
            nn.init.zeros_(self.tz_gate_delta[-1].bias)

    @staticmethod
    def _make_residual_branch(input_dim: int, hidden_size: int, dropout: float) -> nn.Sequential:
        branch = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_size) // 2),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size) // 2, 1),
        )
        nn.init.zeros_(branch[-1].weight)
        nn.init.zeros_(branch[-1].bias)
        return branch

    def residual_limit_tensor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.residual_limit_mode == "learnable":
            return self.residual_limit_param.to(device=device, dtype=dtype)
        if self.residual_limit_mode == "bounded_learnable":
            lower = torch.as_tensor(self.residual_limit_lower, device=device, dtype=dtype)
            upper = torch.as_tensor(self.residual_limit_upper, device=device, dtype=dtype)
            return lower + (upper - lower) * torch.sigmoid(self.residual_limit_param.to(device=device, dtype=dtype))
        return torch.as_tensor(self.residual_limit, device=device, dtype=dtype)

    def residual_limit_value(self) -> float:
        if self.residual_limit_mode == "learnable":
            return float(self.residual_limit_param.detach().cpu())
        if self.residual_limit_mode == "bounded_learnable":
            raw = float(self.residual_limit_param.detach().cpu())
            sig = 1.0 / (1.0 + float(np.exp(-raw)))
            return float(self.residual_limit_lower + (self.residual_limit_upper - self.residual_limit_lower) * sig)
        return float(self.residual_limit)

    def residual_limit_initial(self) -> float:
        return float(self.residual_limit_initial_value)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dynamic.encode_sequence(x)
        if not self.mha_pooling:
            return h
        attn_out, _ = self.mha(h, h, h, need_weights=False)
        return self.mha_norm(h + self.mha_dropout(attn_out))

    def anchor_sequence(self, x: torch.Tensor) -> torch.Tensor:
        anchor_x = x.index_select(dim=2, index=torch.as_tensor(self.anchor_indices, device=x.device))
        return torch.sigmoid(self.anchor_head(anchor_x))

    def _regime_sequence(self, x: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(1, int(x.shape[1]) + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        current = x[:, :, self.current_idx:self.current_idx + 1]
        abs_current = x[:, :, self.abs_i_idx:self.abs_i_idx + 1].abs() if self.abs_i_idx is not None else current.abs()
        abs_i_mean = abs_current.cumsum(dim=1) / steps
        i_mean = current.cumsum(dim=1) / steps
        i2_mean = current.square().cumsum(dim=1) / steps
        i_std = (i2_mean - i_mean.square()).clamp_min(0.0).sqrt()
        if self.di_idx is not None:
            di = x[:, :, self.di_idx:self.di_idx + 1]
        else:
            di = torch.diff(current, dim=1, prepend=current[:, :1, :])
        di_energy = di.square().cumsum(dim=1) / steps
        vcorr = x[:, :, self.vcorr_idx:self.vcorr_idx + 1]
        vcorr_max = torch.cummax(vcorr, dim=1).values
        vcorr_min = torch.cummin(vcorr, dim=1).values
        vcorr_span = vcorr_max - vcorr_min
        return torch.cat([abs_i_mean, i_std, di_energy, vcorr_span, anchor], dim=2)

    def _temperature_gates(self, x: torch.Tensor) -> torch.Tensor:
        temp = x[:, :, self.temp_idx:self.temp_idx + 1]
        sharpness = 8.0
        cold = torch.sigmoid((-0.45 - temp) * sharpness)
        mid = torch.sigmoid((temp + 0.45) * sharpness) * torch.sigmoid((0.65 - temp) * sharpness)
        hot = torch.sigmoid((temp - 0.65) * sharpness)
        gates = torch.cat([cold, mid, hot], dim=2)
        return gates / gates.sum(dim=2, keepdim=True).clamp_min(1e-6)

    def _residual_gates(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        gates = self._temperature_gates(x)
        if self.gate_mode == "tz":
            temp = x[:, :, self.temp_idx:self.temp_idx + 1]
            delta = self.tz_gate_delta(torch.cat([temp, z], dim=2)).clamp(-5.0, 5.0)
            gates = gates * torch.exp(delta)
            gates = gates / gates.sum(dim=2, keepdim=True).clamp_min(1e-6)
        return gates

    @staticmethod
    def _soc_basis(anchor: torch.Tensor) -> torch.Tensor:
        centers = anchor.new_tensor([0.2, 0.5, 0.8]).view(1, 1, 3)
        width = anchor.new_tensor(0.18)
        logits = -0.5 * ((anchor - centers) / width).square()
        return torch.softmax(logits, dim=2)

    def _branch_context(self, x: torch.Tensor, anchor: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.regime_norm(self._regime_sequence(x, anchor))
        parts = [h, z]
        if self.soc_basis:
            parts.append(self._soc_basis(anchor))
        return z, torch.cat(parts, dim=2)

    def _residual_scale_from_regime(self, z: torch.Tensor) -> torch.Tensor:
        if not self.dynamic_residual_scale:
            return z.new_ones((*z.shape[:2], 1))
        return 0.5 + torch.sigmoid(self.residual_scale_head(z))

    def _raw_residual_and_scale_from_anchor_hidden(
        self,
        x: torch.Tensor,
        anchor: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z, branch_input = self._branch_context(x, anchor, h)
        cold_residual = self.cold_head(branch_input)
        hot_residual = self.hot_head(branch_input)
        mid_residual = self.mid_head(branch_input)
        if self.mid_interpolation:
            alpha = torch.sigmoid(self.mid_interp_gate(z))
            mid_residual = alpha * cold_residual + (1.0 - alpha) * hot_residual + mid_residual
        if self.mid_regime_dynamic:
            mid_residual = mid_residual + torch.sigmoid(self.mid_dynamic_gate(z)) * self.mid_dynamic_head(branch_input)
        residuals = torch.cat(
            [
                cold_residual,
                mid_residual,
                hot_residual,
            ],
            dim=2,
        )
        gates = self._residual_gates(x, z)
        raw_residual = (gates * residuals).sum(dim=2, keepdim=True)
        if self.transition_branch:
            transition_gate = torch.sigmoid(self.transition_gate(z))
            raw_residual = raw_residual + transition_gate * self.transition_head(branch_input)
        return raw_residual, self._residual_scale_from_regime(z)

    def _raw_residual_from_anchor_hidden(self, x: torch.Tensor, anchor: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        raw_residual, _scale = self._raw_residual_and_scale_from_anchor_hidden(x, anchor, h)
        return raw_residual

    def mid_delta_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if not self.mid_interpolation:
            return x.new_zeros((int(x.shape[0]), int(x.shape[1]), 1))
        anchor = self.anchor_sequence(x)
        h = self.encode_sequence(x)
        _z, branch_input = self._branch_context(x, anchor, h)
        return self.mid_head(branch_input)

    def raw_residual_sequence(self, x: torch.Tensor) -> torch.Tensor:
        anchor = self.anchor_sequence(x)
        h = self.encode_sequence(x)
        return self._raw_residual_from_anchor_hidden(x, anchor, h)

    def residual_sequence(self, x: torch.Tensor) -> torch.Tensor:
        anchor = self.anchor_sequence(x)
        h = self.encode_sequence(x)
        raw_residual, scale = self._raw_residual_and_scale_from_anchor_hidden(x, anchor, h)
        limit = self.residual_limit_tensor(device=x.device, dtype=x.dtype)
        return limit * scale * torch.tanh(raw_residual)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        anchor = self.anchor_sequence(x)
        h = self.encode_sequence(x)
        raw_residual, scale = self._raw_residual_and_scale_from_anchor_hidden(x, anchor, h)
        limit = self.residual_limit_tensor(device=h.device, dtype=h.dtype)
        residual = limit * scale * torch.tanh(raw_residual)
        return (anchor + residual).clamp(0.0, 1.0)

    def logits_sequence(self, x: torch.Tensor) -> torch.Tensor:
        soc = self.forward_sequence(x).clamp(1e-5, 1.0 - 1e-5)
        return torch.logit(soc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1, :]


class BaseMidResidualModel(nn.Module):
    """Base sequence model with a 25C-gated regime residual branch."""

    def __init__(
        self,
        input_dim: int,
        feature_cols: list[str],
        hidden_size: int,
        layers: int,
        kernel_size: int,
        dropout: float,
        residual_limit: float,
        recurrent: str = "transformer",
        head_kind: str = "linear",
    ):
        super().__init__()
        self.feature_cols = list(feature_cols)
        if "V_corr_raw" not in self.feature_cols or "I_raw" not in self.feature_cols or "T" not in self.feature_cols:
            raise RuntimeError("BaseMidResidualModel requires V_corr_raw, I_raw, and T features.")
        self.vcorr_idx = int(self.feature_cols.index("V_corr_raw"))
        self.current_idx = int(self.feature_cols.index("I_raw"))
        self.temp_idx = int(self.feature_cols.index("T"))
        self.abs_i_idx = self.feature_cols.index("absI") if "absI" in self.feature_cols else None
        self.di_idx = self.feature_cols.index("dI") if "dI" in self.feature_cols else None
        self.residual_limit = float(residual_limit)
        self.base = VcorrITGoalModel(
            input_dim=input_dim,
            hidden_size=int(hidden_size),
            recurrent=str(recurrent),
            layers=int(layers),
            head_kind=str(head_kind),
            temp_mode="none",
            dropout=float(dropout),
            kernel_size=int(kernel_size),
            norm_kind="channel",
        )
        regime_dim = 5
        self.regime_norm = nn.LayerNorm(regime_dim)
        self.residual_head = nn.Sequential(
            nn.Linear(int(hidden_size) + regime_dim, int(hidden_size) // 2),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size) // 2, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def residual_limit_initial(self) -> float:
        return float(self.residual_limit)

    def residual_limit_value(self) -> float:
        return float(self.residual_limit)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.encode_sequence(x)

    def _mid_gate(self, x: torch.Tensor) -> torch.Tensor:
        temp = x[:, :, self.temp_idx:self.temp_idx + 1]
        sharpness = 8.0
        return torch.sigmoid((temp + 0.45) * sharpness) * torch.sigmoid((0.65 - temp) * sharpness)

    def _regime_sequence(self, x: torch.Tensor, anchor_soc: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(1, int(x.shape[1]) + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        current = x[:, :, self.current_idx:self.current_idx + 1]
        abs_current = x[:, :, self.abs_i_idx:self.abs_i_idx + 1].abs() if self.abs_i_idx is not None else current.abs()
        abs_i_mean = abs_current.cumsum(dim=1) / steps
        i_mean = current.cumsum(dim=1) / steps
        i2_mean = current.square().cumsum(dim=1) / steps
        i_std = (i2_mean - i_mean.square()).clamp_min(0.0).sqrt()
        if self.di_idx is not None:
            di = x[:, :, self.di_idx:self.di_idx + 1]
        else:
            di = torch.diff(current, dim=1, prepend=current[:, :1, :])
        di_energy = di.square().cumsum(dim=1) / steps
        vcorr = x[:, :, self.vcorr_idx:self.vcorr_idx + 1]
        vcorr_span = torch.cummax(vcorr, dim=1).values - torch.cummin(vcorr, dim=1).values
        return torch.cat([abs_i_mean, i_std, di_energy, vcorr_span, anchor_soc], dim=2)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.base.encode_sequence(x)
        base_logits = self.base.base_head(h)
        base_soc = torch.sigmoid(base_logits)
        z = self.regime_norm(self._regime_sequence(x, base_soc))
        residual = float(self.residual_limit) * torch.tanh(self.residual_head(torch.cat([h, z], dim=2)))
        return (base_soc + self._mid_gate(x) * residual).clamp(0.0, 1.0)

    def logits_sequence(self, x: torch.Tensor) -> torch.Tensor:
        soc = self.forward_sequence(x).clamp(1e-5, 1.0 - 1e-5)
        return torch.logit(soc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1, :]


class MLPControlModel(nn.Module):
    """Non-recurrent control model for fixed G4 model-class comparisons."""

    def __init__(self, input_dim: int, hidden_size: int, dropout: float, mode: str):
        super().__init__()
        self.mode = str(mode)
        if self.mode not in {"endpoint", "window_summary"}:
            raise ValueError(f"Unknown MLPControlModel mode={mode!r}")
        in_dim = int(input_dim) if self.mode == "endpoint" else int(input_dim) * 4
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, int(hidden_size)),
            nn.LayerNorm(int(hidden_size)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), int(hidden_size)),
            nn.LayerNorm(int(hidden_size)),
            nn.SiLU(),
        )
        self.head = nn.Linear(int(hidden_size), 1)

    def _summarize(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "endpoint":
            return x[:, -1, :]
        x_end = x[:, -1, :]
        x_mean = x.mean(dim=1)
        x_std = x.std(dim=1, unbiased=False)
        x_delta = x_end - x[:, 0, :]
        return torch.cat([x_end, x_mean, x_std, x_delta], dim=1)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self._summarize(x))
        return h[:, None, :].expand(-1, x.shape[1], -1)

    def logits_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode_sequence(x)
        h_end = h[:, -1, :]
        logits = self.head(h_end)
        return logits[:, None, :].expand(-1, x.shape[1], -1)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits_sequence(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1, :]


class AnchorResidualMLPModel(nn.Module):
    """MLP encoder control with the same anchor-residual output head."""

    def __init__(
        self,
        input_dim: int,
        feature_cols: list[str],
        hidden_size: int,
        dropout: float,
        residual_limit: float,
        mode: str,
        residual_limit_init: str = "value",
        residual_limit_mode: str = "fixed",
        residual_limit_lower: float = 0.0,
        residual_limit_upper: float = 0.2,
    ):
        super().__init__()
        self.residual_limit_mode = str(residual_limit_mode)
        residual_init = torch.rand((), dtype=torch.float32).item() if str(residual_limit_init) == "rand01" else float(residual_limit)
        self.residual_limit = float(residual_init)
        self.residual_limit_initial_value = float(residual_init)
        self.residual_limit_init = str(residual_limit_init)
        self.residual_limit_lower = float(residual_limit_lower)
        self.residual_limit_upper = float(residual_limit_upper)
        if self.residual_limit_mode == "learnable":
            self.residual_limit_param = nn.Parameter(torch.tensor(float(residual_init), dtype=torch.float32))
        elif self.residual_limit_mode == "bounded_learnable":
            lower = float(residual_limit_lower)
            upper = float(residual_limit_upper)
            if not lower < upper:
                raise ValueError("anchor residual limit lower bound must be smaller than upper bound.")
            init = min(max(float(residual_init), lower + 1e-6), upper - 1e-6)
            p = (init - lower) / (upper - lower)
            self.residual_limit_param = nn.Parameter(torch.tensor(float(np.log(p / (1.0 - p))), dtype=torch.float32))
        elif self.residual_limit_mode != "fixed":
            raise ValueError(f"Unknown residual_limit_mode={residual_limit_mode!r}")
        self.dynamic = MLPControlModel(
            input_dim=input_dim,
            hidden_size=int(hidden_size),
            dropout=float(dropout),
            mode=str(mode),
        )
        preferred = [
            "V_corr_raw",
            "T",
            "V_corr_raw_ema50",
            "V_corr_raw_dev_ema50",
            "V_corr_raw_ema200",
            "V_corr_raw_dev_ema200",
            "V_corr_raw_ema800",
            "V_corr_raw_dev_ema800",
            "dV_corr",
            "abs_dV_corr",
            "Vcorr_x_absI",
        ]
        idx = [feature_cols.index(col) for col in preferred if col in feature_cols]
        if ("V_corr_raw" not in feature_cols and "V_eq_slow_raw" not in feature_cols) or "T" not in feature_cols:
            raise RuntimeError("AnchorResidualMLPModel requires V_corr_raw or V_eq_slow_raw, plus T features.")
        self.anchor_indices = idx
        anchor_dim = len(idx)
        self.anchor_head = nn.Sequential(
            nn.Linear(anchor_dim, int(hidden_size)),
            nn.LayerNorm(int(hidden_size)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), int(hidden_size) // 2),
            nn.SiLU(),
            nn.Linear(int(hidden_size) // 2, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(int(hidden_size), int(hidden_size) // 2),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size) // 2, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def residual_limit_tensor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.residual_limit_mode == "learnable":
            return self.residual_limit_param.to(device=device, dtype=dtype)
        if self.residual_limit_mode == "bounded_learnable":
            lower = torch.as_tensor(self.residual_limit_lower, device=device, dtype=dtype)
            upper = torch.as_tensor(self.residual_limit_upper, device=device, dtype=dtype)
            return lower + (upper - lower) * torch.sigmoid(self.residual_limit_param.to(device=device, dtype=dtype))
        return torch.as_tensor(self.residual_limit, device=device, dtype=dtype)

    def residual_limit_value(self) -> float:
        if self.residual_limit_mode == "learnable":
            return float(self.residual_limit_param.detach().cpu())
        if self.residual_limit_mode == "bounded_learnable":
            raw = float(self.residual_limit_param.detach().cpu())
            sig = 1.0 / (1.0 + float(np.exp(-raw)))
            return float(self.residual_limit_lower + (self.residual_limit_upper - self.residual_limit_lower) * sig)
        return float(self.residual_limit)

    def residual_limit_initial(self) -> float:
        return float(self.residual_limit_initial_value)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return self.dynamic.encode_sequence(x)

    def anchor_sequence(self, x: torch.Tensor) -> torch.Tensor:
        anchor_x = x.index_select(dim=2, index=torch.as_tensor(self.anchor_indices, device=x.device))
        return torch.sigmoid(self.anchor_head(anchor_x))

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        anchor = self.anchor_sequence(x)
        h = self.encode_sequence(x)
        limit = self.residual_limit_tensor(device=h.device, dtype=h.dtype)
        residual = limit * torch.tanh(self.residual_head(h))
        return (anchor + residual).clamp(0.0, 1.0)

    def logits_sequence(self, x: torch.Tensor) -> torch.Tensor:
        soc = self.forward_sequence(x).clamp(1e-5, 1.0 - 1e-5)
        return torch.logit(soc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1, :]


def _selected_feature_columns(feature_set: str) -> list[str]:
    cols = vit_feature_columns(str(feature_set))
    if str(feature_set).startswith("paper_g4_eqdyn_no_vcorr"):
        prefix = ["V_eq_slow_raw", "I_raw", "T"]
    else:
        prefix = ["V_corr_raw", "I_raw", "T"]
    missing_prefix = [c for c in prefix if c not in cols]
    if missing_prefix:
        raise RuntimeError(f"feature_set={feature_set!r} is missing required leading columns: {missing_prefix}")
    return prefix + [c for c in cols if c not in set(prefix)]


def _make_stage1_model(
    cfg: TrainDSTSelectorConfig,
    variant: CondInvVariant,
    input_dim: int | None = None,
    feature_cols: list[str] | None = None,
) -> nn.Module:
    input_dim = int(input_dim if input_dim is not None else len(FEATURE_COLS))
    if str(cfg.model_kind) == "fusion_h64_h128":
        return H64H128FusionModel(input_dim, dropout=float(cfg.dropout)).to(device)
    if str(cfg.model_kind) == "fusion_h64_h128_fixed":
        return H64H128FusionModel(
            input_dim,
            dropout=float(cfg.dropout),
            fixed_h64_weight=float(cfg.fusion_h64_weight),
        ).to(device)
    if str(cfg.model_kind) in {"anchor_residual_tcn", "anchor_residual_sequence"}:
        return AnchorResidualTCNModel(
            input_dim=input_dim,
            feature_cols=list(feature_cols or _selected_feature_columns(str(cfg.feature_set))),
            hidden_size=int(cfg.hidden_size),
            layers=int(cfg.layers),
            kernel_size=int(cfg.kernel_size),
            dropout=float(cfg.dropout),
            residual_limit=float(cfg.anchor_residual_limit),
            residual_limit_init=str(cfg.anchor_residual_limit_init),
            residual_limit_mode=str(cfg.anchor_residual_limit_mode),
            residual_limit_lower=float(cfg.anchor_residual_limit_lower),
            residual_limit_upper=float(cfg.anchor_residual_limit_upper),
            recurrent=str(cfg.recurrent),
        ).to(device)
    if str(cfg.model_kind) in {"anchor_residual_dynamic_only_sequence", "anchor_residual_dynamic_gated_sequence"}:
        return DynamicOnlyAnchorResidualModel(
            input_dim=input_dim,
            feature_cols=list(feature_cols or _selected_feature_columns(str(cfg.feature_set))),
            hidden_size=int(cfg.hidden_size),
            layers=int(cfg.layers),
            kernel_size=int(cfg.kernel_size),
            dropout=float(cfg.dropout),
            residual_limit=float(cfg.anchor_residual_limit),
            residual_limit_init=str(cfg.anchor_residual_limit_init),
            residual_limit_mode=str(cfg.anchor_residual_limit_mode),
            residual_limit_lower=float(cfg.anchor_residual_limit_lower),
            residual_limit_upper=float(cfg.anchor_residual_limit_upper),
            recurrent=str(cfg.recurrent),
            gated=str(cfg.model_kind) == "anchor_residual_dynamic_gated_sequence",
        ).to(device)
    if str(cfg.model_kind) == "normal_dynamic_correction_sequence":
        return NormalDynamicCorrectionModel(
            input_dim=input_dim,
            feature_cols=list(feature_cols or _selected_feature_columns(str(cfg.feature_set))),
            hidden_size=int(cfg.hidden_size),
            layers=int(cfg.layers),
            kernel_size=int(cfg.kernel_size),
            dropout=float(cfg.dropout),
            residual_limit=float(cfg.anchor_residual_limit),
            recurrent=str(cfg.recurrent),
            head_kind=str(cfg.head_kind),
            temp_mode=str(cfg.temp_mode),
        ).to(device)
    if str(cfg.model_kind) == "anchor_residual_temp_limit_sequence":
        return TemperatureLimitAnchorResidualTCNModel(
            input_dim=input_dim,
            feature_cols=list(feature_cols or _selected_feature_columns(str(cfg.feature_set))),
            hidden_size=int(cfg.hidden_size),
            layers=int(cfg.layers),
            kernel_size=int(cfg.kernel_size),
            dropout=float(cfg.dropout),
            residual_limit=float(cfg.anchor_residual_limit),
            residual_limit_init=str(cfg.anchor_residual_limit_init),
            residual_limit_mode=str(cfg.anchor_residual_limit_mode),
            residual_limit_lower=float(cfg.anchor_residual_limit_lower),
            residual_limit_upper=float(cfg.anchor_residual_limit_upper),
            recurrent=str(cfg.recurrent),
        ).to(device)
    if str(cfg.model_kind) == "anchor_residual_temp_budget_sequence":
        return TemperatureBudgetAnchorResidualModel(
            input_dim=input_dim,
            feature_cols=list(feature_cols or _selected_feature_columns(str(cfg.feature_set))),
            hidden_size=int(cfg.hidden_size),
            layers=int(cfg.layers),
            kernel_size=int(cfg.kernel_size),
            dropout=float(cfg.dropout),
            residual_limit=float(cfg.anchor_residual_limit),
            residual_limit_init=str(cfg.anchor_residual_limit_init),
            residual_limit_mode="fixed",
            residual_limit_lower=float(cfg.anchor_residual_limit_lower),
            residual_limit_upper=float(cfg.anchor_residual_limit_upper),
            recurrent=str(cfg.recurrent),
        ).to(device)
    if str(cfg.model_kind) == "anchor_residual_cold_low_soc_positive_sequence":
        return ColdLowSOCPositiveCorrectorModel(
            input_dim=input_dim,
            feature_cols=list(feature_cols or _selected_feature_columns(str(cfg.feature_set))),
            hidden_size=int(cfg.hidden_size),
            layers=int(cfg.layers),
            kernel_size=int(cfg.kernel_size),
            dropout=float(cfg.dropout),
            residual_limit=float(cfg.anchor_residual_limit),
            residual_limit_init=str(cfg.anchor_residual_limit_init),
            residual_limit_mode=str(cfg.anchor_residual_limit_mode),
            residual_limit_lower=float(cfg.anchor_residual_limit_lower),
            residual_limit_upper=float(cfg.anchor_residual_limit_upper),
            recurrent=str(cfg.recurrent),
            correction_limit=float(cfg.cold_corrector_limit),
            soc_threshold=float(cfg.cold_corrector_soc_threshold),
            temp_threshold=float(cfg.cold_corrector_temp_threshold),
            gate_sharpness=float(cfg.cold_corrector_gate_sharpness),
            sag_gate=bool(cfg.cold_corrector_sag_gate),
        ).to(device)
    if str(cfg.model_kind) == "anchor_residual_cold_low_soc_signed_sequence":
        return ColdLowSOCSignedCorrectorModel(
            input_dim=input_dim,
            feature_cols=list(feature_cols or _selected_feature_columns(str(cfg.feature_set))),
            hidden_size=int(cfg.hidden_size),
            layers=int(cfg.layers),
            kernel_size=int(cfg.kernel_size),
            dropout=float(cfg.dropout),
            residual_limit=float(cfg.anchor_residual_limit),
            residual_limit_init=str(cfg.anchor_residual_limit_init),
            residual_limit_mode=str(cfg.anchor_residual_limit_mode),
            residual_limit_lower=float(cfg.anchor_residual_limit_lower),
            residual_limit_upper=float(cfg.anchor_residual_limit_upper),
            recurrent=str(cfg.recurrent),
            correction_limit=float(cfg.cold_corrector_limit),
            soc_threshold=float(cfg.cold_corrector_soc_threshold),
            temp_threshold=float(cfg.cold_corrector_temp_threshold),
            gate_sharpness=float(cfg.cold_corrector_gate_sharpness),
            sag_gate=bool(cfg.cold_corrector_sag_gate),
        ).to(device)
    if str(cfg.model_kind) == "normal_cold_low_soc_positive_sequence":
        return ColdLowSOCNormalPositiveCorrectorModel(
            input_dim=input_dim,
            feature_cols=list(feature_cols or _selected_feature_columns(str(cfg.feature_set))),
            hidden_size=int(cfg.hidden_size),
            layers=int(cfg.layers),
            kernel_size=int(cfg.kernel_size),
            dropout=float(cfg.dropout),
            recurrent=str(cfg.recurrent),
            head_kind=str(cfg.head_kind),
            temp_mode=str(cfg.temp_mode),
            correction_limit=float(cfg.cold_corrector_limit),
            soc_threshold=float(cfg.cold_corrector_soc_threshold),
            temp_threshold=float(cfg.cold_corrector_temp_threshold),
            gate_sharpness=float(cfg.cold_corrector_gate_sharpness),
            sag_gate=bool(cfg.cold_corrector_sag_gate),
        ).to(device)
    if str(cfg.model_kind) == "normal_cold_low_soc_signed_sequence":
        return ColdLowSOCNormalSignedCorrectorModel(
            input_dim=input_dim,
            feature_cols=list(feature_cols or _selected_feature_columns(str(cfg.feature_set))),
            hidden_size=int(cfg.hidden_size),
            layers=int(cfg.layers),
            kernel_size=int(cfg.kernel_size),
            dropout=float(cfg.dropout),
            recurrent=str(cfg.recurrent),
            head_kind=str(cfg.head_kind),
            temp_mode=str(cfg.temp_mode),
            correction_limit=float(cfg.cold_corrector_limit),
            soc_threshold=float(cfg.cold_corrector_soc_threshold),
            temp_threshold=float(cfg.cold_corrector_temp_threshold),
            gate_sharpness=float(cfg.cold_corrector_gate_sharpness),
            sag_gate=bool(cfg.cold_corrector_sag_gate),
        ).to(device)
    if str(cfg.model_kind) == "normal_cold_high_dynamic_signed_sequence":
        return ColdHighDynamicNormalSignedCorrectorModel(
            input_dim=input_dim,
            feature_cols=list(feature_cols or _selected_feature_columns(str(cfg.feature_set))),
            hidden_size=int(cfg.hidden_size),
            layers=int(cfg.layers),
            kernel_size=int(cfg.kernel_size),
            dropout=float(cfg.dropout),
            recurrent=str(cfg.recurrent),
            head_kind=str(cfg.head_kind),
            temp_mode=str(cfg.temp_mode),
            correction_limit=float(cfg.cold_corrector_limit),
            soc_threshold=float(cfg.cold_corrector_soc_threshold),
            temp_threshold=float(cfg.cold_corrector_temp_threshold),
            gate_sharpness=float(cfg.cold_corrector_gate_sharpness),
            sag_gate=bool(cfg.cold_corrector_sag_gate),
            dynamic_center=float(cfg.high_dynamic_corrector_center),
            dynamic_sharpness=float(cfg.high_dynamic_corrector_sharpness),
            istd_center=float(cfg.high_dynamic_corrector_istd_center),
            istd_sharpness=float(cfg.high_dynamic_corrector_istd_sharpness),
        ).to(device)
    if str(cfg.model_kind) == "normal_low_voltage_tail_sequence":
        return NormalLowVoltageTailCorrectionModel(
            input_dim=input_dim,
            feature_cols=list(feature_cols or _selected_feature_columns(str(cfg.feature_set))),
            hidden_size=int(cfg.hidden_size),
            layers=int(cfg.layers),
            kernel_size=int(cfg.kernel_size),
            dropout=float(cfg.dropout),
            recurrent=str(cfg.recurrent),
            head_kind=str(cfg.head_kind),
            temp_mode=str(cfg.temp_mode),
            correction_limit=float(cfg.cold_corrector_limit),
            soc_threshold=float(cfg.cold_corrector_soc_threshold),
            temp_threshold=float(cfg.cold_corrector_temp_threshold),
            gate_sharpness=float(cfg.cold_corrector_gate_sharpness),
            sag_gate=bool(cfg.cold_corrector_sag_gate),
        ).to(device)
    if str(cfg.model_kind) == "normal_affine_calibration_sequence":
        return NormalAffineCalibrationModel(
            input_dim=input_dim,
            feature_cols=list(feature_cols or _selected_feature_columns(str(cfg.feature_set))),
            hidden_size=int(cfg.hidden_size),
            layers=int(cfg.layers),
            kernel_size=int(cfg.kernel_size),
            dropout=float(cfg.dropout),
            residual_limit=float(cfg.anchor_residual_limit),
            recurrent=str(cfg.recurrent),
            head_kind=str(cfg.head_kind),
            temp_mode=str(cfg.temp_mode),
        ).to(device)
    if str(cfg.model_kind) == "cold_regime_signed_bias_sequence":
        return ColdRegimeSignedBiasModel(
            input_dim=input_dim,
            feature_cols=list(feature_cols or _selected_feature_columns(str(cfg.feature_set))),
            hidden_size=int(cfg.hidden_size),
            layers=int(cfg.layers),
            kernel_size=int(cfg.kernel_size),
            dropout=float(cfg.dropout),
            recurrent=str(cfg.recurrent),
            head_kind=str(cfg.head_kind),
            temp_mode=str(cfg.temp_mode),
            correction_limit=float(cfg.cold_corrector_limit),
            temp_threshold=float(cfg.cold_corrector_temp_threshold),
            gate_sharpness=float(cfg.cold_corrector_gate_sharpness),
        ).to(device)
    if str(cfg.model_kind) == "regime_gated_normal_anchor_signed_sequence":
        return RegimeGatedNormalAnchorSignedModel(
            input_dim=input_dim,
            feature_cols=list(feature_cols or _selected_feature_columns(str(cfg.feature_set))),
            hidden_size=int(cfg.hidden_size),
            layers=int(cfg.layers),
            kernel_size=int(cfg.kernel_size),
            dropout=float(cfg.dropout),
            residual_limit=float(cfg.anchor_residual_limit),
            residual_limit_init=str(cfg.anchor_residual_limit_init),
            residual_limit_mode=str(cfg.anchor_residual_limit_mode),
            residual_limit_lower=float(cfg.anchor_residual_limit_lower),
            residual_limit_upper=float(cfg.anchor_residual_limit_upper),
            recurrent=str(cfg.recurrent),
            head_kind=str(cfg.head_kind),
            temp_mode=str(cfg.temp_mode),
            correction_limit=float(cfg.cold_corrector_limit),
            soc_threshold=float(cfg.cold_corrector_soc_threshold),
            temp_threshold=float(cfg.cold_corrector_temp_threshold),
            gate_sharpness=float(cfg.cold_corrector_gate_sharpness),
        ).to(device)
    if str(cfg.model_kind) == "anchor_residual_vt_additive_mlp_sequence":
        return VTSequenceAdditiveMLPResidualModel(
            input_dim=input_dim,
            feature_cols=list(feature_cols or _selected_feature_columns(str(cfg.feature_set))),
            hidden_size=int(cfg.hidden_size),
            layers=int(cfg.layers),
            kernel_size=int(cfg.kernel_size),
            dropout=float(cfg.dropout),
            residual_limit=float(cfg.anchor_residual_limit),
            residual_limit_init=str(cfg.anchor_residual_limit_init),
            residual_limit_mode=str(cfg.anchor_residual_limit_mode),
            residual_limit_lower=float(cfg.anchor_residual_limit_lower),
            residual_limit_upper=float(cfg.anchor_residual_limit_upper),
            recurrent=str(cfg.recurrent),
            correction_limit=float(cfg.cold_corrector_limit),
        ).to(device)
    if str(cfg.model_kind) == "anchor_residual_vt_input_additive_mlp_sequence":
        return VTInputAdditiveMLPResidualModel(
            input_dim=input_dim,
            feature_cols=list(feature_cols or _selected_feature_columns(str(cfg.feature_set))),
            hidden_size=int(cfg.hidden_size),
            layers=int(cfg.layers),
            kernel_size=int(cfg.kernel_size),
            dropout=float(cfg.dropout),
            residual_limit=float(cfg.anchor_residual_limit),
            residual_limit_init=str(cfg.anchor_residual_limit_init),
            residual_limit_mode=str(cfg.anchor_residual_limit_mode),
            residual_limit_lower=float(cfg.anchor_residual_limit_lower),
            residual_limit_upper=float(cfg.anchor_residual_limit_upper),
            recurrent=str(cfg.recurrent),
            correction_limit=float(cfg.cold_corrector_limit),
        ).to(device)
    if str(cfg.model_kind) in {
        "anchor_residual_tbr_sequence",
        "anchor_residual_tbr_regime_sequence",
        "anchor_residual_tbr_transition_sequence",
        "anchor_residual_tbr_tzgate_sequence",
        "anchor_residual_tbr_interp25_sequence",
        "anchor_residual_tbr_interp25_tzgate_sequence",
        "anchor_residual_tbr_interp25_tzgate_transition_sequence",
        "anchor_residual_tbr_interp25_tzgate_dynscale_sequence",
        "anchor_residual_tbr_interp25_tzgate_dynscale_lstm_mha_sequence",
        "anchor_residual_tbr_interp25_socbasis_sequence",
        "anchor_residual_tbr_socbasis_sequence",
    }:
        model_kind = str(cfg.model_kind)
        return TemperatureBandedAnchorResidualModel(
            input_dim=input_dim,
            feature_cols=list(feature_cols or _selected_feature_columns(str(cfg.feature_set))),
            hidden_size=int(cfg.hidden_size),
            layers=int(cfg.layers),
            kernel_size=int(cfg.kernel_size),
            dropout=float(cfg.dropout),
            residual_limit=float(cfg.anchor_residual_limit),
            residual_limit_init=str(cfg.anchor_residual_limit_init),
            residual_limit_mode=str(cfg.anchor_residual_limit_mode),
            residual_limit_lower=float(cfg.anchor_residual_limit_lower),
            residual_limit_upper=float(cfg.anchor_residual_limit_upper),
            recurrent=str(cfg.recurrent),
            mid_regime_dynamic=model_kind == "anchor_residual_tbr_regime_sequence",
            transition_branch=model_kind
            in {
                "anchor_residual_tbr_transition_sequence",
                "anchor_residual_tbr_interp25_tzgate_transition_sequence",
            },
            mid_interpolation=model_kind
            in {
                "anchor_residual_tbr_interp25_sequence",
                "anchor_residual_tbr_interp25_tzgate_sequence",
                "anchor_residual_tbr_interp25_tzgate_transition_sequence",
                "anchor_residual_tbr_interp25_tzgate_dynscale_sequence",
                "anchor_residual_tbr_interp25_tzgate_dynscale_lstm_mha_sequence",
                "anchor_residual_tbr_interp25_socbasis_sequence",
            },
            soc_basis=model_kind
            in {
                "anchor_residual_tbr_socbasis_sequence",
                "anchor_residual_tbr_interp25_socbasis_sequence",
            },
            dynamic_residual_scale=model_kind
            in {
                "anchor_residual_tbr_interp25_tzgate_dynscale_sequence",
                "anchor_residual_tbr_interp25_tzgate_dynscale_lstm_mha_sequence",
            },
            mha_pooling=model_kind == "anchor_residual_tbr_interp25_tzgate_dynscale_lstm_mha_sequence",
            gate_mode="tz"
            if model_kind
            in {
                "anchor_residual_tbr_tzgate_sequence",
                "anchor_residual_tbr_interp25_tzgate_sequence",
                "anchor_residual_tbr_interp25_tzgate_transition_sequence",
                "anchor_residual_tbr_interp25_tzgate_dynscale_sequence",
                "anchor_residual_tbr_interp25_tzgate_dynscale_lstm_mha_sequence",
            }
            else "temperature",
        ).to(device)
    if str(cfg.model_kind) == "base_mid_residual_sequence":
        return BaseMidResidualModel(
            input_dim=input_dim,
            feature_cols=list(feature_cols or _selected_feature_columns(str(cfg.feature_set))),
            hidden_size=int(cfg.hidden_size),
            layers=int(cfg.layers),
            kernel_size=int(cfg.kernel_size),
            dropout=float(cfg.dropout),
            residual_limit=float(cfg.anchor_residual_limit),
            recurrent=str(cfg.recurrent),
            head_kind=str(cfg.head_kind),
        ).to(device)
    if str(cfg.model_kind) == "endpoint_mlp":
        return MLPControlModel(
            input_dim=input_dim,
            hidden_size=int(cfg.hidden_size),
            dropout=float(cfg.dropout),
            mode="endpoint",
        ).to(device)
    if str(cfg.model_kind) == "window_summary_mlp":
        return MLPControlModel(
            input_dim=input_dim,
            hidden_size=int(cfg.hidden_size),
            dropout=float(cfg.dropout),
            mode="window_summary",
        ).to(device)
    if str(cfg.model_kind) == "anchor_residual_endpoint_mlp":
        return AnchorResidualMLPModel(
            input_dim=input_dim,
            feature_cols=list(feature_cols or _selected_feature_columns(str(cfg.feature_set))),
            hidden_size=int(cfg.hidden_size),
            dropout=float(cfg.dropout),
            residual_limit=float(cfg.anchor_residual_limit),
            residual_limit_init=str(cfg.anchor_residual_limit_init),
            residual_limit_mode=str(cfg.anchor_residual_limit_mode),
            residual_limit_lower=float(cfg.anchor_residual_limit_lower),
            residual_limit_upper=float(cfg.anchor_residual_limit_upper),
            mode="endpoint",
        ).to(device)
    if str(cfg.model_kind) == "anchor_residual_window_summary_mlp":
        return AnchorResidualMLPModel(
            input_dim=input_dim,
            feature_cols=list(feature_cols or _selected_feature_columns(str(cfg.feature_set))),
            hidden_size=int(cfg.hidden_size),
            dropout=float(cfg.dropout),
            residual_limit=float(cfg.anchor_residual_limit),
            residual_limit_init=str(cfg.anchor_residual_limit_init),
            residual_limit_mode=str(cfg.anchor_residual_limit_mode),
            residual_limit_lower=float(cfg.anchor_residual_limit_lower),
            residual_limit_upper=float(cfg.anchor_residual_limit_upper),
            mode="window_summary",
        ).to(device)
    if str(cfg.model_kind) == "single_mha_endpoint":
        return LastQueryMHAEndpointModel(
            input_dim=input_dim,
            hidden_size=int(cfg.hidden_size),
            recurrent=str(cfg.recurrent),
            layers=int(cfg.layers),
            head_kind=str(cfg.head_kind),
            temp_mode=str(cfg.temp_mode),
            dropout=float(cfg.dropout),
            kernel_size=int(cfg.kernel_size),
        ).to(device)
    if str(cfg.model_kind) == "single_multiscale_endpoint":
        return CausalMultiScaleEndpointModel(
            input_dim=input_dim,
            hidden_size=int(cfg.hidden_size),
            recurrent=str(cfg.recurrent),
            layers=int(cfg.layers),
            head_kind=str(cfg.head_kind),
            temp_mode=str(cfg.temp_mode),
            dropout=float(cfg.dropout),
            kernel_size=int(cfg.kernel_size),
            feature_cols=list(feature_cols or _selected_feature_columns(str(cfg.feature_set))),
            local_len=int(cfg.multiscale_local_len),
        ).to(device)
    if str(cfg.model_kind) == "single_dual_context_endpoint":
        return DualContextEndpointModel(
            input_dim=input_dim,
            hidden_size=int(cfg.hidden_size),
            recurrent=str(cfg.recurrent),
            layers=int(cfg.layers),
            head_kind=str(cfg.head_kind),
            temp_mode=str(cfg.temp_mode),
            dropout=float(cfg.dropout),
            kernel_size=int(cfg.kernel_size),
            feature_cols=list(feature_cols or _selected_feature_columns(str(cfg.feature_set))),
            local_len=int(cfg.multiscale_local_len),
        ).to(device)
    if str(cfg.model_kind) == "single_dual_context_dynamic_gate_endpoint":
        return DualContextEndpointModel(
            input_dim=input_dim,
            hidden_size=int(cfg.hidden_size),
            recurrent=str(cfg.recurrent),
            layers=int(cfg.layers),
            head_kind=str(cfg.head_kind),
            temp_mode=str(cfg.temp_mode),
            dropout=float(cfg.dropout),
            kernel_size=int(cfg.kernel_size),
            feature_cols=list(feature_cols or _selected_feature_columns(str(cfg.feature_set))),
            local_len=int(cfg.multiscale_local_len),
            gate_mode="dynamic",
            dynamic_gate_center=float(cfg.dual_context_gate_center),
            dynamic_gate_sharpness=float(cfg.dual_context_gate_sharpness),
            dynamic_gate_local_min=float(cfg.dual_context_gate_local_min),
            dynamic_gate_local_max=float(cfg.dual_context_gate_local_max),
        ).to(device)
    if str(cfg.model_kind) == "single_dual_context_fusion_endpoint":
        return DualContextEndpointModel(
            input_dim=input_dim,
            hidden_size=int(cfg.hidden_size),
            recurrent=str(cfg.recurrent),
            layers=int(cfg.layers),
            head_kind=str(cfg.head_kind),
            temp_mode=str(cfg.temp_mode),
            dropout=float(cfg.dropout),
            kernel_size=int(cfg.kernel_size),
            feature_cols=list(feature_cols or _selected_feature_columns(str(cfg.feature_set))),
            local_len=int(cfg.multiscale_local_len),
            gate_mode="fusion",
            fusion_delta_limit=float(cfg.dual_context_fusion_delta_limit),
        ).to(device)
    if str(cfg.model_kind) == "single_dual_context_eqdyn_g4global_endpoint":
        selected_cols = list(feature_cols or _selected_feature_columns(str(cfg.feature_set)))
        return DualContextEndpointModel(
            input_dim=input_dim,
            hidden_size=int(cfg.hidden_size),
            recurrent=str(cfg.recurrent),
            layers=int(cfg.layers),
            head_kind=str(cfg.head_kind),
            temp_mode=str(cfg.temp_mode),
            dropout=float(cfg.dropout),
            kernel_size=int(cfg.kernel_size),
            feature_cols=selected_cols,
            local_len=int(cfg.multiscale_local_len),
            global_feature_cols=_selected_feature_columns("paper_g4_all_ema"),
        ).to(device)
    if str(cfg.model_kind) == "single_dual_context_dynamic_gate_eqdyn_g4full_endpoint":
        selected_cols = list(feature_cols or _selected_feature_columns(str(cfg.feature_set)))
        return DualContextEndpointModel(
            input_dim=input_dim,
            hidden_size=int(cfg.hidden_size),
            recurrent=str(cfg.recurrent),
            layers=int(cfg.layers),
            head_kind=str(cfg.head_kind),
            temp_mode=str(cfg.temp_mode),
            dropout=float(cfg.dropout),
            kernel_size=int(cfg.kernel_size),
            feature_cols=selected_cols,
            local_len=int(cfg.multiscale_local_len),
            global_feature_cols=_selected_feature_columns("paper_g4_all_ema"),
            gate_mode="dynamic",
            dynamic_gate_center=float(cfg.dual_context_gate_center),
            dynamic_gate_sharpness=float(cfg.dual_context_gate_sharpness),
            dynamic_gate_local_min=float(cfg.dual_context_gate_local_min),
            dynamic_gate_local_max=float(cfg.dual_context_gate_local_max),
        ).to(device)
    if str(cfg.model_kind) == "single_dual_context_fusion_eqdyn_g4full_endpoint":
        selected_cols = list(feature_cols or _selected_feature_columns(str(cfg.feature_set)))
        return DualContextEndpointModel(
            input_dim=input_dim,
            hidden_size=int(cfg.hidden_size),
            recurrent=str(cfg.recurrent),
            layers=int(cfg.layers),
            head_kind=str(cfg.head_kind),
            temp_mode=str(cfg.temp_mode),
            dropout=float(cfg.dropout),
            kernel_size=int(cfg.kernel_size),
            feature_cols=selected_cols,
            local_len=int(cfg.multiscale_local_len),
            global_feature_cols=_selected_feature_columns("paper_g4_all_ema"),
            gate_mode="fusion",
            fusion_delta_limit=float(cfg.dual_context_fusion_delta_limit),
        ).to(device)
    if str(cfg.model_kind) == "single":
        model = make_model(variant, CondInvConfig(hidden_size=cfg.hidden_size, window_len=cfg.window_len))
        if input_dim == len(FEATURE_COLS):
            return model
        return VcorrITGoalModel(
            input_dim=input_dim,
            hidden_size=int(cfg.hidden_size),
            recurrent=variant.recurrent,
            layers=int(variant.layers),
            head_kind=variant.head_kind,
            temp_mode=variant.temp_mode,
            dropout=float(variant.dropout),
            kernel_size=int(variant.kernel_size),
            norm_kind="channel",
        ).to(device)
    raise ValueError(f"Unknown model_kind={cfg.model_kind!r}")


def _metric_value(metrics: pd.DataFrame, split: str, temp: float, drive: str) -> float:
    if metrics.empty or "split" not in metrics.columns:
        return float("inf")
    sub = metrics[
        metrics["split"].eq(split)
        & np.isclose(metrics["temperature_C"], float(temp))
        & metrics["drive_cycle"].astype(str).eq(str(drive))
    ]
    if not len(sub):
        return float("inf")
    return float(sub["MAE_pct"].iloc[0])


def _temp_metric_value(metrics: pd.DataFrame, split: str, temp: float) -> float:
    if metrics.empty or "split" not in metrics.columns:
        return float("inf")
    sub = metrics[metrics["split"].eq(split) & np.isclose(metrics["temperature_C"], float(temp))]
    if not len(sub):
        return float("inf")
    return float(sub["MAE_pct"].iloc[0])


def _finite_mean(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("inf")


def _finite_max(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.max(finite)) if finite else float("inf")


def _target_for_temp(temp: float) -> float:
    if np.isclose(float(temp), 0.0):
        return 1.0
    if np.isclose(float(temp), 25.0):
        return 0.7
    if np.isclose(float(temp), 45.0):
        return 0.3
    return 1.0


def _regime_selector_values(valid_regime_metrics: pd.DataFrame | None, min_windows: int) -> tuple[float, float, int, int]:
    if valid_regime_metrics is None or valid_regime_metrics.empty:
        return float("inf"), float("inf"), 0, 0
    df = valid_regime_metrics.copy()
    if "temperature_C" not in df.columns or "MAE_pct" not in df.columns:
        return float("inf"), float("inf"), 0, 0
    df["target_norm"] = [
        float(mae) / _target_for_temp(float(temp))
        for mae, temp in zip(df["MAE_pct"], df["temperature_C"])
    ]
    total = int(len(df))
    if "n_windows" in df.columns:
        df = df[pd.to_numeric(df["n_windows"], errors="coerce").fillna(0).ge(int(min_windows))]
    finite = df[np.isfinite(df["target_norm"])]
    if finite.empty:
        return float("inf"), float("inf"), 0, total
    return float(finite["target_norm"].mean()), float(finite["target_norm"].max()), int(len(finite)), total


def _selector_score(
    train_metrics: pd.DataFrame,
    rule: str,
    valid_metrics: pd.DataFrame | None = None,
    valid_regime_metrics: pd.DataFrame | None = None,
    regime_min_windows: int = 30,
) -> dict[str, float]:
    dst25 = _metric_value(train_metrics, "train", 25.0, "DST")
    us0625 = _metric_value(train_metrics, "train", 25.0, "US06")
    vals = [dst25, us0625]
    mean25 = _finite_mean(vals)
    worst25 = _finite_max(vals)
    gap25 = float(abs(dst25 - us0625)) if np.isfinite(dst25) and np.isfinite(us0625) else float("inf")
    valid_metrics = valid_metrics if valid_metrics is not None else pd.DataFrame()
    valid0 = _temp_metric_value(valid_metrics, "valid", 0.0)
    valid25 = _temp_metric_value(valid_metrics, "valid", 25.0)
    valid45 = _temp_metric_value(valid_metrics, "valid", 45.0)
    valid_vals = [valid0, valid25, valid45]
    valid_mean = _finite_mean(valid_vals)
    valid_worst = _finite_max(valid_vals)
    valid_finite = [v for v in valid_vals if np.isfinite(v)]
    valid_spread = float(valid_worst - min(valid_finite)) if valid_finite else float("inf")
    valid_hot_gap = float(max(0.0, valid45 - valid25)) if np.isfinite(valid45) and np.isfinite(valid25) else float("inf")
    valid_target_norm = [
        valid0 / 1.0 if np.isfinite(valid0) else float("inf"),
        valid25 / 0.7 if np.isfinite(valid25) else float("inf"),
        valid45 / 0.3 if np.isfinite(valid45) else float("inf"),
    ]
    valid_target_norm_mean = _finite_mean(valid_target_norm)
    valid_target_norm_worst = _finite_max(valid_target_norm)
    (
        valid_regime_target_norm_mean,
        valid_regime_target_norm_worst,
        valid_regime_slices_used,
        valid_regime_slices_total,
    ) = _regime_selector_values(valid_regime_metrics, int(regime_min_windows))
    if rule == "train25_dst":
        score = dst25
    elif rule == "train25_us06":
        score = us0625
    elif rule == "train25_mean_drive":
        score = mean25
    elif rule == "train25_worst_drive":
        score = worst25
    elif rule == "train25_worst_gap":
        score = worst25 + 0.5 * gap25
    elif rule == "valid_worst_temp":
        score = valid_worst
    elif rule == "val_mean_mae":
        score = valid_mean
    elif rule == "val_worst_mae":
        score = valid_worst
    elif rule == "val_mean_plus_worst":
        score = valid_mean + 0.5 * valid_worst
    elif rule == "val_target_worst":
        score = valid_target_norm_worst
    elif rule == "val_target_mean_plus_worst":
        score = valid_target_norm_mean + 0.5 * valid_target_norm_worst
    elif rule == "val_regime_target_worst":
        score = valid_regime_target_norm_worst
    elif rule == "val_regime_target_mean_plus_worst":
        score = valid_regime_target_norm_mean + 0.5 * valid_regime_target_norm_worst
    elif rule == "valid25_temp":
        score = valid25
    elif rule == "last_epoch":
        if len(valid_metrics) and "epoch" in valid_metrics:
            score = -float(valid_metrics["epoch"].iloc[0])
        elif len(train_metrics) and "epoch" in train_metrics:
            score = -float(train_metrics["epoch"].iloc[0])
        else:
            score = 0.0
    elif rule == "train25_dst_valid_worst":
        score = dst25 + 0.7 * valid_worst
    elif rule == "train25_worst_valid_worst":
        score = worst25 + 0.7 * valid_worst
    elif rule == "train25_dst_valid45_guard":
        score = dst25 + 0.7 * valid45 + 0.3 * valid_hot_gap
    elif rule == "train25_worst_valid_balance":
        score = worst25 + 0.5 * gap25 + 0.6 * valid_worst + 0.2 * valid_spread
    else:
        raise ValueError(f"Unknown stage1_selector={rule!r}")
    return {
        "selector_score": float(score),
        "selector_train25_dst": float(dst25),
        "selector_train25_us06": float(us0625),
        "selector_train25_mean_drive": float(mean25),
        "selector_train25_worst_drive": float(worst25),
        "selector_train25_gap": float(gap25),
        "selector_valid0": float(valid0),
        "selector_valid25": float(valid25),
        "selector_valid45": float(valid45),
        "selector_valid_mean": float(valid_mean),
        "selector_valid_worst": float(valid_worst),
        "selector_valid_spread": float(valid_spread),
        "selector_valid_hot_gap": float(valid_hot_gap),
        "selector_valid_target_norm_mean": float(valid_target_norm_mean),
        "selector_valid_target_norm_worst": float(valid_target_norm_worst),
        "selector_valid_regime_target_norm_mean": float(valid_regime_target_norm_mean),
        "selector_valid_regime_target_norm_worst": float(valid_regime_target_norm_worst),
        "selector_valid_regime_min_windows": int(regime_min_windows),
        "selector_valid_regime_slices_used": int(valid_regime_slices_used),
        "selector_valid_regime_slices_total": int(valid_regime_slices_total),
    }


def _standard_train_loader(ds, cfg: TrainDSTSelectorConfig, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    cuda_cached = bool(getattr(ds, "cuda_cached", False))
    num_workers = 0 if cuda_cached else int(cfg.num_workers)
    kwargs = {
        "batch_size": int(cfg.batch_size),
        "shuffle": True,
        "generator": generator,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda" and not cuda_cached,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    if cuda_cached:
        kwargs["collate_fn"] = _identity_collate
    return DataLoader(ds, **kwargs)


def _identity_collate(batch):
    return batch


class PrecomputedWeightedBatchSampler(Sampler[torch.Tensor]):
    def __init__(self, weights: torch.Tensor, *, batch_size: int, epochs: int, seed: int):
        self.batch_size = int(batch_size)
        self.num_samples = int(weights.numel())
        self.epochs = max(1, int(epochs))
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        probs = weights.detach().to(device="cpu", dtype=torch.double)
        chunks = [
            torch.multinomial(probs, self.num_samples, replacement=True, generator=generator).to(dtype=torch.long)
            for _ in range(self.epochs)
        ]
        self.indices = torch.stack(chunks, dim=0)
        self.cursor = 0
        self.num_batches = int(np.ceil(self.num_samples / max(1, self.batch_size)))

    def __iter__(self):
        if self.cursor >= self.epochs:
            self.cursor = 0
        epoch_indices = self.indices[self.cursor]
        self.cursor += 1
        for start in range(0, self.num_samples, self.batch_size):
            yield epoch_indices[start : start + self.batch_size]

    def __len__(self):
        return self.num_batches


def _soc_bin4(soc: float) -> str:
    value = float(soc)
    if value < 0.2:
        return "soc0_0_20"
    if value < 0.5:
        return "soc1_20_50"
    if value < 0.8:
        return "soc2_50_80"
    return "soc3_80_100"


def _endpoint_balance_keys(ds, mode: str) -> list[tuple]:
    keys = []
    for fi, _start, end in ds.index:
        frame = ds.frames[fi]
        temp = round(float(frame["temperature"][end]), 3)
        drive = str(frame["drive_cycle"][end]).upper()
        if str(mode) == "temperature_profile_balanced":
            keys.append((temp, drive))
        elif str(mode) == "temperature_profile_soc_balanced":
            keys.append((temp, drive, _soc_bin4(float(frame["y_physical"][end]))))
        else:
            raise ValueError(f"Unknown profile balance mode={mode!r}")
    return keys


def _metric_bin(value: float, edges: tuple[float, float]) -> str:
    if not np.isfinite(value):
        return "nan"
    lo, hi = edges
    if value <= lo:
        return "low"
    if value <= hi:
        return "mid"
    return "high"


def _regime_edges(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return (0.0, 0.0)
    lo, hi = np.nanquantile(arr, [1.0 / 3.0, 2.0 / 3.0])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (0.0, 0.0)
    if hi <= lo:
        hi = lo + 1e-6
    return (float(lo), float(hi))


def _window_regime_rows(ds) -> pd.DataFrame:
    rows = []
    feature_to_idx = {name: idx for idx, name in enumerate(getattr(ds, "feature_cols", []))}

    def get_feature(frame, col: str, sl: slice) -> np.ndarray | None:
        idx = feature_to_idx.get(col)
        if idx is None:
            return None
        return np.asarray(frame["x"][sl, idx], dtype=np.float64)

    for row_id, (fi, start, end) in enumerate(ds.index):
        frame = ds.frames[fi]
        sl = slice(int(start), int(end) + 1)
        temp = round(float(frame["temperature"][end]), 3)
        drive = str(frame["drive_cycle"][end]).upper()
        i = get_feature(frame, "I_raw", sl)
        if i is None:
            raise RuntimeError("Regime sampler requires I_raw in the selected feature set.")
        abs_i = get_feature(frame, "absI", sl)
        if abs_i is None:
            abs_i = np.abs(i)
        di = get_feature(frame, "dI", sl)
        if di is None:
            di = np.diff(i, prepend=i[0] if len(i) else 0.0)
        v = get_feature(frame, "V_corr_raw", sl)
        if v is None:
            v = get_feature(frame, "V_eq_slow_raw", sl)
        if v is None:
            v = get_feature(frame, "V_ohm_free_raw", sl)
        if v is None:
            raise RuntimeError("Regime sampler requires a voltage proxy feature in the selected feature set.")
        rows.append(
            {
                "row_id": int(row_id),
                "temperature_C": float(temp),
                "drive_cycle": drive,
                "absI_mean": float(np.nanmean(np.abs(abs_i))),
                "I_std": float(np.nanstd(i)),
                "dI_energy": float(np.nanmean(np.square(di))),
                "V_corr_span": float(np.nanmax(v) - np.nanmin(v)) if len(v) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _regime_balance_keys(ds, mode: str) -> tuple[list[tuple], pd.DataFrame, pd.DataFrame]:
    cache = getattr(ds, "_regime_balance_cache", None)
    if cache is None:
        cache = {}
        setattr(ds, "_regime_balance_cache", cache)
    if str(mode) in cache:
        return cache[str(mode)]
    rows = _window_regime_rows(ds)
    metric_cols = ["absI_mean", "I_std", "dI_energy", "V_corr_span"]
    edges = {col: _regime_edges(rows[col].tolist()) for col in metric_cols}
    for col in metric_cols:
        rows[f"{col}_bin"] = [_metric_bin(v, edges[col]) for v in rows[col]]
    if mode == "temperature_profile_soc_regime_balanced":
        soc_bins = []
        for fi, _start, end in ds.index:
            frame = ds.frames[fi]
            soc_bins.append(_soc_bin4(float(frame["y_physical"][end])))
        rows["SOC_bin4"] = soc_bins
    key_cols = ["temperature_C", "drive_cycle", "absI_mean_bin", "I_std_bin", "dI_energy_bin", "V_corr_span_bin"]
    if mode == "temperature_regime_balanced":
        key_cols = ["temperature_C", "absI_mean_bin", "I_std_bin", "dI_energy_bin", "V_corr_span_bin"]
    elif mode == "temperature_profile_soc_regime_balanced":
        key_cols = [
            "temperature_C",
            "drive_cycle",
            "SOC_bin4",
            "absI_mean_bin",
            "I_std_bin",
            "dI_energy_bin",
            "V_corr_span_bin",
        ]
    elif mode != "temperature_profile_regime_balanced":
        raise ValueError(f"Unknown regime balance mode={mode!r}")
    keys = [tuple(row[col] for col in key_cols) for _, row in rows.iterrows()]
    edge_rows = [
        {"metric": col, "low_mid_edge": edges[col][0], "mid_high_edge": edges[col][1]}
        for col in metric_cols
    ]
    counts = rows.assign(regime_key=[repr(k) for k in keys]).groupby("regime_key", as_index=False).size()
    counts = counts.rename(columns={"size": "n_windows"}).sort_values(["n_windows", "regime_key"]).reset_index(drop=True)
    result = (keys, pd.DataFrame(edge_rows), counts)
    cache[str(mode)] = result
    return result


def _write_regime_sampler_audit(ds, cfg: TrainDSTSelectorConfig, out_dir: Path, seed: int) -> None:
    if str(cfg.train_sampler) not in {
        "temperature_regime_balanced",
        "temperature_profile_regime_balanced",
        "temperature_profile_soc_regime_balanced",
    }:
        return
    _keys, edges, counts = _regime_balance_keys(ds, str(cfg.train_sampler))
    prefix = f"{cfg.output_prefix}_seed{seed}_{cfg.train_sampler}"
    edges.to_csv(out_dir / f"{prefix}_regime_edges.csv", index=False)
    counts.to_csv(out_dir / f"{prefix}_regime_counts.csv", index=False)


def _profile_balanced_train_loader(ds, cfg: TrainDSTSelectorConfig, seed: int, mode: str) -> DataLoader:
    weight_cache = getattr(ds, "_balance_weight_cache", None)
    if weight_cache is None:
        weight_cache = {}
        setattr(ds, "_balance_weight_cache", weight_cache)
    weight_key = str(mode)
    if weight_key in weight_cache:
        weights = weight_cache[weight_key]
    else:
        if str(mode) in {
            "temperature_regime_balanced",
            "temperature_profile_regime_balanced",
            "temperature_profile_soc_regime_balanced",
        }:
            keys, _edges, _counts = _regime_balance_keys(ds, str(mode))
        else:
            keys = _endpoint_balance_keys(ds, mode)
        counts = {}
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
        weights = torch.as_tensor([1.0 / counts[key] for key in keys], dtype=torch.double)
        weight_cache[weight_key] = weights
    cuda_cached = bool(getattr(ds, "cuda_cached", False))
    num_workers = 0 if cuda_cached else int(cfg.num_workers)
    if cuda_cached:
        sampler_epochs = max(1, int(cfg.epochs) + int(cfg.finetune_epochs), int(cfg.stage2_epochs))
        batch_sampler = PrecomputedWeightedBatchSampler(
            weights,
            batch_size=int(cfg.batch_size),
            epochs=sampler_epochs,
            seed=int(seed),
        )
        kwargs = {
            "batch_sampler": batch_sampler,
            "num_workers": 0,
            "pin_memory": False,
            "collate_fn": _identity_collate,
        }
    else:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
        kwargs = {
            "batch_size": int(cfg.batch_size),
            "sampler": sampler,
            "num_workers": num_workers,
            "pin_memory": device.type == "cuda",
        }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    return DataLoader(ds, **kwargs)


def _make_train_loader(ds, base_cfg, cfg: TrainDSTSelectorConfig, seed: int):
    loader_seed = int(seed) if str(cfg.sampler_seed_mode) == "seed" else 0
    if str(cfg.train_sampler) == "standard":
        return _standard_train_loader(ds, cfg, int(loader_seed))
    if str(cfg.train_sampler) == "temperature_balanced":
        return temperature_balanced_loader(ds, base_cfg, shuffle=True)
    if str(cfg.train_sampler) in {
        "temperature_profile_balanced",
        "temperature_profile_soc_balanced",
        "temperature_regime_balanced",
        "temperature_profile_regime_balanced",
        "temperature_profile_soc_regime_balanced",
    }:
        return _profile_balanced_train_loader(ds, cfg, int(loader_seed), str(cfg.train_sampler))
    raise ValueError(f"Unknown train_sampler={cfg.train_sampler!r}")


def _make_eval_loader(ds, cfg: TrainDSTSelectorConfig):
    if len(ds) == 0:
        return None
    if bool(getattr(ds, "cuda_cached", False)):
        return DataLoader(
            ds,
            batch_size=int(cfg.batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=_identity_collate,
        )
    return make_eval_loader(ds, cfg)


def _group_loss_weight_stacks(sample_loss: torch.Tensor, sample_weight: torch.Tensor, meta, group_name: str):
    keys = group_keys(meta, group_name)
    group_losses = []
    group_weights = []
    if torch.is_tensor(keys):
        group_ids = keys.to(device=sample_loss.device, dtype=torch.long).reshape(-1)
        if int(group_ids.numel()) != int(sample_loss.numel()):
            return sample_loss.mean().reshape(1), sample_weight.mean().reshape(1)
        for group_id in torch.unique(group_ids).unbind(0):
            mask = group_ids.eq(group_id)
            group_losses.append(sample_loss[mask].mean())
            group_weights.append(sample_weight[mask].mean())
    else:
        for key in sorted(set(keys)):
            idx = torch.as_tensor([i for i, k in enumerate(keys) if k == key], device=sample_loss.device, dtype=torch.long)
            group_losses.append(sample_loss.index_select(0, idx).mean())
            group_weights.append(sample_weight.index_select(0, idx).mean())
    if not group_losses:
        return sample_loss.mean().reshape(1), sample_weight.mean().reshape(1)
    return torch.stack(group_losses), torch.stack(group_weights)


def _eval_by_temp_or_empty(model: nn.Module, loader, split: str, variant_name: str, epoch: int) -> pd.DataFrame:
    if loader is None:
        return pd.DataFrame()
    return eval_by_temp(model, loader, split, variant_name, epoch)


class CudaCachedWindowDataset(Dataset):
    def __init__(self, base: Dataset, name: str):
        if device.type != "cuda":
            raise RuntimeError("CudaCachedWindowDataset requires a CUDA device.")
        self.base = base
        self.name = str(name)
        self.cuda_cached = True
        self.frames = getattr(base, "frames", [])
        self.feature_cols = list(getattr(base, "feature_cols", []))
        self.window_len = int(getattr(base, "window_len", 0))
        self.stride = int(getattr(base, "stride", 1))
        self.target_label = str(getattr(base, "target_label", "physical"))
        self.index = list(getattr(base, "index", []))
        xs = []
        ys = []
        metas = []
        for idx in range(len(base)):
            x, y, meta = base[idx]
            xs.append(x)
            ys.append(y)
            metas.append(dict(meta))
        if xs:
            self.x = torch.stack(xs, dim=0).to(device=device, dtype=torch.float32)
            self.y = torch.stack(ys, dim=0).to(device=device, dtype=torch.float32)
        else:
            self.x = torch.empty((0, self.window_len, len(self.feature_cols)), device=device, dtype=torch.float32)
            y_shape = (self.window_len, 1) if isinstance(base, SequenceWindowDataset) else (1,)
            self.y = torch.empty((0, *y_shape), device=device, dtype=torch.float32)
        keys = sorted({key for meta in metas for key in meta})
        self.meta = {}
        self.meta_tensors = {}
        for key in keys:
            values = [meta.get(key) for meta in metas]
            if values and all(isinstance(v, (bool, int, float, np.integer, np.floating, np.bool_)) for v in values):
                is_integral = all(isinstance(v, (bool, int, np.integer, np.bool_)) for v in values)
                dtype = torch.long if is_integral else torch.float32
                self.meta_tensors[key] = torch.as_tensor(values, device=device, dtype=dtype)
            else:
                self.meta[key] = values
        drives = [str(meta.get("drive_cycle", "")).upper() for meta in metas]
        if drives:
            drive_lookup = {value: idx for idx, value in enumerate(sorted(set(drives)))}
            self.meta_tensors["drive_id"] = torch.as_tensor(
                [drive_lookup[value] for value in drives],
                device=device,
                dtype=torch.long,
            )
            self.meta_tensors["group_drive"] = self.meta_tensors["drive_id"].clone()
        temps = []
        for meta in metas:
            try:
                temps.append(round(float(meta.get("temperature")), 3))
            except (TypeError, ValueError):
                temps.append(float("nan"))
        finite_temps = sorted({temp for temp in temps if np.isfinite(temp)})
        if finite_temps:
            temp_lookup = {value: idx for idx, value in enumerate(finite_temps)}
            temp_ids = [temp_lookup.get(temp, len(finite_temps)) for temp in temps]
            self.meta_tensors["temperature_id"] = torch.as_tensor(temp_ids, device=device, dtype=torch.long)
            self.meta_tensors["group_temperature"] = self.meta_tensors["temperature_id"].clone()
        if "temperature_id" in self.meta_tensors and "drive_id" in self.meta_tensors:
            temp_ids_cpu = self.meta_tensors["temperature_id"].detach().cpu().tolist()
            drive_ids_cpu = self.meta_tensors["drive_id"].detach().cpu().tolist()
            combo_lookup = {
                combo: idx
                for idx, combo in enumerate(sorted(set(zip(temp_ids_cpu, drive_ids_cpu))))
            }
            self.meta_tensors["group_temperature_drive"] = torch.as_tensor(
                [combo_lookup[(t, d)] for t, d in zip(temp_ids_cpu, drive_ids_cpu)],
                device=device,
                dtype=torch.long,
            )
        self.cuda_cache_bytes = int(self.x.numel() * self.x.element_size() + self.y.numel() * self.y.element_size())

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        item = int(idx)
        meta = {key: values[item] for key, values in self.meta.items()}
        for key, values in self.meta_tensors.items():
            meta[key] = values[item]
        return self.x[item], self.y[item], meta

    def __getitems__(self, indices):
        if torch.is_tensor(indices):
            idx = indices.to(device=self.x.device, dtype=torch.long).reshape(-1)
            idx_cpu = idx.detach().cpu().tolist() if self.meta else []
        else:
            idx_cpu = [int(i) for i in indices]
            idx = torch.as_tensor(idx_cpu, device=self.x.device, dtype=torch.long)
        meta = {key: [values[i] for i in idx_cpu] for key, values in self.meta.items()}
        for key, values in self.meta_tensors.items():
            meta[key] = values.index_select(0, idx)
        return self.x.index_select(0, idx), self.y.index_select(0, idx), meta


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}GB"


def _epoch_iter(start: int, stop: int, cfg: TrainDSTSelectorConfig, desc: str):
    values = range(int(start), int(stop))
    if not bool(cfg.tqdm_epochs):
        return values
    try:
        from tqdm.auto import tqdm
    except ImportError:
        print("[tqdm unavailable] install tqdm or run without --tqdm-epochs", flush=True)
        return values
    return tqdm(values, desc=desc, unit="epoch", leave=True)


def _maybe_cuda_cache_dataset(ds, cfg: TrainDSTSelectorConfig, name: str):
    if not bool(cfg.cache_dataset_cuda):
        return ds
    if device.type != "cuda":
        print(f"[cuda-cache skip] {name}: device={device}", flush=True)
        return ds
    cached = CudaCachedWindowDataset(ds, name)
    print(
        f"[cuda-cache] {name}: windows={len(cached)} x_shape={tuple(cached.x.shape)} "
        f"y_shape={tuple(cached.y.shape)} tensors={_format_bytes(cached.cuda_cache_bytes)}",
        flush=True,
    )
    return cached


def _meta_float_tensor(meta, key: str) -> torch.Tensor:
    vals = meta[key]
    if torch.is_tensor(vals):
        return vals.to(device=device, dtype=torch.float32)
    return torch.as_tensor(vals, device=device, dtype=torch.float32)


def _batch_dynamic_regime_ids(x: torch.Tensor, meta, feature_cols: list[str]) -> torch.Tensor:
    feature_to_idx = {name: idx for idx, name in enumerate(feature_cols)}
    if "I_raw" not in feature_to_idx or "V_corr_raw" not in feature_to_idx:
        return torch.zeros(int(x.shape[0]), device=x.device, dtype=torch.long)
    current = x[:, :, feature_to_idx["I_raw"]]
    abs_current = x[:, :, feature_to_idx["absI"]].abs() if "absI" in feature_to_idx else current.abs()
    if "dI" in feature_to_idx:
        di = x[:, :, feature_to_idx["dI"]]
    else:
        di = torch.diff(current, dim=1, prepend=current[:, :1])
    vcorr = x[:, :, feature_to_idx["V_corr_raw"]]
    stats = torch.stack(
        [
            abs_current.mean(dim=1),
            current.std(dim=1, unbiased=False),
            di.square().mean(dim=1),
            vcorr.max(dim=1).values - vcorr.min(dim=1).values,
        ],
        dim=1,
    )
    dynamic_score = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0).mean(dim=1)
    if int(dynamic_score.numel()) >= 3:
        q = torch.quantile(dynamic_score.detach(), torch.tensor([1.0 / 3.0, 2.0 / 3.0], device=x.device, dtype=x.dtype))
        dynamic_bin = (dynamic_score > q[0]).long() + (dynamic_score > q[1]).long()
    else:
        dynamic_bin = torch.zeros_like(dynamic_score, dtype=torch.long)
    temps = _meta_float_tensor(meta, "temperature")
    temp_bin = torch.full_like(dynamic_bin, 3)
    temp_bin = torch.where(torch.isclose(temps, torch.zeros_like(temps), atol=0.75), torch.zeros_like(temp_bin), temp_bin)
    temp_bin = torch.where(torch.isclose(temps, torch.full_like(temps, 25.0), atol=0.75), torch.ones_like(temp_bin), temp_bin)
    temp_bin = torch.where(torch.isclose(temps, torch.full_like(temps, 45.0), atol=0.75), torch.full_like(temp_bin, 2), temp_bin)
    return temp_bin * 3 + dynamic_bin


def _regime_cvar_loss(
    sample_loss: torch.Tensor,
    sample_weight: torch.Tensor,
    x: torch.Tensor,
    meta,
    feature_cols: list[str],
    top_frac: float,
) -> torch.Tensor:
    group_ids = _batch_dynamic_regime_ids(x.detach(), meta, feature_cols)
    weighted_loss = sample_loss * sample_weight.detach()
    group_losses = []
    for group_id in torch.unique(group_ids):
        mask = group_ids.eq(group_id)
        if bool(mask.any()):
            group_losses.append(weighted_loss[mask].mean())
    if not group_losses:
        return sample_loss.new_tensor(0.0)
    stack = torch.stack(group_losses)
    k = max(1, int(np.ceil(float(top_frac) * int(stack.numel()))))
    k = min(k, int(stack.numel()))
    return torch.topk(stack, k=k, largest=True).values.mean()


def _regime_variance_loss(
    sample_loss: torch.Tensor,
    sample_weight: torch.Tensor,
    x: torch.Tensor,
    meta,
    feature_cols: list[str],
) -> torch.Tensor:
    group_ids = _batch_dynamic_regime_ids(x.detach(), meta, feature_cols)
    weighted_loss = sample_loss * sample_weight.detach()
    group_losses = []
    for group_id in torch.unique(group_ids):
        mask = group_ids.eq(group_id)
        if bool(mask.any()):
            group_losses.append(weighted_loss[mask].mean())
    if len(group_losses) <= 1:
        return sample_loss.new_tensor(0.0)
    stack = torch.stack(group_losses)
    return stack.std(unbiased=False)


def _quantile_bins(values: torch.Tensor) -> torch.Tensor:
    values = torch.nan_to_num(values.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    if int(values.numel()) >= 3:
        q = torch.quantile(values, torch.tensor([1.0 / 3.0, 2.0 / 3.0], device=values.device, dtype=values.dtype))
        return (values > q[0]).long() + (values > q[1]).long()
    return torch.zeros_like(values, dtype=torch.long)


def _batch_zbin_ids(x: torch.Tensor, model: nn.Module, feature_cols: list[str]) -> torch.Tensor:
    feature_to_idx = {name: idx for idx, name in enumerate(feature_cols)}
    if "I_raw" not in feature_to_idx or "V_corr_raw" not in feature_to_idx:
        return torch.zeros(int(x.shape[0]), device=x.device, dtype=torch.long)
    current = x[:, :, feature_to_idx["I_raw"]]
    abs_current = x[:, :, feature_to_idx["absI"]].abs() if "absI" in feature_to_idx else current.abs()
    if "dI" in feature_to_idx:
        di = x[:, :, feature_to_idx["dI"]]
    else:
        di = torch.diff(current, dim=1, prepend=current[:, :1])
    vcorr = x[:, :, feature_to_idx["V_corr_raw"]]
    abs_i_mean = abs_current.mean(dim=1)
    i_std = current.std(dim=1, unbiased=False)
    di_energy = di.square().mean(dim=1).sqrt()
    vcorr_span = vcorr.max(dim=1).values - vcorr.min(dim=1).values
    dynamic_score = abs_i_mean + i_std + di_energy
    observability_score = vcorr_span / dynamic_score.clamp_min(1e-6)
    if hasattr(model, "anchor_sequence"):
        with torch.no_grad():
            soc_score = model.anchor_sequence(x)[:, -1, 0].detach()
    else:
        soc_score = torch.zeros_like(dynamic_score)
    dynamic_bin = _quantile_bins(dynamic_score)
    observability_bin = _quantile_bins(observability_score)
    soc_bin = _quantile_bins(soc_score)
    return dynamic_bin * 9 + observability_bin * 3 + soc_bin


def _zbin_cvar_loss(
    model: nn.Module,
    sample_loss: torch.Tensor,
    sample_weight: torch.Tensor,
    x: torch.Tensor,
    feature_cols: list[str],
    top_frac: float,
) -> torch.Tensor:
    group_ids = _batch_zbin_ids(x.detach(), model, feature_cols)
    weighted_loss = sample_loss * sample_weight.detach()
    group_losses = []
    for group_id in torch.unique(group_ids):
        mask = group_ids.eq(group_id)
        if bool(mask.any()):
            group_losses.append(weighted_loss[mask].mean())
    if not group_losses:
        return sample_loss.new_tensor(0.0)
    stack = torch.stack(group_losses)
    k = max(1, int(np.ceil(float(top_frac) * int(stack.numel()))))
    k = min(k, int(stack.numel()))
    return torch.topk(stack, k=k, largest=True).values.mean()


def _cold_socbin_cvar_loss(
    sample_loss: torch.Tensor,
    sample_weight: torch.Tensor,
    y_endpoint: torch.Tensor,
    meta,
    *,
    top_frac: float,
    bin_width: float,
    cold_temp: float,
) -> torch.Tensor:
    temps = _meta_float_tensor(meta, "temperature").to(device=sample_loss.device, dtype=sample_loss.dtype).reshape(-1)
    cold_mask = torch.isclose(
        temps,
        torch.full_like(temps, float(cold_temp)),
        atol=0.75,
    )
    if not bool(cold_mask.any()):
        return sample_loss.new_tensor(0.0)
    y_soc = y_endpoint.reshape(int(sample_loss.numel()), -1)[:, 0].detach().clamp(0.0, 1.0)
    width = max(float(bin_width), 1e-6)
    bin_ids = torch.floor(y_soc / width).to(dtype=torch.long)
    weighted_loss = sample_loss * sample_weight.detach()
    group_losses = []
    for bin_id in torch.unique(bin_ids[cold_mask]):
        mask = cold_mask & bin_ids.eq(bin_id)
        if bool(mask.any()):
            group_losses.append(weighted_loss[mask].mean())
    if not group_losses:
        return sample_loss.new_tensor(0.0)
    stack = torch.stack(group_losses)
    k = max(1, int(np.ceil(float(top_frac) * int(stack.numel()))))
    k = min(k, int(stack.numel()))
    return torch.topk(stack, k=k, largest=True).values.mean()


def _cold_profile_socbin_cvar_loss(
    sample_loss: torch.Tensor,
    sample_weight: torch.Tensor,
    y_endpoint: torch.Tensor,
    meta,
    *,
    top_frac: float,
    bin_width: float,
    cold_temp: float,
) -> torch.Tensor:
    temps = _meta_float_tensor(meta, "temperature").to(device=sample_loss.device, dtype=sample_loss.dtype).reshape(-1)
    cold_mask = torch.isclose(
        temps,
        torch.full_like(temps, float(cold_temp)),
        atol=0.75,
    )
    if not bool(cold_mask.any()):
        return sample_loss.new_tensor(0.0)
    drives = _meta_list(meta, "drive_cycle")
    drive_ids = {str(value).upper(): idx for idx, value in enumerate(sorted({str(v).upper() for v in drives}))}
    drive = torch.as_tensor([drive_ids[str(value).upper()] for value in drives], device=sample_loss.device, dtype=torch.long)
    y_soc = y_endpoint.reshape(int(sample_loss.numel()), -1)[:, 0].detach().clamp(0.0, 1.0)
    width = max(float(bin_width), 1e-6)
    bin_ids = torch.floor(y_soc / width).to(dtype=torch.long)
    weighted_loss = sample_loss * sample_weight.detach()
    group_losses = []
    for drive_id in torch.unique(drive[cold_mask]):
        drive_mask = cold_mask & drive.eq(drive_id)
        for bin_id in torch.unique(bin_ids[drive_mask]):
            mask = drive_mask & bin_ids.eq(bin_id)
            if bool(mask.any()):
                group_losses.append(weighted_loss[mask].mean())
    if not group_losses:
        return sample_loss.new_tensor(0.0)
    stack = torch.stack(group_losses)
    k = max(1, int(np.ceil(float(top_frac) * int(stack.numel()))))
    k = min(k, int(stack.numel()))
    return torch.topk(stack, k=k, largest=True).values.mean()


def _cold_socbin_residual_bias_loss(
    pred_endpoint: torch.Tensor,
    y_endpoint: torch.Tensor,
    meta,
    *,
    bin_width: float,
    soc_upper: float,
    cold_temp: float,
) -> torch.Tensor:
    pred_soc = pred_endpoint.reshape(int(pred_endpoint.shape[0]), -1)[:, 0]
    true_soc = y_endpoint.reshape(int(y_endpoint.shape[0]), -1)[:, 0]
    temps = _meta_float_tensor(meta, "temperature").to(device=pred_soc.device, dtype=pred_soc.dtype).reshape(-1)
    cold_mask = torch.isclose(
        temps,
        torch.full_like(temps, float(cold_temp)),
        atol=0.75,
    )
    soc = true_soc.detach().clamp(0.0, 1.0)
    mask = cold_mask & (soc <= float(soc_upper))
    if not bool(mask.any()):
        return pred_soc.new_tensor(0.0)
    width = max(float(bin_width), 1e-6)
    bin_ids = torch.floor(soc / width).to(dtype=torch.long)
    residual = pred_soc - true_soc
    group_biases = []
    for bin_id in torch.unique(bin_ids[mask]):
        group_mask = mask & bin_ids.eq(bin_id)
        if bool(group_mask.any()):
            group_biases.append(residual[group_mask].mean())
    if not group_biases:
        return pred_soc.new_tensor(0.0)
    biases = torch.stack(group_biases)
    return biases.square().mean()


def _cold_profile_socbin_underbias_loss(
    pred_endpoint: torch.Tensor,
    y_endpoint: torch.Tensor,
    meta,
    *,
    bin_width: float,
    soc_upper: float,
    cold_temp: float,
    beta: float,
) -> torch.Tensor:
    pred_soc = pred_endpoint.reshape(int(pred_endpoint.shape[0]), -1)[:, 0]
    true_soc = y_endpoint.reshape(int(y_endpoint.shape[0]), -1)[:, 0]
    temps = _meta_float_tensor(meta, "temperature").to(device=pred_soc.device, dtype=pred_soc.dtype).reshape(-1)
    cold_mask = torch.isclose(
        temps,
        torch.full_like(temps, float(cold_temp)),
        atol=0.75,
    )
    soc = true_soc.detach().clamp(0.0, 1.0)
    mask = cold_mask & (soc <= float(soc_upper))
    if not bool(mask.any()):
        return pred_soc.new_tensor(0.0)
    drives = _meta_list(meta, "drive_cycle")
    drive_ids = {str(value).upper(): idx for idx, value in enumerate(sorted({str(v).upper() for v in drives}))}
    drive = torch.as_tensor([drive_ids[str(value).upper()] for value in drives], device=pred_soc.device, dtype=torch.long)
    width = max(float(bin_width), 1e-6)
    bin_ids = torch.floor(soc / width).to(dtype=torch.long)
    residual = pred_soc - true_soc
    losses = []
    for drive_id in torch.unique(drive[mask]):
        drive_mask = mask & drive.eq(drive_id)
        for bin_id in torch.unique(bin_ids[drive_mask]):
            group_mask = drive_mask & bin_ids.eq(bin_id)
            if bool(group_mask.any()):
                under_bias = F.relu(-residual[group_mask].mean())
                losses.append(
                    torch.where(
                        under_bias < float(beta),
                        0.5 * under_bias.square() / max(float(beta), 1e-8),
                        under_bias - 0.5 * float(beta),
                    )
                )
    if not losses:
        return pred_soc.new_tensor(0.0)
    return torch.stack(losses).mean()


def _update_weight_ema_state(
    ema_state: dict[str, torch.Tensor] | None,
    model: nn.Module,
    decay: float,
) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    if ema_state is None:
        return {name: tensor.detach().clone() for name, tensor in state.items()}
    for name, tensor in state.items():
        value = tensor.detach()
        if torch.is_floating_point(value):
            ema_state[name].mul_(float(decay)).add_(value, alpha=1.0 - float(decay))
        else:
            ema_state[name].copy_(value)
    return ema_state


def _feature_gate(values: torch.Tensor) -> torch.Tensor:
    values = torch.nan_to_num(values.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    if int(values.numel()) < 2:
        return torch.zeros_like(values)
    center = values.median()
    scale = values.std(unbiased=False).clamp_min(1e-6)
    return torch.sigmoid((values - center) / scale)


def _hard_region_sample_weights(
    x: torch.Tensor,
    y_endpoint: torch.Tensor,
    feature_cols: list[str],
    *,
    loss_weight: float,
    soc_threshold: float,
    soc_lower: float = -1.0,
    soc_upper: float = -1.0,
    dynamic_weight: float,
    cold_only: bool = False,
) -> torch.Tensor:
    if float(loss_weight) <= 0.0:
        return torch.ones(int(x.shape[0]), device=x.device, dtype=x.dtype)
    feature_to_idx = {name: idx for idx, name in enumerate(feature_cols)}
    y_soc = y_endpoint.reshape(int(x.shape[0]), -1)[:, 0].detach()
    if float(soc_upper) > float(soc_lower) >= 0.0:
        lower_gate = torch.sigmoid((y_soc - float(soc_lower)) * 40.0)
        upper_gate = torch.sigmoid((float(soc_upper) - y_soc) * 40.0)
        soc_gate = lower_gate * upper_gate
    else:
        soc_gate = torch.sigmoid((float(soc_threshold) - y_soc) * 20.0)
    dynamic_gate = torch.ones_like(soc_gate)
    if "I_raw" in feature_to_idx and "V_corr_raw" in feature_to_idx:
        current = x[:, :, feature_to_idx["I_raw"]]
        abs_current = x[:, :, feature_to_idx["absI"]].abs() if "absI" in feature_to_idx else current.abs()
        vcorr = x[:, :, feature_to_idx["V_corr_raw"]]
        i_std_gate = _feature_gate(current.std(dim=1, unbiased=False))
        abs_i_gate = _feature_gate(torch.quantile(abs_current, 0.95, dim=1))
        vcorr_span_gate = _feature_gate(vcorr.max(dim=1).values - vcorr.min(dim=1).values)
        dynamic_gate = (i_std_gate + abs_i_gate + vcorr_span_gate) / 3.0
    mix = (1.0 - float(dynamic_weight)) + float(dynamic_weight) * dynamic_gate
    hard_gate = (soc_gate * mix).clamp(0.0, 1.0)
    if bool(cold_only) and "T" in feature_to_idx:
        temp = x[:, -1, feature_to_idx["T"]]
        cold_gate = torch.sigmoid((-0.45 - temp) * 8.0)
        hard_gate = hard_gate * cold_gate
    return 1.0 + float(loss_weight) * hard_gate


def _cold_lowsoc_overpred_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    meta,
    *,
    soc_upper: float,
    cold_temp: float,
    beta: float,
) -> torch.Tensor:
    temps = _meta_float_tensor(meta, "temperature").to(device=pred.device, dtype=pred.dtype).reshape(-1)
    cold_gate = torch.isclose(
        temps,
        torch.full_like(temps, float(cold_temp)),
        atol=0.75,
    ).to(dtype=pred.dtype)
    over = F.relu(pred - y)
    over_huber = torch.where(
        over < float(beta),
        0.5 * over.square() / max(float(beta), 1e-8),
        over - 0.5 * float(beta),
    )
    soc_gate = (y.detach() <= float(soc_upper)).to(dtype=pred.dtype)
    if over_huber.ndim == 3:
        cold_gate = cold_gate.reshape(-1, 1, 1)
        denom = (soc_gate * cold_gate).sum(dim=(1, 2)).clamp_min(1.0)
        return (over_huber * soc_gate * cold_gate).sum(dim=(1, 2)) / denom
    cold_gate = cold_gate.reshape(-1, 1)
    denom = (soc_gate * cold_gate).sum(dim=1).clamp_min(1.0)
    return (over_huber * soc_gate * cold_gate).sum(dim=1) / denom


def _cold_lowsoc_underpred_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    meta,
    *,
    soc_upper: float,
    cold_temp: float,
    beta: float,
) -> torch.Tensor:
    temps = _meta_float_tensor(meta, "temperature").to(device=pred.device, dtype=pred.dtype).reshape(-1)
    cold_gate = torch.isclose(
        temps,
        torch.full_like(temps, float(cold_temp)),
        atol=0.75,
    ).to(dtype=pred.dtype)
    under = F.relu(y - pred)
    under_huber = torch.where(
        under < float(beta),
        0.5 * under.square() / max(float(beta), 1e-8),
        under - 0.5 * float(beta),
    )
    soc_gate = (y.detach() <= float(soc_upper)).to(dtype=pred.dtype)
    if under_huber.ndim == 3:
        cold_gate = cold_gate.reshape(-1, 1, 1)
        denom = (soc_gate * cold_gate).sum(dim=(1, 2)).clamp_min(1.0)
        return (under_huber * soc_gate * cold_gate).sum(dim=(1, 2)) / denom
    cold_gate = cold_gate.reshape(-1, 1)
    denom = (soc_gate * cold_gate).sum(dim=1).clamp_min(1.0)
    return (under_huber * soc_gate * cold_gate).sum(dim=1) / denom


def _regime_gated_asymmetric_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    meta,
    x: torch.Tensor,
    feature_cols: list[str],
    *,
    soc_upper: float,
    cold_temp: float,
    under_weight: float,
    over_weight: float,
    beta: float,
) -> torch.Tensor:
    feature_to_idx = {name: idx for idx, name in enumerate(feature_cols)}
    if "I_raw" not in feature_to_idx or "V_corr_raw" not in feature_to_idx:
        return pred.new_zeros(int(pred.shape[0]))
    current = x[:, :, feature_to_idx["I_raw"]]
    abs_current = x[:, :, feature_to_idx["absI"]].abs() if "absI" in feature_to_idx else current.abs()
    if "dI" in feature_to_idx:
        di = x[:, :, feature_to_idx["dI"]]
    else:
        di = torch.diff(current, dim=1, prepend=current[:, :1])
    vcorr = x[:, :, feature_to_idx["V_corr_raw"]]
    abs_i_mean = abs_current.abs().mean(dim=1)
    i_std = current.std(dim=1, unbiased=False)
    di_energy = di.square().mean(dim=1).sqrt()
    vcorr_span = vcorr.max(dim=1).values - vcorr.min(dim=1).values
    dynamic_score = abs_i_mean + i_std + di_energy
    observability_score = vcorr_span / dynamic_score.abs().clamp_min(1e-6)
    high_dynamic = _feature_gate(dynamic_score)
    high_observability = _feature_gate(observability_score)
    under_gate = (1.0 - high_dynamic) * high_observability
    over_gate = high_dynamic * (1.0 - high_observability)

    temps = _meta_float_tensor(meta, "temperature").to(device=pred.device, dtype=pred.dtype).reshape(-1)
    cold_gate = torch.isclose(
        temps,
        torch.full_like(temps, float(cold_temp)),
        atol=0.75,
    ).to(dtype=pred.dtype)
    under_gate = cold_gate * under_gate.to(device=pred.device, dtype=pred.dtype)
    over_gate = cold_gate * over_gate.to(device=pred.device, dtype=pred.dtype)
    under = F.relu(y - pred)
    over = F.relu(pred - y)
    under_huber = torch.where(
        under < float(beta),
        0.5 * under.square() / max(float(beta), 1e-8),
        under - 0.5 * float(beta),
    )
    over_huber = torch.where(
        over < float(beta),
        0.5 * over.square() / max(float(beta), 1e-8),
        over - 0.5 * float(beta),
    )
    soc_gate = (y.detach() <= float(soc_upper)).to(dtype=pred.dtype)
    if under_huber.ndim == 3:
        under_gate = under_gate.reshape(-1, 1, 1)
        over_gate = over_gate.reshape(-1, 1, 1)
        denom = soc_gate.sum(dim=(1, 2)).clamp_min(1.0)
        return (
            float(under_weight) * (under_huber * soc_gate * under_gate).sum(dim=(1, 2))
            + float(over_weight) * (over_huber * soc_gate * over_gate).sum(dim=(1, 2))
        ) / denom
    under_gate = under_gate.reshape(-1, 1)
    over_gate = over_gate.reshape(-1, 1)
    denom = soc_gate.sum(dim=1).clamp_min(1.0)
    return (
        float(under_weight) * (under_huber * soc_gate * under_gate).sum(dim=1)
        + float(over_weight) * (over_huber * soc_gate * over_gate).sum(dim=1)
    ) / denom


def _lowdyn_highspan_under_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    meta,
    x: torch.Tensor,
    feature_cols: list[str],
    *,
    soc_upper: float,
    cold_temp: float,
    dynamic_center: float,
    dynamic_sharpness: float,
    span_center: float,
    span_sharpness: float,
    beta: float,
) -> torch.Tensor:
    feature_to_idx = {name: idx for idx, name in enumerate(feature_cols)}
    if "I_raw" not in feature_to_idx or "V_corr_raw" not in feature_to_idx:
        return pred.new_zeros(int(pred.shape[0]))
    current = x[:, :, feature_to_idx["I_raw"]]
    abs_current = x[:, :, feature_to_idx["absI"]].abs() if "absI" in feature_to_idx else current.abs()
    if "dI" in feature_to_idx:
        di = x[:, :, feature_to_idx["dI"]]
    else:
        di = torch.diff(current, dim=1, prepend=current[:, :1])
    vcorr = x[:, :, feature_to_idx["V_corr_raw"]]
    dynamic = abs_current.abs().mean(dim=1) + current.std(dim=1, unbiased=False) + di.square().mean(dim=1).sqrt()
    vcorr_span = vcorr.max(dim=1).values - vcorr.min(dim=1).values
    low_dynamic_gate = torch.sigmoid((float(dynamic_center) - dynamic) * float(dynamic_sharpness))
    high_span_gate = torch.sigmoid((vcorr_span - float(span_center)) * float(span_sharpness))
    regime_gate = low_dynamic_gate * high_span_gate
    temps = _meta_float_tensor(meta, "temperature").to(device=pred.device, dtype=pred.dtype).reshape(-1)
    cold_gate = torch.isclose(
        temps,
        torch.full_like(temps, float(cold_temp)),
        atol=0.75,
    ).to(dtype=pred.dtype)
    regime_gate = regime_gate.to(device=pred.device, dtype=pred.dtype) * cold_gate
    under = F.relu(y - pred)
    under_huber = torch.where(
        under < float(beta),
        0.5 * under.square() / max(float(beta), 1e-8),
        under - 0.5 * float(beta),
    )
    soc_gate = (y.detach() <= float(soc_upper)).to(dtype=pred.dtype)
    if under_huber.ndim == 3:
        regime_gate = regime_gate.reshape(-1, 1, 1)
        denom = (soc_gate * regime_gate).sum(dim=(1, 2)).clamp_min(1.0)
        return (under_huber * soc_gate * regime_gate).sum(dim=(1, 2)) / denom
    regime_gate = regime_gate.reshape(-1, 1)
    denom = (soc_gate * regime_gate).sum(dim=1).clamp_min(1.0)
    return (under_huber * soc_gate * regime_gate).sum(dim=1) / denom


def _voltage_sag_augment(
    x: torch.Tensor,
    y: torch.Tensor,
    meta,
    feature_cols: list[str],
    *,
    prob: float,
    scale: float,
    soc_upper: float,
    cold_temp: float,
) -> torch.Tensor:
    if float(prob) <= 0.0 or float(scale) <= 0.0:
        return x
    voltage_cols = [
        "V_corr_raw",
        "V_corr_raw_ema50",
        "V_corr_raw_ema200",
        "V_corr_raw_ema800",
    ]
    indices = [feature_cols.index(col) for col in voltage_cols if col in feature_cols]
    if not indices:
        return x
    batch = int(x.shape[0])
    y_endpoint = y[:, -1, 0] if y.ndim == 3 else y.reshape(batch, -1)[:, 0]
    soc_gate = (y_endpoint.detach() <= float(soc_upper)).to(dtype=x.dtype)
    temps = _meta_float_tensor(meta, "temperature").to(device=x.device, dtype=x.dtype).reshape(-1)
    temp_gate = torch.isclose(
        temps,
        torch.full_like(temps, float(cold_temp)),
        atol=0.75,
    ).to(dtype=x.dtype)
    keep = (torch.rand(batch, device=x.device, dtype=x.dtype) < float(prob)).to(dtype=x.dtype)
    gate = (soc_gate * temp_gate * keep).reshape(batch, 1, 1)
    if not bool(gate.any()):
        return x
    shift = torch.rand(batch, 1, 1, device=x.device, dtype=x.dtype) * float(scale)
    x_aug = x.clone()
    idx = torch.as_tensor(indices, device=x.device, dtype=torch.long)
    x_aug[:, :, idx] = x_aug.index_select(dim=2, index=idx) - gate * shift
    return x_aug


def _cold_voltage_shift_consistency_loss(
    model: nn.Module,
    x: torch.Tensor,
    y_endpoint: torch.Tensor,
    pred_endpoint: torch.Tensor,
    meta,
    feature_cols: list[str],
    *,
    sequence_training: bool,
    scale: float,
    soc_upper: float,
    cold_temp: float,
    beta: float,
) -> torch.Tensor:
    if float(scale) <= 0.0:
        return x.new_tensor(0.0)
    voltage_cols = [
        "V_corr_raw",
        "V_corr_raw_ema50",
        "V_corr_raw_ema200",
        "V_corr_raw_ema800",
    ]
    indices = [feature_cols.index(col) for col in voltage_cols if col in feature_cols]
    if not indices:
        return x.new_tensor(0.0)
    batch = int(x.shape[0])
    y_soc = y_endpoint.reshape(batch, -1)[:, 0].detach()
    temps = _meta_float_tensor(meta, "temperature").to(device=x.device, dtype=x.dtype).reshape(-1)
    cold_gate = torch.isclose(
        temps,
        torch.full_like(temps, float(cold_temp)),
        atol=0.75,
    )
    soc_gate = y_soc <= float(soc_upper)
    mask = cold_gate & soc_gate
    if not bool(mask.any()):
        return x.new_tensor(0.0)
    signed_shift = (torch.rand(batch, 1, 1, device=x.device, dtype=x.dtype) * 2.0 - 1.0) * float(scale)
    signed_shift = signed_shift * mask.to(dtype=x.dtype).reshape(batch, 1, 1)
    x_shift = x.clone()
    idx = torch.as_tensor(indices, device=x.device, dtype=torch.long)
    x_shift[:, :, idx] = x_shift.index_select(dim=2, index=idx) + signed_shift
    pred_shift = model.forward_sequence(x_shift)[:, -1, :] if bool(sequence_training) else model(x_shift)
    per_sample = F.smooth_l1_loss(
        pred_shift,
        pred_endpoint.detach(),
        beta=float(beta),
        reduction="none",
    ).mean(dim=1)
    weights = mask.to(device=x.device, dtype=x.dtype)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)


def _residual_auxiliary_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    sequence_training: bool,
    huber_beta: float,
) -> torch.Tensor:
    if not hasattr(model, "anchor_sequence") or not hasattr(model, "residual_sequence"):
        return x.new_tensor(0.0)
    anchor_seq = model.anchor_sequence(x).detach()
    residual_seq = model.residual_sequence(x)
    if bool(sequence_training):
        target_residual = y - anchor_seq
        pred_residual = residual_seq
    else:
        y_endpoint = y[:, -1, :] if y.ndim == 3 else y
        target_residual = y_endpoint - anchor_seq[:, -1, :]
        pred_residual = residual_seq[:, -1, :]
    return F.smooth_l1_loss(pred_residual, target_residual.detach(), beta=float(huber_beta))


def _residual_zero_mean_loss(
    model: nn.Module,
    x: torch.Tensor,
    meta,
    sequence_training: bool,
) -> torch.Tensor:
    if not hasattr(model, "residual_sequence"):
        return x.new_tensor(0.0)
    residual = model.residual_sequence(x)
    if residual.ndim == 3:
        if bool(sequence_training):
            sample_residual = residual.mean(dim=(1, 2))
        else:
            sample_residual = residual[:, -1, :].mean(dim=1)
    else:
        sample_residual = residual.reshape(int(x.shape[0]), -1).mean(dim=1)
    if int(sample_residual.numel()) == 0:
        return x.new_tensor(0.0)
    if isinstance(meta, dict) and "temperature_id" in meta:
        vals = meta["temperature_id"]
        if torch.is_tensor(vals):
            group_ids = vals.to(device=x.device, dtype=torch.long).view(-1)
        else:
            group_ids = torch.as_tensor(vals, device=x.device, dtype=torch.long).view(-1)
    elif isinstance(meta, dict) and "temperature" in meta:
        vals = meta["temperature"]
        if torch.is_tensor(vals):
            temp = vals.to(device=x.device, dtype=torch.float32).view(-1)
        else:
            temp = torch.as_tensor(vals, device=x.device, dtype=torch.float32).view(-1)
        _, group_ids = torch.unique(temp, sorted=True, return_inverse=True)
    else:
        group_ids = torch.zeros_like(sample_residual, dtype=torch.long)
    if int(group_ids.numel()) != int(sample_residual.numel()):
        return sample_residual.mean().square()
    terms = []
    for gid in torch.unique(group_ids):
        mask = group_ids == gid
        if bool(mask.any()):
            terms.append(sample_residual[mask].mean().square())
    if not terms:
        return x.new_tensor(0.0)
    return torch.stack(terms).mean()


def _mid_delta_loss(model: nn.Module, x: torch.Tensor, sequence_training: bool, huber_beta: float) -> torch.Tensor:
    if not hasattr(model, "mid_delta_sequence"):
        return x.new_tensor(0.0)
    delta = model.mid_delta_sequence(x)
    if not bool(sequence_training):
        delta = delta[:, -1:, :]
    target = torch.zeros_like(delta)
    return F.smooth_l1_loss(delta, target, beta=float(huber_beta))


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


def _grad_reverse(x: torch.Tensor, scale: float) -> torch.Tensor:
    return _GradientReverse.apply(x, float(scale))


def _profile_label_tensor(meta, profile_to_label: dict[str, int]) -> torch.Tensor:
    vals = meta["drive_cycle"]
    if torch.is_tensor(vals):
        raise TypeError("drive_cycle metadata is expected to be string-like, not a tensor.")
    labels = []
    for value in vals:
        key = str(value).upper()
        if key not in profile_to_label:
            raise RuntimeError(f"Unexpected drive_cycle={value!r}; known profiles={sorted(profile_to_label)}")
        labels.append(profile_to_label[key])
    return torch.as_tensor(labels, device=device, dtype=torch.long)


def _profile_supcon_loss(
    h: torch.Tensor,
    y_endpoint: torch.Tensor,
    meta,
    *,
    temperature: float,
) -> torch.Tensor:
    """Align same-temperature/SOC-bin samples across drive profiles.

    This keeps SOC information as the condition: positives must be in the
    same temperature and endpoint-SOC bin, but from a different drive profile.
    It uses training labels only and never sees validation/test SOC labels.
    """
    n = int(h.shape[0])
    if n < 2:
        return h.new_tensor(0.0)
    features = F.normalize(h, dim=1)
    logits = features @ features.T / max(float(temperature), 1e-6)
    eye = torch.eye(n, device=h.device, dtype=torch.bool)
    logits = logits.masked_fill(eye, -1e9)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    temps_raw = meta["temperature"]
    if torch.is_tensor(temps_raw):
        temps = temps_raw.to(device=h.device, dtype=torch.float32)
    else:
        temps = torch.as_tensor(temps_raw, device=h.device, dtype=torch.float32)
    temps = torch.round(temps * 10.0) / 10.0
    y_flat = y_endpoint[:, 0].detach() if y_endpoint.ndim > 1 else y_endpoint.detach()
    soc_bins = torch.bucketize(y_flat, torch.as_tensor([0.2, 0.5, 0.8], device=h.device))
    drives = _meta_list(meta, "drive_cycle")
    drive_ids = {str(v).upper(): i for i, v in enumerate(sorted({str(x).upper() for x in drives}))}
    drive = torch.as_tensor([drive_ids[str(v).upper()] for v in drives], device=h.device, dtype=torch.long)

    same_context = temps[:, None].eq(temps[None, :]) & soc_bins[:, None].eq(soc_bins[None, :])
    cross_profile = drive[:, None].ne(drive[None, :])
    positives = same_context & cross_profile & (~eye)
    valid = positives.any(dim=1)
    if not bool(valid.any()):
        return h.new_tensor(0.0)
    exp_logits = torch.exp(logits).masked_fill(eye, 0.0)
    pos_sum = (exp_logits * positives.to(exp_logits.dtype)).sum(dim=1)
    denom = exp_logits.sum(dim=1).clamp_min(1e-12)
    loss = -torch.log((pos_sum / denom).clamp_min(1e-12))
    return loss[valid].mean()


def _burn_global_rng(count: int) -> None:
    for _ in range(max(0, int(count))):
        torch.empty((), dtype=torch.int64).random_()


def _augment_current_channel(
    x: torch.Tensor,
    *,
    noise_std: float,
    dropout_prob: float,
) -> torch.Tensor:
    if float(noise_std) <= 0.0 and float(dropout_prob) <= 0.0:
        return x
    out = x.clone()
    current = out[..., 1:2]
    if float(noise_std) > 0.0:
        scale = 1.0 + float(noise_std) * torch.randn(
            (x.shape[0], 1, 1),
            device=x.device,
            dtype=x.dtype,
        )
        current = current * scale.clamp(0.5, 1.5)
    if float(dropout_prob) > 0.0:
        keep = torch.rand((x.shape[0], 1, 1), device=x.device, dtype=x.dtype).ge(float(dropout_prob)).float()
        current = current * keep
    out[..., 1:2] = current
    return out


SSL_VOLTAGE_TARGET_CANDIDATES = [
    "V_corr_raw",
    "dV_corr",
    "abs_dV_corr",
    "d2V_corr",
    "V_corr_raw_ema50",
    "V_corr_raw_dev_ema50",
    "V_corr_raw_ema200",
    "V_corr_raw_dev_ema200",
    "V_corr_raw_ema800",
    "V_corr_raw_dev_ema800",
]


def _ssl_encoder_modules(model: nn.Module) -> tuple[nn.Module, list[nn.Parameter]]:
    encoder = getattr(model, "dynamic", model)
    params: list[nn.Parameter] = []
    for name in ("input_proj", "tcn_blocks", "rnn", "norm"):
        module = getattr(encoder, name, None)
        if module is not None:
            params.extend(module.parameters())
    if not params:
        raise RuntimeError("Voltage-shape SSL requires a model with input_proj and sequence encoder modules.")
    return encoder, params


def _voltage_shape_ssl_pretrain(
    model: nn.Module,
    loader,
    cfg: TrainDSTSelectorConfig,
    feature_cols: list[str],
    out_dir: Path,
    seed: int,
) -> pd.DataFrame:
    if int(cfg.ssl_pretrain_epochs) <= 0:
        return pd.DataFrame()
    target_cols = [col for col in SSL_VOLTAGE_TARGET_CANDIDATES if col in feature_cols]
    if "V_corr_raw" not in feature_cols or not target_cols:
        raise RuntimeError("Voltage-shape SSL requires V_corr_raw and at least one voltage target column.")
    target_idx = [feature_cols.index(col) for col in target_cols]
    vcorr_idx = int(feature_cols.index("V_corr_raw"))
    encoder, encoder_params = _ssl_encoder_modules(model)
    recon_head = nn.Linear(int(cfg.hidden_size), len(target_idx)).to(device)
    next_head = nn.Linear(int(cfg.hidden_size), 1).to(device)
    slope_head = nn.Linear(int(cfg.hidden_size), 1).to(device)
    opt = torch.optim.AdamW(
        list(encoder_params) + list(recon_head.parameters()) + list(next_head.parameters()) + list(slope_head.parameters()),
        lr=float(cfg.ssl_lr),
        weight_decay=float(cfg.weight_decay),
    )
    rows = []
    for ep in range(1, int(cfg.ssl_pretrain_epochs) + 1):
        model.train()
        recon_head.train()
        next_head.train()
        slope_head.train()
        losses = []
        recon_losses = []
        next_losses = []
        slope_losses = []
        for x, _y, _meta in loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            x_masked = x.clone()
            x_masked[:, :, target_idx] = 0.0
            h_masked = encoder.encode_sequence(x_masked)
            recon = recon_head(h_masked)
            recon_target = x[:, :, target_idx]
            recon_loss = F.smooth_l1_loss(recon, recon_target, beta=float(cfg.huber_beta))
            if x.shape[1] > 1:
                h = encoder.encode_sequence(x)
                next_pred = next_head(h[:, :-1, :])
                next_target = x[:, 1:, vcorr_idx:vcorr_idx + 1]
                next_loss = F.smooth_l1_loss(next_pred, next_target, beta=float(cfg.huber_beta))
                slope_pred = slope_head(h[:, :-1, :])
                slope_target = x[:, 1:, vcorr_idx:vcorr_idx + 1] - x[:, :-1, vcorr_idx:vcorr_idx + 1]
                slope_loss = F.smooth_l1_loss(slope_pred, slope_target, beta=float(cfg.huber_beta))
            else:
                next_loss = recon_loss.new_tensor(0.0)
                slope_loss = recon_loss.new_tensor(0.0)
            loss = (
                float(cfg.ssl_recon_weight) * recon_loss
                + float(cfg.ssl_next_vcorr_weight) * next_loss
                + float(cfg.ssl_slope_weight) * slope_loss
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(encoder_params) + list(recon_head.parameters()) + list(next_head.parameters()) + list(slope_head.parameters()), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            recon_losses.append(float(recon_loss.detach().cpu()))
            next_losses.append(float(next_loss.detach().cpu()))
            slope_losses.append(float(slope_loss.detach().cpu()))
        row = {
            "seed": int(seed),
            "epoch": int(ep),
            "ssl_loss": float(np.mean(losses)),
            "ssl_recon_loss": float(np.mean(recon_losses)),
            "ssl_next_vcorr_loss": float(np.mean(next_losses)),
            "ssl_slope_loss": float(np.mean(slope_losses)),
            "ssl_target_columns": ",".join(target_cols),
            "pretrain_uses_soc_labels": False,
        }
        rows.append(row)
        if ep == 1 or ep == int(cfg.ssl_pretrain_epochs) or ep % max(1, int(cfg.eval_every)) == 0:
            print(
                f"seed={seed} ssl_epoch={ep}/{cfg.ssl_pretrain_epochs} "
                f"loss={row['ssl_loss']:.6f} recon={row['ssl_recon_loss']:.6f} "
                f"next={row['ssl_next_vcorr_loss']:.6f} slope={row['ssl_slope_loss']:.6f}",
                flush=True,
            )
    history = pd.DataFrame(rows)
    history.to_csv(out_dir / f"{cfg.output_prefix}_seed{seed}_voltage_shape_ssl_history.csv", index=False)
    return history


def _stage2_selected_epoch_for_rule(cfg: TrainDSTSelectorConfig, selected_epoch: int) -> int:
    rule = str(cfg.stage2_select_rule)
    if rule == "stage1_epoch_30_35":
        if int(selected_epoch) <= int(cfg.stage2_stage1_threshold):
            return int(cfg.stage2_early_epoch)
        return int(cfg.stage2_late_epoch)
    if rule.startswith("fixed"):
        return int(rule.replace("fixed", ""))
    return int(cfg.stage2_epochs)


def _parse_stage1_ensemble_epochs(raw: str, selected_epoch: int) -> list[int]:
    text = str(raw or "").strip().lower()
    if not text or text in {"none", "selected"}:
        return [int(selected_epoch)]
    if text.startswith("range:"):
        parts = [int(x) for x in text.split(":", 2)[1:]]
        if len(parts) != 2:
            raise ValueError(f"Invalid stage1_ensemble_epochs={raw!r}")
        start, end = parts
        if end < start:
            raise ValueError(f"Invalid stage1_ensemble_epochs range {raw!r}")
        return list(range(start, end + 1))
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _load_stage1_model_for_epoch(
    cfg: TrainDSTSelectorConfig,
    variant: CondInvVariant,
    state_by_epoch: dict[int, dict[str, torch.Tensor]],
    epoch: int,
    input_dim: int,
) -> nn.Module:
    if int(epoch) not in state_by_epoch:
        raise RuntimeError(f"Requested stage1 ensemble epoch {epoch} is missing from state_by_epoch.")
    model = _make_stage1_model(cfg, variant, input_dim=int(input_dim))
    model.load_state_dict({k: v.to(device) for k, v in state_by_epoch[int(epoch)].items()})
    model.eval()
    return model


def _make_selected_stage1_base(
    cfg: TrainDSTSelectorConfig,
    variant: CondInvVariant,
    state_by_epoch: dict[int, dict[str, torch.Tensor]],
    selected_epoch: int,
    input_dim: int,
) -> tuple[nn.Module, list[int]]:
    epochs = _parse_stage1_ensemble_epochs(str(cfg.stage1_ensemble_epochs), int(selected_epoch))
    models = [_load_stage1_model_for_epoch(cfg, variant, state_by_epoch, ep, input_dim=int(input_dim)) for ep in epochs]
    if len(models) == 1:
        setattr(models[0], "input_dim_for_stage2", int(input_dim))
        return models[0], epochs
    ensemble = SnapshotEnsembleBase(models).to(device)
    setattr(ensemble, "input_dim_for_stage2", int(input_dim))
    return ensemble, epochs


def _train_stage2_correction_test_blind(
    cfg: TrainDSTSelectorConfig,
    variant: CondInvVariant,
    base_model,
    train_loader,
    valid_loader,
    test_loader,
    seed: int,
    selected_epoch: int,
) -> pd.DataFrame:
    target_epoch = min(int(cfg.stage2_epochs), _stage2_selected_epoch_for_rule(cfg, int(selected_epoch)))
    stage2_rule = str(cfg.stage2_select_rule)
    stage2_val_rule = stage2_rule in {"val_mean_mae", "val_worst_mae", "val_mean_plus_worst"}
    if valid_loader is None and stage2_val_rule:
        raise RuntimeError(f"stage2_select_rule={stage2_rule!r} requires a validation split.")
    corrected = Stage2CorrectedModel(
        base_model,
        int(cfg.hidden_size),
        float(variant.corr_limit),
        str(variant.corr_mode),
        freeze_base=True,
        input_dim=int(
            getattr(
                base_model,
                "correction_input_dim_for_stage2",
                getattr(base_model, "input_dim_for_stage2", 3),
            )
        ),
        base_input_dim=int(getattr(base_model, "input_dim_for_stage2", 3)),
        corr_zero_init=bool(cfg.corr_zero_init),
    ).to(device)
    opt = torch.optim.AdamW(
        [p for p in corrected.parameters() if p.requires_grad],
        lr=float(cfg.lr_stage2),
        weight_decay=float(cfg.weight_decay),
    )
    rows = []
    history = []
    valid_parts = []
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    for ep in range(1, int(target_epoch) + 1):
        corrected.train()
        losses = []
        supervised_losses = []
        keep_losses = []
        for x, y, meta in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            y = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            y_endpoint = y[:, -1, :] if y.ndim == 3 else y
            with torch.no_grad():
                base_pred = corrected.base_prediction(x)
            pred = corrected(x)
            sample_loss = F.smooth_l1_loss(pred, y_endpoint, beta=float(cfg.huber_beta), reduction="none").mean(dim=1)
            temps = _meta_float_tensor(meta, "temperature")
            temp45 = torch.isclose(temps, torch.full_like(temps, 45.0), atol=0.75).float()
            if variant.corr_mode == "temp_bands":
                temp25 = torch.isclose(temps, torch.full_like(temps, 25.0), atol=0.75).float()
                focus = torch.maximum(temp25, temp45).detach()
            elif variant.corr_mode in {"cold_hot", "cold_hot_deep"}:
                temp0 = torch.isclose(temps, torch.full_like(temps, 0.0), atol=0.75).float()
                focus = torch.maximum(temp0, temp45).detach()
            elif variant.corr_mode == "cold_mid_hot":
                temp0 = torch.isclose(temps, torch.full_like(temps, 0.0), atol=0.75).float()
                temp25 = torch.isclose(temps, torch.full_like(temps, 25.0), atol=0.75).float()
                focus = torch.maximum(torch.maximum(temp0, temp25), temp45).detach()
            elif variant.corr_mode == "cold_hot_midlite":
                temp0 = torch.isclose(temps, torch.full_like(temps, 0.0), atol=0.75).float()
                temp25 = torch.isclose(temps, torch.full_like(temps, 25.0), atol=0.75).float()
                focus = torch.maximum(temp0, temp45) + 0.25 * temp25
                focus = focus.clamp(max=1.0).detach()
            elif variant.corr_mode == "all_temp":
                focus = torch.ones_like(temp45).detach()
            elif variant.corr_mode == "cold_only":
                temp0 = torch.isclose(temps, torch.full_like(temps, 0.0), atol=0.75).float()
                focus = temp0.detach()
            elif variant.corr_mode == "hot_only":
                focus = temp45.detach()
            else:
                low_voltage_focus = torch.sigmoid(-(x[:, -1, 0] + 0.25) * 5.0)
                focus = (temp45 * low_voltage_focus).detach()
            weights = 1.0 + (float(cfg.focus45_weight) - 1.0) * focus
            supervised = (sample_loss * weights).sum() / weights.sum().clamp_min(1e-6)
            keep_weight = (1.0 - focus).clamp_min(0.0).unsqueeze(1)
            keep = ((pred - base_pred).square() * keep_weight).sum() / keep_weight.sum().clamp_min(1e-6)
            loss = supervised + float(cfg.keep_lambda) * keep
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(corrected.correction.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            supervised_losses.append(float(supervised.detach().cpu()))
            keep_losses.append(float(keep.detach().cpu()))
        history.append(
            {
                "seed": int(seed),
                "variant": f"{variant.name}_stage2",
                "epoch": int(ep),
                "loss": float(np.mean(losses)),
                "supervised_loss": float(np.mean(supervised_losses)),
                "keep_loss": float(np.mean(keep_losses)),
                "test_blind": True,
                "stage2_train_until_epoch": int(target_epoch),
                "stage2_select_rule": stage2_rule,
            }
        )
        if ep == 1 or ep % int(cfg.eval_every) == 0 or ep == int(target_epoch):
            valid = _eval_by_temp_or_empty(corrected, valid_loader, "valid", f"{variant.name}_stage2", ep)
            if len(valid):
                valid["test_blind"] = True
                valid_parts.append(valid.copy())
            if stage2_val_rule:
                snapshots[int(ep)] = copy.deepcopy({k: v.detach().cpu() for k, v in corrected.state_dict().items()})
            if int(cfg.test_blind_rng_burn) > 0:
                _burn_global_rng(int(cfg.test_blind_rng_burn))
            if len(valid):
                rows.append(valid)
            if len(valid):
                piv = valid.pivot_table(
                    index=["variant", "epoch", "split"], columns="temperature_C", values="MAE_pct"
                ).reset_index()
                mae0 = float(piv.get(0.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
                mae25 = float(piv.get(25.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
                mae45 = float(piv.get(45.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
                valid_summary = f"valid0={mae0:.3f}% valid25={mae25:.3f}% valid45={mae45:.3f}%"
            else:
                valid_summary = "valid=none"
            print(
                f"{variant.name}_stage2 epoch={ep} loss={np.mean(losses):.5f} "
                f"supervised={np.mean(supervised_losses):.5f} keep={np.mean(keep_losses):.5f} "
                f"{valid_summary} test=hidden",
                flush=True,
            )
    selected_stage2_epoch = int(target_epoch)
    stage2_selector_score = float("nan")
    stage2_val_mean = float("nan")
    stage2_val_worst = float("nan")
    if stage2_val_rule:
        if not valid_parts:
            raise RuntimeError("Stage 2 validation selector has no validation rows.")
        valid_all = pd.concat(valid_parts, ignore_index=True)
        per_epoch = (
            valid_all.groupby("epoch", as_index=False)["MAE_pct"]
            .agg(val_mean_mae="mean", val_worst_mae="max")
        )
        if stage2_rule == "val_mean_plus_worst":
            per_epoch["selector_score"] = per_epoch["val_mean_mae"] + 0.5 * per_epoch["val_worst_mae"]
        else:
            per_epoch["selector_score"] = per_epoch[stage2_rule]
        best = per_epoch.sort_values(["selector_score", "epoch"]).iloc[0]
        selected_stage2_epoch = int(best["epoch"])
        stage2_selector_score = float(best["selector_score"])
        stage2_val_mean = float(best["val_mean_mae"])
        stage2_val_worst = float(best["val_worst_mae"])
        if selected_stage2_epoch not in snapshots:
            raise RuntimeError(f"Missing Stage 2 snapshot for selected epoch {selected_stage2_epoch}.")
        corrected.load_state_dict(snapshots[selected_stage2_epoch], strict=True)
    test = eval_by_temp(corrected, test_loader, "test", f"{variant.name}_stage2", int(selected_stage2_epoch))
    test["test_blind"] = True
    test["stage2_selected_epoch"] = int(selected_stage2_epoch)
    test["stage2_select_rule"] = stage2_rule
    test["stage2_selector_score"] = stage2_selector_score
    test["val_mean_mae"] = stage2_val_mean
    test["val_worst_mae"] = stage2_val_worst
    rows.append(test)
    if bool(cfg.save_predictions):
        pred_dir = cfg.base_dir / "nmc_goal_vcorr_it_train_dst_selector_results"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_prefix = f"{cfg.output_prefix}_seed{seed}_sel{selected_epoch}_{variant.name}_stage2"
        valid_pred = _predict_rows(corrected, valid_loader, "valid", f"{variant.name}_stage2", int(selected_stage2_epoch))
        if len(valid_pred):
            valid_pred.to_csv(
                pred_dir / f"{pred_prefix}_valid_prediction_rows.csv.gz",
                index=False,
                compression="gzip",
            )
        _predict_rows(corrected, test_loader, "test", f"{variant.name}_stage2", int(selected_stage2_epoch)).to_csv(
            pred_dir / f"{pred_prefix}_test_prediction_rows.csv.gz",
            index=False,
            compression="gzip",
        )
    history_dir = cfg.base_dir / "nmc_goal_vcorr_it_conditional_invariant_results"
    history_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(
        history_dir / f"{cfg.output_prefix}_seed{seed}_sel{selected_epoch}_{variant.name}_stage2_test_blind_history.csv",
        index=False,
    )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


REGIME_SELECTOR_METRICS = ["absI_mean", "I_std", "dI_energy", "V_corr_span"]


def _regime_edges_from_train_ds(ds) -> dict[str, tuple[float, float]]:
    rows = _window_regime_rows(ds)
    return {col: _regime_edges(rows[col].tolist()) for col in REGIME_SELECTOR_METRICS}


def _regime_stats_from_batch(x_cpu: np.ndarray, feature_cols: list[str], edges: dict[str, tuple[float, float]]) -> pd.DataFrame:
    feature_to_idx = {name: idx for idx, name in enumerate(feature_cols)}
    if "I_raw" not in feature_to_idx or "V_corr_raw" not in feature_to_idx:
        raise RuntimeError("Regime-sliced selector requires I_raw and V_corr_raw in the selected feature set.")
    i = x_cpu[:, :, feature_to_idx["I_raw"]].astype(np.float64)
    v = x_cpu[:, :, feature_to_idx["V_corr_raw"]].astype(np.float64)
    if "absI" in feature_to_idx:
        abs_i = np.abs(x_cpu[:, :, feature_to_idx["absI"]].astype(np.float64))
    else:
        abs_i = np.abs(i)
    if "dI" in feature_to_idx:
        d_i = x_cpu[:, :, feature_to_idx["dI"]].astype(np.float64)
    else:
        d_i = np.diff(i, axis=1, prepend=i[:, :1])
    rows = pd.DataFrame(
        {
            "absI_mean": np.nanmean(abs_i, axis=1),
            "I_std": np.nanstd(i, axis=1),
            "dI_energy": np.nanmean(np.square(d_i), axis=1),
            "V_corr_span": np.nanmax(v, axis=1) - np.nanmin(v, axis=1),
        }
    )
    for col in REGIME_SELECTOR_METRICS:
        rows[f"{col}_bin"] = [_metric_bin(float(value), edges[col]) for value in rows[col]]
    rows["regime_key"] = [
        "|".join(str(row[f"{col}_bin"]) for col in REGIME_SELECTOR_METRICS)
        for _, row in rows.iterrows()
    ]
    return rows


@torch.no_grad()
def eval_by_temp_regime(
    model: nn.Module,
    loader,
    split: str,
    variant_name: str,
    seed: int,
    epoch: int,
    feature_cols: list[str],
    train_regime_edges: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    model.eval()
    rows = []
    for x, y, meta in loader:
        x_cpu = x.detach().cpu().numpy().astype(np.float32)
        regime = _regime_stats_from_batch(x_cpu, feature_cols, train_regime_edges)
        x_dev = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        pred = model(x_dev).detach().cpu().numpy()[:, 0]
        y_cpu = y.detach().cpu()
        true = y_cpu[:, -1, 0].numpy() if y_cpu.ndim == 3 else y_cpu.numpy()[:, 0]
        mdf = collate_meta_to_frame(meta)
        mdf = pd.concat([mdf.reset_index(drop=True), regime.reset_index(drop=True)], axis=1)
        mdf["y_pred"] = pred
        mdf["y_true"] = true
        mdf["error"] = mdf["y_pred"] - mdf["y_true"]
        for (temp, regime_key), g in mdf.groupby(["temperature", "regime_key"], dropna=False):
            err = g["error"].to_numpy(np.float64)
            row = {
                "variant": variant_name,
                "seed": int(seed),
                "epoch": int(epoch),
                "split": split,
                "temperature_C": float(temp),
                "regime_key": str(regime_key),
                "n_windows": int(len(g)),
                "sum_abs_error": float(np.sum(np.abs(err))),
                "sum_sq_error": float(np.sum(np.square(err))),
            }
            for col in REGIME_SELECTOR_METRICS:
                row[f"{col}_bin"] = str(g[f"{col}_bin"].iloc[0])
            rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        key_cols = [
            "variant",
            "seed",
            "epoch",
            "split",
            "temperature_C",
            "regime_key",
            *[f"{col}_bin" for col in REGIME_SELECTOR_METRICS],
        ]
        out = out.groupby(key_cols, as_index=False).agg(
            n_windows=("n_windows", "sum"),
            sum_abs_error=("sum_abs_error", "sum"),
            sum_sq_error=("sum_sq_error", "sum"),
        )
        denom = out["n_windows"].clip(lower=1).astype(float)
        out["MAE_pct"] = out["sum_abs_error"] / denom * 100.0
        out["RMSE_pct"] = np.sqrt(out["sum_sq_error"] / denom) * 100.0
        out = out.drop(columns=["sum_abs_error", "sum_sq_error"])
    return out


@torch.no_grad()
def _predict_rows(model: nn.Module, loader, split: str, variant_name: str, epoch: int) -> pd.DataFrame:
    if loader is None:
        return pd.DataFrame()
    model.eval()
    rows = []
    offset = 0
    for x, y, meta in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        pred = model(x).detach().cpu().numpy()[:, 0]
        y_cpu = y.detach().cpu()
        true = y_cpu[:, -1, 0].numpy() if y_cpu.ndim == 3 else y_cpu.numpy()[:, 0]
        mdf = collate_meta_to_frame(meta)
        n = len(mdf)
        mdf["row_id"] = np.arange(offset, offset + n, dtype=np.int64)
        offset += n
        mdf["split"] = split
        mdf["variant"] = variant_name
        mdf["epoch"] = int(epoch)
        mdf["target_label"] = "physical"
        mdf["y_true"] = true
        mdf["y_pred"] = pred
        mdf["error"] = mdf["y_pred"] - mdf["y_true"]
        mdf["abs_error"] = np.abs(mdf["error"])
        rows.append(mdf)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _collect_eval_tensors(loader) -> tuple[torch.Tensor, np.ndarray, pd.DataFrame]:
    if loader is None:
        return torch.empty(0), np.empty(0), pd.DataFrame()
    xs = []
    ys = []
    metas = []
    for x, y, meta in loader:
        xs.append(x.detach().cpu().float())
        ys.append(y.detach().cpu().numpy()[:, 0])
        metas.append(collate_meta_to_frame(meta))
    if not xs:
        return torch.empty(0), np.empty(0), pd.DataFrame()
    return torch.cat(xs, dim=0), np.concatenate(ys, axis=0), pd.concat(metas, ignore_index=True)


@torch.no_grad()
def _predict_from_tensor(model: nn.Module, x_cpu: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    preds = []
    n = int(x_cpu.shape[0])
    for start in range(0, n, int(batch_size)):
        xb = x_cpu[start : start + int(batch_size)].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        preds.append(model(xb).detach().cpu().numpy()[:, 0])
    return np.concatenate(preds, axis=0) if preds else np.empty(0, dtype=np.float32)


def _indices_for_feature_group(feature_cols: list[str], group: str) -> list[int]:
    voltage = {
        "V_corr_raw_ema50",
        "V_corr_raw_dev_ema50",
        "V_corr_raw_ema200",
        "V_corr_raw_dev_ema200",
        "V_corr_raw_ema800",
        "V_corr_raw_dev_ema800",
    }
    voltage_dev = {c for c in voltage if "_dev_" in c}
    current = {
        "I_raw_ema50",
        "I_raw_dev_ema50",
        "I_raw_ema200",
        "I_raw_dev_ema200",
    }
    current_dev = {c for c in current if "_dev_" in c}
    abs_current = {
        "absI_ema50",
        "absI_dev_ema50",
        "absI_ema200",
        "absI_dev_ema200",
    }
    abs_current_dev = {c for c in abs_current if "_dev_" in c}
    groups = {
        "voltage_ema": voltage,
        "voltage_ema_dev": voltage_dev,
        "current_ema": current,
        "current_ema_dev": current_dev,
        "abs_current_ema": abs_current,
        "abs_current_ema_dev": abs_current_dev,
        "all_ema": voltage | current | abs_current,
    }
    wanted = groups[group]
    return [idx for idx, col in enumerate(feature_cols) if col in wanted]


def _shuffle_within_temp_profile(x: torch.Tensor, meta: pd.DataFrame, indices: list[int], seed: int) -> torch.Tensor:
    out = x.clone()
    if not indices or meta.empty:
        return out
    rng = np.random.default_rng(int(seed))
    key_cols = [c for c in ["temperature", "drive_cycle", "file_name"] if c in meta.columns]
    if not key_cols:
        perm = rng.permutation(int(out.shape[0]))
        out[:, :, indices] = out[perm][:, :, indices]
        return out
    for _, idx in meta.groupby(key_cols, sort=False).groups.items():
        arr_idx = np.asarray(list(idx), dtype=np.int64)
        if len(arr_idx) <= 1:
            continue
        perm = rng.permutation(arr_idx)
        arr_t = torch.as_tensor(arr_idx, dtype=torch.long)
        perm_t = torch.as_tensor(perm, dtype=torch.long)
        for feat_idx in indices:
            out[arr_t, :, feat_idx] = out[perm_t, :, feat_idx]
    return out


def _replace_ema_with_endpoint_memoryless(x: torch.Tensor, feature_cols: list[str]) -> torch.Tensor:
    out = x.clone()
    raw_index = {col: idx for idx, col in enumerate(feature_cols)}
    for idx, col in enumerate(feature_cols):
        if "_ema" not in col:
            continue
        if col.startswith("V_corr_raw_"):
            base = raw_index.get("V_corr_raw")
            if base is not None:
                out[:, :, idx] = out[:, -1:, base]
        elif col.startswith("I_raw_"):
            base = raw_index.get("I_raw")
            if base is not None:
                out[:, :, idx] = out[:, -1:, base]
        elif col.startswith("absI_"):
            base = raw_index.get("I_raw")
            if base is not None:
                out[:, :, idx] = out[:, -1:, base].abs()
    return out


def _ema_perturbation_importance(
    model: nn.Module,
    loader,
    feature_cols: list[str],
    variant_name: str,
    epoch: int,
    seed: int,
    batch_size: int,
) -> pd.DataFrame:
    x, y_true, meta = _collect_eval_tensors(loader)
    if x.numel() == 0:
        return pd.DataFrame()
    perturbations: list[tuple[str, torch.Tensor, str]] = []
    perturbations.append(("P0_no_perturbation", x, "original selected G4 input"))
    x1 = x.clone()
    x1[:, :, _indices_for_feature_group(feature_cols, "voltage_ema_dev")] = 0.0
    perturbations.append(("P1_zero_voltage_ema_deviation", x1, "zero V_corr_raw_dev_ema* channels"))
    x2 = x.clone()
    x2[:, :, _indices_for_feature_group(feature_cols, "current_ema_dev")] = 0.0
    perturbations.append(("P2_zero_current_ema_deviation", x2, "zero I_raw_dev_ema* channels"))
    x3 = x.clone()
    x3[:, :, _indices_for_feature_group(feature_cols, "abs_current_ema_dev")] = 0.0
    perturbations.append(("P3_zero_abs_current_ema_deviation", x3, "zero absI_dev_ema* channels"))
    perturbations.append(
        (
            "P4_shuffle_voltage_ema",
            _shuffle_within_temp_profile(x, meta, _indices_for_feature_group(feature_cols, "voltage_ema"), int(seed) + 4001),
            "shuffle V_corr EMA channels within same temperature/profile/file groups",
        )
    )
    perturbations.append(
        (
            "P5_shuffle_current_ema",
            _shuffle_within_temp_profile(
                x,
                meta,
                _indices_for_feature_group(feature_cols, "current_ema")
                + _indices_for_feature_group(feature_cols, "abs_current_ema"),
                int(seed) + 5001,
            ),
            "shuffle current and abs-current EMA channels within same temperature/profile/file groups",
        )
    )
    perturbations.append(
        (
            "P6_endpoint_raw_memoryless_ema",
            _replace_ema_with_endpoint_memoryless(x, feature_cols),
            "replace EMA channels with endpoint raw proxy values to destroy within-window EMA memory",
        )
    )
    rows = []
    baseline_mae_by_temp: dict[float, float] = {}
    for perturbation, x_pert, detail in perturbations:
        y_pred = _predict_from_tensor(model, x_pert, int(batch_size))
        err = y_pred - y_true
        m = meta.copy()
        m["error"] = err
        m["abs_error"] = np.abs(err)
        for temp, g in m.groupby("temperature", sort=True):
            mae = float(g["abs_error"].mean() * 100.0)
            if perturbation == "P0_no_perturbation":
                baseline_mae_by_temp[float(temp)] = mae
            rows.append(
                {
                    "seed": int(seed),
                    "variant": variant_name,
                    "epoch": int(epoch),
                    "perturbation": perturbation,
                    "temperature_C": float(temp),
                    "n_windows": int(len(g)),
                    "MAE_pct": mae,
                    "RMSE_pct": float(np.sqrt(np.mean(np.square(g["error"].to_numpy(np.float64)))) * 100.0),
                    "delta_MAE_vs_P0_pct": mae - float(baseline_mae_by_temp.get(float(temp), mae)),
                    "detail": detail,
                }
            )
        mae_all = float(m["abs_error"].mean() * 100.0)
        base_all = baseline_mae_by_temp.get(float("inf"), mae_all)
        if perturbation == "P0_no_perturbation":
            baseline_mae_by_temp[float("inf")] = mae_all
            base_all = mae_all
        rows.append(
            {
                "seed": int(seed),
                "variant": variant_name,
                "epoch": int(epoch),
                "perturbation": perturbation,
                "temperature_C": "ALL",
                "n_windows": int(len(m)),
                "MAE_pct": mae_all,
                "RMSE_pct": float(np.sqrt(np.mean(np.square(m["error"].to_numpy(np.float64)))) * 100.0),
                "delta_MAE_vs_P0_pct": mae_all - float(base_all),
                "detail": detail,
            }
        )
    return pd.DataFrame(rows)


def _make_base_cfg_for_seed(cfg: TrainDSTSelectorConfig, seed: int):
    base_cfg = make_cfg()
    base_cfg.output_dir = cfg.base_dir / "nmc_goal_vcorr_it_train_dst_selector_results"
    base_cfg.batch_size = int(cfg.batch_size)
    base_cfg.dataloader_num_workers = 0 if bool(cfg.cache_dataset_cuda) and device.type == "cuda" else int(cfg.num_workers)
    base_cfg.dataloader_prefetch_factor = int(cfg.prefetch_factor)
    base_cfg.dataloader_pin_memory = not (bool(cfg.cache_dataset_cuda) and device.type == "cuda")
    base_cfg.dataloader_persistent_workers = int(base_cfg.dataloader_num_workers) > 0
    if str(cfg.sampler_seed_mode) == "seed":
        base_cfg.sampler_seed = int(seed)
    elif str(cfg.sampler_seed_mode) == "none":
        base_cfg.sampler_seed = None
    else:
        raise ValueError(f"Unknown sampler_seed_mode={cfg.sampler_seed_mode!r}")
    return base_cfg


def _prepare_seed_shared_assets(cfg: TrainDSTSelectorConfig, frames, out_dir: Path) -> dict:
    feature_cols = _selected_feature_columns(str(cfg.feature_set))
    stage2_feature_set = str(cfg.stage2_feature_set or cfg.feature_set)
    stage2_feature_cols = _selected_feature_columns(stage2_feature_set)
    scaled, _ = _cached_scaled_frames_for_ablation(frames, feature_cols)
    if stage2_feature_cols == feature_cols:
        scaled_stage2 = scaled
    else:
        scaled_stage2, _ = _cached_scaled_frames_for_ablation(frames, stage2_feature_cols)
    scaled_stage1 = _filter_scaled_frames_by_temperatures(
        scaled,
        train_temperatures=tuple(cfg.train_temperatures),
        valid_temperatures=tuple(cfg.valid_temperatures),
        test_temperatures=tuple(cfg.test_temperatures),
        name="stage1",
    )
    stage2_train_temperatures = tuple(cfg.stage2_train_temperatures or cfg.train_temperatures)
    stage2_valid_temperatures = tuple(cfg.stage2_valid_temperatures or cfg.valid_temperatures)
    stage2_test_temperatures = tuple(cfg.stage2_test_temperatures or cfg.test_temperatures)
    scaled_stage2_filtered = _filter_scaled_frames_by_temperatures(
        scaled_stage2,
        train_temperatures=stage2_train_temperatures,
        valid_temperatures=stage2_valid_temperatures,
        test_temperatures=stage2_test_temperatures,
        name="stage2",
    )

    train_dataset_cls = SequenceWindowDataset if bool(cfg.sequence_training) else DecomposedWindowDataset
    train_ds = train_dataset_cls(scaled_stage1["train"], feature_cols, cfg.window_len, cfg.stride, target_label="physical")
    needs_train_eval = (not bool(cfg.skip_train_final_eval)) or str(cfg.stage1_selector).startswith("train")
    train_eval_ds = (
        DecomposedWindowDataset(scaled_stage1["train"], feature_cols, cfg.window_len, cfg.stride, target_label="physical")
        if needs_train_eval
        else None
    )
    valid_ds = DecomposedWindowDataset(scaled_stage1["valid"], feature_cols, cfg.window_len, 1, target_label="physical")
    test_ds = DecomposedWindowDataset(scaled_stage1["test"], feature_cols, cfg.window_len, 1, target_label="physical")
    train_regime_edges = _regime_edges_from_train_ds(train_ds)
    train_ds = _maybe_cuda_cache_dataset(train_ds, cfg, "stage1_train")
    if train_eval_ds is not None:
        train_eval_ds = _maybe_cuda_cache_dataset(train_eval_ds, cfg, "stage1_train_eval")
    if len(valid_ds):
        valid_ds = _maybe_cuda_cache_dataset(valid_ds, cfg, "stage1_valid")
    test_ds = _maybe_cuda_cache_dataset(test_ds, cfg, "stage1_test")
    train_eval_loader = _make_eval_loader(train_eval_ds, cfg) if train_eval_ds is not None else None
    valid_loader = _make_eval_loader(valid_ds, cfg)
    test_loader = _make_eval_loader(test_ds, cfg)

    stage2_train_ds = stage2_valid_ds = stage2_test_ds = None
    stage2_valid_loader = stage2_test_loader = None
    if not bool(cfg.skip_stage2):
        stage2_train_ds = train_dataset_cls(
            scaled_stage2_filtered["train"],
            stage2_feature_cols,
            cfg.window_len,
            cfg.stride,
            target_label="physical",
        )
        stage2_valid_ds = DecomposedWindowDataset(
            scaled_stage2_filtered["valid"],
            stage2_feature_cols,
            cfg.window_len,
            1,
            target_label="physical",
        )
        stage2_test_ds = DecomposedWindowDataset(
            scaled_stage2_filtered["test"],
            stage2_feature_cols,
            cfg.window_len,
            1,
            target_label="physical",
        )
        if bool(cfg.cache_dataset_cuda):
            stage2_train_ds = _maybe_cuda_cache_dataset(stage2_train_ds, cfg, "stage2_train")
            if len(stage2_valid_ds):
                stage2_valid_ds = _maybe_cuda_cache_dataset(stage2_valid_ds, cfg, "stage2_valid")
            stage2_test_ds = _maybe_cuda_cache_dataset(stage2_test_ds, cfg, "stage2_test")
        stage2_valid_loader = _make_eval_loader(stage2_valid_ds, cfg)
        stage2_test_loader = _make_eval_loader(stage2_test_ds, cfg)

    finetune_train_eval_loader = train_eval_loader
    finetune_train_ds = None
    finetune_train_eval_ds = None
    if int(cfg.finetune_epochs) > 0:
        if not tuple(cfg.finetune_temperatures):
            raise RuntimeError("--finetune-epochs requires --finetune-temperatures.")
        scaled_finetune_train = _filter_frame_list_by_temperatures(
            scaled["train"],
            tuple(cfg.finetune_temperatures),
            split_name="finetune:train",
        )
        finetune_train_ds = train_dataset_cls(
            scaled_finetune_train,
            feature_cols,
            cfg.window_len,
            cfg.stride,
            target_label="physical",
        )
        finetune_train_eval_ds = DecomposedWindowDataset(
            scaled_finetune_train,
            feature_cols,
            cfg.window_len,
            cfg.stride,
            target_label="physical",
        )
        finetune_train_ds = _maybe_cuda_cache_dataset(finetune_train_ds, cfg, "stage1_finetune_train")
        finetune_train_eval_ds = _maybe_cuda_cache_dataset(finetune_train_eval_ds, cfg, "stage1_finetune_train_eval")
        finetune_train_eval_loader = _make_eval_loader(finetune_train_eval_ds, cfg)

    return {
        "feature_cols": feature_cols,
        "stage2_feature_cols": stage2_feature_cols,
        "train_ds": train_ds,
        "train_eval_loader": train_eval_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "stage2_train_ds": stage2_train_ds,
        "stage2_valid_loader": stage2_valid_loader,
        "stage2_test_loader": stage2_test_loader,
        "finetune_train_ds": finetune_train_ds,
        "finetune_train_eval_loader": finetune_train_eval_loader,
        "train_regime_edges": train_regime_edges,
    }


def _pretrain_dual_context_branches(
    model: nn.Module,
    train_loader,
    cfg: TrainDSTSelectorConfig,
    variant: CondInvVariant,
) -> list[dict]:
    epochs = int(cfg.dual_context_branch_pretrain_epochs)
    if epochs <= 0 or not hasattr(model, "branch_auxiliary_loss"):
        return []
    branch_params = list(model.dual_context_branch_parameters()) if hasattr(model, "dual_context_branch_parameters") else []
    branch_params = [param for param in branch_params if param.requires_grad]
    if not branch_params:
        return []
    opt = torch.optim.AdamW(branch_params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    legacy_variant = to_variant(variant)
    history = []
    for ep in _epoch_iter(1, epochs + 1, cfg, "dual-context branch pretrain"):
        model.train()
        losses = []
        for x, y, meta in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            y = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            opt.zero_grad(set_to_none=True)
            sample_loss = model.branch_auxiliary_loss(x, y, float(cfg.huber_beta))
            sw = temp_weights(meta, legacy_variant, int(sample_loss.numel()))
            stack, wstack = _group_loss_weight_stacks(sample_loss, sw, meta, cfg.rex_group)
            mean_loss = (stack * wstack).sum() / wstack.sum().clamp_min(1e-6)
            rex_var = stack.var(unbiased=False) if len(stack) > 1 else stack.new_tensor(0.0)
            loss = mean_loss + float(variant.lambda_rex) * rex_var
            loss.backward()
            nn.utils.clip_grad_norm_(branch_params, 1.0)
            opt.step()
            losses.append(loss.detach())
        loss_mean = _mean_scalar(losses)
        if ep == 1 or ep == epochs or ep % 10 == 0:
            print(f"dual_context_branch_pretrain epoch={ep} loss={loss_mean:.5f}", flush=True)
        history.append({"pretrain_epoch": int(ep), "dual_context_branch_pretrain_loss": loss_mean})
    return history


def _pretrain_base_estimator(
    model: nn.Module,
    train_loader,
    cfg: TrainDSTSelectorConfig,
    variant: CondInvVariant,
) -> list[dict]:
    epochs = int(cfg.base_pretrain_epochs)
    if epochs <= 0 or not hasattr(model, "base_forward_sequence"):
        return []
    base_params = list(model.base_parameters()) if hasattr(model, "base_parameters") else []
    base_params = [param for param in base_params if param.requires_grad]
    if not base_params:
        return []
    opt = torch.optim.AdamW(base_params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    legacy_variant = to_variant(variant)
    history = []
    for ep in _epoch_iter(1, epochs + 1, cfg, "base pretrain"):
        model.train()
        losses = []
        mmds = []
        for x, y, meta in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            y = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            opt.zero_grad(set_to_none=True)
            h = model.encode_sequence(x)
            h_last = h[:, -1, :]
            if bool(variant.sequence_training):
                pred = model.base_forward_sequence(x)
                sample_loss = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=(1, 2))
                y_for_mmd = y[:, -1, :]
            else:
                pred = model.base_forward(x)
                sample_loss = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=1)
                y_for_mmd = y
            sw = temp_weights(meta, legacy_variant, int(sample_loss.numel()))
            stack, wstack = _group_loss_weight_stacks(sample_loss, sw, meta, cfg.rex_group)
            mean_loss = (stack * wstack).sum() / wstack.sum().clamp_min(1e-6)
            rex_var = stack.var(unbiased=False) if len(stack) > 1 else stack.new_tensor(0.0)
            mmd_loss = conditional_profile_mmd(h_last, y_for_mmd, meta)
            loss = mean_loss + float(variant.lambda_rex) * rex_var + float(variant.lambda_condinv) * mmd_loss
            loss.backward()
            nn.utils.clip_grad_norm_(base_params, 1.0)
            opt.step()
            losses.append(loss.detach())
            mmds.append(mmd_loss.detach())
        loss_mean = _mean_scalar(losses)
        if ep == 1 or ep == epochs or ep % 10 == 0:
            print(f"base_pretrain epoch={ep} loss={loss_mean:.5f}", flush=True)
        history.append(
            {
                "pretrain_epoch": int(ep),
                "base_pretrain_loss": loss_mean,
                "base_pretrain_condinv_loss": _mean_scalar(mmds),
            }
        )
    return history


def _train_select_and_correct(
    cfg: TrainDSTSelectorConfig,
    frames,
    out_dir: Path,
    seed: int,
    shared_assets: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    if shared_assets is None:
        shared_assets = _prepare_seed_shared_assets(cfg, frames, out_dir)
    feature_cols = shared_assets["feature_cols"]
    stage2_feature_cols = shared_assets["stage2_feature_cols"]
    base_cfg = _make_base_cfg_for_seed(cfg, int(seed))
    train_ds = shared_assets["train_ds"]
    train_loader = _make_train_loader(train_ds, base_cfg, cfg, int(seed))
    train_eval_loader = shared_assets["train_eval_loader"]
    valid_loader = shared_assets["valid_loader"]
    test_loader = shared_assets["test_loader"]
    stage2_train_ds = shared_assets["stage2_train_ds"]
    stage2_train_loader = (
        _make_train_loader(stage2_train_ds, base_cfg, cfg, int(seed))
        if stage2_train_ds is not None
        else None
    )
    stage2_valid_loader = shared_assets["stage2_valid_loader"]
    stage2_test_loader = shared_assets["stage2_test_loader"]
    finetune_train_ds = shared_assets["finetune_train_ds"]
    finetune_train_loader = (
        _make_train_loader(finetune_train_ds, base_cfg, cfg, int(seed) + 100003)
        if finetune_train_ds is not None
        else None
    )
    finetune_train_eval_loader = shared_assets["finetune_train_eval_loader"]
    train_regime_edges = shared_assets["train_regime_edges"]
    _write_regime_sampler_audit(train_ds, cfg, out_dir, int(seed))

    variant = CondInvVariant(
        f"condinv_{cfg.recurrent}{cfg.layers}_{cfg.temp_mode}_{cfg.head_kind}_mmd0p02_trainDST25_selector_base",
        recurrent=str(cfg.recurrent),
        lambda_condinv=float(cfg.lambda_condinv),
        lambda_rex=float(cfg.lambda_rex),
        weight_0=float(cfg.weight_0),
        weight_25=float(cfg.weight_25),
        weight_45=float(cfg.weight_45),
        layers=int(cfg.layers),
        kernel_size=int(cfg.kernel_size),
        head_kind=str(cfg.head_kind),
        temp_mode=str(cfg.temp_mode),
        dropout=float(cfg.dropout),
        sequence_training=bool(cfg.sequence_training),
    )
    model = _make_stage1_model(cfg, variant, input_dim=len(feature_cols), feature_cols=feature_cols)
    _voltage_shape_ssl_pretrain(model, train_loader, cfg, feature_cols, out_dir, int(seed))
    base_pretrain_history = _pretrain_base_estimator(model, train_loader, cfg, variant)
    if base_pretrain_history:
        pd.DataFrame(base_pretrain_history).to_csv(
            out_dir / f"{cfg.output_prefix}_seed{seed}_base_pretrain_history.csv",
            index=False,
        )
    if bool(cfg.freeze_base_after_pretrain) and hasattr(model, "freeze_base"):
        model.freeze_base()
    branch_pretrain_history = _pretrain_dual_context_branches(model, train_loader, cfg, variant)
    if branch_pretrain_history:
        pd.DataFrame(branch_pretrain_history).to_csv(
            out_dir / f"{cfg.output_prefix}_seed{seed}_dual_context_branch_pretrain_history.csv",
            index=False,
        )
    if bool(cfg.dual_context_freeze_branches_after_pretrain) and hasattr(model, "freeze_dual_context_branches"):
        model.freeze_dual_context_branches()
    profile_adv_head = None
    profile_to_label = {str(profile).upper(): idx for idx, profile in enumerate(cfg.train_profiles)}
    if float(cfg.lambda_profile_adv) > 0.0:
        profile_adv_head = nn.Sequential(
            nn.Linear(int(cfg.hidden_size), int(cfg.hidden_size)),
            nn.SiLU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(int(cfg.hidden_size), max(2, len(profile_to_label))),
        ).to(device)
    opt_params = [param for param in model.parameters() if param.requires_grad]
    if profile_adv_head is not None:
        opt_params.extend(profile_adv_head.parameters())
    opt = torch.optim.AdamW(opt_params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    legacy_variant = to_variant(variant)
    state_by_epoch = {}
    weight_ema_state = None
    metric_rows = []
    regime_metric_rows = []
    history = []
    selector_rows = []
    for ep in _epoch_iter(1, int(cfg.epochs) + 1, cfg, f"seed {seed} stage1"):
        model.train()
        losses = []
        mmds = []
        adv_losses = []
        adv_accs = []
        supcon_losses = []
        consistency_losses = []
        regime_cvar_losses = []
        regime_var_losses = []
        residual_aux_losses = []
        residual_zeromean_losses = []
        dual_context_aux_losses = []
        zbin_cvar_losses = []
        cold_socbin_cvar_losses = []
        cold_profile_socbin_cvar_losses = []
        cold_socbin_bias_losses = []
        cold_profile_socbin_underbias_losses = []
        mid_delta_losses = []
        voltage_shift_consistency_losses = []
        hard_region_weight_means = []
        cold_lowsoc_overpred_losses = []
        cold_lowsoc_underpred_losses = []
        for x, y, meta in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            y = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            x = _voltage_sag_augment(
                x,
                y,
                meta,
                feature_cols,
                prob=float(cfg.voltage_sag_augment_prob),
                scale=float(cfg.voltage_sag_augment_scale),
                soc_upper=float(cfg.voltage_sag_augment_soc_upper),
                cold_temp=float(cfg.voltage_sag_augment_temp),
            )
            h = model.encode_sequence(x)
            h_last = h[:, -1, :]
            if bool(variant.sequence_training):
                pred = model.forward_sequence(x)
                sample_loss = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=(1, 2))
                y_for_mmd = y[:, -1, :]
                pred_endpoint = pred[:, -1, :]
                y_endpoint = y[:, -1, :]
            else:
                pred = model(x)
                sample_loss = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=1)
                y_for_mmd = y
                pred_endpoint = pred
                y_endpoint = y
            hard_region_weights = _hard_region_sample_weights(
                x,
                y_for_mmd,
                feature_cols,
                loss_weight=float(cfg.hard_region_loss_weight),
                soc_threshold=float(cfg.hard_region_soc_threshold),
                soc_lower=float(cfg.hard_region_soc_lower),
                soc_upper=float(cfg.hard_region_soc_upper),
                dynamic_weight=float(cfg.hard_region_dynamic_weight),
                cold_only=bool(cfg.hard_region_cold_only),
            )
            if float(cfg.hard_region_loss_weight) > 0.0:
                sample_loss = sample_loss * hard_region_weights.detach()
                hard_region_weight_means.append(float(hard_region_weights.mean().detach().cpu()))
            if float(cfg.lambda_cold_lowsoc_overpred_loss) > 0.0:
                cold_lowsoc_overpred = _cold_lowsoc_overpred_loss(
                    pred,
                    y,
                    meta,
                    soc_upper=float(cfg.cold_lowsoc_overpred_soc_upper),
                    cold_temp=float(cfg.cold_lowsoc_overpred_temp),
                    beta=float(cfg.huber_beta),
                )
                sample_loss = sample_loss + float(cfg.lambda_cold_lowsoc_overpred_loss) * cold_lowsoc_overpred
                cold_lowsoc_overpred_losses.append(float(cold_lowsoc_overpred.mean().detach().cpu()))
            if float(cfg.lambda_cold_lowsoc_underpred_loss) > 0.0:
                cold_lowsoc_underpred = _cold_lowsoc_underpred_loss(
                    pred,
                    y,
                    meta,
                    soc_upper=float(cfg.cold_lowsoc_underpred_soc_upper),
                    cold_temp=float(cfg.cold_lowsoc_underpred_temp),
                    beta=float(cfg.huber_beta),
                )
                sample_loss = sample_loss + float(cfg.lambda_cold_lowsoc_underpred_loss) * cold_lowsoc_underpred
                cold_lowsoc_underpred_losses.append(float(cold_lowsoc_underpred.mean().detach().cpu()))
            if float(cfg.lambda_regime_asym_loss) > 0.0:
                regime_asym = _regime_gated_asymmetric_loss(
                    pred,
                    y,
                    meta,
                    x,
                    feature_cols,
                    soc_upper=float(cfg.regime_asym_soc_upper),
                    cold_temp=float(cfg.regime_asym_cold_temp),
                    under_weight=float(cfg.regime_asym_under_weight),
                    over_weight=float(cfg.regime_asym_over_weight),
                    beta=float(cfg.huber_beta),
                )
                sample_loss = sample_loss + float(cfg.lambda_regime_asym_loss) * regime_asym
            if float(cfg.lambda_lowdyn_highspan_under_loss) > 0.0:
                lowdyn_highspan_under = _lowdyn_highspan_under_loss(
                    pred,
                    y,
                    meta,
                    x,
                    feature_cols,
                    soc_upper=float(cfg.lowdyn_highspan_soc_upper),
                    cold_temp=float(cfg.lowdyn_highspan_cold_temp),
                    dynamic_center=float(cfg.lowdyn_highspan_dynamic_center),
                    dynamic_sharpness=float(cfg.lowdyn_highspan_dynamic_sharpness),
                    span_center=float(cfg.lowdyn_highspan_span_center),
                    span_sharpness=float(cfg.lowdyn_highspan_span_sharpness),
                    beta=float(cfg.huber_beta),
                )
                sample_loss = sample_loss + float(cfg.lambda_lowdyn_highspan_under_loss) * lowdyn_highspan_under
            if float(cfg.lambda_dual_context_branch_aux_loss) > 0.0 and hasattr(model, "branch_auxiliary_loss"):
                dual_context_aux = model.branch_auxiliary_loss(x, y, float(cfg.huber_beta))
                sample_loss = sample_loss + float(cfg.lambda_dual_context_branch_aux_loss) * dual_context_aux
                dual_context_aux_losses.append(float(dual_context_aux.mean().detach().cpu()))
            if float(cfg.lambda_cold_voltage_shift_consistency) > 0.0:
                voltage_shift_consistency = _cold_voltage_shift_consistency_loss(
                    model,
                    x,
                    y_for_mmd,
                    pred_endpoint,
                    meta,
                    feature_cols,
                    sequence_training=bool(variant.sequence_training),
                    scale=float(cfg.cold_voltage_shift_consistency_scale),
                    soc_upper=float(cfg.cold_voltage_shift_consistency_soc_upper),
                    cold_temp=float(cfg.cold_voltage_shift_consistency_temp),
                    beta=float(cfg.huber_beta),
                )
                sample_loss = sample_loss + float(cfg.lambda_cold_voltage_shift_consistency) * voltage_shift_consistency
                voltage_shift_consistency_losses.append(float(voltage_shift_consistency.detach().cpu()))
            sw = temp_weights(meta, legacy_variant, int(sample_loss.numel()))
            stack, wstack = _group_loss_weight_stacks(sample_loss, sw, meta, cfg.rex_group)
            mean_loss = (stack * wstack).sum() / wstack.sum().clamp_min(1e-6)
            rex_var = stack.var(unbiased=False) if len(stack) > 1 else stack.new_tensor(0.0)
            mmd_loss = conditional_profile_mmd(h_last, y_for_mmd, meta)
            loss = mean_loss + float(variant.lambda_rex) * rex_var + float(variant.lambda_condinv) * mmd_loss
            if float(cfg.lambda_cold_socbin_cvar) > 0.0:
                cold_socbin_cvar_loss = _cold_socbin_cvar_loss(
                    sample_loss,
                    sw,
                    y_for_mmd,
                    meta,
                    top_frac=float(cfg.cold_socbin_cvar_top_frac),
                    bin_width=float(cfg.cold_socbin_cvar_bin_width),
                    cold_temp=float(cfg.cold_socbin_cvar_temp),
                )
                loss = loss + float(cfg.lambda_cold_socbin_cvar) * cold_socbin_cvar_loss
                cold_socbin_cvar_losses.append(float(cold_socbin_cvar_loss.detach().cpu()))
            if float(cfg.lambda_cold_profile_socbin_cvar) > 0.0:
                cold_profile_socbin_cvar_loss = _cold_profile_socbin_cvar_loss(
                    sample_loss,
                    sw,
                    y_for_mmd,
                    meta,
                    top_frac=float(cfg.cold_profile_socbin_cvar_top_frac),
                    bin_width=float(cfg.cold_profile_socbin_cvar_bin_width),
                    cold_temp=float(cfg.cold_profile_socbin_cvar_temp),
                )
                loss = loss + float(cfg.lambda_cold_profile_socbin_cvar) * cold_profile_socbin_cvar_loss
                cold_profile_socbin_cvar_losses.append(float(cold_profile_socbin_cvar_loss.detach().cpu()))
            if float(cfg.lambda_cold_socbin_bias) > 0.0:
                cold_socbin_bias_loss = _cold_socbin_residual_bias_loss(
                    pred_endpoint,
                    y_endpoint,
                    meta,
                    bin_width=float(cfg.cold_socbin_bias_bin_width),
                    soc_upper=float(cfg.cold_socbin_bias_soc_upper),
                    cold_temp=float(cfg.cold_socbin_bias_temp),
                )
                loss = loss + float(cfg.lambda_cold_socbin_bias) * cold_socbin_bias_loss
                cold_socbin_bias_losses.append(float(cold_socbin_bias_loss.detach().cpu()))
            if float(cfg.lambda_cold_profile_socbin_underbias) > 0.0:
                cold_profile_socbin_underbias_loss = _cold_profile_socbin_underbias_loss(
                    pred_endpoint,
                    y_endpoint,
                    meta,
                    bin_width=float(cfg.cold_profile_socbin_underbias_bin_width),
                    soc_upper=float(cfg.cold_profile_socbin_underbias_soc_upper),
                    cold_temp=float(cfg.cold_profile_socbin_underbias_temp),
                    beta=float(cfg.huber_beta),
                )
                loss = loss + float(cfg.lambda_cold_profile_socbin_underbias) * cold_profile_socbin_underbias_loss
                cold_profile_socbin_underbias_losses.append(float(cold_profile_socbin_underbias_loss.detach().cpu()))
            if float(cfg.lambda_regime_cvar) > 0.0:
                regime_cvar_loss = _regime_cvar_loss(
                    sample_loss,
                    sw,
                    x,
                    meta,
                    feature_cols,
                    float(cfg.regime_cvar_top_frac),
                )
                loss = loss + float(cfg.lambda_regime_cvar) * regime_cvar_loss
                regime_cvar_losses.append(float(regime_cvar_loss.detach().cpu()))
            if float(cfg.lambda_regime_var) > 0.0:
                regime_var_loss = _regime_variance_loss(
                    sample_loss,
                    sw,
                    x,
                    meta,
                    feature_cols,
                )
                loss = loss + float(cfg.lambda_regime_var) * regime_var_loss
                regime_var_losses.append(float(regime_var_loss.detach().cpu()))
            if float(cfg.lambda_residual_aux_loss) > 0.0:
                residual_aux_loss = _residual_auxiliary_loss(
                    model,
                    x,
                    y,
                    bool(variant.sequence_training),
                    float(cfg.huber_beta),
                )
                loss = loss + float(cfg.lambda_residual_aux_loss) * residual_aux_loss
                residual_aux_losses.append(float(residual_aux_loss.detach().cpu()))
            if float(cfg.lambda_residual_zeromean_loss) > 0.0:
                residual_zeromean_loss = _residual_zero_mean_loss(
                    model,
                    x,
                    meta,
                    bool(variant.sequence_training),
                )
                loss = loss + float(cfg.lambda_residual_zeromean_loss) * residual_zeromean_loss
                residual_zeromean_losses.append(float(residual_zeromean_loss.detach().cpu()))
            if float(cfg.lambda_zbin_cvar) > 0.0:
                zbin_cvar_loss = _zbin_cvar_loss(
                    model,
                    sample_loss,
                    sw,
                    x,
                    feature_cols,
                    float(cfg.zbin_cvar_top_frac),
                )
                loss = loss + float(cfg.lambda_zbin_cvar) * zbin_cvar_loss
                zbin_cvar_losses.append(float(zbin_cvar_loss.detach().cpu()))
            if float(cfg.lambda_mid_delta_loss) > 0.0:
                mid_delta_loss = _mid_delta_loss(
                    model,
                    x,
                    bool(variant.sequence_training),
                    float(cfg.huber_beta),
                )
                loss = loss + float(cfg.lambda_mid_delta_loss) * mid_delta_loss
                mid_delta_losses.append(float(mid_delta_loss.detach().cpu()))
            if float(cfg.lambda_anchor_loss) > 0.0 and hasattr(model, "anchor_sequence"):
                anchor_seq = model.anchor_sequence(x)
                if bool(variant.sequence_training):
                    anchor_loss = F.smooth_l1_loss(anchor_seq, y, beta=float(cfg.huber_beta))
                else:
                    anchor_loss = F.smooth_l1_loss(anchor_seq[:, -1, :], y, beta=float(cfg.huber_beta))
                loss = loss + float(cfg.lambda_anchor_loss) * anchor_loss
            if float(cfg.lambda_profile_supcon) > 0.0:
                supcon_loss = _profile_supcon_loss(
                    h_last,
                    y_for_mmd,
                    meta,
                    temperature=float(cfg.supcon_temperature),
                )
                loss = loss + float(cfg.lambda_profile_supcon) * supcon_loss
                supcon_losses.append(float(supcon_loss.detach().cpu()))
            if profile_adv_head is not None:
                profile_labels = _profile_label_tensor(meta, profile_to_label)
                adv_logits = profile_adv_head(_grad_reverse(h_last, float(cfg.profile_adv_grl)))
                adv_loss = F.cross_entropy(adv_logits, profile_labels)
                loss = loss + float(cfg.lambda_profile_adv) * adv_loss
                adv_losses.append(float(adv_loss.detach().cpu()))
                adv_accs.append(float((adv_logits.argmax(dim=1) == profile_labels).float().mean().detach().cpu()))
            if float(cfg.lambda_current_consistency) > 0.0:
                x_aug = _augment_current_channel(
                    x,
                    noise_std=float(cfg.current_noise_std),
                    dropout_prob=float(cfg.current_dropout_prob),
                )
                pred_aug = model.forward_sequence(x_aug) if bool(variant.sequence_training) else model(x_aug)
                consistency = F.smooth_l1_loss(pred_aug, pred.detach(), beta=float(cfg.huber_beta))
                loss = loss + float(cfg.lambda_current_consistency) * consistency
                consistency_losses.append(float(consistency.detach().cpu()))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if float(cfg.weight_ema_decay) > 0.0 and int(ep) >= int(cfg.weight_ema_start_epoch):
                weight_ema_state = _update_weight_ema_state(weight_ema_state, model, float(cfg.weight_ema_decay))
            losses.append(loss.detach())
            mmds.append(mmd_loss.detach())
        loss_mean = _mean_scalar(losses)
        mmd_mean = _mean_scalar(mmds)
        if (
            float(cfg.weight_ema_decay) > 0.0
            and weight_ema_state is not None
            and int(ep) == int(cfg.epochs)
            and int(cfg.finetune_epochs) <= 0
        ):
            model.load_state_dict(weight_ema_state, strict=True)
        keep_epoch_state = (not bool(cfg.final_only_eval)) or int(ep) == int(cfg.epochs) or int(ep) == int(cfg.fixed_stage1_epoch)
        if keep_epoch_state:
            state_by_epoch[int(ep)] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(
            {
                "seed": int(seed),
                "epoch": int(ep),
                "variant": variant.name,
                "loss": loss_mean,
                "condinv_loss": mmd_mean,
                "profile_supcon_loss": float(np.mean(supcon_losses)) if supcon_losses else 0.0,
                "profile_adv_loss": float(np.mean(adv_losses)) if adv_losses else 0.0,
                "hard_region_weight_mean": float(np.mean(hard_region_weight_means)) if hard_region_weight_means else 1.0,
                "profile_adv_acc": float(np.mean(adv_accs)) if adv_accs else np.nan,
                "current_consistency_loss": float(np.mean(consistency_losses)) if consistency_losses else 0.0,
                "regime_cvar_loss": float(np.mean(regime_cvar_losses)) if regime_cvar_losses else 0.0,
                "regime_var_loss": float(np.mean(regime_var_losses)) if regime_var_losses else 0.0,
                "residual_aux_loss": float(np.mean(residual_aux_losses)) if residual_aux_losses else 0.0,
                "residual_zeromean_loss": float(np.mean(residual_zeromean_losses)) if residual_zeromean_losses else 0.0,
                "dual_context_branch_aux_loss": float(np.mean(dual_context_aux_losses)) if dual_context_aux_losses else 0.0,
                "zbin_cvar_loss": float(np.mean(zbin_cvar_losses)) if zbin_cvar_losses else 0.0,
                "cold_socbin_cvar_loss": float(np.mean(cold_socbin_cvar_losses)) if cold_socbin_cvar_losses else 0.0,
                "cold_profile_socbin_cvar_loss": float(np.mean(cold_profile_socbin_cvar_losses)) if cold_profile_socbin_cvar_losses else 0.0,
                "cold_socbin_bias_loss": float(np.mean(cold_socbin_bias_losses)) if cold_socbin_bias_losses else 0.0,
                "cold_profile_socbin_underbias_loss": float(np.mean(cold_profile_socbin_underbias_losses)) if cold_profile_socbin_underbias_losses else 0.0,
                "weight_ema_decay": float(cfg.weight_ema_decay),
                "weight_ema_start_epoch": int(cfg.weight_ema_start_epoch),
                "mid_delta_loss": float(np.mean(mid_delta_losses)) if mid_delta_losses else 0.0,
                "cold_voltage_shift_consistency_loss": float(np.mean(voltage_shift_consistency_losses)) if voltage_shift_consistency_losses else 0.0,
                "cold_lowsoc_overpred_loss": float(np.mean(cold_lowsoc_overpred_losses)) if cold_lowsoc_overpred_losses else 0.0,
                "cold_lowsoc_underpred_loss": float(np.mean(cold_lowsoc_underpred_losses)) if cold_lowsoc_underpred_losses else 0.0,
                "residual_limit_initial": model.residual_limit_initial() if hasattr(model, "residual_limit_initial") else np.nan,
                "residual_limit_value": model.residual_limit_value() if hasattr(model, "residual_limit_value") else np.nan,
            }
        )
        do_stage1_eval = (
            int(ep) == int(cfg.epochs)
            if bool(cfg.final_only_eval)
            else (
                int(ep) == 1
                or int(ep) % max(1, int(cfg.stage1_eval_every)) == 0
                or int(ep) == int(cfg.epochs)
            )
        )
        if do_stage1_eval:
            train_metrics = (
                eval_by_temp_drive(model, train_eval_loader, "train", variant.name, seed, ep)
                if train_eval_loader is not None
                else pd.DataFrame()
            )
            valid_metrics = _eval_by_temp_or_empty(model, valid_loader, "valid", variant.name, ep)
            test_metrics = pd.DataFrame()
            diagnostic_test = (
                bool(cfg.test_blind)
                and int(cfg.diagnostic_test_every) > 0
                and (int(ep) % int(cfg.diagnostic_test_every) == 0 or int(ep) == int(cfg.epochs))
            )
            if (not bool(cfg.test_blind)) or diagnostic_test:
                test_metrics = eval_by_temp(model, test_loader, "test", variant.name, ep)
                if diagnostic_test:
                    test_metrics["diagnostic_test_peek"] = True
            elif int(cfg.test_blind_rng_burn) > 0:
                _burn_global_rng(int(cfg.test_blind_rng_burn))
            for df in (valid_metrics, test_metrics):
                df["seed"] = int(seed)
            metric_rows.extend([df for df in (train_metrics, valid_metrics, test_metrics) if len(df)])
            valid_regime_metrics = pd.DataFrame()
            if valid_loader is not None and str(cfg.stage1_selector).startswith("val_regime"):
                valid_regime_metrics = eval_by_temp_regime(
                    model,
                    valid_loader,
                    "valid",
                    variant.name,
                    int(seed),
                    int(ep),
                    feature_cols,
                    train_regime_edges,
                )
                if len(valid_regime_metrics):
                    regime_metric_rows.append(valid_regime_metrics)
            selector_payload = _selector_score(
                train_metrics,
                str(cfg.stage1_selector),
                valid_metrics,
                valid_regime_metrics,
                int(cfg.selector_regime_min_windows),
            )
            selector_rows.append(
                {
                    "seed": int(seed),
                    "epoch": int(ep),
                    "selector": str(cfg.stage1_selector),
                    **selector_payload,
                }
            )
            report_metrics = valid_metrics if bool(cfg.test_blind) else test_metrics
            if len(report_metrics) and {"variant", "epoch", "split", "temperature_C", "MAE_pct"}.issubset(report_metrics.columns):
                piv = report_metrics.pivot_table(index=["variant", "epoch", "split"], columns="temperature_C", values="MAE_pct").reset_index()
                mae0 = float(piv.get(0.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
                mae25 = float(piv.get(25.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
                mae45 = float(piv.get(45.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
            else:
                mae0 = mae25 = mae45 = float("nan")
            label = "valid" if bool(cfg.test_blind) else "test"
            eval_summary = (
                "valid=none"
                if bool(cfg.test_blind) and not len(valid_metrics)
                else f"{label}0={mae0:.3f}% {label}25={mae25:.3f}% {label}45={mae45:.3f}%"
            )
            print(
                f"seed={seed} epoch={ep} loss={loss_mean:.5f} {cfg.stage1_selector}={selector_rows[-1]['selector_score']:.3f}% "
                f"{eval_summary}"
                + (" test=hidden" if bool(cfg.test_blind) else ""),
                flush=True,
            )
        else:
            if int(ep) % 10 == 0:
                print(f"seed={seed} epoch={ep} loss={loss_mean:.5f} eval=skipped", flush=True)

    if finetune_train_loader is not None:
        opt = torch.optim.AdamW(
            opt_params,
            lr=float(cfg.lr) * float(cfg.finetune_lr_scale),
            weight_decay=float(cfg.weight_decay),
        )
        total_epochs = int(cfg.epochs) + int(cfg.finetune_epochs)
        for ft_ep in range(1, int(cfg.finetune_epochs) + 1):
            ep = int(cfg.epochs) + int(ft_ep)
            model.train()
            losses = []
            mmds = []
            adv_losses = []
            adv_accs = []
            supcon_losses = []
            consistency_losses = []
            regime_cvar_losses = []
            regime_var_losses = []
            residual_aux_losses = []
            residual_zeromean_losses = []
            zbin_cvar_losses = []
            cold_socbin_cvar_losses = []
            cold_profile_socbin_cvar_losses = []
            cold_socbin_bias_losses = []
            cold_profile_socbin_underbias_losses = []
            mid_delta_losses = []
            voltage_shift_consistency_losses = []
            hard_region_weight_means = []
            cold_lowsoc_overpred_losses = []
            cold_lowsoc_underpred_losses = []
            for x, y, meta in finetune_train_loader:
                x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
                y = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
                x = _voltage_sag_augment(
                    x,
                    y,
                    meta,
                    feature_cols,
                    prob=float(cfg.voltage_sag_augment_prob),
                    scale=float(cfg.voltage_sag_augment_scale),
                    soc_upper=float(cfg.voltage_sag_augment_soc_upper),
                    cold_temp=float(cfg.voltage_sag_augment_temp),
                )
                h = model.encode_sequence(x)
                h_last = h[:, -1, :]
                if bool(variant.sequence_training):
                    pred = model.forward_sequence(x)
                    sample_loss = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=(1, 2))
                    y_for_mmd = y[:, -1, :]
                    pred_endpoint = pred[:, -1, :]
                    y_endpoint = y[:, -1, :]
                else:
                    pred = model(x)
                    sample_loss = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=1)
                    y_for_mmd = y
                    pred_endpoint = pred
                    y_endpoint = y
                hard_region_weights = _hard_region_sample_weights(
                    x,
                    y_for_mmd,
                    feature_cols,
                    loss_weight=float(cfg.hard_region_loss_weight),
                    soc_threshold=float(cfg.hard_region_soc_threshold),
                    soc_lower=float(cfg.hard_region_soc_lower),
                    soc_upper=float(cfg.hard_region_soc_upper),
                    dynamic_weight=float(cfg.hard_region_dynamic_weight),
                    cold_only=bool(cfg.hard_region_cold_only),
                )
                if float(cfg.hard_region_loss_weight) > 0.0:
                    sample_loss = sample_loss * hard_region_weights.detach()
                    hard_region_weight_means.append(float(hard_region_weights.mean().detach().cpu()))
                if float(cfg.lambda_cold_lowsoc_overpred_loss) > 0.0:
                    cold_lowsoc_overpred = _cold_lowsoc_overpred_loss(
                        pred,
                        y,
                        meta,
                        soc_upper=float(cfg.cold_lowsoc_overpred_soc_upper),
                        cold_temp=float(cfg.cold_lowsoc_overpred_temp),
                        beta=float(cfg.huber_beta),
                    )
                    sample_loss = sample_loss + float(cfg.lambda_cold_lowsoc_overpred_loss) * cold_lowsoc_overpred
                    cold_lowsoc_overpred_losses.append(float(cold_lowsoc_overpred.mean().detach().cpu()))
                if float(cfg.lambda_cold_lowsoc_underpred_loss) > 0.0:
                    cold_lowsoc_underpred = _cold_lowsoc_underpred_loss(
                        pred,
                        y,
                        meta,
                        soc_upper=float(cfg.cold_lowsoc_underpred_soc_upper),
                        cold_temp=float(cfg.cold_lowsoc_underpred_temp),
                        beta=float(cfg.huber_beta),
                    )
                    sample_loss = sample_loss + float(cfg.lambda_cold_lowsoc_underpred_loss) * cold_lowsoc_underpred
                    cold_lowsoc_underpred_losses.append(float(cold_lowsoc_underpred.mean().detach().cpu()))
                if float(cfg.lambda_regime_asym_loss) > 0.0:
                    regime_asym = _regime_gated_asymmetric_loss(
                        pred,
                        y,
                        meta,
                        x,
                        feature_cols,
                        soc_upper=float(cfg.regime_asym_soc_upper),
                        cold_temp=float(cfg.regime_asym_cold_temp),
                        under_weight=float(cfg.regime_asym_under_weight),
                        over_weight=float(cfg.regime_asym_over_weight),
                        beta=float(cfg.huber_beta),
                    )
                    sample_loss = sample_loss + float(cfg.lambda_regime_asym_loss) * regime_asym
                if float(cfg.lambda_lowdyn_highspan_under_loss) > 0.0:
                    lowdyn_highspan_under = _lowdyn_highspan_under_loss(
                        pred,
                        y,
                        meta,
                        x,
                        feature_cols,
                        soc_upper=float(cfg.lowdyn_highspan_soc_upper),
                        cold_temp=float(cfg.lowdyn_highspan_cold_temp),
                        dynamic_center=float(cfg.lowdyn_highspan_dynamic_center),
                        dynamic_sharpness=float(cfg.lowdyn_highspan_dynamic_sharpness),
                        span_center=float(cfg.lowdyn_highspan_span_center),
                        span_sharpness=float(cfg.lowdyn_highspan_span_sharpness),
                        beta=float(cfg.huber_beta),
                    )
                    sample_loss = sample_loss + float(cfg.lambda_lowdyn_highspan_under_loss) * lowdyn_highspan_under
                if float(cfg.lambda_cold_voltage_shift_consistency) > 0.0:
                    voltage_shift_consistency = _cold_voltage_shift_consistency_loss(
                        model,
                        x,
                        y_for_mmd,
                        pred_endpoint,
                        meta,
                        feature_cols,
                        sequence_training=bool(variant.sequence_training),
                        scale=float(cfg.cold_voltage_shift_consistency_scale),
                        soc_upper=float(cfg.cold_voltage_shift_consistency_soc_upper),
                        cold_temp=float(cfg.cold_voltage_shift_consistency_temp),
                        beta=float(cfg.huber_beta),
                    )
                    sample_loss = sample_loss + float(cfg.lambda_cold_voltage_shift_consistency) * voltage_shift_consistency
                    voltage_shift_consistency_losses.append(float(voltage_shift_consistency.detach().cpu()))
                sw = temp_weights(meta, legacy_variant, int(sample_loss.numel()))
                stack, wstack = _group_loss_weight_stacks(sample_loss, sw, meta, cfg.rex_group)
                mean_loss = (stack * wstack).sum() / wstack.sum().clamp_min(1e-6)
                rex_var = stack.var(unbiased=False) if len(stack) > 1 else stack.new_tensor(0.0)
                mmd_loss = conditional_profile_mmd(h_last, y_for_mmd, meta)
                loss = mean_loss + float(variant.lambda_rex) * rex_var + float(variant.lambda_condinv) * mmd_loss
                if float(cfg.lambda_cold_socbin_cvar) > 0.0:
                    cold_socbin_cvar_loss = _cold_socbin_cvar_loss(
                        sample_loss,
                        sw,
                        y_for_mmd,
                        meta,
                        top_frac=float(cfg.cold_socbin_cvar_top_frac),
                        bin_width=float(cfg.cold_socbin_cvar_bin_width),
                        cold_temp=float(cfg.cold_socbin_cvar_temp),
                    )
                    loss = loss + float(cfg.lambda_cold_socbin_cvar) * cold_socbin_cvar_loss
                    cold_socbin_cvar_losses.append(float(cold_socbin_cvar_loss.detach().cpu()))
                if float(cfg.lambda_cold_profile_socbin_cvar) > 0.0:
                    cold_profile_socbin_cvar_loss = _cold_profile_socbin_cvar_loss(
                        sample_loss,
                        sw,
                        y_for_mmd,
                        meta,
                        top_frac=float(cfg.cold_profile_socbin_cvar_top_frac),
                        bin_width=float(cfg.cold_profile_socbin_cvar_bin_width),
                        cold_temp=float(cfg.cold_profile_socbin_cvar_temp),
                    )
                    loss = loss + float(cfg.lambda_cold_profile_socbin_cvar) * cold_profile_socbin_cvar_loss
                    cold_profile_socbin_cvar_losses.append(float(cold_profile_socbin_cvar_loss.detach().cpu()))
                if float(cfg.lambda_cold_socbin_bias) > 0.0:
                    cold_socbin_bias_loss = _cold_socbin_residual_bias_loss(
                        pred_endpoint,
                        y_endpoint,
                        meta,
                        bin_width=float(cfg.cold_socbin_bias_bin_width),
                        soc_upper=float(cfg.cold_socbin_bias_soc_upper),
                        cold_temp=float(cfg.cold_socbin_bias_temp),
                    )
                    loss = loss + float(cfg.lambda_cold_socbin_bias) * cold_socbin_bias_loss
                    cold_socbin_bias_losses.append(float(cold_socbin_bias_loss.detach().cpu()))
                if float(cfg.lambda_cold_profile_socbin_underbias) > 0.0:
                    cold_profile_socbin_underbias_loss = _cold_profile_socbin_underbias_loss(
                        pred_endpoint,
                        y_endpoint,
                        meta,
                        bin_width=float(cfg.cold_profile_socbin_underbias_bin_width),
                        soc_upper=float(cfg.cold_profile_socbin_underbias_soc_upper),
                        cold_temp=float(cfg.cold_profile_socbin_underbias_temp),
                        beta=float(cfg.huber_beta),
                    )
                    loss = loss + float(cfg.lambda_cold_profile_socbin_underbias) * cold_profile_socbin_underbias_loss
                    cold_profile_socbin_underbias_losses.append(float(cold_profile_socbin_underbias_loss.detach().cpu()))
                if float(cfg.lambda_regime_cvar) > 0.0:
                    regime_cvar_loss = _regime_cvar_loss(
                        sample_loss,
                        sw,
                        x,
                        meta,
                        feature_cols,
                        float(cfg.regime_cvar_top_frac),
                    )
                    loss = loss + float(cfg.lambda_regime_cvar) * regime_cvar_loss
                    regime_cvar_losses.append(float(regime_cvar_loss.detach().cpu()))
                if float(cfg.lambda_regime_var) > 0.0:
                    regime_var_loss = _regime_variance_loss(
                        sample_loss,
                        sw,
                        x,
                        meta,
                        feature_cols,
                    )
                    loss = loss + float(cfg.lambda_regime_var) * regime_var_loss
                    regime_var_losses.append(float(regime_var_loss.detach().cpu()))
                if float(cfg.lambda_residual_aux_loss) > 0.0:
                    residual_aux_loss = _residual_auxiliary_loss(
                        model,
                        x,
                        y,
                        bool(variant.sequence_training),
                        float(cfg.huber_beta),
                    )
                    loss = loss + float(cfg.lambda_residual_aux_loss) * residual_aux_loss
                    residual_aux_losses.append(float(residual_aux_loss.detach().cpu()))
                if float(cfg.lambda_residual_zeromean_loss) > 0.0:
                    residual_zeromean_loss = _residual_zero_mean_loss(
                        model,
                        x,
                        meta,
                        bool(variant.sequence_training),
                    )
                    loss = loss + float(cfg.lambda_residual_zeromean_loss) * residual_zeromean_loss
                    residual_zeromean_losses.append(float(residual_zeromean_loss.detach().cpu()))
                if float(cfg.lambda_zbin_cvar) > 0.0:
                    zbin_cvar_loss = _zbin_cvar_loss(
                        model,
                        sample_loss,
                        sw,
                        x,
                        feature_cols,
                        float(cfg.zbin_cvar_top_frac),
                    )
                    loss = loss + float(cfg.lambda_zbin_cvar) * zbin_cvar_loss
                    zbin_cvar_losses.append(float(zbin_cvar_loss.detach().cpu()))
                if float(cfg.lambda_mid_delta_loss) > 0.0:
                    mid_delta_loss = _mid_delta_loss(
                        model,
                        x,
                        bool(variant.sequence_training),
                        float(cfg.huber_beta),
                    )
                    loss = loss + float(cfg.lambda_mid_delta_loss) * mid_delta_loss
                    mid_delta_losses.append(float(mid_delta_loss.detach().cpu()))
                if float(cfg.lambda_anchor_loss) > 0.0 and hasattr(model, "anchor_sequence"):
                    anchor_seq = model.anchor_sequence(x)
                    if bool(variant.sequence_training):
                        anchor_loss = F.smooth_l1_loss(anchor_seq, y, beta=float(cfg.huber_beta))
                    else:
                        anchor_loss = F.smooth_l1_loss(anchor_seq[:, -1, :], y, beta=float(cfg.huber_beta))
                    loss = loss + float(cfg.lambda_anchor_loss) * anchor_loss
                if float(cfg.lambda_profile_supcon) > 0.0:
                    supcon_loss = _profile_supcon_loss(
                        h_last,
                        y_for_mmd,
                        meta,
                        temperature=float(cfg.supcon_temperature),
                    )
                    loss = loss + float(cfg.lambda_profile_supcon) * supcon_loss
                    supcon_losses.append(float(supcon_loss.detach().cpu()))
                if profile_adv_head is not None:
                    profile_labels = _profile_label_tensor(meta, profile_to_label)
                    adv_logits = profile_adv_head(_grad_reverse(h_last, float(cfg.profile_adv_grl)))
                    adv_loss = F.cross_entropy(adv_logits, profile_labels)
                    loss = loss + float(cfg.lambda_profile_adv) * adv_loss
                    adv_losses.append(float(adv_loss.detach().cpu()))
                    adv_accs.append(float((adv_logits.argmax(dim=1) == profile_labels).float().mean().detach().cpu()))
                if float(cfg.lambda_current_consistency) > 0.0:
                    x_aug = _augment_current_channel(
                        x,
                        noise_std=float(cfg.current_noise_std),
                        dropout_prob=float(cfg.current_dropout_prob),
                    )
                    pred_aug = model.forward_sequence(x_aug) if bool(variant.sequence_training) else model(x_aug)
                    consistency = F.smooth_l1_loss(pred_aug, pred.detach(), beta=float(cfg.huber_beta))
                    loss = loss + float(cfg.lambda_current_consistency) * consistency
                    consistency_losses.append(float(consistency.detach().cpu()))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(loss.detach())
                mmds.append(mmd_loss.detach())
            loss_mean = _mean_scalar(losses)
            mmd_mean = _mean_scalar(mmds)
            keep_epoch_state = (not bool(cfg.final_only_eval)) or int(ep) == int(total_epochs) or int(ep) == int(cfg.fixed_stage1_epoch)
            if keep_epoch_state:
                state_by_epoch[int(ep)] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history.append(
                {
                    "seed": int(seed),
                    "epoch": int(ep),
                    "variant": variant.name,
                    "phase": "finetune",
                    "loss": loss_mean,
                    "condinv_loss": mmd_mean,
                        "profile_supcon_loss": float(np.mean(supcon_losses)) if supcon_losses else 0.0,
                        "profile_adv_loss": float(np.mean(adv_losses)) if adv_losses else 0.0,
                        "hard_region_weight_mean": float(np.mean(hard_region_weight_means)) if hard_region_weight_means else 1.0,
                        "profile_adv_acc": float(np.mean(adv_accs)) if adv_accs else np.nan,
                    "current_consistency_loss": float(np.mean(consistency_losses)) if consistency_losses else 0.0,
                    "regime_cvar_loss": float(np.mean(regime_cvar_losses)) if regime_cvar_losses else 0.0,
                    "regime_var_loss": float(np.mean(regime_var_losses)) if regime_var_losses else 0.0,
                    "residual_aux_loss": float(np.mean(residual_aux_losses)) if residual_aux_losses else 0.0,
                    "residual_zeromean_loss": float(np.mean(residual_zeromean_losses)) if residual_zeromean_losses else 0.0,
                    "zbin_cvar_loss": float(np.mean(zbin_cvar_losses)) if zbin_cvar_losses else 0.0,
                    "cold_socbin_cvar_loss": float(np.mean(cold_socbin_cvar_losses)) if cold_socbin_cvar_losses else 0.0,
                    "cold_profile_socbin_cvar_loss": float(np.mean(cold_profile_socbin_cvar_losses)) if cold_profile_socbin_cvar_losses else 0.0,
                    "cold_socbin_bias_loss": float(np.mean(cold_socbin_bias_losses)) if cold_socbin_bias_losses else 0.0,
                    "cold_profile_socbin_underbias_loss": float(np.mean(cold_profile_socbin_underbias_losses)) if cold_profile_socbin_underbias_losses else 0.0,
                    "mid_delta_loss": float(np.mean(mid_delta_losses)) if mid_delta_losses else 0.0,
                    "cold_voltage_shift_consistency_loss": float(np.mean(voltage_shift_consistency_losses)) if voltage_shift_consistency_losses else 0.0,
                    "cold_lowsoc_overpred_loss": float(np.mean(cold_lowsoc_overpred_losses)) if cold_lowsoc_overpred_losses else 0.0,
                    "cold_lowsoc_underpred_loss": float(np.mean(cold_lowsoc_underpred_losses)) if cold_lowsoc_underpred_losses else 0.0,
                    "residual_limit_initial": model.residual_limit_initial() if hasattr(model, "residual_limit_initial") else np.nan,
                    "residual_limit_value": model.residual_limit_value() if hasattr(model, "residual_limit_value") else np.nan,
                }
            )
            do_stage1_eval = (
                int(ep) == int(total_epochs)
                if bool(cfg.final_only_eval)
                else (
                    int(ep) == int(cfg.epochs) + 1
                    or int(ep) % max(1, int(cfg.stage1_eval_every)) == 0
                    or int(ep) == total_epochs
                )
            )
            if do_stage1_eval:
                train_metrics = (
                    eval_by_temp_drive(model, finetune_train_eval_loader, "train", variant.name, seed, ep)
                    if finetune_train_eval_loader is not None
                    else pd.DataFrame()
                )
                valid_metrics = _eval_by_temp_or_empty(model, valid_loader, "valid", variant.name, ep)
                test_metrics = pd.DataFrame()
                diagnostic_test = (
                    bool(cfg.test_blind)
                    and int(cfg.diagnostic_test_every) > 0
                    and (int(ep) % int(cfg.diagnostic_test_every) == 0 or int(ep) == total_epochs)
                )
                if (not bool(cfg.test_blind)) or diagnostic_test:
                    test_metrics = eval_by_temp(model, test_loader, "test", variant.name, ep)
                    if diagnostic_test:
                        test_metrics["diagnostic_test_peek"] = True
                elif int(cfg.test_blind_rng_burn) > 0:
                    _burn_global_rng(int(cfg.test_blind_rng_burn))
                for df in (valid_metrics, test_metrics):
                    df["seed"] = int(seed)
                metric_rows.extend([df for df in (train_metrics, valid_metrics, test_metrics) if len(df)])
                valid_regime_metrics = pd.DataFrame()
                if valid_loader is not None and str(cfg.stage1_selector).startswith("val_regime"):
                    valid_regime_metrics = eval_by_temp_regime(
                        model,
                        valid_loader,
                        "valid",
                        variant.name,
                        int(seed),
                        int(ep),
                        feature_cols,
                        train_regime_edges,
                    )
                    if len(valid_regime_metrics):
                        valid_regime_metrics["stage1_phase"] = "finetune"
                        regime_metric_rows.append(valid_regime_metrics)
                selector_payload = _selector_score(
                    train_metrics,
                    str(cfg.stage1_selector),
                    valid_metrics,
                    valid_regime_metrics,
                    int(cfg.selector_regime_min_windows),
                )
                selector_rows.append(
                    {
                        "seed": int(seed),
                        "epoch": int(ep),
                        "selector": str(cfg.stage1_selector),
                        "stage1_phase": "finetune",
                        **selector_payload,
                    }
                )
                report_metrics = valid_metrics if bool(cfg.test_blind) else test_metrics
                if len(report_metrics) and {"variant", "epoch", "split", "temperature_C", "MAE_pct"}.issubset(report_metrics.columns):
                    piv = report_metrics.pivot_table(index=["variant", "epoch", "split"], columns="temperature_C", values="MAE_pct").reset_index()
                    mae0 = float(piv.get(0.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
                    mae25 = float(piv.get(25.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
                    mae45 = float(piv.get(45.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
                else:
                    mae0 = mae25 = mae45 = float("nan")
                label = "valid" if bool(cfg.test_blind) else "test"
                eval_summary = (
                    "valid=none"
                    if bool(cfg.test_blind) and not len(valid_metrics)
                    else f"{label}0={mae0:.3f}% {label}25={mae25:.3f}% {label}45={mae45:.3f}%"
                )
                print(
                    f"seed={seed} finetune_epoch={ft_ep}/{cfg.finetune_epochs} epoch={ep} "
                    f"loss={loss_mean:.5f} {cfg.stage1_selector}={selector_rows[-1]['selector_score']:.3f}% "
                    f"{eval_summary}"
                    + (" test=hidden" if bool(cfg.test_blind) else ""),
                    flush=True,
                )
            else:
                if int(ft_ep) % 10 == 0:
                    print(
                        f"seed={seed} finetune_epoch={ft_ep}/{cfg.finetune_epochs} epoch={ep} "
                        f"loss={loss_mean:.5f} eval=skipped",
                        flush=True,
                    )

    selector_df = pd.DataFrame(selector_rows)
    selector_df["selector_min_epoch"] = int(cfg.selector_min_epoch)
    selector_df["selector_max_epoch"] = int(cfg.selector_max_epoch)
    if regime_metric_rows:
        pd.concat(regime_metric_rows, ignore_index=True).to_csv(
            out_dir / f"{cfg.output_prefix}_seed{seed}_valid_regime_metrics.csv",
            index=False,
        )
    selector_pool = selector_df.copy()
    if int(cfg.selector_min_epoch) > 1:
        selector_pool = selector_pool[selector_pool["epoch"].ge(int(cfg.selector_min_epoch))].copy()
    if int(cfg.selector_max_epoch) > 0:
        selector_pool = selector_pool[selector_pool["epoch"].le(int(cfg.selector_max_epoch))].copy()
    if not len(selector_pool):
        raise RuntimeError(
            f"No selector candidates remain for selector_min_epoch={cfg.selector_min_epoch}, "
            f"selector_max_epoch={cfg.selector_max_epoch}"
        )
    if int(cfg.fixed_stage1_epoch) > 0:
        selected_epoch = int(cfg.fixed_stage1_epoch)
        if selected_epoch not in state_by_epoch:
            raise RuntimeError(
                f"fixed_stage1_epoch={selected_epoch} is missing from saved states. "
                f"Available epoch range: 1..{max(state_by_epoch)}"
            )
        selected_score = float(
            selector_pool.loc[selector_pool["epoch"].eq(selected_epoch), "selector_score"].iloc[0]
        ) if selected_epoch in set(selector_pool["epoch"].astype(int)) else float("nan")
    else:
        selected_epoch = int(selector_pool.sort_values(["selector_score", "epoch"]).iloc[0]["epoch"])
        selected_score = float(selector_pool.loc[selector_pool["epoch"].eq(selected_epoch), "selector_score"].iloc[0])
    selected_model, ensemble_epochs = _make_selected_stage1_base(
        cfg,
        variant,
        state_by_epoch,
        selected_epoch,
        input_dim=len(feature_cols),
    )
    setattr(selected_model, "correction_input_dim_for_stage2", int(len(stage2_feature_cols)))
    selected_name = f"condinv_trainDST25_selected_seed{seed}_ep{selected_epoch}_base"
    if bool(cfg.save_checkpoints):
        checkpoint_path = out_dir / f"{cfg.output_prefix}_seed{seed}_sel{selected_epoch}_{selected_name}_checkpoint.pt"
        torch.save(
            {
                "model_state_dict": {k: v.detach().cpu() for k, v in selected_model.state_dict().items()},
                "output_prefix": str(cfg.output_prefix),
                "seed": int(seed),
                "selected_epoch": int(selected_epoch),
                "selected_name": str(selected_name),
                "model_kind": str(cfg.model_kind),
                "recurrent": str(cfg.recurrent),
                "feature_set": str(cfg.feature_set),
                "feature_cols": list(feature_cols),
                "stage1_ensemble_epochs": [int(ep) for ep in ensemble_epochs],
            },
            checkpoint_path,
        )
    if bool(cfg.save_final_weights):
        final_weights_path = out_dir / f"{cfg.output_prefix}_seed{seed}_final_epoch{selected_epoch}_weights.pt"
        torch.save(
            {
                "model_state_dict": {k: v.detach().cpu() for k, v in selected_model.state_dict().items()},
                "output_prefix": str(cfg.output_prefix),
                "seed": int(seed),
                "final_epoch": int(selected_epoch),
                "model_kind": str(cfg.model_kind),
                "recurrent": str(cfg.recurrent),
                "feature_set": str(cfg.feature_set),
                "feature_cols": list(feature_cols),
            },
            final_weights_path,
        )
    base_valid = _eval_by_temp_or_empty(selected_model, valid_loader, "valid", selected_name, selected_epoch)
    base_test = (
        pd.DataFrame()
        if bool(cfg.test_blind) and int(cfg.diagnostic_test_every) <= 0
        else eval_by_temp(selected_model, test_loader, "test", selected_name, selected_epoch)
    )
    if len(base_test) and bool(cfg.test_blind):
        base_test["diagnostic_test_peek"] = True
    for df in (base_valid, base_test):
        df["seed"] = int(seed)
        df["selector"] = str(cfg.stage1_selector)
        df["selector_min_epoch"] = int(cfg.selector_min_epoch)
        df["selector_max_epoch"] = int(cfg.selector_max_epoch)
        df["selected_epoch"] = selected_epoch
        df["selector_score"] = selected_score
        df["stage1_ensemble_epochs"] = ",".join(str(ep) for ep in ensemble_epochs)

    if bool(cfg.ema_perturbation_importance):
        pert = _ema_perturbation_importance(
            selected_model,
            test_loader,
            feature_cols,
            selected_name,
            int(selected_epoch),
            int(seed),
            int(cfg.batch_size),
        )
        if len(pert):
            pert.to_csv(
                out_dir / f"{cfg.output_prefix}_seed{seed}_sel{selected_epoch}_ema_perturbation_importance.csv",
                index=False,
            )

    if bool(cfg.skip_stage2):
        if bool(cfg.save_predictions):
            pred_prefix = f"{cfg.output_prefix}_seed{seed}_sel{selected_epoch}_{selected_name}"
            if bool(cfg.save_train_predictions) and train_eval_loader is not None:
                train_pred = _predict_rows(selected_model, train_eval_loader, "train", selected_name, int(selected_epoch))
                if len(train_pred):
                    train_pred.to_csv(
                        out_dir / f"{pred_prefix}_train_prediction_rows.csv.gz",
                        index=False,
                        compression="gzip",
                    )
            valid_pred = _predict_rows(selected_model, valid_loader, "valid", selected_name, int(selected_epoch))
            if len(valid_pred):
                valid_pred.to_csv(
                    out_dir / f"{pred_prefix}_valid_prediction_rows.csv.gz",
                    index=False,
                    compression="gzip",
                )
            test_pred = _predict_rows(selected_model, test_loader, "test", selected_name, int(selected_epoch))
            if bool(cfg.test_blind):
                test_pred["diagnostic_test_peek"] = True
            test_pred.to_csv(
                out_dir / f"{pred_prefix}_test_prediction_rows.csv.gz",
                index=False,
                compression="gzip",
            )
        print(
            f"seed={seed} selected_epoch={selected_epoch} selector_max_epoch={cfg.selector_max_epoch} "
            f"selector_score={selected_score:.3f}% stage1_ensemble_epochs={ensemble_epochs} skip_stage2=True",
            flush=True,
        )
        out_frames = [df for df in (metric_rows + [base_valid, base_test]) if len(df)]
        return pd.concat(out_frames, ignore_index=True), pd.DataFrame(history), selector_df

    stage2_variant = CondInvVariant(
        f"condinv_trainDST25_selected_seed{seed}_ep{selected_epoch}_stage2_0_45_corr",
        lambda_condinv=0.02,
        enable_stage2=True,
        corr_limit=float(cfg.corr_limit),
        corr_mode=str(cfg.corr_mode),
    )
    cond_cfg = CondInvConfig(
        base_dir=cfg.base_dir,
        output_prefix=f"{cfg.output_prefix}_seed{seed}_sel{selected_epoch}",
        hidden_size=cfg.hidden_size,
        window_len=cfg.window_len,
        stage2_epochs=cfg.stage2_epochs,
        lr_stage2=cfg.lr_stage2,
        weight_decay=cfg.weight_decay,
        huber_beta=cfg.huber_beta,
        focus45_weight=cfg.focus45_weight,
        keep_lambda=cfg.keep_lambda,
        eval_every=cfg.eval_every,
    )
    if bool(cfg.test_blind):
        stage2_rows = _train_stage2_correction_test_blind(
            cfg,
            stage2_variant,
            selected_model,
            stage2_train_loader,
            stage2_valid_loader,
            stage2_test_loader,
            int(seed),
            int(selected_epoch),
        )
    else:
        stage2_rows = train_stage2_correction(cond_cfg, stage2_variant, selected_model, stage2_train_loader, stage2_valid_loader, stage2_test_loader)
    if len(stage2_rows):
        stage2_rows["seed"] = int(seed)
        stage2_rows["selector"] = str(cfg.stage1_selector)
        stage2_rows["selector_min_epoch"] = int(cfg.selector_min_epoch)
        stage2_rows["selector_max_epoch"] = int(cfg.selector_max_epoch)
        stage2_rows["selected_epoch"] = selected_epoch
        stage2_rows["selector_score"] = selected_score
        stage2_rows["stage1_ensemble_epochs"] = ",".join(str(ep) for ep in ensemble_epochs)
    print(
        f"seed={seed} selected_epoch={selected_epoch} selector_max_epoch={cfg.selector_max_epoch} "
        f"selector_score={selected_score:.3f}% stage1_ensemble_epochs={ensemble_epochs}",
        flush=True,
    )
    out_frames = [df for df in (metric_rows + [base_valid, base_test, stage2_rows]) if len(df)]
    return pd.concat(out_frames, ignore_index=True), pd.DataFrame(history), selector_df


def run(cfg: TrainDSTSelectorConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg.base_dir = Path(cfg.base_dir).resolve()
    if not Path(cfg.raw_root).is_absolute():
        cfg.raw_root = cfg.base_dir / cfg.raw_root
    cfg.v_corr_variant = _effective_v_corr_variant(cfg)
    out_dir = cfg.base_dir / "nmc_goal_vcorr_it_train_dst_selector_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_torch_runtime()
    files = _cached_find_csv_files(cfg.raw_root)
    raw_source_columns = _cached_raw_source_columns(files[0])
    write_start_audit(files, out_dir / f"{cfg.output_prefix}_file_start_audit.csv")
    r0_df = _cached_r0_by_temperature(cfg, files)
    r0_df.to_csv(out_dir / f"{cfg.output_prefix}_decomposition_params.csv", index=False)
    if str(cfg.v_corr_variant) == "ocvcal_dynfit_v1":
        ocv_cal = estimate_vcorr_ocv_calibration(files, tuple(cfg.train_profiles), r0_df, cfg)
        ocv_cal["params_df"].to_csv(out_dir / f"{cfg.output_prefix}_ocvcal_params.csv", index=False)
        ocv_cal["curve_df"].to_csv(out_dir / f"{cfg.output_prefix}_ocvcal_curve.csv", index=False)
    if str(cfg.v_corr_variant) == "dynr_vit_event_ema10":
        dynr_cal = estimate_dynamic_r_calibration(files, tuple(cfg.train_profiles), r0_df, cfg)
        dynr_cal["params_df"].to_csv(out_dir / f"{cfg.output_prefix}_dynr_params.csv", index=False)
        dynr_cal["coef_df"].to_csv(out_dir / f"{cfg.output_prefix}_dynr_coef.csv", index=False)
        dynr_cal["event_df"].to_csv(out_dir / f"{cfg.output_prefix}_dynr_events.csv", index=False)
    lfp_style_cal = None
    if str(cfg.v_corr_variant) in {"lfpstyle_ocvfit_v1", "lfpstyle_ocvfit"}:
        lfp_style_cal = estimate_lfp_style_vcorr_calibration(files, tuple(cfg.train_profiles), r0_df, cfg)
        lfp_style_cal["params_df"].to_csv(out_dir / f"{cfg.output_prefix}_lfpstyle_params.csv", index=False)
        lfp_style_cal["fit_df"].to_csv(out_dir / f"{cfg.output_prefix}_lfpstyle_train_fit.csv", index=False)
        lfp_style_cal["ocv_curve_df"].to_csv(out_dir / f"{cfg.output_prefix}_lfpstyle_ocv_curve.csv", index=False)
        lfp_style_cal["ocv_params_df"].to_csv(out_dir / f"{cfg.output_prefix}_lfpstyle_ocv_params.csv", index=False)
    nmc_tailored_cal = None
    if str(cfg.v_corr_variant) in {
        "nmc_tailored_minimax_v1",
        "nmc_tailored_minimax",
        "nmc_tailored_minimax_aligned_v1",
        "nmc_tailored_minimax_aligned",
    }:
        nmc_tailored_cal = estimate_nmc_tailored_minimax_calibration(files, tuple(cfg.train_profiles), r0_df, cfg)
        nmc_tailored_cal["params_df"].to_csv(out_dir / f"{cfg.output_prefix}_nmc_tailored_minimax_params.csv", index=False)
        nmc_tailored_cal["curve_df"].to_csv(out_dir / f"{cfg.output_prefix}_nmc_tailored_minimax_ocv_curve.csv", index=False)
        nmc_tailored_cal["alpha_scores_df"].to_csv(out_dir / f"{cfg.output_prefix}_nmc_tailored_minimax_alpha_scores.csv", index=False)
        if len(nmc_tailored_cal.get("alignment_scores_df", pd.DataFrame())):
            nmc_tailored_cal["alignment_scores_df"].to_csv(
                out_dir / f"{cfg.output_prefix}_nmc_tailored_minimax_alignment_scores.csv",
                index=False,
            )
    rtvar_lowv_cal = None
    if str(cfg.v_corr_variant) in {"rtvar_lowv_ema120", "endzero_rtvar_lowv_ema120"}:
        rtvar_lowv_cal = estimate_rtvar_lowv_calibration(files, tuple(cfg.train_profiles), r0_df, cfg)
        rtvar_lowv_cal["params_df"].to_csv(out_dir / f"{cfg.output_prefix}_rtvar_lowv_params.csv", index=False)
        rtvar_lowv_cal["selection_df"].to_csv(out_dir / f"{cfg.output_prefix}_rtvar_lowv_selection.csv", index=False)
    frames = _cached_feature_frames(
        cfg,
        files,
        r0_df,
        lfp_style_calibration=lfp_style_cal,
        nmc_tailored_calibration=nmc_tailored_cal,
        rtvar_lowv_calibration=rtvar_lowv_cal,
    )
    frames = _split_train_profile_blocks_for_validation(
        frames,
        cfg,
        out_dir / f"{cfg.output_prefix}_internal_valid_split_audit.csv",
    )
    feature_cols = _selected_feature_columns(str(cfg.feature_set))
    stage2_feature_set = str(cfg.stage2_feature_set or cfg.feature_set)
    stage2_feature_cols = _selected_feature_columns(stage2_feature_set)
    available = set().union(*(set(frame.columns) for split_frames in frames.values() for frame in split_frames))
    missing = [col for col in feature_cols if col not in available]
    if missing:
        raise RuntimeError(f"Selected feature_set={cfg.feature_set!r} has missing columns: {missing}")
    stage2_missing = [col for col in stage2_feature_cols if col not in available]
    if stage2_missing:
        raise RuntimeError(f"Selected stage2_feature_set={stage2_feature_set!r} has missing columns: {stage2_missing}")
    write_input_schema(feature_cols, out_dir / f"{cfg.output_prefix}_input_schema.csv")
    write_leakage_audit(feature_cols, raw_source_columns, out_dir / f"{cfg.output_prefix}_leakage_audit.csv")
    if stage2_feature_cols != feature_cols:
        write_input_schema(stage2_feature_cols, out_dir / f"{cfg.output_prefix}_stage2_input_schema.csv")
        write_leakage_audit(stage2_feature_cols, raw_source_columns, out_dir / f"{cfg.output_prefix}_stage2_leakage_audit.csv")
    all_metrics = []
    all_history = []
    all_selector = []
    shared_assets = _prepare_seed_shared_assets(cfg, frames, out_dir)
    for seed in cfg.seeds:
        metrics, history, selector = _train_select_and_correct(cfg, frames, out_dir, int(seed), shared_assets)
        all_metrics.append(metrics)
        all_history.append(history)
        all_selector.append(selector)
    metrics = pd.concat(all_metrics, ignore_index=True)
    history = pd.concat(all_history, ignore_index=True)
    selector = pd.concat(all_selector, ignore_index=True)
    metrics.to_csv(out_dir / f"{cfg.output_prefix}_by_temperature.csv", index=False)
    history.to_csv(out_dir / f"{cfg.output_prefix}_history.csv", index=False)
    selector.to_csv(out_dir / f"{cfg.output_prefix}_selector_trace.csv", index=False)
    test = metrics[metrics["split"].eq("test")].copy()
    temp_piv = test.pivot_table(index=["seed", "variant", "epoch"], columns="temperature_C", values="MAE_pct", aggfunc="mean").reset_index()
    for c in [0.0, 25.0, 45.0]:
        if c not in temp_piv.columns:
            temp_piv[c] = np.nan
    temp_piv["max_target"] = temp_piv[[0.0, 25.0, 45.0]].max(axis=1)
    temp_piv["target_met"] = (temp_piv[0.0] < 1.0) & (temp_piv[25.0] < 0.7) & (temp_piv[45.0] < 0.3)
    temp_piv.sort_values(["seed", "target_met", "max_target"], ascending=[True, False, True]).to_csv(
        out_dir / f"{cfg.output_prefix}_test_summary.csv",
        index=False,
    )
    if str(cfg.stage2_select_rule) != "none":
        selected = _stage2_rule_selection(cfg, metrics)
        selected.to_csv(out_dir / f"{cfg.output_prefix}_stage2_rule_selected_by_temperature.csv", index=False)
        selected_test = selected[selected["split"].eq("test")].copy()
        selected_piv = selected_test.pivot_table(
            index=["seed", "variant", "selected_epoch", "stage2_selected_epoch", "stage2_select_rule"],
            columns="temperature_C",
            values="MAE_pct",
            aggfunc="mean",
        ).reset_index()
        for c in [0.0, 25.0, 45.0]:
            if c not in selected_piv.columns:
                selected_piv[c] = np.nan
        selected_piv["max_target"] = selected_piv[[0.0, 25.0, 45.0]].max(axis=1)
        selected_piv["target_met"] = (selected_piv[0.0] < 1.0) & (selected_piv[25.0] < 0.7) & (selected_piv[45.0] < 0.3)
        selected_piv.to_csv(out_dir / f"{cfg.output_prefix}_stage2_rule_selected_test_summary.csv", index=False)
    (out_dir / f"{cfg.output_prefix}_metadata.json").write_text(
        json.dumps(
            {
                **asdict(cfg),
                "feature_columns": feature_cols,
                "input_feature_dim": len(feature_cols),
                "stage2_feature_set_effective": stage2_feature_set,
                "stage2_feature_columns": stage2_feature_cols,
                "stage2_input_feature_dim": len(stage2_feature_cols),
                "voltage_shape_ssl_target_candidates": SSL_VOLTAGE_TARGET_CANDIDATES,
                "voltage_shape_ssl_uses_soc_labels": False,
                "voltage_shape_ssl_uses_future_current_input": False,
                "selector": f"min {cfg.stage1_selector} within selector_max_epoch",
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("Train-DST25 selector test summary:")
    print(temp_piv.sort_values(["seed", "target_met", "max_target"], ascending=[True, False, True]).head(60).to_string(index=False), flush=True)
    if str(cfg.stage2_select_rule) != "none":
        print("Stage2 rule-selected test summary:")
        print(selected_piv.to_string(index=False), flush=True)
    del shared_assets
    _clear_per_config_cuda_cache()
    return metrics, history, selector


def _stage2_rule_selection(cfg: TrainDSTSelectorConfig, metrics: pd.DataFrame) -> pd.DataFrame:
    stage2 = metrics[metrics["variant"].astype(str).str.contains("_stage2", na=False)].copy()
    if not len(stage2):
        return pd.DataFrame()
    if str(cfg.stage2_select_rule) == "stage1_epoch_30_35":
        selected_epoch = pd.to_numeric(stage2["selected_epoch"], errors="coerce")
        stage2["stage2_selected_epoch"] = np.where(
            selected_epoch.le(int(cfg.stage2_stage1_threshold)),
            int(cfg.stage2_early_epoch),
            int(cfg.stage2_late_epoch),
        )
    elif str(cfg.stage2_select_rule).startswith("fixed"):
        fixed_raw = str(cfg.stage2_select_rule).replace("fixed", "")
        stage2["stage2_selected_epoch"] = int(fixed_raw)
    elif str(cfg.stage2_select_rule) in {"val_mean_mae", "val_worst_mae", "val_mean_plus_worst"}:
        key_cols = ["seed", "variant", "selected_epoch"]
        stage2 = stage2.drop(
            columns=[
                "stage2_selected_epoch",
                "stage2_selector_score",
                "val_mean_mae",
                "val_worst_mae",
            ],
            errors="ignore",
        )
        valid = stage2[stage2["split"].eq("valid")].copy()
        if valid.empty:
            raise RuntimeError("stage2_select_rule requires validation rows, but none were found.")
        per_epoch = (
            valid.groupby(key_cols + ["epoch"], as_index=False)["MAE_pct"]
            .agg(val_mean_mae="mean", val_worst_mae="max")
        )
        if str(cfg.stage2_select_rule) == "val_mean_plus_worst":
            per_epoch["stage2_selector_score"] = per_epoch["val_mean_mae"] + 0.5 * per_epoch["val_worst_mae"]
        else:
            per_epoch["stage2_selector_score"] = per_epoch[str(cfg.stage2_select_rule)]
        chosen = (
            per_epoch.sort_values(key_cols + ["stage2_selector_score", "epoch"])
            .groupby(key_cols, as_index=False)
            .first()
            .rename(columns={"epoch": "stage2_selected_epoch"})
        )
        stage2 = stage2.merge(
            chosen[key_cols + ["stage2_selected_epoch", "stage2_selector_score", "val_mean_mae", "val_worst_mae"]],
            on=key_cols,
            how="left",
            validate="many_to_one",
        )
    else:
        raise ValueError(f"Unknown stage2_select_rule={cfg.stage2_select_rule!r}")
    selected = stage2[stage2["epoch"].eq(stage2["stage2_selected_epoch"])].copy()
    selected["stage2_select_rule"] = str(cfg.stage2_select_rule)
    selected["stage2_stage1_threshold"] = int(cfg.stage2_stage1_threshold)
    selected["stage2_early_epoch"] = int(cfg.stage2_early_epoch)
    selected["stage2_late_epoch"] = int(cfg.stage2_late_epoch)
    return selected


def _parse_seeds(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


def _parse_profiles(raw: str) -> tuple[str, ...]:
    profiles = tuple(str(x).strip().upper() for x in str(raw).split(",") if str(x).strip())
    if not profiles:
        raise ValueError("At least one profile is required.")
    return profiles


def _parse_temperatures(raw: str) -> tuple[float, ...]:
    text = str(raw or "").strip()
    if not text or text.lower() == "all":
        return ()
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def _frame_temperature(frame: pd.DataFrame) -> float:
    if "temperature" not in frame.columns or frame.empty:
        return float("nan")
    return float(pd.to_numeric(frame["temperature"], errors="coerce").iloc[0])


def _filter_frame_list_by_temperatures(
    frames: list[pd.DataFrame],
    temperatures: tuple[float, ...],
    *,
    split_name: str,
) -> list[pd.DataFrame]:
    if not temperatures:
        return list(frames)
    wanted = np.asarray([float(t) for t in temperatures], dtype=np.float64)
    out = [
        frame
        for frame in frames
        if np.isfinite(_frame_temperature(frame)) and np.isclose(_frame_temperature(frame), wanted, atol=0.75).any()
    ]
    if not out:
        raise RuntimeError(f"No frames left in split={split_name!r} after temperature filter {tuple(temperatures)}")
    return out


def _filter_scaled_frames_by_temperatures(
    scaled: dict[str, list[pd.DataFrame]],
    *,
    train_temperatures: tuple[float, ...],
    valid_temperatures: tuple[float, ...],
    test_temperatures: tuple[float, ...],
    name: str,
) -> dict[str, list[pd.DataFrame]]:
    filtered = {
        "train": _filter_frame_list_by_temperatures(
            scaled["train"],
            train_temperatures,
            split_name=f"{name}:train",
        ),
        "valid": _filter_frame_list_by_temperatures(
            scaled["valid"],
            valid_temperatures,
            split_name=f"{name}:valid",
        ),
        "test": _filter_frame_list_by_temperatures(
            scaled["test"],
            test_temperatures,
            split_name=f"{name}:test",
        ),
    }
    return filtered


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(mask.astype(bool).tolist()):
        if flag and start is None:
            start = idx
        elif (not flag) and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, int(len(mask))))
    return runs


def _slice_segment(frame: pd.DataFrame, start: int, end: int, split_tag: str, segment_idx: int) -> pd.DataFrame:
    out = frame.iloc[int(start) : int(end)].copy().reset_index(drop=True)
    base_tid = str(frame["trajectory_id"].iloc[0]) if "trajectory_id" in frame.columns and len(frame) else "trajectory"
    out["source_trajectory_id"] = base_tid
    out["trajectory_id"] = f"{base_tid}__{split_tag}{segment_idx:03d}"
    out["split_segment_start"] = int(start)
    out["split_segment_end"] = int(end)
    out["split_segment_role"] = str(split_tag)
    return out


def _split_train_profile_blocks_for_validation(
    frames: dict[str, list[pd.DataFrame]],
    cfg: TrainDSTSelectorConfig,
    out_path: Path,
) -> dict[str, list[pd.DataFrame]]:
    mode = str(cfg.valid_split_mode)
    if mode == "profile":
        return frames
    if mode != "train_profile_blocks":
        raise ValueError(f"Unknown valid_split_mode={mode!r}")
    if int(cfg.valid_block_mod) < 2:
        raise ValueError("--valid-block-mod must be at least 2 for train_profile_blocks.")
    block_rows = int(cfg.valid_block_rows)
    if block_rows <= int(cfg.window_len):
        raise ValueError("--valid-block-rows must be larger than --window-len.")
    valid_mod = int(cfg.valid_block_mod)
    valid_index = int(cfg.valid_block_index) % valid_mod
    min_rows = int(cfg.window_len) + 1
    train_segments: list[pd.DataFrame] = []
    valid_segments: list[pd.DataFrame] = []
    audit_rows = []

    for frame_idx, frame in enumerate(frames["train"]):
        n = int(len(frame))
        if n < min_rows * valid_mod:
            raise RuntimeError(
                f"Training frame {frame_idx} is too short for block validation: n={n}, "
                f"window_len={cfg.window_len}, valid_block_mod={valid_mod}"
            )
        block_id = np.arange(n, dtype=np.int64) // block_rows
        valid_mask = (block_id % valid_mod) == valid_index
        source_tid = str(frame["trajectory_id"].iloc[0])
        temp = float(frame["temperature"].iloc[0]) if "temperature" in frame.columns else float("nan")
        drive = str(frame["drive_cycle"].iloc[0]) if "drive_cycle" in frame.columns else "unknown"
        for role, mask, target in [
            ("trainblk", ~valid_mask, train_segments),
            ("validblk", valid_mask, valid_segments),
        ]:
            for start, end in _contiguous_true_runs(mask):
                if int(end) - int(start) < min_rows:
                    continue
                segment = _slice_segment(frame, int(start), int(end), role, len(target))
                target.append(segment)
                audit_rows.append(
                    {
                        "source_trajectory_id": source_tid,
                        "segment_trajectory_id": str(segment["trajectory_id"].iloc[0]),
                        "split": "train" if role == "trainblk" else "valid",
                        "temperature_C": temp,
                        "drive_cycle": drive,
                        "start_row": int(start),
                        "end_row_exclusive": int(end),
                        "n_rows": int(end) - int(start),
                        "valid_block_rows": block_rows,
                        "valid_block_mod": valid_mod,
                        "valid_block_index": valid_index,
                    }
                )

    if not train_segments or not valid_segments:
        raise RuntimeError("Internal train-profile block validation produced an empty train or valid split.")
    train_keys = {
        (str(seg["source_trajectory_id"].iloc[0]), int(seg["end_index"].iloc[i]))
        for seg in train_segments
        for i in range(len(seg))
    }
    valid_keys = {
        (str(seg["source_trajectory_id"].iloc[0]), int(seg["end_index"].iloc[i]))
        for seg in valid_segments
        for i in range(len(seg))
    }
    overlap = train_keys & valid_keys
    if overlap:
        raise RuntimeError(f"Internal validation split leakage: {len(overlap)} overlapping source rows.")
    pd.DataFrame(audit_rows).to_csv(out_path, index=False)
    return {"train": train_segments, "valid": valid_segments, "test": list(frames["test"])}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run train-only DST25 checkpoint selector with cold/hot correction.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--raw-root", default=TrainDSTSelectorConfig.raw_root)
    p.add_argument("--output-prefix", default=BASE_PREFIX)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--train-profiles", default=",".join(TrainDSTSelectorConfig.train_profiles))
    p.add_argument("--valid-profiles", default=",".join(TrainDSTSelectorConfig.valid_profiles))
    p.add_argument("--test-profiles", default=",".join(TrainDSTSelectorConfig.test_profiles))
    p.add_argument(
        "--train-temperatures",
        default="all",
        help="Comma-separated Stage 1 train temperatures. Use 'all' for the default all-temperature train split.",
    )
    p.add_argument(
        "--valid-temperatures",
        default="all",
        help="Comma-separated Stage 1 validation temperatures. Use 'all' for default.",
    )
    p.add_argument(
        "--test-temperatures",
        default="all",
        help="Comma-separated Stage 1 test temperatures. Use 'all' for default.",
    )
    p.add_argument(
        "--finetune-temperatures",
        default="all",
        help="Comma-separated Stage 1 fine-tune train temperatures. Empty/all disables fine-tune unless --finetune-epochs is positive.",
    )
    p.add_argument("--finetune-epochs", type=int, default=0)
    p.add_argument("--finetune-lr-scale", type=float, default=0.35)
    p.add_argument(
        "--stage2-train-temperatures",
        default="all",
        help="Comma-separated Stage 2 correction train temperatures. Empty/all means follow Stage 1 train temperatures.",
    )
    p.add_argument(
        "--stage2-valid-temperatures",
        default="all",
        help="Comma-separated Stage 2 validation temperatures. Empty/all means follow Stage 1 validation temperatures.",
    )
    p.add_argument(
        "--stage2-test-temperatures",
        default="all",
        help="Comma-separated Stage 2 test temperatures. Empty/all means follow Stage 1 test temperatures.",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--window-len", type=int, default=TrainDSTSelectorConfig.window_len)
    p.add_argument("--stride", type=int, default=TrainDSTSelectorConfig.stride)
    p.add_argument("--selector-min-epoch", type=int, default=1)
    p.add_argument("--selector-max-epoch", type=int, default=0)
    p.add_argument("--stage2-epochs", type=int, default=60)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--stage1-eval-every", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--recurrent", choices=["tcn", "lstm", "gru", "rnn", "transformer"], default="tcn")
    p.add_argument(
        "--head-kind",
        choices=[
            "linear",
            "mlp",
            "residual_mlp_0p5",
            "residual_mlp_1p0",
            "residual_mlp_2p0",
            "gated_mlp",
            "gated_mlp_linstart",
            "gated_mlp_midstart",
        ],
        default="linear",
    )
    p.add_argument("--temp-mode", choices=["none", "bias", "hard_affine", "moe", "hard_heads"], default="none")
    p.add_argument("--dropout", type=float, default=0.06)
    p.add_argument(
        "--model-kind",
        choices=[
            "single",
            "fusion_h64_h128",
            "fusion_h64_h128_fixed",
            "anchor_residual_tcn",
            "anchor_residual_sequence",
            "anchor_residual_dynamic_only_sequence",
            "anchor_residual_dynamic_gated_sequence",
            "normal_dynamic_correction_sequence",
            "anchor_residual_temp_budget_sequence",
            "anchor_residual_cold_low_soc_positive_sequence",
            "anchor_residual_cold_low_soc_signed_sequence",
            "anchor_residual_vt_additive_mlp_sequence",
            "anchor_residual_vt_input_additive_mlp_sequence",
            "normal_cold_low_soc_positive_sequence",
            "normal_cold_low_soc_signed_sequence",
            "normal_cold_high_dynamic_signed_sequence",
            "normal_low_voltage_tail_sequence",
            "normal_affine_calibration_sequence",
            "regime_gated_normal_anchor_signed_sequence",
            "anchor_residual_tbr_sequence",
            "anchor_residual_tbr_regime_sequence",
            "anchor_residual_tbr_transition_sequence",
            "anchor_residual_tbr_tzgate_sequence",
            "anchor_residual_tbr_interp25_sequence",
            "anchor_residual_tbr_interp25_tzgate_sequence",
            "anchor_residual_tbr_interp25_tzgate_transition_sequence",
            "anchor_residual_tbr_interp25_tzgate_dynscale_sequence",
            "anchor_residual_tbr_interp25_tzgate_dynscale_lstm_mha_sequence",
            "anchor_residual_tbr_interp25_socbasis_sequence",
            "anchor_residual_tbr_socbasis_sequence",
            "base_mid_residual_sequence",
            "endpoint_mlp",
            "window_summary_mlp",
            "anchor_residual_endpoint_mlp",
            "anchor_residual_window_summary_mlp",
            "single_mha_endpoint",
            "single_multiscale_endpoint",
            "single_dual_context_endpoint",
            "single_dual_context_dynamic_gate_endpoint",
            "single_dual_context_fusion_endpoint",
            "single_dual_context_eqdyn_g4global_endpoint",
            "single_dual_context_dynamic_gate_eqdyn_g4full_endpoint",
            "single_dual_context_fusion_eqdyn_g4full_endpoint",
        ],
        default="single",
    )
    p.add_argument("--fusion-h64-weight", type=float, default=0.5)
    p.add_argument("--multiscale-local-len", type=int, default=TrainDSTSelectorConfig.multiscale_local_len)
    p.add_argument("--dual-context-gate-center", type=float, default=TrainDSTSelectorConfig.dual_context_gate_center)
    p.add_argument("--dual-context-gate-sharpness", type=float, default=TrainDSTSelectorConfig.dual_context_gate_sharpness)
    p.add_argument("--dual-context-gate-local-min", type=float, default=TrainDSTSelectorConfig.dual_context_gate_local_min)
    p.add_argument("--dual-context-gate-local-max", type=float, default=TrainDSTSelectorConfig.dual_context_gate_local_max)
    p.add_argument("--dual-context-fusion-delta-limit", type=float, default=TrainDSTSelectorConfig.dual_context_fusion_delta_limit)
    p.add_argument("--anchor-residual-limit", type=float, default=0.12)
    p.add_argument("--anchor-residual-limit-init", choices=["value", "rand01"], default="value")
    p.add_argument("--anchor-residual-limit-mode", choices=["fixed", "learnable", "bounded_learnable"], default="fixed")
    p.add_argument("--anchor-residual-limit-lower", type=float, default=0.0)
    p.add_argument("--anchor-residual-limit-upper", type=float, default=0.2)
    p.add_argument("--lambda-anchor-loss", type=float, default=0.2)
    p.add_argument("--lambda-regime-cvar", type=float, default=0.0)
    p.add_argument("--regime-cvar-top-frac", type=float, default=0.34)
    p.add_argument("--lambda-regime-var", type=float, default=0.0)
    p.add_argument("--lambda-residual-aux-loss", type=float, default=0.0)
    p.add_argument("--lambda-residual-zeromean-loss", type=float, default=0.0)
    p.add_argument("--lambda-dual-context-branch-aux-loss", type=float, default=0.0)
    p.add_argument("--dual-context-branch-pretrain-epochs", type=int, default=0)
    p.add_argument("--dual-context-freeze-branches-after-pretrain", action="store_true")
    p.add_argument("--base-pretrain-epochs", type=int, default=0)
    p.add_argument("--freeze-base-after-pretrain", action="store_true")
    p.add_argument("--lambda-mid-delta-loss", type=float, default=0.0)
    p.add_argument("--lambda-zbin-cvar", type=float, default=0.0)
    p.add_argument("--zbin-cvar-top-frac", type=float, default=0.34)
    p.add_argument("--lambda-cold-socbin-cvar", type=float, default=TrainDSTSelectorConfig.lambda_cold_socbin_cvar)
    p.add_argument("--cold-socbin-cvar-top-frac", type=float, default=TrainDSTSelectorConfig.cold_socbin_cvar_top_frac)
    p.add_argument("--cold-socbin-cvar-bin-width", type=float, default=TrainDSTSelectorConfig.cold_socbin_cvar_bin_width)
    p.add_argument("--cold-socbin-cvar-temp", type=float, default=TrainDSTSelectorConfig.cold_socbin_cvar_temp)
    p.add_argument("--lambda-cold-socbin-bias", type=float, default=TrainDSTSelectorConfig.lambda_cold_socbin_bias)
    p.add_argument("--cold-socbin-bias-bin-width", type=float, default=TrainDSTSelectorConfig.cold_socbin_bias_bin_width)
    p.add_argument("--cold-socbin-bias-soc-upper", type=float, default=TrainDSTSelectorConfig.cold_socbin_bias_soc_upper)
    p.add_argument("--cold-socbin-bias-temp", type=float, default=TrainDSTSelectorConfig.cold_socbin_bias_temp)
    p.add_argument("--lambda-cold-profile-socbin-underbias", type=float, default=TrainDSTSelectorConfig.lambda_cold_profile_socbin_underbias)
    p.add_argument("--cold-profile-socbin-underbias-bin-width", type=float, default=TrainDSTSelectorConfig.cold_profile_socbin_underbias_bin_width)
    p.add_argument("--cold-profile-socbin-underbias-soc-upper", type=float, default=TrainDSTSelectorConfig.cold_profile_socbin_underbias_soc_upper)
    p.add_argument("--cold-profile-socbin-underbias-temp", type=float, default=TrainDSTSelectorConfig.cold_profile_socbin_underbias_temp)
    p.add_argument("--weight-ema-decay", type=float, default=TrainDSTSelectorConfig.weight_ema_decay)
    p.add_argument("--weight-ema-start-epoch", type=int, default=TrainDSTSelectorConfig.weight_ema_start_epoch)
    p.add_argument("--lambda-cold-lowsoc-overpred-loss", type=float, default=0.0)
    p.add_argument("--cold-lowsoc-overpred-soc-upper", type=float, default=0.2)
    p.add_argument("--cold-lowsoc-overpred-temp", type=float, default=0.0)
    p.add_argument("--lambda-cold-lowsoc-underpred-loss", type=float, default=0.0)
    p.add_argument("--cold-lowsoc-underpred-soc-upper", type=float, default=0.2)
    p.add_argument("--cold-lowsoc-underpred-temp", type=float, default=0.0)
    p.add_argument("--voltage-sag-augment-prob", type=float, default=TrainDSTSelectorConfig.voltage_sag_augment_prob)
    p.add_argument("--voltage-sag-augment-scale", type=float, default=TrainDSTSelectorConfig.voltage_sag_augment_scale)
    p.add_argument("--voltage-sag-augment-soc-upper", type=float, default=TrainDSTSelectorConfig.voltage_sag_augment_soc_upper)
    p.add_argument("--voltage-sag-augment-temp", type=float, default=TrainDSTSelectorConfig.voltage_sag_augment_temp)
    p.add_argument("--lambda-cold-voltage-shift-consistency", type=float, default=TrainDSTSelectorConfig.lambda_cold_voltage_shift_consistency)
    p.add_argument("--cold-voltage-shift-consistency-scale", type=float, default=TrainDSTSelectorConfig.cold_voltage_shift_consistency_scale)
    p.add_argument("--cold-voltage-shift-consistency-soc-upper", type=float, default=TrainDSTSelectorConfig.cold_voltage_shift_consistency_soc_upper)
    p.add_argument("--cold-voltage-shift-consistency-temp", type=float, default=TrainDSTSelectorConfig.cold_voltage_shift_consistency_temp)
    p.add_argument("--hard-region-loss-weight", type=float, default=0.0)
    p.add_argument("--hard-region-soc-threshold", type=float, default=0.4)
    p.add_argument("--hard-region-soc-lower", type=float, default=-1.0)
    p.add_argument("--hard-region-soc-upper", type=float, default=-1.0)
    p.add_argument("--hard-region-dynamic-weight", type=float, default=0.5)
    p.add_argument("--hard-region-cold-only", action="store_true")
    p.add_argument("--cold-corrector-limit", type=float, default=TrainDSTSelectorConfig.cold_corrector_limit)
    p.add_argument("--cold-corrector-soc-threshold", type=float, default=TrainDSTSelectorConfig.cold_corrector_soc_threshold)
    p.add_argument("--cold-corrector-temp-threshold", type=float, default=TrainDSTSelectorConfig.cold_corrector_temp_threshold)
    p.add_argument("--cold-corrector-gate-sharpness", type=float, default=TrainDSTSelectorConfig.cold_corrector_gate_sharpness)
    p.add_argument("--cold-corrector-sag-gate", action="store_true")
    p.add_argument("--high-dynamic-corrector-center", type=float, default=TrainDSTSelectorConfig.high_dynamic_corrector_center)
    p.add_argument("--high-dynamic-corrector-sharpness", type=float, default=TrainDSTSelectorConfig.high_dynamic_corrector_sharpness)
    p.add_argument("--high-dynamic-corrector-istd-center", type=float, default=TrainDSTSelectorConfig.high_dynamic_corrector_istd_center)
    p.add_argument("--high-dynamic-corrector-istd-sharpness", type=float, default=TrainDSTSelectorConfig.high_dynamic_corrector_istd_sharpness)
    p.add_argument("--ssl-pretrain-epochs", type=int, default=0)
    p.add_argument("--ssl-lr", type=float, default=8e-4)
    p.add_argument("--ssl-recon-weight", type=float, default=1.0)
    p.add_argument("--ssl-next-vcorr-weight", type=float, default=0.5)
    p.add_argument("--ssl-slope-weight", type=float, default=0.25)
    p.add_argument("--corr-mode", default="cold_hot")
    p.add_argument("--corr-limit", type=float, default=1.2)
    p.add_argument("--no-corr-zero-init", action="store_false", dest="corr_zero_init")
    p.set_defaults(corr_zero_init=True)
    p.add_argument("--lr-stage2", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=TrainDSTSelectorConfig.weight_decay)
    p.add_argument("--focus45-weight", type=float, default=12.0)
    p.add_argument("--keep-lambda", type=float, default=4.0)
    p.add_argument("--lambda-rex", type=float, default=2.0)
    p.add_argument("--lambda-condinv", type=float, default=0.02)
    p.add_argument("--weight-0", type=float, default=4.0)
    p.add_argument("--weight-25", type=float, default=2.2)
    p.add_argument("--weight-45", type=float, default=1.0)
    p.add_argument(
        "--stage1-selector",
        choices=[
            "train25_dst",
            "train25_us06",
            "train25_mean_drive",
            "train25_worst_drive",
            "train25_worst_gap",
            "valid_worst_temp",
            "val_mean_mae",
            "val_worst_mae",
            "val_mean_plus_worst",
            "val_target_worst",
            "val_target_mean_plus_worst",
            "val_regime_target_worst",
            "val_regime_target_mean_plus_worst",
            "valid25_temp",
            "last_epoch",
            "train25_dst_valid_worst",
            "train25_worst_valid_worst",
            "train25_dst_valid45_guard",
            "train25_worst_valid_balance",
        ],
        default="train25_dst",
    )
    p.add_argument(
        "--fixed-stage1-epoch",
        type=int,
        default=0,
        help="Override metric-based Stage 1 selection with a predeclared epoch. Useful for inner-LOO-derived fixed horizons.",
    )
    p.add_argument(
        "--selector-regime-min-windows",
        type=int,
        default=30,
        help="Minimum validation windows required for a regime slice to contribute to val_regime_* selectors.",
    )
    p.add_argument("--stage2-select-rule", default="none")
    p.add_argument("--stage2-stage1-threshold", type=int, default=7)
    p.add_argument("--stage2-early-epoch", type=int, default=30)
    p.add_argument("--stage2-late-epoch", type=int, default=35)
    p.add_argument("--test-blind", action="store_true")
    p.add_argument(
        "--train-sampler",
        choices=[
            "temperature_balanced",
            "standard",
            "temperature_profile_balanced",
            "temperature_profile_soc_balanced",
            "temperature_regime_balanced",
            "temperature_profile_regime_balanced",
            "temperature_profile_soc_regime_balanced",
        ],
        default="temperature_balanced",
    )
    p.add_argument(
        "--feature-set",
        choices=[
            "vcorr_it",
            "vcorr_it_excitation_ema",
            "paper_g0_raw",
            "paper_g1_derivatives",
            "paper_g4_all_ema",
            "paper_g4_all_ema_tailaware",
            "paper_g4_all_ema_asymohm",
            "paper_g4_all_ema_lvblend",
            "paper_g4_all_ema_ocvcal",
            "paper_g4_all_ema_start",
            "paper_g4_all_ema_di_abs_mean",
            "paper_g4_all_ema_transient_min",
            "paper_g4_all_ema_shape_norm",
            "paper_g4_all_ema_regime",
            "paper_g4_eqdyn",
            "paper_g4_eqdyn_start",
            "paper_g4_eqdyn_regime",
            "paper_g4_relax_upper",
            "paper_t6_voltage_ema_all",
            "paper_t7_current_abs_ema_all",
            "paper_voltage_ema50_only",
            "paper_voltage_ema200_only",
            "paper_voltage_ema800_only",
            "paper_current_abs_ema50_only",
            "paper_current_abs_ema200_only",
            "paper_g6_full23",
            "paper_g7_no_current_ema",
            "paper_g8_no_voltage_ema",
            "vit_basic",
            "vit_decomp",
            "vit_ema_compact",
            "vit_response_ema",
            "vit_full_ema",
        ],
        default="vcorr_it",
    )
    p.add_argument(
        "--stage2-feature-set",
        choices=[
            "",
            "vcorr_it",
            "vcorr_it_excitation_ema",
            "paper_g0_raw",
            "paper_g1_derivatives",
            "paper_g4_all_ema",
            "paper_g4_all_ema_tailaware",
            "paper_g4_all_ema_asymohm",
            "paper_g4_all_ema_lvblend",
            "paper_g4_all_ema_ocvcal",
            "paper_g4_all_ema_start",
            "paper_g4_all_ema_di_abs_mean",
            "paper_g4_all_ema_transient_min",
            "paper_g4_all_ema_shape_norm",
            "paper_g4_all_ema_regime",
            "paper_g4_eqdyn",
            "paper_g4_eqdyn_start",
            "paper_g4_eqdyn_regime",
            "paper_t6_voltage_ema_all",
            "paper_t7_current_abs_ema_all",
            "paper_voltage_ema50_only",
            "paper_voltage_ema200_only",
            "paper_voltage_ema800_only",
            "paper_current_abs_ema50_only",
            "paper_current_abs_ema200_only",
            "paper_g6_full23",
            "paper_g7_no_current_ema",
            "paper_g8_no_voltage_ema",
            "vit_basic",
            "vit_decomp",
            "vit_ema_compact",
            "vit_response_ema",
            "vit_full_ema",
        ],
        default="",
        help="Optional correction-only feature set. Empty means reuse --feature-set.",
    )
    p.add_argument(
        "--stage1-ensemble-epochs",
        default="",
        help="Optional comma list or range:start:end of stage1 checkpoint epochs to average before stage2.",
    )
    p.add_argument(
        "--sampler-seed-mode",
        choices=["seed", "none"],
        default="seed",
        help="Use an independent per-seed sampler generator, or reproduce the old global-RNG sampler path.",
    )
    p.add_argument(
        "--test-blind-rng-burn",
        type=int,
        default=0,
        help="When test-blind, advance global RNG after validation to mimic old non-blind test-loader RNG side effects without reading test data.",
    )
    p.add_argument(
        "--diagnostic-test-every",
        type=int,
        default=0,
        help="When test-blind, still export diagnostic test metrics every N Stage 1 epochs. Use only for screening, not paper selection.",
    )
    p.add_argument(
        "--skip-stage2",
        action="store_true",
        help="Train/evaluate Stage 1 only and skip the bounded correction stage.",
    )
    p.add_argument("--lambda-current-consistency", type=float, default=0.0)
    p.add_argument("--current-noise-std", type=float, default=0.0)
    p.add_argument("--current-dropout-prob", type=float, default=0.0)
    p.add_argument("--lambda-profile-adv", type=float, default=0.0)
    p.add_argument("--profile-adv-grl", type=float, default=1.0)
    p.add_argument("--lambda-profile-supcon", type=float, default=0.0)
    p.add_argument("--supcon-temperature", type=float, default=0.15)
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--save-train-predictions", action="store_true")
    p.add_argument("--save-checkpoints", action="store_true")
    p.add_argument("--save-final-weights", action="store_true")
    p.add_argument("--final-only-eval", action="store_true")
    p.add_argument(
        "--skip-train-final-eval",
        action="store_true",
        help="Do not compute final train-split metrics. Valid/test metrics and predictions are still produced.",
    )
    p.add_argument(
        "--ema-perturbation-importance",
        action="store_true",
        help="After selecting the fixed Stage 1 model, export inference-only EMA perturbation metrics.",
    )
    p.add_argument("--valid-split-mode", default="profile", choices=["profile", "train_profile_blocks"])
    p.add_argument("--valid-block-rows", type=int, default=800)
    p.add_argument("--valid-block-mod", type=int, default=5)
    p.add_argument("--valid-block-index", type=int, default=4)
    p.add_argument("--sequence-training", action="store_true")
    p.add_argument(
        "--cache-dataset-cuda",
        action="store_true",
        help="Materialize per-job window datasets as CUDA tensors. Forces DataLoader workers to 0 for cached datasets.",
    )
    p.add_argument("--tqdm-epochs", action="store_true", help="Show one tqdm progress bar over training epochs only.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainDSTSelectorConfig(
        base_dir=Path(args.base_dir),
        raw_root=Path(args.raw_root),
        output_prefix=str(args.output_prefix),
        seeds=_parse_seeds(args.seeds),
        train_profiles=_parse_profiles(args.train_profiles),
        valid_profiles=_parse_profiles(args.valid_profiles),
        test_profiles=_parse_profiles(args.test_profiles),
        train_temperatures=_parse_temperatures(args.train_temperatures),
        valid_temperatures=_parse_temperatures(args.valid_temperatures),
        test_temperatures=_parse_temperatures(args.test_temperatures),
        finetune_temperatures=_parse_temperatures(args.finetune_temperatures),
        finetune_epochs=int(args.finetune_epochs),
        finetune_lr_scale=float(args.finetune_lr_scale),
        stage2_train_temperatures=_parse_temperatures(args.stage2_train_temperatures),
        stage2_valid_temperatures=_parse_temperatures(args.stage2_valid_temperatures),
        stage2_test_temperatures=_parse_temperatures(args.stage2_test_temperatures),
        window_len=int(args.window_len),
        stride=int(args.stride),
        epochs=int(args.epochs),
        selector_min_epoch=int(args.selector_min_epoch),
        selector_max_epoch=int(args.selector_max_epoch),
        stage2_epochs=int(args.stage2_epochs),
        eval_every=int(args.eval_every),
        stage1_eval_every=int(args.stage1_eval_every),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        hidden_size=int(args.hidden_size),
        layers=int(args.layers),
        kernel_size=int(args.kernel_size),
        recurrent=str(args.recurrent),
        head_kind=str(args.head_kind),
        temp_mode=str(args.temp_mode),
        dropout=float(args.dropout),
        model_kind=str(args.model_kind),
        fusion_h64_weight=float(args.fusion_h64_weight),
        multiscale_local_len=int(args.multiscale_local_len),
        dual_context_gate_center=float(args.dual_context_gate_center),
        dual_context_gate_sharpness=float(args.dual_context_gate_sharpness),
        dual_context_gate_local_min=float(args.dual_context_gate_local_min),
        dual_context_gate_local_max=float(args.dual_context_gate_local_max),
        dual_context_fusion_delta_limit=float(args.dual_context_fusion_delta_limit),
        corr_mode=str(args.corr_mode),
        corr_limit=float(args.corr_limit),
        corr_zero_init=bool(args.corr_zero_init),
        lr_stage2=float(args.lr_stage2),
        weight_decay=float(args.weight_decay),
        focus45_weight=float(args.focus45_weight),
        keep_lambda=float(args.keep_lambda),
        lambda_rex=float(args.lambda_rex),
        lambda_condinv=float(args.lambda_condinv),
        weight_0=float(args.weight_0),
        weight_25=float(args.weight_25),
        weight_45=float(args.weight_45),
        stage1_selector=str(args.stage1_selector),
        selector_regime_min_windows=int(args.selector_regime_min_windows),
        fixed_stage1_epoch=int(args.fixed_stage1_epoch),
        stage2_select_rule=str(args.stage2_select_rule),
        stage2_stage1_threshold=int(args.stage2_stage1_threshold),
        stage2_early_epoch=int(args.stage2_early_epoch),
        stage2_late_epoch=int(args.stage2_late_epoch),
        test_blind=bool(args.test_blind),
        train_sampler=str(args.train_sampler),
        stage1_ensemble_epochs=str(args.stage1_ensemble_epochs),
        feature_set=str(args.feature_set),
        stage2_feature_set=str(args.stage2_feature_set),
        sampler_seed_mode=str(args.sampler_seed_mode),
        test_blind_rng_burn=int(args.test_blind_rng_burn),
        diagnostic_test_every=int(args.diagnostic_test_every),
        skip_stage2=bool(args.skip_stage2),
        lambda_current_consistency=float(args.lambda_current_consistency),
        current_noise_std=float(args.current_noise_std),
        current_dropout_prob=float(args.current_dropout_prob),
        lambda_profile_adv=float(args.lambda_profile_adv),
        profile_adv_grl=float(args.profile_adv_grl),
        lambda_profile_supcon=float(args.lambda_profile_supcon),
        supcon_temperature=float(args.supcon_temperature),
        anchor_residual_limit=float(args.anchor_residual_limit),
        anchor_residual_limit_init=str(args.anchor_residual_limit_init),
        anchor_residual_limit_mode=str(args.anchor_residual_limit_mode),
        anchor_residual_limit_lower=float(args.anchor_residual_limit_lower),
        anchor_residual_limit_upper=float(args.anchor_residual_limit_upper),
        lambda_anchor_loss=float(args.lambda_anchor_loss),
        lambda_regime_cvar=float(args.lambda_regime_cvar),
        regime_cvar_top_frac=float(args.regime_cvar_top_frac),
        lambda_regime_var=float(args.lambda_regime_var),
        lambda_residual_aux_loss=float(args.lambda_residual_aux_loss),
        lambda_residual_zeromean_loss=float(args.lambda_residual_zeromean_loss),
        lambda_dual_context_branch_aux_loss=float(args.lambda_dual_context_branch_aux_loss),
        dual_context_branch_pretrain_epochs=int(args.dual_context_branch_pretrain_epochs),
        dual_context_freeze_branches_after_pretrain=bool(args.dual_context_freeze_branches_after_pretrain),
        base_pretrain_epochs=int(args.base_pretrain_epochs),
        freeze_base_after_pretrain=bool(args.freeze_base_after_pretrain),
        lambda_mid_delta_loss=float(args.lambda_mid_delta_loss),
        lambda_zbin_cvar=float(args.lambda_zbin_cvar),
        zbin_cvar_top_frac=float(args.zbin_cvar_top_frac),
        lambda_cold_socbin_cvar=float(args.lambda_cold_socbin_cvar),
        cold_socbin_cvar_top_frac=float(args.cold_socbin_cvar_top_frac),
        cold_socbin_cvar_bin_width=float(args.cold_socbin_cvar_bin_width),
        cold_socbin_cvar_temp=float(args.cold_socbin_cvar_temp),
        lambda_cold_socbin_bias=float(args.lambda_cold_socbin_bias),
        cold_socbin_bias_bin_width=float(args.cold_socbin_bias_bin_width),
        cold_socbin_bias_soc_upper=float(args.cold_socbin_bias_soc_upper),
        cold_socbin_bias_temp=float(args.cold_socbin_bias_temp),
        lambda_cold_profile_socbin_underbias=float(args.lambda_cold_profile_socbin_underbias),
        cold_profile_socbin_underbias_bin_width=float(args.cold_profile_socbin_underbias_bin_width),
        cold_profile_socbin_underbias_soc_upper=float(args.cold_profile_socbin_underbias_soc_upper),
        cold_profile_socbin_underbias_temp=float(args.cold_profile_socbin_underbias_temp),
        weight_ema_decay=float(args.weight_ema_decay),
        weight_ema_start_epoch=int(args.weight_ema_start_epoch),
        lambda_cold_lowsoc_overpred_loss=float(args.lambda_cold_lowsoc_overpred_loss),
        cold_lowsoc_overpred_soc_upper=float(args.cold_lowsoc_overpred_soc_upper),
        cold_lowsoc_overpred_temp=float(args.cold_lowsoc_overpred_temp),
        lambda_cold_lowsoc_underpred_loss=float(args.lambda_cold_lowsoc_underpred_loss),
        cold_lowsoc_underpred_soc_upper=float(args.cold_lowsoc_underpred_soc_upper),
        cold_lowsoc_underpred_temp=float(args.cold_lowsoc_underpred_temp),
        voltage_sag_augment_prob=float(args.voltage_sag_augment_prob),
        voltage_sag_augment_scale=float(args.voltage_sag_augment_scale),
        voltage_sag_augment_soc_upper=float(args.voltage_sag_augment_soc_upper),
        voltage_sag_augment_temp=float(args.voltage_sag_augment_temp),
        lambda_cold_voltage_shift_consistency=float(args.lambda_cold_voltage_shift_consistency),
        cold_voltage_shift_consistency_scale=float(args.cold_voltage_shift_consistency_scale),
        cold_voltage_shift_consistency_soc_upper=float(args.cold_voltage_shift_consistency_soc_upper),
        cold_voltage_shift_consistency_temp=float(args.cold_voltage_shift_consistency_temp),
        hard_region_loss_weight=float(args.hard_region_loss_weight),
        hard_region_soc_threshold=float(args.hard_region_soc_threshold),
        hard_region_soc_lower=float(args.hard_region_soc_lower),
        hard_region_soc_upper=float(args.hard_region_soc_upper),
        hard_region_dynamic_weight=float(args.hard_region_dynamic_weight),
        hard_region_cold_only=bool(args.hard_region_cold_only),
        cold_corrector_limit=float(args.cold_corrector_limit),
        cold_corrector_soc_threshold=float(args.cold_corrector_soc_threshold),
        cold_corrector_temp_threshold=float(args.cold_corrector_temp_threshold),
        cold_corrector_gate_sharpness=float(args.cold_corrector_gate_sharpness),
        cold_corrector_sag_gate=bool(args.cold_corrector_sag_gate),
        high_dynamic_corrector_center=float(args.high_dynamic_corrector_center),
        high_dynamic_corrector_sharpness=float(args.high_dynamic_corrector_sharpness),
        high_dynamic_corrector_istd_center=float(args.high_dynamic_corrector_istd_center),
        high_dynamic_corrector_istd_sharpness=float(args.high_dynamic_corrector_istd_sharpness),
        ssl_pretrain_epochs=int(args.ssl_pretrain_epochs),
        ssl_lr=float(args.ssl_lr),
        ssl_recon_weight=float(args.ssl_recon_weight),
        ssl_next_vcorr_weight=float(args.ssl_next_vcorr_weight),
        ssl_slope_weight=float(args.ssl_slope_weight),
        valid_split_mode=str(args.valid_split_mode),
        valid_block_rows=int(args.valid_block_rows),
        valid_block_mod=int(args.valid_block_mod),
        valid_block_index=int(args.valid_block_index),
        save_predictions=bool(args.save_predictions),
        save_train_predictions=bool(args.save_train_predictions),
        save_checkpoints=bool(args.save_checkpoints),
        save_final_weights=bool(args.save_final_weights),
        final_only_eval=bool(args.final_only_eval),
        skip_train_final_eval=bool(args.skip_train_final_eval),
        ema_perturbation_importance=bool(args.ema_perturbation_importance),
        sequence_training=bool(args.sequence_training),
        cache_dataset_cuda=bool(args.cache_dataset_cuda),
        tqdm_epochs=bool(args.tqdm_epochs),
    )
    run(cfg)


if __name__ == "__main__":
    main()
