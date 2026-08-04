from dataclasses import dataclass
from pathlib import Path

# Configuration
def infer_mamba_stateful_dir() -> Path:
    candidates = [
        Path("."),
        Path("torch") / "LSTM_STATELESS_DECOMP_SOC",
        Path("torch") / "MAMBA_STATEFUL",
        Path("LSTM_STATELESS_DECOMP_SOC"),
        Path.cwd(),
        Path.cwd() / "LSTM_STATELESS_DECOMP_SOC",
        Path.cwd() / "torch" / "LSTM_STATELESS_DECOMP_SOC",
        Path.cwd() / "torch" / "MAMBA_STATEFUL",
    ]
    for p in candidates:
        if (p / "LFP_ABS_SOC").exists():
            return p
    # The isolated NMC LOPO runner imports legacy modules that reference this
    # config, but it does not use the LFP_ABS_SOC data path. Keep import-time
    # behavior non-fatal inside this isolated experiment folder.
    return Path.cwd()


BASE_DIR = infer_mamba_stateful_dir()


@dataclass
class CFG:
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "LFP_ABS_SOC"
    output_dir: Path = BASE_DIR
    decomposed_dir: Path = BASE_DIR / "decomposed_features"

    original_notebook: str = "../MAMBA_STATEFUL/V_POL_2_STAGE_EXTRAP_4TEMP.ipynb"

    # CSV columns.
    v_col: str = "Voltage(V)"
    i_col: str = "Current(A)"
    t_col: str = "Temperature (C)_1"
    time_col: str = "t_rel(s)"
    physical_soc_col: str = "SOC_CC"
    usable_soc_col: str = "SOC_use"

    # Label construction. Physical SOC is a coulomb-counted fraction against Q_ref.
    # Usable-to-cutoff is always rebuilt as remaining discharge fraction to this trajectory cutoff.
    use_existing_soc_cc_if_available: bool = False
    use_existing_usable_if_available: bool = False
    q_ref_override_Ah: float | None = None
    soc_start_default: float = 1.0
    dt_sec_default: float = 1.0
    clip_training_soc_to_0_1: bool = True

    # Data split. Split is trajectory/file-level, never window-level.
    all_temps: tuple = ("N10", "0", "10", "20", "25", "30", "40", "50")
    train_temps: tuple = ("N10", "10", "25", "50")
    eval_temps: tuple = ("N10", "0", "10", "20", "25", "30", "40", "50")
    train_drives: tuple = ("DST", "US06")
    eval_drive: str = "FUDS"
    split_mode: str = "train_dst_us06_eval_all_fuds"

    # Smoke mode keeps top-to-bottom execution short. Disable for full experiments.
    smoke_mode: bool = True
    smoke_train_temps: tuple = ("25",)
    smoke_eval_temps: tuple = ("30",)
    smoke_max_rows_per_trajectory: int = 2000

    # Corrector input scaling and recurrence.
    dt_sec: float = 1.0
    corr_use_v_feat: bool = True
    n_pol_fast: int = 4
    n_pol_mid: int = 4
    n_pol_slow: int = 4
    n_hys: int = 6
    pol_limit_V: float = 0.25
    pol_fast_limit_V: float = 0.16
    pol_mid_limit_V: float = 0.12
    pol_slow_limit_V: float = 0.10
    hys_limit_V: float = 0.20
    ohm_R0_min: float = 0.0
    ohm_R0_max: float = 0.18
    r0_sigmoid_temp: float = 2.0
    g_corr_scale: float = 1.35
    enforce_vcorr_floor: bool = True
    v_floor_raw: float = 2.02
    use_soft_vfloor: bool = True
    vfloor_softplus_beta: float = 12.0
    use_corr_temp_expert: bool = True
    corr_temp_gain_min: float = 0.75
    corr_temp_gain_max: float = 1.35
    I_rest_thr_A: float = 0.05
    corrector_variant: str = "base"  # base | temp_tau
    temp_tau_log_scale: float = 1.25
    temp_tau_min_sec: float = 0.25
    temp_tau_max_sec: float = 4096.0
    lambda_temp_tau_reg: float = 0.002
    shift_tau_log_aT_limit: float = 1.6094379124341003  # log(5), bounded temperature shift factor

    # Corrector pretraining. No SOC label is passed to this loss.
    run_corrector_pretraining: bool = True
    corrector_epochs: int = 1
    corrector_lr: float = 1e-3
    corrector_weight_decay: float = 1e-2
    corrector_batch_by_length: bool = True
    corrector_profile_batch_size: int = 8
    corrector_train_segment_len: int | None = None
    corrector_segments_per_profile_per_epoch: int = 1
    grad_clip: float = 1.0
    lambda_recon: float = 1.0
    lambda_smooth_vcorr: float = 0.02
    lambda_component_bound: float = 0.20
    lambda_r0_smooth: float = 0.02
    lambda_hys_smooth: float = 0.02
    lambda_rest_components: float = 0.02
    lambda_pol_hf: float = 0.0
    lambda_hys_hf: float = 0.0
    lambda_R0_smooth: float = 0.0
    lambda_timescale_sep: float = 0.0
    lambda_hys_tv: float = 0.0
    lambda_hys_slope: float = 0.0
    hys_limit_scale: float = 1.0
    lambda_pol_slow_hf: float = 0.0
    lambda_R0_hf: float = 0.0
    lambda_frequency_route: float = 0.0
    frequency_route_margin: float = 0.2

    # Feature extraction.
    run_causal_feature_check: bool = True
    causal_check_len: int = 512
    save_decomposed_features: bool = True
    reuse_cached_decomposed_features: bool = False
    require_cached_decomposed_features: bool = True
    use_future_smoothing: bool = False
    feature_normalization_scope: str = "train_only"

    # Stateless LSTM.
    window_len: int = 128
    stride: int = 64
    lstm_hidden_size: int = 64
    lstm_layers: int = 1
    lstm_dropout: float = 0.0
    lstm_lr: float = 1e-3
    lstm_weight_decay: float = 1e-4
    lstm_epochs: int = 1
    lstm_print_every: int = 1
    batch_size: int = 1024
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int = 4
    cuda_non_blocking: bool = True
    target_labels_to_run: tuple = ("physical", "usable")
    output_mode: str = "single"  # single | multi

    # Shortcut-defense ablations. Smoke mode still runs the full required baseline set with 1 epoch.
    ablation_names_to_run: tuple = (
        "A1_V_raw_only",
        "A2_V_corr_only",
        "A3_V_raw_I_T",
        "A4_V_corr_I_T",
        "A5_I_T_only",
        "A6_I_T_dI_absI_only",
        "F1_full_decomposed",
        "F2_raw_plus_full_decomposed",
        "R1_raw_I_T_pol_like",
        "R2_raw_I_T_hys_like",
        "R3_raw_I_T_ohmic_like",
        "R4_raw_I_T_pol_hys_like",
        "R5_raw_I_T_all_components",
    )
    include_dI_absI_in_ablation_windows: bool = False
    gate_summary_enabled: bool = True
    save_prediction_rows: bool = True

    # Review-defense diagnostics.
    same_voltage_bin_width_V: float = 0.02
    same_voltage_min_count: int = 5
    same_voltage_min_soc_std: float = 0.03
    shortcut_delta_warn_mae: float = 0.005
    cutoff_low_voltage_threshold: float = 2.2



def make_cfg(**overrides):
    cfg = CFG()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"Unknown CFG field: {key}")
        setattr(cfg, key, value)
    cfg.decomposed_dir.mkdir(parents=True, exist_ok=True)
    return cfg
