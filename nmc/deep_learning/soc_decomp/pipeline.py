import pandas as pd

from .config import CFG, make_cfg
from .runtime import configure_torch_runtime, device
from .data import load_and_prepare_data
from .models import build_corrector
from .corrector import run_corrector_pretraining
from .features import extract_all_feature_frames, load_cached_feature_frames
from .training import run_ablation_experiments
from .diagnostics import run_shortcut_diagnostics
from .plots import run_required_plots
from .checks import cutoff_label_warning, final_leakage_check, print_artifact_paths

try:
    from IPython.display import display
except Exception:
    display = print


def display_core_tables(outputs):
    print("trajectory/file-level split verification:")
    display(outputs["data"]["trajectory_split_verification"])

    print("label sources:")
    display(outputs["data"]["label_sources"])

    train_head = outputs["feature_frames"]["train"][0].head(1) if outputs["feature_frames"]["train"] else pd.DataFrame()
    valid_head = outputs["feature_frames"]["valid"][0].head(1) if outputs["feature_frames"]["valid"] else pd.DataFrame()
    test_head = outputs["feature_frames"]["test"][0].head(1) if outputs["feature_frames"]["test"] else pd.DataFrame()
    feature_preview = pd.concat([train_head, valid_head, test_head], ignore_index=True)
    preview_cols = [
        "file_name", "temperature", "drive_cycle", "V_raw", "V_corr_raw",
        "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0", "SOC_physical", "SOC_usable_cutoff"
    ]
    print("feature preview:")
    display(feature_preview[preview_cols])

    ab = outputs["ablation"]
    print("Voltage-only shortcut baseline table:")
    display(ab["voltage_only_baseline_table"])
    print("Current-only baseline table:")
    display(ab["current_only_baseline_table"])
    print("Full decomposed ablation table:")
    display(ab["full_decomposed_ablation_table"])
    print("Plateau 20-80% MAE table:")
    display(ab["plateau_20_80_mae_table"])
    print("Cutoff last 10% MAE table:")
    display(ab["cutoff_last10_mae_table"])
    if "component_gate_df" in ab and len(ab["component_gate_df"]):
        print("Component reliability gate by temperature:")
        display(ab["component_gate_df"])
    print("Core shortcut comparisons:")
    display(ab["core_compare"][[
        "target_label", "ablation", "MAE_pct", "RMSE_pct", "Max_error_pct",
        "plateau_20_80_MAE_pct", "cutoff_last_10pct_MAE_pct"
    ]])
    print("SOC-bin metrics, including 20-80% LFP plateau:")
    display(ab["metrics_by_soc_bin_df"])

    diag = outputs["diagnostics"]
    print("Same-voltage SOC spread table:")
    display(diag["same_voltage_soc_spread"].head(30))
    print("Same-voltage error comparison table:")
    display(diag["same_voltage_error_comparison_df"])
    print("Cutoff-exclusion test table:")
    display(diag["cutoff_exclusion_test_table"])

    print("Fixed cutoff physical vs usable-to-cutoff summary:")
    display(outputs["cutoff_summary"])


def run_pipeline(cfg: CFG | None = None, *, make_plots=True, show_tables=True):
    cfg = make_cfg() if cfg is None else cfg
    cfg.decomposed_dir.mkdir(parents=True, exist_ok=True)
    configure_torch_runtime()
    print("device:", device)
    print("random initialization and DataLoader shuffle are not fixed")
    print("Mamba SOC estimator is intentionally not imported or instantiated in this notebook.")
    print("BASE_DIR:", cfg.base_dir.resolve())
    print("DATA_DIR:", cfg.data_dir.resolve())
    print("DECOMPOSED_DIR:", cfg.decomposed_dir.resolve())

    data = load_and_prepare_data(cfg)

    corrector = None
    corrector_history = pd.DataFrame()
    if getattr(cfg, "reuse_cached_decomposed_features", False):
        try:
            feature_frames = load_cached_feature_frames(cfg, data=data, decomposed_dir=cfg.decomposed_dir)
            print("Skipping corrector training/extraction; loaded cached decomposed features from:", cfg.decomposed_dir.resolve())
        except Exception:
            if getattr(cfg, "require_cached_decomposed_features", True):
                raise
            print("Cached decomposed features were unavailable; falling back to corrector training/extraction.")
            corrector = build_corrector(cfg, device)
            print(f"VoltageCorrector initialized: variant={getattr(cfg, 'corrector_variant', 'base')}. R0 head has no Softplus tail.")
            corrector_history = run_corrector_pretraining(corrector, data["train_profiles"], cfg, data["v_scaler"])
            feature_frames = extract_all_feature_frames(
                corrector,
                data["train_profiles"],
                data["valid_profiles"],
                data["test_profiles"],
                cfg,
                data["v_scaler"],
            )
            print("decomposed feature files written to:", cfg.decomposed_dir.resolve())
    else:
        corrector = build_corrector(cfg, device)
        print(f"VoltageCorrector initialized: variant={getattr(cfg, 'corrector_variant', 'base')}. R0 head has no Softplus tail.")
        corrector_history = run_corrector_pretraining(corrector, data["train_profiles"], cfg, data["v_scaler"])

        feature_frames = extract_all_feature_frames(
            corrector,
            data["train_profiles"],
            data["valid_profiles"],
            data["test_profiles"],
            cfg,
            data["v_scaler"],
        )
        print("decomposed feature files written to:", cfg.decomposed_dir.resolve())

    ablation = run_ablation_experiments(feature_frames, cfg)
    diagnostics = run_shortcut_diagnostics(
        feature_frames,
        ablation["all_predictions"],
        ablation["ablation_results"],
        cfg,
        make_plots=make_plots,
    )
    cutoff_summary = run_required_plots(
        feature_frames,
        ablation["all_predictions"],
        ablation["ablation_results"],
        cfg,
        make_plots=make_plots,
    )

    cutoff_label_warning(cutoff_summary)
    final_leakage_check(cfg, feature_frames, data["v_scaler"], data["i_scaler"], data["t_scaler"])
    print_artifact_paths(cfg)

    outputs = {
        "cfg": cfg,
        "data": data,
        "corrector": corrector,
        "corrector_history": corrector_history,
        "feature_frames": feature_frames,
        "ablation": ablation,
        "diagnostics": diagnostics,
        "cutoff_summary": cutoff_summary,
    }
    if show_tables:
        display_core_tables(outputs)
    return outputs
