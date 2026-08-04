import warnings
import numpy as np

from .corrector import assert_corrector_loss_has_no_soc


def final_leakage_check(cfg, feature_frames, v_scaler, i_scaler, t_scaler):
    train_ids = {f["trajectory_id"].iloc[0] for f in feature_frames["train"]}
    valid_ids = {f["trajectory_id"].iloc[0] for f in feature_frames["valid"]}
    test_ids = {f["trajectory_id"].iloc[0] for f in feature_frames["test"]}
    if not train_ids.isdisjoint(valid_ids) or not train_ids.isdisjoint(test_ids) or not valid_ids.isdisjoint(test_ids):
        warnings.warn("Leakage risk: train/eval share trajectory_id values")
    assert train_ids.isdisjoint(valid_ids)
    assert train_ids.isdisjoint(test_ids)
    assert valid_ids.isdisjoint(test_ids)
    if v_scaler.fit_ids != train_ids or i_scaler.fit_ids != train_ids or t_scaler.fit_ids != train_ids:
        warnings.warn("Scaler leakage risk: a scaler was not fit only on train trajectory IDs")
    assert v_scaler.fit_ids == train_ids
    assert i_scaler.fit_ids == train_ids
    assert t_scaler.fit_ids == train_ids
    assert cfg.feature_normalization_scope == "train_only"
    assert cfg.use_future_smoothing is False
    if getattr(cfg, "split_mode", "") not in {"train_dst_us06_eval_all_fuds"}:
        warnings.warn("Window-level or non-trajectory split mode may be active. Recheck train/test leakage.")
    assert_corrector_loss_has_no_soc()
    print("final leakage checks passed")


def cutoff_label_warning(cutoff_summary):
    if cutoff_summary is not None and len(cutoff_summary):
        physical = cutoff_summary["cutoff_physical_SOC"].to_numpy(np.float32)
        usable = cutoff_summary["cutoff_usable_SOC"].to_numpy(np.float32)
        if np.nanmax(np.abs(usable)) > 0.02:
            warnings.warn("Usable-to-cutoff SOC is not near zero at cutoff for at least one trajectory.")
        if np.all(np.abs(physical) < 0.02):
            warnings.warn("Physical SOC appears forced to zero at cutoff for all trajectories. Recheck label generation.")
        if np.allclose(physical, usable, atol=1e-5, rtol=1e-5):
            warnings.warn("Physical SOC and usable-to-cutoff SOC are identical at cutoff. Recheck label separation.")


def print_artifact_paths(cfg):
    print("Generated artifacts:")
    print("  notebook:", (cfg.output_dir / "LSTM_stateless_decomposed_voltage_SOC.ipynb").resolve())
    print("  original:", (cfg.output_dir / cfg.original_notebook).resolve())
    print("  decomposed_features:", cfg.decomposed_dir.resolve())
    for name in [
        "ablation_results_fixed_labels.csv",
        "ablation_results.csv",
        "physical_soc_ablation_results.csv",
        "usable_cutoff_ablation_results.csv",
        "plateau_20_80_mae_table_fixed.csv",
        "cutoff_last10_mae_table_fixed.csv",
        "metrics_by_soc_bin.csv",
        "metrics_by_soc_bin_fixed_labels.csv",
        "metrics_by_temperature.csv",
        "metrics_by_temperature_fixed_labels.csv",
        "metrics_by_cycle.csv",
        "metrics_by_cycle_fixed_labels.csv",
        "trajectory_final_soc_error.csv",
        "trajectory_final_soc_error_fixed_labels.csv",
        "cutoff_physical_soc_summary_fixed.csv",
        "cutoff_physical_soc_summary.csv",
        "trajectory_split_verification.csv",
        "voltage_only_baseline_table.csv",
        "current_only_baseline_table.csv",
        "full_decomposed_ablation_table.csv",
        "plateau_20_80_mae_table.csv",
        "same_voltage_soc_spread_fixed.csv",
        "same_voltage_soc_spread.csv",
        "same_voltage_error_comparison_fixed.csv",
        "same_voltage_error_comparison.csv",
        "cutoff_exclusion_test_fixed.csv",
        "cutoff_exclusion_test_table.csv",
        "all_predictions_fixed_labels.csv",
        "component_gate_by_temperature.csv",
        "component_temperature_distance.csv",
        "component_temperature_distance_heatmap.png",
        "event_aligned_component_response_by_temperature.csv",
        "event_aligned_pol_response_plot.png",
        "event_aligned_hys_response_plot.png",
        "temp10_error_diagnostic.csv",
        "temp10_error_summary.csv",
        "temp10_error_over_time.png",
        "temp10_error_vs_components.png",
    ]:
        print(f"  {name}:", (cfg.output_dir / name).resolve())
