#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
LEGACY_ROOT = Path("/home/user/바탕화면/DL/CEMA-TCN").resolve()
DATA_ROOT = (ROOT / "nmc_ocvstart_lopo_clean").resolve()
S2_SOURCE = LEGACY_ROOT / "Scripts/figure_S1.py"
FREQ_SOURCE = LEGACY_ROOT / "Scripts/build_frequency_structure_analysis.py"
PROFILES_12 = ("DST", "US06", "FUDS", "VALIDATION")
PROFILES_9 = ("DST", "US06", "FUDS")
TEMPERATURES = (0.0, 25.0, 45.0)
ACF_LAGS = (1, 50, 200, 800)
CAUSAL_LAGS = (1, 10, 20, 50, 200)


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_s2(module, profiles: tuple[str, ...]):
    mappings, diagnostics = module.discover_terminal_files([DATA_ROOT], PROFILES_12, TEMPERATURES)
    mappings = [mapping for mapping in mappings if mapping.profile in profiles]
    expected = {(temperature, profile) for temperature in TEMPERATURES for profile in profiles}
    actual = {(mapping.temperature_C, mapping.profile) for mapping in mappings}
    if actual != expected:
        raise RuntimeError(f"Record discovery mismatch: missing={expected-actual}, extra={actual-expected}")
    frames = [module.load_terminal_frame(mapping) for mapping in mappings]
    return mappings, frames, pd.concat(frames, ignore_index=True), diagnostics


def autocorr_output(lag_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in lag_table.itertuples(index=False):
        for lag in ACF_LAGS:
            rows.append(
                {
                    "row_type": "record",
                    "profile": record.profile,
                    "temperature_C": record.temperature_C,
                    "file_name": record.file_name,
                    "lag_samples": lag,
                    "rho_V": getattr(record, f"rho_V_lag{lag}"),
                    "rho_I": getattr(record, f"rho_I_lag{lag}"),
                    "n_records": 1,
                }
            )
    records = pd.DataFrame(rows)
    summary = []
    for lag, group in records.groupby("lag_samples"):
        for statistic in ("mean", "min", "max"):
            summary.append(
                {
                    "row_type": statistic,
                    "profile": "__SUMMARY__",
                    "temperature_C": np.nan,
                    "file_name": "__SUMMARY__",
                    "lag_samples": int(lag),
                    "rho_V": float(getattr(group.rho_V, statistic)()),
                    "rho_I": float(getattr(group.rho_I, statistic)()),
                    "n_records": len(group),
                }
            )
    return pd.concat([records, pd.DataFrame(summary)], ignore_index=True)


def lagcorr_output(lag_corr: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "corr_I_t_minus_lag_with_V_t",
        "corr_V_t_minus_lag_with_SOC_t",
        "corr_I_t_minus_lag_with_SOC_t",
    ]
    records = lag_corr[lag_corr.lag_samples.isin(CAUSAL_LAGS)].copy()
    records.insert(0, "row_type", "record")
    summary = []
    for lag, group in records.groupby("lag_samples"):
        for statistic in ("mean", "min", "max"):
            row = {
                "row_type": statistic,
                "temperature_C": np.nan,
                "profile": "__SUMMARY__",
                "file_name": "__SUMMARY__",
                "lag_samples": int(lag),
                "n_pairs": int(group.n_pairs.sum()),
            }
            for column in columns:
                row[column] = float(getattr(group[column], statistic)())
            summary.append(row)
    return pd.concat([records, pd.DataFrame(summary)], ignore_index=True)


def ambiguity_output(spread: pd.DataFrame) -> pd.DataFrame:
    output = spread.copy()
    for column in [name for name in output if "fraction" in name]:
        output[column.replace("_fraction", "_percent")] = 100.0 * output[column]
    return output


def strat_output(history: pd.DataFrame) -> pd.DataFrame:
    output = history.copy()
    output["median_SOC_IQR_percent"] = 100.0 * output.median_SOC_IQR_fraction
    output["p90_SOC_IQR_percent"] = 100.0 * output.p90_SOC_IQR_fraction
    output["ambiguous_sample_fraction_percent"] = 100.0 * output.fraction_samples_in_ambiguous_bins_IQR_ge_0p10
    return output


def spectra(module, profiles: tuple[str, ...]) -> pd.DataFrame:
    mappings, _ = module.discover_terminal_files(DATA_ROOT, LEGACY_ROOT)
    mappings = [mapping for mapping in mappings if mapping.profile in profiles]
    if len(mappings) != len(profiles) * len(TEMPERATURES):
        raise RuntimeError(f"Expected {len(profiles)*3} spectral mappings, found {len(mappings)}")
    welch = module.get_scipy_welch()
    rows = []
    for mapping in mappings:
        frame, info = module.load_terminal_record(mapping)
        for signal, column in (("Voltage", "raw_voltage"), ("Current", "raw_current"), ("Reference SOC", "reference_soc")):
            frequency, psd, psd_info = module.compute_psd(frame[column].to_numpy(float), "constant", welch)
            summary = module.spectral_summary(frequency, psd)
            rows.append(
                {
                    "row_type": "record",
                    "profile": mapping.profile,
                    "temperature_C": mapping.temperature_C,
                    "file_name": mapping.file_name,
                    "signal_name": signal,
                    "n_samples": len(frame),
                    "low_frequency_energy_fraction": summary["low_frequency_energy_fraction"],
                    "mid_frequency_energy_fraction": summary["mid_frequency_energy_fraction"],
                    "high_frequency_energy_fraction": summary["high_frequency_energy_fraction"],
                    "median_frequency_cycles_per_sample": summary["median_frequency_cycles_per_sample"],
                    "welch_nperseg": psd_info["nperseg"],
                    "welch_method": psd_info["method"],
                }
            )
    records = pd.DataFrame(rows)
    summary_rows = []
    metric_columns = [
        "low_frequency_energy_fraction", "mid_frequency_energy_fraction",
        "high_frequency_energy_fraction", "median_frequency_cycles_per_sample",
    ]
    for signal, group in records.groupby("signal_name"):
        for statistic in ("mean", "min", "max"):
            row = {
                "row_type": statistic,
                "profile": "__SUMMARY__",
                "temperature_C": np.nan,
                "file_name": "__SUMMARY__",
                "signal_name": signal,
                "n_samples": int(group.n_samples.sum()),
                "welch_nperseg": 1024,
                "welch_method": "scipy.signal.welch",
            }
            for column in metric_columns:
                row[column] = float(getattr(group[column], statistic)())
            summary_rows.append(row)
    return pd.concat([records, pd.DataFrame(summary_rows)], ignore_index=True)


def plot_ready_trajectory(frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for frame in frames:
        output = frame[["file_name", "profile", "temperature_C", "voltage_V", "current_A", "SOC"]].copy()
        output.insert(3, "sample_index", np.arange(len(output), dtype=np.int64))
        output.insert(4, "time_s", frame["time_s"].to_numpy(float))
        rows.append(output)
    return pd.concat(rows, ignore_index=True)


def equivalence_checks(spread, history, lag, spectra12) -> pd.DataFrame:
    raw = history[history.condition == "A_raw_VI_bins"]
    abs_history = history[history.condition == "B_raw_plus_absI_mean_past200_tertile"]
    spectrum_soc = spectra12[(spectra12.row_type == "record") & (spectra12.signal_name == "Reference SOC")]
    checks = [
        ("median_IQR_min_pct", 100.0 * spread.median_SOC_IQR_fraction.min(), 1.3054521501462857, 1e-9),
        ("median_IQR_max_pct", 100.0 * spread.median_SOC_IQR_fraction.max(), 1.715026257428276, 1e-9),
        ("max_IQR_max_pct", 100.0 * spread.max_SOC_IQR_fraction.max(), 24.880101626320583, 1e-9),
        ("rho_V_lag1_mean", lag.rho_V_lag1.mean(), 0.9752146247733758, 1e-12),
        ("SOC_low_frequency_energy_pct", 100.0 * spectrum_soc.low_frequency_energy_fraction.mean(), 98.85, 0.01),
        ("raw_VI_median_IQR_mean_pct", 100.0 * raw.median_SOC_IQR_fraction.mean(), 1.5179, 0.001),
        ("absI_history_median_IQR_mean_pct", 100.0 * abs_history.median_SOC_IQR_fraction.mean(), 0.6139, 0.001),
    ]
    return pd.DataFrame(
        [
            {
                "check": name,
                "observed": observed,
                "legacy_expected": expected,
                "absolute_tolerance": tolerance,
                "absolute_difference": abs(observed - expected),
                "passed": abs(observed - expected) <= tolerance,
                "record_set": "12_records_with_VALIDATION",
                "analysis_stage": "code_equivalence_check",
            }
            for name, observed, expected, tolerance in checks
        ]
    )


def make_figures(out: Path, trajectory: pd.DataFrame, ambiguity: pd.DataFrame, spectral: pd.DataFrame, strat: pd.DataFrame) -> None:
    colors = {0.0: "#3b78a8", 25.0: "#2f9c68", 45.0: "#d95f43"}
    fig, axes = plt.subplots(3, 3, figsize=(13, 8), sharex="col")
    for column, profile in enumerate(PROFILES_9):
        for temperature, group in trajectory[trajectory.profile == profile].groupby("temperature_C"):
            axes[0, column].plot(group.sample_index, group.voltage_V, color=colors[temperature], lw=0.7, label=f"{int(temperature)} C")
            axes[1, column].plot(group.sample_index, group.current_A, color=colors[temperature], lw=0.55)
            axes[2, column].plot(group.sample_index, 100.0 * group.SOC, color=colors[temperature], lw=0.8)
        axes[0, column].set_title(profile)
        axes[2, column].set_xlabel("Sample")
    axes[0, 0].set_ylabel("Voltage (V)")
    axes[1, 0].set_ylabel("Current (A)")
    axes[2, 0].set_ylabel("SOC (%)")
    axes[0, 0].legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(out / "fig1_trajectory_9rec.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(ambiguity.temperature_C.astype(str), 100.0 * ambiguity.median_SOC_IQR_fraction, color="#4c78a8")
    axes[0].set(xlabel="Temperature (C)", ylabel="Median SOC IQR (%SOC)", title="Raw V-I ambiguity")
    cond = strat[strat.condition.isin(["A_raw_VI_bins", "B_raw_plus_absI_mean_past200_tertile"])]
    for condition, marker in (("A_raw_VI_bins", "o"), ("B_raw_plus_absI_mean_past200_tertile", "s")):
        group = cond[cond.condition == condition]
        axes[1].plot(group.temperature_C, group.median_SOC_IQR_percent, marker=marker, label=condition)
    axes[1].set(xlabel="Temperature (C)", ylabel="Median SOC IQR (%SOC)", title="History stratification")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig2_ambiguity_9rec.png", dpi=220)
    plt.close(fig)

    means = spectral[spectral.row_type == "record"].groupby("signal_name")[[
        "low_frequency_energy_fraction", "mid_frequency_energy_fraction", "high_frequency_energy_fraction"
    ]].mean() * 100.0
    means = means.reindex(["Voltage", "Current", "Reference SOC"])
    fig, axis = plt.subplots(figsize=(7, 4.4))
    bottom = np.zeros(len(means))
    for column, label, color in (
        ("low_frequency_energy_fraction", "Low", "#4c78a8"),
        ("mid_frequency_energy_fraction", "Mid", "#f58518"),
        ("high_frequency_energy_fraction", "High", "#54a24b"),
    ):
        axis.bar(means.index, means[column], bottom=bottom, label=label, color=color)
        bottom += means[column].to_numpy()
    axis.set_ylabel("Energy fraction (%)")
    axis.set_title("Raw-signal spectral energy, 9 records")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig3_spectra_9rec.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "nmc_s2_regen_9records")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing analysis directory: {output}")
    output.mkdir(parents=True)

    s2 = import_file("legacy_section2", S2_SOURCE)
    frequency = import_file("legacy_frequency", FREQ_SOURCE)
    mappings12, frames12, data12, diagnostics12 = load_s2(s2, PROFILES_12)
    lag12, lagcorr12 = s2.table_lag_statistics(frames12)
    spread12, _, details12, edges12 = s2.raw_vi_bin_tables(data12, 40, 40, 50)
    history12, _ = s2.history_conditioned_spread(data12, edges12, 50)
    spectra12 = spectra(frequency, PROFILES_12)
    equivalence = equivalence_checks(spread12, history12, lag12, spectra12)
    equivalence.to_csv(output / "equivalence_check_12rec.csv", index=False)
    equivalence_passed = bool(equivalence.passed.all())

    mappings9, frames9, data9, diagnostics9 = load_s2(s2, PROFILES_9)
    lag9, lagcorr9 = s2.table_lag_statistics(frames9)
    spread9, _, details9, edges9 = s2.raw_vi_bin_tables(data9, 40, 40, 50)
    history9, history_details9 = s2.history_conditioned_spread(data9, edges9, 50)
    support9 = s2.profile_shift_overlap(data9, edges9, PROFILES_9, "FUDS")
    support9 = support9[support9.comparison_kind == "leave_one_profile_out"].reset_index(drop=True)
    spectral9 = spectra(frequency, PROFILES_9)

    autocorr_output(lag9).to_csv(output / "autocorr_9rec.csv", index=False)
    lagcorr_output(lagcorr9).to_csv(output / "lagcorr_9rec.csv", index=False)
    ambiguity_output(spread9).to_csv(output / "vi_ambiguity_9rec.csv", index=False)
    strat9 = strat_output(history9)
    strat9.to_csv(output / "strat_ambiguity_9rec.csv", index=False)
    spectral9.to_csv(output / "spectra_9rec.csv", index=False)
    support9.to_csv(output / "vi_support_9rec.csv", index=False)

    trajectory9 = plot_ready_trajectory(frames9)
    trajectory9.to_csv(output / "fig1_trajectory_plot_ready.csv", index=False)
    details9.to_csv(output / "fig2_ambiguity_plot_ready.csv", index=False)
    spectral9[spectral9.row_type == "record"].to_csv(output / "fig3_spectra_plot_ready.csv", index=False)
    make_figures(output, trajectory9, spread9, spectral9, strat9)

    metadata = {
        "analysis_stage": "s2_regen_9records",
        "data_root": str(DATA_ROOT),
        "profiles_9": list(PROFILES_9),
        "profiles_equivalence_12": list(PROFILES_12),
        "temperatures_C": list(TEMPERATURES),
        "legacy_section2_source": str(S2_SOURCE),
        "legacy_frequency_source": str(FREQ_SOURCE),
        "section2_script_version": s2.SCRIPT_VERSION,
        "frequency_script_version": frequency.SCRIPT_VERSION,
        "voltage_bins": 40,
        "current_bins": 40,
        "minimum_bin_count": 50,
        "ambiguous_IQR_threshold_fraction": 0.10,
        "history_window_samples": 200,
        "equivalence_all_passed": equivalence_passed,
        "equivalence_note": (
            "The current frozen legacy function reproduces six checks but yields 0.430064% for the "
            "history-stratified mean versus the older locked table value 0.6139%; this mismatch is retained."
        ),
        "frequency_bands_cycles_per_sample": {"low": "f<1/200", "mid": "1/200<=f<1/50", "high": "f>=1/50"},
        "record_count_12": len(mappings12),
        "record_count_9": len(mappings9),
        "samples_12": len(data12),
        "samples_9": len(data9),
        "discovery_12": diagnostics12,
        "discovery_9": diagnostics9,
    }
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "analysis_code.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"output": str(output), "equivalence_pass": equivalence_passed, "records_9": len(mappings9)}, indent=2))


if __name__ == "__main__":
    main()
