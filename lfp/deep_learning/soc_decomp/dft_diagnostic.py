from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COMPONENT_COLUMNS = [
    "V_raw",
    "V_corr_raw",
    "V_pol_raw",
    "V_pol_fast_raw",
    "V_pol_mid_raw",
    "V_pol_slow_raw",
    "V_hys_raw",
    "V_ohm_raw",
    "R0",
]

BANDS_HZ = [
    ("ultra_low_0_0p001Hz", 0.0, 0.001),
    ("low_0p001_0p01Hz", 0.001, 0.01),
    ("mid_0p01_0p05Hz", 0.01, 0.05),
    ("high_0p05_Nyquist", 0.05, np.inf),
]

TEMP_ORDER = [-10.0, 0.0, 10.0, 20.0, 25.0, 30.0, 40.0, 50.0]
DRIVE_ORDER = ["DST", "US06", "FUDS"]


def _temp_sort_key(temp: float) -> int:
    arr = np.asarray(TEMP_ORDER)
    idx = np.where(np.isclose(arr, float(temp)))[0]
    return int(idx[0]) if len(idx) else 999


def _trajectory_label(temp: float, drive: str) -> str:
    if np.isclose(temp, -10.0):
        return f"-10C_{drive}"
    if abs(temp - round(temp)) < 1e-6:
        return f"{int(round(temp))}C_{drive}"
    return f"{temp:g}C_{drive}"


def _detrend_linear(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    if y.size < 3:
        return y - np.nanmean(y)
    t = np.linspace(-1.0, 1.0, y.size)
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return y - np.nanmean(y)
    coef = np.polyfit(t[mask], y[mask], deg=1)
    return y - np.polyval(coef, t)


def _rfft_metrics(values: np.ndarray, dt_sec: float = 1.0) -> tuple[pd.DataFrame, dict, list[dict]]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 8:
        empty = pd.DataFrame(columns=["frequency_Hz", "period_s", "power", "power_fraction", "amplitude"])
        return empty, {}, []

    mean_value = float(np.mean(x))
    std_value = float(np.std(x))
    rms_value = float(np.sqrt(np.mean(x**2)))
    x_ac = x - mean_value
    x_det = _detrend_linear(x)
    ac_rms = float(np.sqrt(np.mean(x_ac**2)))
    detrended_rms = float(np.sqrt(np.mean(x_det**2)))

    # Hann window makes pulse-related spectral peaks easier to compare across trajectories.
    win = np.hanning(n)
    xw = x_det * win
    freq = np.fft.rfftfreq(n, d=dt_sec)
    fft = np.fft.rfft(xw)
    amp = 2.0 * np.abs(fft) / max(np.sum(win), 1e-12)
    power = np.abs(fft) ** 2
    if power.size:
        power[0] = 0.0
    total_power = float(np.sum(power))
    if total_power <= 0:
        frac = np.zeros_like(power)
    else:
        frac = power / total_power
    period = np.full_like(freq, np.inf, dtype=np.float64)
    nz = freq > 0
    period[nz] = 1.0 / freq[nz]

    spec = pd.DataFrame({
        "frequency_Hz": freq,
        "period_s": period,
        "power": power,
        "power_fraction": frac,
        "amplitude": amp,
    })
    spec = spec[spec["frequency_Hz"] > 0].reset_index(drop=True)

    if len(spec):
        f = spec["frequency_Hz"].to_numpy(np.float64)
        pf = spec["power_fraction"].to_numpy(np.float64)
        centroid = float(np.sum(f * pf) / max(np.sum(pf), 1e-12))
        cumsum = np.cumsum(pf)
        f95 = float(f[np.searchsorted(cumsum, 0.95, side="left").clip(0, len(f) - 1)])
        dom_idx = int(np.argmax(spec["power"].to_numpy(np.float64)))
        dominant_frequency = float(spec["frequency_Hz"].iloc[dom_idx])
        dominant_period = float(spec["period_s"].iloc[dom_idx])
        dominant_power_fraction = float(spec["power_fraction"].iloc[dom_idx])
    else:
        centroid = np.nan
        f95 = np.nan
        dominant_frequency = np.nan
        dominant_period = np.nan
        dominant_power_fraction = np.nan

    band_vals = {}
    for name, lo, hi in BANDS_HZ:
        if np.isinf(hi):
            m = spec["frequency_Hz"].to_numpy(np.float64) >= lo
        elif lo == 0.0:
            m = (spec["frequency_Hz"].to_numpy(np.float64) > lo) & (spec["frequency_Hz"].to_numpy(np.float64) <= hi)
        else:
            m = (spec["frequency_Hz"].to_numpy(np.float64) >= lo) & (spec["frequency_Hz"].to_numpy(np.float64) < hi)
        band_vals[f"{name}_power_fraction"] = float(spec.loc[m, "power_fraction"].sum()) if len(spec) else np.nan

    summary = {
        "n_samples": n,
        "dt_sec": float(dt_sec),
        "duration_s": float(n * dt_sec),
        "mean": mean_value,
        "std": std_value,
        "rms": rms_value,
        "ac_rms": ac_rms,
        "detrended_rms": detrended_rms,
        "total_dft_power": total_power,
        "spectral_centroid_Hz": centroid,
        "frequency_95pct_energy_Hz": f95,
        "dominant_frequency_Hz": dominant_frequency,
        "dominant_period_s": dominant_period,
        "dominant_power_fraction": dominant_power_fraction,
        **band_vals,
    }

    peak_rows = []
    if len(spec):
        top = spec.sort_values("power", ascending=False).head(5).reset_index(drop=True)
        for rank, row in top.iterrows():
            peak_rows.append({
                "peak_rank": int(rank + 1),
                "peak_frequency_Hz": float(row["frequency_Hz"]),
                "peak_period_s": float(row["period_s"]),
                "peak_power_fraction": float(row["power_fraction"]),
                "peak_amplitude": float(row["amplitude"]),
            })
    return spec, summary, peak_rows


def _log_bin_spectrum(spec: pd.DataFrame, n_bins: int = 120) -> pd.DataFrame:
    if spec.empty:
        return spec
    f = spec["frequency_Hz"].to_numpy(np.float64)
    lo = max(float(np.nanmin(f[f > 0])), 1e-6)
    hi = float(np.nanmax(f))
    if hi <= lo:
        return spec.copy()
    edges = np.geomspace(lo, hi, n_bins + 1)
    bins = np.digitize(f, edges, right=False) - 1
    rows = []
    for b in range(n_bins):
        m = bins == b
        if not m.any():
            continue
        sub = spec.iloc[np.where(m)[0]]
        freq_mid = float(np.sqrt(edges[b] * edges[b + 1]))
        rows.append({
            "frequency_Hz": freq_mid,
            "period_s": float(1.0 / freq_mid),
            "power_fraction": float(sub["power_fraction"].sum()),
            "amplitude_mean": float(sub["amplitude"].mean()),
            "n_fft_bins": int(len(sub)),
        })
    return pd.DataFrame(rows)


def _load_feature_frames(feature_dir: Path) -> list[pd.DataFrame]:
    frames = []
    for path in sorted(feature_dir.glob("*_features.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["_source_feature_file"] = path.name
        frames.append(frame)
    return frames


def _coverage_table(frames: list[pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    have = set()
    for f in frames:
        have.add((float(f["temperature"].iloc[0]), str(f["drive_cycle"].iloc[0]).upper()))
    rows = []
    for temp in TEMP_ORDER:
        for drive in DRIVE_ORDER:
            rows.append({
                "temperature_C": temp,
                "drive_cycle": drive,
                "has_decomposed_feature_csv": (temp, drive) in have,
            })
    cov = pd.DataFrame(rows)
    cov.to_csv(out_dir / "dft_feature_coverage.csv", index=False)
    return cov


def _plot_heatmap(summary: pd.DataFrame, value_col: str, out_path: Path, title: str) -> None:
    pivot = summary.pivot_table(
        index=["temperature_C", "drive_cycle"],
        columns="component",
        values=value_col,
        aggfunc="mean",
    )
    order = sorted(pivot.index, key=lambda x: (_temp_sort_key(x[0]), DRIVE_ORDER.index(x[1]) if x[1] in DRIVE_ORDER else 99))
    pivot = pivot.loc[order]
    fig, ax = plt.subplots(figsize=(13, max(5, 0.34 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(np.float64), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([_trajectory_label(t, d) for t, d in pivot.index])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_band_energy(summary: pd.DataFrame, out_path: Path) -> None:
    band_cols = [f"{name}_power_fraction" for name, _, _ in BANDS_HZ]
    comps = [c for c in COMPONENT_COLUMNS if c in summary["component"].unique()]
    fig, axes = plt.subplots(len(comps), 1, figsize=(14, max(4, 2.3 * len(comps))), sharex=True)
    if len(comps) == 1:
        axes = [axes]
    grouped = summary.groupby(["component", "drive_cycle"], as_index=False)[band_cols].mean()
    xlabels = DRIVE_ORDER
    for ax, comp in zip(axes, comps):
        sub = grouped[grouped["component"].eq(comp)].set_index("drive_cycle").reindex(xlabels)
        bottom = np.zeros(len(xlabels))
        for col in band_cols:
            vals = sub[col].fillna(0).to_numpy(np.float64)
            ax.bar(xlabels, vals, bottom=bottom, label=col.replace("_power_fraction", ""))
            bottom += vals
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(comp)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.35))
    axes[-1].set_xlabel("drive cycle")
    fig.suptitle("DFT band energy fraction by component and drive cycle", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_component_spectra(spectrum: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for comp in sorted(spectrum["component"].unique()):
        subc = spectrum[spectrum["component"].eq(comp)]
        if subc.empty:
            continue
        fig, axes = plt.subplots(1, len(DRIVE_ORDER), figsize=(18, 4.8), sharey=True)
        for ax, drive in zip(axes, DRIVE_ORDER):
            subd = subc[subc["drive_cycle"].eq(drive)]
            if subd.empty:
                ax.set_title(f"{drive} (missing)")
                ax.axis("off")
                continue
            for temp in sorted(subd["temperature_C"].unique(), key=_temp_sort_key):
                g = subd[np.isclose(subd["temperature_C"].astype(float), temp)]
                if g.empty:
                    continue
                ax.plot(
                    g["frequency_Hz"],
                    g["power_fraction"],
                    linewidth=0.9,
                    label=f"{temp:g}C",
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(drive)
            ax.set_xlabel("frequency [Hz]")
            ax.grid(True, which="both", alpha=0.25)
        axes[0].set_ylabel("log-binned power fraction")
        axes[-1].legend(fontsize=7, bbox_to_anchor=(1.02, 1.0), loc="upper left")
        fig.suptitle(f"DFT spectrum: {comp} (detrended, Hann-windowed)")
        fig.tight_layout()
        fig.savefig(out_dir / f"dft_spectrum_{comp}.png", dpi=180)
        plt.close(fig)


def _write_markdown(summary: pd.DataFrame, coverage: pd.DataFrame, out_dir: Path) -> None:
    lines = ["# DFT Component Diagnostic Summary\n"]
    lines.append("Signals are analyzed per trajectory after mean removal and linear detrending. The DFT uses a Hann window. Band fractions are fractions of non-DC detrended DFT power.\n")
    lines.append("\n## Frequency Bands\n")
    lines.append("- ultra_low: 0-0.001 Hz, periods longer than about 1000 s.\n")
    lines.append("- low: 0.001-0.01 Hz, slow trajectory-scale response.\n")
    lines.append("- mid: 0.01-0.05 Hz, pulse/drive-cycle-scale response.\n")
    lines.append("- high: >=0.05 Hz, fast local jitter/pulse-sensitive response.\n")
    missing = coverage[~coverage["has_decomposed_feature_csv"]]
    if len(missing):
        lines.append("\n## Missing Decomposed Feature Coverage\n")
        lines.append(missing.to_markdown(index=False))
        lines.append("\n")
    comp_cols = [
        "component",
        "high_0p05_Nyquist_power_fraction",
        "mid_0p01_0p05Hz_power_fraction",
        "spectral_centroid_Hz",
        "detrended_rms",
    ]
    by_comp = summary.groupby("component", as_index=False)[comp_cols[1:]].mean()
    by_comp = by_comp.sort_values("high_0p05_Nyquist_power_fraction", ascending=False)
    lines.append("\n## Component-Level Average Spectral Summary\n")
    lines.append(by_comp.to_markdown(index=False, floatfmt=".6g"))
    lines.append("\n\n## GPT-Readable Interpretation Guide\n")
    lines.append("- Larger high-band fraction means the component is more pulse-sensitive or jitter-like.\n")
    lines.append("- Larger ultra-low/low fraction means the component behaves more like a slow state or trajectory-scale trend.\n")
    lines.append("- Compare the same component across temperatures within the same drive cycle before making a temperature claim.\n")
    lines.append("- These are learned voltage decomposition features, not uniquely identified physical polarization/hysteresis quantities.\n")
    (out_dir / "dft_gpt_readable_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_dft_component_diagnostic(
    feature_dir: str | Path = "decomposed_features_train_temp_minus10_0_10_20_25_50",
    output_dir: str | Path = "dft_component_diagnostic",
    *,
    dt_sec: float = 1.0,
) -> dict[str, pd.DataFrame]:
    feature_dir = Path(feature_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = _load_feature_frames(feature_dir)
    if not frames:
        raise FileNotFoundError(f"No *_features.csv files in {feature_dir}")
    coverage = _coverage_table(frames, output_dir)

    summary_rows = []
    band_rows = []
    peak_rows_all = []
    spectrum_rows = []
    for frame in frames:
        meta = {
            "trajectory_id": str(frame["trajectory_id"].iloc[0]),
            "file_name": str(frame["file_name"].iloc[0]) if "file_name" in frame else "",
            "source_feature_file": str(frame["_source_feature_file"].iloc[0]),
            "temperature_C": float(frame["temperature"].iloc[0]),
            "temperature_key": str(frame["temperature_key"].iloc[0]) if "temperature_key" in frame else "",
            "drive_cycle": str(frame["drive_cycle"].iloc[0]).upper(),
        }
        for comp in [c for c in COMPONENT_COLUMNS if c in frame.columns]:
            spec, summary, peaks = _rfft_metrics(frame[comp].to_numpy(np.float64), dt_sec=dt_sec)
            row = {**meta, "component": comp, **summary}
            summary_rows.append(row)
            for name, lo, hi in BANDS_HZ:
                band_rows.append({
                    **meta,
                    "component": comp,
                    "band": name,
                    "freq_low_Hz": float(lo),
                    "freq_high_Hz": float(hi) if np.isfinite(hi) else np.nan,
                    "power_fraction": row.get(f"{name}_power_fraction", np.nan),
                })
            for p in peaks:
                peak_rows_all.append({**meta, "component": comp, **p})
            binned = _log_bin_spectrum(spec, n_bins=120)
            if len(binned):
                binned.insert(0, "component", comp)
                for k, v in reversed(meta.items()):
                    binned.insert(0, k, v)
                spectrum_rows.append(binned)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        ["temperature_C", "drive_cycle", "component"],
        key=lambda s: s.map(_temp_sort_key) if s.name == "temperature_C" else s,
    )
    band_df = pd.DataFrame(band_rows)
    peaks_df = pd.DataFrame(peak_rows_all)
    spectrum_df = pd.concat(spectrum_rows, ignore_index=True) if spectrum_rows else pd.DataFrame()

    summary_df.to_csv(output_dir / "dft_component_summary_by_temp_cycle.csv", index=False)
    band_df.to_csv(output_dir / "dft_component_band_energy.csv", index=False)
    peaks_df.to_csv(output_dir / "dft_component_top_peaks.csv", index=False)
    spectrum_df.to_csv(output_dir / "dft_component_spectrum_long_logbinned.csv", index=False)

    _plot_heatmap(
        summary_df,
        "high_0p05_Nyquist_power_fraction",
        output_dir / "dft_high_frequency_fraction_heatmap.png",
        "High-frequency DFT power fraction (>=0.05 Hz)",
    )
    _plot_heatmap(
        summary_df,
        "spectral_centroid_Hz",
        output_dir / "dft_spectral_centroid_heatmap.png",
        "Spectral centroid [Hz]",
    )
    _plot_band_energy(summary_df, output_dir / "dft_band_energy_stacked_by_component.png")
    if len(spectrum_df):
        _plot_component_spectra(spectrum_df, output_dir / "component_spectrum_plots")
    _write_markdown(summary_df, coverage, output_dir)
    return {
        "summary": summary_df,
        "band_energy": band_df,
        "top_peaks": peaks_df,
        "spectrum": spectrum_df,
        "coverage": coverage,
    }


if __name__ == "__main__":
    run_dft_component_diagnostic()
