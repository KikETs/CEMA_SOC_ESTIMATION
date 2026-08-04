import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .training import cutoff_physical_soc_summary

try:
    from IPython.display import display
except Exception:
    display = print

# Required plots and cutoff physical-SOC summary
def label_display_name(label):
    return "usable-to-cutoff fraction" if label == "usable" else f"{label} SOC"


def plot_soc_predictions(pred_df, title):
    if pred_df.empty:
        return
    for (label, tid), g in pred_df.groupby(["target_label", "trajectory_id"]):
        gg = g.sort_values("end_index")
        plt.figure(figsize=(12, 3))
        shown_label = label_display_name(label)
        plt.plot(gg["end_index"], gg["y_true"], label=f"true {shown_label}", linewidth=1.2)
        plt.plot(gg["end_index"], gg["y_pred"], label=f"pred {shown_label}", linewidth=1.0)
        plt.ylim(-0.05, 1.05)
        plt.xlabel("time index")
        plt.ylabel("fraction")
        plt.title(f"{title} | {shown_label} | {tid}")
        plt.legend()
        plt.show()


def plot_decomposition(frame):
    plt.figure(figsize=(12, 6))
    x = frame["end_index"]
    ax1 = plt.subplot(5, 1, 1)
    ax1.plot(x, frame["V_raw"], label="V_raw", linewidth=0.8)
    ax1.plot(x, frame["V_corr_raw"], label="V_corr_raw", linewidth=0.8)
    ax1.legend(loc="best")
    ax1.set_title(f"learned dynamic voltage components | {frame['trajectory_id'].iloc[0]}")
    ax2 = plt.subplot(5, 1, 2, sharex=ax1)
    ax2.plot(x, frame["V_pol_raw"], linewidth=0.8)
    ax2.set_ylabel("polarization-like V")
    ax3 = plt.subplot(5, 1, 3, sharex=ax1)
    ax3.plot(x, frame["V_hys_raw"], linewidth=0.8)
    ax3.set_ylabel("hysteresis-like V")
    ax4 = plt.subplot(5, 1, 4, sharex=ax1)
    ax4.plot(x, frame["V_ohm_raw"], linewidth=0.8)
    ax4.set_ylabel("ohmic-like V")
    ax5 = plt.subplot(5, 1, 5, sharex=ax1)
    ax5.plot(x, frame["R0"], linewidth=0.8)
    ax5.set_ylabel("R0")
    ax5.set_xlabel("time index")
    plt.tight_layout()
    plt.show()


def plot_plateau_errors(pred_df, title):
    if pred_df.empty:
        return
    g = pred_df[(pred_df["y_true"] >= 0.2) & (pred_df["y_true"] <= 0.8)].copy()
    if g.empty:
        print("No 20-80% plateau samples to plot.")
        return
    for (label, tid), gg in g.groupby(["target_label", "trajectory_id"]):
        gg = gg.sort_values("end_index")
        shown_label = label_display_name(label)
        plt.figure(figsize=(12, 3))
        plt.plot(gg["end_index"], gg["error"] * 100.0, linewidth=0.8)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.ylabel("error (% fraction)")
        plt.xlabel("time index")
        plt.title(f"{title} plateau error 20-80% | {shown_label} | {tid}")
        plt.show()


def plot_ablation_bars(ablation_results):
    if ablation_results.empty:
        return
    for label, g in ablation_results.groupby("target_label"):
        gg = g.sort_values("MAE_pct")
        plt.figure(figsize=(10, 3))
        plt.bar(gg["ablation"], gg["MAE_pct"])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("MAE (%SOC)")
        plt.title(f"ablation result | {label}")
        plt.tight_layout()
        plt.show()


def plot_cutoff_physical_soc(cutoff_df):
    if cutoff_df.empty:
        return
    plt.figure(figsize=(10, 3))
    labels = cutoff_df["trajectory_id"].astype(str)
    plt.bar(labels, cutoff_df["cutoff_physical_SOC"] * 100.0)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("physical SOC at cutoff (%)")
    plt.title("cutoff physical SOC is not forced to zero")
    plt.tight_layout()
    plt.show()



def run_required_plots(feature_frames, all_predictions, ablation_results, cfg, make_plots=True):
    cutoff_summary = cutoff_physical_soc_summary(feature_frames["train"] + feature_frames["valid"] + feature_frames["test"])
    cutoff_summary.to_csv(cfg.output_dir / "cutoff_physical_soc_summary_fixed.csv", index=False)
    cutoff_summary.to_csv(cfg.output_dir / "cutoff_physical_soc_summary.csv", index=False)
    display(cutoff_summary)

    if make_plots:
        if len(feature_frames["test"]):
            plot_decomposition(feature_frames["test"][0])

        for key, d in all_predictions.items():
            target_label, ablation_name = key
            if ablation_name in ("A1_V_raw_only", "A2_V_corr_only", "F1_full_decomposed", "F2_raw_plus_full_decomposed"):
                plot_soc_predictions(d["test"], title=ablation_name)
                plot_plateau_errors(d["test"], title=ablation_name)

        plot_ablation_bars(ablation_results)
        plot_cutoff_physical_soc(cutoff_summary)

    return cutoff_summary
