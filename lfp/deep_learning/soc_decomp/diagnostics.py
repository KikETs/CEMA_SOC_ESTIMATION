from collections import OrderedDict
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import CFG

try:
    from IPython.display import display
except Exception:
    display = print


def run_shortcut_diagnostics(feature_frames, all_predictions, ablation_results, cfg: CFG, make_plots=True):
    # Shortcut diagnostics: same-voltage SOC spread, plateau tests, cutoff memorization tests
    def build_feature_lookup(feature_frames):
        frames = []
        for split, fs in feature_frames.items():
            for f in fs:
                frames.append(f.assign(split=split))
        cols = [
            "split", "trajectory_id", "end_index", "V_raw", "V_corr_raw",
            "SOC_physical", "SOC_usable_cutoff", "temperature", "drive_cycle"
        ]
        return pd.concat([f[cols] for f in frames], ignore_index=True)


    feature_lookup = build_feature_lookup(feature_frames)


    def attach_voltage_features(pred_df, feature_lookup):
        if pred_df.empty:
            return pred_df.copy()
        keep = ["trajectory_id", "end_index", "V_raw", "V_corr_raw", "SOC_physical", "SOC_usable_cutoff"]
        out = pred_df.merge(feature_lookup[keep], on=["trajectory_id", "end_index"], how="left", validate="many_to_one")
        assert out["V_raw"].notna().all(), "Prediction rows failed to join voltage features"
        return out


    def concatenate_predictions(all_predictions, split="test"):
        rows = []
        for (_, ablation_name), d in all_predictions.items():
            df = d.get(split, pd.DataFrame())
            if len(df):
                rows.append(df.assign(ablation=ablation_name))
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


    all_test_predictions = attach_voltage_features(concatenate_predictions(all_predictions, "test"), feature_lookup)


    def same_voltage_soc_spread_table(frames, voltage_col="V_raw", bin_width=0.02, min_count=5):
        df = pd.concat(frames, ignore_index=True)
        b = np.floor(df[voltage_col].to_numpy(np.float32) / float(bin_width)) * float(bin_width)
        out = df.assign(voltage_col=voltage_col, voltage_bin=np.round(b, 4))
        tab = (
            out.groupby(["voltage_col", "voltage_bin"])["SOC_physical"]
            .agg(count="size", soc_mean="mean", soc_std="std", soc_min="min", soc_max="max")
            .reset_index()
        )
        tab["soc_range"] = tab["soc_max"] - tab["soc_min"]
        tab = tab[tab["count"] >= int(min_count)].sort_values(["soc_std", "soc_range"], ascending=False).reset_index(drop=True)
        return tab


    spread_vraw = same_voltage_soc_spread_table(
        feature_frames["test"], voltage_col="V_raw",
        bin_width=cfg.same_voltage_bin_width_V, min_count=cfg.same_voltage_min_count
    )
    spread_vcorr = same_voltage_soc_spread_table(
        feature_frames["test"], voltage_col="V_corr_raw",
        bin_width=cfg.same_voltage_bin_width_V, min_count=cfg.same_voltage_min_count
    )
    same_voltage_soc_spread = pd.concat([spread_vraw, spread_vcorr], ignore_index=True)
    same_voltage_soc_spread.to_csv(cfg.output_dir / "same_voltage_soc_spread_fixed.csv", index=False)
    same_voltage_soc_spread.to_csv(cfg.output_dir / "same_voltage_soc_spread.csv", index=False)
    print("Same-voltage SOC spread table:")
    display(same_voltage_soc_spread.head(30))


    def same_voltage_error_comparison(pred_df, spread_df, voltage_col, cfg: CFG, plateau_only=False):
        if pred_df.empty or spread_df.empty:
            return pd.DataFrame()
        df = pred_df[pred_df["target_label"] == "physical"].copy()
        if plateau_only:
            df = df[(df["y_true"] >= 0.2) & (df["y_true"] <= 0.8)].copy()
        if df.empty:
            return pd.DataFrame()
        df["voltage_col"] = voltage_col
        df["voltage_bin"] = np.round(np.floor(df[voltage_col].to_numpy(np.float32) / float(cfg.same_voltage_bin_width_V)) * float(cfg.same_voltage_bin_width_V), 4)
        hi = spread_df[
            (spread_df["voltage_col"] == voltage_col)
            & (spread_df["count"] >= int(cfg.same_voltage_min_count))
            & (spread_df["soc_std"] >= float(cfg.same_voltage_min_soc_std))
        ][["voltage_col", "voltage_bin", "count", "soc_std", "soc_range"]]
        joined = df.merge(hi, on=["voltage_col", "voltage_bin"], how="inner")
        if joined.empty:
            return pd.DataFrame()
        rows = []
        for ab, g in joined.groupby("ablation"):
            rows.append({
                "voltage_col": voltage_col,
                "plateau_only": bool(plateau_only),
                "ablation": ab,
                "n_windows": int(len(g)),
                "n_bins": int(g["voltage_bin"].nunique()),
                "mean_bin_soc_std": float(g["soc_std"].mean()),
                "mean_bin_soc_range": float(g["soc_range"].mean()),
                "MAE": float(g["abs_error"].mean()),
                "MAE_pct": float(g["abs_error"].mean() * 100.0),
            })
        return pd.DataFrame(rows).sort_values(["voltage_col", "plateau_only", "MAE"])


    same_voltage_error_comparison_df = pd.concat([
        same_voltage_error_comparison(all_test_predictions, same_voltage_soc_spread, "V_raw", cfg, plateau_only=False),
        same_voltage_error_comparison(all_test_predictions, same_voltage_soc_spread, "V_corr_raw", cfg, plateau_only=False),
        same_voltage_error_comparison(all_test_predictions, same_voltage_soc_spread, "V_raw", cfg, plateau_only=True),
        same_voltage_error_comparison(all_test_predictions, same_voltage_soc_spread, "V_corr_raw", cfg, plateau_only=True),
    ], ignore_index=True)
    same_voltage_error_comparison_df.to_csv(cfg.output_dir / "same_voltage_error_comparison_fixed.csv", index=False)
    same_voltage_error_comparison_df.to_csv(cfg.output_dir / "same_voltage_error_comparison.csv", index=False)
    print("Same-voltage error comparison table:")
    display(same_voltage_error_comparison_df)


    def cutoff_exclusion_metrics(pred_df, cfg: CFG):
        if pred_df.empty:
            return pd.DataFrame()
        rows = []
        compare_ablations = {"A1_V_raw_only", "A3_V_raw_I_T", "F1_full_decomposed", "F2_raw_plus_full_decomposed"}
        conditions = OrderedDict({
            "all_test": lambda g, max_idx: np.ones(len(g), dtype=bool),
            "drop_last_5pct": lambda g, max_idx: g["end_index"].to_numpy() < 0.95 * max_idx,
            "drop_last_10pct": lambda g, max_idx: g["end_index"].to_numpy() < 0.90 * max_idx,
            "drop_V_raw_lt_2p2": lambda g, max_idx: g["V_raw"].to_numpy() >= 2.2,
            "drop_V_raw_lt_2p3": lambda g, max_idx: g["V_raw"].to_numpy() >= 2.3,
        })
        for (target, ablation, tid), g in pred_df.groupby(["target_label", "ablation", "trajectory_id"]):
            if ablation not in compare_ablations:
                continue
            max_idx = float(g["end_index"].max())
            for condition, fn in conditions.items():
                mask = fn(g, max_idx)
                gg = g.loc[mask]
                if gg.empty:
                    continue
                rows.append({
                    "target_label": target,
                    "ablation": ablation,
                    "trajectory_id": tid,
                    "condition": condition,
                    "n_windows": int(len(gg)),
                    "MAE": float(gg["abs_error"].mean()),
                    "MAE_pct": float(gg["abs_error"].mean() * 100.0),
                    "RMSE_pct": float(np.sqrt(np.mean(gg["error"].to_numpy(np.float32) ** 2)) * 100.0),
                })
        detail = pd.DataFrame(rows)
        if detail.empty:
            return detail
        summary = (
            detail.groupby(["target_label", "ablation", "condition"])[["n_windows", "MAE", "MAE_pct", "RMSE_pct"]]
            .agg({"n_windows": "sum", "MAE": "mean", "MAE_pct": "mean", "RMSE_pct": "mean"})
            .reset_index()
        )
        return summary


    cutoff_exclusion_test_table = cutoff_exclusion_metrics(all_test_predictions, cfg)
    cutoff_exclusion_test_table.to_csv(cfg.output_dir / "cutoff_exclusion_test_fixed.csv", index=False)
    cutoff_exclusion_test_table.to_csv(cfg.output_dir / "cutoff_exclusion_test_table.csv", index=False)
    print("Cutoff-exclusion test table:")
    display(cutoff_exclusion_test_table)


    def shortcut_warning_checks(ablation_results, cutoff_exclusion_test_table, cfg: CFG):
        if ablation_results.empty:
            return
        pairs = [
            ("A1_V_raw_only", "F1_full_decomposed"),
            ("A2_V_corr_only", "F1_full_decomposed"),
            ("A5_I_T_only", "F1_full_decomposed"),
            ("A1_V_raw_only", "F2_raw_plus_full_decomposed"),
            ("A2_V_corr_only", "F2_raw_plus_full_decomposed"),
            ("A3_V_raw_I_T", "F1_full_decomposed"),
            ("A3_V_raw_I_T", "F2_raw_plus_full_decomposed"),
        ]
        for target in ablation_results["target_label"].unique():
            sub = ablation_results[ablation_results["target_label"] == target].set_index("ablation")
            for base, full in pairs:
                if base in sub.index and full in sub.index:
                    delta = float(sub.loc[base, "MAE"] - sub.loc[full, "MAE"])
                    if abs(delta) < float(cfg.shortcut_delta_warn_mae):
                        warnings.warn(
                            f"Voltage/current shortcut risk: {full} and {base} differ by only {delta:.4f} MAE for {target}."
                        )
                    if base == "A3_V_raw_I_T" and abs(float(sub.loc[base, "MAE_pct"] - sub.loc[full, "MAE_pct"])) < 0.1:
                        warnings.warn(
                            f"{full} vs A3_V_raw_I_T is not a meaningful improvement for {target}: "
                            "absolute MAE difference is below 0.1 percentage points."
                        )
                    if delta < 0:
                        warnings.warn(
                            f"Full decomposed did not beat shortcut baseline: {full} MAE > {base} MAE for {target}."
                        )
        if len(cutoff_exclusion_test_table):
            piv = cutoff_exclusion_test_table.pivot_table(
                index=["target_label", "condition"], columns="ablation", values="MAE", aggfunc="mean"
            )
            for full in ["F1_full_decomposed", "F2_raw_plus_full_decomposed"]:
                for base in ["A1_V_raw_only", "A3_V_raw_I_T"]:
                    if full in piv.columns and base in piv.columns:
                        near = (piv[base] - piv[full]).abs() < float(cfg.shortcut_delta_warn_mae)
                        for idx in piv.index[near.fillna(False)]:
                            warnings.warn(f"Cutoff-exclusion shortcut risk: {full} and {base} are nearly tied at {idx}.")
                        worse = piv[full] > piv[base]
                        for idx in piv.index[worse.fillna(False)]:
                            warnings.warn(f"Cutoff-exclusion result: {full} does not beat {base} at {idx}.")


    shortcut_warning_checks(ablation_results, cutoff_exclusion_test_table, cfg)


    def plot_same_voltage_spread(spread_df, voltage_col):
        g = spread_df[spread_df["voltage_col"] == voltage_col].sort_values("voltage_bin")
        if g.empty:
            print(f"No spread rows for {voltage_col}")
            return
        plt.figure(figsize=(10, 3))
        plt.plot(g["voltage_bin"], g["soc_range"] * 100.0, marker=".", linewidth=0.8)
        plt.xlabel(f"{voltage_col} bin (V)")
        plt.ylabel("physical SOC range (%SOC)")
        plt.title(f"Same-voltage physical SOC range | {voltage_col}")
        plt.tight_layout()
        plt.show()

    def plot_same_voltage_error_comparison(comp_df, *, plateau_only):
        if comp_df.empty:
            return
        compare = ["A3_V_raw_I_T", "F1_full_decomposed", "F2_raw_plus_full_decomposed"]
        g = comp_df[(comp_df["plateau_only"] == plateau_only) & (comp_df["ablation"].isin(compare))].copy()
        if g.empty:
            print(f"No same-voltage comparison rows for plateau_only={plateau_only}")
            return
        for voltage_col, gg in g.groupby("voltage_col"):
            gg = gg.set_index("ablation").reindex(compare).dropna(subset=["MAE_pct"]).reset_index()
            if gg.empty:
                continue
            plt.figure(figsize=(8, 3))
            plt.bar(gg["ablation"], gg["MAE_pct"])
            plt.xticks(rotation=20, ha="right")
            plt.ylabel("MAE (%SOC)")
            scope = "20-80% plateau" if plateau_only else "all high-spread bins"
            plt.title(f"Same-voltage error comparison | {voltage_col} | {scope}")
            plt.tight_layout()
            plt.show()


    if make_plots:
        plot_same_voltage_spread(same_voltage_soc_spread, "V_raw")
        plot_same_voltage_spread(same_voltage_soc_spread, "V_corr_raw")
        plot_same_voltage_error_comparison(same_voltage_error_comparison_df, plateau_only=False)
        plot_same_voltage_error_comparison(same_voltage_error_comparison_df, plateau_only=True)

    return {
        "feature_lookup": feature_lookup,
        "all_test_predictions": all_test_predictions,
        "same_voltage_soc_spread": same_voltage_soc_spread,
        "same_voltage_error_comparison_df": same_voltage_error_comparison_df,
        "cutoff_exclusion_test_table": cutoff_exclusion_test_table,
    }
