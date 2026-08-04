#!/usr/bin/env python3
"""Summarize isolated LOPO CEMA-LSTM learnable-L runs."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ISOLATED_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ISOLATED_ROOT / "nmc_goal_vcorr_it_train_dst_selector_results"
LOG_DIR = ISOLATED_ROOT / "lopo_cema_lstm_anchorL_logs"
REPORT_DIR = ISOLATED_ROOT / "reports"
MANIFEST_PATH = REPORT_DIR / "run_manifest.json"
TEMPERATURES = (0.0, 25.0, 45.0)
PROFILES = ("VALIDATION", "DST", "FUDS", "US06")


def _load_jobs() -> list[dict[str, object]]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return list(manifest.get("jobs", [])) + list(manifest.get("aliases", []))


def _format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return ""
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(value):
        return ""
    total = max(0, int(round(value)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _parse_job_timing(prefix: str) -> dict[str, object]:
    path = LOG_DIR / f"{prefix}.log"
    out: dict[str, object] = {
        "job_started_at": "",
        "job_finished_at": "",
        "job_status_log": "",
        "job_returncode": np.nan,
        "job_elapsed_seconds": np.nan,
        "job_elapsed_minutes": np.nan,
        "job_elapsed_hms": "",
    }
    if not path.exists():
        return out
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        key = key.strip()
        if key in {
            "started_at_local",
            "finished_at_local",
            "job_status",
            "returncode",
            "elapsed_seconds",
            "elapsed_hms",
        }:
            values[key] = value.strip()
    out["job_started_at"] = values.get("started_at_local", "")
    out["job_finished_at"] = values.get("finished_at_local", "")
    out["job_status_log"] = values.get("job_status", "")
    if "returncode" in values:
        try:
            out["job_returncode"] = int(values["returncode"])
        except ValueError:
            out["job_returncode"] = np.nan
    if "elapsed_seconds" in values:
        try:
            elapsed = float(values["elapsed_seconds"])
            out["job_elapsed_seconds"] = elapsed
            out["job_elapsed_minutes"] = elapsed / 60.0
            out["job_elapsed_hms"] = values.get("elapsed_hms") or _format_duration(elapsed)
        except ValueError:
            pass
    return out


def _read_metadata(prefix: str) -> dict[str, object]:
    path = RESULT_DIR / f"{prefix}_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_test_metrics(prefix: str) -> pd.DataFrame:
    path = RESULT_DIR / f"{prefix}_by_temperature.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "split" not in df.columns:
        return pd.DataFrame()
    out = df[df["split"].astype(str).eq("test")].copy()
    if out.empty:
        return out
    if "variant" in out.columns and out["variant"].astype(str).str.contains("selected", na=False).any():
        out = out[out["variant"].astype(str).str.contains("selected", na=False)].copy()
    if "epoch" in out.columns:
        out["epoch"] = pd.to_numeric(out["epoch"], errors="coerce")
        out = out[out["epoch"].eq(out["epoch"].max())].copy()
    return out


def _load_l_values(prefix: str) -> dict[int, dict[str, float]]:
    path = RESULT_DIR / f"{prefix}_history.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if not {"seed", "residual_limit_initial", "residual_limit_value"}.issubset(df.columns):
        return {}
    out: dict[int, dict[str, float]] = {}
    for seed, group in df.groupby("seed"):
        values = pd.to_numeric(group["residual_limit_value"], errors="coerce").dropna()
        initial = pd.to_numeric(group["residual_limit_initial"], errors="coerce").dropna()
        if values.empty or initial.empty:
            continue
        out[int(seed)] = {
            "L_initial": float(initial.iloc[0]),
            "L_after_epoch1": float(values.iloc[0]),
            "L_final": float(values.iloc[-1]),
            "L_min": float(values.min()),
            "L_max": float(values.max()),
        }
    return out


def _prediction_files(prefix: str) -> list[Path]:
    return sorted(RESULT_DIR.glob(f"{prefix}_seed*_sel*_*_test_prediction_rows.csv.gz"))


def _seed_from_prediction_path(path: Path) -> int | None:
    match = re.search(r"_seed(\d+)_sel", path.name)
    return int(match.group(1)) if match else None


def _prediction_max_errors(prefix: str) -> dict[tuple[int, float], dict[str, float]]:
    out: dict[tuple[int, float], dict[str, float]] = {}
    for path in _prediction_files(prefix):
        seed = _seed_from_prediction_path(path)
        if seed is None:
            continue
        with gzip.open(path, "rt") as f:
            df = pd.read_csv(f)
        temp_col = "temperature" if "temperature" in df.columns else "temperature_C"
        if temp_col not in df.columns or "abs_error" not in df.columns:
            continue
        df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
        df["abs_error_pct"] = pd.to_numeric(df["abs_error"], errors="coerce") * 100.0
        for temp, group in df.groupby(temp_col):
            out[(int(seed), float(temp))] = {
                "MaxAE_pct": float(group["abs_error_pct"].max()),
                "prediction_rows": int(len(group)),
                "source_prediction_file": path.name,
            }
    return out


def _mean_std(series: pd.Series) -> tuple[float, float]:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return float("nan"), float("nan")
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return float(vals.mean()), std


def _spec_from_job(job: dict[str, object]) -> dict[str, object]:
    spec = job.get("spec", {})
    return spec if isinstance(spec, dict) else {}


def _expected_seeds_for_job(job: dict[str, object]) -> set[int]:
    cmd = [str(x) for x in job.get("command", [])]
    if "--seeds" in cmd:
        idx = cmd.index("--seeds")
        if idx + 1 < len(cmd):
            return {int(x.strip()) for x in cmd[idx + 1].split(",") if x.strip()}
    return {0, 1, 2}


def build_seed_rows(jobs: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for job in jobs:
        prefix = str(job["prefix"])
        source_prefix = str(job.get("source_prefix", prefix))
        is_alias = source_prefix != prefix
        spec = _spec_from_job(job)
        timing = _parse_job_timing(source_prefix)
        meta = _read_metadata(source_prefix)
        l_values = _load_l_values(source_prefix)
        max_errors = _prediction_max_errors(source_prefix)
        metrics = _selected_test_metrics(source_prefix)
        if metrics.empty:
            rows.append(
                {
                    "status": "missing",
                    "prefix": prefix,
                    "source_prefix": source_prefix,
                    "is_alias": bool(is_alias),
                    "run_group": spec.get("run_group", ""),
                    "heldout_profile": job.get("holdout_profile"),
                    "train_profiles": job.get("train_profiles", ""),
                    "model": spec.get("model_id"),
                    "feature_set": spec.get("feature_set"),
                    "input_dim": np.nan,
                    "model_kind": spec.get("model_kind", ""),
                    "recurrent": spec.get("recurrent", ""),
                    "layers": spec.get("layers", ""),
                    "hidden_size": spec.get("hidden_size", ""),
                    "head": "",
                    "seed": np.nan,
                    "temperature": np.nan,
                    "MAE_pct": np.nan,
                    "RMSE_pct": np.nan,
                    "MaxAE_pct": np.nan,
                    "Bias_pct": np.nan,
                    "n_windows": np.nan,
                    "residual_limit_mode": "",
                    "residual_limit_init": "",
                    "L_initial": np.nan,
                    "L_after_epoch1": np.nan,
                    "L_final": np.nan,
                    "L_min": np.nan,
                    "L_max": np.nan,
                    "source_prediction_file": "",
                    **timing,
                }
            )
            continue
        for _, row in metrics.iterrows():
            seed = int(row["seed"])
            temp = float(row["temperature_C"])
            l_meta = l_values.get(seed, {})
            max_meta = max_errors.get((seed, temp), {})
            train_profiles = job.get("train_profiles", meta.get("train_profiles", ""))
            rows.append(
                {
                    "status": "complete",
                    "prefix": prefix,
                    "source_prefix": source_prefix,
                    "is_alias": bool(is_alias),
                    "run_group": spec.get("run_group", ""),
                    "model": spec.get("model_id", ""),
                    "feature_set": spec.get("feature_set", meta.get("feature_set", "")),
                    "input_dim": meta.get("input_feature_dim", np.nan),
                    "model_kind": spec.get("model_kind", meta.get("model_kind", "")),
                    "recurrent": spec.get("recurrent", meta.get("recurrent", "")),
                    "layers": spec.get("layers", meta.get("layers", "")),
                    "hidden_size": spec.get("hidden_size", meta.get("hidden_size", "")),
                    "head": "anchor_residual_head" if "anchor_residual" in str(spec.get("model_kind", "")) else "ordinary_linear_head",
                    "heldout_profile": job.get("holdout_profile", ""),
                    "train_profiles": train_profiles,
                    "seed": seed,
                    "temperature": temp,
                    "MAE_pct": float(row["MAE_pct"]),
                    "RMSE_pct": float(row["RMSE_pct"]) if "RMSE_pct" in row else np.nan,
                    "MaxAE_pct": max_meta.get("MaxAE_pct", np.nan),
                    "Bias_pct": float(row["bias_pct"]) if "bias_pct" in row else np.nan,
                    "n_windows": int(row["n_windows"]) if "n_windows" in row and not pd.isna(row["n_windows"]) else np.nan,
                    "residual_limit_mode": meta.get("anchor_residual_limit_mode", ""),
                    "residual_limit_init": meta.get("anchor_residual_limit_init", ""),
                    "L_initial": l_meta.get("L_initial", np.nan),
                    "L_after_epoch1": l_meta.get("L_after_epoch1", np.nan),
                    "L_final": l_meta.get("L_final", np.nan),
                    "L_min": l_meta.get("L_min", np.nan),
                    "L_max": l_meta.get("L_max", np.nan),
                    "source_prediction_file": max_meta.get("source_prediction_file", ""),
                    **timing,
                }
            )
    return pd.DataFrame(rows)


def proposed_performance(seed_rows: pd.DataFrame) -> pd.DataFrame:
    df = seed_rows[
        seed_rows["status"].eq("complete")
        & seed_rows["model"].eq("CEMA-LSTM_proposed")
        & seed_rows["feature_set"].eq("paper_g4_all_ema")
    ].copy()
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (heldout, temp), group in df.groupby(["heldout_profile", "temperature"]):
        mae_mean, mae_std = _mean_std(group["MAE_pct"])
        rmse_mean, rmse_std = _mean_std(group["RMSE_pct"])
        maxae_mean, _ = _mean_std(group["MaxAE_pct"])
        bias_mean, _ = _mean_std(group["Bias_pct"])
        rows.append(
            {
                "heldout_profile": heldout,
                "temperature": temp,
                "MAE_mean": mae_mean,
                "MAE_std": mae_std,
                "RMSE_mean": rmse_mean,
                "RMSE_std": rmse_std,
                "MaxAE_mean": maxae_mean,
                "Bias_mean": bias_mean,
                "seed_count": int(group["seed"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["heldout_profile", "temperature"])


def model_comparison(seed_rows: pd.DataFrame) -> pd.DataFrame:
    df = seed_rows[seed_rows["status"].eq("complete") & seed_rows["run_group"].eq("model_comparison")].copy()
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for model, group in df.groupby("model"):
        per_seed = group.groupby("seed", as_index=False)["MAE_pct"].mean()
        profile_temp = group.groupby(["heldout_profile", "temperature"], as_index=False)["MAE_pct"].mean()
        profile_mean = profile_temp.groupby("heldout_profile", as_index=False)["MAE_pct"].mean()
        rows.append(
            {
                "model": model,
                "input": "G4 causal EMA",
                "LOPO_mean_MAE": float(group["MAE_pct"].mean()),
                "seed_std": float(per_seed["MAE_pct"].std(ddof=1)) if len(per_seed) > 1 else 0.0,
                "worst_profile_temperature_MAE": float(profile_temp["MAE_pct"].max()),
                "worst_profile_mean_MAE": float(profile_mean["MAE_pct"].max()),
                "notes": str(group["head"].iloc[0]),
            }
        )
    order = ["CEMA-LSTM_proposed", "Vanilla_LSTM_G4", "GRU_G4", "Transformer_G4_L2", "Endpoint_MLP_G4"]
    out = pd.DataFrame(rows)
    out["_order"] = out["model"].map({name: i for i, name in enumerate(order)}).fillna(999)
    return out.sort_values("_order").drop(columns="_order")


def feature_ablation(seed_rows: pd.DataFrame) -> pd.DataFrame:
    df = seed_rows[seed_rows["status"].eq("complete") & seed_rows["run_group"].eq("feature_ablation")].copy()
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    g4_mean = float(df[df["feature_set"].eq("paper_g4_all_ema")]["MAE_pct"].mean()) if df["feature_set"].eq("paper_g4_all_ema").any() else math.nan
    for feature_set, group in df.groupby("feature_set"):
        per_seed = group.groupby("seed", as_index=False)["MAE_pct"].mean()
        profile_temp = group.groupby(["heldout_profile", "temperature"], as_index=False)["MAE_pct"].mean()
        rows.append(
            {
                "feature_set": feature_set,
                "input_role": str(group["model"].iloc[0]).replace("CEMA-LSTM_", ""),
                "input_dim": int(pd.to_numeric(group["input_dim"], errors="coerce").dropna().iloc[0]) if pd.to_numeric(group["input_dim"], errors="coerce").notna().any() else np.nan,
                "LOPO_mean_MAE": float(group["MAE_pct"].mean()),
                "seed_std": float(per_seed["MAE_pct"].std(ddof=1)) if len(per_seed) > 1 else 0.0,
                "worst_profile_temperature_MAE": float(profile_temp["MAE_pct"].max()),
                "delta_vs_G4": float(group["MAE_pct"].mean() - g4_mean) if not math.isnan(g4_mean) else np.nan,
                "notes": "CEMA-LSTM anchor-residual head; L is learnable scalar",
            }
        )
    order = ["paper_g0_raw", "paper_g1_derivatives", "paper_g4_all_ema", "paper_g6_full23", "paper_g7_no_current_ema", "paper_g8_no_voltage_ema"]
    out = pd.DataFrame(rows)
    out["_order"] = out["feature_set"].map({name: i for i, name in enumerate(order)}).fillna(999)
    return out.sort_values("_order").drop(columns="_order")


def missing_combinations(seed_rows: pd.DataFrame, jobs: list[dict[str, object]]) -> pd.DataFrame:
    done = seed_rows[seed_rows["status"].eq("complete")].copy()
    rows: list[dict[str, object]] = []
    for job in jobs:
        spec = _spec_from_job(job)
        prefix = str(job["prefix"])
        sub = done[done["prefix"].eq(prefix)]
        expected_seeds = _expected_seeds_for_job(job)
        for seed in expected_seeds:
            for temp in TEMPERATURES:
                if not ((sub["seed"].eq(seed)) & (sub["temperature"].eq(temp))).any():
                    rows.append(
                        {
                            "prefix": prefix,
                            "run_group": spec.get("run_group", ""),
                            "model": spec.get("model_id", ""),
                            "feature_set": spec.get("feature_set", ""),
                            "heldout_profile": job.get("holdout_profile", ""),
                            "seed": seed,
                            "temperature": temp,
                        }
                    )
    return pd.DataFrame(rows)


def job_timing_summary(jobs: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for job in jobs:
        prefix = str(job["prefix"])
        source_prefix = str(job.get("source_prefix", prefix))
        spec = _spec_from_job(job)
        summary_path = RESULT_DIR / f"{source_prefix}_test_summary.csv"
        timing = _parse_job_timing(source_prefix)
        rows.append(
            {
                "prefix": prefix,
                "source_prefix": source_prefix,
                "is_alias": bool(source_prefix != prefix),
                "result_status": "complete" if summary_path.exists() else "missing",
                "run_group": spec.get("run_group", ""),
                "model": spec.get("model_id", ""),
                "feature_set": spec.get("feature_set", ""),
                "model_kind": spec.get("model_kind", ""),
                "recurrent": spec.get("recurrent", ""),
                "layers": spec.get("layers", ""),
                "hidden_size": spec.get("hidden_size", ""),
                "heldout_profile": job.get("holdout_profile", ""),
                "train_profiles": job.get("train_profiles", ""),
                **timing,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_cols = ["heldout_profile", "run_group", "model", "prefix"]
    return out.sort_values([col for col in sort_cols if col in out.columns]).reset_index(drop=True)


def write_notes(seed_rows: pd.DataFrame, missing: pd.DataFrame) -> None:
    complete = seed_rows[seed_rows["status"].eq("complete")].copy()
    text = [
        "# Implementation notes",
        "",
        f"Isolated path: `{ISOLATED_ROOT}`",
        "",
        "- The previous release-folder reconstruction is not used.",
        "- The runner uses the copied experimental module that already implements the anchor-residual head.",
        "- Proposed model label is CEMA-LSTM: `model_kind=anchor_residual_sequence`, `recurrent=lstm`, `layers=1`.",
        "- The residual scale L is a learnable scalar parameter with `anchor_residual_limit_mode=learnable` and seed-dependent `rand01` initialization.",
        "- TCN is not scheduled as a proposed model or baseline in this LOPO suite.",
        "- Validation is built from training-profile blocks only; held-out profile records are not used for scaler fitting or checkpoint selection.",
        "- MaxAE is filled only when prediction-row files exist. Run with `--save-predictions` for manuscript-ready MaxAE.",
        "- Per-job training wall time is read from `lopo_cema_lstm_anchorL_logs/*.log` and written to `reports/lopo_job_timing_summary.csv`.",
        "",
        f"Complete seed-temperature rows: {len(complete)}",
        f"Missing expected rows: {len(missing)}",
    ]
    (REPORT_DIR / "implementation_notes.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize isolated LOPO CEMA-LSTM learnable-L runs.")
    parser.add_argument("--print", action="store_true", dest="print_summary")
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = _load_jobs()
    seed_rows = build_seed_rows(jobs)
    seed_rows.to_csv(REPORT_DIR / "lopo_seed_level_metrics.csv", index=False)
    seed_rows[seed_rows["status"].eq("complete")].to_csv(REPORT_DIR / "lopo_profile_temperature_metrics.csv", index=False)
    proposed_performance(seed_rows).to_csv(REPORT_DIR / "lopo_cema_lstm_proposed_performance.csv", index=False)
    model_comparison(seed_rows).to_csv(REPORT_DIR / "lopo_model_comparison_summary.csv", index=False)
    feature_ablation(seed_rows).to_csv(REPORT_DIR / "lopo_feature_ablation_summary.csv", index=False)
    timing = job_timing_summary(jobs)
    timing.to_csv(REPORT_DIR / "lopo_job_timing_summary.csv", index=False)
    missing = missing_combinations(seed_rows, jobs)
    missing.to_csv(REPORT_DIR / "lopo_missing_combinations.csv", index=False)
    write_notes(seed_rows, missing)
    if args.print_summary:
        print("outputs:")
        for name in [
            "lopo_seed_level_metrics.csv",
            "lopo_profile_temperature_metrics.csv",
            "lopo_cema_lstm_proposed_performance.csv",
            "lopo_model_comparison_summary.csv",
            "lopo_feature_ablation_summary.csv",
            "lopo_job_timing_summary.csv",
            "lopo_missing_combinations.csv",
            "implementation_notes.md",
        ]:
            print(REPORT_DIR / name)
        complete_timing = timing[pd.to_numeric(timing.get("job_elapsed_seconds"), errors="coerce").notna()].copy()
        if not complete_timing.empty:
            print("\njob timing:")
            cols = ["prefix", "source_prefix", "is_alias", "result_status", "job_elapsed_hms", "job_elapsed_minutes"]
            print(complete_timing[cols].head(40).to_string(index=False))
        if not missing.empty:
            print("\nmissing combinations:")
            print(missing.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
