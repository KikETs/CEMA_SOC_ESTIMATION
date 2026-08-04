#!/usr/bin/env python3
"""Replay the frozen v2.2 LFP KF baselines on freshly prepared records."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from runtime_env import configure_runtime


RUNTIME_ENV = configure_runtime()

import numpy as np
import pandas as pd
import yaml

from run_v2 import MAIN_METHODS, rows_from_result, summarize
from run_v2_1 import gamma_for, run_filter
from src.data_io import load_all
from src.ekf import coulomb_count
from src.v2_core import FoldParameterMap, OCVGrid


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
LOCKED = ROOT / "locked_parameters"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=REPO / "runs/lfp_kf")
    parser.add_argument("--profiles", default="DST,FUDS,US06")
    parser.add_argument("--temperatures", default="-10,0,10,20,25,30,40,50")
    parser.add_argument(
        "--methods",
        default="coulomb_count,plain_2rc_ekf,hysteresis_2rc_ekf,adaptive_hysteresis_2rc_ekf,hysteresis_2rc_ukf",
    )
    parser.add_argument("--initial-offsets", default="0,-5,-10")
    return parser.parse_args()


def paper_cc_metrics(
    predictions: pd.DataFrame,
    output_root: Path,
) -> pd.DataFrame:
    """Aggregate every CC initialization as an unweighted mean of 24 slices."""
    direct = predictions[predictions.method == "coulomb_count"]
    slice_frame = summarize(direct, ["method", "initial_condition", "profile", "temperature_C"])
    slice_frame["aggregation_scope"] = "full_evaluation_slice"
    slice_frame.to_csv(output_root / "paper_cc_openloop_slices.csv", index=False)

    rows = []
    for label in ("oracle", "minus5pp", "minus10pp"):
        values = slice_frame.loc[slice_frame.initial_condition == label, "MAE_pct"]
        if len(values) != 24:
            raise RuntimeError(f"Expected 24 CC slices for {label}, found {len(values)}")
        rows.append({
            "method": "coulomb_count", "initial_condition": label,
            "MAE_pct": float(values.mean()),
            "aggregation": "slice_unweighted_mean_of_24_full_evaluation_slices",
            "n_slices": int(len(values)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load((ROOT / "configs/remote.yaml").read_text(encoding="utf-8"))
    v2 = yaml.safe_load((ROOT / "configs/v2.yaml").read_text(encoding="utf-8"))
    lfp_preprocessed = REPO / "Data/Preprocessed/LFP"
    proposed_results = REPO / "runs/lfp_dl/nmc_goal_vcorr_it_train_dst_selector_results"
    base["inputs"].update({
        "prepared_root": str(lfp_preprocessed / "prepared_data_ocv_discharge_soc"),
        "prepared_manifest": str(
            lfp_preprocessed / "manifests_ocv_discharge_3lopo/prepared_dataset_manifest.csv"
        ),
        "ocv_root": str(lfp_preprocessed / "ocv_csv"),
        "proposed_results_root": str(proposed_results),
    })
    v2["proposed_confirmatory"].update({
        "tier1_root": str(proposed_results),
        "winner_promotion_root": str(proposed_results),
    })
    trajectories = load_all(base)
    profiles = {value.strip().upper() for value in args.profiles.split(",") if value.strip()}
    temperatures = {float(value) for value in args.temperatures.split(",") if value.strip()}
    trajectories = [tr for tr in trajectories if tr.profile in profiles and tr.temperature_C in temperatures]
    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    offsets = tuple(float(value) for value in args.initial_offsets.split(",") if value.strip())

    ocv = OCVGrid(pd.read_csv(LOCKED / "artifacts/ocv_table.csv"), v2["slope_noise"])
    ecm = pd.read_csv(LOCKED / "v2_1/ecm_fit_quality.csv")
    unique = ecm.groupby(["fold_holdout", "temperature_C"], as_index=False).first()
    pmap = FoldParameterMap(unique, pd.read_csv(LOCKED / "artifacts/ecm_parameter_map.csv"))
    selected = pd.read_csv(LOCKED / "v2_1/qr_selection.csv").sort_values("range_cycle").groupby("fold_holdout").tail(1)
    noises = {
        row.fold_holdout: {key: float(getattr(row, key)) for key in ("q_soc", "q_vp", "q_h", "r_voltage")}
        for row in selected.itertuples()
    }
    labels = {0.0: "oracle", -5.0: "minus5pp", -10.0: "minus10pp", 5.0: "plus5pp"}
    frames = []
    failures = []
    for tr in trajectories:
        for offset in offsets:
            label = labels.get(offset, f"offset{offset:+g}pp")
            soc0 = float(np.clip(tr.soc_ref[0] + offset / 100.0, 0.0, 1.0))
            for method in methods:
                if method == "coulomb_count":
                    result = coulomb_count(tr, soc0)
                elif method in MAIN_METHODS:
                    result = run_filter(
                        method, tr, tr.profile, ocv, pmap, noises[tr.profile],
                        gamma_for(ecm, tr.profile, tr.temperature_C), v2, offset,
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")
                frames.append(rows_from_result(method, label, tr, result, ocv))
                if result.diverged:
                    failures.append({
                        "method": method, "initial_condition": label, "profile": tr.profile,
                        "temperature_C": tr.temperature_C,
                        "reason": getattr(result, "divergence_reason", "diverged"),
                    })
            print(f"[{tr.profile} {tr.temperature_C:g}C] {label} complete", flush=True)

    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(args.output_root / "prediction_rows.csv.gz", index=False, compression="gzip")
    temperature = summarize(predictions, ["method", "initial_condition", "profile", "temperature_C"])
    temperature.to_csv(args.output_root / "temperature_metrics.csv", index=False)
    rows = []
    for (method, initial), group in predictions.groupby(["method", "initial_condition"]):
        slices = summarize(group, ["profile", "temperature_C"])
        rows.append({
            "method": method, "initial_condition": initial, "weighting": "slice_unweighted_primary",
            "MAE_pct": float(slices["MAE_pct"].mean()), "RMSE_pct": float(slices["RMSE_pct"].mean()),
            "n_slices": len(slices), "n_points": len(group),
        })
    pd.DataFrame(rows).to_csv(args.output_root / "main_table.csv", index=False)
    paper_rows = pd.DataFrame(rows)
    paper_rows["aggregation"] = paper_rows.pop("weighting")
    cc_paper = paper_cc_metrics(predictions, args.output_root)
    paper_rows = paper_rows[paper_rows.method != "coulomb_count"]
    paper_rows = pd.concat([paper_rows, cc_paper], ignore_index=True, sort=False)
    paper_rows.to_csv(args.output_root / "paper_reference_metrics.csv", index=False)
    pd.DataFrame(failures, columns=["method", "initial_condition", "profile", "temperature_C", "reason"]).to_csv(
        args.output_root / "failures.csv", index=False,
    )
    parameter_dir = args.output_root / "parameters"
    parameter_dir.mkdir(exist_ok=True)
    for source in (
        LOCKED / "artifacts/ocv_table.csv",
        LOCKED / "v2_1/ecm_fit_quality.csv", LOCKED / "v2_1/r0_estimates.csv",
        LOCKED / "v2_1/qr_selection.csv",
    ):
        shutil.copy2(source, parameter_dir / source.name)
    (args.output_root / "run_manifest.json").write_text(json.dumps({
        "status": "PASS", "profiles": sorted(profiles), "temperatures_C": sorted(temperatures),
        "methods": list(methods), "initial_offsets_pp": list(offsets),
        "prediction_rows": len(predictions), "failed_runs": len(failures),
        "parameter_source": str(LOCKED.resolve()),
        "ecm_policy": "v2.2 frozen per-fold, per-temperature training-profile-only fit",
        "hppc_used": False,
        "runtime_environment": RUNTIME_ENV,
        "paper_metric_file": "paper_reference_metrics.csv",
        "paper_cc_aggregation_note": (
            "All CC initializations use the unweighted mean of 24 full-evaluation profile-temperature slices."
        ),
        "ukf_numerical_note": (
            "The pure-NumPy UKF is sensitive to LAPACK eigensolver roundoff on unstable slices."
        ),
    }, indent=2) + "\n")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
