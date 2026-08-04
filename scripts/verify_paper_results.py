#!/usr/bin/env python3
"""Compare fresh paper runs with the locked manuscript reference values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]


def row(
    name: str,
    klass: str,
    expected: float,
    actual: float,
    tolerance: float,
    note: str = "",
    reference_scope: str = "paper reference",
) -> dict:
    delta = float(actual - expected)
    return {
        "metric": name,
        "reproduction_class": klass,
        "expected": expected,
        "actual": actual,
        "absolute_delta": abs(delta),
        "tolerance": tolerance,
        "status": "PASS" if abs(delta) <= tolerance else "FAIL",
        "reference_scope": reference_scope,
        "note": note,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO / "runs/reproduction_verification.csv")
    parser.add_argument("--nmc-dl-root", type=Path, default=REPO / "runs/nmc_dl")
    parser.add_argument("--lfp-dl-root", type=Path, default=REPO / "runs/lfp_dl")
    parser.add_argument("--nmc-kf-root", type=Path, default=REPO / "runs/nmc_kf")
    parser.add_argument("--lfp-kf-root", type=Path, default=REPO / "runs/lfp_kf")
    args = parser.parse_args()
    rows: list[dict] = []

    summary_name = "paper_t6_plain_gru_linear_auxoff_summary.csv"
    nmc_dl = pd.read_csv(args.nmc_dl_root / summary_name).iloc[0]
    lfp_dl = pd.read_csv(args.lfp_dl_root / summary_name).iloc[0]
    rows.append(row("NMC T6-plain GRU 10-seed MAE", "C", 0.348279078267482, nmc_dl.slice_unweighted_MAE_pct, 0.04,
                    "Full retraining; tolerance covers the locked ten-seed standard deviation."))
    rows.append(row("LFP T6-plain GRU 10-seed MAE", "C", 0.644679400016474, lfp_dl.slice_unweighted_MAE_pct, 0.04,
                    "Full retraining; tolerance covers the locked ten-seed standard deviation."))

    nmc = pd.read_csv(args.nmc_kf_root / "results/temperature_metrics.csv")
    nmc = nmc[nmc["condition"] == "oracle_initial_soc"]
    nmc_actual = nmc.groupby("method")["mae_pct"].mean().to_dict()
    for method, (expected, tolerance, klass) in {
        "CC": (0.099450, 1e-6, "B"),
        "Adaptive_2RC_EKF": (1.704694, 0.01, "B-platform"),
        "2RC_EKF": (1.727215, 1e-6, "B"),
        "1RC_EKF": (1.963149, 1e-6, "B"),
        "2RC_UKF": (5.664560, 0.12, "B-platform"),
    }.items():
        rows.append(row(
            f"NMC {method} oracle slice MAE", klass, expected, nmc_actual[method], tolerance,
            "Frozen training-only ECM/bounds/Q-R replay from raw-workbook-regenerated trajectories. "
            "The adaptive EKF and UKF gates include the measured deterministic single-thread "
            "Linux/Windows covariance-repair range."
        ))

    lfp = pd.read_csv(args.lfp_kf_root / "paper_reference_metrics.csv")
    lookup = {(r.method, r.initial_condition): float(r.MAE_pct) for r in lfp.itertuples()}
    expected_lfp = {
        ("plain_2rc_ekf", "oracle"): 0.2417909308,
        ("plain_2rc_ekf", "minus5pp"): 1.0491127566,
        ("plain_2rc_ekf", "minus10pp"): 6.1630418558,
        ("hysteresis_2rc_ekf", "oracle"): 0.2071081229,
        ("hysteresis_2rc_ekf", "minus5pp"): 1.4110274434,
        ("hysteresis_2rc_ekf", "minus10pp"): 6.7080578422,
        ("adaptive_hysteresis_2rc_ekf", "oracle"): 0.2162369251,
        ("adaptive_hysteresis_2rc_ekf", "minus5pp"): 1.4257474045,
        ("adaptive_hysteresis_2rc_ekf", "minus10pp"): 6.1684694124,
        ("coulomb_count", "oracle"): 0.1129039483397598,
        ("coulomb_count", "minus5pp"): 5.029036223673301,
        ("coulomb_count", "minus10pp"): 9.864444057320716,
        ("hysteresis_2rc_ukf", "oracle"): 7.8005395442,
    }
    for key, expected in expected_lfp.items():
        klass = "B"
        tolerance = 1e-6
        note = "Fresh frozen-v2.2 training-profile-only ECM/Q-R replay."
        if key[1] == "minus10pp" and key[0] != "coulomb_count":
            klass = "B-platform"
            tolerance = 0.001
            note += " The -10 pp transient amplifies bounded Linux/Windows arithmetic differences."
        if key[0] == "coulomb_count":
            note += " The paper table uses the unweighted mean of 24 full-evaluation slices."
        elif key == ("hysteresis_2rc_ukf", "oracle"):
            klass = "B-platform"
            tolerance = 0.02
            note += (
                " UKF covariance eigendecomposition is LAPACK-sensitive on diverging slices; "
                "0.02 %SOC covers the observed fresh Linux/Windows OpenBLAS range without substituting "
                "archived predictions."
            )
        rows.append(row(
            f"LFP {key[0]} {key[1]} paper MAE", klass, expected, lookup[key], tolerance, note,
        ))

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    summary = {
        "status": "PASS" if output["status"].eq("PASS").all() else "FAIL",
        "passed": int(output["status"].eq("PASS").sum()),
        "failed": int(output["status"].eq("FAIL").sum()),
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(output.to_string(index=False))
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
