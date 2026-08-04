#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SOC80 dataset using temperature OCV-derived Qeff as the SOC denominator and cutting below 0% SOC."
    )
    parser.add_argument("--src", default="nmc_soc80_train_nofloor_qmax")
    parser.add_argument("--dst", default="nmc_soc80_ocvqeff_cut")
    parser.add_argument("--soc0", type=float, default=0.8)
    return parser.parse_args()


def recompute_file(src_path: Path, dst_path: Path, soc0: float) -> dict:
    df = pd.read_csv(src_path)
    if "Qeff_ocv_Ah(debug)" not in df.columns:
        raise RuntimeError(f"{src_path} is missing Qeff_ocv_Ah(debug)")
    if "Qnet_removed(Ah)" not in df.columns:
        raise RuntimeError(f"{src_path} is missing Qnet_removed(Ah)")
    qeff_values = pd.to_numeric(df["Qeff_ocv_Ah(debug)"], errors="coerce").dropna().unique()
    if len(qeff_values) == 0:
        raise RuntimeError(f"{src_path} has no finite Qeff_ocv_Ah(debug)")
    qdenom = float(qeff_values[0])
    if not np.isfinite(qdenom) or qdenom <= 0.0:
        raise RuntimeError(f"{src_path} has invalid Qeff_ocv_Ah(debug)={qdenom}")

    qremoved = pd.to_numeric(df["Qnet_removed(Ah)"], errors="coerce").to_numpy(np.float64)
    soc_unclipped = float(soc0) - qremoved / qdenom
    valid = np.isfinite(soc_unclipped) & (soc_unclipped >= 0.0)
    if bool(valid.all()):
        keep_end = len(df) - 1
    elif bool(valid.any()):
        first_bad = int(np.argmax(~valid))
        keep_end = first_bad
    else:
        keep_end = 0

    out = df.iloc[: keep_end + 1].copy()
    qremoved_out = pd.to_numeric(out["Qnet_removed(Ah)"], errors="coerce").to_numpy(np.float64)
    soc_out_unclipped = float(soc0) - qremoved_out / qdenom
    soc_out = np.clip(soc_out_unclipped, 0.0, float(soc0))
    if len(out) and soc_out_unclipped[-1] < 0.0:
        soc_out[-1] = 0.0

    out["Qnet_denom(Ah)"] = qdenom
    out["Q_available_from_SOC80_Ah"] = float(soc0) * qdenom
    out["Q_removed_for_SOC80_label_Ah"] = qremoved_out
    out["Q_removed_from_current_recomputed_Ah"] = qremoved_out
    out["SOC_CC"] = soc_out
    out["SOC_CC(%)"] = soc_out * 100.0
    out["SOC_CC_unclipped"] = soc_out_unclipped
    out["SOC_scale_mode"] = "soc80_ocvqeff_cut"
    out["DENOM_MODE"] = "temperature_ocv_qeff_cut_at_zero"
    out["ENFORCE_MONOTONIC"] = False
    if "SOC_CC_prev" in out.columns:
        out["SOC_CC_prev"] = out["SOC_CC"]
    if "SOC_CC_prev(%)" in out.columns:
        out["SOC_CC_prev(%)"] = out["SOC_CC(%)"]
    if "Qnet_denom_prev(Ah)" in out.columns:
        out["Qnet_denom_prev(Ah)"] = qdenom
    if "SOC_CC_floorclip_prev" in out.columns:
        out["SOC_CC_floorclip_prev"] = out["SOC_CC"]
    if "SOC_CC_floorclip_prev(%)" in out.columns:
        out["SOC_CC_floorclip_prev(%)"] = out["SOC_CC(%)"]

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst_path, index=False)
    return {
        "file": str(src_path.relative_to(src_path.parents[1])),
        "out_file": str(dst_path),
        "rows_in": int(len(df)),
        "rows_out": int(len(out)),
        "rows_cut": int(len(df) - len(out)),
        "qeff_ocv_Ah": qdenom,
        "qremoved_end_in_Ah": float(qremoved[-1]) if len(qremoved) else float("nan"),
        "qremoved_end_out_Ah": float(qremoved_out[-1]) if len(qremoved_out) else float("nan"),
        "soc_end_out_pct": float(out["SOC_CC(%)"].iloc[-1]) if len(out) else float("nan"),
        "voltage_end_out_V": float(out["Voltage(V)"].iloc[-1]) if "Voltage(V)" in out.columns and len(out) else float("nan"),
    }


def main() -> None:
    args = parse_args()
    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()
    if not src_root.exists():
        raise FileNotFoundError(src_root)
    rows = []
    for src_path in sorted(src_root.glob("*/*.csv")):
        dst_path = dst_root / src_path.relative_to(src_root)
        rows.append(recompute_file(src_path, dst_path, float(args.soc0)))
    manifest = pd.DataFrame(rows)
    manifest.to_csv(dst_root / "dataset_manifest.tsv", index=False, sep="\t")
    print(f"[done] wrote {len(rows)} files to {dst_root}")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
