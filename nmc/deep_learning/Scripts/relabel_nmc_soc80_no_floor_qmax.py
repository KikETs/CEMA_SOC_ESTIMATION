#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "nmc_soc80_train_clean"
OUT_ROOT = ROOT / "nmc_soc80_train_nofloor_qmax"
AUDIT_ROOT = ROOT / "nmc_soc80_train_nofloor_qmax_audit"
SUMMARY_PATH = AUDIT_ROOT / "soc80_nofloor_qmax_summary.csv"
REPORT_PATH = AUDIT_ROOT / "soc80_nofloor_qmax_report.md"
SOC0 = 0.8


def trailing_true(mask: np.ndarray) -> int:
    count = 0
    for value in mask[::-1]:
        if bool(value):
            count += 1
        else:
            break
    return count


def relabel_file(src: Path) -> dict:
    rel = src.relative_to(SRC_ROOT)
    dst = OUT_ROOT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    required = {"Qnet_removed(Ah)", "SOC_CC", "SOC_CC(%)"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{src} missing required columns: {missing}")

    old_soc = pd.to_numeric(df["SOC_CC"], errors="coerce").to_numpy(np.float64)
    old_soc_pct = pd.to_numeric(df["SOC_CC(%)"], errors="coerce").to_numpy(np.float64)
    q = pd.to_numeric(df["Qnet_removed(Ah)"], errors="coerce").to_numpy(np.float64)
    q = q - q[0]
    q_max = float(np.nanmax(q))
    if not np.isfinite(q_max) or q_max <= 0:
        raise ValueError(f"{src} has invalid q_max={q_max}")

    soc_unclipped = SOC0 * (1.0 - q / q_max)
    soc = np.clip(soc_unclipped, 0.0, 1.0)
    soc_pct = soc * 100.0
    q_ref_eff = q_max / SOC0

    if "SOC_CC_floorclip_prev" not in df.columns:
        df["SOC_CC_floorclip_prev"] = old_soc
    if "SOC_CC_floorclip_prev(%)" not in df.columns:
        df["SOC_CC_floorclip_prev(%)"] = old_soc_pct
    if "SOC_CC_unclipped_globalQref" not in df.columns and "SOC_CC_unclipped" in df.columns:
        df["SOC_CC_unclipped_globalQref"] = df["SOC_CC_unclipped"]
    if "Q_ref_initial_capacity_Ah_global_prev" not in df.columns and "Q_ref_initial_capacity_Ah" in df.columns:
        df["Q_ref_initial_capacity_Ah_global_prev"] = df["Q_ref_initial_capacity_Ah"]
    if "Qnet_denom_global_prev(Ah)" not in df.columns and "Qnet_denom(Ah)" in df.columns:
        df["Qnet_denom_global_prev(Ah)"] = df["Qnet_denom(Ah)"]

    df["Q_removed_for_SOC80_label_Ah"] = q
    df["Qnet_denom(Ah)"] = q_ref_eff
    df["Q_ref_initial_capacity_Ah"] = q_ref_eff
    df["Q_available_from_SOC80_Ah"] = q_max
    df["SOC0_used"] = SOC0
    df["SOC_CC_unclipped"] = soc_unclipped
    df["SOC_CC"] = soc
    df["SOC_CC(%)"] = soc_pct
    df["SOC_scale_mode"] = "soc80_qmax_no_floor"
    df["DENOM_MODE"] = "per_file_qnet_removed_max_over_soc0"
    df["ENFORCE_MONOTONIC"] = False

    df.to_csv(dst, index=False)

    zero_mask = soc <= 1e-12
    old_zero_mask = old_soc <= 1e-12
    return {
        "file_name": src.name,
        "relative_path": str(rel),
        "temperature_label": str(df["TempLabel"].iloc[0]) if "TempLabel" in df.columns else "",
        "profile": str(df["Profile"].iloc[0]) if "Profile" in df.columns else "",
        "rows": int(len(df)),
        "q_removed_max_Ah": q_max,
        "q_removed_end_Ah": float(q[-1]),
        "q_ref_eff_Ah": q_ref_eff,
        "old_zero_rows": int(old_zero_mask.sum()),
        "old_trailing_zero_rows": trailing_true(old_zero_mask),
        "new_zero_rows": int(zero_mask.sum()),
        "new_trailing_zero_rows": trailing_true(zero_mask),
        "new_start_SOC_pct": float(soc_pct[0]),
        "new_min_SOC_pct": float(np.nanmin(soc_pct)),
        "new_final_SOC_pct": float(soc_pct[-1]),
    }


def main() -> None:
    if not SRC_ROOT.exists():
        raise FileNotFoundError(SRC_ROOT)
    if OUT_ROOT.exists():
        for path in sorted(OUT_ROOT.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

    rows = [relabel_file(path) for path in sorted(SRC_ROOT.rglob("*.csv"))]
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_PATH, index=False)

    report = [
        "# NMC SOC80 no-floor qmax relabel",
        "",
        f"- Source root: `{SRC_ROOT}`",
        f"- Output root: `{OUT_ROOT}`",
        f"- Formula: `SOC_CC = clip({SOC0} * (1 - Qnet_removed(Ah) / max(Qnet_removed(Ah))), 0, 1)`",
        "- The maximum net removed capacity within each trajectory is mapped to 0 SOC.",
        "- Existing floor-clipped labels are preserved as `SOC_CC_floorclip_prev` and `SOC_CC_floorclip_prev(%)`.",
        "",
        "## Zero-row audit",
        "",
        summary[
            [
                "file_name",
                "old_zero_rows",
                "old_trailing_zero_rows",
                "new_zero_rows",
                "new_trailing_zero_rows",
                "new_final_SOC_pct",
            ]
        ].to_markdown(index=False),
        "",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    print(f"wrote {OUT_ROOT}")
    print(f"summary {SUMMARY_PATH}")
    print(f"report {REPORT_PATH}")


if __name__ == "__main__":
    main()
