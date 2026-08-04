#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "nmc_ocvstart_lopo_clean"
DST = ROOT / "nmc_ocvstart_endzero_lopo_clean"


def relabel_file(src_path: Path, dst_path: Path) -> dict:
    df = pd.read_csv(src_path)
    if "SOC0_used" not in df.columns:
        raise RuntimeError(f"{src_path} is missing SOC0_used")
    soc0 = float(pd.to_numeric(df["SOC0_used"], errors="coerce").iloc[0])
    if not np.isfinite(soc0) or soc0 <= 0:
        raise RuntimeError(f"{src_path} has invalid SOC0_used={soc0}")

    if "Q_removed_for_OCV_start_label_Ah" in df.columns:
        q_removed = pd.to_numeric(df["Q_removed_for_OCV_start_label_Ah"], errors="coerce")
    elif "Qnet_removed(Ah)" in df.columns:
        q_removed = pd.to_numeric(df["Qnet_removed(Ah)"], errors="coerce")
    else:
        raise RuntimeError(f"{src_path} has no Q_removed column")
    q_removed = q_removed.ffill().fillna(0.0).astype(float)
    q_removed = q_removed - float(q_removed.iloc[0])
    q_end = float(q_removed.iloc[-1])
    if not np.isfinite(q_end) or q_end <= 0:
        raise RuntimeError(f"{src_path} has invalid final Q_removed={q_end}")

    q_eff = q_end / soc0
    soc_unclipped = soc0 - q_removed.to_numpy(np.float64) / q_eff
    soc = np.clip(soc_unclipped, 0.0, 1.0)

    out = df.copy()
    if "Q_ref_lc_ocv_Ah" in out.columns and "Q_ref_lc_ocv_Ah_original" not in out.columns:
        out["Q_ref_lc_ocv_Ah_original"] = out["Q_ref_lc_ocv_Ah"]
    if "Qnet_denom(Ah)" in out.columns and "Qnet_denom_original(Ah)" not in out.columns:
        out["Qnet_denom_original(Ah)"] = out["Qnet_denom(Ah)"]
    if "Qeff_ocv_Ah(debug)" in out.columns and "Qeff_ocv_Ah_original(debug)" not in out.columns:
        out["Qeff_ocv_Ah_original(debug)"] = out["Qeff_ocv_Ah(debug)"]

    out["Qnet_removed(Ah)"] = q_removed.to_numpy(np.float64)
    out["Q_removed_for_OCV_start_label_Ah"] = q_removed.to_numpy(np.float64)
    out["Qnet_denom(Ah)"] = q_eff
    out["Q_ref_lc_ocv_Ah"] = q_eff
    out["Qeff_ocv_Ah(debug)"] = q_end
    out["Q_available_from_SOC0_OCV_Ah"] = q_end
    out["Q_eff_endzero_Ah"] = q_eff
    out["Q_removed_end_Ah"] = q_end
    out["SOC_CC_unclipped"] = soc_unclipped
    out["SOC_CC"] = soc
    out["SOC_CC(%)"] = soc * 100.0
    out["SOC_percent"] = soc * 100.0
    out["Qeff_reaches_cutoff(debug)"] = True
    out["SOC_scale_mode"] = "SOC0_ocv_file_minus_Qremoved_over_profile_endzero_Qeff"
    out["DENOM_MODE"] = "profile_endzero_from_OCV_SOC0_and_final_Qremoved"
    out["ENFORCE_MONOTONIC"] = True

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst_path, index=False)
    return {
        "file": dst_path.name,
        "temperature": dst_path.parent.name,
        "profile": str(out["Profile"].iloc[0]) if "Profile" in out.columns else src_path.stem,
        "n": int(len(out)),
        "soc0_pct": soc0 * 100.0,
        "old_end_soc_pct": float(pd.to_numeric(df["SOC_CC"], errors="coerce").iloc[-1]) * (100.0 if float(pd.to_numeric(df["SOC_CC"], errors="coerce").iloc[-1]) <= 1.5 else 1.0),
        "new_end_soc_pct": float(soc[-1] * 100.0),
        "old_q_ref_Ah": float(pd.to_numeric(df.get("Q_ref_lc_ocv_Ah", pd.Series([np.nan])), errors="coerce").iloc[0]),
        "q_removed_end_Ah": q_end,
        "q_eff_endzero_Ah": q_eff,
    }


def main() -> None:
    rows = []
    for path in sorted(SRC.glob("*C/*.csv")):
        rel = path.relative_to(SRC)
        rows.append(relabel_file(path, DST / rel))
    audit = pd.DataFrame(rows).sort_values(["temperature", "profile"])
    audit_path = ROOT / "nmc_goal_vcorr_it_train_dst_selector_results" / "ocvstart_endzero_label_audit.csv"
    audit_path.parent.mkdir(exist_ok=True)
    audit.to_csv(audit_path, index=False)
    print("dataset", DST)
    print("audit", audit_path)
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
