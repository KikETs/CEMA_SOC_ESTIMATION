#!/usr/bin/env python3
"""Apply the paper's declared 25 C US06 SOC0 correction.

This is a path-adapted extraction of the original 2026-06-25 execution: only
25 C US06 is relabeled using mean(SOC0_25C_DST, SOC0_25C_FUDS).
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


PROFILES = ("DST", "FUDS", "US06")
TEMPS = ("0C", "25C", "45C")
AFFECTED = {("25C", "US06")}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def resolve_record(root: Path, temp: str, profile: str) -> Path:
    matches = list(root.rglob(f"NMC_{temp}_{profile}.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {temp}/{profile} record under {root}, found {matches}")
    return matches[0]


def run(source: Path, destination: Path, manifest_path: Path) -> pd.DataFrame:
    frames = {(temp, profile): pd.read_csv(resolve_record(source, temp, profile)) for temp in TEMPS for profile in PROFILES}
    ref_soc0 = float(np.mean([float(frames[("25C", p)]["SOC0_used"].iloc[0]) for p in ("DST", "FUDS")]))
    if destination.exists():
        shutil.rmtree(destination)
    rows: list[dict[str, object]] = []
    for temp in TEMPS:
        output_dir = destination / temp
        output_dir.mkdir(parents=True, exist_ok=True)
        for profile in PROFILES:
            frame = frames[(temp, profile)].copy()
            old_soc0 = float(frame["SOC0_used"].iloc[0])
            changed = (temp, profile) in AFFECTED
            if changed:
                qref = float(frame["Q_ref_lc_ocv_Ah"].iloc[0])
                qnet = pd.to_numeric(frame["Qnet_removed(Ah)"], errors="raise").to_numpy(float)
                unclipped = ref_soc0 - qnet / qref
                soc = np.clip(unclipped, 0.0, 1.0)
                frame["SOC0_used"] = ref_soc0
                if "SOC0_OCV_inferred" in frame:
                    frame["SOC0_OCV_inferred"] = ref_soc0
                if "SOC0_OCV_inferred(%)" in frame:
                    frame["SOC0_OCV_inferred(%)"] = ref_soc0 * 100.0
                if "Q_available_from_SOC0_OCV_Ah" in frame:
                    frame["Q_available_from_SOC0_OCV_Ah"] = ref_soc0 * qref
                if "Qeff_ocv_Ah(debug)" in frame:
                    frame["Qeff_ocv_Ah(debug)"] = ref_soc0 * qref
                frame["SOC_CC"] = soc
                frame["SOC_CC(%)"] = soc * 100.0
                if "SOC_percent" in frame:
                    frame["SOC_percent"] = soc * 100.0
                if "SOC_CC_unclipped" in frame:
                    frame["SOC_CC_unclipped"] = unclipped
                frame["SOC_scale_mode"] = "25C_US06_SOC0_set_to_DST_FUDS_mean_for_label_audit"
                if "SOC0_anchor_source" in frame:
                    frame["SOC0_anchor_source"] = "25C_DST_FUDS_mean_label_audit"
                if "SOC0_mode" in frame:
                    frame["SOC0_mode"] = "25C_DST_FUDS_mean_label_audit"
            output = output_dir / f"NMC_{temp}_{profile}.csv"
            frame.to_csv(output, index=False, lineterminator="\n")
            rows.append({
                "temperature": temp, "profile": profile, "changed": changed,
                "old_SOC0_pct": old_soc0 * 100.0, "new_SOC0_pct": float(frame["SOC0_used"].iloc[0]) * 100.0,
                "rows": len(frame), "output_file": str(output.resolve()), "output_sha256": sha256_file(output),
                "status": "PASS",
            })
    manifest = pd.DataFrame(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    return manifest


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=repo / "Data/Preprocessed/NMC/nmc_ocvstart_lopo_clean")
    parser.add_argument("--destination", type=Path, default=repo / "Data/Preprocessed/NMC/nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean")
    parser.add_argument("--manifest", type=Path, default=repo / "Data/Preprocessed/NMC/manifests/soc0fix25_manifest.csv")
    args = parser.parse_args()
    manifest = run(args.source.resolve(), args.destination.resolve(), args.manifest.resolve())
    print(f"NMC SOC0 correction: pass={len(manifest)} changed={int(manifest.changed.sum())}")


if __name__ == "__main__":
    main()
