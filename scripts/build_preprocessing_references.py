#!/usr/bin/env python3
"""Build compact preprocessing references from a trusted completed Linux run."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from verify_preprocessing import (
    NMC_TOLERANT_PREFIXES,
    REPO,
    frame_fingerprint,
    sha256,
)


def raw_sources() -> list[Path]:
    required_profiles = ("DST", "FUDS", "US06")
    nmc_profiles = sorted(
        path for path in (REPO / "Data/NMC/Profiles").glob("*80SOC.xls*")
        if any(f"_{profile}_" in path.stem.upper() for profile in required_profiles)
    )
    nmc_ocv = sorted((REPO / "Data/NMC/OCV").glob("*.xls*"))
    adapter = pd.read_csv(REPO / "Data/Preprocessed/LFP/raw_adapter_manifest.csv")
    lfp_sources = sorted({Path(value) for value in adapter["source_file"]})
    paths = nmc_profiles + nmc_ocv + lfp_sources
    if len(nmc_profiles) != 9 or len(nmc_ocv) != 3 or len(lfp_sources) != 16 or len(paths) != 28:
        raise RuntimeError(
            f"Unexpected raw scope: NMC profiles={len(nmc_profiles)}, "
            f"NMC OCV={len(nmc_ocv)}, LFP={len(lfp_sources)}"
        )
    return paths


def main() -> None:
    reference = REPO / "reference"
    reference.mkdir(exist_ok=True)
    raw_rows = [
        {"chemistry": path.parts[path.parts.index("Data") + 1], "path": path.relative_to(REPO).as_posix(), "sha256": sha256(path)}
        for path in raw_sources()
    ]
    pd.DataFrame(raw_rows).sort_values(["chemistry", "path"]).to_csv(
        reference / "raw_source_hashes.csv", index=False, lineterminator="\n"
    )

    nmc_root = REPO / "Data/Preprocessed/NMC/nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean"
    rows = []
    for path in sorted(nmc_root.rglob("NMC_*C_*.csv")):
        frame = pd.read_csv(path)
        tolerant = [column for column in frame.columns if column.startswith(NMC_TOLERANT_PREFIXES)]
        strict = [column for column in frame.columns if column not in tolerant]
        rows.append({
            "path": path.relative_to(REPO).as_posix(),
            "rows": len(frame),
            "all_columns_json": json.dumps(list(frame.columns), ensure_ascii=True, separators=(",", ":")),
            "strict_columns_json": json.dumps(strict, ensure_ascii=True, separators=(",", ":")),
            "strict_sha256": frame_fingerprint(frame, strict),
            "tolerant_columns_json": json.dumps(tolerant, ensure_ascii=True, separators=(",", ":")),
        })
    if len(rows) != 9:
        raise RuntimeError(f"Expected 9 NMC records, found {len(rows)}")
    pd.DataFrame(rows).to_csv(
        reference / "nmc_preprocessed_fingerprints.csv", index=False, lineterminator="\n"
    )
    print("Wrote 28 raw hashes and 9 NMC semantic fingerprints")


if __name__ == "__main__":
    main()
