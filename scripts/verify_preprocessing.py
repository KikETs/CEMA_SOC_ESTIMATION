#!/usr/bin/env python3
"""Fail loudly when the raw-to-paper preprocessing contract is incomplete."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
PREPARED = REPO / "Data/Preprocessed"
PROFILES = {"DST", "FUDS", "US06"}
HASH_REFERENCE = REPO / "reference/preprocessed_file_hashes.csv"
RAW_HASH_REFERENCE = REPO / "reference/raw_source_hashes.csv"
NMC_FINGERPRINT_REFERENCE = REPO / "reference/nmc_preprocessed_fingerprints.csv"
NMC_TOLERANT_PREFIXES = ("V_corr_raw",)

sys.path.insert(0, str(REPO))
from nmc.preprocessing.prepare_calce_nmc import add_features, estimate_r0  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_fingerprint(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    float_decimals: int | None = None,
) -> str:
    """Hash parsed values independently of CSV formatting and host endianness."""
    digest = hashlib.sha256(b"cema-frame-fingerprint-v1\0")
    digest.update(np.asarray([len(frame)], dtype="<i8").tobytes())
    for column in columns:
        encoded_name = column.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "little"))
        digest.update(encoded_name)
        series = frame[column]
        if pd.api.types.is_float_dtype(series.dtype):
            digest.update(b"float64\0")
            values = series.to_numpy(dtype=np.float64, copy=True)
            missing = np.isnan(values)
            values[missing] = 0.0
            if float_decimals is not None:
                values = np.round(values, decimals=float_decimals)
            values[values == 0.0] = 0.0
            digest.update(np.packbits(missing, bitorder="little").tobytes())
            digest.update(values.astype("<f8", copy=False).tobytes())
        elif pd.api.types.is_integer_dtype(series.dtype):
            digest.update(b"int64\0")
            digest.update(series.to_numpy(dtype="<i8", copy=False).tobytes())
        elif pd.api.types.is_bool_dtype(series.dtype):
            digest.update(b"bool\0")
            digest.update(series.to_numpy(dtype=np.uint8, copy=False).tobytes())
        else:
            digest.update(b"text\0")
            for value in series.astype("string"):
                if pd.isna(value):
                    digest.update((2**32 - 1).to_bytes(4, "little"))
                    continue
                encoded = str(value).encode("utf-8")
                digest.update(len(encoded).to_bytes(4, "little"))
                digest.update(encoded)
    return digest.hexdigest()


def verify_raw_hashes() -> None:
    reference = pd.read_csv(RAW_HASH_REFERENCE)
    if len(reference) != 31 or reference["path"].duplicated().any():
        raise AssertionError("Raw source hash reference must contain 31 unique workbooks")
    mismatches = []
    for row in reference.itertuples(index=False):
        path = REPO / row.path
        if not path.is_file():
            mismatches.append(f"MISSING {row.path}")
        elif sha256(path) != row.sha256:
            mismatches.append(f"HASH {row.path}: expected={row.sha256} actual={sha256(path)}")
    if mismatches:
        raise AssertionError("Raw source workbooks differ from the locked reference:\n" + "\n".join(mismatches))


def verify_file_hashes() -> tuple[int, int]:
    reference = pd.read_csv(HASH_REFERENCE)
    fingerprints = pd.read_csv(NMC_FINGERPRINT_REFERENCE).set_index("path")
    if len(reference) != 36 or reference["path"].duplicated().any():
        raise AssertionError("Preprocessing hash reference must contain 36 unique records")
    if len(fingerprints) != 12 or fingerprints.index.duplicated().any():
        raise AssertionError("NMC semantic fingerprint reference must contain 12 unique records")

    mismatches: list[str] = []
    exact_count = 0
    semantic_count = 0
    for row in reference.itertuples(index=False):
        path = REPO / row.path
        if not path.is_file():
            mismatches.append(f"MISSING {row.path}")
            continue
        actual = sha256(path)
        if actual == row.sha256:
            exact_count += 1
            continue
        if row.chemistry != "NMC" or row.path not in fingerprints.index:
            mismatches.append(f"HASH {row.path}: expected={row.sha256} actual={actual}")
            continue

        expected = fingerprints.loc[row.path]
        frame = pd.read_csv(path)
        strict_columns = json.loads(expected.strict_columns_json)
        if list(frame.columns) != json.loads(expected.all_columns_json):
            mismatches.append(f"SCHEMA {row.path}: columns differ from locked reference")
            continue
        if len(frame) != int(expected.rows):
            mismatches.append(f"ROWS {row.path}: expected={expected.rows} actual={len(frame)}")
            continue
        strict_actual = frame_fingerprint(frame, strict_columns)
        if strict_actual != expected.strict_sha256:
            mismatches.append(f"CORE {row.path}: strict semantic fingerprint differs")
        else:
            semantic_count += 1

    if mismatches:
        raise AssertionError("Preprocessed files differ from the locked reference:\n" + "\n".join(mismatches))
    return exact_count, semantic_count


def verify_status() -> None:
    frame = pd.read_csv(PREPARED / "preprocessing_status.csv")
    if not frame["status"].eq("PASS").all():
        raise AssertionError(f"Preprocessing has non-PASS stages:\n{frame[frame.status != 'PASS']}")


def verify_nmc() -> None:
    root = PREPARED / "NMC/nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean"
    files = sorted(root.rglob("NMC_*C_*.csv"))
    if len(files) != 9:
        raise AssertionError(f"Expected 9 NMC records, found {len(files)}")
    required = {"Voltage(V)", "Current(A)", "T", "SOC_CC", "SOC_CC(%)", "Q_ref_lc_ocv_Ah"}
    seen = set()
    records: list[tuple[Path, pd.DataFrame]] = []
    for path in files:
        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise AssertionError(f"{path}: missing {sorted(missing)}")
        temp = float(frame["T"].iloc[0]); profile = str(frame["Profile"].iloc[0]).upper()
        seen.add((temp, profile))
        if not np.allclose(frame["SOC_CC(%)"], 100.0 * frame["SOC_CC"], atol=2e-7, rtol=0):
            raise AssertionError(f"{path}: SOC percent/fraction mismatch")
        records.append((path, frame))
    expected = {(temp, profile) for temp in (0.0, 25.0, 45.0) for profile in PROFILES}
    if seen != expected:
        raise AssertionError(f"NMC scope mismatch: {sorted(seen ^ expected)}")
    manifest = pd.read_csv(PREPARED / "NMC/manifests/soc0fix25_manifest.csv")
    if len(manifest) != 9 or int(manifest["changed"].sum()) != 1:
        raise AssertionError("NMC 25 C SOC0 correction must contain 9 records and exactly one change")

    estimates = pd.DataFrame(
        [estimate_r0(frame).__dict__ for _, frame in records]
    )
    r0_by_temp = estimates.groupby("temperature_C")["r0_ohm"].median().to_dict()
    for path, frame in records:
        temp = float(frame["temperature_C"].iloc[0])
        core = frame.drop(columns=[c for c in frame if c.startswith(NMC_TOLERANT_PREFIXES)])
        rebuilt = add_features(core, r0_by_temp[temp])
        for column in (c for c in frame if c.startswith(NMC_TOLERANT_PREFIXES)):
            if not np.allclose(
                frame[column].to_numpy(float), rebuilt[column].to_numpy(float),
                atol=1e-12, rtol=0.0, equal_nan=True,
            ):
                delta = np.nanmax(np.abs(frame[column].to_numpy(float) - rebuilt[column].to_numpy(float)))
                raise AssertionError(f"{path}: {column} recomputation differs (max_abs={delta:.3e})")


def verify_lfp() -> None:
    root = PREPARED / "LFP/prepared_data_ocv_discharge_soc"
    files = sorted(root.rglob("LFP_*C_*.csv"))
    if len(files) != 24:
        raise AssertionError(f"Expected 24 LFP records, found {len(files)}")
    failures = pd.read_csv(PREPARED / "LFP/raw_adapter_failures.csv")
    if len(failures):
        raise AssertionError(f"LFP adapter failures:\n{failures}")
    required = {"Voltage(V)", "Current(A)", "Temperature(C)", "SOC_CC", "SOC_CC(%)", "Q_ref_lc_ocv_discharge_Ah"}
    seen = set()
    for path in files:
        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise AssertionError(f"{path}: missing {sorted(missing)}")
        temp = float(path.parent.name.removesuffix("C")); profile = path.stem.rsplit("_", 1)[-1].upper()
        seen.add((temp, profile))
        if not np.allclose(frame["SOC_CC(%)"], 100.0 * frame["SOC_CC"], atol=2e-7, rtol=0):
            raise AssertionError(f"{path}: SOC percent/fraction mismatch")
    expected = {(temp, profile) for temp in (-10.0, 0.0, 10.0, 20.0, 25.0, 30.0, 40.0, 50.0) for profile in PROFILES}
    if seen != expected:
        raise AssertionError(f"LFP scope mismatch: {sorted(seen ^ expected)}")


def main() -> None:
    verify_status(); verify_raw_hashes(); verify_nmc(); verify_lfp()
    exact_count, semantic_count = verify_file_hashes()
    print(
        "PASS: preprocessing contract "
        f"(raw workbooks 31 exact; prepared byte-exact={exact_count}, "
        f"NMC semantic-fallback={semantic_count}; no failed stage)"
    )


if __name__ == "__main__":
    main()
