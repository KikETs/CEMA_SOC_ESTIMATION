#!/usr/bin/env python3
"""Run the frozen NMC and LFP preprocessing chains from ./Data."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "Data"
PREPARED = DATA / "Preprocessed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def run_step(name: str, command: list[str], status_rows: list[dict[str, object]]) -> bool:
    started = time.time()
    result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
    log_dir = DATA / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{name}.stdout.log").write_text(result.stdout, encoding="utf-8")
    (log_dir / f"{name}.stderr.log").write_text(result.stderr, encoding="utf-8")
    status_rows.append({
        "chemistry": name.split("_", 1)[0].upper(), "stage": name,
        "status": "PASS" if result.returncode == 0 else "FAIL", "returncode": result.returncode,
        "elapsed_s": time.time() - started, "command": json.dumps(command),
        "stdout_log": str((log_dir / f"{name}.stdout.log").resolve()),
        "stderr_log": str((log_dir / f"{name}.stderr.log").resolve()),
        "error_tail": result.stderr[-2000:].replace("\n", " ") if result.stderr else "",
    })
    print(f"[{name}] {'PASS' if result.returncode == 0 else 'FAIL'} ({result.returncode})")
    return result.returncode == 0


def stage_nmc_80soc() -> tuple[Path, pd.DataFrame]:
    source = DATA / "NMC" / "Profiles"
    stage = PREPARED / "NMC" / "selected_80soc_raw"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    required_profiles = ("DST", "FUDS", "US06")
    for path in sorted(source.glob("*.xls*")):
        upper = path.stem.upper()
        selected = "80SOC" in upper and any(f"_{profile}_" in upper for profile in required_profiles)
        destination = stage / path.name
        if selected:
            shutil.copy2(path, destination)
        rows.append({
            "source_file": str(path.resolve()), "source_sha256": sha256_file(path),
            "selected_for_paper": selected, "selection_reason": "80SOC" if selected else "excluded_non_80SOC",
            "staged_file": str(destination.resolve()) if selected else "",
        })
    manifest = pd.DataFrame(rows)
    manifest_path = PREPARED / "NMC" / "raw_selection_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    return stage, manifest


def main() -> int:
    PREPARED.mkdir(parents=True, exist_ok=True)
    statuses: list[dict[str, object]] = []

    try:
        nmc_stage, selection = stage_nmc_80soc()
        selected = int(selection["selected_for_paper"].sum()) if len(selection) else 0
        if selected != 9:
            raise RuntimeError(f"Expected 9 NMC 80SOC records, found {selected}")
        statuses.append({"chemistry": "NMC", "stage": "nmc_select_80soc", "status": "PASS", "returncode": 0, "selected_files": selected})
        nmc_base = PREPARED / "NMC" / "nmc_ocvstart_lopo_clean"
        nmc_ok = run_step("nmc_base", [
            sys.executable, str(REPO / "nmc/preprocessing/prepare_calce_nmc.py"),
            "--raw-dir", str(nmc_stage), "--reference-dir", str(DATA / "NMC/OCV"),
            "--reference-metadata-dir", str(REPO / "nmc/preprocessing/locked_metadata"),
            "--out-dir", str(nmc_base),
        ], statuses)
        if nmc_ok:
            run_step("nmc_soc0fix25", [
                sys.executable, str(REPO / "nmc/preprocessing/apply_soc0fix25.py"),
                "--source", str(nmc_base), "--destination",
                str(PREPARED / "NMC/nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean"),
                "--manifest", str(PREPARED / "NMC/manifests/soc0fix25_manifest.csv"),
            ], statuses)
    except Exception as exc:
        statuses.append({
            "chemistry": "NMC", "stage": "nmc_select_80soc", "status": "FAIL", "returncode": 1,
            "error_tail": f"{type(exc).__name__}: {exc}",
        })
        print(f"[NMC] FAIL: {type(exc).__name__}: {exc}")

    lfp_adapter_ok = run_step("lfp_raw_adapter", [
        sys.executable, str(REPO / "lfp/preprocessing/prepare_lfp_raw_excel.py"),
        "--profile-root", str(DATA / "LFP/Profiles"), "--ocv-root", str(DATA / "LFP/OCV"),
        "--output-root", str(PREPARED / "LFP"),
    ], statuses)
    if lfp_adapter_ok:
        adapter_failures = PREPARED / "LFP/raw_adapter_failures.csv"
        failures = pd.read_csv(adapter_failures) if adapter_failures.exists() and adapter_failures.stat().st_size else pd.DataFrame()
        if failures.empty:
            run_step("lfp_labels", [
                sys.executable, str(REPO / "lfp/preprocessing/prepare_lfp_ocv_discharge_soc.py"),
                "--dynamic-root", str(PREPARED / "LFP/raw_adapter"),
                "--ocv-root", str(PREPARED / "LFP/ocv_csv"),
                "--prepared-root", str(PREPARED / "LFP/prepared_data_ocv_discharge_soc"),
                "--manifest-dir", str(PREPARED / "LFP/manifests_ocv_discharge_3lopo"), "--force",
            ], statuses)
        else:
            statuses.append({
                "chemistry": "LFP", "stage": "lfp_labels", "status": "SKIP", "returncode": 0,
                "error_tail": f"raw adapter recorded {len(failures)} failures; see {adapter_failures}",
            })

    status = pd.DataFrame(statuses)
    status.to_csv(PREPARED / "preprocessing_status.csv", index=False)
    incomplete = int(status["status"].isin(["FAIL", "SKIP"]).sum())
    print(
        f"Preprocessing complete: stages={len(status)} incomplete={incomplete}; "
        f"status={PREPARED / 'preprocessing_status.csv'}"
    )
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
