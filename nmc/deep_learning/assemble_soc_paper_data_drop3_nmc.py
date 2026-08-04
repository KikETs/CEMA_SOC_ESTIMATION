#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tarfile
from datetime import date, datetime
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent
DROP_ROOT = Path.home() / "soc_paper_data_drop3_nmc"
LOCKED_ROOT = DROP_ROOT / "data/locked/CEMA_MLP_OCVSTART_FULLGRID"
T6_SOURCE = SOURCE_ROOT / "nmc_5seed_promotion/t6_gru_residual_5seed"
S2_SOURCE = SOURCE_ROOT / "nmc_s2_regen_9records_final"
T6_DEST = LOCKED_ROOT / "nmc_5seed_promotion/t6_gru_residual_5seed"
S2_DEST = LOCKED_ROOT / "s2_regen_9records"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_for(relative: Path) -> str:
    text = relative.as_posix()
    if text.endswith("equivalence_check_12rec.csv"):
        return "code_equivalence_check"
    if text.startswith("s2_regen_9records/"):
        return "s2_regen_9records"
    if text.endswith(("_slice_rows.csv", "_by_seed.csv", "_aggregate.csv", "carrier_5seed.csv")):
        return "posthoc_5seed"
    return "posthoc_5seed_training"


def source_for(relative: Path) -> Path:
    parts = relative.parts
    if parts[0] == "s2_regen_9records":
        return S2_SOURCE.joinpath(*parts[1:])
    return T6_SOURCE.joinpath(*parts[2:])


def main() -> None:
    if DROP_ROOT.exists():
        raise SystemExit(f"Refusing to overwrite existing drop: {DROP_ROOT}")
    archive = Path.home() / f"soc_paper_data_drop3_nmc_{date.today():%Y%m%d}.tar.gz"
    if archive.exists():
        raise SystemExit(f"Refusing to overwrite existing archive: {archive}")
    for required in (T6_SOURCE, S2_SOURCE):
        if not required.is_dir():
            raise FileNotFoundError(required)

    T6_DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(T6_SOURCE, T6_DEST, copy_function=shutil.copy2)
    shutil.copytree(S2_SOURCE, S2_DEST, copy_function=shutil.copy2)

    data_files = sorted(path for path in LOCKED_ROOT.rglob("*") if path.is_file())
    promotion = json.loads((T6_DEST / "promotion_manifest.json").read_text())
    equivalence = list(csv.DictReader((S2_DEST / "equivalence_check_12rec.csv").open()))
    failed = [row["check"] for row in equivalence if row["passed"].lower() != "true"]
    missing = []
    required_names = {
        "t6_gru_residual_5seed_slice_rows.csv", "t6_gru_residual_5seed_by_seed.csv",
        "t6_gru_residual_5seed_aggregate.csv", "carrier_5seed.csv", "autocorr_9rec.csv",
        "lagcorr_9rec.csv", "vi_ambiguity_9rec.csv", "strat_ambiguity_9rec.csv",
        "spectra_9rec.csv", "vi_support_9rec.csv", "equivalence_check_12rec.csv",
        "fig1_trajectory_plot_ready.csv", "fig2_ambiguity_plot_ready.csv",
        "fig3_spectra_plot_ready.csv",
    }
    present_names = {path.name for path in data_files}
    missing.extend(sorted(required_names - present_names))

    manifest_lines = [
        "# SOC Paper Data Drop 3 - NMC",
        "",
        "## Notes",
        "",
        "- Original repositories and existing artifacts were treated as read-only; this drop contains copies only.",
        "- Part A executed only six T6 GRU anchor-residual runs: seeds 3/4 x DST/FUDS/US06.",
        "- Seeds 0/1/2 were reused; batch=2048, epochs=200, last-epoch selection; no tuning.",
        "- The excluded 70-grid five-seed expansion was not run: 396 runs.",
        f"- Excluded-grid serial GPU estimate: {promotion['excluded_full_grid_estimated_serial_gpu_hours']:.3f} h, extrapolated from the measured six-run wall time.",
        "- Section 2 uses the prepared OCV-start labels and excludes VALIDATION only after exact four-profile discovery, preventing VALIDATION from substring-matching DST.",
        "- The 12-record equivalence check retains all outcomes. Six checks pass; the current frozen legacy history function gives 0.430064% rather than the older locked 0.6139% value.",
        f"- MISSING: {', '.join(missing) if missing else 'NONE'}",
        f"- EQUIVALENCE_FAILED_CHECKS: {', '.join(failed) if failed else 'NONE'}",
        "",
        "## File Inventory",
        "",
        "| action | original_absolute_path | size_bytes | mtime | sha256 | analysis_stage | drop_path |",
        "|---|---|---:|---|---|---|---|",
    ]
    for path in data_files:
        relative = path.relative_to(LOCKED_ROOT)
        source = source_for(relative)
        stat = path.stat()
        manifest_lines.append(
            f"| copy | {source} | {stat.st_size} | {datetime.fromtimestamp(stat.st_mtime).isoformat()} | "
            f"{sha256(path)} | {stage_for(relative)} | data/locked/CEMA_MLP_OCVSTART_FULLGRID/{relative.as_posix()} |"
        )
    (DROP_ROOT / "MANIFEST.md").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    header_lines = []
    for path in sorted(LOCKED_ROOT.rglob("*.csv")):
        header_lines.append(f"===== {path.relative_to(DROP_ROOT).as_posix()} =====")
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for _ in range(3):
                line = handle.readline()
                if not line:
                    break
                header_lines.append(line.rstrip("\n"))
        header_lines.append("")
    (DROP_ROOT / "headers.txt").write_text("\n".join(header_lines), encoding="utf-8")

    checksum_files = sorted(path for path in DROP_ROOT.rglob("*") if path.is_file())
    checksum_lines = [f"{sha256(path)}  {path.relative_to(DROP_ROOT).as_posix()}" for path in checksum_files]
    (DROP_ROOT / "sha256sums.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    with tarfile.open(archive, "w:gz") as handle:
        handle.add(DROP_ROOT, arcname=DROP_ROOT.name)
    print(json.dumps({"drop": str(DROP_ROOT), "archive": str(archive), "missing": missing, "equivalence_failed": failed}, indent=2))


if __name__ == "__main__":
    main()
