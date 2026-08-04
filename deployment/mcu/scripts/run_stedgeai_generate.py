#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


DROP = Path(__file__).resolve().parents[1]
STEDGEAI = Path(os.environ.get("STEDGEAI", "stedgeai"))
GCC_BIN = Path(
    os.environ.get("ARM_GCC_BIN", "/usr/bin")
)


def main() -> None:
    parity = pd.read_csv(DROP / "onnx_parity_pc.csv")
    if len(parity) != 45 or set(parity["status"]) != {"PASS"}:
        raise SystemExit("G2 is not a complete 45/45 PASS")
    manifest = pd.read_csv(DROP / "onnx_export_manifest.csv")
    manifest = manifest[manifest["graph_kind"] == "static"].copy()
    if len(manifest) != 45:
        raise SystemExit(f"Expected 45 static ONNX models, found {len(manifest)}")

    root = DROP / "stedgeai" / "models"
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PATH"] = f"{GCC_BIN}:{env.get('PATH', '')}"
    rows = []
    failures = []
    for number, item in enumerate(manifest.itertuples(index=False), start=1):
        model_id = (
            f"{item.chemistry.lower()}_{item.feature.lower()}_"
            f"{item.fold.lower()}_s{item.seed}"
        )
        print(f"[ST] {number:02d}/45 {model_id}", flush=True)
        model_root = root / model_id
        if model_root.exists():
            shutil.rmtree(model_root)
        output = model_root / "output"
        model_root.mkdir(parents=True)
        log_path = model_root / "generate.log"
        # ST Edge AI 4.0.1 cannot serialize Korean absolute paths into its text
        # report. Run in an ASCII-only staging path, then preserve the outputs.
        with tempfile.TemporaryDirectory(
            prefix=f"cema_stedgeai_{model_id}_", dir="/tmp"
        ) as staging_name:
            staging = Path(staging_name)
            shutil.copy2(DROP / item.onnx_file, staging / "model.onnx")
            (staging / "workspace").mkdir()
            (staging / "output").mkdir()
            command = [
                str(STEDGEAI),
                "generate",
                "--model",
                "model.onnx",
                "--type",
                "onnx",
                "--target",
                "stm32h5",
                "--name",
                "cema",
                "--optimization",
                "time",
                "--workspace",
                "workspace",
                "--output",
                "output",
                "--with-report",
                "--verbosity",
                "1",
            ]
            result = subprocess.run(
                command,
                cwd=staging,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            shutil.copytree(staging / "output", output)
        log_path.write_text(result.stdout, encoding="utf-8")
        info_path = output / "cema_c_info.json"
        report_path = output / "cema_generate_report.txt"
        status = (
            "PASS"
            if result.returncode == 0
            and info_path.exists()
            and report_path.exists()
            and report_path.stat().st_size > 1000
            else "FAIL"
        )
        if status == "FAIL":
            failures.append(f"{model_id}: returncode={result.returncode}")
            rows.append(
                {
                    "chemistry": item.chemistry,
                    "feature": item.feature,
                    "fold": item.fold,
                    "seed": item.seed,
                    "model_id": model_id,
                    "status": status,
                    "rewrites_applied": "none",
                    "precision": "fp32",
                    "target": "stm32h5",
                    "optimization": "time",
                    "compression": "lossless",
                    "macc": "",
                    "weights_bytes": "",
                    "activations_bytes": "",
                    "runtime_flash_bytes": "",
                    "runtime_ram_bytes": "",
                    "toolchain_flash_bytes": "",
                    "toolchain_ram_bytes": "",
                    "input_bytes": "",
                    "report": str(report_path.relative_to(DROP)),
                    "log": str(log_path.relative_to(DROP)),
                }
            )
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        memory = info["memory_footprint"]
        graph = info["graphs"][0]
        input_buffer = next(
            buffer
            for buffer in info["buffers"]
            if buffer["name"] == "window_output_array"
        )
        rows.append(
            {
                "chemistry": item.chemistry,
                "feature": item.feature,
                "fold": item.fold,
                "seed": item.seed,
                "model_id": model_id,
                "status": status,
                "rewrites_applied": "none",
                "precision": "fp32",
                "target": "stm32h5",
                "optimization": "time",
                "compression": "lossless",
                "macc": sum(int(node.get("macc", 0)) for node in graph["nodes"]),
                "weights_bytes": int(memory["weights"]),
                "activations_bytes": int(memory["activations"]),
                "runtime_flash_bytes": int(memory["kernel_flash"]),
                "runtime_ram_bytes": int(memory["kernel_ram"]),
                "toolchain_flash_bytes": int(memory["toolchain_flash"]),
                "toolchain_ram_bytes": int(memory["toolchain_ram"]),
                "input_bytes": int(input_buffer["size_bytes"]),
                "report": str(report_path.relative_to(DROP)),
                "log": str(log_path.relative_to(DROP)),
            }
        )
    fields = list(rows[0])
    with (DROP / "stedgeai_generate_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if failures:
        raise SystemExit("ST Edge AI generation failed:\n" + "\n".join(failures))
    print("ST Edge AI 45/45 PASS", flush=True)


if __name__ == "__main__":
    main()
