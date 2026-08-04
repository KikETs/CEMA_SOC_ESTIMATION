#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


DROP = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def fmt(value: float) -> str:
    return f"{value:.12g}"


def main() -> None:
    checkpoints = pd.read_csv(DROP / "checkpoints_manifest.csv")
    golden = pd.read_csv(DROP / "golden_gate_results.csv")
    golden_agg = pd.read_csv(DROP / "golden_gate_aggregate.csv")
    onnx = pd.read_csv(DROP / "onnx_parity_pc.csv")
    onnx_agg = pd.read_csv(DROP / "onnx_parity_agg.csv")
    st = pd.read_csv(DROP / "stedgeai_generate_manifest.csv")
    host_parity = pd.read_csv(DROP / "mcu_onchip_parity.csv")
    host_latency = pd.read_csv(DROP / "mcu_latency.csv")
    host_memory = pd.read_csv(DROP / "mcu_memory.csv")
    host_raw = pd.read_csv(DROP / "mcu_latency_raw.csv")
    vit_parity = pd.read_csv(DROP / "raw_vit_onchip_parity.csv")
    vit_latency = pd.read_csv(DROP / "raw_vit_latency.csv")
    vit_memory = pd.read_csv(DROP / "raw_vit_memory.csv")
    vit_raw = pd.read_csv(DROP / "raw_vit_latency_raw.csv")
    vit_detail = pd.read_csv(DROP / "raw_vit_onchip_parity_rows.csv")

    require(len(checkpoints) == 45, "Checkpoint manifest is not 45 rows")
    require(len(golden) == 45 and (golden["status"] == "PASS").all(), "G1 failed")
    require(len(onnx) == 45 and onnx["bit_identical"].all(), "G2 failed")
    require(len(st) == 45 and (st["status"] == "PASS").all(), "ST generation failed")
    for name, frame in (
        ("host parity", host_parity),
        ("host latency", host_latency),
        ("host memory", host_memory),
        ("raw VIT parity", vit_parity),
        ("raw VIT latency", vit_latency),
        ("raw VIT memory", vit_memory),
    ):
        require(len(frame) == 45, f"{name} is not 45 rows")
        require((frame["status"] == "PASS").all(), f"{name} has failures")
    require(len(host_raw) == 45 * 1024, "Host raw cycle count mismatch")
    require(len(vit_raw) == 45 * 1024, "Raw VIT cycle count mismatch")
    require(len(vit_detail) == 45 * 3 * 22, "Raw VIT parity detail mismatch")
    require(host_parity["max_abs_diff_pct"].max() < 1e-3, "Host G3 gate failed")
    require(vit_parity["max_abs_diff_pct"].max() < 1e-3, "Raw VIT G3 gate failed")

    families = (
        checkpoints[["chemistry", "feature"]]
        .drop_duplicates()
        .sort_values(["chemistry", "feature"])
    )
    summary_rows = []
    for family in families.itertuples(index=False):
        key = (family.chemistry, family.feature)
        delta = onnx_agg.loc[
            (onnx_agg["chemistry"] == key[0])
            & (onnx_agg["feature"] == key[1])
        ].iloc[0]
        for pipeline, parity, latency, memory in (
            (
                "host_preprocessed_window_network_only",
                host_parity,
                host_latency,
                host_memory,
            ),
            (
                "raw_vit_full_preprocessing_on_mcu",
                vit_parity,
                vit_latency,
                vit_memory,
            ),
        ):
            p = parity.loc[
                (parity["chemistry"] == key[0]) & (parity["feature"] == key[1])
            ]
            l = latency.loc[
                (latency["chemistry"] == key[0]) & (latency["feature"] == key[1])
            ]
            m = memory.loc[
                (memory["chemistry"] == key[0]) & (memory["feature"] == key[1])
            ]
            summary_rows.append(
                {
                    "chemistry": key[0],
                    "feature": key[1],
                    "pipeline": pipeline,
                    "precision": "fp32",
                    "n_checkpoints": len(p),
                    "clock_mhz": float(l["clock_mhz"].median()),
                    "latency_us_checkpoint_median": float(l["us_median"].median()),
                    "latency_us_checkpoint_min": float(l["us_median"].min()),
                    "latency_us_checkpoint_max": float(l["us_median"].max()),
                    "estimates_per_second_checkpoint_median": float(
                        l["estimates_per_second"].median()
                    ),
                    "preprocessing_window_us_median": (
                        float(l["preprocessing_window_us_median"].median())
                        if "preprocessing_window_us_median" in l
                        else 0.0
                    ),
                    "flash_model_bytes": int(m["flash_model_bytes"].median()),
                    "flash_total_bytes": int(m["flash_total_bytes"].median()),
                    "ram_arena_bytes": int(m["ram_arena_bytes"].median()),
                    "ram_window_buffer_bytes": int(
                        m["ram_window_buffer_bytes"].median()
                    ),
                    "ram_total_bytes": int(m["ram_total_bytes"].median()),
                    "stack_highwater_bytes": int(m["stack_highwater_bytes"].max()),
                    "onchip_max_abs_diff_pct": float(
                        p["max_abs_diff_pct"].max()
                    ),
                    "onnx_delta_mae_pct": float(delta["agg_delta_pct"]),
                    "display_3dp_changed": bool(delta["display_3dp_changed"]),
                    "includes_preprocessing": bool(
                        l["includes_preprocessing"].iloc[0]
                    ),
                    "status": "PASS",
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(DROP / "mcu_summary.csv", index=False)

    gate_rows = []
    for family in families.itertuples(index=False):
        mask = (golden["chemistry"] == family.chemistry) & (
            golden["feature"] == family.feature
        )
        onnx_mask = (onnx["chemistry"] == family.chemistry) & (
            onnx["feature"] == family.feature
        )
        for gate, status, evidence, note in (
            (
                "G0",
                "PASS",
                "environment_host.txt;environment_mcu.txt;checkpoints_manifest.csv",
                "45 frozen checkpoints; parameter counts match",
            ),
            (
                "G1",
                "PASS" if (golden.loc[mask, "status"] == "PASS").all() else "FAIL",
                "golden_gate_results.csv;golden_gate_aggregate.csv",
                "PyTorch package vs archived predictions",
            ),
            (
                "G2",
                "PASS"
                if onnx.loc[onnx_mask, "bit_identical"].all()
                and onnx.loc[onnx_mask, "golden_max_abs_pct"].max() < 1e-4
                else "FAIL",
                "onnx_parity_pc.csv;onnx_parity_agg.csv",
                "Static ONNX fp32 parity; expected 1e-5 %SOC is diagnostic only",
            ),
            (
                "G3_network_only",
                "PASS",
                "mcu_onchip_parity.csv;mcu_latency.csv;mcu_memory.csv",
                "Normalized feature window sent by host",
            ),
            (
                "G3_raw_vit_end_to_end",
                "PASS",
                "raw_vit_onchip_parity.csv;raw_vit_latency.csv;raw_vit_memory.csv",
                "Only raw V/I/T sent; R0, EMA, scaler, window and NN on MCU",
            ),
        ):
            gate_rows.append(
                {
                    "chemistry": family.chemistry,
                    "feature": family.feature,
                    "gate": gate,
                    "status": status,
                    "evidence": evidence,
                    "note": note,
                }
            )
    gates = pd.DataFrame(gate_rows)
    gates.to_csv(DROP / "gate_summary.csv", index=False)
    require((gates["status"] == "PASS").all(), "Final gate table contains failures")

    anomalies = pd.DataFrame(
        [
            {
                "id": "A1",
                "severity": "diagnostic",
                "finding": (
                    "All 45 static ONNX record_max_abs_pct values exceed the "
                    "non-gating 1e-5 %SOC expectation, while all remain below "
                    "the binding 1e-6-fraction parity gate."
                ),
                "evidence": "onnx_parity_pc.csv;onnx_bisection_diagnostics.csv",
            },
            {
                "id": "A2",
                "severity": "tooling",
                "finding": (
                    "ST Edge AI 4.0.1 report serialization failed under a Korean "
                    "absolute path; conversion was repeated in an ASCII /tmp staging "
                    "path and complete reports were copied back without graph rewrites."
                ),
                "evidence": "stedgeai_generate_manifest.csv;scripts/run_stedgeai_generate.py",
            },
            {
                "id": "A3",
                "severity": "interpretation",
                "finding": (
                    "Network-only firmware RAM includes an 8192-byte cycle capture "
                    "array used by the benchmark harness. Raw V/I/T firmware streams "
                    "cycle results and omits that array, so firmware-total RAM values "
                    "must not be interpreted as preprocessing reducing deployment RAM."
                ),
                "evidence": "mcu_memory.csv;raw_vit_memory.csv;firmware/",
            },
            {
                "id": "A4",
                "severity": "contract",
                "finding": (
                    "Raw V/I/T transport assumes exactly 1.0 s sampling because no "
                    "timestamp is transmitted. A variable-rate deployment must add dt "
                    "to the protocol and use it in the Vcorr EMA."
                ),
                "evidence": "raw_vit_onchip_parity.csv;firmware/h563_raw_vit/Src/main.c",
            },
        ]
    )
    anomalies.to_csv(DROP / "anomalies.csv", index=False)

    lines = [
        "# CEMA SOC STM32H563ZI benchmark",
        "",
        "## Scope and status",
        "",
        "This drop contains 45 frozen checkpoints: NMC G4/T6 and LFP G4/T6/T7,",
        "three LOPO folds and seeds 0/1/2. No retraining, checkpoint selection,",
        "quantization, approximation, or ONNX graph rewrite was performed.",
        "",
        f"- G0/G1/G2/G3 status: all PASS ({len(gates)} family-gate rows).",
        f"- Network-only MCU G3: {len(host_parity)}/45 PASS; maximum absolute "
        f"difference {fmt(host_parity['max_abs_diff_pct'].max())} %SOC.",
        f"- Raw V/I/T end-to-end MCU G3: {len(vit_parity)}/45 PASS; maximum "
        f"absolute difference {fmt(vit_parity['max_abs_diff_pct'].max())} %SOC.",
        "- Board: NUCLEO-H563ZI, STM32H563ZIT6, Cortex-M33, measured 250 MHz.",
        "- Neural inference precision: FP32.",
        "- KF/CC precision: CC FP32; EKF/UKF FP64 state, assets, transport, and output.",
        "",
        "## Two measured boundaries",
        "",
        "1. `mcu_*`: the host sends a normalized `(1, 50, C)` float32 window.",
        "   DWT timing covers only `stai_cema_run`; `includes_preprocessing=False`.",
        "2. `raw_vit_*`: the host sends only raw float32 voltage, current, and",
        "   temperature. The MCU performs training-fold R0 lookup/interpolation,",
        "   causal Vcorr EMA, channel EMAs, scaler normalization, rolling-window",
        "   update, and the GRU residual inference. DWT timing covers the complete",
        "   compute path; `includes_preprocessing=True`.",
        "",
        "The raw V/I/T protocol assumes a fixed 1.0 s sample period. Record reset",
        "reinitializes all EMA states and the rolling window. The first 49 samples",
        "produce no estimate; sample 50 produces the first estimate.",
        "",
        "## Aggregate results",
        "",
        "All values below are generated from `mcu_summary.csv`.",
        "",
        "| Chemistry | Feature | Boundary | Median latency (us) | Preprocess (us) | "
        "Flash total (B) | RAM total (B) | Max MCU delta (%SOC) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        boundary = "raw V/I/T end-to-end" if row.includes_preprocessing else "network only"
        lines.append(
            f"| {row.chemistry} | {row.feature} | {boundary} | "
            f"{fmt(row.latency_us_checkpoint_median)} | "
            f"{fmt(row.preprocessing_window_us_median)} | "
            f"{row.flash_total_bytes} | {row.ram_total_bytes} | "
            f"{fmt(row.onchip_max_abs_diff_pct)} |"
        )
    lines += [
        "",
        "The network-only RAM total includes an 8192-byte cycle capture buffer used",
        "only by that measurement firmware. The raw V/I/T firmware sends each cycle",
        "result immediately and therefore does not contain this buffer. Consult",
        "`ram_arena_bytes`, `ram_window_buffer_bytes`, and linker maps for component",
        "comparisons.",
        "",
        "## Measurement protocol",
        "",
        "- ST Edge AI Core 4.0.1 / STM32CubeAI 12.0.1-RC2, target `stm32h5`,",
        "  optimization `time`, compression `lossless`.",
        "- GNU Arm 14.3.1, `-Os`, Cortex-M33 FPv5-SP-D16 hard-float.",
        "- DWT CYCCNT, measured SystemCoreClock 250 MHz.",
        "- 10 valid estimates discarded, followed by 1024 measured estimates per",
        "  checkpoint. UART transfer and response time are outside DWT timing.",
        "- Interrupts are disabled only around measured compute.",
        "- Network-only parity uses 64 deterministic windows per checkpoint.",
        "- Raw V/I/T parity uses 66 outputs per checkpoint: 22 outputs at each of",
        "  three temperatures after reset and 49-sample warm-up.",
        "- Raw V/I/T PC reference is the static ONNX model fed by the frozen Python",
        "  feature builder using float32 raw inputs and the same fixed 1.0 s timebase.",
        "",
        "## Reproduction commands",
        "",
        "Run from this directory with the board connected:",
        "",
        "```bash",
        "python scripts/run_g0_g1.py",
        "python scripts/run_g2_onnx.py",
        "python scripts/diagnose_g2_fp32.py",
        "python scripts/run_stedgeai_generate.py",
        "python scripts/prepare_onchip_windows.py",
        "python scripts/run_mcu_benchmark.py",
        "python scripts/run_raw_vit_mcu_benchmark.py",
        "python scripts/recompute_raw_vit_onnx_reference.py",
        "python scripts/finalize_drop.py",
        "```",
        "",
        "Both MCU runners accept `--start-at <model_id>` and preserve preceding CSV",
        "rows for resumable execution. Per-model build and flash logs are stored in",
        "`mcu_logs/` and `raw_vit_logs/`.",
        "",
        "## Firmware protocols",
        "",
        "All multibyte fields are little-endian over UART3 ST-LINK VCP at 921600 8N1.",
        "",
        "- Network-only protocol v1: `Q`, `W + 50*C float32`, and `B + uint32 reps`.",
        "- Raw V/I/T protocol v2: `Q`, `R`, and `S + {V,I,T} float32`.",
        "- KF/CC protocol v5: float64 reset, V/I/T/dt, and SOC output.",
        "- Protocol sources: `firmware/h563_bench/Src/main.c` and",
        "  `firmware/h563_raw_vit/Src/main.c`; KF uses `firmware/h563_kf/Src/main.c`.",
        "",
        "## Failures and retries",
        "",
        "1. The initial network-only benchmark reused an input tensor placed inside",
        "   the activation arena. ST inference overwrote it between repetitions.",
        "   A dedicated input-window buffer and an untimed copy before each run fixed",
        "   the issue; repeated outputs then became identical.",
        "2. The first automated `B` request used a 3 s host timeout, shorter than 16",
        "   inferences plus the initial 10 warm-ups. The timeout was changed to 15 s.",
        "3. ST Edge AI report output failed under the Korean absolute path. Generation",
        "   was staged under ASCII `/tmp` paths; generated C and reports were copied",
        "   back. No model graph rewrite was applied.",
        "",
        "## Diagnostic deviations",
        "",
        "The non-binding FP32 expectation `record_max_abs_pct < 1e-5 %SOC` was not",
        "met by any of the 45 ONNX exports. All passed the binding `<1e-6` fraction",
        "gate, and no three-decimal MAE display changed. Bisection attributes the",
        "difference primarily to FP32 operation ordering in the anchor branch.",
        "See `anomalies.csv` and `onnx_bisection_diagnostics.csv`.",
        "",
        "## Evidence",
        "",
        "- `gate_summary.csv`: gate status by chemistry and feature.",
        "- `mcu_summary.csv`: generated comparison table.",
        "- `*_latency_raw.csv`: all 1024 cycle measurements per checkpoint.",
        "- `firmware/**/Build/*/*.map` and `size.txt`: linker and image evidence.",
        "- `stedgeai/models/*/output/cema_generate_report.txt`: vendor model evidence.",
        "- `MANIFEST.sha256`: hashes every drop file except itself.",
        "",
        "The original frozen packages, training repositories, manuscript files, and",
        "locked result directories were not modified.",
    ]
    readme = DROP / "README_rerun.md"
    if readme.exists():
        existing = readme.read_text(encoding="utf-8")
        marker = "## KF and coulomb-counting benchmark"
        if marker in existing:
            lines.extend(["", existing[existing.index(marker):].rstrip()])
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = DROP / "MANIFEST.sha256"
    if manifest.exists():
        manifest.unlink()
    command = (
        "find . -type f ! -name MANIFEST.sha256 -print0 | "
        "sort -z | xargs -0 sha256sum"
    )
    result = subprocess.run(
        ["bash", "-lc", command],
        cwd=DROP,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    manifest.write_text(result.stdout, encoding="utf-8")
    print(
        f"PASS: {len(checkpoints)} checkpoints, "
        f"{len(result.stdout.splitlines())} hashed files"
    )


if __name__ == "__main__":
    main()
