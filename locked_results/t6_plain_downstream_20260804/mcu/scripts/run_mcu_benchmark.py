#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import serial


DROP = Path(__file__).resolve().parents[1]
FIRMWARE = DROP / "firmware" / "h563_bench"
SERIAL_PORT = os.environ.get("CEMA_SERIAL_PORT", "/dev/ttyACM0")
PROGRAMMER = os.environ.get("STM32_PROGRAMMER_CLI", "STM32_Programmer_CLI")
ARM_SIZE = os.environ.get("ARM_SIZE", "arm-none-eabi-size")
STLINK_SERIAL = "0024003C3235511138363730"
MAGIC = 0x43454D41
N_WINDOWS = 64
REPS_PER_WINDOW = 16
N_REPS = N_WINDOWS * REPS_PER_WINDOW
BASE_TEXT = 20192
BASE_DATA = 12
BASE_BSS = 12468


def run_logged(command: list[str], log_path: Path, cwd: Path | None = None) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}); see {log_path.relative_to(DROP)}"
        )


def read_exact(port: serial.Serial, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        block = port.read(size - len(chunks))
        if not block:
            raise TimeoutError(f"UART timeout: received {len(chunks)}/{size} bytes")
        chunks.extend(block)
    return bytes(chunks)


def connect_query() -> tuple[serial.Serial, tuple[int, ...]]:
    last_error: Exception | None = None
    for _ in range(5):
        try:
            port = serial.Serial(SERIAL_PORT, 921600, timeout=15)
            time.sleep(0.25)
            port.reset_input_buffer()
            port.write(b"Q")
            port.flush()
            query = struct.unpack("<6I", read_exact(port, 24))
            if query[0] != MAGIC:
                raise RuntimeError(f"Invalid query magic: 0x{query[0]:08x}")
            return port, query
        except Exception as error:
            last_error = error
            try:
                port.close()
            except Exception:
                pass
            time.sleep(0.5)
    raise RuntimeError(f"Cannot query board: {last_error}")


def parse_size(path: Path) -> tuple[int, int, int]:
    output = subprocess.check_output([ARM_SIZE, str(path)], text=True)
    match = re.search(r"\n\s*(\d+)\s+(\d+)\s+(\d+)\s+", output)
    if not match:
        raise RuntimeError(f"Cannot parse arm size output: {output}")
    return tuple(int(value) for value in match.groups())


def write_partial(
    parity_rows: list[dict],
    latency_rows: list[dict],
    memory_rows: list[dict],
    raw_rows: list[dict],
) -> None:
    parity_columns = [
        "model",
        "precision",
        "n_windows",
        "max_abs_diff_pct",
        "mean_abs_diff_pct",
        "transport",
        "chemistry",
        "feature",
        "fold",
        "seed",
        "status",
    ]
    latency_columns = [
        "model",
        "precision",
        "clock_mhz",
        "n_reps",
        "cycles_median",
        "cycles_p90",
        "cycles_max",
        "us_median",
        "us_p90",
        "estimates_per_second",
        "includes_preprocessing",
        "chemistry",
        "feature",
        "fold",
        "seed",
        "status",
    ]
    memory_columns = [
        "model",
        "precision",
        "flash_model_bytes",
        "flash_total_bytes",
        "ram_arena_bytes",
        "ram_window_buffer_bytes",
        "ram_other_bytes",
        "ram_total_bytes",
        "evidence",
        "chemistry",
        "feature",
        "fold",
        "seed",
        "flash_increment_vs_base_bytes",
        "ram_increment_vs_base_bytes",
        "stack_highwater_bytes",
        "status",
    ]
    pd.DataFrame(parity_rows, columns=parity_columns).to_csv(
        DROP / "mcu_onchip_parity.csv", index=False
    )
    pd.DataFrame(latency_rows, columns=latency_columns).to_csv(
        DROP / "mcu_latency.csv", index=False
    )
    pd.DataFrame(memory_rows, columns=memory_columns).to_csv(
        DROP / "mcu_memory.csv", index=False
    )
    pd.DataFrame(raw_rows).to_csv(DROP / "mcu_latency_raw.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-at", default=None)
    parser.add_argument("--stop-after", default=None)
    args = parser.parse_args()

    manifest = pd.read_csv(DROP / "stedgeai_generate_manifest.csv")
    if len(manifest) != 18 or not (manifest["status"] == "PASS").all():
        raise SystemExit("ST Edge AI manifest is not 18/18 PASS")
    onchip_manifest = pd.read_csv(DROP / "onchip_windows_manifest.csv")
    if len(onchip_manifest) != 18 or not (onchip_manifest["n_windows"] == 64).all():
        raise SystemExit("On-chip input manifest is not 18 x 64")

    def prior_rows(name: str) -> list[dict]:
        path = DROP / name
        if args.start_at is None or not path.is_file():
            return []
        return pd.read_csv(path).to_dict("records")

    parity_rows = prior_rows("mcu_onchip_parity.csv")
    latency_rows = prior_rows("mcu_latency.csv")
    memory_rows = prior_rows("mcu_memory.csv")
    raw_rows = prior_rows("mcu_latency_raw.csv")
    started = args.start_at is None
    total = len(manifest)
    for index, row in enumerate(manifest.itertuples(index=False), start=1):
        if not started:
            started = row.model_id == args.start_at
            if not started:
                continue
        print(f"[G3] {index:02d}/{total} {row.model_id}: build", flush=True)
        build_log = DROP / "mcu_logs" / f"{row.model_id}_build.log"
        flash_log = DROP / "mcu_logs" / f"{row.model_id}_flash.log"
        run_logged(
            [str(FIRMWARE / "build_model.sh"), row.model_id],
            build_log,
            cwd=FIRMWARE,
        )
        build_dir = FIRMWARE / "Build" / row.model_id
        elf = build_dir / "cema_h563_bench.elf"
        run_logged(
            [
                PROGRAMMER,
                "-c",
                f"port=SWD",
                f"sn={STLINK_SERIAL}",
                "freq=8000",
                "-w",
                str(elf),
                "-v",
                "-rst",
            ],
            flash_log,
            cwd=DROP,
        )
        port, query = connect_query()
        _, protocol, clock_hz, input_floats, input_bytes, macc = query
        if protocol != 1 or clock_hz != 250_000_000:
            port.close()
            raise RuntimeError(f"{row.model_id}: unexpected query {query}")

        data = np.load(DROP / "onchip_windows" / f"{row.model_id}.npz")
        windows = np.ascontiguousarray(data["windows"], dtype="<f4")
        if windows.shape[0] != N_WINDOWS or windows.nbytes // N_WINDOWS != input_bytes:
            port.close()
            raise RuntimeError(
                f"{row.model_id}: board input={input_bytes}, windows={windows.shape}"
            )
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(DROP / "onnx" / f"{row.model_id}_static.onnx"),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name
        expected = np.asarray(
            [
                session.run(None, {input_name: window[None].astype(np.float32)})[0]
                .reshape(-1)[0]
                for window in windows
            ],
            dtype=np.float32,
        )

        print(
            f"[G3] {index:02d}/{total} {row.model_id}: "
            f"{N_WINDOWS} parity windows, {N_REPS} timed runs",
            flush=True,
        )
        observed = []
        cycles = []
        stack_highwater = 0
        try:
            for window_index, window in enumerate(windows):
                port.write(b"W" + window.tobytes())
                port.flush()
                response = struct.unpack("<IIIIf", read_exact(port, 20))
                magic, status, _, stack_bytes, output = response
                if magic != MAGIC or status != 0:
                    raise RuntimeError(
                        f"{row.model_id}/window{window_index}: response={response}"
                    )
                observed.append(output)
                stack_highwater = max(stack_highwater, stack_bytes)

                port.write(b"B" + struct.pack("<I", REPS_PER_WINDOW))
                port.flush()
                header = struct.unpack("<IIIIf", read_exact(port, 20))
                raw = np.frombuffer(
                    read_exact(port, 4 * REPS_PER_WINDOW), dtype="<u4"
                ).copy()
                if header[0] != MAGIC or header[1] != 0:
                    raise RuntimeError(
                        f"{row.model_id}/window{window_index}: benchmark={header}"
                    )
                if not np.float32(header[4]) == np.float32(output):
                    raise RuntimeError(
                        f"{row.model_id}/window{window_index}: repeated output changed"
                    )
                cycles.extend(raw.tolist())
        finally:
            port.close()

        observed_np = np.asarray(observed, dtype=np.float32)
        diff_pct = np.abs(observed_np.astype(np.float64) - expected) * 100.0
        cycles_np = np.asarray(cycles, dtype=np.uint64)
        if len(cycles_np) != N_REPS:
            raise RuntimeError(f"{row.model_id}: {len(cycles_np)} != {N_REPS}")
        max_diff = float(np.max(diff_pct))
        status = "PASS" if max_diff < 1e-3 else "FAIL"

        text_bytes, data_bytes, bss_bytes = parse_size(elf)
        flash_total = text_bytes + data_bytes
        ram_total = data_bytes + bss_bytes
        arena = int(row.activations_bytes)
        window_buffer = int(input_bytes)
        ram_other = ram_total - arena - window_buffer
        evidence = (
            f"firmware/h563_bench/Build/{row.model_id}/cema_h563_bench.map;"
            f"firmware/h563_bench/Build/{row.model_id}/size.txt;"
            f"{row.report}"
        )
        parity_rows.append(
            {
                "model": row.model_id,
                "precision": "fp32",
                "n_windows": N_WINDOWS,
                "max_abs_diff_pct": max_diff,
                "mean_abs_diff_pct": float(np.mean(diff_pct)),
                "transport": "UART3 ST-LINK VCP 921600 8N1 little-endian float32",
                "chemistry": row.chemistry,
                "feature": row.feature,
                "fold": row.fold,
                "seed": row.seed,
                "status": status,
            }
        )
        latency_rows.append(
            {
                "model": row.model_id,
                "precision": "fp32",
                "clock_mhz": clock_hz / 1e6,
                "n_reps": len(cycles_np),
                "cycles_median": float(np.median(cycles_np)),
                "cycles_p90": float(np.percentile(cycles_np, 90)),
                "cycles_max": int(np.max(cycles_np)),
                "us_median": float(np.median(cycles_np) / (clock_hz / 1e6)),
                "us_p90": float(np.percentile(cycles_np, 90) / (clock_hz / 1e6)),
                "estimates_per_second": float(clock_hz / np.median(cycles_np)),
                "includes_preprocessing": False,
                "chemistry": row.chemistry,
                "feature": row.feature,
                "fold": row.fold,
                "seed": row.seed,
                "status": status,
            }
        )
        memory_rows.append(
            {
                "model": row.model_id,
                "precision": "fp32",
                "flash_model_bytes": int(row.weights_bytes),
                "flash_total_bytes": flash_total,
                "ram_arena_bytes": arena,
                "ram_window_buffer_bytes": window_buffer,
                "ram_other_bytes": ram_other,
                "ram_total_bytes": ram_total,
                "evidence": evidence,
                "chemistry": row.chemistry,
                "feature": row.feature,
                "fold": row.fold,
                "seed": row.seed,
                "flash_increment_vs_base_bytes": flash_total
                - (BASE_TEXT + BASE_DATA),
                "ram_increment_vs_base_bytes": ram_total - (BASE_DATA + BASE_BSS),
                "stack_highwater_bytes": stack_highwater,
                "status": status,
            }
        )
        raw_rows.extend(
            {
                "model": row.model_id,
                "precision": "fp32",
                "window_index": sample_index // REPS_PER_WINDOW,
                "repeat_index": sample_index % REPS_PER_WINDOW,
                "cycles": int(value),
                "clock_hz": clock_hz,
            }
            for sample_index, value in enumerate(cycles_np)
        )
        write_partial(parity_rows, latency_rows, memory_rows, raw_rows)
        print(
            f"[G3] {index:02d}/{total} {row.model_id}: {status}, "
            f"max_diff={max_diff:.9g}%SOC, "
            f"median={np.median(cycles_np)/(clock_hz/1e6):.3f} us",
            flush=True,
        )
        if status != "PASS":
            raise SystemExit(f"G3 parity failed for {row.model_id}")
        if args.stop_after == row.model_id:
            break


if __name__ == "__main__":
    main()
