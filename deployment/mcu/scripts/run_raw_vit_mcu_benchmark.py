#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import serial


DROP = Path(__file__).resolve().parents[1]
FIRMWARE = DROP / "firmware" / "h563_raw_vit"
REPO = DROP.parents[1]
PACKAGE_ROOTS = {
    "NMC": REPO / "nmc/kf/inference_pkg_nmc",
    "LFP": REPO / "lfp/deep_learning/inference_pkg_lfp",
}
RECORD_ROOTS = {
    "NMC": REPO
    / "Data/Preprocessed/NMC/"
    "nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean",
    "LFP": REPO / "Data/Preprocessed/LFP/prepared_data_ocv_discharge_soc",
}
TEMPS = {"NMC": (0, 25, 45), "LFP": (-10, 0, 10, 20, 25, 30, 40, 50)}
PARITY_TEMPS = {"NMC": (0, 25, 45), "LFP": (-10, 25, 50)}
SERIAL_PORT = os.environ.get("CEMA_SERIAL_PORT", "/dev/ttyACM0")
PROGRAMMER = os.environ.get("STM32_PROGRAMMER_CLI", "STM32_Programmer_CLI")
ARM_SIZE = os.environ.get("ARM_SIZE", "arm-none-eabi-size")
STLINK_SERIAL = "0024003C3235511138363730"
MAGIC = 0x43454D41
PARITY_SAMPLES_PER_RECORD = 71
PARITY_OUTPUTS_PER_RECORD = 22
WARMUP_OUTPUTS = 10
N_REPS = 1024
LATENCY_RAW_SAMPLES = 49 + WARMUP_OUTPUTS + N_REPS
BASE_TEXT = 20192
BASE_DATA = 12
BASE_BSS = 12468


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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
    output = bytearray()
    while len(output) < size:
        block = port.read(size - len(output))
        if not block:
            raise TimeoutError(f"UART timeout: {len(output)}/{size} bytes")
        output.extend(block)
    return bytes(output)


def connect_query() -> tuple[serial.Serial, tuple[int, ...]]:
    last_error: Exception | None = None
    for _ in range(5):
        try:
            port = serial.Serial(SERIAL_PORT, 921600, timeout=15)
            time.sleep(0.25)
            port.reset_input_buffer()
            port.write(b"Q")
            port.flush()
            query = struct.unpack("<8I", read_exact(port, 32))
            if query[0] != MAGIC:
                raise RuntimeError(f"Bad query magic: 0x{query[0]:08x}")
            return port, query
        except Exception as error:
            last_error = error
            try:
                port.close()
            except Exception:
                pass
            time.sleep(0.5)
    raise RuntimeError(f"Cannot query raw V/I/T firmware: {last_error}")


def reset(port: serial.Serial) -> None:
    port.write(b"R")
    port.flush()
    magic, status = struct.unpack("<2I", read_exact(port, 8))
    if magic != MAGIC or status != 0:
        raise RuntimeError(f"Reset failed: {(magic, status)}")


def send_sample(
    port: serial.Serial, voltage: np.float32, current: np.float32, temperature: np.float32
) -> tuple:
    port.write(
        b"S"
        + struct.pack(
            "<fff",
            float(voltage),
            float(current),
            float(temperature),
        )
    )
    port.flush()
    response = struct.unpack("<7If", read_exact(port, 32))
    if response[0] != MAGIC or response[1] != 0:
        raise RuntimeError(f"Sample failed: {response}")
    return response


def record_path(chemistry: str, temperature: int, fold: str) -> Path:
    return (
        RECORD_ROOTS[chemistry]
        / f"{temperature}C"
        / f"{chemistry}_{temperature}C_{fold}.csv"
    )


def raw_frame(path: Path, temperature: int, count: int) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["Voltage(V)", "Current(A)"]).iloc[:count].copy()
    if len(frame) != count:
        raise RuntimeError(f"{path}: need {count}, got {len(frame)}")
    return pd.DataFrame(
        {
            "Voltage(V)": frame["Voltage(V)"].to_numpy(np.float32),
            "Current(A)": frame["Current(A)"].to_numpy(np.float32),
            "T": np.full(count, temperature, dtype=np.float32),
        }
    )


def parse_size(path: Path) -> tuple[int, int, int]:
    output = subprocess.check_output([ARM_SIZE, str(path)], text=True)
    match = re.search(r"\n\s*(\d+)\s+(\d+)\s+(\d+)\s+", output)
    if not match:
        raise RuntimeError(output)
    return tuple(int(value) for value in match.groups())


def prior_rows(name: str, start_at: str | None) -> list[dict]:
    path = DROP / name
    if start_at is None or not path.is_file():
        return []
    return pd.read_csv(path).to_dict("records")


def write_outputs(parity, latency, memory, raw_cycles, parity_detail) -> None:
    pd.DataFrame(parity).to_csv(DROP / "raw_vit_onchip_parity.csv", index=False)
    pd.DataFrame(latency).to_csv(DROP / "raw_vit_latency.csv", index=False)
    pd.DataFrame(memory).to_csv(DROP / "raw_vit_memory.csv", index=False)
    pd.DataFrame(raw_cycles).to_csv(DROP / "raw_vit_latency_raw.csv", index=False)
    pd.DataFrame(parity_detail).to_csv(
        DROP / "raw_vit_onchip_parity_rows.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-at", default=None)
    parser.add_argument("--stop-after", default=None)
    args = parser.parse_args()

    manifest = pd.read_csv(DROP / "stedgeai_generate_manifest.csv")
    if len(manifest) != 45 or not (manifest["status"] == "PASS").all():
        raise SystemExit("ST Edge AI manifest is not 45/45 PASS")
    loaders = {
        chemistry: load_module(
            root / "loader.py", f"_raw_vit_loader_{chemistry.lower()}"
        )
        for chemistry, root in PACKAGE_ROOTS.items()
    }
    parity_rows = prior_rows("raw_vit_onchip_parity.csv", args.start_at)
    latency_rows = prior_rows("raw_vit_latency.csv", args.start_at)
    memory_rows = prior_rows("raw_vit_memory.csv", args.start_at)
    raw_rows = prior_rows("raw_vit_latency_raw.csv", args.start_at)
    detail_rows = prior_rows("raw_vit_onchip_parity_rows.csv", args.start_at)

    started = args.start_at is None
    for index, row in enumerate(manifest.itertuples(index=False), start=1):
        if not started:
            started = row.model_id == args.start_at
            if not started:
                continue
        print(f"[raw VIT] {index:02d}/45 {row.model_id}: build", flush=True)
        build_log = DROP / "raw_vit_logs" / f"{row.model_id}_build.log"
        flash_log = DROP / "raw_vit_logs" / f"{row.model_id}_flash.log"
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
                "port=SWD",
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
        _, protocol, clock_hz, raw_fields, raw_bytes, macc, channels, window = query
        if (
            protocol != 2
            or clock_hz != 250_000_000
            or raw_fields != 3
            or raw_bytes != 12
            or channels * window * 4 != int(row.input_bytes)
        ):
            port.close()
            raise RuntimeError(f"{row.model_id}: bad query {query}")

        package_path = (
            PACKAGE_ROOTS[row.chemistry] / row.feature / row.fold / str(row.seed)
        )
        package = loaders[row.chemistry].load_package(package_path)
        inference_module = sys.modules[package.model.__class__.__module__]
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        onnx_session = ort.InferenceSession(
            str(DROP / "onnx" / f"{row.model_id}_static.onnx"),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        onnx_input = onnx_session.get_inputs()[0].name
        observed = []
        expected = []
        model_details = []
        stack_highwater = 0
        print(
            f"[raw VIT] {index:02d}/45 {row.model_id}: "
            "66 end-to-end parity outputs",
            flush=True,
        )
        try:
            for temperature in PARITY_TEMPS[row.chemistry]:
                frame = raw_frame(
                    record_path(row.chemistry, temperature, row.fold),
                    temperature,
                    PARITY_SAMPLES_PER_RECORD,
                )
                feature_matrix, segments = inference_module.build_feature_matrix(
                    frame, package.config, package.r0_table
                )
                if segments != [(0, len(frame))]:
                    raise RuntimeError(f"{row.model_id}: unexpected segments {segments}")
                scaled = np.ascontiguousarray(
                    (feature_matrix - package.mean) / package.std, dtype=np.float32
                )
                reference = np.asarray(
                    [
                        onnx_session.run(
                            None,
                            {
                                onnx_input: scaled[
                                    end_index - 49 : end_index + 1
                                ][None]
                            },
                        )[0]
                        .reshape(-1)[0]
                        for end_index in range(49, len(frame))
                    ],
                    dtype=np.float32,
                )
                reset(port)
                valid_count = 0
                for sample_index, sample in frame.iterrows():
                    response = send_sample(
                        port,
                        np.float32(sample["Voltage(V)"]),
                        np.float32(sample["Current(A)"]),
                        np.float32(sample["T"]),
                    )
                    _, _, ready, total, network, stack, sample_count, output = response
                    stack_highwater = max(stack_highwater, stack)
                    if ready:
                        expected_value = np.float32(reference[sample_index - 49])
                        observed.append(np.float32(output))
                        expected.append(expected_value)
                        model_details.append(
                            {
                                "model": row.model_id,
                                "temperature_C": temperature,
                                "sample_index": sample_index,
                                "expected_onnx_boundary_fraction": expected_value,
                                "mcu_fraction": np.float32(output),
                                "abs_diff_pct": abs(
                                    float(np.float32(output)) - float(expected_value)
                                )
                                * 100.0,
                                "total_cycles": total,
                                "network_cycles": network,
                            }
                        )
                        valid_count += 1
                if valid_count != PARITY_OUTPUTS_PER_RECORD:
                    raise RuntimeError(
                        f"{row.model_id}/{temperature}: {valid_count} parity outputs"
                    )

            latency_temperature = TEMPS[row.chemistry][0]
            latency_frame = raw_frame(
                record_path(row.chemistry, latency_temperature, row.fold),
                latency_temperature,
                LATENCY_RAW_SAMPLES,
            )
            reset(port)
            timed = []
            ready_index = 0
            for sample_index, sample in latency_frame.iterrows():
                response = send_sample(
                    port,
                    np.float32(sample["Voltage(V)"]),
                    np.float32(sample["Current(A)"]),
                    np.float32(sample["T"]),
                )
                _, _, ready, total, network, stack, sample_count, output = response
                stack_highwater = max(stack_highwater, stack)
                if ready:
                    if ready_index >= WARMUP_OUTPUTS:
                        timed.append((sample_index, total, network))
                    ready_index += 1
        finally:
            port.close()

        if len(timed) != N_REPS:
            raise RuntimeError(f"{row.model_id}: {len(timed)} != {N_REPS}")
        observed_np = np.asarray(observed, dtype=np.float32)
        expected_np = np.asarray(expected, dtype=np.float32)
        diff_pct = np.abs(
            observed_np.astype(np.float64) - expected_np.astype(np.float64)
        ) * 100.0
        total_cycles = np.asarray([value[1] for value in timed], dtype=np.uint64)
        network_cycles = np.asarray([value[2] for value in timed], dtype=np.uint64)
        preprocess_cycles = total_cycles - network_cycles
        max_diff = float(np.max(diff_pct))
        status = "PASS" if max_diff < 1e-3 else "FAIL"
        print(
            f"[raw VIT] {index:02d}/45 {row.model_id}: "
            f"1024 timed end-to-end estimates",
            flush=True,
        )

        text_bytes, data_bytes, bss_bytes = parse_size(elf)
        flash_total = text_bytes + data_bytes
        ram_total = data_bytes + bss_bytes
        arena = int(row.activations_bytes)
        window_bytes = int(row.input_bytes)
        ram_other = ram_total - arena - window_bytes
        evidence = (
            f"firmware/h563_raw_vit/Build/{row.model_id}/cema_h563_bench.map;"
            f"firmware/h563_raw_vit/Build/{row.model_id}/size.txt;"
            f"{row.report}"
        )
        parity_rows.append(
            {
                "model": row.model_id,
                "precision": "fp32",
                "n_windows": len(observed_np),
                "max_abs_diff_pct": max_diff,
                "mean_abs_diff_pct": float(np.mean(diff_pct)),
                "transport": "UART3 raw float32 V/I/T, 921600 8N1 little-endian",
                "chemistry": row.chemistry,
                "feature": row.feature,
                "fold": row.fold,
                "seed": row.seed,
                "timebase": "fixed 1.0 s; timestamp not transmitted",
                "status": status,
            }
        )
        latency_rows.append(
            {
                "model": row.model_id,
                "precision": "fp32",
                "clock_mhz": clock_hz / 1e6,
                "n_reps": N_REPS,
                "cycles_median": float(np.median(total_cycles)),
                "cycles_p90": float(np.percentile(total_cycles, 90)),
                "cycles_max": int(np.max(total_cycles)),
                "us_median": float(np.median(total_cycles) / (clock_hz / 1e6)),
                "us_p90": float(
                    np.percentile(total_cycles, 90) / (clock_hz / 1e6)
                ),
                "estimates_per_second": float(clock_hz / np.median(total_cycles)),
                "includes_preprocessing": True,
                "network_cycles_median": float(np.median(network_cycles)),
                "preprocessing_window_cycles_median": float(
                    np.median(preprocess_cycles)
                ),
                "preprocessing_window_us_median": float(
                    np.median(preprocess_cycles) / (clock_hz / 1e6)
                ),
                "warmup_outputs_discarded": WARMUP_OUTPUTS,
                "latency_temperature_C": latency_temperature,
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
                "ram_window_buffer_bytes": window_bytes,
                "ram_other_bytes": ram_other,
                "ram_total_bytes": ram_total,
                "evidence": evidence,
                "ram_preprocessing_state_bytes": 72,
                "flash_increment_vs_base_bytes": flash_total
                - (BASE_TEXT + BASE_DATA),
                "ram_increment_vs_base_bytes": ram_total - (BASE_DATA + BASE_BSS),
                "stack_highwater_bytes": stack_highwater,
                "chemistry": row.chemistry,
                "feature": row.feature,
                "fold": row.fold,
                "seed": row.seed,
                "status": status,
            }
        )
        raw_rows.extend(
            {
                "model": row.model_id,
                "precision": "fp32",
                "sample_index": int(sample_index),
                "repeat_index": repeat_index,
                "total_cycles": int(total),
                "network_cycles": int(network),
                "preprocessing_window_cycles": int(total - network),
                "clock_hz": clock_hz,
            }
            for repeat_index, (sample_index, total, network) in enumerate(timed)
        )
        detail_rows.extend(model_details)
        write_outputs(
            parity_rows, latency_rows, memory_rows, raw_rows, detail_rows
        )
        print(
            f"[raw VIT] {index:02d}/45 {row.model_id}: {status}, "
            f"max_diff={max_diff:.9g}%SOC, "
            f"end_to_end={np.median(total_cycles)/(clock_hz/1e6):.3f} us, "
            f"preprocess={np.median(preprocess_cycles)/(clock_hz/1e6):.3f} us",
            flush=True,
        )
        if status != "PASS":
            raise SystemExit(f"Raw V/I/T G3 parity failed for {row.model_id}")
        if args.stop_after == row.model_id:
            break


if __name__ == "__main__":
    main()
