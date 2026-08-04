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
import pandas as pd
import serial


DROP = Path(__file__).resolve().parents[1]
REPO = DROP.parents[1]
FIRMWARE = DROP / "firmware/h563_kf"
SERIAL_PORT = os.environ.get("CEMA_SERIAL_PORT", "/dev/ttyACM0")
PROGRAMMER = os.environ.get("STM32_PROGRAMMER_CLI", "STM32_Programmer_CLI")
ARM_SIZE = os.environ.get("ARM_SIZE", "arm-none-eabi-size")
STLINK_SERIAL = "0024003C3235511138363730"
MAGIC = 0x43454D41
CLOCK_HZ = 250_000_000
N_REPS = 1024
WARMUP = 10
FOLDS = ("DST", "FUDS", "US06")
TEMPERATURES = {
    "NMC": (0, 25, 45),
    "LFP": (-10, 0, 10, 20, 25, 30, 40, 50),
}
METHODS = {
    "NMC": {
        "cc": "CC",
        "1rc_ekf": "1RC_EKF",
        "2rc_ekf": "2RC_EKF",
        "adaptive_2rc_ekf": "Adaptive_2RC_EKF",
        "2rc_ukf": "2RC_UKF",
    },
    "LFP": {
        "coulomb_count": "coulomb_count",
        "plain_2rc_ekf": "plain_2rc_ekf",
        "hysteresis_2rc_ekf": "hysteresis_2rc_ekf",
        "adaptive_hysteresis_2rc_ekf": "adaptive_hysteresis_2rc_ekf",
        "hysteresis_2rc_ukf": "hysteresis_2rc_ukf",
    },
}
DATA_ROOTS = {
    "NMC": REPO
    / "Data/Preprocessed/NMC/"
    "nmc_ocvstart_25c_us06_soc0_dstfudsmean_lopo_clean",
    "LFP": REPO
    / "Data/Preprocessed/LFP/prepared_data_ocv_discharge_soc",
}
OUTPUTS = {
    "parity": DROP / "kf_mcu_onchip_parity.csv",
    "rows": DROP / "kf_mcu_onchip_parity_rows.csv.gz",
    "latency": DROP / "kf_mcu_latency.csv",
    "latency_raw": DROP / "kf_mcu_latency_raw.csv",
    "memory": DROP / "kf_mcu_memory.csv",
    "summary": DROP / "kf_mcu_summary.csv",
    "manifest": DROP / "kf_config_manifest.csv",
    "failures": DROP / "kf_mcu_failures.csv",
}


def method_precision(method: str) -> str:
    return (
        "FP32"
        if method.upper() in {"CC", "COULOMB_COUNT"}
        else "FP64_SOFTWARE"
    )


def model_ids() -> list[tuple[str, str, str, str]]:
    rows = []
    for chemistry in ("NMC", "LFP"):
        for fold in FOLDS:
            for slug, archive_name in METHODS[chemistry].items():
                rows.append(
                    (
                        f"{chemistry.lower()}_{fold.lower()}_{slug}",
                        chemistry,
                        fold,
                        archive_name,
                    )
                )
    return rows


def run_logged(command: list[str], path: Path, cwd: Path | None = None) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {path}")


def read_exact(port: serial.Serial, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        block = port.read(size - len(output))
        if not block:
            raise TimeoutError(f"UART timeout {len(output)}/{size}")
        output.extend(block)
    return bytes(output)


def connect_query() -> tuple[serial.Serial, tuple[int, ...]]:
    last_error: Exception | None = None
    for _ in range(6):
        try:
            port = serial.Serial(SERIAL_PORT, 921600, timeout=10)
            time.sleep(0.2)
            port.reset_input_buffer()
            port.write(b"Q")
            port.flush()
            query = struct.unpack("<10I", read_exact(port, 40))
            if query[0] != MAGIC or query[1] != 5 or query[2] != CLOCK_HZ:
                raise RuntimeError(f"bad query {query}")
            return port, query
        except Exception as error:
            last_error = error
            try:
                port.close()
            except Exception:
                pass
            time.sleep(0.5)
    raise RuntimeError(f"cannot query KF firmware: {last_error}")


def reset(
    port: serial.Serial,
    initial_soc: float,
    initial_voltage: float,
    initial_temperature: float,
    nominal_temperature: float,
) -> None:
    port.write(
        b"R"
        + struct.pack(
            "<dddd",
            initial_soc,
            initial_voltage,
            initial_temperature,
            nominal_temperature,
        )
    )
    port.flush()
    magic, status = struct.unpack("<2I", read_exact(port, 8))
    if magic != MAGIC or status != 0:
        raise RuntimeError(f"reset failed: {(magic, status)}")


def sample(
    port: serial.Serial,
    voltage: float,
    current: float,
    temperature: float,
    dt: float | None,
) -> tuple:
    if dt is None:
        port.write(b"S" + struct.pack("<ddd", voltage, current, temperature))
    else:
        port.write(
            b"D" + struct.pack("<dddd", voltage, current, temperature, dt)
        )
    port.flush()
    response = struct.unpack("<8Id", read_exact(port, 40))
    if response[0] != MAGIC:
        raise RuntimeError(f"bad response magic: {response}")
    return response


def record_path(chemistry: str, fold: str, temperature: int) -> Path:
    return (
        DATA_ROOTS[chemistry]
        / f"{temperature}C"
        / f"{chemistry}_{temperature}C_{fold}.csv"
    )


def load_record(chemistry: str, fold: str, temperature: int) -> pd.DataFrame:
    path = record_path(chemistry, fold, temperature)
    frame = pd.read_csv(path)
    time_values = frame["Test_Time(s)"].to_numpy(float)
    delta = np.diff(time_values, prepend=np.nan)
    valid = delta[np.isfinite(delta) & (delta > 0)]
    fallback = float(np.median(valid)) if len(valid) else 1.0
    dt = np.where(np.isfinite(delta) & (delta > 0), delta, fallback)
    if chemistry == "NMC":
        measured_temperature = np.full(len(frame), temperature, dtype=float)
    else:
        measured_temperature = frame["Temperature(C)"].to_numpy(float)
    return pd.DataFrame(
        {
            "end_index": np.arange(len(frame), dtype=int),
            "voltage": frame["Voltage(V)"].to_numpy(float),
            "current": frame["Current(A)"].to_numpy(float),
            "temperature": measured_temperature,
            "dt": dt,
            "soc_true": frame["SOC_CC"].to_numpy(float),
        }
    )


def load_nmc_reference(fold: str, temperature: int, method: str) -> pd.DataFrame:
    path = (
        REPO
        / "nmc/kf/results/predictions"
        / f"{fold.lower()}_{temperature}C_{method.lower()}_oracle.csv.gz"
    )
    frame = pd.read_csv(path)
    return frame.loc[
        frame.eval_mask.astype(bool),
        ["end_index", "reference_soc", "predicted_soc"],
    ].rename(
        columns={
            "reference_soc": "soc_true_archive",
            "predicted_soc": "soc_python",
        }
    )


def load_lfp_references() -> pd.DataFrame:
    frame = pd.read_csv(REPO / "runs/lfp_kf/prediction_rows.csv.gz")
    return frame[
        frame.initial_condition.eq("oracle")
    ][
        [
            "method",
            "profile",
            "temperature_C",
            "end_index",
            "soc_true",
            "soc_pred",
        ]
    ].rename(columns={"soc_true": "soc_true_archive", "soc_pred": "soc_python"})


def reference_rows(
    chemistry: str,
    fold: str,
    temperature: int,
    method: str,
    lfp: pd.DataFrame,
) -> pd.DataFrame:
    if chemistry == "NMC":
        return load_nmc_reference(fold, temperature, method)
    return lfp[
        (lfp.method == method)
        & (lfp.profile == fold)
        & np.isclose(lfp.temperature_C, temperature)
    ][["end_index", "soc_true_archive", "soc_python"]].copy()


def parse_size(path: Path) -> tuple[int, int, int]:
    output = subprocess.check_output([ARM_SIZE, str(path)], text=True)
    match = re.search(r"\n\s*(\d+)\s+(\d+)\s+(\d+)\s+", output)
    if not match:
        raise RuntimeError(output)
    return tuple(int(value) for value in match.groups())


def write_outputs(
    parity: list[dict],
    detail: list[pd.DataFrame],
    latency: list[dict],
    raw_latency: list[dict],
    memory: list[dict],
    manifest: list[dict],
    failures: list[dict],
    write_detail: bool = True,
) -> None:
    pd.DataFrame(parity).to_csv(OUTPUTS["parity"], index=False)
    if detail and write_detail:
        pd.concat(detail, ignore_index=True).to_csv(
            OUTPUTS["rows"], index=False, compression="gzip"
        )
    pd.DataFrame(latency).to_csv(OUTPUTS["latency"], index=False)
    pd.DataFrame(raw_latency).to_csv(OUTPUTS["latency_raw"], index=False)
    pd.DataFrame(memory).to_csv(OUTPUTS["memory"], index=False)
    pd.DataFrame(manifest).to_csv(OUTPUTS["manifest"], index=False)
    pd.DataFrame(
        failures,
        columns=[
            "model_id",
            "chemistry",
            "fold",
            "method",
            "temperature_C",
            "stage",
            "reason",
        ],
    ).to_csv(OUTPUTS["failures"], index=False)


def summarize(parity: pd.DataFrame, latency: pd.DataFrame, memory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (chemistry, method), group in parity.groupby(["chemistry", "method"]):
        latency_group = latency[
            (latency.chemistry == chemistry) & (latency.method == method)
        ]
        memory_group = memory[
            (memory.chemistry == chemistry) & (memory.method == method)
        ]
        rows.append(
            {
                "chemistry": chemistry,
                "method": method,
                "precision": method_precision(method),
                "initial_soc": "oracle",
                "online_sensor_signals": "V/I/T",
                "dt_source": "1Hz scheduler for latency; recorded dt for archived parity",
                "n_fold_temperature_slices": len(group),
                "n_successful_slices": int(group.onchip_status.eq("PASS").sum()),
                "mcu_slice_mean_mae_pct": float(group.mcu_mae_pct.mean()),
                "python_slice_mean_mae_pct": float(group.python_mae_pct.mean()),
                "mae_delta_mcu_minus_python_pct": float(
                    group.mcu_mae_pct.mean() - group.python_mae_pct.mean()
                ),
                "mean_abs_mcu_python_diff_pct_point": float(
                    group.mean_abs_mcu_python_diff_pct_point.mean()
                ),
                "worst_max_abs_mcu_python_diff_pct_point": float(
                    group.max_abs_mcu_python_diff_pct_point.max()
                ),
                "median_latency_us": float(latency_group.median_us.median()),
                "p95_latency_us": float(latency_group.p95_us.median()),
                "max_flash_bytes": int(memory_group.flash_bytes.max()),
                "max_static_ram_bytes": int(memory_group.static_ram_bytes.max()),
                "max_stack_highwater_bytes": int(
                    memory_group.stack_highwater_bytes.max()
                ),
                "max_runtime_state_bytes": int(
                    memory_group.runtime_state_bytes.max()
                ),
                "max_stored_asset_bytes": int(memory_group.asset_bytes.max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["chemistry", "mcu_slice_mean_mae_pct"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-at")
    parser.add_argument("--stop-after")
    args = parser.parse_args()

    parity_rows: list[dict] = []
    detail_frames: list[pd.DataFrame] = []
    latency_rows: list[dict] = []
    raw_latency_rows: list[dict] = []
    memory_rows: list[dict] = []
    manifest_rows: list[dict] = []
    failures: list[dict] = []
    lfp_reference = load_lfp_references()
    configs = model_ids()
    started = args.start_at is None
    overall_started = time.time()

    for config_index, (model_id, chemistry, fold, method) in enumerate(
        configs, start=1
    ):
        if not started:
            started = model_id == args.start_at
            if not started:
                continue
        print(
            f"[KF] {config_index:02d}/{len(configs)} {model_id}: build",
            flush=True,
        )
        build_log = DROP / "kf_mcu_logs" / f"{model_id}_build.log"
        flash_log = DROP / "kf_mcu_logs" / f"{model_id}_flash.log"
        try:
            run_logged(
                [str(FIRMWARE / "build_model.sh"), model_id],
                build_log,
                cwd=FIRMWARE,
            )
            build = FIRMWARE / "Build" / model_id
            elf = build / "cema_h563_kf_bench.elf"
            text_bytes, data_bytes, bss_bytes = parse_size(elf)
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
            (
                _,
                protocol,
                clock_hz,
                sensor_fields,
                sensor_bytes,
                chemistry_code,
                method_code,
                state_dim,
                asset_bytes,
                runtime_state_bytes,
            ) = query
            if (
                protocol != 5
                or clock_hz != CLOCK_HZ
                or sensor_fields != 3
                or sensor_bytes != 24
            ):
                raise RuntimeError(f"unexpected query: {query}")

            first_temperature = TEMPERATURES[chemistry][0]
            first_record = load_record(
                chemistry, fold, first_temperature
            )
            reset(
                port,
                float(first_record.soc_true.iloc[0]),
                float(first_record.voltage.iloc[0]),
                float(first_record.temperature.iloc[0]),
                float(first_temperature),
            )
            latency_cycles = []
            stack_max = 0
            for index in range(WARMUP + N_REPS):
                source_index = index % len(first_record)
                row = first_record.iloc[source_index]
                response = sample(
                    port,
                    float(row.voltage),
                    float(row.current),
                    float(row.temperature),
                    None,
                )
                if response[1] != 0:
                    raise RuntimeError(
                        f"fixed-dt latency run failed at {index}: {response}"
                    )
                stack_max = max(stack_max, int(response[3]))
                if index >= WARMUP:
                    latency_cycles.append(int(response[2]))
            cycles = np.asarray(latency_cycles, dtype=np.uint32)
            latency_rows.append(
                {
                    "model_id": model_id,
                    "chemistry": chemistry,
                    "fold": fold,
                    "method": method,
                    "precision": method_precision(method),
                    "clock_hz": CLOCK_HZ,
                    "n_repetitions": N_REPS,
                    "warmup_steps": WARMUP,
                    "input_path": "raw_V_I_T_fixed_1Hz_scheduler",
                    "median_cycles": float(np.median(cycles)),
                    "p95_cycles": float(np.percentile(cycles, 95)),
                    "min_cycles": int(cycles.min()),
                    "max_cycles": int(cycles.max()),
                    "median_us": float(np.median(cycles) / 250.0),
                    "p95_us": float(np.percentile(cycles, 95) / 250.0),
                    "min_us": float(cycles.min() / 250.0),
                    "max_us": float(cycles.max() / 250.0),
                }
            )
            raw_latency_rows.extend(
                {
                    "model_id": model_id,
                    "chemistry": chemistry,
                    "fold": fold,
                    "method": method,
                    "repetition": repetition,
                    "cycles": int(value),
                    "latency_us": float(value / 250.0),
                }
                for repetition, value in enumerate(cycles)
            )

            for temperature in TEMPERATURES[chemistry]:
                print(
                    f"[KF] {model_id} {temperature:g}C: replay",
                    flush=True,
                )
                record = load_record(chemistry, fold, temperature)
                reference = reference_rows(
                    chemistry, fold, temperature, method, lfp_reference
                )
                reset(
                    port,
                    float(record.soc_true.iloc[0]),
                    float(record.voltage.iloc[0]),
                    float(record.temperature.iloc[0]),
                    float(temperature),
                )
                output = np.empty(len(record), dtype=np.float32)
                status = 0
                for index, row in enumerate(record.itertuples(index=False)):
                    response = sample(
                        port,
                        float(row.voltage),
                        float(row.current),
                        float(row.temperature),
                        float(row.dt),
                    )
                    output[index] = response[-1]
                    stack_max = max(stack_max, int(response[3]))
                    status = int(response[1])
                    if status:
                        output[index + 1 :] = np.nan
                        failures.append(
                            {
                                "model_id": model_id,
                                "chemistry": chemistry,
                                "fold": fold,
                                "method": method,
                                "temperature_C": temperature,
                                "stage": "onchip_replay",
                                "reason": f"MCU status {status} at index {index}",
                            }
                        )
                        break
                aligned = reference.merge(
                    record[["end_index", "soc_true"]], on="end_index", how="inner"
                )
                aligned["soc_mcu"] = output[
                    aligned.end_index.to_numpy(dtype=int)
                ].astype(float)
                aligned["chemistry"] = chemistry
                aligned["model_id"] = model_id
                aligned["fold"] = fold
                aligned["method"] = method
                aligned["temperature_C"] = temperature
                aligned["mcu_minus_python_pct_point"] = (
                    aligned.soc_mcu - aligned.soc_python
                ) * 100.0
                aligned["mcu_abs_error_pct"] = (
                    aligned.soc_mcu - aligned.soc_true
                ).abs() * 100.0
                aligned["python_abs_error_pct"] = (
                    aligned.soc_python - aligned.soc_true_archive
                ).abs() * 100.0
                finite = np.isfinite(aligned.soc_mcu) & np.isfinite(
                    aligned.soc_python
                )
                valid = aligned.loc[finite]
                if valid.empty:
                    parity_rows.append(
                        {
                            "model_id": model_id,
                            "chemistry": chemistry,
                            "fold": fold,
                            "method": method,
                            "temperature_C": temperature,
                            "onchip_status": "FAIL",
                            "n_evaluation_points": 0,
                            "mcu_mae_pct": np.nan,
                            "python_mae_pct": np.nan,
                            "mae_delta_mcu_minus_python_pct": np.nan,
                            "mean_abs_mcu_python_diff_pct_point": np.nan,
                            "p95_abs_mcu_python_diff_pct_point": np.nan,
                            "max_abs_mcu_python_diff_pct_point": np.nan,
                        }
                    )
                else:
                    difference = valid.mcu_minus_python_pct_point.abs()
                    mcu_mae = float(valid.mcu_abs_error_pct.mean())
                    python_mae = float(valid.python_abs_error_pct.mean())
                    parity_rows.append(
                        {
                            "model_id": model_id,
                            "chemistry": chemistry,
                            "fold": fold,
                            "method": method,
                            "temperature_C": temperature,
                            "onchip_status": (
                                "PASS"
                                if status == 0 and len(valid) == len(aligned)
                                else "FAIL"
                            ),
                            "n_evaluation_points": len(valid),
                            "mcu_mae_pct": mcu_mae,
                            "python_mae_pct": python_mae,
                            "mae_delta_mcu_minus_python_pct": mcu_mae
                            - python_mae,
                            "mean_abs_mcu_python_diff_pct_point": float(
                                difference.mean()
                            ),
                            "p95_abs_mcu_python_diff_pct_point": float(
                                difference.quantile(0.95)
                            ),
                            "max_abs_mcu_python_diff_pct_point": float(
                                difference.max()
                            ),
                        }
                    )
                detail_frames.append(
                    aligned[
                        [
                            "chemistry",
                            "model_id",
                            "fold",
                            "method",
                            "temperature_C",
                            "end_index",
                            "soc_true",
                            "soc_true_archive",
                            "soc_python",
                            "soc_mcu",
                            "mcu_minus_python_pct_point",
                            "mcu_abs_error_pct",
                            "python_abs_error_pct",
                        ]
                    ]
                )
            port.close()
            memory_rows.append(
                {
                    "model_id": model_id,
                    "chemistry": chemistry,
                    "fold": fold,
                    "method": method,
                    "precision": method_precision(method),
                    "state_dim": state_dim,
                    "text_bytes": text_bytes,
                    "data_bytes": data_bytes,
                    "bss_bytes": bss_bytes,
                    "flash_bytes": text_bytes + data_bytes,
                    "static_ram_bytes": data_bytes + bss_bytes,
                    "stack_highwater_bytes": stack_max,
                    "runtime_state_bytes": runtime_state_bytes,
                    "asset_bytes": asset_bytes,
                }
            )
            manifest_rows.append(
                {
                    "model_id": model_id,
                    "chemistry": chemistry,
                    "fold": fold,
                    "method": method,
                    "precision": method_precision(method),
                    "method_code": method_code,
                    "chemistry_code": chemistry_code,
                    "state_dim": state_dim,
                    "temperatures_C": ";".join(
                        map(str, TEMPERATURES[chemistry])
                    ),
                    "input_signals": "raw V/I/T",
                    "initial_soc_condition": "oracle at reset",
                    "dt_policy": "1.0s latency; recorded per-step dt parity",
                    "build_status": "PASS",
                    "flash_status": "PASS",
                }
            )
        except Exception as error:
            failures.append(
                {
                    "model_id": model_id,
                    "chemistry": chemistry,
                    "fold": fold,
                    "method": method,
                    "temperature_C": np.nan,
                    "stage": "build_flash_or_protocol",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            print(f"[KF] {model_id}: FAILED {error}", flush=True)
            try:
                port.close()
            except Exception:
                pass
        write_outputs(
            parity_rows,
            detail_frames,
            latency_rows,
            raw_latency_rows,
            memory_rows,
            manifest_rows,
            failures,
            write_detail=(
                config_index % 5 == 0
                or config_index == len(configs)
                or model_id == args.stop_after
            ),
        )
        if model_id == args.stop_after:
            break
        elapsed = time.time() - overall_started
        completed = config_index
        eta = elapsed / completed * (len(configs) - completed)
        print(
            f"[KF] completed {completed}/{len(configs)}; ETA {eta/60:.1f} min",
            flush=True,
        )

    parity = pd.DataFrame(parity_rows)
    latency = pd.DataFrame(latency_rows)
    memory = pd.DataFrame(memory_rows)
    if not parity.empty and not latency.empty and not memory.empty:
        summary = summarize(parity, latency, memory)
        summary.to_csv(OUTPUTS["summary"], index=False)
        print(summary.to_string(index=False), flush=True)
    print(
        f"[KF] finished in {(time.time()-overall_started)/60:.1f} min; "
        f"failures={len(failures)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
