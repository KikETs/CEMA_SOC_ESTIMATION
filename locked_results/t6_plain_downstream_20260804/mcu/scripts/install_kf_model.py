#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter


DROP = Path(__file__).resolve().parents[1]
REPO = DROP.parents[1]
FIRMWARE = DROP / "firmware/h563_kf"
GENERATED = FIRMWARE / "Generated"

CHEMISTRY = {"nmc": 1, "lfp": 2}
METHOD_CODE = {
    "cc": 0,
    "1rc_ekf": 1,
    "2rc_ekf": 2,
    "adaptive_2rc_ekf": 3,
    "2rc_ukf": 4,
    "coulomb_count": 0,
    "plain_2rc_ekf": 2,
    "hysteresis_2rc_ekf": 5,
    "adaptive_hysteresis_2rc_ekf": 3,
    "hysteresis_2rc_ukf": 4,
}
FOLDS = ("dst", "fuds", "us06")
NMC_METHODS = ("cc", "1rc_ekf", "2rc_ekf", "adaptive_2rc_ekf", "2rc_ukf")
LFP_METHODS = (
    "coulomb_count",
    "plain_2rc_ekf",
    "hysteresis_2rc_ekf",
    "adaptive_hysteresis_2rc_ekf",
    "hysteresis_2rc_ukf",
)


def c_float(value: float) -> str:
    return float(np.float32(value)).hex() + "F"


def c_number(value: float, fp64: bool) -> str:
    return float(value).hex() if fp64 else c_float(value)


def c_array(name: str, values: np.ndarray, fp64: bool = False) -> str:
    array = np.asarray(values, dtype=np.float64 if fp64 else np.float32)
    c_type = "double" if fp64 else "float"
    if array.ndim == 1:
        body = ", ".join(c_number(value, fp64) for value in array)
        return f"static const {c_type} {name}[CEMA_TEMP_COUNT] = {{{body}}};\n"
    if array.ndim != 2:
        raise ValueError(f"{name}: expected 1D/2D, got {array.shape}")
    rows = []
    for row in array:
        rows.append(
            "  {" + ", ".join(c_number(value, fp64) for value in row) + "}"
        )
    return (
        f"static const {c_type} {name}[CEMA_TEMP_COUNT][CEMA_SOC_COUNT] = {{\n"
        + ",\n".join(rows)
        + "\n};\n"
    )


def c_nd_array(name: str, values: np.ndarray, fp64: bool = False) -> str:
    array = np.asarray(values, dtype=np.float64 if fp64 else np.float32)
    c_type = "double" if fp64 else "float"

    def initializer(value: np.ndarray, indent: int = 0) -> str:
        if value.ndim == 1:
            return (
                "{"
                + ", ".join(c_number(item, fp64) for item in value)
                + "}"
            )
        prefix = " " * (indent + 2)
        return (
            "{\n"
            + ",\n".join(
                prefix + initializer(item, indent + 2) for item in value
            )
            + "\n"
            + " " * indent
            + "}"
        )

    dimensions = "".join(f"[{size}]" for size in array.shape)
    return f"static const {c_type} {name}{dimensions} = {initializer(array)};\n"


def nmc_assets(fold: str, method: str) -> dict:
    root = REPO / "nmc/kf"
    temperatures = np.array([0.0, 25.0, 45.0], dtype=float)
    curves = pd.read_csv(root / "results/ocv_curves.csv")
    knots = []
    coefficients = []
    for temperature in temperatures:
        frame = curves[np.isclose(curves.temperature_C, temperature)].sort_values("soc")
        pchip = PchipInterpolator(frame.soc.to_numpy(), frame.ocv_V.to_numpy())
        knots.append(pchip.x)
        coefficients.append(pchip.c)
    sources = pd.read_csv(root / "results/ocv_sources.csv")
    qref = np.array(
        [
            float(
                sources.loc[
                    np.isclose(sources.temperature_C, temperature), "q_ref_Ah"
                ].iloc[0]
            )
            for temperature in temperatures
        ]
    )

    output = {
        "temperatures": temperatures,
        "qref": qref,
        "soc_grid": np.asarray(knots[0]),
        "pchip_knots": np.asarray(knots),
        "pchip_coefficients": np.asarray(coefficients),
        "order": 0 if method == "cc" else (1 if method == "1rc_ekf" else 2),
        "with_hysteresis": False,
        "adaptive": method == "adaptive_2rc_ekf",
        "ukf": method == "2rc_ukf",
        "slope_min": 0.001,
        "slope_max": 8.0,
        "innovation_floor": 1.0e-10,
        "covariance_floor": 1.0e-12,
        "voltage_state_limit": 1.0,
        "adaptive_alpha": 0.01,
        "adaptive_r_min": 1.0e-7,
        "adaptive_r_max": 1.0e-3,
        "adaptive_r_min_scale": 0.25,
        "adaptive_r_max_scale": 25.0,
        "slope_sref": 0.2,
        "slope_smin": 0.005,
        "slope_factor_min": 1.0,
        "slope_factor_max": 400.0,
    }
    if method == "cc":
        return output

    params = pd.read_csv(root / "locked_parameters/ecm_parameters.csv")
    order = output["order"]
    selected_params = params[
        (params.fold.str.upper() == fold.upper()) & (params.order == order)
    ].sort_values("temperature_C")
    if list(selected_params.temperature_C.astype(float)) != list(temperatures):
        raise RuntimeError(f"NMC ECM rows incomplete for {fold}/{method}")
    output.update(
        {
            "r0": selected_params.R0_ohm.to_numpy(float),
            "r1": selected_params.R1_ohm.to_numpy(float),
            "r2": selected_params.R2_ohm.to_numpy(float),
            "tau1": selected_params.tau1_s.to_numpy(float),
            "tau2": selected_params.tau2_s.to_numpy(float),
            "gamma": np.zeros(len(temperatures)),
        }
    )

    config = yaml.safe_load((root / "configs/ecm.yaml").read_text())
    candidate_by_name = {
        row["name"]: row for row in config["filter"]["noise_candidates"]
    }
    selection = pd.read_csv(root / "locked_parameters/filter_noise_selection.csv")
    noise_method = "2RC_EKF" if method == "adaptive_2rc_ekf" else {
        "1rc_ekf": "1RC_EKF",
        "2rc_ekf": "2RC_EKF",
        "2rc_ukf": "2RC_UKF",
    }[method]
    q_soc = []
    q_vp = []
    r_voltage = []
    beta = []
    for temperature in temperatures:
        row = selection[
            (selection.fold.str.upper() == fold.upper())
            & np.isclose(selection.temperature_C, temperature)
            & (selection.method == noise_method)
            & selection.selected
            & (selection.validation_profile == "CV_MEAN")
        ]
        if len(row) != 1:
            raise RuntimeError(
                f"NMC noise selection incomplete for {fold}/{temperature}/{noise_method}"
            )
        candidate = candidate_by_name[str(row.iloc[0].noise_name)]
        q_soc.append(candidate["q_soc"])
        q_vp.append(candidate["q_vp"])
        r_voltage.append(candidate["r_voltage"])
        if method == "adaptive_2rc_ekf":
            adaptive = selection[
                (selection.fold.str.upper() == fold.upper())
                & np.isclose(selection.temperature_C, temperature)
                & (selection.method == "Adaptive_2RC_EKF")
                & selection.selected
                & (selection.validation_profile == "CV_MEAN")
            ]
            if len(adaptive) != 1:
                raise RuntimeError(
                    f"NMC adaptive beta incomplete for {fold}/{temperature}"
                )
            beta.append(float(adaptive.iloc[0].adaptive_beta))
        else:
            beta.append(0.0)
    output.update(
        {
            "q_soc": np.asarray(q_soc),
            "q_vp": np.asarray(q_vp),
            "q_h": np.zeros(len(temperatures)),
            "r_voltage": np.asarray(r_voltage),
            "adaptive_beta": np.asarray(beta),
        }
    )
    return output


def lfp_assets(fold: str, method: str) -> dict:
    root = REPO / "lfp/kf"
    table = pd.read_csv(root / "locked_parameters/artifacts/ocv_table.csv")
    temperatures = np.sort(table.temperature_C.unique().astype(float))
    first = table[np.isclose(table.temperature_C, temperatures[0])].sort_values("soc")
    soc_grid = first.soc.to_numpy(float)
    base = []
    hmag = []
    raw_mid = []
    dbase = []
    dhmag = []
    discharge_slope = []
    for temperature in temperatures:
        frame = table[np.isclose(table.temperature_C, temperature)].sort_values("soc")
        if not np.allclose(frame.soc.to_numpy(float), soc_grid, rtol=0, atol=1e-12):
            raise RuntimeError(f"LFP SOC grid mismatch at {temperature:g}C")
        row_base = frame.ocv_base_monotonic_V.to_numpy(float)
        row_hmag = frame.hysteresis_half_monotonic_V.to_numpy(float)
        discharge = frame.ocv_discharge_raw_V.to_numpy(float)
        charge = frame.ocv_charge_raw_V.to_numpy(float)
        base.append(row_base)
        hmag.append(row_hmag)
        raw_mid.append(0.5 * (discharge + charge))
        dbase.append(np.gradient(row_base, soc_grid))
        dhmag.append(np.gradient(row_hmag, soc_grid))
        discharge_slope.append(
            np.gradient(
                savgol_filter(
                    discharge, window_length=11, polyorder=3, mode="interp"
                ),
                soc_grid,
            )
        )
    data_root = REPO / "Data/Preprocessed/LFP/prepared_data_ocv_discharge_soc"
    qref = []
    for temperature in temperatures:
        paths = sorted((data_root / f"{int(temperature)}C").glob("LFP_*C_*.csv"))
        values = []
        for path in paths:
            frame = pd.read_csv(
                path, usecols=["Q_ref_lc_ocv_discharge_Ah"]
            )
            values.append(float(frame.iloc[:, 0].median()))
        if not values or not np.allclose(values, values[0], rtol=0, atol=1e-10):
            raise RuntimeError(f"LFP Qref mismatch at {temperature:g}C")
        qref.append(values[0])
    output = {
        "temperatures": temperatures,
        "qref": np.asarray(qref),
        "soc_grid": soc_grid,
        "base": np.asarray(base),
        "hmag": np.asarray(hmag),
        "raw_mid": np.asarray(raw_mid),
        "dbase": np.asarray(dbase),
        "dhmag": np.asarray(dhmag),
        "discharge_slope": np.asarray(discharge_slope),
        "order": 0 if method == "coulomb_count" else 2,
        "with_hysteresis": method
        in {
            "hysteresis_2rc_ekf",
            "adaptive_hysteresis_2rc_ekf",
            "hysteresis_2rc_ukf",
        },
        "adaptive": method == "adaptive_hysteresis_2rc_ekf",
        "ukf": method == "hysteresis_2rc_ukf",
        "slope_min": -10.0,
        "slope_max": 10.0,
        "innovation_floor": (
            1.0e-12 if method == "hysteresis_2rc_ukf" else 1.0e-16
        ),
        "covariance_floor": 1.0e-12,
        "voltage_state_limit": 1.0,
        "adaptive_alpha": 0.01,
        "adaptive_r_min": 1.0e-7,
        "adaptive_r_max": 1.0e-3,
        "adaptive_r_min_scale": 0.25,
        "adaptive_r_max_scale": 25.0,
        "slope_sref": 0.2,
        "slope_smin": 0.005,
        "slope_factor_min": 1.0,
        "slope_factor_max": 400.0,
    }
    if method == "coulomb_count":
        return output

    ecm = pd.read_csv(root / "locked_parameters/v2_1/ecm_fit_quality.csv")
    unique = (
        ecm.groupby(["fold_holdout", "temperature_C"], as_index=False)
        .first()
        .sort_values(["fold_holdout", "temperature_C"])
    )
    selected_params = unique[
        unique.fold_holdout.str.upper() == fold.upper()
    ].sort_values("temperature_C")
    if list(selected_params.temperature_C.astype(float)) != list(temperatures):
        raise RuntimeError(f"LFP ECM rows incomplete for {fold}/{method}")
    output.update(
        {
            "r0": selected_params.R0_ohm.to_numpy(float),
            "r1": selected_params.R1_ohm.to_numpy(float),
            "r2": selected_params.R2_ohm.to_numpy(float),
            "tau1": selected_params.tau1_s.to_numpy(float),
            "tau2": selected_params.tau2_s.to_numpy(float),
            "gamma": selected_params.gamma.to_numpy(float),
        }
    )
    qr = pd.read_csv(root / "locked_parameters/v2_1/qr_selection.csv")
    selected = (
        qr.sort_values("range_cycle")
        .groupby("fold_holdout", as_index=False)
        .tail(1)
    )
    row = selected[selected.fold_holdout.str.upper() == fold.upper()]
    if len(row) != 1:
        raise RuntimeError(f"LFP Q/R row incomplete for {fold}")
    row = row.iloc[0]
    output.update(
        {
            key: np.full(len(temperatures), float(row[key]))
            for key in ("q_soc", "q_vp", "q_h", "r_voltage")
        }
    )
    output["adaptive_beta"] = np.zeros(len(temperatures))
    return output


def parse_model_id(model_id: str) -> tuple[str, str, str]:
    match = re.fullmatch(r"(nmc|lfp)_(dst|fuds|us06)_(.+)", model_id)
    if not match:
        raise SystemExit(f"Invalid KF model_id: {model_id}")
    chemistry, fold, method = match.groups()
    allowed = NMC_METHODS if chemistry == "nmc" else LFP_METHODS
    if method not in allowed:
        raise SystemExit(
            f"Invalid method for {chemistry}: {method}; expected {allowed}"
        )
    return chemistry, fold, method


def asset_bytes(chemistry: str, method: str, assets: dict) -> int:
    bytes_per_value = 4 if METHOD_CODE[method] == 0 else 8
    count = 2 * len(assets["temperatures"])
    if METHOD_CODE[method] == 0:
        return count * bytes_per_value
    count += 11 * len(assets["temperatures"])
    if chemistry == "nmc":
        count += assets["pchip_knots"].size
        count += assets["pchip_coefficients"].size
    else:
        count += 2 * assets["base"].size
        count += assets["discharge_slope"].size
        if assets["with_hysteresis"]:
            count += (
                assets["hmag"].size
                + assets["raw_mid"].size
                + assets["dhmag"].size
            )
    return int(count * bytes_per_value)


def render(model_id: str, chemistry: str, fold: str, method: str, assets: dict) -> str:
    temperature_count = len(assets["temperatures"])
    soc_count = len(assets["soc_grid"])
    method_code = METHOD_CODE[method]
    fp64 = method_code != 0
    state_dim = 1 if method_code == 0 else 1 + assets["order"] + int(assets["with_hysteresis"])
    lines = [
        "#ifndef CEMA_KF_CONFIG_H",
        "#define CEMA_KF_CONFIG_H",
        "",
        "#define CEMA_CHEMISTRY_NMC 1U",
        "#define CEMA_CHEMISTRY_LFP 2U",
        "#define CEMA_METHOD_CC 0U",
        f'#define CEMA_MODEL_ID "{model_id}"',
        f"#define CEMA_CHEMISTRY {CHEMISTRY[chemistry]}U",
        f"#define CEMA_METHOD {method_code}U",
        f"#define CEMA_ORDER {assets['order']}U",
        f"#define CEMA_STATE_DIM {state_dim}U",
        f"#define CEMA_WITH_HYSTERESIS {int(assets['with_hysteresis'])}",
        f"#define CEMA_ADAPTIVE {int(assets['adaptive'])}",
        f"#define CEMA_UKF {int(assets['ukf'])}",
        f"#define CEMA_INTERNAL_FP64 {int(fp64)}",
        f"#define CEMA_TEMP_COUNT {temperature_count}U",
        f"#define CEMA_SOC_COUNT {soc_count}U",
        f"#define CEMA_PCHIP_SEGMENT_COUNT {soc_count - 1}U",
        f"#define CEMA_ASSET_BYTES {asset_bytes(chemistry, method, assets)}U",
        f"#define CEMA_SLOPE_MIN {c_number(assets['slope_min'], fp64)}",
        f"#define CEMA_SLOPE_MAX {c_number(assets['slope_max'], fp64)}",
        f"#define CEMA_INNOVATION_FLOOR {c_number(assets['innovation_floor'], fp64)}",
        f"#define CEMA_COVARIANCE_FLOOR {c_number(assets['covariance_floor'], fp64)}",
        f"#define CEMA_VOLTAGE_STATE_LIMIT {c_number(assets['voltage_state_limit'], fp64)}",
        f"#define CEMA_ADAPTIVE_ALPHA {c_number(assets['adaptive_alpha'], fp64)}",
        f"#define CEMA_ADAPTIVE_R_MIN {c_number(assets['adaptive_r_min'], fp64)}",
        f"#define CEMA_ADAPTIVE_R_MAX {c_number(assets['adaptive_r_max'], fp64)}",
        f"#define CEMA_ADAPTIVE_R_MIN_SCALE {c_number(assets['adaptive_r_min_scale'], fp64)}",
        f"#define CEMA_ADAPTIVE_R_MAX_SCALE {c_number(assets['adaptive_r_max_scale'], fp64)}",
        f"#define CEMA_SLOPE_SREF {c_number(assets['slope_sref'], fp64)}",
        f"#define CEMA_SLOPE_SMIN {c_number(assets['slope_smin'], fp64)}",
        f"#define CEMA_SLOPE_FACTOR_MIN {c_number(assets['slope_factor_min'], fp64)}",
        f"#define CEMA_SLOPE_FACTOR_MAX {c_number(assets['slope_factor_max'], fp64)}",
        "",
        c_array("CEMA_TEMPERATURES", assets["temperatures"], fp64),
        c_array("CEMA_QREF", assets["qref"], fp64),
    ]
    if method_code != 0:
        for name, key in (
            ("CEMA_R0", "r0"),
            ("CEMA_R1", "r1"),
            ("CEMA_R2", "r2"),
            ("CEMA_TAU1", "tau1"),
            ("CEMA_TAU2", "tau2"),
            ("CEMA_GAMMA", "gamma"),
            ("CEMA_Q_SOC", "q_soc"),
            ("CEMA_Q_VP", "q_vp"),
            ("CEMA_Q_H", "q_h"),
            ("CEMA_R_VOLTAGE", "r_voltage"),
            ("CEMA_ADAPTIVE_BETA", "adaptive_beta"),
        ):
            lines.append(c_array(name, assets[key], fp64))
        if chemistry == "nmc":
            lines.append(
                c_nd_array(
                    "CEMA_OCV_PCHIP_KNOTS", assets["pchip_knots"], fp64
                )
            )
            lines.append(
                c_nd_array(
                    "CEMA_OCV_PCHIP_COEFFICIENTS",
                    assets["pchip_coefficients"],
                    fp64,
                )
            )
        else:
            lines.append(c_array("CEMA_OCV_BASE", assets["base"], fp64))
            lines.append(c_array("CEMA_OCV_DBASE", assets["dbase"], fp64))
            lines.append(
                c_array(
                    "CEMA_OCV_DISCHARGE_SLOPE",
                    assets["discharge_slope"],
                    fp64,
                )
            )
            if assets["with_hysteresis"]:
                lines.append(c_array("CEMA_OCV_HMAG", assets["hmag"], fp64))
                lines.append(c_array("CEMA_OCV_DHMAG", assets["dhmag"], fp64))
                lines.append(c_array("CEMA_OCV_RAW_MID", assets["raw_mid"], fp64))
    lines.extend(["", "#endif", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    args = parser.parse_args()
    chemistry, fold, method = parse_model_id(args.model_id)
    assets = (
        nmc_assets(fold, method)
        if chemistry == "nmc"
        else lfp_assets(fold, method)
    )
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / "kf_config.h").write_text(
        render(args.model_id, chemistry, fold, method, assets),
        encoding="ascii",
    )
    print(args.model_id)


if __name__ == "__main__":
    main()
