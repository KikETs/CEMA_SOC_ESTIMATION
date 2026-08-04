#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


DROP = Path(__file__).resolve().parents[1]
FIRMWARE = DROP / "firmware" / "h563_raw_vit"
REPO = DROP.parents[1]
PACKAGE_ROOTS = {
    "NMC": REPO / "nmc/kf/inference_pkg_nmc",
    "LFP": REPO / "lfp/deep_learning/inference_pkg_lfp",
}
CHANNEL_IDS = {
    "V_corr_raw": 0,
    "I_raw": 1,
    "T": 2,
    "V_corr_raw_ema50": 3,
    "V_corr_raw_dev_ema50": 4,
    "V_corr_raw_ema200": 5,
    "V_corr_raw_dev_ema200": 6,
    "V_corr_raw_ema800": 7,
    "V_corr_raw_dev_ema800": 8,
    "I_raw_ema50": 9,
    "I_raw_dev_ema50": 10,
    "I_raw_ema200": 11,
    "I_raw_dev_ema200": 12,
    "absI_ema50": 13,
    "absI_dev_ema50": 14,
    "absI_ema200": 15,
    "absI_dev_ema200": 16,
}


def c_float(value: float) -> str:
    return float(np.float32(value)).hex() + "F"


def c_double(value: float) -> str:
    return float(value).hex()


def array(values, formatter) -> str:
    return ", ".join(formatter(value) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    args = parser.parse_args()

    manifest = pd.read_csv(DROP / "stedgeai_generate_manifest.csv")
    matches = manifest.loc[manifest["model_id"] == args.model_id]
    if len(matches) != 1:
        raise SystemExit(f"Unknown model_id: {args.model_id}")
    row = matches.iloc[0]
    package = (
        PACKAGE_ROOTS[row["chemistry"]]
        / row["feature"]
        / row["fold"]
        / str(int(row["seed"]))
    )
    config = json.loads((package / "config.json").read_text(encoding="utf-8"))
    scaler = json.loads((package / "scaler.json").read_text(encoding="utf-8"))
    r0_table = json.loads((package / "r0_table.json").read_text(encoding="utf-8"))
    channels = list(config["channels"])
    if channels != list(scaler["columns"]):
        raise SystemExit(f"Scaler channel mismatch for {args.model_id}")
    if int(config["window"]) != 50:
        raise SystemExit(f"Unexpected window for {args.model_id}")

    source = DROP / "stedgeai" / "models" / args.model_id / "output"
    destination = FIRMWARE / "AI" / "Generated"
    required = ("cema.c", "cema.h", "cema_data.c", "cema_data.h", "cema_details.h")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise SystemExit(f"Missing ST Edge AI files for {args.model_id}: {missing}")
    destination.mkdir(parents=True, exist_ok=True)
    for old in destination.glob("cema*"):
        old.unlink()
    for name in required:
        shutil.copy2(source / name, destination / name)
    (destination / "model_config.h").write_text(
        "#ifndef CEMA_MODEL_CONFIG_H\n"
        "#define CEMA_MODEL_CONFIG_H\n"
        f'#define CEMA_MODEL_ID "{args.model_id}"\n'
        "#endif\n",
        encoding="ascii",
    )

    r0_rows = sorted(
        r0_table["table"], key=lambda item: float(item["temperature_C"])
    )
    temperatures = [float(item["temperature_C"]) for item in r0_rows]
    resistances = [float(item["r0_ohm"]) for item in r0_rows]
    means = [float(value) for value in scaler["mean"]]
    stds = [float(value) for value in scaler["std"]]
    ids = [CHANNEL_IDS[channel] for channel in channels]
    alpha = {
        50: float(np.exp(-1.0 / 50.0)),
        120: float(np.exp(-1.0 / 120.0)),
        200: float(np.exp(-1.0 / 200.0)),
        800: float(np.exp(-1.0 / 800.0)),
    }
    header = f"""#ifndef CEMA_PREPROCESS_CONFIG_H
#define CEMA_PREPROCESS_CONFIG_H

#include <stdint.h>

#define CEMA_ALL_FEATURE_COUNT 17U
#define CEMA_CHANNEL_COUNT {len(channels)}U
#define CEMA_WINDOW_SIZE 50U
#define CEMA_R0_COUNT {len(r0_rows)}U

static const uint8_t CEMA_CHANNEL_IDS[CEMA_CHANNEL_COUNT] = {{{array(ids, str)}}};
static const float CEMA_SCALER_MEAN[CEMA_CHANNEL_COUNT] = {{{array(means, c_float)}}};
static const float CEMA_SCALER_STD[CEMA_CHANNEL_COUNT] = {{{array(stds, c_float)}}};
static const double CEMA_R0_TEMPERATURES[CEMA_R0_COUNT] = {{{array(temperatures, c_double)}}};
static const double CEMA_R0_VALUES[CEMA_R0_COUNT] = {{{array(resistances, c_double)}}};
static const double CEMA_ALPHA_VCORR = {c_double(alpha[120])};
static const double CEMA_ALPHA_50 = {c_double(alpha[50])};
static const double CEMA_ALPHA_200 = {c_double(alpha[200])};
static const double CEMA_ALPHA_800 = {c_double(alpha[800])};

#endif
"""
    (destination / "preprocess_config.h").write_text(header, encoding="ascii")
    print(args.model_id)


if __name__ == "__main__":
    main()
