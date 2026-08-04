#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DROP = Path(__file__).resolve().parents[1]
FIRMWARE = DROP / "firmware" / "h563_bench"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    args = parser.parse_args()
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
    print(args.model_id)


if __name__ == "__main__":
    main()
