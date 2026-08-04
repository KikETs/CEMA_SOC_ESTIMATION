from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "inference_pkg_nmc"


def test_all_18_entries_are_complete_and_golden() -> None:
    index = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert len(index["entries"]) == 18
    expected = {(feature, fold, seed) for feature in ("T6", "G4") for fold in ("DST", "FUDS", "US06") for seed in (0, 1, 2)}
    actual = {(row["feature"], row["fold"], row["seed"]) for row in index["entries"]}
    assert actual == expected
    for row in index["entries"]:
        entry = ROOT / row["path"]
        for name in ("config.json", "scaler.json", "r0_table.json", "manifest.json", "code/inference.py", "code/__init__.py"):
            assert (entry / name).is_file(), f"missing {entry / name}"
        config = json.loads((entry / "config.json").read_text(encoding="utf-8"))
        assert (entry / config["weights_file"]).is_file()
        manifest = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
        for item in manifest["package_files"]:
            path = entry / item["path"]
            assert path.stat().st_size == item["size_bytes"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    golden = pd.read_csv(ROOT / "golden_test_results.csv")
    assert len(golden) == 18
    assert golden["passed"].astype(bool).all()
    assert golden["deterministic_bit_identical"].astype(bool).all()
    assert (golden["max_abs_diff"] <= golden["atol"]).all()
