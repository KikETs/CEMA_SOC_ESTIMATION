from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MCU = ROOT / "deployment" / "mcu"


def test_mcu_source_and_summary_are_present() -> None:
    required = [
        MCU / "README.md",
        MCU / "BENCHMARK_PROTOCOL.md",
        MCU / "scripts" / "run_raw_vit_mcu_benchmark.py",
        MCU / "scripts" / "run_kf_mcu_benchmark.py",
        MCU / "firmware" / "h563_raw_vit" / "Src" / "main.c",
        MCU / "firmware" / "h563_kf" / "Src" / "kf_core.c",
        MCU / "results" / "REPORT.md",
        MCU / "results" / "summary" / "nn_latency.csv",
        MCU / "results" / "summary" / "kf_latency.csv",
    ]
    assert all(path.is_file() for path in required)


def test_mcu_tree_contains_no_generated_or_raw_artifacts() -> None:
    forbidden_directories = {"AI", "Generated", "Build", "Debug", "Release"}
    forbidden_suffixes = {".elf", ".bin", ".hex", ".o", ".a"}

    assert not any(
        path.is_dir() and path.name in forbidden_directories
        for path in MCU.rglob("*")
    )
    assert not any(
        path.is_file() and path.suffix.lower() in forbidden_suffixes
        for path in MCU.rglob("*")
    )
    assert all(
        path.parent == MCU / "onnx"
        for path in MCU.rglob("*.onnx")
    )
    assert not any(
        path.is_file()
        and ("_raw.csv" in path.name or "_rows.csv" in path.name)
        for path in MCU.rglob("*")
    )
