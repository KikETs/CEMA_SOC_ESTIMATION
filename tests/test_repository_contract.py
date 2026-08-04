from pathlib import Path
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys

import pandas as pd

from scripts.run_reproduction import COMMANDS, conda_sibling_interpreter


REPO = Path(__file__).resolve().parents[1]


def git_executable() -> str:
    discovered = shutil.which("git")
    if discovered:
        return discovered
    candidates = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git/cmd/git.exe",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Git/cmd/git.exe",
        Path(os.environ.get("LocalAppData", "")) / "Programs/Git/cmd/git.exe",
    ]
    return str(next(path for path in candidates if path.is_file()))


def test_reproduction_entrypoints_exist():
    required = [
        "scripts/prepare_data.py",
        "scripts/verify_preprocessing.py",
        "scripts/verify_environment.py",
        "scripts/install_torch.py",
        "scripts/verify_paper_results.py",
        "nmc/preprocessing/prepare_calce_nmc.py",
        "lfp/preprocessing/prepare_lfp_raw_excel.py",
        "lfp/preprocessing/prepare_lfp_ocv_discharge_soc.py",
        "nmc/deep_learning/run_paper_t6_plain_10seed.py",
        "lfp/deep_learning/run_paper_t6_plain_10seed.py",
        "nmc/deep_learning/run_paper_g4_5seed.py",
        "lfp/deep_learning/run_paper_g4_5seed.py",
        "nmc/kf/run_all_lopo.py",
        "lfp/kf/run_paper_filters.py",
    ]
    assert all((REPO / path).is_file() for path in required)


def test_data_payloads_are_ignored():
    candidates = [
        "Data/NMC/OCV/raw.xlsx",
        "Data/NMC/Profiles/raw.xls",
        "Data/LFP/OCV/raw.xlsx",
        "Data/LFP/Profiles/raw.xlsx",
        "Data/Preprocessed/example.csv",
        "runs/nmc_dl/final_weights.pt",
        "runs/lfp_dl/prediction_rows.csv",
        "nmc/kf/results/generated.csv",
    ]
    result = subprocess.run(
        [git_executable(), "check-ignore", "-z", "--stdin"],
        cwd=REPO,
        input=("\0".join(candidates) + "\0").encode("utf-8"),
        capture_output=True,
        check=True,
    )
    ignored = {value.decode("utf-8") for value in result.stdout.split(b"\0") if value}
    assert ignored == set(candidates)


def test_no_battery_workbooks_are_tracked():
    result = subprocess.run(
        [git_executable(), "ls-files"], cwd=REPO, text=True, capture_output=True, check=True
    )
    tracked = result.stdout.lower().splitlines()
    assert not any(path.endswith((".xls", ".xlsx", ".zip")) for path in tracked)
    assert not any("/data/preprocessed/" in path for path in tracked)


def test_data_readme_names_every_required_raw_workbook():
    required = pd.read_csv(REPO / "reference/raw_source_hashes.csv")
    readme = (REPO / "Data/README.md").read_text(encoding="utf-8")
    missing = [Path(path).name for path in required["path"] if Path(path).name not in readme]
    assert not missing
    assert readme.count("https://web.calce.umd.edu/batteries/data/") == 28
    assert "12 archives" in readme
    assert "16 archives" in readme


def test_data_provenance_is_explicit_and_does_not_invent_download_date():
    provenance = (REPO / "Data/PROVENANCE.md").read_text(encoding="utf-8")
    assert "Local acquisition into this reproducibility workspace: 2026-07-21" in provenance
    assert "28/28 HTTP 200" in provenance
    assert "original-download date cannot be independently recovered" in provenance


def test_lfp_frozen_inference_package_is_complete():
    root = REPO / "lfp/deep_learning/inference_pkg_lfp"
    summary = json.loads((root / "validation_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert summary["all_passed"] is True
    assert summary["entries"] == 27
    assert len(manifest["entries"]) == 27
    golden = pd.read_csv(root / "golden_test_results.csv")
    assert len(golden) == 27
    assert golden["passed"].all()
    assert golden["max_abs_delta"].max() < 1e-6

    for entry in manifest["entries"]:
        entry_root = root / entry["path"]
        entry_manifest = json.loads((entry_root / "manifest.json").read_text(encoding="utf-8"))
        for record in entry_manifest["package_files"]:
            path = entry_root / record["path"]
            assert path.stat().st_size == record["size_bytes"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_t6_plain_locked_packages_and_headline_are_complete():
    locked = REPO / "locked_results"
    headline = pd.read_csv(
        locked
        / "CEMA_CARRIER_FAIRNESS_ABLATIONS_20260803"
        / "normalhead_auxoff"
        / "seed_slice_unweighted_mae.csv"
    )
    expected = {"NMC": 0.348279078267482, "LFP": 0.644679400016474}
    for chemistry, target in expected.items():
        values = headline[(headline["chemistry"] == chemistry) & (headline["feature"] == "T6")]
        assert sorted(values["seed"].tolist()) == list(range(10))
        assert abs(values["mae"].mean() - target) < 1e-12

    package_base = locked / "t6_plain_downstream_20260804"
    for chemistry in ("nmc", "lfp"):
        root = package_base / f"inference_pkg_{chemistry}"
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        assert len(manifest["entries"]) == 30
        golden = pd.read_csv(root / "golden_test_results.csv")
        assert len(golden) == 30
        assert golden["passed"].all()
        assert golden["deterministic_bit_identical"].all()
        assert golden["max_abs_diff"].max() < 1e-6
        for entry in manifest["entries"]:
            entry_root = root / entry["path"]
            entry_manifest = json.loads((entry_root / "manifest.json").read_text(encoding="utf-8"))
            for record in entry_manifest["package_files"]:
                path = entry_root / record["path"]
                assert path.stat().st_size == record["size_bytes"]
                assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_lfp_cc_locked_slice_unweighted_values():
    frame = pd.read_csv(
        REPO / "locked_results/lfp_kf_v2_2/cc_openloop_init_sweep.csv"
    ).set_index("initial_condition")
    expected = {
        "oracle": 0.1129039483397598,
        "minus5pp": 5.029036223673301,
        "minus10pp": 9.864444057320716,
    }
    for condition, target in expected.items():
        assert abs(frame.loc[condition, "slice_unweighted_MAE_pct"] - target) < 1e-12


def test_nmc_paper_kf_uses_frozen_training_only_identification():
    locked = str(REPO / "nmc/kf/locked_parameters")
    assert COMMANDS["nmc-kf"][-4:] == [
        "--locked-parameters-dir", locked, "--output-root", str(REPO / "runs/nmc_kf")
    ]
    assert "--locked-parameters-dir" not in COMMANDS["nmc-kf-refit"]
    assert str(REPO / "runs/nmc_kf_refit") in COMMANDS["nmc-kf-refit"]


def test_result_verifier_accepts_isolated_run_roots():
    source = (REPO / "scripts/verify_paper_results.py").read_text(encoding="utf-8")
    for option in ("--nmc-dl-root", "--lfp-dl-root", "--nmc-kf-root", "--lfp-kf-root"):
        assert option in source


def test_t6_plain_runners_lock_the_adopted_recipe():
    required = {
        '"lambda_rex": 0.0',
        '"lambda_condinv": 0.0',
        '"lambda_anchor_loss": 0.0',
        '"model_kind": "single"',
        '"head_kind": "linear"',
        '"batch_size": 2048',
        '"epochs": 200',
        '"window_len": 50',
        '"stride": 3',
    }
    for chemistry in ("nmc", "lfp"):
        source = (REPO / f"{chemistry}/deep_learning/run_paper_t6_plain_10seed.py").read_text(
            encoding="utf-8"
        )
        assert all(token in source for token in required)
        assert 'default="0,1,2,3,4,5,6,7,8,9"' in source
        assert "paper_t6_voltage_ema_all" in source


def test_numerically_sensitive_stages_have_backend_preflights():
    from scripts.run_reproduction import PREFLIGHTS

    assert PREFLIGHTS["nmc-kf"] == ["--profile", "nmc", "--require-mkl"]
    assert PREFLIGHTS["nmc-kf-refit"] == ["--profile", "nmc", "--require-mkl"]
    assert PREFLIGHTS["nmc-dl"] == ["--profile", "nmc", "--require-torch"]
    assert PREFLIGHTS["lfp-kf"] == ["--profile", "lfp-kf"]


def test_torch_installer_resolves_platform_backends():
    from scripts.install_torch import install_command, resolve_backend

    assert resolve_backend("auto", system="Linux", machine="x86_64", has_nvidia=True) == "cu128"
    assert resolve_backend("auto", system="Windows", machine="AMD64", has_nvidia=False) == "cpu"
    assert resolve_backend("auto", system="Darwin", machine="arm64") == "mps"
    assert resolve_backend("auto", system="Darwin", machine="x86_64") == "cpu"
    assert "https://download.pytorch.org/whl/cu128" in install_command("cu128", system="Linux")
    assert "https://download.pytorch.org/whl/cpu" in install_command("cpu", system="Windows")
    assert "--index-url" not in install_command("mps", system="Darwin")


def test_dl_runtime_supports_cuda_mps_and_cpu():
    for chemistry in ("nmc", "lfp"):
        source = (REPO / f"{chemistry}/deep_learning/soc_decomp/runtime.py").read_text(encoding="utf-8")
        assert 'os.environ.get("CEMA_TORCH_DEVICE", "auto")' in source
        assert 'requested = "mps"' in source
        assert 'requested = "cpu"' in source


def test_preprocessed_hash_reference_is_complete():
    frame = pd.read_csv(REPO / "reference/preprocessed_file_hashes.csv")
    assert len(frame) == 33
    assert frame["path"].is_unique
    assert frame["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert frame["chemistry"].value_counts().to_dict() == {"LFP": 24, "NMC": 9}

    raw = pd.read_csv(REPO / "reference/raw_source_hashes.csv")
    assert len(raw) == 28
    assert raw["path"].is_unique
    assert raw["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert raw["chemistry"].value_counts().to_dict() == {"LFP": 16, "NMC": 12}

    fingerprints = pd.read_csv(REPO / "reference/nmc_preprocessed_fingerprints.csv")
    assert len(fingerprints) == 9
    assert fingerprints["path"].is_unique
    assert fingerprints["strict_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    for value in fingerprints["tolerant_columns_json"]:
        assert set(__import__("json").loads(value)) == {
            "V_corr_raw", "V_corr_raw_ema50", "V_corr_raw_dev_ema50",
            "V_corr_raw_ema200", "V_corr_raw_dev_ema200",
            "V_corr_raw_ema800", "V_corr_raw_dev_ema800",
        }


def test_platform_lock_files_are_complete_and_portable():
    for platform in ("linux-64", "win-64"):
        for environment in ("cema_soc_repro", "cema_soc_lfp_kf"):
            conda_lock = REPO / f"locks/{platform}/{environment}-conda-explicit.txt"
            pip_lock = REPO / f"locks/{platform}/{environment}-pip-freeze.txt"
            assert conda_lock.is_file() and "@EXPLICIT" in conda_lock.read_text(encoding="utf-8")
            pip_text = pip_lock.read_text(encoding="utf-8")
            assert pip_lock.is_file() and "file://" not in pip_text


def test_lfp_kf_runtime_is_configured_before_numpy_import():
    source = (REPO / "lfp/kf/run_paper_filters.py").read_text(encoding="utf-8")
    top_level = ast.parse(source).body
    configure_index = next(
        i for i, node in enumerate(top_level)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "RUNTIME_ENV" for target in node.targets)
    )
    numpy_index = next(
        i for i, node in enumerate(top_level)
        if isinstance(node, ast.Import)
        and any(alias.name == "numpy" for alias in node.names)
    )
    assert configure_index < numpy_index


def test_nmc_kf_runtime_is_configured_before_numpy_import():
    source = (REPO / "nmc/kf/run_all_lopo.py").read_text(encoding="utf-8")
    assert source.index('os.environ[name] = "1"') < source.index("import numpy as np")


def test_lfp_paper_metric_contract_is_explicit():
    source = (REPO / "lfp/kf/run_paper_filters.py").read_text(encoding="utf-8")
    assert "paper_reference_metrics.csv" in source
    assert "paper_cc_openloop_slices.csv" in source
    assert "slice_unweighted_mean_of_24_full_evaluation_slices" in source
    assert (REPO / "lfp/kf/locked_parameters/artifacts/ecm_parameter_map.csv").is_file()


def test_supported_kf_runners_rebase_machine_specific_data_paths():
    lfp_source = (REPO / "lfp/kf/run_paper_filters.py").read_text(encoding="utf-8")
    nmc_source = (REPO / "nmc/kf/run_all_lopo.py").read_text(encoding="utf-8")
    assert 'REPO / "Data/Preprocessed/LFP"' in lfp_source
    assert 'REPO / "Data/Preprocessed/NMC/' in nmc_source
    assert 'REPO / "Data/NMC/OCV"' in nmc_source


def test_official_entrypoints_have_no_machine_specific_posix_paths():
    sources = {Path(command[1]).resolve() for command in COMMANDS.values()}
    sources.add((REPO / "scripts/run_reproduction.py").resolve())
    forbidden = ("/home/user", "/home/lab", "/bin/python", "바탕화면")
    for path in sources:
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path


def test_conda_sibling_interpreter_supports_windows_and_posix(tmp_path):
    envs = tmp_path / "conda" / "envs"
    current = envs / "cema_soc_repro"
    windows_python = envs / "cema_soc_lfp_kf" / "python.exe"
    windows_python.parent.mkdir(parents=True)
    windows_python.touch()
    assert conda_sibling_interpreter("cema_soc_lfp_kf", current) == windows_python

    windows_python.unlink()
    posix_python = envs / "cema_soc_lfp_kf" / "bin" / "python"
    posix_python.parent.mkdir(parents=True)
    posix_python.touch()
    assert conda_sibling_interpreter("cema_soc_lfp_kf", current) == posix_python


def test_runner_subprocesses_use_argument_lists_without_shell_strings():
    tree = ast.parse((REPO / "scripts/run_reproduction.py").read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
    ]
    assert calls
    for call in calls:
        assert not any(
            keyword.arg == "shell"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )


def test_argument_list_execution_handles_spaces(tmp_path):
    script = tmp_path / "folder with spaces" / "echo argument.py"
    script.parent.mkdir(parents=True)
    script.write_text("import sys; print(sys.argv[1])\n", encoding="utf-8")
    value = "value with spaces"
    result = subprocess.run(
        [sys.executable, str(script), value], text=True, capture_output=True, check=True
    )
    assert result.stdout.strip() == value


def test_nmc_discovery_handles_hangul_path(tmp_path, monkeypatch):
    from scripts import prepare_data

    data = tmp_path / "한글 데이터"
    profiles = data / "NMC" / "Profiles"
    profiles.mkdir(parents=True)
    source = profiles / "샘플_DST_80SOC.xlsx"
    source.write_bytes(b"workbook-placeholder")
    monkeypatch.setattr(prepare_data, "DATA", data)
    monkeypatch.setattr(prepare_data, "PREPARED", data / "Preprocessed")
    stage, manifest = prepare_data.stage_nmc_80soc()
    assert (stage / source.name).is_file()
    assert manifest.loc[0, "selected_for_paper"]
