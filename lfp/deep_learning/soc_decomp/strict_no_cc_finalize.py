from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import pandas as pd

from .branch_bands_no_cc_audit import build_audit as build_branch_audit
from .branch_bands_no_cc_audit import write_report as write_branch_audit_report
from .no_cc_bandtcn_experiment import NoCCBandTCNConfig, aggregate_and_write
from .strict_no_cc_artifact_manifest import config_from_metadata, generate_manifest
from .strict_no_cc_completion_audit import audit as completion_audit


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def active_bandtcn_processes() -> list[dict]:
    rows: list[dict] = []
    proc = Path("/proc")
    if not proc.exists():
        return rows
    for cmdline_path in proc.glob("[0-9]*/cmdline"):
        try:
            raw = cmdline_path.read_bytes()
        except OSError:
            continue
        parts = [p.decode(errors="replace") for p in raw.split(b"\0") if p]
        if not parts:
            continue
        cmd = " ".join(parts)
        if "soc_decomp.no_cc_bandtcn_experiment" not in cmd:
            continue
        status = {}
        try:
            for line in (cmdline_path.parent / "status").read_text(encoding="utf-8").splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    status[key] = value.strip()
        except OSError:
            status = {}
        rows.append({
            "pid": int(cmdline_path.parent.name),
            "ppid": int(status.get("PPid", "0").split()[0]),
            "cmd": cmd[:1200],
        })
    active_pids = {int(row["pid"]) for row in rows}
    for row in rows:
        row["role"] = "worker_or_child" if int(row["ppid"]) in active_pids else "main"
    return rows


def write_branch_audit(base_dir: Path) -> dict:
    summary, rows = build_branch_audit(base_dir)
    rows.to_csv(base_dir / "branch_bands_strict_no_cc_audit.csv", index=False)
    write_json(base_dir / "branch_bands_strict_no_cc_audit.json", summary)
    write_branch_audit_report(base_dir, summary, rows)
    return {
        "strict_no_cc_input_pass": bool(summary.get("strict_no_cc_input_pass")),
        "n_input_features": int(summary.get("n_input_features", 0)),
        "selected_forbidden_features": summary.get("selected_forbidden_features", []),
        "missing_selected_features_after_runtime_derivations": summary.get(
            "missing_selected_features_after_runtime_derivations",
            [],
        ),
    }


def build_bandtcn_cfg(base_dir: Path, *, save_predictions: bool = True) -> NoCCBandTCNConfig:
    meta_cfg = config_from_metadata(base_dir)
    return NoCCBandTCNConfig(
        base_dir=base_dir,
        output_prefix=str(meta_cfg["output_prefix"]),
        folds=tuple(meta_cfg["folds"]),
        variants=tuple(meta_cfg["variants"]),
        seeds=tuple(int(s) for s in meta_cfg["seeds"]),
        save_predictions=bool(save_predictions),
    )


def process_counts(processes: list[dict]) -> dict:
    return {
        "active_bandtcn_main_process_count": sum(1 for p in processes if p.get("role") == "main"),
        "active_bandtcn_worker_process_count": sum(1 for p in processes if p.get("role") != "main"),
        "active_bandtcn_process_count": len(processes),
    }


def run_finalize(base_dir: Path, *, skip_aggregate: bool, allow_partial: bool, force_while_active: bool) -> dict:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    active_processes = active_bandtcn_processes()
    active_counts = process_counts(active_processes)
    if active_counts["active_bandtcn_main_process_count"] and not skip_aggregate and not force_while_active:
        result = {
            "base_dir": str(base_dir),
            "finalized": False,
            "finalize_mode": "aborted_active_sweep",
            **active_counts,
            "active_bandtcn_processes": active_processes,
            "allow_partial": bool(allow_partial),
            "skip_aggregate": bool(skip_aggregate),
            "force_while_active": bool(force_while_active),
            "next_step": (
                "A BandTCN sweep is still active. Wait for it to finish, or rerun with "
                "--skip-aggregate --allow-partial for a safe partial refresh."
            ),
        }
        write_json(base_dir / "strict_no_cc_finalize_summary.json", result)
        return result

    branch_summary = write_branch_audit(base_dir)
    if not skip_aggregate:
        aggregate_and_write(build_bandtcn_cfg(base_dir))

    manifest_summary = generate_manifest(base_dir)
    audit_result, audit_rows = completion_audit(base_dir)
    audit_rows.to_csv(base_dir / "strict_no_cc_completion_audit.csv", index=False)
    write_json(base_dir / "strict_no_cc_completion_audit.json", audit_result)
    active_processes = active_bandtcn_processes()
    active_counts = process_counts(active_processes)

    result = {
        "base_dir": str(base_dir),
        "branch_bands_audit": branch_summary,
        "manifest_summary": manifest_summary,
        "completion_audit": audit_result,
        "finalized": bool(audit_result.get("strict_no_cc_goal_complete")),
        "finalize_mode": "final" if audit_result.get("strict_no_cc_goal_complete") else "partial_refresh",
        **active_counts,
        "active_bandtcn_processes": active_processes,
        "allow_partial": bool(allow_partial),
        "skip_aggregate": bool(skip_aggregate),
        "force_while_active": bool(force_while_active),
        "authoritative_outputs_after_finalize": [
            "branch_bands_strict_no_cc_audit.json",
            "branch_bands_strict_no_cc_audit.csv",
            "branch_bands_strict_no_cc_audit.md",
            "no_cc_bandtcn_results.csv",
            "no_cc_thermoguard_results.csv",
            "no_cc_observability_metrics.csv",
            "no_cc_observability_vs_error.csv",
            "strict_no_cc_artifact_manifest.json",
            "strict_no_cc_completion_audit.json",
            "strict_no_cc_final_comparison_report.md",
        ],
        "next_step": (
            "Strict No-CC goal is complete."
            if audit_result.get("strict_no_cc_goal_complete")
            else "Wait for the BandTCN sweep to reach 144/144, then rerun this module without --allow-partial."
        ),
    }
    write_json(base_dir / "strict_no_cc_finalize_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Finalize strict No-CC artifacts after BandTCN runs complete.",
    )
    parser.add_argument("--base-dir", default=".")
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip no_cc_bandtcn aggregate-only regeneration and only refresh audits/reports.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit 0 even when the BandTCN sweep is still incomplete.",
    )
    parser.add_argument(
        "--force-while-active",
        action="store_true",
        help="Allow aggregate regeneration even while a BandTCN sweep process is active.",
    )
    args = parser.parse_args()

    result = run_finalize(
        Path(args.base_dir),
        skip_aggregate=bool(args.skip_aggregate),
        allow_partial=bool(args.allow_partial),
        force_while_active=bool(args.force_while_active),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["finalized"] or args.allow_partial:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
