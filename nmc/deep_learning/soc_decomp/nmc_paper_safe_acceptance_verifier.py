from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


TARGETS = {
    "mae_0c": 1.0,
    "mae_25c": 0.7,
    "mae_45c": 0.3,
}


def _boolish(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _safe_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _load_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing decision matrix: {path}")
    df = pd.read_csv(path)
    required = {
        "candidate",
        "protocol_type",
        "selection_rule",
        "stage2_used",
        "strict_gate_result",
        "mae_0c",
        "mae_25c",
        "mae_45c",
        "decision",
        "evidence_file",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Decision matrix is missing columns: {missing}")
    return df


def _failed_targets(row: pd.Series) -> list[str]:
    failed: list[str] = []
    for col, target in TARGETS.items():
        value = float(row[col])
        if not value < target:
            failed.append(f"{col.replace('mae_', '').upper()} {value:.3f}>={target:.3f}")
    return failed


def _contains_any(texts: Iterable[object], needles: Iterable[str]) -> bool:
    blob = " ".join(_safe_text(x).lower() for x in texts)
    return any(needle in blob for needle in needles)


def _row_has_shortcut_risk(row: pd.Series) -> bool:
    protocol_blob = " ".join(
        _safe_text(x).lower()
        for x in [row["candidate"], row["protocol_type"], row["selection_rule"], row["decision"]]
    )
    if any(needle in protocol_blob for needle in ["train25", "train 25c", "25c dst", "exploratory"]):
        return True

    failure = _safe_text(row.get("key_failure", "")).lower()
    return any(
        needle in failure
        for needle in [
            "checkpoint chosen by train",
            "shortcut suspicion remains",
            "paper-ready",
        ]
    )


def _metric_target_met(row: pd.Series) -> bool:
    return not _failed_targets(row)


def _is_causal_observer(row: pd.Series) -> bool:
    blob = " ".join(
        _safe_text(x).lower()
        for x in [row["candidate"], row["protocol_type"], row["selection_rule"], row["decision"], row.get("key_failure", "")]
    )
    return any(
        needle in blob
        for needle in [
            "causal observer",
            "capacity-anchor",
            "capacity anchor",
            "current integration",
            "known initial soc",
        ]
    )


def _observer_audit_ok(audit_path: Path | None) -> tuple[bool, str]:
    if audit_path is None:
        return False, "missing_observer_audit_path"
    if not audit_path.exists():
        return False, f"missing_observer_audit_file:{audit_path}"
    audit = pd.read_csv(audit_path)
    required = {"scope", "result", "interpretation"}
    missing = sorted(required - set(audit.columns))
    if missing:
        return False, f"observer_audit_missing_columns:{missing}"
    bad = audit[~audit["result"].astype(str).str.startswith("PASS") & ~audit["result"].astype(str).eq("NOT_APPLICABLE_TO_OBSERVER")]
    if not bad.empty:
        return False, "observer_audit_has_nonpass_rows:" + ",".join(bad["scope"].astype(str).head(5))
    text = " ".join(audit["interpretation"].astype(str).str.lower().tolist())
    if "not strict nocc" not in text or "known initial soc" not in text:
        return False, "observer_audit_missing_required_caveats"
    return True, "observer_audit_passed"


def _base_first_pass_exists(df: pd.DataFrame) -> bool:
    for _, row in df.iterrows():
        if _is_causal_observer(row):
            continue
        stage2 = _boolish(row["stage2_used"])
        strict = _safe_text(row["strict_gate_result"]).strip().lower()
        protocol_type = _safe_text(row["protocol_type"]).lower()
        decision = _safe_text(row["decision"]).lower()
        if (
            not stage2
            and strict == "pass"
            and "base" in protocol_type
            and "reject" not in decision
            and _metric_target_met(row)
            and not _row_has_shortcut_risk(row)
        ):
            return True
    return False


def _classify_strict_nocc(row: pd.Series, has_base_first_pass: bool) -> tuple[bool, str, str]:
    if _is_causal_observer(row):
        return False, "not_strict_nocc_track", "This row is a causal observer candidate, not a strict NoCC main-model candidate."

    stage2 = _boolish(row["stage2_used"])
    strict = _safe_text(row["strict_gate_result"]).strip().lower()
    protocol_type = _safe_text(row["protocol_type"]).lower()
    decision = _safe_text(row["decision"]).lower()
    shortcut = _row_has_shortcut_risk(row)
    failed = _failed_targets(row)

    if failed:
        return False, "reject_metric_failed", "Target miss: " + "; ".join(failed)

    if strict == "apparent_pass" or shortcut:
        return False, "diagnostic_only_shortcut_risk", "Good numeric result, but selection/candidate has shortcut or exploratory risk."

    if "partial" in strict:
        return False, "reject_partial_evidence", "Evidence is partial; it cannot support a main-model claim."

    if strict != "pass":
        return False, "reject_gate_failed", f"Strict gate result is {strict!r}, not 'pass'."

    if stage2 and not has_base_first_pass:
        return False, "diagnostic_only_no_base_first_pass", "Stage2/correction result has no preceding base-first pass."

    if "diagnostic" in decision or "exploratory" in decision:
        return False, "diagnostic_only", "Decision matrix marks this as diagnostic/exploratory."

    if (not stage2) and ("base" in protocol_type):
        return True, "paper_safe_base_main_candidate", "Base-first candidate passes numeric targets and strict gate."

    if stage2:
        return True, "paper_safe_corrected_candidate_needs_3seed", "Corrected candidate has base-first support; promote only after 3-seed confirmation."

    return True, "paper_safe_candidate_needs_review", "Numeric targets and strict gate pass; review protocol details before promotion."


def _classify_observer(row: pd.Series, audit_ok: bool, audit_reason: str) -> tuple[bool, str, str]:
    if not _is_causal_observer(row):
        return False, "not_observer_track", "This row is not a causal observer candidate."
    failed = _failed_targets(row)
    if failed:
        return False, "observer_reject_metric_failed", "Target miss: " + "; ".join(failed)
    if _row_has_shortcut_risk(row):
        return False, "observer_reject_shortcut_risk", "Observer row still contains shortcut/exploratory risk markers."
    if not audit_ok:
        return False, "observer_reject_audit_incomplete", audit_reason
    return (
        True,
        "observer_candidate_with_declared_current_integration",
        "Causal observer candidate passes numeric profile-heldout targets with explicit current-integration and initial-SOC caveats.",
    )


def verify(matrix: pd.DataFrame, observer_audit_path: Path | None = None) -> pd.DataFrame:
    has_base_first_pass = _base_first_pass_exists(matrix)
    observer_audit_ok, observer_audit_reason = _observer_audit_ok(observer_audit_path)
    rows = []
    for _, row in matrix.iterrows():
        strict_allowed, strict_status, strict_reason = _classify_strict_nocc(row, has_base_first_pass)
        observer_allowed, observer_status, observer_reason = _classify_observer(
            row,
            audit_ok=observer_audit_ok,
            audit_reason=observer_audit_reason,
        )
        rows.append(
            {
                "rank": row.get("rank", ""),
                "candidate": row["candidate"],
                "claim_track": "causal_observer" if _is_causal_observer(row) else "strict_nocc_or_control",
                "protocol_type": row["protocol_type"],
                "selection_rule": row["selection_rule"],
                "stage2_used": _boolish(row["stage2_used"]),
                "strict_gate_result": row["strict_gate_result"],
                "mae_0c": float(row["mae_0c"]),
                "mae_25c": float(row["mae_25c"]),
                "mae_45c": float(row["mae_45c"]),
                "metric_target_met": _metric_target_met(row),
                "shortcut_risk": _row_has_shortcut_risk(row),
                "strict_nocc_main_allowed": strict_allowed,
                "strict_nocc_status": strict_status,
                "strict_nocc_reason": strict_reason,
                "observer_main_candidate_allowed": observer_allowed,
                "observer_status": observer_status,
                "observer_reason": observer_reason,
                "evidence_file": row["evidence_file"],
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df.loc[:, columns].copy()
    for col in ["mae_0c", "mae_25c", "mae_45c"]:
        if col in view:
            view[col] = view[col].map(lambda x: f"{float(x):.3f}")
    try:
        return view.to_markdown(index=False)
    except Exception:
        return view.to_csv(index=False)


def write_report(out_path: Path, verdict: pd.DataFrame) -> None:
    strict_accepted = verdict[verdict["strict_nocc_main_allowed"].astype(bool)].copy()
    observer_accepted = verdict[verdict["observer_main_candidate_allowed"].astype(bool)].copy()
    apparent = verdict[verdict["strict_gate_result"].astype(str).str.lower().eq("apparent_pass")]
    stage2_no_main = verdict[
        verdict["stage2_used"].astype(bool)
        & (~verdict["strict_nocc_main_allowed"].astype(bool))
        & (~verdict["observer_main_candidate_allowed"].astype(bool))
    ]

    if strict_accepted.empty:
        strict_overall = "NO_UNIVERSAL_STRICT_NOCC_MAIN_MODEL_FOUND"
        nocc_position = "NOCC_ABLATION_ONLY"
    else:
        strict_overall = "STRICT_NOCC_MAIN_MODEL_CANDIDATE_EXISTS"
        nocc_position = "PROMOTE_ONLY_AFTER_PROTOCOL_REVIEW_AND_3SEEDS"
    if observer_accepted.empty:
        observer_overall = "NO_CAUSAL_OBSERVER_MAIN_TRACK_CANDIDATE_FOUND"
    else:
        observer_overall = "CAUSAL_OBSERVER_MAIN_TRACK_CANDIDATE_EXISTS_WITH_CAVEATS"

    lines = [
        "# Paper-Safe Acceptance Verdict",
        "",
        "This verifier intentionally separates good-looking numbers from paper-safe adoption and separates strict NoCC from causal-observer claims.",
        "",
        "## Overall Verdict",
        "",
        "```text",
        f"Universal strict NoCC main model: {strict_overall}",
        f"Recommended NoCC position: {nocc_position}",
        f"Causal observer main-track candidate: {observer_overall}",
        "Clean adoption protocol: AVAILABLE",
        "Shortcut suspicion audit: ENFORCED",
        "```",
        "",
        "Targets:",
        "",
        "- 0C MAE < 1.0",
        "- 25C MAE < 0.7",
        "- 45C MAE < 0.3",
        "",
        "Acceptance rule:",
        "",
        "- Numeric targets alone are insufficient.",
        "- `apparent_pass`, train-25C selector, or shortcut-risk rows are diagnostic only.",
        "- Stage2/correction cannot become the main claim unless a base-first model already passes.",
        "- Partial ProfileLOO evidence cannot be promoted.",
        "- Causal observer rows are not strict NoCC rows; they require explicit current-integration and known-initial-SOC caveats.",
        "",
        "## Counts",
        "",
        f"- Accepted strict NoCC main-model candidates: {len(strict_accepted)}",
        f"- Accepted causal observer main-track candidates: {len(observer_accepted)}",
        f"- Apparent-pass rows excluded: {len(apparent)}",
        f"- Stage2 diagnostic rows excluded: {len(stage2_no_main)}",
        f"- Total checked rows: {len(verdict)}",
        "",
        "## Candidate Verdicts",
        "",
        _markdown_table(
            verdict,
            [
                "rank",
                "candidate",
                "stage2_used",
                "strict_gate_result",
                "mae_0c",
                "mae_25c",
                "mae_45c",
                "metric_target_met",
                "strict_nocc_status",
                "observer_status",
                "strict_nocc_reason",
                "observer_reason",
            ],
        ),
        "",
        "## Working Conclusion",
        "",
        "The 25C-foundation -> 0/45C-fine-tune -> Stage2 one-check should not be continued as a main-model path. "
        "It is useful as a diagnostic because it shows cold/hot adaptation can improve 0C and 45C, but it leaves 25C above the paper target and depends on correction after a weak base.",
        "",
        "Strict NoCC remains an ablation/failure-analysis track. The current positive main-track evidence is the separate causal observer row, which must be reported with known-initial-SOC and explicit-current-integration caveats.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify whether NMC strict NoCC candidates are paper-safe main-model candidates.")
    parser.add_argument("--matrix", default="paper_results/universal_soc_final_decision_matrix.csv")
    parser.add_argument("--out-dir", default="paper_results")
    parser.add_argument("--observer-audit", default="paper_results/nmc_three_profile_capacity_anchor_leakage_audit.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix_path = Path(args.matrix).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = _load_matrix(matrix_path)
    observer_audit_path = Path(args.observer_audit).resolve()
    verdict = verify(matrix, observer_audit_path=observer_audit_path)
    csv_path = out_dir / "paper_safe_acceptance_verdict.csv"
    md_path = out_dir / "paper_safe_acceptance_verdict.md"
    verdict.to_csv(csv_path, index=False)
    write_report(md_path, verdict)

    accepted_count = int(verdict["strict_nocc_main_allowed"].sum())
    observer_count = int(verdict["observer_main_candidate_allowed"].sum())
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"accepted_strict_nocc_main_model_candidates={accepted_count}")
    print(f"accepted_causal_observer_candidates={observer_count}")


if __name__ == "__main__":
    main()
