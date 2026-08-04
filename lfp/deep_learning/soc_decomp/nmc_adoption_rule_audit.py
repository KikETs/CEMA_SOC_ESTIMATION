from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


TARGETS = {"0C": 1.0, "25C": 0.7, "45C": 0.3}
RULE_PRIORITY = {
    "val_target_worst_min": 0,
    "val_target_worst_early_within_3pct": 1,
    "val_target_mean_min": 2,
    "val_worst_min": 3,
    "val_mean_min": 4,
    "final_last5_predeclared": 5,
    "fixed_epoch_100": 6,
    "fixed_epoch_200": 6,
    "fixed_epoch_300": 6,
    "fixed_epoch_400": 6,
    "fixed_epoch_500": 6,
    "diagnostic_best_test_25C": 99,
}


def _protocol_from_name(path: Path) -> str:
    name = path.name
    m = re.match(r"nmc_universal_(.+?)_E1_DST_US06_to_FUDS_seed\d+_test_summary\.csv$", name)
    if not m:
        return path.stem
    return m.group(1)


def _seed_from_name(path: Path) -> int:
    m = re.search(r"_seed(\d+)_", path.name)
    return int(m.group(1)) if m else -1


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _mae_payload(row: pd.Series) -> dict[str, float | bool]:
    mae_0 = _float_or_nan(row.get("0.0"))
    mae_25 = _float_or_nan(row.get("25.0"))
    mae_45 = _float_or_nan(row.get("45.0"))
    target_norm_worst = max(
        mae_0 / TARGETS["0C"],
        mae_25 / TARGETS["25C"],
        mae_45 / TARGETS["45C"],
    )
    return {
        "test_MAE_0C": mae_0,
        "test_MAE_25C": mae_25,
        "test_MAE_45C": mae_45,
        "test_target_norm_worst": target_norm_worst,
        "strict_target_met": bool(
            mae_0 < TARGETS["0C"] and mae_25 < TARGETS["25C"] and mae_45 < TARGETS["45C"]
        ),
    }


def _best_row(df: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in df.columns or df.empty:
        return None
    metric = pd.to_numeric(df[column], errors="coerce")
    if metric.notna().sum() == 0:
        return None
    idx = metric.idxmin()
    return df.loc[idx]


def _row_for_epoch(test_epochs: pd.DataFrame, epoch: int) -> pd.Series | None:
    rows = test_epochs[pd.to_numeric(test_epochs["epoch"], errors="coerce").astype("Int64") == int(epoch)]
    if rows.empty:
        return None
    return rows.iloc[0]


def _append_selection(
    rows: list[dict[str, Any]],
    *,
    protocol: str,
    seed: int,
    rule: str,
    selection_signal: str,
    paper_safe: bool,
    uses_test_for_selection: bool,
    selected_epoch: int,
    test_row: pd.Series,
    selector_score: float | None = None,
    selector_detail: str = "",
) -> None:
    payload = _mae_payload(test_row)
    rows.append(
        {
            "protocol": protocol,
            "seed": seed,
            "selection_rule": rule,
            "selection_signal": selection_signal,
            "paper_safe": paper_safe,
            "uses_test_for_selection": uses_test_for_selection,
            "selected_epoch": selected_epoch,
            "selector_score": selector_score,
            "selector_detail": selector_detail,
            **payload,
        }
    )


def audit_one(test_path: Path) -> list[dict[str, Any]]:
    protocol = _protocol_from_name(test_path)
    seed = _seed_from_name(test_path)
    prefix = test_path.name.replace("_test_summary.csv", "")
    selector_path = test_path.with_name(prefix + "_selector_trace.csv")

    test = pd.read_csv(test_path)
    if test.empty:
        return []

    test_epochs = test[~test["variant"].astype(str).str.contains("_selected_", regex=False)].copy()
    test_epochs["epoch"] = pd.to_numeric(test_epochs["epoch"], errors="coerce").astype("Int64")
    test_epochs = test_epochs.dropna(subset=["epoch"]).sort_values("epoch")

    rows: list[dict[str, Any]] = []

    selected = test[test["variant"].astype(str).str.contains("_selected_", regex=False)]
    if not selected.empty:
        selected_row = selected.iloc[0]
        _append_selection(
            rows,
            protocol=protocol,
            seed=seed,
            rule="final_last5_predeclared",
            selection_signal="predeclared_end_of_training",
            paper_safe=True,
            uses_test_for_selection=False,
            selected_epoch=int(_float_or_nan(selected_row["epoch"])),
            test_row=selected_row,
            selector_score=None,
            selector_detail="Last-5 snapshot ensemble at the predeclared final training horizon.",
        )

    available_epochs = [int(ep) for ep in sorted(test_epochs["epoch"].dropna().unique())]
    for epoch in available_epochs:
        test_row = _row_for_epoch(test_epochs, epoch)
        if test_row is None:
            continue
        _append_selection(
            rows,
            protocol=protocol,
            seed=seed,
            rule=f"fixed_epoch_{epoch}",
            selection_signal="predeclared_fixed_horizon",
            paper_safe=True,
            uses_test_for_selection=False,
            selected_epoch=epoch,
            test_row=test_row,
            selector_score=None,
            selector_detail="Fixed horizon is valid only if declared before looking at test diagnostics.",
        )

    if selector_path.exists():
        selector = pd.read_csv(selector_path)
        selector["epoch"] = pd.to_numeric(selector["epoch"], errors="coerce").astype("Int64")
        selector = selector[selector["epoch"].isin(available_epochs)].copy()
        rules = [
            ("val_mean_min", "selector_valid_mean", "validation_mean"),
            ("val_worst_min", "selector_valid_worst", "validation_worst"),
            ("val_target_mean_min", "selector_valid_target_norm_mean", "validation_target_normalized_mean"),
            ("val_target_worst_min", "selector_valid_target_norm_worst", "validation_target_normalized_worst"),
        ]
        for rule, column, signal in rules:
            srow = _best_row(selector, column)
            if srow is None:
                continue
            epoch = int(srow["epoch"])
            test_row = _row_for_epoch(test_epochs, epoch)
            if test_row is None:
                continue
            _append_selection(
                rows,
                protocol=protocol,
                seed=seed,
                rule=rule,
                selection_signal=signal,
                paper_safe=True,
                uses_test_for_selection=False,
                selected_epoch=epoch,
                test_row=test_row,
                selector_score=_float_or_nan(srow.get(column)),
                selector_detail=f"Selected minimum {column} on validation VALIDATION only.",
            )

        if "selector_valid_target_norm_worst" in selector.columns:
            metric = pd.to_numeric(selector["selector_valid_target_norm_worst"], errors="coerce")
            if metric.notna().any():
                best = float(metric.min())
                candidates = selector[metric <= best * 1.03].sort_values("epoch")
                if not candidates.empty:
                    srow = candidates.iloc[0]
                    epoch = int(srow["epoch"])
                    test_row = _row_for_epoch(test_epochs, epoch)
                    if test_row is not None:
                        _append_selection(
                            rows,
                            protocol=protocol,
                            seed=seed,
                            rule="val_target_worst_early_within_3pct",
                            selection_signal="validation_target_normalized_worst_with_stability_margin",
                            paper_safe=True,
                            uses_test_for_selection=False,
                            selected_epoch=epoch,
                            test_row=test_row,
                            selector_score=_float_or_nan(srow.get("selector_valid_target_norm_worst")),
                            selector_detail=(
                                "Earliest epoch within 3% of the best validation target-normalized worst score."
                            ),
                        )

    if not test_epochs.empty:
        metric_25 = pd.to_numeric(test_epochs["25.0"], errors="coerce")
        if metric_25.notna().any():
            test_row = test_epochs.loc[metric_25.idxmin()]
            _append_selection(
                rows,
                protocol=protocol,
                seed=seed,
                rule="diagnostic_best_test_25C",
                selection_signal="test_peek_not_adoptable",
                paper_safe=False,
                uses_test_for_selection=True,
                selected_epoch=int(test_row["epoch"]),
                test_row=test_row,
                selector_score=float(metric_25.min()),
                selector_detail="For diagnosis only. This must not be used as an adoption rule.",
            )

    return rows


def build_report(audit: pd.DataFrame, out_md: Path) -> None:
    safe = audit[audit["paper_safe"]].copy()
    safe["rule_priority"] = safe["selection_rule"].map(RULE_PRIORITY).fillna(50)
    safe = safe.sort_values(["test_target_norm_worst", "rule_priority", "test_MAE_25C", "test_MAE_45C"])
    validation_safe = safe[safe["selection_signal"].astype(str).str.startswith("validation")].copy()
    diagnostic = audit[~audit["paper_safe"]].copy()
    diagnostic = diagnostic.sort_values(["test_MAE_25C", "test_target_norm_worst"])

    lines: list[str] = []
    lines.append("# Universal Adoption Rule Audit")
    lines.append("")
    lines.append("이 문서는 기존 E1 seed0 base-only run들을 대상으로, test를 보지 않는 채택 규칙과 test-peek diagnostic을 분리해 재평가한 결과다.")
    lines.append("")
    lines.append("Strict target은 0C < 1.0, 25C < 0.7, 45C < 0.3 MAE이다.")
    lines.append("")

    lines.append("## Best Validation-Only Rule")
    lines.append("")
    if validation_safe.empty:
        lines.append("No validation-only rows were found.")
    else:
        cols = [
            "protocol",
            "selection_rule",
            "selected_epoch",
            "test_MAE_0C",
            "test_MAE_25C",
            "test_MAE_45C",
            "test_target_norm_worst",
            "strict_target_met",
        ]
        lines.append(validation_safe.head(10)[cols].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## Paper-Safe Rule Top 15")
    lines.append("")
    if safe.empty:
        lines.append("No paper-safe rows were found.")
    else:
        cols = [
            "protocol",
            "selection_rule",
            "selected_epoch",
            "test_MAE_0C",
            "test_MAE_25C",
            "test_MAE_45C",
            "test_target_norm_worst",
            "strict_target_met",
        ]
        lines.append(safe.head(15)[cols].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## Diagnostic Test-Peek Best 25C")
    lines.append("")
    lines.append("아래 표는 어디까지나 진단용이다. test 25C를 보고 epoch를 고른 것이므로 채택 규칙으로 쓰면 안 된다.")
    lines.append("")
    if diagnostic.empty:
        lines.append("No diagnostic rows were found.")
    else:
        cols = [
            "protocol",
            "selection_rule",
            "selected_epoch",
            "test_MAE_0C",
            "test_MAE_25C",
            "test_MAE_45C",
            "test_target_norm_worst",
            "strict_target_met",
        ]
        lines.append(diagnostic.head(12)[cols].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## Adoption Rule Interpretation")
    lines.append("")
    lines.append("- `final_last5_predeclared`: 끝까지 학습한 뒤 마지막 snapshot ensemble을 쓰는 가장 단순하고 깨끗한 규칙이다.")
    lines.append("- `fixed_epoch_N`: epoch N을 실험 전에 선언했다면 깨끗하지만, 사후에 test를 보고 N을 고르면 안 된다.")
    lines.append("- `val_target_worst_min`: validation의 온도별 목표 난이도를 정규화한 뒤 최악값을 최소화한다. 25C만 맞추는 selector보다 논리적으로 낫다.")
    lines.append("- `val_target_worst_early_within_3pct`: validation 최적점 주변에서 가장 이른 epoch를 선택해 후반 drift를 줄이는 규칙이다.")
    lines.append("")

    lines.append("## Current Conclusion")
    lines.append("")
    if not validation_safe.empty:
        best = validation_safe.iloc[0]
        lines.append(
            f"가장 좋은 validation-only row는 `{best['protocol']}` / `{best['selection_rule']}`이며 "
            f"test MAE는 0C {best['test_MAE_0C']:.3f}, 25C {best['test_MAE_25C']:.3f}, "
            f"45C {best['test_MAE_45C']:.3f}이다."
        )
        if bool(best["strict_target_met"]):
            lines.append("이 row는 strict target을 통과한다. 다음 단계는 E1-E4 rotation seed0이다.")
        else:
            lines.append("하지만 이 row도 strict target을 통과하지 못한다. 따라서 아직 universal main model로 승격하면 안 된다.")
    lines.append("")
    lines.append("현재 증거상 가장 논리적인 방향은 Stage 2 correction이 아니라 base-only Stage 1에서 validation-target-normalized 채택 규칙과 voltage-anchor bounded residual 구조를 결합해 더 안정화하는 것이다.")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("remote_result_summaries"))
    parser.add_argument("--out-dir", type=Path, default=Path("paper_results"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.result_dir.glob("nmc_universal_base_*_e*_eval*_E1_DST_US06_to_FUDS_seed*_test_summary.csv"))

    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(audit_one(path))

    audit = pd.DataFrame(rows)
    if audit.empty:
        raise SystemExit("No audit rows found.")

    audit["rule_priority"] = audit["selection_rule"].map(RULE_PRIORITY).fillna(50)
    audit = audit.sort_values(
        ["paper_safe", "test_target_norm_worst", "rule_priority", "test_MAE_25C"],
        ascending=[False, True, True, True],
    )
    out_csv = args.out_dir / "universal_adoption_rule_audit.csv"
    out_md = args.out_dir / "universal_adoption_rule_audit.md"
    audit.to_csv(out_csv, index=False)
    build_report(audit, out_md)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
