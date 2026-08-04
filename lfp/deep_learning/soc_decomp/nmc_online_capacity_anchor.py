from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = {0.0: 1.0, 25.0: 0.7, 45.0: 0.3}


@dataclass
class OnlineCapacityAnchorConfig:
    raw_root: Path = Path("nmc_samsung_inr_18650_2ah_raw/NMC SAMSUNG INR 18650 2Ah")
    output_dir: Path = Path("paper_results")
    output_prefix: str = "nmc_online_capacity_anchor"


def _temp(value: object) -> float:
    return float(str(value).replace("C", "").strip())


def _load(root: Path) -> dict[tuple[str, float], dict[str, np.ndarray | float]]:
    out: dict[tuple[str, float], dict[str, np.ndarray | float]] = {}
    for path in sorted(root.rglob("*.csv")):
        head = pd.read_csv(path, nrows=2)
        profile = str(head["Profile"].iloc[0])
        temp = _temp(head["TempLabel"].iloc[0])
        df = pd.read_csv(path, usecols=["Step_Time(s)", "Current(A)", "Voltage(V)", "SOC_CC", "Qnet_denom(Ah)"])
        t = pd.to_numeric(df["Step_Time(s)"], errors="coerce").to_numpy(np.float64)
        i_dis = -pd.to_numeric(df["Current(A)"], errors="coerce").to_numpy(np.float64)
        v = pd.to_numeric(df["Voltage(V)"], errors="coerce").to_numpy(np.float64)
        y = pd.to_numeric(df["SOC_CC"], errors="coerce").to_numpy(np.float64)
        dt = np.diff(t, prepend=t[0])
        ok = np.isfinite(dt) & (dt > 0)
        dt[~ok] = float(np.nanmedian(dt[ok])) if np.any(ok) else 1.0
        dah = i_dis * dt / 3600.0
        run_max = np.maximum.accumulate(v)
        run_min = np.minimum.accumulate(v)
        out[(profile, temp)] = {
            "dah": dah,
            "ah": np.cumsum(dah),
            "y": y,
            "v": v,
            "v_range_run": run_max - run_min,
            "q_true": float(df["Qnet_denom(Ah)"].iloc[0]) / max(float(y[0]), 1e-9),
        }
    if not out:
        raise FileNotFoundError(f"No NMC CSV files under {root}")
    return out


def _q_values(data: dict[tuple[str, float], dict[str, np.ndarray | float]], profiles: list[str], temp: float) -> np.ndarray:
    return np.asarray([float(data[(profile, temp)]["q_true"]) for profile in profiles], dtype=np.float64)


def _v_range_at(rec: dict[str, np.ndarray | float], prefix: int) -> float:
    vr = rec["v_range_run"]
    assert isinstance(vr, np.ndarray)
    idx = min(max(int(prefix), 1), len(vr)) - 1
    return float(vr[idx])


def _constant_q_mae(rec: dict[str, np.ndarray | float], q_full: float) -> float:
    y = rec["y"]
    ah = rec["ah"]
    assert isinstance(y, np.ndarray) and isinstance(ah, np.ndarray)
    pred = np.clip(float(y[0]) - ah / max(float(q_full), 1e-9), 0.0, 1.0)
    return float(np.mean(np.abs(pred - y)) * 100.0)


def _q_bounded_at_prefix(
    data: dict[tuple[str, float], dict[str, np.ndarray | float]],
    train_profiles: list[str],
    test_profile: str,
    temp: float,
    prefix: int,
    shrink: float,
    margin: float,
) -> tuple[float, float]:
    qs = _q_values(data, train_profiles, temp)
    xs = np.asarray([_v_range_at(data[(profile, temp)], prefix) for profile in train_profiles], dtype=np.float64)
    x = _v_range_at(data[(test_profile, temp)], prefix)
    q_base = float(np.mean(qs))
    if abs(float(xs[1] - xs[0])) < 1e-12:
        q_pred = q_base
    else:
        q_pred = q_base + float(shrink) * float((qs[1] - qs[0]) / (xs[1] - xs[0])) * (x - float(np.mean(xs)))
    extra = float(margin) * abs(float(qs[1] - qs[0]))
    q_pred = float(np.clip(q_pred, float(np.min(qs) - extra), float(np.max(qs) + extra)))
    return q_base, q_pred


def _twophase_mae(
    rec: dict[str, np.ndarray | float],
    q_base: float,
    q_adapt: float,
    prefix: int,
) -> float:
    y = rec["y"]
    ah = rec["ah"]
    assert isinstance(y, np.ndarray) and isinstance(ah, np.ndarray)
    sw = min(max(int(prefix), 1), len(y))
    pred = np.empty_like(y)
    pred[:sw] = np.clip(float(y[0]) - ah[:sw] / max(float(q_base), 1e-9), 0.0, 1.0)
    if sw < len(y):
        pred[sw:] = np.clip(pred[sw - 1] - (ah[sw:] - ah[sw - 1]) / max(float(q_adapt), 1e-9), 0.0, 1.0)
    return float(np.mean(np.abs(pred - y)) * 100.0)


def _online_q_series(
    data: dict[tuple[str, float], dict[str, np.ndarray | float]],
    train_profiles: list[str],
    test_profile: str,
    temp: float,
    *,
    min_prefix: int,
    shrink: float,
    margin: float,
    alpha: float,
    update_period: int,
) -> tuple[float, np.ndarray]:
    rec = data[(test_profile, temp)]
    test_vr = rec["v_range_run"]
    assert isinstance(test_vr, np.ndarray)
    n = len(test_vr)
    qs = _q_values(data, train_profiles, temp)
    q_base = float(np.mean(qs))
    train_vr0 = data[(train_profiles[0], temp)]["v_range_run"]
    train_vr1 = data[(train_profiles[1], temp)]["v_range_run"]
    assert isinstance(train_vr0, np.ndarray) and isinstance(train_vr1, np.ndarray)

    prefix_idx = np.arange(1, n + 1, dtype=np.int64)
    idx0 = np.minimum(prefix_idx, len(train_vr0)) - 1
    idx1 = np.minimum(prefix_idx, len(train_vr1)) - 1
    x0 = train_vr0[idx0]
    x1 = train_vr1[idx1]
    denom = x1 - x0
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = q_base + float(shrink) * ((qs[1] - qs[0]) / denom) * (test_vr - 0.5 * (x0 + x1))
    raw[~np.isfinite(raw) | (np.abs(denom) < 1e-12)] = q_base
    extra = float(margin) * abs(float(qs[1] - qs[0]))
    raw = np.clip(raw, float(np.min(qs) - extra), float(np.max(qs) + extra))

    target = np.empty(n, dtype=np.float64)
    last = q_base
    min_prefix = max(int(min_prefix), 1)
    update_period = max(int(update_period), 1)
    for k in range(n):
        step = k + 1
        if step >= min_prefix and ((step - min_prefix) % update_period == 0):
            last = float(raw[k])
        target[k] = last

    alpha = float(alpha)
    if alpha >= 0.999:
        q_eff = target
    else:
        q_eff = np.empty_like(target)
        prev = q_base
        for k, val in enumerate(target):
            prev = (1.0 - alpha) * prev + alpha * float(val)
            q_eff[k] = prev
    if min_prefix > 1:
        q_eff[: min(min_prefix - 1, n)] = q_base
    return q_base, q_eff


def _online_mae(
    rec: dict[str, np.ndarray | float],
    q_eff: np.ndarray,
) -> float:
    y = rec["y"]
    dah = rec["dah"]
    assert isinstance(y, np.ndarray) and isinstance(dah, np.ndarray)
    q_eff = np.maximum(np.asarray(q_eff, dtype=np.float64), 1e-9)
    pred = np.clip(float(y[0]) - np.cumsum(dah / q_eff), 0.0, 1.0)
    return float(np.mean(np.abs(pred - y)) * 100.0)


def _rotation_rows_for_candidate(
    data: dict[tuple[str, float], dict[str, np.ndarray | float]],
    *,
    mode: str,
    min_prefix: int | None,
    shrink: float | None,
    margin: float | None,
    alpha: float | None,
    update_period: int | None,
) -> pd.DataFrame:
    profiles = sorted({profile for profile, _temp in data})
    temps = sorted({temp for _profile, temp in data})
    rows = []
    for test_profile in profiles:
        remaining = [p for p in profiles if p != test_profile]
        for valid_profile in remaining:
            train_profiles = [p for p in remaining if p != valid_profile]
            row = {
                "mode": mode,
                "feature": "causal_v_range",
                "min_prefix": "" if min_prefix is None else int(min_prefix),
                "shrink": "" if shrink is None else float(shrink),
                "margin": "" if margin is None else float(margin),
                "alpha": "" if alpha is None else float(alpha),
                "update_period": "" if update_period is None else int(update_period),
                "test_profile": test_profile,
                "valid_profile": valid_profile,
                "train_profiles": ",".join(train_profiles),
            }
            worst = 0.0
            failures = []
            for temp in temps:
                q_base = float(np.mean(_q_values(data, train_profiles, temp)))
                rec = data[(test_profile, temp)]
                if mode == "train_temp_Qfull":
                    q_used = q_base
                    mae = _constant_q_mae(rec, q_used)
                elif mode == "fixed_prefix_twophase":
                    assert min_prefix is not None and shrink is not None and margin is not None
                    q_base, q_used = _q_bounded_at_prefix(
                        data,
                        train_profiles,
                        test_profile,
                        temp,
                        int(min_prefix),
                        float(shrink),
                        float(margin),
                    )
                    mae = _twophase_mae(rec, q_base, q_used, int(min_prefix))
                elif mode == "online_bounded_vrange":
                    assert (
                        min_prefix is not None
                        and shrink is not None
                        and margin is not None
                        and alpha is not None
                        and update_period is not None
                    )
                    q_base, q_series = _online_q_series(
                        data,
                        train_profiles,
                        test_profile,
                        temp,
                        min_prefix=int(min_prefix),
                        shrink=float(shrink),
                        margin=float(margin),
                        alpha=float(alpha),
                        update_period=int(update_period),
                    )
                    q_used = float(q_series[-1])
                    mae = _online_mae(rec, q_series)
                    row[f"{temp:g}C_Qend_Ah"] = q_used
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                row[f"{temp:g}C_Qbase_Ah"] = q_base
                if mode != "online_bounded_vrange":
                    row[f"{temp:g}C_Qused_Ah"] = q_used
                row[f"{temp:g}C_MAE_pct"] = mae
                if temp in TARGETS:
                    worst = max(worst, mae / TARGETS[temp])
                    if mae >= TARGETS[temp]:
                        failures.append(f"{temp:g}C {mae:.3f}>={TARGETS[temp]:.3f}")
            row["target_met"] = not failures
            row["target_norm_worst"] = float(worst)
            row["failure_detail"] = "; ".join(failures) if failures else "pass"
            rows.append(row)
    return pd.DataFrame(rows)


def run(cfg: OnlineCapacityAnchorConfig) -> dict[str, pd.DataFrame]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    data = _load(cfg.raw_root)

    frames = [_rotation_rows_for_candidate(data, mode="train_temp_Qfull", min_prefix=None, shrink=None, margin=None, alpha=None, update_period=None)]
    for prefix in [128, 256, 512]:
        for shrink in [0.50, 0.75]:
            for margin in [0.5, 1.0, 2.0]:
                frames.append(
                    _rotation_rows_for_candidate(
                        data,
                        mode="fixed_prefix_twophase",
                        min_prefix=prefix,
                        shrink=shrink,
                        margin=margin,
                        alpha=None,
                        update_period=None,
                    )
                )
                for alpha in [0.25, 0.50, 1.0]:
                    frames.append(
                        _rotation_rows_for_candidate(
                            data,
                            mode="online_bounded_vrange",
                            min_prefix=prefix,
                            shrink=shrink,
                            margin=margin,
                            alpha=alpha,
                            update_period=32,
                        )
                    )

    rows = pd.concat(frames, ignore_index=True)
    summary = (
        rows.groupby(["mode", "feature", "min_prefix", "shrink", "margin", "alpha", "update_period"], dropna=False)
        .agg(
            pass_count=("target_met", "sum"),
            mean_target_norm_worst=("target_norm_worst", "mean"),
            max_target_norm_worst=("target_norm_worst", "max"),
            mean_0C_MAE_pct=("0C_MAE_pct", "mean"),
            mean_25C_MAE_pct=("25C_MAE_pct", "mean"),
            mean_45C_MAE_pct=("45C_MAE_pct", "mean"),
        )
        .reset_index()
        .sort_values(["pass_count", "mean_target_norm_worst", "max_target_norm_worst"], ascending=[False, True, True])
        .reset_index(drop=True)
    )

    best_parts = []
    for mode in ["train_temp_Qfull", "fixed_prefix_twophase", "online_bounded_vrange"]:
        best = summary[summary["mode"].eq(mode)].iloc[0]
        mask = rows["mode"].eq(mode)
        for col in ["min_prefix", "shrink", "margin", "alpha", "update_period"]:
            mask &= rows[col].astype(str).eq(str(best[col]))
        best_parts.append(rows[mask])
    best_rows = pd.concat(best_parts, ignore_index=True)

    rows.to_csv(cfg.output_dir / f"{cfg.output_prefix}_rotation_rows.csv", index=False)
    summary.to_csv(cfg.output_dir / f"{cfg.output_prefix}_summary.csv", index=False)
    best_rows.to_csv(cfg.output_dir / f"{cfg.output_prefix}_best_rows.csv", index=False)
    write_report(cfg, summary, best_rows)
    print(summary.head(20).to_string(index=False))
    return {"rows": rows, "summary": summary, "best_rows": best_rows}


def write_report(cfg: OnlineCapacityAnchorConfig, summary: pd.DataFrame, best_rows: pd.DataFrame) -> None:
    best = pd.concat(
        [
            summary[summary["mode"].eq("train_temp_Qfull")].iloc[[0]],
            summary[summary["mode"].eq("fixed_prefix_twophase")].iloc[[0]],
            summary[summary["mode"].eq("online_bounded_vrange")].iloc[[0]],
        ],
        ignore_index=True,
    )
    e1 = best_rows[
        best_rows["test_profile"].eq("FUDS")
        & best_rows["valid_profile"].eq("VALIDATION")
        & best_rows["train_profiles"].eq("DST,US06")
    ].copy()
    lines = [
        "# NMC Online Capacity-Anchor Screen",
        "",
        "## Purpose",
        "",
        "This is not a strict NoCC experiment. It is a causal current-integration observer sanity screen.",
        "",
        "The test trajectory is rolled forward from the file-level initial SOC. Current is integrated explicitly, while `Q_eff` is adapted only from voltage range observed up to the current time.",
        "",
        "No future test voltage is used in the online mode. Full-trajectory descriptors are not used.",
        "",
        "## Best Settings By Mode",
        "",
        best.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## E1 FUDS Diagnostic Rows",
        "",
        e1.to_markdown(index=False, floatfmt=".3f") if not e1.empty else "(missing)",
        "",
        "## Best Rotation Details",
        "",
        best_rows.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Interpretation",
        "",
        "- This screen tests whether causal voltage-range adaptation is enough before adding a neural observer.",
        "- Passing E1 alone is not sufficient; the main criterion is profile rotation.",
        "- If the online mode fails profile rotation, the next observer needs stronger causal regime features or an abstention/coverage gate.",
        "",
    ]
    (cfg.output_dir / f"{cfg.output_prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Causal online v_range capacity-anchor profile-rotation screen.")
    parser.add_argument("--raw-root", default=OnlineCapacityAnchorConfig.raw_root)
    parser.add_argument("--output-dir", default=OnlineCapacityAnchorConfig.output_dir)
    parser.add_argument("--output-prefix", default=OnlineCapacityAnchorConfig.output_prefix)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        OnlineCapacityAnchorConfig(
            raw_root=Path(args.raw_root),
            output_dir=Path(args.output_dir),
            output_prefix=str(args.output_prefix),
        )
    )


if __name__ == "__main__":
    main()
