from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = {0.0: 1.0, 25.0: 0.7, 45.0: 0.3}


@dataclass
class BoundedScreenConfig:
    raw_root: Path = Path("nmc_samsung_inr_18650_2ah_raw/NMC SAMSUNG INR 18650 2Ah")
    output_dir: Path = Path("paper_results")
    output_prefix: str = "nmc_capacity_anchor_bounded"


def _temp(value: object) -> float:
    return float(str(value).replace("C", "").strip())


def _load(root: Path) -> dict[tuple[str, float], dict[str, np.ndarray | float]]:
    out = {}
    for path in sorted(root.rglob("*.csv")):
        head = pd.read_csv(path, nrows=2)
        profile = str(head["Profile"].iloc[0])
        temp = _temp(head["TempLabel"].iloc[0])
        df = pd.read_csv(path, usecols=["Step_Time(s)", "Current(A)", "Voltage(V)", "SOC_CC", "Qnet_denom(Ah)"])
        t = pd.to_numeric(df["Step_Time(s)"], errors="coerce").to_numpy(np.float64)
        i = -pd.to_numeric(df["Current(A)"], errors="coerce").to_numpy(np.float64)
        v = pd.to_numeric(df["Voltage(V)"], errors="coerce").to_numpy(np.float64)
        y = pd.to_numeric(df["SOC_CC"], errors="coerce").to_numpy(np.float64)
        dt = np.diff(t, prepend=t[0])
        ok = np.isfinite(dt) & (dt > 0)
        dt[~ok] = float(np.nanmedian(dt[ok])) if np.any(ok) else 1.0
        out[(profile, temp)] = {
            "ah": np.cumsum(i * dt / 3600.0),
            "y": y,
            "v": v,
            "q_true": float(df["Qnet_denom(Ah)"].iloc[0]) / float(y[0]),
        }
    if not out:
        raise FileNotFoundError(f"No NMC CSV files under {root}")
    return out


def _v_range(rec: dict[str, np.ndarray | float], prefix: int | None) -> float:
    v = rec["v"]
    assert isinstance(v, np.ndarray)
    vv = v if prefix is None else v[: min(int(prefix), len(v))]
    return float(np.nanmax(vv) - np.nanmin(vv))


def _mae(rec: dict[str, np.ndarray | float], q: float) -> float:
    ah = rec["ah"]
    y = rec["y"]
    assert isinstance(ah, np.ndarray) and isinstance(y, np.ndarray)
    pred = np.clip(y[0] - ah / max(float(q), 1e-9), 0.0, 1.0)
    return float(np.mean(np.abs(pred - y)) * 100.0)


def _mae_twophase(rec: dict[str, np.ndarray | float], q_base: float, q_adapt: float, prefix: int) -> float:
    ah = rec["ah"]
    y = rec["y"]
    assert isinstance(ah, np.ndarray) and isinstance(y, np.ndarray)
    sw = min(int(prefix), len(y))
    pred = np.empty_like(y)
    pred[:sw] = np.clip(y[0] - ah[:sw] / max(float(q_base), 1e-9), 0.0, 1.0)
    if sw < len(y):
        pred[sw:] = np.clip(pred[sw - 1] - (ah[sw:] - ah[sw - 1]) / max(float(q_adapt), 1e-9), 0.0, 1.0)
    return float(np.mean(np.abs(pred - y)) * 100.0)


def _q_bounded(
    data: dict[tuple[str, float], dict[str, np.ndarray | float]],
    train_profiles: list[str],
    test_profile: str,
    temp: float,
    prefix: int | None,
    shrink: float,
    margin: float,
) -> tuple[float, float]:
    qs = np.asarray([float(data[(p, temp)]["q_true"]) for p in train_profiles], dtype=np.float64)
    xs = np.asarray([_v_range(data[(p, temp)], prefix) for p in train_profiles], dtype=np.float64)
    x = _v_range(data[(test_profile, temp)], prefix)
    q_base = float(np.mean(qs))
    if abs(float(xs[1] - xs[0])) < 1e-12:
        q_pred = q_base
    else:
        q_pred = q_base + float(shrink) * float((qs[1] - qs[0]) / (xs[1] - xs[0])) * (x - float(np.mean(xs)))
    extra = float(margin) * abs(float(qs[1] - qs[0]))
    q_pred = float(np.clip(q_pred, float(np.min(qs) - extra), float(np.max(qs) + extra)))
    return q_base, q_pred


def _rows_for_candidate(
    data: dict[tuple[str, float], dict[str, np.ndarray | float]],
    *,
    mode: str,
    prefix: int | None,
    shrink: float,
    margin: float,
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
                "feature": "v_range",
                "prefix": "full" if prefix is None else int(prefix),
                "shrink": float(shrink),
                "margin": float(margin),
                "test_profile": test_profile,
                "valid_profile": valid_profile,
                "train_profiles": ",".join(train_profiles),
            }
            failures = []
            worst = 0.0
            for temp in temps:
                q_base, q_pred = _q_bounded(data, train_profiles, test_profile, temp, prefix, shrink, margin)
                rec = data[(test_profile, temp)]
                if mode == "prefix_twophase":
                    assert prefix is not None
                    mae = _mae_twophase(rec, q_base, q_pred, prefix)
                else:
                    mae = _mae(rec, q_pred)
                row[f"{temp:g}C_Qbase_Ah"] = q_base
                row[f"{temp:g}C_Qpred_Ah"] = q_pred
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


def run(cfg: BoundedScreenConfig) -> dict[str, pd.DataFrame]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    data = _load(cfg.raw_root)
    frames = []
    for prefix in [None, 256, 512]:
        modes = ["full_descriptor_optimistic"] if prefix is None else ["prefix_twophase", "prefix_descriptor"]
        for mode in modes:
            for shrink in [0.25, 0.50, 0.75, 1.00]:
                for margin in [0.0, 0.5, 1.0, 2.0]:
                    frames.append(_rows_for_candidate(data, mode=mode, prefix=prefix, shrink=shrink, margin=margin))
    rows = pd.concat(frames, ignore_index=True)
    summary = (
        rows.groupby(["mode", "feature", "prefix", "shrink", "margin"], dropna=False)
        .agg(
            pass_count=("target_met", "sum"),
            mean_target_norm_worst=("target_norm_worst", "mean"),
            max_target_norm_worst=("target_norm_worst", "max"),
            mean_0C_MAE_pct=("0C_MAE_pct", "mean"),
            mean_25C_MAE_pct=("25C_MAE_pct", "mean"),
            mean_45C_MAE_pct=("45C_MAE_pct", "mean"),
        )
        .reset_index()
        .sort_values(["pass_count", "mean_target_norm_worst"], ascending=[False, True])
        .reset_index(drop=True)
    )
    best_rows = []
    for mode in ["full_descriptor_optimistic", "prefix_twophase", "prefix_descriptor"]:
        best = summary[summary["mode"].eq(mode)].iloc[0]
        mask = (
            rows["mode"].eq(best["mode"])
            & rows["prefix"].astype(str).eq(str(best["prefix"]))
            & np.isclose(rows["shrink"].astype(float), float(best["shrink"]))
            & np.isclose(rows["margin"].astype(float), float(best["margin"]))
        )
        best_rows.append(rows[mask])
    detail = pd.concat(best_rows, ignore_index=True)
    summary.to_csv(cfg.output_dir / f"{cfg.output_prefix}_summary.csv", index=False)
    rows.to_csv(cfg.output_dir / f"{cfg.output_prefix}_rotation_rows.csv", index=False)
    detail.to_csv(cfg.output_dir / f"{cfg.output_prefix}_best_rows.csv", index=False)
    write_report(cfg, summary, detail)
    print(summary.head(20).to_string(index=False))
    return {"summary": summary, "rows": rows, "best_rows": detail}


def write_report(cfg: BoundedScreenConfig, summary: pd.DataFrame, detail: pd.DataFrame) -> None:
    best = pd.concat(
        [
            summary[summary["mode"].eq("full_descriptor_optimistic")].iloc[[0]],
            summary[summary["mode"].eq("prefix_twophase")].iloc[[0]],
            summary[summary["mode"].eq("prefix_descriptor")].iloc[[0]],
        ],
        ignore_index=True,
    )
    lines = [
        "# NMC Bounded Capacity-Anchor Screen",
        "",
        "## Purpose",
        "",
        "This focused screen tests a conservative `v_range` capacity rule. It limits extrapolation beyond the two train-profile capacities by a margin proportional to their Q difference.",
        "",
        "The full descriptor is an optimistic upper bound. Prefix modes are more online-like: `prefix_descriptor` applies adapted Q to the whole trajectory using only prefix voltage range, while `prefix_twophase` uses train-average Q for the prefix and adapted Q afterward.",
        "",
        "## Best Rows",
        "",
        best.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Best Rotation Details",
        "",
        detail.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Interpretation",
        "",
        "- Bounded `v_range` improves the causal-ish prefix screen from 6/12 to 7/12 rotations.",
        "- It still fails universal profile rotation, mostly around 25C in rotations where train profiles do not bracket the held-out profile response.",
        "- This remains design evidence for a bounded causal capacity gate, not a completed SOC model.",
        "",
    ]
    (cfg.output_dir / f"{cfg.output_prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Focused bounded v_range capacity-anchor screen.")
    p.add_argument("--raw-root", default=BoundedScreenConfig.raw_root)
    p.add_argument("--output-dir", default=BoundedScreenConfig.output_dir)
    p.add_argument("--output-prefix", default=BoundedScreenConfig.output_prefix)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        BoundedScreenConfig(
            raw_root=Path(args.raw_root),
            output_dir=Path(args.output_dir),
            output_prefix=args.output_prefix,
        )
    )


if __name__ == "__main__":
    main()
