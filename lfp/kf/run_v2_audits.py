#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import PchipInterpolator

from src.data_io import load_all
from src.hysteresis_model import propagate_hysteresis
from src.lfp_ecm import OCVMap


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_v2" / "audits"


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False]) + 1
    return list(zip(starts.tolist(), ends.tolist()))


class RawBranches:
    def __init__(self, table: pd.DataFrame):
        self.temperatures = np.sort(table.temperature_C.unique().astype(float))
        self.curves: dict[float, dict[str, PchipInterpolator]] = {}
        for temp in self.temperatures:
            d = table[table.temperature_C == temp].sort_values("soc")
            self.curves[float(temp)] = {
                "discharge": PchipInterpolator(d.soc, d.ocv_discharge_raw_V),
                "charge": PchipInterpolator(d.soc, d.ocv_charge_raw_V),
            }

    def evaluate(self, soc: float, temperature_C: float, branch: str) -> float:
        soc = float(np.clip(soc, 0.0, 1.0))
        temperature_C = float(np.clip(temperature_C, self.temperatures[0], self.temperatures[-1]))
        hi = int(np.searchsorted(self.temperatures, temperature_C))
        if hi == 0:
            lo_t = hi_t = float(self.temperatures[0]); weight = 0.0
        elif hi == len(self.temperatures):
            lo_t = hi_t = float(self.temperatures[-1]); weight = 0.0
        else:
            lo_t = float(self.temperatures[hi - 1]); hi_t = float(self.temperatures[hi])
            weight = (temperature_C - lo_t) / (hi_t - lo_t)
        lo = float(self.curves[lo_t][branch](soc))
        high = lo if lo_t == hi_t else float(self.curves[hi_t][branch](soc))
        return (1.0 - weight) * lo + weight * high


def alignment_audit(trajectories, branches: RawBranches) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for tr in trajectories:
        for start, end in contiguous_runs(np.abs(tr.current_A) < 0.02):
            duration = float(tr.time_s[end - 1] - tr.time_s[start] + tr.dt_s[end - 1])
            if duration < 120.0:
                continue
            idx = end - 1
            soc = float(tr.soc_ref[idx]); temp = float(tr.temperature_series_C[idx])
            v_rest = float(tr.voltage_V[idx])
            v_dis = branches.evaluate(soc, temp, "discharge")
            v_chg = branches.evaluate(soc, temp, "charge")
            rows.append({
                "profile": tr.profile,
                "temperature_C": tr.temperature_C,
                "start_index": start,
                "end_index": idx,
                "rest_duration_s": duration,
                "soc_label": soc,
                "V_rest_V": v_rest,
                "OCV_discharge_V": v_dis,
                "OCV_charge_V": v_chg,
                "r_dis_V": v_rest - v_dis,
                "r_chg_V": v_rest - v_chg,
            })
    event_columns = [
        "profile", "temperature_C", "start_index", "end_index", "rest_duration_s",
        "soc_label", "V_rest_V", "OCV_discharge_V", "OCV_charge_V", "r_dis_V", "r_chg_V",
    ]
    events = pd.DataFrame(rows, columns=event_columns)
    summaries = []
    for temp in sorted({tr.temperature_C for tr in trajectories}):
        g = events[events.temperature_C == temp]
        row = {"temperature_C": temp, "n_rest_ends": len(g)}
        for residual in ("r_dis_V", "r_chg_V"):
            values = 1000.0 * g[residual].to_numpy(float)
            row[f"{residual}_median_mV"] = float(np.median(values)) if len(values) else np.nan
            row[f"{residual}_q1_mV"] = float(np.quantile(values, 0.25)) if len(values) else np.nan
            row[f"{residual}_q3_mV"] = float(np.quantile(values, 0.75)) if len(values) else np.nan
            row[f"{residual}_iqr_mV"] = row[f"{residual}_q3_mV"] - row[f"{residual}_q1_mV"]
            row[f"{residual}_systematic_offset"] = bool(abs(row[f"{residual}_median_mV"]) > 1.0) if len(values) else False
            row[f"{residual}_assessment"] = "FLAG" if len(values) and row[f"{residual}_systematic_offset"] else ("PASS" if len(values) else "NOT_ASSESSABLE_NO_QUALIFYING_RESTS")
        summaries.append(row)
    return events, pd.DataFrame(summaries)


def hysteresis_ratio_audit(table: pd.DataFrame, ocv: OCVMap) -> pd.DataFrame:
    rows = []
    soc_grid = np.linspace(0.0, 1.0, 101)
    for temp in sorted(table.temperature_C.unique()):
        d = table[table.temperature_C == temp].sort_values("soc")
        raw_half = 0.5 * np.abs(
            np.interp(soc_grid, d.soc, d.ocv_charge_raw_V)
            - np.interp(soc_grid, d.soc, d.ocv_discharge_raw_V)
        )
        model_m = np.asarray(ocv.hmag(soc_grid, temp), dtype=float)
        ratio = np.divide(model_m, raw_half, out=np.full_like(model_m, np.nan), where=raw_half > 1e-9)
        for soc, raw, model, value in zip(soc_grid, raw_half, model_m, ratio):
            rows.append({
                "temperature_C": float(temp), "soc": float(soc),
                "table_half_gap_V": float(raw), "model_M_V": float(model),
                "M_to_table_half_gap_ratio": float(value),
            })
    return pd.DataFrame(rows)


def sign_unit_audit(table: pd.DataFrame, ocv: OCVMap) -> dict:
    q_ref = 1.0; gamma = 30.0; dt_s = 1.0; current_A = 1.0; steps = 36000
    checks = []
    for temp in sorted(table.temperature_C.unique()):
        for direction, current, target in (("discharge", current_A, -1.0), ("charge", -current_A, 1.0)):
            h = 0.0
            for _ in range(steps):
                h, _ = propagate_hysteresis(h, current, dt_s, q_ref, gamma)
            soc = 0.5
            mid = float(ocv.ocv(soc, temp)); magnitude = float(ocv.hmag(soc, temp))
            modeled_ocv = mid + magnitude * h
            discharge_ocv = mid - magnitude; charge_ocv = mid + magnitude
            target_ocv = discharge_ocv if direction == "discharge" else charge_ocv
            wrong_ocv = charge_ocv if direction == "discharge" else discharge_ocv
            passed = bool(np.sign(h) == np.sign(target) and abs(modeled_ocv - target_ocv) < abs(modeled_ocv - wrong_ocv))
            checks.append({
                "temperature_C": float(temp), "direction": direction,
                "final_h": float(h), "expected_h": target,
                "modeled_ocv_V": modeled_ocv, "target_branch_ocv_V": target_ocv,
                "opposite_branch_ocv_V": wrong_ocv, "passed": passed,
            })
    if not all(row["passed"] for row in checks):
        raise RuntimeError(f"A3 sign/unit test failed: {checks}")
    return {"status": "PASS", "current_convention": "discharge_positive_internal", "checks": checks}


def save_plots(events: pd.DataFrame, summary: pd.DataFrame, ratio: pd.DataFrame) -> None:
    temps = summary.temperature_C.to_numpy(float)
    fig, ax = plt.subplots(figsize=(9, 5))
    if len(events):
        for x, col, color, label in (
            (-0.12, "r_dis_V", "#2f6f9f", "discharge branch"),
            (0.12, "r_chg_V", "#d18f25", "charge branch"),
        ):
            grouped = [1000 * events.loc[events.temperature_C == t, col].to_numpy(float) for t in temps]
            positions = np.arange(len(temps)) + x
            bp = ax.boxplot(grouped, positions=positions, widths=0.20, patch_artist=True, showfliers=False)
            for box in bp["boxes"]: box.set(facecolor=color, alpha=0.55, edgecolor=color)
            for item in bp["medians"]: item.set(color="#222222", linewidth=1.2)
            ax.plot([], [], color=color, linewidth=8, alpha=0.55, label=label)
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No |I| < 0.02 A rest segment reached 120 s", transform=ax.transAxes,
                ha="center", va="center", fontsize=12)
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=0.9)
    ax.axhline(-1.0, color="#555555", linestyle="--", linewidth=0.9)
    ax.set_xticks(np.arange(len(temps)), [f"{t:g}" for t in temps])
    ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Rest-end residual (mV)")
    ax.set_title("Rest-end voltage minus tabulated OCV branch")
    fig.tight_layout()
    fig.savefig(OUT / "a1_ocv_label_alignment.png", dpi=180); plt.close(fig)

    pivot = ratio.pivot(index="temperature_C", columns="soc", values="M_to_table_half_gap_ratio")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=2)
    ax.set_yticks(np.arange(len(pivot.index)), [f"{t:g}" for t in pivot.index])
    ticks = np.linspace(0, len(pivot.columns) - 1, 6).astype(int)
    ax.set_xticks(ticks, [f"{100*pivot.columns[i]:.0f}" for i in ticks])
    ax.set_xlabel("SOC (%)"); ax.set_ylabel("Temperature (°C)")
    ax.set_title("Model hysteresis magnitude / raw OCV half-gap")
    fig.colorbar(image, ax=ax, label="ratio (clipped display at 2)")
    fig.tight_layout(); fig.savefig(OUT / "a2_hysteresis_ratio.png", dpi=180); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "configs" / "remote.yaml").read_text())
    trajectories = load_all(cfg)
    table = pd.read_csv(ROOT / "artifacts" / "ocv_table.csv")
    branches = RawBranches(table); ocv = OCVMap(table, "monotonic")
    events, summary = alignment_audit(trajectories, branches)
    ratio = hysteresis_ratio_audit(table, ocv)
    sign = sign_unit_audit(table, ocv)
    events.to_csv(OUT / "a1_rest_end_residuals.csv", index=False)
    summary.to_csv(OUT / "a1_temperature_summary.csv", index=False)
    ratio.to_csv(OUT / "a2_hysteresis_ratio.csv", index=False)
    (OUT / "a3_sign_unit_test.json").write_text(json.dumps(sign, indent=2) + "\n")
    save_plots(events, summary, ratio)
    chart_notes = """# Audit chart map

| section | question | form | fields | supported claim | palette |
| --- | --- | --- | --- | --- | --- |
| A1 | Do rest-end voltages align to either OCV branch within 1 mV median? | grouped box plot | temperature, branch residual mV | median and spread of label/OCV alignment | blue/gold plus neutral thresholds |
| A2 | Does model M match the tabulated branch half-gap across SOC and temperature? | heatmap | SOC, temperature, ratio | systematic over/under-scaling of hysteresis magnitude | single viridis scale |
"""
    (OUT / "chart_map.md").write_text(chart_notes)
    result = {
        "status": "PASS" if len(events) else "PASS_WITH_A1_NOT_ASSESSABLE",
        "records_checked": len(trajectories),
        "rest_ends_found": len(events),
        "temperatures_flagged_discharge": summary.loc[summary.r_dis_V_systematic_offset, "temperature_C"].tolist(),
        "temperatures_flagged_charge": summary.loc[summary.r_chg_V_systematic_offset, "temperature_C"].tolist(),
        "sign_unit_test": sign["status"],
    }
    (OUT / "audit_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
