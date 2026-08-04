from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def _causal_time_ema(values: np.ndarray, times_s: np.ndarray, tau_s: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    t = np.asarray(times_s, dtype=np.float64)
    if len(x) == 0:
        return x.astype(np.float32)
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    if len(x) == 1:
        return y.astype(np.float32)
    delta = np.diff(t)
    positive = delta[np.isfinite(delta) & (delta > 0)]
    dt_default = float(np.nanmedian(positive)) if len(positive) else 1.0
    if not np.isfinite(dt_default) or dt_default <= 0:
        dt_default = 1.0
    for index in range(1, len(x)):
        dt = t[index] - t[index - 1]
        if not np.isfinite(dt) or dt <= 0:
            dt = dt_default
        alpha = float(np.exp(-dt / max(float(tau_s), 1e-6)))
        alpha = min(max(alpha, 0.0), 0.999999)
        y[index] = alpha * y[index - 1] + (1.0 - alpha) * x[index]
    return y.astype(np.float32)


def _causal_index_ema(values: np.ndarray, tau: int) -> np.ndarray:
    # The final V/I feature pass overwrites earlier EMA columns using float64
    # recurrence, then stores the completed series as float32.
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return array.astype(np.float32)
    alpha = float(np.exp(-1.0 / max(float(tau), 1e-6)))
    alpha = min(max(alpha, 0.0), 0.999999)
    output = np.empty_like(array, dtype=np.float64)
    output[0] = array[0]
    for index in range(1, len(array)):
        output[index] = alpha * output[index - 1] + (1.0 - alpha) * array[index]
    return output.astype(np.float32)


def _temperature_from_frame(frame: pd.DataFrame) -> float:
    for column in ("T", "temperature_C", "temperature"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
            if len(values) == 1:
                return float(values[0])
    if "TempLabel" in frame.columns:
        values = frame["TempLabel"].dropna().astype(str).unique()
        if len(values) == 1:
            match = re.search(r"(-?\d+(?:\.\d+)?)", values[0])
            if match:
                return float(match.group(1))
    raise ValueError("df_record must contain one constant temperature in T, temperature_C, temperature, or TempLabel")


def _time_from_frame(frame: pd.DataFrame, nominal_period_s: float) -> np.ndarray:
    for column in ("Step_Time(s)", "Test_Time(s)", "time_s"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
            if np.isfinite(values).all():
                return values
    return np.arange(len(frame), dtype=np.float64) * float(nominal_period_s)


def _r0_for_temperature(r0_table: dict, temperature_C: float) -> float:
    rows = sorted(r0_table["table"], key=lambda row: float(row["temperature_C"]))
    temperatures = np.asarray([float(row["temperature_C"]) for row in rows], dtype=np.float64)
    values = np.asarray([float(row["r0_ohm"]) for row in rows], dtype=np.float64)
    if np.any(np.isclose(temperatures, float(temperature_C), rtol=0.0, atol=1e-8)):
        return float(values[np.flatnonzero(np.isclose(temperatures, float(temperature_C), rtol=0.0, atol=1e-8))[0]])
    return float(np.interp(float(temperature_C), temperatures, values))


def _normalize_reset_indices(length: int, reset_indices) -> list[int]:
    if reset_indices is None:
        return [0]
    values = [int(value) for value in reset_indices]
    if any(value < 0 or value >= int(length) for value in values):
        raise ValueError(f"reset_indices must be within [0, {length})")
    if len(values) != len(set(values)):
        raise ValueError("reset_indices contains duplicates")
    return sorted(set([0] + values))


def _perturbation(noise, length: int, name: str) -> np.ndarray:
    if noise is None:
        return np.zeros(length, dtype=np.float64)
    array = np.asarray(noise, dtype=np.float64)
    if array.ndim == 0:
        return np.full(length, float(array), dtype=np.float64)
    if array.shape != (length,):
        raise ValueError(f"{name} must be None, a scalar, or shape ({length},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _segment_features(
    voltage: np.ndarray,
    current: np.ndarray,
    times_s: np.ndarray,
    temperature_C: float,
    r0_ohm: float,
    channels: list[str],
    vcorr_tau_s: float,
) -> np.ndarray:
    v_raw = np.asarray(voltage, dtype=np.float64)
    i_raw = np.asarray(current, dtype=np.float64)
    v_ohm = (i_raw * float(r0_ohm)).astype(np.float32)
    v_ohm_removed = v_raw - v_ohm
    v_corr = _causal_time_ema(v_ohm_removed, times_s, float(vcorr_tau_s))
    values: dict[str, np.ndarray] = {
        "V_corr_raw": v_corr.astype(np.float32),
        "I_raw": i_raw.astype(np.float32),
        "T": np.full(len(v_raw), float(temperature_C), dtype=np.float32),
    }
    for base, taus in (("V_corr_raw", (50, 200, 800)), ("I_raw", (50, 200)), ("absI", (50, 200))):
        raw = np.abs(i_raw).astype(np.float32) if base == "absI" else values[base].astype(np.float32)
        for tau in taus:
            ema = _causal_index_ema(raw, tau)
            values[f"{base}_ema{tau}"] = ema
            values[f"{base}_dev_ema{tau}"] = (raw - ema).astype(np.float32)
    missing = [channel for channel in channels if channel not in values]
    if missing:
        raise RuntimeError(f"Frozen feature builder does not implement channels: {missing}")
    return np.ascontiguousarray(np.column_stack([values[channel] for channel in channels]), dtype=np.float32)


def build_feature_matrix(
    frame: pd.DataFrame,
    config: dict,
    r0_table: dict,
    reset_indices=None,
    r0_scale: float = 1.0,
    v_noise=None,
    i_noise=None,
    v_bias: float = 0.0,
    i_bias: float = 0.0,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("df_record must be a pandas.DataFrame")
    if len(frame) == 0:
        return np.empty((0, len(config["channels"])), dtype=np.float32), []
    required = ["Voltage(V)", "Current(A)"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"df_record is missing raw columns: {missing}")
    voltage = pd.to_numeric(frame["Voltage(V)"], errors="coerce").to_numpy(np.float64)
    current = pd.to_numeric(frame["Current(A)"], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(voltage).all() or not np.isfinite(current).all():
        raise ValueError("Raw voltage/current contain non-finite values")
    voltage = voltage + float(v_bias) + _perturbation(v_noise, len(frame), "v_noise")
    current = current + float(i_bias) + _perturbation(i_noise, len(frame), "i_noise")
    temperature = _temperature_from_frame(frame)
    times = _time_from_frame(frame, float(config["sampling_period_s"]["nominal"]))
    r0 = _r0_for_temperature(r0_table, temperature) * float(r0_scale)
    resets = _normalize_reset_indices(len(frame), reset_indices)
    boundaries = resets + [len(frame)]
    matrix = np.empty((len(frame), len(config["channels"])), dtype=np.float32)
    segments = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop <= start:
            continue
        matrix[start:stop] = _segment_features(
            voltage[start:stop],
            current[start:stop],
            times[start:stop],
            temperature,
            r0,
            list(config["channels"]),
            float(config["vcorr_tau_s"]),
        )
        segments.append((start, stop))
    return matrix, segments


class _DynamicGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.rnn = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=float(dropout) if layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.base_head = nn.Linear(hidden_size, 1)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.input_proj(x)
        output, _ = self.rnn(projected)
        return self.norm(output)


class _AnchorResidualGRU(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        hidden_size = int(config["hidden_size"])
        dropout = float(config["dropout"])
        self.residual_limit_mode = "learnable"
        self.residual_limit_param = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.dynamic = _DynamicGRU(
            input_dim=len(config["channels"]),
            hidden_size=hidden_size,
            layers=int(config["layers"]),
            dropout=dropout,
        )
        self.anchor_indices = list(config["anchor_indices"])
        self.anchor_head = nn.Sequential(
            nn.Linear(len(self.anchor_indices), hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        index = torch.as_tensor(self.anchor_indices, device=x.device)
        anchor = torch.sigmoid(self.anchor_head(x.index_select(dim=2, index=index)))
        hidden = self.dynamic.encode_sequence(x)
        residual = self.residual_limit_param.to(hidden.dtype) * torch.tanh(self.residual_head(hidden))
        return (anchor + residual).clamp(0.0, 1.0)[:, -1, :]


class FrozenPackage:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.config = json.loads((self.path / "config.json").read_text(encoding="utf-8"))
        self.scaler = json.loads((self.path / "scaler.json").read_text(encoding="utf-8"))
        self.r0_table = json.loads((self.path / "r0_table.json").read_text(encoding="utf-8"))
        self.manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        self.mean = np.asarray(self.scaler["mean"], dtype=np.float32)
        self.std = np.asarray(self.scaler["std"], dtype=np.float32)
        if self.mean.shape != (len(self.config["channels"]),) or self.std.shape != self.mean.shape:
            raise RuntimeError("Scaler/channel shape mismatch")
        weights_path = self.path / self.config["weights_file"]
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        if list(checkpoint["feature_cols"]) != list(self.config["channels"]):
            raise RuntimeError("Checkpoint feature columns do not match config.json")
        self.model = _AnchorResidualGRU(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()

    def predict(
        self,
        df_record,
        reset_indices=None,
        r0_scale=1.0,
        v_noise=None,
        i_noise=None,
        v_bias=0.0,
        i_bias=0.0,
    ) -> np.ndarray:
        features, segments = build_feature_matrix(
            df_record,
            self.config,
            self.r0_table,
            reset_indices=reset_indices,
            r0_scale=r0_scale,
            v_noise=v_noise,
            i_noise=i_noise,
            v_bias=v_bias,
            i_bias=i_bias,
        )
        output = np.full(len(df_record), np.nan, dtype=np.float32)
        if len(features) == 0:
            return output
        scaled = np.ascontiguousarray((features - self.mean) / self.std, dtype=np.float32)
        window = int(self.config["window"])
        batch_size = int(self.config.get("inference_batch_size", 2048))
        with torch.inference_mode():
            for start, stop in segments:
                if stop - start < window:
                    continue
                tensor = torch.from_numpy(scaled[start:stop])
                windows = tensor.unfold(0, window, 1).permute(0, 2, 1).contiguous()
                predictions = []
                for batch_start in range(0, len(windows), batch_size):
                    predictions.append(self.model(windows[batch_start : batch_start + batch_size]).cpu())
                values = torch.cat(predictions, dim=0).numpy()[:, 0].astype(np.float32, copy=False)
                output[start + window - 1 : stop] = values
        return output


def load_package(path) -> FrozenPackage:
    """Load one frozen feature/fold/seed package directory."""
    return FrozenPackage(path)
