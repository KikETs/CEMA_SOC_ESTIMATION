from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .config import CFG, make_cfg
from .runtime import configure_torch_runtime, device
from .data import FeatureStandardizer
from .models import collate_meta_to_frame
from .training import attach_prediction_features, build_prediction_feature_lookup
from .variance_control import R5_GATED_FEATURES, variance_by_temperature, _overall_metrics
from .extrapolation_robustness import (
    EXP_SPECS,
    exp_cfg,
    load_filtered_feature_frames,
    temp_key_to_c,
)

try:
    from IPython.display import display
except Exception:
    display = print


FEATURES = R5_GATED_FEATURES
COMPONENTS = ["V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0"]


@dataclass
class RobustSpec:
    name: str
    thermo: bool = False
    correction_limit: float = 0.05
    lambda_rex: float = 0.5
    lambda_worst: float = 0.2
    lambda_selfsup: float = 0.05
    lambda_shift_smooth: float = 0.01
    lambda_R0_mono: float = 0.01
    state_dropout: float = 0.05
    component_mask_p: float = 0.10
    shift_mode: str = "hybrid"


class SeqEndpointDataset(Dataset):
    def __init__(self, frames, feature_cols, window_len, stride, target_col="SOC_physical"):
        self.frames = []
        self.feature_cols = list(feature_cols)
        self.window_len = int(window_len)
        self.stride = int(stride)
        self.index = []
        for fi, frame in enumerate(frames):
            f = frame.reset_index(drop=True)
            cache = {
                "x": np.ascontiguousarray(f[self.feature_cols].to_numpy(np.float32)),
                "y": np.ascontiguousarray(f[target_col].to_numpy(np.float32)),
                "file_name": f["file_name"].to_numpy(),
                "trajectory_id": f["trajectory_id"].to_numpy(),
                "end_index": f["end_index"].to_numpy(np.int64),
                "temperature": f["temperature"].to_numpy(np.float32),
                "drive_cycle": f["drive_cycle"].to_numpy(),
            }
            self.frames.append(cache)
            n = len(f)
            if n < self.window_len:
                continue
            for start in range(0, n - self.window_len + 1, self.stride):
                end = start + self.window_len - 1
                y = cache["y"][start:end + 1]
                if np.isfinite(y).all():
                    self.index.append((fi, start, end))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, start, end = self.index[idx]
        f = self.frames[fi]
        x = f["x"][start:end + 1]
        y_seq = f["y"][start:end + 1, None]
        meta = {
            "file_name": f["file_name"][end],
            "trajectory_id": f["trajectory_id"][end],
            "end_index": int(f["end_index"][end]),
            "temperature": float(f["temperature"][end]),
            "drive_cycle": f["drive_cycle"][end],
        }
        return torch.from_numpy(x), torch.from_numpy(y_seq), meta


def scale_frames(feature_frames, feature_cols):
    train_ids = [f["trajectory_id"].iloc[0] for f in feature_frames["train"]]
    scaler = FeatureStandardizer().fit(feature_frames["train"], feature_cols, fit_ids=train_ids)
    test_ids = {f["trajectory_id"].iloc[0] for f in feature_frames["test"]}
    assert scaler.fit_ids.isdisjoint(test_ids), "Scaler leakage: test IDs included in fit"
    return {split: [scaler.transform_frame(f) for f in frames] for split, frames in feature_frames.items()}, scaler


def endpoint_temperatures(ds):
    vals = []
    for fi, _, end in ds.index:
        vals.append(float(ds.frames[fi]["temperature"][end]))
    return np.asarray(vals, dtype=np.float32)


def balanced_loader(ds, cfg: CFG, shuffle=True):
    if not shuffle:
        return DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)
    temps = endpoint_temperatures(ds)
    unique, counts = np.unique(temps, return_counts=True)
    count = {float(t): int(c) for t, c in zip(unique, counts)}
    weights = np.asarray([1.0 / count[float(t)] for t in temps], dtype=np.float64)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(ds, batch_size=int(cfg.batch_size), sampler=sampler, num_workers=0)


class CausalLatentSOC(nn.Module):
    def __init__(self, feature_cols, hidden_size=64, correction_limit=0.05, thermo=False, shift_mode="hybrid", state_dropout=0.05):
        super().__init__()
        self.feature_cols = list(feature_cols)
        self.hidden_size = int(hidden_size)
        self.correction_limit = float(correction_limit)
        self.thermo = bool(thermo)
        self.shift_mode = str(shift_mode)
        self.state_dropout = float(state_dropout)
        self.input_dim = len(feature_cols)
        self.t_idx = self.feature_cols.index("T")
        self.vcorr_idx = self.feature_cols.index("V_corr_raw")
        self.component_indices = [self.feature_cols.index(c) for c in COMPONENTS if c in self.feature_cols]
        self.r0_idx = self.feature_cols.index("R0") if "R0" in self.feature_cols else None
        self.input_proj = nn.Linear(self.input_dim, hidden_size)
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.base_head = nn.Linear(hidden_size, 1)
        self.corr_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, 1))
        self.next_vcorr_head = nn.Linear(hidden_size, 1)
        self.comp_recon_head = nn.Linear(hidden_size, len(self.component_indices))
        self.relax_head = nn.Linear(hidden_size, 1)
        if self.thermo:
            self.tau_raw = nn.Parameter(torch.zeros(hidden_size))
            self.arr_a = nn.Parameter(torch.zeros(hidden_size))
            self.arr_b = nn.Parameter(torch.zeros(hidden_size))
            self.shift_mlp = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, hidden_size))
            self.gain_mlp = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, len(self.component_indices)))
            self.r0_scale_mlp = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 1))
            nn.init.zeros_(self.shift_mlp[-1].weight)
            nn.init.zeros_(self.shift_mlp[-1].bias)
            nn.init.zeros_(self.gain_mlp[-1].weight)
            nn.init.zeros_(self.gain_mlp[-1].bias)

    def _log_aT(self, T):
        if not self.thermo:
            return None
        if self.shift_mode == "arrhenius":
            log_at = self.arr_a.view(1, -1) + self.arr_b.view(1, -1) * (-T)
        elif self.shift_mode == "mlp_bounded":
            log_at = self.shift_mlp(T)
        else:
            log_at = self.arr_a.view(1, -1) + self.arr_b.view(1, -1) * (-T)
            log_at = log_at + 0.25 * torch.tanh(self.shift_mlp(T))
        return log_at.clamp(min=-np.log(5.0), max=np.log(5.0))

    def canonicalize(self, x):
        if not self.thermo or not self.component_indices:
            return x
        T = x[..., self.t_idx:self.t_idx + 1]
        log_gain = torch.tanh(self.gain_mlp(T)) * np.log(2.0)
        by_idx = {}
        for j, idx in enumerate(self.component_indices):
            by_idx[idx] = x[..., idx:idx + 1] * torch.exp(-log_gain[..., j:j + 1])
        if self.r0_idx is not None:
            r0_scale = 0.25 + F.softplus(self.r0_scale_mlp(T))
            by_idx[self.r0_idx] = x[..., self.r0_idx:self.r0_idx + 1] / r0_scale
        cols = []
        for idx in range(x.size(-1)):
            cols.append(by_idx.get(idx, x[..., idx:idx + 1]))
        return torch.cat(cols, dim=-1)

    def encode_sequence(self, x):
        x = self.canonicalize(x)
        B, L, _ = x.shape
        h = x.new_zeros(B, self.hidden_size)
        hs = []
        z = self.input_proj(x)
        for t in range(L):
            h_in = F.dropout(h, p=self.state_dropout, training=self.training) if self.state_dropout > 0 else h
            cand = self.gru(z[:, t, :], h_in)
            if self.thermo:
                log_at = self._log_aT(x[:, t, self.t_idx:self.t_idx + 1])
                tau = (0.25 + F.softplus(self.tau_raw).view(1, -1)) * torch.exp(log_at)
                alpha = torch.exp(-1.0 / tau.clamp(0.25, 4096.0)).clamp(1e-4, 0.9999)
                h = alpha * h + (1.0 - alpha) * cand
            else:
                h = cand
            hs.append(self.norm(h).unsqueeze(1))
        return torch.cat(hs, dim=1)

    def forward_seq(self, x):
        h = self.encode_sequence(x)
        base = torch.sigmoid(self.base_head(h))
        delta = self.correction_limit * torch.tanh(self.corr_head(h))
        return (base + delta).clamp(0.0, 1.0)

    def forward(self, x):
        return self.forward_seq(x)[:, -1, :]

    def selfsup_loss(self, x, h=None, mask_p=0.10):
        if h is None:
            h = self.encode_sequence(x)
        losses = []
        if x.size(1) > 1:
            next_pred = self.next_vcorr_head(h[:, :-1, :])
            losses.append(F.smooth_l1_loss(next_pred, x[:, 1:, self.vcorr_idx:self.vcorr_idx + 1], beta=0.05))
        if self.component_indices:
            comp_true = x[..., self.component_indices]
            comp_pred = self.comp_recon_head(h)
            mask = (torch.rand_like(comp_true) < float(mask_p)).float()
            denom = mask.sum().clamp(min=1.0)
            losses.append((F.smooth_l1_loss(comp_pred, comp_true, beta=0.05, reduction="none") * mask).sum() / denom)
        if "absI" in self.feature_cols and x.size(1) > 1:
            abs_idx = self.feature_cols.index("absI")
            rest = (x[:, :-1, abs_idx:abs_idx + 1].abs() < 0.2).float()
            if rest.sum() > 0:
                relax_target = x[:, 1:, self.vcorr_idx:self.vcorr_idx + 1] - x[:, :-1, self.vcorr_idx:self.vcorr_idx + 1]
                relax_pred = self.relax_head(h[:, :-1, :])
                losses.append((F.smooth_l1_loss(relax_pred, relax_target, beta=0.05, reduction="none") * rest).sum() / rest.sum().clamp(min=1.0))
        return torch.stack(losses).mean() if losses else x.new_tensor(0.0)

    def shift_smoothness_loss(self):
        if not self.thermo:
            return next(self.parameters()).new_tensor(0.0)
        grid = torch.linspace(-1.2, 1.2, 64, device=next(self.parameters()).device).view(-1, 1)
        log_at = self._log_aT(grid)
        return torch.abs(log_at[1:] - log_at[:-1]).mean()

    def r0_mono_loss(self):
        if not self.thermo or self.r0_idx is None:
            return next(self.parameters()).new_tensor(0.0)
        grid = torch.linspace(-1.2, 1.2, 64, device=next(self.parameters()).device).view(-1, 1)
        scale = 0.25 + F.softplus(self.r0_scale_mlp(grid))
        return F.relu(scale[1:] - scale[:-1]).mean()


def robust_loss(model, x, y_seq, meta, spec: RobustSpec):
    pred_seq = model.forward_seq(x)
    y_last = y_seq[:, -1, :]
    pred_last = pred_seq[:, -1, :]
    temps = meta["temperature"]
    temps = temps.to(device=pred_last.device, dtype=torch.float32) if torch.is_tensor(temps) else torch.as_tensor(temps, device=pred_last.device, dtype=torch.float32)
    losses = []
    for t in torch.unique(temps):
        mask = temps == t
        if mask.any():
            losses.append(F.l1_loss(pred_last[mask], y_last[mask]))
    if losses:
        domain = torch.stack(losses)
        l_soc = domain.mean()
        l_rex = domain.var(unbiased=False) if len(losses) > 1 else domain.new_tensor(0.0)
        l_worst = domain.max()
    else:
        l_soc = F.l1_loss(pred_last, y_last)
        l_rex = pred_last.new_tensor(0.0)
        l_worst = l_soc
    l_self = model.selfsup_loss(x, mask_p=spec.component_mask_p)
    l_shift = model.shift_smoothness_loss()
    l_r0 = model.r0_mono_loss()
    total = (
        l_soc
        + spec.lambda_rex * l_rex
        + spec.lambda_worst * l_worst
        + spec.lambda_selfsup * l_self
        + spec.lambda_shift_smooth * l_shift
        + spec.lambda_R0_mono * l_r0
    )
    return total, {
        "soc": float(l_soc.detach().cpu()),
        "rex": float(l_rex.detach().cpu()),
        "worst": float(l_worst.detach().cpu()),
        "selfsup": float(l_self.detach().cpu()),
        "shift": float(l_shift.detach().cpu()),
        "r0_mono": float(l_r0.detach().cpu()),
    }


def move(x, cfg: CFG):
    return x.to(device=device, dtype=torch.float32, non_blocking=bool(getattr(cfg, "cuda_non_blocking", True)) and device.type == "cuda")


@torch.no_grad()
def predict(model, loader, cfg: CFG):
    model.eval()
    rows = []
    for x, y_seq, meta in loader:
        yp = model(move(x, cfg)).detach().cpu().numpy()[:, 0]
        yy = y_seq.numpy()[:, -1, 0]
        mdf = collate_meta_to_frame(meta)
        mdf["target_label"] = "physical"
        mdf["y_true"] = yy
        mdf["y_pred"] = yp
        rows.append(mdf)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out):
        out["error"] = out["y_pred"] - out["y_true"]
        out["abs_error"] = np.abs(out["error"])
    return out


def train_robust_model(feature_frames, cfg: CFG, spec: RobustSpec, experiment: str):
    scaled, _ = scale_frames(feature_frames, FEATURES)
    train_ds = SeqEndpointDataset(scaled["train"], FEATURES, cfg.window_len, cfg.stride)
    test_ds = SeqEndpointDataset(scaled["test"], FEATURES, cfg.window_len, cfg.stride)
    train_loader = balanced_loader(train_ds, cfg, shuffle=True)
    test_loader = balanced_loader(test_ds, cfg, shuffle=False)
    model = CausalLatentSOC(
        FEATURES,
        hidden_size=int(cfg.lstm_hidden_size),
        correction_limit=spec.correction_limit,
        thermo=spec.thermo,
        shift_mode=spec.shift_mode,
        state_dropout=spec.state_dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lstm_lr, weight_decay=cfg.lstm_weight_decay)
    hist = []
    for ep in range(1, int(cfg.lstm_epochs) + 1):
        model.train()
        meters = []
        for x, y_seq, meta in train_loader:
            x = move(x, cfg)
            y_seq = move(y_seq, cfg)
            loss, parts = robust_loss(model, x, y_seq, meta, spec)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            opt.step()
            parts["total"] = float(loss.detach().cpu())
            meters.append(parts)
        row = {"experiment": experiment, "model_name": spec.name, "epoch": ep}
        if meters:
            for k in meters[0]:
                row[f"train_{k}"] = float(np.mean([m[k] for m in meters]))
        hist.append(row)
        print_every = max(1, int(getattr(cfg, "lstm_print_every", 20)))
        if ep == 1 or ep == int(cfg.lstm_epochs) or ep % print_every == 0:
            print(f"{experiment} {spec.name} epoch={ep} loss={row.get('train_total', np.nan):.5f}")
    return model, pd.DataFrame(hist), predict(model, test_loader, cfg)


def attach_predictions(pred_pairs, feature_frames, experiment):
    lookup = build_prediction_feature_lookup(feature_frames)
    rows = []
    for name, pred in pred_pairs:
        rows.append(attach_prediction_features(pred.assign(split="test", ablation=name), lookup, ablation_name=name, target_label="physical"))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out["experiment"] = experiment
    return out


def summarize(pred, omitted_temp):
    overall = _overall_metrics(pred)
    by_temp = variance_by_temperature(pred)
    focus = []
    for model, g in by_temp.groupby("model_name"):
        omitted = g[np.isclose(g["temperature_C"].astype(float), float(omitted_temp))]
        seen = g[~np.isclose(g["temperature_C"].astype(float), float(omitted_temp))]
        focus.append({
            "model_name": model,
            "omitted_temperature_C": float(omitted_temp),
            "omitted_MAE_pct": float(omitted["MAE_pct"].iloc[0]) if len(omitted) else np.nan,
            "omitted_RMSE_pct": float(omitted["RMSE_pct"].iloc[0]) if len(omitted) else np.nan,
            "omitted_jitter_ratio": float(omitted["jitter_ratio"].iloc[0]) if len(omitted) else np.nan,
            "seen_MAE_pct": float(seen["MAE_pct"].mean()) if len(seen) else np.nan,
            "worst_temperature_MAE_pct": float(g["MAE_pct"].max()) if len(g) else np.nan,
            "temperature_MAE_variance": float(g["MAE_pct"].var()) if len(g) > 1 else np.nan,
        })
    return overall, by_temp, pd.DataFrame(focus)


def risk_aware_summary(pred):
    rows = []
    for (experiment, model), g in pred.groupby(["experiment", "model_name"]):
        train_temps = {temp_key_to_c(t) for t in EXP_SPECS.get(experiment, {}).get("train_temps", ())}
        if not train_temps:
            continue
        temps = g["temperature_C"].to_numpy(float)
        risk = np.asarray([min(abs(t - tr) for tr in train_temps) for t in temps], dtype=float)
        gg = g.copy()
        gg["temperature_gap_risk"] = risk
        for cov in [1.0, 0.9, 0.8, 0.7, 0.5]:
            thr = gg["temperature_gap_risk"].quantile(cov)
            keep = gg[gg["temperature_gap_risk"] <= thr]
            rej = gg[gg["temperature_gap_risk"] > thr]
            rows.append({
                "experiment": experiment,
                "model_name": model,
                "coverage": cov,
                "risk_score": "nearest_train_temperature_gap",
                "retained_MAE_pct": keep["abs_error"].mean() * 100.0 if len(keep) else np.nan,
                "rejected_MAE_pct": rej["abs_error"].mean() * 100.0 if len(rej) else np.nan,
                "rejected_fraction": len(rej) / max(len(gg), 1),
            })
    return pd.DataFrame(rows)


def default_specs():
    return [
        RobustSpec(
            name="CausalLatent_REX_lim003",
            thermo=False,
            correction_limit=0.03,
            lambda_rex=0.5,
            lambda_worst=0.2,
        ),
        RobustSpec(
            name="CausalLatent_REX_lim005",
            thermo=False,
            correction_limit=0.05,
            lambda_rex=0.5,
            lambda_worst=0.2,
        ),
        RobustSpec(
            name="ThermoCanonical_CausalLatent_REX_lim003",
            thermo=True,
            correction_limit=0.03,
            lambda_rex=0.5,
            lambda_worst=0.2,
            lambda_shift_smooth=0.01,
            lambda_R0_mono=0.01,
            shift_mode="hybrid",
        ),
        RobustSpec(
            name="ThermoCanonical_CausalLatent_REX_lim005",
            thermo=True,
            correction_limit=0.05,
            lambda_rex=0.5,
            lambda_worst=0.2,
            lambda_shift_smooth=0.01,
            lambda_R0_mono=0.01,
            shift_mode="hybrid",
        ),
    ]


def run_thermo_robust_experiment(
    cfg: CFG | None = None,
    *,
    experiments=("Exp A", "Exp B", "Exp C"),
    outside_experiments=("Omit N10", "Omit 50"),
    specs=None,
):
    configure_torch_runtime()
    cfg = cfg or make_cfg()
    specs = specs or default_specs()
    all_pred, all_overall, all_by_temp, all_focus, all_hist = [], [], [], [], []
    fold_specs = dict(EXP_SPECS)
    fold_specs["Omit N10"] = {"train_temps": ("0", "10", "25", "50"), "omitted_temp_C": -10.0, "feature_dir": "decomposed_features"}
    fold_specs["Omit 50"] = {"train_temps": ("N10", "0", "10", "25"), "omitted_temp_C": 50.0, "feature_dir": "decomposed_features"}
    for experiment in list(experiments) + list(outside_experiments):
        base = fold_specs[experiment]
        if experiment in EXP_SPECS:
            ecfg = exp_cfg(cfg, experiment)
        else:
            ecfg = make_cfg()
            ecfg.smoke_mode = False
            ecfg.train_temps = base["train_temps"]
            ecfg.eval_temps = ("N10", "0", "10", "20", "25", "30", "40", "50")
            ecfg.train_drives = ("DST", "US06")
            ecfg.eval_drive = "FUDS"
            ecfg.decomposed_dir = ecfg.output_dir / base["feature_dir"]
        ecfg.batch_size = cfg.batch_size
        ecfg.window_len = cfg.window_len
        ecfg.stride = cfg.stride
        ecfg.lstm_epochs = cfg.lstm_epochs
        ecfg.lstm_hidden_size = cfg.lstm_hidden_size
        ecfg.lstm_lr = cfg.lstm_lr
        feature_frames = load_filtered_feature_frames(ecfg)
        pred_pairs = []
        for spec in specs:
            model, hist, pred = train_robust_model(feature_frames, ecfg, spec, experiment)
            pred_pairs.append((spec.name, pred))
            all_hist.append(hist)
        pred = attach_predictions(pred_pairs, feature_frames, experiment)
        overall, by_temp, focus = summarize(pred, base["omitted_temp_C"])
        overall["experiment"] = experiment
        by_temp["experiment"] = experiment
        focus["experiment"] = experiment
        all_pred.append(pred)
        all_overall.append(overall)
        all_by_temp.append(by_temp)
        all_focus.append(focus)
    pred = pd.concat(all_pred, ignore_index=True)
    overall = pd.concat(all_overall, ignore_index=True)
    by_temp = pd.concat(all_by_temp, ignore_index=True)
    focus = pd.concat(all_focus, ignore_index=True)
    hist = pd.concat(all_hist, ignore_index=True)
    risk = risk_aware_summary(pred)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(cfg.output_dir / "thermo_robust_prediction_rows.csv", index=False)
    overall.to_csv(cfg.output_dir / "thermo_robust_results.csv", index=False)
    focus[focus["experiment"].isin(experiments)].to_csv(cfg.output_dir / "loto_extrapolation_results.csv", index=False)
    by_temp.groupby(["experiment", "model_name"]).agg(
        worst_temperature_MAE_pct=("MAE_pct", "max"),
        temperature_MAE_variance=("MAE_pct", "var"),
    ).reset_index().to_csv(cfg.output_dir / "worst_temperature_results.csv", index=False)
    focus[focus["experiment"].isin(outside_experiments)].to_csv(cfg.output_dir / "outside_range_extrapolation_results.csv", index=False)
    risk.to_csv(cfg.output_dir / "risk_aware_results.csv", index=False)
    hist.to_csv(cfg.output_dir / "thermo_robust_training_history.csv", index=False)
    print("Thermo robust LOTO focus:")
    display(focus.sort_values(["experiment", "omitted_MAE_pct"]).head(60))
    return {
        "predictions": pred,
        "overall": overall,
        "by_temperature": by_temp,
        "focus": focus,
        "risk": risk,
        "history": hist,
    }
