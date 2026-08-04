import warnings
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG
from .runtime import device

# Corrector pretraining loss. No SOC label is used here.
def tensor_profile(profile, start=0, end=None):
    end = len(profile["V"]) if end is None else end
    sl = slice(start, end)
    def tt(name):
        return torch.from_numpy(profile[name][sl]).float().view(1, -1, 1).to(device)
    return {
        "V_s": tt("V_s"),
        "I_s": tt("I_s"),
        "T_s": tt("T_s"),
        "V_raw": tt("V"),
        "I_raw": tt("I"),
    }


def tensor_profile_batch(profiles, start=0, end=None):
    if not profiles:
        raise ValueError("tensor_profile_batch requires at least one profile")
    n0 = len(profiles[0]["V"])
    end = n0 if end is None else end
    sl = slice(start, end)
    for p in profiles:
        if len(p["V"]) != n0:
            raise ValueError("Batched corrector pretraining requires equal-length profiles")

    def tt(name):
        arr = np.stack([p[name][sl] for p in profiles], axis=0).astype(np.float32, copy=False)
        return torch.from_numpy(arr).float().unsqueeze(-1).to(device)

    return {
        "V_s": tt("V_s"),
        "I_s": tt("I_s"),
        "T_s": tt("T_s"),
        "V_raw": tt("V"),
        "I_raw": tt("I"),
    }


def tensor_profile_segments(segments):
    if not segments:
        raise ValueError("tensor_profile_segments requires at least one segment")
    lengths = {end - start for profile, start, end in segments}
    if len(lengths) != 1:
        raise ValueError("Batched corrector segments must have equal lengths")

    def tt(name):
        arr = np.stack(
            [profile[name][start:end] for profile, start, end in segments],
            axis=0,
        ).astype(np.float32, copy=False)
        return torch.from_numpy(arr).float().unsqueeze(-1).to(device)

    return {
        "V_s": tt("V_s"),
        "I_s": tt("I_s"),
        "T_s": tt("T_s"),
        "V_raw": tt("V"),
        "I_raw": tt("I"),
    }


def corrector_profile_batches(train_profiles, cfg: CFG):
    seg_len = getattr(cfg, "corrector_train_segment_len", None)
    if seg_len is not None:
        seg_len = int(seg_len)
        rng = np.random.default_rng()
        segments = []
        repeats = max(1, int(getattr(cfg, "corrector_segments_per_profile_per_epoch", 1)))
        for profile in train_profiles:
            n = len(profile["V"])
            if n < 2:
                continue
            length = min(seg_len, n)
            for _ in range(repeats):
                start = 0 if n <= length else int(rng.integers(0, n - length + 1))
                segments.append((profile, start, start + length))
        groups = OrderedDict()
        for segment in segments:
            groups.setdefault(segment[2] - segment[1], []).append(segment)
        batch_size = max(1, int(getattr(cfg, "corrector_profile_batch_size", 8)))
        batches = []
        for _, group_segments in groups.items():
            rng.shuffle(group_segments)
            for i in range(0, len(group_segments), batch_size):
                batches.append(group_segments[i:i + batch_size])
        rng.shuffle(batches)
        return batches

    if not bool(getattr(cfg, "corrector_batch_by_length", True)):
        return [(p, 0, len(p["V"])) for p in train_profiles if len(p["V"]) >= 2]
    groups = OrderedDict()
    for profile in train_profiles:
        if len(profile["V"]) < 2:
            continue
        groups.setdefault(len(profile["V"]), []).append((profile, 0, len(profile["V"])))
    batch_size = max(1, int(getattr(cfg, "corrector_profile_batch_size", 8)))
    batches = []
    for _, segments in groups.items():
        for i in range(0, len(segments), batch_size):
            batches.append(segments[i:i + batch_size])
    return batches


def masked_mean(x):
    return torch.mean(x)


def high_frequency_energy_torch(x):
    if x.size(1) < 3:
        return x.new_tensor(0.0)
    d1 = x[:, 1:, :] - x[:, :-1, :]
    d2 = d1[:, 1:, :] - d1[:, :-1, :]
    return d2.square().mean()


def absolute_curvature_torch(x):
    if x.size(1) < 3:
        return x.new_tensor(0.0)
    d1 = x[:, 1:, :] - x[:, :-1, :]
    d2 = d1[:, 1:, :] - d1[:, :-1, :]
    return d2.abs().mean()


def frequency_route_order_loss(aux, margin=0.2, eps=1e-12):
    energies = [
        high_frequency_energy_torch(aux["v_pol_fast_raw"]),
        high_frequency_energy_torch(aux["v_pol_mid_raw"]),
        high_frequency_energy_torch(aux["v_pol_slow_raw"]),
        high_frequency_energy_torch(aux["v_hys_raw"]),
        high_frequency_energy_torch(aux["R0"]),
    ]
    logs = [torch.log(e + eps) for e in energies]
    loss = logs[0].new_tensor(0.0)
    # Enforce fast >= mid >= slow >= hys >= R0 by a soft log-energy margin.
    for prev, nxt in zip(logs, logs[1:]):
        loss = loss + F.relu(nxt - prev + float(margin)).square()
    return loss


def normalized_derivative_overlap(a, b, eps=1e-8):
    if a.size(1) < 2 or b.size(1) < 2:
        return a.new_tensor(0.0)
    da = a[:, 1:, :] - a[:, :-1, :]
    db = b[:, 1:, :] - b[:, :-1, :]
    da = da - da.mean(dim=1, keepdim=True)
    db = db - db.mean(dim=1, keepdim=True)
    num = (da * db).mean()
    den = da.square().mean().sqrt() * db.square().mean().sqrt()
    return torch.abs(num / (den + eps))


def corrector_pretraining_loss(aux, I_raw, cfg: CFG):
    losses = {}
    # This is a sign-convention consistency term. It is usually near-identity by construction.
    recon = aux["V_corr_raw"] - aux["v_drop_capped"]
    losses["recon"] = F.smooth_l1_loss(recon, aux["V_raw"], beta=0.01)

    if aux["V_corr_raw"].size(1) > 1:
        losses["smooth_vcorr"] = masked_mean(torch.abs(aux["V_corr_raw"][:, 1:, :] - aux["V_corr_raw"][:, :-1, :]))
        losses["r0_smooth"] = masked_mean(torch.abs(aux["R0"][:, 1:, :] - aux["R0"][:, :-1, :]))
        losses["hys_smooth"] = masked_mean(torch.abs(aux["v_hys_raw"][:, 1:, :] - aux["v_hys_raw"][:, :-1, :]))
    else:
        z = aux["V_corr_raw"].new_tensor(0.0)
        losses["smooth_vcorr"] = z
        losses["r0_smooth"] = z
        losses["hys_smooth"] = z

    losses["pol_hf"] = high_frequency_energy_torch(aux["v_pol_raw"])
    losses["hys_hf"] = high_frequency_energy_torch(aux["v_hys_raw"])
    losses["R0_hf"] = high_frequency_energy_torch(aux["R0"])
    losses["pol_slow_hf"] = high_frequency_energy_torch(aux["v_pol_slow_raw"])
    losses["hys_slope"] = absolute_curvature_torch(aux["v_hys_raw"])
    losses["hys_tv"] = losses["hys_smooth"]
    losses["R0_smooth_extra"] = losses["r0_smooth"]
    losses["timescale_sep"] = (
        normalized_derivative_overlap(aux["v_pol_fast_raw"], aux["v_pol_mid_raw"])
        + normalized_derivative_overlap(aux["v_pol_fast_raw"], aux["v_pol_slow_raw"])
        + normalized_derivative_overlap(aux["v_pol_mid_raw"], aux["v_pol_slow_raw"])
        + high_frequency_energy_torch(aux["v_pol_mid_raw"])
        + high_frequency_energy_torch(aux["v_pol_slow_raw"])
    )
    losses["frequency_route"] = frequency_route_order_loss(
        aux,
        margin=float(getattr(cfg, "frequency_route_margin", 0.2)),
    )

    pol_lim = float(cfg.pol_limit_V)
    hys_lim = float(cfg.hys_limit_V) * float(getattr(cfg, "hys_limit_scale", 1.0))
    r0_lim = float(cfg.ohm_R0_max)
    losses["component_bound"] = (
        F.relu(torch.abs(aux["v_pol_raw"]) - pol_lim).square().mean()
        + F.relu(torch.abs(aux["v_hys_raw"]) - hys_lim).square().mean()
        + F.relu(aux["R0"] - r0_lim).square().mean()
    )

    rest = (torch.abs(I_raw) < float(cfg.I_rest_thr_A)).float()
    denom = rest.sum().clamp(min=1.0)
    losses["rest_components"] = (
        rest * (torch.abs(aux["v_pol_raw"]) + torch.abs(aux["v_hys_raw"]) + torch.abs(aux["v_ohm_raw"]))
    ).sum() / denom

    losses["total"] = (
        cfg.lambda_recon * losses["recon"]
        + cfg.lambda_smooth_vcorr * losses["smooth_vcorr"]
        + cfg.lambda_component_bound * losses["component_bound"]
        + cfg.lambda_r0_smooth * losses["r0_smooth"]
        + cfg.lambda_hys_smooth * losses["hys_smooth"]
        + cfg.lambda_rest_components * losses["rest_components"]
        + float(getattr(cfg, "lambda_pol_hf", 0.0)) * losses["pol_hf"]
        + float(getattr(cfg, "lambda_hys_hf", 0.0)) * losses["hys_hf"]
        + float(getattr(cfg, "lambda_R0_smooth", 0.0)) * losses["R0_smooth_extra"]
        + float(getattr(cfg, "lambda_timescale_sep", 0.0)) * losses["timescale_sep"]
        + float(getattr(cfg, "lambda_hys_tv", 0.0)) * losses["hys_tv"]
        + float(getattr(cfg, "lambda_hys_slope", 0.0)) * losses["hys_slope"]
        + float(getattr(cfg, "lambda_pol_slow_hf", 0.0)) * losses["pol_slow_hf"]
        + float(getattr(cfg, "lambda_R0_hf", 0.0)) * losses["R0_hf"]
        + float(getattr(cfg, "lambda_frequency_route", 0.0)) * losses["frequency_route"]
    )
    if "temp_tau_reg" in aux:
        losses["temp_tau_reg"] = aux["temp_tau_reg"]
        losses["total"] = losses["total"] + float(getattr(cfg, "lambda_temp_tau_reg", 0.0)) * losses["temp_tau_reg"]
    return losses


def assert_corrector_loss_has_no_soc():
    names = set(corrector_pretraining_loss.__code__.co_varnames)
    assert not any("soc" in n.lower() for n in names), "Corrector pretraining loss must not accept SOC labels"
    print("corrector loss SOC-leakage check passed")


def train_voltage_corrector(corrector, train_profiles, cfg: CFG, v_scaler):
    assert_corrector_loss_has_no_soc()
    corrector.train()
    opt = torch.optim.AdamW(corrector.parameters(), lr=cfg.corrector_lr, weight_decay=cfg.corrector_weight_decay)
    vmin = torch.tensor(v_scaler.min_, device=device, dtype=torch.float32)
    vmax = torch.tensor(v_scaler.max_, device=device, dtype=torch.float32)
    history = []
    warnings.warn("Corrector pretraining excludes SOC labels. Reconstruction consistency alone is not an identifiability guarantee.")

    for ep in range(1, int(cfg.corrector_epochs) + 1):
        meters = OrderedDict()
        n_steps = 0
        for profile_batch in corrector_profile_batches(train_profiles, cfg):
            batch = tensor_profile_segments(profile_batch)
            _, _, aux = corrector(
                batch["V_s"], batch["I_s"], batch["T_s"],
                batch["V_raw"], batch["I_raw"], vmin, vmax, state=None
            )
            losses = corrector_pretraining_loss(aux, batch["I_raw"], cfg)
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(corrector.parameters(), float(cfg.grad_clip))
            opt.step()
            for k, v in losses.items():
                meters[k] = meters.get(k, 0.0) + float(v.detach().cpu())
            n_steps += 1
        row = {k: v / max(1, n_steps) for k, v in meters.items()}
        row["epoch"] = ep
        history.append(row)
        print("corrector epoch", ep, {k: round(row[k], 6) for k in row if k != "epoch"})
    return pd.DataFrame(history)



def run_corrector_pretraining(corrector, train_profiles, cfg: CFG, v_scaler):
    if cfg.run_corrector_pretraining:
        return train_voltage_corrector(corrector, train_profiles, cfg, v_scaler)
    print("Corrector pretraining skipped by cfg.run_corrector_pretraining=False")
    return pd.DataFrame()
