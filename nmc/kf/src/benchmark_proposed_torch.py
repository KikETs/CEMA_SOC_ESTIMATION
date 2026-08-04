import json
from time import perf_counter_ns

import numpy as np
import torch
from torch import nn


class ProposedG4ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(17 * 4, 128), nn.LayerNorm(128), nn.SiLU(), nn.Dropout(0.07),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.SiLU(),
        )
        self.unused_normal_head = nn.Linear(128, 1)
        self.anchor = nn.Sequential(
            nn.Linear(8, 128), nn.LayerNorm(128), nn.SiLU(), nn.Dropout(0.07),
            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1),
        )
        self.residual = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Dropout(0.07), nn.Linear(64, 1))
        self.residual_limit = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        summary = torch.cat([x[:, -1], x.mean(1), x.std(1, unbiased=False), x[:, -1] - x[:, 0]], dim=1)
        encoded = self.encoder(summary)
        anchor = torch.sigmoid(self.anchor(x[:, :, :8]))
        residual = self.residual_limit * torch.tanh(self.residual(encoded))
        return (anchor[:, -1] + residual).clamp(0.0, 1.0)


torch.set_num_threads(1)
model = ProposedG4ResidualMLP().eval()
inputs = torch.zeros((1, 50, 17), dtype=torch.float32)
with torch.inference_mode():
    for _ in range(100):
        model(inputs)
    timings = []
    for _ in range(1000):
        started = perf_counter_ns()
        model(inputs)
        timings.append(perf_counter_ns() - started)
count = sum(parameter.numel() for parameter in model.parameters())
if count != 44036:
    raise RuntimeError(count)
print(json.dumps({"latency_ns": timings, "parameter_count": count, "parameter_bytes": count * 4, "macs": 901888}))
