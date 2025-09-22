from typing import Literal

import torch.nn as nn


class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super(MLPProbe, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class MultiLayerProbe(nn.Module):
    def __init__(
        self, probe: Literal["linear", "mlp"] = "linear", num_probes: int = 12, **kwargs
    ):
        super(MultiLayerProbe, self).__init__()
        probe_cls = LinearProbe if probe == "linear" else MLPProbe
        self.probes = nn.ModuleList([probe_cls(**kwargs) for _ in range(num_probes)])

    def forward(self, activations):
        return [probe(act) for probe, act in zip(self.probes, activations)]
