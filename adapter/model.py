from pathlib import Path

import torch
from torch import nn


class FeatureAdapter(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.dim = int(dim)
        self.linear = nn.Linear(self.dim, self.dim)

    def forward(self, features):
        return self.linear(features)


def build_identity_adapter(dim=1024):
    adapter = FeatureAdapter(dim)
    with torch.no_grad():
        nn.init.eye_(adapter.linear.weight)
        nn.init.zeros_(adapter.linear.bias)
    return adapter


def save_adapter_checkpoint(path, adapter, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": adapter.state_dict(), "dim": adapter.dim, "metadata": dict(metadata or {})}, path)


def load_adapter_checkpoint(path, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    adapter = FeatureAdapter(checkpoint["dim"])
    adapter.load_state_dict(checkpoint["state_dict"])
    return adapter, dict(checkpoint.get("metadata", {}))
