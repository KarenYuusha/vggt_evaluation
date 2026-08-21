from pathlib import Path

import torch
from torch import nn


class FeatureAdapter(nn.Module):
    adapter_type = "linear"

    def __init__(self, dim=1024):
        super().__init__()
        self.dim = int(dim)
        self.linear = nn.Linear(self.dim, self.dim)

    def forward(self, features):
        return self.linear(features)


class ResidualMLPAdapter(nn.Module):
    adapter_type = "mlp"

    def __init__(self, dim=1024, hidden_dim=2048):
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.fc1 = nn.Linear(self.dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_dim, self.dim)

    def forward(self, features):
        return features + self.fc2(self.activation(self.fc1(features)))


def build_identity_adapter(dim=1024):
    adapter = FeatureAdapter(dim)
    with torch.no_grad():
        nn.init.eye_(adapter.linear.weight)
        nn.init.zeros_(adapter.linear.bias)
    return adapter


def build_residual_mlp_adapter(dim=1024, hidden_dim=2048):
    adapter = ResidualMLPAdapter(dim=dim, hidden_dim=hidden_dim)
    with torch.no_grad():
        nn.init.zeros_(adapter.fc2.weight)
        nn.init.zeros_(adapter.fc2.bias)
    return adapter


def save_adapter_checkpoint(path, adapter, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": adapter.state_dict(),
        "dim": adapter.dim,
        "adapter_type": adapter.adapter_type,
        "metadata": dict(metadata or {}),
    }
    if isinstance(adapter, ResidualMLPAdapter):
        checkpoint["hidden_dim"] = adapter.hidden_dim
    torch.save(checkpoint, path)


def load_adapter_checkpoint(path, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    adapter_type = checkpoint.get("adapter_type", "linear")
    if adapter_type == "linear":
        adapter = FeatureAdapter(checkpoint["dim"])
    elif adapter_type == "mlp":
        adapter = ResidualMLPAdapter(checkpoint["dim"], checkpoint["hidden_dim"])
    else:
        raise ValueError(f"Unsupported adapter type: {adapter_type}")
    adapter.load_state_dict(checkpoint["state_dict"])
    return adapter, dict(checkpoint.get("metadata", {}))
