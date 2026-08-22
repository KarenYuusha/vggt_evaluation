from pathlib import Path

import torch
from torch import nn


class FeatureAdapter(nn.Module):
    adapter_type = "linear"

    def __init__(self, dim=1024, *, input_dim=None, output_dim=None):
        super().__init__()
        if input_dim is None:
            input_dim = dim
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.dim = self.output_dim if self.input_dim == self.output_dim else None
        self.linear = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, features):
        return self.linear(features)


class ResidualMLPAdapter(nn.Module):
    adapter_type = "mlp"

    def __init__(self, dim=1024, hidden_dim=2048, *, input_dim=None, output_dim=None):
        super().__init__()
        if input_dim is None:
            input_dim = dim
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.dim = self.output_dim if self.input_dim == self.output_dim else None
        self.hidden_dim = int(hidden_dim)
        self.use_residual = self.input_dim == self.output_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, features):
        update = self.fc2(self.activation(self.fc1(features)))
        return features + update if self.use_residual else update


def build_linear_adapter(input_dim=1024, output_dim=None, identity_if_possible=True):
    adapter = FeatureAdapter(input_dim=input_dim, output_dim=output_dim)
    if identity_if_possible and adapter.input_dim == adapter.output_dim:
        with torch.no_grad():
            nn.init.eye_(adapter.linear.weight)
            nn.init.zeros_(adapter.linear.bias)
    return adapter


def build_identity_adapter(dim=1024):
    return build_linear_adapter(input_dim=dim, output_dim=dim, identity_if_possible=True)


def build_residual_mlp_adapter(dim=None, hidden_dim=2048, input_dim=None, output_dim=None):
    if input_dim is None:
        input_dim = 1024 if dim is None else dim
    if output_dim is None:
        output_dim = input_dim if dim is None else dim
    adapter = ResidualMLPAdapter(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)
    if adapter.use_residual:
        with torch.no_grad():
            nn.init.zeros_(adapter.fc2.weight)
            nn.init.zeros_(adapter.fc2.bias)
    return adapter


def save_adapter_checkpoint(path, adapter, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": adapter.state_dict(),
        "input_dim": adapter.input_dim,
        "output_dim": adapter.output_dim,
        "adapter_type": adapter.adapter_type,
        "metadata": dict(metadata or {}),
    }
    if adapter.input_dim == adapter.output_dim:
        checkpoint["dim"] = adapter.input_dim
    if isinstance(adapter, ResidualMLPAdapter):
        checkpoint["hidden_dim"] = adapter.hidden_dim
    torch.save(checkpoint, path)


def load_adapter_checkpoint(path, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    adapter_type = checkpoint.get("adapter_type", "linear")
    legacy_dim = checkpoint.get("dim")
    input_dim = int(checkpoint.get("input_dim", legacy_dim))
    output_dim = int(checkpoint.get("output_dim", legacy_dim))
    if adapter_type == "linear":
        adapter = FeatureAdapter(input_dim=input_dim, output_dim=output_dim)
    elif adapter_type == "mlp":
        adapter = ResidualMLPAdapter(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=checkpoint["hidden_dim"],
        )
    else:
        raise ValueError(f"Unsupported adapter type: {adapter_type}")
    adapter.load_state_dict(checkpoint["state_dict"])
    return adapter, dict(checkpoint.get("metadata", {}))
