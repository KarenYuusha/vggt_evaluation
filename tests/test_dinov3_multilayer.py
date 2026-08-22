from types import SimpleNamespace

import pytest
import torch

from evaluation.dinov3_backbone import (
    DINOV3_MULTILAYER_INDICES,
    extract_dinov3_patch_tokens,
)


class FakeHFBackbone(torch.nn.Module):
    def __init__(self, patch_count=6, num_register_tokens=4):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.patch_count = patch_count
        self.config = SimpleNamespace(num_register_tokens=num_register_tokens)
        self.norm = torch.nn.Identity()

    def forward(self, pixel_values, output_hidden_states=False):
        batch = pixel_values.shape[0]
        total = 1 + self.config.num_register_tokens + self.patch_count
        hidden_states = []
        # hidden_states[0] is embedding output; state k+1 represents transformer block k.
        for state_idx in range(25):
            hidden_states.append(
                torch.full((batch, total, 1024), float(state_idx), device=pixel_values.device)
            )
        return SimpleNamespace(
            last_hidden_state=hidden_states[-1],
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
        )


class AddConstantNorm(torch.nn.Module):
    def forward(self, x):
        return x + 100


def test_multilayer_indices_match_official_dinov3_paper_indices():
    assert DINOV3_MULTILAYER_INDICES == (4, 11, 17, 23)


def test_multilayer_extraction_uses_block_k_as_hidden_state_k_plus_one_and_concatenates():
    model = FakeHFBackbone()
    images = torch.zeros(2, 3, 32, 48)
    patches = extract_dinov3_patch_tokens(model, images, feature_mode="multilayer")

    assert patches.shape == (2, 6, 4096)
    chunks = patches.chunk(4, dim=-1)
    for chunk, value in zip(chunks, [5.0, 12.0, 18.0, 24.0]):
        assert torch.all(chunk == value)


def test_multilayer_extraction_normalizes_each_selected_intermediate_state():
    model = FakeHFBackbone()
    model.norm = AddConstantNorm()
    patches = extract_dinov3_patch_tokens(
        model, torch.zeros(1, 3, 32, 48), feature_mode="multilayer"
    )

    chunks = patches.chunk(4, dim=-1)
    for chunk, value in zip(chunks, [105.0, 112.0, 118.0, 124.0]):
        assert torch.all(chunk == value)


def test_final_extraction_remains_1024_dimensional():
    model = FakeHFBackbone()
    patches = extract_dinov3_patch_tokens(
        model, torch.zeros(1, 3, 32, 48), feature_mode="final"
    )
    assert patches.shape == (1, 6, 1024)
    assert torch.all(patches == 24)


def test_unknown_feature_mode_is_rejected():
    with pytest.raises(ValueError, match="feature mode"):
        extract_dinov3_patch_tokens(
            FakeHFBackbone(), torch.zeros(1, 3, 32, 48), feature_mode="bad"
        )
