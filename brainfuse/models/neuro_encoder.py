"""Neuro-Enhanced Encoder modules.

The implementation here is a compact reference version of the encoder described
in the BrainFuse-LLM manuscript. It focuses on the core data flow:

voxel values + coordinates + region labels -> fixed-length neural tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class NeuroEncoderOutput:
    tokens: torch.Tensor
    attention_weights: torch.Tensor | None = None


class MultiScaleValueFusion(nn.Module):
    """Fuse voxel responses with several local 1D convolutional kernels."""

    def __init__(self, hidden_dim: int, kernel_sizes: tuple[int, ...] = (1, 3, 5)) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(1, hidden_dim, kernel_size=k, padding=k // 2)
                for k in kernel_sizes
            ]
        )
        self.fusion_logits = nn.Parameter(torch.zeros(len(kernel_sizes)))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, voxel_values: torch.Tensor) -> torch.Tensor:
        if voxel_values.ndim != 2:
            raise ValueError("voxel_values must have shape [batch, num_voxels]")

        x = voxel_values.unsqueeze(1)
        branch_outputs = [branch(x).transpose(1, 2) for branch in self.branches]
        weights = torch.softmax(self.fusion_logits, dim=0)
        fused = sum(weight * output for weight, output in zip(weights, branch_outputs))
        return self.norm(fused)


class CoordinateRegionKeyBuilder(nn.Module):
    """Build key embeddings from voxel coordinates and parcellation labels."""

    def __init__(
        self,
        key_dim: int,
        region_vocab_sizes: tuple[int, ...] = (512, 1024),
        region_dim: int = 32,
    ) -> None:
        super().__init__()
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, key_dim),
        )
        self.region_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, region_dim) for vocab_size in region_vocab_sizes]
        )
        self.region_proj = nn.Linear(region_dim * len(region_vocab_sizes), key_dim)
        self.output_norm = nn.LayerNorm(key_dim)

    def forward(self, coords: torch.Tensor, region_ids: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError("coords must have shape [batch, num_voxels, 3]")
        if region_ids.ndim != 3:
            raise ValueError("region_ids must have shape [batch, num_voxels, num_parcellations]")
        if region_ids.shape[-1] != len(self.region_embeddings):
            raise ValueError("region_ids does not match the configured parcellations")

        coord_features = self.coord_mlp(coords.float())
        region_features = []
        for idx, embedding in enumerate(self.region_embeddings):
            labels = region_ids[..., idx].clamp(min=0, max=embedding.num_embeddings - 1)
            region_features.append(embedding(labels))
        region_features_cat = torch.cat(region_features, dim=-1)
        keys = coord_features + self.region_proj(region_features_cat)
        return self.output_norm(apply_coordinate_rope(keys, coords))


def apply_coordinate_rope(keys: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Apply a lightweight coordinate-conditioned rotary modulation."""

    dim = keys.shape[-1]
    if dim % 2 != 0:
        return keys

    coord_phase = coords.float().mean(dim=-1, keepdim=True)
    freqs = torch.linspace(1.0, 3.0, dim // 2, device=keys.device, dtype=keys.dtype)
    angles = coord_phase * freqs
    sin, cos = torch.sin(angles), torch.cos(angles)
    even, odd = keys[..., 0::2], keys[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class NeuroEnhancedEncoder(nn.Module):
    """Attention-based subject-agnostic fMRI encoder."""

    def __init__(
        self,
        token_count: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 4096,
        value_kernel_sizes: tuple[int, ...] = (1, 3, 5),
        region_vocab_sizes: tuple[int, ...] = (512, 1024),
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.value_fusion = MultiScaleValueFusion(hidden_dim, value_kernel_sizes)
        self.key_builder = CoordinateRegionKeyBuilder(hidden_dim, region_vocab_sizes)
        self.queries = nn.Parameter(torch.randn(token_count, hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        voxel_values: torch.Tensor,
        coords: torch.Tensor,
        region_ids: torch.Tensor,
        return_attention: bool = False,
    ) -> NeuroEncoderOutput:
        batch_size = voxel_values.shape[0]
        values = self.value_fusion(voxel_values)
        keys = self.key_builder(coords, region_ids)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        hidden, attention_weights = self.attention(
            query=queries,
            key=keys,
            value=values,
            need_weights=return_attention,
        )
        tokens = self.output(hidden)
        return NeuroEncoderOutput(
            tokens=tokens,
            attention_weights=attention_weights if return_attention else None,
        )
