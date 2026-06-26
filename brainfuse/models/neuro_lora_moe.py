"""Neuro LoRA-MoE reference modules."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LoRAExpert(nn.Module):
    """A single low-rank update path."""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x)) * self.scale


class NeuroLoRAMoELinear(nn.Module):
    """Linear layer augmented with neural-conditioned LoRA experts.

    The base projection is frozen by default. A lightweight router selects a
    sparse mixture of LoRA experts for each token.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        num_experts: int = 4,
        top_k: int = 1,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")

        self.base = nn.Linear(in_features, out_features)
        self.experts = nn.ModuleList(
            [LoRAExpert(in_features, out_features, rank, alpha) for _ in range(num_experts)]
        )
        self.router = nn.Linear(in_features, num_experts)
        self.top_k = top_k

        if freeze_base:
            for parameter in self.base.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base_output = self.base(x)
        router_logits = self.router(x)
        top_values, top_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        top_weights = F.softmax(top_values, dim=-1)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        selected = torch.gather(
            expert_outputs,
            dim=-2,
            index=top_indices.unsqueeze(-1).expand(*top_indices.shape, expert_outputs.shape[-1]),
        )
        moe_update = (selected * top_weights.unsqueeze(-1)).sum(dim=-2)
        return base_output + moe_update, router_logits
