"""Run a synthetic forward pass through BrainFuse-LLM core modules."""

from __future__ import annotations

import torch

from brainfuse import NeuroEnhancedEncoder, NeuroLoRAMoELinear


def main() -> None:
    torch.manual_seed(7)
    batch_size = 2
    num_voxels = 512
    output_dim = 128

    voxel_values = torch.randn(batch_size, num_voxels)
    coords = torch.randn(batch_size, num_voxels, 3)
    region_ids = torch.randint(0, 128, (batch_size, num_voxels, 2))

    encoder = NeuroEnhancedEncoder(
        token_count=16,
        hidden_dim=64,
        output_dim=output_dim,
        region_vocab_sizes=(128, 128),
        num_heads=4,
    )
    adapter = NeuroLoRAMoELinear(
        in_features=output_dim,
        out_features=output_dim,
        rank=4,
        alpha=8,
        num_experts=4,
        top_k=1,
    )

    encoder_output = encoder(voxel_values, coords, region_ids, return_attention=True)
    adapted_tokens, router_logits = adapter(encoder_output.tokens)

    print("neural_tokens:", tuple(encoder_output.tokens.shape))
    print("adapted_tokens:", tuple(adapted_tokens.shape))
    print("router_logits:", tuple(router_logits.shape))
    if encoder_output.attention_weights is not None:
        print("attention_weights:", tuple(encoder_output.attention_weights.shape))


if __name__ == "__main__":
    main()
