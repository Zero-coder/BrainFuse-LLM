import torch

from brainfuse import NeuroEnhancedEncoder, NeuroLoRAMoELinear


def test_neuro_encoder_shapes():
    encoder = NeuroEnhancedEncoder(
        token_count=8,
        hidden_dim=32,
        output_dim=64,
        region_vocab_sizes=(16, 16),
        num_heads=4,
    )
    voxel_values = torch.randn(2, 64)
    coords = torch.randn(2, 64, 3)
    region_ids = torch.randint(0, 16, (2, 64, 2))

    output = encoder(voxel_values, coords, region_ids)

    assert output.tokens.shape == (2, 8, 64)


def test_neuro_lora_moe_shapes():
    layer = NeuroLoRAMoELinear(64, 64, rank=4, num_experts=3, top_k=1)
    x = torch.randn(2, 8, 64)

    y, logits = layer(x)

    assert y.shape == x.shape
    assert logits.shape == (2, 8, 3)
