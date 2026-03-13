#!/usr/bin/env python3
"""
Lemma Projector Model for Metric Learning
Projects 768-dim AraBERT embeddings to 128-dim metric space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LemmaProjector(nn.Module):
    """
    Neural network that projects high-dimensional embeddings to a lower-dimensional
    metric space where embeddings with the same lemma are close together.

    Architecture:
        Input: 768 (AraBERT embedding)
        Hidden: 512 with BatchNorm, ReLU, Dropout
        Output: 128 (normalized to unit sphere)
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 128,
        dropout: float = 0.2
    ):
        super(LemmaProjector, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Layer 1: 768 -> 512
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Layer 2: 512 -> 128
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with L2 normalization.

        Args:
            x: Input embeddings (batch_size, 768)

        Returns:
            Normalized embeddings (batch_size, 128) on unit sphere
        """
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.fc2(x)

        # L2 normalization - critical for metric learning!
        # Projects all vectors onto the surface of a unit sphere
        x = F.normalize(x, p=2, dim=1)

        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward pass (for clarity in inference)."""
        return self.forward(x)


def test_model():
    """Test the model to verify architecture and output."""
    print("="*80)
    print("TESTING LEMMA PROJECTOR MODEL")
    print("="*80)

    # Create model
    model = LemmaProjector()
    model.eval()  # Set to eval mode for testing
    print(f"\nModel architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    print("\nTesting forward pass:")
    batch_size = 8
    input_dim = 768

    # Create random input
    x = torch.randn(batch_size, input_dim)
    print(f"Input shape: {x.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")

    # Verify normalization
    norms = torch.norm(output, p=2, dim=1)
    print(f"\nL2 norms of output vectors:")
    print(f"  Mean: {norms.mean():.6f}")
    print(f"  Std:  {norms.std():.6f}")
    print(f"  Min:  {norms.min():.6f}")
    print(f"  Max:  {norms.max():.6f}")

    # All norms should be ≈ 1.0
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        "Output vectors are not properly normalized!"

    print("\n✓ All output vectors have unit norm (L2 = 1.0)")

    # Test with different batch sizes
    print("\nTesting different batch sizes:")
    model.eval()  # Set to eval mode for single sample testing
    for bs in [1, 16, 64, 128]:
        x = torch.randn(bs, input_dim)
        with torch.no_grad():
            output = model(x)
        print(f"  Batch size {bs:3d}: output shape = {output.shape}, "
              f"mean norm = {torch.norm(output, p=2, dim=1).mean():.6f}")

    print("\n" + "="*80)
    print("✓ Model test passed!")
    print("="*80)


if __name__ == "__main__":
    test_model()

