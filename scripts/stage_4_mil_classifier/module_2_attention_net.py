"""Module 2: Attention-based MIL network for slide classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionMIL(nn.Module):
    """Attention MIL model for binary slide-level classification."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, dropout: float = 0.25) -> None:
        super().__init__()
        self.compressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for one slide bag with shape [1, N, 768]."""
        x = x.squeeze(0)               # [N, 768]
        h = self.compressor(x)         # [N, 512]
        a = self.attention(h)          # [N, 1]
        A = torch.softmax(a, dim=0)    # [N, 1]
        A = A.t()                      # [1, N]
        M = torch.mm(A, h)             # [1, 512]
        logits = self.classifier(M)    # [1, 1]
        return logits, A


if __name__ == "__main__":
    model = AttentionMIL()
    x = torch.randn(1, 16041, 768)
    logits, A = model(x)
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"attention shape: {tuple(A.shape)}")
