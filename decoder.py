"""Nonlinear edge decoder used by SCGAE."""

from __future__ import annotations

import torch
from torch import nn


class EdgeDecoder(nn.Module):
    """Decode Hadamard node-pair representations with an MLP."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the consistency decoder.

        Args:
            hidden_dim: Size of each node embedding.
            num_layers: Number of linear layers in the decoder.
            dropout: Dropout probability before hidden decoder layers.
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        middle_dim = max(hidden_dim // 2, 1)
        widths = [hidden_dim] + [middle_dim] * (num_layers - 1) + [1]
        self.layers = nn.ModuleList(
            nn.Linear(widths[index], widths[index + 1])
            for index in range(len(widths) - 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def reset_parameters(self) -> None:
        """Reset every decoder layer to its initial distribution."""
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return raw link logits for supplied node pairs.

        Args:
            z: Node embedding matrix with shape ``[num_nodes, hidden_dim]``.
            edge_index: Node pairs to score, with shape ``[2, num_pairs]``.

        Returns:
            Raw edge logits with shape ``[num_pairs]``.
        """
        source, target = edge_index.long()
        hidden = z[source] * z[target]
        for layer in self.layers[:-1]:
            hidden = self.activation(layer(self.dropout(hidden)))
        return self.layers[-1](hidden).squeeze(-1)

    def predict_proba(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return link probabilities for supplied node pairs.

        Args:
            z: Node embedding matrix with shape ``[num_nodes, hidden_dim]``.
            edge_index: Node pairs to score, with shape ``[2, num_pairs]``.

        Returns:
            Link probabilities with shape ``[num_pairs]``.
        """
        return self(z, edge_index).sigmoid()
