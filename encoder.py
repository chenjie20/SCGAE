"""Shared graph encoder used by both SCGAE views."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv

from edges import to_bidirectional


class GCNEncoder(nn.Module):
    """Multi-layer GCN encoder with shared parameters across graph views."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.8,
    ) -> None:
        """Initialize the shared GCN encoder.

        Args:
            input_dim: Number of input node features.
            hidden_dim: Size of the node embedding.
            num_layers: Number of graph convolution layers.
            dropout: Feature dropout probability before each layer.
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.layers = nn.ModuleList([GCNConv(input_dim, hidden_dim)])
        self.layers.extend(GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 1))
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()

    def reset_parameters(self) -> None:
        """Reset every graph convolution layer to its initial distribution."""
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode node features on one undirected graph view.

        Args:
            x: Node feature matrix with shape ``[num_nodes, input_dim]``.
            edge_index: View edges with shape ``[2, num_edges]``.

        Returns:
            Node embeddings with shape ``[num_nodes, hidden_dim]``.
        """
        message_edges = to_bidirectional(edge_index)
        hidden = x
        for layer in self.layers:
            hidden = self.dropout(hidden)
            hidden = self.activation(layer(hidden, message_edges))
        return hidden
