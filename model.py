"""Top-level SCGAE model interface."""

from __future__ import annotations

import torch
from torch import nn

from decoder import EdgeDecoder
from encoder import GCNEncoder
from losses import SCGAELoss, combine_view_losses, edge_reconstruction_loss


class SCGAE(nn.Module):
    """Structure-Aware Consistent Graph Autoencoder for link prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        encoder_dropout: float = 0.8,
        decoder_dropout: float = 0.2,
    ) -> None:
        """Initialize the shared encoder and nonlinear consistency decoder.

        Args:
            input_dim: Number of input node features.
            hidden_dim: Size of the learned node embedding.
            encoder_layers: Number of GCN layers.
            decoder_layers: Number of MLP decoder layers.
            encoder_dropout: Feature dropout probability in the encoder.
            decoder_dropout: Hidden dropout probability in the decoder.
        """
        super().__init__()
        self.encoder = GCNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            dropout=encoder_dropout,
        )
        self.decoder = EdgeDecoder(
            hidden_dim=hidden_dim,
            num_layers=decoder_layers,
            dropout=decoder_dropout,
        )

    def reset_parameters(self) -> None:
        """Reset encoder and decoder parameters."""
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode all nodes using one graph view.

        Args:
            x: Node feature matrix with shape ``[num_nodes, input_dim]``.
            edge_index: Message-passing edges with shape ``[2, num_edges]``.

        Returns:
            Node embeddings with shape ``[num_nodes, hidden_dim]``.
        """
        return self.encoder(x, edge_index)

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Decode raw link logits from node embeddings.

        Args:
            z: Node embedding matrix with shape ``[num_nodes, hidden_dim]``.
            edge_index: Node pairs to score, with shape ``[2, num_pairs]``.

        Returns:
            Raw link logits with shape ``[num_pairs]``.
        """
        return self.decoder(z, edge_index)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        query_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Encode an observed graph and return logits for query node pairs.

        Args:
            x: Node feature matrix with shape ``[num_nodes, input_dim]``.
            edge_index: Observed message-passing edges with shape
                ``[2, num_edges]``.
            query_edge_index: Node pairs to score, with shape
                ``[2, num_queries]``.

        Returns:
            Raw link logits with shape ``[num_queries]``.
        """
        return self.decode(self.encode(x, edge_index), query_edge_index)

    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        query_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Encode an observed graph and return query link probabilities.

        Args:
            x: Node feature matrix with shape ``[num_nodes, input_dim]``.
            edge_index: Observed message-passing edges with shape
                ``[2, num_edges]``.
            query_edge_index: Node pairs to score, with shape
                ``[2, num_queries]``.

        Returns:
            Link probabilities with shape ``[num_queries]``.
        """
        return self.forward(x, edge_index, query_edge_index).sigmoid()

    def _loss_for_targets(
        self,
        z: torch.Tensor,
        positive_edge_index: torch.Tensor,
        negative_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute one embedding-to-topology reconstruction loss.

        Args:
            z: Node embeddings learned from one graph view.
            positive_edge_index: Positive topology targets.
            negative_edge_index: Negative topology targets.

        Returns:
            Balanced binary edge reconstruction loss.
        """
        return edge_reconstruction_loss(
            self.decode(z, positive_edge_index),
            self.decode(z, negative_edge_index),
        )

    def compute_loss(
        self,
        x: torch.Tensor,
        views: tuple[torch.Tensor, torch.Tensor],
        negative_edges: tuple[torch.Tensor, torch.Tensor],
        alpha: float = 1.0,
    ) -> SCGAELoss:
        """Compute SCGAE self-topology and cross-view reconstruction losses.

        Embeddings from each view reconstruct both their own topology and the
        complementary view. Negative targets correspond to the view at the same
        tuple position and should exclude all known positive graph edges.

        Args:
            x: Node feature matrix with shape ``[num_nodes, input_dim]``.
            views: Two positive edge tensors produced by ``make_views``.
            negative_edges: Negative target tensors for the two views.
            alpha: Weight of cross-view reconstruction.

        Returns:
            Structured total and component losses.
        """
        view1, view2 = views
        negative1, negative2 = negative_edges
        z1 = self.encode(x, view1)
        z2 = self.encode(x, view2)

        self_losses = (
            self._loss_for_targets(z1, view1, negative1),
            self._loss_for_targets(z2, view2, negative2),
        )
        cross_losses = (
            self._loss_for_targets(z1, view2, negative2),
            self._loss_for_targets(z2, view1, negative1),
        )
        return combine_view_losses(self_losses, cross_losses, alpha)
