"""Reconstruction objectives for SCGAE."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SCGAELoss:
    """Total SCGAE objective and its two interpretable components."""

    total: torch.Tensor
    self_reconstruction: torch.Tensor
    cross_view: torch.Tensor


def edge_reconstruction_loss(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute balanced binary reconstruction loss for edges and non-edges.

    Args:
        positive_logits: Raw logits for positive target edges.
        negative_logits: Raw logits for negative target edges.

    Returns:
        Mean positive BCE plus mean negative BCE.
    """
    positive_loss = F.binary_cross_entropy_with_logits(
        positive_logits,
        torch.ones_like(positive_logits),
    )
    negative_loss = F.binary_cross_entropy_with_logits(
        negative_logits,
        torch.zeros_like(negative_logits),
    )
    return positive_loss + negative_loss


def combine_view_losses(
    self_losses: tuple[torch.Tensor, torch.Tensor],
    cross_losses: tuple[torch.Tensor, torch.Tensor],
    alpha: float,
) -> SCGAELoss:
    """Average two-view losses and apply the cross-view weight.

    Args:
        self_losses: Self-topology reconstruction losses from both views.
        cross_losses: Complementary-view reconstruction losses from both views.
        alpha: Weight of cross-view consistency reconstruction.

    Returns:
        Structured total, self-reconstruction, and cross-view losses.
    """
    self_reconstruction = (self_losses[0] + self_losses[1]) / 2
    cross_view = (cross_losses[0] + cross_losses[1]) / 2
    total = self_reconstruction + alpha * cross_view
    return SCGAELoss(
        total=total,
        self_reconstruction=self_reconstruction,
        cross_view=cross_view,
    )
