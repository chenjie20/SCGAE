"""Distribution-aware binary graph-type identification used by SCGAE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from edges import canonicalize_undirected

GraphType = Literal["homophilic", "heterophilic"]


@dataclass(frozen=True)
class GraphTypeResult:
    """Graph-type decision and the statistics used to obtain it."""

    graph_type: GraphType
    boundary: float
    positive_high_ratio: float
    negative_low_ratio: float
    positive_similarities: torch.Tensor
    negative_similarities: torch.Tensor


def centered_cosine_similarity(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute mean-centered cosine similarity for every supplied node pair.

    Args:
        x: Node feature matrix with shape ``[num_nodes, num_features]``.
        edge_index: Node pairs with shape ``[2, num_edges]``.
        eps: Numerical stability constant used by cosine similarity.

    Returns:
        Similarity tensor with shape ``[num_edges]``.
    """
    centered = x.float() - x.float().mean(dim=0, keepdim=True)
    source, target = edge_index
    return F.cosine_similarity(centered[source], centered[target], dim=-1, eps=eps)


def _youden_boundary(
    positive_similarities: torch.Tensor,
    negative_similarities: torch.Tensor,
) -> torch.Tensor:
    """Select the ROC boundary maximizing Youden's ``TPR - FPR`` statistic.

    Args:
        positive_similarities: Similarities of observed graph edges.
        negative_similarities: Similarities of sampled non-edges.

    Returns:
        A scalar threshold tensor on the input device.
    """
    scores = torch.cat((positive_similarities, negative_similarities))
    labels = torch.cat(
        (
            torch.ones_like(positive_similarities, dtype=torch.bool),
            torch.zeros_like(negative_similarities, dtype=torch.bool),
        )
    )
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    last_at_score = torch.ones_like(sorted_scores, dtype=torch.bool)
    last_at_score[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    threshold_indices = torch.where(last_at_score)[0]
    true_positives = sorted_labels.cumsum(0)[threshold_indices].float()
    false_positives = (~sorted_labels).cumsum(0)[threshold_indices].float()
    statistic = true_positives / positive_similarities.numel()
    statistic = statistic - false_positives / negative_similarities.numel()

    statistic = torch.cat((statistic.new_zeros(1), statistic))
    thresholds = torch.cat((scores.new_full((1,), torch.inf), sorted_scores[threshold_indices]))
    return thresholds[torch.argmax(statistic)]


@torch.no_grad()
def identify_graph_type(
    x: torch.Tensor,
    positive_edge_index: torch.Tensor,
    negative_edge_index: torch.Tensor,
    delta: float = 0.6,
) -> GraphTypeResult:
    """Identify a graph as homophilic or heterophilic without node labels.

    The method implements the manuscript's distribution-aware rule: find the
    feature-similarity boundary maximizing ``TPR - FPR`` and select the
    homophilic route only when both the positive-high and negative-low ratios
    are at least ``delta``.

    Args:
        x: Node feature matrix with shape ``[num_nodes, num_features]``.
        positive_edge_index: Observed positive edges with shape
            ``[2, num_positive_edges]``.
        negative_edge_index: Sampled non-edges with shape
            ``[2, num_negative_edges]``.
        delta: Minimum ratio for both parts of the homophilic decision.

    Returns:
        The binary decision, boundary, ratios, and edge similarities.
    """
    if not 0.0 <= delta <= 1.0:
        raise ValueError("delta must be in [0, 1]")
    positive_edges = canonicalize_undirected(positive_edge_index)
    negative_edges = canonicalize_undirected(negative_edge_index)
    if positive_edges.size(1) == 0 or negative_edges.size(1) == 0:
        raise ValueError("positive and negative edge sets must be non-empty")

    positive = centered_cosine_similarity(x, positive_edges)
    negative = centered_cosine_similarity(x, negative_edges)
    boundary = _youden_boundary(positive, negative)
    positive_high = float((positive > boundary).float().mean())
    negative_low = float((negative < boundary).float().mean())
    graph_type: GraphType = (
        "homophilic"
        if positive_high >= delta and negative_low >= delta
        else "heterophilic"
    )
    return GraphTypeResult(
        graph_type=graph_type,
        boundary=float(boundary),
        positive_high_ratio=positive_high,
        negative_low_ratio=negative_low,
        positive_similarities=positive,
        negative_similarities=negative,
    )
