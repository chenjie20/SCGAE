"""Type-adaptive construction of SCGAE's complementary graph views."""

from __future__ import annotations

import torch

from edges import canonicalize_undirected
from graph_type import GraphType


def _make_generator(device: torch.device) -> torch.Generator:
    """Create a device-matched generator seeded from PyTorch's current seed.

    Args:
        device: Device on which the graph edges are stored.

    Returns:
        A device-compatible random-number generator.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(torch.initial_seed())
    return generator


def _critical_indices(
    base_indices: torch.Tensor,
    similarities: torch.Tensor,
    graph_type: GraphType,
    count: int,
) -> torch.Tensor:
    """Select source-view edges to inject into the complementary view.

    Args:
        base_indices: Indices of edges assigned to one base view.
        similarities: Similarity value aligned with every original edge.
        graph_type: Binary graph route returned by graph identification.
        count: Injection budget before the heterophilic half-budget rule.

    Returns:
        Selected indices into the original canonical edge tensor.
    """
    if count <= 0:
        return base_indices[:0]
    count = min(count, base_indices.numel())
    order = torch.argsort(similarities[base_indices])
    if graph_type == "homophilic":
        return base_indices[order[-count:]]
    if graph_type != "heterophilic":
        raise ValueError(f"unsupported graph type: {graph_type}")

    half = count // 2
    if half == 0:
        return base_indices[:0]
    return torch.cat((base_indices[order[:half]], base_indices[order[-half:]]))


def make_views(
    edge_index: torch.Tensor,
    similarities: torch.Tensor,
    graph_type: GraphType,
    overlap: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the two type-adaptive graph views used by SCGAE.

    The canonical edge set is randomly divided into two disjoint base sets.
    For a homophilic graph, each source base contributes its highest-similarity
    edges to the other view. For a heterophilic graph, each source contributes
    ``floor(K / 2)`` edges from both similarity tails, following Eq. (9).

    Args:
        edge_index: Observed positive edges with shape ``[2, num_edges]``.
        similarities: Similarities aligned with the canonicalized edge set.
        graph_type: ``"homophilic"`` or ``"heterophilic"``.
        overlap: Cross-view injection ratio in ``[0, 1]``.
        generator: Optional device-matched generator controlling the partition.

    Returns:
        Two canonical edge tensors, each with shape ``[2, num_view_edges]``.
    """
    if not 0.0 <= overlap <= 1.0:
        raise ValueError("overlap must be in [0, 1]")
    edges = canonicalize_undirected(edge_index)
    if similarities.ndim != 1 or similarities.numel() != edges.size(1):
        raise ValueError("similarities must contain one value per canonical edge")
    if generator is None:
        generator = _make_generator(edges.device)

    permutation = torch.randperm(edges.size(1), generator=generator, device=edges.device)
    split = edges.size(1) // 2
    base1 = permutation[:split]
    base2 = permutation[split:]
    count1 = int(base1.numel() * overlap)
    count2 = int(base2.numel() * overlap)
    inject1 = _critical_indices(base1, similarities, graph_type, count1)
    inject2 = _critical_indices(base2, similarities, graph_type, count2)

    view1 = canonicalize_undirected(edges[:, torch.cat((base1, inject2))])
    view2 = canonicalize_undirected(edges[:, torch.cat((base2, inject1))])
    return view1, view2
