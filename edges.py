"""Tensor operations for simple undirected graphs."""

from __future__ import annotations

import torch


def canonicalize_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    """Return unique non-self-loop edges in canonical ``source < target`` form.

    Args:
        edge_index: Integer tensor with shape ``[2, num_edges]``. It may contain
            one or both directions of an undirected edge.

    Returns:
        A ``torch.long`` tensor with shape ``[2, num_unique_edges]``.
    """
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

    source, target = edge_index.long()
    low = torch.minimum(source, target)
    high = torch.maximum(source, target)
    keep = low != high
    pairs = torch.stack((low[keep], high[keep]), dim=0)
    if pairs.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    return torch.unique(pairs, dim=1)


def to_bidirectional(edge_index: torch.Tensor) -> torch.Tensor:
    """Expand canonical undirected edges into the two message-passing directions.

    Args:
        edge_index: Integer tensor with shape ``[2, num_edges]``.

    Returns:
        A coalesced edge tensor containing ``(u, v)`` and ``(v, u)``.
    """
    canonical = canonicalize_undirected(edge_index)
    if canonical.numel() == 0:
        return canonical
    return torch.cat((canonical, canonical.flip(0)), dim=1)


def _edge_ids(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Map canonical node pairs to unique scalar identifiers.

    Args:
        edge_index: Edge tensor with shape ``[2, num_edges]``.
        num_nodes: Number of nodes in the graph.

    Returns:
        One scalar identifier per unique undirected edge.
    """
    canonical = canonicalize_undirected(edge_index)
    return canonical[0] * int(num_nodes) + canonical[1]


def _make_generator(device: torch.device) -> torch.Generator:
    """Create a device-matched generator seeded from PyTorch's current seed.

    Args:
        device: Device on which random tensors will be created.

    Returns:
        A device-compatible random-number generator.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(torch.initial_seed())
    return generator


def sample_negative_edges(
    all_positive_edge_index: torch.Tensor,
    num_nodes: int,
    num_samples: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Uniformly sample unique undirected non-edges.

    Args:
        all_positive_edge_index: All node pairs that must be excluded from the
            negative set, with shape ``[2, num_positive_edges]``.
        num_nodes: Number of nodes in the graph.
        num_samples: Number of negative edges to return.
        generator: Optional device-matched generator controlling randomness.

    Returns:
        A canonical ``torch.long`` tensor with shape ``[2, num_samples]``.
    """
    device = all_positive_edge_index.device
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if num_samples == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    positive_ids = _edge_ids(all_positive_edge_index, num_nodes).sort().values
    available = num_nodes * (num_nodes - 1) // 2 - positive_ids.numel()
    if num_samples > available:
        raise ValueError("num_samples exceeds the number of available non-edges")
    if generator is None:
        generator = _make_generator(device)

    selected = torch.empty(0, dtype=torch.long, device=device)
    for _ in range(200):
        remaining = num_samples - selected.numel()
        if remaining == 0:
            break
        batch_size = max(remaining * 4, 1024)
        source = torch.randint(num_nodes, (batch_size,), generator=generator, device=device)
        target = torch.randint(num_nodes, (batch_size,), generator=generator, device=device)
        low = torch.minimum(source, target)
        high = torch.maximum(source, target)
        candidate_ids = torch.unique(low[low != high] * num_nodes + high[low != high])

        if positive_ids.numel():
            positions = torch.searchsorted(positive_ids, candidate_ids)
            safe_positions = positions.clamp(max=positive_ids.numel() - 1)
            is_positive = (positions < positive_ids.numel()) & (
                positive_ids[safe_positions] == candidate_ids
            )
            candidate_ids = candidate_ids[~is_positive]
        if selected.numel():
            candidate_ids = candidate_ids[~torch.isin(candidate_ids, selected)]
        selected = torch.cat((selected, candidate_ids[:remaining]))

    if selected.numel() != num_samples:
        raise RuntimeError("unable to sample the requested number of negative edges")
    return torch.stack((selected // num_nodes, selected % num_nodes), dim=0)
