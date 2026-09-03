#!/usr/bin/env python3
"""Convert processed ReviewGraph data into a flat SCGAE input dictionary."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "processed" / "data.pt"

PROTOCOL_FIELDS = {
    ("transductive", "train"): ("transductive_edge_index", "transductive_edge_type"),
    ("transductive", "valid"): ("transductive_edge_index", "transductive_edge_type"),
    ("transductive", "test"): ("transductive_edge_index", "transductive_edge_type"),
    ("inductive", "train"): ("inductive_edge_index", "inductive_edge_type"),
    ("inductive", "valid"): ("valid_query_edge_index", "valid_query_edge_type"),
    ("inductive", "test"): ("test_query_edge_index", "test_query_edge_type"),
}

NODE_TYPE_NAMES = {
    0: "context",
    1: "review_element",
    2: "risk_type",
    3: "policy_clause",
}
EDGE_TYPE_NAMES = {
    0: "context-element",
    1: "element-risk",
    2: "element-policy",
    3: "risk-policy",
}
DIRECTED_TO_UNDIRECTED_TYPE = {
    0: 0,
    1: 0,
    2: 3,
    3: 3,
    4: 1,
    5: 2,
    6: 1,
    7: 2,
}


def load_source() -> dict:
    """Load the trusted tensor-only processed ReviewGraph dictionary.

    Returns:
        Processed features, protocol graphs, types, and supervision tensors.
    """
    source = torch.load(SOURCE, map_location="cpu", weights_only=True)
    if not isinstance(source, dict):
        raise TypeError("processed/data.pt must contain a dictionary")
    return source


def canonicalize_edges(
    edge_index: torch.Tensor,
    directed_edge_type: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse bidirectional ReviewGraph edges into unique undirected pairs.

    Args:
        edge_index: Bidirectional edge tensor with shape ``[2, num_edges]``.
        directed_edge_type: Direction-specific type ID for every edge.

    Returns:
        Canonical edge pairs and aligned four-way undirected edge types.
    """
    edges: dict[tuple[int, int], int] = {}
    for (source, target), directed_type in zip(
        edge_index.t().tolist(),
        directed_edge_type.tolist(),
    ):
        if source == target:
            continue
        pair = (min(source, target), max(source, target))
        edge_type = DIRECTED_TO_UNDIRECTED_TYPE[int(directed_type)]
        previous = edges.get(pair)
        if previous is not None and previous != edge_type:
            raise ValueError(f"edge {pair} has conflicting edge types")
        edges[pair] = edge_type

    pairs = sorted(edges)
    canonical_edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    canonical_edge_type = torch.tensor([edges[pair] for pair in pairs], dtype=torch.long)
    return canonical_edge_index, canonical_edge_type


def convert(protocol: str, split: str) -> dict:
    """Select one processed ReviewGraph message graph for direct SCGAE input.

    Args:
        protocol: ``"transductive"`` or ``"inductive"``.
        split: ``"train"``, ``"valid"``, or ``"test"`` message graph.

    Returns:
        Flat dictionary whose ``x`` and ``edge_index`` fields enter SCGAE.
    """
    source = load_source()
    edge_field, type_field = PROTOCOL_FIELDS[(protocol, split)]
    edge_index, edge_type = canonicalize_edges(source[edge_field], source[type_field])
    return {
        "format_version": 1,
        "dataset": "ReviewGraph",
        "protocol": protocol,
        "split": split,
        "x": source["x"].float(),
        "edge_index": edge_index,
        "node_type": source["node_type"].long(),
        "edge_type": edge_type,
        "node_type_names": NODE_TYPE_NAMES,
        "edge_type_names": EDGE_TYPE_NAMES,
        "source_edge_field": edge_field,
    }


def save_data(data: dict, output: Path) -> None:
    """Atomically save a converted SCGAE input dictionary.

    Args:
        data: Converted dataset dictionary.
        output: Destination ``.pt`` file.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(data, temporary)
    temporary.replace(output)


def parse_args() -> argparse.Namespace:
    """Parse protocol, split, and output options.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=("transductive", "inductive"),
        default="transductive",
    )
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    """Convert one processed graph and print the resulting tensor dimensions."""
    args = parse_args()
    output = args.output or ROOT / "processed" / f"scgae_{args.protocol}_{args.split}.pt"
    data = convert(args.protocol, args.split)
    save_data(data, output)
    print(
        f"saved {output}: x={tuple(data['x'].shape)}, "
        f"edge_index={tuple(data['edge_index'].shape)}"
    )


if __name__ == "__main__":
    main()
