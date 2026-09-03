#!/usr/bin/env python3
"""Convert processed ToxiGraph data for direct SCGAE input."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "processed" / "data.pt"

EDGE_FIELDS = {
    ("transductive", "train"): ("train_edge_index", "train_edge_type"),
    ("transductive", "valid"): ("train_edge_index", "train_edge_type"),
    ("transductive", "test"): ("train_edge_index", "train_edge_type"),
    ("inductive", "train"): ("train_edge_index", "train_edge_type"),
    ("inductive", "valid"): ("valid_query_edge_index", "valid_query_edge_type"),
    ("inductive", "test"): ("test_query_edge_index", "test_query_edge_type"),
    ("domain_holdout", "train"): ("train_edge_index", "train_edge_type"),
    ("domain_holdout", "valid"): ("valid_query_edge_index", "valid_query_edge_type"),
    ("domain_holdout", "test"): ("test_query_edge_index", "test_query_edge_type"),
}


def load_source() -> dict:
    """Load the tensor-only processed ToxiGraph dictionary.

    Returns:
        Processed base data and all available protocol entries.
    """
    source = torch.load(SOURCE, map_location="cpu", weights_only=True)
    if not isinstance(source, dict):
        raise TypeError("processed/data.pt must contain a dictionary")
    return source


def canonicalize_edges(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse a bidirectional graph into unique undirected typed edges.

    Args:
        edge_index: Bidirectional edge tensor with shape ``[2, num_edges]``.
        edge_type: Edge-type ID aligned with every directed edge.

    Returns:
        Canonical edge pairs and their aligned edge-type IDs.
    """
    edges: dict[tuple[int, int], int] = {}
    for (source, target), relation in zip(edge_index.t().tolist(), edge_type.tolist()):
        if source == target:
            continue
        pair = (min(source, target), max(source, target))
        relation = int(relation)
        previous = edges.get(pair)
        if previous is not None and previous != relation:
            raise ValueError(f"edge {pair} has conflicting edge types")
        edges[pair] = relation

    pairs = sorted(edges)
    canonical_edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    canonical_edge_type = torch.tensor([edges[pair] for pair in pairs], dtype=torch.long)
    return canonical_edge_index, canonical_edge_type


def type_names(rows: list[dict], id_field: str, name_field: str) -> dict[int, str]:
    """Convert stored mapping rows into an integer-to-name dictionary.

    Args:
        rows: Mapping dictionaries from the processed dataset.
        id_field: Field containing the integer identifier.
        name_field: Field containing the symbolic name.

    Returns:
        Mapping from integer identifiers to symbolic names.
    """
    return {int(row[id_field]): str(row[name_field]) for row in rows}


def convert(protocol: str, split: str) -> dict:
    """Select one processed protocol graph for direct SCGAE input.

    Args:
        protocol: ``"transductive"``, ``"inductive"``, or
            ``"domain_holdout"``.
        split: ``"train"``, ``"valid"``, or ``"test"`` message graph.

    Returns:
        Flat dictionary whose ``x`` and ``edge_index`` fields enter SCGAE.
    """
    source = load_source()
    selected = source["protocols"][protocol]
    edge_field, type_field = EDGE_FIELDS[(protocol, split)]
    edge_index, edge_type = canonicalize_edges(selected[edge_field], selected[type_field])
    x = selected["x"] if protocol == "domain_holdout" else source["x"]
    return {
        "format_version": 1,
        "dataset": str(source["dataset_name"]),
        "protocol": protocol,
        "split": split,
        "x": x.float(),
        "edge_index": edge_index,
        "node_ids": list(source["node_ids"]),
        "node_type": source["node_type"].long(),
        "edge_type": edge_type,
        "node_type_names": type_names(
            source["node_type_mapping"],
            "node_type_id",
            "node_type",
        ),
        "edge_type_names": type_names(
            source["edge_type_mapping"],
            "edge_type_id",
            "edge_type",
        ),
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
        choices=("transductive", "inductive", "domain_holdout"),
        default="domain_holdout",
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
