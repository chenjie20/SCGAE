"""Public entry point for the model-only SCGAE implementation."""

from decoder import EdgeDecoder
from edges import canonicalize_undirected, sample_negative_edges, to_bidirectional
from encoder import GCNEncoder
from graph_type import GraphTypeResult, identify_graph_type
from losses import SCGAELoss
from model import SCGAE
from views import make_views

__all__ = [
    "EdgeDecoder",
    "GCNEncoder",
    "GraphTypeResult",
    "SCGAE",
    "SCGAELoss",
    "canonicalize_undirected",
    "identify_graph_type",
    "make_views",
    "sample_negative_edges",
    "to_bidirectional",
]
