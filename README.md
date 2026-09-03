# SCGAE

Model-only implementation of **Structure-Aware Consistent Graph Autoencoder
(SCGAE)**. This repository contains the complete model path and no dataset,
training loop, evaluation metric, hyperparameter search, or experiment result.

## Structure

```text
main.py             # Public SCGAE entry point
decoder.py          # Hadamard edge representation and MLP decoder
edges.py            # Undirected edge normalization and negative sampling
encoder.py          # Shared GCN encoder
graph_type.py       # Distribution-aware binary graph-type identification
losses.py           # Self- and cross-view reconstruction objectives
model.py            # SCGAE model interface
views.py            # Type-adaptive complementary view construction

data/
├── ReviewGraph/             # processed/data.pt and one converter
└── CrossDomainReviewGraph/  # processed/data.pt and one converter
```

The two datasets are independent of the model package. Each dataset directory
contains only `processed/` and a single `convert.py`. The converter produces a
flat dictionary with the exact `x` and `edge_index` inputs described below. Run
`python data/<dataset>/convert.py --help` for protocol options.

## Input format

SCGAE uses plain PyTorch tensors and does not require a dataset wrapper.

Import the public model interface from `main.py`:

```python
from main import SCGAE
```

| Input | Shape | Dtype | Meaning |
|---|---:|---|---|
| `x` | `[num_nodes, num_features]` | floating point | Node feature matrix. |
| `edge_index` | `[2, num_edges]` | `torch.long` | Observed positive edges used for message passing. |
| `query_edge_index` | `[2, num_queries]` | `torch.long` | Node pairs whose link logits are requested. |
| `negative_edge_index` | `[2, num_negatives]` | `torch.long` | Known non-edges used by graph-type identification or reconstruction loss. |

## Installation

```bash
pip install -e .
```
