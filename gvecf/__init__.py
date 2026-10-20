"""GVECF: Graph Variational Embedding Collaborative Filtering.

Reference implementation of
    N. S. Fazeli Dehkordi, H. Zare, P. Moradi, M. Jalili,
    "Neural Graph Collaborative Filtering Using Variational Inference",
    arXiv:2311.11824, 2023.
"""

from .data import InteractionDataset
from .metrics import evaluate
from .ngcf import NGCF
from .vgae import GraphAutoEncoder

__all__ = ["InteractionDataset", "GraphAutoEncoder", "NGCF", "evaluate"]
__version__ = "1.0.0"
