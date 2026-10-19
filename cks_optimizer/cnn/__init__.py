"""CNN / vision optimization utilities based on ModelOpt."""

from .api import optimize_cnn_model
from .config import CNNConfig

__all__ = ["CNNConfig", "optimize_cnn_model"]
