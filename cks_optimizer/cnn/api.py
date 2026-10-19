from __future__ import annotations

from typing import Any, Optional

from cks_optimizer.cnn.config import CNNConfig
from cks_optimizer.cnn.modelopt_adapter import ModelOptTorchAdapter


def optimize_cnn_model(
    model: Any,
    sample_input: Any,
    config: Optional[CNNConfig] = None,
    output_path: Optional[str] = None,
    **kwargs: Any,
):
    """Public API for CNN optimization using ModelOpt torch mode."""
    adapter = ModelOptTorchAdapter(config=config or CNNConfig())
    return adapter.optimize(model, sample_input, output_path=output_path, **kwargs)
