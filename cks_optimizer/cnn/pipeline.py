from __future__ import annotations

from typing import Any

from cks_optimizer.cnn.api import optimize_cnn_model
from cks_optimizer.cnn.config import CNNConfig
from cks_optimizer.cnn.data.registry import get_dataset_adapter
from cks_optimizer.cnn.tasks import get_task_name


def run_cnn_pipeline(
    model: Any,
    model_name: str,
    dataset_root: str,
    task: str,
    dataset_kind: str = "imagefolder",
    sample_input: Any | None = None,
    output_path: str = "tmp/quantized_model.onnx",
    **kwargs: Any,
):
    """High-level pipeline for CNN optimization tasks."""
    task_name = get_task_name(task)
    adapter = get_dataset_adapter(dataset_kind, dataset_root, **kwargs)

    config = CNNConfig(
        model_name=model_name,
        task=task_name,
        dataset_kind=dataset_kind,
        dataset_root=dataset_root,
        output_path=output_path,
    )

    if sample_input is None:
        raise ValueError("sample_input is required for ModelOpt Torch quantization")

    return optimize_cnn_model(model, sample_input, config=config, output_path=output_path)
