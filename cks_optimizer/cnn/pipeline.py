from __future__ import annotations

from typing import Any, Optional

from torch.utils.data import DataLoader

from cks_optimizer.cnn.api import optimize_cnn_model
from cks_optimizer.cnn.config import CNNConfig
from cks_optimizer.cnn.data.preprocess import build_calibration_loader
from cks_optimizer.cnn.data.registry import get_dataset_adapter
from cks_optimizer.cnn.tasks import get_task_name


def build_calibration_loader_from_config(
    config: CNNConfig,
    task: str,
    **adapter_kwargs: Any,
) -> Optional[DataLoader]:
    """Build a calibration DataLoader for the dataset referenced by `config`.

    The loader's batches flow through the model during calibration so that
    TensorQuantizer hooks collect statistics over real activation
    distributions. Returns None when the config carries no dataset root or
    the dataset yields no usable images, so callers can fall back to a
    static sample input.
    """
    if not config.dataset_root or not config.dataset_kind:
        return None

    adapter = get_dataset_adapter(
        config.dataset_kind,
        config.dataset_root,
        **adapter_kwargs,
    )
    return build_calibration_loader(
        adapter,
        task=task,
        batch_size=config.calibration_batch_size,
        max_samples=config.calibration_samples,
        mean=config.normalize_mean,
        std=config.normalize_std,
    )


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

    config = CNNConfig(
        model_name=model_name,
        task=task_name,
        dataset_kind=dataset_kind,
        dataset_root=dataset_root,
        output_path=output_path,
    )

    if sample_input is None:
        raise ValueError("sample_input is required for ModelOpt Torch quantization")

    calibration_loader = build_calibration_loader_from_config(
        config,
        task_name,
        **kwargs,
    )

    return optimize_cnn_model(
        model,
        sample_input,
        config=config,
        output_path=output_path,
        calibration_loader=calibration_loader,
    )
