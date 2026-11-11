"""Prepare task datasets and calibration batches for CNN optimization."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from cks_optimizer.cnn.config import OptimizeConfig
from cks_optimizer.cnn.datasets.dataloaders import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    get_dataloader,
)

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_MEAN", "DEFAULT_STD", "ProcessedData", "data_process"]


@dataclass
class ProcessedData:
    """Data prepared for a CNN optimization run.

    Attributes:
        task: Selected task name.
        input_shape: Model input shape in ``[batch, channels, height, width]`` order.
        calib_data: Deterministically sampled calibration image batches.
        num_samples: Number of images in the complete calibration split.
        evaluation_dataset: Held-out dataset, or ``None`` if not requested.
        class_to_idx: ImageNet class-name to label-index mapping when available.
    """

    task: str = ""
    input_shape: tuple[int, ...] = (1, 3, 224, 224)
    calib_data: list[torch.Tensor] = field(default_factory=list)
    num_samples: int = 0
    evaluation_dataset: Any = None
    class_to_idx: dict[str, int] = field(default_factory=dict)


def data_process(
    config: OptimizeConfig,
    *,
    dataset: str | None = None,
    task: str | None = None,
    model: torch.nn.Module | None = None,
    include_evaluation: bool = False,
    include_calibration: bool = True,
    mean: tuple[float, ...] = DEFAULT_MEAN,
    std: tuple[float, ...] = DEFAULT_STD,
) -> tuple[ProcessedData, torch.Tensor]:
    """Load task datasets and prepare model input and calibration batches.

    The dataset loader is selected from ``dataset`` and ``task``. Supported
    pairs are ImageNet/classification, COCO/detection, and VOC/segmentation.
    If ``dataset`` is omitted, the task selects its default dataset.

    Args:
        config: CNN settings, including dataset root, split names, calibration
            limits, and device.
        dataset: Optional dataset name override.
        task: Optional task override; defaults to ``config.task``.
        model: Optional loaded model used to infer input shape and model-specific
            preprocessing or fixed batch-size metadata.
        include_evaluation: Whether to load the held-out evaluation split.
        include_calibration: Whether to load the calibration split and build
            calibration batches.
        mean: Channel means used by default image transforms.
        std: Channel standard deviations used by default image transforms.

    Returns:
        A pair containing :class:`ProcessedData` and a sample input tensor.
        ``sample_input`` is suitable for a model forward pass and ONNX export.

    Raises:
        ValueError: If no data split is requested, a split is empty, or the
            dataset/task combination is unsupported.
        TypeError: If a dataset transform does not produce CHW image tensors.
    """
    task = task or config.task
    shape = config.input_shape
    if shape is None and model is not None:
        from cks_optimizer.cnn.models import CNNModelMetadata

        shape = CNNModelMetadata.infer_input_shape(model)
    shape = shape or (1, 3, 224, 224)
    loader = get_dataloader(
        dataset, task, config, shape, mean=mean, std=std, model=model
    )
    training, evaluation = loader.splits(include_calibration, include_evaluation)
    if training is None and evaluation is None:
        raise ValueError("Request calibration data, evaluation data, or both.")

    parameter = next(model.parameters(), None) if model is not None else None
    device = (
        parameter.device
        if parameter is not None
        else torch.device(config.device or "cpu")
    )
    calib_data, calibration_count = (
        loader.calibration_batches(training, device)
        if training is not None
        else ([], 0)
    )
    num_samples = len(training) if training is not None else 0
    if training is not None:
        logger.info(
            "Loaded %d %s calibration sample(s) from %s.",
            calibration_count,
            task,
            config.dataset_root,
        )

    sample_source = training if training is not None else evaluation
    sample_input = (
        calib_data[0] if calib_data else sample_source[0][0].unsqueeze(0).to(device)
    )
    return ProcessedData(
        task=task,
        input_shape=shape,
        calib_data=calib_data,
        num_samples=num_samples,
        evaluation_dataset=evaluation,
        class_to_idx=dict(getattr(training, "class_to_idx", {}))
        if loader.dataset == "imagenet" and training is not None
        else {},
    ), sample_input
