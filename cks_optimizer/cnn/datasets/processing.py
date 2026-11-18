"""Prepare task datasets and calibration batches for CNN optimization."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image

from cks_optimizer.cnn.datasets.handler import DatasetHandler, get_handler_for_task

if TYPE_CHECKING:
    from cks_optimizer.cnn.config import OptimizeConfig

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_MEAN", "DEFAULT_STD", "ProcessedData", "data_process"]

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)

_ImageTransform = Callable[[Image.Image], torch.Tensor]


@dataclass
class ProcessedData:
    """Prepared calibration inputs and the held-out evaluation dataset."""

    task: str = ""
    input_shape: tuple[int, ...] = (1, 3, 224, 224)
    calib_data: list[torch.Tensor] = field(default_factory=list)
    num_samples: int = 0
    evaluation_dataset: Any = None
    class_to_idx: dict[str, int] = field(default_factory=dict)


def _input_shape(
    config: OptimizeConfig, model: torch.nn.Module | None
) -> tuple[int, ...]:
    shape = config.input_shape
    if shape is not None:
        return shape
    if model is not None:
        from cks_optimizer.cnn.models import infer_model_input_shape

        inferred_shape = infer_model_input_shape(model)
        if inferred_shape is not None:
            return inferred_shape
    return (1, 3, 224, 224)


def _build_dataset(
    handler: DatasetHandler,
    root: str,
    split: str,
    shape: tuple[int, ...],
    mean: tuple[float, ...],
    std: tuple[float, ...],
    image_transform: _ImageTransform | None = None,
) -> Any:
    default_transform, target_transform = handler.make_transforms(shape, mean, std)
    return handler.build(
        Path(root),
        split,
        image_transform=(
            image_transform if image_transform is not None else default_transform
        ),
        target_transform=target_transform,
    )


def _load_datasets(
    config: OptimizeConfig,
    handler: DatasetHandler,
    shape: tuple[int, ...],
    mean: tuple[float, ...],
    std: tuple[float, ...],
    *,
    include_evaluation: bool,
    include_calibration: bool = True,
    image_transform: _ImageTransform | None = None,
) -> tuple[Any | None, Any | None]:
    training = None
    if include_calibration:
        training = _build_dataset(
            handler,
            config.dataset_root,
            config.calibration.split,
            shape,
            mean,
            std,
            image_transform,
        )
        if len(training) == 0:
            raise ValueError(
                f"No samples found for {handler.name} split "
                f"{config.calibration.split!r} under {config.dataset_root!r}."
            )

    evaluation = None
    if include_evaluation:
        evaluation = _build_dataset(
            handler,
            config.dataset_root,
            config.evaluation_split,
            shape,
            mean,
            std,
            image_transform,
        )
        if len(evaluation) == 0:
            raise ValueError(
                f"No samples found for evaluation split "
                f"{config.evaluation_split!r} under {config.dataset_root!r}."
            )
        if training is not None:
            handler.validate_splits(training, evaluation)
    return training, evaluation


def _calibration_batches(
    dataset: Any,
    *,
    dataset_name: str,
    num_samples: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], int]:
    sample_count = min(num_samples, len(dataset))
    indices = random.Random(0).sample(range(len(dataset)), sample_count)
    images = []
    for index in indices:
        image, _ = dataset[index]
        if not isinstance(image, torch.Tensor) or image.ndim != 3:
            raise TypeError(
                f"{dataset_name} sample {index} did not produce a CHW image tensor."
            )
        images.append(image)

    batches = [
        torch.stack(images[start : start + batch_size]).to(device)
        for start in range(0, len(images), batch_size)
    ]
    if not batches:
        raise ValueError(
            "No calibration images were produced from the selected dataset split."
        )
    return batches, sample_count


def data_process(
    config: OptimizeConfig,
    *,
    model: torch.nn.Module | None = None,
    include_evaluation: bool = False,
    include_calibration: bool = True,
    mean: tuple[float, ...] = DEFAULT_MEAN,
    std: tuple[float, ...] = DEFAULT_STD,
) -> tuple[ProcessedData, torch.Tensor]:
    """Load task datasets and prepare deterministic calibration batches.

    Add a dataset by subclassing
    :class:`~cks_optimizer.cnn.datasets.handler.DatasetHandler` and registering
    it with
    :func:`~cks_optimizer.cnn.datasets.handler.register`; see
    :mod:`cks_optimizer.cnn.datasets.registry` for the built-in handlers.
    """
    handler = get_handler_for_task(config.task)

    shape = _input_shape(config, model)
    image_transform = (
        handler.model_image_transform(model, shape) if model is not None else None
    )
    if image_transform is not None and not callable(image_transform):
        raise TypeError("Model image preprocessing must be callable.")
    training, evaluation = _load_datasets(
        config,
        handler,
        shape,
        mean,
        std,
        include_evaluation=include_evaluation,
        include_calibration=include_calibration,
        image_transform=image_transform,
    )
    if training is None and evaluation is None:
        raise ValueError(
            "At least one dataset split must be requested for calibration or evaluation."
        )
    parameter = next(model.parameters(), None) if model is not None else None
    device = (
        parameter.device
        if parameter is not None
        else torch.device(config.device or "cpu")
    )
    if training is not None:
        calibration_batch_size = config.calibration.batch_size
        max_batch_size = getattr(model, "_cks_max_batch_size", None)
        if max_batch_size is not None:
            calibration_batch_size = min(calibration_batch_size, max_batch_size)
            logger.warning(
                "Model input is fixed to batch size %d; limiting calibration batches "
                "to that size.",
                max_batch_size,
            )
        calib_data, sample_count = _calibration_batches(
            training,
            dataset_name=handler.name,
            num_samples=config.calibration.num_samples,
            batch_size=calibration_batch_size,
            device=device,
        )
        logger.info(
            "Loaded %d %s calibration sample(s) from %s.",
            sample_count,
            config.task,
            config.dataset_root,
        )
        num_samples = len(training)
    else:
        calib_data = []
        num_samples = 0

    sample_source = training if training is not None else evaluation
    if sample_source is None:
        raise ValueError("No dataset is available to create a model input.")
    if calib_data:
        sample_input = calib_data[0]
    else:
        image, _ = sample_source[0]
        sample_input = image.unsqueeze(0).to(device)

    processed_data = ProcessedData(
        task=config.task,
        input_shape=shape,
        calib_data=calib_data,
        num_samples=num_samples,
        evaluation_dataset=evaluation,
        class_to_idx=handler.extract_metadata(training)
        if training is not None
        else {},
    )
    return processed_data, sample_input
