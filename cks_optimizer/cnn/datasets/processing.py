"""Prepare task datasets and calibration batches for CNN optimization."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, transforms
from torchvision.transforms import functional as TF

from cks_optimizer.cnn.config import OptimizeConfig
from cks_optimizer.cnn.datasets.registry import get_dataset

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_MEAN", "DEFAULT_STD", "ProcessedData", "data_process"]

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)

_TASK_DATASETS = {
    "classification": "imagenet",
    "detection": "coco",
    "segmentation": "voc",
}

_ImageTransform = Callable[[Image.Image], torch.Tensor]
_TargetTransform = Callable[[Any], Any] | None
_TransformFactory = Callable[
    [tuple[int, ...], tuple[float, ...], tuple[float, ...]],
    tuple[_ImageTransform, _TargetTransform],
]


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


def _image_transform(
    shape: tuple[int, ...],
    mean: tuple[float, ...],
    std: tuple[float, ...],
    *,
    normalize: bool = True,
) -> _ImageTransform:
    steps = [
        transforms.Resize(
            (shape[2], shape[3]),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
    ]
    if normalize:
        steps.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(steps)


def _mask_transform(shape: tuple[int, ...]) -> Callable[[Any], torch.Tensor]:
    def transform(mask: Any) -> torch.Tensor:
        resized = TF.resize(
            mask,
            (shape[2], shape[3]),
            interpolation=InterpolationMode.NEAREST,
        )
        return TF.pil_to_tensor(resized).squeeze(0).long()

    return transform


def _classification_transforms(
    shape: tuple[int, ...],
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> tuple[_ImageTransform, _TargetTransform]:
    return _image_transform(shape, mean, std), None


def _detection_transforms(
    shape: tuple[int, ...],
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> tuple[_ImageTransform, _TargetTransform]:
    return _image_transform(shape, mean, std, normalize=False), None


def _segmentation_transforms(
    shape: tuple[int, ...],
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> tuple[_ImageTransform, _TargetTransform]:
    return _image_transform(shape, mean, std), _mask_transform(shape)


# Register each dataset's image and target preprocessing here.
_TRANSFORM_FACTORIES: dict[str, _TransformFactory] = {
    "imagenet": _classification_transforms,
    "coco": _detection_transforms,
    "voc": _segmentation_transforms,
}


def _build_dataset(
    dataset_name: str,
    root: str,
    split: str,
    shape: tuple[int, ...],
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> Any:
    try:
        transform_factory = _TRANSFORM_FACTORIES[dataset_name]
    except KeyError as exc:
        raise ValueError(
            f"No preprocessing registered for dataset {dataset_name!r}. "
            f"Registered datasets: {sorted(_TRANSFORM_FACTORIES)}."
        ) from exc

    image_transform, target_transform = transform_factory(shape, mean, std)
    return get_dataset(
        dataset_name,
        root,
        split,
        transform=image_transform,
        target_transform=target_transform,
    )


def _load_datasets(
    config: OptimizeConfig,
    dataset_name: str,
    shape: tuple[int, ...],
    mean: tuple[float, ...],
    std: tuple[float, ...],
    *,
    include_evaluation: bool,
) -> tuple[Any, Any | None]:
    training = _build_dataset(
        dataset_name,
        config.dataset_root,
        config.calibration.split,
        shape,
        mean,
        std,
    )
    if len(training) == 0:
        raise ValueError(
            f"No samples found for {dataset_name} split "
            f"{config.calibration.split!r} under {config.dataset_root!r}."
        )

    evaluation = None
    if include_evaluation:
        evaluation = _build_dataset(
            dataset_name,
            config.dataset_root,
            config.evaluation_split,
            shape,
            mean,
            std,
        )
        if len(evaluation) == 0:
            raise ValueError(
                f"No samples found for evaluation split "
                f"{config.evaluation_split!r} under {config.dataset_root!r}."
            )
        if dataset_name == "imagenet" and getattr(
            evaluation, "class_to_idx", {}
        ) != getattr(training, "class_to_idx", {}):
            raise ValueError(
                "ImageNet training and evaluation splits must use the same class directories."
            )
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
    mean: tuple[float, ...] = DEFAULT_MEAN,
    std: tuple[float, ...] = DEFAULT_STD,
) -> tuple[ProcessedData, torch.Tensor]:
    """Load task datasets and prepare deterministic calibration batches.

    Add a dataset by registering its task in ``_TASK_DATASETS``, its dataset
    loader in ``datasets.registry``, and its transform factory in
    ``_TRANSFORM_FACTORIES``.
    """
    try:
        dataset_name = _TASK_DATASETS[config.task]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported CNN task {config.task!r}; expected one of {sorted(_TASK_DATASETS)}."
        ) from exc

    shape = _input_shape(config, model)
    training, evaluation = _load_datasets(
        config,
        dataset_name,
        shape,
        mean,
        std,
        include_evaluation=include_evaluation,
    )
    parameter = next(model.parameters(), None) if model is not None else None
    device = (
        parameter.device
        if parameter is not None
        else torch.device(config.device or "cpu")
    )
    calib_data, sample_count = _calibration_batches(
        training,
        dataset_name=dataset_name,
        num_samples=config.calibration.num_samples,
        batch_size=config.calibration.batch_size,
        device=device,
    )

    logger.info(
        "Loaded %d %s calibration sample(s) from %s.",
        sample_count,
        config.task,
        config.dataset_root,
    )
    processed_data = ProcessedData(
        task=config.task,
        input_shape=shape,
        calib_data=calib_data,
        num_samples=len(training),
        evaluation_dataset=evaluation,
        class_to_idx=dict(getattr(training, "class_to_idx", {}))
        if dataset_name == "imagenet"
        else {},
    )
    return processed_data, calib_data[0]
