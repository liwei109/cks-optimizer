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


def _yolov5_letterbox_transform(shape: tuple[int, ...]) -> _ImageTransform:
    """Resize with aspect ratio preserved and YOLOv5's 114-valued padding."""
    height, width = shape[2:]

    def transform(image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        ratio = min(width / image.width, height / image.height)
        resized_size = (round(image.width * ratio), round(image.height * ratio))
        resized = image.resize(resized_size, Image.Resampling.BILINEAR)
        horizontal_padding = (width - resized.width) / 2
        vertical_padding = (height - resized.height) / 2
        left = round(horizontal_padding - 0.1)
        top = round(vertical_padding - 0.1)
        canvas = Image.new("RGB", (width, height), color=(114, 114, 114))
        canvas.paste(resized, (left, top))
        return TF.to_tensor(canvas)

    return transform


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
    image_transform: _ImageTransform | None = None,
) -> Any:
    try:
        transform_factory = _TRANSFORM_FACTORIES[dataset_name]
    except KeyError as exc:
        raise ValueError(
            f"No preprocessing registered for dataset {dataset_name!r}. "
            f"Registered datasets: {sorted(_TRANSFORM_FACTORIES)}."
        ) from exc

    default_transform, target_transform = transform_factory(shape, mean, std)
    return get_dataset(
        dataset_name,
        root,
        split,
        transform=(
            image_transform if image_transform is not None else default_transform
        ),
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
    include_calibration: bool = True,
    image_transform: _ImageTransform | None = None,
) -> tuple[Any | None, Any | None]:
    training = None
    if include_calibration:
        training = _build_dataset(
            dataset_name,
            config.dataset_root,
            config.calibration.split,
            shape,
            mean,
            std,
            image_transform,
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
            image_transform,
        )
        if len(evaluation) == 0:
            raise ValueError(
                f"No samples found for evaluation split "
                f"{config.evaluation_split!r} under {config.dataset_root!r}."
            )
        if dataset_name == "imagenet" and training is not None and getattr(
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
    include_calibration: bool = True,
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
    image_transform = (
        getattr(model, "_cks_image_transform", None)
        if dataset_name == "imagenet" and model is not None
        else None
    )
    if (
        dataset_name == "coco"
        and model is not None
        and getattr(model, "_cks_yolov5_raw_output", False)
    ):
        image_transform = _yolov5_letterbox_transform(shape)
    if image_transform is not None and not callable(image_transform):
        raise TypeError("Model image preprocessing must be callable.")
    training, evaluation = _load_datasets(
        config,
        dataset_name,
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
            dataset_name=dataset_name,
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
        class_to_idx=dict(getattr(training, "class_to_idx", {}))
        if dataset_name == "imagenet"
        else {},
    )
    return processed_data, sample_input
