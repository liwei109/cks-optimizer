"""Dataset-specific loading and preprocessing for supported CNN tasks."""

from __future__ import annotations

import logging
import random
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, transforms
from torchvision.transforms import functional as TF

from cks_optimizer.cnn.config import OptimizeConfig
from cks_optimizer.cnn.models import CNNModelMetadata
from cks_optimizer.cnn.datasets.registry import get_dataset, normalize_dataset

logger = logging.getLogger(__name__)
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def _image_transform(
    size: tuple[int, int],
    mean: tuple[float, ...],
    std: tuple[float, ...],
    *,
    normalize: bool = True,
) -> Any:
    ops = [
        transforms.Resize(
            size, interpolation=InterpolationMode.BILINEAR, antialias=True
        ),
        transforms.ToTensor(),
    ]
    if normalize:
        ops.append(transforms.Normalize(mean, std))
    return transforms.Compose(ops)


class _Letterbox:
    def __init__(self, size: tuple[int, int]):
        self.height, self.width = size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        scale = min(self.width / image.width, self.height / image.height)
        size = round(image.width * scale), round(image.height * scale)
        resized = image.resize(size, Image.Resampling.BILINEAR)
        left = round((self.width - resized.width) / 2 - 0.1)
        top = round((self.height - resized.height) / 2 - 0.1)
        canvas = Image.new("RGB", (self.width, self.height), (114, 114, 114))
        canvas.paste(resized, (left, top))
        return TF.to_tensor(canvas)


class _Mask:
    def __init__(self, size: tuple[int, int]):
        self.size = size

    def __call__(self, mask: Image.Image) -> torch.Tensor:
        mask = TF.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)
        return TF.pil_to_tensor(mask).squeeze(0).long()


class _DatasetLoader:
    """Shared split loading and calibration batching for task loaders."""

    dataset: str
    task: str

    def __init__(
        self,
        config: OptimizeConfig,
        shape: tuple[int, ...],
        mean: tuple[float, ...],
        std: tuple[float, ...],
        model: torch.nn.Module | None,
    ):
        self.config, self.shape = config, shape
        self.mean, self.std, self.model = mean, std, model
        self.size = shape[2:]

    def transforms(self) -> tuple[Any, Any]:
        """Return image and optional target transforms for this dataset."""
        raise NotImplementedError

    def load(self, split: str) -> Any:
        """Load a configured dataset split with this loader's transforms.

        Args:
            split: Dataset split name, such as ``"train"`` or ``"val"``.

        Returns:
            The torchvision dataset for the selected split.
        """
        image, target = self.transforms()
        return get_dataset(
            self.dataset,
            self.config.dataset_root,
            split,
            transform=image,
            target_transform=target,
        )

    def splits(self, calibration: bool, evaluation: bool) -> tuple[Any, Any]:
        """Load requested calibration and evaluation datasets.

        Args:
            calibration: Whether to load the configured calibration split.
            evaluation: Whether to load the configured evaluation split.

        Returns:
            A ``(training, evaluation)`` pair; either item is ``None`` when
            its split was not requested.

        Raises:
            ValueError: If a requested split is empty or ImageNet splits have
                inconsistent class directories.
        """
        train = self.load(self.config.calibration.split) if calibration else None
        valid = self.load(self.config.evaluation_split) if evaluation else None
        for data, split in (
            (train, self.config.calibration.split),
            (valid, self.config.evaluation_split),
        ):
            if data is not None and not len(data):
                raise ValueError(
                    f"No samples found for {self.dataset} split {split!r} "
                    f"under {self.config.dataset_root!r}."
                )
        if (
            self.dataset == "imagenet"
            and train is not None
            and valid is not None
            and getattr(train, "class_to_idx", {}) != getattr(valid, "class_to_idx", {})
        ):
            raise ValueError("ImageNet splits must use the same class directories.")
        return train, valid

    def calibration_batches(
        self, dataset: Any, device: torch.device
    ) -> tuple[list[torch.Tensor], int]:
        """Select deterministic calibration samples and stack them into batches.

        Args:
            dataset: Dataset returning a CHW tensor as the first sample item.
            device: Device to which each image batch is moved.

        Returns:
            The list of image batches and the number of sampled images.

        Raises:
            TypeError: If a sample does not contain a CHW tensor.
            ValueError: If no calibration images are available.
        """
        count = min(self.config.calibration.num_samples, len(dataset))
        indices = random.Random(0).sample(range(len(dataset)), count)
        images = [dataset[i][0] for i in indices]
        if any(
            not isinstance(image, torch.Tensor) or image.ndim != 3 for image in images
        ):
            raise TypeError(f"{self.dataset} samples must produce CHW image tensors.")

        batch_size = self.config.calibration.batch_size
        metadata = (
            CNNModelMetadata.for_model(self.model) if self.model is not None else None
        )
        max_size = metadata.max_batch_size if metadata else None
        if max_size is not None:
            batch_size = min(batch_size, max_size)
            logger.warning(
                "Limiting calibration batch size to fixed model input %d.", max_size
            )
        batches = [
            torch.stack(images[i : i + batch_size]).to(device)
            for i in range(0, count, batch_size)
        ]
        if not batches:
            raise ValueError("No calibration images were produced.")
        return batches, count


class ImagenetDataloader(_DatasetLoader):
    """ImageNet classification loader with model-aware image preprocessing."""

    dataset, task = "imagenet", "classification"

    def transforms(self) -> tuple[Any, None]:
        """Return ImageNet image preprocessing and no target transform."""
        metadata = (
            CNNModelMetadata.for_model(self.model) if self.model is not None else None
        )
        image = metadata.image_transform if metadata else None
        if image is None:
            image = _image_transform(self.size, self.mean, self.std)
        if not callable(image):
            raise TypeError("Model image preprocessing must be callable.")
        return image, None


class CocoDataloader(_DatasetLoader):
    """COCO detection loader with raw-image transforms for detection models."""

    dataset, task = "coco", "detection"

    def transforms(self) -> tuple[Any, None]:
        """Return COCO image preprocessing and no target transform."""
        metadata = (
            CNNModelMetadata.for_model(self.model) if self.model is not None else None
        )
        if metadata and metadata.detection_output == "yolov5_raw":
            return _Letterbox(self.size), None
        return _image_transform(self.size, self.mean, self.std, normalize=False), None


class VocDataloader(_DatasetLoader):
    """VOC segmentation loader with nearest-neighbor mask resizing."""

    dataset, task = "voc", "segmentation"

    def transforms(self) -> tuple[Any, _Mask]:
        """Return VOC image preprocessing and a nearest-neighbor mask transform."""
        return _image_transform(self.size, self.mean, self.std), _Mask(self.size)


_LOADERS = {
    ("imagenet", "classification"): ImagenetDataloader,
    ("coco", "detection"): CocoDataloader,
    ("voc", "segmentation"): VocDataloader,
}
_TASK_DATASETS = {task: dataset for dataset, task in _LOADERS}


def get_dataloader(
    dataset: str | None,
    task: str,
    config: OptimizeConfig,
    shape: tuple[int, ...],
    *,
    mean: tuple[float, ...] = DEFAULT_MEAN,
    std: tuple[float, ...] = DEFAULT_STD,
    model: torch.nn.Module | None = None,
) -> _DatasetLoader:
    """Create the loader registered for a dataset and task.

    Args:
        dataset: Dataset name or alias. If omitted, infer it from ``task``.
        task: One of ``classification``, ``detection``, or ``segmentation``.
        config: Dataset paths, split names, and calibration settings.
        shape: Model input shape in ``[batch, channels, height, width]`` order.
        mean: Channel means used by default image transforms.
        std: Channel standard deviations used by default image transforms.
        model: Optional model metadata source for preprocessing and batch limits.

    Returns:
        The task-specific dataset loader.

    Raises:
        ValueError: If the task is unsupported or the dataset does not support
            the requested task.
    """
    if task not in _TASK_DATASETS:
        raise ValueError(f"Unsupported CNN task {task!r}.")
    name = normalize_dataset(dataset or _TASK_DATASETS[task])
    try:
        loader = _LOADERS[name, task]
    except KeyError as exc:
        raise ValueError(f"Dataset {name!r} does not support task {task!r}.") from exc
    return loader(config, shape, mean, std, model)


__all__ = [
    "CocoDataloader",
    "DEFAULT_MEAN",
    "DEFAULT_STD",
    "ImagenetDataloader",
    "VocDataloader",
    "get_dataloader",
]
