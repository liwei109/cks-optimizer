"""Task-specific dataset preprocessing for CNN calibration.

Migrated from the ModelOpt torch-route demos:

  - classification (ImageNet): Resize(256) -> CenterCrop(224) -> ToTensor
    -> Normalize. The pipeline MUST match deployment preprocessing, otherwise
    calibration statistics (amax) do not reflect real activation distributions.
  - detection / segmentation (COCO / VOC): ToTensor ONLY (RGB, [0, 1], CHW).
    The detection model's internal transform performs resize + normalize;
    doing it here would double-normalize and skew calibration statistics.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cks_optimizer.cnn.data.base import VisionDatasetAdapter

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(
    task: str,
    mean: Optional[Iterable[float]] = None,
    std: Optional[Iterable[float]] = None,
) -> Any:
    """Build the preprocessing pipeline for a task.

    `mean` / `std` override the default ImageNet normalization statistics
    (only used by the classification pipeline).
    """
    if task.lower().strip() == "classification":
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=list(mean) if mean is not None else IMAGENET_MEAN,
                    std=list(std) if std is not None else IMAGENET_STD,
                ),
            ]
        )
    return torchvision.transforms.ToTensor()


def detection_collate(batch: list[Any]) -> list[Any]:
    """Keep a python list of tensors — torchvision detection models take
    List[Tensor], not a stacked batch."""
    return batch


class _CalibrationImageDataset(Dataset):
    """Adapter image paths -> preprocessed tensors."""

    def __init__(self, image_paths: list[str], transform: Any):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Any:
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.transform(image)


def build_calibration_loader(
    adapter: VisionDatasetAdapter,
    task: str,
    batch_size: int = 16,
    max_samples: Optional[int] = None,
    mean: Optional[Iterable[float]] = None,
    std: Optional[Iterable[float]] = None,
) -> Optional[DataLoader]:
    """Build a calibration DataLoader from a dataset adapter.

    This is the dataset entry point for quantization: the loader's batches
    flow through the model inside the calibration forward loop, where
    TensorQuantizer hooks collect activation statistics.

    Returns None when the adapter yields no usable image paths, so callers
    can fall back to a static sample input.
    """
    samples = list(adapter)
    if max_samples is not None:
        samples = samples[:max_samples]
    image_paths = [
        sample["image_path"]
        for sample in samples
        if sample.get("image_path")
    ]
    if not image_paths:
        return None

    transform = build_transform(task, mean=mean, std=std)
    dataset = _CalibrationImageDataset(image_paths, transform)
    collate_fn = detection_collate if task.lower().strip() == "detection" else None
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
