"""Torchvision dataset construction for the supported vision tasks."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

__all__ = ["DATASET_REGISTRY", "get_dataset", "normalize_dataset"]

DATASET_REGISTRY = {
    "imagenet": "torchvision.datasets.ImageFolder",
    "coco": "torchvision.datasets.CocoDetection",
    "voc": "torchvision.datasets.VOCSegmentation",
}

_DATASET_ALIASES = {
    "imagenet1k": "imagenet",
    "coco2017": "coco",
    "coco2014": "coco",
    "voc2012": "voc",
    "voc2007": "voc",
}


def normalize_dataset(dataset: str | None) -> str:
    """Return one of the canonical dataset names."""
    key = (dataset or "").strip().lower()
    key = _DATASET_ALIASES.get(key, key)
    if key not in DATASET_REGISTRY:
        raise ValueError(
            f"Unsupported dataset: {dataset!r}. Supported datasets: {sorted(DATASET_REGISTRY)}."
        )
    return key


def _imagenet_root(root: Path, split: str) -> Path:
    split_root = root / split
    return split_root if split_root.is_dir() else root


class _ImageNetValidation(Dataset[tuple[Tensor, int]]):
    def __init__(
        self,
        image_root: Path,
        ground_truth: Path,
        class_to_idx: dict[str, int],
        transform: Callable[[Image.Image], Tensor],
    ):
        self.image_paths = sorted(
            path
            for path in image_root.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpeg", ".jpg"}
        )
        labels = [
            int(label.strip()) - 1
            for label in ground_truth.read_text().splitlines()
            if label.strip()
        ]
        if len(self.image_paths) != len(labels):
            raise ValueError(
                f"ImageNet validation has {len(self.image_paths)} images but "
                f"{len(labels)} labels in {ground_truth}."
            )
        if any(label < 0 or label >= len(class_to_idx) for label in labels):
            raise ValueError("ImageNet validation labels are outside the training class range.")
        self.targets = labels
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        with Image.open(self.image_paths[index]) as image:
            return self.transform(image.convert("RGB")), self.targets[index]


def _official_imagenet_validation(
    root: Path,
    split_root: Path,
    transform: Callable[[Image.Image], Tensor],
) -> Dataset[Any] | None:
    ground_truth = next(
        (
            path
            for path in (
                root / "ILSVRC2012_validation_ground_truth.txt",
                split_root / "ILSVRC2012_validation_ground_truth.txt",
            )
            if path.is_file()
        ),
        None,
    )
    train_root = root / "train"
    if ground_truth is None or not train_root.is_dir():
        return None

    classes = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    if not classes:
        raise ValueError(f"No ImageNet training class directories found under {train_root}.")
    class_to_idx = {name: index for index, name in enumerate(classes)}
    return _ImageNetValidation(split_root, ground_truth, class_to_idx, transform)


def _coco_paths(root: Path, split: str) -> tuple[Path, Path]:
    split_names = [split]
    normalized_split = split.lower()
    if normalized_split in {"train", "val", "validation"}:
        split_names.append(
            {"train": "train2017", "val": "val2017", "validation": "val2017"}[
                normalized_split
            ]
        )

    for split_name in dict.fromkeys(split_names):
        image_dirs = (
            root / split_name / "images",
            root / split_name,
            root / "images" / split_name,
            root / "images",
        )
        annotation_names = [f"instances_{split_name}.json"]
        if split_name == "test2017":
            annotation_names.append("image_info_test2017.json")
        annotation_files = tuple(
            candidate
            for name in annotation_names
            for candidate in (
                root / split_name / "annotations" / name,
                root / "annotations" / name,
                root / split_name / name,
                root / name,
            )
        )
        image_dir = next((path for path in image_dirs if path.is_dir()), None)
        annotation_file = next(
            (path for path in annotation_files if path.is_file()), None
        )
        if image_dir is not None and annotation_file is not None:
            return image_dir, annotation_file

    raise FileNotFoundError(
        f"COCO split {split!r} was not found under {root}. Expected an image "
        "directory and annotations/instances_<split>.json."
    )


def _voc_root(root: Path) -> Path:
    candidates = (
        root,
        root / "VOCdevkit",
        root.parent if root.name == "VOC2012" else root,
    )
    for candidate in candidates:
        if (candidate / "VOCdevkit" / "VOC2012").is_dir():
            return candidate
        if (candidate / "VOC2012").is_dir():
            return candidate.parent
        if (candidate / "JPEGImages").is_dir() and (candidate / "SegmentationClass").is_dir():
            return candidate.parent.parent
    return root


def get_dataset(
    dataset: str,
    root: str | Path,
    split: str,
    *,
    transform: Any,
    target_transform: Any = None,
) -> Dataset[Any]:
    """Create an ImageNet, COCO, or VOC dataset using torchvision."""
    from torchvision.datasets import CocoDetection, ImageFolder, VOCSegmentation

    dataset_name = normalize_dataset(dataset)
    dataset_root = Path(root)
    if dataset_name == "imagenet":
        split_root = _imagenet_root(dataset_root, split)
        if split.lower() in {"val", "validation"}:
            validation = _official_imagenet_validation(dataset_root, split_root, transform)
            if validation is not None:
                return validation
        return ImageFolder(split_root, transform=transform)
    if dataset_name == "coco":
        image_dir, annotation_file = _coco_paths(dataset_root, split)
        return CocoDetection(
            image_dir,
            annotation_file,
            transform=transform,
            target_transform=target_transform,
        )
    return VOCSegmentation(
        _voc_root(dataset_root),
        year="2012",
        image_set=split,
        transform=transform,
        target_transform=target_transform,
        download=False,
    )
