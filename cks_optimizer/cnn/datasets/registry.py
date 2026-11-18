"""Built-in dataset handlers (ImageNet, COCO, VOC) and compatibility APIs.

Add a new dataset by subclassing :class:`~cks_optimizer.cnn.datasets.handler.
DatasetHandler` and decorating it with
:func:`~cks_optimizer.cnn.datasets.handler.register` — see the handlers below
for reference implementations.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from cks_optimizer.cnn.datasets.handler import (
    DATASET_HANDLER_REGISTRY,
    DatasetHandler,
    get_handler,
    image_transform,
    layout_search_dirs,
    mask_transform,
    normalize_dataset,
    register,
)

__all__ = [
    "DATASET_REGISTRY",
    "ImageNetHandler",
    "CocoHandler",
    "VocHandler",
    "get_dataset",
    "normalize_dataset",
]


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


def _yolov5_letterbox_transform(shape: tuple[int, ...]) -> Callable[[Image.Image], Tensor]:
    """Resize with aspect ratio preserved and YOLOv5's 114-valued padding."""
    from torchvision.transforms import functional as TF

    height, width = shape[2:]

    def transform(image: Image.Image) -> Tensor:
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


@register
class ImageNetHandler(DatasetHandler):
    """ImageNet-style classification folders (``train/<class>/*.jpg``)."""

    name = "imagenet"
    task = "classification"
    aliases = ("imagenet1k",)
    torchvision_dataset = "torchvision.datasets.ImageFolder"

    def matches_layout(self, root: Path) -> bool:
        return True  # Fallback dataset: any layout is potentially ImageNet.

    def build(
        self,
        root: Path,
        split: str,
        *,
        image_transform: Callable[[Image.Image], Tensor],
        target_transform: Any = None,
    ) -> Dataset[Any]:
        from torchvision.datasets import ImageFolder

        split_root = _imagenet_root(root, split)
        if split.lower() in {"val", "validation"}:
            validation = _official_imagenet_validation(root, split_root, image_transform)
            if validation is not None:
                return validation
        return ImageFolder(split_root, transform=image_transform)

    def make_transforms(
        self,
        shape: tuple[int, ...],
        mean: tuple[float, ...],
        std: tuple[float, ...],
    ) -> tuple[Callable[[Image.Image], Tensor], Any]:
        return image_transform(shape, mean, std), None

    def model_image_transform(
        self, model: Any, shape: tuple[int, ...]
    ) -> Callable[[Image.Image], Tensor] | None:
        if model is None:
            return None
        return getattr(model, "_cks_image_transform", None)

    def validate_splits(self, training: Any, evaluation: Any) -> None:
        if getattr(evaluation, "class_to_idx", {}) != getattr(
            training, "class_to_idx", {}
        ):
            raise ValueError(
                "ImageNet training and evaluation splits must use the same class directories."
            )

    def extract_metadata(self, training: Any) -> dict[str, Any]:
        return dict(getattr(training, "class_to_idx", {}))


@register
class CocoHandler(DatasetHandler):
    """COCO instance detection with ``instances_<split>.json`` annotations."""

    name = "coco"
    task = "detection"
    aliases = ("coco2017", "coco2014")
    torchvision_dataset = "torchvision.datasets.CocoDetection"
    layout_priority = 10

    def matches_layout(self, root: Path) -> bool:
        annotation_dirs = [
            directory
            for directory in layout_search_dirs(root)
            if directory == root or directory.name.lower() == "annotations"
        ]
        return any(
            any("instances" in path.name.lower() for path in directory.glob("*.json"))
            for directory in annotation_dirs
        )

    def build(
        self,
        root: Path,
        split: str,
        *,
        image_transform: Callable[[Image.Image], Tensor],
        target_transform: Any = None,
    ) -> Dataset[Any]:
        from torchvision.datasets import CocoDetection

        image_dir, annotation_file = _coco_paths(root, split)
        return CocoDetection(
            image_dir,
            annotation_file,
            transform=image_transform,
            target_transform=target_transform,
        )

    def make_transforms(
        self,
        shape: tuple[int, ...],
        mean: tuple[float, ...],
        std: tuple[float, ...],
    ) -> tuple[Callable[[Image.Image], Tensor], Any]:
        return image_transform(shape, mean, std, normalize=False), None

    def model_image_transform(
        self, model: Any, shape: tuple[int, ...]
    ) -> Callable[[Image.Image], Tensor] | None:
        if model is not None and getattr(model, "_cks_yolov5_raw_output", False):
            return _yolov5_letterbox_transform(shape)
        return None


@register
class VocHandler(DatasetHandler):
    """PASCAL VOC 2012 segmentation (``JPEGImages`` + ``SegmentationClass``)."""

    name = "voc"
    task = "segmentation"
    aliases = ("voc2012", "voc2007")
    torchvision_dataset = "torchvision.datasets.VOCSegmentation"
    layout_priority = 20

    def matches_layout(self, root: Path) -> bool:
        return any(
            (directory / "JPEGImages").is_dir()
            and (directory / "SegmentationClass").is_dir()
            for directory in layout_search_dirs(root)
        )

    def build(
        self,
        root: Path,
        split: str,
        *,
        image_transform: Callable[[Image.Image], Tensor],
        target_transform: Any = None,
    ) -> Dataset[Any]:
        from torchvision.datasets import VOCSegmentation

        return VOCSegmentation(
            _voc_root(root),
            year="2012",
            image_set=split,
            transform=image_transform,
            target_transform=target_transform,
            download=False,
        )

    def make_transforms(
        self,
        shape: tuple[int, ...],
        mean: tuple[float, ...],
        std: tuple[float, ...],
    ) -> tuple[Callable[[Image.Image], Tensor], Any]:
        return image_transform(shape, mean, std), mask_transform(shape)


# Compatibility constant: canonical dataset name -> backing torchvision class.
DATASET_REGISTRY = {
    handler.name: handler.torchvision_dataset
    for handler in DATASET_HANDLER_REGISTRY.values()
}


def get_dataset(
    dataset: str,
    root: str | Path,
    split: str,
    *,
    transform: Any,
    target_transform: Any = None,
) -> Dataset[Any]:
    """Create a registered dataset (ImageNet, COCO, or VOC) via its handler."""
    return get_handler(dataset).build(
        Path(root),
        split,
        image_transform=transform,
        target_transform=target_transform,
    )
