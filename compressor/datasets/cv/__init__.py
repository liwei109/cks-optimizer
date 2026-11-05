"""Data processing adapters for CNN / CV workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from compressor.datasets.cv.base import (
    CustomDatasetAdapter,
    SUPPORTED_IMAGE_EXTENSIONS,
    VisionDatasetAdapter,
    iter_cv_batches,
    load_image_tensor,
)
from compressor.datasets.cv.coco import CocoDatasetAdapter
from compressor.datasets.cv.imagenet import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ImageFolderDatasetAdapter,
    ImageNetDatasetAdapter,
)
from compressor.datasets.cv.voc import VocDatasetAdapter

CV_DATASET_REGISTRY: dict[str, type[VisionDatasetAdapter]] = {
    "imagefolder": ImageFolderDatasetAdapter,
    "imagenet": ImageNetDatasetAdapter,
    "coco": CocoDatasetAdapter,
    "voc": VocDatasetAdapter,
    "custom": CustomDatasetAdapter,
}


def infer_dataset_kind_from_root(dataset_root: str | Path | None) -> str:
    root = Path(dataset_root) if dataset_root else Path(".")
    if not root.exists() or not root.is_dir():
        return "imagefolder"

    if (root / "JPEGImages").exists() and (root / "Annotations").exists():
        return "voc"

    json_candidates = list(root.rglob("*.json"))
    if any(
        "instances" in candidate.name.lower()
        for candidate in json_candidates
    ):
        return "coco"
    if json_candidates:
        return "custom"

    return "imagefolder"


def get_cv_dataset(
    kind: str,
    root: str,
    split: str = "train",
    **kwargs: Any,
) -> VisionDatasetAdapter:
    resolved = (kind or "auto").strip().lower()
    if resolved == "auto":
        resolved = infer_dataset_kind_from_root(root)
    adapter_cls = CV_DATASET_REGISTRY.get(resolved)
    if adapter_cls is None:
        raise ValueError(
            f"Unsupported dataset kind: {kind}; "
            f"supported: {sorted(CV_DATASET_REGISTRY)} + auto"
        )
    return adapter_cls(root=root, split=split, **kwargs)


__all__ = [
    "CV_DATASET_REGISTRY",
    "CocoDatasetAdapter",
    "CustomDatasetAdapter",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "ImageFolderDatasetAdapter",
    "ImageNetDatasetAdapter",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "VisionDatasetAdapter",
    "VocDatasetAdapter",
    "get_cv_dataset",
    "infer_dataset_kind_from_root",
    "iter_cv_batches",
    "load_image_tensor",
]
