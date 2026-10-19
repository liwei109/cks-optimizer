from __future__ import annotations

from typing import Any

from cks_optimizer.cnn.data.base import (
    CocoDatasetAdapter,
    CustomDatasetAdapter,
    ImageFolderDatasetAdapter,
    VisionDatasetAdapter,
    VocDatasetAdapter,
)


DATASET_REGISTRY: dict[str, type[VisionDatasetAdapter]] = {
    "imagefolder": ImageFolderDatasetAdapter,
    "coco": CocoDatasetAdapter,
    "voc": VocDatasetAdapter,
    "custom": CustomDatasetAdapter,
}


def get_dataset_adapter(kind: str, root: str, split: str = "train", **kwargs: Any):
    """Return a dataset adapter according to the dataset kind."""
    adapter_cls = DATASET_REGISTRY.get(kind)
    if adapter_cls is None:
        raise ValueError(f"Unsupported dataset kind: {kind}")
    return adapter_cls(root=root, split=split, **kwargs)
