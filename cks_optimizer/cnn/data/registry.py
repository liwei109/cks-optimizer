from __future__ import annotations

from typing import Any

from cks_optimizer.cnn.data.base import (
    CocoDatasetAdapter,
    CustomDatasetAdapter,
    ImageFolderDatasetAdapter,
    VisionDatasetAdapter,
    VocDatasetAdapter,
)


# Insertion order doubles as layout-inference priority: the first adapter
# whose matches() accepts a directory wins, so specific layouts (voc, coco)
# must be registered before generic ones (custom). "imagefolder" never
# matches and acts as the fallback kind.
DATASET_REGISTRY: dict[str, type[VisionDatasetAdapter]] = {
    "voc": VocDatasetAdapter,
    "coco": CocoDatasetAdapter,
    "custom": CustomDatasetAdapter,
    "imagefolder": ImageFolderDatasetAdapter,
}


def get_dataset_adapter(kind: str, root: str, split: str = "train", **kwargs: Any):
    """Return a dataset adapter according to the dataset kind."""
    adapter_cls = DATASET_REGISTRY.get(kind)
    if adapter_cls is None:
        raise ValueError(f"Unsupported dataset kind: {kind}")
    return adapter_cls(root=root, split=split, **kwargs)
