from __future__ import annotations

import pytest

from cks_optimizer.cnn.datasets.registry import (
    DATASET_REGISTRY,
    normalize_dataset,
)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("imagenet", "imagenet"),
        ("ImageNet", "imagenet"),
        ("COCO", "coco"),
        ("coco2017", "coco"),
        ("voc", "voc"),
        ("voc2012", "voc"),
    ],
)
def test_normalize_dataset(kind, expected):
    assert normalize_dataset(kind) == expected


@pytest.mark.parametrize("kind", [None, "", "custom", "ade20k"])
def test_normalize_dataset_rejects_unsupported_datasets(kind):
    with pytest.raises(ValueError, match="Supported datasets"):
        normalize_dataset(kind)


def test_registry_contains_only_task_datasets():
    assert set(DATASET_REGISTRY) == {"imagenet", "coco", "voc"}
