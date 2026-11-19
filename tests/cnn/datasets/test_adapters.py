from __future__ import annotations

import pytest

from cks_optimizer.cnn.datasets.registry import (
    _coco_paths,
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


def test_coco_paths_resolve_2017_splits_from_short_names(tmp_path):
    root = tmp_path / "coco"
    annotations = root / "annotations"
    annotations.mkdir(parents=True)
    for split in ("train2017", "val2017"):
        (root / split).mkdir(parents=True)
        (annotations / f"instances_{split}.json").touch()

    assert _coco_paths(root, "train") == (
        root / "train2017",
        root / "annotations" / "instances_train2017.json",
    )
    assert _coco_paths(root, "val") == (
        root / "val2017",
        root / "annotations" / "instances_val2017.json",
    )
    assert _coco_paths(root, "validation") == (
        root / "val2017",
        root / "annotations" / "instances_val2017.json",
    )


def test_coco_paths_resolve_unlabeled_test2017_images(tmp_path):
    root = tmp_path / "coco"
    (root / "images" / "test2017").mkdir(parents=True)
    (root / "annotations").mkdir()
    (root / "annotations" / "image_info_test2017.json").touch()

    assert _coco_paths(root, "test2017") == (
        root / "images" / "test2017",
        root / "annotations" / "image_info_test2017.json",
    )
