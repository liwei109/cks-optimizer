from __future__ import annotations

import pytest
import torch
from PIL import Image

from cks_optimizer.cnn.data.base import ImageFolderDatasetAdapter
from cks_optimizer.cnn.data.preprocess import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_calibration_loader,
    build_transform,
    detection_collate,
)


def _write_image(path, size=(300, 300)):
    Image.new("RGB", size, color=(120, 60, 30)).save(path)


@pytest.fixture
def image_root(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    for index in range(5):
        _write_image(root / f"img{index}.jpg")
    return root


def test_imagenet_constants_match_demo():
    assert IMAGENET_MEAN == (0.485, 0.456, 0.406)
    assert IMAGENET_STD == (0.229, 0.224, 0.225)


def test_classification_transform_matches_imagenet_eval_pipeline(tmp_path):
    img_path = tmp_path / "img.jpg"
    _write_image(img_path, size=(300, 200))

    transform = build_transform("classification")
    tensor = transform(Image.open(img_path).convert("RGB"))

    assert tensor.shape == (3, 224, 224)


def test_classification_transform_supports_custom_mean_std(tmp_path):
    img_path = tmp_path / "img.jpg"
    _write_image(img_path)

    transform = build_transform(
        "classification",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    tensor = transform(Image.open(img_path).convert("RGB"))

    # ToTensor yields [0, 1]; (x - 0.5) / 0.5 lands in [-1, 1].
    assert -1.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0


def test_detection_transform_is_totensor_only(tmp_path):
    img_path = tmp_path / "img.jpg"
    _write_image(img_path, size=(320, 240))

    transform = build_transform("detection")
    tensor = transform(Image.open(img_path).convert("RGB"))

    # CHW layout, original resolution preserved, values in [0, 1].
    assert tensor.shape == (3, 240, 320)
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0


def test_detection_collate_keeps_list_of_tensors():
    batch = [torch.zeros(3, 4, 4), torch.zeros(3, 8, 8)]
    result = detection_collate(batch)

    assert isinstance(result, list)
    assert result[0] is batch[0]
    assert result[1] is batch[1]


def test_build_calibration_loader_stacks_classification_batches(image_root):
    adapter = ImageFolderDatasetAdapter(str(image_root))

    loader = build_calibration_loader(
        adapter,
        task="classification",
        batch_size=2,
    )
    batches = list(loader)

    assert len(batches) == 3  # 5 images, batch size 2 -> 2 + 2 + 1
    assert batches[0].shape == (2, 3, 224, 224)
    assert batches[-1].shape == (1, 3, 224, 224)


def test_build_calibration_loader_caps_samples(image_root):
    adapter = ImageFolderDatasetAdapter(str(image_root))

    loader = build_calibration_loader(
        adapter,
        task="classification",
        batch_size=4,
        max_samples=3,
    )
    batches = list(loader)

    assert len(batches) == 1
    assert batches[0].shape[0] == 3


@pytest.fixture
def large_image_root(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    for index in range(40):
        _write_image(root / f"img{index:02d}.jpg")
    return root


def test_build_calibration_loader_samples_randomly_not_prefix(large_image_root):
    adapter = ImageFolderDatasetAdapter(str(large_image_root))
    all_paths = [sample["image_path"] for sample in adapter]

    loader = build_calibration_loader(
        adapter,
        task="classification",
        batch_size=8,
        max_samples=8,
    )

    selected = loader.dataset.image_paths
    assert len(selected) == 8
    # Random draw must not degenerate to the sorted first-N prefix.
    assert selected != all_paths[:8]
    assert set(selected) <= set(all_paths)


def test_build_calibration_loader_sampling_is_reproducible(large_image_root):
    adapter = ImageFolderDatasetAdapter(str(large_image_root))

    first = build_calibration_loader(
        adapter, task="classification", batch_size=8, max_samples=8
    )
    second = build_calibration_loader(
        adapter, task="classification", batch_size=8, max_samples=8
    )
    other_seed = build_calibration_loader(
        adapter, task="classification", batch_size=8, max_samples=8, seed=7
    )

    assert first.dataset.image_paths == second.dataset.image_paths
    assert other_seed.dataset.image_paths != first.dataset.image_paths


def test_build_calibration_loader_keeps_all_when_cap_exceeds_size(image_root):
    adapter = ImageFolderDatasetAdapter(str(image_root))
    all_paths = [sample["image_path"] for sample in adapter]

    loader = build_calibration_loader(
        adapter,
        task="classification",
        batch_size=2,
        max_samples=100,
    )

    assert sorted(loader.dataset.image_paths) == sorted(all_paths)


def test_build_calibration_loader_detection_keeps_image_list(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    for index in range(3):
        _write_image(root / f"img{index}.jpg", size=(640, 480))

    adapter = ImageFolderDatasetAdapter(str(root))

    loader = build_calibration_loader(
        adapter,
        task="detection",
        batch_size=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    assert isinstance(batches[0], list)
    assert batches[0][0].shape == (3, 480, 640)
    assert isinstance(batches[1], list)
    assert batches[1][0].shape == (3, 480, 640)


def test_build_calibration_loader_returns_none_for_empty_adapter(tmp_path):
    adapter = ImageFolderDatasetAdapter(str(tmp_path / "missing"))

    loader = build_calibration_loader(adapter, task="classification")

    assert loader is None
