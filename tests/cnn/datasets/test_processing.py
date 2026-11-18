from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from cks_optimizer.cnn import CalibrationConfig, OptimizeConfig, data_process


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, inputs):
        return self.net(inputs)


def _create_image(root: Path, split: str, label: str = "cat", size: int = 64) -> None:
    image_path = root / split / label / "sample.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color=(120, 120, 120)).save(image_path)


def test_imagenet_uses_torchvision_transforms_and_validation_split(tmp_path):
    _create_image(tmp_path, "train")
    _create_image(tmp_path, "val")
    config = OptimizeConfig(
        model_name="resnet50",
        dataset_root=str(tmp_path),
        input_shape=(1, 3, 32, 32),
        calibration=CalibrationConfig(num_samples=4, batch_size=2),
    )

    data, sample_input = data_process(config, include_evaluation=True)

    assert data.task == "classification"
    assert data.num_samples == 1
    assert data.class_to_idx == {"cat": 0}
    assert len(data.calib_data) == 1
    assert data.calib_data[0].shape == (1, 3, 32, 32)
    assert sample_input is data.calib_data[0]
    assert len(data.evaluation_dataset) == 1


def test_pretrained_model_transform_is_shared_by_calibration_and_evaluation(tmp_path):
    _create_image(tmp_path, "train")
    _create_image(tmp_path, "val")
    model = TinyModel().eval()
    model._cks_image_transform = lambda image: torch.full((3, 32, 32), 0.25)
    config = OptimizeConfig(
        model_name="resnet50",
        dataset_root=str(tmp_path),
        input_shape=(1, 3, 32, 32),
    )

    data, _ = data_process(config, model=model, include_evaluation=True)

    assert torch.all(data.calib_data[0] == 0.25)
    evaluation_image, _ = data.evaluation_dataset[0]
    assert torch.all(evaluation_image == 0.25)


def test_yolov5_coco_preprocessing_preserves_aspect_ratio(tmp_path):
    root = tmp_path / "coco"
    for split in ("train2017", "val2017"):
        image_dir = root / "images" / split
        image_dir.mkdir(parents=True)
        Image.new("RGB", (64, 32), color=(255, 0, 0)).save(image_dir / "sample.jpg")
        annotation_dir = root / "annotations"
        annotation_dir.mkdir(exist_ok=True)
        (annotation_dir / f"instances_{split}.json").write_text(
            '{"images":[{"id":1,"file_name":"sample.jpg","width":64,"height":32}],'
            '"annotations":[],"categories":[{"id":1,"name":"person"}]}'
        )

    model = TinyModel().eval()
    model._cks_yolov5_raw_output = True
    config = OptimizeConfig(
        model_name="yolov5s",
        task="detection",
        dataset_root=str(root),
        input_shape=(1, 3, 32, 32),
        evaluation_split="val",
        calibration=CalibrationConfig(split="train"),
    )
    data, _ = data_process(config, model=model, include_evaluation=True)

    image, _ = data.evaluation_dataset[0]
    assert image.shape == (3, 32, 32)
    assert image[0, 0, 16].item() == pytest.approx(114 / 255)
    assert image[0, 16, 16].item() == pytest.approx(254 / 255)


def test_imagenet_reads_official_flat_validation_images_and_labels(tmp_path):
    train_class = tmp_path / "train" / "n00000001"
    train_class.mkdir(parents=True)
    Image.new("RGB", (16, 16), color=(0, 0, 0)).save(train_class / "train.png")
    val_root = tmp_path / "val"
    val_root.mkdir()
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(
        val_root / "ILSVRC2012_val_00000001.JPEG"
    )
    (tmp_path / "ILSVRC2012_validation_ground_truth.txt").write_text("1\n")
    config = OptimizeConfig(
        model_name="resnet50",
        dataset_root=str(tmp_path),
        input_shape=(1, 3, 32, 32),
    )

    data, _ = data_process(config, include_evaluation=True)

    assert data.class_to_idx == {"n00000001": 0}
    image, target = data.evaluation_dataset[0]
    assert image.shape == (3, 32, 32)
    assert target == 0


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ("", "1 images but 0 labels"),
        ("2\n", "outside the training class range"),
    ],
)
def test_imagenet_rejects_invalid_official_validation_labels(tmp_path, labels, message):
    train_class = tmp_path / "train" / "n00000001"
    train_class.mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(train_class / "train.png")
    val_root = tmp_path / "val"
    val_root.mkdir()
    Image.new("RGB", (16, 16)).save(val_root / "ILSVRC2012_val_00000001.JPEG")
    (tmp_path / "ILSVRC2012_validation_ground_truth.txt").write_text(labels)
    config = OptimizeConfig(
        model_name="resnet50",
        dataset_root=str(tmp_path),
        input_shape=(1, 3, 32, 32),
    )

    with pytest.raises(ValueError, match=message):
        data_process(config, include_evaluation=True)


def test_imagenet_infers_shape_from_model(tmp_path):
    _create_image(tmp_path, "train", size=64)

    class SizedModel(TinyModel):
        default_cfg = {"input_size": (3, 40, 40)}

    config = OptimizeConfig(model_name="tiny", dataset_root=str(tmp_path))
    data, _ = data_process(config, model=SizedModel().eval())

    assert data.input_shape == (1, 3, 40, 40)
    assert data.calib_data[0].shape == (1, 3, 40, 40)


def test_imagenet_uses_default_shape_when_model_shape_is_unknown(tmp_path):
    _create_image(tmp_path, "train", size=64)
    config = OptimizeConfig(model_name="resnet50", dataset_root=str(tmp_path))

    data, sample_input = data_process(config, model=TinyModel().eval())

    assert data.input_shape == (1, 3, 224, 224)
    assert sample_input.shape == (1, 3, 224, 224)


def test_imagenet_requires_shared_classes_in_train_and_validation(tmp_path):
    _create_image(tmp_path, "train", "cat")
    _create_image(tmp_path, "val", "dog")
    config = OptimizeConfig(
        model_name="tiny",
        dataset_root=str(tmp_path),
        input_shape=(1, 3, 32, 32),
    )

    with pytest.raises(ValueError, match="same class directories"):
        data_process(config, include_evaluation=True)


def test_empty_calibration_split_fails_instead_of_using_random_input(tmp_path):
    (tmp_path / "train").mkdir()
    config = OptimizeConfig(
        model_name="tiny",
        dataset_root=str(tmp_path),
        input_shape=(1, 3, 32, 32),
    )

    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        data_process(config)


def test_voc_uses_nearest_neighbor_masks(tmp_path):
    voc_root = tmp_path / "VOCdevkit" / "VOC2012"
    (voc_root / "JPEGImages").mkdir(parents=True)
    (voc_root / "SegmentationClass").mkdir()
    split_dir = voc_root / "ImageSets" / "Segmentation"
    split_dir.mkdir(parents=True)
    for split in ("train", "val"):
        (split_dir / f"{split}.txt").write_text("sample\n")
    Image.new("RGB", (8, 8), color=(120, 120, 120)).save(voc_root / "JPEGImages" / "sample.jpg")
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[4:, 4:] = 1
    Image.fromarray(mask).save(voc_root / "SegmentationClass" / "sample.png")
    config = OptimizeConfig(
        model_name="deeplabv3_resnet50",
        dataset_root=str(tmp_path),
        input_shape=(1, 3, 4, 4),
    )

    data, _ = data_process(config, include_evaluation=True)
    _, target = data.evaluation_dataset[0]

    assert data.task == "segmentation"
    assert target.shape == (4, 4)
    assert set(target.unique().tolist()) == {0, 1}


def test_explicit_dataset_selects_handler_over_model_name_hint(tmp_path):
    voc_root = tmp_path / "VOCdevkit" / "VOC2012"
    (voc_root / "JPEGImages").mkdir(parents=True)
    (voc_root / "SegmentationClass").mkdir()
    split_dir = voc_root / "ImageSets" / "Segmentation"
    split_dir.mkdir(parents=True)
    for split in ("train", "val"):
        (split_dir / f"{split}.txt").write_text("sample\n")
    Image.new("RGB", (8, 8), color=(120, 120, 120)).save(voc_root / "JPEGImages" / "sample.jpg")
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[4:, 4:] = 1
    Image.fromarray(mask).save(voc_root / "SegmentationClass" / "sample.png")
    config = OptimizeConfig(
        model_name="resnet50",
        dataset="voc",
        dataset_root=str(tmp_path),
        input_shape=(1, 3, 4, 4),
    )

    data, _ = data_process(config, include_evaluation=True)
    _, target = data.evaluation_dataset[0]

    assert data.task == "segmentation"
    assert target.shape == (4, 4)
    assert set(target.unique().tolist()) == {0, 1}
