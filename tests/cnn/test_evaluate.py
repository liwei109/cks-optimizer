from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import TensorDataset

from cks_optimizer.cnn import (
    CalibrationConfig,
    ExportConfig,
    OptimizeConfig,
    QuantConfig,
    export_onnx,
)
from cks_optimizer.cnn.datasets import data_process
from cks_optimizer.cnn.evaluate import evaluate, resolve_input_name


class ClassificationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("logits", torch.tensor([[2.0, 0.0]]))

    def forward(self, image):
        input_signal = image.sum(dim=(1, 2, 3), keepdim=True).reshape(-1, 1) * 1e-6
        return self.logits.expand(image.shape[0], -1) + input_signal


class SegmentationModel(torch.nn.Module):
    def forward(self, image):
        background = image[:, :1] * 1e-6 + 2
        foreground = torch.zeros_like(image[:, :1])
        return torch.cat((background, foreground), dim=1)


class CocoDetectionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("boxes", torch.tensor([[[4.0, 4.0, 12.0, 12.0]]]))
        self.register_buffer("labels", torch.tensor([[1]], dtype=torch.int64))
        self.register_buffer("scores", torch.tensor([[0.99]]))

    def forward(self, image):
        input_signal = image.sum() * 1e-6
        return self.boxes + input_signal, self.labels, self.scores + input_signal


def _export(
    model,
    path: Path,
    *,
    output_names: list[str] | None = None,
    dynamic_batch: bool = False,
):
    return export_onnx(
        model.eval(),
        torch.zeros(1, 3, 32, 32),
        config=OptimizeConfig(
            model_name="test-model",
            input_shape=(1, 3, 32, 32),
            export=ExportConfig(
                output_path=str(path),
                input_names=["pixels"],
                output_names=output_names or ["logits"],
                simplify=False,
                check_onnx=False,
                use_dynamic_axes=dynamic_batch,
            ),
        ),
    )


def test_classification_reports_accuracy_against_labels(tmp_path):
    onnx_path = _export(
        ClassificationModel(),
        tmp_path / "classifier.onnx",
        dynamic_batch=True,
    )
    dataset = TensorDataset(
        torch.zeros(2, 3, 32, 32),
        torch.tensor([0, 1]),
    )

    result = evaluate(onnx_path, dataset, task="classification")

    assert result.task == "classification"
    assert result.num_samples == 2
    assert result.accuracy == pytest.approx(0.5)
    assert result.summary().endswith("accuracy=0.5000")


def test_voc_reports_mean_iou(tmp_path):
    onnx_path = _export(SegmentationModel(), tmp_path / "segmenter.onnx")
    dataset = TensorDataset(
        torch.zeros(1, 3, 32, 32),
        torch.tensor([[[0, 0], [1, 255]]], dtype=torch.int64),
    )

    result = evaluate(onnx_path, dataset, task="segmentation")

    assert result.num_samples == 1
    assert result.mean_iou == pytest.approx(1 / 3)


def _create_coco_split(root: Path, split: str) -> None:
    image_dir = root / split / "images"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(100, 100, 100)).save(image_dir / "sample.jpg")
    annotation_dir = root / "annotations"
    annotation_dir.mkdir(exist_ok=True)
    payload = {
        "info": {},
        "licenses": [],
        "images": [{"id": 7, "file_name": "sample.jpg", "width": 32, "height": 32}],
        "annotations": [
            {
                "id": 1,
                "image_id": 7,
                "category_id": 1,
                "bbox": [4, 4, 8, 8],
                "area": 64,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "object", "supercategory": "object"}],
    }
    (annotation_dir / f"instances_{split}.json").write_text(json.dumps(payload))


def test_coco_reports_standard_bbox_map(tmp_path):
    pytest.importorskip("pycocotools")
    root = tmp_path / "coco"
    _create_coco_split(root, "train2017")
    _create_coco_split(root, "val2017")
    config = OptimizeConfig(
        model_name="detector",
        task="detection",
        dataset_root=str(root),
        input_shape=(1, 3, 32, 32),
        evaluation_split="val2017",
        calibration=CalibrationConfig(split="train2017"),
    )
    data, _ = data_process(config, include_evaluation=True)
    assert data.calib_data[0].mean().item() == pytest.approx(100 / 255, abs=1e-3)
    onnx_path = _export(
        CocoDetectionModel(),
        tmp_path / "detector.onnx",
        output_names=["boxes", "labels", "scores"],
    )

    result = evaluate(onnx_path, data.evaluation_dataset, task="detection")

    assert result.num_samples == 1
    assert result.mean_average_precision == pytest.approx(1.0)
    assert result.map_50 == pytest.approx(1.0)
    assert result.map_75 == pytest.approx(1.0)


def test_evaluate_skips_when_disabled():
    assert evaluate("unused.onnx", None, enabled=False) is None


def test_evaluate_skips_unsupported_quantization_mode():
    config = OptimizeConfig(model_name="resnet50", quant=QuantConfig(mode="fp8"))
    assert evaluate("unused.onnx", None, config=config) is None


def test_resolve_input_name_prefers_real_graph_input(tmp_path):
    onnx_path = _export(ClassificationModel(), tmp_path / "named.onnx")

    assert resolve_input_name(onnx_path) == "pixels"
    assert resolve_input_name(onnx_path, preferred="input") == "pixels"
    assert resolve_input_name(onnx_path, preferred="pixels") == "pixels"
