from __future__ import annotations

import os

import pytest
import torch
from PIL import Image

from cks_optimizer import cli
from cks_optimizer.cnn import CNNConfig, optimize_cnn_model
from cks_optimizer.cnn.pipeline import (
    build_calibration_loader_from_config,
    run_cnn_pipeline,
)
from cks_optimizer.common import ensure_parent_dir


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(8, 10),
        )

    def forward(self, x):
        return self.net(x)


def test_optimize_cnn_model_creates_onnx(tmp_path):
    model = TinyModel().eval()
    sample_input = torch.randn(1, 3, 32, 32)
    output_path = tmp_path / "tiny_quantized.onnx"

    config = CNNConfig(
        model_name="tiny",
        task="classification",
        dataset_kind="imagefolder",
        quant_mode="int8",
        output_path=str(output_path),
        use_dynamic_axes=False,
    )

    result = optimize_cnn_model(model, sample_input, config=config)

    assert result == output_path
    assert result.exists()
    assert result.suffix == ".onnx"
    assert os.path.getsize(result) > 0


def test_run_cnn_pipeline_requires_sample_input(tmp_path):
    model = TinyModel().eval()
    with pytest.raises(ValueError, match="sample_input"):
        run_cnn_pipeline(
            model=model,
            model_name="tiny",
            dataset_root=str(tmp_path),
            task="classification",
            dataset_kind="imagefolder",
            output_path=str(tmp_path / "pipeline.onnx"),
        )


def test_config_and_output_file_are_consistent(tmp_path):
    config = CNNConfig(output_path=str(tmp_path / "custom.onnx"))
    assert config.output_file().name == "custom.onnx"
    assert config.task == "classification"


def test_ensure_parent_dir_creates_parent_folders(tmp_path):
    target = tmp_path / "nested" / "artifact" / "model.onnx"
    result = ensure_parent_dir(target)
    assert result == target
    assert result.parent.exists()


def _write_dataset_images(root, count, size=(300, 300)):
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        Image.new(
            "RGB",
            size,
            color=(10 * index % 255, 60, 120),
        ).save(root / f"img{index:03d}.jpg")


def test_build_calibration_loader_from_config_uses_config_fields(tmp_path):
    root = tmp_path / "images"
    _write_dataset_images(root, count=5)

    config = CNNConfig(
        model_name="tiny",
        task="classification",
        dataset_kind="imagefolder",
        dataset_root=str(root),
        calibration_samples=3,
        calibration_batch_size=2,
    )

    loader = build_calibration_loader_from_config(config, "classification")

    assert loader is not None
    batches = list(loader)
    assert len(batches) == 2  # 3 capped samples / batch size 2
    assert batches[0].shape == (2, 3, 224, 224)


def test_build_calibration_loader_from_config_returns_none_without_root():
    config = CNNConfig(model_name="tiny", task="classification")

    assert (
        build_calibration_loader_from_config(config, "classification") is None
    )


def test_cli_cnn_calibrates_on_dataset_images(tmp_path, monkeypatch):
    # 17 images > default batch size 16, so a full 16-image batch can only
    # come from the calibration loader (the static sample input is batch 1).
    root = tmp_path / "images"
    _write_dataset_images(root, count=17)

    model = TinyModel().eval()
    batch_sizes = []
    model.register_forward_hook(
        lambda m, args, output: batch_sizes.append(args[0].shape[0])
    )

    def fake_resolve(model_spec, *, model_kind="auto", input_shape=None):
        return model, torch.randn(1, 3, 32, 32)

    monkeypatch.setattr(
        cli,
        "resolve_cnn_model_and_sample_input",
        fake_resolve,
    )

    output_path = tmp_path / "cli_calib.onnx"
    exit_code = cli.main(
        [
            "cnn",
            "--model",
            "tiny",
            "--dataset",
            str(root),
            "--quant-mode",
            "int8",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    # 17 images / batch 16 -> calibration batches of 16 and 1, plus the
    # single-sample ONNX export trace.
    assert 16 in batch_sizes
    assert batch_sizes.count(1) >= 1
