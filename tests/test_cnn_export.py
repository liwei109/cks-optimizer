from __future__ import annotations

import os

import pytest
import torch

from cks_optimizer.cnn import CNNConfig, optimize_cnn_model
from cks_optimizer.cnn.pipeline import run_cnn_pipeline
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
