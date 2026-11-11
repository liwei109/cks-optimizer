from __future__ import annotations

import torch
from PIL import Image

from cks_optimizer.cnn import (
    ExportConfig,
    OptimizeConfig,
    QuantConfig,
    run_cnn_pipeline,
)
from cks_optimizer.cnn import models


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def test_run_cnn_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(models.torchvision_models, "list_models", lambda: ["testnet"])
    monkeypatch.setattr(
        models, "_load_torchvision_model", lambda name, pretrained=False: TinyModel()
    )
    for split in ("train", "val"):
        image_dir = tmp_path / "images" / split / "cat"
        image_dir.mkdir(parents=True)
        Image.new("RGB", (32, 32)).save(image_dir / "sample.png")

    config = OptimizeConfig(
        model_name="testnet",
        dataset_root=str(tmp_path / "images"),
        device="cpu",
        input_shape=(1, 3, 32, 32),
        quant=QuantConfig(mode="int8"),
        export=ExportConfig(
            output_path=str(tmp_path / "optimized.onnx"),
            use_dynamic_axes=False,
        ),
    )
    result = run_cnn_pipeline(config=config)

    assert result.onnx_path is not None and result.onnx_path.is_file()
    assert result.evaluation is not None
