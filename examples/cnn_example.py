from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cks_optimizer.cnn import (
    ExportConfig,
    OptimizeConfig,
    QuantConfig,
    run_cnn_pipeline,
)

IMAGE_SIZE = (64, 64)


class TinyResNetLike(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.backbone(x)


def _write_sample_images(root: Path, size: tuple[int, int] = IMAGE_SIZE) -> None:
    """Create a small ImageNet-format sample dataset."""
    rng = np.random.default_rng(0)
    labels = []
    image_index = 0
    for class_index in range(1, 3):
        class_dir = root / "train" / f"n{class_index:08d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(2):
            array = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
            Image.fromarray(array).save(class_dir / f"train_{class_index}_{index}.JPEG")

        for _ in range(2):
            val_dir = root / "val"
            val_dir.mkdir(parents=True, exist_ok=True)
            array = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
            Image.fromarray(array).save(val_dir / f"ILSVRC2012_val_{image_index + 1:08d}.JPEG")
            labels.append(str(class_index))
            image_index += 1
    (root / "ILSVRC2012_validation_ground_truth.txt").write_text("\n".join(labels) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    with tempfile.TemporaryDirectory() as dataset_root:
        _write_sample_images(Path(dataset_root))
        config = OptimizeConfig(
            model_name="resnet",
            dataset_root=dataset_root,
            quant=QuantConfig(mode="int8"),
            export=ExportConfig(
                output_path="tmp/resnet.int8.onnx",
                use_dynamic_axes=False,
            ),
        )

        result = run_cnn_pipeline(
            config=config,
            model=TinyResNetLike().eval(),
        )

        logging.getLogger(__name__).info("%s", result.summary())
        logging.getLogger(__name__).info("Quantized ONNX saved to: %s", result.onnx_path)
