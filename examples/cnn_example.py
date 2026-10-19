from __future__ import annotations

from pathlib import Path

import torch

from cks_optimizer.cnn import CNNConfig, optimize_cnn_model


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


if __name__ == "__main__":
    model = TinyResNetLike().eval()
    sample_input = torch.randn(1, 3, 224, 224)
    output_path = Path("tmp/resnet.int8.onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = CNNConfig(
        model_name="resnet",
        dataset_root="./tmp",
        quant_mode="int8",
        output_path=str(output_path),
        use_dynamic_axes=False,
    )

    result = optimize_cnn_model(model, sample_input, config=config)
    print(f"Quantized ONNX saved to: {result}")
