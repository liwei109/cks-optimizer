from __future__ import annotations

import torch

from cks_optimizer.cnn import ExportConfig, OptimizeConfig, export_onnx


class TinyModel(torch.nn.Module):
    def __init__(self, out_features: int = 10):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(8, out_features),
        )

    def forward(self, x):
        return self.net(x)


def test_dynamic_axes_keep_batch_dimension_symbolic(tmp_path):
    import onnx

    model = TinyModel().eval()
    onnx_path = export_onnx(
        model,
        torch.randn(1, 3, 32, 32),
        config=OptimizeConfig(
            model_name="tiny",
            input_shape=(1, 3, 32, 32),
            export=ExportConfig(
                output_path=str(tmp_path / "dynamic.onnx"),
                input_names=["pixels"],
                output_names=["logits"],
                use_dynamic_axes=True,
            ),
        ),
    )

    graph = onnx.load(str(onnx_path)).graph
    batch_dims = [dim.dim_param for inp in graph.input for dim in inp.type.tensor_type.shape.dim]
    assert any(not str(dim).isdigit() for dim in batch_dims), batch_dims


def test_dynamic_axes_apply_to_each_output(tmp_path):
    import onnx

    class MultiOutputModel(torch.nn.Module):
        def forward(self, image):
            logits = image.mean(dim=(2, 3))
            return logits, logits + 1

    onnx_path = export_onnx(
        MultiOutputModel().eval(),
        torch.randn(1, 3, 32, 32),
        config=OptimizeConfig(
            model_name="tiny",
            input_shape=(1, 3, 32, 32),
            export=ExportConfig(
                output_path=str(tmp_path / "multi-output.onnx"),
                output_names=["logits", "aux"],
                use_dynamic_axes=True,
            ),
        ),
    )

    graph = onnx.load(str(onnx_path)).graph
    for output in graph.output:
        first_dim = output.type.tensor_type.shape.dim[0]
        assert first_dim.dim_param == "batch_size"
