import onnx
import torch

from cks_optimizer.cnn import ExportConfig, OptimizeConfig, export_onnx
from cks_optimizer.cnn.export import ensure_parent_dir


def test_export_onnx(tmp_path):
    config = OptimizeConfig(
        model_name="test",
        export=ExportConfig(
            output_path=str(tmp_path / "model.onnx"),
            simplify=False,
            input_names=["images"],
            output_names=["scores"],
        ),
    )
    path = export_onnx(torch.nn.Flatten(), torch.randn(1, 3, 4, 4), config)
    graph_input = onnx.load(str(path)).graph.input[0]
    assert path.is_file()
    assert graph_input.name == "images"
    assert graph_input.type.tensor_type.shape.dim[0].dim_param == "batch_size"


def test_ensure_parent_dir(tmp_path):
    path = tmp_path / "nested" / "model.onnx"
    assert ensure_parent_dir(path) == path
    assert path.parent.is_dir()
