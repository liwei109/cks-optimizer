from __future__ import annotations

import pytest
import torch


def test_load_model_rejects_code_bearing_checkpoint_by_default(tmp_path, monkeypatch):
    from cks_optimizer.cnn import models as models_module

    checkpoint = tmp_path / "weights.pt"
    torch.save({"state_dict": {"w": torch.zeros(1)}}, checkpoint)

    def unsafe_load(path, map_location=None, weights_only=True, **kwargs):  # noqa: ANN001
        if weights_only:
            raise RuntimeError("Unpickling the checkpoint requires code execution")
        return {"state_dict": {}}

    monkeypatch.setattr(torch, "load", unsafe_load)
    with pytest.raises(ValueError, match="trust_checkpoint_code"):
        models_module._load_local_torch_model(str(checkpoint), weights_only=True)


def test_nested_torchvision_model_does_not_download_backbone_weights(monkeypatch):
    import torchvision.models as torchvision_models

    from cks_optimizer.cnn import models as models_module

    called = {}

    def factory(*, weights=None, weights_backbone="DEFAULT"):
        called["weights"] = weights
        called["weights_backbone"] = weights_backbone
        return torch.nn.Identity()

    monkeypatch.setattr(
        torchvision_models.segmentation,
        "deeplabv3_resnet50",
        factory,
    )

    model = models_module._load_torchvision_model(
        "segmentation.deeplabv3_resnet50",
        pretrained=False,
    )

    assert isinstance(model, torch.nn.Module)
    assert called == {"weights": None, "weights_backbone": None}


def test_pretrained_torchvision_model_exposes_weight_preprocessing(monkeypatch):
    import torchvision.models as torchvision_models

    from cks_optimizer.cnn import models as models_module

    def create_transform():
        return lambda image: image

    class DefaultWeights:
        transforms = staticmethod(create_transform)

    class Weights:
        DEFAULT = DefaultWeights()

    called = {}

    def factory(*, weights=None):
        called["weights"] = weights
        return torch.nn.Identity()

    monkeypatch.setattr(torchvision_models, "resnet50", factory)
    monkeypatch.setattr(torchvision_models, "get_model_weights", lambda model: Weights)

    model = models_module._load_torchvision_model("resnet50", pretrained=True)

    assert called["weights"] is Weights.DEFAULT
    assert callable(model._cks_image_transform)


def test_static_yolov5_onnx_is_converted_with_dynamic_batch():
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnx2torch")
    from onnx import helper
    from onnx import numpy_helper

    from cks_optimizer.cnn import models as models_module

    shape = numpy_helper.from_array(torch.tensor([1, 3]).numpy())
    model_proto = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["shape"],
                    value=shape,
                ),
                helper.make_node("Reshape", ["images", "shape"], ["output"]),
            ],
            "static_batch_detector",
            [helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 3])],
            [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 3])],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    converted = models_module._make_yolov5_batch_dynamic(model_proto)
    assert converted is True
    from onnx2torch import convert

    pytorch_model = convert(model_proto).eval()
    assert pytorch_model(torch.zeros(2, 3)).shape == (2, 3)
