import torch

from cks_optimizer.cnn import models


def test_load_torchvision_model(monkeypatch):
    called = {}

    def factory(*, weights=None):
        called["weights"] = weights
        return torch.nn.Identity()

    monkeypatch.setattr(models.torchvision_models, "test_model", factory, raising=False)
    model = models._load_torchvision_model("test_model")

    assert isinstance(model, torch.nn.Module)
    assert called["weights"] is None


def test_load_timm_model(monkeypatch):
    called = {}
    expected = torch.nn.Identity()

    def create_model(name, *, pretrained=False, **kwargs):
        called.update(name=name, pretrained=pretrained, **kwargs)
        return expected

    monkeypatch.setattr(models.timm, "create_model", create_model)
    model = models._load_timm_model("test_model", pretrained=True, num_classes=3)

    assert model is expected
    assert called == {"name": "test_model", "pretrained": True, "num_classes": 3}


def test_load_onnx_model(tmp_path):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    model_path = tmp_path / "test_model.onnx"
    weight = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        [3, 3, 1, 1],
        np.ones((3, 3, 1, 1), dtype=np.float32).tobytes(),
        raw=True,
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv", ["input", "weight"], ["output"], kernel_shape=[1, 1]
            )
        ],
        "test_model",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 4])],
        [weight],
    )
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        model_path,
    )

    model = models._load_onnx_model(model_path)

    assert model(torch.ones(1, 3, 4, 4)).shape == (1, 3, 4, 4)
    assert models.CNNModelMetadata.for_model(model).input_shape == (1, 3, 4, 4)


def test_load_cnn_model(monkeypatch):
    expected = torch.nn.Identity()
    monkeypatch.setattr(
        models.torchvision_models, "list_models", lambda: ["test_model"]
    )
    monkeypatch.setattr(
        models, "_load_torchvision_model", lambda name, pretrained=False: expected
    )

    model = models.load_cnn_model("test_model")

    assert model is expected
    assert not model.training
