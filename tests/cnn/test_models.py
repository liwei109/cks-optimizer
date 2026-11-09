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
