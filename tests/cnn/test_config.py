import pytest

from cks_optimizer.cnn import build_optimize_config


def test_build_optimize_config():
    config = build_optimize_config("recipes/cnn/base_det.yml")
    assert config.task == "detection"

    source = {"model_name": "resnet50", "device": "cpu"}
    overridden = build_optimize_config(source, device="cuda")

    assert overridden.device == "cuda"
    assert source["device"] == "cpu"
    with pytest.raises(ValueError, match="Unknown config override"):
        build_optimize_config(source, unknown_option=True)
