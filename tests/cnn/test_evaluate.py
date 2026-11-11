import pytest
import torch

from cks_optimizer.cnn.evaluate import (
    _classification_accuracy,
    _coco_map,
    _segmentation_miou,
    evaluate,
)


def test_evaluate():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.copy_(torch.tensor([1.0, 0.0]))

    result = evaluate(model, [(torch.zeros(1, 2, 2), 0)], task="classification")

    assert result.accuracy == 1.0
    assert model.training
    assert evaluate(model, [], task="classification", enabled=False) is None


def test__classification_accuracy():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.copy_(torch.tensor([1.0, 0.0]))

    result = _classification_accuracy(
        model, [(torch.zeros(1, 2, 2), 0)], 1, torch.device("cpu")
    )

    assert result.accuracy == 1.0


def test__segmentation_miou():
    logits = torch.tensor([[[[2.0, 0.0]], [[0.0, 2.0]]]])
    result = _segmentation_miou(
        torch.nn.Identity(),
        [(logits[0], torch.tensor([[0, 1]]))],
        1,
        torch.device("cpu"),
    )

    assert result.mean_iou == 1.0


def test__coco_map_requires_annotations():
    with pytest.raises(ValueError, match="COCO dataset with annotations"):
        _coco_map(torch.nn.Identity(), [], 1, torch.device("cpu"))
