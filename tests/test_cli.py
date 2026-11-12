from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from cks_optimizer.cli import main
from cks_optimizer.cnn.datasets import ProcessedData


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


def _make_image_folder(root, *, count: int = 4, size: int = 64):
    from PIL import Image

    class_dir = Path(root) / "cat"
    class_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        image = Image.new("RGB", (size, size), color=(40 + index, 90, 140))
        image.save(str(class_dir / f"cat{index}.png"))
    return Path(root)


def _fake_result(config, model):
    from cks_optimizer.cnn.pipeline import OptimizationResult

    return OptimizationResult(
        config=config,
        data=ProcessedData(task=config.task),
        model=model,
        sample_input=torch.empty(1, 3, 32, 32),
        optimized_model=model,
        onnx_path=Path(config.output_path),
        algorithms=tuple(config.effective_algorithms),
        evaluation=None,
    )


def test_cli_rejects_unknown_quant_mode():
    with pytest.raises(SystemExit) as excinfo:
        main(["cnn", "--model", "tiny", "--quant-mode", "int4"])
    assert excinfo.value.code == 2


def test_cli_emits_parseable_json(tmp_path, monkeypatch, capsys):
    from cks_optimizer import cli

    _make_image_folder(tmp_path / "images", count=1)
    model = TinyModel().eval()

    monkeypatch.setattr(
        cli,
        "run_cnn_pipeline",
        lambda **kwargs: _fake_result(kwargs["config"], model),
    )

    code = cli.main(
        [
            "cnn",
            "--model",
            "tiny",
            "--dataset",
            str(tmp_path / "images"),
            "--quant-mode",
            "int8",
            "--output",
            str(tmp_path / "cli.onnx"),
            "--num-samples",
            "2",
            "--batch-size",
            "2",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["group"] == "cnn"
    assert payload["quant_mode"] == "int8"
    assert payload["algorithms"] == ["quantize"]
    assert payload["output"].endswith("cli.onnx")


def test_cli_explicit_flags_override_yaml_config(tmp_path, monkeypatch, capsys):
    from cks_optimizer import cli

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model_name: tiny\npretrained: true\nexport:\n  output_path: config.onnx\n"
    )
    model = TinyModel().eval()
    captured = {}

    def fake_run_cnn_pipeline(**kwargs):
        captured["config"] = kwargs["config"]
        return _fake_result(kwargs["config"], model)

    monkeypatch.setattr(cli, "run_cnn_pipeline", fake_run_cnn_pipeline)

    code = cli.main(
        [
            "cnn",
            "--config",
            str(config_path),
            "--no-pretrained",
            "--calibration-split",
            "val2017",
            "--evaluation-split",
            "val2017",
            "--evaluation-batch-size",
            "8",
            "--output",
            str(tmp_path / "cli.onnx"),
            "--json",
        ]
    )

    assert code == 0
    assert captured["config"].pretrained is False
    assert captured["config"].calibration.split == "val2017"
    assert captured["config"].evaluation_split == "val2017"
    assert captured["config"].evaluation_batch_size == 8
    assert captured["config"].output_path == str(tmp_path / "cli.onnx")
    payload = json.loads(capsys.readouterr().out)
    assert payload["output"] == str(tmp_path / "cli.onnx")


def test_cli_unspecified_boolean_keeps_yaml_value(tmp_path, monkeypatch):
    from cks_optimizer import cli

    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_name: tiny\npretrained: true\n")
    model = TinyModel().eval()
    captured = {}

    def fake_run_cnn_pipeline(**kwargs):
        captured["pretrained"] = kwargs["config"].pretrained
        return _fake_result(kwargs["config"], model)

    monkeypatch.setattr(cli, "run_cnn_pipeline", fake_run_cnn_pipeline)

    assert cli.main(["cnn", "--config", str(config_path)]) == 0
    assert captured["pretrained"] is True


def test_cli_evaluation_only_defaults_named_model_to_pretrained(monkeypatch, capsys):
    from cks_optimizer import cli

    captured = {}

    def fake_run_cnn_pipeline(**kwargs):
        captured["config"] = kwargs["config"]
        return _fake_result(kwargs["config"], TinyModel().eval())

    monkeypatch.setattr(cli, "run_cnn_pipeline", fake_run_cnn_pipeline)

    assert cli.main(["cnn", "--model", "resnet50", "--dataset", "datasets/imagenet"]) == 0

    assert captured["config"].effective_algorithms == []
    assert captured["config"].pretrained is True
    capsys.readouterr()


def test_cli_evaluation_only_respects_no_pretrained(monkeypatch, capsys):
    from cks_optimizer import cli

    captured = {}

    def fake_run_cnn_pipeline(**kwargs):
        captured["config"] = kwargs["config"]
        return _fake_result(kwargs["config"], TinyModel().eval())

    monkeypatch.setattr(cli, "run_cnn_pipeline", fake_run_cnn_pipeline)

    assert cli.main(
        [
            "cnn",
            "--model",
            "resnet50",
            "--dataset",
            "datasets/imagenet",
            "--no-pretrained",
        ]
    ) == 0

    assert captured["config"].pretrained is False
    capsys.readouterr()


def test_cli_rejects_invalid_lm_model_name():
    assert main(["lm", "--model", "tiny"]) == 2


def test_cli_logs_lm_errors_without_writing_stdout(monkeypatch, caplog, capsys):
    class UnavailableOptimizer:
        def optimize(self, **kwargs):
            raise RuntimeError("backend unavailable")

    monkeypatch.setattr("cks_optimizer.cli.LMOptimizer", UnavailableOptimizer)

    assert main(["lm", "--model", "org/model"]) == 2

    assert capsys.readouterr().out == ""
    assert "Large-model optimization unavailable: backend unavailable" in caplog.text


def test_cli_writes_lm_result_as_json(monkeypatch, capsys):
    class SuccessfulOptimizer:
        def optimize(self, **kwargs):
            return {"status": "completed", "model": kwargs["model"]}

    monkeypatch.setattr("cks_optimizer.cli.LMOptimizer", SuccessfulOptimizer)

    assert main(["lm", "--model", "org/model"]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "status": "completed",
        "model": "org/model",
    }


def test_cli_accepts_valid_lm_model_argument_shape():
    assert main(["lm", "--model", "Qwen/Qwen2.5-0.5B-Instruct"]) in {0, 2}
