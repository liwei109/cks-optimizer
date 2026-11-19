import json
from types import SimpleNamespace

from cks_optimizer import cli


def test_run_lm(monkeypatch, capsys, tmp_path):
    captured = {}

    class Optimizer:
        def optimize(self, **kwargs):
            captured.update(kwargs)
            return {"status": "completed"}

    monkeypatch.setattr(cli, "LMOptimizer", Optimizer)

    assert cli.main(["lm", "--model", "org/model", "--output-dir", str(tmp_path)]) == 0
    assert captured["model"] == "org/model"
    assert captured["output_dir"] == str(tmp_path)
    assert json.loads(capsys.readouterr().out) == {"status": "completed"}


def test_run_cnn(monkeypatch, capsys):
    config = SimpleNamespace(
        effective_algorithms=["int8"],
        model_name="test-model",
        dataset_root="dataset",
        task="classification",
        quant_mode="int8",
    )
    result = SimpleNamespace(
        config=config,
        algorithms=("quantization",),
        onnx_path="output.onnx",
        evaluation=None,
    )
    monkeypatch.setattr(cli, "build_optimize_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(cli, "run_cnn_pipeline", lambda **kwargs: result)

    assert cli.main(["cnn", "--model", "test-model", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "group": "cnn",
        "model": "test-model",
        "dataset": "dataset",
        "task": "classification",
        "quant_mode": "int8",
        "algorithms": ["quantization"],
        "output": "output.onnx",
    }
