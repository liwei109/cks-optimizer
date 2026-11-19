from __future__ import annotations

import sys
from types import ModuleType

import pytest

from cks_optimizer.lm import optimizer as optimizer_module
from cks_optimizer.lm.optimizer import LMOptimizer


class TestLMOptimizer:
    def test_init(self):
        config = {"model": "test_model"}
        assert LMOptimizer(config).config is config

    def test__validate_model_id(self):
        LMOptimizer._validate_model_id("org/model")
        with pytest.raises(ValueError, match="cannot be empty"):
            LMOptimizer._validate_model_id("")
        with pytest.raises(ValueError, match="valid local path or Hugging Face repo ID"):
            LMOptimizer._validate_model_id("model")

    def test_optimize(self, monkeypatch, tmp_path):
        captured = {}

        class Oneshot:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def __call__(self):
                captured["called"] = True

        llmcompressor = ModuleType("llmcompressor")
        llmcompressor.Oneshot = Oneshot
        monkeypatch.setitem(sys.modules, "llmcompressor", llmcompressor)
        monkeypatch.setattr(LMOptimizer, "is_available", lambda self: True)
        monkeypatch.setattr(optimizer_module, "_ensure_llmcompressor_compat", lambda: None)
        output_dir = tmp_path / "output"

        result = LMOptimizer().optimize(
            "org/model", output_dir, dataset="wikitext", custom_option=True
        )

        assert captured["model"] == "org/model"
        assert captured["dataset"] == "wikitext"
        assert captured["output_dir"] == str(output_dir)
        assert captured["custom_option"] is True
        assert captured["called"] is True
        assert result["status"] == "completed"
        assert output_dir.is_dir()
