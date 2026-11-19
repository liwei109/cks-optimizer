from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cks_optimizer.cnn import (
    OptimizeConfig,
    QuantConfig,
    SparsityConfig,
    optimize_cnn_model,
)
from cks_optimizer.cnn import optimizer as optimizer_module
from cks_optimizer.cnn.optimizer import CNNOptimizer


class TestCNNOptimizer:
    def test__as_optimize_config(self):
        config = optimizer_module._as_optimize_config({"model_name": "test_model"})
        assert isinstance(config, OptimizeConfig)

    def test__build_forward_loop(self):
        sample = torch.ones(1, 2)
        batches = [torch.zeros(1, 2), torch.full((1, 2), 2)]
        seen = []

        def model(batch):
            seen.append(batch)

        optimizer_module._build_forward_loop(sample, batches)(model)
        assert all(
            torch.equal(actual, expected) for actual, expected in zip(seen, batches)
        )

        seen.clear()
        optimizer_module._build_forward_loop(sample, None)(model)
        assert seen == [sample]

    def test_init(self):
        optimizer = CNNOptimizer({"model_name": "test_model"})
        assert isinstance(optimizer.config, OptimizeConfig)

    def test__validate_backend(self, monkeypatch):
        optimizer = CNNOptimizer()
        monkeypatch.setattr(optimizer_module, "mtq", None)
        with pytest.raises(ImportError, match="ModelOpt quantization backend"):
            optimizer._validate_backend()

    def test_quantize(self, monkeypatch):
        model = torch.nn.Linear(2, 1).train()
        captured = {}

        def quantize(model, config, *, forward_loop):
            captured["config"] = config
            forward_loop(model)
            return model

        monkeypatch.setattr(optimizer_module, "mtq", SimpleNamespace(quantize=quantize))
        optimizer = CNNOptimizer(OptimizeConfig(quant=QuantConfig(mode="int8")))

        result = optimizer.quantize(
            model, torch.ones(1, 2), [torch.zeros(1, 2)], custom_option=True
        )

        assert result is model
        assert not model.training
        assert captured["config"]["custom_option"] is True

    def test_sparsify(self, monkeypatch):
        model = torch.nn.Linear(2, 1).train()
        captured = {}

        def sparsify(model, mode, config):
            captured["mode"] = mode
            config["forward_loop"](model)
            return model

        monkeypatch.setattr(optimizer_module, "_sparsify", sparsify)
        optimizer = CNNOptimizer(
            OptimizeConfig(sparsity=SparsityConfig(mode="sparsegpt"))
        )

        result = optimizer.sparsify(
            model, torch.ones(1, 2), calib_data=[torch.zeros(1, 2)]
        )

        assert result is model
        assert captured["mode"] == "sparsegpt"
        assert not model.training

    def test_optimize(self, monkeypatch):
        optimizer = CNNOptimizer(
            OptimizeConfig(
                quant=QuantConfig(mode="int8"),
                sparsity=SparsityConfig(mode="sparse_magnitude"),
            )
        )
        calls = []

        def sparsify(model, sample, **kwargs):
            calls.append(("sparsify", kwargs))
            return model

        def quantize(model, sample, **kwargs):
            calls.append(("quantize", kwargs))
            return model

        monkeypatch.setattr(optimizer, "sparsify", sparsify)
        monkeypatch.setattr(optimizer, "quantize", quantize)
        model, sample = torch.nn.Identity(), torch.ones(1)

        assert (
            optimizer.optimize(
                model,
                sample,
                quantize={"custom": True},
                sparsify={"mode": "sparsegpt"},
            )
            is model
        )
        assert [name for name, _ in calls] == ["sparsify", "quantize"]
        with pytest.raises(ValueError, match="No optimization algorithm configured"):
            CNNOptimizer().optimize(model, sample)

    def test_optimize_cnn_model(self, monkeypatch):
        expected = torch.nn.Identity()

        def optimize(self, model, sample, *, calib_data=None, **kwargs):
            assert calib_data == ["batch"]
            assert kwargs == {"option": True}
            return expected

        monkeypatch.setattr(CNNOptimizer, "optimize", optimize)

        result = optimize_cnn_model(
            torch.nn.Identity(),
            torch.ones(1),
            OptimizeConfig(model_name="test_model"),
            calib_data=["batch"],
            option=True,
        )

        assert result is expected
