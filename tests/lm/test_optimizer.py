from __future__ import annotations

from unittest.mock import patch

import pytest

from cks_optimizer.lm.optimizer import LMOptimizer


def test_optimizer_rejects_invalid_model_ids(tmp_path):
    with pytest.raises(ValueError):
        LMOptimizer().optimize(
            model="tiny",
            output_dir=tmp_path / "invalid-lm",
        )


def test_optimizer_rejects_missing_model_name(tmp_path):
    with pytest.raises(ValueError):
        LMOptimizer().optimize(
            model="",
            output_dir=tmp_path / "invalid-lm",
        )


def test_optimizer_passes_compression_options_without_loading_model(
    tmp_path,
):
    output_dir = tmp_path / "lm-output"

    with patch("llmcompressor.Oneshot") as mock_oneshot:
        mock_instance = mock_oneshot.return_value
        mock_instance.return_value = None

        optimizer = LMOptimizer()
        result = optimizer.optimize(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            output_dir=output_dir,
            recipe="quant_8bit",
            dataset="wikitext",
            dataset_config_name="wikitext-2-raw-v1",
            splits="train[:1]",
            num_calibration_samples=1,
            max_seq_length=32,
            trust_remote_code_model=True,
        )

    assert result["group"] == "lm"
    assert result["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert result["status"] == "completed"
    assert mock_oneshot.called

    call_kwargs = mock_oneshot.call_args.kwargs
    assert call_kwargs["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert call_kwargs["recipe"] == "quant_8bit"
    assert call_kwargs["dataset"] == "wikitext"
    assert call_kwargs["dataset_config_name"] == "wikitext-2-raw-v1"
    assert call_kwargs["splits"] == "train[:1]"
    assert call_kwargs["num_calibration_samples"] == 1
    assert call_kwargs["max_seq_length"] == 32
    assert call_kwargs["trust_remote_code_model"] is True
