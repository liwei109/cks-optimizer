"""End-to-end CNN optimization pipeline (six stages)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from cks_optimizer.cnn.config import OptimizeConfig, build_optimize_config
from cks_optimizer.cnn.datasets.processing import ProcessedData, data_process
from cks_optimizer.cnn.evaluate import EvaluationResult, evaluate
from cks_optimizer.cnn.export import export_onnx
from cks_optimizer.cnn.models import load_cnn_model
from cks_optimizer.cnn.optimizer import optimize_cnn_model

__all__ = ["OptimizationResult", "run_cnn_pipeline"]


@dataclass
class OptimizationResult:
    """Models, data, and artifacts produced by a CNN optimization run."""

    config: OptimizeConfig
    data: ProcessedData
    model: torch.nn.Module
    sample_input: torch.Tensor
    optimized_model: torch.nn.Module
    onnx_path: Path
    algorithms: tuple[str, ...]
    evaluation: EvaluationResult | None = None

    def summary(self) -> str:
        lines = [
            f"model={self.config.model_name}",
            f"task={self.config.task}",
            f"algorithms={list(self.algorithms)}",
            f"onnx_path={self.onnx_path}",
        ]
        if self.config.quant is not None:
            lines.insert(2, f"quant_mode={self.config.quant.mode}")
        if self.evaluation is not None:
            lines.append(self.evaluation.summary())
        return "\n".join(lines)


def run_cnn_pipeline(
    config: Any = None,
    *,
    model: Any = None,
    evaluate_model: bool = True,
    providers: list[str] | None = None,
    **overrides: Any,
) -> OptimizationResult:
    """Run the CNN optimization pipeline.

    ``model`` accepts a model instance, model name, or model path and
    overrides ``config.model_name`` when provided. Calibration and export
    inputs come from the dataset where possible.
    Algorithms are selected by the configured ``quant`` and ``sparsity``
    sections.
    """
    # Normalize the config and apply explicit overrides.
    cfg = build_optimize_config(config, **overrides)
    if not cfg.effective_algorithms:
        raise ValueError(
            "No optimization algorithm configured. Add a 'quant' and/or "
            "'sparsity' section to the config."
        )

    # Load onnx or pytorch model.
    model = load_cnn_model(
        model if model is not None else cfg.model_name,
        pretrained=cfg.pretrained,
        device=cfg.device or None,
    )

    # Prepare calibration data and sample input.
    data, sample_input = data_process(
        cfg,
        model=model,
        include_evaluation=evaluate_model and cfg.evaluation_supported,
    )

    # Sparsify and quantize the model.
    optimized_model = optimize_cnn_model(
        model,
        sample_input,
        config=cfg,
        calib_data=data.calib_data,
    )

    # Export the optimized model.
    optimized_onnx_path = export_onnx(optimized_model, sample_input, config=cfg)

    # Measure metrics and analyze the optimized error.
    evaluation = evaluate(
        optimized_onnx_path,
        data.evaluation_dataset,
        task=cfg.task,
        config=cfg,
        enabled=evaluate_model,
        input_name=cfg.export.input_names[0],
        providers=providers,
    )

    return OptimizationResult(
        config=cfg,
        data=data,
        model=model,
        sample_input=sample_input,
        optimized_model=optimized_model,
        onnx_path=optimized_onnx_path,
        evaluation=evaluation,
        algorithms=tuple(cfg.effective_algorithms),
    )
