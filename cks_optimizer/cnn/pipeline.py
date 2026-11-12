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
    if not cfg.effective_algorithms and not evaluate_model:
        raise ValueError(
            "No quantization or sparsity algorithm is configured, and evaluation "
            "is disabled. Enable evaluation or configure an optimization algorithm."
        )

    # Load onnx or pytorch model.
    model_source = model if model is not None else cfg.model_name
    model = load_cnn_model(
        model_source,
        pretrained=cfg.pretrained,
        device=cfg.device or None,
    )

    # Evaluation-only runs need the validation set, but not a calibration split.
    data, sample_input = data_process(
        cfg,
        model=model,
        include_evaluation=evaluate_model and cfg.evaluation_supported,
        include_calibration=bool(cfg.effective_algorithms),
    )

    if cfg.effective_algorithms:
        optimized_model = optimize_cnn_model(
            model,
            sample_input,
            config=cfg,
            calib_data=data.calib_data,
        )
        optimized_onnx_path = export_onnx(
            optimized_model, sample_input, config=cfg
        )
    else:
        optimized_model = model
        source_path = Path(model_source) if isinstance(model_source, (str, Path)) else None
        if source_path is not None and source_path.suffix.lower() == ".onnx":
            optimized_onnx_path = source_path
        else:
            optimized_onnx_path = export_onnx(
                model, sample_input, config=cfg
            )

    evaluation = evaluate(
        optimized_onnx_path,
        data.evaluation_dataset,
        task=cfg.task,
        config=cfg,
        enabled=evaluate_model,
        input_name=cfg.export.input_names[0],
        batch_size=cfg.evaluation_batch_size,
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
