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
    """Models, data, metrics, and artifacts produced by a CNN run.

    Attributes:
        config: Normalized configuration used for the run.
        data: Prepared calibration and held-out evaluation data.
        model: Loaded input model.
        sample_input: Representative tensor used for optimization and export.
        optimized_model: Model after configured optimization algorithms.
        onnx_path: Path to the exported ONNX model.
        algorithms: Optimization algorithms that ran, in execution order.
        evaluation: Evaluation metrics, or ``None`` when evaluation was skipped.
    """

    config: OptimizeConfig
    data: ProcessedData
    model: torch.nn.Module
    sample_input: torch.Tensor
    optimized_model: torch.nn.Module
    onnx_path: Path
    algorithms: tuple[str, ...]
    evaluation: EvaluationResult | None = None

    def summary(self) -> str:
        """Return a concise multiline summary of the optimization result."""
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
    **overrides: Any,
) -> OptimizationResult:
    """Run the CNN optimization pipeline.

    The model source may be supplied as ``model`` or ``config.model_name``.
    Calibration and export inputs are prepared from the configured dataset.
    Algorithms are selected by the configured ``quant`` and ``sparsity``
    sections; sparsification runs before quantization when both are set.

    Args:
        config: Configuration mapping, file path, or configuration object.
        model: Optional model source that overrides ``config.model_name``.
        evaluate_model: Whether to load an evaluation split and compute metrics.
        **overrides: Configuration values applied on top of ``config``.

    Returns:
        An :class:`OptimizationResult` containing the optimized model, metrics,
        and exported ONNX path.

    Raises:
        ValueError: If configuration is invalid or no work is enabled.
        RuntimeError: If CUDA is the selected device but is unavailable.
    """
    # Normalize the config and apply explicit overrides.
    cfg = build_optimize_config(config, **overrides)
    if not cfg.effective_algorithms and not evaluate_model:
        raise ValueError(
            "No quantization or sparsity algorithm is configured, and evaluation "
            "is disabled. Enable evaluation or configure an optimization algorithm."
        )

    device = torch.device(str(cfg.device).strip() or "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is the default device for CNN calibration and evaluation, but it "
            "is unavailable. Specify device='cpu' or pass --device cpu to use CPU."
        )
    cfg.device = str(device)

    # Load onnx or pytorch model.
    model_source = model if model is not None else cfg.model_name
    model = load_cnn_model(
        model_source,
        pretrained=cfg.pretrained,
        device=device,
    )

    # Evaluation-only runs need the validation set, but not a calibration split.
    data, sample_input = data_process(
        cfg,
        model=model,
        include_evaluation=evaluate_model and cfg.evaluation_supported,
        include_calibration=bool(cfg.effective_algorithms),
    )

    # Optimize the model with the configured algorithms.
    optimized_model = (
        optimize_cnn_model(
            model,
            sample_input,
            config=cfg,
            calib_data=data.calib_data,
        )
        if cfg.effective_algorithms
        else model
    )

    # Evaluate the optimized model on the held-out split.
    evaluation = evaluate(
        optimized_model,
        data.evaluation_dataset,
        task=cfg.task,
        config=cfg,
        enabled=evaluate_model,
        device=device,
        batch_size=cfg.evaluation_batch_size,
    )

    # Export the optimized model to ONNX at the configured output path.
    onnx_path = export_onnx(optimized_model.to("cpu"), sample_input.cpu(), cfg)

    return OptimizationResult(
        config=cfg,
        data=data,
        model=model,
        sample_input=sample_input,
        optimized_model=optimized_model,
        onnx_path=onnx_path,
        evaluation=evaluation,
        algorithms=tuple(cfg.effective_algorithms),
    )
