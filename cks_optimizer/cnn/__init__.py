"""CNN model optimization and evaluation APIs."""

from .config import (
    QUANT_MODES,
    SPARSITY_MODES,
    TRT_ONLY_QUANT_MODES,
    CalibrationConfig,
    CNNConfig,
    ExportConfig,
    OptimizeConfig,
    QuantConfig,
    QuantizerSettings,
    SparsityConfig,
    build_optimize_config,
)
from .datasets import ProcessedData, data_process
from .datasets.registry import normalize_dataset
from .evaluate import EvaluationResult, evaluate
from .export import export_onnx
from .models import (
    infer_model_input_shape,
    load_cnn_model,
    load_model,
    resolve_cnn_model,
    resolve_cnn_model_and_sample_input,
)
from .optimizer import (
    CNNOptimizer,
    optimize_cnn_model,
)
from .pipeline import OptimizationResult, run_cnn_pipeline

__all__ = [
    "CNNConfig",
    "CalibrationConfig",
    "CNNOptimizer",
    "EvaluationResult",
    "ExportConfig",
    "OptimizeConfig",
    "OptimizationResult",
    "ProcessedData",
    "QUANT_MODES",
    "QuantConfig",
    "QuantizerSettings",
    "SPARSITY_MODES",
    "SparsityConfig",
    "TRT_ONLY_QUANT_MODES",
    "build_optimize_config",
    "data_process",
    "evaluate",
    "export_onnx",
    "infer_model_input_shape",
    "load_cnn_model",
    "load_model",
    "normalize_dataset",
    "optimize_cnn_model",
    "resolve_cnn_model",
    "resolve_cnn_model_and_sample_input",
    "run_cnn_pipeline",
]
