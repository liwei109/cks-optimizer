"""port contract: domain layer depends on tool-agnostic interfaces with DTO.

- adapter/modelopt.py      → CvCompressor(CNN/ViT: ModelOpt)
- adapter/llmcompressor.py → LmCompressor(LLM/VLM: llmcompressor)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Technique",
    "CvQuantizeRequest",
    "CvQuantizeResult",
    "CvSparsifyRequest",
    "CvSparsifyResult",
    "CvPruneRequest",
    "CvPruneResult",
    "LmCompressRequest",
    "LmCompressResult",
    "ToolAdapter",
    "CvCompressor",
    "LmCompressor",
]


class Technique(str, Enum):

    QUANTIZE = "quantize"
    SPARSIFY = "sparsify"
    PRUNE = "prune"


@dataclass
class CvQuantizeRequest:
    """CV quantize request: drives ModelOpt forward_loop """

    model: Any
    sample_input: Any
    quant_mode: str = "int8"  # map to ModelOpt quant cfg
    output_path: str | Path | None = None
    options: dict[str, Any] = field(default_factory=dict)
    # Full calibration set (list of batch tensors). When present the
    # adapter's forward_loop iterates all batches for calibration;
    # sample_input stays the export dummy input (first batch).
    calibration_batches: Any | None = None


@dataclass
class CvQuantizeResult:

    quantized_model: Any
    export_path: Path | None = None  # export.onnx_export


@dataclass
class CvSparsifyRequest:
    """CV sparsify request: drives ModelOpt sparsity """

    model: Any
    sample_input: Any
    sparsity_cfg: dict[str, Any] = field(default_factory=dict)


@dataclass
class CvSparsifyResult:

    model: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CvPruneRequest:
    """CV prune request(placeholder): drives ModelOpt pruning """

    model: Any
    sample_input: Any
    prune_cfg: dict[str, Any] = field(default_factory=dict)


@dataclass
class CvPruneResult:

    model: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LmCompressRequest:
    """LM compress request: model id + dataset + recipe
    llmcompressor quantize/sparsify/prune are unified by recipe(modifier list)
    """

    model: str
    output_dir: str | Path
    recipe: Any | None = None
    dataset: str = "wikitext"
    dataset_config_name: str = "wikitext-2-raw-v1"
    splits: str = "train[:1]"
    num_calibration_samples: int = 8
    max_seq_length: int = 128
    trust_remote_code_model: bool = True
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class LmCompressResult:

    output_dir: Path
    model: str
    recipe: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)  # status / ppl_eval 


@runtime_checkable
class ToolAdapter(Protocol):
    """All scene adapters share this interface for: 
    capability declaration, availability detection, idempotent wiring.
    """

    def supported_techniques(self) -> frozenset[Technique]:
        """Supported compression techniques(pipeline verifies beforehand)."""
        ...

    def is_available(self) -> bool:
        """External tool is installed and importable."""
        ...

    def install(self) -> None:
        """Idempotent wiring: compat patch, observer/backend injection, model registration.
        Called by pipeline on first use; repeated calls have no side effects.
        """
        ...


@runtime_checkable
class CvCompressor(ToolAdapter, Protocol):
    """CV(CNN/ViT) scene compressor: torch memory model driven.

    Implementation: adapter.modelopt.ModelOptAdapter。
    """

    def quantize(self, request: CvQuantizeRequest) -> CvQuantizeResult:
        """Calibrate and quantize torch model.
        If request.output_path is set, export ONNX model synchronously.
        """
        ...

    def sparsify(self, request: CvSparsifyRequest) -> CvSparsifyResult:
        """Apply sparsity to torch model.
        If request.sparsify_cfg is set, use sparsify_cfg.
        Otherwise, use default sparsity_cfg.
        """
        ...

    def prune(self, request: CvPruneRequest) -> CvPruneResult:
        """Prune torch model.
        If request.prune_cfg is set, use prune_cfg.
        Otherwise, use default pruning."""
        ...


@runtime_checkable
class LmCompressor(ToolAdapter, Protocol):
    """LM/VLM scene compressor: work directory driven.

    All compression techniques share the LmCompressRequest(recipe driven);
    if recipe is None, adapter inject default modifier(quantize: GPTQ/AWQ..., sparsify: 2:4 SparseGPT...)

    Implementation: adapter.llmcompressor.LLMCompressorAdapter。
    """

    def quantize(self, request: LmCompressRequest) -> LmCompressResult:
        """Quantize model with recipe.
        Oneshot + quantize recipe.
        """
        ...

    def sparsify(self, request: LmCompressRequest) -> LmCompressResult:
        """Apply sparsity to model.
        If request.sparsify_cfg is set, use sparsify_cfg.
        Otherwise, use default sparsity_cfg.
        """
        ...

    def prune(self, request: LmCompressRequest) -> LmCompressResult:
        """Prune model.
        If request.prune_cfg is set, use prune_cfg.
        Otherwise, use default pruning.
        """
