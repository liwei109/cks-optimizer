"""adapter layer: unique aware of external tools(ModelOpt / llmcompressor).
"""

from __future__ import annotations

from compressor.adapter.contract import (
    CvCompressor,
    CvPruneRequest,
    CvPruneResult,
    CvQuantizeRequest,
    CvQuantizeResult,
    CvSparsifyRequest,
    CvSparsifyResult,
    LmCompressRequest,
    LmCompressResult,
    LmCompressor,
    Technique,
    ToolAdapter,
)

__all__ = [
    "Technique",
    "ToolAdapter",
    "CvCompressor",
    "LmCompressor",
    "CvQuantizeRequest",
    "CvQuantizeResult",
    "CvSparsifyRequest",
    "CvSparsifyResult",
    "CvPruneRequest",
    "CvPruneResult",
    "LmCompressRequest",
    "LmCompressResult",
    "get_cv_compressor",
    "get_lm_compressor",
]


def get_cv_compressor() -> CvCompressor:
    from compressor.adapter.modelopt import ModelOptAdapter

    return ModelOptAdapter()


def get_lm_compressor() -> LmCompressor:
    from compressor.adapter.llmcompressor import LLMCompressorAdapter

    return LLMCompressorAdapter()
