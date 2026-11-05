"""TU (Tensor Unit) semantic reference kernels.

Usage::

    from compressor.kernels.reference import fake_quant, TuFormat, RoundingMode

    result = fake_quant(weight, TuFormat.MXFP4)
    result = fake_quant(act, TuFormat.FP8_E4M3, block_size=128,
                        rounding=RoundingMode.RZ)
    result = fake_quant(weight, TuFormat.INT4, per_channel=True, symmetric=True)

Format routing:
- MXFP4 / MXFP8 / NVFP4 → zqinfer semantic core (C++ kernel preferred, sim
  fallback). ``fake_quant`` defaults to sim (verified truth); opt into the
  C++ acceleration only after confirming bit-exact parity.
- FP8                   → pure-torch grid reference (per-tensor / per-block,
                          FP32 scale).
- INT8 / INT4           → pure-torch reference (symmetric / asymmetric,
                          zero-point, per-channel, INT32 accumulation).
"""

from __future__ import annotations

import torch

from compressor.hardware.formats import (
    SUPPORTED_BLOCK_SIZES,
    RoundingMode,
    TuFormat,
)
from compressor.kernels.reference.fp8 import fp8_fake_quant
from compressor.kernels.reference.int import int_fake_quant
from compressor.kernels.reference.result import FakeQuantResult
from compressor.kernels.reference.zq_bridge import (
    ZQINFER_BLOCK_SIZES,
    ZqInferUnavailable,
    default_nvfp4_global_scale,
    zq_cpp_ops_available,
    zq_fake_quant,
)

_ZQ_FORMATS = {TuFormat.MXFP4, TuFormat.MXFP8, TuFormat.NVFP4}
_FP8_FORMATS = {TuFormat.FP8_E4M3, TuFormat.FP8_E5M2}
_INT_FORMATS = {TuFormat.INT8, TuFormat.INT4}


def fake_quant(
    tensor: torch.Tensor,
    fmt: TuFormat,
    *,
    block_size: int | None = None,
    rounding: RoundingMode | None = None,
    global_scale: torch.Tensor | float | None = None,
    symmetric: bool = False,
    per_channel: bool = False,
    unsigned: bool = False,
    prefer_zq_kernel: bool = False,
) -> FakeQuantResult:
    """Fake-quantize ``tensor`` with TU hardware semantics and return the
    dequantised tensor plus kernel metadata.

    ``rounding=None`` picks the hardware default for that format: zqinfer
    kernels decide for themselves (TE uses RZ), FP8 / INT default to RN.

    ``prefer_zq_kernel`` is off by default — the verified sim kernel is the
    source of truth; the C++ extension is a pure acceleration path.
    """
    if not isinstance(fmt, TuFormat):
        raise TypeError(f"fmt must be TuFormat, got {type(fmt).__name__}")

    if block_size is not None and block_size not in SUPPORTED_BLOCK_SIZES:
        raise ValueError(
            f"block_size={block_size} is not supported by hardware; "
            f"supported: {sorted(SUPPORTED_BLOCK_SIZES)}"
        )

    if fmt in _ZQ_FORMATS:
        allowed = ZQINFER_BLOCK_SIZES[fmt]
        if block_size is not None and block_size not in allowed:
            raise NotImplementedError(
                f"{fmt.value} semantic core currently only supports "
                f"block_size={allowed}; {block_size} needs zqinfer-side work"
            )
        return zq_fake_quant(
            tensor,
            fmt,
            prefer_zq_kernel=prefer_zq_kernel,
            global_scale=global_scale,
        )

    if fmt in _FP8_FORMATS:
        return fp8_fake_quant(
            tensor,
            fmt,
            block_size=block_size,
            rounding=rounding or RoundingMode.RN,
        )

    if fmt in _INT_FORMATS:
        return int_fake_quant(
            tensor,
            fmt,
            symmetric=symmetric,
            per_channel=per_channel,
            unsigned=unsigned,
            rounding=rounding or RoundingMode.RN,
        )

    raise ValueError(f"unknown format {fmt}")


__all__ = [
    "FakeQuantResult",
    "RoundingMode",
    "SUPPORTED_BLOCK_SIZES",
    "TuFormat",
    "ZqInferUnavailable",
    "default_nvfp4_global_scale",
    "fake_quant",
    "zq_cpp_ops_available",
]
