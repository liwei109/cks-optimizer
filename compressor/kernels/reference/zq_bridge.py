"""ZQInfer semantic bridge: 
Semantics come from zqinfer's verified pure-torch ``*_sim`` kernels. The
``*_zq`` C++ kernels default to **off** — the currently installed build
predates the sim kernels and is known to diverge on E8M0 scale rules and
subnormal encoding. Confirm bit-exact parity before opting in.
"""

from __future__ import annotations

from typing import Any

import torch

from compressor.hardware.formats import (
    FP4_MAX,
    FP8_E4M3_MAX,
    RoundingMode,
    TuFormat,
)
from compressor.kernels.reference.result import FakeQuantResult

# Block sizes currently wired up in zqinfer's MX-format kernels. This is a
# software-availability fact, not a hardware one — hence it lives here
# (beside the bridge) rather than in the hardware format model.
ZQINFER_BLOCK_SIZES = {
    TuFormat.MXFP4: (32,),
    TuFormat.MXFP8: (32,),
    TuFormat.NVFP4: (16,),
}


class ZqInferUnavailable(ImportError):
    """Raised when the zqinfer semantic core can't be imported or the
    compiled ``_ops`` extension isn't available."""


def import_zq_quantization():
    """Import ``zqinfer.quantization``, failing with an actionable prompt
    when it is not installed.

    zqinfer is an optional local source package (not on PyPI). It is only
    detected here, at import time — install it (editable) from the zqinfer
    source tree if MX/NVFP quantization is needed.
    """
    try:
        from zqinfer import quantization as zq
    except ImportError as exc:
        raise ZqInferUnavailable(
            "zqinfer is not installed; MX/NVFP quantization requires it. "
            "Install from source: pip install -e <path-to-zqinfer>"
        ) from exc
    return zq


def zq_cpp_ops_available() -> bool:
    """True when the compiled ``zqinfer._ops`` C++ extension is loadable."""
    try:
        import zqinfer  # noqa: F401

        import zqinfer._ops  # noqa: F401
        return True
    except Exception:
        return False


def _select_quant_entry(zq: Any, fmt: TuFormat, prefer_zq: bool):
    suffix = "zq" if prefer_zq else "sim"
    name = f"{fmt.value}_quantize_{suffix}"
    entry = getattr(zq, name, None)
    if entry is None:
        raise ZqInferUnavailable(f"zqinfer is missing kernel {name}")
    return entry, f"zqinfer-{suffix}"


def _select_dequant_entry(zq: Any, fmt: TuFormat, prefer_zq: bool):
    suffix = "zq" if prefer_zq else "sim"
    name = f"{fmt.value}_dequantize_{suffix}"
    entry = getattr(zq, name, None)
    if entry is None:
        raise ZqInferUnavailable(f"zqinfer is missing kernel {name}")
    return entry


def default_nvfp4_global_scale(tensor: torch.Tensor) -> torch.Tensor:
    """Per-tensor global scale for NVFP4's two-level scale hierarchy
    (NVIDIA TRT convention).

    ``global_scale = FP8_E4M3_MAX * FP4_MAX / amax`` keeps the block-level
    scales inside an FP8 representable range with minimal saturation risk.
    Callers with a custom global-scale contract should pass theirs in
    explicitly.
    """
    amax = tensor.abs().amax().float().clamp(min=1e-12)
    return (FP8_E4M3_MAX * FP4_MAX) / amax


def zq_fake_quant(
    tensor: torch.Tensor,
    fmt: TuFormat,
    *,
    prefer_zq_kernel: bool = False,
    global_scale: torch.Tensor | float | None = None,
) -> FakeQuantResult:
    """Route MXFP4 / MXFP8 / NVFP4 through the zqinfer semantic core
    (sim kernel wins unless the caller opts into the C++ extension and it's
    actually compiled).
    """
    zq = import_zq_quantization()
    prefer_zq = prefer_zq_kernel and zq_cpp_ops_available()
    quant_entry, backend = _select_quant_entry(zq, fmt, prefer_zq)
    dequant_entry = _select_dequant_entry(zq, fmt, prefer_zq)

    original_device = tensor.device
    source = tensor.detach().to(device="cpu", dtype=torch.float32)

    if fmt is TuFormat.MXFP4:
        packed, scale = quant_entry(source)
        dequant = dequant_entry(packed, scale)
        block_size = 32
        zero_point = None
    elif fmt is TuFormat.MXFP8:
        # The _zq signature has a default sf_vec_size; _sim requires it
        # explicitly — pass 32 in both to keep the call uniform.
        packed, scale = quant_entry(source, 32)
        dequant = dequant_entry(packed, scale, 32)
        block_size = 32
        zero_point = None
    elif fmt is TuFormat.NVFP4:
        if global_scale is None:
            global_scale = default_nvfp4_global_scale(source)
        gs = (
            global_scale.detach().to(device="cpu", dtype=torch.float32)
            if isinstance(global_scale, torch.Tensor)
            else torch.tensor(float(global_scale), dtype=torch.float32)
        )
        packed, scale = quant_entry(source, gs)
        dequant = dequant_entry(packed, scale, gs)
        block_size = 16
        zero_point = None
    else:
        raise ValueError(f"{fmt} is not bridged by zqinfer")

    result = FakeQuantResult(
        dequant=dequant.to(original_device),
        packed=packed,
        scale=scale,
        fmt=fmt,
        block_size=block_size,
        rounding=RoundingMode.RZ,  # TE matrix pipeline uses RZ truncation semantics
        backend=backend,
        meta={"source": source, "global_scale": global_scale},
    )
    return result