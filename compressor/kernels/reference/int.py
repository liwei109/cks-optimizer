"""Pure-torch INT8 / INT4 fake-quant reference implementation.

Hardware semantics (confirmed with the chip team):
- TE uses an unsigned multiplier with the sign bit handled separately —
  asymmetric quant goes through the signed path with zero-point; symmetric
  quant has no unsigned step and zero_point = 0.
- Scale granularity: activation per-tensor, weight per-channel; scale
  dtype stays FP32.
- Accumulator is fixed INT32.

Note: sign handling and unsigned storage layout are kernel-side details.
This reference gives the standard mathematical semantics
(q = round(x / s) + zp) and lets the kernel align storage layout. If a
sign-bit offset ever shows up in the kernel format, fix it here — that's
the single source of truth.
"""

from __future__ import annotations

import torch

from compressor.hardware.formats import RoundingMode, TuFormat
from compressor.kernels.reference.result import FakeQuantResult

_INT_LIMITS = {
    8: (-128, 127),
    4: (-8, 7),
}
_UINT_LIMITS = {
    8: (0, 255),
    4: (0, 15),
}


def _round_fn(rounding: RoundingMode):
    return {
        RoundingMode.RN: torch.round,
        RoundingMode.RZ: torch.trunc,
        RoundingMode.RU: torch.ceil,
        RoundingMode.RD: torch.floor,
    }.get(rounding)


def int_fake_quant(
    tensor: torch.Tensor,
    fmt: TuFormat,
    *,
    symmetric: bool = False,
    per_channel: bool = False,
    unsigned: bool = False,
    rounding: RoundingMode = RoundingMode.RN,
) -> FakeQuantResult:
    """INT8 / INT4 fake-quant.

    ``per_channel=True`` computes per-channel scales along the last dim
    (canonical weight ``out_channels``); otherwise per-tensor (activation
    path). ``unsigned=True`` forces the unsigned UINT range and implies
    asymmetric semantics.
    """
    bits = 8 if fmt is TuFormat.INT8 else 4
    if unsigned:
        symmetric = False

    lo, hi = (
        _INT_LIMITS[bits] if symmetric else _UINT_LIMITS[bits]
    )

    rnd = _round_fn(rounding)
    if rnd is None:
        raise NotImplementedError(f"integer quant doesn't support {rounding}")

    source = tensor.detach().to(dtype=torch.float32)
    if per_channel:
        xmin = source.amin(dim=-1, keepdim=True)
        xmax = source.amax(dim=-1, keepdim=True)
    else:
        xmin = source.amin()
        xmax = source.amax()

    if symmetric:
        amax = torch.maximum(xmin.abs(), xmax.abs()).clamp(min=1e-12)
        scale = amax / hi
        zero_point = torch.zeros_like(scale)
    else:
        # Asymmetric: map the observed range into [lo, hi] with a zero-point.
        xmin = torch.minimum(xmin, torch.zeros_like(xmin))
        xmax = torch.maximum(xmax, torch.zeros_like(xmax))
        scale = ((xmax - xmin) / (hi - lo)).clamp(min=1e-12)
        zero_point = torch.round(-xmin / scale).clamp(lo, hi)

    scaled = source / scale
    q = rnd(scaled) + zero_point
    q = torch.clamp(q, lo, hi)
    dequant = (q - zero_point) * scale

    packed = q.to(torch.int32)
    if not symmetric:
        packed = q.to(torch.int32) - lo  # unsigned layout starts at 0 for readability

    return FakeQuantResult(
        dequant=dequant,
        packed=packed.to(torch.int8 if symmetric else torch.uint8),
        scale=scale,
        fmt=fmt,
        block_size=None,
        rounding=rounding,
        backend="torch-reference",
        zero_point=zero_point,
        meta={
            "source": source,
            "symmetric": symmetric,
            "per_channel": per_channel,
            "unsigned": unsigned,
            "accumulator_dtype": "int32",
            "bits": bits,
        },
    )
