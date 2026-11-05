"""TU semantics QDQ (quantize-dequantize) implementation and
grid search based on QDQ errors, used for independent testing and GPTQ backend calls.

The scale derivation convention is kept consistent with compressed_tensors' `calculate_qparams`,
so that the range selected by the observer is optimal for the qparams that are finally saved:

"""

from __future__ import annotations

import torch

from compressor.kernels.reference.fp8 import build_fp8_grid, round_to_grid
from compressor.hardware.formats import RoundingMode


_FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}
_FP4_MAX = 6.0

_GRID_CACHE: dict[str, torch.Tensor] = {}


def build_fp4_grid() -> torch.Tensor:

    def _decode(byte: torch.Tensor) -> torch.Tensor:
        s = (byte & 0x08) != 0
        e = (byte >> 1) & 0x03
        m = byte & 0x01
        val = torch.where(
            e == 0,
            0.5 * m,  # Subnormal: 0, 0.5
            (1.0 + 0.5 * m) * 2.0 ** (e.to(torch.int32) - 1),
        )
        return torch.where(s, -val, val)

    pos = _decode(torch.arange(0, 8, dtype=torch.uint8))
    return torch.cat([-pos.flip(0), pos])


def _get_grid(num_bits: int, qtype: str) -> torch.Tensor:
    key = f"{num_bits}:{qtype}"
    if key not in _GRID_CACHE:
        if num_bits == 4 and qtype == "float":
            _GRID_CACHE[key] = build_fp4_grid()
        elif num_bits == 8 and qtype == "float":
            _GRID_CACHE[key] = build_fp8_grid(e4m3=True)
        else:
            raise ValueError(f"Invalid grid format: num_bits={num_bits}, type={qtype}")
    return _GRID_CACHE[key]


def _float_max(num_bits: int, qtype: str) -> float:
    if num_bits == 4:
        return _FP4_MAX
    if num_bits == 8 and qtype == "float":
        return _FP8_MAX["e4m3"]
    raise ValueError(f"Invalid float range: num_bits={num_bits}, type={qtype}")


def int_limits(num_bits: int, symmetric: bool) -> tuple[int, int]:
    """Symmetric → signed range; asymmetric → unsigned range. See kernels' int reference."""
    if symmetric:
        return -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1
    return 0, 2**num_bits - 1


def scale_from_range(
    cmin: torch.Tensor,
    cmax: torch.Tensor,
    num_bits: int,
    qtype: str,
    symmetric: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Calculate (scale, zero_point), mirroring compressed_tensors convention."""
    if qtype == "float":
        amax = torch.maximum(cmin.abs(), cmax.abs()).clamp(min=1e-12)
        return amax / _float_max(num_bits, qtype), None
    lo, hi = int_limits(num_bits, symmetric)
    if symmetric:
        amax = torch.maximum(cmin.abs(), cmax.abs()).clamp(min=1e-12)
        return amax / hi, None
    cmin = torch.minimum(cmin, torch.zeros_like(cmin))
    cmax = torch.maximum(cmax, torch.zeros_like(cmax))
    scale = ((cmax - cmin) / (hi - lo)).clamp(min=1e-12)
    zp = torch.round(-cmin / scale).clamp(float(lo), float(hi))
    return scale, zp


def tu_qdq(
    values: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    num_bits: int,
    qtype: str,
    symmetric: bool = True,
    rounding: RoundingMode = RoundingMode.RZ,
) -> torch.Tensor:
    """TU semantic QDQ."""
    if qtype == "float":
        grid = _get_grid(num_bits, qtype)
        quantized = round_to_grid(values / scale, grid, rounding)
        return quantized * scale

    lo, hi = int_limits(num_bits, symmetric)
    rounded = {
        RoundingMode.RN: torch.round,
        RoundingMode.RZ: torch.trunc,
        RoundingMode.RU: torch.ceil,
        RoundingMode.RD: torch.floor,
    }[rounding](values / scale)
    q = rounded if zero_point is None else rounded + zero_point
    q = torch.clamp(q, float(lo), float(hi))
    q = q if zero_point is None else q - zero_point
    return q * scale


def tu_qdq_with_args(
    values: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    quant_args,
    rounding: RoundingMode = RoundingMode.RZ,
) -> torch.Tensor:
    """duck-typed QuantizationArgs version."""
    return tu_qdq(
        values,
        scale,
        zero_point,
        num_bits=int(quant_args.num_bits),
        qtype=str(getattr(quant_args.type, "value", quant_args.type)).lower(),
        symmetric=bool(quant_args.symmetric),
        rounding=rounding,
    )


def tu_grid_search_minmax(
    observed: torch.Tensor,
    num_bits: int,
    qtype: str,
    symmetric: bool = True,
    rounding: RoundingMode = RoundingMode.RZ,
    maxshrink: float = 0.20,
    grid: int = 100,
    norm: float = 2.4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TU grid search for minmax range.

    For each observation group, within the shrinkage candidates [amin, amax],
    calculate the error using TU semantics QDQ and select the optimal range.
    This is a TU semantics alternative to the Triton/eager search of the llmcompressor MSE observer,
    with behavior aligned to its eager implementation (maxshrink/grid/norm parameter semantics are consistent).

    :param observed: Flattened observations, shape ``(num_observations, group_size)``
    :return: ``(min_vals, max_vals)``, shape ``(num_observations,)``
    """
    if observed.ndim == 1:
        observed = observed.unsqueeze(0)

    min_vals = observed.amin(dim=-1)
    max_vals = observed.amax(dim=-1)

    best_min = min_vals.clone()
    best_max = max_vals.clone()
    best_error = torch.full_like(min_vals, torch.inf)

    for i in range(int(grid)):
        shrink = 1.0 - i / float(grid)
        if shrink < 1.0 - maxshrink:
            break
        cand_min = min_vals * shrink
        cand_max = max_vals * shrink

        scale, zp = scale_from_range(cand_min, cand_max, num_bits, qtype, symmetric)
        quantized = tu_qdq(
            observed, scale.unsqueeze(-1), None if zp is None else zp.unsqueeze(-1),
            num_bits, qtype, symmetric, rounding,
        )
        error = (quantized - observed).abs().pow(norm).sum(dim=-1)

        improved = error < best_error
        best_error = torch.where(improved, error, best_error)
        best_min = torch.where(improved, cand_min, best_min)
        best_max = torch.where(improved, cand_max, best_max)

    return best_min, best_max
