"""Pure-torch FP8 fake-quant reference implementation.

Scale semantics: per-tensor or per-block (default 128), scale itself stays
FP32 (not quantised). Rounding uses an explicit finite-value grid plus
interval selection — the five modes (RZ/RN/RU/RD/RS) share one index
path, and correctness is covered by grid-enumeration tests.

E4M3 byte layout: 0x00..0x7E are monotonic non-negative finite values
(``fn`` — no inf); 0x80..0xFF mirror the negatives. E5M2 has the same
shape with 0x00..0x7B finite. ``sorted grid`` index == byte value, so
RN's tie-even ("even mantissa LSB") rule maps directly to even index.
"""

from __future__ import annotations

import torch

from compressor.hardware.formats import (
    FP8_E4M3_MAX,
    FP8_E5M2_MAX,
    RoundingMode,
    TuFormat,
)
from compressor.kernels.reference.result import FakeQuantResult

_E4M3_MAX_BYTE = 0x7E  # e4m3fn: 0x7F is NaN
_E5M2_MAX_BYTE = 0x7B  # e5m2: 0x7C/0x7D are ±inf, 0x7E/0x7F are NaN


def build_fp8_grid(e4m3: bool = True) -> torch.Tensor:
    """Build the ordered finite FP8 grid: negatives asc → ±0 → positives asc.

    ``sorted`` index == byte encoding (see ``round_to_grid``), which is why
    RN's tie-even ("even byte") rule reduces to checking index parity.
    """
    dtype = torch.float8_e4m3fn if e4m3 else torch.float8_e5m2
    max_byte = _E4M3_MAX_BYTE if e4m3 else _E5M2_MAX_BYTE
    positive = (
        torch.arange(0, max_byte + 1, dtype=torch.uint8)
        .view(dtype)
        .to(torch.float32)
    )
    return torch.cat([-positive.flip(0), positive])


# Lazily built global caches for each grid variant.
_GRID_E4M3: torch.Tensor | None = None
_GRID_E5M2: torch.Tensor | None = None


def _get_grid(e4m3: bool) -> torch.Tensor:
    global _GRID_E4M3, _GRID_E5M2
    if e4m3 and _GRID_E4M3 is None:
        _GRID_E4M3 = build_fp8_grid(e4m3=True)
    if not e4m3 and _GRID_E5M2 is None:
        _GRID_E5M2 = build_fp8_grid(e4m3=False)
    return _GRID_E4M3 if e4m3 else _GRID_E5M2


def round_to_grid(
    values: torch.Tensor, grid: torch.Tensor, mode: RoundingMode
) -> torch.Tensor:
    """Map pre-scaled ``values`` onto the ordered finite-value ``grid``.

    Grid layout: negatives asc (k entries) → ±0 → positives asc (k
    entries); n = 2k. Sorted-index → byte encoding:
        positive zone  byte = idx - k
        negative zone   byte = (n - 1) - idx
    so RN's tie-even rule is just "pick even byte".
    """
    n = grid.numel()
    k = n // 2
    values = values.clamp(grid[0].item(), grid[-1].item())
    idx_hi = torch.searchsorted(grid, values.contiguous()).clamp(1, n - 1)
    idx_lo = idx_hi - 1
    v_hi = grid[idx_hi]
    v_lo = grid[idx_lo]

    if mode is RoundingMode.RD:
        return v_lo
    if mode is RoundingMode.RU:
        return v_hi
    if mode is RoundingMode.RZ:
        # Truncate toward zero: v >= 0 picks lo, v < 0 picks hi.
        return torch.where(values >= 0, v_lo, v_hi)

    dist_lo = (values - v_lo).abs()
    dist_hi = (v_hi - values).abs()
    nearer_lo = dist_lo < dist_hi
    tie = dist_lo == dist_hi

    if mode is RoundingMode.RS:
        # Ties away from zero: pick whichever side has the larger magnitude.
        pick_lo = tie & (v_lo.abs() > v_hi.abs())
        return torch.where(nearer_lo | pick_lo, v_lo, v_hi)

    # RN: ties to even byte (byte bit 0 == mantissa LSB).
    byte_lo = torch.where(idx_lo >= k, idx_lo - k, (k - 1) - idx_lo)
    even_lo = (byte_lo % 2) == 0
    pick_lo = nearer_lo | (tie & even_lo)
    return torch.where(pick_lo, v_lo, v_hi)


def fp8_fake_quant(
    tensor: torch.Tensor,
    fmt: TuFormat,
    *,
    block_size: int | None = None,
    rounding: RoundingMode = RoundingMode.RN,
) -> FakeQuantResult:
    """FP8 per-tensor / per-block fake-quant; scale stays FP32."""
    e4m3 = fmt is TuFormat.FP8_E4M3
    grid = _get_grid(e4m3)
    fmax = FP8_E4M3_MAX if e4m3 else FP8_E5M2_MAX

    source = tensor.detach().to(dtype=torch.float32)

    if block_size is None:
        scale = source.abs().amax().clamp(min=1e-12) / fmax
        scale = scale.reshape(1)
        scaled = source / scale
        quantized = round_to_grid(scaled, grid, rounding)
        packed = quantized.to(
            torch.float8_e4m3fn if e4m3 else torch.float8_e5m2
        )
        dequant = packed.to(torch.float32) * scale
        packed_meta = packed.view(torch.uint8).reshape(source.shape)
        dequant = dequant.reshape(source.shape)
    else:
        last = source.shape[-1]
        if last % block_size != 0:
            raise ValueError(
                f"last dim {last} is not divisible by block_size={block_size}"
            )
        blocked = source.reshape(-1, block_size)
        amax = blocked.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        # Keep ``scale_blocked`` with ``(num_blocks, 1)`` layout through the
        # broadcast path — reshape *after* computation, otherwise 3-D inputs
        # would silently mis-broadcast (old implementation bug we explicitly
        # avoid here).
        scale_blocked = amax / fmax
        scaled = blocked / scale_blocked
        quantized = round_to_grid(scaled, grid, rounding)
        packed = quantized.to(
            torch.float8_e4m3fn if e4m3 else torch.float8_e5m2
        )
        dequant = (packed.to(torch.float32) * scale_blocked).reshape(source.shape)
        packed_meta = packed.view(torch.uint8).reshape(source.shape)
        scale = scale_blocked.reshape(*source.shape[:-1], last // block_size)

    return FakeQuantResult(
        dequant=dequant,
        packed=packed_meta,
        scale=scale.reshape(scale.shape),
        fmt=fmt,
        block_size=block_size,
        rounding=rounding,
        backend="torch-reference",
        meta={"source": source},
    )
