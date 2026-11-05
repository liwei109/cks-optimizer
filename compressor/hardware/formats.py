"""Hardware data-format model: formats, rounding modes, and numeric bounds
for the CSK tensor unit (TU).

This module is the single source of truth for *hardware* format facts — the
formats the chip's TE/TU can represent, the block sizes it accepts, and the
rounding modes available to each unit. It carries no kernel logic; the
numerical semantics live in ``compressor.kernels.reference``.

Scheme-level acceptance rules (which weight/activation schemes the chip
admits) live in ``compressor.hardware.profile``, in the compressed-tensors
``(num_bits, type)`` vocabulary. Its block sizes are subsets of
``SUPPORTED_BLOCK_SIZES`` below.
"""

from __future__ import annotations

import enum

__all__ = [
    "FP4_MAX",
    "FP8_E4M3_MAX",
    "FP8_E5M2_MAX",
    "SUPPORTED_BLOCK_SIZES",
    "RoundingMode",
    "TuFormat",
]


class RoundingMode(enum.Enum):
    """Hardware rounding modes. TE matrix compute only supports RZ; the
    ALU supports all five."""

    RZ = "rz"  # round toward zero (truncation)
    RN = "rn"  # round to nearest, ties to even
    RU = "ru"  # round toward +inf
    RD = "rd"  # round toward -inf
    RS = "rs"  # round to nearest, ties away from zero


class TuFormat(enum.Enum):
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    MXFP4 = "mxfp4"    # E8M0 scale, block 32 (OCP MX)
    MXFP8 = "mxfp8"    # E8M0 scale, block 32 (OCP MX)
    NVFP4 = "nvfp4"    # two-level scale: block 16 + FP32 global scale
    INT8 = "int8"
    INT4 = "int4"      # signed / unsigned decided by symmetric / asymmetric


# Block sizes the chip actually supports.
SUPPORTED_BLOCK_SIZES = frozenset({16, 32, 64, 128, 256})

# FP8 / FP4 max representable values.
FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0
FP4_MAX = 6.0