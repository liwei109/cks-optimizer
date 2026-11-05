"""Kernel result DTO: ``FakeQuantResult``.

Format facts (``TuFormat`` / ``RoundingMode``) live in
``compressor.hardware.formats``; this module only carries the per-call output
of a fake-quant kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from compressor.hardware.formats import RoundingMode, TuFormat

__all__ = ["FakeQuantResult"]


@dataclass
class FakeQuantResult:
    """Full output of ``fake_quant``.

    ``dequant`` is the dequantised float tensor used by calibration / eval
    paths; ``packed`` / ``scale`` / ``zero_point`` are kernel-layout
    metadata ready for golden-vector export and deployment-format alignment.
    """

    dequant: torch.Tensor
    packed: torch.Tensor
    scale: torch.Tensor
    fmt: TuFormat
    block_size: int | None
    rounding: RoundingMode
    backend: str  # "zqinfer-zq" | "zqinfer-sim" | "torch-reference"
    zero_point: torch.Tensor | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def reconstruction_error(self) -> torch.Tensor:
        """Absolute vs. the original source tensor (reused by
        per-layer sensitivity analysis)."""
        return (self.dequant - self.meta["source"]).abs()