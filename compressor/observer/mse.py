"""MSE grid search observer: based on TU semantic grid search for min/max.

Replace: ``MemorylessMSEObserver`` as base class, replace QDQ in search with TU semantic
(zqinfer alignment grid rounding). Write ``observer: tu_mse`` in recipe to enable.
TU semantic QDQ and grid search are from ``compressor.quantization.qdq``.

MX strategy (E8M0 block scale) does not use shrink search —— scale is determined by block amax
via OCP rules, here we only collect amax. MX semantic is guaranteed by quantizer forward.
"""

from __future__ import annotations

import logging

import torch

from compressor.hardware.formats import RoundingMode
from compressor.quantization.qdq import tu_grid_search_minmax

logger = logging.getLogger(__name__)

_REGISTERED = False


def _parse_rounding(name: str | None) -> RoundingMode:
    if not name:
        return RoundingMode.RZ
    return RoundingMode(name.lower())


def install_tu_observer() -> None:
    """Register tu_mse observer (idempotent). Requires llmcompressor importable."""
    global _REGISTERED
    if _REGISTERED:
        return

    from compressed_tensors.registry import get_from_registry  # noqa: F401
    from compressed_tensors.quantization.utils.mxfp_utils import (
        should_generate_mx_scales,
    )
    from llmcompressor.observers.base import Observer
    from llmcompressor.observers.mse import MemorylessMSEObserver

    try:
        get_from_registry(Observer, "tu_mse")
        _REGISTERED = True
        return
    except KeyError:
        pass

    @Observer.register("tu_mse")
    class TuMSEObserver(MemorylessMSEObserver):
        """TU semantic QDQ's eager grid search observer.

        Extra observer_kwargs:
            tu_rounding (str): TU rounding mode, default "rz"(TE semantic).
        """

        def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
            if should_generate_mx_scales(self.args):
                # MX: scale is determined by OCP rules from amax uniquely, no shrink search required
                self.min_vals = observed.amin(dim=-1)
                self.max_vals = observed.amax(dim=-1)
                return

            rounding = _parse_rounding(
                self.args.observer_kwargs.get("tu_rounding", "rz")
            )
            self.min_vals, self.max_vals = tu_grid_search_minmax(
                observed,
                num_bits=int(self.args.num_bits),
                qtype=str(getattr(self.args.type, "value", self.args.type)).lower(),
                symmetric=bool(self.args.symmetric),
                rounding=rounding,
                maxshrink=self.maxshrink,
                grid=int(self.grid),
                norm=self.norm,
            )

    _REGISTERED = True
    logger.debug("registered llmcompressor observer 'tu_mse'")
