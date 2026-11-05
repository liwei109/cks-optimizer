"""GPTQ quantization algorithm for LM: TU semantic block update backend injection.
"""

from __future__ import annotations

import logging
from copy import copy

logger = logging.getLogger(__name__)

_REGISTERED = False


def install_tu_gptq_backend() -> None:
    """Register TU GPTQ block update backend."""
    global _REGISTERED
    if _REGISTERED:
        return

    from compressed_tensors.quantization import QuantizationStrategy
    from compressed_tensors.utils.impl_backend import ImplBackend
    from llmcompressor.modifiers.gptq.helpers import column_scale_window

    from compressor.hardware.formats import RoundingMode
    from compressor.quantization.qdq import tu_qdq_with_args

    def _tu_gptq_req(
        W1, Hinv1, Q1, Err1, losses1, *,
        scale, zero_point, global_scale, g_idx, quant_args, i1,
    ) -> bool:
        return str(getattr(quant_args, "observer", "") or "").startswith("tu_")

    @ImplBackend.register("gptq_block_update", _tu_gptq_req, -1)
    def tu_gptq_block_update(
        W1, Hinv1, Q1, Err1, losses1, *,
        scale, zero_point, global_scale, g_idx, quant_args, i1,
    ) -> None:
        """Same semantics as llmcompressor eager implementation, QDQ to TU semantic."""
        if W1.dim() != 3:
            raise ValueError("The TU GPTQ block backend requires a 3D weight block")

        rounding_name = (quant_args.observer_kwargs or {}).get("tu_rounding", "rz")
        rounding = RoundingMode(str(rounding_name).lower())

        block_width = W1.shape[-1]
        altered_qargs = copy(quant_args)
        altered_qargs.strategy = QuantizationStrategy.CHANNEL
        eff, zp = column_scale_window(
            scale,
            zero_point,
            global_scale,
            g_idx,
            quant_args,
            num_rows=W1.shape[-2],
            i1=i1,
            i2=i1 + block_width,
        )
        for i in range(block_width):
            w = W1[:, :, i]
            q = tu_qdq_with_args(
                w.transpose(0, 1),
                eff[:, :, i].transpose(0, 1),
                None if zp is None else zp[:, :, i].transpose(0, 1),
                altered_qargs,
                rounding=rounding,
            ).transpose(0, 1)

            diagonal = Hinv1[:, i, i]
            error = (w - q) / diagonal[:, None]
            Q1[:, :, i] = q
            Err1[:, :, i] = error
            losses1[:, :, i] = error.square()
            W1[:, :, i:] -= error.unsqueeze(-1) * Hinv1[:, i, i:].unsqueeze(1)

    _REGISTERED = True
    logger.debug("registered TU backend for llmcompressor gptq_block_update")
