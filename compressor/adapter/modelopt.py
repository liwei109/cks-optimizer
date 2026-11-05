"""ModelOpt adapter: CvCompressor interface implementation.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from compressor.adapter.contract import (
    CvPruneRequest,
    CvPruneResult,
    CvQuantizeRequest,
    CvQuantizeResult,
    CvSparsifyRequest,
    CvSparsifyResult,
    Technique,
)

try:
    import torch

    import modelopt.torch.quantization as mtq
except Exception:  # pragma: no cover
    torch = None
    mtq = None

logger = logging.getLogger(__name__)

# request.options export reserved keys: does not participate in quant_cfg override
_RESERVED_EXPORT_KEYS = ("use_dynamic_axes", "model_name", "simplify", "opset_version")


class ModelOptAdapter:
    """ModelOpt torch-mode adapter: CV 场景量化/稀疏/剪枝。"""

    def supported_techniques(self) -> frozenset[Technique]:
        # SPARSIFY: ModelOpt sparsity(mto) will be supported later(current sparsify is TODO)
        # PRUNE: ModelOpt provides modelopt.torch.pruning, will be supported later
        return frozenset({Technique.QUANTIZE})

    def is_available(self) -> bool:
        return mtq is not None and hasattr(mtq, "quantize")

    def install(self) -> None:
        """Install ModelOpt torch backend."""
        if not self.is_available():
            raise ImportError(
                "ModelOpt torch backend is not available. "
                "Please install modelopt in the current environment."
            )


    def _build_quant_cfg(self, quant_mode: str) -> dict[str, Any]:
        mode = (quant_mode or "int8").strip().lower()
        if mode == "fp8" and hasattr(mtq, "FP8_DEFAULT_CFG"):
            return copy.deepcopy(mtq.FP8_DEFAULT_CFG)
        if mode != "int8":
            logger.warning(
                "quant_mode=%r has no dedicated ModelOpt quant cfg, fallback to INT8_DEFAULT_CFG",
                mode,
            )
        return copy.deepcopy(mtq.INT8_DEFAULT_CFG)


    def quantize(self, request: CvQuantizeRequest) -> CvQuantizeResult:
        if not self.is_available():
            raise ImportError(
                "ModelOpt torch backend is not available. "
                "Please install modelopt in the current environment."
            )

        model = request.model
        model.eval()

        quant_cfg = self._build_quant_cfg(request.quant_mode)
        export_opts = {
            key: request.options[key]
            for key in _RESERVED_EXPORT_KEYS
            if key in request.options
        }
        cfg_overrides = {
            k: v for k, v in request.options.items() if k not in export_opts
        }
        if cfg_overrides:
            quant_cfg.update(cfg_overrides)

        def forward_loop(m):
            # Full-dataset calibration when the pipeline materialized the
            # whole calibration set; single-batch fallback otherwise.
            batches = request.calibration_batches
            if batches:
                try:
                    from tqdm import tqdm

                    batches = tqdm(batches, desc="calibrating", unit="batch")
                except ImportError:  # tqdm not installed — iterate silently
                    pass
                with torch.no_grad():
                    return [m(batch) for batch in batches]
            return m(request.sample_input)

        quantized_model = mtq.quantize(
            model,
            quant_cfg,
            forward_loop=forward_loop,
        )

        export_path = None
        if request.output_path is not None:
            from compressor.export.onnx_export import export_cv_deployment

            bundle = export_cv_deployment(
                quantized_model,
                request.sample_input,
                request.output_path,
                model_name=export_opts.get(
                    "model_name", type(model).__name__
                ),
                quant_mode=request.quant_mode,
                use_dynamic_axes=export_opts.get("use_dynamic_axes", True),
                simplify=export_opts.get("simplify", True),
                opset_version=export_opts.get("opset_version", 17),
            )
            export_path = bundle.onnx_path

        return CvQuantizeResult(
            quantized_model=quantized_model,
            export_path=export_path,
        )

    def sparsify(self, request: CvSparsifyRequest) -> CvSparsifyResult:
        """TODO: Delegate ModelOpt sparsity(mto) to implement 2:4 sparsity."""
        raise NotImplementedError("TODO: Integrate ModelOpt sparsity interface")

    def prune(self, request: CvPruneRequest) -> CvPruneResult:
        """TODO: Delegate modelopt.torch.pruning to implement pruning."""
        raise NotImplementedError("TODO: Integrate ModelOpt pruning interface")
