from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import torch

from cks_optimizer.common import export_torch_to_onnx
from cks_optimizer.cnn.config import CNNConfig

try:
    import modelopt.torch.quantization as mtq
except Exception:  # pragma: no cover
    mtq = None


class ModelOptTorchAdapter:
    """ModelOpt torch-mode adapter for CNN / CV models."""

    def __init__(self, config: Optional[CNNConfig] = None):
        self.config = config or CNNConfig()

    def _validate_backend(self) -> None:
        if mtq is None or not hasattr(mtq, "quantize"):
            raise ImportError(
                "ModelOpt torch backend is not available. "
                "Please install modelopt in the current environment."
            )

    def _build_quant_cfg(self) -> dict[str, Any]:
        if mtq is None:
            return {
                "quant_cfg": [
                    {"quantizer_name": "*", "enable": False},
                    {
                        "quantizer_name": "*weight_quantizer",
                        "cfg": {"num_bits": 8, "axis": 0},
                    },
                    {
                        "quantizer_name": "*input_quantizer",
                        "cfg": {"num_bits": 8, "axis": None},
                    },
                ],
                "algorithm": "max",
            }

        if self.config.quant_mode == "int8":
            return copy.deepcopy(mtq.INT8_DEFAULT_CFG)

        return copy.deepcopy(mtq.INT8_DEFAULT_CFG)

    def optimize(
        self,
        model: torch.nn.Module,
        sample_input: Any,
        output_path: Optional[str] = None,
        calibration_loader: Any = None,
        **kwargs: Any,
    ) -> Path:
        self._validate_backend()

        model.eval()
        quant_cfg = self._build_quant_cfg()
        if kwargs:
            quant_cfg.update(kwargs)

        if calibration_loader is not None:
            # The dataset consumption point: data flows through the model here.
            # Each forward triggers TensorQuantizer hooks that collect absmax
            # statistics over real activation distributions. Batches are stacked
            # tensors (classification) or List[Tensor] (detection) — both are
            # passed to the model as-is.
            def forward_loop(m):
                for batch in calibration_loader:
                    m(batch)
        else:
            def forward_loop(m):
                return m(sample_input)

        quantized_model = mtq.quantize(
            model,
            quant_cfg,
            forward_loop=forward_loop,
        )
        final_path = (
            Path(output_path)
            if output_path
            else Path(self.config.output_path)
        )
        final_path.parent.mkdir(parents=True, exist_ok=True)

        export_torch_to_onnx(
            quantized_model,
            final_path,
            sample_input,
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            } if self.config.use_dynamic_axes else None,
            input_names=["input"],
            output_names=["output"],
        )
        return final_path
