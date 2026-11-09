"""ModelOpt-backed CNN quantization and sparsification."""

from __future__ import annotations

import copy
import logging
import warnings
from collections.abc import Sequence
from typing import Any

import torch

from cks_optimizer.cnn.config import (
    CNNConfig,
    OptimizeConfig,
    build_optimize_config,
)

logger = logging.getLogger(__name__)

__all__ = ["CNNOptimizer", "optimize_cnn_model"]

try:
    import modelopt.torch.quantization as mtq
except ImportError:  # pragma: no cover - depends on optional backend
    mtq = None

try:
    from modelopt.torch.sparsity.weight_sparsity.sparsification import (
        sparsify as _sparsify,
    )
except ImportError:  # pragma: no cover - depends on optional backend
    _sparsify = None


def _as_optimize_config(config: Any) -> OptimizeConfig:
    """Normalize optimizer config through the central config builder."""
    return build_optimize_config(config)


def _build_forward_loop(
    sample_input: Any,
    calib_data: Sequence[torch.Tensor] | None,
):
    """Build a ModelOpt forward loop over calibration batches or one sample."""

    def forward_loop(model: torch.nn.Module) -> None:
        if calib_data:
            for tensor in calib_data:
                model(tensor)
        else:
            model(sample_input)

    return forward_loop


class CNNOptimizer:
    """CNN model optimizer (quantization + sparsity) backed by ModelOpt.

    The configured sections select algorithms: ``sparsity`` runs
    :meth:`sparsify` and ``quant`` runs
    :meth:`quantize` (in that order). See :attr:`OptimizeConfig.effective_algorithms`.
    """

    def __init__(self, config: Any | None = None):
        self.config = _as_optimize_config(config)

    def _validate_backend(self) -> None:
        if mtq is None or not hasattr(mtq, "quantize"):
            raise ImportError(
                "ModelOpt quantization backend is not available. Please install "
                "the vendored backend with 'pip install -e thirdparty/Model-Optimizer'."
            )

    def quantize(
        self,
        model: torch.nn.Module,
        sample_input: Any,
        calib_data: Sequence[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.nn.Module:
        """Quantize the model (ModelOpt torch mode) and return it.

        Args:
            model: The PyTorch model to quantize.
            sample_input: A sample input tensor (used when no calibration
                tensors are provided).
            calib_data: Optional list of preprocessed input tensors
                from :func:`data_process` used for calibration.
            **kwargs: Extra configuration forwarded to ModelOpt's
                ``quantize``.
        """
        self._validate_backend()

        model.eval()
        quant_cfg = copy.deepcopy(self.config._quant_cfg)
        quant_cfg.update(kwargs)

        forward_loop = _build_forward_loop(sample_input, calib_data)
        return mtq.quantize(
            model,
            quant_cfg,
            forward_loop=forward_loop,
        )

    def sparsify(
        self,
        model: torch.nn.Module,
        sample_input: Any,
        mode: str | None = None,
        sparsity: float | None = None,
        calib_data: Sequence[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.nn.Module:
        """Sparsify the model (ModelOpt sparsity) and return it.

        Args:
            model: The PyTorch model to sparsify.
            sample_input: A sample input tensor used for the forward loop.
            mode: Sparsity mode, e.g. ``"sparse_magnitude"`` (unstructured,
                no calibration data needed) or ``"sparsegpt"`` (calibration
                based). Defaults to ``config.sparsity.mode``.
            sparsity: Target sparsity ratio in ``[0, 1)``. Defaults to
                ``config.sparsity.sparsity``. Note that ModelOpt's
                magnitude sparsifier applies a fixed 2:4 (50%) pattern, so
                this value is informational and a warning is emitted when it
                differs from 0.5.
            calib_data: Optional calibration tensors for
                calibration-based modes.
            **kwargs: Extra configuration forwarded to ModelOpt's
                ``sparsify``.
        """
        if _sparsify is None:
            raise ImportError(
                "ModelOpt sparsity backend is not available. "
                "Please install the vendored backend with "
                "'pip install -e thirdparty/Model-Optimizer'."
            )

        settings = self.config.sparsity
        if settings is None:
            raise ValueError("Sparsity settings are missing from the config.")
        mode = mode or settings.mode
        sparsity = settings.sparsity if sparsity is None else sparsity

        if abs(sparsity - 0.5) > 1e-6:
            warnings.warn(
                f"ModelOpt magnitude sparsification applies a fixed 2:4 "
                f"(50%) pattern; the requested target sparsity of "
                f"{sparsity} is not directly supported and will be ignored.",
                stacklevel=2,
            )

        model.eval()

        config: dict[str, Any] = {}
        if mode == "sparsegpt":
            config["forward_loop"] = _build_forward_loop(
                sample_input,
                calib_data,
            )
        config.update(settings.extra)
        config.update(kwargs)

        return _sparsify(model, mode, config)

    def optimize(
        self,
        model: torch.nn.Module,
        sample_input: Any,
        calib_data: Sequence[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.nn.Module:
        """Run the configured optimization algorithms and return the model.

        ONNX export is handled separately by :func:`export_onnx`.

        A configured ``sparsity`` section runs :meth:`sparsify`; a configured
        ``quant`` section runs :meth:`quantize` (sparsity first).

        Args:
            model: The PyTorch model to optimize.
            sample_input: A sample input tensor used for calibration and
                export.
            calib_data: Optional calibration tensors from
                :func:`data_process`.
            **kwargs: Per-algorithm options keyed by algorithm name, e.g.
                ``sparsify={"mode": "sparsegpt", "sparsity": 0.5}``.

        Returns:
            The optimized ``torch.nn.Module``.
        """
        if not self.config.effective_algorithms:
            raise ValueError(
                "No optimization algorithm configured. Add a 'quant' and/or "
                "'sparsity' section to the config."
            )

        if self.config.sparsity is not None:
            model = self.sparsify(
                model,
                sample_input,
                calib_data=calib_data,
                **kwargs.get("sparsify", {}),
            )

        if self.config.quant is not None:
            model = self.quantize(
                model,
                sample_input,
                calib_data=calib_data,
                **kwargs.get("quantize", {}),
            )

        return model


def optimize_cnn_model(
    model: Any,
    sample_input: Any,
    config: Any | None = None,
    calib_data: Sequence[torch.Tensor] | None = None,
    **kwargs: Any,
) -> torch.nn.Module:
    """Public API for CNN optimization using ModelOpt torch mode.

    ONNX export is handled separately by :func:`export_onnx`.

    The algorithms to run are driven by the ``sparsity`` and ``quant``
    sections; see :attr:`OptimizeConfig.effective_algorithms`.

    Args:
        model: The PyTorch model to optimize.
        sample_input: A sample input tensor used for calibration.
        config: Optional :class:`OptimizeConfig` (or legacy :class:`CNNConfig`).
        calib_data: Optional calibration tensors from
            :func:`data_process`.
        **kwargs: Per-algorithm options keyed by algorithm name, e.g.
            ``sparsify={"mode": "sparsegpt", "sparsity": 0.5}``.

    Returns:
        The optimized ``torch.nn.Module``.
    """
    optimizer = CNNOptimizer(config=config)
    return optimizer.optimize(
        model,
        sample_input,
        calib_data=calib_data,
        **kwargs,
    )
