"""Export and validate ONNX models."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any

import onnx
import torch

logger = logging.getLogger(__name__)

__all__ = [
    "ensure_parent_dir",
    "export_onnx",
]


def _simplify_onnx_model(
    model_path: str | Path,
    *,
    check: bool = True,
) -> Path:
    """Run onnxsim simplification and optionally check the resulting model."""
    try:
        import onnxsim
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "onnxsim is required for ONNX simplification. Install it with 'pip install onnxsim'."
        ) from exc

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    model = onnx.load(str(path))
    # ``check_n`` is an int (number of times to run the checker), not a bool.
    model_simplified, check_ok = onnxsim.simplify(
        model,
        check_n=1 if check else 0,
        skip_fuse_bn=False,
    )
    if check and not check_ok:
        raise ValueError(f"onnxsim reported a failed validation for '{path}'.")

    onnx.save(model_simplified, str(path))
    onnx.checker.check_model(model_simplified)
    logger.info(
        "Simplified %s: %d -> %d nodes",
        path,
        len(model.graph.node),
        len(model_simplified.graph.node),
    )
    return path


def ensure_parent_dir(path: str | Path) -> Path:
    """Create a file path's parent directories and return the path.

    Args:
        path: File path whose parent directory should exist.

    Returns:
        The same path represented as a :class:`~pathlib.Path`.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def export_onnx(
    model: torch.nn.Module,
    sample_input: Any,
    config: Any = None,
    **kwargs: Any,
) -> Path:
    """Export a PyTorch model to ONNX using configured export settings.

    Args:
        model: Model to export. It is switched to evaluation mode.
        sample_input: Example input passed to ``torch.onnx.export``.
        config: ``OptimizeConfig``, legacy ``CNNConfig``, or ``None`` for
            default export settings.
        **kwargs: Additional keyword arguments forwarded to
            :func:`torch.onnx.export`, overriding matching defaults.

    Returns:
        The output ONNX file path.

    Raises:
        TypeError: If ``config`` is not a supported configuration type.
        RuntimeError: If PyTorch fails to export the model.
        ValueError: If ONNX validation or simplification fails.
    """
    from cks_optimizer.cnn.config import CNNConfig, OptimizeConfig

    if config is None:
        config = OptimizeConfig()
    elif isinstance(config, CNNConfig):
        config = config.to_optimize_config()
    elif not isinstance(config, OptimizeConfig):
        raise TypeError(
            f"config must be an OptimizeConfig, a CNNConfig or None, got {type(config).__name__}."
        )

    cfg = config.export
    dynamic_axes = (
        {name: {0: "batch_size"} for name in (*cfg.input_names, *cfg.output_names)}
        if cfg.use_dynamic_axes
        else None
    )
    path = ensure_parent_dir(cfg.output_path)
    model.eval()

    export_kwargs: dict[str, Any] = {
        "export_params": True,
        "opset_version": cfg.opset_version,
        "input_names": cfg.input_names,
        "output_names": cfg.output_names,
        "dynamic_axes": dynamic_axes,
    }
    export_signature = inspect.signature(torch.onnx.export)
    if "dynamo" in export_signature.parameters:
        export_kwargs["dynamo"] = False
    if "enable_onnx_checker" in export_signature.parameters:
        export_kwargs["enable_onnx_checker"] = False
    export_kwargs.update(kwargs)

    with torch.inference_mode():
        torch.onnx.export(model, sample_input, str(path), **export_kwargs)

    if cfg.simplify:
        _simplify_onnx_model(path, check=cfg.check_onnx)
    elif cfg.check_onnx:
        onnx.checker.check_model(onnx.load(str(path)))

    return path
