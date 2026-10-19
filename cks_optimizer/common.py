from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import onnx
import torch


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
            "onnxsim is required for ONNX simplification. "
            "Install it with 'pip install onnxsim'."
        ) from exc

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    model = onnx.load(str(path))
    model_simplified, check_ok = onnxsim.simplify(
        model,
        check_n=check,
        skip_fuse_bn=False,
    )
    if not check_ok:
        raise ValueError(f"onnxsim reported a failed validation for '{path}'.")

    onnx.save(model_simplified, str(path))
    onnx.checker.check_model(model_simplified)
    return path


def ensure_parent_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def export_torch_to_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    dummy_input: Any,
    opset_version: int = 17,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, Any] | None = None,
    simplify: bool = True,
    check_onnx: bool = True,
    **kwargs: Any,
) -> Path:
    """Export a PyTorch model to ONNX using the ModelOpt-compatible path.

    In the current cks-opt stack, the torch-export / dynamo-based path
    triggers a ModelOpt quantizer guard error during tracing. Setting
    `dynamo=False` uses the legacy export path that ModelOpt's own test
    suite expects and successfully produces ONNX for quantized CNN models.
    """
    path = ensure_parent_dir(output_path)
    model.eval()

    export_kwargs: dict[str, Any] = {
        "export_params": True,
        "opset_version": opset_version,
        "input_names": input_names or ["input"],
        "output_names": output_names or ["output"],
        "dynamic_axes": dynamic_axes,
    }

    export_signature = inspect.signature(torch.onnx.export)
    if "dynamo" in export_signature.parameters:
        export_kwargs["dynamo"] = False
    if "enable_onnx_checker" in export_signature.parameters:
        export_kwargs["enable_onnx_checker"] = False

    export_kwargs.update(kwargs)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            str(path),
            **export_kwargs,
        )

    if simplify:
        _simplify_onnx_model(path, check=check_onnx)

    if check_onnx:
        onnx_model = onnx.load(str(path))
        onnx.checker.check_model(onnx_model)

    return path
