from __future__ import annotations

import inspect
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx
import torch
from torch import nn
from torch.jit import TracerWarning

from compressor.export.contract import DeploymentManifest, TensorQuant

logger = logging.getLogger(__name__)


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

    # ModelOpt's TensorQuantizer.forward branches on tensor booleans, which
    # the legacy tracer cannot record. The resulting TracerWarning is benign
    # here: quantization scales are already frozen by calibration, so the
    # exported graph is deterministic by design. Suppress only this category.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=TracerWarning)
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


@dataclass
class CVDeploymentBundle:
    """CV deployment bundle: QDQ ONNX + manifest(+ optional rlym, see rlym_builder)"""

    onnx_path: Path
    manifest_path: Path
    manifest: DeploymentManifest


def _quantizer_scale(module: nn.Module, attr_prefix: str) -> torch.Tensor | None:
    """Derive the quantization scale from a ModelOpt quantizer.

    ModelOpt stores quantizers as ``<prefix>_quantizer`` submodules
    (e.g. ``weight_quantizer`` / ``input_quantizer``). This version of
    TensorQuantizer has no ``.scale`` attribute — calibration results live
    in the ``.amax`` buffer, and the scale is ``amax / qmax`` (symmetric,
    zero-point 0). Weight amax is per-channel (axis 0); activation amax is
    per-tensor. Returns None when the quantizer is absent or uncalibrated.
    """
    quantizer = getattr(module, f"{attr_prefix}_quantizer", None)
    if quantizer is None:
        return None
    amax = getattr(quantizer, "amax", None)
    if amax is None:
        return None
    if isinstance(amax, torch.Tensor):
        if amax.numel() == 0:
            return None
        amax = amax.detach().float().cpu()
    else:
        amax = torch.tensor(float(amax))
    num_bits = int(getattr(quantizer, "num_bits", 8) or 8)
    unsigned = bool(getattr(quantizer, "unsigned", False) or False)
    qmax = (2**num_bits - 1) if unsigned else (2 ** (num_bits - 1) - 1)
    return amax / qmax


def _tensor_quant_from_scale(
    scale: torch.Tensor, axis: int | None, scheme: str
) -> TensorQuant:
    flat = scale.reshape(-1).tolist()
    zp = None  # ModelOpt INT8 default symmetric zero-point 0
    return TensorQuant(
        scale=flat,
        zero_point=zp,
        axis=None if len(flat) == 1 else (axis if axis is not None else 0),
        dtype="int8",
        scheme=scheme,
    )


def build_manifest_from_model(
    model: nn.Module,
    *,
    model_name: str = "",
    quant_mode: str = "int8",
) -> DeploymentManifest:
    """Extract manifest from quantizer attributes of a fake-quant model.

    ModelOpt wraps quantizable layers (QuantConv2d / QuantLinear) with
    ``weight_quantizer`` / ``input_quantizer`` submodules, so quantized
    layers are NOT leaves — iterating must not filter by children.
    Containers and the quantizer modules themselves are naturally skipped
    because they carry no ``*_quantizer`` attributes.
    """
    manifest = DeploymentManifest(model_source=model_name, quant_mode=quant_mode)
    for name, module in model.named_modules():
        w_scale = _quantizer_scale(module, "weight")
        if w_scale is not None:
            manifest.set_tensor(
                f"{name}.weight", _tensor_quant_from_scale(w_scale, 0, "INT8")
            )
        in_scale = _quantizer_scale(module, "input")
        if in_scale is not None:
            manifest.set_tensor(
                f"{name}.input", _tensor_quant_from_scale(in_scale, None, "INT8")
            )
    return manifest


def export_cv_deployment(
    model: nn.Module,
    sample_input: Any,
    output_path: str | Path,
    *,
    model_name: str = "",
    quant_mode: str = "int8",
    use_dynamic_axes: bool = False,
    manifest: DeploymentManifest | None = None,
    simplify: bool = True,
    opset_version: int = 17,
) -> CVDeploymentBundle:
    """export CV deployment bundle: QDQ ONNX + manifest sidecar.
    """
    output_path = Path(output_path)
    export_torch_to_onnx(
        model,
        output_path,
        sample_input,
        opset_version=opset_version,
        dynamic_axes=(
            {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
            if use_dynamic_axes
            else None
        ),
        input_names=["input"],
        output_names=["output"],
        simplify=simplify,
    )

    if manifest is None:
        manifest = build_manifest_from_model(
            model, model_name=model_name, quant_mode=quant_mode
        )
        if not manifest.tensors:
            logger.warning(
                "no scale found in model, manifest is empty; "
                "please explicitly construct DeploymentManifest or check quantizer naming convention"
            )

    manifest_path = output_path.with_suffix(".manifest.json")
    manifest.save(manifest_path)
    return CVDeploymentBundle(
        onnx_path=output_path, manifest_path=manifest_path, manifest=manifest
    )
