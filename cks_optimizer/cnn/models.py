"""Load CNN models and infer their input shapes."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "TORCH_CHECKPOINT_SUFFIXES",
    "infer_model_input_shape",
    "load_cnn_model",
    "load_model",
    "resolve_cnn_model",
    "resolve_cnn_model_and_sample_input",
]

#: File extensions treated as local torch checkpoints.
TORCH_CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt", ".bin")

#: Export opset used by the ONNX exporter; also the fallback opset for
#: ``onnx2torch`` on torch versions that hide the real value.
DEFAULT_OPSET_VERSION = 17


def _shape_from_input_size(
    input_size: Iterable[int] | tuple[int, ...] | None,
) -> tuple[int, ...] | None:
    if input_size is None:
        return None
    size = tuple(int(v) for v in input_size)
    if len(size) == 4:
        return size
    if len(size) == 3:
        _, height, width = size
        return (1, 3, height, width)
    if len(size) == 2:
        height, width = size
        return (1, 3, height, width)
    return None


def infer_model_input_shape(
    model: torch.nn.Module,
    model_name: str | None = None,
) -> tuple[int, ...] | None:
    """Best-effort NCHW input shape for ``model``, or ``None`` if unknown.

    Unlike :func:`_model_input_shape_or_default`, this never falls back to a
    hardcoded 224x224 default, so callers can tell "not found" apart from a
    guess and fall back to their own source of truth (e.g. the dataset
    preprocessing in :func:`~cks_optimizer.cnn.datasets.processing.data_process`).
    """
    candidates = []

    for attr in ("default_cfg", "pretrained_cfg"):
        cfg = getattr(model, attr, None)
        if isinstance(cfg, dict) and "input_size" in cfg:
            candidates.append(cfg["input_size"])

    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "input_size"):
        candidates.append(config.input_size)

    for attr in ("input_size", "img_size"):
        if hasattr(model, attr):
            candidates.append(getattr(model, attr))

    # Set by :func:`_load_local_onnx_model` from the ONNX graph signature.
    onnx_shape = getattr(model, "_cks_onnx_input_shape", None)
    if onnx_shape is not None:
        candidates.append(onnx_shape)

    for candidate in candidates:
        shape = _shape_from_input_size(candidate)
        if shape is not None:
            return shape

    name = (model_name or "").lower()
    if "256" in name:
        return (1, 3, 256, 256)
    if "384" in name:
        return (1, 3, 384, 384)

    return None


def _model_input_shape_or_default(
    model: torch.nn.Module,
    model_name: str | None = None,
) -> tuple[int, ...]:
    """Like :func:`infer_model_input_shape` but always returns a shape.

    Used only to build the random ``sample_input`` fallback, where a plausible
    default beats failing outright.
    """
    return infer_model_input_shape(model, model_name) or (1, 3, 224, 224)


def _load_torchvision_model(model_name: str, *, pretrained: bool = False) -> torch.nn.Module:
    import torchvision.models as torchvision_models

    resolved = model_name.strip()
    for prefix in ("torchvision.models.", "torchvision."):
        if resolved.startswith(prefix):
            resolved = resolved[len(prefix) :]
            break
    model_factory: Any = torchvision_models
    for component in resolved.split("."):
        if not hasattr(model_factory, component):
            model_factory = None
            break
        model_factory = getattr(model_factory, component)
    if model_factory is None:
        raise ValueError(
            f"Model '{model_name}' is not available in torchvision.models. "
            "Use one of the torchvision model names such as resnet18, "
            "resnet50, segmentation.deeplabv3_resnet50, or vit_b_16."
        )

    weights: Any = None
    if pretrained:
        # Prefer the official weights enum; fall back to the legacy
        # ``pretrained=`` flag for factories that predate it.
        weights_enum = getattr(model_factory, "Weights", None)
        weights = weights_enum.DEFAULT if weights_enum is not None else "DEFAULT"
    model_kwargs: dict[str, Any] = {"weights": weights}
    if not pretrained and "weights_backbone" in inspect.signature(model_factory).parameters:
        model_kwargs["weights_backbone"] = None
    try:
        model = model_factory(**model_kwargs)
    except TypeError:
        model = model_factory(pretrained=pretrained)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Resolved torchvision model '{resolved}' is not a nn.Module.")
    return model.eval()


def _load_timm_model(
    model_name: str, *, pretrained: bool = False, **kwargs: Any
) -> torch.nn.Module:
    import timm

    model = timm.create_model(model_name, pretrained=pretrained, **kwargs)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Resolved timm model '{model_name}' is not a nn.Module.")
    return model.eval()


def _load_local_torch_model(
    model_path: str | Path, *, weights_only: bool = True
) -> torch.nn.Module:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file '{path}' does not exist.")

    try:
        loaded = torch.load(path, map_location="cpu", weights_only=weights_only)
    except Exception as exc:
        if not weights_only:
            raise
        # Pickles that contain full model objects cannot be loaded with
        # ``weights_only=True``; retry only when the user opted in.
        raise ValueError(
            f"Could not safely load '{path}' with weights_only=True ({exc}). "
            "If this checkpoint comes from a trusted source, pass "
            "trust_checkpoint_code=True to load the embedded model code."
        ) from exc

    if isinstance(loaded, torch.nn.Module):
        return loaded.eval()

    if isinstance(loaded, dict):
        for key in ("model", "network", "module"):
            value = loaded.get(key)
            if isinstance(value, torch.nn.Module):
                return value.eval()
        if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            raise ValueError(
                "The checkpoint contains only a state_dict, not a complete "
                "model object. Please pass a model object from "
                "torchvision/timm or provide a full model file."
            )

    raise TypeError(
        f"Unsupported torch checkpoint format in '{path}'. "
        "Expected a nn.Module instance or a dict containing a model object."
    )


def _has_external_data(path: Path) -> bool:
    """Whether an ONNX file keeps any tensor in an external data file."""
    try:
        import onnx

        model = onnx.load(str(path), load_external_data=False)
    except Exception as exc:  # pragma: no cover - malformed file
        logger.debug("Cannot inspect external data of %s: %s", path, exc)
        return False
    return any(
        tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer
    )


def _onnx_declared_input_shape(source: Any) -> tuple[int, ...] | None:
    """Return the first concrete NCHW input shape declared by an ONNX graph.

    ``onnx2torch`` drops the graph metadata, so ``infer_model_input_shape``
    cannot recover the shape from the converted module. Recording it at load
    time lets the pipeline use the model's own shape instead of guessing from
    the calibration dataset.
    """
    try:
        import onnx

        # Graph inputs never live in external data files, so a plain
        # load without external data is enough to read the signature.
        model = onnx.load(source, load_external_data=False) if isinstance(source, str) else source
    except Exception as exc:  # pragma: no cover - malformed file
        logger.debug("Cannot read declared input shape: %s", exc)
        return None

    initializer_names = {initializer.name for initializer in model.graph.initializer}
    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue
        dims = [
            dim.dim_value
            for dim in value_info.type.tensor_type.shape.dim
            if dim.HasField("dim_value")
        ]
        if len(dims) not in (3, 4) or any(dim <= 0 for dim in dims):
            continue
        if len(dims) == 4:
            return tuple(int(dim) for dim in dims)
        _, height, width = dims
        return (1, 3, height, width)
    return None


def _patch_onnx2torch_opset_version() -> None:
    """Keep ``onnx2torch`` working on torch versions that removed the opset global.

    torch 2.13 dropped both ``symbolic_helper.GLOBALS`` and the private
    ``_export_onnx_opset_version`` read by ``onnx2torch.get_onnx_version()``,
    which breaks conversion of graphs with ``Shape``-like nodes. The value only
    matters when re-exporting to ONNX, so our export opset is a fine stand-in.
    """
    import torch.onnx.symbolic_helper as symbolic_helper

    if not hasattr(symbolic_helper, "GLOBALS") and not hasattr(
        symbolic_helper, "_export_onnx_opset_version"
    ):
        symbolic_helper._export_onnx_opset_version = DEFAULT_OPSET_VERSION  # type: ignore[attr-defined]


def _load_local_onnx_model(model_path: str | Path) -> torch.nn.Module:
    """Load a local ONNX model and convert it to a ``torch.nn.Module``.

    Uses ``onnx2torch`` when available. The dependency is imported lazily so
    that the rest of the toolkit works without it installed.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model file '{path}' does not exist.")

    try:
        from onnx2torch import convert
    except ImportError as exc:
        raise ImportError(
            "Loading a local ONNX model requires the 'onnx2torch' package. "
            "Install it with 'pip install -e thirdparty/onnx2torch'."
        ) from exc

    # ``convert`` takes a path or a ModelProto. Models with external tensor
    # data must be handed over as a proto with the weights already loaded,
    # because onnx2torch's own shape inference does not resolve external files.
    source: str | Any = str(path)
    declared_shape = _onnx_declared_input_shape(str(path))
    if _has_external_data(path):
        import onnx

        source = onnx.load(str(path), load_external_data=True)

    _patch_onnx2torch_opset_version()
    onnx_model = convert(source)
    if not isinstance(onnx_model, torch.nn.Module):
        raise TypeError(f"onnx2torch did not return a nn.Module for '{path}'.")
    if declared_shape is not None:
        # Read back by infer_model_input_shape(); nn.Module.__setattr__
        # stores plain values normally, only Parameters/Modules are special.
        onnx_model._cks_onnx_input_shape = declared_shape  # type: ignore[assignment]
    return onnx_model.eval()


def _to_device(
    model: torch.nn.Module,
    device: str | torch.device | None,
) -> torch.nn.Module:
    """Move a model to ``device`` and put it in eval mode."""
    target = torch.device(device) if device else torch.device("cpu")
    return model.to(target).eval()


def resolve_cnn_model(
    model_source: str | torch.nn.Module | Path,
    *,
    pretrained: bool = False,
    device: str | torch.device | None = None,
    trust_checkpoint_code: bool = False,
    **model_kwargs: Any,
) -> torch.nn.Module:
    """Resolve a CNN model from a model object, model name, or path.

    The model is moved to ``device`` and put in evaluation mode. This function
    does not create an input tensor.

    Args:
        model_source: Model object, torchvision/timm name, or local path.
        pretrained: Load pretrained weights for torchvision/timm models.
        device: Target device for the model.
        trust_checkpoint_code: Allow ``torch.load(weights_only=False)`` for
            local checkpoints that pickle a full model object. Off by default
            because unpickling executes arbitrary code.
        **model_kwargs: Extra keyword arguments forwarded to ``timm.create_model``.
    """
    if isinstance(model_source, torch.nn.Module):
        return _to_device(model_source, device)

    if not isinstance(model_source, str):
        model_source = str(model_source)

    cleaned = model_source.strip()
    if not cleaned:
        raise ValueError("Model specification cannot be empty.")

    model_path = Path(cleaned)
    suffix = model_path.suffix.lower()
    if suffix == ".onnx":
        return _to_device(_load_local_onnx_model(cleaned), device)

    if suffix in TORCH_CHECKPOINT_SUFFIXES or model_path.is_file():
        return _to_device(
            _load_local_torch_model(cleaned, weights_only=not trust_checkpoint_code),
            device,
        )

    try:
        model = _load_torchvision_model(cleaned, pretrained=pretrained)
    except ValueError as exc:
        logger.debug("torchvision resolution failed for %r: %s", cleaned, exc)
    else:
        return _to_device(model, device)

    try:
        model = _load_timm_model(cleaned, pretrained=pretrained, **model_kwargs)
    except (ValueError, TypeError, AttributeError, RuntimeError) as exc:
        logger.debug("timm resolution failed for %r: %s", cleaned, exc)
    else:
        return _to_device(model, device)

    raise ValueError(
        f"Could not resolve CNN model '{cleaned}'. Use a torchvision model "
        "name, a timm model name, a local torch checkpoint path, or a local "
        "ONNX model path."
    )


def resolve_cnn_model_and_sample_input(
    model_source: str | torch.nn.Module | Path,
    *,
    input_shape: Iterable[int] | None = None,
    pretrained: bool = False,
    device: str | torch.device | None = None,
    trust_checkpoint_code: bool = False,
    **model_kwargs: Any,
) -> tuple[torch.nn.Module, torch.Tensor]:
    """Resolve a CNN model and create a random sample input.

    The sample input is a placeholder for callers without real input data.
    """
    model = resolve_cnn_model(
        model_source,
        pretrained=pretrained,
        device=device,
        trust_checkpoint_code=trust_checkpoint_code,
        **model_kwargs,
    )
    name = model_source if isinstance(model_source, str) else None
    shape = (
        tuple(int(v) for v in input_shape)
        if input_shape is not None
        else _model_input_shape_or_default(model, name)
    )
    target = torch.device(device) if device else torch.device("cpu")
    return model, torch.randn(*shape, device=target)


def load_model(
    model_source: str | torch.nn.Module | Path,
    *,
    input_shape: Iterable[int] | None = None,
    pretrained: bool = False,
    device: str | torch.device | None = None,
    trust_checkpoint_code: bool = False,
    **model_kwargs: Any,
) -> tuple[torch.nn.Module, torch.Tensor]:
    """Load a CNN model and create a matching sample input.

    This is a thin, explicitly-named wrapper around
    :func:`resolve_cnn_model_and_sample_input`. Supported sources:

    * a ``torch.nn.Module`` instance,
    * a torchvision model name (e.g. ``"resnet50"``),
    * a timm model name,
    * a local torch checkpoint (``.pt``/``.pth``/``.ckpt``/``.bin``),
    * a local ONNX model (``.onnx``), converted via ``onnx2torch``.

    Args:
        model_source: The model object, name, or path.
        input_shape: Optional explicit input shape (NCHW).
        pretrained: Load pretrained weights for torchvision/timm models.
        device: Target device for the model and sample input.
        trust_checkpoint_code: Allow unpickling full model objects from local
            checkpoints (executes arbitrary code; trusted files only).
        **model_kwargs: Extra keyword arguments forwarded to ``timm.create_model``.

    Returns:
        A ``(model, sample_input)`` tuple.
    """
    return resolve_cnn_model_and_sample_input(
        model_source,
        input_shape=input_shape,
        pretrained=pretrained,
        device=device,
        trust_checkpoint_code=trust_checkpoint_code,
        **model_kwargs,
    )


def load_cnn_model(
    model_source: str | torch.nn.Module | Path,
    *,
    pretrained: bool = False,
    device: str | torch.device | None = None,
    trust_checkpoint_code: bool = False,
    **model_kwargs: Any,
) -> torch.nn.Module:
    """Load a CNN model without creating a sample input.

    Thin, explicitly-named wrapper around :func:`resolve_cnn_model`; see
    :func:`load_model` for the supported ``model_source`` forms.
    """
    return resolve_cnn_model(
        model_source,
        pretrained=pretrained,
        device=device,
        trust_checkpoint_code=trust_checkpoint_code,
        **model_kwargs,
    )
