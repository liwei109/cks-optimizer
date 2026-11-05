from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from compressor.model.loaders.local_ckpt import (
    load_torch_model,
    looks_like_torch_checkpoint,
)


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


def _infer_model_input_shape(
    model: torch.nn.Module,
    model_name: str | None = None,
) -> tuple[int, ...]:
    """Best-effort NCHW input shape for ``model``, or ``None`` if unknown.

    Unlike :func:`_infer_model_input_shape`, this never falls back to a
    hardcoded 224x224 default, so callers can tell "not found" apart from a
    guess and fall back to their own source of truth (e.g. the dataset
    preprocessing in :func:`~cks_optimizer.cnn.data.data_process`).
    """
    candidates = []

    for attr in ("default_cfg", "pretrained_cfg"):
        cfg = getattr(model, attr, None)
        if isinstance(cfg, dict) and "input_size" in cfg:
            candidates.append(cfg["input_size"])

    if hasattr(model, "config"):
        config = getattr(model, "config")
        if hasattr(config, "input_size"):
            candidates.append(config.input_size)

    if hasattr(model, "input_size"):
        candidates.append(getattr(model, "input_size"))

    if hasattr(model, "img_size"):
        candidates.append(getattr(model, "img_size"))

    for candidate in candidates:
        shape = _shape_from_input_size(candidate)
        if shape is not None:
            return shape

    name = (model_name or "").lower()
    if "256" in name or "_256" in name:
        return (1, 3, 256, 256)
    if "384" in name or "_384" in name:
        return (1, 3, 384, 384)

    return (1, 3, 224, 224)


def _instantiate_torchvision(model_name: str, *, weights: Any = None) -> torch.nn.Module:
    import torchvision.models as torchvision_models

    resolved = model_name.strip()
    if resolved.startswith("torchvision."):
        resolved = resolved.split(".", 1)[1]

    if not hasattr(torchvision_models, resolved):
        raise ValueError(
            "Model '{model_name}' is not available in torchvision.models. "
            "Use one of the torchvision model names such as resnet18, "
            "resnet50, mobilenet_v3_small, vit_b_16, etc."
        )

    model_factory = getattr(torchvision_models, resolved)
    try:
        model = model_factory(weights=weights)
    except TypeError:  # legacy factory without a weights kwarg
        model = model_factory()
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"Resolved torchvision model '{resolved}' is not a nn.Module."
        )
    return model.eval()


def _load_torchvision_model(model_name: str) -> torch.nn.Module:
    try:
        # A bare model name implies downloading pretrained weights online.
        return _instantiate_torchvision(model_name, weights="DEFAULT")
    except (ValueError, RuntimeError):
        # Either the architecture has no published weights (fall back to
        # random init), or the name is unknown (the retry re-raises the
        # same ValueError so the caller can try other sources).
        return _instantiate_torchvision(model_name, weights=None)


def _onnx_input_shape(onnx_model: Any) -> tuple[int, ...] | None:
    """Best-effort NCHW input shape from the ONNX graph's first input.

    Dynamic dims (dim_param / dim_value 0) become 1 (batch size).
    """
    try:
        tensor_type = onnx_model.graph.input[0].type.tensor_type
        dims = list(tensor_type.shape.dim)
    except (AttributeError, IndexError):
        return None
    shape: list[int] = []
    for dim in dims:
        value = int(dim.dim_value)
        shape.append(value if value > 0 else 1)
    if len(shape) == 4:
        return tuple(shape)
    if len(shape) == 3:  # CHW without batch
        return (1, *shape)
    return None


def _load_onnx_model(model_path: str) -> tuple[torch.nn.Module, tuple[int, ...] | None]:
    """Load an ONNX model as a torch nn.Module via onnx2torch.

    Returns ``(model, input_shape)`` where input_shape is read from the ONNX
    graph when statically known, so calibration matches the original input
    contract of the exported model.
    """
    try:
        import onnx
        import onnx2torch
    except ImportError as exc:
        raise ImportError(
            "onnx and onnx2torch are required to load ONNX models. "
            "Install them with 'pip install onnx onnx2torch'."
        ) from exc

    onnx_model = onnx.load(model_path)
    model = onnx2torch.convert(onnx_model)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"onnx2torch.convert('{model_path}') did not return a nn.Module."
        )
    return model.eval(), _onnx_input_shape(onnx_model)


def _rebuild_torchvision_from_state_dict(model_path: str) -> torch.nn.Module:
    """Rebuild a torchvision model from a state_dict-only checkpoint.

    Torchvision model-zoo files (e.g. ``resnet18-f37072fd.pth``) are bare
    state_dicts named ``<arch>-<tag>.pth``. Derive the architecture from the
    filename, instantiate it with random weights, and load the state_dict
    (strict). Raises a descriptive error when the filename does not encode
    a known architecture or the keys do not match.
    """
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and isinstance(
        payload.get("state_dict"), dict
    ):  # training-style checkpoint: {..., "state_dict": {...}}
        payload = payload["state_dict"]
    if not isinstance(payload, dict) or not payload:
        raise TypeError(
            f"Unsupported torch checkpoint format in '{model_path}'. "
            "Expected an nn.Module pickle or a state_dict."
        )

    arch = Path(model_path).stem.split("-")[0]
    # weights=None: random init — the state_dict overwrites everything, so
    # downloading pretrained weights here would be wasted work.
    model = _instantiate_torchvision(arch, weights=None)
    try:
        model.load_state_dict(payload)
    except RuntimeError as exc:
        raise ValueError(
            f"The state_dict in '{model_path}' does not match torchvision "
            f"architecture '{arch}' (inferred from the filename): {exc}"
        ) from exc
    return model.eval()


def _load_timm_model(model_name: str) -> torch.nn.Module:
    import timm

    model = timm.create_model(model_name, pretrained=False)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "Resolved timm model '{model_name}' is not a nn.Module."
        )
    return model.eval()


def resolve_cv_model_and_sample_input(
    model_spec: str | torch.nn.Module | Path,
    *,
    model_kind: str = "auto",
    input_shape: Iterable[int] | None = None,
) -> tuple[torch.nn.Module, torch.Tensor]:
    """Resolve a CV model from a model object, model name, or path.

    Accepted specs: nn.Module, torchvision/timm model name (torchvision
    names download pretrained weights online), local torch checkpoint
    (.pt/.pth/.ckpt), or a local ONNX file (converted via onnx2torch).

    :param model_kind: auto / torchvision / timm / local / onnx
    :param input_shape: Explicit input shape.
    :return: (eval mode model, random sample input)
    """
    if isinstance(model_spec, torch.nn.Module):
        shape = (
            tuple(int(v) for v in input_shape)
            if input_shape is not None
            else _infer_model_input_shape(model_spec, None)
        )
        return model_spec.eval(), torch.randn(*shape)

    if not isinstance(model_spec, str):
        model_spec = str(model_spec)

    cleaned = model_spec.strip()
    if not cleaned:
        raise ValueError("Model specification cannot be empty.")

    # ONNX path: convert to torch via onnx2torch, then run the standard
    # ModelOpt PTQ path and re-export as QDQ ONNX. Checked before the
    # generic existing-path branch, which would otherwise feed the .onnx
    # file to torch.load.
    if cleaned.endswith(".onnx"):
        if model_kind not in {"auto", "local", "onnx"}:
            raise ValueError(
                f"model_kind='{model_kind}' is incompatible with an ONNX "
                f"model path: '{cleaned}'."
            )
        model, onnx_shape = _load_onnx_model(cleaned)
        shape = (
            tuple(int(v) for v in input_shape)
            if input_shape is not None
            else onnx_shape or _infer_model_input_shape(model, cleaned)
        )
        return model, torch.randn(*shape)

    path_like = Path(cleaned)
    if looks_like_torch_checkpoint(cleaned) or path_like.exists():
        if model_kind not in {"auto", "local", "torch", "checkpoint"}:
            raise ValueError(
                f"model_kind='{model_kind}' is incompatible with a local "
                f"model path: '{cleaned}'."
            )
        try:
            model = load_torch_model(cleaned)
        except (TypeError, ValueError):
            # state_dict-only checkpoint (e.g. torchvision zoo files named
            # <arch>-<tag>.pth) — rebuild the architecture from the filename.
            model = _rebuild_torchvision_from_state_dict(cleaned)
        shape = (
            tuple(int(v) for v in input_shape)
            if input_shape is not None
            else _infer_model_input_shape(model, cleaned)
        )
        return model.eval(), torch.randn(*shape)

    if model_kind in {"auto", "torchvision"}:
        try:
            model = _load_torchvision_model(cleaned)
            shape = (
                tuple(int(v) for v in input_shape)
                if input_shape is not None
                else _infer_model_input_shape(model, cleaned)
            )
            return model, torch.randn(*shape)
        except ValueError:
            pass

    if model_kind in {"auto", "timm"}:
        try:
            model = _load_timm_model(cleaned)
            shape = (
                tuple(int(v) for v in input_shape)
                if input_shape is not None
                else _infer_model_input_shape(model, cleaned)
            )
            return model, torch.randn(*shape)
        except (ValueError, TypeError, AttributeError):
            pass

    raise ValueError(
        "Could not resolve CV model '{cleaned}'. Use a torchvision model "
        "name (downloads pretrained weights), a timm model name, a local "
        "torch checkpoint (.pt/.pth/.ckpt), or a local ONNX file."
    )
