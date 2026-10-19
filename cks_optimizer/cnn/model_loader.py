from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch


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


def _default_input_shape_for_name(model_name: str | None) -> tuple[int, ...]:
    name = (model_name or "").lower()
    if "yolo" in name or "detection" in name:
        return (1, 3, 640, 640)
    if "seg" in name or "unet" in name:
        return (1, 3, 224, 224)
    if "256" in name or "_256" in name:
        return (1, 3, 256, 256)
    if "384" in name or "_384" in name:
        return (1, 3, 384, 384)
    return (1, 3, 224, 224)


def _normalize_model_name(value: str) -> str:
    return value.strip().strip("./")


def _load_torchvision_model(model_name: str) -> torch.nn.Module:
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
        model = model_factory(weights=None)
    except TypeError:
        model = model_factory()
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"Resolved torchvision model '{resolved}' is not a nn.Module."
        )
    return model.eval()


def _load_timm_model(model_name: str) -> torch.nn.Module:
    import timm

    model = timm.create_model(model_name, pretrained=False)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "Resolved timm model '{model_name}' is not a nn.Module."
        )
    return model.eval()


def _load_local_torch_model(model_path: str | Path) -> torch.nn.Module:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file '{path}' does not exist.")

    loaded = torch.load(path, map_location="cpu", weights_only=False)

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


def resolve_cnn_model_and_sample_input(
    model_spec: str | torch.nn.Module | Path,
    *,
    model_kind: str = "auto",
    input_shape: Iterable[int] | None = None,
) -> tuple[torch.nn.Module, torch.Tensor]:
    """Resolve a CNN model from a model object, model name, or path."""
    if isinstance(model_spec, torch.nn.Module):
        shape = (
            tuple(input_shape)
            if input_shape is not None
            else _infer_model_input_shape(model_spec, None)
        )
        return model_spec.eval(), torch.randn(*shape)

    if not isinstance(model_spec, str):
        model_spec = str(model_spec)

    cleaned = model_spec.strip()
    if not cleaned:
        raise ValueError("Model specification cannot be empty.")

    path_like = Path(cleaned)
    if (
        cleaned.endswith((".pt", ".pth", ".ckpt", ".bin"))
        or path_like.exists()
    ):
        if model_kind not in {"auto", "local", "torch", "checkpoint"}:
            raise ValueError(
                f"model_kind='{model_kind}' is incompatible with a local "
                f"model path: '{cleaned}'."
            )
        model = _load_local_torch_model(cleaned)
        shape = (
            tuple(input_shape)
            if input_shape is not None
            else _infer_model_input_shape(model, cleaned)
        )
        return model.eval(), torch.randn(*shape)

    if cleaned.endswith(".onnx"):
        raise ValueError(
            "Direct ONNX model loading is not supported in the "
            "torch-quantization path. Please pass a PyTorch model object or "
            "a local torch checkpoint (.pt/.pth/.ckpt) or choose a "
            "torchvision/timm model by name."
        )

    if model_kind in {"auto", "torchvision"}:
        try:
            model = _load_torchvision_model(cleaned)
            shape = (
                tuple(input_shape)
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
                tuple(input_shape)
                if input_shape is not None
                else _infer_model_input_shape(model, cleaned)
            )
            return model, torch.randn(*shape)
        except (ValueError, TypeError, AttributeError):
            pass

    raise ValueError(
        "Could not resolve CNN model '{cleaned}'. Use a torchvision model "
        "name, a timm model name, or a local torch checkpoint path."
    )
