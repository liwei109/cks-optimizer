"""Load CNN models and infer their input shapes."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import torch
import timm
import torchvision.models as torchvision_models
from PIL import Image

logger = logging.getLogger(__name__)

__all__ = [
    "CNNModelMetadata",
    "load_cnn_model",
]

#: Export opset used by the ONNX exporter; also the fallback opset for
#: ``onnx2torch`` on torch versions that hide the real value.
DEFAULT_OPSET_VERSION = 17


@dataclass
class CNNModelMetadata:
    """Model-specific input preprocessing and fixed-shape information.

    Attributes:
        input_shape: Known model input dimensions in NCHW order.
        image_transform: Optional PIL image transform associated with the model.
        detection_output: Output convention for supported detection models.
        max_batch_size: Maximum supported input batch size, if fixed by the model.
    """

    input_shape: tuple[int, ...] | None = None
    image_transform: Callable[[Image.Image], torch.Tensor] | None = None
    detection_output: Literal["yolov5_raw"] | None = None
    max_batch_size: int | None = None

    @classmethod
    def for_model(cls, model: torch.nn.Module) -> CNNModelMetadata:
        """Return the model's metadata object, creating it when absent.

        Args:
            model: Model whose ``cnn_metadata`` attribute is read or initialized.
        """
        metadata = getattr(model, "cnn_metadata", None)
        if not isinstance(metadata, cls):
            metadata = cls()
            model.cnn_metadata = metadata  # type: ignore[attr-defined]
        return metadata

    @classmethod
    def infer_input_shape(
        cls, model: torch.nn.Module, model_name: str | None = None
    ) -> tuple[int, ...] | None:
        """Infer NCHW input dimensions from model metadata or its name.

        Args:
            model: Model to inspect for input-size metadata.
            model_name: Optional name used for known-size name heuristics.

        Returns:
            A four-dimensional NCHW shape, or ``None`` if no shape is known.
        """
        candidates = []
        for attr in ("default_cfg", "pretrained_cfg"):
            cfg = getattr(model, attr, None)
            if isinstance(cfg, dict) and "input_size" in cfg:
                candidates.append(cfg["input_size"])

        config = getattr(model, "config", None)
        if config is not None and hasattr(config, "input_size"):
            candidates.append(config.input_size)

        candidates.extend(
            getattr(model, attr)
            for attr in ("input_size", "img_size")
            if hasattr(model, attr)
        )
        if shape := cls.for_model(model).input_shape:
            candidates.append(shape)

        for candidate in candidates:
            if candidate is None:
                continue
            shape = tuple(int(value) for value in candidate)
            if len(shape) == 4 and all(dim > 0 for dim in shape):
                return shape
            if len(shape) == 3 and all(dim > 0 for dim in shape):
                return (1, *shape)
            if len(shape) == 2 and all(dim > 0 for dim in shape):
                return (1, 3, *shape)

        name = (model_name or "").lower()
        if "256" in name:
            return (1, 3, 256, 256)
        if "384" in name:
            return (1, 3, 384, 384)
        return None


def _load_torchvision_model(
    model_name: str, *, pretrained: bool = False
) -> torch.nn.Module:
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
        # Resolve the official weights enum both for the model factory and
        # its preprocessing metadata.
        get_model_weights = getattr(torchvision_models, "get_model_weights", None)
        weights_enum = (
            get_model_weights(model_factory)
            if get_model_weights is not None
            else getattr(model_factory, "Weights", None)
        )
        weights = weights_enum.DEFAULT if weights_enum is not None else "DEFAULT"
    model_kwargs: dict[str, Any] = {"weights": weights}
    if (
        not pretrained
        and "weights_backbone" in inspect.signature(model_factory).parameters
    ):
        model_kwargs["weights_backbone"] = None
    try:
        model = model_factory(**model_kwargs)
    except TypeError:
        model = model_factory(pretrained=pretrained)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Resolved torchvision model '{resolved}' is not a nn.Module.")
    weight_transforms = getattr(weights, "transforms", None)
    preprocessing = (
        weight_transforms() if pretrained and callable(weight_transforms) else None
    )
    CNNModelMetadata.for_model(model).image_transform = preprocessing
    return model


def _load_timm_model(
    model_name: str, *, pretrained: bool = False, **model_kwargs: Any
) -> torch.nn.Module:
    import timm

    model = timm.create_model(model_name, pretrained=pretrained, **model_kwargs)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Resolved timm model {model_name!r} is not a nn.Module.")
    return model


def _load_onnx_model(model_path: str | Path) -> torch.nn.Module:
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

    import onnx

    graph = onnx.load(str(path), load_external_data=False)
    initializer_names = {initializer.name for initializer in graph.graph.initializer}
    declared_shape = None
    for value_info in graph.graph.input:
        if value_info.name in initializer_names:
            continue
        dims = [
            dim.dim_value
            for dim in value_info.type.tensor_type.shape.dim
            if dim.HasField("dim_value")
        ]
        if len(dims) in (3, 4) and all(dim > 0 for dim in dims):
            declared_shape = (
                tuple(int(dim) for dim in dims)
                if len(dims) == 4
                else (1, 3, dims[-2], dims[-1])
            )
            break
    has_yolov5_raw_output = any(
        len(output.type.tensor_type.shape.dim) == 3
        and output.type.tensor_type.shape.dim[-1].HasField("dim_value")
        and output.type.tensor_type.shape.dim[-1].dim_value == 85
        for output in graph.graph.output
    )
    external_data = any(
        tensor.data_location == onnx.TensorProto.EXTERNAL
        for tensor in graph.graph.initializer
    )
    source: str | Any = str(path)
    max_batch_size = None
    if external_data:
        source = onnx.load(str(path), load_external_data=True)

    if has_yolov5_raw_output and declared_shape and declared_shape[0] == 1:
        if isinstance(source, str):
            source = onnx.load(source, load_external_data=True)
        from onnx import numpy_helper

        shape_tensors = {
            initializer.name: initializer for initializer in source.graph.initializer
        }
        constant_tensors = {}
        for node in source.graph.node:
            if node.op_type == "Constant":
                tensor = next(
                    (
                        attribute.t
                        for attribute in node.attribute
                        if attribute.name == "value"
                    ),
                    None,
                )
                if tensor is not None:
                    constant_tensors[node.output[0]] = tensor

        patched_shapes = 0
        for node in source.graph.node:
            if node.op_type != "Reshape" or len(node.input) < 2:
                continue
            tensor = shape_tensors.get(node.input[1]) or constant_tensors.get(
                node.input[1]
            )
            if tensor is None:
                continue
            shape = numpy_helper.to_array(tensor)
            if shape.ndim != 1 or not shape.size or int(shape[0]) != 1:
                continue
            shape = shape.copy()
            shape[0] = 0
            tensor.CopyFrom(numpy_helper.from_array(shape, name=tensor.name))
            patched_shapes += 1

        if patched_shapes:
            for value_info in (*source.graph.input, *source.graph.output):
                if (
                    value_info.name in shape_tensors
                    or not value_info.type.tensor_type.shape.dim
                ):
                    continue
                batch_dimension = value_info.type.tensor_type.shape.dim[0]
                batch_dimension.ClearField("dim_value")
                batch_dimension.dim_param = "batch_size"
            logger.info("Enabled dynamic batch support for YOLOv5 ONNX model %s.", path)
        else:
            max_batch_size = 1
            logger.warning(
                "YOLOv5 ONNX model %s has a fixed batch size and could not be "
                "converted to dynamic batch; using batch size 1.",
                path,
            )

    import torch.onnx.symbolic_helper as symbolic_helper

    if not hasattr(symbolic_helper, "GLOBALS") and not hasattr(
        symbolic_helper, "_export_onnx_opset_version"
    ):
        symbolic_helper._export_onnx_opset_version = DEFAULT_OPSET_VERSION  # type: ignore[attr-defined]
    onnx_model = convert(source)
    if not isinstance(onnx_model, torch.nn.Module):
        raise TypeError(f"onnx2torch did not return a nn.Module for '{path}'.")
    metadata = CNNModelMetadata.for_model(onnx_model)
    metadata.input_shape = declared_shape
    metadata.detection_output = "yolov5_raw" if has_yolov5_raw_output else None
    metadata.max_batch_size = max_batch_size
    return onnx_model


def load_cnn_model(
    model_source: str | Path,
    *,
    pretrained: bool = False,
    device: str | torch.device | None = None,
    **model_kwargs: Any,
) -> torch.nn.Module:
    """Load an ONNX file or a torchvision/timm model by name.

    The model is moved to ``device`` and put in evaluation mode. This function
    does not create a sample input; the pipeline uses real dataset inputs.

    Args:
        model_source: Torchvision/timm model name or ONNX file path.
        pretrained: Load pretrained weights for torchvision/timm models.
        device: Target device for the model.
        **model_kwargs: Extra keyword arguments forwarded to ``timm.create_model``.

    Returns:
        The loaded model on the requested device in evaluation mode.

    Raises:
        ValueError: If the model name is not registered or the source is
            unsupported.
        FileNotFoundError: If an ONNX model path does not exist.
        ImportError: If loading an ONNX model requires the unavailable
            ``onnx2torch`` package.
    """
    name = str(model_source)
    if name.lower().endswith(".onnx"):
        model = _load_onnx_model(name)
    elif name in torchvision_models.list_models():
        model = _load_torchvision_model(name, pretrained=pretrained)
    elif name in timm.list_models():
        model = _load_timm_model(name, pretrained=pretrained, **model_kwargs)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model.to(device or "cpu").eval()
