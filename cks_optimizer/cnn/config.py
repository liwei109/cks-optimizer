from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cks_optimizer.cnn.data.registry import DATASET_REGISTRY


def infer_dataset_kind_from_root(dataset_root: str | Path | None) -> str:
    """Infer a CNN dataset kind from the filesystem layout.

    Iterates DATASET_REGISTRY in priority order and returns the first
    adapter whose ``matches`` accepts the layout; falls back to
    "imagefolder" when nothing matches.
    """
    root = Path(dataset_root) if dataset_root else Path(".")
    if not root.exists() or not root.is_dir():
        return "imagefolder"

    for kind, adapter_cls in DATASET_REGISTRY.items():
        if adapter_cls.matches(root):
            return kind

    return "imagefolder"


def build_default_output_path(model_name: str, quant_mode: str) -> str:
    """Build the default ONNX output path from model and quantization mode."""
    safe_model = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "-",
        (model_name or "model").strip(),
    )
    safe_model = safe_model.strip(".-") or "model"
    safe_quant = (quant_mode or "int8").strip().lower()
    return f"tmp/{safe_model}.{safe_quant}.onnx"


def infer_task_from_model_and_dataset(
    model_name: str,
    dataset_kind: str | None = None,
) -> str:
    """Infer the task from the model and dataset automatically."""
    model = (model_name or "").lower()
    dataset = (dataset_kind or "").lower()

    if any(token in model for token in ("seg", "unet", "deeplab", "mask")):
        return "segmentation"
    if any(token in model for token in ("yolo", "det", "detection")):
        return "detection"
    if dataset in {"coco", "voc"}:
        return "detection"
    return "classification"


@dataclass
class CNNConfig:
    """Configuration for CNN / CV optimization using ModelOpt torch mode."""

    model_name: str = "resnet"
    task: str = ""
    dataset_kind: str = ""
    dataset_root: str = ""
    quant_mode: str = "int8"
    precision: str = "int8"
    output_path: str = ""
    input_shape: tuple[int, ...] = (1, 3, 224, 224)
    use_dynamic_axes: bool = True
    calibration_samples: int = 32
    calibration_batch_size: int = 16
    normalize_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, ...] = (0.229, 0.224, 0.225)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.dataset_kind:
            self.dataset_kind = infer_dataset_kind_from_root(self.dataset_root)
        if not self.task:
            self.task = infer_task_from_model_and_dataset(
                self.model_name,
                self.dataset_kind,
            )
        if not self.output_path:
            self.output_path = build_default_output_path(
                self.model_name,
                self.quant_mode,
            )

    def output_file(self) -> Path:
        return Path(self.output_path)
