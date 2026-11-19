"""Ground-truth evaluation for classification, detection, and segmentation."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassJaccardIndex,
)
from torchmetrics.detection import MeanAveragePrecision
from tqdm.auto import tqdm

from cks_optimizer.cnn.config import OptimizeConfig
from cks_optimizer.cnn.detection import detection_prediction
from cks_optimizer.cnn.models import CNNModelMetadata

logger = logging.getLogger(__name__)

__all__ = ["EvaluationResult", "evaluate", "resolve_input_name"]


@dataclass
class EvaluationResult:
    """Task metric values measured against dataset annotations.

    Attributes:
        task: Evaluated task name.
        num_samples: Number of images evaluated.
        accuracy: Top-1 classification accuracy, or ``None`` for other tasks.
        mean_average_precision: COCO bbox mAP, or ``None`` for other tasks.
        map_50: COCO bbox mAP at IoU 0.50.
        map_75: COCO bbox mAP at IoU 0.75.
        mean_iou: Mean intersection over union, or ``None`` for other tasks.
    """

    task: str
    num_samples: int
    accuracy: float | None = None
    mean_average_precision: float | None = None
    map_50: float | None = None
    map_75: float | None = None
    mean_iou: float | None = None

    def summary(self) -> str:
        """Format the result as a concise human-readable metric summary."""
        if self.task == "classification":
            metric = f"accuracy={self.accuracy:.4f}"
        elif self.task == "detection":
            metric = (
                f"mAP={self.mean_average_precision:.4f}, "
                f"mAP50={self.map_50:.4f}, mAP75={self.map_75:.4f}"
            )
        else:
            metric = f"mIoU={self.mean_iou:.4f}"
        return f"Evaluated {self.num_samples} {self.task} sample(s): {metric}"


def resolve_input_name(onnx_path: str | Path, *, preferred: str | None = None) -> str:
    """Return a usable input name from an ONNX graph.

    Initializer-backed graph inputs are excluded. If ``preferred`` names a
    usable input, it is returned; otherwise the graph's first usable input is
    selected.

    Args:
        onnx_path: ONNX model file to inspect.
        preferred: Optional input name to prefer when present.

    Returns:
        The selected graph input name.

    Raises:
        FileNotFoundError: If the ONNX file cannot be loaded.
        ValueError: If the graph has no usable inputs.
    """
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover
        raise ImportError("onnx is required for ONNX input-name resolution.") from exc

    model = onnx.load(str(onnx_path))
    initializers = {initializer.name for initializer in model.graph.initializer}
    names = [item.name for item in model.graph.input if item.name not in initializers]
    if not names:
        raise ValueError(f"No usable graph input found in {onnx_path}.")
    if preferred is not None and preferred in names:
        return preferred
    return names[0]


def _batches(dataset: Any, batch_size: int) -> Iterator[tuple[torch.Tensor, list[Any]]]:
    for start in range(0, len(dataset), batch_size):
        samples = [
            dataset[index]
            for index in range(start, min(start + batch_size, len(dataset)))
        ]
        yield (
            torch.stack([sample[0] for sample in samples]),
            [sample[1] for sample in samples],
        )


def _classification_accuracy(
    model: torch.nn.Module,
    dataset: Any,
    batch_size: int,
    device: torch.device,
) -> EvaluationResult:
    accuracy = None
    with tqdm(
        total=len(dataset), desc="Classification evaluation", unit="image"
    ) as progress:
        for images, targets in _batches(dataset, batch_size):
            logits = model(images.to(device))
            if isinstance(logits, Mapping):
                logits = logits.get("logits")
            elif hasattr(logits, "logits"):
                logits = logits.logits
            if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                raise ValueError(
                    "Classification output must have shape [batch, classes]."
                )
            if logits.shape[0] != len(targets):
                raise ValueError(
                    "Classification output batch size does not match targets."
                )
            if accuracy is None:
                accuracy = MulticlassAccuracy(
                    num_classes=logits.shape[1], average="micro"
                ).to(device)
            accuracy.update(
                logits, torch.as_tensor(targets, device=device, dtype=torch.long)
            )
            progress.update(len(targets))

    if accuracy is None:
        raise ValueError("The classification evaluation split is empty.")
    return EvaluationResult(
        task="classification",
        num_samples=len(dataset),
        accuracy=float(accuracy.compute().item()),
    )


def _segmentation_miou(
    model: torch.nn.Module,
    dataset: Any,
    batch_size: int,
    device: torch.device,
) -> EvaluationResult:
    confusion = jaccard = None
    with tqdm(
        total=len(dataset), desc="Segmentation evaluation", unit="image"
    ) as progress:
        for images, targets in _batches(dataset, batch_size):
            outputs = model(images.to(device))
            if isinstance(outputs, Mapping):
                outputs = outputs.get("out")
            if not isinstance(outputs, torch.Tensor) or outputs.ndim != 4:
                shape = (
                    tuple(outputs.shape)
                    if isinstance(outputs, torch.Tensor)
                    else type(outputs)
                )
                raise ValueError(
                    "Segmentation output must have shape [batch, classes, height, width], "
                    f"got {shape}."
                )
            if outputs.shape[0] != len(targets):
                raise ValueError(
                    f"Segmentation output batch size {outputs.shape[0]} does not "
                    f"match target batch size {len(targets)}."
                )
            predictions = outputs.argmax(dim=1)
            target_batch = torch.stack(
                [torch.as_tensor(target, dtype=torch.long) for target in targets]
            ).to(device)
            if predictions.shape[-2:] != target_batch.shape[-2:]:
                predictions = torch.nn.functional.interpolate(
                    predictions[:, None].float(),
                    size=target_batch.shape[-2:],
                    mode="nearest",
                )[:, 0].long()

            classes = outputs.shape[1]
            valid = target_batch != 255
            if torch.any((target_batch[valid] < 0) | (target_batch[valid] >= classes)):
                raise ValueError(
                    f"Segmentation target contains a class outside the model's {classes} outputs."
                )
            if confusion is None:
                confusion = MulticlassConfusionMatrix(
                    num_classes=classes, ignore_index=255
                ).to(device)
                jaccard = MulticlassJaccardIndex(
                    num_classes=classes,
                    average="none",
                    ignore_index=255,
                    zero_division=0,
                ).to(device)
            confusion.update(predictions, target_batch)
            jaccard.update(predictions, target_batch)
            progress.update(len(targets))

    if confusion is None or jaccard is None:
        raise ValueError("The segmentation evaluation split is empty.")
    matrix = confusion.compute()
    present = (matrix.sum(dim=0) + matrix.sum(dim=1)) > 0
    if not present.any():
        raise ValueError(
            "The segmentation evaluation split contains no labeled pixels."
        )
    per_class_iou = jaccard.compute()
    return EvaluationResult(
        task="segmentation",
        num_samples=len(dataset),
        mean_iou=float(per_class_iou[present].mean().item()),
    )


def _coco_map(
    model: torch.nn.Module,
    dataset: Any,
    batch_size: int,
    device: torch.device,
) -> EvaluationResult:
    if not hasattr(dataset, "coco") or not hasattr(dataset, "ids"):
        raise ValueError(
            "Detection evaluation requires a COCO dataset with annotations."
        )
    try:
        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox").to(device)
    except (
        ImportError,
        ModuleNotFoundError,
    ) as exc:  # pragma: no cover - optional COCO backend
        raise ImportError(
            "COCO mAP requires pycocotools. Install it with "
            "`pip install cks-optimizer[coco]`."
        ) from exc

    image_ids = [int(image_id) for image_id in dataset.ids]
    valid_category_ids = {int(category_id) for category_id in dataset.coco.getCatIds()}
    with tqdm(
        total=len(image_ids), desc="Detection evaluation", unit="image"
    ) as progress:
        for start in range(0, len(image_ids), batch_size):
            batch_ids = image_ids[start : start + batch_size]
            samples = [
                dataset[index]
                for index in range(start, min(start + batch_size, len(image_ids)))
            ]
            images = torch.stack([sample[0] for sample in samples]).to(device)
            torchvision_model = all(
                hasattr(model, name) for name in ("transform", "rpn", "roi_heads")
            )
            outputs = model(list(images)) if torchvision_model else model(images)
            predictions, targets = [], []
            for index, (image_id, (_, annotations)) in enumerate(
                zip(batch_ids, samples, strict=True)
            ):
                original = dataset.coco.imgs[image_id]
                predictions.append(
                    detection_prediction(
                        outputs,
                        index,
                        len(samples),
                        valid_category_ids,
                        (images.shape[-2], images.shape[-1]),
                        (original["height"], original["width"]),
                        device,
                    )
                )
                boxes, labels = [], []
                for annotation in annotations or []:
                    x, y, width, height = annotation["bbox"]
                    boxes.append((x, y, x + width, y + height))
                    labels.append(annotation["category_id"])
                targets.append(
                    {
                        "boxes": torch.tensor(
                            boxes, dtype=torch.float32, device=device
                        ).reshape(-1, 4),
                        "labels": torch.tensor(labels, dtype=torch.long, device=device),
                    }
                )
            metric.update(predictions, targets)
            progress.update(len(samples))

    values = metric.compute()
    return EvaluationResult(
        task="detection",
        num_samples=len(image_ids),
        mean_average_precision=float(values["map"].item()),
        map_50=float(values["map_50"].item()),
        map_75=float(values["map_75"].item()),
    )


def evaluate(
    model: torch.nn.Module,
    dataset: Any,
    *,
    task: str | None = None,
    config: OptimizeConfig | None = None,
    enabled: bool = True,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
) -> EvaluationResult | None:
    """Evaluate a model against a held-out classification, detection, or segmentation set.

    Classification reports top-1 accuracy, detection reports COCO bounding-box
    mAP, and segmentation reports mean IoU. Evaluation preserves the model's
    original training/evaluation state.

    Args:
        model: PyTorch model to evaluate.
        dataset: Non-empty held-out dataset returning image/target pairs.
        task: Task name; defaults to ``config.task`` when config is provided.
        config: Optional configuration for task support and evaluation batch size.
        enabled: If false, skip evaluation and return ``None``.
        device: Device for model execution and metric accumulation. If omitted,
            use the model's current device.
        batch_size: Evaluation batch size. Defaults to the config value or 16.

    Returns:
        An :class:`EvaluationResult`, or ``None`` if evaluation is disabled or
        unsupported by the configured quantization mode.

    Raises:
        TypeError: If ``model`` is not a PyTorch module.
        ValueError: If the task is unsupported, the dataset is empty, or model
            outputs do not match the requested task.
        ImportError: If COCO mAP is requested but ``pycocotools`` is unavailable.
    """
    if not enabled:
        return None
    if config is not None and not config.evaluation_supported:
        quant_mode = config.quant.mode if config.quant is not None else "unknown"
        logger.info(
            "Skipping evaluation: quantization mode %r is not supported by this "
            "evaluation pipeline.",
            quant_mode,
        )
        return None
    task = task or (config.task if config is not None else None)
    if task not in {"classification", "detection", "segmentation"}:
        raise ValueError(f"Unsupported evaluation task: {task!r}.")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Evaluation requires a torch.nn.Module.")
    if dataset is None or len(dataset) == 0:
        raise ValueError("A non-empty held-out dataset is required for evaluation.")

    if device is not None and str(device).strip():
        device = torch.device(device)
    else:
        parameter = next(model.parameters(), None)
        buffer = next(model.buffers(), None)
        device = (
            parameter.device
            if parameter is not None
            else (buffer.device if buffer is not None else torch.device("cpu"))
        )
    if batch_size is None:
        batch_size = config.evaluation_batch_size if config else 16
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    input_shape = CNNModelMetadata.for_model(model).input_shape
    if input_shape and isinstance(input_shape[0], int):
        batch_size = min(batch_size, input_shape[0])
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            if task == "classification":
                return _classification_accuracy(model, dataset, batch_size, device)
            if task == "detection":
                return _coco_map(model, dataset, batch_size, device)
            return _segmentation_miou(model, dataset, batch_size, device)
    finally:
        model.train(was_training)
