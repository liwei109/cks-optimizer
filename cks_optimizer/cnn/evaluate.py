"""Ground-truth evaluation for classification, detection, and segmentation."""

from __future__ import annotations

import io
import logging
from collections.abc import Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cks_optimizer.cnn.config import OptimizeConfig

logger = logging.getLogger(__name__)

__all__ = ["EvaluationResult", "evaluate", "resolve_input_name"]


@dataclass
class EvaluationResult:
    """Task metric values measured against dataset annotations."""

    task: str
    num_samples: int
    accuracy: float | None = None
    mean_average_precision: float | None = None
    map_50: float | None = None
    map_75: float | None = None
    mean_iou: float | None = None

    def summary(self) -> str:
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
    """Return a real input name from an ONNX graph."""
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover
        raise ImportError("onnx is required for ONNX evaluation.") from exc

    model = onnx.load(str(onnx_path))
    initializers = {initializer.name for initializer in model.graph.initializer}
    names = [item.name for item in model.graph.input if item.name not in initializers]
    if not names:
        raise ValueError(f"No usable graph input found in {onnx_path}.")
    if preferred is not None and preferred in names:
        return preferred
    return names[0]


def _create_session(onnx_path: str | Path, providers: Sequence[str] | None) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise ImportError("onnxruntime is required for ONNX evaluation.") from exc

    available = ort.get_available_providers()
    selected = [provider for provider in (providers or available) if provider in available]
    if not selected:
        logger.warning("Requested ONNX Runtime providers are unavailable; using CPU.")
        selected = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(onnx_path), providers=selected)


def _session_input_name(session: Any, requested: str | None) -> str:
    names = [item.name for item in session.get_inputs()]
    if not names:
        raise ValueError("The ONNX model has no inputs.")
    if requested and requested in names:
        return requested
    if requested:
        logger.warning("ONNX input %r not found; using %r.", requested, names[0])
    return names[0]


def _batch_size(session: Any) -> int:
    dimension = session.get_inputs()[0].shape[0]
    return 1 if isinstance(dimension, int) else 16


def _batches(dataset: Any, batch_size: int):
    for start in range(0, len(dataset), batch_size):
        samples = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        images = torch.stack([sample[0] for sample in samples])
        targets = [sample[1] for sample in samples]
        yield images, targets


def _classification_accuracy(session: Any, input_name: str, dataset: Any) -> EvaluationResult:
    correct = 0
    for images, targets in _batches(dataset, _batch_size(session)):
        outputs = session.run(None, {input_name: images.numpy()})
        logits = np.asarray(outputs[0])
        if logits.ndim != 2 or logits.shape[0] != len(targets):
            raise ValueError(
                f"Classification output must have shape [batch, classes], got {logits.shape}."
            )
        correct += sum(
            int(prediction) == int(target)
            for prediction, target in zip(logits.argmax(axis=1), targets, strict=True)
        )
    return EvaluationResult(
        task="classification",
        num_samples=len(dataset),
        accuracy=correct / len(dataset),
    )


def _segmentation_miou(session: Any, input_name: str, dataset: Any) -> EvaluationResult:
    confusion: torch.Tensor | None = None
    for images, targets in _batches(dataset, _batch_size(session)):
        outputs = session.run(None, {input_name: images.numpy()})
        logits = torch.as_tensor(np.asarray(outputs[0]))
        if logits.ndim != 4 or logits.shape[0] != len(targets):
            raise ValueError(
                f"Segmentation output must have shape [batch, classes, height, width], "
                f"got {tuple(logits.shape)}."
            )
        classes = logits.shape[1]
        predictions = logits.argmax(dim=1).to(torch.int64)
        for prediction, target in zip(predictions, targets, strict=True):
            target = torch.as_tensor(target, dtype=torch.int64)
            if prediction.shape != target.shape:
                prediction = torch.nn.functional.interpolate(
                    prediction[None, None].float(),
                    size=target.shape,
                    mode="nearest",
                )[0, 0].to(torch.int64)

            valid = target != 255
            if torch.any((target[valid] < 0) | (target[valid] >= classes)):
                raise ValueError(
                    f"VOC target contains a class outside the model's {classes} outputs."
                )
            encoded = target[valid] * classes + prediction[valid]
            counts = torch.bincount(encoded, minlength=classes * classes).reshape(classes, classes)
            confusion = counts if confusion is None else confusion + counts

    if confusion is None:
        raise ValueError("The VOC evaluation split is empty.")
    true_positive = confusion.diag()
    union = confusion.sum(dim=0) + confusion.sum(dim=1) - true_positive
    present = union > 0
    if not present.any():
        raise ValueError("The VOC evaluation split contains no labeled pixels.")
    mean_iou = float((true_positive[present] / union[present]).mean())
    return EvaluationResult(task="segmentation", num_samples=len(dataset), mean_iou=mean_iou)


def _named_outputs(session: Any, outputs: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    metadata = session.get_outputs()
    if len(metadata) != len(outputs):
        raise ValueError("ONNX output metadata does not match model output count.")
    return {metadata[index].name.lower(): np.asarray(value) for index, value in enumerate(outputs)}


def _detection_outputs(named: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    aliases = {
        "boxes": ("boxes", "detection_boxes"),
        "labels": ("labels", "detection_classes", "classes"),
        "scores": ("scores", "detection_scores"),
    }
    values = {
        key: next((named[name] for name in names if name in named), None)
        for key, names in aliases.items()
    }
    if any(value is None for value in values.values()):
        raise ValueError("COCO mAP requires ONNX outputs named boxes, labels, and scores.")
    boxes = np.asarray(values["boxes"]).reshape(-1, 4)
    labels = np.asarray(values["labels"]).reshape(-1)
    scores = np.asarray(values["scores"]).reshape(-1)
    if len(boxes) != len(labels) or len(labels) != len(scores):
        raise ValueError("Detection boxes, labels, and scores must have equal lengths.")
    return boxes, labels, scores


def _coco_map(session: Any, input_name: str, dataset: Any) -> EvaluationResult:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "COCO mAP requires pycocotools. Install it with `pip install cks-optimizer[coco]`."
        ) from exc

    predictions: list[dict[str, Any]] = []
    image_ids = [int(image_id) for image_id in dataset.ids]
    for index, image_id in enumerate(image_ids):
        image, _ = dataset[index]
        outputs = session.run(None, {input_name: image.unsqueeze(0).numpy()})
        boxes, labels, scores = _detection_outputs(_named_outputs(session, outputs))
        original = dataset.coco.imgs[image_id]
        width_scale = original["width"] / image.shape[-1]
        height_scale = original["height"] / image.shape[-2]
        for box, label, score in zip(boxes, labels, scores, strict=True):
            x1, y1, x2, y2 = (float(value) for value in box)
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [
                        x1 * width_scale,
                        y1 * height_scale,
                        (x2 - x1) * width_scale,
                        (y2 - y1) * height_scale,
                    ],
                    "score": float(score),
                }
            )

    coco_ground_truth = dataset.coco
    if predictions:
        coco_predictions = coco_ground_truth.loadRes(predictions)
    else:
        coco_predictions = COCO()
        coco_predictions.dataset = {
            "images": coco_ground_truth.dataset["images"],
            "categories": coco_ground_truth.dataset["categories"],
            "annotations": [],
        }
        coco_predictions.createIndex()
    evaluator = COCOeval(coco_ground_truth, coco_predictions, "bbox")
    evaluator.params.imgIds = image_ids
    with redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return EvaluationResult(
        task="detection",
        num_samples=len(image_ids),
        mean_average_precision=float(evaluator.stats[0]),
        map_50=float(evaluator.stats[1]),
        map_75=float(evaluator.stats[2]),
    )


def evaluate(
    onnx_path: str | Path,
    dataset: Any,
    *,
    task: str | None = None,
    config: OptimizeConfig | None = None,
    enabled: bool = True,
    input_name: str | None = None,
    providers: Sequence[str] | None = None,
) -> EvaluationResult | None:
    """Evaluate an ONNX model against ImageNet, COCO, or VOC annotations."""
    if not enabled:
        return None
    if config is not None and not config.evaluation_supported:
        quant_mode = config.quant.mode if config.quant is not None else "unknown"
        logger.info(
            "Skipping evaluation: quantization mode %r emits operators that "
            "ONNX Runtime cannot execute.",
            quant_mode,
        )
        return None
    task = task or (config.task if config is not None else None)
    if task not in {"classification", "detection", "segmentation"}:
        raise ValueError(f"Unsupported evaluation task: {task!r}.")
    if dataset is None or len(dataset) == 0:
        raise ValueError("A non-empty held-out dataset is required for evaluation.")

    session = _create_session(onnx_path, providers)
    input_name = _session_input_name(session, input_name)
    if task == "classification":
        return _classification_accuracy(session, input_name, dataset)
    if task == "detection":
        return _coco_map(session, input_name, dataset)
    return _segmentation_miou(session, input_name, dataset)
