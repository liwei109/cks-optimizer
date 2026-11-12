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


def _batch_size(session: Any, requested: int | None = None) -> int:
    dimension = session.get_inputs()[0].shape[0]
    preferred = requested or 16
    return min(dimension, preferred) if isinstance(dimension, int) else preferred


def _batches(dataset: Any, batch_size: int):
    for start in range(0, len(dataset), batch_size):
        samples = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        images = torch.stack([sample[0] for sample in samples])
        targets = [sample[1] for sample in samples]
        yield images, targets


def _classification_accuracy(
    session: Any,
    input_name: str,
    dataset: Any,
    batch_size: int,
) -> EvaluationResult:
    correct = 0
    for images, targets in _batches(dataset, _batch_size(session, batch_size)):
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


def _segmentation_miou(
    session: Any,
    input_name: str,
    dataset: Any,
    batch_size: int,
) -> EvaluationResult:
    confusion: torch.Tensor | None = None
    for images, targets in _batches(dataset, _batch_size(session, batch_size)):
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


_COCO80_CATEGORY_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
)


def _nms_indices(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Return score-ordered indices after class-wise non-maximum suppression."""
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    remaining = np.argsort(scores)[::-1]
    keep: list[int] = []
    while remaining.size:
        current = int(remaining[0])
        keep.append(current)
        if remaining.size == 1:
            break
        rest = remaining[1:]
        intersection = (
            np.maximum(0, np.minimum(x2[current], x2[rest]) - np.maximum(x1[current], x1[rest]))
            * np.maximum(0, np.minimum(y2[current], y2[rest]) - np.maximum(y1[current], y1[rest]))
        )
        union = areas[current] + areas[rest] - intersection
        iou = intersection / np.maximum(union, np.finfo(np.float32).eps)
        remaining = rest[iou <= threshold]
    return keep


def _yolov5_outputs(
    raw: np.ndarray,
    valid_category_ids: set[int],
    input_size: tuple[int, int],
    original_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode YOLOv5 [batch, boxes, 5 + classes] output into COCO detections."""
    if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[2] != 85:
        raise ValueError(
            "Unsupported raw detection output; expected YOLOv5 COCO shape "
            f"[1, boxes, 85], got {raw.shape}."
        )

    candidates = raw[0]
    class_ids = np.argmax(candidates[:, 5:], axis=1)
    scores = candidates[:, 4] * candidates[np.arange(len(candidates)), 5 + class_ids]
    category_ids = np.asarray(_COCO80_CATEGORY_IDS, dtype=np.int64)[class_ids]
    valid = (
        np.isfinite(scores)
        & (scores >= 0.001)
        & np.isin(category_ids, tuple(valid_category_ids))
    )
    candidates = candidates[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]
    if not len(candidates):
        return np.empty((0, 4)), np.empty((0,), dtype=np.int64), np.empty((0,))

    centers = candidates[:, :2]
    half_sizes = candidates[:, 2:4] / 2
    boxes = np.concatenate((centers - half_sizes, centers + half_sizes), axis=1)
    finite_boxes = np.isfinite(boxes).all(axis=1)
    boxes = boxes[finite_boxes]
    scores = scores[finite_boxes]
    class_ids = class_ids[finite_boxes]
    valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid_boxes]
    scores = scores[valid_boxes]
    class_ids = class_ids[valid_boxes]

    kept: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        kept.extend(indices[index] for index in _nms_indices(boxes[indices], scores[indices], 0.6))
    kept = sorted(kept, key=lambda index: float(scores[index]), reverse=True)[:300]

    boxes = boxes[kept]
    original_height, original_width = original_size
    ratio = min(
        input_size[1] / original_width,
        input_size[0] / original_height,
    )
    resized_width = round(original_width * ratio)
    resized_height = round(original_height * ratio)
    pad_x = (input_size[1] - resized_width) / 2
    pad_y = (input_size[0] - resized_height) / 2
    boxes[:, [0, 2]] = np.clip(
        (boxes[:, [0, 2]] - pad_x) / ratio, 0, original_width
    )
    boxes[:, [1, 3]] = np.clip(
        (boxes[:, [1, 3]] - pad_y) / ratio, 0, original_height
    )
    labels = np.asarray(_COCO80_CATEGORY_IDS, dtype=np.int64)[class_ids[kept]]
    return boxes, labels, scores[kept]


def _detection_outputs(
    named: dict[str, np.ndarray],
    valid_category_ids: set[int],
    input_size: tuple[int, int],
    original_size: tuple[int, int],
) -> tuple[np.ndarray, ...]:
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
        raw_outputs = list(named.values())
        if len(raw_outputs) == 1 and raw_outputs[0].ndim == 3:
            return _yolov5_outputs(
                raw_outputs[0], valid_category_ids, input_size, original_size
            )
        raise ValueError("COCO mAP requires ONNX outputs named boxes, labels, and scores.")
    boxes = np.asarray(values["boxes"]).reshape(-1, 4)
    labels = np.asarray(values["labels"]).reshape(-1)
    scores = np.asarray(values["scores"]).reshape(-1)
    if len(boxes) != len(labels) or len(labels) != len(scores):
        raise ValueError("Detection boxes, labels, and scores must have equal lengths.")
    return boxes, labels, scores


def _coco_map(
    session: Any,
    input_name: str,
    dataset: Any,
    batch_size: int,
) -> EvaluationResult:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "COCO mAP requires pycocotools. Install it with `pip install cks-optimizer[coco]`."
        ) from exc

    predictions: list[dict[str, Any]] = []
    image_ids = [int(image_id) for image_id in dataset.ids]
    valid_category_ids = {int(category_id) for category_id in dataset.coco.getCatIds()}
    batch_size = _batch_size(session, batch_size)
    for start in range(0, len(image_ids), batch_size):
        batch_ids = image_ids[start : start + batch_size]
        samples = [
            dataset[index]
            for index in range(start, min(start + batch_size, len(image_ids)))
        ]
        images = torch.stack([sample[0] for sample in samples])
        outputs = session.run(None, {input_name: images.numpy()})
        named = _named_outputs(session, outputs)
        raw_yolo = (
            len(named) == 1
            and next(iter(named.values())).ndim == 3
            and next(iter(named.values())).shape[-1] == 85
        )
        for offset, (image_id, (image, _)) in enumerate(zip(batch_ids, samples, strict=True)):
            image_outputs = (
                {name: value[offset : offset + 1] for name, value in named.items()}
                if raw_yolo
                else {
                    name: value[offset] if value.ndim > 1 and value.shape[0] == len(samples) else value
                    for name, value in named.items()
                }
            )
            original = dataset.coco.imgs[image_id]
            boxes, labels, scores = _detection_outputs(
                image_outputs,
                valid_category_ids,
                (image.shape[-2], image.shape[-1]),
                (original["height"], original["width"]),
            )
            width_scale = 1.0 if raw_yolo else original["width"] / image.shape[-1]
            height_scale = 1.0 if raw_yolo else original["height"] / image.shape[-2]
            processed = start + offset + 1
            if processed % 100 == 0 or processed == len(image_ids):
                logger.info("COCO evaluation: %d/%d images", processed, len(image_ids))
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
    batch_size: int | None = None,
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
    batch_size = batch_size or (config.evaluation_batch_size if config else 16)
    if task == "classification":
        return _classification_accuracy(session, input_name, dataset, batch_size)
    if task == "detection":
        return _coco_map(session, input_name, dataset, batch_size)
    return _segmentation_miou(session, input_name, dataset, batch_size)
