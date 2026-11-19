"""Detection output adapters used by task evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

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


def yolov5_outputs(
    raw: np.ndarray,
    valid_category_ids: set[int],
    input_size: tuple[int, int],
    original_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one YOLOv5 [1, boxes, 5 + classes] output into COCO detections."""
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
    candidates, scores, class_ids = candidates[valid], scores[valid], class_ids[valid]
    if not len(candidates):
        return np.empty((0, 4)), np.empty((0,), dtype=np.int64), np.empty((0,))

    centers = candidates[:, :2]
    half_sizes = candidates[:, 2:4] / 2
    boxes = np.concatenate((centers - half_sizes, centers + half_sizes), axis=1)
    finite_boxes = np.isfinite(boxes).all(axis=1)
    boxes, scores, class_ids = boxes[finite_boxes], scores[finite_boxes], class_ids[finite_boxes]
    valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores, class_ids = boxes[valid_boxes], scores[valid_boxes], class_ids[valid_boxes]

    kept: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        kept.extend(indices[index] for index in _nms_indices(boxes[indices], scores[indices], 0.6))
    kept = sorted(kept, key=lambda index: float(scores[index]), reverse=True)[:300]

    boxes = boxes[kept]
    original_height, original_width = original_size
    ratio = min(input_size[1] / original_width, input_size[0] / original_height)
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


def detection_prediction(
    outputs: Any,
    index: int,
    batch_size: int,
    valid_category_ids: set[int],
    input_size: tuple[int, int],
    original_size: tuple[int, int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Normalize supported model detection outputs to TorchMetrics format."""
    raw_yolo = (
        isinstance(outputs, torch.Tensor)
        and outputs.ndim == 3
        and outputs.shape[-1] == 85
    )
    if raw_yolo:
        boxes, labels, scores = yolov5_outputs(
            outputs[index : index + 1].detach().float().cpu().numpy(),
            valid_category_ids,
            input_size,
            original_size,
        )
        return {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32, device=device),
            "labels": torch.as_tensor(labels, dtype=torch.long, device=device),
            "scores": torch.as_tensor(scores, dtype=torch.float32, device=device),
        }

    if isinstance(outputs, (list, tuple)) and len(outputs) == batch_size and all(
        isinstance(output, Mapping) for output in outputs
    ):
        output = outputs[index]
        boxes, labels, scores = output["boxes"], output["labels"], output["scores"]
    elif isinstance(outputs, Mapping):
        boxes, labels, scores = (
            outputs.get("boxes"),
            outputs.get("labels"),
            outputs.get("scores"),
        )
        if boxes is None or labels is None or scores is None:
            raise ValueError("Detection output must include boxes, labels, and scores.")
        if boxes.ndim >= 3 and boxes.shape[0] == batch_size:
            boxes, labels, scores = boxes[index], labels[index], scores[index]
    elif isinstance(outputs, (tuple, list)) and len(outputs) == 3:
        boxes, labels, scores = outputs
        if boxes.ndim >= 3 and boxes.shape[0] == batch_size:
            boxes, labels, scores = boxes[index], labels[index], scores[index]
    else:
        raise ValueError(
            "Detection output must be a YOLOv5 raw tensor, a (boxes, labels, scores) "
            "tuple, a matching mapping, or a list of torchvision prediction mappings."
        )

    boxes = torch.as_tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4)
    labels = torch.as_tensor(labels, dtype=torch.long, device=device).reshape(-1)
    scores = torch.as_tensor(scores, dtype=torch.float32, device=device).reshape(-1)
    if len(boxes) != len(labels) or len(labels) != len(scores):
        raise ValueError("Detection boxes, labels, and scores must have equal lengths.")

    original_height, original_width = original_size
    input_height, input_width = input_size
    scale = boxes.new_tensor(
        (
            original_width / input_width,
            original_height / input_height,
            original_width / input_width,
            original_height / input_height,
        )
    )
    return {"boxes": boxes * scale, "labels": labels, "scores": scores}
