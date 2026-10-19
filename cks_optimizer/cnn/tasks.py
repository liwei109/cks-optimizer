from __future__ import annotations

from typing import Any


TASK_REGISTRY = {
    "classification": "classification",
    "detection": "detection",
    "segmentation": "segmentation",
}


def get_task_name(task: str) -> str:
    """Normalize a task name."""
    task_name = task.lower().strip()
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unsupported task: {task}")
    return TASK_REGISTRY[task_name]
