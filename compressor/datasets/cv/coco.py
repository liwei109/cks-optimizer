from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from compressor.datasets.cv.base import VisionDatasetAdapter


class CocoDatasetAdapter(VisionDatasetAdapter):
    """COCO dataset adapter for detection / segmentation tasks."""

    def _load_samples(self) -> list[dict[str, Any]]:
        root = self._resolve_split_root()
        annotation_candidates: list[Path] = []
        for candidate in sorted(root.rglob("*.json")):
            if "instances" in candidate.name.lower():
                annotation_candidates.append(candidate)
        if not annotation_candidates:
            annotation_candidates = sorted(root.rglob("*.json"))

        if annotation_candidates:
            annotation_file = annotation_candidates[0]
            try:
                payload = json.loads(annotation_file.read_text())
            except json.JSONDecodeError:
                return []

            images = payload.get("images", [])
            if not images:
                return []

            samples: list[dict[str, Any]] = []
            for item in images:
                image_name = item.get("file_name")
                if not image_name:
                    continue
                image_path = (root / image_name).resolve()
                if image_path.exists():
                    samples.append(
                        {
                            "image_path": str(image_path),
                            "annotation_path": str(annotation_file),
                            "image_id": item.get("id"),
                            "annotations": payload.get("annotations", []),
                        }
                    )
            return samples

        return [
            {
                "image_path": str(path),
                "annotation_path": None,
            }
            for path in self._iter_image_files(root)
        ]
