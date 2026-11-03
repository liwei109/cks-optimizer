from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator


SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".webp",
}


class VisionDatasetAdapter(ABC):
    """Abstract adapter for CNN vision datasets."""

    @classmethod
    def matches(cls, root: Path) -> bool:
        """Whether the directory layout matches this dataset kind.

        Used by kind inference: the first registered adapter whose
        ``matches`` returns True wins, so specific layouts must be
        checked before generic ones.
        """
        return False

    def __init__(self, root: str, split: str = "train", **kwargs: Any):
        self.root = Path(root)
        self.split = split
        self.kwargs = kwargs
        self.samples = self._load_samples()

    def _resolve_split_root(self) -> Path:
        split_root = self.root / self.split
        if split_root.exists() and split_root.is_dir():
            return split_root
        return self.root

    def _iter_image_files(self, directory: Path | None = None) -> list[Path]:
        target = directory or self._resolve_split_root()
        files: list[Path] = []
        if target.exists() and target.is_dir():
            for child in sorted(target.rglob("*")):
                if (
                    child.is_file()
                    and child.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                ):
                    files.append(child)
        return files

    @abstractmethod
    def _load_samples(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


class ImageFolderDatasetAdapter(VisionDatasetAdapter):
    """Generic ImageFolder-style dataset for classification tasks."""

    def _load_samples(self) -> list[dict[str, Any]]:
        files: list[dict[str, Any]] = []
        target = self._resolve_split_root()

        if target.exists() and target.is_dir():
            class_dirs = sorted(
                [child for child in target.iterdir() if child.is_dir()]
            )
            if class_dirs:
                for class_dir in class_dirs:
                    for image_path in self._iter_image_files(class_dir):
                        files.append(
                            {
                                "image_path": str(image_path),
                                "label": class_dir.name,
                                "class_id": class_dir.name,
                            }
                        )
                return files

        for image_path in self._iter_image_files(target):
            label = self.kwargs.get("label", "unknown")
            files.append(
                {
                    "image_path": str(image_path),
                    "label": label,
                    "class_id": label,
                }
            )
        return files


class CocoDatasetAdapter(VisionDatasetAdapter):
    """COCO dataset adapter for detection / segmentation tasks."""

    @classmethod
    def matches(cls, root: Path) -> bool:
        return any(
            "instances" in candidate.name.lower()
            for candidate in root.rglob("*.json")
        )

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


class VocDatasetAdapter(VisionDatasetAdapter):
    """VOC dataset adapter for detection / segmentation tasks."""

    @classmethod
    def matches(cls, root: Path) -> bool:
        return (root / "JPEGImages").exists() and (root / "Annotations").exists()

    def _load_samples(self) -> list[dict[str, Any]]:
        target = self._resolve_split_root()
        jpeg_root = target / "JPEGImages"
        annotation_root = target / "Annotations"

        if jpeg_root.exists() and jpeg_root.is_dir():
            files: list[dict[str, Any]] = []
            for image_path in sorted(jpeg_root.iterdir()):
                if (
                    image_path.is_file()
                    and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                ):
                    annotation_path = annotation_root / (
                        f"{image_path.stem}.xml"
                    )
                    files.append(
                        {
                            "image_path": str(image_path),
                            "annotation_path": str(annotation_path),
                            "label": self.kwargs.get("label", "object"),
                        }
                    )
            if files:
                return files

        return [
            {
                "image_path": str(path),
                "annotation_path": None,
                "label": self.kwargs.get("label", "object"),
            }
            for path in self._iter_image_files(target)
        ]


class CustomDatasetAdapter(VisionDatasetAdapter):
    """Custom dataset adapter for user-specific formats."""

    @classmethod
    def matches(cls, root: Path) -> bool:
        return any(root.rglob("*.json"))

    def _load_samples(self) -> list[dict[str, Any]]:
        root = self._resolve_split_root()
        manifest_candidates = [
            root / "manifest.json",
            root / "data.json",
            root / "dataset.json",
        ]

        for manifest in manifest_candidates:
            if manifest.exists() and manifest.is_file():
                try:
                    payload = json.loads(manifest.read_text())
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, list):
                    return [
                        {
                            "image_path": str(
                                item.get("image_path")
                                or item.get("file_path")
                                or item.get("path")
                            ),
                            "label": item.get("label", "custom"),
                            "metadata": item,
                        }
                        for item in payload
                    ]
                if isinstance(payload, dict):
                    items = payload.get("items", [])
                    if isinstance(items, list):
                        return [
                            {
                                "image_path": str(
                                    item.get("image_path")
                                    or item.get("file_path")
                                    or item.get("path")
                                ),
                                "label": item.get("label", "custom"),
                                "metadata": item,
                            }
                            for item in items
                        ]

        csv_path = root / "manifest.csv"
        if csv_path.exists() and csv_path.is_file():
            rows: list[dict[str, Any]] = []
            with csv_path.open("r", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    rows.append(
                        {
                            "image_path": (
                                row.get("image_path")
                                or row.get("file_path")
                                or row.get("path")
                            ),
                            "label": row.get("label", "custom"),
                            "metadata": row,
                        }
                    )
            return rows

        file_items = [
            {
                "image_path": str(path),
                "label": self.kwargs.get("label", "custom"),
            }
            for path in self._iter_image_files(root)
        ]
        return file_items
