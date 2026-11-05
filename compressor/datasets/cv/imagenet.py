from __future__ import annotations

from typing import Any

from compressor.datasets.cv.base import VisionDatasetAdapter


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


class ImageNetDatasetAdapter(ImageFolderDatasetAdapter):
    """ImageNet dataset adapter for classification tasks."""

    default_mean = IMAGENET_MEAN
    default_std = IMAGENET_STD
