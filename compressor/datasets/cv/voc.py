from __future__ import annotations

from typing import Any

from compressor.datasets.cv.base import (
    SUPPORTED_IMAGE_EXTENSIONS,
    VisionDatasetAdapter,
)


class VocDatasetAdapter(VisionDatasetAdapter):
    """VOC dataset adapter for detection / segmentation tasks."""

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
                    annotation_path = annotation_root / f"{image_path.stem}.xml"
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
