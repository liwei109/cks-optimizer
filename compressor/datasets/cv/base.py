from __future__ import annotations

import csv
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator

import torch

logger = logging.getLogger(__name__)

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


class CustomDatasetAdapter(VisionDatasetAdapter):
    """Custom dataset adapter for user-specific formats."""

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
                    return self._from_manifest_items(payload)
                if isinstance(payload, dict):
                    items = payload.get("items", [])
                    if isinstance(items, list):
                        return self._from_manifest_items(items)

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

        return [
            {
                "image_path": str(path),
                "label": self.kwargs.get("label", "custom"),
            }
            for path in self._iter_image_files(root)
        ]

    @staticmethod
    def _from_manifest_items(items: list[Any]) -> list[dict[str, Any]]:
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


def load_image_tensor(
    image_path: str | Path,
    image_size: int | tuple[int, int] | None = None,
    *,
    mean: list[float] | tuple[float, ...] | None = None,
    std: list[float] | tuple[float, ...] | None = None,
) -> torch.Tensor:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Image loading requires Pillow; please install pillow first"
        ) from exc

    img = Image.open(image_path).convert("RGB")
    if image_size is not None:
        height, width = (
            (int(image_size), int(image_size))
            if isinstance(image_size, int)
            else (int(image_size[0]), int(image_size[1]))
        )
        img = img.resize((width, height), Image.BILINEAR)

    tensor = torch.from_numpy(
        __import__("numpy").asarray(img, dtype="float32").copy()
    ).permute(2, 0, 1) / 255.0

    if mean is not None and std is not None:
        mean_t = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        tensor = (tensor - mean_t) / std_t
    return tensor


def iter_cv_batches(
    dataset: VisionDatasetAdapter,
    *,
    batch_size: int = 1,
    image_size: int | tuple[int, int] | None = None,
    mean: list[float] | tuple[float, ...] | None = None,
    std: list[float] | tuple[float, ...] | None = None,
    progress: str | None = None,
) -> Iterator[torch.Tensor]:
    """Yield preprocessed image batches from a CV dataset adapter.

    :param progress: when set, show a tqdm progress bar (image decode +
        preprocess) with this description; tqdm is optional at runtime.
    """
    iterator = iter(dataset)
    if progress is not None:
        try:
            from tqdm import tqdm

            iterator = tqdm(
                iterator, total=len(dataset), desc=progress, unit="img"
            )
        except ImportError:  # tqdm not installed — iterate silently
            pass

    buffer: list[torch.Tensor] = []
    for sample in iterator:
        path = sample.get("image_path")
        if not path:
            continue
        try:
            buffer.append(
                load_image_tensor(path, image_size, mean=mean, std=std)
            )
        except Exception as exc:
            logger.warning("Skipping unreadable image '%s': %s", path, exc)
            continue
        if len(buffer) >= batch_size:
            yield torch.stack(buffer[:batch_size])
            buffer = buffer[batch_size:]
    if buffer:
        yield torch.stack(buffer)
