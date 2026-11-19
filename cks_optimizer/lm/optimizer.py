"""Large-model optimization backed by llmcompressor."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["LMOptimizer"]

# The vendored llm-compressor source imports ``exec_jobs_dynamic`` from
# ``compressed_tensors.entrypoints.convert``; newer compressed_tensors releases
# expose the renamed ``exec_jobs`` instead. Patching lazily (instead of at
# import time) keeps importing this module side-effect free.
_COMPAT_MODULES = (
    "compressed_tensors.entrypoints.convert",
    "compressed_tensors.entrypoints",
)


def _ensure_llmcompressor_compat() -> None:
    """Add the historical ``exec_jobs_dynamic`` alias expected by llmcompressor."""
    for module_name in _COMPAT_MODULES:
        try:
            module = __import__(module_name, fromlist=["*"])
        except Exception:  # pragma: no cover - optional dependency
            continue
        if not hasattr(module, "exec_jobs_dynamic") and hasattr(module, "exec_jobs"):
            module.exec_jobs_dynamic = module.exec_jobs  # type: ignore[attr-defined]
            logger.debug("Aliased %s.exec_jobs_dynamic -> exec_jobs", module_name)


class LMOptimizer:
    """Large-model optimizer backed by llmcompressor.

    The optimizer is intentionally small and focuses on a real smoke-test
    path for the local llm-compressor stack in the cks-opt environment.
    """

    def __init__(self, config: Any | None = None):
        self.config = config or {}

    def is_available(self) -> bool:
        """Return whether ``llmcompressor`` imports successfully here."""
        try:
            import llmcompressor  # noqa: F401
        except Exception as exc:
            logger.debug("llmcompressor is unavailable: %s", exc)
            return False
        return True

    def unavailable_reason(self) -> str:
        """Return the import error explaining why the backend is unavailable.

        Returns:
            An empty string if ``llmcompressor`` is available, otherwise the
            exception type and message from its failed import.
        """
        try:
            import llmcompressor  # noqa: F401
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        return ""

    @staticmethod
    def _validate_model_id(model: str) -> None:
        if not model or not model.strip():
            raise ValueError("Model name/path cannot be empty.")

        candidate = model.strip()
        if os.path.exists(candidate):
            return

        if candidate.startswith(("http://", "https://")):
            return

        if "/" in candidate:
            return

        raise ValueError(
            "Model must be a valid local path or Hugging Face repo ID; "
            f"'{candidate}' is not a valid model identifier."
        )

    def optimize(
        self,
        model: str,
        output_dir: str | Path,
        recipe: str | None = None,
        dataset: str | None = None,
        dataset_config_name: str | None = None,
        splits: str | None = None,
        num_calibration_samples: int = 8,
        max_seq_length: int = 128,
        trust_remote_code_model: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Optimize a large model with the llm-compressor one-shot workflow.

        Args:
            model: Local model path or Hugging Face model/repository identifier.
            output_dir: Directory where compressed model artifacts are written.
            recipe: Optional llm-compressor recipe path or recipe text.
            dataset: Calibration dataset name or local dataset path.
            dataset_config_name: Optional dataset configuration.
            splits: Dataset split selector, such as ``"train[:128]"``.
            num_calibration_samples: Maximum calibration examples to consume.
            max_seq_length: Maximum token sequence length.
            trust_remote_code_model: Whether to allow custom model code from a
                remote repository.
            **kwargs: Additional arguments forwarded to llm-compressor's
                ``Oneshot`` constructor.

        Returns:
            A result mapping with the model, output directory, dataset, and
            completion status.

        Raises:
            ValueError: If ``model`` is not a local path, URL, or repository ID.
            RuntimeError: If the llm-compressor backend cannot be imported or
                used in the current environment.
        """
        self._validate_model_id(model)

        if not self.is_available():
            raise RuntimeError(
                "llmcompressor is not usable in this environment "
                f"({self.unavailable_reason()}). Install the vendored backend "
                "with 'pip install -e thirdparty/llm-compressor'."
            )

        _ensure_llmcompressor_compat()

        try:
            from llmcompressor import Oneshot
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "llmcompressor import failed. Please verify the environment "
                "matches the installed llm-compressor sources."
            ) from exc

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        recipe_value = (recipe or "").strip() or None

        dataset_value = dataset or "wikitext"
        dataset_cfg = dataset_config_name or "wikitext-2-raw-v1"
        splits_value = splits or "train[:1]"

        oneshot = Oneshot(
            model=model,
            recipe=recipe_value,
            dataset=dataset_value,
            dataset_config_name=dataset_cfg,
            splits=splits_value,
            output_dir=str(output_path),
            num_calibration_samples=num_calibration_samples,
            max_seq_length=max_seq_length,
            trust_remote_code_model=trust_remote_code_model,
            **kwargs,
        )
        oneshot()

        return {
            "group": "lm",
            "model": model,
            "recipe": recipe_value or "",
            "output_dir": str(output_path),
            "dataset": dataset_value,
            "status": "completed",
        }
