from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _ensure_llmcompressor_compat() -> None:
    """Patch compressed_tensors API drift expected by the local llmcompressor.

    The local llm-compressor source imports `exec_jobs_dynamic` from
    `compressed_tensors.entrypoints.convert`; newer releases expose the renamed
    `exec_jobs` instead. We add the historical alias so the local source can be
    imported without failing as soon as the project loads the adapter.
    """
    try:
        import compressed_tensors.entrypoints.convert as convert_module
    except Exception:  # pragma: no cover
        return

    if not hasattr(convert_module, "exec_jobs_dynamic") and hasattr(
        convert_module, "exec_jobs"
    ):
        convert_module.exec_jobs_dynamic = convert_module.exec_jobs

    try:
        import compressed_tensors.entrypoints as entrypoints_module
    except Exception:  # pragma: no cover
        return

    try:
        convert = entrypoints_module.convert
    except AttributeError:  # pragma: no cover
        return

    if not hasattr(convert, "exec_jobs_dynamic") and hasattr(convert, "exec_jobs"):
        convert.exec_jobs_dynamic = convert.exec_jobs


_ensure_llmcompressor_compat()


class LLMCompressorAdapter:
    """Minimal adapter around llmcompressor for runtime validation.

    The adapter is intentionally small and focuses on a real smoke-test
    path for the local llm-compressor stack in the cks-opt environment.
    """

    def __init__(self, config: Any | None = None):
        self.config = config or {}

    def is_available(self) -> bool:
        try:
            import llmcompressor  # noqa: F401
        except Exception:
            return False
        return True

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
            "LLM model must be a valid local path or Hugging Face repo ID; "
            f"'{candidate}' is not a valid model identifier."
        )

    def compress(
        self,
        model: str,
        output_dir: str | Path,
        recipe: str | None = None,
        dataset: str | None = None,
        dataset_config_name: str | None = None,
        splits: str | None = None,
        num_calibration_samples: int = 8,
        max_seq_length: int = 128,
        trust_remote_code_model: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run one-shot LLM compression if llmcompressor is usable."""
        self._validate_model_id(model)

        if not self.is_available():
            raise RuntimeError(
                "llmcompressor is installed in the environment but is not "
                "compatible with the active compressed_tensors stack. "
                "Please fix the package versions before running LLM "
                "compression."
            )

        try:
            from llmcompressor import Oneshot
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "llmcompressor import failed. Please verify the environment "
                "matches the installed llm-compressor sources."
            ) from exc

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if isinstance(recipe, str) and recipe.strip():
            recipe_value = recipe.strip()
        else:
            recipe_value = None

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
            "group": "llm",
            "model": model,
            "recipe": recipe_value or "",
            "output_dir": str(output_path),
            "dataset": dataset_value,
            "status": "completed",
        }
