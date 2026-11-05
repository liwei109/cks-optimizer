"""llmcompressor adapter: LmCompressor interface implementation.
"""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import Any

from compressor.adapter.contract import (
    LmCompressRequest,
    LmCompressResult,
    Technique,
)

logger = logging.getLogger(__name__)


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


def _import_class(candidates: tuple[str, ...], what: str):
    for path in candidates:
        module_path, _, name = path.rpartition(".")
        try:
            module = importlib.import_module(module_path)
            return getattr(module, name)
        except (ImportError, AttributeError):
            continue
    raise ImportError(
        f"Failed to import {what}; tried: {candidates}. "
        "Please check llmcompressor version or explicitly pass recipe."
    )


class LLMCompressorAdapter:
    """llmcompressor adapter: LLM/VLM scene compression."""

    def __init__(self) -> None:
        self._installed = False

    def supported_techniques(self) -> frozenset[Technique]:
        # PRUNE: llmcompressor pruning capability pending confirmation
        return frozenset({Technique.QUANTIZE, Technique.SPARSIFY})

    def is_available(self) -> bool:
        try:
            import llmcompressor  # noqa: F401
        except Exception:
            return False
        return True

    def install(self) -> None:
        if self._installed:
            return

        if not self.is_available():
            raise ImportError(
                "llmcompressor is not available. "
                "Please install llmcompressor in the current environment."
            )

        _ensure_llmcompressor_compat()

        from compressor.algorithms.lm.gptq import install_tu_gptq_backend
        from compressor.observer.mse import install_tu_observer

        install_tu_observer()
        install_tu_gptq_backend()
        self._installed = True

 
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

    def _default_quantize_recipe(self, request: LmCompressRequest):
        gptq_cls = _import_class(
            (
                "llmcompressor.modifiers.quantization.gptq.GPTQModifier",
                "llmcompressor.modifiers.gptq.GPTQModifier",
            ),
            "GPTQModifier",
        )
        return gptq_cls(
            scheme=request.options.get("scheme", "W4A16"),
            targets="Linear",
            ignore=["lm_head"],
        )

    def _default_sparsify_recipe(self, request: LmCompressRequest):
        sparse_cls = _import_class(
            (
                "llmcompressor.modifiers.sparsify.SparseGPTModifier",
                "llmcompressor.modifiers.sparsify.sparsify.SparseGPTModifier",
            ),
            "SparseGPTModifier",
        )
        return sparse_cls(
            sparsity=request.options.get("sparsity", 0.5),
            mask_structure="2:4",
        )

    def _resolve_recipe(self, request: LmCompressRequest, technique: Technique):
        if request.recipe is not None:
            if isinstance(request.recipe, str):
                stripped = request.recipe.strip()
                return stripped if stripped else None
            return request.recipe
        if technique is Technique.SPARSIFY:
            return self._default_sparsify_recipe(request)
        return self._default_quantize_recipe(request)

    def _run_oneshot(
        self, request: LmCompressRequest, technique: Technique
    ) -> LmCompressResult:
        """quantize / sparsify common path: Oneshot + recipe。"""
        self.install()
        self._validate_model_id(request.model)

        try:
            from llmcompressor import Oneshot
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "llmcompressor import failed. Please verify the environment "
                "matches the installed llm-compressor sources."
            ) from exc

        recipe = self._resolve_recipe(request, technique)
        oneshot_kwargs = dict(request.options)
        if request.recipe is None:
            for consumed in (("scheme",) if technique is Technique.QUANTIZE
                             else ("sparsity",)):
                oneshot_kwargs.pop(consumed, None)
        output_path = Path(request.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        from compressor.datasets.lm.hub_cache import resolve_hf_dataset_revision
        dataset_value = request.dataset
        force_offline = False
        if isinstance(dataset_value, str):
            resolved = resolve_hf_dataset_revision(
                dataset_value, request.dataset_config_name
            )
            if resolved is not None:
                canonical, revision = resolved
                logger.info(
                    "dataset '%s' local cache hit(%s@%s), offline loading",
                    dataset_value, canonical, revision[:12],
                )
                raw_kwargs = dict(oneshot_kwargs.get("raw_kwargs") or {})
                raw_kwargs.setdefault("revision", revision)
                oneshot_kwargs["raw_kwargs"] = raw_kwargs
                force_offline = bool(
                    isinstance(request.model, str) and os.path.exists(request.model)
                )

        oneshot = Oneshot(
            model=request.model,
            recipe=recipe,
            dataset=dataset_value,
            dataset_config_name=request.dataset_config_name,
            splits=request.splits,
            output_dir=str(output_path),
            num_calibration_samples=request.num_calibration_samples,
            max_seq_length=request.max_seq_length,
            trust_remote_code_model=request.trust_remote_code_model,
            **oneshot_kwargs,
        )
        if force_offline:
            from compressor.datasets.lm.hub_cache import set_hub_offline
            restore_offline = set_hub_offline(True)
            try:
                oneshot()
            finally:
                restore_offline()
        else:
            oneshot()

        return LmCompressResult(
            output_dir=output_path,
            model=request.model,
            recipe=recipe,
            metadata={
                "group": "llm",
                "technique": technique.value,
                "status": "completed",
                "dataset": request.dataset,
            },
        )


    def quantize(self, request: LmCompressRequest) -> LmCompressResult:
        return self._run_oneshot(request, Technique.QUANTIZE)

    def sparsify(self, request: LmCompressRequest) -> LmCompressResult:
        return self._run_oneshot(request, Technique.SPARSIFY)

    def prune(self, request: LmCompressRequest) -> LmCompressResult:
        raise NotImplementedError("TODO: prune capability placeholder")
