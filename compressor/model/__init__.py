"""Model layer: a shared registry plus source-aware loaders.

Design rule — no ``cnn`` / ``vit`` / ``llm`` / ``vlm`` sub-packages here.
Instead, every registered model carries metadata (``domain`` / ``modality``
/ sample-input shape) so the registry can answer capability queries without
re-importing any heavy framework.

Dependency rule — ``model`` never imports ``adapter``. External-tool models
(ModelOpt-registered CV backbones, llmcompressor-registered LM families)
push themselves into the registry via :func:`register_model` during
``adapter.install()``; the domain layer only reads back.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from compressor.model.loaders.cv_loader import resolve_cv_model_and_sample_input
from compressor.model.loaders.hf_loader import load_lm_model, looks_like_hf_repo_id
from compressor.model.loaders.local_ckpt import looks_like_torch_checkpoint
from compressor.utils.registry import Registry

__all__ = [
    "ModelEntry",
    "LoadedModel",
    "MODEL_REGISTRY",
    "register_model",
    "list_models",
    "resolve_model",
]


@dataclass
class ModelEntry:
    """A registry record: metadata + optional construction factory.

    ``factory`` returns a fully-configured model object. Sample-input shape
    for CV models lives in ``input_shape``; LM/vision-language models carry
    theirs inside ``metadata``. Adapters populate entries on the way in, the
    domain layer consumes them on the way out.
    """

    name: str
    domain: str  # "cv" | "lm"
    modality: str = ""  # "image" | "text" | "image+text" ...
    loader: str = ""  # "torchvision" | "timm" | "local" | "hf" | custom identifier
    input_shape: tuple[int, ...] | None = None  # sample input shape (CV only)
    factory: Callable[[], Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedModel:
    """Container returned by :func:`resolve_model`.

    ``sample_input`` is populated for CV models (needed by ModelOpt's
    forward-loop); LM models leave it ``None``. ``source`` and ``entry``
    are mostly informational — callers that need factory access can pull
    it back through ``entry``.
    """

    model: Any
    sample_input: Any | None = None
    source: str = ""  # "module" | "registry" | "torchvision" | "timm" | "local" | "hf"
    entry: ModelEntry | None = None


# Central registry; adapters install external-tool models here, other layers
# query capabilities with list_models() / MODEL_REGISTRY.has().
MODEL_REGISTRY = Registry("model")


def register_model(
    name: str,
    *,
    domain: str,
    factory: Callable[[], Any] | None = None,
    modality: str = "",
    loader: str = "",
    input_shape: tuple[int, ...] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ModelEntry:
    """Register a model entry. Returns the created ``ModelEntry`` so callers
    can mutate metadata post-registration (e.g. attach hardware quirks).
    """

    entry = ModelEntry(
        name=name,
        domain=domain,
        modality=modality,
        loader=loader,
        input_shape=input_shape,
        factory=factory,
        metadata=metadata or {},
    )
    MODEL_REGISTRY.register(entry, name=name)
    return entry


def list_models(domain: str | None = None) -> list[str]:
    """List registered model names, optionally filtered by domain."""
    return [
        name
        for name, item in MODEL_REGISTRY.items().items()
        if domain is None or getattr(item.target, "domain", None) == domain
    ]


# ---------- resolve helpers ------------------------------------------------


def _resolve_from_entry(entry: ModelEntry, input_shape=None) -> LoadedModel:
    """Materialise a registry entry into a LoadedModel.

    CV entries always go through the loader (sample-input generation lives
    there); LM entries trust the factory alone since their input spec lives
    elsewhere (HF config).
    """
    if entry.factory is None:
        raise ValueError(
            f"Model entry '{entry.name}' has no factory."
        )
    model = entry.factory()
    if entry.domain == "cv":
        shape = input_shape or entry.input_shape
        model, sample = resolve_cv_model_and_sample_input(
            model, input_shape=shape
        )
        return LoadedModel(model=model, sample_input=sample, source="registry", entry=entry)
    return LoadedModel(model=model, source="registry", entry=entry)


def resolve_model(
    spec: Any,
    *,
    domain: str = "auto",
    model_kind: str = "auto",
    input_shape: Iterable[int] | None = None,
    **loader_kwargs: Any,
) -> LoadedModel:
    """Resolve a model identifier into a ``LoadedModel``.

    :param spec: ``nn.Module`` / registry name / torchvision or timm name /
                 local checkpoint path / HF repo id / local HF directory.
    :param domain:     ``"auto"`` (default) routes heuristically; ``"cv"``
                       forces the CV loader; ``"lm"`` forces the HF loader.
    :param model_kind: CV source hint — ``"auto"`` / ``"torchvision"`` /
                       ``"timm"`` / ``"local"``.
    :param input_shape: Explicit NCHW shape for CV sample input; takes
                        precedence over best-effort inference.
    """
    # 1) Direct nn.Module — wrap as CV semantics, no registry lookup needed.
    if isinstance(spec, torch.nn.Module):
        model, sample = resolve_cv_model_and_sample_input(
            spec, input_shape=input_shape
        )
        return LoadedModel(model=model, sample_input=sample, source="module")

    name = str(spec).strip()
    if not name:
        raise ValueError("Model specification cannot be empty.")

    # 2) Exact registry hit — highest priority, adapters may pre-register
    #    external-tool models before any pipeline runs.
    if MODEL_REGISTRY.has(name):
        return _resolve_from_entry(MODEL_REGISTRY.get(name), input_shape=input_shape)

    # 3) Explicit domain skips auto-heuristics.
    if domain == "cv":
        model, sample = resolve_cv_model_and_sample_input(
            spec, model_kind=model_kind, input_shape=input_shape
        )
        return LoadedModel(
            model=model, sample_input=sample, source=_infer_cv_source(name, model_kind)
        )

    if domain == "lm":
        model = load_lm_model(spec, **loader_kwargs)
        return LoadedModel(model=model, source="hf")

    # 4) auto-dispatch: checkpoint → CV, HF repo id → LM, everything else →
    #    CV name (torchvision → timm fallback inside the loader).
    if looks_like_torch_checkpoint(name) or str(spec).endswith(".onnx"):
        model, sample = resolve_cv_model_and_sample_input(
            spec, model_kind=model_kind, input_shape=input_shape
        )
        return LoadedModel(
            model=model, sample_input=sample, source=_infer_cv_source(name, model_kind)
        )
    if looks_like_hf_repo_id(name):
        model = load_lm_model(spec, **loader_kwargs)
        return LoadedModel(model=model, source="hf")

    model, sample = resolve_cv_model_and_sample_input(
        spec, model_kind=model_kind, input_shape=input_shape
    )
    return LoadedModel(
        model=model, sample_input=sample, source=_infer_cv_source(name, model_kind)
    )


def _infer_cv_source(name: str, model_kind: str) -> str:
    """Best-effort source tag for LoadedModel.source."""
    if Path(name).exists() or looks_like_torch_checkpoint(name):
        return "local"
    if model_kind == "timm":
        return "timm"
    if model_kind == "torchvision":
        return "torchvision"
    return "torchvision"
