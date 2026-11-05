"""custom recipe loader.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from compressor.adapter.contract import Technique
from compressor.algorithms import AlgorithmRegistry

logger = logging.getLogger(__name__)

_RESERVED_TOP_KEYS = ("domain", "steps")
_RESERVED_STEP_KEYS = ("technique", "algorithm")


@dataclass
class RecipeStep:
    """recipe step representation: technique(type) + algorithm(implementation) + Modifier kwargs."""

    technique: Technique
    algorithm: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_modifier(self, registry: AlgorithmRegistry) -> Any:
        return registry.build_modifier(self.algorithm, **self.params)


@dataclass
class Recipe:
    """custom recipe representation."""

    domain: str
    steps: list[RecipeStep]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Recipe":
        if not isinstance(data, dict):
            raise ValueError(f"Recipe top level must be a mapping, got {type(data).__name__}")
        unknown = set(data) - set(_RESERVED_TOP_KEYS)
        if unknown:
            raise ValueError(f"Recipe top level keys not supported: {sorted(unknown)}; "
                             f"available: {list(_RESERVED_TOP_KEYS)}")
        domain = data.get("domain", "lm")
        if domain not in ("lm", "cv"):
            raise ValueError(f"domain must be 'lm'/'cv', got '{domain}'")
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("steps must be a non-empty list")

        steps: list[RecipeStep] = []
        for i, raw in enumerate(raw_steps):
            if not isinstance(raw, dict):
                raise ValueError(f"steps[{i}] must be a mapping, got {type(raw).__name__}")
            technique_name = raw.get("technique")
            try:
                technique = Technique(str(technique_name).lower())
            except ValueError:
                valid = [t.value for t in Technique]
                raise ValueError(
                    f"steps[{i}].technique '{technique_name}' is invalid technique; available: {valid}"
                ) from None
            algorithm = raw.get("algorithm")
            if not algorithm or not str(algorithm).strip():
                raise ValueError(f"steps[{i}] is missing algorithm")
            params = {
                k: v for k, v in raw.items() if k not in _RESERVED_STEP_KEYS
            }
            steps.append(RecipeStep(technique, str(algorithm).strip(), params))

        recipe = cls(domain=domain, steps=steps)
        recipe._check_registry()
        return recipe

    @classmethod
    def from_yaml(cls, source: str | Path) -> "Recipe":
        if isinstance(source, Path) or (
            isinstance(source, str)
            and source.endswith((".yaml", ".yml"))
            and "\n" not in source
        ):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Recipe file not found: {path}")
            text = path.read_text(encoding="utf-8")
        else:
            text = source
        data = yaml.safe_load(text)
        return cls.from_dict(data)

    def _check_registry(self) -> None:
        for i, step in enumerate(self.steps):
            try:
                spec = AlgorithmRegistry.get(step.algorithm)
            except KeyError as exc:
                raise ValueError(
                    f"steps[{i}].algorithm '{step.algorithm}' is not registered: {exc}"
                ) from None
            if spec.domain != self.domain:
                raise ValueError(
                    f"steps[{i}].algorithm '{step.algorithm}' is for domain {spec.domain}, not matching recipe domain '{self.domain}'"
                )

    def to_modifiers(self, registry: AlgorithmRegistry) -> list[Any]:
        """Convert to llmcompressor Modifier list for adapter Oneshot."""
        return [step.to_modifier(registry) for step in self.steps]


def load_recipe(source: Any) -> Any:

    if isinstance(source, dict) and "steps" in source:
        return Recipe.from_dict(source)
    if isinstance(source, (str, Path)):
        looks_like_path = isinstance(source, Path) or (
            source.endswith((".yaml", ".yml")) and "\n" not in source
        )
        if looks_like_path:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Recipe file not found: {path}")
            text = path.read_text(encoding="utf-8")
        else:
            text = source
        # llmcompressor native recipe use custom tag(!QuantizationModifier...),
        # safe_load cannot parse custom tag, pass-through to llmcompressor recipe factory
        if re.search(r"^\s*-?\s*!\w", text, re.MULTILINE):
            logger.debug("Recipe with custom tag, pass-through to llmcompressor recipe factory")
            return source
        data = yaml.safe_load(text)
        if isinstance(data, dict) and "steps" in data:
            return Recipe.from_dict(data)
        
        logger.debug("Recipe not a valid own schema, pass-through to llmcompressor recipe factory")
        return source
    return source
