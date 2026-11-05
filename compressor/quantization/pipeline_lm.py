from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compressor.adapter import get_lm_compressor
from compressor.adapter.contract import (
    LmCompressRequest,
    LmCompressResult,
    Technique,
)
from compressor.algorithms import AlgorithmRegistry
from compressor.hardware import (
    HardwareProfile,
    RecipeValidationError,
    validate_recipe_for_profile,
)
from compressor.quantization.recipe import Recipe, load_recipe

logger = logging.getLogger(__name__)


_DEFAULT_TARGETS = "Linear"
_DEFAULT_IGNORE = ("lm_head",)
_SCHEME_ALGOS = ("quantize", "gptq", "awq")  # algorithms accepting scheme


def _modifier_class_name(name: str) -> str:
    spec = AlgorithmRegistry.get(name)
    handler = spec.quant_op
    if isinstance(handler, str):
        return handler.rpartition(".")[2]
    return getattr(handler, "__name__", str(handler))


def _default_params(algorithm: str) -> dict[str, Any]:
    params: dict[str, Any] = {
        "targets": _DEFAULT_TARGETS,
        "ignore": list(_DEFAULT_IGNORE),
    }
    if algorithm in _SCHEME_ALGOS:
        params["scheme"] = "W8A8"
    return params


def _mapping_from_steps(steps: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Convert (algorithm, params) list to hardware validation modifiers mapping."""
    return {
        "modifiers": [
            {_modifier_class_name(algo): params} for algo, params in steps
        ]
    }


@dataclass
class _RecipePlan:
    """recipe two-stage parsing plan: first static validation, then materialize Modifier.
    """

    steps: list[tuple[str, dict[str, Any]]] | None = None
    passthrough: Any = None
    mapping: dict[str, Any] | None = None

    def materialize(self) -> Any:
        """Materialize Modifier."""
        if self.steps is not None:
            return [
                AlgorithmRegistry.build_modifier(algo, **params)
                for algo, params in self.steps
            ]
        return self.passthrough


def _plan_recipe(recipe: Any) -> _RecipePlan:
    if recipe is None:
        return _RecipePlan()

    if (
        isinstance(recipe, str)
        and recipe in AlgorithmRegistry
        and not Path(recipe).exists()
    ):
        steps = [(recipe, _default_params(recipe))]
        return _RecipePlan(steps=steps, mapping=_mapping_from_steps(steps))

    # custom recipe: Recipe object passthrough, or yaml path/text/dict parsed via load_recipe
    if isinstance(recipe, Recipe):
        steps = [(s.algorithm, dict(s.params)) for s in recipe.steps]
        return _RecipePlan(steps=steps, mapping=_mapping_from_steps(steps))
    if isinstance(recipe, (str, dict, Path)):
        recipe_obj = load_recipe(recipe)
        if isinstance(recipe_obj, Recipe):
            steps = [(s.algorithm, dict(s.params)) for s in recipe_obj.steps]
            return _RecipePlan(steps=steps, mapping=_mapping_from_steps(steps))
        return _RecipePlan(
            passthrough=recipe_obj,
            mapping=recipe_obj if isinstance(recipe_obj, (str, dict)) else None,
        )

    return _RecipePlan(passthrough=recipe)


def run_lm_compression(
    model: str,
    output_dir: str | Path,
    recipe: Any | None = None,
    *,
    technique: Technique = Technique.QUANTIZE,
    dataset: str = "wikitext",
    dataset_config_name: str | None = "wikitext-2-raw-v1",
    splits: str = "train[:1]",
    num_calibration_samples: int = 8,
    max_seq_length: int = 128,
    hardware_profile: HardwareProfile | None = None,
    validate_hardware: bool = True,
    trust_remote_code_model: bool = True,
    options: dict[str, Any] | None = None,
) -> LmCompressResult:
    """LM compression pipeline.

    :param recipe: None(default) / algorithm / custom yaml / raw recipe pass
    :param hardware_profile: Hardware profile for validation; None for default profile(CKS default rules)
    :param validate_hardware: False to skip hardware validation (for debug)
    """
    plan = _plan_recipe(recipe)

    if validate_hardware and plan.mapping is not None:
        profile = hardware_profile or HardwareProfile()
        verdict = validate_recipe_for_profile(plan.mapping, profile)
        if verdict.skipped:
            logger.debug(
                "Hardware profile '%s' skipped validation: %s", profile.name, verdict.skip_reason
            )
        if verdict.errors:
            details = "\n".join(
                f"  - [{v.where}] {v.message}" for v in verdict.errors
            )
            raise RecipeValidationError(
                "recipe failed hardware validation, submission blocked:\n" f"{details}"
            )
        for warning in verdict.warnings:
            logger.warning(
                "[hw-profile:%s] [%s] %s",
                profile.name, warning.where, warning.message,
            )

    compressor = get_lm_compressor()
    compressor.install()

    request = LmCompressRequest(
        model=model,
        output_dir=str(output_dir),
        recipe=plan.materialize(),
        dataset=dataset,
        dataset_config_name=dataset_config_name,
        splits=splits,
        num_calibration_samples=num_calibration_samples,
        max_seq_length=max_seq_length,
        trust_remote_code_model=trust_remote_code_model,
        options=options or {},
    )
    if technique is Technique.QUANTIZE:
        return compressor.quantize(request)
    return compressor.sparsify(request)
