"""HardwareProfile: recipe acceptance rules for the CSK tensor unit.

Format facts (``TuFormat`` / ``RoundingMode`` / block sizes) live in
``compressor.hardware.formats``. This module is about *acceptance*: which
weight/activation schemes the chip admits, expressed in the
compressed-tensors ``(num_bits, type)`` vocabulary that recipes use.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from compressor.hardware.formats import SUPPORTED_BLOCK_SIZES

logger = logging.getLogger(__name__)


class RecipeValidationError(RuntimeError):
    """Raised when a recipe violates the hardware profile."""


# (num_bits, type) — compressed-tensors recipe vocabulary. Coarser than
# formats.TuFormat (kernel vocabulary): e.g. FP8_SPEC covers both
# TuFormat.FP8_E4M3 and FP8_E5M2, FP4_SPEC covers MXFP4 and NVFP4.
FP8_SPEC = (8, "float")
INT8_SPEC = (8, "int")
FP4_SPEC = (4, "float")
INT4_SPEC = (4, "int")


@dataclass(frozen=True)
class SchemeRule:
    """Acceptance rule for one quantization scheme family. """

    name: str
    weight: tuple[int, str] | None
    activation: tuple[int, str] | None
    group_sizes: tuple[int, ...] = ()  # subset of formats.SUPPORTED_BLOCK_SIZES


DEFAULT_SCHEMES: dict[str, SchemeRule] = {
    rule.name: rule
    for rule in (
        SchemeRule("FP8", FP8_SPEC, FP8_SPEC),
        SchemeRule("FP8-DYNAMIC", FP8_SPEC, FP8_SPEC),
        SchemeRule("FP8-STATIC", FP8_SPEC, FP8_SPEC),
        SchemeRule("FP8-BLOCK", FP8_SPEC, FP8_SPEC),
        SchemeRule("INT8", INT8_SPEC, INT8_SPEC),

        SchemeRule("W8A16", INT8_SPEC, None),
        SchemeRule("W4A16", INT4_SPEC, None),
        SchemeRule("W4A8", INT4_SPEC, FP8_SPEC),
        SchemeRule("NVFP4", FP4_SPEC, FP4_SPEC, group_sizes=(16,)),
        SchemeRule("NVFP4A16", FP4_SPEC, None, group_sizes=(16,)),
        SchemeRule("MXFP4", FP4_SPEC, FP4_SPEC, group_sizes=(32, 64, 128, 256)),
        SchemeRule("MXFP4A8", FP4_SPEC, FP8_SPEC, group_sizes=(32, 64, 128, 256)),
        SchemeRule("MXFP8", FP8_SPEC, FP8_SPEC, group_sizes=(32, 64, 128, 256)),
    )
}

DEFAULT_SCHEME_ALIASES: dict[str, str] = {
    "W8A8": "INT8",
    "FLOAT8": "FP8",
}

ATTENTION_TARGET_PATTERN = (
    r"(attention|attn|qkv|query_key_value|q_proj|k_proj|v_proj|qkv_proj)"
)

BROAD_TARGETS = {"linear", "conv1d", "conv2d", "conv3d", "linearmodule"}

_QUANT_PARAM_KEYS = ("weights", "input_activations", "output_activations")


def _default_attention_pattern() -> "re.Pattern[str]":
    return re.compile(ATTENTION_TARGET_PATTERN, re.IGNORECASE)


@dataclass
class HardwareProfile:
    """Machine-readable hardware acceptance rules for quantization recipes."""

    name: str = "csk-generic"
    schemes: dict[str, SchemeRule] = field(
        default_factory=lambda: {k: v for k, v in DEFAULT_SCHEMES.items()}
    )
    scheme_aliases: dict[str, str] = field(
        default_factory=lambda: {k: v for k, v in DEFAULT_SCHEME_ALIASES.items()}
    )
    allowed_weights: frozenset[tuple[int, str]] = frozenset(
        {FP8_SPEC, INT8_SPEC, FP4_SPEC, INT4_SPEC}
    )
    allowed_activations: frozenset[tuple[int, str]] = frozenset(
        {FP8_SPEC, INT8_SPEC, FP4_SPEC}
    )
    kv_cache_specs: frozenset[tuple[int, str]] = frozenset({FP8_SPEC})
    # The subset of formats.SUPPORTED_BLOCK_SIZES that actually lands on
    # microscale operators (smaller than the chip's full block-size range).
    microscale_group_sizes: frozenset[int] = frozenset({16, 32})
    attention_targets: "re.Pattern[str]" = field(
        default_factory=_default_attention_pattern
    )
    attention_requires_fp8_activation: bool = True
    recommended_ignore: tuple[str, ...] = ("lm_head",)

    def __post_init__(self) -> None:
        """Keep declared block sizes consistent with the format model.

        ``SUPPORTED_BLOCK_SIZES`` (in ``compressor.hardware.formats``) is the
        single source of truth for block sizes; every scheme / microscale
        group size declared here must be a subset of it.
        """
        for rule in self.schemes.values():
            for group_size in rule.group_sizes:
                if group_size not in SUPPORTED_BLOCK_SIZES:
                    raise ValueError(
                        f"scheme '{rule.name}' declares group_size={group_size} "
                        f"outside SUPPORTED_BLOCK_SIZES={sorted(SUPPORTED_BLOCK_SIZES)}"
                    )
        for group_size in self.microscale_group_sizes:
            if group_size not in SUPPORTED_BLOCK_SIZES:
                raise ValueError(
                    f"microscale_group_sizes contains {group_size} outside "
                    f"SUPPORTED_BLOCK_SIZES={sorted(SUPPORTED_BLOCK_SIZES)}"
                )

    def resolve_scheme(self, name: str) -> SchemeRule | None:
        normalized = str(name).strip().upper()
        normalized = self.scheme_aliases.get(normalized, normalized)
        return self.schemes.get(normalized)

    def supported_scheme_names(self) -> list[str]:
        return sorted(self.schemes)


@dataclass(frozen=True)
class Violation:
    level: str  # "error" | "warning"
    where: str
    message: str


@dataclass
class ProfileVerdict:
    profile_name: str
    skipped: bool = False
    skip_reason: str | None = None
    violations: list[Violation] = field(default_factory=list)

    @property
    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.level == "error"]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.level == "warning"]


def _spec_tuple(spec: Any) -> tuple[int, str] | None:
    """Extract a (num_bits, type) tuple from a param spec, if quantized."""
    if not isinstance(spec, dict):
        return None
    num_bits = spec.get("num_bits")
    if num_bits is None:
        return None
    qtype = spec.get("type")
    return (int(num_bits), str(qtype).lower() if qtype is not None else "int")


def _flatten_targets(targets: Any) -> list[str]:
    if targets is None:
        return []
    if isinstance(targets, str):
        return [targets]
    if isinstance(targets, (list, tuple, set)):
        flattened: list[str] = []
        for item in targets:
            flattened.extend(_flatten_targets(item))
        return flattened
    return [str(targets)]


def _is_broad_target(target: str) -> bool:
    return target.strip().lower() in BROAD_TARGETS or target.strip().startswith("re:")


def _recipe_to_mapping(recipe: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Normalize recipe inputs into a dict mapping.

    Returns ``(mapping, None)`` on success or ``(None, reason)`` when the
    recipe cannot be statically validated (e.g. a preset name such as
    ``"quant_8bit"``).
    """
    if recipe is None:
        return None, "recipe is None (no quantization recipe)"
    if isinstance(recipe, dict):
        return recipe, None
    if not isinstance(recipe, str):
        return None, f"unsupported recipe type: {type(recipe).__name__}"

    text = recipe.strip()
    if not text:
        return None, "recipe is empty"

    try:
        path = Path(text)
        if path.exists() and path.is_file():
            text = path.read_text(encoding="utf-8")
    except OSError:  # pragma: no cover - path too long / invalid
        pass

    try:
        import yaml
    except ImportError:
        return None, "pyyaml is not installed; cannot parse recipe string"

    try:
        mapping = yaml.safe_load(text)
    except Exception:
        return None, "recipe is neither a file nor valid YAML"

    if not isinstance(mapping, dict):
        return None, f"recipe '{text[:40]}' looks like a preset name; skipped"

    return mapping, None


def _iter_modifiers(mapping: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Yield ``(modifier_name, params)`` from a recipe mapping.

    Supports stage-nested structures (``<stage>_modifiers``), a bare
    ``modifiers`` list, a top-level modifier list, and a single modifier
    mapping.
    """
    modifiers: list[tuple[str, dict[str, Any]]] = []

    def _collect(entries: Any) -> None:
        if not isinstance(entries, list):
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for name, params in entry.items():
                modifiers.append(
                    (str(name), params if isinstance(params, dict) else {})
                )

    if "modifiers" in mapping:
        _collect(mapping["modifiers"])
    else:
        for value in mapping.values():
            if not isinstance(value, dict):
                continue
            for key, nested in value.items():
                if key.endswith("modifiers"):
                    _collect(nested)
        if not modifiers:
            # a single modifier mapping, e.g. {"QuantizationModifier": {...}}
            _collect([mapping])

    return modifiers


def _check_scheme_name(
    scheme: Any,
    profile: HardwareProfile,
    where: str,
    violations: list[Violation],
) -> SchemeRule | None:
    if scheme is None or not str(scheme).strip():
        return None
    rule = profile.resolve_scheme(scheme)
    if rule is None:
        violations.append(
            Violation(
                "error",
                where,
                f"scheme '{scheme}' is not in the supported list for hardware '{profile.name}' "
                f"supported: {', '.join(profile.supported_scheme_names())}",
            )
        )
    return rule


def _check_param_spec(
    key: str,
    spec: Any,
    profile: HardwareProfile,
    where: str,
    violations: list[Violation],
) -> tuple[int, str] | None:
    spec_tuple = _spec_tuple(spec)
    if spec_tuple is None:
        return None

    if key == "kv_cache":
        if spec_tuple not in profile.kv_cache_specs:
            violations.append(
                Violation(
                    "error",
                    where,
                    f"kv_cache spec {spec_tuple} is not supported for hardware '{profile.name}' "
                    f"supported: {sorted(profile.kv_cache_specs)}",
                )
            )
        return spec_tuple

    allowed = (
        profile.allowed_weights if key == "weights" else profile.allowed_activations
    )
    if spec_tuple not in allowed:
        violations.append(
            Violation(
                "error",
                where,
                f"{key} spec {spec_tuple} is not in the supported list for hardware '{profile.name}' "
                f"supported: {sorted(allowed)}",
            )
        )

    if key == "weights" and spec_tuple == FP4_SPEC and isinstance(spec, dict):
        group_size = spec.get("group_size")
        if group_size is not None and int(group_size) not in profile.microscale_group_sizes:
            violations.append(
                Violation(
                    "error",
                    where,
                    f"FP4 weight group_size={group_size} is not supported for hardware '{profile.name}' "
                    f"supported: {sorted(profile.microscale_group_sizes)}",
                )
            )
        strategy = spec.get("strategy")
        if strategy is not None and str(strategy).lower() != "group":
            violations.append(
                Violation(
                    "warning",
                    where,
                    "FP4 weight usually requires strategy=group to land on microscale operators",
                )
            )
    return spec_tuple


def _check_group_spec(
    group_name: str,
    spec: Any,
    profile: HardwareProfile,
    where: str,
    violations: list[Violation],
    state: dict[str, bool],
) -> None:
    targets = _flatten_targets(spec.get("targets") if isinstance(spec, dict) else None)
    explicit_attention = any(profile.attention_targets.search(t) for t in targets)
    broad_targets = bool(targets) and all(_is_broad_target(t) for t in targets)

    if isinstance(spec, str):
        rule = _check_scheme_name(spec, profile, where, violations)
        _check_attention_coverage(rule, explicit_attention, broad_targets,
                                  profile, where, violations, state)
        return

    if not isinstance(spec, dict):
        return

    rule = _check_scheme_name(spec.get("scheme"), profile, where, violations)

    param_specs: list[tuple[str, Any]] = []
    for key in _QUANT_PARAM_KEYS:
        if key in spec:
            param_specs.append((key, spec.get(key)))
    kv_spec = spec.get("kv_cache")
    if kv_spec is not None:
        param_specs.append(("kv_cache", kv_spec))

    act_tuple: tuple[int, str] | None = None
    for key, param_spec in param_specs:
        checked = _check_param_spec(key, param_spec, profile, where, violations)
        if key in ("input_activations", "output_activations"):
            act_tuple = checked

    _check_attention_coverage(
        rule, explicit_attention, broad_targets, profile, where, violations, state
    )
    if (
        profile.attention_requires_fp8_activation
        and explicit_attention
        and act_tuple == FP4_SPEC
    ):
        violations.append(
            Violation(
                "error",
                f"{where}/{group_name}",
                "attention operator activation only supports FP8; "
                "please change attention layers to FP8 activation or add to ignore",
            )
        )


def _check_attention_coverage(
    rule: SchemeRule | None,
    explicit_attention: bool,
    broad_targets: bool,
    profile: HardwareProfile,
    where: str,
    violations: list[Violation],
    state: dict[str, bool],
) -> None:
    if not profile.attention_requires_fp8_activation:
        return
    activation = rule.activation if rule is not None else None
    if activation != FP4_SPEC:
        return
    if explicit_attention:
        return
    if broad_targets and not state.get("broad_attention_warned"):
        state["broad_attention_warned"] = True
        violations.append(
            Violation(
                "warning",
                where,
                "4-bit activation scheme covers attention layers with broad targets (e.g., Linear); "
                "please confirm attention is processed as FP8 activation or is already in ignore",
            )
        )


def _check_ignore_recommendation(
    modifier_name: str,
    params: dict[str, Any],
    profile: HardwareProfile,
    where: str,
    violations: list[Violation],
) -> None:
    if not any(token in modifier_name for token in ("Quantization", "GPTQ", "AWQ")):
        return
    ignore = params.get("ignore") or []
    ignore_text = " ".join(str(item) for item in ignore).lower()
    for token in profile.recommended_ignore:
        if token.lower() not in ignore_text:
            violations.append(
                Violation(
                    "warning",
                    where,
                    f"recommend adding '{token}' to ignore to keep precision (hardware does not use quantization operators)",
                )
            )


def validate_recipe_for_profile(
    recipe: Any,
    profile: HardwareProfile | None = None,
) -> ProfileVerdict:
    """Statically validate a recipe against a hardware profile.
    """

    profile = profile or HardwareProfile()
    verdict = ProfileVerdict(profile_name=profile.name)
    state: dict[str, bool] = {}

    if not isinstance(recipe, dict):
        mapping, skip_reason = _recipe_to_mapping(recipe)
        if mapping is None:
            verdict.skipped = True
            verdict.skip_reason = skip_reason
            return verdict
    else:
        mapping = recipe

    modifiers = _iter_modifiers(mapping)
    if not modifiers:
        verdict.skipped = True
        verdict.skip_reason = "recipe does not contain any valid modifier to validate"
        return verdict

    for modifier_name, params in modifiers:
        where = f"{profile.name}/{modifier_name}"
        _check_ignore_recommendation(
            modifier_name, params, profile, where, verdict.violations
        )

        config_groups = params.get("config_groups")
        if config_groups is None:
            # `modifier` level scheme/targets(e.g., QuantizationModifier directly configured)
            group_spec = {k: v for k, v in params.items() if k != "ignore"}
            if any(key in group_spec for key in ("scheme", *_QUANT_PARAM_KEYS, "kv_cache")):
                _check_group_spec(
                    "default", group_spec, profile, where, verdict.violations, state
                )
            continue

        if isinstance(config_groups, str):
            _check_scheme_name(config_groups, profile, where, verdict.violations)
            continue

        groups = (
            config_groups.items()
            if isinstance(config_groups, dict)
            else enumerate(config_groups)
        )
        for group_name, group_spec in groups:
            _check_group_spec(
                str(group_name),
                group_spec,
                profile,
                where,
                verdict.violations,
                state,
            )

    return verdict
