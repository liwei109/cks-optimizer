"""CNN optimization configuration models, validation, and loading."""

from __future__ import annotations

import copy
import json
import logging
import re
from collections.abc import Mapping, MutableMapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Importing the built-in handler implementations registers the supported
# datasets; the handler registry below then powers task and dataset inference.
from cks_optimizer.cnn.datasets import registry as _builtin_dataset_handlers  # noqa: E402, F401
from cks_optimizer.cnn.datasets.handler import (  # noqa: E402
    DATASET_HANDLER_REGISTRY,
    iter_dataset_handlers,
)

__all__ = [
    "QUANT_MODES",
    "SPARSITY_MODES",
    "TRT_ONLY_QUANT_MODES",
    "CalibrationConfig",
    "CNNConfig",
    "ExportConfig",
    "OptimizeConfig",
    "QuantConfig",
    "SparsityConfig",
    "build_default_output_path",
    "build_optimize_config",
    "infer_dataset_from_root",
    "infer_task_from_model_and_dataset",
]

# Dataset names and their corresponding vision tasks, derived from the
# registered dataset handlers (see ``cks_optimizer.cnn.datasets.handler``).
_DATASET_TASKS = {
    handler.name: handler.task
    for handler in DATASET_HANDLER_REGISTRY.values()
}

# Model-name fragments that imply a specific task.
_SEGMENTATION_NAME_HINTS = ("seg", "unet", "deeplab", "mask")
_DETECTION_NAME_HINTS = ("yolo", "det", "detection")

# Supported quantization modes mapped to ModelOpt preset names.
QUANT_MODES: dict[str, str] = {
    "int8": "INT8_DEFAULT_CFG",
    "mxint8": "MXINT8_DEFAULT_CFG",
    "fp8": "FP8_DEFAULT_CFG",
    "nvfp4": "NVFP4_DEFAULT_CFG",
    "nvfp4_svdquant": "NVFP4_SVDQUANT_DEFAULT_CFG",
    "mxfp4": "MXFP4_DEFAULT_CFG",
    "mxfp6": "MXFP6_DEFAULT_CFG",
    "mxfp8": "MXFP8_DEFAULT_CFG",
}

# Sparsity modes supported by ModelOpt. ``sparse_magnitude`` is the only mode
# that does not require a calibration dataloader.
SPARSITY_MODES: frozenset[str] = frozenset({"sparse_magnitude", "sparsegpt"})

# Modes whose exported ONNX graph contains TensorRT-only custom ops, which
# onnxruntime cannot load; accuracy evaluation is skipped for these.
TRT_ONLY_QUANT_MODES: frozenset[str] = frozenset({"fp8", "nvfp4", "nvfp4_svdquant"})

# Nested section names accepted by ``build_optimize_config``.
_SECTION_NAMES = frozenset({"quant", "sparsity", "export", "calibration"})

# Legacy flat keys that map onto a nested section:
#   legacy flat key -> (section attribute, field within that section)
_LEGACY_FLAT_KEYS: dict[str, tuple[str, str]] = {
    "quant_mode": ("quant", "mode"),
    "quant_preset": ("quant", "preset"),
    "sparsity_mode": ("sparsity", "mode"),
    "sparsity_level": ("sparsity", "sparsity"),
    "opset_version": ("export", "opset_version"),
    "use_dynamic_axes": ("export", "use_dynamic_axes"),
    "simplify": ("export", "simplify"),
    "check_onnx": ("export", "check_onnx"),
    "calibration_samples": ("calibration", "num_samples"),
    "calibration_batch_size": ("calibration", "batch_size"),
    "calibration_split": ("calibration", "split"),
    # Short aliases for the calibration section, accepted alongside the
    # ``calibration_*`` spellings above.
    "num_samples": ("calibration", "num_samples"),
    "batch_size": ("calibration", "batch_size"),
    "split": ("calibration", "split"),
}

# Keys accepted for backward compatibility that are not dataclass fields.
_IGNORED_KEYS = frozenset({"algorithms"})


def _normalize_grouped_config(data: dict[str, Any]) -> dict[str, Any]:
    """Translate grouped recipe sections into the internal flat config schema."""
    groups = {
        name: data.pop(name)
        for name in ("model", "data", "optimize")
        if name in data
    }

    def require_mapping(section_name: str, value: Any) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError(f"Config section {section_name!r} must be a mapping")
        return value

    def assign(key: str, value: Any) -> None:
        if key in data:
            raise ValueError(
                f"Config value {key!r} is specified both inside a grouped section "
                "and at the top level."
            )
        data[key] = copy.deepcopy(value)

    model = require_mapping("model", groups.get("model", {}))
    for key, value in model.items():
        target = {"name": "model_name"}.get(key, key)
        if target not in {"model_name", "pretrained", "input_shape"}:
            raise ValueError(f"Unknown model config key {key!r}")
        assign(target, value)

    data_group = require_mapping("data", groups.get("data", {}))
    unknown_data_keys = set(data_group) - {
        "root",
        "dataset_root",
        "calibration",
        "evaluation",
        "evaluation_split",
    }
    if unknown_data_keys:
        raise ValueError(f"Unknown data config key(s): {sorted(unknown_data_keys)}")
    for key in ("root", "dataset_root"):
        if key in data_group:
            assign("dataset_root", data_group[key])
    if "calibration" in data_group:
        assign("calibration", data_group["calibration"])
    if "evaluation" in data_group:
        evaluation = require_mapping("data.evaluation", data_group["evaluation"])
        if set(evaluation) - {"split", "batch_size"}:
            raise ValueError(
                "data.evaluation only accepts the 'split' and 'batch_size' keys"
            )
        if "split" in evaluation:
            assign("evaluation_split", evaluation["split"])
        if "batch_size" in evaluation:
            assign("evaluation_batch_size", evaluation["batch_size"])
    if "evaluation_split" in data_group:
        assign("evaluation_split", data_group["evaluation_split"])

    optimize = require_mapping("optimize", groups.get("optimize", {}))
    for key, value in optimize.items():
        target = {"quantization": "quant"}.get(key, key)
        if target not in {"quant", "sparsity"}:
            raise ValueError(
                f"Unknown optimize config key {key!r}; expected 'quantization' "
                "and/or 'sparsity'."
            )
        assign(target, value)

    return data


# Inference


def infer_dataset_from_root(dataset_root: str | Path | None) -> str:
    """Infer the dataset from the layout of ``dataset_root``.

    Each registered dataset handler is asked, in order of layout specificity
    (most specific first), whether the root matches its layout; ImageNet is
    the fallback. Only the root, its immediate children, and immediate
    ``annotations`` directories are inspected, avoiding a recursive walk over
    image datasets.
    """
    root = Path(dataset_root) if dataset_root else Path(".")
    if not root.is_dir():
        return "imagenet"
    for handler in iter_dataset_handlers():
        if handler.matches_layout(root):
            return handler.name
    return "imagenet"


def build_default_output_path(model_name: str, quant_mode: str) -> str:
    """Build a default ONNX output path like ``tmp/<model>.<mode>.onnx``.

    Only the file name is used, so a local file such as ``tmp/resnet50.onnx``
    (or a hub id like ``microsoft/resnet-50``) yields ``tmp/resnet50.int8.onnx``
    instead of embedding the directory in the name.
    """
    name = Path((model_name or "model").strip()).name
    if name.lower().endswith(".onnx"):
        name = name[: -len(".onnx")]
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "-", name)
    safe_model = safe_model.strip(".-") or "model"
    suffix = (quant_mode or "optimized").strip().lower()
    return f"tmp/{safe_model}.{suffix}.onnx"


def infer_task_from_model_and_dataset(
    model_name: str | None, dataset_name: str | None = None
) -> str:
    """Infer the CNN task from the model name and dataset name.

    Dataset layout determines the task when it is a supported benchmark;
    otherwise model-name hints are used.
    """
    model = (model_name or "").lower()
    dataset = (dataset_name or "").lower()

    if dataset in _DATASET_TASKS:
        return _DATASET_TASKS[dataset]
    if any(token in model for token in _SEGMENTATION_NAME_HINTS):
        return "segmentation"
    if any(token in model for token in _DETECTION_NAME_HINTS):
        return "detection"
    return "classification"


# Validation


def _normalize_quant_mode(mode: Any) -> str:
    """Validate and normalise a quantization mode name."""
    normalized = str(mode).strip().lower()
    if normalized not in QUANT_MODES:
        raise ValueError(
            f"Unsupported quantization mode {mode!r}; expected one of {sorted(QUANT_MODES)}"
        )
    return normalized


def _normalize_name_list(value: Any, field_name: str) -> list[str]:
    """Validate an ONNX input/output name list."""
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of strings")
    names = [str(name) for name in value]
    if not names or any(not name for name in names):
        raise ValueError(f"{field_name} must contain at least one non-empty name")
    return names


def _validate_input_shape(shape: Any) -> tuple[int, ...]:
    """Validate a fully-specified NCHW input shape."""
    if isinstance(shape, int) or not isinstance(shape, (list, tuple)):
        raise ValueError("input_shape must be a 4-element sequence (NCHW)")
    validated = tuple(int(dim) for dim in shape)
    if len(validated) != 4:
        raise ValueError(f"input_shape must have exactly 4 dimensions (NCHW), got {validated}")
    if any(dim < 1 for dim in validated):
        raise ValueError(f"input_shape dimensions must be positive, got {validated}")
    return validated


# Configuration models


@dataclass
class QuantizerSettings:
    """INT8 quantizer settings mapped to ModelOpt quantizer attributes."""

    bits: int = 8
    scheme: str = "symmetric"
    granularity: str = "per_tensor"
    group_size: int | None = None
    axis: int | None = None

    def __post_init__(self) -> None:
        self.bits = int(self.bits)
        if self.bits < 1:
            raise ValueError(f"quantizer bits must be >= 1, got {self.bits}")

        self.scheme = str(self.scheme).strip().lower().replace("-", "_")
        if self.scheme != "symmetric":
            raise ValueError(
                f"Unsupported quantization scheme {self.scheme!r}; the ModelOpt "
                "PyTorch backend supports symmetric integer quantization only."
            )

        self.granularity = str(self.granularity).strip().lower().replace("-", "_")
        if self.granularity not in {"per_tensor", "per_channel", "per_group"}:
            raise ValueError(
                f"Unsupported quantization granularity {self.granularity!r}; "
                "expected per_tensor, per_channel, or per_group."
            )

        if self.group_size is not None:
            self.group_size = int(self.group_size)
            if self.group_size < 1:
                raise ValueError(
                    f"quantizer group_size must be >= 1, got {self.group_size}"
                )

        if self.granularity == "per_group":
            if self.group_size is None:
                raise ValueError("group_size is required for per_group quantization")
            if self.axis is not None:
                raise ValueError("axis cannot be set for per_group quantization")
        elif self.group_size is not None:
            raise ValueError("group_size is only valid for per_group quantization")

        if self.granularity == "per_tensor" and self.axis is not None:
            raise ValueError("axis cannot be set for per_tensor quantization")


@dataclass
class QuantConfig:
    """Quantization settings (ModelOpt)."""

    mode: str = "int8"
    preset: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    weights: QuantizerSettings | Mapping[str, Any] | None = None
    activations: QuantizerSettings | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        self.mode = _normalize_quant_mode(self.mode)
        for name in ("weights", "activations"):
            settings = getattr(self, name)
            if isinstance(settings, Mapping):
                settings = QuantizerSettings(**settings)
                setattr(self, name, settings)
            elif settings is not None and not isinstance(settings, QuantizerSettings):
                raise ValueError(f"quantization.{name} must be a mapping")


@dataclass
class SparsityConfig:
    """Sparsity settings (ModelOpt)."""

    mode: str = "sparse_magnitude"
    sparsity: float = 0.5
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mode = str(self.mode).strip().lower()
        if self.mode not in SPARSITY_MODES:
            raise ValueError(
                f"Unsupported sparsity mode {self.mode!r}; expected one of {sorted(SPARSITY_MODES)}"
            )
        self.sparsity = float(self.sparsity)
        if not 0.0 <= self.sparsity < 1.0:
            raise ValueError(f"sparsity must be in [0.0, 1.0), got {self.sparsity}")


@dataclass
class ExportConfig:
    """ONNX export settings."""

    output_path: str = ""
    opset_version: int = 17
    use_dynamic_axes: bool = True
    simplify: bool = True
    check_onnx: bool = True
    input_names: list[str] = field(default_factory=lambda: ["input"])
    output_names: list[str] = field(default_factory=lambda: ["output"])

    def __post_init__(self) -> None:
        self.opset_version = int(self.opset_version)
        if self.opset_version < 7:
            raise ValueError(f"opset_version must be >= 7, got {self.opset_version}")
        self.input_names = _normalize_name_list(self.input_names, "input_names")
        self.output_names = _normalize_name_list(self.output_names, "output_names")


@dataclass
class CalibrationConfig:
    """Calibration settings used by ``data_process`` and quantization."""

    num_samples: int = 32
    batch_size: int = 1
    split: str = "train"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.num_samples = int(self.num_samples)
        if self.num_samples < 1:
            raise ValueError(f"calibration.num_samples must be >= 1, got {self.num_samples}")
        self.batch_size = int(self.batch_size)
        if self.batch_size < 1:
            raise ValueError(f"calibration.batch_size must be >= 1, got {self.batch_size}")


@dataclass
class OptimizeConfig:
    """Top-level configuration for the CNN optimization pipeline."""

    model_name: str = "resnet"
    dataset_root: str = ""
    task: str = ""
    input_shape: tuple[int, ...] | None = None
    evaluation_split: str = "val"
    evaluation_batch_size: int = 16
    pretrained: bool = False
    device: str = ""
    quant: QuantConfig | None = None
    sparsity: SparsityConfig | None = None
    export: ExportConfig = field(default_factory=ExportConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    extra: dict[str, Any] = field(default_factory=dict)
    # Set when the caller supplied an output path, so ``__post_init__`` must
    # not overwrite it with a value derived from ``quant.mode``.
    _output_path_explicit: bool = field(default=False, repr=False, compare=False)
    _quant_cfg: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.input_shape is not None:
            self.input_shape = _validate_input_shape(self.input_shape)
        if not self.task:
            dataset = infer_dataset_from_root(self.dataset_root or ".")
            self.task = infer_task_from_model_and_dataset(self.model_name, dataset)
        self.evaluation_split = str(self.evaluation_split).strip() or "val"
        self.evaluation_batch_size = int(self.evaluation_batch_size)
        if self.evaluation_batch_size < 1:
            raise ValueError(
                "evaluation_batch_size must be >= 1, "
                f"got {self.evaluation_batch_size}"
            )
        if not self.export.output_path and not self._output_path_explicit:
            self.export.output_path = build_default_output_path(
                self.model_name, self.quant.mode if self.quant else ""
            )

    @property
    def output_path(self) -> str:
        """The ONNX output path (mirrors ``export.output_path``)."""
        return self.export.output_path

    @output_path.setter
    def output_path(self, value: str) -> None:
        self.export.output_path = value
        self._output_path_explicit = bool(value)

    @property
    def quant_mode(self) -> str | None:
        """The quantization mode, or ``None`` when quantization is not configured."""
        return self.quant.mode if self.quant else None

    @quant_mode.setter
    def quant_mode(self, value: str) -> None:
        if self.quant is None:
            self.quant = QuantConfig(mode=value)
        else:
            self.quant.mode = _normalize_quant_mode(value)
        # Keep a derived output path in sync; an explicit path is left alone.
        if not self._output_path_explicit:
            self.export.output_path = build_default_output_path(self.model_name, value)

    @property
    def effective_algorithms(self) -> list[str]:
        """Algorithms selected by the configured sections.

        Sparsity runs before quantization when both are configured, matching the
        order expected by ModelOpt's ``optimize`` API.
        """
        algorithms: list[str] = []
        if self.sparsity is not None:
            algorithms.append("sparsify")
        if self.quant is not None:
            algorithms.append("quantize")
        return algorithms

    @property
    def requires_calibration(self) -> bool:
        """Whether a configured algorithm needs calibration data."""
        if self.quant is not None:
            return True
        return self.sparsity is not None and self.sparsity.mode == "sparsegpt"

    @property
    def evaluation_supported(self) -> bool:
        """Whether ONNX Runtime supports the configured quantization mode."""
        return self.quant is None or self.quant.mode not in TRT_ONLY_QUANT_MODES

    def output_file(self) -> Path:
        """Return the ONNX output path as a :class:`~pathlib.Path`."""
        return Path(self.export.output_path)


# Config loading


def _load_config_file(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML config file into a plain dict."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    text = file_path.read_text(encoding="utf-8")
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "PyYAML is required to load YAML config files. Install it with "
                "`pip install PyYAML` or use a JSON config file instead."
            ) from exc
        data = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported config file extension {suffix!r}; use .json, .yaml or .yml")
    if not isinstance(data, dict):
        raise ValueError(f"Config file {file_path} must contain a mapping")
    return data


def _build_section(cls: type, data: Mapping[str, Any]) -> Any:
    """Build a dataclass from a mapping, routing unknown keys into ``extra``."""
    field_names = {f.name for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in data.items():
        if key == "enabled":
            raise ValueError(
                "'enabled' is not supported; include or omit the config section "
                "to select the algorithm."
            )
        if key in field_names:
            kwargs[key] = value
        else:
            extra[key] = value
    section = cls(**kwargs)
    if extra:
        section.extra.update(extra)  # type: ignore[attr-defined]
    return section


def _coerce_to_mapping(source: Any) -> dict[str, Any]:
    """Normalise any accepted ``build_optimize_config`` source to a dict."""
    if isinstance(source, OptimizeConfig):
        # ``asdict`` recurses into nested dataclasses and deep-copies values,
        # giving a clean base to apply overrides onto.
        data = asdict(source)
        # A derived (non-explicit) output path must be re-derived after the
        # overrides are applied, so drop it here.
        if not data.pop("_output_path_explicit", False):
            data.get("export", {})["output_path"] = ""
        data.pop("_quant_cfg", None)
        return data
    if isinstance(source, CNNConfig):
        return _coerce_to_mapping(source.to_optimize_config())
    if isinstance(source, (str, Path)):
        return _load_config_file(source)
    if isinstance(source, Mapping):
        return copy.deepcopy(dict(source))
    raise TypeError(
        "source must be a dict, OptimizeConfig, CNNConfig, or a path to a "
        f"JSON/YAML file, got {type(source).__name__}"
    )


def _merge_overrides(base: MutableMapping[str, Any], overrides: Mapping[str, Any]) -> None:
    """Merge flat/nested override kwargs into ``base`` in place.

    Nested dict overrides are merged key-by-key so that passing
    ``quant={"mode": "fp8"}`` keeps the other ``quant`` settings intact.
    """
    for key, value in overrides.items():
        if key in _LEGACY_FLAT_KEYS:
            section_name, field_name = _LEGACY_FLAT_KEYS[key]
            section = base.get(section_name)
            if section is None:
                section = {}
                base[section_name] = section
            if not isinstance(section, dict):
                raise ValueError(f"Config section {section_name!r} must be a mapping")
            section[field_name] = copy.deepcopy(value)
        elif key in _SECTION_NAMES:
            if value is None:
                base[key] = None
                continue
            section = base.get(key)
            if section is None:
                section = {}
                base[key] = section
            if not isinstance(section, dict):
                raise ValueError(f"Config section {key!r} must be a mapping")
            section.update(copy.deepcopy(_section_to_dict(key, value)))
        else:
            base[key] = copy.deepcopy(value)


def _section_to_dict(name: str, value: Any) -> Mapping[str, Any]:
    """Coerce a section override (mapping or dataclass instance) to a mapping."""
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise ValueError(f"Config section {name!r} must be a mapping or dataclass")


def _apply_legacy_keys(
    data: dict[str, Any], sections: dict[str, dict[str, Any] | None]
) -> None:
    """Move legacy flat keys from ``data`` into their nested ``sections``."""
    for legacy_key, (section_name, field_name) in _LEGACY_FLAT_KEYS.items():
        if legacy_key not in data:
            continue
        value = data.pop(legacy_key)
        if value is None:
            continue
        section = sections[section_name]
        if section is None:
            section = {}
            sections[section_name] = section
        # An explicit nested value always wins over a legacy flat alias.
        section.setdefault(field_name, value)


def build_optimize_config(
    source: dict[str, Any] | str | Path | CNNConfig | OptimizeConfig | None = None,
    **overrides: Any,
) -> OptimizeConfig:
    """Build an :class:`OptimizeConfig` from a dict, file, or legacy config.

    ``source`` may be a mapping, a path to a JSON/YAML file, a legacy
    :class:`CNNConfig`, or an existing :class:`OptimizeConfig`. Keyword
    arguments in ``overrides`` are applied on top and accept both nested keys
    (``quant={"mode": "fp8"}``) and legacy flat keys (``quant_mode="fp8"``).

    A *new* :class:`OptimizeConfig` is always returned, even when ``source`` is
    already one, so overrides can never be observed half-applied.

    Raises:
        ValueError: If an unknown key is supplied, or a value fails validation
            (unknown quantization mode, malformed ``input_shape``, out-of-range
            ``sparsity``, or unavailable ModelOpt quantization preset).
        ImportError: If quantization is configured but ModelOpt is unavailable.
    """
    data = _coerce_to_mapping(source) if source is not None else {}
    data = _normalize_grouped_config(data)
    _merge_overrides(data, overrides)

    # Pull out the nested sections; unknown top-level keys go to ``extra``.
    sections: dict[str, dict[str, Any] | None] = {}
    for name in ("quant", "sparsity", "export", "calibration"):
        raw = data.pop(name, None)
        if raw is None:
            sections[name] = None
        elif isinstance(raw, Mapping) or is_dataclass(raw):
            sections[name] = dict(_section_to_dict(name, raw))
        else:
            raise ValueError(f"Config section {name!r} must be a mapping")

    _apply_legacy_keys(data, sections)

    # ``output_path`` is a top-level alias for ``export.output_path``.
    top_level_output_path = data.pop("output_path", None)
    export_values = sections["export"] or {}
    if top_level_output_path:
        export_values.setdefault("output_path", str(top_level_output_path))

    # ``algorithms`` is derived from configured sections, not stored separately.
    data.pop("algorithms", None)

    top_level_fields = {f.name for f in fields(OptimizeConfig) if not f.name.startswith("_")}
    unknown = sorted(set(data) - top_level_fields - _IGNORED_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown config override(s): {', '.join(unknown)}. "
            f"Allowed keys: {', '.join(sorted(top_level_fields))}"
        )

    config = OptimizeConfig(
        **data,
        quant=(
            _build_section(QuantConfig, sections["quant"])
            if sections["quant"] is not None
            else None
        ),
        sparsity=(
            _build_section(SparsityConfig, sections["sparsity"])
            if sections["sparsity"] is not None
            else None
        ),
        export=_build_section(ExportConfig, export_values),
        calibration=_build_section(CalibrationConfig, sections["calibration"] or {}),
        _output_path_explicit=bool(export_values.get("output_path")),
    )
    if config.quant is not None:
        config._quant_cfg = _build_modelopt_quant_cfg(config.quant)
    return config


def _build_modelopt_quant_cfg(quant: QuantConfig) -> dict[str, Any]:
    """Resolve and validate the ModelOpt preset while building the config."""
    try:
        import modelopt.torch.quantization as mtq
    except ImportError as exc:
        raise ImportError(
            "ModelOpt quantization backend is not available. Please install "
            "the vendored backend with 'pip install -e thirdparty/Model-Optimizer'."
        ) from exc

    preset_name = quant.preset or QUANT_MODES[quant.mode]
    preset = getattr(mtq, preset_name, None)
    if preset is None:
        raise ValueError(
            f"ModelOpt quant preset '{preset_name}' is not available in "
            "the installed modelopt version."
        )
    quant_cfg = copy.deepcopy(preset)
    if quant.weights is not None or quant.activations is not None:
        if quant.mode != "int8":
            raise ValueError(
                "quantization.weights and quantization.activations are currently "
                "supported only with mode 'int8'."
            )
        entries = quant_cfg.get("quant_cfg")
        if not isinstance(entries, list):
            raise ValueError("ModelOpt quant preset must define a quant_cfg list")
        for name, settings, default_axis in (
            ("weight_quantizer", quant.weights, 0),
            ("input_quantizer", quant.activations, 1),
        ):
            if settings is None:
                continue
            attributes: dict[str, Any] = {
                "num_bits": settings.bits,
                "unsigned": False,
                "narrow_range": False,
            }
            if settings.granularity == "per_tensor":
                attributes["axis"] = None
            elif settings.granularity == "per_channel":
                attributes["axis"] = (
                    settings.axis if settings.axis is not None else default_axis
                )
            else:
                attributes["block_sizes"] = {-1: settings.group_size}

            matching_entries = [
                entry
                for entry in entries
                if entry.get("quantizer_name") == f"*{name}"
            ]
            if not matching_entries:
                raise ValueError(
                    f"ModelOpt quant preset '{preset_name}' does not configure "
                    f"{name}."
                )
            for entry in matching_entries:
                quantizer_attributes = entry.setdefault("cfg", {})
                if "block_sizes" in attributes:
                    quantizer_attributes.pop("axis", None)
                else:
                    quantizer_attributes.pop("block_sizes", None)
                quantizer_attributes.update(attributes)
    quant_cfg.update(quant.extra)
    return quant_cfg


# Legacy config


@dataclass
class CNNConfig:
    """Legacy flat CNN optimization configuration.

    .. deprecated::
        Use :class:`OptimizeConfig` (built via :func:`build_optimize_config`)
        instead. This class is retained so existing scripts keep working; call
        :meth:`to_optimize_config` to convert.
    """

    model_name: str = "resnet"
    dataset_root: str = ""
    task: str = ""
    input_shape: tuple[int, ...] | None = None
    output_path: str = ""
    quant_mode: str = "int8"
    quant_preset: str | None = None
    opset_version: int = 17
    use_dynamic_axes: bool = True
    simplify: bool = True
    check_onnx: bool = True
    calibration_samples: int = 32
    calibration_batch_size: int = 1
    calibration_split: str = "train"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task:
            dataset = infer_dataset_from_root(self.dataset_root or ".")
            self.task = infer_task_from_model_and_dataset(self.model_name, dataset)
        if not self.output_path:
            self.output_path = build_default_output_path(self.model_name, self.quant_mode)

    def to_optimize_config(self) -> OptimizeConfig:
        """Convert this legacy config into an :class:`OptimizeConfig`."""
        return OptimizeConfig(
            model_name=self.model_name,
            dataset_root=self.dataset_root,
            task=self.task,
            input_shape=self.input_shape,
            quant=QuantConfig(
                mode=self.quant_mode,
                preset=self.quant_preset,
                extra=copy.deepcopy(self.extra),
            ),
            sparsity=None,
            export=ExportConfig(
                output_path=self.output_path,
                opset_version=self.opset_version,
                use_dynamic_axes=self.use_dynamic_axes,
                simplify=self.simplify,
                check_onnx=self.check_onnx,
            ),
            calibration=CalibrationConfig(
                num_samples=self.calibration_samples,
                batch_size=self.calibration_batch_size,
                split=self.calibration_split,
            ),
            extra=copy.deepcopy(self.extra),
            _output_path_explicit=bool(self.output_path),
        )
