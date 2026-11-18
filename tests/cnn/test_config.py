from __future__ import annotations

import pytest

from cks_optimizer.cnn import (
    CalibrationConfig,
    OptimizeConfig,
    QuantConfig,
    QuantizerSettings,
    SparsityConfig,
    build_optimize_config,
)
from cks_optimizer.cnn.config import QUANT_MODES, TRT_ONLY_QUANT_MODES


def test_build_optimize_config_does_not_alias_source_config():
    original = OptimizeConfig(
        model_name="resnet50",
        quant=QuantConfig(mode="int8"),
        calibration=CalibrationConfig(num_samples=8),
    )

    derived = build_optimize_config(original, quant_mode="mxfp8", num_samples=2)

    assert derived is not original
    assert derived.quant.mode == "mxfp8"
    assert derived.calibration.num_samples == 2
    # The source object must be untouched.
    assert original.quant.mode == "int8"
    assert original.calibration.num_samples == 8
    assert original.output_path.endswith(".int8.onnx")
    assert derived.output_path.endswith(".mxfp8.onnx")


def test_nested_section_override_keeps_sibling_fields():
    base = {"model_name": "resnet50", "calibration": {"num_samples": 16}}

    cfg = build_optimize_config(base, calibration={"batch_size": 4})

    assert cfg.calibration.batch_size == 4
    assert cfg.calibration.num_samples == 16
    assert base["calibration"] == {"num_samples": 16}


def test_build_optimize_config_accepts_dataclass_section():
    cfg = build_optimize_config(
        {"model_name": "resnet50"},
        quant=QuantConfig(mode="mxint8"),
    )
    assert cfg.quant is not None
    assert cfg.quant.mode == "mxint8"
    assert cfg._quant_cfg


def test_build_optimize_config_resolves_quant_preset_and_extra():
    cfg = build_optimize_config(
        {
            "model_name": "resnet50",
            "quant": {
                "mode": "int8",
                "extra": {"custom_option": True},
            },
        }
    )
    assert cfg._quant_cfg
    assert cfg._quant_cfg["custom_option"] is True


def test_quantizer_settings_map_to_modelopt_attributes():
    import modelopt.torch.quantization as mtq

    cfg = build_optimize_config(
        {
            "model_name": "resnet50",
            "quant": {
                "mode": "int8",
                "weights": {
                    "bits": 6,
                    "scheme": "symmetric",
                    "granularity": "per_group",
                    "group_size": 32,
                },
                "activations": {
                    "bits": 8,
                    "scheme": "symmetric",
                    "granularity": "per_channel",
                },
            },
        }
    )
    weight_cfg = next(
        entry["cfg"]
        for entry in cfg._quant_cfg["quant_cfg"]
        if entry.get("quantizer_name") == "*weight_quantizer"
    )
    activation_cfg = next(
        entry["cfg"]
        for entry in cfg._quant_cfg["quant_cfg"]
        if entry.get("quantizer_name") == "*input_quantizer"
    )

    assert weight_cfg["num_bits"] == 6
    assert weight_cfg["block_sizes"] == {-1: 32}
    assert weight_cfg["unsigned"] is False
    assert weight_cfg["narrow_range"] is False
    assert activation_cfg["num_bits"] == 8
    assert activation_cfg["axis"] == 1
    mtq.QuantizeConfig(**cfg._quant_cfg)


def test_quantizer_settings_reject_asymmetric_scheme():
    with pytest.raises(ValueError, match="supports symmetric integer quantization only"):
        QuantizerSettings(scheme="asymmetric")


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"granularity": "per_group"}, "group_size is required"),
        (
            {"granularity": "per_tensor", "group_size": 32},
            "group_size is only valid",
        ),
    ],
)
def test_quantizer_settings_validate_granularity_options(settings, message):
    with pytest.raises(ValueError, match=message):
        QuantizerSettings(**settings)


def test_build_optimize_config_rejects_unavailable_quant_preset():
    with pytest.raises(ValueError, match="ModelOpt quant preset"):
        build_optimize_config(
            {"model_name": "resnet50", "quant": {"preset": "MISSING_PRESET"}}
        )


def test_algorithm_sections_are_optional_and_overridable():
    config = build_optimize_config({"model_name": "resnet50"})
    assert config.quant is None
    assert config.sparsity is None
    assert config.effective_algorithms == []

    sparse_only = build_optimize_config(
        config,
        sparsity={"mode": "sparse_magnitude"},
    )
    assert sparse_only.quant is None
    assert sparse_only.sparsity is not None
    assert sparse_only.effective_algorithms == ["sparsify"]


def test_build_optimize_config_rejects_non_mapping_section():
    with pytest.raises(ValueError, match="must be a mapping"):
        build_optimize_config({"model_name": "resnet50"}, quant="int8")


def test_build_optimize_config_rejects_unknown_quant_mode():
    with pytest.raises(ValueError, match="Unsupported quantization mode"):
        build_optimize_config({"model_name": "resnet50"}, quant_mode="int4")


def test_build_optimize_config_rejects_unknown_sparsity_mode():
    with pytest.raises(ValueError, match="Unsupported sparsity mode"):
        build_optimize_config(
            {"model_name": "resnet50"},
            sparsity={"mode": "wanda"},
        )


def test_output_path_follows_quant_mode_when_not_explicit():
    cfg = build_optimize_config({"model_name": "resnet50", "quant_mode": "int8"})
    assert cfg.output_path.endswith(".int8.onnx")
    cfg.quant_mode = "fp8"
    assert cfg.output_path.endswith(".fp8.onnx")


def test_explicit_output_path_survives_quant_mode_change():
    cfg = build_optimize_config({"model_name": "resnet50", "output_path": "out/keep.onnx"})
    cfg.quant_mode = "fp8"
    assert cfg.output_path == "out/keep.onnx"


def test_evaluation_supported_flags_trt_only_modes():
    for mode in TRT_ONLY_QUANT_MODES:
        cfg = OptimizeConfig(model_name="resnet50", quant=QuantConfig(mode=mode))
        assert cfg.evaluation_supported is False
    cfg = OptimizeConfig(model_name="resnet50", quant=QuantConfig(mode="int8"))
    assert cfg.evaluation_supported is True
    assert set(QUANT_MODES) >= {"int8", "fp8", "mxfp8"}


def test_config_sections_select_algorithms_and_calibration():
    cfg = OptimizeConfig(
        model_name="resnet50",
        quant=QuantConfig(mode="int8"),
        sparsity=SparsityConfig(mode="sparse_magnitude"),
    )
    assert cfg.effective_algorithms == ["sparsify", "quantize"]
    assert cfg.requires_calibration is True

    sparse_only = OptimizeConfig(
        model_name="resnet50", sparsity=SparsityConfig(mode="sparsegpt")
    )
    assert sparse_only.effective_algorithms == ["sparsify"]
    assert sparse_only.requires_calibration is True

    plain = OptimizeConfig(model_name="resnet50")
    assert plain.effective_algorithms == []
    assert plain.requires_calibration is False


def test_config_rejects_enabled_flag():
    with pytest.raises(ValueError, match="include or omit the config section"):
        build_optimize_config({"model_name": "resnet50", "quant": {"enabled": True}})


def test_explicit_dataset_pins_task():
    cfg = build_optimize_config({"model_name": "resnet50"}, dataset="voc")

    assert cfg.dataset == "voc"
    assert cfg.task == "segmentation"


def test_explicit_dataset_normalizes_aliases():
    cfg = OptimizeConfig(model_name="resnet50", dataset="coco2017")

    assert cfg.dataset == "coco"
    assert cfg.task == "detection"


def test_explicit_dataset_conflicting_task_raises():
    with pytest.raises(ValueError, match="serves task"):
        OptimizeConfig(model_name="resnet50", task="detection", dataset="voc")


def test_grouped_data_section_accepts_dataset():
    cfg = build_optimize_config(
        {"model": {"name": "resnet50"}, "data": {"dataset": "coco"}}
    )

    assert cfg.dataset == "coco"
    assert cfg.task == "detection"
