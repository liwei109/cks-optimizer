from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from cks_optimizer.cnn import (
    CNNConfig,
    ExportConfig,
    OptimizeConfig,
    QuantConfig,
    SparsityConfig,
    build_optimize_config,
    export_onnx,
    load_model,
    optimize_cnn_model,
    run_cnn_pipeline,
)
from cks_optimizer.cnn.config import (
    build_default_output_path,
    infer_dataset_from_root,
    infer_task_from_model_and_dataset,
)
from cks_optimizer.cnn.export import ensure_parent_dir


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(8, 10),
        )

    def forward(self, x):
        return self.net(x)


# Configuration


def test_build_default_output_path():
    assert build_default_output_path("resnet50", "int8") == "tmp/resnet50.int8.onnx"
    assert build_default_output_path("my model!", "fp8") == "tmp/my-model.fp8.onnx"
    # Local paths / hub ids must not leak their directory into the file name.
    assert (
        build_default_output_path("tmp/resnet50.onnx", "int8")
        == "tmp/resnet50.int8.onnx"
    )
    assert (
        build_default_output_path("microsoft/resnet-50", "int8")
        == "tmp/resnet-50.int8.onnx"
    )


def test_infer_dataset_from_root(tmp_path):
    assert infer_dataset_from_root(str(tmp_path)) == "imagenet"

    voc = tmp_path / "voc"
    (voc / "JPEGImages").mkdir(parents=True)
    (voc / "SegmentationClass").mkdir(parents=True)
    assert infer_dataset_from_root(str(voc)) == "voc"

    coco = tmp_path / "coco"
    coco.mkdir()
    (coco / "instances.json").write_text("{}")
    assert infer_dataset_from_root(str(coco)) == "coco"

    imagenet = tmp_path / "imagenet"
    imagenet.mkdir()
    assert infer_dataset_from_root(str(imagenet)) == "imagenet"


def test_infer_dataset_checks_common_nested_metadata_without_recursive_walk(
    tmp_path, monkeypatch
):
    coco = tmp_path / "coco" / "train2017" / "annotations"
    coco.mkdir(parents=True)
    (coco / "instances_train2017.json").write_text("{}")

    def reject_recursive_walk(*args, **kwargs):
        raise AssertionError(
            "dataset inference must not recursively walk the data root"
        )

    monkeypatch.setattr(Path, "rglob", reject_recursive_walk)

    assert infer_dataset_from_root(tmp_path / "coco") == "coco"


def test_infer_task_from_model_and_dataset():
    assert infer_task_from_model_and_dataset("resnet50", "imagenet") == "classification"
    assert infer_task_from_model_and_dataset("yolov5", "imagenet") == "classification"
    assert infer_task_from_model_and_dataset("resnet50", "coco") == "detection"
    assert infer_task_from_model_and_dataset("resnet50", "voc") == "segmentation"


def test_optimize_config_post_init_fills_defaults(tmp_path):
    config = OptimizeConfig(
        model_name="resnet50",
        dataset_root=str(tmp_path),
    )
    assert config.task == "classification"
    assert infer_dataset_from_root(config.dataset_root) == "imagenet"
    assert config.output_path == "tmp/resnet50.optimized.onnx"


def test_build_optimize_config_from_dict():
    config = build_optimize_config(
        {
            "model_name": "resnet50",
            "quant": {"mode": "fp8"},
            "export": {"output_path": "out/model.onnx"},
        }
    )
    assert config.model_name == "resnet50"
    assert config.quant.mode == "fp8"
    assert config.export.output_path == "out/model.onnx"


def test_build_optimize_config_accepts_task_alongside_grouped_sections():
    config = build_optimize_config(
        {
            "task": "segmentation",
            "model": {"name": "deeplabv3", "input_shape": [1, 3, 520, 520]},
            "data": {"root": "datasets/voc"},
        }
    )

    assert config.task == "segmentation"
    assert config.model_name == "deeplabv3"


def test_build_optimize_config_reads_evaluation_batch_size():
    config = build_optimize_config(
        {
            "model": {"name": "detector"},
            "data": {"evaluation": {"split": "val2017", "batch_size": 8}},
        }
    )

    assert config.evaluation_split == "val2017"
    assert config.evaluation_batch_size == 8


def test_build_optimize_config_rejects_nonpositive_evaluation_batch_size():
    with pytest.raises(ValueError, match="evaluation_batch_size must be >= 1"):
        build_optimize_config({"evaluation_batch_size": 0})


def test_build_optimize_config_rejects_task_inside_model_section():
    with pytest.raises(ValueError, match="Unknown model config key 'task'"):
        build_optimize_config(
            {"model": {"name": "deeplabv3", "task": "segmentation"}}
        )


def test_build_optimize_config_legacy_flat_keys():
    config = build_optimize_config(
        {
            "model_name": "resnet50",
            "quant_mode": "fp8",
            "output_path": "out/model.onnx",
            "calibration_samples": 16,
        }
    )
    assert config.quant.mode == "fp8"
    assert config.export.output_path == "out/model.onnx"
    assert config.calibration.num_samples == 16


def test_build_optimize_config_from_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "model_name: resnet50\nquant:\n  mode: int8\nexport:\n  output_path: out/model.onnx\n"
    )
    config = build_optimize_config(str(path))
    assert config.model_name == "resnet50"
    assert config.quant.mode == "int8"
    assert config.export.output_path == "out/model.onnx"


@pytest.mark.parametrize(
    "recipe",
    sorted((Path(__file__).resolve().parents[2] / "recipes" / "cnn").glob("*.yaml")),
    ids=lambda path: path.name,
)
def test_shipped_recipes_are_valid(recipe):
    config = build_optimize_config(str(recipe))
    assert config.model_name
    assert config.dataset_root
    expected_tasks = {
        "base_cls.yaml": "classification",
        "base_det.yaml": "detection",
        "base_seg.yaml": "segmentation",
        "base_sparse_cls.yaml": "classification",
    }
    assert config.task == expected_tasks[recipe.name]
    if recipe.name == "base_det.yaml":
        assert config.calibration.split == "val2017"
        assert config.evaluation_split == "val2017"
        assert config.evaluation_batch_size == 8
        assert config.export.output_names == ["output0"]
    if recipe.name == "base_sparse_cls.yaml":
        assert config.quant is None
        assert config.sparsity is not None
    else:
        assert config.quant.weights.bits == 8
        assert config.quant.weights.scheme == "symmetric"
        assert config.quant.weights.granularity == "per_channel"
        assert config.quant.activations.bits == 8
        assert config.quant.activations.scheme == "symmetric"
        assert config.quant.activations.granularity == "per_tensor"
    assert config.effective_algorithms
    if recipe.name == "base_sparse_cls.yaml":
        assert config.quant is None
        assert config.effective_algorithms == ["sparsify"]
    else:
        assert config.quant is not None


def test_build_optimize_config_from_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"model_name": "resnet50", "quant": {"mode": "fp8"}}')
    config = build_optimize_config(str(path))
    assert config.model_name == "resnet50"
    assert config.quant.mode == "fp8"


def test_build_optimize_config_overrides():
    config = build_optimize_config(
        {"model_name": "resnet50"},
        model_name="resnet101",
    )
    assert config.model_name == "resnet101"


def test_build_optimize_config_rejects_unknown_override():
    with pytest.raises(ValueError, match="Unknown config override"):
        build_optimize_config({"model_name": "resnet50"}, bogus=1)


@pytest.mark.parametrize("kind_key", ["model_type", "dataset_type"])
def test_build_optimize_config_rejects_manual_type_fields(kind_key):
    with pytest.raises(ValueError, match="Unknown config override"):
        build_optimize_config({"model_name": "resnet50", kind_key: "manual"})


def test_effective_algorithms_derived_from_config_sections():
    assert OptimizeConfig().effective_algorithms == []
    assert OptimizeConfig(
        sparsity=SparsityConfig(),
    ).effective_algorithms == ["sparsify"]
    # Both configured -> sparsity first, then quantize.
    cfg = OptimizeConfig(
        quant=QuantConfig(mode="int8"),
        sparsity=SparsityConfig(),
    )
    assert cfg.effective_algorithms == ["sparsify", "quantize"]


def test_build_optimize_config_drops_algorithms_key():
    # Algorithms are derived from configured sections, not stored in ``extra``.
    config = build_optimize_config(
        {
            "model_name": "resnet50",
            "quant": {"mode": "int8"},
            "sparsity": {},
            "algorithms": ["sparsify", "quantize"],
        }
    )
    assert "algorithms" not in config.extra
    assert config.effective_algorithms == ["sparsify", "quantize"]


def test_legacy_cnn_config_converts():
    legacy = CNNConfig(
        model_name="resnet50",
        quant_mode="fp8",
        output_path="out/model.onnx",
    )
    config = legacy.to_optimize_config()
    assert config.model_name == "resnet50"
    assert config.quant.mode == "fp8"
    assert config.export.output_path == "out/model.onnx"


# Model loading


def test_load_model_accepts_module():
    model, sample = load_model(TinyModel().eval(), input_shape=(1, 3, 32, 32))
    assert isinstance(model, torch.nn.Module)
    assert tuple(sample.shape) == (1, 3, 32, 32)


def test_load_model_rejects_empty_spec():
    with pytest.raises(ValueError, match="cannot be empty"):
        load_model("   ")


def test_load_cnn_model_detects_uppercase_onnx_suffix(tmp_path, monkeypatch):
    from cks_optimizer.cnn import models

    model_path = tmp_path / "model.ONNX"
    model_path.write_bytes(b"onnx")
    model = TinyModel().eval()
    monkeypatch.setattr(models, "_load_local_onnx_model", lambda path: model)

    assert models.load_cnn_model(model_path) is model


def test_load_cnn_model_returns_model_only():
    from cks_optimizer.cnn import load_cnn_model

    model = load_cnn_model(TinyModel().eval())
    assert isinstance(model, torch.nn.Module)
    assert next(model.parameters()).device.type == "cpu"


def test_infer_model_input_shape_returns_none_when_unknown():
    from cks_optimizer.cnn import infer_model_input_shape

    assert infer_model_input_shape(TinyModel().eval()) is None

    class Sized(torch.nn.Module):
        default_cfg = {"input_size": (3, 160, 160)}

        def forward(self, x):
            return x

    assert infer_model_input_shape(Sized(), "some_model") == (1, 3, 160, 160)


def test_load_model_onnx_requires_onnx2torch(tmp_path, monkeypatch):
    # Ensure onnx2torch is not importable so we hit the clear error path.
    monkeypatch.setitem(__import__("sys").modules, "onnx2torch", None)
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake")
    with pytest.raises(ImportError, match="onnx2torch"):
        load_model(str(onnx_path))


def _write_tiny_onnx(path) -> None:
    """Write a minimal 2D-conv ONNX model usable by onnx2torch."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    weights = np.random.default_rng(0).random((8, 3, 3, 3)).astype(np.float32)
    # raw=True keeps the payload in TensorProto.raw_data, which is what the
    # external-data conversion looks at.
    init = helper.make_tensor(
        "w", TensorProto.FLOAT, [8, 3, 3, 3], weights.tobytes(), raw=True
    )
    node = helper.make_node(
        "Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
    )
    graph = helper.make_graph(
        [node],
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 8, 8])],
        [init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, str(path))


def test_load_model_onnx_converts_via_onnx2torch(tmp_path):
    pytest.importorskip("onnx2torch")
    from cks_optimizer.cnn import infer_model_input_shape

    onnx_path = tmp_path / "tiny.onnx"
    _write_tiny_onnx(onnx_path)

    model, sample_input = load_model(str(onnx_path), input_shape=[1, 3, 8, 8])
    assert isinstance(model, torch.nn.Module)
    assert tuple(sample_input.shape) == (1, 3, 8, 8)
    with torch.no_grad():
        assert tuple(model(sample_input).shape) == (1, 8, 8, 8)
    # The declared graph shape is recorded so the pipeline does not have to
    # guess the input shape from the calibration dataset.
    assert infer_model_input_shape(model) == (1, 3, 8, 8)


def test_load_model_onnx_with_external_data(tmp_path):
    pytest.importorskip("onnx2torch")
    import onnx

    onnx_path = tmp_path / "tiny_external.onnx"
    _write_tiny_onnx(onnx_path)
    # Move the weights to an external data file; onnx2torch's own shape
    # inference cannot resolve those, so the loader must pre-load the proto.
    model = onnx.load(str(onnx_path))
    onnx.save_model(
        model,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="tiny_external.data",
        size_threshold=0,
    )
    external = onnx.load(str(onnx_path), load_external_data=False)
    assert all(
        tensor.data_location == onnx.TensorProto.EXTERNAL
        for tensor in external.graph.initializer
    )

    loaded, sample_input = load_model(str(onnx_path), input_shape=[1, 3, 8, 8])
    with torch.no_grad():
        assert tuple(loaded(sample_input).shape) == (1, 8, 8, 8)


# Optimization


def test_optimize_cnn_model_returns_model(tmp_path):
    model = TinyModel().eval()
    sample_input = torch.randn(1, 3, 32, 32)

    config = OptimizeConfig(
        model_name="tiny",
        quant=QuantConfig(mode="int8"),
        export=ExportConfig(
            output_path=str(tmp_path / "tiny.onnx"),
            use_dynamic_axes=False,
        ),
    )

    result = optimize_cnn_model(model, sample_input, config=config)
    assert isinstance(result, torch.nn.Module)


def test_optimize_cnn_model_sparsify_then_quantize(tmp_path):
    # Both configured -> derived ["sparsify", "quantize"].
    model = TinyModel().eval()
    sample_input = torch.randn(1, 3, 32, 32)

    config = OptimizeConfig(
        model_name="tiny",
        quant=QuantConfig(mode="int8"),
        sparsity=SparsityConfig(
            mode="sparse_magnitude",
            sparsity=0.5,
        ),
        export=ExportConfig(
            output_path=str(tmp_path / "tiny_sparse_quant.onnx"),
            use_dynamic_axes=False,
        ),
    )

    result = optimize_cnn_model(
        model,
        sample_input,
        config=config,
        sparsify={"mode": "sparse_magnitude", "sparsity": 0.5},
    )
    assert isinstance(result, torch.nn.Module)


def test_optimize_cnn_model_sparsify_from_config(tmp_path):
    # Algorithms are selected by configured sections.
    model = TinyModel().eval()
    sample_input = torch.randn(1, 3, 32, 32)

    config = OptimizeConfig(
        model_name="tiny",
        quant=QuantConfig(mode="int8"),
        sparsity=SparsityConfig(
            mode="sparse_magnitude",
            sparsity=0.5,
        ),
        export=ExportConfig(
            output_path=str(tmp_path / "tiny_cfg_sparse.onnx"),
            use_dynamic_axes=False,
        ),
    )

    result = optimize_cnn_model(model, sample_input, config=config)
    assert isinstance(result, torch.nn.Module)


def test_sparse_recipe_runs_modelopt_sparsification():
    recipe = (
        Path(__file__).resolve().parents[2] / "recipes" / "cnn" / "base_sparse_cls.yaml"
    )
    config = build_optimize_config(str(recipe))

    class SparseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(128, 128, bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(
                    torch.arange(1, 128 * 128 + 1, dtype=torch.float32).reshape(
                        128, 128
                    )
                )

        def forward(self, inputs):
            return self.linear(inputs)

    model = SparseModel().eval()
    before = torch.count_nonzero(model.linear.weight).item()
    result = optimize_cnn_model(
        model,
        torch.randn(1, 128),
        config=config,
    )

    assert config.effective_algorithms == ["sparsify"]
    assert isinstance(result, SparseModel)
    assert torch.count_nonzero(result.linear.weight).item() == before // 2


def test_optimize_cnn_model_rejects_no_enabled_algorithms():
    model = TinyModel().eval()
    sample_input = torch.randn(1, 3, 32, 32)

    config = OptimizeConfig(model_name="tiny")

    with pytest.raises(ValueError, match="No optimization algorithm configured"):
        optimize_cnn_model(model, sample_input, config=config)


def test_optimize_cnn_model_dispatches_on_config_sections():
    # Dispatch is driven by configured sections (sparsity first).
    model = TinyModel().eval()
    sample_input = torch.randn(1, 3, 32, 32)

    config = OptimizeConfig(
        model_name="tiny",
        quant=QuantConfig(mode="int8"),
        sparsity=SparsityConfig(
            mode="sparse_magnitude",
            sparsity=0.5,
        ),
    )
    assert config.effective_algorithms == ["sparsify", "quantize"]

    result = optimize_cnn_model(model, sample_input, config=config)
    assert isinstance(result, torch.nn.Module)


# ONNX export


def test_export_onnx_creates_file(tmp_path):
    model = TinyModel().eval()
    sample_input = torch.randn(1, 3, 32, 32)
    output_path = tmp_path / "tiny.onnx"

    config = OptimizeConfig(
        model_name="tiny",
        export=ExportConfig(
            output_path=str(output_path),
            use_dynamic_axes=False,
        ),
    )

    result = export_onnx(model, sample_input, config=config)
    assert result == output_path
    assert result.exists()
    assert result.suffix == ".onnx"
    assert os.path.getsize(result) > 0


# Pipeline


def test_run_cnn_pipeline_end_to_end(tmp_path):
    from PIL import Image

    root = tmp_path / "images"
    for split in ("train", "val"):
        class_a = root / split / "cat"
        class_a.mkdir(parents=True)
        image = Image.new("RGB", (64, 64), color=(120, 120, 120))
        image.save(str(class_a / "cat1.png"))

    config = OptimizeConfig(
        model_name="tiny",
        dataset_root=str(root),
        input_shape=(1, 3, 32, 32),
        quant=QuantConfig(mode="int8"),
        export=ExportConfig(
            output_path=str(tmp_path / "pipeline.onnx"),
            use_dynamic_axes=False,
        ),
    )

    # Quant-only config selects quantization.
    result = run_cnn_pipeline(
        config=config,
        model=TinyModel().eval(),
    )

    assert result.algorithms == ("quantize",)
    assert result.onnx_path is not None
    assert result.onnx_path.exists()
    assert result.evaluation is not None
    assert result.evaluation.num_samples >= 1
    assert result.evaluation.task == "classification"
    assert result.evaluation.accuracy is not None


def test_run_cnn_pipeline_evaluates_without_optimization_or_calibration(tmp_path):
    from PIL import Image

    image_path = tmp_path / "images" / "val" / "cat" / "sample.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(120, 120, 120)).save(image_path)
    output_path = tmp_path / "tiny-evaluation.onnx"
    config = OptimizeConfig(
        model_name="tiny",
        dataset_root=str(tmp_path / "images"),
        input_shape=(1, 3, 32, 32),
        export=ExportConfig(use_dynamic_axes=False),
    )
    config.output_path = str(output_path)

    result = run_cnn_pipeline(config=config, model=TinyModel().eval())

    assert result.algorithms == ()
    assert result.data.calib_data == []
    assert result.data.num_samples == 0
    assert result.sample_input.shape == (1, 3, 32, 32)
    assert result.onnx_path == output_path
    assert result.evaluation is not None
    assert result.evaluation.num_samples == 1


def test_run_cnn_pipeline_sample_input_comes_from_dataset(tmp_path):
    from PIL import Image

    root = tmp_path / "images"
    for split in ("train", "val"):
        (root / split / "cat").mkdir(parents=True)
    # A distinctive constant image so we can tell it apart from randn noise.
    for split in ("train", "val"):
        Image.new("RGB", (64, 64), color=(255, 255, 255)).save(
            str(root / split / "cat" / "a.png")
        )

    config = OptimizeConfig(
        model_name="tiny",
        dataset_root=str(root),
        input_shape=(1, 3, 32, 32),
        quant=QuantConfig(mode="int8"),
        export=ExportConfig(
            output_path=str(tmp_path / "pipeline.onnx"),
            use_dynamic_axes=False,
        ),
    )
    result = run_cnn_pipeline(config=config, model=TinyModel().eval())

    assert result.sample_input is not None
    assert result.sample_input is result.data.calib_data[0]
    # White images normalize to a known per-channel constant, never to randn noise.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    expected = (torch.ones(3, 32, 32) - mean) / std
    assert torch.allclose(result.sample_input, expected)


def test_run_cnn_pipeline_uses_processed_data_as_sample_input(tmp_path):
    from PIL import Image

    image_path = tmp_path / "train" / "cat" / "sample.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(image_path)
    config = OptimizeConfig(
        model_name="tiny",
        dataset_root=str(tmp_path),
        input_shape=(1, 3, 32, 32),
        quant=QuantConfig(mode="int8"),
        export=ExportConfig(
            output_path=str(tmp_path / "pipeline.onnx"),
            use_dynamic_axes=False,
        ),
    )
    result = run_cnn_pipeline(
        config=config,
        model=TinyModel().eval(),
        evaluate_model=False,
    )

    assert tuple(result.sample_input.shape) == (1, 3, 32, 32)
    assert result.sample_input is result.data.calib_data[0]


def test_run_cnn_pipeline_skips_evaluation(tmp_path):
    from PIL import Image

    image_path = tmp_path / "train" / "cat" / "sample.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(image_path)
    config = OptimizeConfig(
        model_name="tiny",
        dataset_root=str(tmp_path),
        input_shape=(1, 3, 32, 32),
        quant=QuantConfig(mode="int8"),
        export=ExportConfig(
            output_path=str(tmp_path / "pipeline.onnx"),
            use_dynamic_axes=False,
        ),
    )

    result = run_cnn_pipeline(
        config=config,
        model=TinyModel().eval(),
        evaluate_model=False,
    )

    assert result.evaluation is None
    assert result.onnx_path is not None
    assert result.onnx_path.exists()


# Miscellaneous


def test_config_and_output_file_are_consistent(tmp_path):
    # dataset_root is pinned so inference cannot pick up stray files in CWD.
    config = OptimizeConfig(
        dataset_root=str(tmp_path),
        export=ExportConfig(output_path=str(tmp_path / "custom.onnx")),
    )
    assert config.export.output_path.endswith("custom.onnx")
    assert config.task == "classification"


def test_ensure_parent_dir_creates_parent_folders(tmp_path):
    target = tmp_path / "nested" / "artifact" / "model.onnx"
    result = ensure_parent_dir(target)
    assert result == target
    assert result.parent.exists()
