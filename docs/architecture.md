# Architecture

This document describes how the CNN optimization path is put together. The
[README](https://github.com/liwei109/cks-optimizer#readme) covers installation
and everyday usage; this page covers the module layout and the per-stage API.

## Repository layout

The package intentionally lives at the repository root, without a nested `src/`
folder.

```text
cks_optimizer/
  cli.py           # cks-opt entrypoint (cnn / lm subcommands)
  cnn/
    __init__.py
    config.py      # OptimizeConfig + build_optimize_config (stage 1)
    models.py      # load_cnn_model (stage 2)
    datasets/
      __init__.py
      dataloaders.py # ImageNet / COCO / VOC transforms and split loading
      processing.py # data_process: dataset/task dispatch (stage 3)
      registry.py   # torchvision dataset construction
    optimizer.py   # optimize_cnn_model: quantize / sparsify (stage 4)
    export.py      # export_onnx (stage 5)
    evaluate.py    # evaluate (stage 6)
    pipeline.py    # run_cnn_pipeline: orchestrates all six stages
  lm/
    __init__.py
    optimizer.py   # LMOptimizer: optimize large models
thirdparty/
  Model-Optimizer/ # NVIDIA ModelOpt (git submodule)
  llm-compressor/  # vLLM llm-compressor (git submodule)
```

## The six stages

| # | Stage | Responsibility |
|---|-------|----------------|
| 1 | `build_optimize_config` | Normalize a YAML file, JSON file, or dict into an `OptimizeConfig` |
| 2 | `load_cnn_model` | Load a torchvision / timm model by name, or a local `.onnx` file |
| 3 | `data_process` | Dispatch a dataset/task pair and prepare samples/calibration tensors |
| 4 | `optimize_cnn_model` | Run ModelOpt `quantize` and/or `sparsify` |
| 5 | `export_onnx` | Export the optimized model and simplify the graph |
| 6 | `evaluate` | Measure task metrics against held-out dataset labels |

:func:`cks_optimizer.cnn.run_cnn_pipeline` drives all six in order and
returns an `OptimizationResult`. A configured `quant` section runs quantization;
a configured `sparsity` section runs sparsification. If both are present, the
pipeline runs `["sparsify", "quantize"]` — sparsity first, then quantization.

CNN recipes keep `task` at the top level, alongside the `model`, `data`,
`optimize`, and `export` sections. The `model` section describes only the model
itself; for example, use `task: segmentation` rather than nesting `task` under
`model`.

The `data_process` API dispatches to `ImagenetDataloader`, `CocoDataloader`,
or `VocDataloader` using the dataset/task pair. Supported pairs are
`imagenet`/`classification`, `coco`/`detection`, and `voc`/`segmentation`.
Callers can pass `dataset=` explicitly; otherwise the task selects its default
dataset. Mismatched pairs fail with a descriptive error.

INT8 recipes may define `weights` and `activations` under
`optimize.quantization`, with `bits`, `scheme`, and `granularity`
(`per_tensor`, `per_channel`, or `per_group`). `per_group` also requires a
positive `group_size`; `axis` optionally overrides the channel axis. The
current PyTorch ModelOpt backend supports symmetric integer quantization only,
so `scheme: asymmetric` is rejected rather than silently treated as symmetric.
Weight per-channel quantization defaults to axis 0, activation per-channel to
axis 1, and per-group quantization groups values along the last tensor axis.

Calibration batches use `data.calibration.batch_size` (or `--batch-size`), and
evaluation batches use `data.evaluation.batch_size` (or
`--evaluation-batch-size`). Evaluation defaults to 16 and runs the optimized
PyTorch model directly on CUDA by default; set `device: cpu` or pass
`--device cpu` to use the CPU. A missing CUDA device is reported rather than
silently falling back. Calibration and evaluation use the same device.
TorchMetrics computes
classification accuracy, COCO bbox mAP, and VOC mean IoU. The exported ONNX
artifact is still produced by the pipeline, but is not used for evaluation.

Two details worth knowing:

- **Stage 3 supplies the input tensors.** The calibration batches it returns are
  reused as the `sample_input` for stages 4 and 5, so `config.input_shape` is
  optional. When unset, the shape comes from model metadata or defaults to
  224 × 224.
- **Stage 6 uses a held-out split.** Classification reports top-1 accuracy,
  COCO reports bbox mAP, and VOC reports mean IoU.

## Running stages individually

Every stage is importable from `cks_optimizer.cnn`, which is useful when you
want to cache an artifact between steps or swap out one stage:

```python
import logging

from cks_optimizer.cnn import (
    OptimizeConfig,
    QuantConfig,
    SparsityConfig,
    data_process,
    evaluate,
    export_onnx,
    load_cnn_model,
    optimize_cnn_model,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = OptimizeConfig(
    model_name="mobilenetv2_100",
    dataset_root="datasets/imagenet",
    quant=QuantConfig(mode="int8"),
)

# Stage 2: the model is built from its name alone; no input shape required.
model = load_cnn_model("mobilenetv2_100")

# Stage 3: the model is passed so the input shape can be inferred from it
# when config.input_shape is not set.
data, sample_input = data_process(config, model=model, include_evaluation=True)

# Stage 4: run the configured algorithms, sparsity before quantization.
config.sparsity = SparsityConfig()
optimized = optimize_cnn_model(
    model,
    sample_input,
    config=config,
    calib_data=data.calib_data,
)

# Stage 5: export to ONNX.
onnx_path = export_onnx(optimized, sample_input, config)

# Stage 6: evaluate the optimized PyTorch model against the held-out split.
logger.info(
    "%s",
    evaluate(
        optimized,
        data.evaluation_dataset,
        task=config.task,
        config=config,
    ),
)
```

## Model sources

`load_cnn_model` accepts a torchvision or timm model name, or a local ONNX
file. The optional `model` argument to `run_cnn_pipeline` overrides the model
source in the configuration; it is not a PyTorch module instance.

```python
from cks_optimizer.cnn import OptimizeConfig, run_cnn_pipeline

config = OptimizeConfig(model_name="resnet18", dataset_root="datasets/imagenet")
result = run_cnn_pipeline(config, model="resnet18")
```

## `OptimizationResult`

The object returned by `run_cnn_pipeline` carries every stage's output:

| Field | Contents |
|-------|----------|
| `config` | The normalized `OptimizeConfig` used for the run |
| `data` | `ProcessedData` with calibration tensors and evaluation dataset |
| `model` | The loaded model before optimization |
| `sample_input` | The tensor used for tracing, export, and calibration |
| `optimized_model` | The model after quantization / sparsification |
| `onnx_path` | Path of the exported ONNX artifact |
| `algorithms` | Stages that actually ran, e.g. `("sparsify", "quantize")` |
| `evaluation` | Task metric from stage 6, or `None` if evaluation was skipped |

`OptimizationResult.summary()` renders these fields as multi-line text for
callers that want a concise human-readable report.

## Sparsity

`CNNOptimizer.sparsify` wraps `modelopt.torch.sparsity.sparsify` and returns the
sparsified model; exporting to ONNX remains a separate stage. ModelOpt's
magnitude sparsifier applies a fixed 2:4 pattern, so the requested ratio is
informational and a value other than `0.5` produces a warning. `sparsegpt` is
calibration-based and consumes the stage 3 tensors.

`LMOptimizer.optimize` applies the configured recipe through `Oneshot`. Recipes
can target large models including LLMs and VLMs.

## Dataset processing

Datasets are loaded through torchvision and selected by task. `data_process`
creates calibration batches from the configured training split and, when
requested, loads the held-out split for evaluation.

| Task | Dataset | Metric |
|------|---------|--------|
| `classification` | ImageNet | Top-1 accuracy |
| `detection` | COCO | Bounding-box mAP |
| `segmentation` | VOC 2012 | Mean IoU |

Small local examples are stored under the ignored repository-level
`datasets/imagenet`, `datasets/coco`, and `datasets/voc` directories. They
follow the dataset layouts but contain synthetic samples.

## Extending the pipeline

- **New task or dataset** — add its torchvision dataset construction,
  preprocessing, and task metric.
- **New model source** — extend `load_cnn_model` in `cnn/models.py`.
- **New quantization mode** — add an entry to `QUANT_MODES` in
  `cnn/config.py` mapping it to a ModelOpt default config, and add it to
  `TRT_ONLY_QUANT_MODES` if `onnxruntime` cannot execute the resulting graph.
