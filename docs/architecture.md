# Architecture

This document describes how the CNN optimization path is put together. The
[README](../README.md) covers installation and everyday usage; this page covers
the module layout and the per-stage API.

## Repository layout

The package intentionally lives at the repository root, without a nested `src/`
folder.

```text
cks_optimizer/
  cli.py           # cks-opt entrypoint (cnn / lm subcommands)
  cnn/
    __init__.py
    config.py      # OptimizeConfig + build_optimize_config (stage 1)
    models.py      # load_cnn_model / load_model (stage 2)
    datasets/
      __init__.py
      processing.py  # data_process (stage 3)
      registry.py
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
| 2 | `load_cnn_model` | Load a torchvision / timm model by name, or a local `.pt` / `.pth` / `.onnx` file |
| 3 | `data_process` | Turn a dataset root into samples and calibration tensors |
| 4 | `optimize_cnn_model` | Run ModelOpt `quantize` and/or `sparsify` |
| 5 | `export_onnx` | Export the optimized model and simplify the graph |
| 6 | `evaluate` | Measure task metrics against held-out dataset labels |

[run_cnn_pipeline](../cks_optimizer/cnn/pipeline.py) drives all six in order and
returns an `OptimizationResult`. A configured `quant` section runs quantization;
a configured `sparsity` section runs sparsification. If both are present, the
pipeline runs `["sparsify", "quantize"]` — sparsity first, then quantization.

CNN recipes keep `task` at the top level, alongside the `model`, `data`,
`optimize`, and `export` sections. The `model` section describes only the model
itself; for example, use `task: segmentation` rather than nesting `task` under
`model`.

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
`--evaluation-batch-size`). Evaluation defaults to 16. The pipeline enables
dynamic batches when loading static-batch-one YOLOv5 ONNX models; other static
ONNX inputs are safely limited to their declared batch size.

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

# Stage 6: evaluate against the held-out split.
logger.info(
    "%s",
    evaluate(
        onnx_path,
        data.evaluation_dataset,
        task=config.task,
        config=config,
    ),
)
```

## Running the pipeline on an in-memory model

`run_cnn_pipeline` accepts any `torch.nn.Module` via `model`, which is how
the tests and [examples/cnn_example.py](../examples/cnn_example.py) run the full
pipeline without downloading weights:

```python
import logging

import torch

from cks_optimizer.cnn import ExportConfig, OptimizeConfig, QuantConfig, run_cnn_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TinyResNetLike(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.backbone(x)


result = run_cnn_pipeline(
    OptimizeConfig(
        model_name="resnet",
        dataset_root="datasets/imagenet",
        quant=QuantConfig(mode="int8"),
        export=ExportConfig(output_path="tmp/resnet.int8.onnx", use_dynamic_axes=False),
    ),
    model=TinyResNetLike().eval(),
)

logger.info("%s", result.summary())
logger.info("Quantized ONNX saved to: %s", result.onnx_path)
```

## `OptimizationResult`

The object returned by `run_cnn_pipeline` carries every stage's output:

| Field | Contents |
|-------|----------|
| `config` | The normalized `OptimizeConfig` used for the run |
| `data` | `ProcessedData` with calibration tensors and evaluation dataset |
| `model` | The loaded and optimized model |
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
  [cnn/config.py](../cks_optimizer/cnn/config.py) mapping it to a ModelOpt
  default config, and add it to `TRT_ONLY_QUANT_MODES` if `onnxruntime` cannot
  execute the resulting graph.
