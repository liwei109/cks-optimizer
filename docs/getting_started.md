# Getting started

CKS Model Optimizer has a `cks-opt` command and a Python API. The CNN pipeline
loads a model, prepares task-specific data, optionally applies quantization or
sparsity, exports ONNX, and can evaluate the optimized model.

## Installation

Install the package and its vendored backends from the repository root:

```bash
python -m pip install -e .
python -m pip install -e thirdparty/Model-Optimizer
python -m pip install -e thirdparty/llm-compressor
python -m pip install -e thirdparty/onnx2torch
```

The CNN optimization backend requires NVIDIA Model Optimizer. ONNX model
loading requires `onnx2torch`; large-model optimization requires
`llm-compressor`. These are installed separately because they are vendored
backends. COCO evaluation also requires `pycocotools` (included in the project
dependencies, or install with `pip install -e '.[coco]'`).

## Command line

Run a CNN recipe:

```bash
cks-opt cnn --config recipes/cnn/base_cls.yaml
```

Run the COCO detection example:

```bash
cks-opt cnn --config recipes/cnn/base_det.yaml
```

The default CNN device is CUDA. Use `--device cpu` to select CPU explicitly.
Pass `--no-evaluate` to skip held-out evaluation. See
`cks-opt cnn --help` for the available command-line options.

## Python API

Use the same stages directly when you need to customize the workflow:

```python
from cks_optimizer.cnn import (
    build_optimize_config,
    data_process,
    evaluate,
    export_onnx,
    load_cnn_model,
    optimize_cnn_model,
)

config = build_optimize_config(
    {
        "model_name": "resnet18",
        "dataset_root": "datasets/imagenet",
        "task": "classification",
        "quant": {"mode": "int8"},
        "device": "cuda",
    }
)
model = load_cnn_model(config.model_name, pretrained=True, device=config.device)
data, sample_input = data_process(config, model=model, include_evaluation=True)
optimized = optimize_cnn_model(
    model, sample_input, config=config, calib_data=data.calib_data
)
onnx_path = export_onnx(optimized, sample_input, config)
metrics = evaluate(
    optimized,
    data.evaluation_dataset,
    task=config.task,
    config=config,
    device=config.device,
)
```

For the complete orchestration entry point, use
[`run_cnn_pipeline`](api) and configure the desired optimization in
[`OptimizeConfig`](configuration).

## Large-model optimization

Install the vendored llm-compressor backend, then run a recipe from the CLI:

```bash
cks-opt lm \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --recipe recipes/lm/base_int8.yaml \
  --dataset wikitext \
  --dataset-config-name wikitext-2-raw-v1 \
  --splits 'train[:1]' \
  --num-calibration-samples 1 \
  --max-seq-length 32 \
  --output-dir tmp/qwen2.5-0.5b-instruct.int8
```

The Python entry point is [`LMOptimizer`](api). The model and dataset are
loaded by llm-compressor, so their availability and any remote-code settings
are controlled by that backend.
