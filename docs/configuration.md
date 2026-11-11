# Configuration

The CNN configuration can be supplied as a Python mapping, a YAML or JSON
file, an `OptimizeConfig`, or the legacy `CNNConfig`. The recommended format
is YAML with `task`, `model`, `data`, `optimize`, and `export` sections.

## Recipe structure

```yaml
task: classification

model:
  name: resnet50
  pretrained: true
  input_shape: [1, 3, 224, 224]

data:
  root: datasets/imagenet
  calibration:
    split: train
    num_samples: 128
    batch_size: 8
  evaluation:
    split: val
    batch_size: 16

optimize:
  quantization:
    mode: int8
    weights:
      bits: 8
      scheme: symmetric
      granularity: per_channel
    activations:
      bits: 8
      scheme: symmetric
      granularity: per_tensor

export:
  output_path: tmp/resnet50.int8.onnx
  opset_version: 17
```

`optimize.quantization` and `optimize.sparsity` are independent optional
sections. If both are configured, sparsity runs before quantization.
Calibration data is required for quantization and for calibration-based
sparsification such as `sparsegpt`. Evaluation uses the configured evaluation
split and batch size.

## Supported tasks and datasets

| Task | Dataset | Evaluation metric |
|---|---|---|
| `classification` | ImageNet | Top-1 accuracy |
| `detection` | COCO | Bounding-box mAP |
| `segmentation` | VOC 2012 | Mean IoU |

Supported dataset/task pairs are ImageNet/classification, COCO/detection, and
VOC/segmentation. The dataset may be inferred from its root directory or
specified explicitly. Dataset layouts and split names must follow the
corresponding dataset conventions.

## Quantization and sparsity

Quantizer settings support `bits`, `scheme`, and `granularity`. The PyTorch
ModelOpt backend currently supports symmetric quantization with
`per_tensor`, `per_channel`, or `per_group` granularity. `per_group` requires
`group_size`; `axis` may be used for per-channel quantization.

Supported quantization and sparsity modes are exposed as
`QUANT_MODES` and `SPARSITY_MODES` in the API. Backend-specific settings can
be supplied through the `extra` mapping on their configuration objects.

## Python configuration

```python
from cks_optimizer.cnn import (
    CalibrationConfig,
    ExportConfig,
    OptimizeConfig,
    QuantConfig,
    QuantizerSettings,
    SparsityConfig,
)

config = OptimizeConfig(
    model_name="resnet50",
    dataset_root="datasets/imagenet",
    task="classification",
    quant=QuantConfig(
        mode="int8",
        weights=QuantizerSettings(granularity="per_channel"),
    ),
    calibration=CalibrationConfig(num_samples=128, batch_size=8),
    export=ExportConfig(output_path="tmp/resnet50.int8.onnx"),
)
```

Use [`build_optimize_config`](api) to load a recipe and apply overrides.
Configuration classes and their constructor signatures are listed in the
[API reference](api).
