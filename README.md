# CKS Model Optimizer

A model optimization toolkit for two major scenarios:

1. CNN / CV models using ModelOpt torch-mode quantization
2. LLM compression workflows using llmcompressor

This project is intentionally organized as a root-level package without a `src/` wrapper, as requested.

## Design overview

- `cks_optimizer/cnn/` for traditional vision models such as ResNet, YOLO, UNet, Swin, etc.
- `cks_optimizer/llm/` for LLM compression using llmcompressor
- `cks_optimizer/cli.py` for developer-facing CLI entrypoints

## CNN path

The CNN path supports:

- torch model inputs
- ONNX model inputs
- ModelOpt-based quantization in torch mode
- ONNX export of the quantized model
- dataset adapters for classification, detection, segmentation and custom data

### CNN package structure

```text
cks_optimizer/
  cnn/
    __init__.py
    api.py
    config.py
    modelopt_adapter.py
    tasks.py
    pipeline.py
    data/
      __init__.py
      base.py
      registry.py
```

### CNN Python example

```python
import torch
from cks_optimizer.cnn import CNNConfig, optimize_cnn_model


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


model = TinyResNetLike().eval()
sample_input = torch.randn(1, 3, 224, 224)
config = CNNConfig(
    model_name="resnet",
    dataset_root="./tmp",
    quant_mode="int8",
    output_path="artifacts/resnet_quantized.onnx",
    use_dynamic_axes=False,
)

result = optimize_cnn_model(model, sample_input, config=config)
print(f"Saved quantized ONNX to: {result}")
```

### CNN CLI example

```bash
python -m cks_optimizer cnn \
  --model mobilenetv2_100 \
  --dataset ./tmp \
  --quant-mode int8
```

The CLI automatically infers the dataset kind and task from the model/data context and writes the result to `tmp/mobilenetv2_100.int8.onnx` by default. You can still override the output path with `--output`, but it is not required.

## LLM path

The LLM path uses `llmcompressor` as the backend. It validates model identifiers and calls the compression workflow with calibration dataset arguments.

### LLM Python example

```python
from cks_optimizer.llm.llmcompressor_adapter import LLMCompressorAdapter

adapter = LLMCompressorAdapter()
result = adapter.compress(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    output_dir="artifacts/llm_example",
    recipe="quant_8bit",
    dataset="wikitext",
    dataset_config_name="wikitext-2-raw-v1",
    splits="train[:1]",
    num_calibration_samples=1,
    max_seq_length=32,
    trust_remote_code_model=True,
)
print(result)
```

### LLM CLI example

```bash
python -m cks_optimizer llm \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset wikitext \
  --dataset-config-name wikitext-2-raw-v1 \
  --splits train[:1] \
  --num-calibration-samples 1 \
  --max-seq-length 32 \
  --output-dir artifacts/llm_example
```

## Dataset support

The CNN pipeline includes dataset adapters for the following categories:

- `imagefolder` for ImageFolder-style classification datasets
- `coco` for COCO detection / segmentation annotations
- `voc` for Pascal VOC-style annotations
- `custom` for manifest-based or CSV-based custom datasets

## Notes

- CNN optimization is the core path and produces quantized ONNX artifacts.
- LLM compression is implemented through a thin adapter around llmcompressor.
- The project intentionally keeps the package at the repository root without a nested `src/` folder.
