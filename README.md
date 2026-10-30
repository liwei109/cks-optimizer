# CKS Model Optimizer

[![CI](https://github.com/liwei109/cks-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/liwei109/cks-optimizer/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](./LICENSE)

CKS Model Optimizer is a model optimization toolkit with two backends:

- **CNN / CV models** — quantization and sparsity via [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer), exported to ONNX
- **Large models** — optimization recipes for LLMs, VLMs, and related models via [llm-compressor](https://github.com/vllm-project/llm-compressor), saved in `compressed-tensors` format

Both backends can be used from the `cks-opt` CLI or Python API.

## Installation

Clone the repository with its submodules, then install the project and backends from the repository root:

```bash
git clone --recurse-submodules https://github.com/liwei109/cks-optimizer.git
cd cks-optimizer
python -m pip install -e ".[coco]" \
  -e thirdparty/Model-Optimizer \
  -e thirdparty/llm-compressor \
  -e thirdparty/onnx2torch
```

Use Python 3.11 or newer.

## Guide

Run quantization with CLI options:

```bash
cks-opt cnn --model resnet50 --dataset datasets/imagenet --quant-mode int8
```

Or run a recipe from [`recipes/cnn/base_cls.yaml`](recipes/cnn/base_cls.yaml):

```bash
cks-opt cnn --config recipes/cnn/base_cls.yaml
```

Run large-model optimization with CLI options, using the recipe from [`recipes/lm/base_int8.yaml`](recipes/lm/base_int8.yaml):

```bash
cks-opt lm --model Qwen/Qwen2.5-0.5B-Instruct \
  --recipe recipes/lm/base_int8.yaml \
  --dataset wikitext --dataset-config-name wikitext-2-raw-v1 \
  --splits "train[:1]" --num-calibration-samples 1 --max-seq-length 32 \
  --output-dir tmp/qwen2.5-0.5b-instruct.int8
```

### License

Apache License 2.0 — see [LICENSE](LICENSE).
