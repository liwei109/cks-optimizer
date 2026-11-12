#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

cks-opt cnn \
  --model resnet50 \
  --dataset ./datasets/imagenet \
  --quant-mode int8 \
  --pretrained \
  --num-samples 128 \
  --batch-size 8
