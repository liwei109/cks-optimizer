#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cks-opt

python -m cks_optimizer cnn \
  --model resnet50 \
  --dataset ./datasets/imagenet \
  --quant-mode int8
