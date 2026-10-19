#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cks-opt

mkdir -p tmp

python -m cks_optimizer cnn \
  --model resnet50 \
  --dataset ./tmp \
  --quant-mode int8
