#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cks-opt

mkdir -p tmp

python -m cks_optimizer llm \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --recipe '' \
  --dataset wikitext \
  --dataset-config-name wikitext-2-raw-v1 \
  --splits train[:1] \
  --num-calibration-samples 1 \
  --max-seq-length 32 \
