#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

cks-opt cnn \
  --model models/yolov5s.onnx \
  --dataset ./datasets/coco \
  --quant-mode int8 \
  --calibration-split val2017 \
  --evaluation-split val2017
