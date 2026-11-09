# LM recipes

`base_int8.yaml` is an INT8 W8A8 dynamic-activation large-model recipe based
on the upstream `recipe_w8a8_dynamic_asym.yaml` recipe.

Pass the recipe file path to `--recipe`:

```bash
cks-opt lm \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --recipe recipes/lm/base_int8.yaml \
  --dataset wikitext \
  --dataset-config-name wikitext-2-raw-v1 \
  --splits "train[:1]" \
  --num-calibration-samples 1 \
  --max-seq-length 32 \
  --output-dir tmp/qwen2.5-0.5b-instruct.int8
```

The recipe applies SmoothQuant followed by GPTQ, uses symmetric INT8 weights
and dynamic asymmetric INT8 activations, and excludes `lm_head` from GPTQ.
