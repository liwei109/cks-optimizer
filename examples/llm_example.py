from __future__ import annotations

import argparse
from pathlib import Path

from compressor.quantization import run_lm_compression


def main() -> None:
    parser = argparse.ArgumentParser(description="LM Quantization Example")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B",
                        help="HF repo id or local model path")
    parser.add_argument("--output", default="tmp/llm_example.out")
    parser.add_argument("--algorithm", default="quantize",
                        choices=["quantize", "gptq", "awq", "smoothquant"],
                        help="algorithm to use (quantize=W8A8 validated; "
                             "gptq affected by llmcompressor×transformers5 autowrap defect)")
    parser.add_argument("--recipe", default=None,
                        help="custom yaml recipe path(quantization/recipe.py); "
                             "provide when override --algorithm")
    parser.add_argument("--num-calibration-samples", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=32)
    args = parser.parse_args()

    result = run_lm_compression(
        model=args.model,
        output_dir=args.output,
        recipe=args.recipe or args.algorithm,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
    )
    print(f"[pipeline] compression done: output_dir={result.output_dir}, "
          f"metadata={result.metadata}")


if __name__ == "__main__":
    main()
