from __future__ import annotations

import argparse
import os
import sys

from compressor.adapter.contract import Technique


def _default_output_path(model: str, *suffixes: str) -> str:
    name = model.rstrip("/").rsplit("/", 1)[-1] or "model"
    return os.path.join("tmp", ".".join((name, *suffixes)))


def _add_lm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="HF repo id or local model path")
    parser.add_argument(
        "--output-dir", default=None,
        help="output directory; default tmp/<model_name>",
    )
    parser.add_argument(
        "--recipe", default=None,
        help="compression recipe: algorithm name(quantize/gptq/awq/smoothquant) / custom yaml path;"
             " default: inject default modifier from model adapter",
    )
    parser.add_argument(
        "--dataset", default="wikitext",
        help="dataset name or local path(llmcompressor supported)",
    )
    parser.add_argument(
        "--dataset-config-name", default="wikitext-2-raw-v1",
        help="dataset config name",
    )
    parser.add_argument(
        "--splits", default="train[:1]",
        help="dataset split selector",
    )
    parser.add_argument(
        "--num-calibration-samples", type=int, default=1,
        help="oneshot calibration samples",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=32,
        help="max sequence length after tokenize",
    )
    parser.add_argument(
        "--no-trust-remote-code-model", action="store_true",
        help="disable trust remote code model(default allow)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cks-compressor",
        description="cks-compressor: model compression unified workflow for CV/LM",
    )
    sub = parser.add_subparsers(dest="group", required=True)

    # ------------------------------------------------------------------
    # quantize: CV/LM quantize workflow for CNN/ViT quantize/LLMcompressor
    # ------------------------------------------------------------------
    quantize = sub.add_parser("quantize", help="quantize workflow for CV/LM")
    qsub = quantize.add_subparsers(dest="scenario", required=True)

    cv = qsub.add_parser("cv", help="CNN/ViT quantize(ModelOpt → ONNX)")
    cv.add_argument("--model", required=True, help="torchvision model name or local ckpt path")
    cv.add_argument(
        "--dataset", required=True,
        help="calibration dataset root directory(imagefolder/coco/voc)",
    )
    cv.add_argument(
        "--quant-mode", default="int8",
        help="quantization mode(int8/fp8, default int8)",
    )
    cv.add_argument(
        "--output", default=None,
        help="ONNX output path; default tmp/<model_name>.<quant_mode>.onnx",
    )

    lm = qsub.add_parser("lm", help="LLM/VLM quantize(llmcompressor)")
    _add_lm_arguments(lm)

    # ------------------------------------------------------------------
    # sparsify: LM supported; CV pending ModelOpt sparsity integration
    # ------------------------------------------------------------------
    sparsify = sub.add_parser("sparsify", help="2:4 sparsify workflow")
    spsub = sparsify.add_subparsers(dest="scenario", required=True)
    sparsify_lm = spsub.add_parser("lm", help="LLM/VLM 2:4 sparseify(SparseGPT)")
    _add_lm_arguments(sparsify_lm)

    search = sub.add_parser("search", help="mixed precision search and recipe generation")
    search.add_argument("--model", required=True)
    search.add_argument("--profile", default=None, help="hardware profile file")
    search.add_argument("--budget-bits", type=float, default=None)

    evaluate = sub.add_parser("eval", help="compression quality evaluation")
    evaluate.add_argument("--baseline", required=True)
    evaluate.add_argument("--quantized", required=True)

    return parser


def _run_quantize_cv(args: argparse.Namespace) -> int:
    output = args.output or _default_output_path(args.model, args.quant_mode, "onnx")
    try:
        from compressor.quantization import run_cv_quantization

        result = run_cv_quantization(
            model=args.model,
            dataset_root=args.dataset,
            quant_mode=args.quant_mode,
            output_path=output,
        )
    except Exception as exc:  # pragma: no cover
        print(f"CV quantization failed: {exc}", file=sys.stderr)
        return 2
    print(f"[quantize:cv] done: model={args.model}, "
          f"quant_mode={args.quant_mode}, export={result.export_path}")
    return 0


def _run_lm(args: argparse.Namespace, technique_name: str) -> int:
    technique = Technique(technique_name)
    output_dir = args.output_dir or _default_output_path(args.model, technique.value)
    try:
        from compressor.quantization import run_lm_compression

        result = run_lm_compression(
            model=args.model,
            output_dir=output_dir,
            recipe=args.recipe or None,
            technique=technique,
            dataset=args.dataset,
            dataset_config_name=args.dataset_config_name,
            splits=args.splits,
            num_calibration_samples=args.num_calibration_samples,
            max_seq_length=args.max_seq_length,
            trust_remote_code_model=not args.no_trust_remote_code_model,
        )
    except Exception as exc:  # pragma: no cover
        print(f"LM compression failed: {exc}", file=sys.stderr)
        return 2
    print(f"[{technique_name}:lm] done: model={args.model}, "
          f"output_dir={result.output_dir}, metadata={result.metadata}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.group == "quantize":
        if args.scenario == "cv":
            return _run_quantize_cv(args)
        return _run_lm(args, "quantize")
    if args.group == "sparsify":
        return _run_lm(args, "sparsify")

    print(f"TODO: pipeline not implemented for: {args.group}", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
