from __future__ import annotations

import argparse
from pathlib import Path

from cks_optimizer.cnn import CNNConfig, optimize_cnn_model
from cks_optimizer.cnn.config import (
    build_default_output_path,
    infer_dataset_kind_from_root,
    infer_task_from_model_and_dataset,
)
from cks_optimizer.cnn.model_loader import resolve_cnn_model_and_sample_input
from cks_optimizer.llm.llmcompressor_adapter import LLMCompressorAdapter


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CKS model optimizer")
    subparsers = parser.add_subparsers(dest="group")

    cnn_parser = subparsers.add_parser(
        "cnn",
        help="CNN / CV optimization with ModelOpt",
    )
    cnn_parser.add_argument(
        "--model",
        required=True,
        help="Model name or path",
    )
    cnn_parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset root or annotation file",
    )
    cnn_parser.add_argument("--quant-mode", default="int8")
    cnn_parser.add_argument(
        "--output",
        default="",
        help="Output path; defaults to <model>.<quant_mode>.onnx under tmp/",
    )

    llm_parser = subparsers.add_parser(
        "llm",
        help="LLM compression with llmcompressor",
    )
    llm_parser.add_argument(
        "--model",
        required=True,
        help="Model name or path",
    )
    llm_parser.add_argument("--recipe", default="", help="Compression recipe")
    llm_parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory; defaults to tmp/<model>.<quant_mode>",
    )
    llm_parser.add_argument(
        "--dataset",
        default="wikitext",
        help=(
            "Calibration dataset name or local path "
            "supported by llmcompressor"
        ),
    )
    llm_parser.add_argument(
        "--dataset-config-name",
        default="wikitext-2-raw-v1",
        help=(
            "Dataset config name when the calibration dataset "
            "requires one"
        ),
    )
    llm_parser.add_argument(
        "--splits",
        default="train[:1]",
        help="Dataset split selector for calibration",
    )
    llm_parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=1,
        help="Number of calibration samples to use for oneshot",
    )
    llm_parser.add_argument(
        "--max-seq-length",
        type=int,
        default=32,
        help="Maximum sequence length for calibration tokenization",
    )
    llm_parser.add_argument(
        "--trust-remote-code-model",
        action="store_true",
        default=True,
        help="Allow custom model code from remote repositories",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.group == "cnn":
        try:
            model, sample_input = resolve_cnn_model_and_sample_input(
                args.model,
                model_kind="auto",
            )

            dataset_kind = infer_dataset_kind_from_root(args.dataset)
            task_name = infer_task_from_model_and_dataset(
                args.model,
                dataset_kind,
            )

            output_path_value = args.output or build_default_output_path(
                args.model,
                args.quant_mode,
            )
            output_path = Path(output_path_value)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            config = CNNConfig(
                model_name=args.model,
                task=task_name,
                dataset_kind=dataset_kind,
                dataset_root=args.dataset,
                quant_mode=args.quant_mode,
                output_path=str(output_path),
                use_dynamic_axes=False,
            )

            result = optimize_cnn_model(model, sample_input, config=config)
            print({
                "group": "cnn",
                "model": args.model,
                "dataset": args.dataset,
                "task": task_name,
                "dataset_kind": dataset_kind,
                "quant_mode": args.quant_mode,
                "output": str(result),
            })
            return 0
        except Exception as exc:  # pragma: no cover
            print(f"CNN optimization failed: {exc}")
            return 2

    if args.group == "llm":
        if not args.model or args.model == "tiny":
            print(
                "LLM model must be a valid local path or Hugging Face "
                "repo ID; 'tiny' is not a valid model identifier."
            )
            return 2

        adapter = LLMCompressorAdapter()
        try:
            output_dir_value = args.output_dir or build_default_output_path(
                args.model,
                args.quant_mode if hasattr(args, 'quant_mode') else 'int8',
            )
            if output_dir_value.endswith('.onnx'):
                output_dir_value = output_dir_value.rsplit('.onnx', 1)[0]

            result = adapter.compress(
                model=args.model,
                output_dir=output_dir_value,
                recipe=args.recipe or None,
                dataset=args.dataset,
                dataset_config_name=args.dataset_config_name,
                splits=args.splits,
                num_calibration_samples=args.num_calibration_samples,
                max_seq_length=args.max_seq_length,
                trust_remote_code_model=args.trust_remote_code_model,
            )
            print(result)
            return 0
        except RuntimeError as exc:
            print(f"LLM compression unavailable: {exc}")
            return 2
        except Exception as exc:  # pragma: no cover
            print(f"LLM compression failed: {exc}")
            return 2

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
