"""Command-line interface for the CKS model optimizer."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from cks_optimizer.cnn import run_cnn_pipeline
from cks_optimizer.cnn.config import (
    QUANT_MODES,
    build_default_output_path,
    build_optimize_config,
)
from cks_optimizer.cnn.datasets.handler import DATASET_HANDLER_REGISTRY
from cks_optimizer.lm.optimizer import LMOptimizer

logger = logging.getLogger(__name__)

__all__ = ["main"]


def _write_stdout(text: str) -> None:
    """Write one user-facing CLI result without mixing it with log messages."""
    sys.stdout.write(f"{text}\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cks-optimizer",
        description=(
            "CKS model optimizer for CNN and LM."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    subparsers = parser.add_subparsers(dest="model_type")

    cnn_parser = subparsers.add_parser(
        "cnn",
        help="CNN / CV optimization with ModelOpt",
    )
    cnn_parser.add_argument(
        "--config",
        default="",
        help="Path to a YAML/JSON optimization config file.",
    )
    cnn_parser.add_argument(
        "--model",
        default="",
        help="Model name or path (overrides the config).",
    )
    cnn_parser.add_argument(
        "--data-dir",
        default="",
        help="Dataset root directory (overrides the config).",
    )
    cnn_parser.add_argument(
        "--data-types",
        default="",
        choices=sorted(DATASET_HANDLER_REGISTRY),
        help=(
            "Dataset type used to select the preprocessing pipeline "
            "(imagenet, coco, voc); overrides automatic task inference."
        ),
    )
    cnn_parser.add_argument(
        "--quant-mode",
        default="",
        choices=sorted(QUANT_MODES),
        help="Quantization mode (int8, fp8, nvfp4, mxfp8, ...).",
    )
    cnn_parser.add_argument(
        "--sparsity",
        type=float,
        default=None,
        help=(
            "Target sparsity ratio for the 'sparsify' algorithm. When "
            "provided, a sparsity config section is added."
        ),
    )
    cnn_parser.add_argument(
        "--sparsity-mode",
        default=None,
        help=(
            "Sparsity mode for the 'sparsify' algorithm. When provided, "
            "a sparsity config section is added."
        ),
    )
    cnn_parser.add_argument(
        "--output",
        default="",
        help="Output path; defaults to tmp/<model>.<quant_mode-or-optimized>.onnx.",
    )
    cnn_parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Maximum number of calibration samples.",
    )
    cnn_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Calibration batch size.",
    )
    cnn_parser.add_argument(
        "--calibration-split",
        default="",
        help="Dataset split used for calibration (e.g. train2017 or val2017).",
    )
    cnn_parser.add_argument(
        "--evaluation-split",
        default="",
        help="Held-out dataset split used for evaluation (e.g. val2017).",
    )
    cnn_parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=None,
        help="Evaluation batch size; static ONNX inputs still limit the maximum.",
    )
    cnn_parser.add_argument(
        "--device",
        default="",
        help="Torch device used to load the model (e.g. 'cpu', 'cuda:0').",
    )
    cnn_parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Load pretrained weights for torchvision/timm models.",
    )
    cnn_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print the result as JSON instead of key=value lines.",
    )
    cnn_parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Skip the evaluation stage.",
    )

    lm_parser = subparsers.add_parser(
        "lm",
        help="Large-model optimization with llmcompressor",
    )
    lm_parser.add_argument(
        "--model",
        required=True,
        help="Model name or path",
    )
    lm_parser.add_argument("--recipe", default="", help="Compression recipe")
    lm_parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory; defaults to tmp/<model>.<quant_mode>",
    )
    lm_parser.add_argument(
        "--dataset",
        default="wikitext",
        help="Calibration dataset name or local path supported by llmcompressor",
    )
    lm_parser.add_argument(
        "--dataset-config-name",
        default="wikitext-2-raw-v1",
        help=("Dataset config name when the calibration dataset requires one"),
    )
    lm_parser.add_argument(
        "--splits",
        default="train[:1]",
        help="Dataset split selector for calibration",
    )
    lm_parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=1,
        help="Number of calibration samples to use for oneshot",
    )
    lm_parser.add_argument(
        "--max-seq-length",
        type=int,
        default=32,
        help="Maximum sequence length for calibration tokenization",
    )
    lm_parser.add_argument(
        "--trust-remote-code-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow custom model code from remote repositories",
    )

    return parser


def _cnn_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Translate CNN CLI flags into ``build_optimize_config`` overrides."""
    overrides: dict[str, Any] = {}
    if args.model:
        overrides["model_name"] = args.model
    if args.data_dir:
        overrides["dataset_root"] = args.data_dir
    if args.data_types:
        # An explicit dataset type pins the task and the dataset handler
        # used for preprocessing in ``data_process``.
        overrides["dataset"] = args.data_types
    if args.quant_mode:
        overrides["quant_mode"] = args.quant_mode
    if args.device:
        overrides["device"] = args.device
    if args.pretrained is not None:
        overrides["pretrained"] = args.pretrained

    calibration: dict[str, Any] = {}
    if args.num_samples is not None:
        calibration["num_samples"] = args.num_samples
    if args.batch_size is not None:
        calibration["batch_size"] = args.batch_size
    if args.calibration_split:
        calibration["split"] = args.calibration_split
    if calibration:
        overrides["calibration"] = calibration
    if args.evaluation_split:
        overrides["evaluation_split"] = args.evaluation_split
    if args.evaluation_batch_size is not None:
        overrides["evaluation_batch_size"] = args.evaluation_batch_size

    # Supplying sparsity settings selects the sparsify algorithm.
    sparsity: dict[str, Any] = {}
    if args.sparsity is not None:
        sparsity["sparsity"] = args.sparsity
    if args.sparsity_mode is not None:
        sparsity["mode"] = args.sparsity_mode
    if sparsity:
        overrides["sparsity"] = sparsity

    return overrides


def _run_cnn(args: argparse.Namespace) -> int:
    config = build_optimize_config(args.config or None, **_cnn_overrides(args))
    if args.output:
        config.output_path = args.output
    if (
        not config.effective_algorithms
        and not args.config
        and args.pretrained is None
        and not Path(config.model_name).is_file()
        and Path(config.model_name).suffix.lower()
        not in {".onnx", ".pt", ".pth", ".ckpt", ".bin"}
    ):
        config.pretrained = True
        logger.info(
            "Evaluation-only run: loading pretrained weights for model %r.",
            config.model_name,
        )

    result = run_cnn_pipeline(
        config=config,
        evaluate_model=not args.no_evaluate,
    )

    payload: dict[str, Any] = {
        "group": "cnn",
        "model": result.config.model_name,
        "dataset": result.config.dataset_root,
        "task": result.config.task,
        "quant_mode": (
            result.config.quant.mode if result.config.quant is not None else None
        ),
        "algorithms": list(result.algorithms),
        "output": str(result.onnx_path),
    }
    if result.evaluation is not None:
        evaluation_payload: dict[str, Any] = {
            "task": result.evaluation.task,
            "num_samples": result.evaluation.num_samples,
        }
        for metric in (
            "accuracy",
            "mean_average_precision",
            "map_50",
            "map_75",
            "mean_iou",
        ):
            value = getattr(result.evaluation, metric)
            if value is not None:
                evaluation_payload[metric] = value
        payload["evaluation"] = evaluation_payload
    if args.as_json:
        _write_stdout(json.dumps(payload, indent=2))
    else:
        for key, value in payload.items():
            if key == "evaluation":
                continue
            _write_stdout(f"{key}={value}")
        if result.evaluation is not None:
            _write_stdout(result.evaluation.summary())
    return 0


def _run_lm(args: argparse.Namespace) -> int:
    output_dir_value = args.output_dir or build_default_output_path(args.model, "int8")
    if output_dir_value.endswith(".onnx"):
        output_dir_value = output_dir_value.rsplit(".onnx", 1)[0]

    optimizer = LMOptimizer()
    result = optimizer.optimize(
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
    _write_stdout(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    if args.model_type == "cnn":
        try:
            return _run_cnn(args)
        except Exception as exc:  # pragma: no cover - CLI error path
            logger.error("CNN optimization failed: %s", exc, exc_info=True)
            return 2

    if args.model_type == "lm":
        try:
            return _run_lm(args)
        except RuntimeError as exc:
            logger.error("Large-model optimization unavailable: %s", exc)
            return 2
        except Exception as exc:  # pragma: no cover - CLI error path
            logger.error("Large-model optimization failed: %s", exc, exc_info=True)
            return 2

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
