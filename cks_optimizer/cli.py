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
from cks_optimizer.lm.optimizer import LMOptimizer

logger = logging.getLogger(__name__)

__all__ = ["main"]


def _write_stdout(text: str) -> None:
    """Write one user-facing CLI result without mixing it with log messages."""
    sys.stdout.write(f"{text}\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cks-optimizer", description="CKS model optimizer for CNN and LM."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    subparsers = parser.add_subparsers(dest="model_type")

    cnn_parser = subparsers.add_parser("cnn", help="CNN optimization with ModelOpt")
    cnn_parser.add_argument("--config", default="", help="YAML optimization config.")
    cnn_parser.add_argument("--model", default="", help="Model name or path.")
    cnn_parser.add_argument("--dataset", default="", help="Dataset root directory.")
    cnn_parser.add_argument(
        "--quant-mode",
        default="",
        choices=sorted(QUANT_MODES),
        help="Quantization mode; omitted means no quantization unless configured.",
    )
    cnn_parser.add_argument("--sparsity", type=float, help="Target sparsity ratio.")
    cnn_parser.add_argument("--sparsity-mode", help="Sparsity algorithm.")
    cnn_parser.add_argument("--output", default="", help="Output ONNX path.")
    cnn_parser.add_argument("--num-samples", type=int, help="Calibration sample limit.")
    cnn_parser.add_argument("--batch-size", type=int, help="Calibration batch size.")
    cnn_parser.add_argument(
        "--calibration-split", default="", help="Calibration split."
    )
    cnn_parser.add_argument("--evaluation-split", default="", help="Evaluation split.")
    cnn_parser.add_argument(
        "--evaluation-batch-size", type=int, help="Evaluation batch size."
    )
    cnn_parser.add_argument("--device", default="", help="Device (default: cuda).")
    cnn_parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Load pretrained torchvision/timm weights.",
    )
    cnn_parser.add_argument(
        "--json", action="store_true", dest="as_json", help="Print JSON."
    )
    cnn_parser.add_argument(
        "--no-evaluate", action="store_true", help="Skip evaluation."
    )

    lm_parser = subparsers.add_parser(
        "lm", help="Large-model optimization with llmcompressor"
    )
    lm_parser.add_argument("--model", required=True, help="Model name or path.")
    lm_parser.add_argument("--recipe", default="", help="Compression recipe.")
    lm_parser.add_argument("--output-dir", default="", help="Output directory.")
    lm_parser.add_argument(
        "--dataset", default="wikitext", help="Calibration dataset or path."
    )
    lm_parser.add_argument(
        "--dataset-config-name",
        default="wikitext-2-raw-v1",
        help="Dataset config name.",
    )
    lm_parser.add_argument(
        "--splits", default="train[:1]", help="Calibration split selector."
    )
    lm_parser.add_argument(
        "--num-calibration-samples", type=int, default=1, help="Sample count."
    )
    lm_parser.add_argument(
        "--max-seq-length", type=int, default=32, help="Tokenized sequence length."
    )
    lm_parser.add_argument(
        "--trust-remote-code-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow custom model code.",
    )
    return parser


def _cnn_overrides(args: argparse.Namespace) -> dict[str, Any]:
    values = (
        ("model", "model_name"),
        ("dataset", "dataset_root"),
        ("quant_mode", "quant_mode"),
        ("device", "device"),
        ("pretrained", "pretrained"),
    )
    overrides = {
        target: value
        for name, target in values
        if (value := getattr(args, name)) not in (None, "")
    }
    calibration = {
        target: getattr(args, name)
        for name, target in (
            ("num_samples", "num_samples"),
            ("batch_size", "batch_size"),
            ("calibration_split", "split"),
        )
        if getattr(args, name) not in (None, "")
    }
    if calibration:
        overrides["calibration"] = calibration
    for name in ("evaluation_split", "evaluation_batch_size"):
        if (value := getattr(args, name)) not in (None, ""):
            overrides[name] = value
    sparsity = {
        target: getattr(args, name)
        for name, target in (("sparsity", "sparsity"), ("sparsity_mode", "mode"))
        if getattr(args, name) is not None
    }
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
        "quant_mode": result.config.quant_mode,
        "algorithms": list(result.algorithms),
        "output": str(result.onnx_path),
    }
    if result.evaluation is not None:
        payload["evaluation"] = {
            key: value
            for key, value in vars(result.evaluation).items()
            if value is not None
        }
    if args.as_json:
        _write_stdout(json.dumps(payload, indent=2))
    else:
        for key, value in payload.items():
            if key != "evaluation":
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
    """Run the command-line interface.

    Args:
        argv: Optional argument vector. When omitted, arguments are read from
            the process command line.

    Returns:
        Process-style status code: ``0`` for success, ``1`` for missing command,
        and ``2`` when an optimization command fails.
    """
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
            logger.error("LM optimization failed: %s", exc, exc_info=True)
            return 2

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
