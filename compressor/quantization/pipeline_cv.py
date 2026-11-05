from __future__ import annotations

import torch
from typing import Any

from compressor.adapter import get_cv_compressor
from compressor.adapter.contract import CvQuantizeRequest, CvQuantizeResult
from compressor.datasets import get_cv_dataset, iter_cv_batches
from compressor.model import resolve_model


def run_cv_quantization(
    model: Any,
    dataset_root: str | None = None,
    sample_input: Any | None = None,
    *,
    dataset_kind: str = "auto",
    split: str = "train",
    quant_mode: str = "int8",
    output_path: str | None = None,
    image_size: tuple[int, ...] | None = None,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
    options: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> CvQuantizeResult:
    """CV quantization pipeline

    :param model: nn or model name/path.resolve_model)
    :param dataset_root: calibration dataset root path or sample_input
    :param sample_input: calibration/export input batch
    :param image_size/mean/std: calibration image preprocessing
    :param options: pass-through to CvQuantizeRequest.options(quant_cfg/export override)
    :param model_kwargs: pass-through to resolve_model
    """
    
    loaded = None
    if not isinstance(model, torch.nn.Module):
        loaded = resolve_model(model, domain="cv", **(model_kwargs or {}))
        model = loaded.model
        
        if image_size is None:
            image_size = tuple(loaded.sample_input.shape[-2:])

    calibration_batches = None
    if sample_input is None and dataset_root is not None:
        dataset = get_cv_dataset(dataset_kind, dataset_root, split=split)
        print(f"[datasets] {type(dataset).__name__}: {len(dataset)} images")
        batches = list(
            iter_cv_batches(
                dataset,
                image_size=image_size,
                mean=mean,
                std=std,
                progress="preprocess",
            )
        )
        if not batches:
            raise ValueError(
                f"No readable images found under dataset root: {dataset_root}"
            )
        sample_input = batches[0]
        calibration_batches = batches
    if sample_input is None:
        if loaded is not None:
            sample_input = loaded.sample_input
        else:
            raise ValueError(
                "Provide either sample_input or dataset_root when using nn.Module directly"
            )

    compressor = get_cv_compressor()
    compressor.install()
    return compressor.quantize(CvQuantizeRequest(
        model=model,
        sample_input=sample_input,
        quant_mode=quant_mode,
        output_path=output_path,
        options=options or {},
        calibration_batches=calibration_batches,
    ))
