from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
from PIL import Image

from compressor.datasets.cv import IMAGENET_MEAN, IMAGENET_STD
from compressor.quantization import run_cv_quantization


def build_tiny_imagefolder(root: str) -> None:
    """custom tiny imagefolder: 2 categories x 2 random PNG, demo only."""
    for cls in ("cat", "dog"):
        (Path(root) / cls).mkdir(parents=True, exist_ok=True)
        for i in range(2):
            Image.fromarray(
                torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8).numpy()
            ).save(Path(root) / cls / f"{cls}_{i}.png")
    print(f"[datasets] custom tiny imagefolder: {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CV Quantization Example")
    parser.add_argument("--model", default="resnet18",
                        help="torchvision model name (downloads pretrained "
                             "weights online) / local .pt/.pth ckpt / local .onnx")
    parser.add_argument("--dataset", default=None,
                        help="dataset root dir; default tiny tiny imagefolder")
    parser.add_argument("--quant-mode", default="int8", choices=["int8", "fp8"],
                        help="quantization mode (int8 or fp8)")
    parser.add_argument("--output", default="tmp/cv_example.quantized.onnx")
    args = parser.parse_args()

    dataset_root = args.dataset or tempfile.mkdtemp(prefix="cv_example_data_")
    if args.dataset is None:
        build_tiny_imagefolder(dataset_root)

    result = run_cv_quantization(
        model=args.model,
        dataset_root=dataset_root,
        quant_mode=args.quant_mode,
        output_path=args.output,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        options={"simplify": True, "opset_version": 19},  # onnxsim reference requires opset_version 19/21
    )
    print(f"[pipeline] quantization done: quant_mode={args.quant_mode}, "
          f"export={result.export_path}")

    onnx_path = Path(result.export_path)
    manifest_path = onnx_path.with_suffix(".manifest.json")
    assert onnx_path.exists() and manifest_path.exists()

    import onnx
    onnx_model = onnx.load(str(onnx_path))
    n_qdq = sum(
        1 for node in onnx_model.graph.node
        if node.op_type in ("QuantizeLinear", "DequantizeLinear")
    )
    print(f"[export] {onnx_path} ({onnx_path.stat().st_size // 1024} KB), "
          f"QDQ nodes: {n_qdq} nodes; manifest: {manifest_path}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
