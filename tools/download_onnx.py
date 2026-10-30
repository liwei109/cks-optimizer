"""Download a model (torchvision first, then timm) and export it to ONNX.

Usage:
  python tools/download_onnx.py --list                       # list torchvision + timm models
  python tools/download_onnx.py resnet50                     # -> tmp/resnet50.onnx (batch=1, static)
  python tools/download_onnx.py mobilenet_v2 -o m.onnx --opset 17 --size 256 256
  python tools/download_onnx.py tf_efficientnet_b0           # resolved via timm
  python tools/download_onnx.py resnet50 --batch 8           # fixed batch of 8
  python tools/download_onnx.py resnet50 --dynamic           # symbolic batch dim

  from tools.download_onnx import export_model
  export_model("resnet50", "tmp/resnet50.onnx")
"""

import argparse
import logging

import timm
import torch
import torchvision.models as models

logger = logging.getLogger(__name__)


def list_models():
    """Log available torchvision and timm model names."""
    tv = models.list_models()
    logger.info("torchvision (%d):\n%s", len(tv), ", ".join(tv))
    tm = timm.list_models()
    logger.info("timm (%d):\n%s", len(tm), ", ".join(tm))


def export_model(model_name, output=None, opset=17, size=None, batch=1, dynamic=False):
    """Download pretrained weights (torchvision first, then timm) and export to ONNX.

    ``size`` defaults to the model's own default input size (224x224 for torchvision).
    The batch dim is frozen at ``batch`` unless ``dynamic`` is set.
    Returns the output path.
    """
    if hasattr(models, model_name):
        model = getattr(models, model_name)(weights="DEFAULT").eval()
        size = tuple(size or (224, 224))
    else:
        model = timm.create_model(model_name, pretrained=True).eval()
        if size is None:
            size = tuple(getattr(model, "default_input_size", (3, 224, 224))[1:])

    x = torch.randn(batch, 3, *size)
    out = output or f"tmp/{model_name.replace('/', '_')}.onnx"
    axes = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic else None
    torch.onnx.export(
        model,
        x,
        out,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=axes,
        opset_version=opset,
        dynamo=False,
    )
    logger.info(
        "%s -> %s  input=%s  opset=%d  batch=%s",
        model_name,
        out,
        tuple(x.shape),
        opset,
        "dynamic" if dynamic else batch,
    )
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Download a model (torchvision, then timm) and export it to ONNX"
    )
    parser.add_argument("model", nargs="?", help="Model name, e.g. resnet50 / tf_efficientnet_b0")
    parser.add_argument("-o", "--output", help="Output .onnx path (default: tmp/<model>.onnx)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Input spatial size",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Fixed batch size to freeze (default: 1; ignored with --dynamic)",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Keep the batch dim symbolic instead of freezing it",
    )
    parser.add_argument("--list", action="store_true", help="List all available models")
    args = parser.parse_args()

    if args.list or not args.model:
        list_models()
    else:
        size = tuple(args.size) if args.size else None
        export_model(
            args.model,
            args.output,
            args.opset,
            size,
            batch=args.batch,
            dynamic=args.dynamic,
        )


if __name__ == "__main__":
    main()
