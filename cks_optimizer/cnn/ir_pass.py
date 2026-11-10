"""Perform IR(Graph) optimization pass on the original/quantized model"""
from pyexpat import model
import torch

from transforms import OptimizeAndFuse, SimplifyPass, FoldBNConv2dPass


def _get_ctx(
    config: dict = None,
    disable_pass=None,
    instruments=None,
    required_pass=None,
):
    """Get the context for the optimization pass."""
    return model


def optimize(model: torch.nn.Module) -> torch.nn.Module:
    # TODO: Implement optimization and fuse pass
    with _get_ctx():
        model = OptimizeAndFuse(model)

        return model


def get_fold_bn_conv2d_passes():
    """Get the passes to fold BN and Conv2d."""
    return [FoldBNConv2dPass()]


def to_low_precision(model: torch.nn.Module) -> torch.nn.Module:
    """Convert the model to low precision."""
    return model


def get_common_simplify_passes():
    """Get the common simplification passes."""
    return [SimplifyPass()]
