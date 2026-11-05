"""algorithm registry: custom algorithms + adapter wrapped algorithms, unified registration and discovery.
"""

from .lm.smoothquant import SmoothQuant
from .register_algo import Algorithm
from .registry_factory import ALGO_REGISTRY


AlgorithmRegistry = Algorithm()


AlgorithmRegistry.register(
    "quantize", "Linear",
    "llmcompressor.modifiers.quantization.QuantizationModifier",
    deploy_op=None, domain="lm",
)
AlgorithmRegistry.register(
    "gptq", "Linear",
    "llmcompressor.modifiers.gptq.GPTQModifier",
    deploy_op=None, domain="lm",
)
AlgorithmRegistry.register(
    "awq", "Linear",
    "llmcompressor.modifiers.awq.AWQModifier",
    deploy_op=None, domain="lm",
)
AlgorithmRegistry.register(
    "smoothquant", "Linear", SmoothQuant, deploy_op=None, domain="lm",
)


AlgorithmRegistry.register("admm", "Conv", None, deploy_op=None, domain="cv")

__all__ = [
    "Algorithm",
    "AlgorithmRegistry",
    "ALGO_REGISTRY",
    "SmoothQuant",
]
