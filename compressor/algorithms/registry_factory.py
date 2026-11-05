from .base import QuantAlgorithmBase
from compressor.utils.registry import Registry


class QuantAlgorithmRegistry(Registry):
    def _register(self, key, obj, force, metadata):
        if not isinstance(obj, type) or not issubclass(obj, QuantAlgorithmBase):
            raise TypeError(
                f"Algorithm '{key}' must inherit QuantAlgorithmBase, got {obj!r}."
            )
        super()._register(key, obj, force, metadata)


ALGO_REGISTRY = QuantAlgorithmRegistry("algo")