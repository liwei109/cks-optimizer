import importlib
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AlgorithmSpec:
    """AlgoSpec item

    :param quant_op: algorithm processor, 3 forms:
        - llmcompressor Modifier's dotted path(str, lazy parsed, import llmcompressor)
        - custom algorithm processor
        - None: registered but not implemented
    :param deploy_op: deployment operator mapping
    """

    name: str
    src_op: str = "Linear"
    quant_op: Any = None
    deploy_op: Any = None
    domain: str = "lm"  # "cv" | "lm"
    metadata: dict[str, Any] = field(default_factory=dict)


class Algorithm:
    '''Algorithm registry for quantization algorithms.

    Can register cv algorithms, as well as lm algorithms

    Two purposes:
    1. Register llmcompressor existing Modifier(QuantizationModifier /
       GPTQModifier / AWQModifier / SmoothQuantModifier...), pipeline by
       algorithm name to construct default recipe
    2. Register custom algorithms, and custom Modifier's for specific tasks
    '''

    def __init__(self):
        self.algo: dict[str, AlgorithmSpec] = {}

    def register(
        self,
        name: str,
        src_op: str = "Linear",
        quant_op: Any = None,
        deploy_op: Any = None,
        *,
        domain: str = "lm",
        force: bool = False,
    ) -> AlgorithmSpec:
        """Register an algorithm item.

        Default reject re-registering existing names, force=True to override.
        """
        if not name or not name.strip():
            raise ValueError("algorithm name cannot be empty")
        name = name.strip()
        if name in self.algo and not force:
            raise KeyError(
                f"algorithm '{name}' already registered: {self.algo[name]}; "
                "use force=True to override"
            )
        spec = AlgorithmSpec(
            name=name,
            src_op=src_op,
            quant_op=quant_op,
            deploy_op=deploy_op,
            domain=domain,
        )
        self.algo[name] = spec
        logger.info(f"register algorithm '{name}' ({domain}/{src_op}) success.")
        return spec

    def get(self, name: str) -> AlgorithmSpec:
        spec = self.algo.get(name)
        if spec is None:
            raise KeyError(
                f"algorithm '{name}' not registered; available: {sorted(self.algo)}"
            )
        return spec

    def build_modifier(self, name: str, **kwargs: Any) -> Any:
        """Build llmcompressor Modifier by algorithm name.

        dotted path lazy import; custom build_modifier for custom algorithms.
        """
        spec = self.get(name)
        handler = spec.quant_op
        if handler is None:
            raise NotImplementedError(
                f"algorithm '{name}' registered but not implemented"
            )
        if isinstance(handler, str):
            module_path, _, attr = handler.rpartition(".")
            handler = getattr(importlib.import_module(module_path), attr)
        if hasattr(handler, "build_modifier"):
            # custom build_modifier: project default params + custom point in wrapper
            return handler().build_modifier(**kwargs)
        return handler(**kwargs)

    def __contains__(self, name: str) -> bool:
        return name in self.algo

    def list_algorithms(self, domain: str | None = None) -> list[str]:
        """List algorithm names by domain."""
        return [
            name
            for name, spec in self.algo.items()
            if domain is None or spec.domain == domain
        ]
