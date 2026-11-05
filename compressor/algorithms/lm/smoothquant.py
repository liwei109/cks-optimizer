"""SmoothQuant quantization algorithm for LM: integrate llmcompressor SmoothQuantModifier.
"""

from __future__ import annotations

from typing import Any

from compressor.algorithms.base import BaseAlgo


DEFAULTS: dict[str, Any] = {
    "smoothing_strength": 0.5,  # alpha: 0=only weight migration, 1=only activation migration
}


class SmoothQuant(BaseAlgo):
    """SmoothQuant quantization algorithm for LM: default params + external Modifier customization points."""

    def __init__(self, config: dict[str, Any] | None = None):
        merged = {**DEFAULTS, **(config or {})}
        super().__init__(merged)

    def get_config(self) -> dict[str, Any]:
        return dict(self.config)

    def validate_config(self) -> None:
        strength = self.config.get("smoothing_strength", 0.5)
        if not 0.0 <= strength <= 1.0:
            raise ValueError(
                f"smoothing_strength must be in [0, 1], got {strength}"
            )

    def build_modifier(self, **overrides: Any) -> Any:
        """Construct llmcompressor SmoothQuantModifier lazily.

        Customization point: subclass / parameter injection.
        """
        from llmcompressor.modifiers.transform.smoothquant import (
            SmoothQuantModifier,
        )

        self.validate_config()
        params = {**self.config, **overrides}
        return SmoothQuantModifier(**params)

    def apply(self, model: Any, *args: Any, **kwargs: Any) -> Any:
        """Recipe semantics: return Modifier to be scheduled by pipeline_lm in Oneshot.

        SmoothQuant model transformation is performed by llmcompressor in Oneshot.
        """
        return self.build_modifier(**kwargs)
