"""Contract types between the quantization and the compilation: TensorQuant / DeploymentManifest.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TensorQuant:
    """Tensor quant parameters."""

    scale: list[float]
    zero_point: list[int] | None = None
    axis: int | None = None          # per-channel axis; None = per-tensor
    dtype: str = "int8"              # storage dtype: int8/uint8/int4/uint4
    scheme: str = "INT8"             # hardware scheme name(align with hardware.profile)

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.scale or any(s <= 0 for s in self.scale):
            problems.append(f"scale must be positive, got {self.scale[:3]}")
        if self.zero_point is not None and len(self.zero_point) != len(self.scale):
            problems.append("zero_point must match scale length")
        if self.axis is not None and self.axis != 0 and len(self.scale) != 1:
            problems.append(f"per-channel axis={self.axis} must match scale length")
        return problems


@dataclass
class DeploymentManifest:
    """CV deployment manifest: QDQ ONNX + optional rlym"""

    schema: str = "cks.deployment_manifest/1"
    format: str = "qdq_onnx+rlym"
    model_source: str = ""
    quant_mode: str = "int8"
    tensors: dict[str, TensorQuant] = field(default_factory=dict)
    ignore: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def set_tensor(self, name: str, quant: TensorQuant) -> None:
        self.tensors[name] = quant

    def validate(self) -> list[str]:
        problems: list[str] = []
        for name, tq in self.tensors.items():
            for problem in tq.validate():
                problems.append(f"{name}: {problem}")
        return problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "format": self.format,
            "model_source": self.model_source,
            "quant_mode": self.quant_mode,
            "ignore": self.ignore,
            "extra": self.extra,
            "tensors": {k: asdict(v) for k, v in self.tensors.items()},
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DeploymentManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != "cks.deployment_manifest/1":
            raise ValueError(f"Unsupported manifest schema: {payload.get('schema')}")
        manifest = cls(
            format=payload.get("format", "qdq_onnx+rlym"),
            model_source=payload.get("model_source", ""),
            quant_mode=payload.get("quant_mode", "int8"),
            ignore=payload.get("ignore", []),
            extra=payload.get("extra", {}),
        )
        for name, entry in payload.get("tensors", {}).items():
            manifest.tensors[name] = TensorQuant(**entry)
        return manifest
