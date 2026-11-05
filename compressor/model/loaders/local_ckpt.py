from __future__ import annotations

from pathlib import Path

import torch


_MODEL_DICT_KEYS = ("model", "network", "module")

TORCH_CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt", ".bin")


def load_torch_model(model_path: str | Path) -> torch.nn.Module:
    """from local torch model checkpoint.

    Accepts: pickled nn.Module, or dict with model/network/module keys;
    state_dict-only checkpoint throws error.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file '{path}' does not exist.")

    loaded = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(loaded, torch.nn.Module):
        return loaded.eval()

    if isinstance(loaded, dict):
        for key in _MODEL_DICT_KEYS:
            value = loaded.get(key)
            if isinstance(value, torch.nn.Module):
                return value.eval()
        if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            raise ValueError(
                "The checkpoint contains only a state_dict, not a complete "
                "model object. Please pass a model object from "
                "torchvision/timm or provide a full model file."
            )

    raise TypeError(
        f"Unsupported torch checkpoint format in '{path}'. "
        "Expected a nn.Module instance or a dict containing a model object."
    )


def load_state_dict(path: str | Path, model: torch.nn.Module) -> torch.nn.Module:
    """load state_dict-only checkpoint into model."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file '{path}' does not exist.")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(
            f"Unsupported state_dict format in '{path}': {type(payload).__name__}"
        )
    model.load_state_dict(payload)
    return model.eval()


def looks_like_torch_checkpoint(path: str | Path) -> bool:
    return str(path).endswith(TORCH_CHECKPOINT_SUFFIXES)
