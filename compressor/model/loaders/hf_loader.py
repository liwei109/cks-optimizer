from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_lm_model(
    model_id: str | Path,
    *,
    torch_dtype: Any = torch.float32,
    device: str = "cpu",
    trust_remote_code: bool = True,
) -> torch.nn.Module:
    """load causal LLM weights. VLM(visual encoder+LLM) TODO."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    return model.to(device).eval()


def load_tokenizer(
    model_id: str | Path,
    *,
    trust_remote_code: bool = True,
):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )


def looks_like_hf_repo_id(model_id: str | Path) -> bool:
    candidate = str(model_id).strip()
    if not candidate:
        return False
    if Path(candidate).exists():
        return True
    if candidate.startswith(("http://", "https://")):
        return True
    return "/" in candidate
