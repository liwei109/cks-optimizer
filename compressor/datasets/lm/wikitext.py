from __future__ import annotations


def load_wikitext_batches(
    model_path: str,
    dataset: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    seq_len: int = 2048,
    max_sequences: int = 20,
    stride: int | None = None,
) -> list[dict]:
    """loading and slicing wikitext corpus(transformers/datasets core).

    :param model_path: providing tokenizer model name/path
    :param split: calibration using "train", evaluation using "test"
    :param seq_len: window length; ``stride`` default to seq_len(non-overlapping)
    :param max_sequences: max number of sequences to return
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "wikitext loading requires transformers and datasets first"
        ) from exc

    from compressor.datasets.lm.hub_cache import load_dataset_local_first

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    corpus = "\n\n".join(
        item["text"]
        for item in load_dataset_local_first(
            dataset, dataset_config, split=split
        )
    )
    token_ids = tokenizer(corpus, return_tensors="pt").input_ids[0]

    stride = stride or seq_len
    batches: list[dict] = []
    for start in range(0, len(token_ids) - seq_len, stride):
        if len(batches) >= max_sequences:
            break
        chunk = token_ids[start : start + seq_len].unsqueeze(0)
        batches.append({"input_ids": chunk})
    if not batches:
        raise ValueError("corpus is too short, please reduce seq_len")
    return batches
