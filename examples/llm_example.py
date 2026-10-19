from __future__ import annotations

from cks_optimizer.llm.llmcompressor_adapter import LLMCompressorAdapter


if __name__ == "__main__":
    adapter = LLMCompressorAdapter()
    result = adapter.compress(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        output_dir="tmp/qwen2.5-0.5b-instruct.int8",
        recipe=None,
        dataset="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        splits="train[:1]",
        num_calibration_samples=1,
        max_seq_length=32,
        trust_remote_code_model=True,
    )
    print(result)
