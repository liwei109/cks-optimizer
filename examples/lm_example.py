from __future__ import annotations

import logging

from cks_optimizer.lm.optimizer import LMOptimizer

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    optimizer = LMOptimizer()
    result = optimizer.optimize(
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
    logging.getLogger(__name__).info("Large-model optimization result: %s", result)
