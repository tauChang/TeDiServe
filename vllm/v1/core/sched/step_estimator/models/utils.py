import pandas as pd
from vllm.v1.core.sched.step_estimator import StepStats
from typing import Union


def transform_features(step_stats: Union[StepStats, list[StepStats]], 
                       features: list[str]) -> pd.DataFrame:
    """Transform one or many StepStats objects into a DataFrame."""
    if not isinstance(step_stats, list):
        step_stats = [step_stats]

    base_attrs = {
        "id": lambda s: s.id,
        "num_denoise_ran": lambda s: s.num_denoise_ran,
        "num_unmasked_tokens": lambda s: s.num_unmasked_tokens,
        "num_cur_unmasked_tokens": lambda s: s.num_cur_unmasked_tokens,
        "output_length": lambda s: s.output_length,
        "block": lambda s: s.block,
        "block_num_denoise_ran": lambda s: s.block_num_denoise_ran,
        "block_num_unmasked_tokens": lambda s: s.block_num_unmasked_tokens,
        "block_size": lambda s: s.block_size,
        "confidence_threshold": lambda s: s.confidence_threshold,
    }

    derived_attrs = {
        "full_progress": lambda s: (s.num_denoise_ran + 1) / s.output_length,
        "full_unmask_progress": lambda s: s.num_unmasked_tokens / s.output_length,
        "full_cur_unmask_progress": lambda s: s.num_cur_unmasked_tokens / s.output_length,
        "block_progress": lambda s: s.block / (s.output_length // s.block_size),
        "block_unmask_progress": lambda s: s.block_num_unmasked_tokens / s.block_size,
    }

    all_attrs = {**base_attrs, **derived_attrs}

    rows = []
    for s in step_stats:
        row = {f: func(s) for f, func in all_attrs.items() if f in features}
        rows.append(row)

    return pd.DataFrame(rows)
