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
        "max_confidence_threshold": lambda s: s.max_confidence_threshold,
        "min_confidence": lambda s: s.min_confidence,
        "q25_confidence": lambda s: s.q25_confidence,
        "median_confidence": lambda s: s.median_confidence,
        "q75_confidence": lambda s: s.q75_confidence,
        "avg_confidence": lambda s: s.avg_confidence,
        "output_min_confidence": lambda s: s.output_min_confidence,
        "output_q25_confidence": lambda s: s.output_q25_confidence,
        "output_median_confidence": lambda s: s.output_median_confidence,
        "output_q75_confidence": lambda s: s.output_q75_confidence,
        "output_avg_confidence": lambda s: s.output_avg_confidence,
        "last_recompute_avg_output_confidence": lambda s: s.output_avg_confidence,
        "cur_avg_output_confidence": lambda s: s.output_avg_confidence,
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
    
    df = pd.DataFrame(rows)
    df = df[features]

    return df
