import pandas as pd
from vllm.v1.core.sched.step_estimator import StepStats
from typing import Union
import numpy as np


# def transform_features(step_stats: Union[StepStats, list[StepStats]], 
#                        features: list[str]) -> pd.DataFrame:
#     """Transform one or many StepStats objects into a DataFrame."""
#     if not isinstance(step_stats, list):
#         step_stats = [step_stats]

#     base_attrs = {
#         "id": lambda s: s.id,
#         "num_denoise_ran": lambda s: s.num_denoise_ran,
#         "num_unmasked_tokens": lambda s: s.num_unmasked_tokens,
#         "num_cur_unmasked_tokens": lambda s: s.num_cur_unmasked_tokens,
#         "output_length": lambda s: s.output_length,
#         "block": lambda s: s.block,
#         "block_num_denoise_ran": lambda s: s.block_num_denoise_ran,
#         "block_num_unmasked_tokens": lambda s: s.block_num_unmasked_tokens,
#         "block_size": lambda s: s.block_size,
#         "confidence_threshold": lambda s: s.confidence_threshold,
#         "max_confidence_threshold": lambda s: s.max_confidence_threshold,
#         "min_confidence": lambda s: s.min_confidence,
#         "q25_confidence": lambda s: s.q25_confidence,
#         "median_confidence": lambda s: s.median_confidence,
#         "q75_confidence": lambda s: s.q75_confidence,
#         "avg_confidence": lambda s: s.avg_confidence,
#         "output_min_confidence": lambda s: s.output_min_confidence,
#         "output_q25_confidence": lambda s: s.output_q25_confidence,
#         "output_median_confidence": lambda s: s.output_median_confidence,
#         "output_q75_confidence": lambda s: s.output_q75_confidence,
#         "output_avg_confidence": lambda s: s.output_avg_confidence,
#         "last_recompute_avg_output_confidence": lambda s: s.output_avg_confidence,
#         "cur_avg_output_confidence": lambda s: s.output_avg_confidence,
#     }

#     derived_attrs = {
#         "full_progress": lambda s: (s.num_denoise_ran + 1) / s.output_length,
#         "full_unmask_progress": lambda s: s.num_unmasked_tokens / s.output_length,
#         "full_cur_unmask_progress": lambda s: s.num_cur_unmasked_tokens / s.output_length,
#         "block_progress": lambda s: s.block / (s.output_length // s.block_size),
#         "block_unmask_progress": lambda s: s.block_num_unmasked_tokens / s.block_size,
#     }

#     all_attrs = {**base_attrs, **derived_attrs}

#     rows = []
#     for s in step_stats:
#         row = {f: func(s) for f, func in all_attrs.items() if f in features}
#         rows.append(row)
    
#     df = pd.DataFrame(rows)
#     df = df[features]

#     return df

# import numpy as np
# import pandas as pd

# SPECIAL_MAP = {
#     "last_recompute_avg_output_confidence": "output_avg_confidence",
#     "cur_avg_output_confidence": "output_avg_confidence",
# }

# def transform_features(step_stats, features):
#     if not isinstance(step_stats, list):
#         step_stats = [step_stats]

#     out = {}

#     for f in features:
#         if f == "full_progress":
#             out[f] = np.array([(s.num_denoise_ran + 1) / s.output_length for s in step_stats])
#         elif f == "full_unmask_progress":
#             out[f] = np.array([s.num_unmasked_tokens / s.output_length for s in step_stats])
#         elif f == "full_cur_unmask_progress":
#             out[f] = np.array([s.num_cur_unmasked_tokens / s.output_length for s in step_stats])
#         elif f == "block_progress":
#             out[f] = np.array([s.block / (s.output_length // s.block_size) for s in step_stats])
#         elif f == "block_unmask_progress":
#             out[f] = np.array([s.block_num_unmasked_tokens / s.block_size for s in step_stats])
#         else:
#             attr = SPECIAL_MAP.get(f, f)
#             out[f] = np.array([getattr(s, attr) for s in step_stats])

#     return pd.DataFrame(out)

# import numpy as np
# import pandas as pd

SPECIAL_MAP = {
    "last_recompute_avg_output_confidence": "output_avg_confidence",
    "cur_avg_output_confidence": "output_avg_confidence",
}

# def transform_features(step_stats, features):
#     if not isinstance(step_stats, list):
#         step_stats = [step_stats]

#     # ------------------------------------------
#     # 1) One full scan over step_stats (fast!)
#     # ------------------------------------------
#     cols = {}
#     for f in features:
#         base_attr = SPECIAL_MAP.get(f, f)
#         if f not in (
#             "full_progress",
#             "full_unmask_progress",
#             "full_cur_unmask_progress",
#             "block_progress",
#             "block_unmask_progress",
#         ):
#             # extract direct fields
#             cols[f] = np.array([getattr(s, base_attr) for s in step_stats])

#     # Extract common arrays only once
#     num_denoise_ran      = np.array([s.num_denoise_ran for s in step_stats])
#     num_unmasked         = np.array([s.num_unmasked_tokens for s in step_stats])
#     num_cur_unmasked     = np.array([s.num_cur_unmasked_tokens for s in step_stats])
#     output_length        = np.array([s.output_length for s in step_stats])
#     block                = np.array([s.block for s in step_stats])
#     block_num_unmasked   = np.array([s.block_num_unmasked_tokens for s in step_stats])
#     block_size           = np.array([s.block_size for s in step_stats])

#     # ------------------------------------------
#     # 2) Compute derived attributes in bulk
#     # ------------------------------------------
#     if "full_progress" in features:
#         cols["full_progress"] = (num_denoise_ran + 1) / output_length

#     if "full_unmask_progress" in features:
#         cols["full_unmask_progress"] = num_unmasked / output_length

#     if "full_cur_unmask_progress" in features:
#         cols["full_cur_unmask_progress"] = num_cur_unmasked / output_length

#     if "block_progress" in features:
#         cols["block_progress"] = block / (output_length // block_size)

#     if "block_unmask_progress" in features:
#         cols["block_unmask_progress"] = block_num_unmasked / block_size

#     # ------------------------------------------
#     # 3) Create DataFrame
#     # ------------------------------------------
#     return pd.DataFrame({f: cols[f] for f in features})

def transform_features(step_stats, features):
    if not isinstance(step_stats, list):
        step_stats = [step_stats]

    cols = []

    for f in features:

        if f == "full_progress":
            cols.append([(s.num_denoise_ran + 1) / s.output_length for s in step_stats])

        elif f == "full_unmask_progress":
            cols.append([s.num_unmasked_tokens / s.output_length for s in step_stats])

        elif f == "full_cur_unmask_progress":
            cols.append([s.num_cur_unmasked_tokens / s.output_length for s in step_stats])

        elif f == "block_progress":
            cols.append([s.block / (s.output_length // s.block_size) for s in step_stats])

        elif f == "block_unmask_progress":
            cols.append([s.block_num_unmasked_tokens / s.block_size for s in step_stats])

        else:
            attr = SPECIAL_MAP.get(f, f)
            cols.append([getattr(s, attr) for s in step_stats])

    # shape (N, D)
    return np.column_stack(cols)
