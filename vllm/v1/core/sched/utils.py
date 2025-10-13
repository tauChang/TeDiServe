# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import bisect
import json
import torch
import os
from datetime import datetime

from vllm.v1.request import Request, RequestStatus


def check_stop(request: Request,
               max_model_len: int,
               pooler_output: Optional[torch.Tensor] = None) -> bool:
    # if (request.num_tokens >= max_model_len
    #         or request.num_output_tokens >= request.max_tokens):
    #     request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    #     return True
    if (request.num_unmasked_tokens == request.max_tokens):
        request.status = RequestStatus.FINISHED_STOPPED
        return True
    
    return False

    if request.pooling_params:
        if pooler_output is not None:
            request.status = RequestStatus.FINISHED_STOPPED
            return True
        return False

    sampling_params = request.sampling_params
    assert sampling_params is not None
    last_token_id = request.output_token_ids[-1]
    if (not sampling_params.ignore_eos
            and last_token_id == request.eos_token_id):
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True
    return False

class LatencyProfile:
    """Latency profile for a specific TP degree."""

    def __init__(self, path: str):
        assert os.path.exists(path), f"Latency profile not found: {path}"
        with open(path, "r") as f:
            profile = {int(k): float(v) for k, v in json.load(f).items()}

        # Pre-sort batch sizes and latencies for fast lookup
        self.batch_sizes = sorted(profile.keys())
        self.latencies = [profile[k] for k in self.batch_sizes]

        self.cached_lookup = {}

    def lookup(self, num_tokens: int) -> tuple[int, float]:
        """Find the closest batch size >= num_tokens and return (batch_size, latency)."""
        if num_tokens in self.cached_lookup:
            return self.cached_lookup[num_tokens]

        idx = bisect.bisect_left(self.batch_sizes, num_tokens)
        if idx == len(self.batch_sizes):
            idx -= 1

        self.cached_lookup[num_tokens] = (self.batch_sizes[idx], self.latencies[idx])
        return self.batch_sizes[idx], self.latencies[idx]
    

def get_cur_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")[:-3]