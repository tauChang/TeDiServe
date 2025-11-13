# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import bisect
import json
import torch
import os
from datetime import datetime

from vllm.v1.request import Request, RequestStatus
from dataclasses import dataclass, field
from typing import Dict, List
from vllm.logger import init_logger

logger = init_logger(__name__)

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
        self.cached_lookup_max_batch_size = {}

    def lookup(self, num_tokens: int) -> tuple[int, float]:
        """Find the closest batch size >= num_tokens and return (batch_size, latency)."""
        if num_tokens in self.cached_lookup:
            return self.cached_lookup[num_tokens]

        idx = bisect.bisect_left(self.batch_sizes, num_tokens)
        if idx == len(self.batch_sizes):
            idx -= 1

        self.cached_lookup[num_tokens] = (self.batch_sizes[idx], self.latencies[idx])
        return self.batch_sizes[idx], self.latencies[idx]
    
    def lookup_max_batch_size(self, max_latency: float) -> int:
        """Find the maximum batch size that can be processed within max_latency."""
        # max_latency to 4 decimal places
        max_latency = round(max_latency, 4)
        idx = bisect.bisect_right(self.latencies, max_latency)
        if idx == 0:
            return 0
        self.cached_lookup_max_batch_size[max_latency] = self.batch_sizes[idx - 1]
        return self.batch_sizes[idx - 1]
    

def get_cur_timestamp(include_ms: bool = True) -> str:
    if not include_ms:
        return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")[:-3]

@dataclass
class SystemSnapshot:
    executor_tp_degree: Dict[str, int]
    executor_to_requests: Dict[str, List[str]]
    num_tokens: int
    request_confidence_thresholds: Dict[str, float]
    timestamp: str = field(default_factory=get_cur_timestamp)

    @staticmethod
    def from_states(scheduler) -> "SystemSnapshot":
        executor_tp_degree = {
            es.executor_id: es.tp_degree for es in scheduler.executor_states.values()
        }
        executor_to_requests = {
            es.executor_id: list(es.req_ids) for es in scheduler.executor_states.values()
        }
        num_tokens = sum(r.num_tokens for r in scheduler.requests.values())
        request_confidence_thresholds = {
            r.request_id: r.confidence_threshold for r in scheduler.request_states.values()
        }
        return SystemSnapshot(
            executor_tp_degree=executor_tp_degree,
            executor_to_requests=executor_to_requests,
            num_tokens=num_tokens,
            request_confidence_thresholds=request_confidence_thresholds,
        )
    
    def __str__(self) -> str:
        return json.dumps(self.__dict__, indent=2)
    
class SystemLogger:
    def __init__(self, log_dir: str, scheduler: object):
        self.scheduler = scheduler
        # self.log_dir = log_dir
        self.log_file = os.path.join(scheduler.vllm_config.experiment_config.experiment_dir, "system_log.log")
        # os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"system_log_{get_cur_timestamp(include_ms=False)}.log")
        with open(self.log_file, "w") as f:
            f.write("")  # Create or clear the log file
        self.snapshots = []

    def log(self, write_out: bool = False):
        snapshot = SystemSnapshot.from_states(self.scheduler)
        self.snapshots.append(snapshot)
        logger.debug(f"System Snapshot at {snapshot.timestamp}:\n{snapshot}")

        if write_out:
            logger.info(f"Writing {len(self.snapshots)} snapshots to {self.log_file}")
            with open(self.log_file, "a") as f:
                for snap in self.snapshots:
                    f.write(str(snap) + "\n")
            
            self.snapshots.clear()
