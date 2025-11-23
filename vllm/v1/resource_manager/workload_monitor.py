from vllm.logger import init_logger
from vllm.v1.request import Request
from vllm.v1.utils import BufferedAsyncFileWriter
from vllm.config import VllmConfig
from typing import Dict
from dataclasses import dataclass, asdict
from collections import deque
import time
import math
import os
# import datetime
from datetime import datetime
import json
import threading

logger = init_logger(__name__)
TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

@dataclass
class RequestArrivalStats:
    request_id: str
    arrival_time: float
    prompt_length: int
    output_length: int
    latency_slo: float

    def from_request(request: Request) -> "RequestArrivalStats":
        return RequestArrivalStats(
            request_id=request.request_id,
            arrival_time=request.arrival_time,
            prompt_length=len(request.prompt_token_ids),
            output_length=request.output_length,
            latency_slo=request.latency_slo,
        )
    
    def asdict(self):
        return {
            "request_id": self.request_id,
            "arrival_time": datetime.fromtimestamp(self.arrival_time).strftime(TIME_FORMAT),
            "prompt_length": self.prompt_length,
            "output_length": self.output_length,
            "latency_slo": self.latency_slo,
        }

@dataclass
class RequestCompletionStats:
    request_id: str
    completion_time: str

@dataclass
class WorkloadClass:
    prompt_length: float
    output_length: float
    slo: float  # in seconds
    rps: float  # rate per second

    def __post_init__(self):
        self.name = f"{self.prompt_length}_{self.output_length}"


class WorkloadMonitor:
    def __init__(self, vllm_config: VllmConfig,
                 time_window: float = 30.0):
        self.request_arrival_stats_queue: deque[RequestArrivalStats] = deque()
        self.time_window = time_window
        self.start_time = time.time()
        self.workload_history_path = f"{vllm_config.experiment_config.experiment_dir}/workload_history.json"
        
        self.file_writer = BufferedAsyncFileWriter(self.workload_history_path)
    
    def record_request_arrival(self, request: Request):
        stats = RequestArrivalStats.from_request(request)
        self.request_arrival_stats_queue.append(stats)
        logger.info(f"Logged arrival of request {stats.request_id} at time {stats.arrival_time} "
                    f"with prompt length {stats.prompt_length} and output length {stats.output_length}.")
        
        self.file_writer.add(stats.asdict())
    
    def record_request_completion(self, request_id: str):
        logger.info(f"Request {request_id} has completed processing.")
        stats = RequestCompletionStats(
            request_id=request_id,
            completion_time=datetime.now().strftime(TIME_FORMAT)
        )
        
        self.file_writer.add(asdict(stats))

    def _purge_old_requests(self):
        """Drop outdated requests (arrival_time < now - time_window)."""
        now = time.time()
        cutoff = now - self.time_window

        # Pop from left until we reach a recent one
        while self.request_arrival_stats_queue and self.request_arrival_stats_queue[0].arrival_time < cutoff:
            old = self.request_arrival_stats_queue.popleft()
            logger.debug(f"Dropped expired request {old.request_id} (age={now - old.arrival_time:.1f}s)")
    
    def get_workload_classes(self) -> Dict[str, WorkloadClass]:
        workload_classes: list[WorkloadClass] = []
        self._purge_old_requests()

        # classify request based on (prompt_length, output_length)
        # in 256-token bins
        binsize = 256

        # group requests by ceiling-binned (prompt_len, output_len)
        grouped: Dict[tuple[int, int], list[RequestArrivalStats]] = {}
        for req in self.request_arrival_stats_queue:
            p_bin = math.ceil(req.prompt_length / binsize) * binsize
            o_bin = math.ceil(req.output_length / binsize) * binsize
            grouped.setdefault((p_bin, o_bin), []).append(req)

        # compute per-class RPS and average SLO
        for (p_bin, o_bin), group in grouped.items():
            count = len(group)
            duration = min(self.time_window, time.time() - self.start_time)
            rps = count / duration
            avg_slo = sum(r.latency_slo for r in group) / count

            wc = WorkloadClass(
                prompt_length=p_bin,
                output_length=o_bin,
                slo=avg_slo,
                rps=rps,
            )
            workload_classes.append(wc)

        logger.info(f"Detected {len(workload_classes)} workload classes "
                    f"in the past {self.time_window:.0f}s window:")
        for wc in workload_classes:
            logger.info(f"  {wc.name}: rps={wc.rps:.2f}, slo={wc.slo:.2f}, "
                        f"prompt={wc.prompt_length}, output={wc.output_length}")

        return workload_classes
