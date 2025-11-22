from dataclasses import dataclass, asdict
import pandas as pd
import os
import time
from typing import Optional
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import resolve_obj_by_qualname
from typing import Union
import threading
import json
from vllm.v1.utils import TimeProfiler

logger = init_logger(__name__)

@dataclass
class StepStats:
    id: str
    timestamp: str
    num_denoise_ran: int
    num_unmasked_tokens: int # cumulative
    num_cur_unmasked_tokens: int # in current step
    output_length: int
    block: int
    block_num_denoise_ran: int
    block_num_unmasked_tokens: int
    block_size: int

    confidence_threshold: Optional[float] = None
    max_confidence_threshold: Optional[float] = None
    min_confidence: Optional[float] = None
    q25_confidence: Optional[float] = None
    median_confidence: Optional[float] = None
    q75_confidence: Optional[float] = None
    avg_confidence: Optional[float] = None
    output_min_confidence: Optional[float] = None
    output_q25_confidence: Optional[float] = None
    output_median_confidence: Optional[float] = None
    output_q75_confidence: Optional[float] = None
    output_avg_confidence: Optional[float] = None
    # last_recompute_avg_output_confidence: Optional[float] = None
    # cur_avg_output_confidence: Optional[float] = None

class StepEstimator:
    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.model_config = vllm_config.model_config

        self.llm_model = self.model_config.model
        self.cache_prefix = self.model_config.cache_prefix
        self.cache_suffix = self.model_config.cache_suffix

        self.step_data_dir = self.scheduler_config.step_data_dir

        estimator_model_class_str = self.scheduler_config.step_estimator_model_class
        if estimator_model_class_str is not None:
            Model = resolve_obj_by_qualname(estimator_model_class_str)
            self.estimator_model = Model(
                model_path=self.scheduler_config.step_estimator_model_path,
                features_path=self.scheduler_config.step_estimator_features_path,
                features=self.scheduler_config.step_estimator_features
            )
        else:
            self.estimator_model = None
        
        # Load data (disabled in your version)
        self.df = self._load_data()

        # Single append-only buffer used by main thread
        self._row_buffer: list[dict] = []

        # One lock to protect buffer during atomic swap
        self._lock = threading.Lock()

        # Flush every N seconds (default 5s)
        self.flush_interval = 10

        # Start background flush thread
        t = threading.Thread(target=self._periodic_flush, daemon=True)
        t.start()

        profiler_path = os.path.join(
            self.vllm_config.experiment_config.experiment_dir,
            "profiles/step_estimator/predict.jsonl")
        self.predict_profiler = TimeProfiler(
            name="step_estimator_predict",
            file_path=profiler_path,
            flush_interval=10,   # seconds
        )

    # ------------------------------------------------------------
    # File utilities
    # ------------------------------------------------------------

    def _get_file_path(self):
        path = f"{self.step_data_dir}/step_data.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    
    def _load_data(self):
        # TODO: load existing data from file
        return pd.DataFrame(columns=[
            "num_denoise_ran",
            "num_unmasked_tokens",
            "output_length",
            "block",
            "block_num_denoise_ran",
            "block_num_unmasked_tokens",
            "block_size",
            "confidence_threshold",
        ])

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def add_data_point(self, stats: StepStats):
        """Fast, thread-safe append."""
        row = asdict(stats)
        with self._lock:
            self._row_buffer.append(row)

    def predict(self, stats):
        # Ensure list-like for counting
        if isinstance(stats, StepStats):
            batch_size = 1
        else:
            batch_size = len(stats)

        # ---------------------------
        # Timed profiling section
        # ---------------------------
        with self.predict_profiler.section("predict"):
            result = self.estimator_model.predict(stats)

        # Extra metadata (optional)
        self.predict_profiler.add_info("batch_size", batch_size)

        # Commit this predict() entry
        self.predict_profiler.commit()

        return result

    # ------------------------------------------------------------
    # Background flush
    # ------------------------------------------------------------

    def _periodic_flush(self):
        while True:
            time.sleep(self.flush_interval)
            self.flush()

    def flush(self):
        """
        Thread-safe atomic swap flush:
        - lock just long enough to steal row_buffer
        - release lock immediately
        - write & concat outside lock
        """
        # -------- atomic swap --------
        with self._lock:
            if not self._row_buffer:
                return
            to_write = self._row_buffer
            self._row_buffer = []     # new empty buffer
        # -------- lock released --------

        # Convert to DataFrame (outside lock)
        new_df = pd.DataFrame(to_write)

        # Update in-memory df
        self.df = pd.concat([self.df, new_df], ignore_index=True)

        # Append to JSONL file
        file_path = self._get_file_path()
        with open(file_path, "a") as f:
            for row in to_write:
                f.write(json.dumps(row) + "\n")

        logger.debug(
            f"StepEstimator: flushed {len(to_write)} rows to {file_path}"
        )
