from dataclasses import dataclass, asdict
import pandas as pd
import os
import time
from typing import Optional, Dict
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import resolve_obj_by_qualname
from vllm.v1.utils import BufferedAsyncFileWriter
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

    pred_num_steps_left: Optional[Dict[float, float]] = None # confidence_threshold -> pred_num_steps_left

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
                features=self.scheduler_config.step_estimator_features,
                scheduler_config=self.scheduler_config,
                model_config=self.model_config,
            )
        else:
            self.estimator_model = None
        
        # Load data (disabled in your version)
        self.df = self._load_data()

        # Single append-only buffer used by main thread
        self._row_buffer: list[dict] = []

        # One lock to protect buffer during atomic swap
        self._lock = threading.Lock()

        # for writing step data
        self.file_writer = BufferedAsyncFileWriter(file_path=self._get_file_path())

        profiler_path = os.path.join(
            self.vllm_config.experiment_config.experiment_dir,
            "profiles/step_estimator/predict.jsonl")
        self.predict_profiler = TimeProfiler(
            name="step_estimator_predict",
            file_path=profiler_path,
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

    def add_data_point(self, stats: StepStats, pred_num_steps_left: Optional[Dict[float, float]] = None):
        if pred_num_steps_left is not None:
            stats.pred_num_steps_left = pred_num_steps_left
        """Fast, thread-safe append."""
        row = asdict(stats)
        self.file_writer.add(row)

    def predict(self, stats: list[StepStats]):
        # Ensure list-like for counting
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

    def get_initial_default_predictions(self, stats: StepStats) -> dict[float, float]:
        """Get initial default predictions from stats.
        
        Delegates to the estimator model.
        
        Args:
            stats (StepStats): The stat to get initial defaults for.
        
        Returns:
            dict[float, float]: Predictions for each confidence threshold.
        """
        if self.estimator_model is None:
            return {}
        
        return self.estimator_model.get_initial_default_predictions(stats)
