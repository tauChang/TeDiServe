from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Optional

from vllm.logger import init_logger
from vllm.v1.core.sched.step_estimator import StepStats

from .base import BaseModel

logger = init_logger(__name__)


def _default_counts_path() -> Path:
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        candidate = parent / "analysis" / "gsm8k_oracle" / "req_denoise_counts_by_conf.json"
        if candidate.exists():
            return candidate

    return current_path.parents[6] / "analysis" / "gsm8k_oracle" / "req_denoise_counts_by_conf.json"


class Oracle(BaseModel):
    def __init__(
        self,
        model_path: str | None,
        features_path: str,
        features: Optional[list[str]] = None,
        scheduler_config = None,
        model_config = None,
    ):
        # Initialize parent class (which will call load_model and _build_initial_default_predictions)
        super().__init__(
            model_path=model_path or str(_default_counts_path()),
            features_path=features_path,
            features=features,
            scheduler_config=scheduler_config,
            model_config=model_config,
        )
        self.model: dict[str, dict[float, int]] = {}
        self._default_remaining_by_conf: dict[float, float] = {}
        # Reload the Oracle-specific model (override load_model call from parent)
        self.load_model()
        # Rebuild cache with Oracle-specific logic
        if scheduler_config is not None and model_config is not None:
            self._build_initial_default_predictions()

    def _normalize_confidence(self, confidence_threshold: float | None) -> float:
        if confidence_threshold is None:
            return -1.0
        return round(float(confidence_threshold), 1)

    def _resolve_counts_path(self) -> Path:
        counts_path = Path(self.model_path)
        if counts_path.is_dir():
            counts_path = counts_path / "req_denoise_counts_by_conf.json"

        if counts_path.exists():
            return counts_path

        default_path = _default_counts_path()
        if default_path.exists():
            return default_path

        raise FileNotFoundError(
            f"Could not find Oracle counts JSON at {counts_path} or {default_path}"
        )

    def _build_default_remaining(self) -> dict[float, float]:
        remaining_by_conf: dict[float, list[int]] = {}
        for req_counts in self.model.values():
            for confidence, total_rows in req_counts.items():
                remaining_by_conf.setdefault(confidence, []).append(
                    max(0, total_rows - 1)
                )

        return {
            confidence: float(mean(remaining_values))
            for confidence, remaining_values in remaining_by_conf.items()
            if remaining_values
        }

    def _predict_one(self, stats: StepStats) -> float:
        confidence = self._normalize_confidence(stats.confidence_threshold)
        req_counts = self.model.get(stats.id)
        if req_counts is None:
            return self._default_remaining_by_conf.get(confidence, 0.0)

        total_rows = req_counts.get(confidence)
        if total_rows is None:
            return self._default_remaining_by_conf.get(confidence, 0.0)

        remaining_steps = total_rows - 1 - stats.num_denoise_ran
        return float(max(0, remaining_steps))

    def predict(self, X: list[StepStats] | StepStats) -> list[float] | float:
        single_input = isinstance(X, StepStats)
        if single_input:
            X = [X]

        predictions = [self._predict_one(stats) for stats in X]
        return float(predictions[0]) if single_input else predictions

    def get_initial_default_predictions(self, stats: StepStats) -> dict[float, float]:
        """Get initial default predictions using ground truth from Oracle.
        
        For real request IDs, uses ground truth from the JSON.
        For synthetic __initial_default__ IDs, uses cached predictions as fallback.
        
        Args:
            stats (StepStats): The stats for which to get defaults.
        
        Returns:
            dict[float, float]: Confidence threshold -> prediction.
        """
        if self.scheduler_config is None:
            return {}
        
        req_counts = self.model.get(stats.id)
        
        # If we have ground truth for this request id, use all confidences' ground truth
        if req_counts is not None:
            result = {}
            for conf_key, total_rows in req_counts.items():
                remaining_steps = total_rows - 1 - stats.num_denoise_ran
                result[conf_key] = float(max(0, remaining_steps))
            return result
        
        # Otherwise fallback to cached predictions
        output_length = stats.output_length
        return {
            confidence_threshold: self._initial_default_predictions.get(
                (confidence_threshold, output_length), 0.0
            )
            for confidence_threshold in self.scheduler_config.candidate_confidence_thresholds
        }

    def train(self, X, y):
        raise NotImplementedError("Oracle is a read-only estimator.")

    def save_model(self, file_path=None):
        raise NotImplementedError("Oracle is a read-only estimator.")

    def load_model(self):
        counts_path = self._resolve_counts_path()
        logger.info("Loading Oracle step counts from %s", counts_path)

        with counts_path.open("r", encoding="utf-8") as file_handle:
            raw_counts = json.load(file_handle)

        self.model = {
            req_id: {
                self._normalize_confidence(confidence): int(total_rows)
                for confidence, total_rows in confidence_counts.items()
            }
            for req_id, confidence_counts in raw_counts.items()
        }
        self._default_remaining_by_conf = self._build_default_remaining()