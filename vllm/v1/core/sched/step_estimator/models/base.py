from abc import ABC, abstractmethod
from vllm.v1.core.sched.step_estimator import StepStats
import os
from vllm.logger import init_logger
from typing import Optional, Any

logger = init_logger(__name__)

class BaseModel(ABC):
    def __init__(self, 
                 model_path: str,
                 features_path: str,
                 features: Optional[list[str]] = None,
                 scheduler_config: Optional[Any] = None,
                 model_config: Optional[Any] = None):
        self.model_path = model_path
        self.model = None
        self.features_path = features_path
        self.scheduler_config = scheduler_config
        self.model_config = model_config
        self._initial_default_predictions: dict[tuple[float, int], float] = {}

        if not os.path.exists(self.features_path):
            assert features is not None, "Features must be provided if features_path does not exist."
            self.features = features
        else:
            features_loaded = self.load_features()
            if features is not None:
                assert set(features) == set(features_loaded), "Provided features do not match features in features_path."
            self.features = features_loaded
        # self.features is now set

        self.load_model()
        
        # Build initial default predictions after model is loaded
        if scheduler_config is not None and model_config is not None:
            self._build_initial_default_predictions()

    @abstractmethod
    def predict(self, X: StepStats) -> float:
        """Predict the number of denoise steps for the given input features.
        
        Args:
            X (StepStats): The input features for prediction.
        """
        pass
        
    @abstractmethod
    def train(self, X: StepStats):
        """Train the model with the given input features and target values.
        
        Args:
            X (StepStats): The input features for training.
        """
        pass

    @abstractmethod
    def save_model(self, model_path: str = None):
        """Save the trained model to the specified path.
        
        Args:
            model_path (str, optional): The path to save the model. If None, use the default path.
        """
        pass

    @abstractmethod
    def load_model(self):
        """Load a trained model from the specified path.
        
        Args:
            model_path (str): The path to load the model from.
        """
        pass

    def get_initial_default_predictions(self, stats: StepStats) -> dict[float, float]:
        """Get initial default predictions for the given stats.
        
        Default implementation uses cached predictions by output_length.
        Subclasses can override to provide custom logic (e.g., Oracle uses ground truth).
        
        Args:
            stats (StepStats): The stats for which to get initial defaults.
        
        Returns:
            dict[float, float]: Predictions for each confidence threshold.
        """
        output_length = stats.output_length
        return {
            confidence_threshold: self._initial_default_predictions.get(
                (confidence_threshold, output_length), 0.0
            )
            for confidence_threshold in self.scheduler_config.candidate_confidence_thresholds
        }

    def load_features(self) -> list[str]:
        """Load the feature list from the specified path.
        
        Returns:
            list[str]: The list of features.
        """
        with open(self.features_path, 'r') as f:
            features = f.read().splitlines()
        return features
    
    def _build_initial_default_predictions(self) -> None:
        """Build cache of initial default predictions for synthetic stats.
        
        Called automatically during init if scheduler_config and model_config are provided.
        Subclasses can override to customize cache building.
        """
        if self.scheduler_config is None or self.model_config is None:
            return
        
        confidence_thresholds = self.scheduler_config.candidate_confidence_thresholds
        output_length_bucket = self.model_config.denoise_block_size
        if output_length_bucket <= 0:
            output_length_bucket = 32

        max_output_length = max(
            output_length_bucket,
            (self.scheduler_config.max_model_len // output_length_bucket) * output_length_bucket,
        )
        output_length_buckets = list(range(
            output_length_bucket,
            max_output_length + 1,
            output_length_bucket,
        ))

        synthetic_stats: list[StepStats] = []
        for output_length in output_length_buckets:
            for confidence_threshold in confidence_thresholds:
                synthetic_stats.append(
                    StepStats(
                        id="__initial_default__",
                        timestamp="",
                        num_denoise_ran=0,
                        num_unmasked_tokens=0,
                        num_cur_unmasked_tokens=0,
                        output_length=output_length,
                        block=0,
                        block_num_denoise_ran=0,
                        block_num_unmasked_tokens=0,
                        block_size=self.model_config.denoise_block_size if self.model_config.denoise_block_size > 0 else 32,
                        confidence_threshold=confidence_threshold,
                        max_confidence_threshold=confidence_threshold,
                        min_confidence=0.0,
                        q25_confidence=0.0,
                        median_confidence=0.0,
                        q75_confidence=0.0,
                        avg_confidence=0.0,
                        output_min_confidence=0.0,
                        output_q25_confidence=0.0,
                        output_median_confidence=0.0,
                        output_q75_confidence=0.0,
                        output_avg_confidence=0.0,
                    )
                )

        predictions = self.predict(synthetic_stats)
        for stats, prediction in zip(synthetic_stats, predictions):
            logger.debug(f"Initial default prediction for confidence_threshold={stats.confidence_threshold}, output_length={stats.output_length}: {prediction}")
        self._initial_default_predictions = {
            (stats.confidence_threshold, stats.output_length): float(prediction)
            for stats, prediction in zip(synthetic_stats, predictions)
        }
        # self._initial_default_predictions = {
        #     (0.9, 256): 85,
        #     (0.8, 256): 66,
        #     (0.7, 256): 54,
        #     (0.6, 256): 45,
        #     (0.5, 256): 39
        # }
    