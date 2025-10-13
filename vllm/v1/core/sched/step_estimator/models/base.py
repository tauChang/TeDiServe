from abc import ABC, abstractmethod
from vllm.v1.core.sched.step_estimator import StepStats
import os
from vllm.logger import init_logger
from typing import Optional

logger = init_logger(__name__)

class BaseModel(ABC):
    def __init__(self, 
                 model_path: str,
                 features_path: str,
                 features: Optional[list[str]] = None):
        self.model_path = model_path
        self.model = None
        self.features_path = features_path

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

    def load_features(self) -> list[str]:
        """Load the feature list from the specified path.
        
        Returns:
            list[str]: The list of features.
        """
        with open(self.features_path, 'r') as f:
            features = f.read().splitlines()
        return features
    