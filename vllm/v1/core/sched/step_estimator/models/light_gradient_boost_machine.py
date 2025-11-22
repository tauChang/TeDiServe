import lightgbm as lgb
from .base import BaseModel
from .utils import transform_features
import os
from vllm.logger import init_logger
from typing import Union
import pandas as pd
# import time
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.width', None)

from vllm.v1.core.sched.step_estimator import StepStats
logger = init_logger(__name__)

class LightGradientBoostMachine(BaseModel):
    def __init__(self, 
                 model_path: str, 
                 features_path: str,
                 features: list[str]):
        logger.info(f"Creating LightGradientBoostMachine with model_path: {model_path}, features_path: {features_path}, features: {features}")
        super().__init__(model_path, features_path, features)
        
    def predict(self, X: Union[StepStats, list[StepStats]]) -> Union[float, list[float]]:
        X_df = transform_features(X, self.features)
        # logger.debug(f"Predicting with features: {X_df}")
        # raw_predict = self.model.predict(X_df)
        # predict one row at a time
        predictions = self.model.predict(X_df) * (X.output_length if isinstance(X, StepStats) else X[0].output_length)
        logger.debug(f"Predictions: {predictions}")
        return predictions[0] if len(predictions) == 1 else predictions

    def train(self, X, y):
        # TODO
        self.model.fit(X, y)

    def save_model(self, file_path):
        # TODO
        self.model.save_model(file_path)

    def load_model(self):
        self.model = lgb.Booster(model_file=self.model_path)