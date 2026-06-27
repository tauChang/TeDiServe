import lightgbm as lgb
import numpy as np
from .base import BaseModel
from .utils import transform_features
import os
from vllm.logger import init_logger
from typing import Union
import pandas as pd
import time
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
                 features: list[str],
                 scheduler_config = None,
                 model_config = None
                 ):
        logger.info(f"Creating LightGradientBoostMachine with model_path: {model_path}, features_path: {features_path}, features: {features}")
        super().__init__(
            model_path=model_path, 
            features_path=features_path, 
            features=features,
            scheduler_config=scheduler_config,
            model_config=model_config
        )

    def predict(self, X: list[StepStats]) -> list[float]:
        # start_time = time.perf_counter()
        single_input = isinstance(X, StepStats)
        if single_input:
            X = [X]

        X_np = transform_features(X, self.features)
        # end_time = time.perf_counter()
        # logger.debug(f"Transformed features in {(end_time - start_time) * 1000:.3f} ms.")
        # logger.debug(f"Predicting with features: {X_df}")
        # raw_predict = self.model.predict(X_df)
        # predict one row at a time
        # start_time = time.perf_counter()
        denoise_ratios = np.asarray(self.model.predict(X_np))
        logger.debug(f"predicted denoise ratio is {denoise_ratios}")
        output_lengths = np.asarray([step_stats.output_length for step_stats in X], dtype=denoise_ratios.dtype)
        predictions = denoise_ratios * output_lengths
        # end_time = time.perf_counter()
        # logger.debug(f"Made predictions in {(end_time - start_time) * 1000:.3f} ms.")
        #
        logger.debug(f"Predictions: {predictions}")
        return float(predictions[0]) if single_input else predictions.tolist()

    def train(self, X, y):
        # TODO
        self.model.fit(X, y)

    def save_model(self, file_path):
        # TODO
        self.model.save_model(file_path)

    def load_model(self):
        self.model = lgb.Booster(model_file=self.model_path)