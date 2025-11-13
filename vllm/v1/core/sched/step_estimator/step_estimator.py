from dataclasses import dataclass, asdict
import pandas as pd
import os
import time
from typing import Optional
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import resolve_obj_by_qualname
from typing import Union

logger = init_logger(__name__)

@dataclass
class StepStats:
    id: str
    num_denoise_ran: int
    num_unmasked_tokens: int # cumulative
    num_cur_unmasked_tokens: int # in current step
    output_length: int
    block: int
    block_num_denoise_ran: int
    block_num_unmasked_tokens: int
    block_size: int

    confidence_threshold: Optional[float] = None
    last_recompute_avg_output_confidence: Optional[float] = None
    cur_avg_output_confidence: Optional[float] = None
    

class StepEstimator:
    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.model_config = vllm_config.model_config

        self.llm_model = self.model_config.model
        self.cache_prefix = self.model_config.cache_prefix
        self.cache_suffix = self.model_config.cache_suffix

        self.step_data_dir = self.scheduler_config.step_data_dir
        self.eval_task = self.scheduler_config.eval_task # just for file naming
        self.gen_len = self.scheduler_config.gen_len # just for file naming

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
        
        # load from file if exists
        self.df = self._load_data()
        self._new_rows = 0
    
    def _get_file_path(self):
        safe_model_name = self.llm_model.replace("/", "_")
        cache_prefix = "_prefix" if self.cache_prefix else ""
        cache_suffix = "_suffix" if self.cache_suffix else ""
        block_size = f"_block{self.vllm_config.model_config.denoise_block_size}"
        confidence = f"_conf{self.vllm_config.scheduler_config.default_confidence_threshold}"
        timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
        
        path = f"{self.step_data_dir}/{self.eval_task}/{self.gen_len}/{safe_model_name}{cache_prefix}{cache_suffix}{block_size}{confidence}.json"
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return path
    
    def _load_data(self):
        file_path = self._get_file_path()
        # if os.path.exists(file_path):
        if False:
            self.df = pd.read_json(file_path)
            logger.info(f"Loaded step estimator data from {file_path}")
        else:
            self.df = pd.DataFrame(columns=[
                "num_denoise_ran",
                "num_unmasked_tokens",
                "output_length",
                "block",
                "block_num_denoise_ran",
                "block_num_unmasked_tokens",
                "block_size"
            ])
            logger.info(f"No existing step estimator data found at {file_path}. Starting fresh.")
    
    def save_data(self):
        file_path = self._get_file_path()
        if self._new_rows > 0:
            self.df.tail(self._new_rows).to_json(
                file_path, 
                orient="records", 
                lines=True,
                mode='a',
                index=False
            )
            logger.info(f"Saved {self._new_rows} new rows to step estimator data at {file_path}")
            self._new_rows = 0
        else:
            logger.info("No new data to save for step estimator.")
    
    def add_data_point(self, stats: StepStats):
        # Convert dataclass to dict and append as a single-row DataFrame
        new_row = pd.DataFrame([asdict(stats)])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self._new_rows += 1
        
    def predict(self, stats: Union[StepStats, list[StepStats]]) -> Union[float, list[float]]:
        logger.debug(f"stats for prediction: {stats}")
        assert self.estimator_model is not None, "Estimator model is not initialized."
        return self.estimator_model.predict(stats)