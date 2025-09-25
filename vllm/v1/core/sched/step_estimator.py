from dataclasses import dataclass, asdict
import pandas as pd
import os
from typing import Optional
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

@dataclass
class StepStats:
    id: str
    num_denoise_ran: int
    num_unmasked_tokens: int # cumulative
    output_length: int
    block: int
    block_num_denoise_ran: int
    block_num_unmasked_tokens: int
    block_size: int

    confidence_threshold: Optional[float] = None # to be updated by scheduler
    

class StepEstimator:
    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.model = vllm_config.model_config.model
        self.cache_prefix = vllm_config.model_config.cache_prefix
        self.cache_suffix = vllm_config.model_config.cache_suffix
        # load from file if exists
        self.df = self._load_data()
        self._new_rows = 0
    
    def _get_file_path(self):
        dir = "step_data/"
        # create directory if not exists
        if not os.path.exists(dir):
            os.makedirs(dir)
            
        safe_model_name = self.model.replace("/", "_")
        cache_prefix = "_prefix" if self.cache_prefix else ""
        cache_suffix = "_suffix" if self.cache_suffix else ""
        block_size = f"_block{self.vllm_config.model_config.denoise_block_size}"
        confidence = f"_conf{self.vllm_config.scheduler_config.default_confidence_threshold}"
        output_length = f"_out1234"
        
        # return f"{dir}{safe_model_name}{cache_prefix}{cache_suffix}{block_size}_step_estimator.json"
        return f"{dir}{safe_model_name}{cache_prefix}{cache_suffix}{block_size}{confidence}{output_length}.json"
    
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
        