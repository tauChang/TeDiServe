from vllm.logger import init_logger
from vllm.config import ClusterConfig, VllmConfig

logger = init_logger(__name__)

class ResourceManager:
    
    def __init__(self,
                 vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.cluster_config = vllm_config.cluster_config
        
    def reconfig(self):
        """
        Updates ClusterConfig
        """
        current_bundle_id = 0
        if self.cluster_config.model_executor_to_bundles is None:
            self.cluster_config.model_executor_to_bundles = {}
            
        for me_id, num_gpus in \
            self.cluster_config.num_gpus_per_model_executor.items():
            self.cluster_config.model_executor_to_bundles[me_id] = []
            for _ in range(num_gpus):
                self.cluster_config.model_executor_to_bundles[me_id].\
                    append(current_bundle_id)
                current_bundle_id += 1
        
        logger.info("Reconfigured model executor to bundles mapping: %s",
                    self.cluster_config.model_executor_to_bundles)


    
    
        