from vllm.executor.ray_utils import _wait_until_pg_ready, _verify_bundles
from vllm.logger import init_logger
from vllm.config import ClusterConfig, VllmConfig
from vllm.platforms import current_platform
from collections import defaultdict

import ray
from ray.util import placement_group_table
from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

class ResourceManager:
    
    def __init__(self,
                 vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.cluster_config = vllm_config.cluster_config
    
    def initialize_placement_group(self):
        device_str = current_platform.ray_device_key
        cluster_config = self.cluster_config

        if not device_str:
            raise ValueError(
                f"current platform {current_platform.device_name} does not "
                "support ray.")

        if isinstance(cluster_config.num_gpus_per_model_executor,
                        dict):
            device_required = sum(
                cluster_config.num_gpus_per_model_executor.values())
        else:
            device_required = None

        # Create or get the placement group for worker processes
        if cluster_config.placement_group:
            current_placement_group = cluster_config.placement_group
        else:
            current_placement_group = ray.util.get_current_placement_group()

        if current_placement_group:
            logger.info("Using the existing placement group")

            # We are in a placement group
            bundles = current_placement_group.bundle_specs
            # Verify that we can use the placement group.
            device_bundles = 0
            for bundle in bundles:
                bundle_devices = bundle.get(device_str, 0)
                if bundle_devices > 1:
                    raise ValueError(
                        "Placement group bundle cannot have more than 1 "
                        f"{device_str}.")
                if bundle_devices:
                    device_bundles += 1
                    
            if device_required is not None and device_required > device_bundles:
                raise ValueError(
                    f"The number of required {device_str}s exceeds the total "
                    f"number of available {device_str}s in the placement group. "
                    f"Required number of devices: {cluster_config.world_size}. "
                    f"Total number of devices: {device_bundles}.")
        else:
            logger.info("No current placement group found. "
                        "Creating a new placement group.")
            logger.info("Current cluster resources: %s", ray.cluster_resources())

            bundles = []
            for node in ray.nodes():
                logger.info("Node resources: %s", node["Resources"])
                devices = int(node["Resources"].get(device_str, 0))
                for _ in range(devices):
                    ip = node["NodeManagerAddress"]
                    bundles.append({
                        device_str: 1,
                        f"node:{ip}": 0.001
                    })

            if device_required is not None and \
            device_required > len(bundles):
                raise ValueError(
                    f"The number of required {device_str}s exceeds the total "
                    f"number of available {device_str}s in the cluster. "
                    f"Required number of devices: {device_required}. "
                    f"Total number of devices: {len(bundles)}.")
            
            current_placement_group = ray.util.placement_group(
                bundles, strategy="PACK")
            _wait_until_pg_ready(current_placement_group)
        
        # num_devices_in_cluster = ray.cluster_resources().get(device_str, 0)
        # logger.info(f"num_devices_in_cluster={num_devices_in_cluster}, ")
        # num_devices_in_cluster = int(num_devices_in_cluster)
        # # Log a warning message and delay resource allocation failure response.
        # # Avoid immediate rejection to allow user-initiated placement group
        # # created and wait cluster to be ready
        # if device_required is not None and \
        #    device_required > num_devices_in_cluster:
        #     raise ValueError(
        #         f"The number of required {device_str}s exceeds the total "
        #         f"number of available {device_str}s in the placement group. "
        #         f"Required number of devices: {cluster_config.world_size}. "
        #         f"Total number of devices: {device_bundles}.")
        #     # logger.warning(
        #     #     "The number of required %ss exceeds the total "
        #     #     "number of available %ss in the placement group.", device_str,
        #     #     device_str)
        # # Create a new placement group
        # placement_group_specs: List[Dict[str, float]] = ([{
        #     device_str: 1.0
        # } for _ in range(num_devices_in_cluster)])

        # # vLLM engine is also a worker to execute model with an accelerator,
        # # so it requires to have the device in a current node. Check if
        # # the current node has at least one device.
        # current_ip = get_ip()
        # current_node_id = ray.get_runtime_context().get_node_id()
        # current_node_resource = available_resources_per_node()[current_node_id]
        # if current_node_resource.get(device_str, 0) < 1:
        #     raise ValueError(
        #         f"Current node has no {device_str} available. "
        #         f"{current_node_resource=}. vLLM engine cannot start without "
        #         f"{device_str}. Make sure you have at least 1 {device_str} "
        #         f"available in a node {current_node_id=} {current_ip=}.")
        # # This way, at least bundle is required to be created in a current
        # # node.
        # placement_group_specs[0][f"node:{current_ip}"] = 0.001

        # # By default, Ray packs resources as much as possible.
        # current_placement_group = ray.util.placement_group(
        #     placement_group_specs, strategy="PACK")
        # _wait_until_pg_ready(current_placement_group)

        assert current_placement_group is not None
        _verify_bundles(current_placement_group, cluster_config, device_str)
        # Set the placement group in the cluster config
        cluster_config.placement_group = current_placement_group
        
    # def reconfig(self):
    #     """
    #     Updates ClusterConfig naively.
    #     """
    #     current_bundle_id = 0
    #     if self.cluster_config.model_executor_to_bundles is None:
    #         self.cluster_config.model_executor_to_bundles = defaultdict(list)
            
    #     for me_id, num_gpus in \
    #         self.cluster_config.num_gpus_per_model_executor.items():
            
            
    #         for _ in range(num_gpus):
    #             self.cluster_config.model_executor_to_bundles[me_id].\
    #                 append(current_bundle_id)
    #             current_bundle_id += 1
        
    #     logger.info("Reconfigured model executor to bundles mapping: %s",
    #                 self.cluster_config.model_executor_to_bundles)

    def reconfig(self):
        """
        Updates ClusterConfig: assigns bundles to model executors
        ensuring all bundles of a single executor are on the same node.
        Prioritizes nodes with fewer bundles already assigned.
        """
        if self.cluster_config.model_executor_to_bundles is None:
            self.cluster_config.model_executor_to_bundles = defaultdict(list)
        
        # mapping from node_ip -> list of bundle ids on that node
        node_to_bundles = defaultdict(list)
        for bundle_id, bundle in \
            enumerate(self.cluster_config.placement_group.bundle_specs):
            for key in bundle:
                if key.startswith("node:"):
                    node_ip = key.split(":")[1]
                    node_to_bundles[node_ip].append(bundle_id)
                    break
        
        used_bundles = set()
        node_assigned_count = defaultdict(int)

        for me_id, num_gpus in \
            self.cluster_config.num_gpus_per_model_executor.items():
            assigned = False
            # sort nodes by number of bundles already assigned (ascending)
            sorted_nodes = sorted(
                node_to_bundles.keys(), key=lambda n: node_assigned_count[n])
            for node_ip in sorted_nodes:
                bundles_on_node = node_to_bundles[node_ip]
                available = [b for b in bundles_on_node if b not in used_bundles]
                if len(available) >= num_gpus:
                    # assign first `num_gpus` bundles to this executor
                    self.cluster_config.model_executor_to_bundles[me_id] = \
                        available[:num_gpus]
                    used_bundles.update(available[:num_gpus])
                    node_assigned_count[node_ip] += num_gpus
                    assigned = True
                    break
            
            if not assigned:
                raise RuntimeError(
                    f"Cannot assign {num_gpus} GPUs to model executor {me_id} "
                    "on a single node; not enough free bundles.")