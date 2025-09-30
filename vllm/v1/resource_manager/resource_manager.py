from vllm.executor.ray_utils import _wait_until_pg_ready, _verify_bundles
from vllm.logger import init_logger
from vllm.config import ClusterConfig, VllmConfig
from vllm.platforms import current_platform
from vllm.v1.engine.reconfig_commands import ReconfigCommand, KillCommand, \
    LaunchCommand
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.executors_manager import ExecutorsManager
from collections import defaultdict

import asyncio
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
import ray
from ray.util import placement_group_table
from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

def create_command(old_config: Dict[int, List[int]], 
                    new_config:Dict[int, List[int]]) -> ReconfigCommand:
    """
    Create a parallel command to kill and launch executors
    based on the difference between old and new executor to bundles mapping.
    """
    to_kill = set()
    to_launch = set()
    for executor_id in old_config.keys():
        if executor_id not in new_config:
            to_kill.add(executor_id)
    
    for executor_id, bundles in new_config.items():
        if executor_id not in old_config:
            to_launch.add(executor_id)
    
    parents = defaultdict(set) # eid -> set of commands it depends on
    children = defaultdict(set) # eid -> set of commands depending on it

    
    for launch_eid in to_launch:
        launch_bundles = set(new_config[launch_eid])
        for kill_eid in to_kill:
            kill_bundles = set(old_config[kill_eid])
            if launch_bundles & kill_bundles:
                parents[launch_eid].add(kill_eid)
                children[kill_eid].add(launch_eid)
    
    # dummy
    # if an eid does not have a parent, add a dummy parent
    dummy = -1
    for eid in to_launch | to_kill:
        if eid not in parents:
            parents[eid].add(dummy)
            children[dummy].add(eid)

    # create command
    cmds = {}
    for eid in to_launch:
        cmds[eid] = LaunchCommand(eid, new_config[eid], set(), set())
    for eid in to_kill:
        cmds[eid] = KillCommand(eid, set(), set())
    cmds[dummy] = ReconfigCommand(dummy, set(), set())
    
    for eid, pids in parents.items():
        for pid in pids:
            cmds[eid].parents.add(cmds[pid])
    for eid, cids in children.items():
        for cid in cids:
            cmds[eid].children.add(cmds[cid])
    
    return cmds[dummy]
                                  
class ResourceManager:
    
    def __init__(self,
                 vllm_config: VllmConfig,
                 executor_manager: ExecutorsManager
        ):
        self.vllm_config = vllm_config

        self.cluster_config = vllm_config.cluster_config
        self.node_to_bundles = defaultdict(list)
        self.config = {}
    
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
                accelerator_type = None
                for key in node["Resources"].keys():
                    if key.startswith("accelerator_type:"):
                        accelerator_type = key.split(":")[1]
                        break
                else:
                    raise ValueError(
                        f"Node {node} has {device_str} but no accelerator_type.")
                for _ in range(devices):
                    ip = node["NodeManagerAddress"]
                    bundles.append({
                        device_str: 1,
                        f"node:{ip}": 0.001,
                        f"accelerator_type:{accelerator_type}": 0.001
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

        for bundle_id, bundle in \
            enumerate(self.cluster_config.placement_group.bundle_specs):
            for key in bundle:
                if key.startswith("node:"):
                    node_ip = key.split(":")[1]
                    self.node_to_bundles[node_ip].append(bundle_id)
                    break
        logger.info("Node to bundles mapping: %s", self.node_to_bundles)
        
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
    def reconfig(self) -> ReconfigCommand:
        """
        Assigns bundles to model executors
        ensuring all bundles of a single executor are on the same node.
        Prioritizes nodes with fewer bundles already assigned.
        
        Returns a ([me_ids to kill], )
        """
        logger.debug(f"Starting reconfiguration")
        new_config = {}

        if len(self.config) == 0:
            # use cluster config
            logger.debug("Initial reconfiguration")
            logger.debug(f"num_gpus_per_model_executor: "
                         f"{self.cluster_config.num_gpus_per_model_executor}")

            used_bundles = set()
            node_assigned_count = defaultdict(int)
            for me_id, num_gpus in \
                self.cluster_config.num_gpus_per_model_executor.items():
                assigned = False
                # sort nodes by number of bundles already assigned (ascending)
                sorted_nodes = sorted(
                    self.node_to_bundles.keys(), key=lambda n: node_assigned_count[n])
                logger.debug(f"sorted_nodes: {sorted_nodes}")
                for node_ip in sorted_nodes:
                    bundles_on_node = self.node_to_bundles[node_ip]
                    logger.debug(f"node_ip: {node_ip}, bundles_on_node: {bundles_on_node}")
                    available = [b for b in bundles_on_node if b not in used_bundles]
                    if len(available) >= num_gpus:
                        # assign first `num_gpus` bundles to this executor
                        new_config[me_id] = available[:num_gpus]
                        logger.debug(f"Assigning bundles {available[:num_gpus]} to model executor {me_id}")
                        used_bundles.update(available[:num_gpus])
                        node_assigned_count[node_ip] += num_gpus
                        assigned = True
                        break
                
                if not assigned:
                    raise RuntimeError(
                        f"Cannot assign {num_gpus} GPUs to model executor {me_id} "
                        "on a single node; not enough free bundles.")
        else:
            # one bundle per executor
            # copy old config
            new_config = self.config.copy()

            next_me_id = max(self.config.keys()) + 1
            last_me_id = list(self.config.keys())[-1]
            new_config[next_me_id] = self.config[last_me_id]
            del new_config[last_me_id]
            
            # for node_ip in self.node_to_bundles.keys():
            #     bundles_on_node = self.node_to_bundles[node_ip]
            #     for bundle in bundles_on_node:
            #         new_config[next_me_id] = [bundle]
            #         next_me_id += 1

            # for me_id in self.executors.keys():
            #     assigned = False
            #     # sort nodes by number of bundles already assigned (ascending)
            #     sorted_nodes = sorted(
            #         self.node_to_bundles.keys(), key=lambda n: node_assigned_count[n])
            #     for node_ip in sorted_nodes:
            #         bundles_on_node = self.node_to_bundles[node_ip]
            #         available = [b for b in bundles_on_node if b not in used_bundles]
            #         if len(available) >= 1:
            #             # assign first `num_gpus` bundles to this executor
            #             new_config[me_id] = available[:1]
            #             used_bundles.update(available[:1])
            #             node_assigned_count[node_ip] += 1
            #             assigned = True
            #             break
                
            #     if not assigned:
            #         raise RuntimeError(
            #             f"Cannot assign a GPU to model executor {me_id} "
            #             "on a single node; not enough free bundles.")
        
        logger.debug(f"Old config: {self.config}")
        logger.debug(f"New config: {new_config}")
        cmd = create_command(self.config, new_config)
        logger.debug(f"Reconfiguration command created: {cmd}")
        self.config = new_config
        return cmd

    