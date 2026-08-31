from vllm.executor.ray_utils import _wait_until_pg_ready, _verify_bundles
from vllm.logger import init_logger
from vllm.config import ClusterConfig, VllmConfig
from vllm.platforms import current_platform
from vllm.v1.engine.reconfig_commands import ReconfigCommand, KillCommand, \
    LaunchCommand, print_command_tree
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.executors_manager import ExecutorsManager
from vllm.v1.request import Request
from vllm.v1.resource_manager.workload_monitor import WorkloadMonitor
from vllm.v1.resource_manager.workload_monitor import WorkloadClass
from collections import defaultdict

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import ray
from ray.util import placement_group_table
from ray.util.placement_group import PlacementGroup
from dataclasses import dataclass, asdict
import os
import time
import json
from collections import deque

logger = init_logger(__name__)
TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

def create_stagewise_parallel_command(old_config: Dict[int, List[int]], 
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

from collections import defaultdict
from typing import Dict, List
from vllm.v1.engine.reconfig_commands import ReconfigCommand, KillCommand, LaunchCommand

def create_stagewise_sequential_command(
    old_config: Dict[int, List[int]],
    new_config: Dict[int, List[int]],
) -> ReconfigCommand:
    """
    Create a reconfiguration command DAG where kills/launches are grouped
    into stages by bundle overlap.

    Example:
        Stage 1: {kill0, kill1} -> {launch5}
        Stage 2: {kill2, kill3} -> {launch6}
        dummy -> Stage1 -> Stage2
    """
    def build_stages(old_config, new_config):
        to_kill = set(old_config) - set(new_config)
        to_launch = set(new_config) - set(old_config)

        # Build adjacency between kills and launches based on overlapping bundles
        adj = defaultdict(set)
        for k in to_kill:
            old_bundles = set(old_config[k])
            for l in to_launch:
                if old_bundles & set(new_config[l]):
                    adj[k].add(l)
                    adj[l].add(k)

        # BFS/DFS over connected components
        visited = set()
        stages = []
        for node in list(to_kill) + list(to_launch):
            if node in visited:
                continue
            queue = deque([node])
            comp_kills, comp_launches = set(), set()
            while queue:
                u = queue.popleft()
                if u in visited:
                    continue
                visited.add(u)
                if u in to_kill:
                    comp_kills.add(u)
                else:
                    comp_launches.add(u)
                for v in adj[u]:
                    if v not in visited:
                        queue.append(v)
            if comp_kills or comp_launches:
                stages.append((comp_kills, comp_launches))
        return stages

    # --- Step 1. Build command objects ---------------------------------
    cmds = {}
    to_kill = [eid for eid in old_config if eid not in new_config]
    to_launch = [eid for eid in new_config if eid not in old_config]

    for eid in to_launch:
        bundles = new_config[eid]
        cmds[eid] = LaunchCommand(eid, bundles, set(), set())
    for eid in to_kill:
        cmds[eid] = KillCommand(eid, set(), set())

    dummy = -1
    cmds[dummy] = ReconfigCommand(dummy, set(), set())

    # --- Step 2. Compute bundle → executor maps -------------------------
    # bundle_to_kill = defaultdict(set)
    # bundle_to_launch = defaultdict(set)
    # for eid in to_kill:
    #     for b in old_config[eid]:
    #         bundle_to_kill[b].add(eid)
    # for eid in to_launch:
    #     for b in new_config[eid]:
    #         bundle_to_launch[b].add(eid)

    # --- Step 3. Build stages based on overlapping bundles --------------
    # For each launch, find all kills that share bundles → stage
    # stages = []
    # used_kills = set()

    # for lid in to_launch:
    #     launch_bundles = set(new_config[lid])
    #     stage_kills = set()
    #     for b in launch_bundles:
    #         stage_kills |= bundle_to_kill[b]

    #     # mark kills as used to prevent reuse in later stages
    #     used_kills |= stage_kills
    #     stages.append((stage_kills, {lid}))

    # # Add any leftover kills (if any) as a pre-stage
    # unused_kills = [k for k in to_kill if k not in used_kills]
    # # if unused_kills:
    # #     stages.insert(0, (set(unused_kills), set()))
    # assert len(unused_kills) == 0, \
    #     f"Some kills were not assigned to any stage: {unused_kills}"
    stages = build_stages(old_config, new_config)
    logger.debug(f"Reconfiguration stages: {stages}")

    # --- Step 4. Wire dependencies -------------------------------------
    prev_stage = {dummy}

    for stage_kills, stage_launches in stages:
        # prev_stage → kills
        for kid in stage_kills:
            for p in prev_stage:
                cmds[p].children.add(cmds[kid])
                cmds[kid].parents.add(cmds[p])

        # kills → launches (only if overlap)
        for lid in stage_launches:
            launch_bundles = set(new_config[lid])
            for kid in stage_kills:
                kill_bundles = set(old_config[kid])
                if launch_bundles & kill_bundles:
                    cmds[kid].children.add(cmds[lid])
                    cmds[lid].parents.add(cmds[kid])

        # move to next stage
        prev_stage = stage_launches

    return cmds[dummy]


def create_nodewise_sequential_command(
    old_config: Dict[int, List[int]],
    new_config: Dict[int, List[int]],
    bundle_to_node: Dict[int, str],
) -> ReconfigCommand:
    """
    Create reconfiguration commands grouped node-by-node:

        dummy -> {kills@node0} -> {launches@node0} -> {kills@node1} -> {launches@node1} -> ...

    Within each node, a KillCommand only depends on a LaunchCommand
    if they share common bundles (i.e., reuse the same physical GPUs).
    This ensures service continuity while minimizing unnecessary dependencies.
    """

    # --- Step 2: Identify kill / launch sets per node ---------------------
    to_kill_by_node = defaultdict(set)
    to_launch_by_node = defaultdict(set)

    for eid, bundles in old_config.items():
        if eid not in new_config:
            node = bundle_to_node[bundles[0]]
            to_kill_by_node[node].add(eid)

    for eid, bundles in new_config.items():
        if eid not in old_config:
            node = bundle_to_node[bundles[0]]
            to_launch_by_node[node].add(eid)

    # --- Step 3: Build command objects ------------------------------------
    cmds = {}
    for eid, bundles in new_config.items():
        if eid not in old_config:  # new executor
            cmds[eid] = LaunchCommand(eid, bundles, set(), set())
    for eid in old_config:
        if eid not in new_config:  # executor to kill
            cmds[eid] = KillCommand(eid, set(), set())

    dummy = -1
    cmds[dummy] = ReconfigCommand(dummy, set(), set())

    # --- Step 4: Sequentialize across nodes -------------------------------
    prev_stage = {dummy}
    # assumption: on a node, there's a kill iff there's a launch
    assert set(to_kill_by_node.keys()) == set(to_launch_by_node.keys()), \
        f"Mismatch in nodes: kills={set(to_kill_by_node.keys())}, launches={set(to_launch_by_node.keys())}"

    for node in to_kill_by_node.keys():
        kills = to_kill_by_node[node]
        launches = to_launch_by_node[node]

        # dummy / previous node’s launches → all kills on this node
        for kid in kills:
            for p in prev_stage:
                cmds[p].children.add(cmds[kid])
                cmds[kid].parents.add(cmds[p])

        # fine-grained: kills(node) → launches(node) only if bundles overlap
        for lid in launches:
            launch_bundles = set(new_config[lid])
            for kid in kills:
                kill_bundles = set(old_config[kid])
                if launch_bundles & kill_bundles:
                    cmds[kid].children.add(cmds[lid])
                    cmds[lid].parents.add(cmds[kid])

        # move to next node: last launches (or kills if no launches)
        prev_stage = launches

    return cmds[dummy]



@dataclass
class ConfigRecord:
    timestamp: float
    config: Dict[int, List[int]]
    workload_classes: List[WorkloadClass]
    aggregate_rps: float

class ResourceManager:
    
    def __init__(self,
                 vllm_config: VllmConfig,
                 executor_manager: ExecutorsManager
        ):
        self.vllm_config = vllm_config

        self.cluster_config: ClusterConfig = vllm_config.cluster_config
        self.node_to_bundles: Dict[str, List[int]] = defaultdict(list)
        self.bundle_to_node: Dict[int, str] = {}
        self.config: Dict[int, List[int]] = {}

        self.workload_monitor = WorkloadMonitor(vllm_config, time_window=60)
        # delay until first time reconfig planner is called to create, since reconfig planner needs latency profile,
        # which requires vllm_config.cluster_config.placement_group.bundle_specs to be initialized
        self.reconfig_planner = None 
        self.config_history: List[ConfigRecord] = []
        self.config_history_path = f"{self.vllm_config.experiment_config.experiment_dir}/config_history.json"

    @staticmethod
    def _aggregate_workload_rps(workload_classes: List[WorkloadClass]) -> float:
        return sum(workload_class.rps for workload_class in workload_classes)
    
    def write_config_history(self):
        """Write config history to JSON file on disk."""
        # Convert dataclass objects to serializable dicts
        serializable_records = []
        for record in self.config_history:
            rec_dict = asdict(record)
            # Each WorkloadClass inside the list is also a dataclass, so asdict() handles it recursively
            serializable_records.append(rec_dict)

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.config_history_path), exist_ok=True)

        # Write as JSON (append or overwrite depending on your preference)
        with open(self.config_history_path, "w") as f:
            json.dump(serializable_records, f, indent=2)

        logger.debug(f"Wrote {len(serializable_records)} config history records to {self.config_history_path}")

    
    def initialize_placement_group(self):
        device_str = current_platform.ray_device_key
        cluster_config = self.cluster_config

        if not device_str:
            raise ValueError(
                f"current platform {current_platform.device_name} does not "
                "support ray.")

        # Skip GPU requirement check for fake executor since it doesn't use GPUs
        if self.vllm_config.parallel_config.distributed_executor_backend == "fake":
            device_required = None
        elif isinstance(cluster_config.num_gpus_per_model_executor,
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
                    self.bundle_to_node[bundle_id] = node_ip
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
    def get_initial_config(self):
        config = {}
        logger.debug("Initial reconfiguration")
        logger.debug(f"num_gpus_per_model_executor: "
                        f"{self.cluster_config.num_gpus_per_model_executor}")

        # Skip GPU assignment for fake executor since it doesn't use GPUs
        if self.vllm_config.parallel_config.distributed_executor_backend == "fake":
            logger.debug("Using fake executor; skipping GPU bundle assignment")
            # just act like we have one bundle per executor for simplicity
            for me_id in self.cluster_config.num_gpus_per_model_executor.keys():
                config[me_id] = [me_id]  # assign a unique dummy bundle ID to each executor
            return config

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
                    config[me_id] = available[:num_gpus]
                    logger.debug(f"Assigning bundles {available[:num_gpus]} to model executor {me_id}")
                    used_bundles.update(available[:num_gpus])
                    node_assigned_count[node_ip] += num_gpus
                    assigned = True
                    break
            
            if not assigned:
                raise RuntimeError(
                    f"Cannot assign {num_gpus} GPUs to model executor {me_id} "
                    "on a single node; not enough free bundles.")
        
        return config
        
    def naive_reconfig(self) -> ReconfigCommand:
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
            new_config = self.get_initial_config()
            workload_classes = []
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
            workload_classes = self.workload_monitor.get_workload_classes()
        
        logger.debug(f"Old config: {self.config}")
        logger.debug(f"New config: {new_config}")
        cmd = create_stagewise_parallel_command(self.config, new_config)
        logger.debug(f"Reconfiguration command created: {cmd}")
        self.config = new_config

        timestamp = time.time()
        record = ConfigRecord(
            timestamp=datetime.fromtimestamp(timestamp).strftime(TIME_FORMAT),
            config=new_config,
            workload_classes=workload_classes,
            aggregate_rps=self._aggregate_workload_rps(workload_classes),
        )
        self.config_history.append(record)
        self.write_config_history()
        return cmd
    
    async def reconfig(self) -> ReconfigCommand:
        # return self.naive_reconfig()
        timestamp = time.time()
        if len(self.config) == 0:
            logger.info("Initial config")
            workload_classes = []
            new_config = self.get_initial_config()
            cmd = create_stagewise_parallel_command(self.config, new_config)
        else:
            logger.info(f"Reconfiguration based on workload")
            workload_classes = self.workload_monitor.get_workload_classes()
            if self.reconfig_planner is None:
                # Gurobi is an optional dependency: it is only needed after
                # the initial configuration when dynamic reconfiguration is
                # actually enabled.
                from vllm.v1.resource_manager.reconfig_planner \
                    .milp_reconfig_planner import MILPReconfigPlanner
                self.reconfig_planner = MILPReconfigPlanner(self.vllm_config)
            new_config = await self.reconfig_planner.plan_reconfiguration_async(
                self.node_to_bundles, self.config, workload_classes)

            # if there's only one node, and the reconfig plan involves killing
            # all executors at the same time, skip reconfiguration (new_config = old_config) to avoid downtime
            if len(self.node_to_bundles) == 1:
                tmp_cmd = create_stagewise_sequential_command(self.config, new_config)
                # if the dummy has ${num of executor} kill commands as children, it means all executors are killed at the same time
                if len(tmp_cmd.children) == len(self.config):
                    logger.info("Single node detected with full kill plan; skipping reconfiguration to avoid downtime.")
                    new_config = self.config
                cmd = create_stagewise_sequential_command(
                    self.config, new_config)
            else:
                # cmd = create_nodewise_sequential_command(
                #     self.config, new_config, self.bundle_to_node
                # )
                cmd = create_stagewise_sequential_command(
                    self.config, new_config)


        logger.info(f"Old config: {self.config}")
        logger.info(f"New config: {new_config}")
        logger.debug(f"Reconfiguration command created: {cmd}")
        print_command_tree(cmd)
        self.config = new_config

        # record
        record = ConfigRecord(
            timestamp=datetime.fromtimestamp(timestamp).strftime(TIME_FORMAT),
            config=new_config,
            workload_classes=workload_classes,
            aggregate_rps=self._aggregate_workload_rps(workload_classes),
        )
        self.config_history.append(record)
        self.write_config_history()
        return cmd
        

    def record_request_arrival(self, req: Request):
        self.workload_monitor.record_request_arrival(req)
    
    def record_request_completion(self, req_id: str):
        self.workload_monitor.record_request_completion(req_id)
