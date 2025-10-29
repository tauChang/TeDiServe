from vllm.v1.resource_manager.reconfig_planner.milp_reconfig_planner import MILPReconfigPlanner
from vllm.v1.resource_manager.workload_monitor import WorkloadClass


if __name__ == "__main__":
    latency_profile_paths = {
        1: "./latency_profiles/GSAI-ML_LLaDA-8B-Instruct/GH200/TP1.json",
        2: "./latency_profiles/GSAI-ML_LLaDA-8B-Instruct/GH200/TP2.json",
        4: "./latency_profiles/GSAI-ML_LLaDA-8B-Instruct/GH200/TP4.json",
    }

    num_instances = 2 
    K = ["1024_256", "1280_256", "1536_256"]
    # L_k  = {"len256":256, "len512":512}
    P_k = {"1024_256":1024, "1280_256":1280, "1536_256":1536}
    O_k = {"1024_256":256, "1280_256":256, "1536_256":256}
    SLO_k = {"1024_256":6.0, "1280_256":6.0, "1536_256":5.0}
    RPS_k = {"1024_256":1.03, "1280_256": 1.03, "1536_256":0.1}

    workload_classes = [
        WorkloadClass(
            prompt_length=P_k[k],
            output_length=O_k[k],
            slo=SLO_k[k],
            rps=RPS_k[k],
        )
        for k in K
    ]

    current_config = {
        0: [0,1,2,3],
        1: [4,5,6,7],
        # 2: [8,9,10,11],
        # 3: [12,13,14,15],
        # 2: [2],
        # 3: [3],
    }

    node_to_bundles = {
        f"node{i}": [i*4 + j for j in range(4)]
        for i in range(num_instances)
    }
    

    planner = MILPReconfigPlanner(
        latency_profile_paths=latency_profile_paths,
    )
        
    new_config = planner.plan_reconfiguration(
        node_to_bundles=node_to_bundles,
        current_config=current_config,
        workload_classes=workload_classes,
    )
    print("New config:", new_config)