# gurobi_milp_bigM.py
from gurobipy import Model, GRB, quicksum
import json

import json
import numpy as np
import os

def load_Tc_from_logs(base_dir="step_data_kiet/gsm8k_100/256/", pattern_prefix="GSAI-ML_LLaDA-8B-Instruct_block32_conf", conf_list=[0.5, 0.6, 0.7, 0.8, 0.9]):
    """
    Read diffusion-LM step logs and compute expected tokens unmasked per step (T_c)
    for each confidence threshold.

    Args:
        base_dir (str): Directory containing the .json trace files.
        pattern_prefix (str): Common prefix before "confX.json"
        conf_list (list[float]): Confidence thresholds to load

    Returns:
        dict[float, float]: T_c mapping {confidence -> mean(num_cur_unmasked_tokens)}
    """
    T_c = {}

    for conf in conf_list:
        path = os.path.join(base_dir, f"{pattern_prefix}{conf}.json")

        values = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                values.append(obj.get("num_cur_unmasked_tokens", 0))

        if len(values) == 0:
            print(f"⚠️  No values found in {path}")
            continue

        mean_val = float(np.mean(values))
        T_c[conf] = mean_val

        print(f"Confidence {conf:.1f}: mean={mean_val:.2f}, median={np.median(values):.2f}, std={np.std(values):.2f}, min={np.min(values)}, max={np.max(values)}")

    return T_c


def build_capacity_planner_bigM(
    K, C, G, B, N,
    P_k, O_k,
    SLO_k, RPS_k,
    T_c,
    L_gb,
    G_n, 
):
    """
    MILP with s_{k,g} (no s_{k,g,b}); batch-bin choice linearized with big-M.
    """
    m = Model("dlm_capacity_planner_bigM")

    # --- Decision variables ---
    # Steps per class k at confidence c
    s_kc = m.addVars(K, C, lb=0.0, vtype=GRB.CONTINUOUS, name="s_kc")

    # Steps per class k executed on TP degree g
    s_kg = m.addVars(K, G, lb=0.0, vtype=GRB.CONTINUOUS, name="s_kg")

    # Binary batch-bin choice per TP degree g
    z_gb = {(g,b): m.addVar(vtype=GRB.BINARY, name=f"z[{g},{b}]")
            for g in G for b in B[g]}

    # Integer number of TP=g instances on node n
    x_ng = m.addVars(N, G, lb=0.0, vtype=GRB.INTEGER, name="x_ng")

    # slack
    r_k = m.addVars(K, lb=0.0, vtype=GRB.CONTINUOUS, name="r_k")

    m.update()

    # --- Objective ---
    # Maximize confidence weighted token throughput
    total_RPS = sum(RPS_k[k] for k in K)

    obj = quicksum(
        (RPS_k[k] / O_k[k]) * quicksum(c * T_c[c] * s_kc[k, c] for c in C)
        for k in K
    ) / total_RPS
    # m.setObjective(obj, GRB.MAXIMIZE)
    m.setObjectiveN(-obj, index=0, priority=2, name="primary")
    m.setObjectiveN(
        -quicksum(r_k[k] / SLO_k[k] * RPS_k[k] / total_RPS for k in K),
        index=1, priority=1, name="max_slack"
    )

    # --- Constraints ---

    # 1) Token completion
    for k in K:
        m.addConstr(quicksum(T_c[c] * s_kc[k, c] for c in C) == O_k[k],
                    name=f"token_completion[{k}]")

    # 2) Step accounting
    for k in K:
        m.addConstr(quicksum(s_kc[k, c] for c in C)
                    == quicksum(s_kg[k, g] for g in G),
                    name=f"step_accounting[{k}]")

    # 3) SLO per class
    # sum_g (sum_b L_{g,b} z_{g,b}) * s_{k,g} <= SLO_k
    for k in K:
        lhs = quicksum(
            s_kg[k, g] * quicksum(L_gb[(g,b)] * z_gb[(g,b)] for b in B[g])
            for g in G
        )
        m.addConstr(lhs + r_k[k] <= SLO_k[k], name=f"slo[{k}]")

    # 4) One batch bin per TP degree
    for g in G:
        m.addConstr(quicksum(z_gb[(g,b)] for b in B[g]) == 1,
                    name=f"one_bin[{g}]")

    # 5) Node GPU capacity
    for n in N:
        m.addConstr(
            quicksum(g * x_ng[n,g] for g in G) <= G_n[n],
            name=f"gpu_cap[{n}]"
        )

    # # 6) Step-throughput capacity
    # # suO_k RPS_k * s_{k,g} <= sum_b (1/L_{g,b}) * sum_n x_{n,g} * z_{g,b}
    # for g in G:
    #     lhs = quicksum(RPS_k[k] * s_kg[k, g] for k in K)
    #     rhs = quicksum(
    #         (1 / L_gb[(g,b)]) * quicksum(x_ng[n,g] for n in N) * z_gb[(g,b)]
    #         for b in B[g]
    #     )
    #     m.addConstr(lhs <= rhs, name=f"step_capacity[{g}]")

    # # 7) Aggregate batch-size (token throughput) (Incorrect)
    # lhs_tokens = quicksum(RPS_k[k] * (P_k[k] + O_k[k]) for k in K)
    # rhs_tokens = quicksum(
    #     b * (1 / L_gb[(g,b)]) * quicksum(x_ng[n,g] for n in N) * z_gb[(g,b)]
    #     for g in G for b in B[g]
    # )
    # m.addConstr(lhs_tokens <= rhs_tokens, name="token_throughput_capacity")

    # # 8) Batch size chosen must fit prompt + mask lengths
    # max_seq_len = max(P_k[k] + O_k[k] for k in K)
    # for g in G:
    #     m.addConstr(quicksum(b * z_gb[(g, b)] for b in B[g]) >= max_seq_len,
    #                 name=f"batch_size_lb[{g}]")

    # 6) Step-throughput capacity
    # suO_k RPS_k * s_{k,g} * (P_k + O_k) / (sum_b b * z_{g,b}) <= sum_b (1/L_{g,b}) * sum_n x_{n,g} * z_{g,b}
    # for g in G:
    #     lhs = quicksum(RPS_k[k] * s_kg[k, g] * (P_k[k] + O_k[k]) / quicksum(b * z_gb[(g,b)] for b in B[g]) for k in K)
    #     rhs = quicksum(
    #         (1 / L_gb[(g,b)]) * quicksum(x_ng[n,g] for n in N) * z_gb[(g,b)]
    #         for b in B[g]
    #     )
    #     m.addConstr(lhs <= rhs, name=f"step_capacity[{g}]")
    M_big = 1e6  # Big-M constant
    for g in G:
        for b in B[g]:
            lhs = quicksum(RPS_k[k] * s_kg[k,g] * (P_k[k] + O_k[k]) for k in K)
            rhs = (b / L_gb[(g,b)]) * quicksum(x_ng[n,g] for n in N)
            m.addConstr(lhs <= rhs + M_big * (1 - z_gb[(g,b)]),
                        name=f"token_throughput_{g}_{b}")

    # 8) Each batch must have one or more requests
    for g in G:
        m.addConstr(
            quicksum(b * z_gb[(g,b)] for b in B[g]) >= max(P_k[k] + O_k[k] for k in K),
            name=f"batch_nonzero_req[{k},{g}]"
        )

    m.update()
    return m, {"s_kc": s_kc, "s_kg": s_kg, "x_ng": x_ng, "z_gb": z_gb}

def get_new_config_ilp(
    G, # candidate TP degrees
    node_to_bundles: dict,  # {node: [bundle_ids]}
    x_ng: dict,                       # {(node, g): Var/float/int} -> solved MILP counts (aggregate per g)
    old_config: dict[int, list[int]], # {me_id: [bundle_ids]}
) -> dict[int, list[int]]:
    """
    ILP that:
        - Meets required counts per TP degree (node-agnostic).
        - Keeps every executor's bundles on a single node.
        - Maximizes # of unchanged old executors (exact same bundle set).
        - Returns new_config: {me_id: [bundle_ids]} with re-used IDs when kept
        and reuses surplus IDs before creating new ones.

    Assumptions:
        - self.node_to_bundles: Dict[node, List[bundle_id]] (node is hashable)
        - Each old executor is already node-local (all bundles on the same node)
    """
    # ---------- Preprocess -------------------------------------------------
    # Required counts per degree g (aggregate across nodes)
    required_counts: dict[int, int] = {}
    for (n, g), var in x_ng.items():
        val = getattr(var, "X", var)
        cnt = int(round(val))
        if cnt > 0:
            required_counts[g] = required_counts.get(g, 0) + cnt

    # Build quick maps
    N = list(node_to_bundles.keys())
    Bn: dict = {n: list(node_to_bundles[n]) for n in N}
    bundle_to_node = {}
    for n in N:
        for b in Bn[n]:
            bundle_to_node[b] = n
    print(bundle_to_node)
    # 3/0

    # Old executors grouped; also verify each is node-local
    E = list(old_config.keys())
    old_exec_info = {}  # e -> (n_e, g_e, S_e)
    for e in E:
        bundles = old_config[e]
        if not bundles:
            # Treat empty as trivially node-local but useless; skip keeps
            continue
        nodes = {bundle_to_node[b] for b in bundles}
        if len(nodes) != 1:
            raise ValueError(
                f"Old executor {e} spans multiple nodes: {nodes}. "
                "ILP requires per-executor single-node bundles."
            )
        n_e = next(iter(nodes))
        g_e = len(bundles)
        old_exec_info[e] = (n_e, g_e, tuple(sorted(bundles)))

    # Per-node slot upper bounds: K_{n,g} = floor(|B_n| / g)
    K = {}
    for n in N:
        for g in G:
            if g <= 0:
                continue
            cap = len(Bn[n]) // g
            assert len(Bn[n]) % g == 0, f"Node {n} has {len(Bn[n])} bundles not divisible by g={g}"
            if cap > 0:
                K[(n, g)] = cap

    # ---------- Build ILP --------------------------------------------------
    m = Model("reconfig_slots")
    m.Params.OutputFlag = 0  # quiet; toggle to 1 if you want logs

    # u[n,g,k] ∈ {0,1} : slot used
    u = {}
    for (n, g), cap in K.items():
        for k in range(cap):
            u[(n, g, k)] = m.addVar(vtype=GRB.BINARY, name=f"u[{n},{g},{k}]")

    # a[n,g,k,b] ∈ {0,1} : bundle b assigned to slot (n,g,k)
    a = {}
    for (n, g), cap in K.items():
        for k in range(cap):
            for b in Bn[n]:
                a[(n, g, k, b)] = m.addVar(vtype=GRB.BINARY, name=f"a[{n},{g},{k},{b}]")

    # m_keep[e,k] ∈ {0,1} : old executor e kept by mapping to slot (n_e,g_e,k)
    m_keep = {}
    for e, (n_e, g_e, S_e) in old_exec_info.items():
        cap = K.get((n_e, g_e), 0)
        for k in range(cap):
            m_keep[(e, k)] = m.addVar(vtype=GRB.BINARY, name=f"mkeep[{e},{k}]")

    m.update()

    # ---------- Constraints -----------------------------------------------
    # 1) Slot fill and node locality: sum_b a = g * u
    for (n, g), cap in K.items():
        for k in range(cap):
            m.addConstr(quicksum(a[(n, g, k, b)] for b in Bn[n]) == g * u[(n, g, k)],
                        name=f"fill[{n},{g},{k}]")

    # 2) Bundle exclusivity: each bundle used at most once
    all_bundles = [b for n in N for b in Bn[n]]
    for b in all_bundles:
        n_b = bundle_to_node[b]
        sum_terms = []
        for (n, g), cap in K.items():
            if n != n_b:
                continue
            for k in range(cap):
                sum_terms.append(a[(n, g, k, b)])
        if sum_terms:
            m.addConstr(quicksum(sum_terms) <= 1, name=f"exclusive[{b}]")

    # 3) Meet required counts per TP degree
    for g in G:
        req = required_counts.get(g, 0)
        # Sum of used slots over all nodes for this g equals R_g
        sum_u = []
        for (n2, g2), cap in K.items():
            if g2 == g:
                for k in range(cap):
                    sum_u.append(u[(n2, g2, k)])
        if sum_u:
            m.addConstr(quicksum(sum_u) == req, name=f"req[{g}]")
        else:
            # No capacity for this g anywhere; must require 0
            if req != 0:
                raise RuntimeError(f"Infeasible: no capacity to place TP={g} while R_g={req}")

    # 4) Keep mapping: each old executor kept at most once
    for e, (n_e, g_e, S_e) in old_exec_info.items():
        cap = K.get((n_e, g_e), 0)
        assert cap > 0 # it's already assigned, so there must be capacity
        m.addConstr(quicksum(m_keep[(e, k)] for k in range(cap)) <= 1, name=f"keep_once[{e}]")

    # 5) If kept, that slot must be active
    for e, (n_e, g_e, S_e) in old_exec_info.items():
        cap = K.get((n_e, g_e), 0)
        assert cap > 0
        for k in range(cap):
            m.addConstr(m_keep[(e, k)] <= u[(n_e, g_e, k)], name=f"keep_implies_u[{e},{k}]")

    # 6) If kept, all bundles in S_e must be assigned to that slot
    for e, (n_e, g_e, S_e) in old_exec_info.items():
        cap = K.get((n_e, g_e), 0)
        assert cap > 0
        for k in range(cap):
            for b in S_e:
                m.addConstr(m_keep[(e, k)] <= a[(n_e, g_e, k, b)],
                            name=f"keep_implies_a[{e},{k},{b}]")

    # ---------- Objective ---------------------------------------------------
    # Maximize sum_e keep_e  - epsilon * sum_{n,g,k} u[n,g,k]
    # keep_e = sum_k m_keep[e,k]
    obj_keep = quicksum(m_keep[(e, k)] for e in old_exec_info for k in range(K.get((old_exec_info[e][0], old_exec_info[e][1]), 0)))
    m.setObjective(obj_keep, GRB.MAXIMIZE)

    # ---------- Solve -------------------------------------------------------
    m.Params.MIPGap = 0.01
    m.optimize()
    status = m.Status
    if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(f"Reconfig ILP ended with status {status}")

    # ---------- Extract solution -------------------------------------------
    # Gather used slots and their assigned bundles
    used_slots = []  # (n, g, k, [bundles])
    for (n, g), cap in K.items():
        for k in range(cap):
            if u[(n, g, k)].X > 0.5:
                # Collect bundles
                bundles = [b for b in Bn[n] if a[(n, g, k, b)].X > 0.5]
                # Sanity: must be exactly g bundles
                if len(bundles) != g:
                    raise RuntimeError(f"Slot ({n},{g},{k}) has {len(bundles)} bundles, expected {g}")
                used_slots.append((n, g, k, sorted(bundles)))
    # assert num bundles used = total bundles available
    total_used_bundles = sum(len(bundles) for (_, _, _, bundles) in used_slots)
    assert total_used_bundles == len(all_bundles)

    # Map kept old executors first
    new_config: dict[int, list[int]] = {}
    assigned_slots = set()
    kept_ids = set()
    for e, (n_e, g_e, S_e) in old_exec_info.items():
        cap = K.get((n_e, g_e), 0)
        keep_k = None
        for k in range(cap):
            if m_keep[(e, k)].X > 0.5:
                keep_k = k
                break
        if keep_k is not None:
            # Find the corresponding slot in used_slots
            for idx, (n, g, k, bundles) in enumerate(used_slots):
                if (n == n_e) and (g == g_e) and (k == keep_k):
                    # bundles should equal S_e (by constraints)
                    new_config[e] = list(bundles)
                    assigned_slots.add((n, g, k))
                    kept_ids.add(e)
                    break

    next_id = (max(E) + 1) if E else 0
    # Assign remaining used slots to IDs (reuse surplus first)
    for (n, g, k, bundles) in used_slots:
        if (n, g, k) in assigned_slots:
            continue
        me_id = next_id
        next_id += 1
        new_config[me_id] = list(bundles)

    return new_config

# ---------------- Example -----------------
if __name__ == "__main__":
    num_instances = 2
    K = ["1024_256"] # P_M
    C = [.9, .8, .7, .6, .5]
    G = [1, 2, 4]
    B = {}
    N = [f"node{i}" for i in range(0, num_instances)]

    # L_k  = {"len256":256, "len512":512}
    P_k = {"1024_256":1024, "1280_256":1280, "1536_256":1536}
    O_k = {"1024_256":256, "1280_256":256, "1536_256":256}
    SLO_k = {"1024_256":50.0, "1280_256":5.0, "1536_256":5.0}
    RPS_k = {"1024_256":1.1333, "1280_256":0.1, "1536_256":0.0333}

    T_c   = {.9: 3.18, .8: 4.12, .7: 5.14, .6:6.25, .5: 7.14}
    # T_c = load_Tc_from_logs()

    # read L_gb from latency_profiles/GSAI-ML_LLaDA-8B-Instruct/GH200/TP{i}.json
    L_gb = {}
    for g in G:
        with open(f"latency_profiles/GSAI-ML_LLaDA-8B-Instruct/GH200/TP{g}.json", "r") as f:
            data = json.load(f)
            # make keys int
            data = {int(k): v for k, v in data.items()}
            for b in data.keys():
                L_gb[(g, b)] = data[b]
            B[g] = list(data.keys())
    # L_gb = {
    #     (1, 1280): 0.04741176962852478, (1, 2560): 0.09430396556854248, (1, 3840): 0.14542371034622192, (1, 5120): 0.18064334988594055,
    #     (2, 1280): 0.05359116196632385, (2, 2560): 0.0690348744392395, (2, 3840): 0.10225313901901245, (2, 5120): 0.13222801685333252,
    #     (4, 1280): 0.035597145557403564, (4, 2560): 0.041252315044403076, (4, 3840): 0.06059566140174866, (4, 5120): 0.07760381698608398,
    # }

    G_n = {f"node{i}": 4 for i in range(0, num_instances)}

    m, v = build_capacity_planner_bigM(
        K, C, G, B, N,
        O_k=O_k, P_k=P_k,
        SLO_k=SLO_k, RPS_k=RPS_k,
        T_c=T_c,
        L_gb=L_gb,
        G_n=G_n,
    )

    m.setParam("MIPGap", 0.01)
    m.setParam("TimeLimit", 10)  # Set timeout to 300 seconds
    m.optimize()
    print(f"Status: {m.status}")
    print(f"SolCount: {m.SolCount}")

    if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
        if m.SolCount == 0:
            print("No feasible solution found.")
            exit(0)
        # print(f"Objective value: {m.ObjVal}")
        for i in range(m.NumObj):
            m.setParam('ObjNumber', i)
            print(f"Objective {i}: value = {m.ObjNVal}")
        x_ng = v["x_ng"]
        z_gb = v["z_gb"]
        for g in G:
            chosen_bin = [b for b in B[g] if z_gb[(g,b)].X > 0.5]
            print(f"TP={g} batch={chosen_bin}")
            total_inst = sum(x_ng[n,g].X for n in N)
            print(f"  total instances = {total_inst:.2f}")
        
        # pring s_kc
        s_kc = v["s_kc"]
        for k in K:
            print(f"Class {k}:")
            for c in C:
                print(f"  Confidence {c}: s_kc = {s_kc[k,c].X:.2f}")
        
        # print s_kg
        s_kg = v["s_kg"]
        for k in K:
            print(f"Class {k}:")
            for g in G:
                print(f"  TP {g}: s_kg = {s_kg[k,g].X:.2f}")
                
        old_config = {
            0: [0,1,2,3],
            1: [4,5,6,7],
            # 2: [8,9],
            # 3: [10],
            # 4: [11],
            # 5: [12, 13],
            # 6: [14, 15]
        }
        assert len(G_n) == num_instances
        # derive from G_n
        node_to_bundles = {}
        cur_bundle_id = 0
        for i in range(num_instances):
            n = f"node{i}"
            node_to_bundles[n] = []
            for _ in range(G_n[n]): # 4 bundles per GPU
                node_to_bundles[n].append(cur_bundle_id)
                cur_bundle_id += 1
        
        # new_config = get_new_config_ilp(
        #     G,
        #     node_to_bundles,
        #     x_ng=x_ng,
        #     old_config=old_config,
        # )
        # print(f"Old config:", old_config)
        # print("New config:", new_config)
