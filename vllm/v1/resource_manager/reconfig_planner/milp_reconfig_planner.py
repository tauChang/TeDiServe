from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.utils import LatencyProfile
from vllm.v1.engine.utils import get_slurm_assigned_cpus
from vllm.v1.executor.executors_manager import get_latency_profile_path
from vllm.v1.resource_manager.workload_monitor import WorkloadClass

from datetime import datetime
from gurobipy import Model, GRB, quicksum
import asyncio
from concurrent.futures import ProcessPoolExecutor
import os
import logging
from typing import Optional
import psutil

NUM_THREADS = 8
PLANNER_PINNED_CPUS: Optional[list[int]] = None

logger = init_logger(__name__)

def get_timestamp(include_ms: bool = False) -> str:
    if not include_ms:
        return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")[:-3]


def _attach_file_logger(log_file: str):
    """Attach a module-level FileHandler writing DEBUG logs to `log_file`.

    This is idempotent: if a FileHandler already exists for the same
    absolute path it will not be added again. Uses append mode so the
    Gurobi log (written by the solver) and our logger can share the file.
    """
    if not log_file:
        return
    try:
        abs_path = os.path.abspath(log_file)
    except Exception:
        return

    # Remove any FileHandlers pointing to a different file to avoid log mixing
    for h in list(logger.handlers):
        try:
            if isinstance(h, logging.FileHandler):
                h_path = os.path.abspath(getattr(h, "baseFilename", ""))
                if h_path != abs_path:
                    logger.removeHandler(h)
                    h.close()
        except Exception:
            continue

    # Add handler for current file if not already present
    for h in logger.handlers:
        try:
            if isinstance(h, logging.FileHandler) and os.path.abspath(getattr(h, "baseFilename", "")) == abs_path:
                return  # Already attached
        except Exception:
            continue

    fh = logging.FileHandler(abs_path, mode="a")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s Line %(lineno)d: %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Ensure there's a console/stream handler so messages are visible on CLI.
    has_stream = False
    for h in logger.handlers:
        try:
            if isinstance(h, logging.StreamHandler):
                has_stream = True
                break
        except Exception:
            continue
    if not has_stream:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s Line %(lineno)d: %(message)s"))
        logger.addHandler(ch)

def get_minimum_movement_config(
        x_ng: dict,                       # {(node, g): Var/float/int} -> solved MILP counts (aggregate per g)
        old_config: dict[int, list[int]], # {me_id: [bundle_ids]}
        node_to_bundles: dict[int, list[int]],
        candidate_tp_degree: list[int],
        log_file: str,
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
    # Candidate TP degrees (e.g., [4,2,1]) – you already have this:
    G = list(candidate_tp_degree)

    # Required counts per degree g (aggregate across nodes)
    required_counts: dict[int, int] = {}
    for (n, g), var in x_ng.items():
        val = getattr(var, "X", var)
        logger.debug(f"x_ng[{n},{g}] = {val}")
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
    # m.Params.OutputFlag = 0  # quiet; toggle to 1 if you want logs

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
            m.addConstr(quicksum(sum_terms) == 1, name=f"exclusive[{b}]")

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
    m.setParam("LogFile", log_file)
    m.setParam("LogToConsole", 0)
    m.setParam("Threads", NUM_THREADS)
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


def _apply_fixed_xng_constraints(m, x_ng, current_config, node_to_bundles, candidate_tp_degree):
    # Convert {me_id: [bundle_ids]} into counts per node
    # Example: node0 has 4 bundles assigned to 1 executor (so TP4=1)

    node_assignment = {n: {} for n in node_to_bundles}
    # Count TP choices
    for me_id, bundles in current_config.items():
        tp_degree = len(bundles)
        # find which node the bundles belong to
        for n, node_bundles in node_to_bundles.items():
            if all(b in node_bundles for b in bundles):
                node_assignment[n][tp_degree] = node_assignment[n].get(tp_degree, 0) + 1
    
    # for others, set to zero
    for n in node_assignment:
        for g in candidate_tp_degree:
            if g not in node_assignment[n]:
                node_assignment[n][g] = 0

    # Now enforce
    for n, tp_counts in node_assignment.items():
        for g, cnt in tp_counts.items():
            m.addConstr(x_ng[n, g] == cnt, name=f"fix_xng[{n},{g}]")


def get_maximum_confidence_abstract_config(
        node_to_bundles: dict[int, list[int]],
        current_config: dict[int, list[int]],
        workload_classes: list[WorkloadClass],
        fix_x_ng: bool,
        candidate_tp_degree: list[int],
        candidate_confidence_thresholds: list[float],
        latency_profiles: dict[int, LatencyProfile],
        confidence_unmasked_tokens_per_step: dict[float, int],
        cache_prefix: bool,
        cache_suffix: bool,
        denoise_block_size: int,
        log_file: str,
    ):
    # returns x_ng, objective
        # --- Inputs ---
    K = [workload_class.name for workload_class in workload_classes]
    C = candidate_confidence_thresholds
    G = candidate_tp_degree
    B = {tp_degree: latency_profiles[tp_degree].batch_sizes for tp_degree in G}
    N = list(node_to_bundles.keys())

    P_k = {workload_class.name: workload_class.prompt_length for workload_class in workload_classes}
    O_k = {workload_class.name: workload_class.output_length for workload_class in workload_classes}
    SLO_k = {workload_class.name: workload_class.slo for workload_class in workload_classes}
    RPS_k = {workload_class.name: workload_class.rps for workload_class in workload_classes}
    
    T_c = confidence_unmasked_tokens_per_step
    L_gb = {(tp_degree, batch_size): latency_profiles[tp_degree].lookup(batch_size)[1] for tp_degree in G for batch_size in B[tp_degree]}

    G_n = {node: len(node_to_bundles[node]) for node in N}

    DB_k = {}
    for k in K:
        if cache_prefix and cache_suffix:
            DB_k[k] = denoise_block_size
        elif cache_prefix and not cache_suffix:
            DB_k[k] = O_k[k] // 2
        else:
            DB_k[k] = 0

    # Pre-solve SLO diagnostics.
    # This is not the exact constraint value, but it gives a quick lower-bound
    # style check using the fastest latency profile available.
    # slo_lb_by_class = {}
    # for k in K:
    #     fastest_latency = min(L_gb[(g, b)] for g in G for b in B[g])
    #     slo_lb_by_class[k] = RPS_k[k] * (P_k[k] + O_k[k]) * fastest_latency
    # logger.info(
    #     "SLO lower-bound diagnostics: "
    #     + ", ".join(
    #         f"{k}: demand_rps={RPS_k[k]:.4f}, prompt={P_k[k]}, output={O_k[k]}, "
    #         f"fastest_latency={min(L_gb[(g, b)] for g in G for b in B[g]):.6f}, "
    #         f"lhs_lower_bound={slo_lb_by_class[k]:.6f}, rhs={SLO_k[k]:.6f}"
    #         for k in K
    #     )
    # )

    # Create model
    # Ensure module logger writes to the same Gurobi log file
    # _attach_file_logger(log_file)

    m = Model("reconfig_planner")
    
    # --- Decision Variables ---
    # ------ Main decision variables ------
    # Integer number of TP=g instances on node n
    x_ng = m.addVars(N, G, lb=0.0, vtype=GRB.INTEGER, name="x_ng")

    # ------ Auxiliary decision variables ------
    # Steps per class k at confidence c
    s_kc = m.addVars(K, C, lb=0.0, vtype=GRB.CONTINUOUS, name="s_kc")
    sp_kc = m.addVars(K, C, lb=0.0, vtype=GRB.CONTINUOUS, name="sp_kc")

    # Steps per class k executed on TP degree g
    s_kg = m.addVars(K, G, lb=0.0, vtype=GRB.CONTINUOUS, name="s_kg")
    sp_kg = m.addVars(K, G, lb=0.0, vtype=GRB.CONTINUOUS, name="sp_kg")

    # Binary batch-bin choice per TP degree g
    z_gb = {(g,b): m.addVar(vtype=GRB.BINARY, name=f"z[{g},{b}]")
            for g in G for b in B[g]}
    zp_gb = {(g,b): m.addVar(vtype=GRB.BINARY, name=f"zp[{g},{b}]")
                for g in G for b in B[g]}
    
    # SLO slack per class k
    r_k = m.addVars(K, lb=-1000000.0, vtype=GRB.CONTINUOUS, name="r_k")

    m.update()
    
    # --- Objective ---
    # Maximize avg confidence per token
    total_RPS = sum(RPS_k[k] for k in K)
    token_conf_obj = quicksum(
        (RPS_k[k] / O_k[k]) * quicksum(c * T_c[c] * (s_kc[k, c]+sp_kc[k, c]) for c in C)
        for k in K
    ) / total_RPS
    # m.setObjective(obj, GRB.MAXIMIZE)
    slack_obj = quicksum(r_k[k] / SLO_k[k] * RPS_k[k] / total_RPS for k in K)

    # minimize number of executors used as third objective
    executor_count_obj = quicksum(x_ng[n,g] for n in N for g in G)

    m.setObjectiveN(-token_conf_obj, index=0, priority=2, name="max_token_conf")
    m.setObjectiveN(-slack_obj,index=1, priority=1, name="max_slack")
    m.setObjectiveN(executor_count_obj, index=2, priority=0, name="min_executors")
    
    # --- Constraints ---
    # -1) Constrain x_ng if specified
    if fix_x_ng:
        _apply_fixed_xng_constraints(m, x_ng, current_config, node_to_bundles, candidate_tp_degree)
        # print and make sure constraints are applied
        for constr in m.getConstrs():
            if constr.ConstrName.startswith("fix_xng"):
                logger.info(f"Applied constraint: {constr.ConstrName}")
        
    # 0) If no caching, enforce sp_kc, sp_kg to be zero (all steps are recomputes)
    if not cache_prefix and not cache_suffix:
        for k in K:
            for c in C:
                m.addConstr(sp_kc[k,c] == 0, name=f"no_cache_spkc[{k},{c}]")
            for g in G:
                m.addConstr(sp_kg[k,g] == 0, name=f"no_cache_spkg[{k},{g}]")
    else:
        # If caching enabled, ensure sp_kc and sp_kg are consistent
        for k in K:
            m.addConstr(
                quicksum(sp_kc[k,c] for c in C) == O_k[k] // DB_k[k],
                name=f"cache_step_consistency[{k}]"
            )
    
    # 1) Token completion
    for k in K:
        m.addConstr(quicksum(T_c[c] * (s_kc[k, c] + sp_kc[k, c]) for c in C) == O_k[k],
                    name=f"token_completion[{k}]")
        
    # 2) Step accounting
    for k in K:
        # m.addConstr(quicksum((s_kc[k, c] + sp_kc[k, c]) for c in C)
        #             == quicksum((s_kg[k, g] + sp_kg[k, g]) for g in G),
        #             name=f"step_accounting[{k}]")
        m.addConstr(quicksum(s_kc[k, c] for c in C)
                    == quicksum(s_kg[k, g] for g in G),
                    name=f"step_accounting[{k}]")
        m.addConstr(quicksum(sp_kc[k, c] for c in C)
                    == quicksum(sp_kg[k, g] for g in G),
                    name=f"step_accounting_p[{k}]")
    
    # 3) SLO
    for k in K:
        lhs = quicksum(
            s_kg[k, g] * quicksum(L_gb[(g,b)] * z_gb[(g,b)] for b in B[g]) + \
            sp_kg[k, g] * quicksum(L_gb[(g,b)] * zp_gb[(g,b)] for b in B[g])
            for g in G
        )
        m.addConstr(lhs + r_k[k] == SLO_k[k], name=f"SLO[{k}]")
    
    # 4) Batch size selection
    for g in G:
        m.addConstr(quicksum(z_gb[(g,b)] for b in B[g]) == 1,
                    name=f"one_bin[{g}]")
        m.addConstr(quicksum(zp_gb[(g,b)] for b in B[g]) == 1,
                    name=f"one_binp[{g}]")
        # limit zp_gb <= 512
        # for b in B[g]:
        #     if b > 512:
        #         m.addConstr(zp_gb[(g,b)] == 0, name=f"cache_bin_limit[{g},{b}]")
    
    # 5) Node capacity
    for n in N:
        m.addConstr(
            quicksum(g * x_ng[n,g] for g in G) <= G_n[n],
            name=f"gpu_cap[{n}]"
        )

    # 6) Aggregate Token Throughput
    for g in G:
        # sec per request-step for recompute, linear via z
        rec_sec_per_reqstep = quicksum(z_gb[(g,b)]  * (L_gb[(g,b)] / b) for b in B[g])
        # sec per request-step for cache, linear via zp
        cache_sec_per_reqstep = quicksum(zp_gb[(g,b)] * (L_gb[(g,b)] / b) for b in B[g])

        lhs = quicksum(
            RPS_k[k] * (
                s_kg[k,g]  * (P_k[k] + O_k[k]) * rec_sec_per_reqstep +
                sp_kg[k,g] * DB_k[k]           * cache_sec_per_reqstep
            )
            for k in K
        )
        rhs = quicksum(x_ng[n,g] for n in N)  # executor-seconds per second
        m.addConstr(lhs <= rhs, name=f"time_budget_g{g}")
    # M_big = 1e9  # Big-M constant
    # for g in G:
    #     for b in B[g]:
    #         lhs = quicksum(RPS_k[k] * (s_kg[k,g] * (P_k[k] + O_k[k]) + (sp_kg[k,g] * DB_k[k])) for k in K)
    #         rhs = (b / L_gb[(g,b)]) * quicksum(x_ng[n,g] for n in N)
    #         m.addConstr(lhs <= rhs + M_big * (1 - z_gb[(g,b)]),
    #                     name=f"token_throughput_{g}_{b}")

    
    # for g in G:
    #     lhs = quicksum(RPS_k[k] * s_kg[k,g] * (P_k[k] + O_k[k]) for k in K)
    #     # rhs = (b / L_gb[(g,b)]) * quicksum(x_ng[n,g] for n in N)
    #     rhs = quicksum(z_gb[g,b] * (b / L_gb[g,b]) for b in B[g]) \
    #         * quicksum(x_ng[n,g] for n in N)
    #     m.addConstr(lhs <= rhs,
    #                 name=f"token_throughput_{g}")
    
    # lhs = quicksum(
    #     RPS_k[k] * quicksum(s_kg[k, g] * (P_k[k] + O_k[k]) for g in G) / SLO_k[k]
    #     for k in K
    # )
    # rhs = quicksum(
    #     z_gb[(g, b)] * b * (1 / L_gb[(g, b)]) *
    #     quicksum(x_ng[n, g] for n in N)
    #     for g in G
    #     for b in B[g]
    # )
    # m.addConstr(lhs <= rhs, name="global_token_throughput")
    
    # 7) Each batch must have at least one request
    for g in G:
        m.addConstr(
            quicksum(b * z_gb[(g,b)] for b in B[g]) >= max(P_k[k] + O_k[k] for k in K),
            name=f"batch_nonzero_req[{k},{g}]"
        )

    # 8) All GPUs must be used
    for n in N:
        m.addConstr(
            quicksum(g * x_ng[n,g] for g in G) == G_n[n],
            name=f"all_gpu_used[{n}]"
        )
    
    # --- Solve ---
    m.update()
    # m.setParam("OutputFlag", 0)
    m.setParam("LogFile", log_file)
    m.setParam("LogToConsole", 0) 
    m.setParam("Threads", NUM_THREADS)
    m.setParam("MIPGap", 0.01)
    m.setParam("TimeLimit", 20)
    m.ObjNAbsTol = 0.0
    m.ObjNRelTol = 0.0
    m.optimize()

    if m.SolCount > 0:
        for k in K:
            lhs_val = 0.0
            for g in G:
                rec_sec_per_reqstep = sum(L_gb[(g, b)] * z_gb[(g, b)].X for b in B[g])
                cache_sec_per_reqstep = sum(L_gb[(g, b)] * zp_gb[(g, b)].X for b in B[g])
                lhs_val += (
                    s_kg[k, g].X * rec_sec_per_reqstep +
                    sp_kg[k, g].X * cache_sec_per_reqstep
                )
            logger.info(
                "SLO solution value for %s: lhs=%.6f rhs=%.6f slack=%.6f",
                k,
                lhs_val,
                SLO_k[k],
                r_k[k].X,
            )

    x_ng_sol = {}
    if m.SolCount == 0:
        logger.info("No feasible solution found.")
        for n in N:
            for g in G:
                if g == 1:
                    x_ng_sol[n,g] = G_n[n]
                else:
                    x_ng_sol[n,g] = 0
    else:
        if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
            # logger.info(f"Objective value: {m.objVal:.3f}")
            for i in range(m.NumObj):
                m.setParam('ObjNumber', i)
                logger.info(f"Objective {i}: value = {m.ObjNVal:.3f}")
            for g in G:
                recompute_chosen_bin = [b for b in B[g] if z_gb[(g,b)].X > 0.5]
                cache_chosen_bin = [b for b in B[g] if zp_gb[(g,b)].X > 0.5]
                logger.info(f"TP={g} recompute batch={recompute_chosen_bin} cache batch={cache_chosen_bin}")
                total_inst = sum(x_ng[n,g].X for n in N)
                logger.info(f"  total instances = {total_inst:.2f}")
            
            # pring s_kc
            for k in K:
                logger.info(f"Class {k}:")
                for c in C:
                    logger.info(f"  Confidence {c}: s_kc = {s_kc[k,c].X:.2f}")
                    logger.info(f"  Confidence {c}: sp_kc = {sp_kc[k,c].X:.2f}")
                
            for k in K:
                logger.info(f"Class {k}: r_k = {r_k[k].X:.2f}")
            
            # print s_kg
            for k in K:
                logger.info(f"Class {k}:")
                for g in G:
                    logger.info(f"  TP {g}: s_kg = {s_kg[k,g].X:.2f}")
                    logger.info(f"  TP {g}: sp_kg = {sp_kg[k,g].X:.2f}")
            
            # extract x_ng solution
            x_ng_sol = {}
            for n in N:
                for g in G:
                    x_ng_sol[n,g] = int(round(x_ng[n,g].X))

            # Prepare new configuration
        else:
            raise RuntimeError(f"Reconfiguration MILP ended with status {m.status}")
        
    if m.SolCount == 0:
        return x_ng_sol, 0, (None, None, None)
    else:
        return x_ng_sol, m.SolCount, (token_conf_obj.getValue(), slack_obj.getValue(), executor_count_obj.getValue())

def plan_reconfiguration(
        node_to_bundles: dict[int, list[int]],
        current_config: dict[int, list[int]], # me_id -> [bundle_ids]
        workload_classes: list[WorkloadClass],
        candidate_tp_degree: list[int],
        candidate_confidence_thresholds: list[float],
        latency_profile_paths: dict[int, str],
        confidence_unmasked_tokens_per_step: dict[float, int],
        cache_prefix: bool,
        cache_suffix: bool,
        denoise_block_size: int,
        log_dir: str,
        compare_with_fixed: bool = True
    ) -> dict[int, list[int]]:
    # create a subfolder using date/time
    log_dir = os.path.join(log_dir, f"{datetime.now().strftime("%Y%m%d")}", f"{datetime.now().strftime("%H%M%S")}")
    os.makedirs(log_dir, exist_ok=True)
    # attach a file handler so module logger writes to the same files
    _attach_file_logger(f"{log_dir}/planner.log")


    p = psutil.Process()
    global NUM_THREADS, PLANNER_PINNED_CPUS
    if PLANNER_PINNED_CPUS is None:
        all_cpus = get_slurm_assigned_cpus()
        eligible_cpus = p.cpu_affinity()
        cpus_to_use = [cpu for cpu in all_cpus if cpu in eligible_cpus]
        if not cpus_to_use:
            cpus_to_use = eligible_cpus
        cpus_to_use = cpus_to_use[len(cpus_to_use)//2:len(cpus_to_use) * 3//4]  # use half of the assigned CPUs
        if not cpus_to_use:
            cpus_to_use = eligible_cpus[:max(1, len(eligible_cpus)//2)]

        p.cpu_affinity(cpus_to_use)
        PLANNER_PINNED_CPUS = list(cpus_to_use)
        logger.info(f"EngineCore process pinned to CPUs: {PLANNER_PINNED_CPUS}")
    else:
        logger.info(
            "EngineCore process affinity already pinned; reusing CPUs: %s",
            PLANNER_PINNED_CPUS,
        )

    cpus_to_use = PLANNER_PINNED_CPUS
    NUM_THREADS = min(32, len(cpus_to_use))

    logger.debug(f"Reconfig planner in process {p.pid}")
    if len(workload_classes) == 0:
        logger.info("No workload detected; keeping current configuration.")
        return current_config
    
    # scale workload classes RPS by 2 to provide buffer
    for wc in workload_classes:
        wc.rps = wc.rps * 1.5
        logger.debug(f"Scaled workload class {wc.name} RPS to {wc.rps}")
    
    # read latency profile
    latency_profiles = {}
    for tp_degree, path in latency_profile_paths.items():
        latency_profiles[tp_degree] = LatencyProfile(path)
    
    # Free
    x_ng_free, sol_count_free, objectives_free = get_maximum_confidence_abstract_config(
        node_to_bundles,
        current_config,
        workload_classes,
        fix_x_ng=False,
        candidate_tp_degree=candidate_tp_degree,
        candidate_confidence_thresholds=candidate_confidence_thresholds,
        latency_profiles=latency_profiles,
        confidence_unmasked_tokens_per_step=confidence_unmasked_tokens_per_step,
        cache_prefix=cache_prefix,
        cache_suffix=cache_suffix,
        denoise_block_size=denoise_block_size,
        log_file=f"{log_dir}/free_abstract.log"
    )

    new_config_free = get_minimum_movement_config(
        x_ng_free, current_config, node_to_bundles, candidate_tp_degree, f"{log_dir}/free_movement.log")
    
    if not compare_with_fixed:
        new_config = new_config_free
    else:
        # constrained
        x_ng_fixed, sol_count_fixed, objectives_fixed = get_maximum_confidence_abstract_config(
            node_to_bundles,
            current_config,
            workload_classes,
            fix_x_ng=True,
            candidate_tp_degree=candidate_tp_degree,
            candidate_confidence_thresholds=candidate_confidence_thresholds,
            latency_profiles=latency_profiles,
            confidence_unmasked_tokens_per_step=confidence_unmasked_tokens_per_step,
            cache_prefix=cache_prefix,
            cache_suffix=cache_suffix,
            denoise_block_size=denoise_block_size,
            log_file=f"{log_dir}/fixed_abstract.log"
        )
        new_config_fixed = current_config

        logger.debug(f"Free: {sol_count_free} solutions found with objectives {objectives_free}")
        logger.debug(f"Fixed: {sol_count_fixed} solutions found with objectives {objectives_fixed}")

        # if no feasible solution with fixed x_ng, use free one
        if sol_count_fixed == 0:
            logger.info("No feasible reconfiguration found that preserves current executor counts; using unconstrained solution.")
            new_config = new_config_free
        else:
            # if objective improve by 10% or more, use free
            conf_free = objectives_free[0]
            slack_free = objectives_free[1]
            conf_fixed = objectives_fixed[0]
            slack_fixed = objectives_fixed[1]
            logger.debug(f"Free confidence: {conf_free}, slack: {slack_free}")
            logger.debug(f"Fixed confidence: {conf_fixed}, slack: {slack_fixed}")
            if conf_free is not None and conf_fixed is not None:
                # make positive
                if conf_free > conf_fixed * 1.1 or \
                    (conf_free >= conf_fixed and slack_free > slack_fixed * 1.1):
                    logger.info("Significant objective improvement with unconstrained solution; using it.")
                    new_config = new_config_free
                else:
                    logger.info("Using constrained solution that preserves current executor counts.")
                    new_config = new_config_fixed
            else:
                # no solution found for both free and fixed. Use fixed (all TP=1's)
                logger.debug("No solutions found for both constrained and unconstrained; using TP=1 configuration.")
                new_config = new_config_fixed
    

    # check new_config validity
    # each bundle is assigned exactly once
    assigned_bundles = [b for bundles in new_config.values() for b in bundles]
    all_bundles = [b for bundles in node_to_bundles.values() for b in bundles]
    assert sorted(assigned_bundles) == sorted(all_bundles), f"Bundle assignment mismatch: assigned {assigned_bundles}, all {all_bundles}"
    assert len(assigned_bundles) == len(set(assigned_bundles)), f"Some bundles assigned multiple times {assigned_bundles}"
    # if an old executor is kept, its bundles must match
    for me_id, bundles in new_config.items():
        if me_id in current_config:
            old_bundles = current_config[me_id]
            assert set(bundles) == set(old_bundles), f"Kept executor {me_id} has different bundles: old {old_bundles}, new {bundles}"
    return new_config
        
class MILPReconfigPlanner:
    def __init__(self,
                 vllm_config: VllmConfig = None,
                 latency_profile_paths: dict[int, str] = None,
                 cache_prefix: bool = None, # for simulation
                 cache_suffix: bool = None, # for simulation
                 denoise_block_size: int = None, # for simulation
                 ):
        self.vllm_config = vllm_config
        if vllm_config is not None:
            self.cache_prefix = vllm_config.model_config.cache_prefix
            self.cache_suffix = vllm_config.model_config.cache_suffix
            self.denoise_block_size = vllm_config.model_config.denoise_block_size
            self.log_dir = f"{vllm_config.experiment_config.experiment_dir}/reconfig_planner_logs/"
        else:
            self.cache_prefix = cache_prefix
            self.cache_suffix = cache_suffix
            self.denoise_block_size = denoise_block_size
            self.log_dir = "./simulation_reconfig_planner_logs"
        
        os.makedirs(self.log_dir, exist_ok=True)

        # TODO
        self.candidate_confidence_thresholds = [.9, .8, .7, .6, .5]
        # self.candidate_confidence_thresholds = [.9]
        # # from "step_data_kiet/gsm8k_100/256/"
        # self.confidence_unmasked_tokens_per_step = {
        #     .9: 3.18, 
        #     .8: 4.12, 
        #     .7: 5.14, 
        #     .6: 6.25, 
        #     .5: 7.14
        #     }
        # from "step_data_dual_cache_with_output/dual_*_256"
        self.confidence_unmasked_tokens_per_step = {
            .9: 3.02, 
            .8: 3.89, 
            .7: 4.77, 
            .6: 5.65, 
            .5: 6.58
            # .9: 1.86,
            # .8: 2.09,
            # .7: 2.36,
            # .6: 2.6,
            # .5: 2.94
            }
        # self.candidate_tp_degree = [4, 2, 1]
        self.candidate_tp_degree = [4,2,1]
        self.latency_profile_paths = {}
        if not latency_profile_paths:
            for tp_degree in self.candidate_tp_degree:
                path = get_latency_profile_path(vllm_config, tp_degree)
                self.latency_profile_paths[tp_degree] = path
        else:
            self.latency_profile_paths = latency_profile_paths
        # if not latency_profile_paths:
        #     self.latency_profile_dir = vllm_config.profile_config.latency_profile_dir

        #     self.latency_profiles: dict[int, LatencyProfile] = {}
        #     # for tp_degree in [1, 2, 4]:
        #     for tp_degree in self.candidate_tp_degree:
        #         path = get_latency_profile_path(vllm_config, tp_degree)
        #         self.latency_profiles[tp_degree] = LatencyProfile(path)
        # else:
        #     self.latency_profiles: dict[int, LatencyProfile] = {}
        #     for tp_degree, path in latency_profile_paths.items():
        #         self.latency_profiles[tp_degree] = LatencyProfile(path)
        
        self.pool = ProcessPoolExecutor(max_workers=1)
    
    async def plan_reconfiguration_async(self,
                                    node_to_bundles: dict[int, list[int]],
                                    current_config: dict[int, list[int]], # me_id -> [bundle_ids]
                                    workload_classes: list[WorkloadClass],
                                    compare_with_fixed: bool = True
                                    ) -> dict[int, list[int]]:
        loop = asyncio.get_event_loop()
        logger.debug(f"main process {os.getpid()}")
        return await loop.run_in_executor(
            # None,
            self.pool,
            plan_reconfiguration,
            node_to_bundles,
            current_config,
            workload_classes,
            self.candidate_tp_degree,
            self.candidate_confidence_thresholds,
            self.latency_profile_paths,
            self.confidence_unmasked_tokens_per_step,
            self.cache_prefix,
            self.cache_suffix,
            self.denoise_block_size,
            self.log_dir,
            compare_with_fixed
        )

                

    # def get_new_config(self,
    #                    x_ng: dict,
    #                    old_config: dict[int, list[int]],
    #                    ) -> dict[int, list[int]]:
    #     # config is {me_id: [bundle_ids]}
    #     # abstract_config is [(tp_degree, me_id)]
    #     old_abstract_config = sorted(
    #         [(len(bundle_ids), me_id) for me_id, bundle_ids in old_config.items()],
    #         reverse=True
    #     )
    #     new_abstract_config = []
    #     for (n, g), var in x_ng.items():
    #         num_instances = int(var.X)
    #         new_abstract_config.extend([(g, None)] * num_instances)
    #     new_abstract_config = sorted(new_abstract_config, reverse=True)

    #     # Map old me_id to new me_id
            

    # def get_new_config(self,
    #                    x_ng: dict,
    #                    old_config: dict[int, list[int]],
    #                    ) -> dict[int, list[int]]:
    #     """
    #     Build a new mapping: model_executor_id -> [bundle_ids]
    #     using the solved counts x_ng[(node, g)] (number of TP=g instances on node).

    #     Heuristic to minimize movement:
    #       1) Keep executors that already match (node, g) exactly.
    #       2) Reuse surplus executor IDs for new placements before creating new IDs.
    #       3) Allocate bundles greedily on the requested node, largest g first.

    #     Assumes:
    #       - self.node_to_bundles: Dict[node_ip, List[bundle_id]]
    #       - Each existing executor in old_config already uses bundles all on the same node.
    #     """
    #     # ---- Helpers / derived structures ----
    #     # Map bundle -> node
    #     bundle_to_node: dict[int, str] = {}
    #     for node_ip, blist in self.node_to_bundles.items():
    #         for b in blist:
    #             bundle_to_node[b] = node_ip

    #     # Group current executors by (node, g) and validate single-node placement
    #     current_by_node_g: dict[tuple[int, int], list[tuple[int, list[int]]]] = {}
    #     for me_id, bundles in old_config.items():
    #         if not bundles:
    #             continue
    #         nodes = {bundle_to_node[b] for b in bundles}
    #         if len(nodes) != 1:
    #             raise ValueError(
    #                 f"Executor {me_id} spans multiple nodes: {nodes}. "
    #                 "This planner requires per-executor single-node bundles."
    #             )
    #         node = next(iter(nodes))
    #         g = len(bundles)
    #         current_by_node_g.setdefault((node, g), []).append((me_id, bundles))

    #     # Requested counts per (node, g) from solution
    #     req_counts: dict[tuple[str, int], int] = {}
    #     for (n, g), var in x_ng.items():
    #         count = int(round(getattr(var, "X", var)))  # support gurobi Var or plain int for testing
    #         if count > 0:
    #             req_counts[(n, g)] = req_counts.get((n, g), 0) + count

    #     # Available bundles per node (we will subtract kept allocations)
    #     available_by_node: dict[str, list[int]] = {
    #         n: list(self.node_to_bundles.get(n, [])) for n in self.node_to_bundles
    #     }

    #     # New config we are constructing
    #     new_config: dict[int, list[int]] = {}

    #     # Pool of executor IDs we can reuse (surplus after keeps)
    #     surplus_ids: list[int] = []

    #     # Keep track of max id to create new ones if needed
    #     next_id = (max(old_config.keys()) + 1) if old_config else 0

    #     # ---- Phase 1: KEEP matching executors (node, g) ----
    #     # Sort by larger g first to reduce fragmentation
    #     for (node, g) in sorted(req_counts.keys(), key=lambda t: (-t[1], str(t[0]))):
    #         needed = req_counts[(node, g)]
    #         cur_execs = list(current_by_node_g.get((node, g), []))
    #         keep_cnt = min(needed, len(cur_execs))

    #         # Keep 'keep_cnt' executors unchanged
    #         for me_id, bundles in cur_execs[:keep_cnt]:
    #             new_config[me_id] = list(bundles)
    #             # remove used bundles from availability
    #             for b in bundles:
    #                 if b in available_by_node[node]:
    #                     available_by_node[node].remove(b)
    #         # Update what's still needed
    #         req_counts[(node, g)] = needed - keep_cnt

    #         # Any leftover current executors of this (node, g) become surplus IDs
    #         for me_id, _ in cur_execs[keep_cnt:]:
    #             surplus_ids.append(me_id)

    #     # Any executors that were in old_config but not kept belong in surplus
    #     kept_ids = set(new_config.keys())
    #     for me_id in old_config.keys():
    #         if me_id not in kept_ids and me_id not in surplus_ids:
    #             surplus_ids.append(me_id)

    #     # ---- Phase 2: ASSIGN new placements for remaining needs ----
    #     # Again, allocate larger g first per node
    #     for (node, g) in sorted(req_counts.keys(), key=lambda t: (-t[1], -t[1] and t[1] or 0, -t[1])):
    #         need = req_counts[(node, g)]
    #         if need <= 0:
    #             continue

    #         # Greedily carve g bundles per executor from node's available pool
    #         # Use stable order to be deterministic
    #         available_by_node[node].sort()
    #         idx = 0  # pointer into available list (we'll pop from front)

    #         while need > 0:
    #             if len(available_by_node[node]) < g:
    #                 raise RuntimeError(
    #                     f"Insufficient free bundles on node {node} to place TP={g} "
    #                     f"executors (need {need}, available bundles={len(available_by_node[node])})."
    #                 )
    #             # take first g bundles
    #             chosen = available_by_node[node][:g]
    #             del available_by_node[node][:g]

    #             # pick an executor id: reuse surplus if possible, else new id
    #             if surplus_ids:
    #                 me_id = surplus_ids.pop(0)
    #             else:
    #                 me_id = next_id
    #                 next_id += 1

    #             new_config[me_id] = chosen
    #             need -= 1

    #     # ---- Optional: clean up any remaining surplus IDs (they are "killed" implicitly by diff)
    #     # They just won't appear in new_config; the diff engine will produce KillCommands.

    #     return new_config

    # def get_new_config(self,
    #                 x_ng: dict,
    #                 old_config: dict[int, list[int]],
    #                 ) -> dict[int, list[int]]:
    #     """
    #     Build a new mapping {model_executor_id: [bundle_ids]}.

    #     - Only total counts of each TP degree matter (node placement flexible).
    #     - Each executor must occupy all its bundles on one node.
    #     - Minimize movement by keeping executors with the same TP degree if possible.
    #     """

    #     # ---- Step 1: derive required counts per TP degree ----
    #     required_counts: dict[int, int] = {}
    #     for (node, g), var in x_ng.items():
    #         count = int(round(getattr(var, "X", var)))  # support Var or float
    #         required_counts[g] = required_counts.get(g, 0) + count

    #     # ---- Step 2: current executors by TP degree ----
    #     bundle_to_node = {}
    #     for node_id, bundles in self.node_to_bundles.items():
    #         for b in bundles:
    #             bundle_to_node[b] = node_id

    #     current_by_g: dict[int, list[tuple[int, str, list[int]]]] = {}
    #     for me_id, bundles in old_config.items():
    #         if not bundles:
    #             continue
    #         node = bundle_to_node[bundles[0]]
    #         g = len(bundles)
    #         current_by_g.setdefault(g, []).append((me_id, node, bundles))

    #     # ---- Step 3: initialize available bundles ----
    #     available_by_node = {n: list(blist) for n, blist in self.node_to_bundles.items()}

    #     new_config: dict[int, list[int]] = {}
    #     surplus_ids: list[int] = []
    #     next_id = (max(old_config.keys()) + 1) if old_config else 0

    #     # ---- Step 4: keep existing executors with same g ----
    #     for g, need in sorted(required_counts.items(), reverse=True):
    #         current_execs = current_by_g.get(g, [])
    #         keep_cnt = min(need, len(current_execs))
    #         for me_id, node, bundles in current_execs[:keep_cnt]:
    #             new_config[me_id] = list(bundles)
    #             for b in bundles:
    #                 if b in available_by_node[node]:
    #                     available_by_node[node].remove(b)
    #         required_counts[g] = need - keep_cnt
    #         for me_id, _, _ in current_execs[keep_cnt:]:
    #             surplus_ids.append(me_id)

    #     # Add any others not reused
    #     kept_ids = set(new_config.keys())
    #     for me_id in old_config.keys():
    #         if me_id not in kept_ids and me_id not in surplus_ids:
    #             surplus_ids.append(me_id)

    #     # ---- Step 5: assign new executors to nodes ----
    #     for g, need in sorted(required_counts.items(), reverse=True):
    #         if need <= 0:
    #             continue
    #         # Try to pack large g's first on any node with enough bundles
    #         for _ in range(need):
    #             placed = False
    #             for node_ip, blist in available_by_node.items():
    #                 if len(blist) >= g:
    #                     chosen = blist[:g]
    #                     del blist[:g]
    #                     if surplus_ids:
    #                         me_id = surplus_ids.pop(0)
    #                     else:
    #                         me_id = next_id
    #                         next_id += 1
    #                     new_config[me_id] = chosen
    #                     placed = True
    #                     break
    #             if not placed:
    #                 raise RuntimeError(
    #                     f"Cannot allocate {g} bundles for an executor; "
    #                     f"not enough contiguous bundles across nodes."
    #                 )

    #     return new_config
        