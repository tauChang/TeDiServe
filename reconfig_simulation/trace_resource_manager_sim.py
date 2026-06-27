#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import os
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from vllm.v1.resource_manager.reconfig_planner.milp_reconfig_planner import MILPReconfigPlanner
from vllm.v1.resource_manager.workload_monitor import WorkloadClass
from transformers import AutoTokenizer


TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


@dataclass
class TraceRequestEvent:
    request_id: str
    arrival_time_s: float
    prompt_length: int
    output_length: int
    latency_slo: float


@dataclass
class TriggerSnapshot:
    trigger_time_s: float
    config: dict[int, list[int]]
    workload_classes: list[WorkloadClass]


def load_trace_arrivals(trace_path: str) -> list[float]:
    path = Path(trace_path)
    raw_text = path.read_text().strip()
    if not raw_text:
        return []

    if path.suffix.lower() == ".json" or raw_text.lstrip().startswith("["):
        raw = json.loads(raw_text)
        if not raw:
            return []
        if isinstance(raw[0], dict):
            for key in ("arrival_time", "arrival_time_s", "timestamp", "time"):
                if key in raw[0]:
                    return [float(item[key]) for item in raw]
            raise ValueError(f"Unsupported JSON trace format in {trace_path}")
        return [float(item) for item in raw]

    loaded = np.loadtxt(path)
    return [float(value) for value in np.atleast_1d(loaded).tolist()]


def load_workload_templates(
    workload_spec_path: str | None,
    prompt_length: int,
    output_length: int,
    slo: float,
) -> list[WorkloadClass]:
    if not workload_spec_path:
        return [WorkloadClass(prompt_length=prompt_length, output_length=output_length, slo=slo, rps=1.0)]

    raw = json.loads(Path(workload_spec_path).read_text())
    if isinstance(raw, dict):
        raw = raw.get("classes", raw.get("workload_classes", raw.get("workloads", [])))

    templates: list[WorkloadClass] = []
    for item in raw:
        templates.append(
            WorkloadClass(
                prompt_length=int(item.get("prompt_length", prompt_length)),
                output_length=int(item.get("output_length", output_length)),
                slo=float(item.get("slo", slo)),
                rps=float(item.get("rps", item.get("share", 1.0))),
            )
        )

    if not templates:
        templates.append(WorkloadClass(prompt_length=prompt_length, output_length=output_length, slo=slo, rps=1.0))
    return templates


def sample_sharegpt_requests_local(
    dataset_path: str,
    num_requests: int,
    tokenizer,
    max_seqlen: int,
):
    prompts = []
    prompt_lens = []
    response_lens = []
    with open(dataset_path) as f:
        for line in f:
            data = json.loads(line)
            if len(data.get("conversations", [])) >= 2:
                prompt = data["conversations"][0]["value"]
                res = data["conversations"][1]["value"]
                prompt_token_ids = tokenizer(prompt).input_ids
                completion_token_ids = tokenizer(res).input_ids
                rounded_up_completion_len = ((len(completion_token_ids) + 31) // 32) * 32
                if len(prompt_token_ids) + rounded_up_completion_len < max_seqlen and \
                    len(prompt_token_ids) > 0 and rounded_up_completion_len > 0:
                    prompts.append(prompt)
                    prompt_lens.append(len(prompt_token_ids))
                    response_lens.append(rounded_up_completion_len)

    if not prompts:
        raise ValueError(f"No valid ShareGPT examples found in {dataset_path}")

    sampled_ids = [random.randint(0, len(prompts) - 1) for _ in range(num_requests)]
    sampled_prompts = [prompts[idx] for idx in sampled_ids]
    sampled_prompt_lens = [prompt_lens[idx] for idx in sampled_ids]
    sampled_response_lens = [response_lens[idx] for idx in sampled_ids]
    return sampled_prompts, sampled_prompt_lens, sampled_response_lens


def build_initial_config(node_to_bundles: dict[str, list[int]]) -> dict[int, list[int]]:
    initial_config: dict[int, list[int]] = {}
    # for executor_id, bundles in enumerate(node_to_bundles.values()):
    #     initial_config[executor_id] = list(bundles)
    # return initial_config
    executor_id = 0
    for bundles in node_to_bundles.values():
        for bundle_id in bundles:
            initial_config[executor_id] = [bundle_id]
            executor_id += 1
    return initial_config


def build_node_to_bundles(num_nodes: int, bundles_per_node: int) -> dict[str, list[int]]:
    node_to_bundles: dict[str, list[int]] = {}
    bundle_id = 0
    for node_index in range(num_nodes):
        bundles = list(range(bundle_id, bundle_id + bundles_per_node))
        node_to_bundles[f"node{node_index}"] = bundles
        bundle_id += bundles_per_node
    return node_to_bundles


class SyntheticWorkloadMonitor:
    def __init__(self, time_window_s: float, start_time_s: float = 0.0, binsize: int = 256):
        self.time_window_s = time_window_s
        self.start_time_s = start_time_s
        self.binsize = binsize
        self.request_arrival_stats_queue: Deque[TraceRequestEvent] = deque()

    def record_request_arrival(self, event: TraceRequestEvent) -> None:
        self.request_arrival_stats_queue.append(event)

    def _purge_old_requests(self, now_s: float) -> None:
        cutoff = now_s - self.time_window_s
        while self.request_arrival_stats_queue and self.request_arrival_stats_queue[0].arrival_time_s < cutoff:
            self.request_arrival_stats_queue.popleft()

    def get_workload_classes(self, now_s: float) -> list[WorkloadClass]:
        self._purge_old_requests(now_s)

        duration = max(1e-6, min(self.time_window_s, now_s - self.start_time_s))
        grouped: dict[tuple[int, int], list[TraceRequestEvent]] = defaultdict(list)
        for event in self.request_arrival_stats_queue:
            p_bin = math.ceil(event.prompt_length / self.binsize) * self.binsize
            o_bin = math.ceil(event.output_length / self.binsize) * self.binsize
            grouped[(p_bin, o_bin)].append(event)

        workload_classes: list[WorkloadClass] = []
        for (prompt_bin, output_bin), events in grouped.items():
            rps = len(events) / duration
            avg_slo = sum(event.latency_slo for event in events) / len(events)
            workload_classes.append(
                WorkloadClass(
                    prompt_length=prompt_bin,
                    output_length=output_bin,
                    slo=avg_slo,
                    rps=rps,
                )
            )

        workload_classes.sort(key=lambda wc: (wc.prompt_length, wc.output_length))
        return workload_classes


def summarize_configurations(snapshots: list[TriggerSnapshot]) -> pd.DataFrame:
    records = []
    for snapshot in snapshots:
        tp_counts = defaultdict(int)
        for bundles in snapshot.config.values():
            tp_counts[len(bundles)] += 1

        row = {"time_s": snapshot.trigger_time_s}
        for tp_degree, count in tp_counts.items():
            row[f"TP{tp_degree}"] = count
        records.append(row)

    return pd.DataFrame(records).fillna(0).sort_values("time_s")


def plot_reconfig_timeline(df: pd.DataFrame, snapshots: list[TriggerSnapshot], output_path: str) -> None:
    if df.empty:
        return

    tp_cols = [col for col in ["TP4", "TP2", "TP1"] if col in df.columns]
    time_steps = df["time_s"].to_numpy()
    total_rps = [sum(wc.rps for wc in snapshot.workload_classes) for snapshot in snapshots]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.2]},
    )

    ax_top.plot(time_steps, total_rps, "k--o", linewidth=2.5, label="Total RPS", markersize=5)
    ax_top.set_ylabel("RPS")
    ax_top.set_title("Trace-driven reconfiguration timeline")
    ax_top.grid(True, linestyle="--", alpha=0.5)
    ax_top.legend()

    bar_width = max(1.0, (time_steps[1] - time_steps[0]) * 0.35) if len(time_steps) > 1 else 0.6
    tp_colors = {"TP1": "#1f77b4", "TP2": "#2ca02c", "TP4": "#d62728"}
    bottom = np.zeros(len(time_steps))
    for tp in tp_cols:
        ax_bottom.bar(
            time_steps,
            df[tp],
            bottom=bottom,
            width=bar_width,
            label=tp,
            color=tp_colors[tp],
            edgecolor="black",
            linewidth=0.8,
        )
        bottom += df[tp].to_numpy()

    ax_bottom.set_ylabel("Executors")
    ax_bottom.set_xlabel("Simulation time (s)")
    ax_bottom.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax_bottom.legend()

    plt.xticks(time_steps)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)


class TraceResourceManagerSimulator:
    def __init__(
        self,
        planner: MILPReconfigPlanner,
        node_to_bundles: dict[str, list[int]],
        initial_config: dict[int, list[int]],
        monitor: SyntheticWorkloadMonitor,
        trigger_interval_s: float,
        trace_start_s: float,
        workload_templates: list[WorkloadClass],
        compare_with_fixed: bool = False,
    ) -> None:
        self.planner = planner
        self.node_to_bundles = node_to_bundles
        self.current_config = copy.deepcopy(initial_config)
        self.monitor = monitor
        self.trigger_interval_s = trigger_interval_s
        self.trace_start_s = trace_start_s
        self.workload_templates = workload_templates
        self.compare_with_fixed = compare_with_fixed
        self.snapshots: list[TriggerSnapshot] = []
        self.workload_history_records: list[dict] = []
        self.sampled_prompt_lens: list[int] | None = None
        self.sampled_response_lens: list[int] | None = None

    def _assign_template(self, index: int) -> WorkloadClass:
        return self.workload_templates[index % len(self.workload_templates)]

    async def run(self, arrival_times: list[float]) -> list[TriggerSnapshot]:
        if not arrival_times:
            self.snapshots.append(TriggerSnapshot(0.0, copy.deepcopy(self.current_config), []))
            return self.snapshots

        # Build events mapping arrival times to prompt/output lengths.
        events: list[TraceRequestEvent] = []
        for index, arrival_time in enumerate(arrival_times):
            if self.sampled_prompt_lens is not None and self.sampled_response_lens is not None:
                p_len = int(self.sampled_prompt_lens[index])
                o_len = int(self.sampled_response_lens[index])
                slo = self._assign_template(index).slo
            else:
                tpl = self._assign_template(index)
                p_len = tpl.prompt_length
                o_len = tpl.output_length
                slo = tpl.slo

            events.append(
                TraceRequestEvent(
                    request_id=f"req_{index}",
                    arrival_time_s=arrival_time,
                    prompt_length=p_len,
                    output_length=o_len,
                    latency_slo=slo,
                )
            )

        event_index = 0
        trigger_time_s = max(self.trigger_interval_s, 0.0)
        final_trigger_s = max(arrival_times) + self.trigger_interval_s

        self.snapshots.append(TriggerSnapshot(0.0, copy.deepcopy(self.current_config), []))

        while trigger_time_s <= final_trigger_s + 1e-9:
            while event_index < len(events) and events[event_index].arrival_time_s <= trigger_time_s:
                self.monitor.record_request_arrival(events[event_index])
                event_index += 1

            workload_classes = self.monitor.get_workload_classes(trigger_time_s)
            if workload_classes:
                new_config = await self.planner.plan_reconfiguration_async(
                    self.node_to_bundles,
                    self.current_config,
                    workload_classes,
                    compare_with_fixed=self.compare_with_fixed,
                )
                self.current_config = new_config

            snapshot = TriggerSnapshot(trigger_time_s, copy.deepcopy(self.current_config), workload_classes)
            self.snapshots.append(snapshot)
            self.workload_history_records.append(
                {
                    "timestamp": f"T+{trigger_time_s:.3f}s",
                    "trigger_time_s": trigger_time_s,
                    "workload_classes": [
                        {
                            "name": wc.name,
                            "prompt_length": wc.prompt_length,
                            "output_length": wc.output_length,
                            "slo": wc.slo,
                            "rps": wc.rps,
                        }
                        for wc in workload_classes
                    ],
                }
            )
            trigger_time_s += self.trigger_interval_s

        return self.snapshots


def write_json(path: str, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def default_latency_profile_paths() -> dict[int, str]:
    base = Path("../latency_profiles/GSAI-ML_LLaDA-8B-Instruct/GH200")
    return {
        1: str(base / "TP1.json"),
        2: str(base / "TP2.json"),
        4: str(base / "TP4.json"),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a BurstGPT trace through the resource-manager planning pipeline.")
    parser.add_argument("--trace-file", required=True, help="BurstGPT-style trace file with arrival times")
    parser.add_argument("--workload-spec", default=None, help="Optional JSON file with workload class templates")
    parser.add_argument("--dataset-path", default="../eval/sharegpt_gpt4_clean_train.jsonl", help="Optional ShareGPT dataset to sample request lengths from")
    parser.add_argument("--tokenizer", default="GSAI-ML/LLaDA-8B-Instruct", help="Tokenizer name or path required when --dataset-path is provided")
    parser.add_argument("--max-seqlen", type=int, default=4096, help="Max sequence length used when sampling dataset items")
    parser.add_argument("--trigger-interval-seconds", type=float, default=300.0, help="How often to trigger reconfiguration")
    parser.add_argument("--monitor-window-seconds", type=float, default=300.0, help="Synthetic workload monitor window")
    parser.add_argument("--prompt-length", type=int, default=1024, help="Default prompt length when no workload spec is provided")
    parser.add_argument("--output-length", type=int, default=256, help="Default output length when no workload spec is provided")
    parser.add_argument("--slo", type=float, default=12.5, help="Default SLO when no workload spec is provided")
    parser.add_argument("--num-nodes", type=int, default=4, help="Number of nodes in the synthetic cluster")
    parser.add_argument("--bundles-per-node", type=int, default=4, help="Bundles available per node")
    parser.add_argument("--output-dir", default=None, help="Directory for simulator outputs")
    parser.add_argument("--compare-with-fixed", action="store_true", help="Also evaluate the fixed-plan branch in the planner")
    parser.add_argument("--cache-prefix", action="store_true", default=True)
    parser.add_argument("--no-cache-prefix", dest="cache_prefix", action="store_false")
    parser.add_argument("--cache-suffix", action="store_true", default=True)
    parser.add_argument("--no-cache-suffix", dest="cache_suffix", action="store_false")
    parser.add_argument("--denoise-block-size", type=int, default=32)
    args = parser.parse_args()

    random.seed(0xCADE)
    np.random.seed(0xCADE)

    arrival_times = load_trace_arrivals(args.trace_file)
    workload_templates = load_workload_templates(args.workload_spec, args.prompt_length, args.output_length, args.slo)

    sampled_prompt_lens = None
    sampled_response_lens = None
    if args.dataset_path:
        if not args.tokenizer:
            raise ValueError("--tokenizer must be provided when --dataset-path is used")
        print("Loading tokenizer for dataset sampling...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        print("Sampling request lengths from dataset...", flush=True)
        _, sampled_prompt_lens, sampled_response_lens = sample_sharegpt_requests_local(
            args.dataset_path, len(arrival_times), tokenizer, args.max_seqlen
        )

    node_to_bundles = build_node_to_bundles(args.num_nodes, args.bundles_per_node)
    initial_config = build_initial_config(node_to_bundles)

    planner = MILPReconfigPlanner(
        latency_profile_paths=default_latency_profile_paths(),
        cache_prefix=args.cache_prefix,
        cache_suffix=args.cache_suffix,
        denoise_block_size=args.denoise_block_size,
    )

    monitor = SyntheticWorkloadMonitor(
        time_window_s=args.monitor_window_seconds,
        start_time_s=arrival_times[0] if arrival_times else 0.0,
    )
    simulator = TraceResourceManagerSimulator(
        planner=planner,
        node_to_bundles=node_to_bundles,
        initial_config=initial_config,
        monitor=monitor,
        trigger_interval_s=args.trigger_interval_seconds,
        trace_start_s=arrival_times[0] if arrival_times else 0.0,
        workload_templates=workload_templates,
        compare_with_fixed=args.compare_with_fixed,
    )

    if sampled_prompt_lens is not None:
        simulator.sampled_prompt_lens = sampled_prompt_lens
        simulator.sampled_response_lens = sampled_response_lens

    snapshots = await simulator.run(arrival_times)

    trace_name = Path(args.trace_file).stem
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{trace_name}"
    out_dir = args.output_dir or os.path.join("reconfig_simulation_outputs", run_name)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    config_history_path = os.path.join(out_dir, "config_history.json")
    workload_history_path = os.path.join(out_dir, "workload_history.json")
    summary_plot_path = os.path.join(out_dir, "reconfig_timeline.png")

    write_json(config_history_path, [asdict(snapshot) for snapshot in snapshots])
    write_json(workload_history_path, simulator.workload_history_records)

    df_summary = summarize_configurations(snapshots)
    plot_reconfig_timeline(df_summary, snapshots, summary_plot_path)

    print(f"Wrote config history: {config_history_path}")
    print(f"Wrote workload history: {workload_history_path}")
    print(f"Wrote plot: {summary_plot_path}")
    print(df_summary)


if __name__ == "__main__":
    asyncio.run(main())