# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
from functools import partial
import aiohttp
import argparse
import asyncio
import json
import os
import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
from tqdm.asyncio import tqdm
import numpy as np

from scipy.stats import zipf
from enum import Enum
from transformers import AutoTokenizer
from typing import List


num_finished_requests = 0
server_num_requests = {}
num_request_in_progress = 0
finished_pbar = None


def get_wait_time(mean_time_between_requests: float, distribution: str, coefficient_variation: float = 0.0) -> float:
    if distribution == "uniform":
        return mean_time_between_requests
    elif distribution == "gamma":
        variance = (coefficient_variation * mean_time_between_requests) ** 2
        shape = mean_time_between_requests ** 2 / variance
        scale = variance / mean_time_between_requests
        return np.random.gamma(shape, scale)
    else:
        return np.random.exponential(mean_time_between_requests)

def request_gen(generator, qps: float, distribution="uniform"):
    while True:
        try:
            item = next(generator)
            yield item
            if distribution != "burst":
                time.sleep(get_wait_time(1.0 / qps, distribution))
        except StopIteration:
            return

async def async_request_gen(generator, qps: float, distribution="uniform", coefficient_variation: float = 0.0):
    while True:
        try:
            item = next(generator)
            yield item
            if distribution != "burst":
                await asyncio.sleep(get_wait_time(1.0 / qps, distribution, coefficient_variation))
        except StopIteration:
            return

def vllm_v1_server_req_func(prompt, output_len):
    request_dict = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        # "temperature": 0.0,
        # "top_p": 0.5,
        # "top_k": 10,
        # "stream": "false",
        # "presence_penalty": 1.1,
        # "repetition_penalty": 1.1,
        "max_tokens": max(output_len, 1),
        "ignore_eos": "true",
    }

    return request_dict

async def inner_query_model_vllm_v1(prompt, verbose, ip_ports, server_req_func):
    prompt, prompt_len, expected_response_len = prompt

    # Evenly dispatch request to the given api servers.
    global server_num_requests
    server_id = min(server_num_requests, key=server_num_requests.get)
    server_num_requests[server_id] += 1
    timeout = aiohttp.ClientTimeout(total=4*60*60)
    global num_finished_requests
    global num_request_in_progress
    global finished_pbar

    async with aiohttp.ClientSession(timeout=timeout) as session:
        request = server_req_func(prompt, expected_response_len)
        # if verbose:
            # print('Querying model')
            # print(f"prompt: {prompt}")
            # print(f"expected_response_len: {expected_response_len}")
        try:
            async with session.post(f'http://{ip_ports[server_id]}/v1/chat/completions', json=request) as resp:
                # if verbose:
                #     print('Done')
                output = await resp.json()
                # necessary for latency calc
                output['response_len'] = expected_response_len
                # if verbose and 'choices' in output:
                #     print(json.dumps(output["choices"][0]["message"]["content"]))
                num_finished_requests += 1
                # print("num_finised_requests: {}".format(num_finished_requests))
                num_request_in_progress -= 1

                if finished_pbar is not None:
                    finished_pbar.update(1)
                return (prompt, output)
        except aiohttp.ClientError as e:
            print(f"Connect to {ip_ports[server_id]} failed with: {str(e)}")
            sys.exit(1)

def load_prompts(prompt_file):
    with open(prompt_file) as f:
        prompts = [json.loads(l) for l in f.readlines()]
    return prompts

def get_tok_id_lens(tokenizer, batch):
    tokenized = tokenizer.batch_encode_plus(batch)
    lens = [len(s) for s in tokenized['input_ids']]
    return lens

def calculate_throughput(queries,
                         dur_s,
                         tokenizer,
                         median_token_latency,
                         median_e2e_latency,
                         median_inference_latency,
                         all_e2e_latencies,
                         all_per_token_latencies,
                         all_inference_latencies,
                         all_request_ids,
                         all_decode_token_latencies,
                         all_request_lens,
                         log_latencies,
                         fail_on_response_failure):
    prompts = []
    responses = []
    naive_hf_lens = []
    ft_lens = []
    expected_response_lens = []
    ray_gen_lens = []
    cf_gen_lens = []
    for prompt, response in queries:
        # print(response)
        assistant_text = None
        if isinstance(response, dict):
            try:
                assistant_text = response["choices"][0]["message"]["content"]
            except:
                pass

        if assistant_text is not None:
            prompts.append(prompt)
            responses.append(assistant_text)
        # if 'generated_text' in response:
        #     prompts.append(prompt)
        #     responses.append(response['generated_text'])
        # if 'naive_hf_lens' in response:
        #     naive_hf_lens.append(response['naive_hf_lens'])
        # if 'ray_gen_len' in response:
        #     ray_gen_lens.append(response['ray_gen_len'])
        # if 'num_output_tokens_cf' in response:
        #     cf_gen_lens.append(response['num_output_tokens_cf'])
        # if 'response_len' in response:
        #     expected_response_lens.append(response['response_len'])
    # prompt_ids = [p for p in tokenizer.batch_encode_plus(prompts)['input_ids']]
    # response_ids = [r for r in tokenizer.batch_encode_plus(responses)['input_ids']]

    # print(f'check_len actual {list(sorted(len(response) for response in response_ids))}')
    # print(f'check_len expect {list(sorted(expected_response_lens))}')
    # print(f'self-reported {list(sorted(cf_gen_lens))}')
    # for prompt, response, expected_response_len in zip(prompt_ids, response_ids, expected_response_lens):
    #     print(f'check lens {len(prompt)=} {len(response)=} {expected_response_len=}')

    try:
        prompt_lens = get_tok_id_lens(tokenizer, prompts)
        response_lens = get_tok_id_lens(tokenizer, responses)
    except Exception:
        print(prompts)
        print(responses)
        raise

    # if naive_hf_lens:
    #     print(f'naive_hf_lens {list(sorted(naive_hf_lens))}')
    print(f'prompt_lens {list(sorted(prompt_lens))}')
    print(f'response_lens {list(sorted(response_lens))}')
    print(f'expected_response_lens {list(sorted(expected_response_lens))}')
    if ray_gen_lens:
        print(f'ray_gen_lens {list(sorted(ray_gen_lens))}')

    prompt_token_count = sum(prompt_lens)
    response_token_count = sum(response_lens)

    all_prompt_lens = prompt_lens
    all_response_lens = response_lens
    all_total_tokens = [all_prompt_lens[i] + all_response_lens[i] for i in range(len(all_prompt_lens))]

    if len(all_inference_latencies) == len(all_e2e_latencies):
        all_waiting_latencies = [all_e2e_latencies[i] - all_inference_latencies[i] for i in range(len(all_e2e_latencies))]
    else:
        all_waiting_latencies = []

    # if naive_hf_lens:
    #     # Manually count naive hf tok len
    #     total_resp_tokens = sum(
    #         [response_len for _, response_len in naive_hf_lens])
    #     total_prompt_tokens = sum(
    #         [prompt_len for prompt_len, _ in naive_hf_lens])
    #     response_token_count = total_prompt_tokens + total_resp_tokens
    # if ray_gen_lens:
    #     response_token_count = sum(ray_gen_lens)
    # if cf_gen_lens:
    #     response_token_count = sum(cf_gen_lens)

    # print(f'prompt_token_count {prompt_token_count} response_token_count {response_token_count}')
    throughput_tok_s = (prompt_token_count + response_token_count) / dur_s
    print(f'throughput_tok_s {throughput_tok_s:.02f}')
    qps = len(responses) / dur_s
    msg1 = f'dur_s {dur_s:.04f} tokens_per_s {throughput_tok_s:.02f} qps {qps:.04f}\n'
    msg2 = f'successful_responses {len(responses)} prompt_token_count {prompt_token_count} response_token_count {response_token_count}\n'
    msg3 = f'{median_token_latency=:.04f}, {median_e2e_latency=:.04f}, {median_inference_latency=:.04f}\n'
    msg = msg1 + msg2 + msg3
    if log_latencies:
        msg += f'{all_request_lens=}\n{all_request_ids=}\n'
        msg += f'{all_total_tokens=}\n{all_prompt_lens=}\n{all_response_lens=}\n'
        msg += f'{all_e2e_latencies=}\n{all_per_token_latencies=}\n{all_inference_latencies=}\n{all_waiting_latencies=}\n{all_decode_token_latencies=}\n'
    print(msg)

    if fail_on_response_failure:
        assert len(responses) == len(queries), \
            f"{fail_on_response_failure=}, expected number of successful respones to equal number of queries, got {len(responses)} vs {len(queries)}"

    return throughput_tok_s, prompts, responses

def calculate_cdf(latencies):
    hist, bin_edges = np.histogram(latencies, bins=50)
    cumsum = np.cumsum(hist)
    print("Latency: ")
    print(f"{bin_edges=}")
    print(f"{hist=}")
    print(f"{cumsum=}")

def plot_latency_cdf(req_latencies, prefill_latencies, decode_latencies, log_filename):
    fig_filename = os.path.splitext(log_filename)[0] + "_latency.png"
    fig, (ax_req, ax_prefill, ax_decode) = plt.subplots(1, 3, figsize=(3*7, 4.8))

    def plot_single(ax, latencies, is_prefill=False):
        hist, bin_edges = np.histogram(latencies, bins=50)
        cumsum = np.cumsum(hist)
        p50 = np.percentile(latencies, 50)
        p80 = np.percentile(latencies, 80)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        p999 = np.percentile(latencies, 99.9)
        ax.plot(bin_edges[1:], cumsum/np.sum(hist)*100, color='red')
        ax.axvline(p50, color='blue', linestyle='--', label='P50')
        ax.text(p50, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p50:.2f}", va='bottom', ha='right', color='blue')
        ax.axvline(p80, color='green', linestyle='--', label='P80')
        ax.text(p80, ax.get_ylim()[0] + 0.10 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p80:.2f}", va='bottom', ha='right', color='green')
        ax.axvline(p95, color='orange', linestyle='--', label='P95')
        ax.text(p95, ax.get_ylim()[0] + 0.15 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p95:.2f}", va='bottom', ha='right', color='orange')
        ax.axvline(p99, color='purple', linestyle='--', label='P99')
        ax.text(p99, ax.get_ylim()[0] + 0.20 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p99:.2f}", va='bottom', ha='right', color='purple')
        ax.axvline(p999, color='gray', linestyle='--', label='P99.9')
        ax.text(p999, ax.get_ylim()[0] + 0.25 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p999:.2f}", va='bottom', ha='right', color='gray')
        mean = np.mean(latencies)
        mean_value = bin_edges[:-1][np.where(bin_edges[:-1] <= mean)][-1]
        mean_percentage = cumsum[np.where(bin_edges[:-1] <= mean)][-1] / np.sum(hist) * 100
        ax.axvline(mean_value, color='black', linestyle='-', label='mean={:.2f}'.format(mean))
        ax.text(mean_value, mean_percentage, f"{mean_percentage:.2f}", va='bottom', ha='right', color='black')
        ax.legend(loc='upper right')
        ax.set_ylabel('Cumulative Percentage(%)')

    plot_single(ax_req, req_latencies)
    # plot_single(ax_prefill, prefill_latencies, is_prefill=True)
    # plot_single(ax_decode, decode_latencies)
    ax_req.set_xlabel('Latency/req(s)')
    ax_req.set_title('request cdf')
    ax_prefill.set_xlabel('Latency/token(ms)')
    ax_prefill.set_title('prefill cdf')
    ax_decode.set_xlabel('Latency/token(ms)')
    ax_decode.set_title('decode cdf')
    index1 = fig_filename.rfind('/')
    index2 = fig_filename.rfind('/', 0, index1)
    fig_filename_title = fig_filename[index2 + 1:]
    plt.suptitle(fig_filename_title, fontsize=6)
    fig.savefig(fig_filename)

def plot_len_cdf(prompt_lens, response_lens, total_tokens, log_filename):
    fig_filename = os.path.splitext(log_filename)[0] + "_len.png"
    fig, (ax_prompt, ax_response, ax_total) = plt.subplots(1, 3, figsize=(3*7, 4.8))

    def plot_single(ax, lens, x_label_str, title_str):
        hist, bin_edges = np.histogram(lens, bins=50)
        cumsum = np.cumsum(hist)
        p50 = np.percentile(lens, 50)
        p80 = np.percentile(lens, 80)
        p95 = np.percentile(lens, 95)
        p99 = np.percentile(lens, 99)
        ax.plot(bin_edges[1:], cumsum/np.sum(hist)*100, color='red')
        ax.axvline(p50, color='blue', linestyle='--', label='P50')
        ax.text(p50, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p50:.2f}", va='bottom', ha='right', color='blue')
        ax.axvline(p80, color='green', linestyle='--', label='P80')
        ax.text(p80, ax.get_ylim()[0] + 0.10 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p80:.2f}", va='bottom', ha='right', color='green')
        ax.axvline(p95, color='orange', linestyle='--', label='P95')
        ax.text(p95, ax.get_ylim()[0] + 0.15 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p95:.2f}", va='bottom', ha='right', color='orange')
        ax.axvline(p99, color='purple', linestyle='--', label='P99')
        ax.text(p99, ax.get_ylim()[0] + 0.20 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p99:.2f}", va='bottom', ha='right', color='purple')
        mean = np.mean(lens)
        mean_value = bin_edges[:-1][np.where(bin_edges[:-1] <= mean)][-1]
        mean_percentage = cumsum[np.where(bin_edges[:-1] <= mean)][-1] / np.sum(hist) * 100
        ax.axvline(mean_value, color='black', linestyle='-', label='mean={:.2f}'.format(mean))
        ax.text(mean_value, mean_percentage, f"{mean_percentage:.2f}", va='bottom', ha='right', color='black')
        ax.legend(loc='upper right')
        ax.set_xlabel(x_label_str)
        ax.set_ylabel('Cumulative Percentage(%)')
        ax.set_title(title_str)

    plot_single(ax_prompt, prompt_lens, 'prompt len', 'prompt len cdf')
    plot_single(ax_response, response_lens, 'response len', 'response len cdf')
    plot_single(ax_total, total_tokens, 'total token', 'total token cdf')
    index1 = fig_filename.rfind('/')
    index2 = fig_filename.rfind('/', 0, index1)
    fig_filename_title = fig_filename[index2 + 1:]
    plt.suptitle(fig_filename_title, fontsize=6)
    fig.savefig(fig_filename)

def plot_instance(log_filename_0):
    current_dir = os.path.dirname(os.path.abspath(log_filename_0))
    log_files = glob.glob(os.path.join(current_dir, '*.log_instance.csv'))
    log_files.sort(key=os.path.getmtime, reverse=True)
    df_0 = pd.read_csv(log_files[0]).sort_values(by=["timestamp"])
    timestamp_list_0 = df_0["timestamp"].to_numpy()
    num_instances_list_0 = df_0["num_instances"].to_numpy()
    time_0 = 0
    sum_0 = 0
    for idx, t in enumerate(timestamp_list_0):
        if t > time_0:
            time_0 += 1
            sum_0 += num_instances_list_0[idx]
    print(f"{sum_0/time_0} gpu/s")
    avg_instance_num = np.round(sum_0/time_0, 2)

    fig, ax = plt.subplots()
    ax.plot(timestamp_list_0, num_instances_list_0, color="red", label=f"instance_num(avg {avg_instance_num} /s)")
    ax.legend(loc='upper left')
    fig_filename = os.path.splitext(log_filename_0)[0] + "_instance.png"
    index1 = fig_filename.rfind('/')
    index2 = fig_filename.rfind('/', 0, index1)
    fig_filename_title = fig_filename[index2 + 1:]
    plt.suptitle(fig_filename_title, fontsize=6)
    fig.savefig(fig_filename)

    return avg_instance_num

def save_all_decode_token_latencies_npy(all_token_latencies: List[np.ndarray], log_filename):
    dtype = [('timestamp',float),('latency',float)]
    all_lat_pairs = []
    for arr in all_token_latencies:
        # use decode latencies
        for pair in arr[1:]:
            all_lat_pairs.append((pair[0],pair[1]))
    all_lat_pairs = np.array(all_lat_pairs,dtype=dtype)
    all_lat_pairs = np.sort(all_lat_pairs,order='timestamp')
    np.save(os.path.splitext(log_filename)[0], all_lat_pairs)

class MeasureLatency:
    def __init__(self):
        self._request_ids = []
        self._request_lens = []
        self._request_latencies = []
        self._per_token_latencies = []
        self._decode_token_latencies = []
        self._prefill_token_latencies = []
        self._all_token_latencies = []
        self._decode_sum_latencies = []
        self._all_decode_token_latencies = []
        self._inference_latencies = []
        self._per_token_latency_breakdown_list = []

    def measure(self, f):
        async def measured(*args, **kwargs):
            start = time.time()
            prompt, output = await f(*args, **kwargs)
            # Do not record latency if request failed.
            latency = time.time() - start
            self._request_latencies.append(latency)
            try:
                self._per_token_latencies.append(
                    latency / output['response_len'])
            except ZeroDivisionError:
                # Not currently using this metric..
                pass
            if 'request_id' in output:
                self._request_ids.append(output['request_id'])
            if 'per_token_latency' in output:
                lat_arr = np.array(output['per_token_latency'])
                mean_decode_token_latency = 0 if len(lat_arr) == 1 else np.mean(lat_arr[1:,1])
                decode_sum_latency = 0 if len(lat_arr) == 1 else np.sum(lat_arr[1:,1])
                self._decode_token_latencies.append(mean_decode_token_latency)
                self._request_lens.append(len(lat_arr[1:,1]))
                self._prefill_token_latencies.append(lat_arr[0][1])
                self._all_token_latencies.append(lat_arr)
                self._decode_sum_latencies.append(decode_sum_latency)
                self._all_decode_token_latencies.extend(lat_arr[1:,1])
                self._inference_latencies.append(0.0)
            if 'per_token_latency_breakdown_list' in output:
                self._per_token_latency_breakdown_list.append(output['per_token_latency_breakdown_list'])
            return prompt, output
        return measured

def get_token_ids(input_str, tokenizer):
    t = tokenizer(input_str)
    return t['input_ids']


async def benchmark(
    tokenizer,
    prompts: List[str],
    arrival_times: List[float],
    verbose: bool,
    log_filename: str,
    ip_ports: List[int],
    log_latencies: bool,
    fail_on_response_failure: bool,
    limit: int
):
    # either limit or arrival_times should be set
    assert (limit is None) != (arrival_times is None)

    query_model = partial(inner_query_model_vllm_v1, server_req_func=vllm_v1_server_req_func)

    global server_num_requests
    num_servers = len(ip_ports)
    for server_id in range(num_servers):
        server_num_requests[server_id] = 0

    m = MeasureLatency()
    query_model = m.measure(query_model)

    print(f"Starting with num_prompts={len(prompts)}")

    total_requests = len(prompts)

    tasks = []
    pbar = tqdm(total=len(prompts), desc="Dispatching requests")
    global finished_pbar
    finished_pbar = tqdm(total=len(prompts), desc="Finished requests")

    start_wall = time.time()

    if arrival_times:
        assert len(prompts) == len(arrival_times), "Arrival times length must match prompts"
        for i, prompt in enumerate(prompts):
            scheduled_arrival = arrival_times[i]

            # compute real time delta from benchmark start
            now = time.time()
            wait = scheduled_arrival - (now - start_wall)

            if wait > 0:
                await asyncio.sleep(wait)

            # Launch the request
            tasks.append(asyncio.create_task(
                query_model(prompt, verbose, ip_ports)
            ))

            pbar.update(1)
    elif limit:
        global num_request_in_progress
        # at most `limit` concurrent requests
        for i in range(len(prompts)):
            prompt = prompts[i]

            while True:
                if num_request_in_progress < limit:
                    # Launch the request
                    tasks.append(asyncio.create_task(
                        query_model(prompt, verbose, ip_ports)
                    ))
                    num_request_in_progress += 1
                    pbar.update(1)
                    break
                else:
                    await asyncio.sleep(0.1)  # wait a bit before checking again

    pbar.close()

    # ============================================================
    # Wait for all requests to complete
    # ============================================================
    queries = await asyncio.gather(*tasks)
    finished_pbar.close()

    dur_s = time.time() - start_wall
    median_token_latency = np.median(m._per_token_latencies)
    median_e2e_latency = np.median(m._request_latencies)
    median_inference_latency = np.median(m._inference_latencies)

    throughput, prompts, responses = calculate_throughput(queries,
                                      dur_s,
                                      tokenizer,
                                      median_token_latency,
                                      median_e2e_latency,
                                      median_inference_latency,
                                      m._request_latencies,
                                      m._per_token_latencies,
                                      m._inference_latencies,
                                      m._request_ids,
                                      m._decode_token_latencies,
                                      m._request_lens,
                                      log_latencies,
                                      fail_on_response_failure)
    calculate_cdf(m._request_latencies)
    plot_latency_cdf(m._request_latencies, m._prefill_token_latencies, m._decode_token_latencies, log_filename)
    save_all_decode_token_latencies_npy(m._all_token_latencies, log_filename)
    # avg_instance_num = plot_instance(log_filename)
    avg_instance_num = 0.0

    return throughput, \
           m._prefill_token_latencies, \
           m._decode_token_latencies, \
           m._inference_latencies, \
           avg_instance_num, \
           m._request_latencies, \
           m._request_ids, \
           m._decode_sum_latencies, \
           m._request_lens, \
           m._all_decode_token_latencies, \
           m._per_token_latency_breakdown_list, \
            prompts, \
            responses

def gen_random_response_lens(distribution: str, len_mean, len_range, num_prompts):
    if distribution == 'uniform':
        if len_range == 0:
            return [len_mean for _ in range(num_prompts)]

        low = len_mean - (len_range // 2)
        high = len_mean + (len_range // 2)
        response_lens = list(
            map(lambda _: random.randint(low, high), range(num_prompts)))
    elif distribution == 'exponential':
        response_lens = [min(round(s), len_range) for s in np.random.exponential(scale=len_mean, size=num_prompts)]
    elif distribution == 'capped_exponential':
        response_lens = []
        while len(response_lens) < num_prompts:
            sample = round(np.random.exponential(scale=len_mean))
            if sample <= len_range and sample >= 1:
                response_lens.append(sample)
    elif distribution == 'zipf':
        rank = np.arange(1, len_mean*2)
        if len_mean == 1024 and len_range == 6144:
            alpha = 1.0005
        elif len_mean == 512 and len_range == 6144:
            alpha = 1.15
        elif len_mean == 256 and len_range == 6144:
            alpha = 1.5
        elif len_mean == 128 and len_range == 6144:
            alpha = 2.0
        else:
            alpha = 1.0
        probabilities = zipf.pmf(rank, alpha)
        probabilities /= np.sum(probabilities)
        response_lens = np.random.choice(np.arange(1, len_mean*2), size=num_prompts, p=probabilities)
    else:
        raise ValueError(f'unknown distribution {distribution=}')

    scaling_factor = len_mean / np.mean(response_lens)
    response_lens = np.ceil(np.array(response_lens) * scaling_factor).astype(int)
    if distribution == 'zipf':
        response_lens = [response_len if response_len <= len_range else len_range for response_len in response_lens]
    elif distribution == 'uniform':
        capped_response_lens = []
        for response_len in response_lens:
            if response_len < low:
                capped_response_lens.append(low)
            elif response_len > high:
                capped_response_lens.append(high)
            else:
                capped_response_lens.append(response_len)
        response_lens = capped_response_lens
    else:
        response_lens = [response_len if response_len <= len_range else len_range for response_len in response_lens]
    response_lens = [int(x) for x in response_lens]

    return response_lens

def fast_sample_sharegpt(dataset_path, num_requests, tokenizer, max_seqlen):
    print(f"Fast loading ShareGPT dataset from {dataset_path}", flush=True)
    prompts, responses = [], []

    with open(dataset_path) as f:
        for line in f:
            data = json.loads(line)
            if len(data["conversations"]) >= 2:
                prompts.append(data["conversations"][0]["value"])
                responses.append(data["conversations"][1]["value"])

    # batch tokenize instead of one-by-one
    prompt_ids = tokenizer(
        prompts, padding=False, truncation=False, add_special_tokens=False
    )["input_ids"]

    response_ids = tokenizer(
        responses, padding=False, truncation=False, add_special_tokens=False
    )["input_ids"]

    sampled_prompts = []
    sampled_prompt_lens = []
    sampled_response_lens = []

    for p, r in zip(prompt_ids, response_ids):
        rounded_r = ((len(r) + 31) // 32) * 32
        if len(p) > 0 and rounded_r > 0 and len(p) + rounded_r < max_seqlen:
            sampled_prompts.append(p)
            sampled_prompt_lens.append(len(p))
            sampled_response_lens.append(rounded_r)
            if len(sampled_prompts) >= num_requests:
                break
            
    print("finish loading dataset", flush=True)
    return sampled_prompts, sampled_prompt_lens, sampled_response_lens


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer,
    max_seqlen:int,
):
    print(f"Loading ShareGPT dataset from {dataset_path}", flush=True)
    # Load the dataset.
    prompts = []
    prompt_lens = []
    response_lens = []
    with open(dataset_path) as f:
        for line in f:
            data = json.loads(line)
            if len(data["conversations"]) >= 2:
                prompt = data["conversations"][0]["value"]
                res = data["conversations"][1]["value"]
                # m = [{"role": "user", "content": prompt}]
                # prompt = tokenizer.apply_chat_template(
                #     m,
                #     add_generation_prompt=True,
                #     tokenize=False
                # )
                prompt_token_ids = tokenizer(prompt).input_ids
                completion_token_ids = tokenizer(res).input_ids
                rounded_up_completion_len = ((len(completion_token_ids) + 31) // 32) * 32
                if len(prompt_token_ids) + rounded_up_completion_len < max_seqlen and \
                    len(prompt_token_ids) > 0 and rounded_up_completion_len > 0:
                    prompts.append(prompt)
                    prompt_lens.append(len(prompt_token_ids))
                    response_lens.append(rounded_up_completion_len)
            # if len(prompts)>num_requests:
            #     break
    sampled_ids = [random.randint(0, len(prompts) - 1) for _ in range(num_requests)]
    sampled_prompts = [prompts[idx] for idx in sampled_ids]
    sampled_prompt_lens = [prompt_lens[idx] for idx in sampled_ids]
    sampled_response_lens = [response_lens[idx] for idx in sampled_ids]
    # print(f"max len:{max(a+b for a,b in zip(prompt_lens, response_lens))}")
    print("finish loading dataset", flush=True)
    return sampled_prompts, sampled_prompt_lens, sampled_response_lens

import httpx
def wait_until_up(url: str, 
                  timeout: float = 600, 
                  interval: float = 1.0):
    start = time.monotonic()
    while True:
        try:
            r = httpx.get(url, timeout=1.0)
            if r.status_code in (200, 404, 405):  # got *some* valid HTTP response
                print(f"Server at {url} is ready.")
                return
            print(f"Server at {url} returned status {r.status_code}, retrying...")
        except Exception as e:
            print(f"Server at {url} not ready yet: {e}")
            pass  # connection refused, keep trying

        if time.monotonic() - start > timeout:
            raise TimeoutError(f"Server at {url} not ready after {timeout:.1f} seconds")
        time.sleep(interval)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument('--trust_remote_code',
                    action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--log_filename', type=str, default='benchmark.log')
    parser.add_argument('--ip_ports', nargs='+', required=True, help='List of ip:port')
    parser.add_argument('--max_request_len', type=int, default=8192)

    parser.add_argument('--log_latencies', action="store_true",
                        help="Whether or not to write all latencies to the log file.")
    parser.add_argument('--fail_on_response_failure', action="store_true",
                        help="Whether or not to fail the benchmarking script if any request fails")

    parser.add_argument('--dataset_path', type=str)

    parser.add_argument('--print_generation_lens_and_exit',
                        action='store_true')

    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of concurrent requests. Either limit or arrival_time_file should be set.")
    parser.add_argument("--arrival_time_file", type=str,
                        help="Path to arrival time file")
    parser.add_argument('--num_requests', type=int, default=500,
                        help='Number of requests to send. Used only if --limit is set.')

    args = parser.parse_args()

    print("start loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    print("finished loading tokenizer")
    # set seed

    # read arrival time file
    if args.arrival_time_file != "":
        print(f"start loading file")
        arrival_times = np.loadtxt(args.arrival_time_file).tolist()
        print(f"loading finished")
        num_prompts = len(arrival_times)
    else:
        arrival_times = None
        num_prompts = args.num_requests
        print(f"using limit {args.limit}, num_prompts {num_prompts}")
        
    # flush
    print("Flushing stdout", flush=True)
        
    random.seed(0xCADE)
    np.random.seed(0xCADE)
    prompts, prompt_lens, response_lens= sample_sharegpt_requests(args.dataset_path, 
                                                                  num_prompts ,tokenizer, args.max_request_len)
    # prompts, prompt_lens, response_lens= fast_sample_sharegpt(args.dataset_path, 
    #                                                         num_prompts ,tokenizer, args.max_request_len)

    for i, (prompt_len, gen_len) in enumerate(zip(prompt_lens, response_lens)):
        total = prompt_len + gen_len
        if total > args.max_request_len:
            print(f'truncating long prompt+gen_len {prompt_len=} {gen_len=}')
            gen_len = args.max_request_len - prompt_len
        response_lens[i] = gen_len

    if args.print_generation_lens_and_exit:
        print(f'{prompt_lens=}')
        print(f'{response_lens=}')
        print('Exiting...')
        return

    if args.verbose or True:
        # print('prompt lens', sorted(list(prompt_lens)))
        # print('response lens', sorted(list(response_lens)))
        total_tokens = []
        for i, (prompt_len, gen_len) in enumerate(zip(prompt_lens, response_lens)):
            total_tokens.append(prompt_len + gen_len)
        # print('total tokens', sorted(list(total_tokens)))

    plot_len_cdf(prompt_lens, response_lens, total_tokens, args.log_filename)

    # print stats (min, p25, p50, p75, p90, p99, max, mean)
    print(f"Prompt lengths: min {np.min(prompt_lens)}, p25 {np.percentile(prompt_lens, 25)}, p50 {np.percentile(prompt_lens, 50)}, p75 {np.percentile(prompt_lens, 75)}, p90 {np.percentile(prompt_lens, 90)}, p99 {np.percentile(prompt_lens, 99)}, max {np.max(prompt_lens)}, mean {np.mean(prompt_lens):.2f}")
    print(f"Response lengths: min {np.min(response_lens)}, p25 {np.percentile(response_lens, 25)}, p50 {np.percentile(response_lens, 50)}, p75 {np.percentile(response_lens, 75)}, p90 {np.percentile(response_lens, 90)}, p99 {np.percentile(response_lens, 99)}, max {np.max(response_lens)}, mean {np.mean(response_lens):.2f}")
    # print('Prompt lengths: min {}, max {}, mean {:.2f}, median {:.2f}'.format(
    #     min(prompt_lens), max(prompt_lens), np.mean(prompt_lens), np.median(prompt_lens)))
    # print('Response lengths: min {}, max {}, mean {:.2f}, median {:.2f}'.format(
    #     min(response_lens), max(response_lens), np.mean(response_lens), np.median(response_lens)))

    prompts = list(zip(prompts, prompt_lens, response_lens))

    for ip_port in args.ip_ports:
        wait_until_up(f'http://{ip_port}/v1/models')

    throughput, \
    prefill_token_latencies, \
    decode_token_latencies, \
    inference_latencies, \
    avg_instance_num, \
    request_latencies, \
    request_ids, \
    decode_sum_latencies, \
    request_lens, \
    all_decode_token_latencies, \
    per_token_latency_breakdown_list, \
    prompts, \
    responses = asyncio.run(benchmark(
        tokenizer,
        prompts,
        arrival_times,
        args.verbose,
        args.log_filename,
        args.ip_ports,
        args.log_latencies,
        args.fail_on_response_failure,
        args.limit
    ))
    
    if throughput is None:
        return

    instances = {
        str(i): {
            "prompt_len": prompt_lens[i],
            "expected_response_len": response_lens[i],
            "request_latency": request_latencies[i],
            "prompt": prompts[i],
            "response": responses[i],
        }
        for i in range(len(prompts))
    }

    file_name = os.path.splitext(args.log_filename)[0] + "_latency_info.json"
    results = []
    import datetime
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    file_name = os.path.splitext(args.log_filename)[0] + "_latency_info.json"
    try:
        with open(file_name, 'r') as f:
            results = json.load(f)
    except json.decoder.JSONDecodeError:
        pass
    except FileNotFoundError:
        os.mknod(file_name)
    with open(file_name, 'w') as f:
        results.append({
                        "request_ids": request_ids,
                        "request_lens": request_lens,
                        "throughput": throughput, 
                        "instance_num": avg_instance_num,
                        "instances": instances,
                        })
        json.dump(results, f,
                  indent=2)


if __name__ == '__main__':
    main()