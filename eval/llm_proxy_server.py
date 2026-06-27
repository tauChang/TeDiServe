import asyncio
import json
import httpx
import logging
import random
import socket
import time
import threading
import uvicorn
import atexit
import os
import hashlib
import numpy as np

from fastapi import FastAPI, Request
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
)

# logger.setLevel(logging.INFO)
app = FastAPI()

import random

_VOLATILE_REQUEST_FIELDS = {"request_id"}


def _normalize_for_hash(obj):
    if isinstance(obj, dict):
        return {k: _normalize_for_hash(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [_normalize_for_hash(v) for v in obj]
    return obj


def make_stable_request_id(payload: dict) -> str:
    base_payload = {k: v for k, v in payload.items() if k not in _VOLATILE_REQUEST_FIELDS}
    canonical = json.dumps(
        _normalize_for_hash(base_payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return "req_" + hashlib.blake2b(canonical.encode("utf-8"), digest_size=12).hexdigest()

def generate_request_arrival_times(arrival_pattern, default_cv=1.0):
    """Generate arrival times. Each tuple may be:
       (num_requests, mean_interarrival)
       (num_requests, mean_interarrival, cv)
       If cv is omitted, Poisson arrivals (cv=1) are used.
    """
    random.seed(42)
    arrival_times = []
    current_time = 0.0

    for item in arrival_pattern:
        # Parse tuple (allow 2 or 3 elements)
        if len(item) == 2:
            num_requests, mean_interarrival = item
            cv = default_cv
        elif len(item) == 3:
            num_requests, mean_interarrival, cv = item
        else:
            raise ValueError("Each pattern element must be 2 or 3 values")
        
        if cv < 0:
            raise ValueError("CV must be >= 0")
        
        if cv == 0:
            # uniform
            for _ in range(num_requests):
                inter_arrival = mean_interarrival
                current_time += inter_arrival
                arrival_times.append(current_time)
        else:
            # Gamma parameters
            k = 1.0 / (cv * cv)           # shape
            theta = mean_interarrival / k # scale

            for _ in range(num_requests):
                if mean_interarrival > 0:
                    inter_arrival = random.gammavariate(k, theta)
                else:
                    inter_arrival = 0

                current_time += inter_arrival
                arrival_times.append(current_time)
    
    t = np.array(arrival_times)
    window = 1.0  # 2 second window
    rates = [
        np.sum((t >= ti - window) & (t <= ti)) / window
        for ti in t
    ]
    logger.info(f"max RPS: {max(rates):.2f}, avg RPS: {len(t)/t[-1]:.2f}")


    return arrival_times


def wait_until_up(url: str, 
                  timeout: float = 600, 
                  interval: float = 1.0):
    start = time.monotonic()
    while True:
        try:
            r = httpx.get(url, timeout=1.0)
            if r.status_code in (200, 404, 405):  # got *some* valid HTTP response
                logger.info(f"Server at {url} is ready.")
                return
            logger.info(f"Server at {url} returned status {r.status_code}, retrying...")
        except Exception as e:
            logger.info(f"Server at {url} not ready yet: {e}")
            pass  # connection refused, keep trying

        if time.monotonic() - start > timeout:
            raise TimeoutError(f"Server at {url} not ready after {timeout:.1f} seconds")
        time.sleep(interval)
        
async def wait_upstream_ready(url, timeout=30000, interval=2.0):
    start = time.monotonic()
    async with httpx.AsyncClient() as client:
        while True:
            try:
                r = await client.get(url, timeout=2.0)
                if r.status_code in (200, 404, 405):
                    return
            except Exception:
                logger.info(f"Upstream at {url} not ready yet, retrying...")
                pass

            if time.monotonic() - start > timeout:
                raise RuntimeError(f"Upstream not ready after {timeout}s")

            await asyncio.sleep(interval)

            
@app.post("/v1/completions")
async def proxy_completions(request: Request):
    data = await request.json()
    # logger.info(f"Received request data: {data}")

    # request_id from caller is arrival-order index used for schedule lookup.
    arrival_req_id = int(data["request_id"])
    stable_req_id = make_stable_request_id(data)

    # ============================================================
    # 1. Warmup using request 0's payload
    # ============================================================
    if arrival_req_id == 0:
        await wait_upstream_ready(app.state.upstream_url)
        # if app.state.does_warmup:
        #     logger.info("⚠️ Warmup triggered by request 0 — sending warmup call.")
            
        #     warmup_data = data.copy()   # SAME payload as normal request 0

        #     # Send warmup immediately, no scheduling, non-blocking
        #     async with httpx.AsyncClient(timeout=200) as client:
        #         try:
        #             _ = await client.post(
        #                 f"{app.state.upstream_url}/completions",
        #                 json=warmup_data,
        #                 headers={"X-Request-Id": "warmup"}
        #             )
        #             logger.info("Warmup request succeeded.")
        #         except Exception as e:
        #             logger.warning(f"Warmup request failed: {e}")

        # await asyncio.sleep(1.0)
        # app.state.first_request_arrival_time = time.monotonic()
        # app.state.warmup_done = True
        # logger.info("Warmup complete. Proceeding with normal handling.")
        if app.state.does_warmup:
            logger.info("⚠️ Warmup triggered — sending 30 warmup calls.")

            warmup_data = data.copy()

            async def send_one(i):
                async with httpx.AsyncClient(timeout=200) as client:
                    try:
                        await client.post(
                            f"{app.state.upstream_url}/completions",
                            json=warmup_data,
                            headers={"X-Request-Id": f"warmup-{i}"},
                        )
                        logger.info(f"Warmup request {i} succeeded.")
                    except Exception as e:
                        logger.warning(f"Warmup request {i} failed: {e}")

            # Launch 30 warmup requests concurrently
            # tasks = [asyncio.create_task(send_one(i)) for i in range(20)]
            tasks = [asyncio.create_task(send_one(i)) for i in range(20)]
            await asyncio.gather(*tasks)

            logger.info("All warmup requests completed.")

        await asyncio.sleep(3.0)
        app.state.first_request_arrival_time = time.monotonic()
        app.state.warmup_done = True
        logger.info("Warmup complete. Proceeding with normal handling.")


    # ============================================================
    # 2. All requests must wait until warmup finishes
    # ============================================================
    while not app.state.warmup_done:
        await asyncio.sleep(0.1)

    # ============================================================
    # 3. Normal request path (request 0 included)
    # ============================================================
    # Apply chat template if enabled
    if app.state.apply_chat_template:
        prompt = data["prompt"]
        m = [{"role": "user", "content": prompt}]
        data["prompt"] = app.state.tokenizer.apply_chat_template(
            m,
            add_generation_prompt=True,
            tokenize=False
        )

    # Compute scheduled release delay
    delay = max(
        0.0,
        app.state.first_request_arrival_time
        + app.state.request_arrival_time[arrival_req_id]
        - time.monotonic()
    )

    # logger.info(
    #     f"Request {req_id}: scheduled after {delay:.3f}s "
    #     f"(gap={app.state.request_arrival_time[req_id]:.3f}s)"
    # )

    await asyncio.sleep(delay)
    # logger.info(f"Request {req_id} released after waiting {delay:.3f}s")
    # logger.info(f"Prompt (first 200 chars): {data['prompt'][:200]}")

    # Forward real request to upstream
    theoretical_release_time = app.state.first_request_arrival_time + app.state.request_arrival_time[arrival_req_id]
    actual_release_time = time.monotonic()
    # combined_req_id = f"{arrival_req_id}-{stable_req_id}"
    combined_req_id = stable_req_id
    async with httpx.AsyncClient(timeout=1200) as client:
        resp = await client.post(
            f"{app.state.upstream_url}/completions",
            json=data,
            headers={"X-Request-Id": combined_req_id},
        )
        resp.raise_for_status()

    elapsed = time.monotonic() - actual_release_time # time since when request was supposed to be released
    app.state.resp_time[arrival_req_id] = elapsed
    app.state.stable_req_id_by_arrival[arrival_req_id] = stable_req_id
    app.state.resp_time_by_stable_req_id.setdefault(stable_req_id, []).append(elapsed)

    resp_json = resp.json()
    output_text = ""
    if "choices" in resp_json and resp_json["choices"]:
        output_text = resp_json["choices"][0].get("text", "")

    # count num output tokens
    num_tokenized = 0
    if app.state.tokenizer is not None:
        tokenized = app.state.tokenizer(
            output_text,
            return_tensors="pt",
            truncation=False,
        )
        num_tokenized = tokenized.input_ids.shape[1]
        
    logger.info(f"num_output_tokens: {num_tokenized}")
    app.state.token_count += num_tokenized

    logger.info(f"tput so far: {app.state.token_count / (time.monotonic() - app.state.first_request_arrival_time):.2f} tokens/s")
    # post again
    resp_json["_proxy_meta"] = {
        "arrival_request_id": arrival_req_id,
        "stable_request_id": stable_req_id,
    }
    # logger.info(f"Request {req_id} got response: {resp_json}")
    # logger.info(f"Time since first request arrival: {time.monotonic() - app.state.first_request_arrival_time:.3f}s")

    return resp_json


def write_response_times(file_path: str):
    # if write-results is true, file_path should already exist. We read it, append the resp_time dict, and write it back
    # else, we don nothing

    if not os.path.exists(file_path):
        logger.warning(f"Output path {file_path} does not exist, not writing response times.")
        return

    with open(file_path, "r") as f:
        data = json.load(f)
    
    data["response_times"] = app.state.resp_time
    data["average_response_time"] = sum(app.state.resp_time.values()) / len(app.state.resp_time)
    data["stable_request_id_by_arrival"] = app.state.stable_req_id_by_arrival
    data["response_times_by_stable_request_id"] = app.state.resp_time_by_stable_req_id
    
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Wrote response times to {file_path}")

def launch_proxy(upstream_url: str, 
                 port: int = 12345, 
                 apply_chat_template: bool = False,
                 tokenizer_name: str = None,
                 arrival_pattern = (100, 0.0),
                 output_path: str = "eval_results.json",
                 warmup: bool = True
                 ):
    # assert avg_inter_arrival_time >= 0, "inter_arrival_time must be non-negative"
    
    if apply_chat_template:
        assert tokenizer_name is not None, "tokenizer_name must be provided if apply_chat_template is True"
    
    app.state.upstream_url = upstream_url
    app.state.apply_chat_template = apply_chat_template
    app.state.request_arrival_time = generate_request_arrival_times(
        arrival_pattern
    )
    logger.info(f"Generated request arrival times: {app.state.request_arrival_time}")
    app.state.first_request_arrival_time = None

    if tokenizer_name:
        app.state.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    else:
        app.state.tokenizer = None

    logger.info(f"Tokenizer loaded")

    app.state.req_count = 0
    app.state.next_release_time = None
    app.state.resp_time = {}
    app.state.stable_req_id_by_arrival = {}
    app.state.resp_time_by_stable_req_id = {}
    app.state.does_warmup = warmup
    app.state.warmup_done = False
    app.state.token_count = 0

     
    def _run():
        logger.info(f"Starting proxy on port {port}, forwarding to {upstream_url}")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")

    # wait_until_up(upstream_url)
        
    threading.Thread(target=_run, daemon=True).start()
    atexit.register(write_response_times, file_path=output_path)
