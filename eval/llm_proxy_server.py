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

from fastapi import FastAPI, Request
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
app = FastAPI()

def generate_request_arrival_times(
        arrival_pattern,
    ):
    random.seed(42)  # for reproducibility
    arrival_times = []
    current_time = 0.0
    for (num_requests, inter_arrival_time) in arrival_pattern:
        for _ in range(num_requests):
            if inter_arrival_time > 0:
                inter_arrival = random.expovariate(1.0 / inter_arrival_time)
            else:
                inter_arrival = 0
            current_time += inter_arrival
            arrival_times.append(current_time)
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
            pass  # connection refused, keep trying

        if time.monotonic() - start > timeout:
            raise TimeoutError(f"Server at {url} not ready after {timeout:.1f} seconds")
        time.sleep(interval)
            
@app.post("/v1/completions")
async def proxy_completions(request: Request):
    data = await request.json()
    if app.state.first_request_arrival_time is None:
        app.state.first_request_arrival_time = time.monotonic()

    req_id = app.state.req_count
    app.state.req_count += 1
    logger.info(f"data: {data}")

    if app.state.apply_chat_template:
        prompt = data["prompt"]
        m = [{"role": "user", "content": prompt}, ]
        prompt = app.state.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

        data["prompt"] = prompt 

        # logger.info(f"Request {req_id} after chat template: {data['prompt']}")
    
        
    # Schedule the release time for this request
    delay = max(0.0, 
                app.state.first_request_arrival_time + app.state.request_arrival_time[req_id] - time.monotonic())
    # delay = max(0.0, app.state.next_release_time - time.monotonic())

    logger.info(
        f"Request {req_id}: "
        f"scheduled after {delay:.3f}s (gap {app.state.request_arrival_time[req_id]:.3f}s)"
    )

    headers = {"X-Request-Id": str(req_id)}

    # Sleep until its release time
    await asyncio.sleep(delay)
    logger.info(f"Request {req_id} released after waiting {delay:.3f}s")

    start_time = time.monotonic()
    # Forward to the real vLLM server
    async with httpx.AsyncClient(timeout=1200) as client:
        resp = await client.post(
            f"{app.state.upstream_url}/completions", 
            json=data,
            headers=headers)
        resp.raise_for_status()
    elapsed = time.monotonic() - start_time
    app.state.resp_time[req_id] = elapsed

    # add elapsed time to resp
    resp = resp.json()
    logger.info(f"Request {req_id} got response {resp}")

    return resp

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
    
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Wrote response times to {file_path}")

def launch_proxy(upstream_url: str, 
                 port: int = 12345, 
                 apply_chat_template: bool = False,
                 tokenizer_name: str = None,
                 arrival_pattern = (100, 0.0),
                 output_path: str = "eval_results.json"
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
        app.state.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        app.state.tokenizer = None

    app.state.req_count = 0
    app.state.next_release_time = None
    app.state.resp_time = {}

     
    def _run():
        logger.info(f"Starting proxy on port {port}, forwarding to {upstream_url}")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")

    wait_until_up(upstream_url)
        
    threading.Thread(target=_run, daemon=True).start()
    atexit.register(write_response_times, file_path=output_path)
