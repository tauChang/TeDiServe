import asyncio, time, random, logging
from fastapi import FastAPI, Request
import httpx
from transformers import AutoTokenizer

logger = logging.getLogger("llm_proxy")
logger.setLevel(logging.INFO)
app = FastAPI()

UPSTREAM_URL = None
LAMBDA = 0.5         # mean 0.5 req/sec (adjust as needed)
APPLY_CHAT_TEMPLATE = False
TOKENIZER = None
req_count = 0
next_release_time = None
resp_time = []

@app.post("/v1/completions")
async def proxy_completions(request: Request):
    global req_count, next_release_time
    if next_release_time is None:
        next_release_time = time.monotonic()

    data = await request.json()
    cur_req_count = req_count
    req_count += 1
    logger.info(f"data: {data}")

    if APPLY_CHAT_TEMPLATE:
        prompt = data["prompt"]
        m = [{"role": "user", "content": prompt}, ]
        prompt = TOKENIZER.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

        data["prompt"] = prompt 

        logger.info(f"Request {cur_req_count} after chat template: {data['prompt']}")
    
        
    # Schedule the release time for this request
    inter_arrival = random.expovariate(LAMBDA)
    next_release_time += inter_arrival
    # delay = max(0.0, next_release_time - time.monotonic())
    delay = 0

    logger.info(
        f"Request {cur_req_count}: "
        f"scheduled after {delay:.3f}s (gap {inter_arrival:.3f}s)"
    )

    # Sleep until its release time
    if delay > 0:
        await asyncio.sleep(delay)
    logger.info(f"Request {cur_req_count} released after waiting {delay:.3f}s")

    start_time = time.monotonic()
    # Forward to the real vLLM server
    async with httpx.AsyncClient(timeout=1200) as client:
        resp = await client.post(f"{UPSTREAM_URL}/completions", json=data)
        resp.raise_for_status()

    elapsed = time.monotonic() - start_time
    resp_time.append(elapsed)
    logger.info(f"Request {cur_req_count} completed in {elapsed:.2f}s")
    logger.info(f"avg response time: {sum(resp_time)/len(resp_time):.2f}s")
    logger.info(f"resp.json(): {resp.json()}")
    return resp.json()

def launch_proxy(upstream_url: str, port: int = 12345, 
                 apply_chat_template: bool = False,
                 tokenizer_name: str = "GSAI-ML/LLaDA-8B-Instruct"):
    global UPSTREAM_URL, APPLY_CHAT_TEMPLATE, TOKENIZER
    UPSTREAM_URL = upstream_url
    APPLY_CHAT_TEMPLATE = apply_chat_template
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name)
    import uvicorn, threading
    random.seed(42)
    def _run():
        logger.info(f"Starting proxy on port {port}, forwarding to {upstream_url}")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")
    threading.Thread(target=_run, daemon=True).start()
