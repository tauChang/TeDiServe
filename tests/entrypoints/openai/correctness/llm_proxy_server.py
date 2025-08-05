from fastapi import FastAPI, Request
import httpx
import asyncio
import logging
import time
import random

# Setup logger
logger = logging.getLogger("llm_proxy")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

app = FastAPI()

UPSTREAM_URL = None


@app.post("/v1/completions")
async def proxy_completions(request: Request):
    global UPSTREAM_URL
    try:
        start_time = time.time()
        data = await request.json()

        prompt = data.get("prompt", "")[:50].replace("\n", " ") + "..."
        model = data.get("model", "unknown")
        max_tokens = data.get("max_tokens", "n/a")

        logger.info(f"Incoming request: model={model}, max_tokens={max_tokens}, prompt='{prompt}'")

        delay = random.uniform(0, 10)
        logger.info(f"sleeping for {delay:.2f}s to simulate input arrival")
        await asyncio.sleep(delay)
        logger.info("woke up, forwarding request")

        timeout = data.get("timeout", 1200)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{UPSTREAM_URL}/completions", json=data)
            response.raise_for_status()

        elapsed = time.time() - start_time
        logger.info(f"Completed request in {elapsed:.2f}s")
        return response.json()

    except Exception as e:
        logger.exception("Proxy encountered an error")
        return {"error": str(e)}


def launch_proxy(upstream_url: str, port: int = 12345):
    global UPSTREAM_URL
    UPSTREAM_URL = upstream_url
    import uvicorn
    import threading

    def _run():
        logger.info(f"Starting proxy on port {port}, forwarding to {upstream_url}")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")  # quiet uvicorn logs

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
