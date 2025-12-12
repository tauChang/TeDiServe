import time
import asyncio
import httpx
import numpy as np
import random
import json                                  # <<< added
from tqdm import tqdm

# ==========================================
# User configs
# ==========================================
SERVER_URL = "http://localhost:8000/v1/completions"

INPUT_TOKENS = 256
OUTPUT_TOKENS = 1024

RPS = 0.1
N_REQUESTS = 50
N_WARMUP = 3

SCHEDULER_NAME = "llumnix"                 # <<< you set this
OUTPUT_FILE = f"{SCHEDULER_NAME}_{INPUT_TOKENS}_{OUTPUT_TOKENS}_{RPS}.jsonl"   # <<< added

BASE_PROMPT = " ".join(["hello"] * INPUT_TOKENS)

random.seed(42)

# ==========================================
# Helpers
# ==========================================
def poisson_interarrival(rps: float) -> float:
    return random.expovariate(rps)


async def send_request(prompt: str):
    async with httpx.AsyncClient(timeout=None) as client:
        start = time.time()
        r = await client.post(
            SERVER_URL,
            json={"prompt": prompt, "max_tokens": OUTPUT_TOKENS},
        )
        end = time.time()
        return end - start


# ==========================================
# Main
# ==========================================
async def main():

    print(f"Warmup phase: {N_WARMUP} concurrent requests\n")
    warmup_tasks = [asyncio.create_task(send_request(BASE_PROMPT)) for _ in range(N_WARMUP)]
    await asyncio.gather(*warmup_tasks)
    print("Warmup complete.\n")

    print(f"Starting measurement: {N_REQUESTS} Poisson arrivals at {RPS} RPS\n")

    pending_tasks = []
    completed_latencies = []

    # open JSONL file for writing                          # <<< added
    outfile = open(OUTPUT_FILE, "w")

    # ------------------------------
    # Completion monitor
    # ------------------------------
    async def completion_monitor(pbar):
        nonlocal completed_latencies

        request_counter = 0

        while len(completed_latencies) < N_REQUESTS:
            for t in list(pending_tasks):
                if t.done():
                    latency = t.result()
                    completed_latencies.append(latency)

                    # write JSONL                                   # <<< added
                    record = {
                        "request_id": request_counter,
                        "latency": float(latency),
                        "arrival_index": request_counter,
                        "timestamp": time.time(),
                    }
                    outfile.write(json.dumps(record) + "\n")
                    outfile.flush()

                    request_counter += 1

                    pending_tasks.remove(t)
                    pbar.update(1)

            await asyncio.sleep(0.01)

    # ------------------------------
    # Arrival loop
    # ------------------------------
    async def arrival_loop():
        for i in range(N_REQUESTS):
            if i > 0:
                wait = poisson_interarrival(RPS)
                print(f"[arrival {i+1}/{N_REQUESTS}] waiting {wait:.3f}s")
                await asyncio.sleep(wait)

            task = asyncio.create_task(send_request(BASE_PROMPT))
            pending_tasks.append(task)

    # ------------------------------
    # Run arrival + completion monitor concurrently
    # ------------------------------
    pbar = tqdm(total=N_REQUESTS, desc="Requests completed")

    await asyncio.gather(
        arrival_loop(),
        completion_monitor(pbar),
    )

    pbar.close()
    outfile.close()   # <<< added

    # ------------------------------
    # Final stats
    # ------------------------------
    latencies = np.array(completed_latencies)
    avg = latencies.mean()
    p90 = np.percentile(latencies, 90)
    p99 = np.percentile(latencies, 99)

    print("\n=== RESULTS (TeDiServe) ===")
    print(f"Avg latency:     {avg:.4f} s")
    print(f"P90 latency:     {p90:.4f} s")
    print(f"P99 latency:     {p99:.4f} s")
    print(f"Latency records written to: {OUTPUT_FILE}")   # <<< added


if __name__ == "__main__":
    asyncio.run(main())
