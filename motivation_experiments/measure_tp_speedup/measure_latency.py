import time
import asyncio
import httpx

SERVER_URL = "http://localhost:8007/v1/completions"
N_REQUESTS = 5     # number of measured requests
N_WARMUP = 1        # ignored warmups
OUTPUT_TOKENS = 256

# Synthetic prompt of ~512 tokens
BASE_PROMPT = " ".join(["fuck"] * 1792)


async def measure_one(prompt: str):
    """Measure prefill, per-token decode latency, and total latency."""
    async with httpx.AsyncClient(timeout=None) as client:
        start_total = time.time()
        start = time.time()
        first_token_time = None
        token_times = []

        async with client.stream(
            "POST",
            SERVER_URL,
            json={
                "model": "meta-llama/Meta-Llama-3.1-8B",
                "prompt": prompt,
                "max_tokens": OUTPUT_TOKENS,
                "ignore_eos": True,
                "stream": True,
            },
        ) as r:
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                
                now = time.time()
                token_times.append(now)

                if first_token_time is None:
                    first_token_time = now

        end_total = time.time()

        prefill = first_token_time - start
        decode_latencies = [
            t2 - t1 for t1, t2 in zip(token_times, token_times[1:])
        ]
        avg_decode = sum(decode_latencies) / len(decode_latencies)
        total_latency = end_total - start_total

        return prefill, avg_decode, total_latency


async def main():
    # -----------------------------
    # Warmup
    # -----------------------------
    print(f"Running {N_WARMUP} warm-up requests...")
    for i in range(N_WARMUP):
        await measure_one(BASE_PROMPT)
    print("Warmup complete.\n")

    # -----------------------------
    # Measured Requests
    # -----------------------------
    prefill_times = []
    decode_times = []
    total_times = []

    for i in range(N_REQUESTS):
        print(f"Request {i+1}/{N_REQUESTS}")
        p, d, t = await measure_one(BASE_PROMPT)
        prefill_times.append(p)
        decode_times.append(d)
        total_times.append(t)

        print(f"  prefill={p:.4f}s  decode={d:.4f}s/token  total={t:.4f}s")

    # -----------------------------
    # Final Summary
    # -----------------------------
    print("\n=== RESULTS ===")
    print(f"Avg prefill latency:        {sum(prefill_times)/len(prefill_times):.4f} s")
    print(f"Avg per-token decode time:  {sum(decode_times)/len(decode_times):.4f} s")
    print(f"Avg total request latency:  {sum(total_times)/len(total_times):.4f} s")


if __name__ == "__main__":
    asyncio.run(main())
