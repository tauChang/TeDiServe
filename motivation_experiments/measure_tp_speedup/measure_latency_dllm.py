import time
import asyncio
import httpx

SERVER_URL = "http://localhost:8000/v1/completions"   # <-- change if needed
N_REQUESTS = 5
N_WARMUP = 1
OUTPUT_TOKENS = 512

# Synthetic prompt of ~512 tokens
BASE_PROMPT = " ".join(["hello"] * 1536)


async def measure_one(prompt: str):
    """Send one TeDiServe request and measure end-to-end latency."""
    async with httpx.AsyncClient(timeout=None) as client:
        start = time.time()

        r = await client.post(
            SERVER_URL,
            json={
                "prompt": prompt,
                "max_tokens": OUTPUT_TOKENS,
            },
        )
        # print the response for debugging
        print("Response:", r.json())

        end = time.time()
        total = end - start

        if r.status_code != 200:
            print("ERROR:", r.status_code, r.text)

        return total


async def main():
    # -----------------------------
    # Warmup
    # -----------------------------
    print(f"Running {N_WARMUP} warmup requests...")
    for _ in range(N_WARMUP):
        await measure_one(BASE_PROMPT)
    print("Warmup complete.\n")

    # -----------------------------
    # Measured requests
    # -----------------------------
    total_times = []

    for i in range(N_REQUESTS):
        print(f"Request {i+1}/{N_REQUESTS}")
        t = await measure_one(BASE_PROMPT)
        total_times.append(t)

        print(f"  total={t:.4f}s")

    # -----------------------------
    # Final Summary
    # -----------------------------
    avg_total = sum(total_times) / len(total_times)

    print("\n=== RESULTS (TeDiServe) ===")
    print(f"Avg total request latency: {avg_total:.4f} s")
    print(f"Min latency:               {min(total_times):.4f} s")
    print(f"Max latency:               {max(total_times):.4f} s")


if __name__ == "__main__":
    asyncio.run(main())
