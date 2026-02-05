#!/usr/bin/env python3
import asyncio
import json
import random
import sys
from openai import AsyncOpenAI

import config
from collections import Counter


# === Test parameters ===
SEED = 12345
NUM_REQUESTS = 200
MAX_CONCURRENCY = 10   # throttle if desired


async def main():
    # ---------------------
    # Load ground truth
    # ---------------------
    with open(config.GROUND_TRUTH_FILE) as f:
        gt = json.load(f)

    items = gt["data"]
    model = gt["model"]

    # ---------------------
    # Deterministic sampling
    # ---------------------
    random.seed(SEED)
    sampled_items = [
        random.choice(items)
        for _ in range(NUM_REQUESTS)
    ]
    
    # Print prompt ID counts
    prompt_counts = Counter(item["id"] for item in sampled_items)
    print("Prompt ID -> Count:")
    for prompt_id, count in sorted(prompt_counts.items()):
        print(f"  {prompt_id} -> {count}")
    print()

    # ---------------------
    # Client
    # ---------------------
    client = AsyncOpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_API_BASE,
    )

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def call_one(req_id, item):
        async with sem:
            print(f"Firing request {req_id}...",flush=True)
            completion = await client.completions.create(
                model=model,
                prompt=item["prompt"],
                max_tokens=config.MAX_TOKENS,
                n=1,
                # If supported by your vLLM fork, this helps debugging
                # seed=SEED,
            )
            print(f"Request {req_id} done.")

            text = completion.choices[0].text

            if text != item["response"]:
                return {
                    "req_id": req_id,
                    "prompt_id": item["id"],
                    "prompt": item["prompt"],
                    "expected": item["response"],
                    "got": text,
                }

            return None

    # ---------------------
    # Fire requests
    # ---------------------
    tasks = [
        call_one(i, item)
        for i, item in enumerate(sampled_items)
    ]

    results = await asyncio.gather(*tasks)

    failures = [r for r in results if r is not None]

    # ---------------------
    # Report
    # ---------------------
    if failures:
        print("❌ ASYNC SAMPLED INVARIANCE FAILED\n")

        for f in failures:
            print(f"Request ID: {f['req_id']}")
            print(f"Prompt ID: {f['prompt_id']}")
            print("Expected:")
            print(repr(f["expected"]))
            print("Got:")
            print(repr(f["got"]))
            print("-" * 80)

        print(f"\nTotal failures: {len(failures)} / {NUM_REQUESTS}")
        print("\nPrompt ID distribution for failed requests:")
        failed_prompt_counts = Counter(f["prompt_id"] for f in failures)
        for prompt_id, count in sorted(failed_prompt_counts.items()):
            print(f"  {prompt_id} -> {count}")
        sys.exit(1)

    print(f"✅ ALL {NUM_REQUESTS} REQUESTS MATCHED GROUND TRUTH")
    print(f"Seed: {SEED}, Concurrency: {MAX_CONCURRENCY}")


if __name__ == "__main__":
    asyncio.run(main())
