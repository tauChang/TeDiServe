import json
import asyncio
import aiohttp
from tqdm import tqdm

JUDGE_PROMPT = """
You are grading an assistant response to a user request.

### USER REQUEST:
{prompt}

### ASSISTANT ANSWER:
{response}

---

Assign one integer score from 0 to 10 using this rubric:

- 10: Fully correct, directly answers the request, complete, clear, and safe.
- 7-9: Mostly correct and useful, with minor omissions or small inaccuracies.
- 4-6: Partially correct or helpful, but missing important content or containing notable mistakes.
- 1-3: Mostly incorrect, off-topic, or seriously incomplete.
- 0: Dangerous, nonsensical, or fails to answer the request.

Evaluate only these criteria:
1. Correctness
2. Relevance
3. Completeness
4. Clarity
5. Safety

Rules:
- Judge only the assistant answer.
- Prefer factual accuracy over style.
- Penalize unsafe or harmful advice heavily.
- If the request does not provide enough information to fully verify facts, score based on likely usefulness and internal consistency.
- Return only valid JSON matching the required schema.
- The score must be an integer from 0 to 10.
- The explanation must be one short sentence.

Return JSON only.

{{"score": <0-10>, "explanation": "<one short sentence>"}}
"""

JUDGE_URL = "http://localhost:8007/v1/chat/completions"


async def judge_one(session, prompt, response):
    payload = {
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a fair LLM evaluator."},
            {"role": "user", "content": JUDGE_PROMPT.format(prompt=prompt, response=response)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "judge_result",
                "schema": {
                    "type": "object",
                    "properties": {
                        "score": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 10,
                        },
                        "explanation": {
                            "type": "string",
                        },
                    },
                    "required": ["score", "explanation"],
                    "additionalProperties": False,
                },
            },
        },
        "temperature": 0.0,
        "max_tokens": 128
    }

    async with session.post(JUDGE_URL, json=payload) as r:
        out = await r.json()
        try:
            text = out["choices"][0]["message"]["content"]
            data = json.loads(text)
        except Exception:
            data = {"score": None, "explanation": "parse_error", "raw": out}
        return data


async def judge_all(pairs, batch_size=16):
    results = []

    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        sem = asyncio.Semaphore(batch_size)

        async def worker(inst_id, p, r):
            async with sem:
                out = await judge_one(session, p, r)
                out["id"] = inst_id   # attach request ID
                return out

        tasks = [
            asyncio.create_task(worker(inst_id, prompt, response))
            for inst_id, prompt, response in pairs
        ]

        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Judging"):
            results.append(await f)

    return results


def print_stats(results):
    scores = [r["score"] for r in results if isinstance(r.get("score"), (int, float))]

    if len(scores) == 0:
        print("No valid scores found.")
        return

    import numpy as np

    print("\n=== LLM Judge Statistics ===")
    print(f"Total items: {len(results)}")
    print(f"Valid scores: {len(scores)}")

    print(f"Mean score: {np.mean(scores):.3f}")
    print(f"Median score: {np.median(scores):.3f}")
    print(f"Min score: {np.min(scores)}")
    print(f"Max score: {np.max(scores)}")

    for p in [5, 25, 75, 95]:
        print(f"P{p}: {np.percentile(scores, p):.2f}")

    parse_err = sum(1 for r in results if r.get("score") is None)
    print(f"Parse errors: {parse_err}")

    hist = {i: 0 for i in range(11)}
    for s in scores:
        hist[int(round(s))] += 1

    print("\nHistogram (score → count):")
    for k, v in hist.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("Loading input...")
    pairs = []
    with open(args.input) as f:
        data = json.load(f)
        for inst_id, inst in data[0]["instances"].items():
            pairs.append((inst_id, inst["prompt"], inst["response"]))

    print(f"Loaded {len(pairs)} instances")

    results = asyncio.run(judge_all(pairs, batch_size=100))

    print("Saving...")
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("Done.")

    print_stats(results)
