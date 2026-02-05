#!/usr/bin/env python3
import json
import sys
from openai import OpenAI

import config


def main():
    with open(config.GROUND_TRUTH_FILE) as f:
        gt = json.load(f)

    client = OpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_API_BASE,
    )

    failures = 0

    for item in gt["data"]:
        prompt = item["prompt"]
        expected = item["response"]

        for run_id in range(config.MULTI_RUN_TRIALS):
            completion = client.completions.create(
                model=gt["model"],
                prompt=prompt,
                max_tokens=config.MAX_TOKENS,
                n=1,
            )

            text = completion.choices[0].text

            if text != expected:
                failures += 1
                print("❌ MISMATCH")
                print(f"Prompt ID: {item['id']}")
                print(f"Run: {run_id}")
                print("Expected:")
                print(repr(expected))
                print("Got:")
                print(repr(text))
                print("-" * 60)
                break
        else:
            print(f"✅ Prompt {item['id']} passed multi-run check")

    if failures > 0:
        print(f"\nFAILED: {failures} prompts are non-deterministic")
        sys.exit(1)

    print("\nALL PROMPTS PASSED MULTI-RUN INVARIANCE")


if __name__ == "__main__":
    main()
