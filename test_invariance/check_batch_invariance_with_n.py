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

        completion = client.completions.create(
            model=gt["model"],
            prompt=prompt,
            max_tokens=config.MAX_TOKENS,
            n=config.BATCH_N,
        )

        for idx, choice in enumerate(completion.choices):
            if choice.text != expected:
                failures += 1
                print("❌ BATCH MISMATCH")
                print(f"Prompt ID: {item['id']}")
                print(f"Batch idx: {idx}")
                print("Expected:")
                print(repr(expected))
                print("Got:")
                print(repr(choice.text))
                print("-" * 60)
        else:
            print(f"✅ Prompt {item['id']} passed batch invariance")

    if failures > 0:
        print(f"\nFAILED: {failures} prompts failed batch invariance")
        sys.exit(1)

    print("\nALL PROMPTS PASSED BATCH INVARIANCE")


if __name__ == "__main__":
    main()
