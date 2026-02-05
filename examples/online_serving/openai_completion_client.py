# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"


def parse_args():
    parser = argparse.ArgumentParser(description="Client for vLLM API server")
    parser.add_argument(
        "--stream", action="store_true", help="Enable streaming response"
    )
    return parser.parse_args()


def main(args):
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    # Completion API
    import time
    start_time = time.time()
    completion = client.completions.create(
        model=model,
        # prompt="Hello " * 1791,
        prompt="Hello " * 511,
        # prompt="Question: Janet\u2019s ducks lay 168 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nAnswer:",
        # prompt="You are given the following problem. Solve it step by step and provide only the final numerical answer on the last line.\n\nA company has 3 warehouses. The first ships 120 items per day, the second ships 95 items per day, and the third ships twice as many items per day as the second. Each item generates $1.75 in revenue. The company operates 6 days per week. What is the total weekly revenue?\n\nExplain.",
        # prompt="Why is PhD so hard?",
        echo=False,
        n=1,
        # min_tokens=16,
        max_tokens=128,
        # stream=args.stream,
        # logprobs=3,
    )
    print(f"Time taken: {time.time() - start_time} seconds")

    print("-" * 50)
    print("Completion results:")
    if args.stream:
        for c in completion:
            print(c)
    else:
        print(completion)
        for x in completion.choices:
            print(f"Choice {x.index}:")
            print(x.text)
            print(f"length: {len(x.text)}")
            print()
    print("-" * 50)


if __name__ == "__main__":
    args = parse_args()
    main(args)
