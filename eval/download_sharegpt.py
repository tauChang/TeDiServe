#!/usr/bin/env python3
import json
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_FILE = "sharegpt_gpt4_clean.jsonl"


def extract_value(turn):
    """Return the text field from the conversation turn."""
    # Dataset variants use either 'value' or 'content'
    return turn.get("value") or turn.get("content") or ""


def normalize_conversation(conv):
    """
    Convert the dataset conversation format into:
    [
        {"role": "...", "value": "..."},
        ...
    ]
    """
    normalized = []
    for t in conv:
        val = extract_value(t)
        if val is None:
            val = ""
        normalized.append({"role": t.get("role", "unknown"), "value": val})
    return normalized


def main():
    print("Loading dataset from HuggingFace...")
    ds = load_dataset("shibing624/sharegpt_gpt4")

    print(f"Dataset loaded. Total samples: {len(ds['train'])}")
    print(f"Writing cleaned JSONL to: {OUTPUT_FILE}")

    count_written = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for item in tqdm(ds["train"], desc="Processing"):

            conv = item.get("conversations")
            if not conv or len(conv) < 2:
                continue

            # Normalize format
            conv_norm = normalize_conversation(conv)

            # Skip if first 2 turns are empty
            if not conv_norm[0]["value"].strip() or not conv_norm[1]["value"].strip():
                continue

            line = json.dumps({"conversations": conv_norm}, ensure_ascii=False)
            out_f.write(line + "\n")
            count_written += 1

    print(f"Done! Wrote {count_written} cleaned samples to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
