import json

def read_sharegpt():
    data = []
    with open('sharegpt_gpt4_clean.jsonl', 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

if __name__ == "__main__":
    conversations = read_sharegpt()
    print(f"Loaded {len(conversations)} conversations")
    # Print first conversation as example
    if conversations:
        print(f"First conversation has {len(conversations[0].get('conversations', []))} messages")