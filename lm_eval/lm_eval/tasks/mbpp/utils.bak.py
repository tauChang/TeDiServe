import re
from typing import Union

import evaluate as hf_evaluate


try:
    pass_at_k = hf_evaluate.load("code_eval")

    # run simple test to check code execution is enabled before model generation
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = pass_at_k.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_1(
    references: Union[str, list[str]], predictions: Union[str, list[list[str]]]
) -> float:
    if isinstance(references, str):
        references = [references]
    if isinstance(predictions[0], str):
        predictions = [[p] for p in predictions]
    return pass_at_k.compute(
        references=references,
        predictions=predictions,
        k=[1],
        num_workers=32,
    )[0]["pass@1"]

def extract_code_blocks(text: str) -> str:
    print("\n")
    print(f"Extracting code blocks from:")
    print("\n" + text + "\n")
    
    # 1. Normal fenced code blocks  ```python ... ```  or ``` ... ```
    pattern_fenced = r"```(?:\w+)?\s*([\s\S]*?)```"
    matches = re.findall(pattern_fenced, text)
    
    if matches:
        print(f"Extracted code block:\n")
        print(matches[0].strip())
        print("\n")
        return matches[0].strip()

    # 2. Degenerate ChatGPT-style blocks: code followed by only ``` at the end
    pattern_trailing = r"([\s\S]*?)```"
    m = re.search(pattern_trailing, text)
    if m:
        print(f"Extracted code block:\n")
        print(m.group(1).strip())
        print("\n")
        return m.group(1).strip()

    print("No code block found.\n")
    return ""

# def extract_code_blocks(text: str) -> str:
#     print("\n")
#     print(f"before extracting code blocks from")
#     print(text)
#     print("-----"*20)
#     print("\n")
#     print(f"after extracting code blocks from")
#     # Pattern to match ```...``` blocks
#     pattern = r"```(?:\w+)?\n?(.*?)\n?```"
#     # (+ ```) as we add the opening "```python" to the gen_prefix
#     matches = re.findall(pattern, r"```" + text, re.DOTALL)
#     # if no matches, try to match ```...``` blocks (after removing the language)
#     if not matches:
#         text_without_lang = re.sub(r"```python", "```", text)
#         matches = re.findall(pattern, text_without_lang, re.DOTALL)
    
#     print(f"{matches[0] if matches else ''}")
#     print("-----"*20)
#     print("\n")
#     if not matches:
#         return ""
#     else:
#         return matches[0]

# def extract_code_blocks(text: str) -> str:
#     # NEW: remove everything after first [DONE]
#     if "[DONE]" in text:
#         text = text.split("[DONE]", 1)[0]
    
#     print("\n")
#     print("-----"*20)
#     print(f"getting first [DONE] from: {text}")

#     # Pattern to match ```...``` blocks
#     pattern = r"```(?:\w+)?\n?(.*?)\n?```"

#     # (+ ```) as we add the opening "```python" to the gen_prefix
#     matches = re.findall(pattern, r"```" + text, re.DOTALL)

#     # if no matches, try to match ```...``` blocks (after removing the language)
#     if not matches:
#         text_without_lang = re.sub(r"```python", "```", text)
#         matches = re.findall(pattern, text_without_lang, re.DOTALL)

#     # NEW: fallback — if still no fenced code, extract first code-looking block
#     if not matches:
#         # try to find the *first function definition*
#         m = re.search(r"(def\s+\w+\(.*?)(?=$|\n\s*\n)", text, re.DOTALL)
#         if m:
#             return m.group(1).strip()
#         return ""

#     print(f"extracted: {matches[0]}")
#     print("-----"*20)
#     print("\n")
#     return matches[0]



def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[extract_code_blocks(r) for r in resp] for resp in resps]


def list_fewshot_samples():
    return [
        {
            "task_id": 2,
            "text": "Write a function to find the similar elements from the given two tuple lists.",
            "code": "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res) ",
            "test_list": [
                "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 3,
            "text": "Write a python function to identify non-prime numbers.",
            "code": "import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result",
            "test_list": [
                "assert is_not_prime(2) == False",
                "assert is_not_prime(10) == True",
                "assert is_not_prime(35) == True",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 4,
            "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
            "code": "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums",
            "test_list": [
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
            ],
            "is_fewshot": True,
        },
    ]
