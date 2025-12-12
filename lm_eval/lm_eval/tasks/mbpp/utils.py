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
# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from Dream repos: https://github.com/HKUNLP/Dream
"""Post-processing LLM-generated Python code implemented using tree-sitter."""

import os
import sys
import pathlib

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([os.path.dirname(ROOT), os.path.dirname(os.path.dirname(ROOT))])

import ast
import traceback

from typing import Dict, List, Optional, Set, Tuple

def refine_text(text: str) -> str:
    text =  text.replace("\t", "    ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip() + "\n"

def syntax_check(code, verbose = False):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False

def extract_longest_valid_code(text: str) -> str:
    lines = text.splitlines()

    if len(lines) > 100:
        lines = lines[:100]
    max_valid_lines = 0
    max_valid_snippet = ""

    for i in range(len(lines)):
        for j in range(i, len(lines)):
            current_snippet = "\n".join(lines[i:j+1])
            if lines[i].lstrip().startswith("def ") and syntax_check(current_snippet):
                # print(current_snippet)
                valid_line_count = sum(1 for line in lines[i:j+1] if line.strip())
                if valid_line_count > max_valid_lines:
                    max_valid_lines = valid_line_count
                    max_valid_snippet = current_snippet

    return max_valid_snippet

def get_deps(nodes: List[Tuple[str, ast.AST]]) -> Dict[str, Set[str]]:
    name2deps = {}
    for name, node in nodes:
        deps = set()
        stack = [node]
        while stack:
            current = stack.pop()
            for child in ast.iter_child_nodes(current):
                if isinstance(child, ast.Name):
                    deps.add(child.id)
                elif isinstance(child, ast.Attribute):
                    deps.add(child.attr)
                else:
                    stack.append(child)
        name2deps[name] = deps
    return name2deps

def get_function_dependency(entrypoint: str, call_graph: Dict[str, Set[str]]) -> Set[str]:
    visited = set()
    to_visit = [entrypoint]

    while to_visit:
        current = to_visit.pop(0)
        if current not in visited:
            visited.add(current)
            to_visit.extend(call_graph.get(current, set()) - visited)

    return visited

def get_definition_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    elif isinstance(node, ast.Assign):
        targets = node.targets
        if targets and isinstance(targets[0], ast.Name):
            return targets[0].id
    return None

def has_return_statement(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Return) for n in ast.walk(node))

def sanitize(text: str, entrypoint: Optional[str] = None) -> str:

    text = refine_text(text)

    # text = python_extract(text)

    code = extract_longest_valid_code(text)
    tree = ast.parse(code)
    
    definitions = {}

    imports = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.ClassDef):
            name = node.name
            definitions[name] = ('class', node)
        elif isinstance(node, ast.FunctionDef):
            name = node.name
            if has_return_statement(node):
                definitions[name] = ('function', node)
        elif isinstance(node, ast.Assign):
            name = get_definition_name(node)
            if name:
                definitions[name] = ('variable', node)

    if entrypoint:
        name2deps = get_deps([(name, node) for name, (_, node) in definitions.items()])
        reachable = get_function_dependency(entrypoint, name2deps)

    sanitized_output = []

    for node in imports:
        sanitized_output.append(ast.unparse(node))

    for name, (_, node) in definitions.items():
        if not entrypoint or name in reachable:
            sanitized_output.append(ast.unparse(node))

    return "\n".join(sanitized_output)

# def pass_at_1(references, predictions):
#     print(references[0])
#     print(predictions[0])
#     return pass_at_k.compute(
#         references=references,
#         predictions=predictions,
#         k=[1],
#     )[0]["pass@1"]
    
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

def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [extract_ref_pred(resp) for resp in resps]

def extract_ref_pred(samples):
    references = []
    predictions = []

    for sample in samples:
        # sanitized model output extraction
        pred_code = sanitize(
            sample.split("```python\n", 1)[-1].split("```")[0],
        )

        predictions.append([pred_code])

    return predictions[0]

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
