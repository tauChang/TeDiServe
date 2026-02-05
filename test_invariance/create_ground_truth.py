#!/usr/bin/env python3
import json
from openai import OpenAI

import config


PROMPTS = [
    # 1. Multi-step arithmetic + explanation
    (
        "You are given the following problem. Solve it step by step and provide "
        "only the final numerical answer on the last line.\n\n" 
        "A company has 3 warehouses. The first ships 120 items per day, the second "
        "ships 95 items per day, and the third ships twice as many items per day "
        "as the second. Each item generates $1.75 in revenue. The company operates "
        "6 days per week. What is the total weekly revenue?\n\n"
    ),

    # 2. Long-form constrained explanation
    (
        "Explain why the sky appears blue during the daytime. "
        "Your explanation must:\n"
        "1. Be exactly three paragraphs.\n"
        "2. Each paragraph must contain exactly two sentences.\n"
        "3. Do not use bullet points or equations.\n"
        "4. Do not mention the words 'physics' or 'wavelength'.\n\n"
        "Begin your explanation below:"
    ),

    # 3. Entity extraction + formatting
    (
        "Read the following paragraph and extract structured information.\n\n"
        "Paragraph:\n"
        "In 1813, Jane Austen published Pride and Prejudice, a novel that was "
        "well received in London and later adapted into multiple films, "
        "including a 2005 adaptation starring Keira Knightley.\n\n"
        "Output the result strictly in JSON with the following fields:\n"
        "- author\n"
        "- title\n"
        "- publication_year\n"
        "- notable_adaptation_year\n"
        "- notable_actor\n\n"
        "JSON:"
    ),

    # 4. Algorithmic reasoning
    (
        "Consider the following pseudocode:\n\n"
        "function f(n):\n"
        "  if n <= 1: return 1\n"
        "  return n * f(n - 1)\n\n"
        "Answer the following questions:\n"
        "1. What mathematical function does f compute?\n"
        "2. What is the value of f(7)?\n"
        "3. What is the time complexity in Big-O notation?\n\n"
        "Provide your answers as:\n"
        "Function:\n"
        "Value:\n"
        "Complexity:"
    ),

    # 5. Translation + constraints
    (
        "Translate the following English text into Spanish.\n\n"
        "Text:\n"
        "Hello! I hope you are doing well. Today is a good day to learn something new.\n\n"
        "Constraints:\n"
        "- Use neutral Latin American Spanish.\n"
        "- Keep punctuation consistent with the original.\n"
        "- Do not add or remove sentences.\n\n"
        "Translation:"
    ),

    # 6. Long factual Q&A
    (
        "Answer the following question concisely but completely.\n\n"
        "Question:\n"
        "What is entropy in information theory, and how does it differ from "
        "the everyday use of the word 'entropy'?\n\n"
        "Answer in exactly five sentences:"
    ),

    # 7. Sequence reasoning
    (
        "Given the sequence below, identify the rule and determine the next two numbers.\n\n"
        "Sequence:\n"
        "2, 4, 8, 16, 32\n\n"
        "Explain the rule in one sentence, then list the next two numbers on a single line "
        "separated by a comma."
    ),

    # 8. Summarization with structure
    (
        "Summarize the following text in exactly four bullet points.\n\n"
        "Text:\n"
        "Machine learning systems are increasingly deployed at scale, requiring "
        "careful consideration of efficiency, latency, and resource utilization. "
        "Modern systems often rely on GPUs to accelerate computation, but GPU resources "
        "are expensive and shared across workloads. Efficient scheduling and batching "
        "can significantly improve throughput, but may introduce fairness and tail-latency "
        "concerns. System designers must balance these trade-offs carefully.\n\n"
        "Summary:"
    ),

    # 9. Deterministic formatting stress
    (
        "Produce a table with exactly three rows (excluding the header) and three columns.\n\n"
        "Columns:\n"
        "Name | Square | Cube\n\n"
        "Rows should correspond to the integers 2, 3, and 4.\n"
        "Use plain text with pipes ('|') as separators.\n"
        "Do not include any extra whitespace or commentary."
    ),

    # 10. Multi-constraint creative task
    (
        "Write a haiku about winter.\n\n"
        "Constraints:\n"
        "- Exactly 3 lines\n"
        "- Syllable pattern: 5-7-5\n"
        "- Do not use the words 'snow', 'cold', or 'ice'\n"
        "- The final line must contain a verb\n\n"
        "Haiku:"
    ),
    # 11. Code generation with specific style
    (
        "Write a Python function that takes a list of integers and returns a new list "
        "containing only the even integers from the original list. \n\n"
        "Constraints:\n"
        "- Use list comprehensions.\n"
        "- Include type hints.\n"
        "- Add a docstring explaining the function.\n\n"
        "Function:"
    ),
    # 12. Historical analysis
    (
        "Analyze the causes and consequences of the fall of the Roman Empire.\n\n"
        "Your analysis must:\n"
        "1. Be structured into an introduction, body, and conclusion.\n"
        "2. Each section must contain at least two paragraphs.\n"
        "3. Avoid using the words 'decline' or 'collapse'.\n\n"
        "Begin your analysis below:"
    ),
    # 13. Debugging task
    (
        "The following Python code contains a bug. Identify and fix the bug.\n\n"
        "Code:\n"
        "def divide(a, b):\n"
        "    return a / b\n\n"
        "print(divide(10, 0))\n\n"
        "Provide the corrected code and explain the fix in one sentence:"
    ),

    # 14. Data transformation
    (
        "Transform the following CSV data into JSON format.\n\n"
        "CSV:\n"
        "name,age,city\n"
        "Alice,30,New York\n"
        "Bob,25,Los Angeles\n"
        "Charlie,35,Chicago\n\n"
        "JSON:"
    ),

    # 15. Logical reasoning
    (
        "Solve the following logical puzzle:\n\n"
        "Three people (Alice, Bob, and Charlie) are wearing hats. Each hat is either red or blue. "
        "They can see the hats of the other two people but not their own. Alice says, 'I don't know my hat color.' "
        "Bob says, 'I don't know my hat color.' Charlie says, 'I know my hat color.'\n\n"
        "What is Charlie's hat color? Explain your reasoning in two sentences:"
    ),

    # 16. SQL query generation
    (
        "Write an SQL query to retrieve the names of employees who earn more than $50,000 per year "
        "from a table named 'employees'. The table has the following columns: id, name, salary, department.\n\n"
        "Query:"
    ),

    # 17. Regex pattern creation
    (
        "Create a regular expression pattern to match email addresses. The pattern must:\n"
        "- Allow alphanumeric characters, dots, underscores, and hyphens before the '@'.\n"
        "- Require a domain name with only alphanumeric characters and dots.\n"
        "- End with a top-level domain of 2 to 6 letters.\n\n"
        "Regex:"
    ),

    # 18. Creative writing with constraints
    (
        "Write a short story in exactly 50 words about a mysterious door. "
        "The story must:\n"
        "- Include dialogue\n"
        "- End with a question\n"
        "- Use past tense throughout\n\n"
        "Story:"
    ),

    # 19. Mathematical proof
    (
        "Prove that the sum of the first n positive integers equals n(n+1)/2.\n\n"
        "Requirements:\n"
        "- Use mathematical induction\n"
        "- Clearly state the base case and inductive step\n"
        "- Show all algebraic work\n\n"
        "Proof:"
    ),

    # 20. Algorithm explanation
    (
        "Explain how the QuickSort algorithm works.\n\n"
        "Your explanation must:\n"
        "1. Describe the partitioning process in detail\n"
        "2. Explain the recursive structure\n"
        "3. Analyze best and worst-case time complexity\n"
        "4. Use exactly 4 paragraphs\n\n"
        "Explanation:"
    ),

    # 21. JSON schema validation
    (
        "Create a JSON schema that validates objects with the following properties:\n"
        "- 'name': required string, minimum 2 characters\n"
        "- 'age': required integer between 0 and 120\n"
        "- 'email': optional string in valid email format\n\n"
        "Schema:"
    ),

    # 22. Error handling code
    (
        "Write a Python function that reads a file and handles potential errors gracefully.\n\n"
        "Requirements:\n"
        "- Handle FileNotFoundError and PermissionError specifically\n"
        "- Return appropriate error messages\n"
        "- Include proper exception logging\n"
        "- Add type hints and docstring\n\n"
        "Function:"
    ),

    # 23. Boolean logic simplification
    (
        "Simplify the following Boolean expression using Boolean algebra rules:\n\n"
        "Expression: (A AND B) OR (NOT A AND B) OR (A AND NOT B)\n\n"
        "Show each step of simplification and state which rule you applied.\n"
        "Provide the final simplified form on the last line."
    ),

    # 24. API design specification
    (
        "Design a REST API endpoint for managing user profiles.\n\n"
        "Specify:\n"
        "- HTTP method and URL pattern\n"
        "- Request body format (JSON)\n"
        "- Response format for success and error cases\n"
        "- HTTP status codes\n\n"
        "API Specification:"
    ),

    # 25. Data structure implementation
    (
        "Implement a stack data structure in Python with the following methods:\n"
        "push(), pop(), peek(), is_empty(), and size().\n\n"
        "Requirements:\n"
        "- Use a list as the underlying storage\n"
        "- Handle edge cases (empty stack operations)\n"
        "- Include comprehensive docstrings\n\n"
        "Implementation:"
    ),
]


def main():
    client = OpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_API_BASE,
    )

    # Auto-discover model once
    models = client.models.list()
    model = models.data[0].id

    records = []

    for i, prompt in enumerate(PROMPTS):
        completion = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=config.MAX_TOKENS,
            n=1,
            echo=config.ECHO,
        )

        text = completion.choices[0].text

        records.append(
            {
                "id": i,
                "prompt": prompt,
                "response": text,
            }
        )

        print(f"[GT] Prompt {i} done")

    with open(config.GROUND_TRUTH_FILE, "w") as f:
        json.dump(
            {
                "model": model,
                "config": {
                    "max_tokens": config.MAX_TOKENS,
                    "temperature": config.TEMPERATURE,
                    "top_p": config.TOP_P,
                },
                "data": records,
            },
            f,
            indent=2,
        )

    print(f"\nGround truth written to {config.GROUND_TRUTH_FILE}")


if __name__ == "__main__":
    main()
