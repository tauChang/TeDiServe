# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Today is good because",
    "My favorite color is",
    "Hello, my name is",
    # "Hello, my name is",
    # "Hello, my name is",
    # "Hello, my name is",
    # "Hello, my name is",
    # "Hello, my name is",
    # "I am a student at the University of Wisconsin-Madison. I",
    # "Lily can run 20000 kilometers per hour. How many kilometers can she run in 8 hours? Explain your answer.",
    # "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours? Explain your answer.",
    # "The weather today is",
    # "The president of the United States is",
    # "The capital of Franc e is",
    # "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, min_tokens=10, max_tokens=10)


def main():
    # Create an LLM.
    # llm = LLM(model="facebook/opt-125m")
    llm = LLM(
        model="GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
        # tensor_parallel_size=2
        # pipeline_parallel_size=3
        )
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
