# Third-party components and data

TeDiServe is a research fork of [vLLM](https://github.com/vllm-project/vllm)
and retains vLLM source and its Apache-2.0 license in the repository root.
Individual source files retain their original license headers where applicable.

The artifact retains the following third-party components:

| Component | Location | License/provenance |
| --- | --- | --- |
| vLLM | `vllm/`, supporting build and documentation files | Apache-2.0; upstream vLLM source, with TeDiServe modifications |
| LightGBM | Python dependency | MIT; installed separately, not vendored |
| LLaDA-8B-Instruct | downloaded only for optional GPU MWE | Subject to the model repository's Hugging Face license and access terms; not redistributed here |

The `artifact/smoke_model/` metadata was created for this artifact's portable
simulated scheduler smoke test. It contains no pretrained weights, benchmark
data, or paper measurement data. The checked-in `latency_profiles/` files are retained
measurements from the research experiments and remain subject to the source
repository's provenance.

The final archive must preserve all license files already present in the source
tree and comply with model/dataset terms when users download optional external
assets.
