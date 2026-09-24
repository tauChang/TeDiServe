# TeDiServe Artifact Evaluation

This repository contains the implementation and artifact for **TeDiServe**.
We seek the EuroSys 2027 **Artifacts Functional** and **Artifacts Available**
badges.

## What should an evaluator run?

| Mode | Execution | Requirements | Purpose |
| --- | --- | --- | --- |
| **Smoke** (recommended) | Runs the real TeDiServe scheduler, control path, step-time predictor, and measured latency profile with two simulated model executors. No language model is loaded; the executors return fixed tokens and confidence values. | Linux x86_64 or ARM64 with Docker; no GPU, CUDA, model weights, Hugging Face access, Ray, or Gurobi. | Verifies that TeDiServe schedules and completes requests across multiple instances. It does not test model quality or real denoising. |
| **GPU** (optional) | Runs the same serving path with actual LLaDA model inference and checks the returned responses. | Tested on GH200 GPUs with the CUDA 12.4 reference environment and model access. | Verifies real model execution and, with two GPUs, multi-instance work distribution. It is not a paper-performance reproduction. |

## Recommended smoke test

The published image supports both Linux **x86_64/AMD64** and
**ARM64/aarch64**. Allow approximately 5 GB of disk space and Internet access
for the one-time image pull.

```bash
docker pull ghcr.io/tauchang/tediserve/tediserve-smoke:2027
mkdir -p artifact-results
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/artifact-results:/opt/tediserve/artifact-results" \
  ghcr.io/tauchang/tediserve/tediserve-smoke:2027
```

The test normally takes under one minute after the image is available. It
prints:

```text
TeDiServe artifact demo PASS (smoke mode).
```

It also writes `artifact-results/smoke/demo_result.json`. Success requires:

```json
{
  "mode": "smoke",
  "requests_completed": 8,
  "used_instances": 2,
  "passed": true
}
```

The result also contains positive assignment counts for both simulated model
instances. This demonstrates genuine multi-instance scheduling, while keeping
the evaluator workflow independent of expensive GPUs and model downloads.

Podman users may replace `docker` with `podman`. The image is a standard OCI
image and may also be pulled by Apptainer using the same `docker://` image URL.
The source recipe is [`Dockerfile.smoke`](Dockerfile.smoke).

## Optional real-GPU demonstration

This path loads `GSAI-ML/LLaDA-8B-Instruct`, executes actual inference on each
model instance, sends eight short GSM8K-style prompts, and checks the returned
answers. It was tested with two GH200 GPUs. One GPU validates model execution;
two GPUs also validate multi-instance distribution.

Create a fresh environment using the tested software stack:

```bash
conda create -n tediserve python=3.12 -y
conda activate tediserve
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1 torchvision==0.20.1
pip install -r requirements/build.txt -r requirements/cuda.txt
pip install lightgbm==4.6.0 pandas==2.2.3
pip install -e . --no-build-isolation
python artifact/check_environment.py --mode gpu
```

Run the two-instance demonstration:

```bash
bash artifact/run_demo.sh --mode gpu --instances 2 --requests 8 \
  --output-dir artifact-results/gpu
```

The first run downloads the model. A successful run writes
`artifact-results/gpu/demo_result.json` with nonempty completions,
per-instance work counts, and `"passed": true`. Use `--instances 1` when only
one GPU is available, `--model PATH_OR_HF_ID` for another compatible model,
and `--hf-home PATH` for an existing model cache.

Dynamic reconfiguration is disabled in both documented demonstrations, so
Gurobi is not required. Gurobi and a valid license are needed only for
reconfiguration experiments.

## Artifact contents

| Location | Role |
| --- | --- |
| `vllm/v1/core/sched/tedi_async_drop_scheduler.py` | Main TeDiServe deadline-aware scheduler and load control |
| `vllm/v1/resource_manager/` and `vllm/v1/resource_manager/reconfig_planner/` | Resource management and optional reconfiguration |
| `analysis/denoise_step_prediction/` | Step-time predictor used by the scheduler |
| `latency_profiles/` | Measured LLaDA and Dream scheduler profiles |
| `artifact/run_demo.py` | Smoke and GPU evaluator harness |
| `artifact/smoke_model/` | Metadata-only configuration for simulated execution |

The artifact focuses on installation and short functional demonstrations. It
does not reproduce every paper figure, long trace replay, or full baseline
comparison; those require large traces, multiple GPUs, and long runs.

## Troubleshooting

- If `artifact-results/smoke` already exists, use a fresh working directory or
  remove only that previous result directory; the harness does not overwrite
  results.
- For GPU mode, the toolkit selected by `nvcc` must match the CUDA family of
  the installed PyTorch wheel. The tested combination is PyTorch 2.5.1 with
  CUDA 12.4.
- Model download or access errors affect only the optional GPU demonstration.
