# TeDiServe: Artifact Evaluation Guide

## Artifact at a glance

TeDiServe is a cluster-level serving system for diffusion language models
(DLMs). Its main artifact claim is that a deadline-aware scheduler can assign
requests across several model instances while accounting for the different
step costs of DLM inference. The supplied MWE makes that behavior inspectable
without requiring the large models, traces, or cluster used for the paper.

This artifact targets the **Artifacts Functional** and **Artifacts Available**
badges for EuroSys 2027. Before camera-ready/AE submission, this exact source
tree will be placed in an immutable public archive (e.g., Zenodo) and the DOI
will replace the placeholder in the root README. The archival release must
include this guide and `LICENSE`.

## Contents and relation to the paper

| Artifact component | Purpose | Paper relationship |
| --- | --- | --- |
| `vllm/v1/core/sched/tedi_async_drop_scheduler.py` | TeDiServe's main deadline-aware scheduler and load control | Core system described in Sections 3–4 |
| `vllm/v1/resource_manager/` and `vllm/v1/reconfigurator/` | Model-instance resource management and optional reconfiguration | Section 4 and the reconfiguration evaluation |
| `analysis/denoise_step_prediction/` | Step-time prediction model and features | Cost model used by TeDiServe |
| `artifact/run_demo.py` | Short evaluator harness | Demonstrates multi-instance scheduling; not a throughput/accuracy figure rerun |
| `artifact/smoke_model/` and `latency_profiles/` | Tiny metadata-only smoke-test model plus checked-in LLaDA/Dream timing profiles | Profiles are measured inputs retained from the paper experiments; the MWE does not claim to reproduce their measurements |
| `vllm/v1/core/sched/{llumnix,infaas}*.py` | Retained baseline scheduler implementations | Baselines used by the paper's comparisons; not needed by the MWE |

The MWE verifies a functional system property: TeDiServe schedules a batch of
requests to multiple independent model instances and completes every request.
It does not claim to reproduce a paper headline number. Paper plots require
large trace inputs, multiple GPUs, long runs, and, for dynamic reconfiguration,
Gurobi. These are intentionally not part of the evaluator journey.

The original paper experiments used two clusters: Sections 5.2 and 5.4 used
GH200 GPUs; the accuracy experiments associated with Figure 9 used H100 GPUs.
The supported GPU evaluator environment below is GH200. Figure 9 is retained
as a research reference, not a promised reproduction target.

## System requirements

### Simulated scheduler smoke test

- Linux (the supplied reference environment is Linux/aarch64 on GH200; the
  portable test itself does not require an NVIDIA device)
- Python 3.12 (Python 3.9–3.12 is accepted by the base project; 3.12 was used
  for this artifact)
- 8 GB RAM and approximately 5 GB free disk space after installation
- No NVIDIA GPU, model weights, Internet access, Ray cluster, or Gurobi license
  is required at run time

### Optional real-GPU demonstration

- Linux with NVIDIA driver compatible with CUDA 12.4 or newer
- one or more GH200 GPUs (two GPUs for the documented two-instance command)
- at least 80 GB GPU memory per instance for `GSAI-ML/LLaDA-8B-Instruct`
- Internet access and any required Hugging Face access approval for the model
- approximately 25 GB free disk space for model download and caches

The real-GPU command intentionally disables dynamic reconfiguration. Gurobi is
therefore not needed for either documented MWE. A valid Gurobi installation and
license are needed only for reconfiguration experiments.

## Containerized smoke test (recommended)

The required evaluator path is also available as a portable OCI image. The
image uses `VLLM_TARGET_DEVICE=empty`, so it does not compile CUDA extensions
and does not require a GPU, NVIDIA container runtime, model weights, or Gurobi.
Build it from the repository root with Docker or Podman:

```bash
podman build -f Dockerfile.smoke -t tediserve-smoke:2027 .
# Docker users: replace `podman` with `docker` in the commands below.
mkdir -p artifact-results/smoke
podman run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/artifact-results:/opt/tediserve/artifact-results" \
  tediserve-smoke:2027
```

The container prints `TeDiServe artifact demo PASS (smoke mode)` and writes
`artifact-results/smoke/demo_result.json`. The image recipe is intentionally
kept in this repository; for the final public release, record the immutable
image digest (and archive the image if practical) alongside the GitHub release.
The manual installation below remains available when containers are not
permitted.

### Running on DeltaAI with Apptainer

[*DeltaAI's container support*](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/containers.html)
uses Apptainer rather than a Docker daemon. After the image has been published
to the artifact's public GitHub Container Registry location,
pull and run it as follows (replace `<owner>` with the release owner):

```bash
export APPTAINER_CACHEDIR=/tmp/$USER/apptainer-cache
mkdir -p "$APPTAINER_CACHEDIR" artifact-results/smoke
apptainer pull tediserve-smoke-2027.sif \
  docker://ghcr.io/tauchang/tediserve/tediserve-smoke:2027
apptainer run --bind "$PWD/artifact-results:/opt/tediserve/artifact-results" \
  tediserve-smoke-2027.sif
```

The smoke image does not need Apptainer's `--nv` flag. For a future GPU image,
DeltaAI's normal `apptainer run --nv` invocation would be required. The
`Dockerfile.smoke` recipe remains the source of truth, and the final release
should record both the GHCR image digest and the SIF checksum if a SIF is
archived.

## Installation

The source tree is a vLLM fork with TeDiServe changes. Build it in a dedicated
environment; do not install it over an unrelated vLLM checkout.

The reference environment is Python 3.12.9, PyTorch 2.5.1 with CUDA 12.4, Ray
2.49.2, Transformers 4.56.2, and LightGBM 4.6.0. CUDA 12.4 is the tested
reference, not a TeDiServe algorithm requirement: another CUDA/PyTorch pair
may be used if the toolkit used by `nvcc` matches the CUDA family of the
installed PyTorch wheel. Do not combine the cu124 wheel below with an
unrelated toolkit (for example, a site module that silently redirects
`cuda/12.4.0` to CUDA 12.9).

The commands below build the fork and install the TeDiServe-only Python dependency:

```bash
conda create -n tediserve python=3.12 -y
conda activate tediserve

# Install a PyTorch build compatible with the host CUDA/driver. The following
# is the reference CUDA 12.4 build; use the corresponding PyTorch command when
# your site provides a different supported CUDA version.
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1 torchvision==0.20.1

pip install -r requirements/build.txt -r requirements/cuda.txt
pip install lightgbm==4.6.0
pip install -e . --no-build-isolation
```

For an evaluator who only wants the portable smoke test, CUDA compilation can
be skipped entirely in a separate environment. Install PyTorch using the
wheel supported by that host, install the same Python requirements above, and
build the package with the no-device target:

```bash
VLLM_TARGET_DEVICE=empty pip install -e . --no-build-isolation
```

This produces an importable package without CUDA extensions; it is sufficient
for `--mode smoke` and cannot be used for the real-GPU demonstration. Keep the
GPU and smoke-only environments separate.

Building this vLLM fork compiles CUDA extensions. On a GH200 login/build node,
expect roughly 20–45 minutes and several GB of temporary build space. The
portable MWE still imports the installed project, so installation is required
even though it does not use a GPU at run time.

The CUDA build also needs a host GCC version supported by PyTorch (GCC 9 or
newer). On module-based clusters, load a current compiler module before the
build (for example, `module load gcc/11.4.0`); verify with `gcc --version`.

If `pip install -e .` cannot find a CUDA compiler, load your site's CUDA
toolkit module first and ensure that `nvcc --version` matches the intended
PyTorch/CUDA family. Do not use the GPU mode on an unsupported driver/runtime
combination.

If CMake reports that it cannot find CUDA headers or `libcudart` even though
`nvcc` is available (common on module-based clusters), export the toolkit root
before the editable install:

```bash
export CUDAToolkit_ROOT="$CUDA_HOME"
export CUDAToolkit_LIBRARY_ROOT="$CUDA_HOME"
export CUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME"
export CMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc"
pip install -e . --no-build-isolation
```

Verify the installation before running the demo:

```bash
python artifact/check_environment.py --mode smoke
```

Expected final line:

```text
TeDiServe environment check PASS (smoke mode).
```

For real GPU mode, run `python artifact/check_environment.py --mode gpu`; it
also checks that PyTorch can see at least one CUDA device.

## Evaluator MWE: no GPU required

From the repository root:

```bash
bash artifact/run_demo.sh --mode smoke --instances 2 --requests 8 \
  --output-dir artifact-results/smoke
```

Expected wall-clock time is under one minute on the reference CPU login node.
The harness runs TeDiServe's actual `TeDiLightScheduler` and the repository's
existing simulated executor. It creates two simulated model instances, sends
eight synthetic requests, and writes:

```text
artifact-results/smoke/demo_result.json
```

The command passes when the file contains all of the following:

```json
{
  "mode": "smoke",
  "requests_completed": 8,
  "used_instances": 2,
  "passed": true
}
```

The exact assignment counts may vary slightly with scheduler timing, but every
selected instance must have a positive assignment count. The program also
prints `TeDiServe artifact demo PASS (smoke mode).`

This is deliberately a simulated scheduler smoke test: no transformer
weights, tokenizer, or CUDA kernel is instantiated. `VLLM_FAKE_EXECUTOR_CPU=1`
is set inside the harness only to allow the existing simulated backend on a
CPU-only host. The main TeDiServe scheduler still runs, including its SLO-aware
admission and confidence-threshold control code, but the simulated executor
returns fixed tokens and confidence values. Consequently this mode validates
request execution, scheduling, profile lookup, and multi-instance dispatch;
it does not validate model quality, real denoising behavior, or meaningful
confidence-threshold adaptation. Those require the GPU path.

## Optional real-GPU MWE

On a machine with at least two supported GPUs, run:

```bash
bash artifact/run_demo.sh --mode gpu --instances 2 --requests 8 \
  --output-dir artifact-results/gpu
```

The harness launches the TeDiServe OpenAI-compatible server with two TP=1
instances, waits for the health endpoint, sends concurrent completion requests,
then stops the server. It uses the open model `GSAI-ML/LLaDA-8B-Instruct` by
default. First execution downloads model files and can take 10–20 minutes;
after download, budget 3–10 minutes depending on the GPU and site startup
overhead. The result file is `artifact-results/gpu/demo_result.json`; success
means all requests completed and every selected instance processed work. The
harness preserves the scheduler's assignment records in `server.log` and
copies their per-instance counts into the result JSON.

The GPU MWE uses the checked-in measured LLaDA GH200 TP1/TP2/TP4 profiles for
scheduler estimates, so it does not spend several minutes rebuilding latency
profiles. These are retained research inputs, not a claim that the small MWE
reproduces paper performance; this artifact deliberately omits the long paper
experiment workflows. The harness also uses eager execution to avoid requiring
fresh CUDA-graph capture artifacts. In addition to the completion and
per-instance counts, `demo_result.json` contains `nonempty_completions` and
short `response_summaries` (status, finish reason, and text preview) for each
request. The bundled workload consists of short GSM8K-style arithmetic
questions; each summary records its expected integer and whether that integer
appears in the completion. This is a transparent execution diagnostic, not a
model-quality or paper-accuracy claim.

Use `--model PATH_OR_HF_ID` to substitute an evaluator-accessible compatible
diffusion model. Use `--instances 1` when only one GPU is available; that
checks the real execution path but cannot demonstrate cross-instance work
distribution.

The harness uses `output-dir/hf-cache` by default, avoiding inaccessible
site-wide cache settings. Pass `--hf-home /path/to/shared/cache` explicitly if
the model is already cached elsewhere.

## What is and is not supported

Supported for Artifact Evaluation:

1. Installation of the TeDiServe/vLLM fork in the reference software stack.
2. A portable, no-GPU, two-instance scheduler demonstration with expected
   machine-readable output.
3. An optional real-GPU, multi-instance DLM serving demonstration.
4. Inspection of the scheduler, predictor, resource manager, and retained
   Llumnix/InFaaS baseline implementations.

Not supported as an evaluator workflow:

1. Full reproduction of every paper figure, long trace replay, or a performance
   comparison against other systems.
2. Reproduction of original trace datasets that are not bundled in the archive.
3. Dynamic reconfiguration without a separately installed and licensed Gurobi.

## Troubleshooting

- **`ModuleNotFoundError` or missing compiled vLLM extension:** activate the
  environment in which `pip install -e . --no-build-isolation` completed, then
  rerun `python artifact/check_environment.py --mode smoke`.
- **No CUDA device:** use `--mode smoke`. The portable MWE is the required
  evaluator path.
- **GPU model download/access fails:** configure Hugging Face credentials or
  pass a local compatible model path with `--model`. This affects only the
  optional GPU demonstration.
- **Gurobi error:** the documented MWE has reconfiguration disabled and should
  not import or require Gurobi. Such an error indicates that a different,
  research evaluation script enabled reconfiguration.
- **Port already in use in GPU mode:** select one with `--port 8001`.

## Provenance and archival checklist

For the final public release, maintainers must: (1) create a versioned GitHub
release, (2) archive the exact release through Zenodo or an equivalent
irrevocable public archive, (3) put the archive DOI/URL in the root README,
and (4) record the release commit in the submitted AE appendix. No secrets,
private model tokens, local paths, model caches, build trees, or experiment
outputs should be included in that archive.
