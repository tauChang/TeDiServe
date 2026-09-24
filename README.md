# TeDiServe

**TeDiServe** is a cluster-level serving system for diffusion language models
(DLMs) that targets high latency-SLO attainment. It is described in the
EuroSys 2027 paper *TeDiServe: High SLO Attainment Serving for Diffusion
Language Models*.

DLMs generate many tokens in parallel per denoising step, which makes them
fast, but serving them under latency SLOs raises problems autoregressive
serving systems do not have. TeDiServe addresses them with:

- **Deadline-aware scheduling through confidence thresholds.** Each request's
  confidence threshold, which trades output quality for fewer denoising
  steps, is adjusted during generation based on how much SLO slack the
  request has left.
- **Adaptive load control.** Under heavy load, the maximum threshold available
  to requests is capped so the cluster keeps up with demand.
- **Step-time prediction that models approximate KV caching.** A learned
  predictor estimates the remaining denoising steps for a request, accounting
  for the uneven per-step cost that approximate KV caching introduces.
- **Cluster reconfiguration.** The cluster's parallelism layout is periodically
  re-planned by solving a quality-aware optimization problem as load changes.

TeDiServe is built on [vLLM](https://github.com/vllm-project/vllm) and has
been trimmed to what it needs: CUDA only, with the LLaDA and Dream model
families.

## Quick start: smoke test (no GPU)

The smoke test runs the real TeDiServe scheduler, load control, step-time
predictor and latency profile against two simulated model executors. It needs
only Docker (or Podman), on Linux x86_64 or ARM64.

```bash
docker pull ghcr.io/uw-mad-dash/tediserve/tediserve-smoke:2027
mkdir -p results
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/results:/opt/tediserve/artifact-results" \
  ghcr.io/uw-mad-dash/tediserve/tediserve-smoke:2027
```

A successful run prints `TeDiServe artifact demo PASS (smoke mode).` and
writes `results/smoke/demo_result.json`.

## Installing from source

TeDiServe is tested on NVIDIA GH200 GPUs with PyTorch 2.5.1 and CUDA 12.4.

```bash
git clone https://github.com/uw-mad-dash/TeDiServe.git
cd TeDiServe
conda create -n tediserve python=3.12 -y
conda activate tediserve
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1 torchvision==0.20.1
pip install -r requirements/build.txt -r requirements/cuda.txt
pip install lightgbm==4.6.0 pandas==2.2.3
pip install -e . --no-build-isolation
python artifact/check_environment.py --mode gpu
```

The CUDA toolkit found by `nvcc` must match the CUDA version of the installed
PyTorch wheel.

## Running TeDiServe

The simplest end-to-end check on GPUs starts a server with two model
instances, sends a small workload and validates the responses:

```bash
bash artifact/run_demo.sh --mode gpu --instances 2 --requests 8 \
  --output-dir results/gpu
```

To run a server yourself, pass TeDiServe's scheduler and settings to
`vllm serve`. This example serves LLaDA-8B-Instruct on two single-GPU
instances:

```bash
vllm serve GSAI-ML/LLaDA-8B-Instruct --trust-remote-code \
  --distributed-executor-backend ray \
  --num-gpus-per-model-executor 1,1 \
  --scheduler-cls vllm.v1.core.sched.tedi_async_drop_scheduler.TeDiLightScheduler \
  --request-latency-slo 30 \
  --default-confidence-threshold 0.9 \
  --candidate-confidence-thresholds 0.9 0.8 0.7 \
  --step-estimator-model-class vllm.v1.core.sched.step_estimator.models.light_gradient_boost_machine.LightGradientBoostMachine \
  --step-estimator-model-path analysis/denoise_step_prediction/models/lgb/gsm8k_0510_all_features/model.bin \
  --step-estimator-features-path analysis/denoise_step_prediction/models/lgb/gsm8k_0510_all_features/features.txt \
  --latency-profile-dir latency_profiles \
  --reconfig-interval -1 \
  --enforce-eager \
  --port 8000
```

The server exposes vLLM's OpenAI-compatible API:

```bash
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model": "GSAI-ML/LLaDA-8B-Instruct", "prompt": "What is 12 * 7?", "max_tokens": 64}'
```

Options specific to TeDiServe:

| Option | Meaning |
| --- | --- |
| `--num-gpus-per-model-executor` | Model instances and the GPUs (tensor-parallel degree) of each, e.g. `1,1` or `2,2` |
| `--scheduler-cls` | The TeDiServe scheduler |
| `--request-latency-slo` | Per-request latency SLO, in seconds |
| `--candidate-confidence-thresholds` | Thresholds the scheduler may choose between for each request |
| `--default-confidence-threshold` | Threshold a request starts with when it is assigned to an instance |
| `--step-estimator-*` | The step-time predictor and its trained model |
| `--latency-profile-dir` | Measured per-batch latencies, `<dir>/<model>/<GPU>/TP<n>.json`, profiled at startup if missing |
| `--cache-prefix`, `--cache-suffix` | Approximate KV caching for the tokens before and after the block being denoised |
| `--reconfig-interval` | Seconds between cluster reconfigurations; `-1` disables them |

**Reconfiguration** solves a mixed-integer program with
[Gurobi](https://www.gurobi.com/), so it needs a valid Gurobi licence. Everything
else runs without one.

## Repository layout

| Location | Contents |
| --- | --- |
| `vllm/v1/core/sched/tedi_async_drop_scheduler.py` | The TeDiServe scheduler, threshold adjustment and load control |
| `vllm/v1/core/sched/step_estimator/` | Step-time predictor |
| `vllm/v1/resource_manager/` | Resource management and the reconfiguration planner |
| `vllm/model_executor/models/llada.py`, `dream.py` | Diffusion language models |
| `analysis/denoise_step_prediction/models/` | Trained step-time predictor |
| `latency_profiles/` | Measured latency profiles for LLaDA and Dream |
| `artifact/` | Smoke and GPU demonstrations used for artifact evaluation |

## Artifact evaluation

Instructions from the EuroSys 2027 artifact evaluation are in
[`artifact/README.md`](artifact/README.md). The evaluated version is tagged
[`eurosys27-ae`](https://github.com/uw-mad-dash/TeDiServe/tree/eurosys27-ae).

## Citation

```bibtex
@inproceedings{chang2027tediserve,
  title     = {{TeDiServe}: High {SLO} Attainment Serving for Diffusion Language Models},
  author    = {Chang, Tzu-Tao and Hong, Benjamin Yuanyang and Pham, Kiet and Venkataraman, Shivaram},
  booktitle = {Proceedings of the European Conference on Computer Systems (EuroSys '27)},
  year      = {2027}
}
```

## License

TeDiServe is released under the Apache License 2.0; see [`LICENSE`](LICENSE).
It is derived from vLLM, and third-party components are listed in
[`THIRD_PARTY.md`](THIRD_PARTY.md).
