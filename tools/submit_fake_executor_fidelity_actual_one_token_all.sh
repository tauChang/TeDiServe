#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-python}
TEMPLATE_PATH=${TEMPLATE_PATH:-"$REPO_ROOT/run_lmeval_multi_sbatch.sh"}
GPU_COUNTS=(1 2 4 8 16)

usage() {
  cat <<'EOF'
Usage: submit_fake_executor_fidelity_actual_one_token_all.sh [--dry-run]

Environment variables:
  PYTHON_BIN     Python executable to use. Defaults to `python`.
  TEMPLATE_PATH  SBATCH template script. Defaults to run_lmeval_multi_sbatch.sh.

This submits:
  sbatch_config/fake_executor_fidelity_actual_one_token_{1,2,4,8,16}.json
EOF
}

DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$TEMPLATE_PATH" ]]; then
  echo "Template not found: $TEMPLATE_PATH" >&2
  exit 1
fi

for gpu_count in "${GPU_COUNTS[@]}"; do
  config_path="$REPO_ROOT/sbatch_config/fake_executor_fidelity_actual_one_token_${gpu_count}.json"
  if [[ ! -f "$config_path" ]]; then
    echo "Config not found: $config_path" >&2
    exit 1
  fi

  echo "=== fake_executor_fidelity_actual_one_token_${gpu_count} ==="
  echo "Config: $config_path"

  if [[ "$DRY_RUN" == true ]]; then
    echo "$PYTHON_BIN $REPO_ROOT/submit_job.py $config_path $TEMPLATE_PATH"
    continue
  fi

  "$PYTHON_BIN" "$REPO_ROOT/submit_job.py" "$config_path" "$TEMPLATE_PATH"
  echo
done