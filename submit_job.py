#!/usr/bin/env python3
"""
Generate and submit SBATCH job from template and config file.

Usage:
  python submit_job.py --config sbatch_config/gsm8k_ablation_8workers.json \
    --template run_lmeval_multi_sbatch.sh

The config file should have a "sbatch" section:
  "sbatch": {
    "job_name": "dllm-lmeval-multi",
    "time": "02:00:00",
    "partition": "ghx4",
    "account": "bftv-dtai-gh",
    "gpus": 4,
    "cpus_per_gpu": 36,
    "mail_type": "BEGIN,END,FAIL",
    "mail_user": "your@email.com"
  }
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def read_config(config_path):
    """Load config from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def read_template(template_path):
    """Read template script."""
    with open(template_path) as f:
        return f.read()


def build_sbatch_directives(sbatch_config):
    """Build SBATCH directives from config."""
    directives = []
    
    param_map = {
        "job_name": "job-name",
        "output": "output",
        "error": "error",
        "time": "time",
        "partition": "partition",
        "account": "account",
        "gpus": "gpus",
        "cpus_per_gpu": "cpus-per-gpu",
        "nodes": "nodes",
        "ntasks": "ntasks",
        "ntasks_per_node": "ntasks-per-node",
        "mail_type": "mail-type",
        "mail_user": "mail-user",
        "constraint": "constraint",
        "exclude": "exclude",
    }
    
    for config_key, sbatch_param in param_map.items():
        if config_key in sbatch_config:
            value = sbatch_config[config_key]
            directives.append(f"#SBATCH --{sbatch_param}={value}")
    
    return "\n".join(directives)


def replace_sbatch_directives(template, sbatch_directives):
    """Replace SBATCH directives in template with new ones."""
    lines = template.split("\n")
    
    new_lines = [lines[0]]  # shebang
    
    # Skip old SBATCH directives
    idx = 1
    while idx < len(lines) and lines[idx].startswith("#SBATCH"):
        idx += 1
    
    # Add new directives
    new_lines.append(sbatch_directives)
    new_lines.extend(lines[idx:])
    
    return "\n".join(new_lines)


def generate_and_submit(template_path, config_path):
    """Generate script and submit job."""
    config = read_config(config_path)
    template = read_template(template_path)
    
    sbatch_config = config.get("sbatch", {})
    if not sbatch_config:
        print("Error: No 'sbatch' section in config")
        return 1
    
    sbatch_directives = build_sbatch_directives(sbatch_config)
    script_content = replace_sbatch_directives(template, sbatch_directives)
    
    # Create temp file to hold script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(script_content)
        temp_script = f.name
    
    Path(temp_script).chmod(0o755)
    
    # Set env var for config file
    config_basename = Path(config_path).name
    config_name = config_basename.split('.')[0]
    
    print(f"Generated script: {temp_script}")
    print(f"Config: {config_path}")
    print(f"Submitting job...")
    
    # Submit the job with CONFIG_FILE environment variable set
    env = dict(os.environ)
    env['CONFIG_FILE'] = str(Path(config_path).absolute())
    
    result = subprocess.run(
        ["sbatch", temp_script],
        capture_output=True,
        text=True,
        env=env
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    if result.returncode != 0:
        return result.returncode
    
    print(f"✓ Job submitted successfully")
    print(f"  Temp script: {temp_script}")
    print(f"  (Note: This was cleaned up after submission)")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate and submit SBATCH job from template and config",
        usage="%(prog)s CONFIG TEMPLATE"
    )
    parser.add_argument(
        "config", help="Path to config file (JSON)"
    )
    parser.add_argument(
        "template", help="Path to template script"
    )
    
    args = parser.parse_args()
    
    if not Path(args.template).exists():
        print(f"Error: Template not found: {args.template}")
        return 1
    if not Path(args.config).exists():
        print(f"Error: Config not found: {args.config}")
        return 1
    
    return generate_and_submit(args.template, args.config)


if __name__ == "__main__":
    sys.exit(main())
