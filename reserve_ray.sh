#!/bin/bash
#SBATCH -N 4                # number of nodes
#SBATCH -n 4                # tasks = number of Ray processes
#SBATCH --ntasks-per-node=1
#SBATCH -p h100          # Queue (partition) name
#SBATCH --time=6:00:00             # Walltime (hh:mm:ss)

module load ray   # if your cluster requires this

# Get all hostnames
# nodes=($(scontrol show hostnames $SLURM_NODELIST))
head_node=${nodes[0]}

echo "Allocated nodes: ${nodes[@]}"
echo "Head node: $head_node"

# Start Ray head on node 0
if [[ $SLURM_PROCID -eq 0 ]]; then
    echo "Starting Ray head on $head_node"
    ray start --head --port=6379 --num-cpus=32
fi

# Wait so head starts fully
sleep 5

# Get head IP from ENV (no SSH!)
HEAD_IP=$(getent hosts $head_node | awk '{print $1}')

echo "Head IP = $HEAD_IP"

# All other nodes start workers
if [[ $SLURM_PROCID -ne 0 ]]; then
    echo "Starting Ray worker on $(hostname)"
    ray start --address=${HEAD_IP}:6379 --num-cpus=32
fi

# Prevent job from exiting so Ray stays alive
sleep 86400  # Sleep for 8 hours (28800 seconds
