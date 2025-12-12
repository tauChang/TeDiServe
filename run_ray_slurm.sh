#!/bin/bash

# Extract node list from SLURM
nodes=($(scontrol show hostnames $SLURM_NODELIST))
head_node=${nodes[0]}

echo "Nodes = ${nodes[@]}"
echo "Head node = ${head_node}"

# Start Ray head on the first node
echo "Starting head on $head_node"
ssh $head_node "
    ray start --head --port=6379 --num-cpus=32 --block &
" &

# Give head a moment to initialize
sleep 3

# Query head IP
HEAD_IP=$(ssh $head_node "hostname -I | awk '{print \$1}'")
echo "Head IP: $HEAD_IP"

# Start Ray worker on remaining nodes
for node in "${nodes[@]:1}"; do
    echo "Starting worker on $node"
    ssh $node "
        ray start --address=${HEAD_IP}:6379 --num-cpus=32 --block &
    " &
done

echo "All Ray nodes started."
wait

ray start --head --port=6379 --num-cpus=32 --block &
ray start --address=c563-002:6379 --num-cpus=32 --block &
ray start --address=c561-006:6379 --num-cpus=32 --block &
ray start --address=c562-005:6379 --num-cpus=32 --block &