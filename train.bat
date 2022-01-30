#!/bin/bash

set NUM_NODES=1
set NUM_GPUS_PER_NODE=1
set NODE_RANK=0
set WORLD_SIZE=1

python -m torch.distributed.launch --nproc_per_node=%NUM_GPUS_PER_NODE% --nnodes=%NUM_NODES% --node_rank %NODE_RANK% main.py
