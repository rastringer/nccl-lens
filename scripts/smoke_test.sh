#!/usr/bin/env bash
set -euo pipefail

mkdir -p traces

NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}" \
NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
mpirun --allow-run-as-root -np 2 ./build/bench_allreduce \
  --iters 20 \
  --warmup 5 \
  --bytes 16777216 \
  --slow-rank 1 \
  --slow-every 10 \
  --slow-us 5000 \
| tee traces/smoke_slow_rank_1.jsonl
