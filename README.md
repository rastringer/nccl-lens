# nccl-lens

(WIP) A lightweight NCCL communication profiler for detecting tail lantency and slow ranks in distributed GPU workloads.

## Features

- CUDA event timing
- MPI + NCCL benchmark
- Configurable (iterations, warmup, bytes, slow rank etc)
- Synthetic slow-rank injection
- JSON trace output

## Installation

[Best on an instance with at least 2 x GPUs]

```bash
git clone https://github.com/rastringer/nccl-lens.git
cd nccl-lens
cmake -S cpp -B build
cmake --build build -j
python scripts/setup_ubuntu_cuda.sh
python scripts/smoke_test.sh
```

And to run:
```bash
NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 \
mpirun -np 2 ./bench_allreduce \
    --iters 100 \
    --warmup 5 \
    --slow-rank 1 \
    --slow-every 10 \
    --slow-us 5000
```