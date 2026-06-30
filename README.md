# nccl-lens

(WIP) A lightweight NCCL communication profiler for detecting tail latency and slow ranks in distributed GPU workloads.

## Features

- CUDA event timing
- MPI + NCCL benchmark
- Configurable (iterations, warmup, bytes, slow rank etc)
- Synthetic slow-rank injection
- JSON trace output

## Architecture

```
            nccl-lens

        +------------------+
        | Benchmark (C++)  |
        +------------------+
                  │
        JSONL telemetry
                  │
                  ▼
        +------------------+
        | Analyzer (Python)|
        +------------------+
                  │
                  ▼
        +------------------+
        | Plotly Reports   |
        +------------------+
```

## Example JSONL output

`{"rank":0,"iter":42,"gpu_ms":8.91}`

## Plots

Include latency over time, latency distribution and heatmaps.

![Latency by Rank](https://github.com/rastringer/nccl-lens/blob/main/plots/latency_by_rank.png?raw=true)

![Heatmap](https://github.com/rastringer/nccl-lens/blob/main/plots/latency_heatmap.png?raw=true)

## Installation

[Best on an instance with at least 2 x GPUs]

```bash
git clone https://github.com/rastringer/nccl-lens.git
cd nccl-lens
cmake -S cpp -B build
cmake --build build -j
./scripts/setup_ubuntu_cuda.sh
./scripts/smoke_test.sh
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
