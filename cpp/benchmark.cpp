#include <benchmark.h>
#include "telemetry.h"
#include <common.h>


// Run timed NCCL all-reduces, emit one JSON row for each rank/iteration.
void runBenchmark(
  const Config& cfg,
  const MpiContext& mpi,
  const NcclContext& nccl,
  BufferContext& buffers,
  cudaStream_t stream
) {
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  if (mpi.rank == 0) {
    std::cout << "starting allreduce loop\n";
  }

  const int total_iters = cfg.warmup + cfg.iters;

  for (int i = 0; i < total_iters; i++) {
    bool isWarmup = i < cfg.warmup;

    // Optional synthetic straggler: delay one rank before entering the collective.
    if (!isWarmup &&
        mpi.rank == cfg.slow_rank &&
        cfg.slow_every > 0 &&
        (i - cfg.warmup) % cfg.slow_every == 0) {
      usleep(cfg.slow_us);
    }

    CHECK_CUDA(cudaEventRecord(start, stream));

    CHECK_NCCL(ncclAllReduce(
      buffers.send,
      buffers.recv,
      buffers.elements,
      ncclFloat,
      ncclSum,
      nccl.comm,
      stream
    ));

    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    emitAllReduceTrace(
        mpi.rank,
        i,
        isWarmup,
        buffers.bytes,
        ms
    );
  }

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
}