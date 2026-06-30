#pragma once 

#include <config.h>
#include <cuda_runtime.h>

void runBenchmark(
  const Config& cfg,
  const MpiContext& mpi,
  const NcclContext& nccl,
  BufferContext& buffers,
  cudaStream_t stream
);