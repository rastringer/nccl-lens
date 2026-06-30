#include <common.h>
#include "config.h"
#include "benchmark.h"

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <string>
#include <iostream>


int main(int argc, char** argv) {

  auto cfg = parseArgs(argc, argv);

  auto mpi = initMPI(argc, argv);

  CHECK_CUDA(cudaSetDevice(mpi.rank % 2));

  auto nccl = initNCCL(mpi);
  auto buffers = allocateBuffers(cfg);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  runBenchmark(cfg, mpi, nccl, buffers, stream);

  cudaStreamDestroy(stream);
  freeBuffers(buffers);
  destroyNCCL(nccl);
  finalizeMPI();

  return 0;
}