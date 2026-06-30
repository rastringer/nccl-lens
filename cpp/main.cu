#include <common.h>
#include "config.h"
#include "benchmark.h"
#include "devices.h"

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <string>
#include <iostream>


int main(int argc, char** argv) {

  auto cfg = parseArgs(argc, argv);

  auto mpi = initMPI(argc, argv);

  auto device = initDeviceForRank(mpi.rank);

  std::cerr << "[rank " << mpi.rank
            << "] cuda_device=" << device.device_id
            << " device_count=" << device.device_count
            << std::endl;

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