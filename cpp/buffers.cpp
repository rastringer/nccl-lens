#include "buffers.h"
#include <common.h>

#include <iostream>


// Allocate and initialize GPU send/receive buffers.
BufferContext allocateBuffers(const Config& cfg) {
  if (cfg.bytes == 0 || cfg.bytes % sizeof(float) != 0) {
    std::cerr << "--bytes must be non-zero and divisible by sizeof(float)\n";
    std::exit(1);
  }

  BufferContext buffers;
  buffers.bytes = cfg.bytes;
  buffers.elements = cfg.bytes / sizeof(float);

  CHECK_CUDA(cudaMalloc(&buffers.send, buffers.bytes));
  CHECK_CUDA(cudaMalloc(&buffers.recv, buffers.bytes));

  CHECK_CUDA(cudaMemset(buffers.send, 1, buffers.bytes));
  CHECK_CUDA(cudaMemset(buffers.recv, 0, buffers.bytes));

  return buffers;
}

void freeBuffers(BufferContext&);