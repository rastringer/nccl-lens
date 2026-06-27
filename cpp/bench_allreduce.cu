#include <nccl.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <string>
#include <iostream>

#define CHECK_CUDA(x) do { auto e=(x); if(e!=cudaSuccess){printf("CUDA %s\n", cudaGetErrorString(e)); exit(1);} } while(0)
#define CHECK_NCCL(x) do { auto e=(x); if(e!=ncclSuccess){printf("NCCL %s\n", ncclGetErrorString(e)); exit(1);} } while(0)

struct Config {
  int iters = 100;
  int warmup = 5;
  size_t bytes = 16 * 1024 * 1024;
  int slow_rank = -1;      // -1 disables synthetic delay
  int slow_every = 10;     // inject delay every N measured iterations
  int slow_us = 5000;      // delay duration in microseconds
};

struct MpiContext {
  int rank = 0;
  int world = 1;
};

struct NcclContext {
  ncclComm_t comm = nullptr;
};

struct BufferContext {
  float* send = nullptr;
  float* recv = nullptr;
  size_t elements = 0;
  size_t bytes = 0;
};

// Parse CLI flags into a config object.
Config parseArgs(int argc, char** argv) {
  Config cfg;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "--iters" && i + 1 < argc) {
      cfg.iters = std::stoi(argv[++i]);
    } else if (arg == "--warmup" && i + 1 < argc) {
      cfg.warmup = std::stoi(argv[++i]);
    } else if (arg == "--bytes" && i + 1 < argc) {
      cfg.bytes = std::stoull(argv[++i]);
    } else if (arg == "--slow-rank" && i + 1 < argc) {
      cfg.slow_rank = std::stoi(argv[++i]);
    } else if (arg == "--slow-every" && i + 1 < argc) {
      cfg.slow_every = std::stoi(argv[++i]);
    } else if (arg == "--slow-us" && i + 1 < argc) {
      cfg.slow_us = std::stoi(argv[++i]);
    } else {
      std::cerr << "Unknown or incomplete arg: " << arg << "\n";
      std::exit(1);
    }
  }

  return cfg;
}

// Initialize MPI, find the rank/world size of the process.
MpiContext initMPI(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  MpiContext ctx;
  MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ctx.world);

  return ctx;
}

// Create one NCCL communicator shared across all MPI ranks.
NcclContext initNCCL(const MpiContext& mpi) {
  ncclUniqueId id;

  if (mpi.rank == 0) {
    CHECK_NCCL(ncclGetUniqueId(&id));
  }

  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  NcclContext ctx;
  CHECK_NCCL(ncclCommInitRank(&ctx.comm, mpi.world, id, mpi.rank));

  return ctx;
}

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

    printf(
      "{\"rank\":%d,\"iter\":%d,\"warmup\":%s,\"op\":\"all_reduce\",\"bytes\":%zu,\"gpu_ms\":%.3f}\n",
      mpi.rank,
      i,
      isWarmup ? "true" : "false",
      buffers.bytes,
      ms
    );
  }

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
}

// Tear down GPU, NCCL, and MPI resources.
void cleanup(NcclContext& nccl, BufferContext& buffers, cudaStream_t stream) {
  if (stream != nullptr) CHECK_CUDA(cudaStreamDestroy(stream));

  if (buffers.send != nullptr) {
    CHECK_CUDA(cudaFree(buffers.send));
    buffers.send = nullptr;
  }

  if (buffers.recv != nullptr) {
    CHECK_CUDA(cudaFree(buffers.recv));
    buffers.recv = nullptr;
  }

  if (nccl.comm != nullptr) {
    CHECK_NCCL(ncclCommDestroy(nccl.comm));
    nccl.comm = nullptr;
  }

  MPI_Finalize();
}

int main(int argc, char** argv) {
  Config cfg = parseArgs(argc, argv);

  MpiContext mpi = initMPI(argc, argv);

  CHECK_CUDA(cudaSetDevice(mpi.rank % 2));

  NcclContext nccl = initNCCL(mpi);
  BufferContext buffers = allocateBuffers(cfg);

  cudaStream_t stream = nullptr;
  CHECK_CUDA(cudaStreamCreate(&stream));

  runBenchmark(cfg, mpi, nccl, buffers, stream);

  cleanup(nccl, buffers, stream);

  return 0;
}