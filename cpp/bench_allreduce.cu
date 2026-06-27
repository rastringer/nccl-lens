#include <nccl.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>
#include <string>
#include <iostream>

#define CHECK_CUDA(x) do { auto e=(x); if(e!=cudaSuccess){printf("CUDA %s\n", cudaGetErrorString(e)); exit(1);} } while(0)
#define CHECK_NCCL(x) do { auto e=(x); if(e!=ncclSuccess){printf("NCCL %s\n", ncclGetErrorString(e)); exit(1);} } while(0)

struct Config {
  int iters = 100;
  int warmup = 5;
  size_t bytes = 16 * 1024 * 1024;
  int slow_rank = -1;
  int slow_every = 10;
  int slow_us = 5000;
};

Config parseArgs(int argc, char** argv) {
  Config cfg;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "--iters" && i + 1 < argc) {
      cfg.iters = std::stoi(argv[++i]);
    } else if (arg == "--warmup" && i + 1 < argc) {
        cfg.iters = std::stoi(argv[++i]);
    } else if (arg == "--bytes" && i + 1 < argc) {
        cfg.iters = std::stoi(argv[++i]);
    } else if (arg == "--slow-rank" && i + 1 < argc) {
        cfg.iters = std::stoi(argv[++i]);
    } else if (arg == "--slow-every" && i + 1 < argc) {
        cfg.iters = std::stoi(argv[++i]);
    } else if (arg == "--slow-us" && i + 1 < argc) {
        cfg.iters = std::stoi(argv[++i]);
    } else {
	std::cerr << "Unknown or incomplete arg: " << arg << "\n";
	std::exit(1);
    }
  }

  return cfg;
}

int main(int argc, char** argv) {
 
  Config cfg = parseArgs(argc, argv);

  MPI_Init(&argc, &argv);
 
  int rank, world;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  CHECK_CUDA(cudaSetDevice(rank % 2));

  ncclUniqueId id;
  if (rank == 0) ncclGetUniqueId(&id);
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  ncclComm_t comm;
  CHECK_NCCL(ncclCommInitRank(&comm, world, id, rank));

  size_t n = cfg.bytes / sizeof(float);
  float *send, *recv;
  CHECK_CUDA(cudaMalloc(&send, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&recv, n * sizeof(float)));
  CHECK_CUDA(cudaMemset(send, 1, n * sizeof(float)));
  CHECK_CUDA(cudaMemset(recv, 0, n * sizeof(float)));

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  
  if (rank == 0) printf("starting allreduce loop\n");
  const int total_iters = cfg.warmup + cfg.iters;
  for (int i = 0; i < total_iters; i++) {
    bool is_warmup = i < cfg.warmup;

    if (!is_warmup && 
	rank == cfg.slow_rank &&
	cfg.slow_every > 0 &&
	(i - cfg.warmup) % cfg.slow_every == 0) {
	usleep(cfg.slow_us);
    }

    CHECK_CUDA(cudaEventRecord(start, stream));
    CHECK_NCCL(ncclAllReduce(send, recv, n, ncclFloat, ncclSum, comm, stream));
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    
    printf("{\"rank\":%d,\"iter\":%d,\"warmup\":%s,\"op\":\"all_reduce\",\"bytes\":%zu,\"gpu_ms\":%.3f}\n",
      rank, 
      i, 
      i < cfg.warmup ? "true" : "false",
      cfg.bytes,
      ms
    );

    if (rank == 1 && i >= cfg.warmup && i % 10 == 0) {
      usleep(5000);  // 5ms CPU-side delay before enqueue
    }
  }
  
  if (rank == 0) printf("finished loop\n");

  CHECK_CUDA(cudaStreamSynchronize(stream));

  if (rank == 0) {
    printf("OK: ran %d NCCL all-reduces across %d ranks\n", cfg.iters, world);
    std::cout << "Configuration\n";
    std::cout << "-------------\n";
    std::cout << "iters:   " << cfg.iters << "\n";
    std::cout << "warmup:  " << cfg.warmup << "\n";
    std::cout << "bytes:   " << cfg.bytes << "\n";
  }

  ncclCommDestroy(comm);
  cudaFree(send);
  cudaFree(recv);
  MPI_Finalize();
}
