#include <nccl.h>
#include <cuda_runtime.h>
#include <mpi.h>

#define CHECK_CUDA(x) do { auto e=(x); if(e!=cudaSuccess){printf("CUDA %s\n", cudaGetErrorString(e)); exit(1);} } while(0)
#define CHECK_NCCL(x) do { auto e=(x); if(e!=ncclSuccess){printf("NCCL %s\n", ncclGetErrorString(e)); exit(1);} } while(0)
