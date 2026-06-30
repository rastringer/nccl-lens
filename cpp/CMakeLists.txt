cmake_minimum_required(VERSION 3.18)
project(nccl_lens LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)

add_executable(nccl_lens
  cpp/main.cu
  cpp/config.cpp
  cpp/benchmark.cpp
  cpp/telemetry.cpp
  cpp/mpi.cpp
  cpp/nccl.cpp
  cpp/buffers.cpp
  cpp/devices.cpp
)

target_link_libraries(bench_allreduce PRIVATE MPI::MPI_CXX nccl cuda cudart)
set_target_properties(bench_allreduce PROPERTIES CUDA_STANDARD 17 CXX_STANDARD 17)
