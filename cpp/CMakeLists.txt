cmake_minimum_required(VERSION 3.18)
project(nccl_lens LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)

add_executable(bench_allreduce bench_allreduce.cu)
target_link_libraries(bench_allreduce PRIVATE MPI::MPI_CXX nccl cuda cudart)
set_target_properties(bench_allreduce PROPERTIES CUDA_STANDARD 17 CXX_STANDARD 17)
