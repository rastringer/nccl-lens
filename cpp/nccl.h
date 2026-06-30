#pragma once

struct NcclContext {
  ncclComm_t comm = nullptr;
};

// Create one NCCL communicator shared across all MPI ranks.
NcclContext initNCCL(const MpiContext& mpi); 