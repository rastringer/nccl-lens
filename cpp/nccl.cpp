#include <common.h>

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

void destroyNCCL(NcclContext&);
