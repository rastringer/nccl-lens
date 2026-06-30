#include "mpi.h"
#include "common.h"

// Initialize MPI, find the rank/world size of the process.
MpiContext initMPI(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  MpiContext ctx;
  MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ctx.world);

  return ctx;
}

  MPI_Finalize();