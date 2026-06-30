#pragma once

struct MpiContext {
  int rank = 0;
  int world = 1;
};

MpiContext initMPI(int argc, char** argv);