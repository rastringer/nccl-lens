#pragma once 

struct Config {
  int iters = 100;
  int warmup = 5;
  size_t bytes = 16 * 1024 * 1024;
  int slow_rank = -1;      // -1 disables synthetic delay
  int slow_every = 10;     // inject delay every N measured iterations
  int slow_us = 5000;      // delay duration in microseconds
};

Config parseArgs(int argc, char** argv);