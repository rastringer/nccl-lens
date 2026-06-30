#include <config.h>

#include <cstdlib>
#include <iostream>
#include <string>

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