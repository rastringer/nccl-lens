#pragma once

#include <iostream>
#include <config.h>

struct BufferContext {
  float* send = nullptr;
  float* recv = nullptr;
  size_t elements = 0;
  size_t bytes = 0;
};

BufferContext allocateBuffers(const Config& cfg);