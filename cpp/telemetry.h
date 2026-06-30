#pragma once

#include <cstddef>

void emitAllReduceTrace(
    int rank,
    int iter,
    bool warmup,
    std::size_t bytes,
    float gpu_ms
);