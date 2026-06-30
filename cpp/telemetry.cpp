#include "telemetry.h"
#include <common.h>

#include <cstdio>

void emitAllReduceTrace(
    int rank,
    int iter,
    bool warmup,
    std::size_t bytes,
    float gpu_ms
) {
    std::printf(
        "{\"rank\":%d,\"iter\":%d,\"warmup\":%s,\"op\":\"all_reduce\",\"bytes\":%zu,\"gpu_ms\":%.3f}\n",
        rank,
        iter,
        warmup ? "true" : "false",
        bytes,
        gpu_ms
    );
}