#include <common.h>
#include "devices.h"
#include <stdexcept>
#include <iostream>

DeviceInfo initDeviceForRank(int rank) {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if(device_count <= 0) {
        throw std::runtime_error("No CUDA devices found");
    }

    int device_id = rank % device_count;
    CHECK_CUDA(cudaSetDevice(device_id));

    return DeviceInfo{
        .device_id = device_id;
        .device_count = device_count;
    };
}
