#pragma once

struct DeviceInfo {
  int device_id;
  int device_count;
};

DeviceInfo initDeviceForRank(int rank);