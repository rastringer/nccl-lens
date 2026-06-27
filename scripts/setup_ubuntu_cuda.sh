#!/usr/bin/env bash
set -euo pipefail

echo "== nccl-lens setup =="

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Are GPUs visible in this container?"
  exit 1
fi

echo "== GPU info =="
nvidia-smi

echo "== Installing system deps =="
apt update
apt install -y \
  build-essential \
  cmake \
  git \
  python3 \
  python3-pip \
  openmpi-bin \
  libopenmpi-dev

echo "== Checking CUDA =="
if ! command -v nvcc >/dev/null 2>&1; then
  echo "WARNING: nvcc not found. You may be in a runtime-only CUDA image."
else
  nvcc --version
fi

echo "== Checking NCCL =="
if ! ldconfig -p | grep -q libnccl; then
  echo "WARNING: libnccl not found via ldconfig."
  echo "Try: find /usr -name 'libnccl.so*' 2>/dev/null"
else
  ldconfig -p | grep libnccl
fi

echo "== Installing Python deps =="
pip3 install --break-system-packages pandas numpy matplotlib || \
pip3 install pandas numpy matplotlib

echo "== Creating local dirs =="
mkdir -p build traces reports plots

echo "== Building =="
cmake -S cpp -B build
cmake --build build -j"$(nproc)"

echo "== Done =="
echo
echo "Smoke test:"
echo "NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 NCCL_DEBUG=WARN \\"
echo "mpirun --allow-run-as-root -np 2 ./build/bench_allreduce --iters 10 --warmup 2"
