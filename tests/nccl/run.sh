#!/bin/bash
# Builds and runs the NCCL device API test (two GPUs, CUDA_VISIBLE_DEVICES) with LibreCUDA's NCCL on LibreCUDA's
# runtime, with LibreCUDA's NCCL on NVIDIA's runtime, and with NVIDIA's NCCL (the reference), as available.
#
# usage: run.sh [build dir]     (default ../../build; NCCL_INCLUDE_DIR / NCCL_LIB_DIR: NVIDIA's NCCL, default: the
#                                nvidia-nccl pip package)
set -u
here="$(cd "$(dirname "$0")" && pwd)"
build="$(cd "${1:-$here/../../build}" && pwd)"
nvcc="${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}"
nccl_pkg="$(python3 -c 'import nvidia.nccl, os; print(os.path.dirname(nvidia.nccl.__file__))' 2>/dev/null ||
            python3 -c 'import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], "nccl"))')"
include="${NCCL_INCLUDE_DIR:-$nccl_pkg/include}"
nvidia_lib="${NCCL_LIB_DIR:-$nccl_pkg/lib}"
out="$build/tests/nccl"
mkdir -p "$out/nvidia-runtime"
"$nvcc" -O2 -std=c++17 -arch="${ARCH:-sm_120}" -cudart=shared -I"$include" "$here/device_api.cu" \
    -L"$build/lib" -lnccl -o "$out/device_api" || exit 1
# LibreCUDA's NCCL alone next to NVIDIA's runtime
ln -sf "$build/lib/libnccl.so.2" "$out/nvidia-runtime/libnccl.so.2"

failed=0
run() {
    echo "== $1"
    shift
    if ! LIBRECUDA_WAIT_TIMEOUT_MS="${LIBRECUDA_WAIT_TIMEOUT_MS:-60000}" timeout 300 env "$@" "$out/device_api"; then
        echo "FAILED"
        failed=$((failed + 1))
    fi
}
run "LibreCUDA's NCCL, LibreCUDA's runtime" LD_LIBRARY_PATH="$build/lib"
run "LibreCUDA's NCCL, NVIDIA's runtime" LD_LIBRARY_PATH="$out/nvidia-runtime"
[ -f "$nvidia_lib/libnccl.so.2" ] && run "NVIDIA's NCCL (reference)" LD_LIBRARY_PATH="$nvidia_lib" CUDA_MODULE_LOADING=EAGER
[ "$failed" = 0 ]
