#!/bin/bash
# Compiles the device side runtime (printf, assert) for every supported architecture and embeds the cubins in
# ../../internal/device_runtime_kernels.h. Like compile_memcpy.sh this is not part of the build: it needs nvcc.
#
# -rdc=true: kernels call vprintf / __assertfail with CALL.ABS and an absolute return address; only relocatable
#            compilation makes ptxas end the functions with the matching RET.ABS.
# -maxrregcount=24: the functions run in the calling kernel's registers, and ptxas gives kernels that call vprintf
#            or __assertfail at least 24.
set -e
cd "$(dirname "$0")"
nvcc="${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}"
mkdir -p output
for arch in sm_75 sm_80 sm_86 sm_87 sm_89 sm_90 sm_120; do
    "$nvcc" -cubin -rdc=true -arch="$arch" -maxrregcount=24 device_runtime.cu -o "output/device_runtime_${arch}.cubin"
    echo "compiled $arch"
done
python3 generate_header.py ../../internal/device_runtime_kernels.h output
