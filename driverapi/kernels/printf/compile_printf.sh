#!/bin/bash
# Compiles the device side printf implementation for every supported architecture and embeds the cubins in
# ../../internal/printf_kernels.h. Like compile_memcpy.sh this is not part of the build: it needs nvcc.
#
# -rdc=true: kernels call vprintf with CALL.ABS and an absolute return address; only relocatable compilation makes
#            ptxas end the function with the matching RET.ABS.
# -maxrregcount=24: librecuda_vprintf runs in the calling kernel's registers, and ptxas gives kernels that call vprintf
#            at least 24.
set -e
cd "$(dirname "$0")"
nvcc="${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}"
mkdir -p output
for arch in sm_75 sm_80 sm_86 sm_87 sm_89 sm_90 sm_120; do
    "$nvcc" -cubin -rdc=true -arch="$arch" -maxrregcount=24 printf.cu -o "output/printf_${arch}.cubin"
    echo "compiled $arch"
done
python3 generate_header.py ../../internal/printf_kernels.h output
