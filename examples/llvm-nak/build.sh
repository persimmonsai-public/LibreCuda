#!/bin/bash
# Prototype: CUDA C -> LLVM (clang, HIP mode) -> SPIR-V -> Mesa (spirv_to_nir) -> NAK -> SM_120 cubin -> LibreCUDA,
# with no NVIDIA compiler and no reverse-engineered encodings. Run setup.sh first.
#
# usage: build.sh [--no-run] [source.cu]       (default: kernels.cu)
#   CLANG      clang with the SPIR-V backend (LLVM 21 tested; opt, llc, llvm-dis next to it)   default: clang
#   WORK       setup.sh's work directory                                                       default: ./work
#   LIBRECUDA  LibreCUDA's build/lib                                                           default: ../../build/lib
#   OUT        output directory                                                                default: ./out
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
run=1
src="$here/kernels.cu"
while [ $# -gt 0 ]; do
    case "$1" in
        --no-run) run=0; shift ;;
        *) src="$1"; shift ;;
    esac
done
clang="${CLANG:-clang}"
bin="$(dirname "$(command -v "$clang")")"
work="${WORK:-$here/work}"
lib="${LIBRECUDA:-$here/../../build/lib}"
out="${OUT:-$here/out}"
mkdir -p "$out"

# 1. CUDA C -> LLVM IR for SPIR-V (HIP mode; prelude_spirv.h maps threadIdx & co. to OpenCL built-ins)
"$clang" -x hip --offload=spirv64 --offload-device-only -nogpuinc -nogpulib -fuse-cuid=none \
    -include "$here/prelude_spirv.h" -O2 -c -emit-llvm "$src" -o "$out/module.bc"
# 2. drop HIP's compilation unit id variable (a global with an initializer, which Mesa's SPIR-V translator rejects),
#    resolve generic pointers where possible, and emit SPIR-V
"$bin/llvm-dis" "$out/module.bc" -o - | grep -vE '^@__hip_cuid_|^@llvm\.compiler\.used' > "$out/module.ll"
"$bin/opt" -passes='globaldce,infer-address-spaces,instcombine,simplifycfg' "$out/module.ll" -o "$out/module.opt.bc"
"$bin/llc" -mtriple=spirv64-unknown-unknown -O2 -filetype=obj "$out/module.opt.bc" -o "$out/module.spv"
# 3. SPIR-V -> NIR -> NAK -> one cubin per kernel
tinymesa="$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib", "posix_user"))')/tinymesa/libtinymesa.so"
mapfile -t kernels < <(PYTHONPATH="$work/tinygrad" MESA_PATH="$tinymesa" \
    python3 "$here/nak_compile.py" "$out/module.spv" "$out" "$work/libspirv_to_nir.so" | tee /dev/stderr | cut -d: -f1)

# 4. run on LibreCUDA
[ "$run" = 1 ] || exit 0
failed=0
for k in "${kernels[@]}"; do
    LIBRECUDA_WAIT_TIMEOUT_MS="${LIBRECUDA_WAIT_TIMEOUT_MS:-20000}" timeout 120 \
        python3 "$here/run_cubin.py" "$lib" "$out/$k.cubin" "$k" || failed=$((failed + 1))
done
echo "nak: $((${#kernels[@]} - failed)) of ${#kernels[@]} kernels passed"
[ "$failed" = 0 ]
