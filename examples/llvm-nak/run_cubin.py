#!/usr/bin/env python3
"""
Loads a cubin with LibreCUDA's driver API (libcuda.so.1 from the LibreCUDA build, through ctypes), launches one of
the example kernels and checks its output against numpy.

usage: run_cubin.py <lib dir> <cubin> <kernel>
  kernel: vector_add | saxpy | reduce_sum | stencil

The lib dir is LibreCUDA's build/lib (it holds libcuda.so.1 and libcudart.so.13).
"""
import ctypes
import sys

import numpy as np


def main():
    lib_dir, cubin_path, kernel = sys.argv[1:4]
    cudart = ctypes.CDLL(f"{lib_dir}/libcudart.so.13", mode=ctypes.RTLD_GLOBAL)  # (libcuda.so.1 runs on it)
    cuda = ctypes.CDLL(f"{lib_dir}/libcuda.so.1")
    del cudart

    def check(err, what):
        if err != 0:
            name = ctypes.c_char_p()
            cuda.cuGetErrorName(err, ctypes.byref(name))
            sys.exit(f"{what}: {name.value.decode() if name.value else err}")

    check(cuda.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    ctx = ctypes.c_void_p()
    check(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
    check(cuda.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

    image = open(cubin_path, "rb").read()
    module = ctypes.c_void_p()
    check(cuda.cuModuleLoadData(ctypes.byref(module), ctypes.c_char_p(image)), "cuModuleLoadData")
    function = ctypes.c_void_p()
    check(cuda.cuModuleGetFunction(ctypes.byref(function), module, kernel.encode()), "cuModuleGetFunction")

    buffers = []

    def upload(array):
        ptr = ctypes.c_uint64()
        check(cuda.cuMemAlloc_v2(ctypes.byref(ptr), array.nbytes), "cuMemAlloc")
        check(cuda.cuMemcpyHtoD_v2(ptr, array.ctypes.data_as(ctypes.c_void_p), array.nbytes), "cuMemcpyHtoD")
        buffers.append(ptr)
        return ptr

    def download(ptr, like):
        out = np.empty_like(like)
        check(cuda.cuMemcpyDtoH_v2(out.ctypes.data_as(ctypes.c_void_p), ptr, out.nbytes), "cuMemcpyDtoH")
        return out

    rng = np.random.default_rng(1)
    n = 1 << 20
    block = 256
    grid = (n + block - 1) // block
    if kernel == "vector_add":
        a = rng.integers(0, 1 << 20, n, dtype=np.uint32)
        b = rng.integers(0, 1 << 20, n, dtype=np.uint32)
        out = upload(np.zeros(n, np.uint32))
        args = [out, upload(a), upload(b), ctypes.c_uint32(n)]
        expected, result_like = a + b, a
    elif kernel == "saxpy":
        x = rng.standard_normal(n).astype(np.float32)
        y = rng.standard_normal(n).astype(np.float32)
        out = upload(y)
        args = [out, upload(x), ctypes.c_float(2.5), ctypes.c_int32(n)]
        expected, result_like = np.float32(2.5) * x + y, y
    elif kernel == "reduce_sum":
        # one partial sum per block (shared memory tree), summed on the host
        x = rng.integers(0, 100, n, dtype=np.int32)
        out = upload(np.zeros(grid, np.int32))
        args = [out, upload(x), ctypes.c_int32(n)]
        expected, result_like = None, np.zeros(grid, np.int32)
    elif kernel == "stencil":
        x = rng.standard_normal(n).astype(np.float32)
        out = upload(np.zeros(n, np.float32))
        args = [out, upload(x), ctypes.c_int32(n)]
        expected = x.copy()
        expected[1:-1] = np.float32(0.25) * x[:-2] + np.float32(0.5) * x[1:-1] + np.float32(0.25) * x[2:]
        result_like = x
    else:
        sys.exit(f"unknown kernel {kernel}")

    params = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    check(cuda.cuLaunchKernel(function, grid, 1, 1, block, 1, 1, 0, None, params, None), "cuLaunchKernel")
    check(cuda.cuCtxSynchronize(), "cuCtxSynchronize")
    result = download(out, result_like)

    if kernel == "reduce_sum":
        ok = int(result.astype(np.int64).sum()) == int(x.astype(np.int64).sum())
        detail = f"sum {int(result.astype(np.int64).sum())} (expected {int(x.astype(np.int64).sum())})"
    elif result.dtype == np.float32:
        err = float(np.max(np.abs(result - expected)))
        ok, detail = err <= 1e-5, f"max error {err:.2e}"
    else:
        bad = int(np.count_nonzero(result != expected))
        ok, detail = bad == 0, f"{bad} wrong elements"
    for ptr in buffers:
        cuda.cuMemFree_v2(ptr)
    cuda.cuModuleUnload(module)
    print(f"{kernel}: {'PASS' if ok else 'FAIL'} ({n} elements, {detail})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
