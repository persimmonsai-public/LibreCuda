// Shared helpers for the CUDA runtime test programs
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t check_err = (call); \
    if (check_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s (%s) at %s:%d: %s\n", cudaGetErrorName(check_err), \
                cudaGetErrorString(check_err), __FILE__, __LINE__, #call); \
        exit(1); \
    } \
} while (0)

#define EXPECT(cond, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAILED %s:%d: %s: ", __FILE__, __LINE__, #cond); \
        fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); \
        exit(1); \
    } \
} while (0)

static inline bool nearlyEqual(double a, double b, double rel = 1e-4, double abs_tol = 1e-5) {
    return std::fabs(a - b) <= abs_tol + rel * std::fabs(b);
}

#define PASS(name) printf("PASS %s\n", name)
