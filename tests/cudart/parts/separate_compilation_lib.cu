// Second translation unit of tests/cudart/separate_compilation.cu
#include <cstdio>

__device__ int g_launches = 100;
__constant__ int c_table[4] = {7, 11, 13, 17};

__device__ int polynomial(int x) {
    return 3 * x * x + c_table[x & 3];
}

__device__ int combine(int a, int b) {
    return a * 1000 + b;
}

__global__ void libKernel(int *out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = combine(polynomial(i), i);
    }
    if (i == 0) {
        atomicAdd(&g_launches, 1);
        printf("libKernel says hello from the second translation unit\n");
    }
}

void launchLibKernel(int *out, int n) {
    libKernel<<<(n + 127) / 128, 128>>>(out, n);
}
