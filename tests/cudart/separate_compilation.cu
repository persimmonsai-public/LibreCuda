// NVCC_FLAGS: -arch=$ARCH -rdc=true
// EXTRA_SOURCES: parts/separate_compilation_lib.cu
// Separate compilation (-rdc=true): device functions, __device__ and __constant__ variables shared between translation
// units, kernels in both, and device printf from the second one. nvcc's device link step registers the linked image;
// calls across translation units are left as relocations (R_CUDA_ABS55_16_34) for the loader.
#include "check.h"
#include <vector>

extern __device__ int g_launches;
extern __constant__ int c_table[4];

__device__ int polynomial(int x);
__device__ int combine(int a, int b);

void launchLibKernel(int *out, int n);

__global__ void mainKernel(int *out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = combine(polynomial(i) + c_table[(i + 1) & 3], -i);
    }
    if (i == 0) {
        atomicAdd(&g_launches, 1);
    }
}

static int polynomialHost(int x, const int *table) {
    return 3 * x * x + table[x & 3];
}

int main() {
    const int n = 1000;
    int *d_main, *d_lib;
    CHECK(cudaMalloc(&d_main, n * sizeof(int)));
    CHECK(cudaMalloc(&d_lib, n * sizeof(int)));

    mainKernel<<<(n + 127) / 128, 128>>>(d_main, n);
    CHECK(cudaGetLastError());
    launchLibKernel(d_lib, n);
    CHECK(cudaGetLastError());

    std::vector<int> main_out(n), lib_out(n);
    CHECK(cudaMemcpy(main_out.data(), d_main, n * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(lib_out.data(), d_lib, n * sizeof(int), cudaMemcpyDeviceToHost));
    const int table[4] = {7, 11, 13, 17};
    for (int i = 0; i < n; i++) {
        const int expected_main = (polynomialHost(i, table) + table[(i + 1) & 3]) * 1000 - i;
        const int expected_lib = polynomialHost(i, table) * 1000 + i;
        EXPECT(main_out[i] == expected_main, "main kernel [%d] = %d, expected %d", i, main_out[i], expected_main);
        EXPECT(lib_out[i] == expected_lib, "lib kernel [%d] = %d, expected %d", i, lib_out[i], expected_lib);
    }

    // the variables of the other translation unit, through the symbol APIs
    int launches = 0;
    CHECK(cudaMemcpyFromSymbol(&launches, g_launches, sizeof(launches)));
    EXPECT(launches == 102, "g_launches is %d, expected 102", launches);
    const int new_table[4] = {1, 2, 3, 4};
    CHECK(cudaMemcpyToSymbol(c_table, new_table, sizeof(new_table)));
    launchLibKernel(d_lib, n);
    CHECK(cudaMemcpy(lib_out.data(), d_lib, n * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) {
        const int expected = polynomialHost(i, new_table) * 1000 + i;
        EXPECT(lib_out[i] == expected, "lib kernel with the new table [%d] = %d, expected %d", i, lib_out[i], expected);
    }

    CHECK(cudaFree(d_main));
    CHECK(cudaFree(d_lib));
    PASS("separate_compilation");
    return 0;
}
