// The fat binary only contains PTX (no SASS): the runtime compiles it for the GPU with the toolkit's ptxas
// NVCC_FLAGS: -gencode arch=compute_120,code=compute_120
#include "check.h"
#include <vector>

__global__ void polynomial(const double *x, double *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double v = x[i];
        y[i] = ((3.0 * v - 2.0) * v + 0.5) * v - sin(v) + exp(-v * v);
    }
}

int main() {
    const int n = 100000;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; i++) x[i] = i * 1e-4 - 5.0;
    double *dx, *dy;
    CHECK(cudaMalloc(&dx, n * 8));
    CHECK(cudaMalloc(&dy, n * 8));
    CHECK(cudaMemcpy(dx, x.data(), n * 8, cudaMemcpyHostToDevice));
    polynomial<<<(n + 127) / 128, 128>>>(dx, dy, n);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(y.data(), dy, n * 8, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) {
        double v = x[i];
        double e = ((3.0 * v - 2.0) * v + 0.5) * v - sin(v) + exp(-v * v);
        EXPECT(nearlyEqual(y[i], e, 1e-12, 1e-12), "y[%d] = %.17g, expected %.17g", i, y[i], e);
    }
    PASS("ptx_jit");
    return 0;
}
