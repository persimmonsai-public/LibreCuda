// __constant__ memory (incl. a full 64 KiB bank), initialized and zero-initialized __device__ globals,
// cudaMemcpyToSymbol / FromSymbol / GetSymbolAddress, device function pointers and virtual functions
#include "check.h"
#include <vector>

__constant__ float c_coeffs[16];
__constant__ int c_table[16000]; // 64000 bytes
__device__ int g_counter;                       // zero initialized (.nv.global)
__device__ float g_init[4] = {1.5f, 2.5f, 3.5f, 4.5f}; // initialized (.nv.global.init)
__device__ unsigned long long g_big[1 << 16];   // 512 KiB of zero initialized globals

typedef int (*op_t)(int, int);
__device__ int opAdd(int a, int b) { return a + b; }
__device__ int opMul(int a, int b) { return a * b; }
__device__ int opSub(int a, int b) { return a - b; }
__device__ op_t g_ops[3] = {opAdd, opMul, opSub};

struct Shape {
    __device__ virtual float area() const = 0;
};
struct Square : Shape {
    float s;
    __device__ explicit Square(float s) : s(s) {}
    __device__ float area() const override { return s * s; }
};
struct Circle : Shape {
    float r;
    __device__ explicit Circle(float r) : r(r) {}
    __device__ float area() const override { return 3.0f * r * r; }
};

__global__ void useConstants(float *out, int *tableOut, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float acc = 0;
        for (int k = 0; k < 16; k++) acc += c_coeffs[k] * (i + k);
        out[i] = acc + g_init[i % 4];
        tableOut[i] = c_table[(i * 7) % 16000];
        atomicAdd(&g_counter, 1);
        atomicAdd(&g_big[i % (1 << 16)], 1ull);
    }
}

__global__ void useFunctionPointers(int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        op_t f = g_ops[i % 3];
        Square sq(i * 0.5f);
        Circle ci(i * 0.25f);
        const Shape *shape = (i % 2) ? static_cast<const Shape *>(&sq) : static_cast<const Shape *>(&ci);
        out[i] = f(i, 3) + (int) (shape->area() * 1e-3f);
    }
}

int main() {
    const int n = 100000;
    float coeffs[16];
    for (int k = 0; k < 16; k++) coeffs[k] = 0.25f * k - 1.0f;
    std::vector<int> table(16000);
    for (int k = 0; k < 16000; k++) table[k] = k * k - 12345;
    CHECK(cudaMemcpyToSymbol(c_coeffs, coeffs, sizeof(coeffs)));
    CHECK(cudaMemcpyToSymbol(c_table, table.data(), table.size() * sizeof(int)));

    float *d_out; int *d_table, *d_fp;
    CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CHECK(cudaMalloc(&d_table, n * sizeof(int)));
    CHECK(cudaMalloc(&d_fp, n * sizeof(int)));
    useConstants<<<(n + 255) / 256, 256>>>(d_out, d_table, n);
    CHECK(cudaGetLastError());
    useFunctionPointers<<<(n + 127) / 128, 128>>>(d_fp, n);
    CHECK(cudaGetLastError());

    std::vector<float> out(n);
    std::vector<int> table_out(n), fp(n);
    CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(table_out.data(), d_table, n * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(fp.data(), d_fp, n * sizeof(int), cudaMemcpyDeviceToHost));
    const float init[4] = {1.5f, 2.5f, 3.5f, 4.5f};
    for (int i = 0; i < n; i++) {
        float acc = 0;
        for (int k = 0; k < 16; k++) acc += coeffs[k] * (i + k);
        EXPECT(nearlyEqual(out[i], acc + init[i % 4]), "out[%d] = %f, expected %f", i, out[i], acc + init[i % 4]);
        EXPECT(table_out[i] == table[(i * 7) % 16000], "table[%d]", i);
        int f = i % 3 == 0 ? i + 3 : i % 3 == 1 ? i * 3 : i - 3;
        int area = (i % 2) ? (int) (i * 0.5f * (i * 0.5f) * 1e-3f) : (int) (3.0f * (i * 0.25f) * (i * 0.25f) * 1e-3f);
        EXPECT(fp[i] == f + area, "fp[%d] = %d, expected %d", i, fp[i], f + area);
    }

    int counter = 0;
    CHECK(cudaMemcpyFromSymbol(&counter, g_counter, sizeof(int)));
    EXPECT(counter == n, "g_counter = %d", counter);
    std::vector<unsigned long long> big(1 << 16);
    CHECK(cudaMemcpyFromSymbol(big.data(), g_big, big.size() * 8));
    for (int k = 0; k < (1 << 16); k++) {
        unsigned long long expect = n / (1 << 16) + (k < n % (1 << 16) ? 1 : 0);
        EXPECT(big[k] == expect, "g_big[%d] = %llu, expected %llu", k, big[k], expect);
    }

    // symbol address & size; write through the address
    int *counter_ptr = nullptr;
    size_t size = 0;
    CHECK(cudaGetSymbolAddress((void **) &counter_ptr, g_counter));
    CHECK(cudaGetSymbolSize(&size, g_counter));
    EXPECT(size == sizeof(int), "symbol size %zu", size);
    CHECK(cudaMemset(counter_ptr, 0, sizeof(int)));
    CHECK(cudaMemcpyFromSymbol(&counter, g_counter, sizeof(int)));
    EXPECT(counter == 0, "g_counter after memset = %d", counter);

    // partial update with an offset
    float two = 2.0f;
    CHECK(cudaMemcpyToSymbol(g_init, &two, sizeof(float), 2 * sizeof(float)));
    float init_now[4];
    CHECK(cudaMemcpyFromSymbol(init_now, g_init, sizeof(init_now)));
    EXPECT(init_now[0] == 1.5f && init_now[2] == 2.0f && init_now[3] == 4.5f, "g_init = %f %f %f %f",
           init_now[0], init_now[1], init_now[2], init_now[3]);
    PASS("globals_constants");
    return 0;
}
