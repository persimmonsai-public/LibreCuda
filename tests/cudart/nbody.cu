// N-body simulation: float4 vector types, rsqrtf, shared memory tiling, __restrict__, many time steps with
// ping-pong buffers; compared against a double precision CPU simulation
#include "check.h"
#include <vector>
#include <random>

constexpr float SOFTENING = 1e-3f;

__global__ void integrate(const float4 *__restrict__ posIn, float4 *__restrict__ posOut,
                          const float4 *__restrict__ velIn, float4 *__restrict__ velOut, int n, float dt) {
    extern __shared__ float4 tile[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float4 p = i < n ? posIn[i] : make_float4(0, 0, 0, 0);
    float3 acc = make_float3(0, 0, 0);
    for (int base = 0; base < n; base += blockDim.x) {
        int j = base + threadIdx.x;
        tile[threadIdx.x] = j < n ? posIn[j] : make_float4(0, 0, 0, 0);
        __syncthreads();
#pragma unroll 8
        for (int k = 0; k < blockDim.x; k++) {
            float4 q = tile[k];
            float dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
            float inv = rsqrtf(dx * dx + dy * dy + dz * dz + SOFTENING);
            float s = q.w * inv * inv * inv;
            acc.x += dx * s; acc.y += dy * s; acc.z += dz * s;
        }
        __syncthreads();
    }
    if (i < n) {
        float4 v = velIn[i];
        v.x += acc.x * dt; v.y += acc.y * dt; v.z += acc.z * dt;
        velOut[i] = v;
        posOut[i] = make_float4(p.x + v.x * dt, p.y + v.y * dt, p.z + v.z * dt, p.w);
    }
}

int main() {
    const int n = 4096, steps = 10, block = 256;
    const float dt = 0.001f;
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> u(-1, 1);
    std::vector<float4> pos(n), vel(n);
    for (int i = 0; i < n; i++) {
        pos[i] = make_float4(u(rng), u(rng), u(rng), 0.5f + 0.5f * (u(rng) + 1));
        vel[i] = make_float4(u(rng) * 0.1f, u(rng) * 0.1f, u(rng) * 0.1f, 0);
    }
    float4 *dp[2], *dv[2];
    for (int k = 0; k < 2; k++) { CHECK(cudaMalloc(&dp[k], n * 16)); CHECK(cudaMalloc(&dv[k], n * 16)); }
    CHECK(cudaMemcpy(dp[0], pos.data(), n * 16, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dv[0], vel.data(), n * 16, cudaMemcpyHostToDevice));
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0));
    for (int s = 0; s < steps; s++) {
        integrate<<<(n + block - 1) / block, block, block * sizeof(float4)>>>(dp[s & 1], dp[(s + 1) & 1], dv[s & 1],
                                                                           dv[(s + 1) & 1], n, dt);
    }
    CHECK(cudaEventRecord(t1));
    CHECK(cudaGetLastError());
    CHECK(cudaEventSynchronize(t1));
    float ms;
    CHECK(cudaEventElapsedTime(&ms, t0, t1));
    std::vector<float4> gpu(n);
    CHECK(cudaMemcpy(gpu.data(), dp[steps & 1], n * 16, cudaMemcpyDeviceToHost));

    // CPU reference in double precision
    std::vector<double> px(n), py(n), pz(n), m(n), vx(n), vy(n), vz(n);
    for (int i = 0; i < n; i++) {
        px[i] = pos[i].x; py[i] = pos[i].y; pz[i] = pos[i].z; m[i] = pos[i].w;
        vx[i] = vel[i].x; vy[i] = vel[i].y; vz[i] = vel[i].z;
    }
    for (int s = 0; s < steps; s++) {
        std::vector<double> nx(n), ny(n), nz(n);
        for (int i = 0; i < n; i++) {
            double ax = 0, ay = 0, az = 0;
            for (int j = 0; j < n; j++) {
                double dx = px[j] - px[i], dy = py[j] - py[i], dz = pz[j] - pz[i];
                double inv = 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                double sc = m[j] * inv * inv * inv;
                ax += dx * sc; ay += dy * sc; az += dz * sc;
            }
            vx[i] += ax * dt; vy[i] += ay * dt; vz[i] += az * dt;
            nx[i] = px[i] + vx[i] * dt; ny[i] = py[i] + vy[i] * dt; nz[i] = pz[i] + vz[i] * dt;
        }
        px = nx; py = ny; pz = nz;
    }
    double max_err = 0;
    for (int i = 0; i < n; i++) {
        max_err = std::max(max_err, std::fabs(gpu[i].x - px[i]));
        max_err = std::max(max_err, std::fabs(gpu[i].y - py[i]));
        max_err = std::max(max_err, std::fabs(gpu[i].z - pz[i]));
    }
    printf("n-body %d bodies x %d steps: %.3f ms/step, %.1f GFLOP/s, max abs error %.2e\n", n, steps, ms / steps,
           20.0 * n * n * steps / (ms * 1e6), max_err);
    EXPECT(max_err < 1e-3, "max error %e", max_err);
    PASS("nbody");
    return 0;
}
