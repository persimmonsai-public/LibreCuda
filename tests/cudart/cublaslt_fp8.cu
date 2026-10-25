// NVCC_FLAGS: -arch=$ARCH -lcublasLt
// cuBLASLt matmuls of 8 bit floats as torch._scaled_mm makes them: e4m3 / e5m2 A (transposed) and B, per-tensor scales
// of A and B, bf16 / fp16 / float / 8 bit float D, a bias, the scale of D and its absolute maximum, a C pointer that
// isn't read (beta 0, C's type the bias'), device alpha and beta, and capture into a CUDA graph. Compared with a double
// precision reference of the dequantized operands. Checked against NVIDIA's cuBLASLt too.
#include "check.h"
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#define BCHECK(call) do { \
    cublasStatus_t bcheck_status = (call); \
    if (bcheck_status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d at %s:%d: %s\n", int(bcheck_status), __FILE__, __LINE__, #call); \
        exit(1); \
    } \
} while (0)

static double hostValue(size_t i, int salt) {
    return ((i * 2654435761u + salt * 40503u) % 2001) / 1000.0 - 1.0;
}

// an element of the given type stored at p (as the float it holds) / stores v there
static double load(const uint8_t *p, cudaDataType type) {
    switch (type) {
        case CUDA_R_8F_E4M3: { __nv_fp8_e4m3 v; memcpy(&v, p, 1); return float(v); }
        case CUDA_R_8F_E5M2: { __nv_fp8_e5m2 v; memcpy(&v, p, 1); return float(v); }
        case CUDA_R_16F: { __half v; memcpy(&v, p, 2); return __half2float(v); }
        case CUDA_R_16BF: { __nv_bfloat16 v; memcpy(&v, p, 2); return __bfloat162float(v); }
        default: { float v; memcpy(&v, p, 4); return v; }
    }
}

static void store(uint8_t *p, cudaDataType type, float value) {
    switch (type) {
        case CUDA_R_8F_E4M3: { __nv_fp8_e4m3 v(value); memcpy(p, &v, 1); break; }
        case CUDA_R_8F_E5M2: { __nv_fp8_e5m2 v(value); memcpy(p, &v, 1); break; }
        case CUDA_R_16F: { __half v = __float2half(value); memcpy(p, &v, 2); break; }
        case CUDA_R_16BF: { __nv_bfloat16 v = __float2bfloat16(value); memcpy(p, &v, 2); break; }
        default: memcpy(p, &value, 4);
    }
}

static size_t sizeOf(cudaDataType type) {
    return type == CUDA_R_8F_E4M3 || type == CUDA_R_8F_E5M2 ? 1 : type == CUDA_R_32F ? 4 : 2;
}

// relative precision of a type (half its spacing)
static double precision(cudaDataType type) {
    switch (type) {
        case CUDA_R_8F_E4M3: return 1.0 / 16;
        case CUDA_R_8F_E5M2: return 1.0 / 8;
        case CUDA_R_16F: return 1.0 / 2048;
        case CUDA_R_16BF: return 1.0 / 256;
        default: return 1e-6;
    }
}

struct Matrix {
    cudaDataType type;
    std::vector<uint8_t> host;
    void *device = nullptr;

    Matrix(cudaDataType t, size_t count, int salt, double range) : type(t), host(count * sizeOf(t)) {
        for (size_t i = 0; i < count; i++) store(&host[i * sizeOf(t)], t, float(hostValue(i, salt) * range));
        CHECK(cudaMalloc(&device, host.size()));
        CHECK(cudaMemcpy(device, host.data(), host.size(), cudaMemcpyHostToDevice));
    }
    ~Matrix() { cudaFree(device); }
    double operator[](size_t i) const { return load(&host[i * sizeOf(type)], type); }
};

struct Case {
    const char *name;
    cudaDataType a, b, d;
    int m, n, k;
    bool bias, scaleD, amax, deviceScalars, graph;
    float alpha;
    // the heuristic refuses it (as NVIDIA's cuBLASLt)
    bool refused = false;
    // the operations on A and B (torch._scaled_mm's layout: T and N)
    char opA = 'T', opB = 'N';
};

static float *deviceFloat(float value) {
    float *p;
    CHECK(cudaMalloc(&p, sizeof(float)));
    CHECK(cudaMemcpy(p, &value, sizeof(float), cudaMemcpyHostToDevice));
    return p;
}

static void run(cublasLtHandle_t lt, cudaStream_t stream, const Case &c) {
    const int m = c.m, n = c.n, k = c.k;
    // A^T stored k x m (op T) or A m x k (op N), B k x n (op N) or B^T n x k (op T), D m x n
    const Matrix a(c.a, size_t(k) * m, 1, 3.0), b(c.b, size_t(k) * n, 2, 3.0);
    const bool ta = c.opA == 'T', tb = c.opB == 'T';
    // element (i, l) of op(A), (l, j) of op(B)
    auto a_at = [&](int i, int l) { return ta ? a[size_t(i) * k + l] : a[i + size_t(l) * m]; };
    auto b_at = [&](int l, int j) { return tb ? b[j + size_t(l) * n] : b[size_t(j) * k + l]; };
    const cudaDataType bias_type = c.d == CUDA_R_16F ? CUDA_R_16F : CUDA_R_16BF;
    const Matrix bias(bias_type, m, 3, 4.0);
    const float scale_a = 0.75f, scale_b = 1.5f, scale_d = 0.125f;
    float *d_scale_a = deviceFloat(scale_a), *d_scale_b = deviceFloat(scale_b), *d_scale_d = deviceFloat(scale_d);
    float *d_amax = deviceFloat(-1), *d_alpha = deviceFloat(c.alpha), *d_beta = deviceFloat(0);
    void *dd;
    const size_t d_bytes = size_t(m) * n * sizeOf(c.d);
    CHECK(cudaMalloc(&dd, d_bytes));
    CHECK(cudaMemset(dd, 0xff, d_bytes));

    cublasLtMatmulDesc_t desc;
    BCHECK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t op_a = ta ? CUBLAS_OP_T : CUBLAS_OP_N, op_b = tb ? CUBLAS_OP_T : CUBLAS_OP_N;
    BCHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
    BCHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)));
    BCHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_a, sizeof(void *)));
    BCHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_b, sizeof(void *)));
    if (c.scaleD) {
        BCHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scale_d, sizeof(void *)));
    }
    if (c.amax) {
        BCHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &d_amax, sizeof(void *)));
    }
    const int8_t fast_accumulation = 1;
    BCHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accumulation, 1));
    const int32_t scalar_mode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    BCHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scalar_mode, sizeof(scalar_mode)));
    BCHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scalar_mode, sizeof(scalar_mode)));
    if (c.bias) {
        const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        const int32_t type = bias_type;
        BCHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias.device, sizeof(void *)));
        BCHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
        BCHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &type, sizeof(type)));
    }
    if (c.deviceScalars) {
        const cublasLtPointerMode_t mode = CUBLASLT_POINTER_MODE_DEVICE;
        BCHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &mode, sizeof(mode)));
    }
    cublasLtMatrixLayout_t la, lb, lc, ld;
    BCHECK(cublasLtMatrixLayoutCreate(&la, c.a, ta ? k : m, ta ? m : k, ta ? k : m));
    BCHECK(cublasLtMatrixLayoutCreate(&lb, c.b, tb ? n : k, tb ? k : n, tb ? n : k));
    // as PyTorch: C of the bias' type (half for 8 bit D), never read
    BCHECK(cublasLtMatrixLayoutCreate(&lc, c.bias ? bias_type : sizeOf(c.d) == 1 ? CUDA_R_16F : c.d, m, n, m));
    BCHECK(cublasLtMatrixLayoutCreate(&ld, c.d, m, n, m));
    cublasLtMatmulPreference_t pref;
    BCHECK(cublasLtMatmulPreferenceCreate(&pref));
    const uint64_t workspace = 32 << 20;
    BCHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace,
                                                sizeof(workspace)));
    cublasLtMatmulHeuristicResult_t result{};
    int count = 0;
    const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, lc, ld, pref, 1, &result, &count);
    // (NVIDIA's cuBLASLt 13.4 refuses B op T on sm_120; LibreCUDA's runs it)
    const bool skip = tb && status == CUBLAS_STATUS_NOT_SUPPORTED;
    if (c.refused) {
        EXPECT(status != CUBLAS_STATUS_SUCCESS && count == 0, "%s: not refused", c.name);
    } else if (skip) {
        printf("%s: not supported\n", c.name);
    } else {
        EXPECT(status == CUBLAS_STATUS_SUCCESS && count == 1, "%s: heuristic status %d, count %d", c.name,
               int(status), count);
    }
    if (!c.refused && !skip) {
        void *ws = nullptr;
        if (result.workspaceSize > 0) CHECK(cudaMalloc(&ws, result.workspaceSize));
        const float alpha = c.alpha, beta = 0;
        const void *alpha_ptr = c.deviceScalars ? static_cast<const void *>(d_alpha) : &alpha;
        const void *beta_ptr = c.deviceScalars ? static_cast<const void *>(d_beta) : &beta;
        auto matmul = [&] {
            BCHECK(cublasLtMatmul(lt, desc, alpha_ptr, a.device, la, b.device, lb, beta_ptr, a.device, lc, dd, ld,
                                  &result.algo, ws, result.workspaceSize, stream));
        };
        if (c.graph) {
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            matmul();
            CHECK(cudaStreamEndCapture(stream, &graph));
            CHECK(cudaGraphInstantiate(&exec, graph, 0));
            for (int i = 0; i < 2; i++) {
                CHECK(cudaMemsetAsync(dd, 0xff, d_bytes, stream));
                CHECK(cudaGraphLaunch(exec, stream));
            }
            CHECK(cudaGraphExecDestroy(exec));
            CHECK(cudaGraphDestroy(graph));
        } else {
            matmul();
        }
        CHECK(cudaStreamSynchronize(stream));
        std::vector<uint8_t> d(d_bytes);
        CHECK(cudaMemcpy(d.data(), dd, d_bytes, cudaMemcpyDeviceToHost));
        float amax = 0;
        CHECK(cudaMemcpy(&amax, d_amax, sizeof(float), cudaMemcpyDeviceToHost));

        // X = alpha scaleA scaleB A^T B + bias, D = scaleD X (8 bit D), amax = max |X|
        const bool d_scaled = c.scaleD;
        std::vector<double> x(size_t(m) * n);
        double ref_amax = 0, max_ref = 0;
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                double acc = 0;
                for (int l = 0; l < k; l++) acc += a_at(i, l) * b_at(l, j);
                double v = double(c.alpha) * scale_a * scale_b * acc + (c.bias ? bias[i] : 0.0);
                x[i + size_t(j) * m] = v;
                ref_amax = std::max(ref_amax, std::fabs(v));
                max_ref = std::max(max_ref, std::fabs(d_scaled ? v * scale_d : v));
            }
        }
        for (size_t i = 0; i < x.size(); i++) {
            const double ref = d_scaled ? x[i] * scale_d : x[i];
            const double got = load(&d[i * sizeOf(c.d)], c.d);
            // the rounding of D plus float accumulation (the bias is added in float, after rounding to its type)
            const double err =
                std::fabs(got - ref) / (precision(c.d) * (std::fabs(ref) + 0.01 * max_ref) + 1e-5 * max_ref);
            EXPECT(std::isfinite(got) && err <= 1.0, "%s: D[%zu] = %g, expected %g", c.name, i, got, ref);
        }
        if (c.amax) {
            EXPECT(std::fabs(amax - ref_amax) <= 1e-5 * ref_amax, "%s: amax %g, expected %g", c.name, amax, ref_amax);
        }
        if (ws) cudaFree(ws);
    }
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(la), cublasLtMatrixLayoutDestroy(lb), cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatrixLayoutDestroy(ld);
    cublasLtMatmulDescDestroy(desc);
    cudaFree(dd), cudaFree(d_scale_a), cudaFree(d_scale_b), cudaFree(d_scale_d), cudaFree(d_amax);
    cudaFree(d_alpha), cudaFree(d_beta);
}

int main() {
    cublasLtHandle_t lt;
    BCHECK(cublasLtCreate(&lt));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    const cudaDataType E4M3 = CUDA_R_8F_E4M3, E5M2 = CUDA_R_8F_E5M2, BF16 = CUDA_R_16BF, F16 = CUDA_R_16F,
                       F32 = CUDA_R_32F;
    // name, A, B, D, m, n, k, bias, scale of D, amax, device alpha / beta, graph, alpha, refused
    const Case cases[] = {
        {"e4m3 -> bf16", E4M3, E4M3, BF16, 256, 128, 512, false, false, false, false, false, 1},
        {"e4m3 -> bf16 + bias", E4M3, E4M3, BF16, 512, 96, 1024, true, false, false, false, false, 1},
        {"e4m3 -> fp16 + bias, alpha 0.5", E4M3, E4M3, F16, 128, 64, 256, true, false, false, false, false, 0.5f},
        {"e4m3 -> float", E4M3, E4M3, F32, 192, 80, 320, false, false, false, false, false, 1},
        {"e4m3 x e5m2 -> bf16", E4M3, E5M2, BF16, 256, 64, 512, false, false, false, false, false, 1},
        {"e5m2 x e4m3 -> fp16", E5M2, E4M3, F16, 256, 64, 512, false, false, false, false, false, 1},
        {"e4m3 -> e4m3, scale of D, amax", E4M3, E4M3, E4M3, 256, 128, 256, false, true, true, false, false, 1},
        {"e4m3 x e5m2 -> e5m2, scale of D", E4M3, E5M2, E5M2, 128, 64, 128, false, true, false, false, false, 1},
        {"e5m2 x e4m3 -> e4m3, scale of D, amax", E5M2, E4M3, E4M3, 256, 64, 512, false, true, true, false, false, 1},
        // (only 8 bit D has a scale and amax)
        {"e4m3 -> bf16, scale of D", E4M3, E4M3, BF16, 128, 32, 128, false, true, false, false, false, 1, true},
        {"e4m3 -> bf16, amax", E4M3, E4M3, BF16, 128, 32, 128, false, false, true, false, false, 1, true},
        // a few tokens (n), deep / shallow k (both tile sizes), large
        {"e4m3 -> bf16, n = 1", E4M3, E4M3, BF16, 512, 1, 512, true, false, false, false, false, 1},
        {"e4m3 -> bf16, n = 7", E4M3, E4M3, BF16, 1024, 7, 2048, false, false, false, false, false, 1},
        {"e4m3 -> bf16, k = 64", E4M3, E4M3, BF16, 384, 160, 64, false, false, false, false, false, 1},
        {"e4m3 -> bf16, 2048 x 256 x 2048", E4M3, E4M3, BF16, 2048, 256, 2048, true, false, false, false, false, 1},
        {"e4m3 -> bf16, device alpha", E4M3, E4M3, BF16, 256, 64, 256, false, false, false, true, false, 2},
        {"e4m3 -> bf16 + bias, graph", E4M3, E4M3, BF16, 256, 48, 512, true, false, false, false, true, 1},
        // the other layouts (supported on Blackwell; transposed into scratch memory)
        {"e4m3 -> bf16, A op N", E4M3, E4M3, BF16, 256, 64, 512, false, false, false, false, false, 1, false, 'N', 'N'},
        {"e4m3 -> bf16 + bias, B op T", E4M3, E4M3, BF16, 256, 64, 512, true, false, false, false, false, 1, false, 'T', 'T'},
        {"e4m3 -> bf16, A op N, B op T", E4M3, E4M3, BF16, 128, 96, 256, false, false, false, false, false, 1, false,
         'N', 'T'},
    };
    for (const Case &c: cases) {
        run(lt, stream, c);
    }
    CHECK(cudaStreamDestroy(stream));
    BCHECK(cublasLtDestroy(lt));
    CHECK(cudaGetLastError());
    PASS("cuBLASLt FP8 matmuls: e4m3 / e5m2, scales, bias, amax, 16 / 32 / 8 bit D, graphs");
    return 0;
}
