// NVCC_FLAGS: -arch=$ARCH -lcublas -lcublasLt
// cuBLAS and cuBLASLt, compared with a CPU reference: GEMM of float, double, complex, half and bfloat16 operands in all
// transposition combinations, sizes that do and don't suit the tensor core kernels (odd leading dimensions), strided and
// pointer array batches, device scalars, TF32, GEMV with strided vectors, dot products, and cuBLASLt matmuls with
// bias / ReLU / GELU epilogues and row-major layouts. Checked against NVIDIA's cuBLAS too.
#include "check.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <complex>
#include <cstring>
#include <vector>

#define BCHECK(call) do { \
    cublasStatus_t bcheck_status = (call); \
    if (bcheck_status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d at %s:%d: %s\n", int(bcheck_status), __FILE__, __LINE__, #call); \
        exit(1); \
    } \
} while (0)

using cd = std::complex<double>;

static double hostValue(int i, int salt) {
    return ((i * 2654435761u + salt * 40503u) % 2001) / 1000.0 - 1.0;
}

// column-major element of op(X) where X is stored with leading dimension ld
static cd opElement(const std::vector<cd> &x, int ld, char op, int row, int col) {
    if (op == 'N') return x[row + size_t(col) * ld];
    cd v = x[col + size_t(row) * ld];
    return op == 'C' ? std::conj(v) : v;
}

static cublasOperation_t cuOp(char op) {
    return op == 'N' ? CUBLAS_OP_N : op == 'T' ? CUBLAS_OP_T : CUBLAS_OP_C;
}

// device buffer of type T from complex host values (real types take the real part)
template<typename T>
static T fromCd(cd v) {
    if constexpr (std::is_same_v<T, cuComplex>) return make_cuComplex(float(v.real()), float(v.imag()));
    else if constexpr (std::is_same_v<T, cuDoubleComplex>) return make_cuDoubleComplex(v.real(), v.imag());
    else if constexpr (std::is_same_v<T, __half>) return __float2half(float(v.real()));
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) return __float2bfloat16(float(v.real()));
    else return T(v.real());
}

template<typename T>
static cd toCd(T v) {
    if constexpr (std::is_same_v<T, cuComplex>) return {v.x, v.y};
    else if constexpr (std::is_same_v<T, cuDoubleComplex>) return {v.x, v.y};
    else if constexpr (std::is_same_v<T, __half>) return __half2float(v);
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) return __bfloat162float(v);
    else return double(v);
}

template<typename T>
static T *upload(const std::vector<cd> &values) {
    std::vector<T> host(values.size());
    for (size_t i = 0; i < values.size(); i++) host[i] = fromCd<T>(values[i]);
    T *dev;
    CHECK(cudaMalloc(&dev, std::max<size_t>(1, host.size()) * sizeof(T)));
    CHECK(cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return dev;
}

template<typename T>
static std::vector<cd> download(const T *dev, size_t count) {
    std::vector<T> host(count);
    CHECK(cudaMemcpy(host.data(), dev, count * sizeof(T), cudaMemcpyDeviceToHost));
    std::vector<cd> out(count);
    for (size_t i = 0; i < count; i++) out[i] = toCd(host[i]);
    return out;
}

// values representable in the type (so the reference sees the same inputs)
template<typename T>
static std::vector<cd> values(size_t count, int salt, bool complex) {
    std::vector<cd> v(count);
    for (size_t i = 0; i < count; i++) {
        v[i] = toCd(fromCd<T>(cd(hostValue(int(i), salt), complex ? hostValue(int(i), salt + 7) : 0)));
    }
    return v;
}

// max |D - reference| relative to the largest reference value
static double compare(const std::vector<cd> &d, const std::vector<cd> &ref) {
    double err = 0, scale = 1e-30;
    for (size_t i = 0; i < d.size(); i++) {
        err = std::max(err, std::abs(d[i] - ref[i]));
        scale = std::max(scale, std::abs(ref[i]));
    }
    return err / scale;
}

static std::vector<cd> reference(char ta, char tb, int m, int n, int k, cd alpha, const std::vector<cd> &a, int lda,
                                 const std::vector<cd> &b, int ldb, cd beta, const std::vector<cd> &c, int ldc) {
    std::vector<cd> d(c);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            cd acc = 0;
            for (int l = 0; l < k; l++) acc += opElement(a, lda, ta, i, l) * opElement(b, ldb, tb, l, j);
            d[i + size_t(j) * ldc] = alpha * acc + beta * c[i + size_t(j) * ldc];
        }
    }
    return d;
}

/**
 * One GEMM through cublasGemmEx (T: operand type, TD: output type), compared with the reference
 */
template<typename T, typename TD>
static void gemmCase(cublasHandle_t handle, cudaDataType type, cudaDataType outType, cublasComputeType_t compute,
                     char ta, char tb, int m, int n, int k, int pad, double tolerance, const char *what) {
    const bool complex = type == CUDA_C_32F || type == CUDA_C_64F;
    const int lda = (ta == 'N' ? m : k) + pad, ldb = (tb == 'N' ? k : n) + pad, ldc = m + pad;
    const auto a = values<T>(size_t(lda) * (ta == 'N' ? k : m), 1, complex);
    const auto b = values<T>(size_t(ldb) * (tb == 'N' ? n : k), 2, complex);
    const auto c = values<TD>(size_t(ldc) * n, 3, complex);
    const cd alpha = complex ? cd(0.75, 0.5) : cd(1.25), beta = complex ? cd(-0.5, 0.25) : cd(-0.5);
    T *da = upload<T>(a), *db = upload<T>(b);
    TD *dc = upload<TD>(c);
    // the scalars as the compute type expects
    double ra[2] = {alpha.real(), alpha.imag()}, rb[2] = {beta.real(), beta.imag()};
    float fa[2] = {float(ra[0]), float(ra[1])}, fb[2] = {float(rb[0]), float(rb[1])};
    __half ha = __float2half(fa[0]), hb = __float2half(fb[0]);
    const void *alpha_ptr = fa, *beta_ptr = fb;
    if (compute == CUBLAS_COMPUTE_64F) {
        alpha_ptr = ra;
        beta_ptr = rb;
    } else if (compute == CUBLAS_COMPUTE_16F) {
        alpha_ptr = &ha;
        beta_ptr = &hb;
    }
    const cublasStatus_t status = cublasGemmEx(handle, cuOp(ta), cuOp(tb), m, n, k, alpha_ptr, da, type, lda, db, type,
                                               ldb, beta_ptr, dc, outType, ldc, compute, CUBLAS_GEMM_DEFAULT);
    EXPECT(status == CUBLAS_STATUS_SUCCESS, "%s %c%c %dx%dx%d (pad %d): status %d", what, ta, tb, m, n, k, pad,
           int(status));
    const auto d = download(dc, c.size());
    const auto ref = reference(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    const double err = compare(d, ref);
    EXPECT(err <= tolerance, "%s %c%c %dx%dx%d (pad %d): error %g", what, ta, tb, m, n, k, pad, err);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}

static void gemmTests(cublasHandle_t handle) {
    const char ops[3] = {'N', 'T', 'C'};
    struct Shape {
        int m, n, k, pad;
    } shapes[] = {{256, 192, 128, 0}, {67, 45, 33, 1}, {130, 7, 257, 3}, {1, 1, 1, 0}, {64, 64, 0, 1},
                  // large enough for 16 bit operands of odd sizes to be repacked for the tensor core kernels
                  {301, 257, 129, 1}};
    for (const Shape &s: shapes) {
        for (char ta: ops) {
            for (char tb: ops) {
                if (ta != 'C' && tb != 'C') {
                    gemmCase<float, float>(handle, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F, ta, tb, s.m, s.n, s.k,
                                           s.pad, 1e-5, "float");
                    gemmCase<double, double>(handle, CUDA_R_64F, CUDA_R_64F, CUBLAS_COMPUTE_64F, ta, tb, s.m, s.n,
                                             s.k, s.pad, 1e-12, "double");
                    gemmCase<__half, __half>(handle, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F, ta, tb, s.m, s.n,
                                             s.k, s.pad, 4e-3, "half");
                    gemmCase<__half, float>(handle, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F, ta, tb, s.m, s.n,
                                            s.k, s.pad, 1e-5, "half -> float");
                    gemmCase<__nv_bfloat16, __nv_bfloat16>(handle, CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F, ta,
                                                           tb, s.m, s.n, s.k, s.pad, 2e-2, "bfloat16");
                    gemmCase<__nv_bfloat16, float>(handle, CUDA_R_16BF, CUDA_R_32F, CUBLAS_COMPUTE_32F, ta, tb, s.m,
                                                   s.n, s.k, s.pad, 1e-5, "bfloat16 -> float");
                    gemmCase<float, float>(handle, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, ta, tb, s.m,
                                           s.n, s.k, s.pad, 5e-3, "tf32");
                    gemmCase<__half, __half>(handle, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F, ta, tb, s.m, s.n,
                                             s.k, s.pad, 2e-2, "half (16 bit compute)");
                }
                gemmCase<cuComplex, cuComplex>(handle, CUDA_C_32F, CUDA_C_32F, CUBLAS_COMPUTE_32F, ta, tb, s.m, s.n,
                                               s.k, s.pad, 1e-5, "complex float");
                gemmCase<cuDoubleComplex, cuDoubleComplex>(handle, CUDA_C_64F, CUDA_C_64F, CUBLAS_COMPUTE_64F, ta, tb,
                                                           s.m, s.n, s.k, s.pad, 1e-12, "complex double");
            }
        }
    }
    PASS("GEMM: types, transpositions, sizes");
}

static void batchTests(cublasHandle_t handle) {
    const int m = 48, n = 40, k = 24, batch = 5;
    const auto a = values<float>(size_t(m) * k * batch, 4, false);
    const auto b = values<float>(size_t(k) * n * batch, 5, false);
    const auto c = values<float>(size_t(m) * n * batch, 6, false);
    float *da = upload<float>(a), *db = upload<float>(b), *dc = upload<float>(c);
    const float alpha = 2, beta = 1;
    BCHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, da, m, long(m) * k, db, k,
                                     long(k) * n, &beta, dc, m, long(m) * n, batch));
    auto d = download(dc, c.size());
    for (int i = 0; i < batch; i++) {
        std::vector<cd> ai(a.begin() + i * m * k, a.begin() + (i + 1) * m * k);
        std::vector<cd> bi(b.begin() + i * k * n, b.begin() + (i + 1) * k * n);
        std::vector<cd> ci(c.begin() + i * m * n, c.begin() + (i + 1) * m * n);
        std::vector<cd> di(d.begin() + i * m * n, d.begin() + (i + 1) * m * n);
        EXPECT(compare(di, reference('N', 'N', m, n, k, alpha, ai, m, bi, k, beta, ci, m)) < 1e-5, "strided batch %d",
               i);
    }
    // pointer arrays (half operands, float output) in reverse order
    const auto ha = values<__half>(size_t(m) * k * batch, 7, false);
    const auto hb = values<__half>(size_t(k) * n * batch, 8, false);
    __half *dha = upload<__half>(ha), *dhb = upload<__half>(hb);
    float *dout;
    CHECK(cudaMalloc(&dout, sizeof(float) * m * n * batch));
    std::vector<const void *> pa(batch), pb(batch);
    std::vector<void *> pc(batch);
    for (int i = 0; i < batch; i++) {
        pa[i] = dha + (batch - 1 - i) * m * k;
        pb[i] = dhb + (batch - 1 - i) * k * n;
        pc[i] = dout + i * m * n;
    }
    const void **dpa;
    const void **dpb;
    void **dpc;
    CHECK(cudaMalloc(&dpa, batch * sizeof(void *)));
    CHECK(cudaMalloc(&dpb, batch * sizeof(void *)));
    CHECK(cudaMalloc(&dpc, batch * sizeof(void *)));
    CHECK(cudaMemcpy(dpa, pa.data(), batch * sizeof(void *), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dpb, pb.data(), batch * sizeof(void *), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dpc, pc.data(), batch * sizeof(void *), cudaMemcpyHostToDevice));
    const float one = 1, zero = 0;
    BCHECK(cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, dpa, CUDA_R_16F, m, dpb, CUDA_R_16F, k,
                               &zero, dpc, CUDA_R_32F, m, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    d = download(dout, size_t(m) * n * batch);
    for (int i = 0; i < batch; i++) {
        const int j = batch - 1 - i;
        std::vector<cd> ai(ha.begin() + j * m * k, ha.begin() + (j + 1) * m * k);
        std::vector<cd> bi(hb.begin() + j * k * n, hb.begin() + (j + 1) * k * n);
        std::vector<cd> di(d.begin() + i * m * n, d.begin() + (i + 1) * m * n);
        std::vector<cd> zeros(size_t(m) * n);
        EXPECT(compare(di, reference('N', 'N', m, n, k, 1, ai, m, bi, k, 0, zeros, m)) < 1e-5, "array batch %d", i);
    }
    // device scalars
    float *dscalars;
    CHECK(cudaMalloc(&dscalars, 2 * sizeof(float)));
    const float scalars[2] = {0.5f, 0.0f};
    CHECK(cudaMemcpy(dscalars, scalars, sizeof(scalars), cudaMemcpyHostToDevice));
    BCHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    BCHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, dscalars, da, m, db, k, dscalars + 1, dc, m));
    BCHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    d = download(dc, size_t(m) * n);
    {
        std::vector<cd> ai(a.begin(), a.begin() + m * k), bi(b.begin(), b.begin() + k * n), zeros(size_t(m) * n);
        EXPECT(compare(d, reference('N', 'N', m, n, k, 0.5, ai, m, bi, k, 0, zeros, m)) < 1e-5, "device scalars");
    }
    cudaFree(da), cudaFree(db), cudaFree(dc), cudaFree(dha), cudaFree(dhb), cudaFree(dout);
    cudaFree(dpa), cudaFree(dpb), cudaFree(dpc), cudaFree(dscalars);
    PASS("batched GEMM, device scalars");
}

static void vectorTests(cublasHandle_t handle) {
    const int m = 70, n = 33;
    for (char t: {'N', 'T'}) {
        for (int inc: {1, 3, -2}) {
            const int xlen = t == 'N' ? n : m, ylen = t == 'N' ? m : n;
            const int ainc = std::abs(inc);
            const auto a = values<float>(size_t(m) * n, 9, false);
            const auto x = values<float>(size_t(xlen) * ainc, 10, false);
            const auto y = values<float>(size_t(ylen) * ainc, 11, false);
            float *da = upload<float>(a), *dx = upload<float>(x), *dy = upload<float>(y);
            const float alpha = 1.5f, beta = 0.5f;
            BCHECK(cublasSgemv(handle, cuOp(t), m, n, &alpha, da, m, dx, inc, &beta, dy, inc));
            const auto out = download(dy, y.size());
            // element i of a vector with increment inc (negative: from the end)
            auto at = [&](int i, int len) { return inc > 0 ? size_t(i) * inc : size_t(len - 1 - i) * ainc; };
            for (int i = 0; i < ylen; i++) {
                cd acc = 0;
                for (int l = 0; l < xlen; l++) acc += opElement(a, m, t, i, l) * x[at(l, xlen)];
                const cd expected = cd(alpha) * acc + cd(beta) * y[at(i, ylen)];
                EXPECT(std::abs(out[at(i, ylen)] - expected) < 1e-4, "gemv %c inc %d [%d]: %f vs %f", t, inc, i,
                       out[at(i, ylen)].real(), expected.real());
            }
            cudaFree(da), cudaFree(dx), cudaFree(dy);
        }
    }
    const int len = 1000;
    const auto x = values<double>(len, 12, false), y = values<double>(len, 13, false);
    double *dx = upload<double>(x), *dy = upload<double>(y);
    double result = 0, expected = 0;
    for (int i = 0; i < len; i++) expected += x[i].real() * y[i].real();
    BCHECK(cublasDdot(handle, len, dx, 1, dy, 1, &result));
    EXPECT(std::abs(result - expected) < 1e-9, "ddot %f vs %f", result, expected);
    const auto cx = values<cuComplex>(len, 14, true), cy = values<cuComplex>(len, 15, true);
    cuComplex *dcx = upload<cuComplex>(cx), *dcy = upload<cuComplex>(cy);
    cuComplex cresult;
    cd cexpected = 0;
    for (int i = 0; i < len / 2; i++) cexpected += std::conj(cx[2 * i]) * cy[2 * i];
    BCHECK(cublasCdotc(handle, len / 2, dcx, 2, dcy, 2, &cresult));
    EXPECT(std::abs(cd(cresult.x, cresult.y) - cexpected) < 1e-3, "cdotc");
    cudaFree(dx), cudaFree(dy), cudaFree(dcx), cudaFree(dcy);
    PASS("GEMV and dot products");
}

static void ltTests() {
    cublasLtHandle_t lt;
    BCHECK(cublasLtCreate(&lt));
    const int m = 96, n = 80, k = 64;
    const auto a = values<__half>(size_t(k) * m, 16, false);   // A^T stored: k x m (op T)
    const auto b = values<__half>(size_t(k) * n, 17, false);
    const auto bias = values<__half>(m, 18, false);
    __half *da = upload<__half>(a), *db = upload<__half>(b), *dbias = upload<__half>(bias), *dd;
    CHECK(cudaMalloc(&dd, sizeof(__half) * m * n));
    struct Case {
        cublasLtEpilogue_t epilogue;
        const char *name;
    } cases[] = {{CUBLASLT_EPILOGUE_DEFAULT, "default"}, {CUBLASLT_EPILOGUE_BIAS, "bias"},
                 {CUBLASLT_EPILOGUE_RELU_BIAS, "relu + bias"}, {CUBLASLT_EPILOGUE_GELU_BIAS, "gelu + bias"},
                 {CUBLASLT_EPILOGUE_RELU, "relu"}};
    for (const Case &c: cases) {
        cublasLtMatmulDesc_t desc;
        BCHECK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        BCHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
        BCHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
        BCHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &c.epilogue, sizeof(c.epilogue)));
        if (c.epilogue & CUBLASLT_EPILOGUE_BIAS) {
            BCHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &dbias, sizeof(dbias)));
        }
        cublasLtMatrixLayout_t la, lb, ld;
        BCHECK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, k, m, k));
        BCHECK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, k, n, k));
        BCHECK(cublasLtMatrixLayoutCreate(&ld, CUDA_R_16F, m, n, m));
        cublasLtMatmulPreference_t pref;
        BCHECK(cublasLtMatmulPreferenceCreate(&pref));
        uint64_t workspace = 1 << 20;
        BCHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace,
                                                    sizeof(workspace)));
        cublasLtMatmulHeuristicResult_t result{};
        int count = 0;
        BCHECK(cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, ld, ld, pref, 1, &result, &count));
        EXPECT(count == 1, "heuristic count %d", count);
        void *ws = nullptr;
        if (result.workspaceSize > 0) CHECK(cudaMalloc(&ws, result.workspaceSize));
        const float alpha = 1, beta = 0;
        BCHECK(cublasLtMatmul(lt, desc, &alpha, da, la, db, lb, &beta, dd, ld, dd, ld, &result.algo, ws,
                              result.workspaceSize, 0));
        const auto d = download(dd, size_t(m) * n);
        std::vector<cd> zeros(size_t(m) * n);
        auto ref = reference('T', 'N', m, n, k, 1, a, k, b, k, 0, zeros, m);
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                double v = ref[i + size_t(j) * m].real();
                if (c.epilogue & CUBLASLT_EPILOGUE_BIAS) v += bias[i].real();
                if (c.epilogue & CUBLASLT_EPILOGUE_RELU) v = std::max(v, 0.0);
                if (c.epilogue & CUBLASLT_EPILOGUE_GELU) v = 0.5 * v * (1 + std::tanh(0.7978845608 * (v + 0.044715 * v * v * v)));
                ref[i + size_t(j) * m] = v;
            }
        }
        EXPECT(compare(d, ref) < 5e-3, "cublasLt %s: error %g", c.name, compare(d, ref));
        if (ws) cudaFree(ws);
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la), cublasLtMatrixLayoutDestroy(lb), cublasLtMatrixLayoutDestroy(ld);
        cublasLtMatmulDescDestroy(desc);
    }
    // row-major layouts (float): D (m x n row-major) = A (m x k row-major) B (k x n row-major)
    {
        const auto fa = values<float>(size_t(m) * k, 19, false), fb = values<float>(size_t(k) * n, 20, false);
        float *dfa = upload<float>(fa), *dfb = upload<float>(fb), *dfd;
        CHECK(cudaMalloc(&dfd, sizeof(float) * m * n));
        cublasLtMatmulDesc_t desc;
        BCHECK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasLtMatrixLayout_t la, lb, ld;
        BCHECK(cublasLtMatrixLayoutCreate(&la, CUDA_R_32F, m, k, k));
        BCHECK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_32F, k, n, n));
        BCHECK(cublasLtMatrixLayoutCreate(&ld, CUDA_R_32F, m, n, n));
        const cublasLtOrder_t row = CUBLASLT_ORDER_ROW;
        for (auto l: {la, lb, ld}) BCHECK(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)));
        const float alpha = 1, beta = 0;
        BCHECK(cublasLtMatmul(lt, desc, &alpha, dfa, la, dfb, lb, &beta, dfd, ld, dfd, ld, nullptr, nullptr, 0, 0));
        const auto d = download(dfd, size_t(m) * n);
        double err = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double acc = 0;
                for (int l = 0; l < k; l++) acc += fa[size_t(i) * k + l].real() * fb[size_t(l) * n + j].real();
                err = std::max(err, std::abs(d[size_t(i) * n + j].real() - acc));
            }
        }
        EXPECT(err < 1e-4, "row-major matmul error %g", err);
        cublasLtMatrixLayoutDestroy(la), cublasLtMatrixLayoutDestroy(lb), cublasLtMatrixLayoutDestroy(ld);
        cublasLtMatmulDescDestroy(desc);
        cudaFree(dfa), cudaFree(dfb), cudaFree(dfd);
    }
    cudaFree(da), cudaFree(db), cudaFree(dbias), cudaFree(dd);
    BCHECK(cublasLtDestroy(lt));
    PASS("cuBLASLt matmul, epilogues, row-major layouts");
}

int main() {
    cublasHandle_t handle;
    BCHECK(cublasCreate(&handle));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    BCHECK(cublasSetStream(handle, stream));
    gemmTests(handle);
    batchTests(handle);
    vectorTests(handle);
    ltTests();
    int version = 0;
    BCHECK(cublasGetVersion(handle, &version));
    EXPECT(version >= 130000, "version %d", version);
    BCHECK(cublasDestroy(handle));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaGetLastError());
    return 0;
}
