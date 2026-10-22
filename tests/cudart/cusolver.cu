// NVCC_FLAGS: -arch=$ARCH -lcublas -lcusolver
// cuBLAS level 1-3 routines, its batched factorizations, and cuSOLVER, for float, double and complex types, checked by
// their defining properties on the CPU: triangular solves and products, rank-k updates, symmetric products, vector
// operations; LU (solve residual), Cholesky (reconstruction), QR (Q orthonormal, QR = A, Q applied), Hermitian eigen
// decomposition (A V = V diag(w), ascending) and SVD (U S V^H = A, orthonormal factors, descending), on matrices small
// enough for one thread block and large enough for the blocked algorithms. Checked against NVIDIA's libraries too.
#include "check.h"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <complex>
#include <cstring>
#include <type_traits>
#include <vector>

#define BCHECK(call) do { cublasStatus_t s_ = (call); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %d at %s:%d: %s\n", int(s_), __FILE__, __LINE__, #call); exit(1); } } while (0)
#define SCHECK(call) do { cusolverStatus_t s_ = (call); if (s_ != CUSOLVER_STATUS_SUCCESS) { \
    fprintf(stderr, "cuSOLVER error %d at %s:%d: %s\n", int(s_), __FILE__, __LINE__, #call); exit(1); } } while (0)

using cd = std::complex<double>;
using Mat = std::vector<cd>;  // column-major

template<typename T> constexpr bool IS_COMPLEX = std::is_same_v<T, cuComplex> || std::is_same_v<T, cuDoubleComplex>;
template<typename T> using RealT = std::conditional_t<std::is_same_v<T, cuComplex>, float,
                                  std::conditional_t<std::is_same_v<T, cuDoubleComplex>, double, T>>;
template<typename T> double eps() { return std::is_same_v<RealT<T>, double> ? 1e-15 : 1e-6; }

template<typename T> T fromCd(cd v) {
    if constexpr (std::is_same_v<T, cuComplex>) return make_cuComplex(float(v.real()), float(v.imag()));
    else if constexpr (std::is_same_v<T, cuDoubleComplex>) return make_cuDoubleComplex(v.real(), v.imag());
    else return T(v.real());
}
template<typename T> cd toCd(T v) {
    if constexpr (IS_COMPLEX<T>) return {v.x, v.y};
    else return double(v);
}

static double rnd(int i, int salt) { return ((i * 2654435761u + salt * 97u) % 2001) / 1000.0 - 1.0; }

template<typename T> Mat random(int rows, int cols, int salt) {
    Mat m(size_t(rows) * cols);
    for (size_t i = 0; i < m.size(); i++) {
        m[i] = toCd(fromCd<T>(cd(rnd(int(i), salt), IS_COMPLEX<T> ? rnd(int(i), salt + 1) : 0)));
    }
    return m;
}

template<typename T> T *upload(const Mat &m) {
    std::vector<T> h(m.size());
    for (size_t i = 0; i < m.size(); i++) h[i] = fromCd<T>(m[i]);
    T *d;
    CHECK(cudaMalloc(&d, std::max<size_t>(1, h.size()) * sizeof(T)));
    CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template<typename T> Mat download(const T *d, size_t n) {
    std::vector<T> h(n);
    CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    Mat m(n);
    for (size_t i = 0; i < n; i++) m[i] = toCd(h[i]);
    return m;
}
template<typename R> std::vector<double> downloadReal(const R *d, size_t n) {
    std::vector<R> h(n);
    CHECK(cudaMemcpy(h.data(), d, n * sizeof(R), cudaMemcpyDeviceToHost));
    return std::vector<double>(h.begin(), h.end());
}

// C = op(A) op(B), A stored rows x cols
static Mat mul(const Mat &a, int ar, int ac, char ta, const Mat &b, int br, int bc, char tb) {
    const int m = ta == 'N' ? ar : ac, k = ta == 'N' ? ac : ar, n = tb == 'N' ? bc : br;
    Mat c(size_t(m) * n);
    for (int j = 0; j < n; j++)
        for (int l = 0; l < k; l++) {
            cd bv = tb == 'N' ? b[l + size_t(j) * br] : b[j + size_t(l) * br];
            if (tb == 'C') bv = std::conj(bv);
            if (bv == cd(0)) continue;
            for (int i = 0; i < m; i++) {
                cd av = ta == 'N' ? a[i + size_t(l) * ar] : a[l + size_t(i) * ar];
                if (ta == 'C') av = std::conj(av);
                c[i + size_t(j) * m] += av * bv;
            }
        }
    return c;
}
static double maxAbs(const Mat &m) {
    double r = 0;
    for (auto v: m) r = std::max(r, std::abs(v));
    return r;
}
static double diff(const Mat &a, const Mat &b) {
    double r = 0;
    for (size_t i = 0; i < a.size(); i++) r = std::max(r, std::abs(a[i] - b[i]));
    return r;
}
// keeps the triangle (unit: ones on the diagonal), zeroes the rest
static Mat triangle(const Mat &a, int n, bool lower, bool unit) {
    Mat t(size_t(n) * n);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            if (i == j) t[i + size_t(j) * n] = unit ? cd(1) : a[i + size_t(j) * n];
            else if (lower ? i > j : i < j) t[i + size_t(j) * n] = a[i + size_t(j) * n];
    return t;
}
static Mat identity(int n) {
    Mat m(size_t(n) * n);
    for (int i = 0; i < n; i++) m[i + size_t(i) * n] = 1;
    return m;
}

// ---- typed API tables
template<typename T> struct Api;
#define API(P, T)                                                                                                      \
template<> struct Api<T> {                                                                                             \
    static constexpr auto trsm = cublas##P##trsm_v2; static constexpr auto trmm = cublas##P##trmm_v2;                  \
    static constexpr auto axpy = cublas##P##axpy_v2; static constexpr auto scal = cublas##P##scal_v2;                  \
    static constexpr auto trsv = cublas##P##trsv_v2; static constexpr auto trmv = cublas##P##trmv_v2;                  \
    static constexpr auto symm = cublas##P##symm_v2; static constexpr auto syr2k = cublas##P##syr2k_v2;                \
    static constexpr auto getrfBatched = cublas##P##getrfBatched;                                                      \
    static constexpr auto getrsBatched = cublas##P##getrsBatched;                                                      \
    static constexpr auto geqrfBatched = cublas##P##geqrfBatched;                                                      \
    static constexpr auto gelsBatched = cublas##P##gelsBatched;                                                        \
    static constexpr auto trsmBatched = cublas##P##trsmBatched;                                                        \
    static constexpr auto getrf_bufferSize = cusolverDn##P##getrf_bufferSize;                                          \
    static constexpr auto getrf = cusolverDn##P##getrf; static constexpr auto getrs = cusolverDn##P##getrs;            \
    static constexpr auto potrf_bufferSize = cusolverDn##P##potrf_bufferSize;                                          \
    static constexpr auto potrf = cusolverDn##P##potrf; static constexpr auto potrs = cusolverDn##P##potrs;            \
    static constexpr auto potrfBatched = cusolverDn##P##potrfBatched;                                                  \
    static constexpr auto geqrf_bufferSize = cusolverDn##P##geqrf_bufferSize;                                          \
    static constexpr auto geqrf = cusolverDn##P##geqrf;                                                                \
    static constexpr auto gesvd_bufferSize = cusolverDn##P##gesvd_bufferSize;                                          \
    static constexpr auto gesvd = cusolverDn##P##gesvd;                                                                \
    static constexpr auto gesvdj_bufferSize = cusolverDn##P##gesvdj_bufferSize;                                        \
    static constexpr auto gesvdj = cusolverDn##P##gesvdj;                                                              \
    static constexpr auto gesvdjBatched_bufferSize = cusolverDn##P##gesvdjBatched_bufferSize;                          \
    static constexpr auto gesvdjBatched = cusolverDn##P##gesvdjBatched;                                                \
};
API(S, float) API(D, double) API(C, cuComplex) API(Z, cuDoubleComplex)

// Q generation / application and eigen solvers are named differently for complex types
static auto qGen(float *) { return std::make_pair(cusolverDnSorgqr_bufferSize, cusolverDnSorgqr); }
static auto qGen(double *) { return std::make_pair(cusolverDnDorgqr_bufferSize, cusolverDnDorgqr); }
static auto qGen(cuComplex *) { return std::make_pair(cusolverDnCungqr_bufferSize, cusolverDnCungqr); }
static auto qGen(cuDoubleComplex *) { return std::make_pair(cusolverDnZungqr_bufferSize, cusolverDnZungqr); }
static auto qMul(float *) { return std::make_pair(cusolverDnSormqr_bufferSize, cusolverDnSormqr); }
static auto qMul(double *) { return std::make_pair(cusolverDnDormqr_bufferSize, cusolverDnDormqr); }
static auto qMul(cuComplex *) { return std::make_pair(cusolverDnCunmqr_bufferSize, cusolverDnCunmqr); }
static auto qMul(cuDoubleComplex *) { return std::make_pair(cusolverDnZunmqr_bufferSize, cusolverDnZunmqr); }
static auto eig(float *) { return std::make_pair(cusolverDnSsyevd_bufferSize, cusolverDnSsyevd); }
static auto eig(double *) { return std::make_pair(cusolverDnDsyevd_bufferSize, cusolverDnDsyevd); }
static auto eig(cuComplex *) { return std::make_pair(cusolverDnCheevd_bufferSize, cusolverDnCheevd); }
static auto eig(cuDoubleComplex *) { return std::make_pair(cusolverDnZheevd_bufferSize, cusolverDnZheevd); }
static auto eigBatched(float *) { return std::make_pair(cusolverDnSsyevjBatched_bufferSize, cusolverDnSsyevjBatched); }
static auto eigBatched(double *) { return std::make_pair(cusolverDnDsyevjBatched_bufferSize, cusolverDnDsyevjBatched); }
static auto eigBatched(cuComplex *) { return std::make_pair(cusolverDnCheevjBatched_bufferSize, cusolverDnCheevjBatched); }
static auto eigBatched(cuDoubleComplex *) {
    return std::make_pair(cusolverDnZheevjBatched_bufferSize, cusolverDnZheevjBatched);
}
static auto herk(float *) { return cublasSsyrk_v2; }
static auto herk(double *) { return cublasDsyrk_v2; }
static auto herk(cuComplex *) { return cublasCherk_v2; }
static auto herk(cuDoubleComplex *) { return cublasZherk_v2; }

static int checkInfo(int *d_info) {
    int info = -1;
    CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    return info;
}

template<typename T> const char *typeName() {
    return std::is_same_v<T, float> ? "float" : std::is_same_v<T, double> ? "double"
           : std::is_same_v<T, cuComplex> ? "complex float" : "complex double";
}

// ---- cuBLAS level 1-3
template<typename T> void blasTests(cublasHandle_t blas) {
    using R = RealT<T>;
    const double tol = eps<T>() * 300;
    const char tops[3] = {'N', 'T', 'C'};
    const cublasOperation_t ops[3] = {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C};
    for (int n: {37, 300}) {
        const int nrhs = 23;
        // well conditioned triangular matrices (with or without unit diagonal): small off-diagonal elements
        Mat a = random<T>(n, n, 1);
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++)
                a[i + size_t(j) * n] = toCd(fromCd<T>(i == j ? cd(2, 0) : a[i + size_t(j) * n] / double(n)));
        T *da = upload<T>(a);
        for (int side = 0; side < 2; side++)
            for (int lower = 0; lower < 2; lower++)
                for (int o = 0; o < 3; o++)
                    for (int unit = 0; unit < 2; unit++) {
                        const int m = side == 0 ? n : nrhs, cols = side == 0 ? nrhs : n;
                        const Mat b = random<T>(m, cols, 2);
                        T *db = upload<T>(b);
                        const T alpha = fromCd<T>(cd(2, IS_COMPLEX<T> ? 1 : 0));
                        BCHECK(Api<T>::trsm(blas, side == 0 ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
                                            lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER, ops[o],
                                            unit ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT, m, cols, &alpha, da, n, db,
                                            m));
                        const Mat x = download(db, b.size());
                        const Mat t = triangle(a, n, lower, unit);
                        const Mat ax = side == 0 ? mul(t, n, n, tops[o], x, m, cols, 'N')
                                                 : mul(x, m, cols, 'N', t, n, n, tops[o]);
                        Mat ab(b.size());
                        for (size_t i = 0; i < b.size(); i++) ab[i] = toCd(alpha) * b[i];
                        EXPECT(diff(ax, ab) <= tol * maxAbs(ab) * n, "%s trsm side %d lower %d op %c unit %d n %d: %g",
                               typeName<T>(), side, lower, tops[o], unit, n, diff(ax, ab));
                        // trmm: C = alpha op(A) B
                        T *dc;
                        CHECK(cudaMalloc(&dc, sizeof(T) * b.size()));
                        T *db2 = upload<T>(b);
                        BCHECK(Api<T>::trmm(blas, side == 0 ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
                                            lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER, ops[o],
                                            unit ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT, m, cols, &alpha, da, n,
                                            db2, m, dc, m));
                        Mat expected = side == 0 ? mul(t, n, n, tops[o], b, m, cols, 'N')
                                                 : mul(b, m, cols, 'N', t, n, n, tops[o]);
                        for (auto &v: expected) v *= toCd(alpha);
                        EXPECT(diff(download(dc, b.size()), expected) <= tol * maxAbs(expected), "%s trmm",
                               typeName<T>());
                        cudaFree(db), cudaFree(db2), cudaFree(dc);
                    }
        // trsv / trmv with a strided vector
        for (int o = 0; o < 3; o++) {
            const Mat x0 = random<T>(n * 2, 1, 3);
            T *dx = upload<T>(x0);
            BCHECK(Api<T>::trmv(blas, CUBLAS_FILL_MODE_UPPER, ops[o], CUBLAS_DIAG_NON_UNIT, n, da, n, dx, 2));
            BCHECK(Api<T>::trsv(blas, CUBLAS_FILL_MODE_UPPER, ops[o], CUBLAS_DIAG_NON_UNIT, n, da, n, dx, 2));
            const Mat x1 = download(dx, x0.size());
            EXPECT(diff(x1, x0) <= tol * n, "%s trmv / trsv op %c: %g", typeName<T>(), tops[o], diff(x1, x0));
            cudaFree(dx);
        }
        cudaFree(da);
    }
    // syrk / herk, syr2k, symm
    {
        const int n = 70, k = 45;
        const Mat a = random<T>(n, k, 4), b = random<T>(n, k, 5), c0 = random<T>(n, n, 6);
        T *da = upload<T>(a), *db = upload<T>(b);
        for (int lower = 0; lower < 2; lower++) {
            // herk: C = alpha A A^H + beta C on the triangle (the other triangle unchanged)
            T *dc = upload<T>(c0);
            const R alpha = 1.5, beta = 0.5;
            BCHECK(herk((T *) nullptr)(blas, lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER,
                                       CUBLAS_OP_N, n, k, &alpha, da, n, &beta, dc, n));
            const Mat c = download(dc, c0.size());
            const Mat aa = mul(a, n, k, 'N', a, n, k, IS_COMPLEX<T> ? 'C' : 'T');
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++) {
                    const bool in = lower ? i >= j : i <= j;
                    cd expected = in ? double(alpha) * aa[i + j * n] + double(beta) * c0[i + j * n] : c0[i + j * n];
                    if (in && i == j && IS_COMPLEX<T>) expected = expected.real();
                    EXPECT(std::abs(c[i + j * n] - expected) <= tol * 50, "%s herk (%d, %d)", typeName<T>(), i, j);
                }
            cudaFree(dc);
            // syr2k: C = alpha (A B^T + B A^T) + beta C
            dc = upload<T>(c0);
            const T alpha_t = fromCd<T>(cd(0.75)), beta_t = fromCd<T>(cd(-1));
            BCHECK(Api<T>::syr2k(blas, lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k,
                                 &alpha_t, da, n, db, n, &beta_t, dc, n));
            const Mat c2 = download(dc, c0.size());
            const Mat ab = mul(a, n, k, 'N', b, n, k, 'T'), ba = mul(b, n, k, 'N', a, n, k, 'T');
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++) {
                    const bool in = lower ? i >= j : i <= j;
                    const cd expected = in ? 0.75 * (ab[i + j * n] + ba[i + j * n]) - c0[i + j * n] : c0[i + j * n];
                    EXPECT(std::abs(c2[i + j * n] - expected) <= tol * 50, "%s syr2k", typeName<T>());
                }
            cudaFree(dc);
            // symm: C = A_sym B (A from its triangle)
            const Mat s = random<T>(n, n, 7), bb = random<T>(n, k, 8);
            T *ds = upload<T>(s), *dbb = upload<T>(bb);
            CHECK(cudaMalloc(&dc, sizeof(T) * n * k));
            const T one = fromCd<T>(1), zero = fromCd<T>(0);
            BCHECK(Api<T>::symm(blas, CUBLAS_SIDE_LEFT, lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER, n, k,
                                &one, ds, n, dbb, n, &zero, dc, n));
            Mat full(size_t(n) * n);
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++) full[i + j * n] = (lower ? i >= j : i <= j) ? s[i + j * n] : s[j + i * n];
            const Mat expected = mul(full, n, n, 'N', bb, n, k, 'N');
            EXPECT(diff(download(dc, size_t(n) * k), expected) <= tol * 50, "%s symm", typeName<T>());
            cudaFree(dc), cudaFree(ds), cudaFree(dbb);
        }
        cudaFree(da), cudaFree(db);
    }
    // level 1
    {
        const int n = 1000;
        const Mat x = random<T>(n, 1, 9), y = random<T>(n, 1, 10);
        T *dx = upload<T>(x), *dy = upload<T>(y);
        const T alpha = fromCd<T>(cd(0.5, IS_COMPLEX<T> ? -0.25 : 0));
        BCHECK(Api<T>::axpy(blas, n, &alpha, dx, 1, dy, 1));
        BCHECK(Api<T>::scal(blas, n, &alpha, dx, 1));
        const Mat y2 = download(dy, n), x2 = download(dx, n);
        for (int i = 0; i < n; i++) {
            EXPECT(std::abs(y2[i] - (y[i] + toCd(alpha) * x[i])) <= tol, "%s axpy", typeName<T>());
            EXPECT(std::abs(x2[i] - toCd(alpha) * x[i]) <= tol, "%s scal", typeName<T>());
        }
        cudaFree(dx), cudaFree(dy);
    }
    printf("PASS cuBLAS level 1-3 (%s)\n", typeName<T>());
}

// ---- batched factorizations of cuBLAS
template<typename T> void batchedTests(cublasHandle_t blas) {
    const int n = 24, nrhs = 5, batch = 7;
    const double tol = eps<T>() * 1e3;
    std::vector<Mat> as, bs;
    std::vector<T *> dA(batch), dB(batch), dA2(batch), dTau(batch);
    for (int b = 0; b < batch; b++) {
        as.push_back(random<T>(n, n, 20 + b));
        bs.push_back(random<T>(n, nrhs, 40 + b));
        dA[b] = upload<T>(as[b]);
        dB[b] = upload<T>(bs[b]);
        dA2[b] = upload<T>(as[b]);
        CHECK(cudaMalloc(&dTau[b], sizeof(T) * n));
    }
    T **dAp, **dBp, **dA2p, **dTaup;
    int *pivots, *infos;
    CHECK(cudaMalloc(&dAp, sizeof(T *) * batch));
    CHECK(cudaMalloc(&dBp, sizeof(T *) * batch));
    CHECK(cudaMalloc(&dA2p, sizeof(T *) * batch));
    CHECK(cudaMalloc(&dTaup, sizeof(T *) * batch));
    CHECK(cudaMalloc(&pivots, sizeof(int) * n * batch));
    CHECK(cudaMalloc(&infos, sizeof(int) * batch));
    CHECK(cudaMemcpy(dAp, dA.data(), sizeof(T *) * batch, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dBp, dB.data(), sizeof(T *) * batch, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dA2p, dA2.data(), sizeof(T *) * batch, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dTaup, dTau.data(), sizeof(T *) * batch, cudaMemcpyHostToDevice));
    BCHECK(Api<T>::getrfBatched(blas, n, dAp, n, pivots, infos, batch));
    int info = 0;
    BCHECK(Api<T>::getrsBatched(blas, CUBLAS_OP_N, n, nrhs, dAp, n, pivots, dBp, n, &info, batch));
    std::vector<int> h_infos(batch);
    CHECK(cudaMemcpy(h_infos.data(), infos, sizeof(int) * batch, cudaMemcpyDeviceToHost));
    for (int b = 0; b < batch; b++) {
        EXPECT(h_infos[b] == 0, "getrfBatched info %d", h_infos[b]);
        const Mat x = download(dB[b], size_t(n) * nrhs);
        EXPECT(diff(mul(as[b], n, n, 'N', x, n, nrhs, 'N'), bs[b]) <= tol * maxAbs(x) * n, "%s getrs batched %d",
               typeName<T>(), b);
    }
    // QR batched: R from geqrf, and least squares (gels) on fresh copies
    BCHECK(Api<T>::geqrfBatched(blas, n, n, dA2p, n, dTaup, &info, batch));
    for (int b = 0; b < batch; b++) {
        const Mat qr = download(dA2[b], size_t(n) * n);
        // |R(0,0)| is the norm of A's first column
        double norm = 0;
        for (int i = 0; i < n; i++) norm += std::norm(as[b][i]);
        EXPECT(std::abs(std::abs(qr[0]) - std::sqrt(norm)) <= tol * 10, "%s geqrf batched R(0,0)", typeName<T>());
        std::vector<T> h(size_t(n) * n);
        for (size_t i = 0; i < h.size(); i++) h[i] = fromCd<T>(as[b][i]);
        CHECK(cudaMemcpy(dA2[b], h.data(), sizeof(T) * h.size(), cudaMemcpyHostToDevice));
        for (size_t i = 0; i < size_t(n) * nrhs; i++) h[i] = fromCd<T>(bs[b][i]);
        CHECK(cudaMemcpy(dB[b], h.data(), sizeof(T) * n * nrhs, cudaMemcpyHostToDevice));
    }
    BCHECK(Api<T>::gelsBatched(blas, CUBLAS_OP_N, n, n, nrhs, dA2p, n, dBp, n, &info, infos, batch));
    for (int b = 0; b < batch; b++) {
        const Mat x = download(dB[b], size_t(n) * nrhs);
        EXPECT(diff(mul(as[b], n, n, 'N', x, n, nrhs, 'N'), bs[b]) <= tol * maxAbs(x) * n, "%s gels batched %d",
               typeName<T>(), b);
    }
    for (int b = 0; b < batch; b++) cudaFree(dA[b]), cudaFree(dB[b]), cudaFree(dA2[b]), cudaFree(dTau[b]);
    cudaFree(dAp), cudaFree(dBp), cudaFree(dA2p), cudaFree(dTaup), cudaFree(pivots), cudaFree(infos);
    printf("PASS cuBLAS batched LU, QR, least squares (%s)\n", typeName<T>());
}

// ---- cuSOLVER
template<typename T> void solverTests(cusolverDnHandle_t solver) {
    using R = RealT<T>;
    const double tol = eps<T>() * 2e3;
    int *d_info;
    CHECK(cudaMalloc(&d_info, sizeof(int)));
    for (int n: {50, 700}) {
        // LU and solve
        const int nrhs = 3;
        const Mat a = random<T>(n, n, 60), b = random<T>(n, nrhs, 61);
        T *da = upload<T>(a), *db = upload<T>(b), *work;
        int lwork = 0, *ipiv;
        SCHECK(Api<T>::getrf_bufferSize(solver, n, n, da, n, &lwork));
        CHECK(cudaMalloc(&work, sizeof(T) * std::max(1, lwork)));
        CHECK(cudaMalloc(&ipiv, sizeof(int) * n));
        SCHECK(Api<T>::getrf(solver, n, n, da, n, work, ipiv, d_info));
        EXPECT(checkInfo(d_info) == 0, "getrf info");
        for (cublasOperation_t op: {CUBLAS_OP_N, CUBLAS_OP_C}) {
            T *dx = upload<T>(b);
            SCHECK(Api<T>::getrs(solver, op, n, nrhs, da, n, ipiv, dx, n, d_info));
            const Mat x = download(dx, b.size());
            const Mat ax = mul(a, n, n, op == CUBLAS_OP_N ? 'N' : 'C', x, n, nrhs, 'N');
            EXPECT(diff(ax, b) <= tol * maxAbs(x) * std::sqrt(n), "%s getrs n %d: residual %g", typeName<T>(), n,
                   diff(ax, b));
            cudaFree(dx);
        }
        cudaFree(da), cudaFree(db), cudaFree(work), cudaFree(ipiv);
        // Cholesky of A A^H + n I
        Mat spd = mul(a, n, n, 'N', a, n, n, 'C');
        for (int i = 0; i < n; i++) spd[i + size_t(i) * n] += double(n);
        for (int lower = 0; lower < 2; lower++) {
            T *ds = upload<T>(spd);
            const cublasFillMode_t uplo = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
            SCHECK(Api<T>::potrf_bufferSize(solver, uplo, n, ds, n, &lwork));
            CHECK(cudaMalloc(&work, sizeof(T) * std::max(1, lwork)));
            SCHECK(Api<T>::potrf(solver, uplo, n, ds, n, work, lwork, d_info));
            EXPECT(checkInfo(d_info) == 0, "potrf info");
            const Mat f = triangle(download(ds, spd.size()), n, lower, false);
            const Mat rebuilt = lower ? mul(f, n, n, 'N', f, n, n, 'C') : mul(f, n, n, 'C', f, n, n, 'N');
            EXPECT(diff(rebuilt, spd) <= tol * maxAbs(spd), "%s potrf lower %d n %d: %g", typeName<T>(), lower, n,
                   diff(rebuilt, spd));
            T *dx = upload<T>(b);
            SCHECK(Api<T>::potrs(solver, uplo, n, nrhs, ds, n, dx, n, d_info));
            const Mat x = download(dx, b.size());
            EXPECT(diff(mul(spd, n, n, 'N', x, n, nrhs, 'N'), b) <= tol * maxAbs(spd) * maxAbs(x), "%s potrs",
                   typeName<T>());
            cudaFree(ds), cudaFree(dx), cudaFree(work);
        }
        // a matrix that isn't positive definite: info is the failing column
        {
            Mat bad = spd;
            bad[5 + size_t(5) * n] = -1e6;
            T *ds = upload<T>(bad);
            SCHECK(Api<T>::potrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, n, ds, n, &lwork));
            CHECK(cudaMalloc(&work, sizeof(T) * std::max(1, lwork)));
            SCHECK(Api<T>::potrf(solver, CUBLAS_FILL_MODE_LOWER, n, ds, n, work, lwork, d_info));
            EXPECT(checkInfo(d_info) == 6, "potrf info for a negative pivot: %d", checkInfo(d_info));
            cudaFree(ds), cudaFree(work);
        }
    }
    // QR: geqrf, orgqr / ungqr, ormqr / unmqr
    for (auto shape: {std::make_pair(40, 30), std::make_pair(400, 260)}) {
        const int m = shape.first, n = shape.second;
        const Mat a = random<T>(m, n, 70);
        T *da = upload<T>(a), *tau, *work;
        CHECK(cudaMalloc(&tau, sizeof(T) * n));
        int lwork = 0;
        SCHECK(Api<T>::geqrf_bufferSize(solver, m, n, da, m, &lwork));
        CHECK(cudaMalloc(&work, sizeof(T) * lwork));
        SCHECK(Api<T>::geqrf(solver, m, n, da, m, tau, work, lwork, d_info));
        Mat rn(size_t(n) * n);
        const Mat packed = download(da, a.size());
        for (int j = 0; j < n; j++)
            for (int i = 0; i <= j; i++) rn[i + size_t(j) * n] = packed[i + size_t(j) * m];
        // C = Q^H A with ormqr: its top n rows are R, the rest zero
        T *dc = upload<T>(a);
        auto [mul_size, mul_fn] = qMul((T *) nullptr);
        int lwork2 = 0;
        SCHECK(mul_size(solver, CUBLAS_SIDE_LEFT, IS_COMPLEX<T> ? CUBLAS_OP_C : CUBLAS_OP_T, m, n, n, da, m, tau, dc,
                        m, &lwork2));
        T *work2;
        CHECK(cudaMalloc(&work2, sizeof(T) * lwork2));
        SCHECK(mul_fn(solver, CUBLAS_SIDE_LEFT, IS_COMPLEX<T> ? CUBLAS_OP_C : CUBLAS_OP_T, m, n, n, da, m, tau, dc, m,
                      work2, lwork2, d_info));
        const Mat qha = download(dc, a.size());
        double err = 0;
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++) {
                err = std::max(err, std::abs(qha[i + size_t(j) * m] - (i < n ? rn[i + size_t(j) * n] : cd(0))));
            }
        EXPECT(err <= tol * maxAbs(a) * 4, "%s ormqr Q^H A = R (%dx%d): %g", typeName<T>(), m, n, err);
        // Q from orgqr: orthonormal columns, Q R = A
        auto [gen_size, gen_fn] = qGen((T *) nullptr);
        SCHECK(gen_size(solver, m, n, n, da, m, tau, &lwork2));
        cudaFree(work2);
        CHECK(cudaMalloc(&work2, sizeof(T) * lwork2));
        SCHECK(gen_fn(solver, m, n, n, da, m, tau, work2, lwork2, d_info));
        const Mat q = download(da, a.size());
        EXPECT(diff(mul(q, m, n, 'C', q, m, n, 'N'), identity(n)) <= tol * 4, "%s orgqr Q^H Q = I", typeName<T>());
        EXPECT(diff(mul(q, m, n, 'N', rn, n, n, 'N'), a) <= tol * maxAbs(a) * 4, "%s Q R = A", typeName<T>());
        cudaFree(da), cudaFree(tau), cudaFree(work), cudaFree(work2), cudaFree(dc);
    }
    // Hermitian eigen decomposition
    for (int n: {33, 200}) {
        Mat a = random<T>(n, n, 80);
        const Mat h = [&] {
            Mat s(size_t(n) * n);
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++) {
                    s[i + size_t(j) * n] = a[i + size_t(j) * n] + std::conj(a[j + size_t(i) * n]);
                }
            return s;
        }();
        // only the lower triangle is passed: garbage above
        Mat input = h;
        for (int j = 0; j < n; j++)
            for (int i = 0; i < j; i++) input[i + size_t(j) * n] = 1e30;
        T *dh = upload<T>(input), *work;
        R *dw;
        CHECK(cudaMalloc(&dw, sizeof(R) * n));
        auto [size_fn, fn] = eig((T *) nullptr);
        int lwork = 0;
        SCHECK(size_fn(solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, dh, n, dw, &lwork));
        CHECK(cudaMalloc(&work, sizeof(T) * lwork));
        SCHECK(fn(solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, dh, n, dw, work, lwork, d_info));
        EXPECT(checkInfo(d_info) == 0, "eigen info");
        const Mat v = download(dh, h.size());
        const std::vector<double> w = downloadReal(dw, n);
        for (int i = 1; i < n; i++) EXPECT(w[i - 1] <= w[i], "eigenvalues not ascending");
        Mat vw = v;
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++) vw[i + size_t(j) * n] *= w[j];
        EXPECT(diff(mul(h, n, n, 'N', v, n, n, 'N'), vw) <= tol * maxAbs(h) * 4, "%s A V = V diag(w) n %d: %g",
               typeName<T>(), n, diff(mul(h, n, n, 'N', v, n, n, 'N'), vw));
        EXPECT(diff(mul(v, n, n, 'C', v, n, n, 'N'), identity(n)) <= tol * 4, "%s V^H V = I", typeName<T>());
        cudaFree(dh), cudaFree(dw), cudaFree(work);
    }
    // batched Jacobi eigen decomposition
    {
        const int n = 12, batch = 9;
        Mat all(size_t(n) * n * batch);
        for (int b = 0; b < batch; b++) {
            const Mat a = random<T>(n, n, 90 + b);
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++)
                    all[b * n * n + i + j * n] = a[i + j * n] + std::conj(a[j + i * n]);
        }
        T *dh = upload<T>(all), *work;
        R *dw;
        int *infos;
        CHECK(cudaMalloc(&dw, sizeof(R) * n * batch));
        CHECK(cudaMalloc(&infos, sizeof(int) * batch));
        syevjInfo_t params;
        SCHECK(cusolverDnCreateSyevjInfo(&params));
        auto [size_fn, fn] = eigBatched((T *) nullptr);
        int lwork = 0;
        SCHECK(size_fn(solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, dh, n, dw, &lwork, params, batch));
        CHECK(cudaMalloc(&work, sizeof(T) * lwork));
        SCHECK(fn(solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, dh, n, dw, work, lwork, infos, params,
                  batch));
        const Mat v = download(dh, all.size());
        const std::vector<double> w = downloadReal(dw, size_t(n) * batch);
        for (int b = 0; b < batch; b++) {
            const Mat hb(all.begin() + b * n * n, all.begin() + (b + 1) * n * n);
            Mat vb(v.begin() + b * n * n, v.begin() + (b + 1) * n * n), vw = vb;
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++) vw[i + j * n] *= w[b * n + j];
            EXPECT(diff(mul(hb, n, n, 'N', vb, n, n, 'N'), vw) <= tol * maxAbs(hb) * 4, "%s syevjBatched %d",
                   typeName<T>(), b);
        }
        cusolverDnDestroySyevjInfo(params);
        cudaFree(dh), cudaFree(dw), cudaFree(work), cudaFree(infos);
    }
    // SVD: gesvd (U, V^H), gesvdj (economy and full), gesvdjBatched
    for (auto shape: {std::make_pair(60, 40), std::make_pair(40, 60), std::make_pair(300, 120)}) {
        const int m = shape.first, n = shape.second, k = std::min(m, n);
        const Mat a = random<T>(m, n, 100);
        auto check = [&](const Mat &u, int ucols, const Mat &v, const std::vector<double> &s, const char *what) {
            for (int i = 1; i < k; i++) EXPECT(s[i - 1] >= s[i], "%s singular values not descending", what);
            const double u_err = diff(mul(u, m, ucols, 'C', u, m, ucols, 'N'), identity(ucols));
            EXPECT(u_err <= tol * 8, "%s %s U^H U = I (%dx%d): %g", typeName<T>(), what, m, n, u_err);
            const int vcols = int(v.size() / n);
            EXPECT(diff(mul(v, n, vcols, 'C', v, n, vcols, 'N'), identity(vcols)) <= tol * 8, "%s %s V^H V = I",
                   typeName<T>(), what);
            // A = U(:, :k) S V(:, :k)^H
            Mat us(size_t(m) * k);
            for (int j = 0; j < k; j++)
                for (int i = 0; i < m; i++) us[i + size_t(j) * m] = u[i + size_t(j) * m] * s[j];
            const Mat vk(v.begin(), v.begin() + size_t(n) * k);
            const Mat rebuilt = mul(us, m, k, 'N', vk, n, k, 'C');
            EXPECT(diff(rebuilt, a) <= tol * maxAbs(a) * 8, "%s %s U S V^H = A (%dx%d): %g", typeName<T>(), what, m, n,
                   diff(rebuilt, a));
        };
        // gesvdj full and economy
        for (int econ = 0; econ < 2; econ++) {
            T *da = upload<T>(a), *du, *dv, *work;
            R *ds;
            const int ucols = econ ? k : m, vcols = econ ? k : n;
            CHECK(cudaMalloc(&du, sizeof(T) * m * ucols));
            CHECK(cudaMalloc(&dv, sizeof(T) * n * vcols));
            CHECK(cudaMalloc(&ds, sizeof(R) * k));
            gesvdjInfo_t params;
            SCHECK(cusolverDnCreateGesvdjInfo(&params));
            int lwork = 0;
            SCHECK(Api<T>::gesvdj_bufferSize(solver, CUSOLVER_EIG_MODE_VECTOR, econ, m, n, da, m, ds, du, m, dv, n,
                                             &lwork, params));
            CHECK(cudaMalloc(&work, sizeof(T) * lwork));
            SCHECK(Api<T>::gesvdj(solver, CUSOLVER_EIG_MODE_VECTOR, econ, m, n, da, m, ds, du, m, dv, n, work, lwork,
                                  d_info, params));
            EXPECT(checkInfo(d_info) == 0, "gesvdj info");
            check(download(du, size_t(m) * ucols), ucols, download(dv, size_t(n) * vcols), downloadReal(ds, k),
                  econ ? "gesvdj economy" : "gesvdj full");
            cusolverDnDestroyGesvdjInfo(params);
            cudaFree(da), cudaFree(du), cudaFree(dv), cudaFree(ds), cudaFree(work);
        }
        // gesvd ('A': U m x m, V^H n x n); cuSOLVER's gesvd needs m >= n
        if (m >= n) {
            T *da = upload<T>(a), *du, *dvt, *work;
            R *ds;
            CHECK(cudaMalloc(&du, sizeof(T) * m * m));
            CHECK(cudaMalloc(&dvt, sizeof(T) * n * n));
            CHECK(cudaMalloc(&ds, sizeof(R) * k));
            int lwork = 0;
            SCHECK(Api<T>::gesvd_bufferSize(solver, m, n, &lwork));
            CHECK(cudaMalloc(&work, sizeof(T) * lwork));
            SCHECK(Api<T>::gesvd(solver, 'A', 'A', m, n, da, m, ds, du, m, dvt, n, work, lwork, nullptr, d_info));
            const Mat vt = download(dvt, size_t(n) * n);
            Mat v(size_t(n) * n);
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++) v[i + size_t(j) * n] = std::conj(vt[j + size_t(i) * n]);
            check(download(du, size_t(m) * m), m, v, downloadReal(ds, k), "gesvd");
            cudaFree(da), cudaFree(du), cudaFree(dvt), cudaFree(ds), cudaFree(work);
        }
    }
    {
        // gesvdjBatched (cuSOLVER: up to 32 x 32)
        const int m = 20, n = 14, batch = 6;
        Mat all;
        for (int b = 0; b < batch; b++) {
            const Mat a = random<T>(m, n, 120 + b);
            all.insert(all.end(), a.begin(), a.end());
        }
        T *da = upload<T>(all), *du, *dv, *work;
        R *ds;
        int *infos;
        CHECK(cudaMalloc(&du, sizeof(T) * m * m * batch));
        CHECK(cudaMalloc(&dv, sizeof(T) * n * n * batch));
        CHECK(cudaMalloc(&ds, sizeof(R) * n * batch));
        CHECK(cudaMalloc(&infos, sizeof(int) * batch));
        gesvdjInfo_t params;
        SCHECK(cusolverDnCreateGesvdjInfo(&params));
        int lwork = 0;
        SCHECK(Api<T>::gesvdjBatched_bufferSize(solver, CUSOLVER_EIG_MODE_VECTOR, m, n, da, m, ds, du, m, dv, n,
                                                &lwork, params, batch));
        CHECK(cudaMalloc(&work, sizeof(T) * lwork));
        SCHECK(Api<T>::gesvdjBatched(solver, CUSOLVER_EIG_MODE_VECTOR, m, n, da, m, ds, du, m, dv, n, work, lwork,
                                     infos, params, batch));
        const Mat u = download(du, size_t(m) * m * batch), v = download(dv, size_t(n) * n * batch);
        const std::vector<double> s = downloadReal(ds, size_t(n) * batch);
        for (int b = 0; b < batch; b++) {
            const Mat ab(all.begin() + b * m * n, all.begin() + (b + 1) * m * n);
            Mat us(size_t(m) * n);
            for (int j = 0; j < n; j++)
                for (int i = 0; i < m; i++) us[i + j * m] = u[b * m * m + i + j * m] * s[b * n + j];
            const Mat vb(v.begin() + b * n * n, v.begin() + (b + 1) * n * n);
            EXPECT(diff(mul(us, m, n, 'N', vb, n, n, 'C'), ab) <= tol * maxAbs(ab) * 8, "%s gesvdjBatched %d",
                   typeName<T>(), b);
        }
        cusolverDnDestroyGesvdjInfo(params);
        cudaFree(da), cudaFree(du), cudaFree(dv), cudaFree(ds), cudaFree(work), cudaFree(infos);
    }
    cudaFree(d_info);
    printf("PASS cuSOLVER LU, Cholesky, QR, eigen decomposition, SVD (%s)\n", typeName<T>());
}

template<typename T> void all(cublasHandle_t blas, cusolverDnHandle_t solver) {
    blasTests<T>(blas);
    batchedTests<T>(blas);
    solverTests<T>(solver);
}

int main() {
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    BCHECK(cublasCreate(&blas));
    SCHECK(cusolverDnCreate(&solver));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    BCHECK(cublasSetStream(blas, stream));
    SCHECK(cusolverDnSetStream(solver, stream));
    all<float>(blas, solver);
    all<double>(blas, solver);
    all<cuComplex>(blas, solver);
    all<cuDoubleComplex>(blas, solver);
    SCHECK(cusolverDnDestroy(solver));
    BCHECK(cublasDestroy(blas));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaGetLastError());
    return 0;
}
