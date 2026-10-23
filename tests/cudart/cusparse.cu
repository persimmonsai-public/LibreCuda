// NVCC_FLAGS: -arch=$ARCH -lcusparse
// cuSPARSE against dense CPU references: SpMV and SpMM (CSR, CSC and row-sorted COO, all operations, 32 / 64 bit
// indices, base 0 / 1, row and column major dense matrices, strided batches, half precision, device scalars), SpGEMM,
// SDDMM, triangular solves (SpSV, SpSM: lower / upper, unit / non-unit, transposed), sparse / dense conversions,
// CSR to CSC, sparse vectors, and the legacy functions PyTorch uses (csrgeam2, COO / CSR sorting, coo2csr, BSR
// products and solves). Checked against NVIDIA's cuSPARSE too (CUSPARSE_TEST_NVIDIA=1).
#include "check.h"
#include <cusparse.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <complex>
#include <cstring>
#include <random>
#include <vector>

#define SCHECK(call) do { \
    cusparseStatus_t scheck_status = (call); \
    if (scheck_status != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE error %d at %s:%d: %s\n", int(scheck_status), __FILE__, __LINE__, #call); \
        exit(1); \
    } \
} while (0)

using cd = std::complex<double>;

static std::mt19937 rng(1234);

static double uniform() {
    return std::uniform_real_distribution<double>(-1, 1)(rng);
}

// value type helpers: conversion from / to the reference's complex<double>
template<typename T> struct V;
template<> struct V<float> {
    static constexpr cudaDataType type = CUDA_R_32F;
    static float from(cd v) { return float(v.real()); }
    static cd to(float v) { return v; }
    static cd random() { return float(uniform()); }
    static constexpr double tol = 2e-4;
};
template<> struct V<double> {
    static constexpr cudaDataType type = CUDA_R_64F;
    static double from(cd v) { return v.real(); }
    static cd to(double v) { return v; }
    static cd random() { return uniform(); }
    static constexpr double tol = 1e-10;
};
template<> struct V<std::complex<float>> {
    static constexpr cudaDataType type = CUDA_C_32F;
    static std::complex<float> from(cd v) { return std::complex<float>(v); }
    static cd to(std::complex<float> v) { return cd(v); }
    static cd random() { return cd(float(uniform()), float(uniform())); }
    static constexpr double tol = 2e-4;
};
template<> struct V<std::complex<double>> {
    static constexpr cudaDataType type = CUDA_C_64F;
    static cd from(cd v) { return v; }
    static cd to(cd v) { return v; }
    static cd random() { return cd(uniform(), uniform()); }
    static constexpr double tol = 1e-10;
};
template<> struct V<__half> {
    static constexpr cudaDataType type = CUDA_R_16F;
    static __half from(cd v) { return __float2half(float(v.real())); }
    static cd to(__half v) { return __half2float(v); }
    static cd random() { return __half2float(__float2half(float(uniform()))); }
    static constexpr double tol = 2e-2;
};

// a dense matrix (row-major reference) with roughly `density` of its entries nonzero
struct Matrix {
    int rows, cols;
    std::vector<cd> a;

    cd &at(int r, int c) { return a[size_t(r) * cols + c]; }
    cd at(int r, int c) const { return a[size_t(r) * cols + c]; }
};

template<typename T>
static Matrix randomSparse(int rows, int cols, double density) {
    Matrix m{rows, cols, std::vector<cd>(size_t(rows) * cols)};
    for (cd &v: m.a) {
        if (std::uniform_real_distribution<double>(0, 1)(rng) < density) v = V<T>::random();
    }
    return m;
}

template<typename T>
static Matrix randomDense(int rows, int cols) {
    return randomSparse<T>(rows, cols, 2.0);
}

static cd opAt(const Matrix &m, cusparseOperation_t op, int r, int c) {
    if (op == CUSPARSE_OPERATION_NON_TRANSPOSE) return m.at(r, c);
    return op == CUSPARSE_OPERATION_TRANSPOSE ? m.at(c, r) : std::conj(m.at(c, r));
}

template<typename T>
static void *upload(const std::vector<T> &h) {
    void *d = nullptr;
    CHECK(cudaMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(T)));
    if (!h.empty()) CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

template<typename T>
static std::vector<T> download(const void *d, size_t n) {
    std::vector<T> h(n);
    if (n > 0) CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

// a sparse matrix on the device in CSR, CSC or COO format (COO sorted by row, as cuSPARSE requires, or shuffled)
template<typename T, typename I>
struct Sparse {
    void *p0 = nullptr, *p1 = nullptr, *values = nullptr;
    int64_t nnz = 0;
    cusparseSpMatDescr_t descr = nullptr;

    Sparse(const Matrix &m, cusparseFormat_t format, int base, bool shuffle = false) {
        std::vector<I> i0, i1;
        std::vector<T> v;
        const bool byColumn = format == CUSPARSE_FORMAT_CSC;
        const int majors = byColumn ? m.cols : m.rows, minors = byColumn ? m.rows : m.cols;
        std::vector<I> offsets(1, I(base));
        for (int a = 0; a < majors; a++) {
            for (int b = 0; b < minors; b++) {
                const cd x = byColumn ? m.at(b, a) : m.at(a, b);
                if (x == cd(0)) continue;
                if (format == CUSPARSE_FORMAT_COO) i0.push_back(I(a + base));
                i1.push_back(I(b + base));
                v.push_back(V<T>::from(x));
            }
            offsets.push_back(I(i1.size() + base));
        }
        nnz = int64_t(v.size());
        if (format == CUSPARSE_FORMAT_COO && shuffle) {
            std::vector<size_t> perm(v.size());
            for (size_t k = 0; k < perm.size(); k++) perm[k] = k;
            std::shuffle(perm.begin(), perm.end(), rng);
            std::vector<I> s0(i0.size()), s1(i1.size());
            std::vector<T> sv(v.size());
            for (size_t k = 0; k < perm.size(); k++) {
                s0[k] = i0[perm[k]];
                s1[k] = i1[perm[k]];
                sv[k] = v[perm[k]];
            }
            i0 = s0;
            i1 = s1;
            v = sv;
        }
        p0 = upload(format == CUSPARSE_FORMAT_COO ? i0 : offsets);
        p1 = upload(i1);
        values = upload(v);
        const cusparseIndexType_t it = sizeof(I) == 8 ? CUSPARSE_INDEX_64I : CUSPARSE_INDEX_32I;
        const cusparseIndexBase_t b = base ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO;
        if (format == CUSPARSE_FORMAT_CSR) {
            SCHECK(cusparseCreateCsr(&descr, m.rows, m.cols, nnz, p0, p1, values, it, it, b, V<T>::type));
        } else if (format == CUSPARSE_FORMAT_CSC) {
            SCHECK(cusparseCreateCsc(&descr, m.rows, m.cols, nnz, p0, p1, values, it, it, b, V<T>::type));
        } else {
            SCHECK(cusparseCreateCoo(&descr, m.rows, m.cols, nnz, p0, p1, values, it, b, V<T>::type));
        }
    }

    ~Sparse() {
        cusparseDestroySpMat(descr);
        cudaFree(p0);
        cudaFree(p1);
        cudaFree(values);
    }
};

// a dense matrix on the device (column or row major, leading dimension with padding)
template<typename T>
struct DenseDev {
    void *values;
    int64_t ld;
    bool rowMajor;
    int rows, cols;
    cusparseDnMatDescr_t descr;

    DenseDev(const Matrix &m, bool rowMajor_, int pad = 3) : rowMajor(rowMajor_), rows(m.rows), cols(m.cols) {
        ld = (rowMajor ? m.cols : m.rows) + pad;
        std::vector<T> h(size_t(ld) * (rowMajor ? m.rows : m.cols), V<T>::from(cd(7)));
        for (int r = 0; r < m.rows; r++) {
            for (int c = 0; c < m.cols; c++) h[rowMajor ? r * ld + c : r + c * ld] = V<T>::from(m.at(r, c));
        }
        values = upload(h);
        SCHECK(cusparseCreateDnMat(&descr, m.rows, m.cols, ld, values, V<T>::type,
                                   rowMajor ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL));
    }

    Matrix get() const {
        std::vector<T> h = download<T>(values, size_t(ld) * (rowMajor ? rows : cols));
        Matrix m{rows, cols, std::vector<cd>(size_t(rows) * cols)};
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) m.at(r, c) = V<T>::to(h[rowMajor ? r * ld + c : r + c * ld]);
        }
        return m;
    }

    ~DenseDev() {
        cusparseDestroyDnMat(descr);
        cudaFree(values);
    }
};

static void expectClose(const Matrix &got, const Matrix &want, double tol, const char *what) {
    double scale = 1;
    for (const cd &v: want.a) scale = std::max(scale, std::abs(v));
    for (int r = 0; r < want.rows; r++) {
        for (int c = 0; c < want.cols; c++) {
            EXPECT(std::abs(got.at(r, c) - want.at(r, c)) <= tol * scale, "%s: (%d, %d) = (%g, %g), expected (%g, %g)",
                   what, r, c, got.at(r, c).real(), got.at(r, c).imag(), want.at(r, c).real(), want.at(r, c).imag());
        }
    }
}

static const char *opName(cusparseOperation_t op) {
    return op == CUSPARSE_OPERATION_NON_TRANSPOSE ? "N" : op == CUSPARSE_OPERATION_TRANSPOSE ? "T" : "C";
}

static const char *formatName(cusparseFormat_t f) {
    return f == CUSPARSE_FORMAT_CSR ? "CSR" : f == CUSPARSE_FORMAT_CSC ? "CSC" : "COO";
}

static cudaDataType computeOf(cudaDataType t) {
    return t == CUDA_R_16F ? CUDA_R_32F : t;
}

// ---- SpMV and SpMM

template<typename T, typename I>
static void spmvCase(cusparseHandle_t h, cusparseFormat_t format, cusparseOperation_t op, int base) {
    if (op == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE && (std::is_same_v<T, float> || std::is_same_v<T, double>))
        return;
    const int rows = 70, cols = 53;
    const Matrix a = randomSparse<T>(rows, cols, 0.15);
    const int xn = op == CUSPARSE_OPERATION_NON_TRANSPOSE ? cols : rows, yn = op == CUSPARSE_OPERATION_NON_TRANSPOSE ? rows : cols;
    const Matrix x = randomDense<T>(xn, 1), y0 = randomDense<T>(yn, 1);
    Sparse<T, I> s(a, format, base);
    DenseDev<T> xd(x, false, 0), yd(y0, false, 0);
    cusparseDnVecDescr_t xv, yv;
    SCHECK(cusparseCreateDnVec(&xv, xn, xd.values, V<T>::type));
    SCHECK(cusparseCreateDnVec(&yv, yn, yd.values, V<T>::type));
    using S = std::conditional_t<std::is_same_v<T, __half>, float, T>;
    const S alpha = V<S>::from(cd(1.5, 0.25)), beta = V<S>::from(cd(-0.5, 0.5));
    size_t bytes = 0;
    SCHECK(cusparseSpMV_bufferSize(h, op, &alpha, s.descr, xv, &beta, yv, computeOf(V<T>::type),
                                   CUSPARSE_SPMV_ALG_DEFAULT, &bytes));
    void *buffer;
    CHECK(cudaMalloc(&buffer, std::max<size_t>(bytes, 1)));
    SCHECK(cusparseSpMV(h, op, &alpha, s.descr, xv, &beta, yv, computeOf(V<T>::type), CUSPARSE_SPMV_ALG_DEFAULT,
                        buffer));
    Matrix want = y0;
    for (int r = 0; r < yn; r++) {
        cd acc = 0;
        for (int k = 0; k < xn; k++) acc += opAt(a, op, r, k) * x.at(k, 0);
        want.at(r, 0) = V<S>::to(alpha) * acc + V<S>::to(beta) * y0.at(r, 0);
    }
    char what[96];
    snprintf(what, sizeof(what), "SpMV %s op %s, %zu byte indices, base %d", formatName(format), opName(op),
             sizeof(I), base);
    expectClose(yd.get(), want, V<T>::tol, what);
    CHECK(cudaFree(buffer));
    cusparseDestroyDnVec(xv);
    cusparseDestroyDnVec(yv);
}

template<typename T, typename I>
static void spmmCase(cusparseHandle_t h, cusparseFormat_t format, cusparseOperation_t opA, cusparseOperation_t opB,
                     bool bRowMajor, bool cRowMajor) {
    if ((opA == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE || opB == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE) &&
        !(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>))
        return;
    const int m = 37, k = 45, n = 70;
    const bool ta = opA != CUSPARSE_OPERATION_NON_TRANSPOSE, tb = opB != CUSPARSE_OPERATION_NON_TRANSPOSE;
    const Matrix a = randomSparse<T>(ta ? k : m, ta ? m : k, 0.2);
    const Matrix b = randomDense<T>(tb ? n : k, tb ? k : n), c0 = randomDense<T>(m, n);
    Sparse<T, I> s(a, format, 0);
    DenseDev<T> bd(b, bRowMajor), cd_(c0, cRowMajor);
    using S = std::conditional_t<std::is_same_v<T, __half>, float, T>;
    const S alpha = V<S>::from(cd(0.75, -0.5)), beta = V<S>::from(cd(0.5, 0.25));
    size_t bytes = 0;
    SCHECK(cusparseSpMM_bufferSize(h, opA, opB, &alpha, s.descr, bd.descr, &beta, cd_.descr, computeOf(V<T>::type),
                                   CUSPARSE_SPMM_ALG_DEFAULT, &bytes));
    void *buffer;
    CHECK(cudaMalloc(&buffer, std::max<size_t>(bytes, 1)));
    SCHECK(cusparseSpMM(h, opA, opB, &alpha, s.descr, bd.descr, &beta, cd_.descr, computeOf(V<T>::type),
                        CUSPARSE_SPMM_ALG_DEFAULT, buffer));
    Matrix want = c0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cd acc = 0;
            for (int q = 0; q < k; q++) acc += opAt(a, opA, i, q) * opAt(b, opB, q, j);
            want.at(i, j) = V<S>::to(alpha) * acc + V<S>::to(beta) * c0.at(i, j);
        }
    }
    char what[128];
    snprintf(what, sizeof(what), "SpMM %s op %s%s, B %s, C %s", formatName(format), opName(opA), opName(opB),
             bRowMajor ? "row" : "col", cRowMajor ? "row" : "col");
    expectClose(cd_.get(), want, V<T>::tol * 4, what);
    CHECK(cudaFree(buffer));
}

template<typename T>
static void productsOf(cusparseHandle_t h) {
    const cusparseFormat_t formats[] = {CUSPARSE_FORMAT_CSR, CUSPARSE_FORMAT_CSC, CUSPARSE_FORMAT_COO};
    const cusparseOperation_t ops[] = {CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                                       CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE};
    for (cusparseFormat_t f: formats) {
        for (cusparseOperation_t op: ops) {
            spmvCase<T, int>(h, f, op, 0);
            spmvCase<T, int64_t>(h, f, op, 1);
            for (cusparseOperation_t opB: ops) {
                spmmCase<T, int>(h, f, op, opB, false, false);
            }
        }
        spmmCase<T, int64_t>(h, f, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, true, true);
        spmmCase<T, int>(h, f, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, true, false);
    }
}

// strided batches of CSR, B and C (PyTorch's batched sparse matmul): every batch with its own row offsets, or one
// set of row offsets for all batches (NVIDIA's batched sample; NVIDIA's cuSPARSE reads only the first batch's offsets,
// so CUSPARSE_TEST_NVIDIA=1 skips the other case)
static void batchedSpmm(cusparseHandle_t h, bool sharedOffsets) {
    const char *nv = getenv("CUSPARSE_TEST_NVIDIA");
    if (!sharedOffsets && nv != nullptr && strcmp(nv, "1") == 0) return;
    const int batches = 3, m = 20, k = 30, n = 9, perRow = 6;
    std::vector<Matrix> a, b, c;
    std::vector<int> offsets, columns, rowCounts(m);
    std::vector<float> values, bh, ch;
    for (int r = 0; r < m; r++) rowCounts[r] = (r * 5) % (perRow + 1);
    for (int i = 0; i < batches; i++) {
        // row r has rowCounts[r] entries (shared offsets) or a batch-dependent number of them
        Matrix x{m, k, std::vector<cd>(size_t(m) * k)};
        for (int r = 0; r < m; r++) {
            const int count = sharedOffsets ? rowCounts[r] : (r * 5 + i * 3) % (perRow + 1);
            std::vector<int> cols(k);
            for (int q = 0; q < k; q++) cols[q] = q;
            std::shuffle(cols.begin(), cols.end(), rng);
            std::sort(cols.begin(), cols.begin() + count);
            for (int e = 0; e < count; e++) x.at(r, cols[e]) = float(uniform());
        }
        a.push_back(x);
    }
    int nnzMax = 0;
    for (const Matrix &x: a) {
        int nnz = 0;
        for (const cd &v: x.a) nnz += v != cd(0);
        nnzMax = std::max(nnzMax, nnz);
    }
    // every batch padded to nnzMax entries (the last row takes the padding as explicit zeros)
    for (int i = 0; i < batches; i++) {
        int nnz = 0;
        if (!sharedOffsets || i == 0) offsets.push_back(0);
        for (int r = 0; r < m; r++) {
            for (int q = 0; q < k; q++) {
                if (a[i].at(r, q) == cd(0)) continue;
                columns.push_back(q);
                values.push_back(float(a[i].at(r, q).real()));
                nnz++;
            }
            if (r == m - 1) {
                for (; nnz < nnzMax; nnz++) {
                    columns.push_back(0);
                    values.push_back(0);
                }
            }
            if (!sharedOffsets || i == 0) offsets.push_back(nnz);
        }
        b.push_back(randomDense<float>(k, n));
        c.push_back(randomDense<float>(m, n));
        for (int q = 0; q < k * n; q++) bh.push_back(float(b[i].a[(q % k) * n + q / k].real()));  // column major
        for (int q = 0; q < m * n; q++) ch.push_back(float(c[i].a[(q % m) * n + q / m].real()));
    }
    void *dOff = upload(offsets), *dCol = upload(columns), *dVal = upload(values), *dB = upload(bh), *dC = upload(ch);
    cusparseSpMatDescr_t sa;
    cusparseDnMatDescr_t db, dc;
    SCHECK(cusparseCreateCsr(&sa, m, k, nnzMax, dOff, dCol, dVal, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                             CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    SCHECK(cusparseCsrSetStridedBatch(sa, batches, sharedOffsets ? 0 : m + 1, nnzMax));
    SCHECK(cusparseCreateDnMat(&db, k, n, k, dB, CUDA_R_32F, CUSPARSE_ORDER_COL));
    SCHECK(cusparseDnMatSetStridedBatch(db, batches, int64_t(k) * n));
    SCHECK(cusparseCreateDnMat(&dc, m, n, m, dC, CUDA_R_32F, CUSPARSE_ORDER_COL));
    SCHECK(cusparseDnMatSetStridedBatch(dc, batches, int64_t(m) * n));
    const float alpha = 2, beta = -1;
    size_t bytes = 0;
    SCHECK(cusparseSpMM_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, sa,
                                   db, &beta, dc, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, &bytes));
    void *buffer;
    CHECK(cudaMalloc(&buffer, std::max<size_t>(bytes, 1)));
    SCHECK(cusparseSpMM(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, sa, db, &beta,
                        dc, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, buffer));
    std::vector<float> got = download<float>(dC, ch.size());
    for (int i = 0; i < batches; i++) {
        for (int r = 0; r < m; r++) {
            for (int j = 0; j < n; j++) {
                cd acc = 0;
                for (int q = 0; q < k; q++) acc += a[i].at(r, q) * b[i].at(q, j);
                const double want = 2 * acc.real() - c[i].at(r, j).real();
                EXPECT(std::fabs(got[size_t(i) * m * n + r + j * m] - want) < 1e-4, "batched SpMM %s offsets: %d (%d, %d)",
                       sharedOffsets ? "shared" : "own", i, r, j);
            }
        }
    }
    cusparseDestroySpMat(sa);
    cusparseDestroyDnMat(db);
    cusparseDestroyDnMat(dc);
    for (void *p: {dOff, dCol, dVal, dB, dC, buffer}) CHECK(cudaFree(p));
}

// ---- SpGEMM

template<typename T, typename I>
static void spgemmCase(cusparseHandle_t h, int base) {
    const int m = 60, k = 45, n = 50;
    const Matrix a = randomSparse<T>(m, k, 0.1), b = randomSparse<T>(k, n, 0.1);
    Sparse<T, I> sa(a, CUSPARSE_FORMAT_CSR, base), sb(b, CUSPARSE_FORMAT_CSR, base);
    const cusparseIndexType_t it = sizeof(I) == 8 ? CUSPARSE_INDEX_64I : CUSPARSE_INDEX_32I;
    const cusparseIndexBase_t ib = base ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO;
    cusparseSpMatDescr_t c;
    void *cOff;
    CHECK(cudaMalloc(&cOff, (m + 1) * sizeof(I)));
    SCHECK(cusparseCreateCsr(&c, m, n, 0, cOff, nullptr, nullptr, it, it, ib, V<T>::type));
    cusparseSpGEMMDescr_t g;
    SCHECK(cusparseSpGEMM_createDescr(&g));
    const T alpha = V<T>::from(cd(1.25, 0.5)), beta = V<T>::from(0);
    const auto N = CUSPARSE_OPERATION_NON_TRANSPOSE;
    size_t size1 = 0, size2 = 0;
    void *b1 = nullptr, *b2 = nullptr;
    SCHECK(cusparseSpGEMM_workEstimation(h, N, N, &alpha, sa.descr, sb.descr, &beta, c, V<T>::type,
                                         CUSPARSE_SPGEMM_DEFAULT, g, &size1, nullptr));
    CHECK(cudaMalloc(&b1, std::max<size_t>(size1, 1)));
    SCHECK(cusparseSpGEMM_workEstimation(h, N, N, &alpha, sa.descr, sb.descr, &beta, c, V<T>::type,
                                         CUSPARSE_SPGEMM_DEFAULT, g, &size1, b1));
    SCHECK(cusparseSpGEMM_compute(h, N, N, &alpha, sa.descr, sb.descr, &beta, c, V<T>::type, CUSPARSE_SPGEMM_DEFAULT,
                                  g, &size2, nullptr));
    CHECK(cudaMalloc(&b2, std::max<size_t>(size2, 1)));
    SCHECK(cusparseSpGEMM_compute(h, N, N, &alpha, sa.descr, sb.descr, &beta, c, V<T>::type, CUSPARSE_SPGEMM_DEFAULT,
                                  g, &size2, b2));
    int64_t rows, cols, nnz;
    SCHECK(cusparseSpMatGetSize(c, &rows, &cols, &nnz));
    EXPECT(rows == m && cols == n, "SpGEMM size %lld x %lld", (long long) rows, (long long) cols);
    void *cCol, *cVal;
    CHECK(cudaMalloc(&cCol, std::max<int64_t>(nnz, 1) * sizeof(I)));
    CHECK(cudaMalloc(&cVal, std::max<int64_t>(nnz, 1) * sizeof(T)));
    SCHECK(cusparseCsrSetPointers(c, cOff, cCol, cVal));
    SCHECK(cusparseSpGEMM_copy(h, N, N, &alpha, sa.descr, sb.descr, &beta, c, V<T>::type, CUSPARSE_SPGEMM_DEFAULT, g));
    std::vector<I> off = download<I>(cOff, m + 1), col = download<I>(cCol, nnz);
    std::vector<T> val = download<T>(cVal, nnz);
    Matrix got{m, n, std::vector<cd>(size_t(m) * n)}, want = got;
    int64_t pattern = 0;
    for (int r = 0; r < m; r++) {
        for (I e = off[r] - base; e < off[r + 1] - base; e++) got.at(r, int(col[e] - base)) += V<T>::to(val[e]);
        for (int j = 0; j < n; j++) {
            cd acc = 0;
            bool structural = false;
            for (int q = 0; q < k; q++) {
                acc += a.at(r, q) * b.at(q, j);
                structural |= a.at(r, q) != cd(0) && b.at(q, j) != cd(0);
            }
            want.at(r, j) = V<T>::to(alpha) * acc;
            pattern += structural;
        }
    }
    EXPECT(nnz == pattern, "SpGEMM nnz %lld, expected %lld", (long long) nnz, (long long) pattern);
    expectClose(got, want, V<T>::tol * 4, "SpGEMM");
    SCHECK(cusparseSpGEMM_destroyDescr(g));
    cusparseDestroySpMat(c);
    for (void *p: {cOff, cCol, cVal, b1, b2}) CHECK(cudaFree(p));
}

// ---- SDDMM

template<typename T>
static void sddmmCase(cusparseHandle_t h, cusparseOperation_t opA, cusparseOperation_t opB) {
    const int m = 31, k = 17, n = 40;
    const Matrix pattern = randomSparse<T>(m, n, 0.2);
    const bool ta = opA != CUSPARSE_OPERATION_NON_TRANSPOSE, tb = opB != CUSPARSE_OPERATION_NON_TRANSPOSE;
    const Matrix a = randomDense<T>(ta ? k : m, ta ? m : k), b = randomDense<T>(tb ? n : k, tb ? k : n);
    Sparse<T, int> c(pattern, CUSPARSE_FORMAT_CSR, 0);
    DenseDev<T> ad(a, false), bd(b, true);
    const T alpha = V<T>::from(cd(0.5, 1)), beta = V<T>::from(cd(2, 0));
    size_t bytes = 0;
    SCHECK(cusparseSDDMM_bufferSize(h, opA, opB, &alpha, ad.descr, bd.descr, &beta, c.descr, V<T>::type,
                                    CUSPARSE_SDDMM_ALG_DEFAULT, &bytes));
    void *buffer;
    CHECK(cudaMalloc(&buffer, std::max<size_t>(bytes, 1)));
    SCHECK(cusparseSDDMM_preprocess(h, opA, opB, &alpha, ad.descr, bd.descr, &beta, c.descr, V<T>::type,
                                    CUSPARSE_SDDMM_ALG_DEFAULT, buffer));
    SCHECK(cusparseSDDMM(h, opA, opB, &alpha, ad.descr, bd.descr, &beta, c.descr, V<T>::type,
                         CUSPARSE_SDDMM_ALG_DEFAULT, buffer));
    std::vector<int> off = download<int>(c.p0, m + 1), col = download<int>(c.p1, c.nnz);
    std::vector<T> val = download<T>(c.values, c.nnz);
    for (int r = 0; r < m; r++) {
        for (int e = off[r]; e < off[r + 1]; e++) {
            cd acc = 0;
            for (int q = 0; q < k; q++) acc += opAt(a, opA, r, q) * opAt(b, opB, q, col[e]);
            const cd want = V<T>::to(alpha) * acc + V<T>::to(beta) * pattern.at(r, col[e]);
            EXPECT(std::abs(V<T>::to(val[e]) - want) <= V<T>::tol * 10 * (1 + std::abs(want)), "SDDMM (%d, %d) %s%s", r,
                   col[e], opName(opA), opName(opB));
        }
    }
    CHECK(cudaFree(buffer));
}

// ---- triangular solves

// a triangular matrix with a dominant diagonal (stored: both triangles, the solve uses one)
template<typename T>
static Matrix triangular(int n) {
    Matrix m = randomSparse<T>(n, n, 0.15);
    for (int i = 0; i < n; i++) m.at(i, i) = V<T>::from(cd(4 + std::fabs(uniform()), 0.5 * uniform()));
    return m;
}

static Matrix usedTriangle(const Matrix &a, bool lower, bool unit) {
    Matrix t = a;
    for (int r = 0; r < a.rows; r++) {
        for (int c = 0; c < a.cols; c++) {
            if ((lower && c > r) || (!lower && c < r)) t.at(r, c) = 0;
            if (r == c && unit) t.at(r, c) = 1;
        }
    }
    return t;
}

// op(T) X = alpha B by substitution on the dense reference
static Matrix solveDense(const Matrix &t, cusparseOperation_t op, const Matrix &b, cd alpha) {
    const int n = t.rows;
    Matrix x{n, b.cols, std::vector<cd>(size_t(n) * b.cols)};
    // op(T) is lower or upper: solve by Gaussian elimination on the dense matrix (it's triangular after op)
    Matrix m{n, n, std::vector<cd>(size_t(n) * n)};
    for (int r = 0; r < n; r++) for (int c = 0; c < n; c++) m.at(r, c) = opAt(t, op, r, c);
    bool lower = true;
    for (int r = 0; r < n; r++) for (int c = r + 1; c < n; c++) if (m.at(r, c) != cd(0)) lower = false;
    for (int j = 0; j < b.cols; j++) {
        for (int s = 0; s < n; s++) {
            const int r = lower ? s : n - 1 - s;
            cd acc = alpha * b.at(r, j);
            for (int c = 0; c < n; c++) {
                if (c != r && m.at(r, c) != cd(0)) acc -= m.at(r, c) * x.at(c, j);
            }
            x.at(r, j) = acc / m.at(r, r);
        }
    }
    return x;
}

template<typename T>
static void spsvCase(cusparseHandle_t h, cusparseFormat_t format, cusparseOperation_t op, bool lower, bool unit) {
    const int n = 300;
    const Matrix a = triangular<T>(n);
    const Matrix b = randomDense<T>(n, 1);
    Sparse<T, int> s(a, format, 0);
    cusparseFillMode_t fill = lower ? CUSPARSE_FILL_MODE_LOWER : CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t diag = unit ? CUSPARSE_DIAG_TYPE_UNIT : CUSPARSE_DIAG_TYPE_NON_UNIT;
    SCHECK(cusparseSpMatSetAttribute(s.descr, CUSPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill)));
    SCHECK(cusparseSpMatSetAttribute(s.descr, CUSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag)));
    DenseDev<T> bd(b, false, 0), xd(randomDense<T>(n, 1), false, 0);
    cusparseDnVecDescr_t bv, xv;
    SCHECK(cusparseCreateDnVec(&bv, n, bd.values, V<T>::type));
    SCHECK(cusparseCreateDnVec(&xv, n, xd.values, V<T>::type));
    cusparseSpSVDescr_t d;
    SCHECK(cusparseSpSV_createDescr(&d));
    const T alpha = V<T>::from(cd(1.5, 0));
    size_t bytes = 0;
    SCHECK(cusparseSpSV_bufferSize(h, op, &alpha, s.descr, bv, xv, V<T>::type, CUSPARSE_SPSV_ALG_DEFAULT, d, &bytes));
    void *buffer;
    CHECK(cudaMalloc(&buffer, std::max<size_t>(bytes, 1)));
    SCHECK(cusparseSpSV_analysis(h, op, &alpha, s.descr, bv, xv, V<T>::type, CUSPARSE_SPSV_ALG_DEFAULT, d, buffer));
    SCHECK(cusparseSpSV_solve(h, op, &alpha, s.descr, bv, xv, V<T>::type, CUSPARSE_SPSV_ALG_DEFAULT, d));
    char what[96];
    snprintf(what, sizeof(what), "SpSV %s op %s %s %s", formatName(format), opName(op), lower ? "lower" : "upper",
             unit ? "unit" : "non-unit");
    expectClose(xd.get(), solveDense(usedTriangle(a, lower, unit), op, b, V<T>::to(alpha)), V<T>::tol * 10, what);
    SCHECK(cusparseSpSV_destroyDescr(d));
    cusparseDestroyDnVec(bv);
    cusparseDestroyDnVec(xv);
    CHECK(cudaFree(buffer));
}

template<typename T>
static void spsmCase(cusparseHandle_t h, cusparseOperation_t opA, cusparseOperation_t opB, bool lower, bool rowMajor) {
    const int n = 200, rhs = 37;
    const Matrix a = triangular<T>(n);
    const bool tb = opB != CUSPARSE_OPERATION_NON_TRANSPOSE;
    const Matrix b = randomDense<T>(tb ? rhs : n, tb ? n : rhs);
    Sparse<T, int64_t> s(a, CUSPARSE_FORMAT_CSR, 0);
    cusparseFillMode_t fill = lower ? CUSPARSE_FILL_MODE_LOWER : CUSPARSE_FILL_MODE_UPPER;
    SCHECK(cusparseSpMatSetAttribute(s.descr, CUSPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill)));
    DenseDev<T> bd(b, rowMajor), xd(randomDense<T>(n, rhs), rowMajor);
    cusparseSpSMDescr_t d;
    SCHECK(cusparseSpSM_createDescr(&d));
    const T alpha = V<T>::from(cd(-0.5, 0));
    size_t bytes = 0;
    SCHECK(cusparseSpSM_bufferSize(h, opA, opB, &alpha, s.descr, bd.descr, xd.descr, V<T>::type,
                                   CUSPARSE_SPSM_ALG_DEFAULT, d, &bytes));
    void *buffer;
    CHECK(cudaMalloc(&buffer, std::max<size_t>(bytes, 1)));
    SCHECK(cusparseSpSM_analysis(h, opA, opB, &alpha, s.descr, bd.descr, xd.descr, V<T>::type,
                                 CUSPARSE_SPSM_ALG_DEFAULT, d, buffer));
    SCHECK(cusparseSpSM_solve(h, opA, opB, &alpha, s.descr, bd.descr, xd.descr, V<T>::type, CUSPARSE_SPSM_ALG_DEFAULT,
                              d));
    Matrix bop{n, rhs, std::vector<cd>(size_t(n) * rhs)};
    for (int r = 0; r < n; r++) for (int j = 0; j < rhs; j++) bop.at(r, j) = opAt(b, opB, r, j);
    char what[96];
    snprintf(what, sizeof(what), "SpSM op %s%s %s %s", opName(opA), opName(opB), lower ? "lower" : "upper",
             rowMajor ? "row" : "col");
    expectClose(xd.get(), solveDense(usedTriangle(a, lower, false), opA, bop, V<T>::to(alpha)), V<T>::tol * 10, what);
    SCHECK(cusparseSpSM_destroyDescr(d));
    CHECK(cudaFree(buffer));
}

template<typename T>
static void solves(cusparseHandle_t h) {
    for (cusparseFormat_t f: {CUSPARSE_FORMAT_CSR, CUSPARSE_FORMAT_COO}) {
        for (cusparseOperation_t op: {CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE}) {
            for (bool lower: {true, false}) {
                for (bool unit: {false, true}) spsvCase<T>(h, f, op, lower, unit);
            }
        }
    }
    for (cusparseOperation_t opA: {CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE}) {
        for (cusparseOperation_t opB: {CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE}) {
            for (bool lower: {true, false}) spsmCase<T>(h, opA, opB, lower, false);
        }
    }
    spsmCase<T>(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, true, true);
}

// ---- conversions and vectors

static void conversions(cusparseHandle_t h) {
    const int m = 23, n = 31;
    const Matrix a = randomSparse<double>(m, n, 0.2);
    // sparse to dense, every format
    for (cusparseFormat_t f: {CUSPARSE_FORMAT_CSR, CUSPARSE_FORMAT_CSC, CUSPARSE_FORMAT_COO}) {
        Sparse<double, int> s(a, f, 1);
        DenseDev<double> d(randomDense<double>(m, n), f == CUSPARSE_FORMAT_CSC);
        size_t bytes = 0;
        SCHECK(cusparseSparseToDense_bufferSize(h, s.descr, d.descr, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &bytes));
        void *buffer;
        CHECK(cudaMalloc(&buffer, std::max<size_t>(bytes, 1)));
        SCHECK(cusparseSparseToDense(h, s.descr, d.descr, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, buffer));
        expectClose(d.get(), a, 0, formatName(f));
        CHECK(cudaFree(buffer));
    }
    // dense to CSR
    {
        DenseDev<double> d(a, true);
        void *off;
        CHECK(cudaMalloc(&off, (m + 1) * sizeof(int64_t)));
        cusparseSpMatDescr_t s;
        SCHECK(cusparseCreateCsr(&s, m, n, 0, off, nullptr, nullptr, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                                 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        size_t bytes = 0;
        SCHECK(cusparseDenseToSparse_bufferSize(h, d.descr, s, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bytes));
        void *buffer;
        CHECK(cudaMalloc(&buffer, std::max<size_t>(bytes, 1)));
        SCHECK(cusparseDenseToSparse_analysis(h, d.descr, s, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, buffer));
        int64_t rows, cols, nnz;
        SCHECK(cusparseSpMatGetSize(s, &rows, &cols, &nnz));
        void *col, *val;
        CHECK(cudaMalloc(&col, std::max<int64_t>(nnz, 1) * 8));
        CHECK(cudaMalloc(&val, std::max<int64_t>(nnz, 1) * 8));
        SCHECK(cusparseCsrSetPointers(s, off, col, val));
        SCHECK(cusparseDenseToSparse_convert(h, d.descr, s, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, buffer));
        std::vector<int64_t> ho = download<int64_t>(off, m + 1), hc = download<int64_t>(col, nnz);
        std::vector<double> hv = download<double>(val, nnz);
        Matrix got{m, n, std::vector<cd>(size_t(m) * n)};
        for (int r = 0; r < m; r++) for (int64_t e = ho[r]; e < ho[r + 1]; e++) got.at(r, int(hc[e])) = hv[e];
        expectClose(got, a, 0, "dense to CSR");
        cusparseDestroySpMat(s);
        for (void *p: {off, col, val, buffer}) CHECK(cudaFree(p));
    }
    // CSR to CSC (base 1)
    {
        Sparse<float, int> s(randomSparse<float>(m, n, 0.25), CUSPARSE_FORMAT_CSR, 1);
        std::vector<int> ho = download<int>(s.p0, m + 1), hc = download<int>(s.p1, s.nnz);
        std::vector<float> hv = download<float>(s.values, s.nnz);
        void *cOff, *cRow, *cVal;
        CHECK(cudaMalloc(&cOff, (n + 1) * 4));
        CHECK(cudaMalloc(&cRow, std::max<int64_t>(s.nnz, 1) * 4));
        CHECK(cudaMalloc(&cVal, std::max<int64_t>(s.nnz, 1) * 4));
        size_t bytes = 0;
        SCHECK(cusparseCsr2cscEx2_bufferSize(h, m, n, int(s.nnz), s.values, (int *) s.p0, (int *) s.p1, cVal,
                                             (int *) cOff, (int *) cRow, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
                                             CUSPARSE_INDEX_BASE_ONE, CUSPARSE_CSR2CSC_ALG1, &bytes));
        void *buffer;
        CHECK(cudaMalloc(&buffer, std::max<size_t>(bytes, 1)));
        SCHECK(cusparseCsr2cscEx2(h, m, n, int(s.nnz), s.values, (int *) s.p0, (int *) s.p1, cVal, (int *) cOff,
                                  (int *) cRow, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ONE,
                                  CUSPARSE_CSR2CSC_ALG1, buffer));
        std::vector<int> co = download<int>(cOff, n + 1), cr = download<int>(cRow, s.nnz);
        std::vector<float> cv = download<float>(cVal, s.nnz);
        // the CSC lists each column's rows in increasing order, as a transposition of the CSR
        int at = 0;
        for (int c = 0; c < n; c++) {
            EXPECT(co[c] == at + 1, "CSC offset of column %d", c);
            for (int r = 0; r < m; r++) {
                for (int e = ho[r] - 1; e < ho[r + 1] - 1; e++) {
                    if (hc[e] - 1 != c) continue;
                    EXPECT(cr[at] == r + 1 && cv[at] == hv[e], "CSC entry %d", at);
                    at++;
                }
            }
        }
        for (void *p: {cOff, cRow, cVal, buffer}) CHECK(cudaFree(p));
    }
    // sparse vectors: SpVV, axpby, gather, scatter
    {
        const int size = 50;
        std::vector<int> idx = {1, 4, 9, 16, 25, 36, 49};
        std::vector<double> xv = {1, -2, 3, -4, 5, -6, 7}, yh(size);
        for (int i = 0; i < size; i++) yh[i] = i * 0.5;
        void *di = upload(idx), *dx = upload(xv), *dy = upload(yh);
        cusparseSpVecDescr_t x;
        cusparseDnVecDescr_t y;
        SCHECK(cusparseCreateSpVec(&x, size, int64_t(idx.size()), di, dx, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_64F));
        SCHECK(cusparseCreateDnVec(&y, size, dy, CUDA_R_64F));
        double dot = 0, want = 0;
        for (size_t i = 0; i < idx.size(); i++) want += xv[i] * yh[idx[i]];
        size_t bytes = 0;
        SCHECK(cusparseSpVV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, x, y, &dot, CUDA_R_64F, &bytes));
        void *buffer;
        CHECK(cudaMalloc(&buffer, std::max<size_t>(bytes, 1)));
        SCHECK(cusparseSpVV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, x, y, &dot, CUDA_R_64F, buffer));
        EXPECT(std::fabs(dot - want) < 1e-12, "SpVV %g, expected %g", dot, want);
        const double alpha = 2, beta = 3;
        SCHECK(cusparseAxpby(h, &alpha, x, &beta, y));
        std::vector<double> got = download<double>(dy, size), ref = yh;
        for (double &v: ref) v *= beta;
        for (size_t i = 0; i < idx.size(); i++) ref[idx[i]] += alpha * xv[i];
        for (int i = 0; i < size; i++) EXPECT(got[i] == ref[i], "axpby %d", i);
        SCHECK(cusparseGather(h, y, x));
        std::vector<double> gathered = download<double>(dx, idx.size());
        for (size_t i = 0; i < idx.size(); i++) EXPECT(gathered[i] == ref[idx[i]], "gather %zu", i);
        CHECK(cudaMemset(dy, 0, size * 8));
        SCHECK(cusparseScatter(h, x, y));
        got = download<double>(dy, size);
        for (size_t i = 0; i < idx.size(); i++) EXPECT(got[idx[i]] == gathered[i], "scatter %zu", i);
        cusparseDestroySpVec(x);
        cusparseDestroyDnVec(y);
        for (void *p: {di, dx, dy, buffer}) CHECK(cudaFree(p));
    }
}

// ---- legacy functions

static void legacy(cusparseHandle_t h) {
    const int m = 40, n = 35;
    cusparseMatDescr_t descr;
    SCHECK(cusparseCreateMatDescr(&descr));
    // csrgeam2: C = alpha A + beta B
    {
        const Matrix a = randomSparse<float>(m, n, 0.2), b = randomSparse<float>(m, n, 0.2);
        Sparse<float, int> sa(a, CUSPARSE_FORMAT_CSR, 0), sb(b, CUSPARSE_FORMAT_CSR, 0);
        const float alpha = 1.5f, beta = -2;
        int *rowC;
        CHECK(cudaMalloc(&rowC, (m + 1) * 4));
        size_t bytes = 0;
        SCHECK(cusparseScsrgeam2_bufferSizeExt(h, m, n, &alpha, descr, int(sa.nnz), (float *) sa.values, (int *) sa.p0,
                                               (int *) sa.p1, &beta, descr, int(sb.nnz), (float *) sb.values,
                                               (int *) sb.p0, (int *) sb.p1, descr, nullptr, rowC, nullptr, &bytes));
        void *buffer;
        CHECK(cudaMalloc(&buffer, std::max<size_t>(bytes, 1)));
        int nnzC = 0;
        SCHECK(cusparseXcsrgeam2Nnz(h, m, n, descr, int(sa.nnz), (int *) sa.p0, (int *) sa.p1, descr, int(sb.nnz),
                                    (int *) sb.p0, (int *) sb.p1, descr, rowC, &nnzC, buffer));
        int *colC;
        float *valC;
        CHECK(cudaMalloc(&colC, std::max(nnzC, 1) * 4));
        CHECK(cudaMalloc(&valC, std::max(nnzC, 1) * 4));
        SCHECK(cusparseScsrgeam2(h, m, n, &alpha, descr, int(sa.nnz), (float *) sa.values, (int *) sa.p0,
                                 (int *) sa.p1, &beta, descr, int(sb.nnz), (float *) sb.values, (int *) sb.p0,
                                 (int *) sb.p1, descr, valC, rowC, colC, buffer));
        std::vector<int> ho = download<int>(rowC, m + 1), hc = download<int>(colC, nnzC);
        std::vector<float> hv = download<float>(valC, nnzC);
        Matrix got{m, n, std::vector<cd>(size_t(m) * n)}, want = got;
        int pattern = 0;
        for (int r = 0; r < m; r++) {
            for (int e = ho[r]; e < ho[r + 1]; e++) got.at(r, hc[e]) = hv[e];
            for (int c = 0; c < n; c++) {
                want.at(r, c) = double(alpha) * a.at(r, c) + double(beta) * b.at(r, c);
                pattern += a.at(r, c) != cd(0) || b.at(r, c) != cd(0);
            }
        }
        EXPECT(nnzC == pattern, "csrgeam2 nnz %d, expected %d", nnzC, pattern);
        expectClose(got, want, 1e-6, "csrgeam2");
        for (void *p: {(void *) rowC, (void *) colC, (void *) valC, buffer}) CHECK(cudaFree(p));
    }
    // COO sorting by row, then coo2csr; CSR column sorting
    {
        const Matrix a = randomSparse<float>(m, n, 0.2);
        Sparse<float, int> coo(a, CUSPARSE_FORMAT_COO, 0, true);
        const int nnz = int(coo.nnz);
        int *perm;
        CHECK(cudaMalloc(&perm, std::max(nnz, 1) * 4));
        SCHECK(cusparseCreateIdentityPermutation(h, nnz, perm));
        size_t bytes = 0;
        SCHECK(cusparseXcoosort_bufferSizeExt(h, m, n, nnz, (int *) coo.p0, (int *) coo.p1, &bytes));
        void *buffer;
        CHECK(cudaMalloc(&buffer, std::max<size_t>(bytes, 1)));
        std::vector<float> before = download<float>(coo.values, nnz);
        SCHECK(cusparseXcoosortByRow(h, m, n, nnz, (int *) coo.p0, (int *) coo.p1, perm, buffer));
        std::vector<int> rows = download<int>(coo.p0, nnz), cols = download<int>(coo.p1, nnz),
                         p = download<int>(perm, nnz);
        for (int i = 0; i < nnz; i++) {
            EXPECT(i == 0 || rows[i] > rows[i - 1] || (rows[i] == rows[i - 1] && cols[i] > cols[i - 1]),
                   "COO not sorted at %d", i);
            EXPECT(a.at(rows[i], cols[i]).real() == double(before[p[i]]), "COO permutation at %d", i);
        }
        int *rowPtr;
        CHECK(cudaMalloc(&rowPtr, (m + 1) * 4));
        SCHECK(cusparseXcoo2csr(h, (int *) coo.p0, nnz, m, rowPtr, CUSPARSE_INDEX_BASE_ZERO));
        std::vector<int> off = download<int>(rowPtr, m + 1);
        for (int r = 0; r <= m; r++) {
            const int want = int(std::lower_bound(rows.begin(), rows.end(), r) - rows.begin());
            EXPECT(off[r] == want, "coo2csr offset %d: %d, expected %d", r, off[r], want);
        }
        // shuffle the columns within each row, then sort them back
        std::vector<int> shuffled = cols;
        for (int r = 0; r < m; r++) std::shuffle(shuffled.begin() + off[r], shuffled.begin() + off[r + 1], rng);
        CHECK(cudaMemcpy(coo.p1, shuffled.data(), nnz * 4, cudaMemcpyHostToDevice));
        SCHECK(cusparseCreateIdentityPermutation(h, nnz, perm));
        SCHECK(cusparseXcsrsort_bufferSizeExt(h, m, n, nnz, rowPtr, (int *) coo.p1, &bytes));
        CHECK(cudaFree(buffer));
        CHECK(cudaMalloc(&buffer, std::max<size_t>(bytes, 1)));
        SCHECK(cusparseXcsrsort(h, m, n, nnz, descr, rowPtr, (int *) coo.p1, perm, buffer));
        std::vector<int> sorted = download<int>(coo.p1, nnz), p2 = download<int>(perm, nnz);
        for (int i = 0; i < nnz; i++) {
            EXPECT(sorted[i] == cols[i], "csrsort at %d", i);
            EXPECT(shuffled[p2[i]] == sorted[i], "csrsort permutation at %d", i);
        }
        for (void *q: {(void *) perm, (void *) rowPtr, buffer}) CHECK(cudaFree(q));
    }
    SCHECK(cusparseDestroyMatDescr(descr));
}

// BSR matrices as dense references; products and triangular solves, both block layouts
static void bsr(cusparseHandle_t h) {
    const int mb = 9, bs = 3, n = mb * bs, rhs = 5;
    for (cusparseDirection_t dir: {CUSPARSE_DIRECTION_ROW, CUSPARSE_DIRECTION_COLUMN}) {
        // block pattern: the diagonal and a few others (lower and upper)
        std::vector<int> rowPtr(1, 0), colInd;
        std::vector<double> values;
        Matrix a{n, n, std::vector<cd>(size_t(n) * n)};
        for (int br = 0; br < mb; br++) {
            for (int bc = 0; bc < mb; bc++) {
                if (bc != br && (br * 7 + bc * 3) % 5 != 0) continue;
                colInd.push_back(bc);
                for (int q = 0; q < bs * bs; q++) {
                    const int r = dir == CUSPARSE_DIRECTION_ROW ? q / bs : q % bs, c = dir == CUSPARSE_DIRECTION_ROW ? q % bs : q / bs;
                    double v = uniform();
                    if (br == bc && r == c) v = 5 + std::fabs(v);
                    values.push_back(v);
                    a.at(br * bs + r, bc * bs + c) = v;
                }
            }
            rowPtr.push_back(int(colInd.size()));
        }
        const int nnzb = int(colInd.size());
        void *dPtr = upload(rowPtr), *dCol = upload(colInd), *dVal = upload(values);
        cusparseMatDescr_t descr;
        SCHECK(cusparseCreateMatDescr(&descr));
        // bsrmv, bsrmm
        const Matrix x = randomDense<double>(n, 1), y0 = randomDense<double>(n, 1);
        void *dx = upload(std::vector<double>(x.a.size())), *dy = upload(std::vector<double>(n));
        std::vector<double> hx(n), hy(n);
        for (int i = 0; i < n; i++) {
            hx[i] = x.a[i].real();
            hy[i] = y0.a[i].real();
        }
        CHECK(cudaMemcpy(dx, hx.data(), n * 8, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dy, hy.data(), n * 8, cudaMemcpyHostToDevice));
        const double alpha = 1.5, beta = -0.5;
        SCHECK(cusparseDbsrmv(h, dir, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &alpha, descr, (double *) dVal,
                              (int *) dPtr, (int *) dCol, bs, (double *) dx, &beta, (double *) dy));
        std::vector<double> gy = download<double>(dy, n);
        for (int r = 0; r < n; r++) {
            double acc = 0;
            for (int c = 0; c < n; c++) acc += a.at(r, c).real() * hx[c];
            EXPECT(std::fabs(gy[r] - (alpha * acc + beta * hy[r])) < 1e-10, "bsrmv row %d", r);
        }
        const Matrix b = randomDense<double>(rhs, n);  // transposed B: rhs x n
        std::vector<double> hb(size_t(rhs) * n), hc(size_t(n) * rhs, 1.0);
        for (int r = 0; r < rhs; r++) for (int c = 0; c < n; c++) hb[r + c * rhs] = b.at(r, c).real();
        void *db = upload(hb), *dc = upload(hc);
        SCHECK(cusparseDbsrmm(h, dir, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, mb, rhs, mb, nnzb,
                              &alpha, descr, (double *) dVal, (int *) dPtr, (int *) dCol, bs, (double *) db, rhs, &beta,
                              (double *) dc, n));
        std::vector<double> gc = download<double>(dc, hc.size());
        for (int r = 0; r < n; r++) {
            for (int j = 0; j < rhs; j++) {
                double acc = 0;
                for (int c = 0; c < n; c++) acc += a.at(r, c).real() * b.at(j, c).real();
                EXPECT(std::fabs(gc[r + j * n] - (alpha * acc + beta)) < 1e-10, "bsrmm (%d, %d)", r, j);
            }
        }
        // bsrsv2: lower, non-unit; bsrsm2: upper, unit
        cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER);
        bsrsv2Info_t sv;
        SCHECK(cusparseCreateBsrsv2Info(&sv));
        int bytes = 0;
        SCHECK(cusparseDbsrsv2_bufferSize(h, dir, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nnzb, descr, (double *) dVal,
                                          (int *) dPtr, (int *) dCol, bs, sv, &bytes));
        void *buffer;
        CHECK(cudaMalloc(&buffer, std::max(bytes, 1)));
        SCHECK(cusparseDbsrsv2_analysis(h, dir, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nnzb, descr, (double *) dVal,
                                        (int *) dPtr, (int *) dCol, bs, sv, CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));
        int pivot = 0;
        EXPECT(cusparseXbsrsv2_zeroPivot(h, sv, &pivot) == CUSPARSE_STATUS_SUCCESS, "bsrsv2 analysis zero pivot");
        SCHECK(cusparseDbsrsv2_solve(h, dir, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nnzb, &alpha, descr, (double *) dVal,
                                     (int *) dPtr, (int *) dCol, bs, sv, (double *) dx, (double *) dy,
                                     CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));
        EXPECT(cusparseXbsrsv2_zeroPivot(h, sv, &pivot) == CUSPARSE_STATUS_SUCCESS, "bsrsv2 zero pivot %d", pivot);
        Matrix xm{n, 1, std::vector<cd>(n)};
        for (int i = 0; i < n; i++) xm.a[i] = hx[i];
        const Matrix wantSv = solveDense(usedTriangle(a, true, false), CUSPARSE_OPERATION_NON_TRANSPOSE, xm, alpha);
        gy = download<double>(dy, n);
        for (int i = 0; i < n; i++) EXPECT(std::fabs(gy[i] - wantSv.a[i].real()) < 1e-10, "bsrsv2 %d", i);
        cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_UPPER);
        cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_UNIT);
        bsrsm2Info_t sm;
        SCHECK(cusparseCreateBsrsm2Info(&sm));
        SCHECK(cusparseDbsrsm2_bufferSize(h, dir, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, mb,
                                          rhs, nnzb, descr, (double *) dVal, (int *) dPtr, (int *) dCol, bs, sm,
                                          &bytes));
        CHECK(cudaFree(buffer));
        CHECK(cudaMalloc(&buffer, std::max(bytes, 1)));
        SCHECK(cusparseDbsrsm2_analysis(h, dir, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, mb,
                                        rhs, nnzb, descr, (double *) dVal, (int *) dPtr, (int *) dCol, bs, sm,
                                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));
        // B: n x rhs column-major (the transposed B above, read as n x rhs with ld n)
        std::vector<double> bn(size_t(n) * rhs), xn(size_t(n) * rhs);
        Matrix bm{n, rhs, std::vector<cd>(size_t(n) * rhs)};
        for (int r = 0; r < n; r++) for (int j = 0; j < rhs; j++) bm.at(r, j) = bn[r + j * n] = uniform();
        void *dbn = upload(bn), *dxn = upload(xn);
        SCHECK(cusparseDbsrsm2_solve(h, dir, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, rhs,
                                     nnzb, &alpha, descr, (double *) dVal, (int *) dPtr, (int *) dCol, bs, sm,
                                     (double *) dbn, n, (double *) dxn, n, CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));
        const Matrix wantSm = solveDense(usedTriangle(a, false, true), CUSPARSE_OPERATION_NON_TRANSPOSE, bm, alpha);
        std::vector<double> gx = download<double>(dxn, xn.size());
        for (int r = 0; r < n; r++) {
            for (int j = 0; j < rhs; j++) {
                EXPECT(std::fabs(gx[r + j * n] - wantSm.at(r, j).real()) < 1e-10, "bsrsm2 (%d, %d)", r, j);
            }
        }
        SCHECK(cusparseDestroyBsrsv2Info(sv));
        SCHECK(cusparseDestroyBsrsm2Info(sm));
        SCHECK(cusparseDestroyMatDescr(descr));
        for (void *p: {dPtr, dCol, dVal, dx, dy, db, dc, buffer, dbn, dxn}) CHECK(cudaFree(p));
    }
}

// device scalars and a stream
static void deviceScalars(cusparseHandle_t h) {
    cudaStream_t s;
    CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    SCHECK(cusparseSetStream(h, s));
    SCHECK(cusparseSetPointerMode(h, CUSPARSE_POINTER_MODE_DEVICE));
    const Matrix a = randomSparse<float>(64, 48, 0.2), x = randomDense<float>(48, 1);
    Sparse<float, int> sa(a, CUSPARSE_FORMAT_CSR, 0);
    DenseDev<float> xd(x, false, 0), yd(randomDense<float>(64, 1), false, 0);
    cusparseDnVecDescr_t xv, yv;
    SCHECK(cusparseCreateDnVec(&xv, 48, xd.values, CUDA_R_32F));
    SCHECK(cusparseCreateDnVec(&yv, 64, yd.values, CUDA_R_32F));
    const std::vector<float> scalars = {3.0f, 0.0f};
    void *ds = upload(scalars);
    size_t bytes = 0;
    SCHECK(cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, ds, sa.descr, xv, (float *) ds + 1, yv,
                                   CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bytes));
    void *buffer;
    CHECK(cudaMalloc(&buffer, std::max<size_t>(bytes, 1)));
    SCHECK(cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, ds, sa.descr, xv, (float *) ds + 1, yv, CUDA_R_32F,
                        CUSPARSE_SPMV_ALG_DEFAULT, buffer));
    CHECK(cudaStreamSynchronize(s));
    Matrix want{64, 1, std::vector<cd>(64)};
    for (int r = 0; r < 64; r++) {
        for (int c = 0; c < 48; c++) want.a[r] += 3.0 * a.at(r, c) * x.a[c];
    }
    expectClose(yd.get(), want, 1e-5, "SpMV with device scalars on a stream");
    SCHECK(cusparseSetPointerMode(h, CUSPARSE_POINTER_MODE_HOST));
    SCHECK(cusparseSetStream(h, nullptr));
    cusparseDestroyDnVec(xv);
    cusparseDestroyDnVec(yv);
    CHECK(cudaFree(ds));
    CHECK(cudaFree(buffer));
    CHECK(cudaStreamDestroy(s));
}

int main() {
    cusparseHandle_t h;
    SCHECK(cusparseCreate(&h));
    productsOf<float>(h);
    productsOf<double>(h);
    productsOf<std::complex<float>>(h);
    productsOf<std::complex<double>>(h);
    spmvCase<__half, int>(h, CUSPARSE_FORMAT_CSR, CUSPARSE_OPERATION_NON_TRANSPOSE, 0);
    spmmCase<__half, int>(h, CUSPARSE_FORMAT_CSR, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          false, false);
    batchedSpmm(h, true);
    batchedSpmm(h, false);
    PASS("SpMV, SpMM: CSR, CSC, COO, all operations, index types and bases, half, batches");
    spgemmCase<float, int>(h, 0);
    spgemmCase<double, int64_t>(h, 1);
    spgemmCase<std::complex<float>, int>(h, 0);
    PASS("SpGEMM");
    for (cusparseOperation_t opA: {CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE}) {
        for (cusparseOperation_t opB: {CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE}) {
            sddmmCase<float>(h, opA, opB);
            sddmmCase<std::complex<double>>(h, opA, opB);
        }
    }
    PASS("SDDMM");
    solves<float>(h);
    solves<double>(h);
    solves<std::complex<float>>(h);
    PASS("SpSV, SpSM: lower / upper, unit / non-unit, transposed, CSR / COO");
    conversions(h);
    PASS("sparse to dense, dense to sparse, CSR to CSC, sparse vectors");
    legacy(h);
    bsr(h);
    PASS("csrgeam2, COO / CSR sorting, coo2csr, BSR products and solves");
    deviceScalars(h);
    PASS("device scalars, streams");
    SCHECK(cusparseDestroy(h));
    return 0;
}
