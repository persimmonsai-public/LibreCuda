/**
 * LibreCUDA's cuSPARSE (libcusparse.so.12): handles, matrix descriptors, the generic API (sparse matrices in CSR, CSC
 * and COO format with strided batches, dense matrices and vectors, sparse vectors; SpMV, SpMM, SpGEMM, SDDMM, SpSV,
 * SpSM, sparse / dense conversions, CSR to CSC, vector operations) and the legacy functions PyTorch uses (BSR products
 * and triangular solves, csrgeam2, COO / CSR sorting and conversion), for half, bfloat16 (computed in float), float,
 * double and complex values with 32 or 64 bit indices.
 *
 * The products and solves work on a CSR view of op(A): CSR matrices as they are, CSC matrices transposed as they are,
 * anything else converted in the caller's buffer (a stable radix sort of the entries by their row in op(A)), so the
 * results don't depend on scheduling. SpMV and SpMM run a warp per row; SpGEMM expands all products, sorts them by
 * (row, column) and adds up equal positions; triangular solves are sync-free (a warp per row in dependency order, waiting
 * for the rows it needs). Other functions of cusparse.h return CUSPARSE_STATUS_NOT_SUPPORTED (stubs).
 */
#include <cusparse.h>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda/std/complex>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

// ---- handles and descriptors

struct cusparseContext {
    uint32_t magic;
    cudaStream_t stream;
    cusparsePointerMode_t pointerMode;
};

struct cusparseMatDescr {
    cusparseMatrixType_t type;
    cusparseFillMode_t fillMode;
    cusparseDiagType_t diagType;
    cusparseIndexBase_t indexBase;
};

// CSR: p0 row offsets, p1 column indices; CSC: p0 column offsets, p1 row indices; COO: p0 rows, p1 columns
struct cusparseSpMatDescr {
    cusparseFormat_t format;
    int64_t rows, cols, nnz;
    void *p0, *p1, *values;
    cusparseIndexType_t type0, type1;
    cusparseIndexBase_t base;
    cudaDataType valueType;
    int batchCount;
    int64_t stride0, stride1;  // batch strides of p0 and of p1 / values
    cusparseFillMode_t fill;
    cusparseDiagType_t diag;
};

struct cusparseDnMatDescr {
    int64_t rows, cols, ld;
    void *values;
    cudaDataType valueType;
    cusparseOrder_t order;
    int batchCount;
    int64_t batchStride;
};

struct cusparseDnVecDescr {
    int64_t size;
    void *values;
    cudaDataType valueType;
};

struct cusparseSpVecDescr {
    int64_t size, nnz;
    void *indices, *values;
    cusparseIndexType_t idxType;
    cusparseIndexBase_t base;
    cudaDataType valueType;
};

struct cusparseSpGEMMDescr {
    int64_t products = 0, nnz = 0, rows = 0, cols = 0;
    void *buffer1 = nullptr, *buffer2 = nullptr;
    cudaDataType valueType = CUDA_R_32F;
};

struct cusparseSpSVDescr {
    void *buffer = nullptr;
    void *values = nullptr;  // cusparseSpSV_updateMatrix
};

struct cusparseSpSMDescr {
    void *buffer = nullptr;
    void *values = nullptr;
};

struct bsrsv2Info {
    int *pivot = nullptr;  // device: the first row with a zero pivot (in the last solve's buffer)
    int blockDim = 1;
};

struct bsrsm2Info {
    int *pivot = nullptr;
    int blockDim = 1;
};

namespace {

constexpr uint32_t SPARSE_MAGIC = 0x4c435350;

bool valid(cusparseHandle_t handle) {
    return handle != nullptr && handle->magic == SPARSE_MAGIC;
}

using cfloat = cuda::std::complex<float>;
using cdouble = cuda::std::complex<double>;

// storage type T, compute type C
template<typename T>
struct Num {
    using C = T;
    static __host__ __device__ C load(T v) { return v; }
    static __host__ __device__ T store(C v) { return v; }
};

template<>
struct Num<__half> {
    using C = float;
    static __device__ C load(__half v) { return __half2float(v); }
    static __device__ __half store(C v) { return __float2half_rn(v); }
};

template<>
struct Num<__nv_bfloat16> {
    using C = float;
    static __device__ C load(__nv_bfloat16 v) { return __bfloat162float(v); }
    static __device__ __nv_bfloat16 store(C v) { return __float2bfloat16_rn(v); }
};

template<typename C>
__host__ __device__ C conjOf(C v) {
    if constexpr (std::is_same_v<C, cfloat> || std::is_same_v<C, cdouble>) {
        return cuda::std::conj(v);
    } else {
        return v;
    }
}

// loads of values other warps write (bypassing the L1 cache)
template<typename C>
__device__ C loadShared(const C *p) {
    if constexpr (std::is_same_v<C, float> || std::is_same_v<C, double>) {
        return __ldcg(p);
    } else if constexpr (std::is_same_v<C, cfloat>) {
        const float2 v = __ldcg(reinterpret_cast<const float2 *>(p));
        return cfloat(v.x, v.y);
    } else {
        const double2 v = __ldcg(reinterpret_cast<const double2 *>(p));
        return cdouble(v.x, v.y);
    }
}

template<typename T>
struct Tag {
    using type = T;
};

template<typename F>
cusparseStatus_t withType(cudaDataType t, F &&f) {
    switch (t) {
        case CUDA_R_32F: return f(Tag<float>{});
        case CUDA_R_64F: return f(Tag<double>{});
        case CUDA_C_32F: return f(Tag<cfloat>{});
        case CUDA_C_64F: return f(Tag<cdouble>{});
        case CUDA_R_16F: return f(Tag<__half>{});
        case CUDA_R_16BF: return f(Tag<__nv_bfloat16>{});
        default: return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
}

size_t typeSize(cudaDataType t) {
    switch (t) {
        case CUDA_R_16F:
        case CUDA_R_16BF: return 2;
        case CUDA_R_32F: return 4;
        case CUDA_R_64F:
        case CUDA_C_32F: return 8;
        case CUDA_C_64F: return 16;
        default: return 0;
    }
}

size_t computeSize(cudaDataType t) {
    return t == CUDA_R_16F || t == CUDA_R_16BF ? 4 : typeSize(t);
}

// an index array (32 or 64 bit), read with the base removed
struct Idx {
    const void *p;
    bool wide;
    int64_t base;

    __device__ int64_t operator[](int64_t i) const {
        return (wide ? static_cast<const int64_t *>(p)[i] : int64_t(static_cast<const int32_t *>(p)[i])) - base;
    }

    __host__ __device__ Idx at(int64_t elements) const {
        return Idx{static_cast<const char *>(p) + elements * (wide ? 8 : 4), wide, base};
    }
};

__device__ void storeIndex(void *p, bool wide, int64_t i, int64_t v) {
    if (wide) {
        static_cast<int64_t *>(p)[i] = v;
    } else {
        static_cast<int32_t *>(p)[i] = int32_t(v);
    }
}

// a CSR matrix as the kernels read it
struct Csr {
    int64_t rows, cols, nnz;
    Idx offsets, columns;
    const void *values;
    bool conj;
};

// a dense matrix (column or row major), element (i, j) of op(M)
struct Dense {
    void *values;
    int64_t ld, batchStride;
    bool rowMajor;
    cusparseOperation_t op;

    __device__ int64_t index(int64_t i, int64_t j) const {
        if (op != CUSPARSE_OPERATION_NON_TRANSPOSE) {
            const int64_t t = i;
            i = j;
            j = t;
        }
        return rowMajor ? i * ld + j : i + j * ld;
    }

    template<typename T>
    __device__ typename Num<T>::C get(int64_t batch, int64_t i, int64_t j) const {
        const auto v = Num<T>::load(static_cast<const T *>(values)[batch * batchStride + index(i, j)]);
        return op == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE ? conjOf(v) : v;
    }
};

Dense dense(cusparseConstDnMatDescr_t d, cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE) {
    return Dense{d->values, d->ld, d->batchCount > 1 ? d->batchStride : 0, d->order == CUSPARSE_ORDER_ROW, op};
}

template<typename C>
struct Scalar {
    C value;
    const C *ptr;

    __device__ C get() const { return ptr != nullptr ? *ptr : value; }
};

template<typename C>
Scalar<C> scalar(cusparseHandle_t h, const void *p) {
    Scalar<C> s{C(0), nullptr};
    if (h->pointerMode == CUSPARSE_POINTER_MODE_HOST) {
        s.value = *static_cast<const C *>(p);
    } else {
        s.ptr = static_cast<const C *>(p);
    }
    return s;
}

// result = alpha * acc + beta * old (old not read when beta is zero)
template<typename C>
__device__ C axpby(C alpha, C acc, C beta, const C *old) {
    return beta == C(0) ? alpha * acc : alpha * acc + beta * *old;
}

cusparseStatus_t launched() {
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_EXECUTION_FAILED;
}

cusparseStatus_t cudaStatus(cudaError_t e) {
    if (e == cudaSuccess) return CUSPARSE_STATUS_SUCCESS;
    cudaGetLastError();
    return e == cudaErrorMemoryAllocation ? CUSPARSE_STATUS_ALLOC_FAILED : CUSPARSE_STATUS_EXECUTION_FAILED;
}

unsigned int blocksOf(int64_t threads, int perBlock = 256) {
    return unsigned(std::max<int64_t>(1, std::min<int64_t>((threads + perBlock - 1) / perBlock, 1 << 30)));
}

int bitsFor(uint64_t maxValue) {
    int bits = 1;
    while (bits < 64 && (maxValue >> bits) != 0) bits++;
    return bits;
}

size_t align(size_t n) {
    return (n + 255) / 256 * 256;
}

// buffer layout: consecutive aligned pieces
struct Layout {
    size_t total = 0;

    size_t add(size_t bytes) {
        const size_t at = total;
        total += align(bytes);
        return at;
    }
};

#define GRID_STRIDE(i, n) \
    for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < (n); i += int64_t(gridDim.x) * blockDim.x)

// ---- CSR views of op(A)

__global__ void fillMajor(Idx offsets, int64_t majors, int64_t *out) {
    const int64_t warp = (blockIdx.x * int64_t(blockDim.x) + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (warp >= majors) return;
    for (int64_t e = offsets[warp] + lane; e < offsets[warp + 1]; e += 32) out[e] = warp;
}

// key (row of op(A)) and minor index (its column) of every entry; `major` holds the expanded offsets' indices
__global__ void conversionKeys(int64_t nnz, const int64_t *major, Idx minor, Idx rowsCoo, Idx colsCoo, bool coo,
                               bool majorIsRow, bool transpose, uint64_t *keys, int64_t *perm, int64_t *minorOut) {
    GRID_STRIDE(e, nnz) {
        int64_t r, c;
        if (coo) {
            r = rowsCoo[e];
            c = colsCoo[e];
        } else if (majorIsRow) {
            r = major[e];
            c = minor[e];
        } else {
            c = major[e];
            r = minor[e];
        }
        keys[e] = uint64_t(transpose ? c : r);
        minorOut[e] = transpose ? r : c;
        perm[e] = e;
    }
}

template<typename T>
__global__ void conversionGather(int64_t nnz, const int64_t *perm, const int64_t *minor, const T *values, bool conj,
                                 int64_t *columns, T *out) {
    GRID_STRIDE(i, nnz) {
        const int64_t e = perm[i];
        columns[i] = minor[e];
        if (values != nullptr) {
            const auto v = Num<T>::load(values[e]);
            out[i] = Num<T>::store(conj ? conjOf(v) : v);
        }
    }
}

// offsets[r] = number of keys < r (keys sorted)
__global__ void offsetsOfSorted(const uint64_t *keys, int64_t n, int64_t rows, void *offsets, bool wide, int64_t base) {
    GRID_STRIDE(r, rows + 1) {
        int64_t lo = 0, hi = n;
        while (lo < hi) {
            const int64_t mid = (lo + hi) / 2;
            if (keys[mid] < uint64_t(r)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        storeIndex(offsets, wide, r, lo + base);
    }
}

struct Conversion {
    size_t keysIn, keysOut, permIn, permOut, minor, major, temp, offsets, columns, values, tempBytes;
    Layout layout;
};

Conversion conversionLayout(int64_t nnz, int64_t rowsOut, size_t valueBytes) {
    Conversion c{};
    const size_t n = size_t(std::max<int64_t>(nnz, 1));
    cub::DeviceRadixSort::SortPairs(nullptr, c.tempBytes, static_cast<const uint64_t *>(nullptr),
                                    static_cast<uint64_t *>(nullptr), static_cast<const int64_t *>(nullptr),
                                    static_cast<int64_t *>(nullptr), int64_t(n));
    c.keysIn = c.layout.add(n * 8);
    c.keysOut = c.layout.add(n * 8);
    c.permIn = c.layout.add(n * 8);
    c.permOut = c.layout.add(n * 8);
    c.minor = c.layout.add(n * 8);
    c.major = c.layout.add(n * 8);
    c.temp = c.layout.add(c.tempBytes);
    c.offsets = c.layout.add(size_t(rowsOut + 1) * 8);
    c.columns = c.layout.add(n * 8);
    c.values = c.layout.add(n * valueBytes);
    return c;
}

// does op(A) need a conversion (else it is A's CSR, or the transpose of its CSC)?
bool needsConversion(cusparseConstSpMatDescr_t a, cusparseOperation_t op) {
    if (a->format == CUSPARSE_FORMAT_CSR) return op != CUSPARSE_OPERATION_NON_TRANSPOSE;
    if (a->format == CUSPARSE_FORMAT_CSC) return op == CUSPARSE_OPERATION_NON_TRANSPOSE;
    return true;
}

bool supportedFormat(cusparseConstSpMatDescr_t a) {
    return a->format == CUSPARSE_FORMAT_CSR || a->format == CUSPARSE_FORMAT_CSC || a->format == CUSPARSE_FORMAT_COO;
}

int64_t opRows(cusparseConstSpMatDescr_t a, cusparseOperation_t op) {
    return op == CUSPARSE_OPERATION_NON_TRANSPOSE ? a->rows : a->cols;
}

int64_t opCols(cusparseConstSpMatDescr_t a, cusparseOperation_t op) {
    return op == CUSPARSE_OPERATION_NON_TRANSPOSE ? a->cols : a->rows;
}

size_t conversionBytes(cusparseConstSpMatDescr_t a, cusparseOperation_t op) {
    if (!needsConversion(a, op)) return 0;
    return conversionLayout(a->nnz, opRows(a, op), typeSize(a->valueType)).layout.total;
}

// op(A) of batch b as CSR (converted into `buffer` when needed)
template<typename T>
cusparseStatus_t csrOf(cudaStream_t s, cusparseConstSpMatDescr_t a, cusparseOperation_t op, int64_t b, void *buffer,
                       const void *values, Csr &out) {
    const bool conj = op == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    const int64_t base = a->base == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;
    const Idx i0{a->p0, a->type0 == CUSPARSE_INDEX_64I, base}, i1{a->p1, a->type1 == CUSPARSE_INDEX_64I, base};
    const T *vals = static_cast<const T *>(values != nullptr ? values : a->values) + b * a->stride1;
    if (!needsConversion(a, op)) {
        out = Csr{opRows(a, op), opCols(a, op), a->nnz, i0.at(b * a->stride0), i1.at(b * a->stride1), vals,
                  a->format == CUSPARSE_FORMAT_CSC && conj};
        return CUSPARSE_STATUS_SUCCESS;
    }
    if (buffer == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    const int64_t rows = opRows(a, op), nnz = a->nnz;
    const Conversion c = conversionLayout(nnz, rows, sizeof(T));
    char *buf = static_cast<char *>(buffer);
    auto *keysIn = reinterpret_cast<uint64_t *>(buf + c.keysIn), *keysOut = reinterpret_cast<uint64_t *>(buf + c.keysOut);
    auto *permIn = reinterpret_cast<int64_t *>(buf + c.permIn), *permOut = reinterpret_cast<int64_t *>(buf + c.permOut);
    auto *minor = reinterpret_cast<int64_t *>(buf + c.minor), *major = reinterpret_cast<int64_t *>(buf + c.major);
    auto *columns = reinterpret_cast<int64_t *>(buf + c.columns);
    T *converted = reinterpret_cast<T *>(buf + c.values);
    const bool coo = a->format == CUSPARSE_FORMAT_COO, transpose = op != CUSPARSE_OPERATION_NON_TRANSPOSE;
    if (nnz > 0) {
        if (!coo) {
            const int64_t majors = a->format == CUSPARSE_FORMAT_CSR ? a->rows : a->cols;
            fillMajor<<<blocksOf(majors * 32), 256, 0, s>>>(i0.at(b * a->stride0), majors, major);
        }
        conversionKeys<<<blocksOf(nnz), 256, 0, s>>>(nnz, major, i1.at(b * a->stride1), i0.at(b * a->stride0),
                                                    i1.at(b * a->stride1), coo, a->format == CUSPARSE_FORMAT_CSR,
                                                    transpose, keysIn, permIn, minor);
        size_t tempBytes = c.tempBytes;
        if (cub::DeviceRadixSort::SortPairs(buf + c.temp, tempBytes, keysIn, keysOut, permIn, permOut, nnz, 0,
                                            bitsFor(uint64_t(std::max<int64_t>(rows, 1))), s) != cudaSuccess) {
            return cudaStatus(cudaGetLastError());
        }
        conversionGather<T><<<blocksOf(nnz), 256, 0, s>>>(nnz, permOut, minor, vals, conj, columns, converted);
    }
    offsetsOfSorted<<<blocksOf(rows + 1), 256, 0, s>>>(keysOut, nnz, rows, buf + c.offsets, true, 0);
    out = Csr{rows, opCols(a, op), nnz, Idx{buf + c.offsets, true, 0}, Idx{columns, true, 0}, converted, false};
    return launched();
}

// ---- SpMV, SpMM

template<typename T>
__global__ void spmv(Csr a, const T *x, T *y, Scalar<typename Num<T>::C> alpha, Scalar<typename Num<T>::C> beta) {
    using C = typename Num<T>::C;
    const int64_t row = (blockIdx.x * int64_t(blockDim.x) + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (row >= a.rows) return;
    const T *values = static_cast<const T *>(a.values);
    C acc = C(0);
    for (int64_t k = a.offsets[row] + lane; k < a.offsets[row + 1]; k += 32) {
        C v = Num<T>::load(values[k]);
        if (a.conj) v = conjOf(v);
        acc += v * Num<T>::load(x[a.columns[k]]);
    }
    for (int o = 16; o > 0; o /= 2) {
        if constexpr (std::is_same_v<C, cfloat> || std::is_same_v<C, cdouble>) {
            acc = C(acc.real() + __shfl_down_sync(0xffffffffu, acc.real(), o),
                    acc.imag() + __shfl_down_sync(0xffffffffu, acc.imag(), o));
        } else {
            acc += __shfl_down_sync(0xffffffffu, acc, o);
        }
    }
    if (lane == 0) {
        const C old = beta.get() == C(0) ? C(0) : Num<T>::load(y[row]);
        y[row] = Num<T>::store(axpby(alpha.get(), acc, beta.get(), &old));
    }
}

// C = alpha op(A) op(B) + beta C: a warp per (row, 32 columns), batch blockIdx.y
template<typename T>
__global__ void spmm(Csr a, int64_t offsetsStride, int64_t columnsStride, Dense b, Dense c, int64_t n,
                     Scalar<typename Num<T>::C> alpha, Scalar<typename Num<T>::C> beta) {
    using C = typename Num<T>::C;
    const int64_t warp = (blockIdx.x * int64_t(blockDim.x) + threadIdx.x) / 32, lane = threadIdx.x % 32;
    const int64_t tiles = (n + 31) / 32, row = warp / tiles, j = (warp % tiles) * 32 + lane, batch = blockIdx.y;
    if (row >= a.rows) return;
    const Idx offsets = a.offsets.at(batch * offsetsStride), columns = a.columns.at(batch * columnsStride);
    const T *values = static_cast<const T *>(a.values) + batch * columnsStride;
    C acc = C(0);
    for (int64_t k = offsets[row]; k < offsets[row + 1]; k++) {
        if (j >= n) continue;
        C v = Num<T>::load(values[k]);
        if (a.conj) v = conjOf(v);
        acc += v * b.get<T>(batch, columns[k], j);
    }
    if (j < n) {
        T *out = static_cast<T *>(c.values) + batch * c.batchStride + c.index(row, j);
        const C old = beta.get() == C(0) ? C(0) : Num<T>::load(*out);
        *out = Num<T>::store(axpby(alpha.get(), acc, beta.get(), &old));
    }
}

// ---- SDDMM: C = alpha (op(A) op(B)) at C's pattern + beta C (CSR)

template<typename T>
__global__ void sddmm(Csr c, int64_t offsetsStride, int64_t columnsStride, Dense a, Dense b, int64_t k,
                      Scalar<typename Num<T>::C> alpha, Scalar<typename Num<T>::C> beta) {
    using C = typename Num<T>::C;
    const int64_t batch = blockIdx.y;
    const Idx offsets = c.offsets.at(batch * offsetsStride), columns = c.columns.at(batch * columnsStride);
    T *values = const_cast<T *>(static_cast<const T *>(c.values)) + batch * columnsStride;
    GRID_STRIDE(e, c.nnz) {
        int64_t lo = 0, hi = c.rows;  // the row: the last one starting at or before e
        while (hi - lo > 1) {
            const int64_t mid = (lo + hi) / 2;
            if (offsets[mid] <= e) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        const int64_t row = lo, col = columns[e];
        C acc = C(0);
        for (int64_t i = 0; i < k; i++) acc += a.get<T>(batch, row, i) * b.get<T>(batch, i, col);
        const C old = beta.get() == C(0) ? C(0) : Num<T>::load(values[e]);
        values[e] = Num<T>::store(axpby(alpha.get(), acc, beta.get(), &old));
    }
}

// ---- triangular solves (sync-free: a warp per row, rows in dependency order through a ticket counter)

struct Solve {
    bool lower, unit;
};

// solution x (compute type, row-major rows x n) of op(A) x = alpha op(B); `done` flags rows that are solved
template<typename T>
__global__ void triangularSolve(Csr a, Solve how, Dense b, int64_t n, Scalar<typename Num<T>::C> alpha,
                                typename Num<T>::C *x, int *done, int *ticket, int *pivot, bool vector) {
    using C = typename Num<T>::C;
    const int lane = threadIdx.x % 32;
    int t = 0;
    if (lane == 0) t = atomicAdd(ticket, 1);
    t = __shfl_sync(0xffffffffu, t, 0);
    if (t >= a.rows) return;
    const int64_t row = how.lower ? t : a.rows - 1 - t;
    const T *values = static_cast<const T *>(a.values);
    const int64_t start = a.offsets[row], end = a.offsets[row + 1];
    // the diagonal
    C diag = C(1);
    bool found = how.unit;
    if (!how.unit) {
        for (int64_t k = start + lane; k < end; k += 32) {
            if (a.columns[k] == row) {
                C v = Num<T>::load(values[k]);
                diag = a.conj ? conjOf(v) : v;
                found = true;
            }
        }
        const unsigned holders = __ballot_sync(0xffffffffu, found);
        found = holders != 0;
        if (found) {
            const int from = __ffs(holders) - 1;
            if constexpr (std::is_same_v<C, cfloat> || std::is_same_v<C, cdouble>) {
                diag = C(__shfl_sync(0xffffffffu, diag.real(), from), __shfl_sync(0xffffffffu, diag.imag(), from));
            } else {
                diag = __shfl_sync(0xffffffffu, diag, from);
            }
        }
        if (lane == 0 && (!found || diag == C(0))) atomicMin(pivot, int(row));
    }
    if (vector) {
        // lanes over the row's entries
        C acc = C(0);
        for (int64_t k = start + lane; k < end; k += 32) {
            const int64_t col = a.columns[k];
            if (how.lower ? col >= row : col <= row) continue;
            while (*static_cast<volatile int *>(&done[col]) == 0) {
            }
            __threadfence();
            C v = Num<T>::load(values[k]);
            if (a.conj) v = conjOf(v);
            acc += v * loadShared(&x[col]);
        }
        for (int o = 16; o > 0; o /= 2) {
            if constexpr (std::is_same_v<C, cfloat> || std::is_same_v<C, cdouble>) {
                acc = C(acc.real() + __shfl_down_sync(0xffffffffu, acc.real(), o),
                        acc.imag() + __shfl_down_sync(0xffffffffu, acc.imag(), o));
            } else {
                acc += __shfl_down_sync(0xffffffffu, acc, o);
            }
        }
        if (lane == 0) x[row] = (alpha.get() * b.get<T>(0, row, 0) - acc) / diag;
    } else {
        // lanes over the right-hand sides
        for (int64_t k = start; k < end; k++) {
            const int64_t col = a.columns[k];
            if (how.lower ? col >= row : col <= row) continue;
            if (lane == 0) {
                while (*static_cast<volatile int *>(&done[col]) == 0) {
                }
            }
            __syncwarp();
            __threadfence();
        }
        for (int64_t j0 = 0; j0 < n; j0 += 32) {
            const int64_t j = j0 + lane;
            C acc = C(0);
            for (int64_t k = start; k < end && j < n; k++) {
                const int64_t col = a.columns[k];
                if (how.lower ? col >= row : col <= row) continue;
                C v = Num<T>::load(values[k]);
                if (a.conj) v = conjOf(v);
                acc += v * loadShared(&x[col * n + j]);
            }
            if (j < n) x[row * n + j] = (alpha.get() * b.get<T>(0, row, j) - acc) / diag;
        }
    }
    __syncwarp();
    __threadfence();
    if (lane == 0) atomicExch(&done[row], 1);
}

template<typename T>
__global__ void storeSolution(const typename Num<T>::C *x, int64_t rows, int64_t n, Dense out) {
    GRID_STRIDE(i, rows * n) {
        const int64_t r = i / n, j = i % n;
        static_cast<T *>(out.values)[out.index(r, j)] = Num<T>::store(x[i]);
    }
}

struct SolveLayout {
    size_t conversion, x, done, ticket, pivot;
    Layout layout;
};

SolveLayout solveLayout(cusparseConstSpMatDescr_t a, cusparseOperation_t op, int64_t n) {
    SolveLayout l{};
    const size_t computeBytes = computeSize(a->valueType);
    l.conversion = l.layout.add(conversionBytes(a, op));
    l.x = l.layout.add(size_t(opRows(a, op)) * size_t(std::max<int64_t>(n, 1)) * computeBytes);
    l.done = l.layout.add(size_t(opRows(a, op)) * sizeof(int));
    l.ticket = l.layout.add(sizeof(int));
    l.pivot = l.layout.add(sizeof(int));
    return l;
}

// op(A) X = alpha op(B) for every batch (B, X dense; vector: one column)
template<typename T>
cusparseStatus_t solveAll(cusparseHandle_t h, cusparseConstSpMatDescr_t a, cusparseOperation_t opA, const void *values,
                          const Dense &b, Dense x, int64_t n, const void *alpha, void *buffer, bool vector,
                          int **pivotOut, int64_t xBatchStride) {
    using C = typename Num<T>::C;
    if (buffer == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    const cudaStream_t s = h->stream;
    const SolveLayout l = solveLayout(a, opA, n);
    char *buf = static_cast<char *>(buffer);
    const Solve how{(a->fill == CUSPARSE_FILL_MODE_LOWER) != (opA != CUSPARSE_OPERATION_NON_TRANSPOSE),
                    a->diag == CUSPARSE_DIAG_TYPE_UNIT};
    const Scalar<C> al = scalar<C>(h, alpha);
    const int64_t rows = opRows(a, opA);
    for (int64_t batch = 0; batch < std::max(1, a->batchCount); batch++) {
        Csr csr{};
        cusparseStatus_t st = csrOf<T>(s, a, opA, batch, buf + l.conversion, values, csr);
        if (st != CUSPARSE_STATUS_SUCCESS) return st;
        if (cudaMemsetAsync(buf + l.done, 0, size_t(rows) * sizeof(int), s) != cudaSuccess ||
            cudaMemsetAsync(buf + l.ticket, 0, sizeof(int), s) != cudaSuccess ||
            cudaMemsetAsync(buf + l.pivot, 0x7f, sizeof(int), s) != cudaSuccess) {
            return cudaStatus(cudaGetLastError());
        }
        Dense bb = b;
        bb.values = static_cast<char *>(b.values) + batch * b.batchStride * sizeof(T);
        if (rows > 0) {
            triangularSolve<T><<<blocksOf(rows * 32), 256, 0, s>>>(
                    csr, how, bb, n, al, reinterpret_cast<C *>(buf + l.x), reinterpret_cast<int *>(buf + l.done),
                    reinterpret_cast<int *>(buf + l.ticket), reinterpret_cast<int *>(buf + l.pivot), vector);
            Dense xb = x;
            xb.values = static_cast<char *>(x.values) + batch * xBatchStride * sizeof(T);
            storeSolution<T><<<blocksOf(rows * n), 256, 0, s>>>(reinterpret_cast<C *>(buf + l.x), rows, n, xb);
        }
        st = launched();
        if (st != CUSPARSE_STATUS_SUCCESS) return st;
    }
    // (zero pivots of the last batch)
    if (pivotOut != nullptr) *pivotOut = reinterpret_cast<int *>(buf + l.pivot);
    return CUSPARSE_STATUS_SUCCESS;
}

// ---- SpGEMM (C = alpha op(A) op(B), CSR): products expanded, sorted by (row, column), equal positions added up

__global__ void productCounts(Csr a, Csr b, int64_t *counts, unsigned long long *total) {
    GRID_STRIDE(row, a.rows) {
        int64_t count = 0;
        for (int64_t k = a.offsets[row]; k < a.offsets[row + 1]; k++) {
            const int64_t inner = a.columns[k];
            count += b.offsets[inner + 1] - b.offsets[inner];
        }
        counts[row] = count;
        if (total != nullptr) atomicAdd(total, (unsigned long long) count);
    }
}

template<typename T>
__global__ void expandProducts(Csr a, Csr b, const int64_t *starts, uint64_t *keys, int64_t *perm,
                               typename Num<T>::C *products) {
    const int64_t row = (blockIdx.x * int64_t(blockDim.x) + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (row >= a.rows) return;
    const T *av = static_cast<const T *>(a.values), *bv = static_cast<const T *>(b.values);
    int64_t at = starts[row];
    for (int64_t k = a.offsets[row]; k < a.offsets[row + 1]; k++) {
        const int64_t inner = a.columns[k], first = b.offsets[inner], count = b.offsets[inner + 1] - first;
        const auto x = Num<T>::load(av[k]);
        for (int64_t q = lane; q < count; q += 32) {
            keys[at + q] = uint64_t(row) * uint64_t(b.cols) + uint64_t(b.columns[first + q]);
            perm[at + q] = at + q;
            products[at + q] = x * Num<T>::load(bv[first + q]);
        }
        at += count;
    }
}

__global__ void uniqueHeads(const uint64_t *keys, int64_t n, int64_t *heads) {
    GRID_STRIDE(i, n) heads[i] = (i == 0 || keys[i] != keys[i - 1]) ? 1 : 0;
}

// starts[u] = first sorted product of unique position u
__global__ void uniqueStarts(const int64_t *positions, const int64_t *heads, int64_t n, int64_t *starts) {
    GRID_STRIDE(i, n) {
        if (heads[i]) starts[positions[i]] = i;
    }
}

template<typename T>
__global__ void reduceProducts(const uint64_t *keys, const int64_t *perm, const typename Num<T>::C *products,
                               const int64_t *starts, int64_t products_n, int64_t unique, int64_t cols,
                               Scalar<typename Num<T>::C> alpha, uint64_t *rowKeys, int64_t *columns,
                               typename Num<T>::C *values) {
    using C = typename Num<T>::C;
    GRID_STRIDE(u, unique) {
        const int64_t first = starts[u], last = u + 1 < unique ? starts[u + 1] : products_n;
        C acc = C(0);
        for (int64_t i = first; i < last; i++) acc += products[perm[i]];
        rowKeys[u] = keys[first] / uint64_t(cols);
        columns[u] = int64_t(keys[first] % uint64_t(cols));
        values[u] = alpha.get() * acc;
    }
}

struct GemmLayout1 {
    size_t counts, starts, keysIn, keysOut, permIn, permOut, products, heads, positions, uniqueStart, temp, tempBytes;
    Layout layout;
};

GemmLayout1 gemmLayout1(int64_t rows, int64_t products, size_t computeBytes) {
    GemmLayout1 l{};
    const size_t n = size_t(std::max<int64_t>(products, 1));
    size_t sortBytes = 0, scanBytes = 0, scanRows = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, sortBytes, static_cast<const uint64_t *>(nullptr),
                                    static_cast<uint64_t *>(nullptr), static_cast<const int64_t *>(nullptr),
                                    static_cast<int64_t *>(nullptr), int64_t(n));
    cub::DeviceScan::ExclusiveSum(nullptr, scanBytes, static_cast<const int64_t *>(nullptr),
                                  static_cast<int64_t *>(nullptr), int64_t(n));
    cub::DeviceScan::ExclusiveSum(nullptr, scanRows, static_cast<const int64_t *>(nullptr),
                                  static_cast<int64_t *>(nullptr), int64_t(rows + 1));
    l.tempBytes = std::max({sortBytes, scanBytes, scanRows});
    l.counts = l.layout.add(size_t(rows + 1) * 8);
    l.starts = l.layout.add(size_t(rows + 1) * 8);
    l.keysIn = l.layout.add(n * 8);
    l.keysOut = l.layout.add(n * 8);
    l.permIn = l.layout.add(n * 8);
    l.permOut = l.layout.add(n * 8);
    l.products = l.layout.add(n * computeBytes);
    l.heads = l.layout.add(n * 8);
    l.positions = l.layout.add(n * 8);
    l.uniqueStart = l.layout.add(n * 8);
    l.temp = l.layout.add(l.tempBytes);
    return l;
}

struct GemmLayout2 {
    size_t rowKeys, columns, values;
    Layout layout;
};

GemmLayout2 gemmLayout2(int64_t nnz, size_t computeBytes) {
    GemmLayout2 l{};
    const size_t n = size_t(std::max<int64_t>(nnz, 1));
    l.rowKeys = l.layout.add(n * 8);
    l.columns = l.layout.add(n * 8);
    l.values = l.layout.add(n * computeBytes);
    return l;
}

template<typename T>
__global__ void copyGemmResult(const int64_t *columns, const typename Num<T>::C *values, int64_t nnz, void *outColumns,
                               bool wide, int64_t base, T *outValues) {
    GRID_STRIDE(i, nnz) {
        storeIndex(outColumns, wide, i, columns[i] + base);
        outValues[i] = Num<T>::store(values[i]);
    }
}

// ---- csrgeam2: C = alpha A + beta B (sorted CSR, 32 bit indices)

__global__ void geamCounts(int m, const int *rowA, const int *colA, int baseA, const int *rowB, const int *colB,
                           int baseB, int64_t *counts) {
    GRID_STRIDE(r, m) {
        int i = rowA[r] - baseA, ie = rowA[r + 1] - baseA, j = rowB[r] - baseB, je = rowB[r + 1] - baseB, count = 0;
        while (i < ie || j < je) {
            const int ca = i < ie ? colA[i] - baseA : INT32_MAX, cb = j < je ? colB[j] - baseB : INT32_MAX;
            if (ca <= cb) i++;
            if (cb <= ca) j++;
            count++;
        }
        counts[r] = count;
    }
}

__global__ void geamOffsets(int m, const int64_t *starts, int baseC, int *rowC, int *total) {
    GRID_STRIDE(r, m + 1) {
        rowC[r] = int(starts[r]) + baseC;
        if (r == m && total != nullptr) *total = int(starts[r]);
    }
}

template<typename T>
__global__ void geamFill(int m, Scalar<T> alpha, const T *valA, const int *rowA, const int *colA, int baseA,
                         Scalar<T> beta, const T *valB, const int *rowB, const int *colB, int baseB, T *valC,
                         const int *rowC, int *colC, int baseC) {
    GRID_STRIDE(r, m) {
        int i = rowA[r] - baseA, ie = rowA[r + 1] - baseA, j = rowB[r] - baseB, je = rowB[r + 1] - baseB;
        int out = rowC[r] - baseC;
        const T al = alpha.get(), be = beta.get();
        while (i < ie || j < je) {
            const int ca = i < ie ? colA[i] - baseA : INT32_MAX, cb = j < je ? colB[j] - baseB : INT32_MAX;
            T v = T(0);
            if (ca <= cb) v += al * valA[i++];
            if (cb <= ca) v += be * valB[j++];
            colC[out] = (ca < cb ? ca : cb) + baseC;
            valC[out] = v;
            out++;
        }
    }
}

// ---- sorting and conversions of 32 bit index arrays

__global__ void sortKeysCoo(int nnz, const int *rows, const int *cols, uint64_t *keys, int64_t *perm) {
    GRID_STRIDE(i, nnz) {
        keys[i] = (uint64_t(uint32_t(rows[i])) << 32) | uint32_t(cols[i]);
        perm[i] = i;
    }
}

__global__ void sortKeysCsr(int m, const int *rowPtr, int base, const int *cols, uint64_t *keys, int64_t *perm) {
    const int64_t row = (blockIdx.x * int64_t(blockDim.x) + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (row >= m) return;
    for (int64_t e = rowPtr[row] - base + lane; e < rowPtr[row + 1] - base; e += 32) {
        keys[e] = (uint64_t(row) << 32) | uint32_t(cols[e]);
        perm[e] = e;
    }
}

__global__ void applyPermutation(int nnz, const int64_t *perm, const int *in0, int *out0, const int *in1, int *out1,
                                 const int *inP, int *outP) {
    GRID_STRIDE(i, nnz) {
        if (in0 != nullptr) out0[i] = in0[perm[i]];
        if (in1 != nullptr) out1[i] = in1[perm[i]];
        if (inP != nullptr) outP[i] = inP[perm[i]];
    }
}

__global__ void coo2csr(const int *rows, int nnz, int m, int *rowPtr, int base) {
    GRID_STRIDE(r, m + 1) {
        int lo = 0, hi = nnz;
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (rows[mid] - base < r) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        rowPtr[r] = lo + base;
    }
}

__global__ void identity(int n, int *p) {
    GRID_STRIDE(i, n) p[i] = int(i);
}

struct SortLayout {
    size_t keysIn, keysOut, permIn, permOut, copy0, copy1, copyP, temp, tempBytes;
    Layout layout;
};

SortLayout sortLayout(int64_t nnz) {
    SortLayout l{};
    const size_t n = size_t(std::max<int64_t>(nnz, 1));
    cub::DeviceRadixSort::SortPairs(nullptr, l.tempBytes, static_cast<const uint64_t *>(nullptr),
                                    static_cast<uint64_t *>(nullptr), static_cast<const int64_t *>(nullptr),
                                    static_cast<int64_t *>(nullptr), int64_t(n));
    l.keysIn = l.layout.add(n * 8);
    l.keysOut = l.layout.add(n * 8);
    l.permIn = l.layout.add(n * 8);
    l.permOut = l.layout.add(n * 8);
    l.copy0 = l.layout.add(n * 4);
    l.copy1 = l.layout.add(n * 4);
    l.copyP = l.layout.add(n * 4);
    l.temp = l.layout.add(l.tempBytes);
    return l;
}

// sorts the keys (and the permutation of their positions) in the buffer, then applies it to up to three arrays
cusparseStatus_t sortAndPermute(cudaStream_t s, int nnz, void *buffer, int *a0, int *a1, int *p) {
    const SortLayout l = sortLayout(nnz);
    char *buf = static_cast<char *>(buffer);
    size_t tempBytes = l.tempBytes;
    if (nnz == 0) return CUSPARSE_STATUS_SUCCESS;
    if (cub::DeviceRadixSort::SortPairs(buf + l.temp, tempBytes, reinterpret_cast<uint64_t *>(buf + l.keysIn),
                                        reinterpret_cast<uint64_t *>(buf + l.keysOut),
                                        reinterpret_cast<int64_t *>(buf + l.permIn),
                                        reinterpret_cast<int64_t *>(buf + l.permOut), int64_t(nnz), 0, 64,
                                        s) != cudaSuccess) {
        return cudaStatus(cudaGetLastError());
    }
    int *c0 = reinterpret_cast<int *>(buf + l.copy0), *c1 = reinterpret_cast<int *>(buf + l.copy1),
        *cp = reinterpret_cast<int *>(buf + l.copyP);
    if (a0 != nullptr) cudaMemcpyAsync(c0, a0, size_t(nnz) * 4, cudaMemcpyDeviceToDevice, s);
    if (a1 != nullptr) cudaMemcpyAsync(c1, a1, size_t(nnz) * 4, cudaMemcpyDeviceToDevice, s);
    if (p != nullptr) cudaMemcpyAsync(cp, p, size_t(nnz) * 4, cudaMemcpyDeviceToDevice, s);
    applyPermutation<<<blocksOf(nnz), 256, 0, s>>>(nnz, reinterpret_cast<int64_t *>(buf + l.permOut),
                                                   a0 ? c0 : nullptr, a0, a1 ? c1 : nullptr, a1, p ? cp : nullptr, p);
    return launched();
}

// ---- BSR (legacy): block rows of blockDim x blockDim blocks, row- or column-major inside

template<typename T>
__device__ T bsrValue(const T *values, int64_t block, int bs, int r, int c, bool rowMajor) {
    return values[block * bs * bs + (rowMajor ? r * bs + c : c * bs + r)];
}

template<typename T>
__global__ void bsrmv(int mb, int bs, Scalar<T> alpha, const T *values, const int *rowPtr, const int *colInd, int base,
                      bool rowMajor, const T *x, Scalar<T> beta, T *y) {
    GRID_STRIDE(row, int64_t(mb) * bs) {
        const int64_t br = row / bs;
        const int r = int(row % bs);
        T acc = T(0);
        for (int64_t k = rowPtr[br] - base; k < rowPtr[br + 1] - base; k++) {
            const int64_t bc = colInd[k] - base;
            for (int c = 0; c < bs; c++) acc += bsrValue(values, k, bs, r, c, rowMajor) * x[bc * bs + c];
        }
        y[row] = axpby(alpha.get(), acc, beta.get(), &y[row]);
    }
}

// C (m x n, column-major) = alpha A op(B) + beta C
template<typename T>
__global__ void bsrmm(int mb, int n, int bs, Scalar<T> alpha, const T *values, const int *rowPtr, const int *colInd,
                      int base, bool rowMajor, const T *b, int ldb, bool transB, Scalar<T> beta, T *c, int ldc) {
    GRID_STRIDE(i, int64_t(mb) * bs * n) {
        const int64_t row = i % (int64_t(mb) * bs), col = i / (int64_t(mb) * bs), br = row / bs;
        const int r = int(row % bs);
        T acc = T(0);
        for (int64_t k = rowPtr[br] - base; k < rowPtr[br + 1] - base; k++) {
            const int64_t bc = colInd[k] - base;
            for (int q = 0; q < bs; q++) {
                const int64_t inner = bc * bs + q;
                acc += bsrValue(values, k, bs, r, q, rowMajor) * (transB ? b[col + inner * ldb] : b[inner + col * ldb]);
            }
        }
        T *out = &c[row + col * ldc];
        *out = axpby(alpha.get(), acc, beta.get(), out);
    }
}

// a BSR matrix's CSR (64 bit indices), entries in each row by column
template<typename T>
__global__ void bsrToCsr(int mb, int bs, const T *values, const int *rowPtr, const int *colInd, int base,
                         bool rowMajor, int64_t *offsets, int64_t *columns, T *out) {
    GRID_STRIDE(row, int64_t(mb) * bs + 1) {
        const int64_t br = row / bs;
        const int r = int(row % bs);
        if (row == int64_t(mb) * bs) {
            offsets[row] = int64_t(rowPtr[mb] - rowPtr[0]) * bs * bs;
            continue;
        }
        const int64_t first = rowPtr[br] - base, count = rowPtr[br + 1] - rowPtr[br];
        int64_t at = first * bs * bs + int64_t(r) * count * bs;
        offsets[row] = at;
        for (int64_t k = first; k < first + count; k++) {
            const int64_t bc = colInd[k] - base;
            for (int c = 0; c < bs; c++) {
                columns[at] = bc * bs + c;
                out[at++] = bsrValue(values, k, bs, r, c, rowMajor);
            }
        }
    }
}

// ---- conversions between sparse and dense, sparse vectors

template<typename T>
__global__ void scatterToDense(Csr a, bool transposed, Dense out) {
    const int64_t row = (blockIdx.x * int64_t(blockDim.x) + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (row >= a.rows) return;
    for (int64_t k = a.offsets[row] + lane; k < a.offsets[row + 1]; k += 32) {
        const int64_t col = a.columns[k];
        static_cast<T *>(out.values)[transposed ? out.index(col, row) : out.index(row, col)] =
                static_cast<const T *>(a.values)[k];
    }
}

template<typename T>
__global__ void zeroDense(Dense d, int64_t rows, int64_t cols) {
    GRID_STRIDE(i, rows * cols) static_cast<T *>(d.values)[d.index(i % rows, i / rows)] = T(0);
}

// nonzeros of each major line (rows: CSR / COO, columns: CSC) of a dense matrix
template<typename T>
__global__ void denseCounts(Dense d, int64_t majors, int64_t minors, bool byRow, int64_t *counts) {
    GRID_STRIDE(m, majors) {
        int64_t count = 0;
        for (int64_t q = 0; q < minors; q++) {
            const T v = static_cast<const T *>(d.values)[byRow ? d.index(m, q) : d.index(q, m)];
            if (!(Num<T>::load(v) == typename Num<T>::C(0))) count++;
        }
        counts[m] = count;
    }
}

template<typename T>
__global__ void denseFill(Dense d, int64_t majors, int64_t minors, bool byRow, const int64_t *starts, int64_t base,
                          void *p0, bool wide0, void *p1, bool wide1, bool coo, T *values) {
    GRID_STRIDE(m, majors) {
        int64_t at = starts[m];
        if (!coo) storeIndex(p0, wide0, m, at + base);
        if (!coo && m == majors - 1) storeIndex(p0, wide0, majors, starts[majors] + base);
        for (int64_t q = 0; q < minors; q++) {
            const T v = static_cast<const T *>(d.values)[byRow ? d.index(m, q) : d.index(q, m)];
            if (Num<T>::load(v) == typename Num<T>::C(0)) continue;
            if (coo) storeIndex(p0, wide0, at, m + base);
            storeIndex(p1, wide1, at, q + base);
            values[at++] = v;
        }
    }
}

template<typename T>
__global__ void spvvKernel(Idx idx, const T *x, const T *y, int64_t nnz, bool conj, typename Num<T>::C *result) {
    using C = typename Num<T>::C;
    __shared__ C partial[256];
    C acc = C(0);
    for (int64_t i = threadIdx.x; i < nnz; i += blockDim.x) {
        C v = Num<T>::load(x[i]);
        if (conj) v = conjOf(v);
        acc += v * Num<T>::load(y[idx[i]]);
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o /= 2) {
        if (threadIdx.x < o) partial[threadIdx.x] += partial[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) *result = partial[0];
}

template<typename T>
__global__ void scaleVector(T *y, int64_t n, Scalar<typename Num<T>::C> beta) {
    GRID_STRIDE(i, n) {
        const auto b = beta.get();
        y[i] = Num<T>::store(b == decltype(b)(0) ? decltype(b)(0) : b * Num<T>::load(y[i]));
    }
}

template<typename T>
__global__ void sparseVectorOp(Idx idx, T *x, T *y, int64_t nnz, int mode, Scalar<typename Num<T>::C> alpha) {
    GRID_STRIDE(i, nnz) {
        const int64_t j = idx[i];
        if (mode == 0) {
            y[j] = Num<T>::store(Num<T>::load(y[j]) + alpha.get() * Num<T>::load(x[i]));  // axpby
        } else if (mode == 1) {
            x[i] = y[j];  // gather
        } else {
            y[j] = x[i];  // scatter
        }
    }
}

bool computeTypeOk(cudaDataType value, cudaDataType compute) {
    if (value == CUDA_R_16F || value == CUDA_R_16BF) return compute == CUDA_R_32F || compute == value;
    return compute == value;
}

} // namespace

// ==== handle

cusparseStatus_t cusparseCreate(cusparseHandle_t *handle) {
    if (handle == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *handle = new cusparseContext{SPARSE_MAGIC, nullptr, CUSPARSE_POINTER_MODE_HOST};
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroy(cusparseHandle_t handle) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    handle->magic = 0;
    delete handle;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseGetVersion(cusparseHandle_t handle, int *version) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (version == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *version = CUSPARSE_VERSION;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseGetProperty(libraryPropertyType type, int *value) {
    if (value == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    switch (type) {
        case MAJOR_VERSION: *value = CUSPARSE_VER_MAJOR; return CUSPARSE_STATUS_SUCCESS;
        case MINOR_VERSION: *value = CUSPARSE_VER_MINOR; return CUSPARSE_STATUS_SUCCESS;
        case PATCH_LEVEL: *value = CUSPARSE_VER_PATCH; return CUSPARSE_STATUS_SUCCESS;
        default: return CUSPARSE_STATUS_INVALID_VALUE;
    }
}

const char *cusparseGetErrorName(cusparseStatus_t status) {
    switch (status) {
        case CUSPARSE_STATUS_SUCCESS: return "CUSPARSE_STATUS_SUCCESS";
        case CUSPARSE_STATUS_NOT_INITIALIZED: return "CUSPARSE_STATUS_NOT_INITIALIZED";
        case CUSPARSE_STATUS_ALLOC_FAILED: return "CUSPARSE_STATUS_ALLOC_FAILED";
        case CUSPARSE_STATUS_INVALID_VALUE: return "CUSPARSE_STATUS_INVALID_VALUE";
        case CUSPARSE_STATUS_ARCH_MISMATCH: return "CUSPARSE_STATUS_ARCH_MISMATCH";
        case CUSPARSE_STATUS_MAPPING_ERROR: return "CUSPARSE_STATUS_MAPPING_ERROR";
        case CUSPARSE_STATUS_EXECUTION_FAILED: return "CUSPARSE_STATUS_EXECUTION_FAILED";
        case CUSPARSE_STATUS_INTERNAL_ERROR: return "CUSPARSE_STATUS_INTERNAL_ERROR";
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        case CUSPARSE_STATUS_ZERO_PIVOT: return "CUSPARSE_STATUS_ZERO_PIVOT";
        case CUSPARSE_STATUS_NOT_SUPPORTED: return "CUSPARSE_STATUS_NOT_SUPPORTED";
        case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES: return "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES";
        default: return "CUSPARSE_STATUS_UNKNOWN";
    }
}

const char *cusparseGetErrorString(cusparseStatus_t status) {
    switch (status) {
        case CUSPARSE_STATUS_SUCCESS: return "success";
        case CUSPARSE_STATUS_NOT_SUPPORTED: return "operation not supported by LibreCUDA's cuSPARSE";
        default: return cusparseGetErrorName(status);
    }
}

cusparseStatus_t cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    handle->stream = streamId;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseGetStream(cusparseHandle_t handle, cudaStream_t *streamId) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (streamId == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *streamId = handle->stream;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseGetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t *mode) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (mode == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *mode = handle->pointerMode;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t mode) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    handle->pointerMode = mode;
    return CUSPARSE_STATUS_SUCCESS;
}

// ==== legacy matrix descriptors

cusparseStatus_t cusparseCreateMatDescr(cusparseMatDescr_t *descrA) {
    if (descrA == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *descrA = new cusparseMatDescr{CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT,
                                   CUSPARSE_INDEX_BASE_ZERO};
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroyMatDescr(cusparseMatDescr_t descrA) {
    delete descrA;
    return CUSPARSE_STATUS_SUCCESS;
}

#define DESCRIPTOR_FIELD(Name, Type, field)                                                                            \
cusparseStatus_t cusparseSet##Name(cusparseMatDescr_t descrA, Type value) {                                            \
    if (descrA == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;                                                       \
    descrA->field = value;                                                                                             \
    return CUSPARSE_STATUS_SUCCESS;                                                                                    \
}                                                                                                                      \
Type cusparseGet##Name(const cusparseMatDescr_t descrA) {                                                              \
    return descrA->field;                                                                                              \
}

DESCRIPTOR_FIELD(MatType, cusparseMatrixType_t, type)
DESCRIPTOR_FIELD(MatFillMode, cusparseFillMode_t, fillMode)
DESCRIPTOR_FIELD(MatDiagType, cusparseDiagType_t, diagType)
DESCRIPTOR_FIELD(MatIndexBase, cusparseIndexBase_t, indexBase)

// ==== generic API descriptors

namespace {

cusparseStatus_t newSpMat(cusparseSpMatDescr_t *d, cusparseFormat_t format, int64_t rows, int64_t cols, int64_t nnz,
                          const void *p0, const void *p1, const void *values, cusparseIndexType_t t0,
                          cusparseIndexType_t t1, cusparseIndexBase_t base, cudaDataType type) {
    if (d == nullptr || rows < 0 || cols < 0 || nnz < 0) return CUSPARSE_STATUS_INVALID_VALUE;
    if ((t0 != CUSPARSE_INDEX_32I && t0 != CUSPARSE_INDEX_64I) || (t1 != CUSPARSE_INDEX_32I && t1 != CUSPARSE_INDEX_64I))
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    if (typeSize(type) == 0) return CUSPARSE_STATUS_NOT_SUPPORTED;
    *d = new cusparseSpMatDescr{format, rows, cols, nnz, const_cast<void *>(p0), const_cast<void *>(p1),
                                const_cast<void *>(values), t0, t1, base, type, 1, 0, 0,
                                CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT};
    return CUSPARSE_STATUS_SUCCESS;
}

} // namespace

cusparseStatus_t cusparseCreateCsr(cusparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
                                   void *csrRowOffsets, void *csrColInd, void *csrValues,
                                   cusparseIndexType_t csrRowOffsetsType, cusparseIndexType_t csrColIndType,
                                   cusparseIndexBase_t idxBase, cudaDataType valueType) {
    return newSpMat(spMatDescr, CUSPARSE_FORMAT_CSR, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues,
                    csrRowOffsetsType, csrColIndType, idxBase, valueType);
}

cusparseStatus_t cusparseCreateConstCsr(cusparseConstSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
                                        const void *csrRowOffsets, const void *csrColInd, const void *csrValues,
                                        cusparseIndexType_t csrRowOffsetsType, cusparseIndexType_t csrColIndType,
                                        cusparseIndexBase_t idxBase, cudaDataType valueType) {
    return newSpMat(const_cast<cusparseSpMatDescr_t *>(spMatDescr), CUSPARSE_FORMAT_CSR, rows, cols, nnz,
                    csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, valueType);
}

cusparseStatus_t cusparseCreateCsc(cusparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
                                   void *cscColOffsets, void *cscRowInd, void *cscValues,
                                   cusparseIndexType_t cscColOffsetsType, cusparseIndexType_t cscRowIndType,
                                   cusparseIndexBase_t idxBase, cudaDataType valueType) {
    return newSpMat(spMatDescr, CUSPARSE_FORMAT_CSC, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues,
                    cscColOffsetsType, cscRowIndType, idxBase, valueType);
}

cusparseStatus_t cusparseCreateConstCsc(cusparseConstSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
                                        const void *cscColOffsets, const void *cscRowInd, const void *cscValues,
                                        cusparseIndexType_t cscColOffsetsType, cusparseIndexType_t cscRowIndType,
                                        cusparseIndexBase_t idxBase, cudaDataType valueType) {
    return newSpMat(const_cast<cusparseSpMatDescr_t *>(spMatDescr), CUSPARSE_FORMAT_CSC, rows, cols, nnz,
                    cscColOffsets, cscRowInd, cscValues, cscColOffsetsType, cscRowIndType, idxBase, valueType);
}

cusparseStatus_t cusparseCreateCoo(cusparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
                                   void *cooRowInd, void *cooColInd, void *cooValues, cusparseIndexType_t cooIdxType,
                                   cusparseIndexBase_t idxBase, cudaDataType valueType) {
    return newSpMat(spMatDescr, CUSPARSE_FORMAT_COO, rows, cols, nnz, cooRowInd, cooColInd, cooValues, cooIdxType,
                    cooIdxType, idxBase, valueType);
}

cusparseStatus_t cusparseCreateConstCoo(cusparseConstSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
                                        const void *cooRowInd, const void *cooColInd, const void *cooValues,
                                        cusparseIndexType_t cooIdxType, cusparseIndexBase_t idxBase,
                                        cudaDataType valueType) {
    return newSpMat(const_cast<cusparseSpMatDescr_t *>(spMatDescr), CUSPARSE_FORMAT_COO, rows, cols, nnz, cooRowInd,
                    cooColInd, cooValues, cooIdxType, cooIdxType, idxBase, valueType);
}

cusparseStatus_t cusparseDestroySpMat(cusparseConstSpMatDescr_t spMatDescr) {
    delete spMatDescr;
    return CUSPARSE_STATUS_SUCCESS;
}

namespace {

cusparseStatus_t getSpMat(cusparseConstSpMatDescr_t d, cusparseFormat_t format, int64_t *rows, int64_t *cols,
                          int64_t *nnz, void **p0, void **p1, void **values, cusparseIndexType_t *t0,
                          cusparseIndexType_t *t1, cusparseIndexBase_t *base, cudaDataType *type) {
    if (d == nullptr || d->format != format) return CUSPARSE_STATUS_INVALID_VALUE;
    if (rows) *rows = d->rows;
    if (cols) *cols = d->cols;
    if (nnz) *nnz = d->nnz;
    if (p0) *p0 = d->p0;
    if (p1) *p1 = d->p1;
    if (values) *values = d->values;
    if (t0) *t0 = d->type0;
    if (t1) *t1 = d->type1;
    if (base) *base = d->base;
    if (type) *type = d->valueType;
    return CUSPARSE_STATUS_SUCCESS;
}

} // namespace

cusparseStatus_t cusparseCsrGet(cusparseSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols, int64_t *nnz,
                                void **csrRowOffsets, void **csrColInd, void **csrValues,
                                cusparseIndexType_t *csrRowOffsetsType, cusparseIndexType_t *csrColIndType,
                                cusparseIndexBase_t *idxBase, cudaDataType *valueType) {
    return getSpMat(spMatDescr, CUSPARSE_FORMAT_CSR, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues,
                    csrRowOffsetsType, csrColIndType, idxBase, valueType);
}

cusparseStatus_t cusparseConstCsrGet(cusparseConstSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols, int64_t *nnz,
                                     const void **csrRowOffsets, const void **csrColInd, const void **csrValues,
                                     cusparseIndexType_t *csrRowOffsetsType, cusparseIndexType_t *csrColIndType,
                                     cusparseIndexBase_t *idxBase, cudaDataType *valueType) {
    return getSpMat(spMatDescr, CUSPARSE_FORMAT_CSR, rows, cols, nnz, const_cast<void **>(csrRowOffsets),
                    const_cast<void **>(csrColInd), const_cast<void **>(csrValues), csrRowOffsetsType, csrColIndType,
                    idxBase, valueType);
}

cusparseStatus_t cusparseCscGet(cusparseSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols, int64_t *nnz,
                                void **cscColOffsets, void **cscRowInd, void **cscValues,
                                cusparseIndexType_t *cscColOffsetsType, cusparseIndexType_t *cscRowIndType,
                                cusparseIndexBase_t *idxBase, cudaDataType *valueType) {
    return getSpMat(spMatDescr, CUSPARSE_FORMAT_CSC, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues,
                    cscColOffsetsType, cscRowIndType, idxBase, valueType);
}

cusparseStatus_t cusparseCooGet(cusparseSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols, int64_t *nnz,
                                void **cooRowInd, void **cooColInd, void **cooValues, cusparseIndexType_t *idxType,
                                cusparseIndexBase_t *idxBase, cudaDataType *valueType) {
    return getSpMat(spMatDescr, CUSPARSE_FORMAT_COO, rows, cols, nnz, cooRowInd, cooColInd, cooValues, idxType,
                    nullptr, idxBase, valueType);
}

cusparseStatus_t cusparseCsrSetPointers(cusparseSpMatDescr_t spMatDescr, void *csrRowOffsets, void *csrColInd,
                                        void *csrValues) {
    if (spMatDescr == nullptr || spMatDescr->format != CUSPARSE_FORMAT_CSR) return CUSPARSE_STATUS_INVALID_VALUE;
    spMatDescr->p0 = csrRowOffsets;
    spMatDescr->p1 = csrColInd;
    spMatDescr->values = csrValues;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCscSetPointers(cusparseSpMatDescr_t spMatDescr, void *cscColOffsets, void *cscRowInd,
                                        void *cscValues) {
    if (spMatDescr == nullptr || spMatDescr->format != CUSPARSE_FORMAT_CSC) return CUSPARSE_STATUS_INVALID_VALUE;
    spMatDescr->p0 = cscColOffsets;
    spMatDescr->p1 = cscRowInd;
    spMatDescr->values = cscValues;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCooSetPointers(cusparseSpMatDescr_t spMatDescr, void *cooRows, void *cooColumns,
                                        void *cooValues) {
    if (spMatDescr == nullptr || spMatDescr->format != CUSPARSE_FORMAT_COO) return CUSPARSE_STATUS_INVALID_VALUE;
    spMatDescr->p0 = cooRows;
    spMatDescr->p1 = cooColumns;
    spMatDescr->values = cooValues;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMatGetSize(cusparseConstSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols,
                                      int64_t *nnz) {
    if (spMatDescr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (rows) *rows = spMatDescr->rows;
    if (cols) *cols = spMatDescr->cols;
    if (nnz) *nnz = spMatDescr->nnz;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMatGetFormat(cusparseConstSpMatDescr_t spMatDescr, cusparseFormat_t *format) {
    if (spMatDescr == nullptr || format == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *format = spMatDescr->format;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMatGetIndexBase(cusparseConstSpMatDescr_t spMatDescr, cusparseIndexBase_t *idxBase) {
    if (spMatDescr == nullptr || idxBase == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *idxBase = spMatDescr->base;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMatGetValues(cusparseSpMatDescr_t spMatDescr, void **values) {
    if (spMatDescr == nullptr || values == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *values = spMatDescr->values;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseConstSpMatGetValues(cusparseConstSpMatDescr_t spMatDescr, const void **values) {
    if (spMatDescr == nullptr || values == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *values = spMatDescr->values;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMatSetValues(cusparseSpMatDescr_t spMatDescr, void *values) {
    if (spMatDescr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    spMatDescr->values = values;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMatGetStridedBatch(cusparseConstSpMatDescr_t spMatDescr, int *batchCount) {
    if (spMatDescr == nullptr || batchCount == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *batchCount = spMatDescr->batchCount;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCsrSetStridedBatch(cusparseSpMatDescr_t spMatDescr, int batchCount,
                                            int64_t offsetsBatchStride, int64_t columnsValuesBatchStride) {
    if (spMatDescr == nullptr || batchCount < 1) return CUSPARSE_STATUS_INVALID_VALUE;
    spMatDescr->batchCount = batchCount;
    spMatDescr->stride0 = offsetsBatchStride;
    spMatDescr->stride1 = columnsValuesBatchStride;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCooSetStridedBatch(cusparseSpMatDescr_t spMatDescr, int batchCount, int64_t batchStride) {
    if (spMatDescr == nullptr || batchCount < 1) return CUSPARSE_STATUS_INVALID_VALUE;
    spMatDescr->batchCount = batchCount;
    spMatDescr->stride0 = spMatDescr->stride1 = batchStride;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMatGetAttribute(cusparseConstSpMatDescr_t spMatDescr, cusparseSpMatAttribute_t attribute,
                                           void *data, size_t dataSize) {
    if (spMatDescr == nullptr || data == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (attribute == CUSPARSE_SPMAT_FILL_MODE && dataSize >= sizeof(cusparseFillMode_t)) {
        *static_cast<cusparseFillMode_t *>(data) = spMatDescr->fill;
    } else if (attribute == CUSPARSE_SPMAT_DIAG_TYPE && dataSize >= sizeof(cusparseDiagType_t)) {
        *static_cast<cusparseDiagType_t *>(data) = spMatDescr->diag;
    } else {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMatSetAttribute(cusparseSpMatDescr_t spMatDescr, cusparseSpMatAttribute_t attribute,
                                           void *data, size_t dataSize) {
    if (spMatDescr == nullptr || data == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (attribute == CUSPARSE_SPMAT_FILL_MODE && dataSize >= sizeof(cusparseFillMode_t)) {
        spMatDescr->fill = *static_cast<const cusparseFillMode_t *>(data);
    } else if (attribute == CUSPARSE_SPMAT_DIAG_TYPE && dataSize >= sizeof(cusparseDiagType_t)) {
        spMatDescr->diag = *static_cast<const cusparseDiagType_t *>(data);
    } else {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateDnMat(cusparseDnMatDescr_t *dnMatDescr, int64_t rows, int64_t cols, int64_t ld,
                                     void *values, cudaDataType valueType, cusparseOrder_t order) {
    if (dnMatDescr == nullptr || rows < 0 || cols < 0) return CUSPARSE_STATUS_INVALID_VALUE;
    if (typeSize(valueType) == 0) return CUSPARSE_STATUS_NOT_SUPPORTED;
    *dnMatDescr = new cusparseDnMatDescr{rows, cols, ld, values, valueType, order, 1, 0};
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateConstDnMat(cusparseConstDnMatDescr_t *dnMatDescr, int64_t rows, int64_t cols,
                                          int64_t ld, const void *values, cudaDataType valueType,
                                          cusparseOrder_t order) {
    return cusparseCreateDnMat(const_cast<cusparseDnMatDescr_t *>(dnMatDescr), rows, cols, ld,
                               const_cast<void *>(values), valueType, order);
}

cusparseStatus_t cusparseDestroyDnMat(cusparseConstDnMatDescr_t dnMatDescr) {
    delete dnMatDescr;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnMatGet(cusparseDnMatDescr_t dnMatDescr, int64_t *rows, int64_t *cols, int64_t *ld,
                                  void **values, cudaDataType *type, cusparseOrder_t *order) {
    if (dnMatDescr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (rows) *rows = dnMatDescr->rows;
    if (cols) *cols = dnMatDescr->cols;
    if (ld) *ld = dnMatDescr->ld;
    if (values) *values = dnMatDescr->values;
    if (type) *type = dnMatDescr->valueType;
    if (order) *order = dnMatDescr->order;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnMatGetValues(cusparseDnMatDescr_t dnMatDescr, void **values) {
    if (dnMatDescr == nullptr || values == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *values = dnMatDescr->values;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnMatSetValues(cusparseDnMatDescr_t dnMatDescr, void *values) {
    if (dnMatDescr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    dnMatDescr->values = values;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnMatSetStridedBatch(cusparseDnMatDescr_t dnMatDescr, int batchCount, int64_t batchStride) {
    if (dnMatDescr == nullptr || batchCount < 1) return CUSPARSE_STATUS_INVALID_VALUE;
    dnMatDescr->batchCount = batchCount;
    dnMatDescr->batchStride = batchStride;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnMatGetStridedBatch(cusparseConstDnMatDescr_t dnMatDescr, int *batchCount,
                                              int64_t *batchStride) {
    if (dnMatDescr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (batchCount) *batchCount = dnMatDescr->batchCount;
    if (batchStride) *batchStride = dnMatDescr->batchStride;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t *dnVecDescr, int64_t size, void *values,
                                     cudaDataType valueType) {
    if (dnVecDescr == nullptr || size < 0) return CUSPARSE_STATUS_INVALID_VALUE;
    if (typeSize(valueType) == 0) return CUSPARSE_STATUS_NOT_SUPPORTED;
    *dnVecDescr = new cusparseDnVecDescr{size, values, valueType};
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateConstDnVec(cusparseConstDnVecDescr_t *dnVecDescr, int64_t size, const void *values,
                                          cudaDataType valueType) {
    return cusparseCreateDnVec(const_cast<cusparseDnVecDescr_t *>(dnVecDescr), size, const_cast<void *>(values),
                               valueType);
}

cusparseStatus_t cusparseDestroyDnVec(cusparseConstDnVecDescr_t dnVecDescr) {
    delete dnVecDescr;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnVecGet(cusparseDnVecDescr_t dnVecDescr, int64_t *size, void **values,
                                  cudaDataType *valueType) {
    if (dnVecDescr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (size) *size = dnVecDescr->size;
    if (values) *values = dnVecDescr->values;
    if (valueType) *valueType = dnVecDescr->valueType;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnVecGetValues(cusparseDnVecDescr_t dnVecDescr, void **values) {
    if (dnVecDescr == nullptr || values == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *values = dnVecDescr->values;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnVecSetValues(cusparseDnVecDescr_t dnVecDescr, void *values) {
    if (dnVecDescr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    dnVecDescr->values = values;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateSpVec(cusparseSpVecDescr_t *spVecDescr, int64_t size, int64_t nnz, void *indices,
                                     void *values, cusparseIndexType_t idxType, cusparseIndexBase_t idxBase,
                                     cudaDataType valueType) {
    if (spVecDescr == nullptr || size < 0 || nnz < 0) return CUSPARSE_STATUS_INVALID_VALUE;
    if ((idxType != CUSPARSE_INDEX_32I && idxType != CUSPARSE_INDEX_64I) || typeSize(valueType) == 0)
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    *spVecDescr = new cusparseSpVecDescr{size, nnz, indices, values, idxType, idxBase, valueType};
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateConstSpVec(cusparseConstSpVecDescr_t *spVecDescr, int64_t size, int64_t nnz,
                                          const void *indices, const void *values, cusparseIndexType_t idxType,
                                          cusparseIndexBase_t idxBase, cudaDataType valueType) {
    return cusparseCreateSpVec(const_cast<cusparseSpVecDescr_t *>(spVecDescr), size, nnz, const_cast<void *>(indices),
                               const_cast<void *>(values), idxType, idxBase, valueType);
}

cusparseStatus_t cusparseDestroySpVec(cusparseConstSpVecDescr_t spVecDescr) {
    delete spVecDescr;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpVecGet(cusparseSpVecDescr_t spVecDescr, int64_t *size, int64_t *nnz, void **indices,
                                  void **values, cusparseIndexType_t *idxType, cusparseIndexBase_t *idxBase,
                                  cudaDataType *valueType) {
    if (spVecDescr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (size) *size = spVecDescr->size;
    if (nnz) *nnz = spVecDescr->nnz;
    if (indices) *indices = spVecDescr->indices;
    if (values) *values = spVecDescr->values;
    if (idxType) *idxType = spVecDescr->idxType;
    if (idxBase) *idxBase = spVecDescr->base;
    if (valueType) *valueType = spVecDescr->valueType;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpVecGetIndexBase(cusparseConstSpVecDescr_t spVecDescr, cusparseIndexBase_t *idxBase) {
    if (spVecDescr == nullptr || idxBase == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *idxBase = spVecDescr->base;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpVecGetValues(cusparseSpVecDescr_t spVecDescr, void **values) {
    if (spVecDescr == nullptr || values == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *values = spVecDescr->values;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpVecSetValues(cusparseSpVecDescr_t spVecDescr, void *values) {
    if (spVecDescr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    spVecDescr->values = values;
    return CUSPARSE_STATUS_SUCCESS;
}

// ==== SpMV

cusparseStatus_t cusparseSpMV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, const void *alpha,
                                         cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX,
                                         const void *beta, cusparseDnVecDescr_t vecY, cudaDataType computeType,
                                         cusparseSpMVAlg_t alg, size_t *bufferSize) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (matA == nullptr || bufferSize == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *bufferSize = std::max<size_t>(conversionBytes(matA, opA), 4);
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMV_preprocess(cusparseHandle_t handle, cusparseOperation_t, const void *,
                                         cusparseConstSpMatDescr_t, cusparseConstDnVecDescr_t, const void *,
                                         cusparseDnVecDescr_t, cudaDataType, cusparseSpMVAlg_t, void *) {
    return valid(handle) ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_NOT_INITIALIZED;
}

cusparseStatus_t cusparseSpMV(cusparseHandle_t handle, cusparseOperation_t opA, const void *alpha,
                              cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, const void *beta,
                              cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t alg,
                              void *externalBuffer) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (matA == nullptr || vecX == nullptr || vecY == nullptr || alpha == nullptr || beta == nullptr)
        return CUSPARSE_STATUS_INVALID_VALUE;
    if (!supportedFormat(matA) || vecX->valueType != matA->valueType || vecY->valueType != matA->valueType ||
        !computeTypeOk(matA->valueType, computeType))
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    if (vecX->size != opCols(matA, opA) || vecY->size != opRows(matA, opA)) return CUSPARSE_STATUS_INVALID_VALUE;
    return withType(matA->valueType, [&](auto tag) {
        using T = typename decltype(tag)::type;
        using C = typename Num<T>::C;
        Csr csr{};
        cusparseStatus_t st = csrOf<T>(handle->stream, matA, opA, 0, externalBuffer, nullptr, csr);
        if (st != CUSPARSE_STATUS_SUCCESS) return st;
        if (csr.rows == 0) return CUSPARSE_STATUS_SUCCESS;
        spmv<T><<<blocksOf(csr.rows * 32), 256, 0, handle->stream>>>(
                csr, static_cast<const T *>(vecX->values), static_cast<T *>(vecY->values), scalar<C>(handle, alpha),
                scalar<C>(handle, beta));
        return launched();
    });
}

// ==== SpMM

cusparseStatus_t cusparseSpMM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
                                         const void *alpha, cusparseConstSpMatDescr_t matA,
                                         cusparseConstDnMatDescr_t matB, const void *beta, cusparseDnMatDescr_t matC,
                                         cudaDataType computeType, cusparseSpMMAlg_t alg, size_t *bufferSize) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (matA == nullptr || bufferSize == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *bufferSize = std::max<size_t>(conversionBytes(matA, opA), 4);
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMM_preprocess(cusparseHandle_t handle, cusparseOperation_t, cusparseOperation_t,
                                         const void *, cusparseConstSpMatDescr_t, cusparseConstDnMatDescr_t,
                                         const void *, cusparseDnMatDescr_t, cudaDataType, cusparseSpMMAlg_t, void *) {
    return valid(handle) ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_NOT_INITIALIZED;
}

cusparseStatus_t cusparseSpMM(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
                              const void *alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB,
                              const void *beta, cusparseDnMatDescr_t matC, cudaDataType computeType,
                              cusparseSpMMAlg_t alg, void *externalBuffer) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (matA == nullptr || matB == nullptr || matC == nullptr || alpha == nullptr || beta == nullptr)
        return CUSPARSE_STATUS_INVALID_VALUE;
    if (!supportedFormat(matA) || matB->valueType != matA->valueType || matC->valueType != matA->valueType ||
        !computeTypeOk(matA->valueType, computeType))
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    const int64_t m = opRows(matA, opA), k = opCols(matA, opA);
    const int64_t bRows = opB == CUSPARSE_OPERATION_NON_TRANSPOSE ? matB->rows : matB->cols;
    const int64_t n = opB == CUSPARSE_OPERATION_NON_TRANSPOSE ? matB->cols : matB->rows;
    if (bRows != k || matC->rows != m || matC->cols != n) return CUSPARSE_STATUS_INVALID_VALUE;
    const int batches = std::max({matA->batchCount, matB->batchCount, matC->batchCount});
    return withType(matA->valueType, [&](auto tag) {
        using T = typename decltype(tag)::type;
        using C = typename Num<T>::C;
        const Dense b = dense(matB, opB), c = dense(matC);
        const bool converted = needsConversion(matA, opA);
        // converted matrices: one batch at a time
        for (int batch = 0; batch < (converted ? batches : 1); batch++) {
            Csr csr{};
            cusparseStatus_t st = csrOf<T>(handle->stream, matA, opA, matA->batchCount > 1 ? batch : 0,
                                           externalBuffer, nullptr, csr);
            if (st != CUSPARSE_STATUS_SUCCESS) return st;
            if (m == 0 || n == 0) return CUSPARSE_STATUS_SUCCESS;
            Dense bb = b, cc = c;
            int64_t offsetsStride = matA->batchCount > 1 ? matA->stride0 : 0;
            int64_t columnsStride = matA->batchCount > 1 ? matA->stride1 : 0;
            unsigned int grid = unsigned(batches);
            if (converted) {
                bb.values = static_cast<char *>(b.values) + batch * b.batchStride * sizeof(T);
                cc.values = static_cast<char *>(c.values) + batch * c.batchStride * sizeof(T);
                offsetsStride = columnsStride = 0;
                grid = 1;
            }
            const int64_t warps = m * ((n + 31) / 32);
            spmm<T><<<dim3(blocksOf(warps * 32), grid), 256, 0, handle->stream>>>(
                    csr, offsetsStride, columnsStride, bb, cc, n, scalar<C>(handle, alpha), scalar<C>(handle, beta));
            st = launched();
            if (st != CUSPARSE_STATUS_SUCCESS) return st;
        }
        return CUSPARSE_STATUS_SUCCESS;
    });
}

// ==== SDDMM

cusparseStatus_t cusparseSDDMM_bufferSize(cusparseHandle_t handle, cusparseOperation_t, cusparseOperation_t,
                                          const void *, cusparseConstDnMatDescr_t, cusparseConstDnMatDescr_t,
                                          const void *, cusparseSpMatDescr_t, cudaDataType, cusparseSDDMMAlg_t,
                                          size_t *bufferSize) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (bufferSize == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *bufferSize = 4;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSDDMM_preprocess(cusparseHandle_t handle, cusparseOperation_t, cusparseOperation_t,
                                          const void *, cusparseConstDnMatDescr_t, cusparseConstDnMatDescr_t,
                                          const void *, cusparseSpMatDescr_t, cudaDataType, cusparseSDDMMAlg_t,
                                          void *) {
    return valid(handle) ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_NOT_INITIALIZED;
}

cusparseStatus_t cusparseSDDMM(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
                               const void *alpha, cusparseConstDnMatDescr_t matA, cusparseConstDnMatDescr_t matB,
                               const void *beta, cusparseSpMatDescr_t matC, cudaDataType computeType,
                               cusparseSDDMMAlg_t alg, void *externalBuffer) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (matA == nullptr || matB == nullptr || matC == nullptr || alpha == nullptr || beta == nullptr)
        return CUSPARSE_STATUS_INVALID_VALUE;
    if (matC->format != CUSPARSE_FORMAT_CSR || matA->valueType != matC->valueType ||
        matB->valueType != matC->valueType || !computeTypeOk(matC->valueType, computeType))
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    const int64_t m = opA == CUSPARSE_OPERATION_NON_TRANSPOSE ? matA->rows : matA->cols;
    const int64_t k = opA == CUSPARSE_OPERATION_NON_TRANSPOSE ? matA->cols : matA->rows;
    const int64_t kb = opB == CUSPARSE_OPERATION_NON_TRANSPOSE ? matB->rows : matB->cols;
    const int64_t n = opB == CUSPARSE_OPERATION_NON_TRANSPOSE ? matB->cols : matB->rows;
    if (k != kb || matC->rows != m || matC->cols != n) return CUSPARSE_STATUS_INVALID_VALUE;
    const int batches = std::max({matA->batchCount, matB->batchCount, matC->batchCount});
    return withType(matC->valueType, [&](auto tag) {
        using T = typename decltype(tag)::type;
        using C = typename Num<T>::C;
        Csr csr{};
        cusparseStatus_t st = csrOf<T>(handle->stream, matC, CUSPARSE_OPERATION_NON_TRANSPOSE, 0, nullptr, nullptr,
                                       csr);
        if (st != CUSPARSE_STATUS_SUCCESS || matC->nnz == 0) return st;
        sddmm<T><<<dim3(blocksOf(matC->nnz), unsigned(batches)), 256, 0, handle->stream>>>(
                csr, matC->batchCount > 1 ? matC->stride0 : 0, matC->batchCount > 1 ? matC->stride1 : 0,
                dense(matA, opA), dense(matB, opB), k, scalar<C>(handle, alpha), scalar<C>(handle, beta));
        return launched();
    });
}

// ==== SpGEMM

cusparseStatus_t cusparseSpGEMM_createDescr(cusparseSpGEMMDescr_t *descr) {
    if (descr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *descr = new cusparseSpGEMMDescr{};
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpGEMM_destroyDescr(cusparseSpGEMMDescr_t descr) {
    delete descr;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpGEMM_getNumProducts(cusparseSpGEMMDescr_t spgemmDescr, int64_t *num_prods) {
    if (spgemmDescr == nullptr || num_prods == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *num_prods = spgemmDescr->products;
    return CUSPARSE_STATUS_SUCCESS;
}

namespace {

cusparseStatus_t gemmCheck(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
                           cusparseConstSpMatDescr_t a, cusparseConstSpMatDescr_t b, cusparseConstSpMatDescr_t c,
                           cudaDataType computeType, cusparseSpGEMMDescr_t d) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (a == nullptr || b == nullptr || c == nullptr || d == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (opA != CUSPARSE_OPERATION_NON_TRANSPOSE || opB != CUSPARSE_OPERATION_NON_TRANSPOSE ||
        a->format != CUSPARSE_FORMAT_CSR || b->format != CUSPARSE_FORMAT_CSR || c->format != CUSPARSE_FORMAT_CSR ||
        a->valueType != b->valueType || a->valueType != c->valueType || !computeTypeOk(a->valueType, computeType) ||
        a->batchCount > 1 || b->batchCount > 1)
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    if (a->cols != b->rows) return CUSPARSE_STATUS_INVALID_VALUE;
    return CUSPARSE_STATUS_SUCCESS;
}

Csr plainCsr(cusparseConstSpMatDescr_t a) {
    const int64_t base = a->base == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;
    return Csr{a->rows, a->cols, a->nnz, Idx{a->p0, a->type0 == CUSPARSE_INDEX_64I, base},
               Idx{a->p1, a->type1 == CUSPARSE_INDEX_64I, base}, a->values, false};
}

} // namespace

cusparseStatus_t cusparseSpGEMM_workEstimation(cusparseHandle_t handle, cusparseOperation_t opA,
                                               cusparseOperation_t opB, const void *alpha,
                                               cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB,
                                               const void *beta, cusparseSpMatDescr_t matC, cudaDataType computeType,
                                               cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr,
                                               size_t *bufferSize1, void *externalBuffer1) {
    cusparseStatus_t st = gemmCheck(handle, opA, opB, matA, matB, matC, computeType, spgemmDescr);
    if (st != CUSPARSE_STATUS_SUCCESS) return st;
    if (bufferSize1 == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    const cudaStream_t s = handle->stream;
    const Csr a = plainCsr(matA), b = plainCsr(matB);
    const size_t cbytes = computeSize(matA->valueType);
    if (externalBuffer1 == nullptr) {
        // the number of products sizes the buffer
        int64_t *counts = nullptr;
        unsigned long long *total = nullptr, products = 0;
        if (cudaMalloc(&counts, size_t(a.rows + 1) * 8) != cudaSuccess || cudaMalloc(&total, 8) != cudaSuccess) {
            cudaFree(counts);
            return CUSPARSE_STATUS_ALLOC_FAILED;
        }
        cudaMemsetAsync(total, 0, 8, s);
        if (a.rows > 0) productCounts<<<blocksOf(a.rows), 256, 0, s>>>(a, b, counts, total);
        const cudaError_t e = cudaMemcpyAsync(&products, total, 8, cudaMemcpyDeviceToHost, s);
        const cudaError_t sync = cudaStreamSynchronize(s);
        cudaFree(counts);
        cudaFree(total);
        if (e != cudaSuccess || sync != cudaSuccess) return cudaStatus(cudaGetLastError());
        spgemmDescr->products = int64_t(products);
        *bufferSize1 = gemmLayout1(a.rows, spgemmDescr->products, cbytes).layout.total;
        return CUSPARSE_STATUS_SUCCESS;
    }
    // products expanded, sorted by position; the number of distinct positions
    const int64_t p = spgemmDescr->products;
    const GemmLayout1 l = gemmLayout1(a.rows, p, cbytes);
    char *buf = static_cast<char *>(externalBuffer1);
    auto *counts = reinterpret_cast<int64_t *>(buf + l.counts), *starts = reinterpret_cast<int64_t *>(buf + l.starts);
    size_t temp = l.tempBytes;
    if (a.rows > 0) productCounts<<<blocksOf(a.rows), 256, 0, s>>>(a, b, counts, nullptr);
    cudaMemsetAsync(counts + a.rows, 0, 8, s);
    cub::DeviceScan::ExclusiveSum(buf + l.temp, temp, counts, starts, a.rows + 1, s);
    int64_t unique = 0;
    if (p > 0) {
        auto *keysIn = reinterpret_cast<uint64_t *>(buf + l.keysIn), *keysOut = reinterpret_cast<uint64_t *>(buf + l.keysOut);
        auto *permIn = reinterpret_cast<int64_t *>(buf + l.permIn), *permOut = reinterpret_cast<int64_t *>(buf + l.permOut);
        auto *heads = reinterpret_cast<int64_t *>(buf + l.heads), *positions = reinterpret_cast<int64_t *>(buf + l.positions);
        st = withType(matA->valueType, [&](auto tag) {
            using T = typename decltype(tag)::type;
            expandProducts<T><<<blocksOf(a.rows * 32), 256, 0, s>>>(
                    a, b, starts, keysIn, permIn, reinterpret_cast<typename Num<T>::C *>(buf + l.products));
            return launched();
        });
        if (st != CUSPARSE_STATUS_SUCCESS) return st;
        temp = l.tempBytes;
        cub::DeviceRadixSort::SortPairs(buf + l.temp, temp, keysIn, keysOut, permIn, permOut, p, 0,
                                        bitsFor(uint64_t(a.rows) * uint64_t(std::max<int64_t>(b.cols, 1))), s);
        uniqueHeads<<<blocksOf(p), 256, 0, s>>>(keysOut, p, heads);
        temp = l.tempBytes;
        cub::DeviceScan::ExclusiveSum(buf + l.temp, temp, heads, positions, p, s);
        uniqueStarts<<<blocksOf(p), 256, 0, s>>>(positions, heads, p, reinterpret_cast<int64_t *>(buf + l.uniqueStart));
        int64_t lastPosition = 0, lastHead = 0;
        cudaMemcpyAsync(&lastPosition, positions + p - 1, 8, cudaMemcpyDeviceToHost, s);
        cudaMemcpyAsync(&lastHead, heads + p - 1, 8, cudaMemcpyDeviceToHost, s);
        if (cudaStreamSynchronize(s) != cudaSuccess) return cudaStatus(cudaGetLastError());
        unique = lastPosition + lastHead;
    }
    spgemmDescr->nnz = unique;
    spgemmDescr->rows = a.rows;
    spgemmDescr->cols = b.cols;
    spgemmDescr->buffer1 = externalBuffer1;
    spgemmDescr->valueType = matA->valueType;
    return launched();
}

cusparseStatus_t cusparseSpGEMM_compute(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
                                        const void *alpha, cusparseConstSpMatDescr_t matA,
                                        cusparseConstSpMatDescr_t matB, const void *beta, cusparseSpMatDescr_t matC,
                                        cudaDataType computeType, cusparseSpGEMMAlg_t alg,
                                        cusparseSpGEMMDescr_t spgemmDescr, size_t *bufferSize2,
                                        void *externalBuffer2) {
    cusparseStatus_t st = gemmCheck(handle, opA, opB, matA, matB, matC, computeType, spgemmDescr);
    if (st != CUSPARSE_STATUS_SUCCESS) return st;
    if (bufferSize2 == nullptr || spgemmDescr->buffer1 == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    const size_t cbytes = computeSize(matA->valueType);
    const GemmLayout2 l2 = gemmLayout2(spgemmDescr->nnz, cbytes);
    if (externalBuffer2 == nullptr) {
        *bufferSize2 = l2.layout.total;
        return CUSPARSE_STATUS_SUCCESS;
    }
    const cudaStream_t s = handle->stream;
    const GemmLayout1 l1 = gemmLayout1(spgemmDescr->rows, spgemmDescr->products, cbytes);
    char *b1 = static_cast<char *>(spgemmDescr->buffer1), *b2 = static_cast<char *>(externalBuffer2);
    if (spgemmDescr->nnz > 0) {
        st = withType(matA->valueType, [&](auto tag) {
            using T = typename decltype(tag)::type;
            using C = typename Num<T>::C;
            reduceProducts<T><<<blocksOf(spgemmDescr->nnz), 256, 0, s>>>(
                    reinterpret_cast<uint64_t *>(b1 + l1.keysOut), reinterpret_cast<int64_t *>(b1 + l1.permOut),
                    reinterpret_cast<C *>(b1 + l1.products), reinterpret_cast<int64_t *>(b1 + l1.uniqueStart),
                    spgemmDescr->products, spgemmDescr->nnz, std::max<int64_t>(spgemmDescr->cols, 1),
                    scalar<C>(handle, alpha), reinterpret_cast<uint64_t *>(b2 + l2.rowKeys),
                    reinterpret_cast<int64_t *>(b2 + l2.columns), reinterpret_cast<C *>(b2 + l2.values));
            return launched();
        });
        if (st != CUSPARSE_STATUS_SUCCESS) return st;
    }
    spgemmDescr->buffer2 = externalBuffer2;
    matC->rows = spgemmDescr->rows;
    matC->cols = spgemmDescr->cols;
    matC->nnz = spgemmDescr->nnz;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpGEMM_copy(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
                                     const void *alpha, cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB,
                                     const void *beta, cusparseSpMatDescr_t matC, cudaDataType computeType,
                                     cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr) {
    cusparseStatus_t st = gemmCheck(handle, opA, opB, matA, matB, matC, computeType, spgemmDescr);
    if (st != CUSPARSE_STATUS_SUCCESS) return st;
    if (spgemmDescr->buffer2 == nullptr || matC->p0 == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    const cudaStream_t s = handle->stream;
    const int64_t nnz = spgemmDescr->nnz, rows = spgemmDescr->rows;
    const GemmLayout2 l2 = gemmLayout2(nnz, computeSize(matA->valueType));
    char *b2 = static_cast<char *>(spgemmDescr->buffer2);
    const int64_t base = matC->base == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;
    offsetsOfSorted<<<blocksOf(rows + 1), 256, 0, s>>>(reinterpret_cast<uint64_t *>(b2 + l2.rowKeys), nnz, rows,
                                                       matC->p0, matC->type0 == CUSPARSE_INDEX_64I, base);
    if (nnz == 0) return launched();
    return withType(matA->valueType, [&](auto tag) {
        using T = typename decltype(tag)::type;
        copyGemmResult<T><<<blocksOf(nnz), 256, 0, s>>>(
                reinterpret_cast<int64_t *>(b2 + l2.columns), reinterpret_cast<typename Num<T>::C *>(b2 + l2.values),
                nnz, matC->p1, matC->type1 == CUSPARSE_INDEX_64I, base, static_cast<T *>(matC->values));
        return launched();
    });
}

// ==== SpSV, SpSM

cusparseStatus_t cusparseSpSV_createDescr(cusparseSpSVDescr_t *descr) {
    if (descr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *descr = new cusparseSpSVDescr{};
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpSV_destroyDescr(cusparseSpSVDescr_t descr) {
    delete descr;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpSV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, const void *alpha,
                                         cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX,
                                         cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg,
                                         cusparseSpSVDescr_t spsvDescr, size_t *bufferSize) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (matA == nullptr || bufferSize == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *bufferSize = solveLayout(matA, opA, 1).layout.total;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpSV_analysis(cusparseHandle_t handle, cusparseOperation_t opA, const void *alpha,
                                       cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX,
                                       cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg,
                                       cusparseSpSVDescr_t spsvDescr, void *externalBuffer) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (matA == nullptr || spsvDescr == nullptr || externalBuffer == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (!supportedFormat(matA) || matA->rows != matA->cols || !computeTypeOk(matA->valueType, computeType))
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    spsvDescr->buffer = externalBuffer;
    spsvDescr->values = nullptr;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpSV_updateMatrix(cusparseHandle_t handle, cusparseSpSVDescr_t spsvDescr, void *newValues,
                                           cusparseSpSVUpdate_t updatePart) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (spsvDescr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (updatePart != CUSPARSE_SPSV_UPDATE_GENERAL) return CUSPARSE_STATUS_NOT_SUPPORTED;
    spsvDescr->values = newValues;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpSV_solve(cusparseHandle_t handle, cusparseOperation_t opA, const void *alpha,
                                    cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX,
                                    cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg,
                                    cusparseSpSVDescr_t spsvDescr) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (matA == nullptr || vecX == nullptr || vecY == nullptr || spsvDescr == nullptr || alpha == nullptr)
        return CUSPARSE_STATUS_INVALID_VALUE;
    if (!supportedFormat(matA) || vecX->valueType != matA->valueType || vecY->valueType != matA->valueType)
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    const Dense x{vecX->values, vecX->size, 0, false, CUSPARSE_OPERATION_NON_TRANSPOSE};
    const Dense y{vecY->values, vecY->size, 0, false, CUSPARSE_OPERATION_NON_TRANSPOSE};
    return withType(matA->valueType, [&](auto tag) {
        using T = typename decltype(tag)::type;
        return solveAll<T>(handle, matA, opA, spsvDescr->values, x, y, 1, alpha, spsvDescr->buffer, true, nullptr, 0);
    });
}

cusparseStatus_t cusparseSpSM_createDescr(cusparseSpSMDescr_t *descr) {
    if (descr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *descr = new cusparseSpSMDescr{};
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpSM_destroyDescr(cusparseSpSMDescr_t descr) {
    delete descr;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpSM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
                                         const void *alpha, cusparseConstSpMatDescr_t matA,
                                         cusparseConstDnMatDescr_t matB, cusparseDnMatDescr_t matC,
                                         cudaDataType computeType, cusparseSpSMAlg_t alg, cusparseSpSMDescr_t spsmDescr,
                                         size_t *bufferSize) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (matA == nullptr || matC == nullptr || bufferSize == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *bufferSize = solveLayout(matA, opA, matC->cols).layout.total;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpSM_analysis(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
                                       const void *alpha, cusparseConstSpMatDescr_t matA,
                                       cusparseConstDnMatDescr_t matB, cusparseDnMatDescr_t matC,
                                       cudaDataType computeType, cusparseSpSMAlg_t alg, cusparseSpSMDescr_t spsmDescr,
                                       void *externalBuffer) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (matA == nullptr || spsmDescr == nullptr || externalBuffer == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (!supportedFormat(matA) || matA->rows != matA->cols || !computeTypeOk(matA->valueType, computeType))
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    spsmDescr->buffer = externalBuffer;
    spsmDescr->values = nullptr;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpSM_updateMatrix(cusparseHandle_t handle, cusparseSpSMDescr_t spsmDescr, void *newValues,
                                           cusparseSpSMUpdate_t updatePart) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (spsmDescr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (updatePart != CUSPARSE_SPSM_UPDATE_GENERAL) return CUSPARSE_STATUS_NOT_SUPPORTED;
    spsmDescr->values = newValues;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpSM_solve(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
                                    const void *alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB,
                                    cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpSMAlg_t alg,
                                    cusparseSpSMDescr_t spsmDescr) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (matA == nullptr || matB == nullptr || matC == nullptr || spsmDescr == nullptr || alpha == nullptr)
        return CUSPARSE_STATUS_INVALID_VALUE;
    if (!supportedFormat(matA) || matB->valueType != matA->valueType || matC->valueType != matA->valueType)
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    const int64_t n = matC->cols;
    const int64_t bRows = opB == CUSPARSE_OPERATION_NON_TRANSPOSE ? matB->rows : matB->cols;
    const int64_t bCols = opB == CUSPARSE_OPERATION_NON_TRANSPOSE ? matB->cols : matB->rows;
    if (bRows != matA->rows || bCols != n || matC->rows != matA->rows) return CUSPARSE_STATUS_INVALID_VALUE;
    return withType(matA->valueType, [&](auto tag) {
        using T = typename decltype(tag)::type;
        return solveAll<T>(handle, matA, opA, spsmDescr->values, dense(matB, opB), dense(matC), n, alpha,
                           spsmDescr->buffer, false, nullptr, dense(matC).batchStride);
    });
}

// ==== conversions

cusparseStatus_t cusparseSparseToDense_bufferSize(cusparseHandle_t handle, cusparseConstSpMatDescr_t matA,
                                                  cusparseDnMatDescr_t, cusparseSparseToDenseAlg_t,
                                                  size_t *bufferSize) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (matA == nullptr || bufferSize == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *bufferSize = std::max<size_t>(matA->format == CUSPARSE_FORMAT_COO
                                           ? conversionBytes(matA, CUSPARSE_OPERATION_NON_TRANSPOSE)
                                           : 0,
                                   4);
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSparseToDense(cusparseHandle_t handle, cusparseConstSpMatDescr_t matA,
                                       cusparseDnMatDescr_t matB, cusparseSparseToDenseAlg_t alg,
                                       void *externalBuffer) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (matA == nullptr || matB == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (!supportedFormat(matA) || matA->valueType != matB->valueType || matA->batchCount > 1)
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    if (matB->rows != matA->rows || matB->cols != matA->cols) return CUSPARSE_STATUS_INVALID_VALUE;
    return withType(matA->valueType, [&](auto tag) {
        using T = typename decltype(tag)::type;
        const cudaStream_t s = handle->stream;
        const Dense out = dense(matB);
        if (matA->rows * matA->cols > 0) {
            zeroDense<T><<<blocksOf(matA->rows * matA->cols), 256, 0, s>>>(out, matA->rows, matA->cols);
        }
        // CSC: its columns are the rows of the transpose
        const bool csc = matA->format == CUSPARSE_FORMAT_CSC;
        Csr csr{};
        cusparseStatus_t st = csrOf<T>(s, matA, csc ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       0, externalBuffer, nullptr, csr);
        if (st != CUSPARSE_STATUS_SUCCESS || csr.rows == 0) return st;
        scatterToDense<T><<<blocksOf(csr.rows * 32), 256, 0, s>>>(csr, csc, out);
        return launched();
    });
}

namespace {

struct DenseToSparseLayout {
    size_t counts, starts, temp, tempBytes;
    Layout layout;
};

DenseToSparseLayout denseToSparseLayout(int64_t majors) {
    DenseToSparseLayout l{};
    cub::DeviceScan::ExclusiveSum(nullptr, l.tempBytes, static_cast<const int64_t *>(nullptr),
                                  static_cast<int64_t *>(nullptr), majors + 1);
    l.counts = l.layout.add(size_t(majors + 1) * 8);
    l.starts = l.layout.add(size_t(majors + 1) * 8);
    l.temp = l.layout.add(l.tempBytes);
    return l;
}

int64_t majorsOf(cusparseConstSpMatDescr_t b) {
    return b->format == CUSPARSE_FORMAT_CSC ? b->cols : b->rows;
}

} // namespace

cusparseStatus_t cusparseDenseToSparse_bufferSize(cusparseHandle_t handle, cusparseConstDnMatDescr_t matA,
                                                  cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t,
                                                  size_t *bufferSize) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (matB == nullptr || bufferSize == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *bufferSize = denseToSparseLayout(majorsOf(matB)).layout.total;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDenseToSparse_analysis(cusparseHandle_t handle, cusparseConstDnMatDescr_t matA,
                                                cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t alg,
                                                void *externalBuffer) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (matA == nullptr || matB == nullptr || externalBuffer == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (!supportedFormat(matB) || matA->valueType != matB->valueType) return CUSPARSE_STATUS_NOT_SUPPORTED;
    if (matA->rows != matB->rows || matA->cols != matB->cols) return CUSPARSE_STATUS_INVALID_VALUE;
    const cudaStream_t s = handle->stream;
    const int64_t majors = majorsOf(matB), minors = matB->format == CUSPARSE_FORMAT_CSC ? matB->rows : matB->cols;
    const DenseToSparseLayout l = denseToSparseLayout(majors);
    char *buf = static_cast<char *>(externalBuffer);
    auto *counts = reinterpret_cast<int64_t *>(buf + l.counts), *starts = reinterpret_cast<int64_t *>(buf + l.starts);
    cusparseStatus_t st = withType(matA->valueType, [&](auto tag) {
        using T = typename decltype(tag)::type;
        cudaMemsetAsync(counts, 0, size_t(majors + 1) * 8, s);
        if (majors > 0) {
            denseCounts<T><<<blocksOf(majors), 256, 0, s>>>(dense(matA), majors, minors,
                                                            matB->format != CUSPARSE_FORMAT_CSC, counts);
        }
        return launched();
    });
    if (st != CUSPARSE_STATUS_SUCCESS) return st;
    size_t temp = l.tempBytes;
    cub::DeviceScan::ExclusiveSum(buf + l.temp, temp, counts, starts, majors + 1, s);
    int64_t nnz = 0;
    cudaMemcpyAsync(&nnz, starts + majors, 8, cudaMemcpyDeviceToHost, s);
    if (cudaStreamSynchronize(s) != cudaSuccess) return cudaStatus(cudaGetLastError());
    matB->nnz = nnz;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDenseToSparse_convert(cusparseHandle_t handle, cusparseConstDnMatDescr_t matA,
                                               cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t alg,
                                               void *externalBuffer) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (matA == nullptr || matB == nullptr || externalBuffer == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (!supportedFormat(matB) || matA->valueType != matB->valueType) return CUSPARSE_STATUS_NOT_SUPPORTED;
    const int64_t majors = majorsOf(matB), minors = matB->format == CUSPARSE_FORMAT_CSC ? matB->rows : matB->cols;
    const DenseToSparseLayout l = denseToSparseLayout(majors);
    const int64_t base = matB->base == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;
    const bool coo = matB->format == CUSPARSE_FORMAT_COO;
    if (majors == 0) return CUSPARSE_STATUS_SUCCESS;
    return withType(matA->valueType, [&](auto tag) {
        using T = typename decltype(tag)::type;
        denseFill<T><<<blocksOf(majors), 256, 0, handle->stream>>>(
                dense(matA), majors, minors, matB->format != CUSPARSE_FORMAT_CSC,
                reinterpret_cast<int64_t *>(static_cast<char *>(externalBuffer) + l.starts), base, matB->p0,
                matB->type0 == CUSPARSE_INDEX_64I, matB->p1, matB->type1 == CUSPARSE_INDEX_64I, coo,
                static_cast<T *>(matB->values));
        return launched();
    });
}

cusparseStatus_t cusparseCsr2cscEx2_bufferSize(cusparseHandle_t handle, int m, int n, int nnz, const void *,
                                               const int *, const int *, void *, int *, int *, cudaDataType valType,
                                               cusparseAction_t, cusparseIndexBase_t, cusparseCsr2CscAlg_t,
                                               size_t *bufferSize) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (bufferSize == nullptr || m < 0 || n < 0 || nnz < 0) return CUSPARSE_STATUS_INVALID_VALUE;
    *bufferSize = conversionLayout(nnz, n, std::max<size_t>(typeSize(valType), 1)).layout.total;
    return CUSPARSE_STATUS_SUCCESS;
}

namespace {

__global__ void storeIndices(const int64_t *in, int64_t n, int *out, int base) {
    GRID_STRIDE(i, n) out[i] = int(in[i]) + base;
}

} // namespace

cusparseStatus_t cusparseCsr2cscEx2(cusparseHandle_t handle, int m, int n, int nnz, const void *csrVal,
                                    const int *csrRowPtr, const int *csrColInd, void *cscVal, int *cscColPtr,
                                    int *cscRowInd, cudaDataType valType, cusparseAction_t copyValues,
                                    cusparseIndexBase_t idxBase, cusparseCsr2CscAlg_t alg, void *buffer) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (buffer == nullptr || m < 0 || n < 0 || nnz < 0) return CUSPARSE_STATUS_INVALID_VALUE;
    const cudaStream_t s = handle->stream;
    const bool numeric = copyValues == CUSPARSE_ACTION_NUMERIC;
    // the CSR of the transpose, in the buffer, then copied out with 32 bit indices
    cusparseSpMatDescr a{CUSPARSE_FORMAT_CSR, m, n, nnz, const_cast<int *>(csrRowPtr), const_cast<int *>(csrColInd),
                         const_cast<void *>(csrVal), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, idxBase, valType, 1, 0, 0,
                         CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT};
    return withType(valType, [&](auto tag) {
        using T = typename decltype(tag)::type;
        if (!numeric) a.values = nullptr;
        Csr csr{};
        cusparseStatus_t st = csrOf<T>(s, &a, CUSPARSE_OPERATION_TRANSPOSE, 0, buffer, nullptr, csr);
        if (st != CUSPARSE_STATUS_SUCCESS) return st;
        const Conversion c = conversionLayout(nnz, n, sizeof(T));
        char *buf = static_cast<char *>(buffer);
        const int base = idxBase == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;
        storeIndices<<<blocksOf(n + 1), 256, 0, s>>>(reinterpret_cast<int64_t *>(buf + c.offsets), n + 1, cscColPtr,
                                                     base);
        if (nnz > 0) {
            storeIndices<<<blocksOf(nnz), 256, 0, s>>>(reinterpret_cast<int64_t *>(buf + c.columns), nnz, cscRowInd,
                                                       base);
            if (numeric) cudaMemcpyAsync(cscVal, buf + c.values, size_t(nnz) * sizeof(T), cudaMemcpyDeviceToDevice, s);
        }
        return launched();
    });
}

// ==== sparse vectors

cusparseStatus_t cusparseAxpby(cusparseHandle_t handle, const void *alpha, cusparseConstSpVecDescr_t vecX,
                               const void *beta, cusparseDnVecDescr_t vecY) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (vecX == nullptr || vecY == nullptr || alpha == nullptr || beta == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (vecX->valueType != vecY->valueType) return CUSPARSE_STATUS_NOT_SUPPORTED;
    return withType(vecX->valueType, [&](auto tag) {
        using T = typename decltype(tag)::type;
        using C = typename Num<T>::C;
        const cudaStream_t s = handle->stream;
        const Idx idx{vecX->indices, vecX->idxType == CUSPARSE_INDEX_64I, vecX->base == CUSPARSE_INDEX_BASE_ONE};
        if (vecY->size > 0) {
            scaleVector<T><<<blocksOf(vecY->size), 256, 0, s>>>(static_cast<T *>(vecY->values), vecY->size,
                                                                scalar<C>(handle, beta));
        }
        if (vecX->nnz > 0) {
            sparseVectorOp<T><<<blocksOf(vecX->nnz), 256, 0, s>>>(idx, static_cast<T *>(vecX->values),
                                                                  static_cast<T *>(vecY->values), vecX->nnz, 0,
                                                                  scalar<C>(handle, alpha));
        }
        return launched();
    });
}

namespace {

cusparseStatus_t vectorOp(cusparseHandle_t handle, cusparseConstSpVecDescr_t x, cusparseConstDnVecDescr_t y,
                          int mode) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (x == nullptr || y == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (x->valueType != y->valueType) return CUSPARSE_STATUS_NOT_SUPPORTED;
    if (x->nnz == 0) return CUSPARSE_STATUS_SUCCESS;
    return withType(x->valueType, [&](auto tag) {
        using T = typename decltype(tag)::type;
        const Idx idx{x->indices, x->idxType == CUSPARSE_INDEX_64I, x->base == CUSPARSE_INDEX_BASE_ONE};
        sparseVectorOp<T><<<blocksOf(x->nnz), 256, 0, handle->stream>>>(
                idx, static_cast<T *>(x->values), static_cast<T *>(y->values), x->nnz, mode,
                Scalar<typename Num<T>::C>{});
        return launched();
    });
}

} // namespace

cusparseStatus_t cusparseGather(cusparseHandle_t handle, cusparseConstDnVecDescr_t vecY, cusparseSpVecDescr_t vecX) {
    return vectorOp(handle, vecX, vecY, 1);
}

cusparseStatus_t cusparseScatter(cusparseHandle_t handle, cusparseConstSpVecDescr_t vecX, cusparseDnVecDescr_t vecY) {
    return vectorOp(handle, vecX, vecY, 2);
}

cusparseStatus_t cusparseSpVV_bufferSize(cusparseHandle_t handle, cusparseOperation_t, cusparseConstSpVecDescr_t,
                                         cusparseConstDnVecDescr_t, const void *, cudaDataType, size_t *bufferSize) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (bufferSize == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *bufferSize = 16;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpVV(cusparseHandle_t handle, cusparseOperation_t opX, cusparseConstSpVecDescr_t vecX,
                              cusparseConstDnVecDescr_t vecY, void *result, cudaDataType computeType,
                              void *externalBuffer) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (vecX == nullptr || vecY == nullptr || result == nullptr || externalBuffer == nullptr)
        return CUSPARSE_STATUS_INVALID_VALUE;
    if (vecX->valueType != vecY->valueType || !computeTypeOk(vecX->valueType, computeType))
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    return withType(vecX->valueType, [&](auto tag) {
        using T = typename decltype(tag)::type;
        using C = typename Num<T>::C;
        const cudaStream_t s = handle->stream;
        const Idx idx{vecX->indices, vecX->idxType == CUSPARSE_INDEX_64I, vecX->base == CUSPARSE_INDEX_BASE_ONE};
        C *out = handle->pointerMode == CUSPARSE_POINTER_MODE_DEVICE ? static_cast<C *>(result)
                                                                     : static_cast<C *>(externalBuffer);
        spvvKernel<T><<<1, 256, 0, s>>>(idx, static_cast<const T *>(vecX->values),
                                        static_cast<const T *>(vecY->values), vecX->nnz,
                                        opX == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, out);
        if (handle->pointerMode == CUSPARSE_POINTER_MODE_HOST) {
            if (cudaMemcpyAsync(result, out, sizeof(C), cudaMemcpyDeviceToHost, s) != cudaSuccess ||
                cudaStreamSynchronize(s) != cudaSuccess)
                return cudaStatus(cudaGetLastError());
        }
        return launched();
    });
}

// ==== legacy: sorting and conversions

cusparseStatus_t cusparseCreateIdentityPermutation(cusparseHandle_t handle, int n, int *p) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (n < 0 || (n > 0 && p == nullptr)) return CUSPARSE_STATUS_INVALID_VALUE;
    if (n > 0) identity<<<blocksOf(n), 256, 0, handle->stream>>>(n, p);
    return launched();
}

cusparseStatus_t cusparseXcoo2csr(cusparseHandle_t handle, const int *cooRowInd, int nnz, int m, int *csrSortedRowPtr,
                                  cusparseIndexBase_t idxBase) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (nnz < 0 || m < 0 || csrSortedRowPtr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    coo2csr<<<blocksOf(m + 1), 256, 0, handle->stream>>>(cooRowInd, nnz, m, csrSortedRowPtr,
                                                         idxBase == CUSPARSE_INDEX_BASE_ONE ? 1 : 0);
    return launched();
}

cusparseStatus_t cusparseXcoosort_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz, const int *,
                                                const int *, size_t *pBufferSizeInBytes) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (pBufferSizeInBytes == nullptr || nnz < 0) return CUSPARSE_STATUS_INVALID_VALUE;
    *pBufferSizeInBytes = sortLayout(nnz).layout.total;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXcoosortByRow(cusparseHandle_t handle, int m, int n, int nnz, int *cooRowsA, int *cooColsA,
                                       int *P, void *pBuffer) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (nnz < 0 || pBuffer == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (nnz == 0) return CUSPARSE_STATUS_SUCCESS;
    const SortLayout l = sortLayout(nnz);
    char *buf = static_cast<char *>(pBuffer);
    sortKeysCoo<<<blocksOf(nnz), 256, 0, handle->stream>>>(nnz, cooRowsA, cooColsA,
                                                           reinterpret_cast<uint64_t *>(buf + l.keysIn),
                                                           reinterpret_cast<int64_t *>(buf + l.permIn));
    return sortAndPermute(handle->stream, nnz, pBuffer, cooRowsA, cooColsA, P);
}

cusparseStatus_t cusparseXcoosortByColumn(cusparseHandle_t handle, int m, int n, int nnz, int *cooRowsA,
                                          int *cooColsA, int *P, void *pBuffer) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (nnz < 0 || pBuffer == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (nnz == 0) return CUSPARSE_STATUS_SUCCESS;
    const SortLayout l = sortLayout(nnz);
    char *buf = static_cast<char *>(pBuffer);
    sortKeysCoo<<<blocksOf(nnz), 256, 0, handle->stream>>>(nnz, cooColsA, cooRowsA,
                                                           reinterpret_cast<uint64_t *>(buf + l.keysIn),
                                                           reinterpret_cast<int64_t *>(buf + l.permIn));
    return sortAndPermute(handle->stream, nnz, pBuffer, cooRowsA, cooColsA, P);
}

cusparseStatus_t cusparseXcsrsort_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz, const int *,
                                                const int *, size_t *pBufferSizeInBytes) {
    return cusparseXcoosort_bufferSizeExt(handle, m, n, nnz, nullptr, nullptr, pBufferSizeInBytes);
}

cusparseStatus_t cusparseXcsrsort(cusparseHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA,
                                  const int *csrRowPtrA, int *csrColIndA, int *P, void *pBuffer) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (nnz < 0 || m < 0 || pBuffer == nullptr || descrA == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (nnz == 0) return CUSPARSE_STATUS_SUCCESS;
    const SortLayout l = sortLayout(nnz);
    char *buf = static_cast<char *>(pBuffer);
    sortKeysCsr<<<blocksOf(int64_t(m) * 32), 256, 0, handle->stream>>>(
            m, csrRowPtrA, descrA->indexBase == CUSPARSE_INDEX_BASE_ONE ? 1 : 0, csrColIndA,
            reinterpret_cast<uint64_t *>(buf + l.keysIn), reinterpret_cast<int64_t *>(buf + l.permIn));
    return sortAndPermute(handle->stream, nnz, pBuffer, csrColIndA, nullptr, P);
}

// ==== legacy: csrgeam2

namespace {

struct GeamLayout {
    size_t counts, starts, temp, tempBytes;
    Layout layout;
};

GeamLayout geamLayout(int m) {
    GeamLayout l{};
    cub::DeviceScan::ExclusiveSum(nullptr, l.tempBytes, static_cast<const int64_t *>(nullptr),
                                  static_cast<int64_t *>(nullptr), int64_t(m) + 1);
    l.counts = l.layout.add(size_t(m + 1) * 8);
    l.starts = l.layout.add(size_t(m + 1) * 8);
    l.temp = l.layout.add(l.tempBytes);
    return l;
}

int baseOf(const cusparseMatDescr_t d) {
    return d->indexBase == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;
}

template<typename T>
cusparseStatus_t geam(cusparseHandle_t handle, int m, int n, const T *alpha, const cusparseMatDescr_t descrA,
                      const T *valA, const int *rowA, const int *colA, const T *beta, const cusparseMatDescr_t descrB,
                      const T *valB, const int *rowB, const int *colB, const cusparseMatDescr_t descrC, T *valC,
                      const int *rowC, int *colC) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || descrA == nullptr || descrB == nullptr || descrC == nullptr)
        return CUSPARSE_STATUS_INVALID_VALUE;
    if (m == 0) return CUSPARSE_STATUS_SUCCESS;
    geamFill<T><<<blocksOf(m), 256, 0, handle->stream>>>(m, scalar<T>(handle, alpha), valA, rowA, colA, baseOf(descrA),
                                                         scalar<T>(handle, beta), valB, rowB, colB, baseOf(descrB),
                                                         valC, rowC, colC, baseOf(descrC));
    return launched();
}

template<typename T>
size_t geamBufferBytes(int m) {
    return geamLayout(std::max(m, 0)).layout.total;
}

} // namespace

cusparseStatus_t cusparseXcsrgeam2Nnz(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, int nnzA,
                                      const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                      const cusparseMatDescr_t descrB, int nnzB, const int *csrSortedRowPtrB,
                                      const int *csrSortedColIndB, const cusparseMatDescr_t descrC,
                                      int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *workspace) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (m < 0 || descrA == nullptr || descrB == nullptr || descrC == nullptr || workspace == nullptr)
        return CUSPARSE_STATUS_INVALID_VALUE;
    const cudaStream_t s = handle->stream;
    const GeamLayout l = geamLayout(m);
    char *buf = static_cast<char *>(workspace);
    auto *counts = reinterpret_cast<int64_t *>(buf + l.counts), *starts = reinterpret_cast<int64_t *>(buf + l.starts);
    cudaMemsetAsync(counts, 0, size_t(m + 1) * 8, s);
    if (m > 0) {
        geamCounts<<<blocksOf(m), 256, 0, s>>>(m, csrSortedRowPtrA, csrSortedColIndA, baseOf(descrA), csrSortedRowPtrB,
                                               csrSortedColIndB, baseOf(descrB), counts);
    }
    size_t temp = l.tempBytes;
    cub::DeviceScan::ExclusiveSum(buf + l.temp, temp, counts, starts, int64_t(m) + 1, s);
    int *deviceTotal = handle->pointerMode == CUSPARSE_POINTER_MODE_DEVICE ? nnzTotalDevHostPtr : nullptr;
    geamOffsets<<<blocksOf(m + 1), 256, 0, s>>>(m, starts, baseOf(descrC), csrSortedRowPtrC, deviceTotal);
    if (handle->pointerMode == CUSPARSE_POINTER_MODE_HOST && nnzTotalDevHostPtr != nullptr) {
        int64_t total = 0;
        cudaMemcpyAsync(&total, starts + m, 8, cudaMemcpyDeviceToHost, s);
        if (cudaStreamSynchronize(s) != cudaSuccess) return cudaStatus(cudaGetLastError());
        *nnzTotalDevHostPtr = int(total);
    }
    return launched();
}

#define GEAM(P, T, Cuda)                                                                                               \
cusparseStatus_t cusparse##P##csrgeam2_bufferSizeExt(                                                                  \
        cusparseHandle_t handle, int m, int, const Cuda *, const cusparseMatDescr_t, int, const Cuda *, const int *,   \
        const int *, const Cuda *, const cusparseMatDescr_t, int, const Cuda *, const int *, const int *,             \
        const cusparseMatDescr_t, const Cuda *, const int *, const int *, size_t *pBufferSizeInBytes) {                \
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;                                                        \
    if (pBufferSizeInBytes == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;                                           \
    *pBufferSizeInBytes = geamBufferBytes<T>(m);                                                                       \
    return CUSPARSE_STATUS_SUCCESS;                                                                                    \
}                                                                                                                      \
cusparseStatus_t cusparse##P##csrgeam2(                                                                                \
        cusparseHandle_t handle, int m, int n, const Cuda *alpha, const cusparseMatDescr_t descrA, int,               \
        const Cuda *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const Cuda *beta,        \
        const cusparseMatDescr_t descrB, int, const Cuda *csrSortedValB, const int *csrSortedRowPtrB,                 \
        const int *csrSortedColIndB, const cusparseMatDescr_t descrC, Cuda *csrSortedValC, int *csrSortedRowPtrC,     \
        int *csrSortedColIndC, void *) {                                                                               \
    return geam<T>(handle, m, n, reinterpret_cast<const T *>(alpha), descrA,                                           \
                   reinterpret_cast<const T *>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA,                     \
                   reinterpret_cast<const T *>(beta), descrB, reinterpret_cast<const T *>(csrSortedValB),              \
                   csrSortedRowPtrB, csrSortedColIndB, descrC, reinterpret_cast<T *>(csrSortedValC),                   \
                   csrSortedRowPtrC, csrSortedColIndC);                                                                \
}

GEAM(S, float, float)
GEAM(D, double, double)
GEAM(C, cfloat, cuComplex)
GEAM(Z, cdouble, cuDoubleComplex)

// ==== legacy: BSR products and triangular solves

namespace {

template<typename T>
cusparseStatus_t bsrmvImpl(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb,
                           int nb, const T *alpha, const cusparseMatDescr_t descrA, const T *val, const int *rowPtr,
                           const int *colInd, int blockDim, const T *x, const T *beta, T *y) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (mb < 0 || nb < 0 || blockDim < 1 || descrA == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (transA != CUSPARSE_OPERATION_NON_TRANSPOSE || descrA->type != CUSPARSE_MATRIX_TYPE_GENERAL)
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    if (mb == 0) return CUSPARSE_STATUS_SUCCESS;
    bsrmv<T><<<blocksOf(int64_t(mb) * blockDim), 256, 0, handle->stream>>>(
            mb, blockDim, scalar<T>(handle, alpha), val, rowPtr, colInd, baseOf(descrA),
            dirA == CUSPARSE_DIRECTION_ROW, x, scalar<T>(handle, beta), y);
    return launched();
}

template<typename T>
cusparseStatus_t bsrmmImpl(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA,
                           cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const T *alpha,
                           const cusparseMatDescr_t descrA, const T *val, const int *rowPtr, const int *colInd,
                           int blockSize, const T *b, int ldb, const T *beta, T *c, int ldc) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (mb < 0 || n < 0 || kb < 0 || blockSize < 1 || descrA == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (transA != CUSPARSE_OPERATION_NON_TRANSPOSE || transB == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE ||
        descrA->type != CUSPARSE_MATRIX_TYPE_GENERAL)
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    if (int64_t(mb) * n == 0) return CUSPARSE_STATUS_SUCCESS;
    bsrmm<T><<<blocksOf(int64_t(mb) * blockSize * n), 256, 0, handle->stream>>>(
            mb, n, blockSize, scalar<T>(handle, alpha), val, rowPtr, colInd, baseOf(descrA),
            dirA == CUSPARSE_DIRECTION_ROW, b, ldb, transB == CUSPARSE_OPERATION_TRANSPOSE, scalar<T>(handle, beta), c,
            ldc);
    return launched();
}

template<typename T>
cudaDataType cudaTypeOf() {
    if constexpr (std::is_same_v<T, float>) return CUDA_R_32F;
    if constexpr (std::is_same_v<T, double>) return CUDA_R_64F;
    if constexpr (std::is_same_v<T, cfloat>) return CUDA_C_32F;
    return CUDA_C_64F;
}

// the expanded CSR of a BSR matrix, then the solver's own space
struct BsrSolveLayout {
    size_t offsets, columns, values, solve;
    Layout layout;
};

BsrSolveLayout bsrSolveLayout(int mb, int nnzb, int blockDim, cudaDataType type, cusparseOperation_t op, int n) {
    const size_t valueBytes = typeSize(type);
    BsrSolveLayout l{};
    const int64_t rows = int64_t(mb) * blockDim, nnz = int64_t(nnzb) * blockDim * blockDim;
    l.offsets = l.layout.add(size_t(rows + 1) * 8);
    l.columns = l.layout.add(size_t(std::max<int64_t>(nnz, 1)) * 8);
    l.values = l.layout.add(size_t(std::max<int64_t>(nnz, 1)) * valueBytes);
    cusparseSpMatDescr probe{CUSPARSE_FORMAT_CSR, rows, rows, nnz, nullptr, nullptr, nullptr, CUSPARSE_INDEX_64I,
                             CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, type, 1, 0, 0,
                             CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT};
    l.solve = l.layout.add(solveLayout(&probe, op, n).layout.total);
    return l;
}

// op(A) X = alpha B with A in BSR: expanded to CSR in the buffer, then the CSR solve
template<typename T>
cusparseStatus_t bsrSolve(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb,
                          int nnzb, const T *alpha, const cusparseMatDescr_t descrA, const T *val, const int *rowPtr,
                          const int *colInd, int blockDim, int **pivot, const T *b, int ldb, T *x, int ldx, int n,
                          bool transXY, void *buffer, bool vector) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (mb < 0 || nnzb < 0 || blockDim < 1 || descrA == nullptr || buffer == nullptr)
        return CUSPARSE_STATUS_INVALID_VALUE;
    if (descrA->type != CUSPARSE_MATRIX_TYPE_GENERAL && descrA->type != CUSPARSE_MATRIX_TYPE_TRIANGULAR)
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    const cudaStream_t s = handle->stream;
    const BsrSolveLayout l = bsrSolveLayout(mb, nnzb, blockDim, cudaTypeOf<T>(), transA, n);
    char *buf = static_cast<char *>(buffer);
    const int64_t rows = int64_t(mb) * blockDim;
    bsrToCsr<T><<<blocksOf(rows + 1), 256, 0, s>>>(mb, blockDim, val, rowPtr, colInd, baseOf(descrA),
                                                    dirA == CUSPARSE_DIRECTION_ROW,
                                                    reinterpret_cast<int64_t *>(buf + l.offsets),
                                                    reinterpret_cast<int64_t *>(buf + l.columns),
                                                    reinterpret_cast<T *>(buf + l.values));
    cusparseSpMatDescr a{CUSPARSE_FORMAT_CSR, rows, rows, int64_t(nnzb) * blockDim * blockDim, buf + l.offsets,
                         buf + l.columns, buf + l.values, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                         CUSPARSE_INDEX_BASE_ZERO, cudaTypeOf<T>(), 1, 0, 0, descrA->fillMode, descrA->diagType};
    const Dense bd{const_cast<T *>(b), vector ? rows : ldb, 0, transXY, CUSPARSE_OPERATION_NON_TRANSPOSE};
    const Dense xd{x, vector ? rows : ldx, 0, transXY, CUSPARSE_OPERATION_NON_TRANSPOSE};
    return solveAll<T>(handle, &a, transA, nullptr, bd, xd, n, alpha, buf + l.solve, vector, pivot, 0);
}

cusparseStatus_t zeroPivot(cusparseHandle_t handle, int *devicePivot, int blockDim, int *position) {
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (position == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    int row = 0x7f7f7f7f;  // (no solve yet: no zero pivot)
    if (devicePivot != nullptr &&
        (cudaMemcpyAsync(&row, devicePivot, sizeof(int), cudaMemcpyDeviceToHost, handle->stream) != cudaSuccess ||
         cudaStreamSynchronize(handle->stream) != cudaSuccess))
        return cudaStatus(cudaGetLastError());
    const bool none = row == 0x7f7f7f7f;
    const int block = none ? -1 : row / std::max(blockDim, 1);
    if (handle->pointerMode == CUSPARSE_POINTER_MODE_DEVICE) {
        cudaMemcpyAsync(position, &block, sizeof(int), cudaMemcpyHostToDevice, handle->stream);
        cudaStreamSynchronize(handle->stream);
    } else {
        *position = block;
    }
    return none ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_ZERO_PIVOT;
}

} // namespace

cusparseStatus_t cusparseCreateBsrsv2Info(bsrsv2Info_t *info) {
    if (info == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *info = new bsrsv2Info{};
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroyBsrsv2Info(bsrsv2Info_t info) {
    delete info;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateBsrsm2Info(bsrsm2Info_t *info) {
    if (info == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *info = new bsrsm2Info{};
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroyBsrsm2Info(bsrsm2Info_t info) {
    delete info;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXbsrsv2_zeroPivot(cusparseHandle_t handle, bsrsv2Info_t info, int *position) {
    if (info == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    return zeroPivot(handle, info->pivot, info->blockDim, position);
}

cusparseStatus_t cusparseXbsrsm2_zeroPivot(cusparseHandle_t handle, bsrsm2Info_t info, int *position) {
    if (info == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    return zeroPivot(handle, info->pivot, info->blockDim, position);
}

#define BSR(P, T, Cuda)                                                                                                \
cusparseStatus_t cusparse##P##bsrmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA,     \
                                    int mb, int nb, int nnzb, const Cuda *alpha, const cusparseMatDescr_t descrA,      \
                                    const Cuda *val, const int *rowPtr, const int *colInd, int blockDim,               \
                                    const Cuda *x, const Cuda *beta, Cuda *y) {                                        \
    return bsrmvImpl<T>(handle, dirA, transA, mb, nb, reinterpret_cast<const T *>(alpha), descrA,                      \
                        reinterpret_cast<const T *>(val), rowPtr, colInd, blockDim, reinterpret_cast<const T *>(x),    \
                        reinterpret_cast<const T *>(beta), reinterpret_cast<T *>(y));                                  \
}                                                                                                                      \
cusparseStatus_t cusparse##P##bsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA,     \
                                    cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const Cuda *alpha,   \
                                    const cusparseMatDescr_t descrA, const Cuda *val, const int *rowPtr,               \
                                    const int *colInd, const int blockSize, const Cuda *B, const int ldb,              \
                                    const Cuda *beta, Cuda *C, int ldc) {                                              \
    return bsrmmImpl<T>(handle, dirA, transA, transB, mb, n, kb, nnzb, reinterpret_cast<const T *>(alpha), descrA,     \
                        reinterpret_cast<const T *>(val), rowPtr, colInd, blockSize, reinterpret_cast<const T *>(B),   \
                        ldb, reinterpret_cast<const T *>(beta), reinterpret_cast<T *>(C), ldc);                        \
}                                                                                                                      \
cusparseStatus_t cusparse##P##bsrsv2_bufferSize(cusparseHandle_t handle, cusparseDirection_t, cusparseOperation_t     \
                                                transA, int mb, int nnzb, const cusparseMatDescr_t, Cuda *,            \
                                                const int *, const int *, int blockDim, bsrsv2Info_t,                  \
                                                int *pBufferSizeInBytes) {                                             \
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;                                                        \
    if (pBufferSizeInBytes == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;                                           \
    *pBufferSizeInBytes = int(bsrSolveLayout(mb, nnzb, blockDim, cudaTypeOf<T>(), transA, 1).layout.total);                  \
    return CUSPARSE_STATUS_SUCCESS;                                                                                    \
}                                                                                                                      \
cusparseStatus_t cusparse##P##bsrsv2_analysis(cusparseHandle_t handle, cusparseDirection_t, cusparseOperation_t, int,  \
                                              int, const cusparseMatDescr_t descrA, const Cuda *, const int *,         \
                                              const int *, int, bsrsv2Info_t info, cusparseSolvePolicy_t, void *) {    \
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;                                                        \
    return info != nullptr && descrA != nullptr ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_INVALID_VALUE;            \
}                                                                                                                      \
cusparseStatus_t cusparse##P##bsrsv2_solve(cusparseHandle_t handle, cusparseDirection_t dirA,                          \
                                           cusparseOperation_t transA, int mb, int nnzb, const Cuda *alpha,            \
                                           const cusparseMatDescr_t descrA, const Cuda *val, const int *rowPtr,        \
                                           const int *colInd, int blockDim, bsrsv2Info_t info, const Cuda *f,          \
                                           Cuda *x, cusparseSolvePolicy_t, void *pBuffer) {                            \
    if (info == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;                                                         \
    info->blockDim = blockDim;                                                                                         \
    return bsrSolve<T>(handle, dirA, transA, mb, nnzb, reinterpret_cast<const T *>(alpha), descrA,                     \
                       reinterpret_cast<const T *>(val), rowPtr, colInd, blockDim, &info->pivot,                       \
                       reinterpret_cast<const T *>(f), 0, reinterpret_cast<T *>(x), 0, 1, false, pBuffer, true);       \
}                                                                                                                      \
cusparseStatus_t cusparse##P##bsrsm2_bufferSize(cusparseHandle_t handle, cusparseDirection_t, cusparseOperation_t     \
                                                transA, cusparseOperation_t, int mb, int n, int nnzb,                  \
                                                const cusparseMatDescr_t, Cuda *, const int *, const int *,            \
                                                int blockSize, bsrsm2Info_t, int *pBufferSizeInBytes) {                \
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;                                                        \
    if (pBufferSizeInBytes == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;                                           \
    *pBufferSizeInBytes = int(bsrSolveLayout(mb, nnzb, blockSize, cudaTypeOf<T>(), transA, n).layout.total);                 \
    return CUSPARSE_STATUS_SUCCESS;                                                                                    \
}                                                                                                                      \
cusparseStatus_t cusparse##P##bsrsm2_analysis(cusparseHandle_t handle, cusparseDirection_t, cusparseOperation_t,      \
                                              cusparseOperation_t, int, int, int, const cusparseMatDescr_t descrA,     \
                                              const Cuda *, const int *, const int *, int, bsrsm2Info_t info,          \
                                              cusparseSolvePolicy_t, void *) {                                         \
    if (!valid(handle)) return CUSPARSE_STATUS_NOT_INITIALIZED;                                                        \
    return info != nullptr && descrA != nullptr ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_INVALID_VALUE;            \
}                                                                                                                      \
cusparseStatus_t cusparse##P##bsrsm2_solve(cusparseHandle_t handle, cusparseDirection_t dirA,                          \
                                           cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n,     \
                                           int nnzb, const Cuda *alpha, const cusparseMatDescr_t descrA,               \
                                           const Cuda *val, const int *rowPtr, const int *colInd, int blockSize,       \
                                           bsrsm2Info_t info, const Cuda *B, int ldb, Cuda *X, int ldx,                \
                                           cusparseSolvePolicy_t, void *pBuffer) {                                     \
    if (info == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;                                                         \
    if (transXY == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE) return CUSPARSE_STATUS_NOT_SUPPORTED;                       \
    info->blockDim = blockSize;                                                                                        \
    return bsrSolve<T>(handle, dirA, transA, mb, nnzb, reinterpret_cast<const T *>(alpha), descrA,                     \
                       reinterpret_cast<const T *>(val), rowPtr, colInd, blockSize, &info->pivot,                      \
                       reinterpret_cast<const T *>(B), ldb, reinterpret_cast<T *>(X), ldx, n,                          \
                       transXY == CUSPARSE_OPERATION_TRANSPOSE, pBuffer, false);                                       \
}

BSR(S, float, float)
BSR(D, double, double)
BSR(C, cfloat, cuComplex)
BSR(Z, cdouble, cuDoubleComplex)
