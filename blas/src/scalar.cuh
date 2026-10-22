/**
 * Arithmetic on the four BLAS / LAPACK element types (float, double, cuComplex, cuDoubleComplex) for the generic
 * kernels of cuBLAS and cuSOLVER.
 */
#pragma once

#include <cuComplex.h>

#include <cmath>
#include <type_traits>

namespace librecuda_blas {

template<typename T>
struct RealOf {
    using type = T;
};
template<>
struct RealOf<cuComplex> {
    using type = float;
};
template<>
struct RealOf<cuDoubleComplex> {
    using type = double;
};
template<typename T>
using Real = typename RealOf<T>::type;

template<typename T>
constexpr bool IS_COMPLEX = std::is_same_v<T, cuComplex> || std::is_same_v<T, cuDoubleComplex>;

template<typename T>
__host__ __device__ inline T make(Real<T> re, Real<T> im = 0) {
    if constexpr (std::is_same_v<T, cuComplex>) {
        return make_cuComplex(re, im);
    } else if constexpr (std::is_same_v<T, cuDoubleComplex>) {
        return make_cuDoubleComplex(re, im);
    } else {
        return re;
    }
}

template<typename T>
__host__ __device__ inline Real<T> re(T x) {
    if constexpr (IS_COMPLEX<T>) {
        return x.x;
    } else {
        return x;
    }
}

template<typename T>
__host__ __device__ inline Real<T> im(T x) {
    if constexpr (IS_COMPLEX<T>) {
        return x.y;
    } else {
        return 0;
    }
}

template<typename T>
__host__ __device__ inline T add(T a, T b) {
    return make<T>(re(a) + re(b), im(a) + im(b));
}

template<typename T>
__host__ __device__ inline T sub(T a, T b) {
    return make<T>(re(a) - re(b), im(a) - im(b));
}

template<typename T>
__host__ __device__ inline T mul(T a, T b) {
    if constexpr (IS_COMPLEX<T>) {
        return make<T>(re(a) * re(b) - im(a) * im(b), re(a) * im(b) + im(a) * re(b));
    } else {
        return a * b;
    }
}

template<typename T>
__host__ __device__ inline T neg(T a) {
    return make<T>(-re(a), -im(a));
}

template<typename T>
__host__ __device__ inline T conjugate(T a) {
    return make<T>(re(a), -im(a));
}

// |a|^2
template<typename T>
__host__ __device__ inline Real<T> norm2(T a) {
    return re(a) * re(a) + im(a) * im(a);
}

template<typename T>
__host__ __device__ inline Real<T> absval(T a) {
    if constexpr (IS_COMPLEX<T>) {
        return hypot(re(a), im(a));
    } else {
        return a < 0 ? -a : a;
    }
}

template<typename T>
__host__ __device__ inline T scale(T a, Real<T> s) {
    return make<T>(re(a) * s, im(a) * s);
}

template<typename T>
__host__ __device__ inline T div(T a, T b) {
    if constexpr (IS_COMPLEX<T>) {
        // (Smith's algorithm: no overflow for large b)
        const Real<T> br = re(b), bi = im(b);
        if (fabs(br) >= fabs(bi)) {
            const Real<T> r = bi / br, d = br + bi * r;
            return make<T>((re(a) + im(a) * r) / d, (im(a) - re(a) * r) / d);
        }
        const Real<T> r = br / bi, d = bi + br * r;
        return make<T>((re(a) * r + im(a)) / d, (im(a) * r - re(a)) / d);
    } else {
        return a / b;
    }
}

template<typename T>
__host__ __device__ inline bool isZero(T a) {
    return re(a) == 0 && im(a) == 0;
}

// a * conj(b) when conj, else a * b
template<typename T>
__host__ __device__ inline T mulConj(T a, T b, bool conj) {
    return mul(a, conj ? conjugate(b) : b);
}

} // namespace librecuda_blas
