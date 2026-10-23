// Example kernels for the open compiler flows (build.sh): compiled to SM_120 cubins without NVIDIA's compiler,
// run on LibreCUDA and checked against numpy by run_cubin.py.

extern "C" __global__ void vector_add(unsigned *out, const unsigned *a, const unsigned *b, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

extern "C" __global__ void saxpy(float *y, const float *x, float a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

// one partial sum per block of 256 threads: a tree in shared memory
extern "C" __global__ void reduce_sum(int *partial, const int *x, int n) {
    __shared__ int sums[256];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    sums[threadIdx.x] = i < n ? x[i] : 0;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = sums[0];
}

// 3 point smoothing; the end points are copied
extern "C" __global__ void stencil(float *out, const float *x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (i == 0 || i == n - 1) {
        out[i] = x[i];
    } else {
        out[i] = 0.25f * x[i - 1] + 0.5f * x[i] + 0.25f * x[i + 1];
    }
}
