extern "C" __global__ void fill(float *dst, float value, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = value + float(i);
    }
}
