// Surface objects on CUDA arrays: writes and reads from kernels (surf1D / 2D / 3D / 2DLayered), checked through copies
// of the array, surface reads and a texture on the same array; 8, 16 and 32 bit texels; cubemap faces; out of range
// accesses in zero mode.
#include "check.h"
#include <vector>

__global__ void write2D(cudaSurfaceObject_t s, int width, int height) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width * height; i += gridDim.x * blockDim.x) {
        const int x = i % width, y = i / width;
        surf2Dwrite(x * 2.0f + y * 1000.0f, s, x * sizeof(float), y);
    }
}

// reads with surf2Dread, adds one and writes back; then out of range accesses in zero mode (ignored / read as 0)
__global__ void readModifyWrite2D(cudaSurfaceObject_t s, int width, int height, float *out) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width * height; i += gridDim.x * blockDim.x) {
        const int x = i % width, y = i / width;
        const float v = surf2Dread<float>(s, x * sizeof(float), y);
        out[i] = v;
        surf2Dwrite(v + 1.0f, s, x * sizeof(float), y);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        surf2Dwrite(-1.0f, s, (width + 3) * sizeof(float), height + 2, cudaBoundaryModeZero);
        out[width * height] = surf2Dread<float>(s, (width + 3) * sizeof(float), 0, cudaBoundaryModeZero);
    }
}

__global__ void readTexture(cudaTextureObject_t t, int width, int height, float *out) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width * height; i += gridDim.x * blockDim.x) {
        out[i] = tex2D<float>(t, i % width + 0.5f, i / width + 0.5f);
    }
}

template<typename T>
__global__ void writeSmall(cudaSurfaceObject_t s, int width, int height) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width * height; i += gridDim.x * blockDim.x) {
        surf2Dwrite(static_cast<T>(i * 7 + 1), s, (i % width) * sizeof(T), i / width);
    }
}

__global__ void writeColors(cudaSurfaceObject_t s, int width, int height) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width * height; i += gridDim.x * blockDim.x) {
        surf2Dwrite(make_uchar4(i, i >> 8, 255 - i, 9), s, (i % width) * sizeof(uchar4), i / width);
    }
}

__global__ void write3D(cudaSurfaceObject_t s, int width, int height, int depth) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width * height * depth; i += gridDim.x * blockDim.x) {
        const int x = i % width, y = i / width % height, z = i / (width * height);
        surf3Dwrite(i * 0.5f, s, x * sizeof(float), y, z);
    }
}

__global__ void writeLayered(cudaSurfaceObject_t s, int width, int height, int layers) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width * height * layers; i += gridDim.x * blockDim.x) {
        const int x = i % width, y = i / width % height, layer = i / (width * height);
        surf2DLayeredwrite(i * 3.0f, s, x * sizeof(float), y, layer);
    }
}

__global__ void write1D(cudaSurfaceObject_t s, int width) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width; i += gridDim.x * blockDim.x) {
        surf1Dwrite(i - 50.0f, s, i * sizeof(float));
    }
}

__global__ void writeCubemap(cudaSurfaceObject_t s, int width) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width * width * 6; i += gridDim.x * blockDim.x) {
        const int x = i % width, y = i / width % width, face = i / (width * width);
        surfCubemapwrite(i + 0.25f, s, x * sizeof(float), y, face);
    }
}

static cudaSurfaceObject_t surfaceOf(cudaArray_t array) {
    cudaResourceDesc res{};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = array;
    cudaSurfaceObject_t surface = 0;
    CHECK(cudaCreateSurfaceObject(&surface, &res));
    return surface;
}

int main() {
    // 2D float: write, read-modify-write, then check by copy, surface read and texture
    {
        const int width = 45, height = 23, n = width * height;
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
        cudaArray_t array;
        CHECK(cudaMallocArray(&array, &desc, width, height, cudaArraySurfaceLoadStore));
        const cudaSurfaceObject_t s = surfaceOf(array);
        float *out;
        CHECK(cudaMalloc(&out, (n + 1) * sizeof(float)));
        write2D<<<8, 128>>>(s, width, height);
        readModifyWrite2D<<<8, 128>>>(s, width, height, out);
        std::vector<float> read(n + 1), copied(n), textured(n);
        CHECK(cudaMemcpy(read.data(), out, (n + 1) * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy2DFromArray(copied.data(), width * sizeof(float), array, 0, 0, width * sizeof(float), height,
                                    cudaMemcpyDeviceToHost));
        cudaResourceDesc res{};
        res.resType = cudaResourceTypeArray;
        res.res.array.array = array;
        cudaTextureDesc tex{};
        cudaTextureObject_t t;
        CHECK(cudaCreateTextureObject(&t, &res, &tex, nullptr));
        readTexture<<<8, 128>>>(t, width, height, out);
        CHECK(cudaMemcpy(textured.data(), out, n * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; i++) {
            const float written = (i % width) * 2.0f + (i / width) * 1000.0f;
            EXPECT(read[i] == written, "surf2Dread [%d] = %g, expected %g", i, read[i], written);
            EXPECT(copied[i] == written + 1.0f, "copied [%d] = %g, expected %g", i, copied[i], written + 1.0f);
            EXPECT(textured[i] == written + 1.0f, "tex2D [%d] = %g, expected %g", i, textured[i], written + 1.0f);
        }
        EXPECT(read[n] == 0.0f, "out of range read in zero mode = %g", read[n]);
        CHECK(cudaDestroyTextureObject(t));
        CHECK(cudaDestroySurfaceObject(s));
        CHECK(cudaFreeArray(array));
        CHECK(cudaFree(out));
    }

    // 8, 16 and 32 bit texels
    {
        const int width = 67, height = 13, n = width * height;
        cudaChannelFormatDesc d8 = cudaCreateChannelDesc<unsigned char>();
        cudaChannelFormatDesc d16 = cudaCreateChannelDesc<unsigned short>();
        cudaChannelFormatDesc d32 = cudaCreateChannelDesc<uchar4>();
        cudaArray_t a8, a16, a32;
        CHECK(cudaMallocArray(&a8, &d8, width, height, cudaArraySurfaceLoadStore));
        CHECK(cudaMallocArray(&a16, &d16, width, height, cudaArraySurfaceLoadStore));
        CHECK(cudaMallocArray(&a32, &d32, width, height, cudaArraySurfaceLoadStore));
        const cudaSurfaceObject_t s8 = surfaceOf(a8), s16 = surfaceOf(a16), s32 = surfaceOf(a32);
        writeSmall<unsigned char><<<8, 128>>>(s8, width, height);
        writeSmall<unsigned short><<<8, 128>>>(s16, width, height);
        writeColors<<<8, 128>>>(s32, width, height);
        std::vector<unsigned char> r8(n);
        std::vector<unsigned short> r16(n);
        std::vector<uchar4> r32(n);
        CHECK(cudaMemcpy2DFromArray(r8.data(), width, a8, 0, 0, width, height, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy2DFromArray(r16.data(), width * 2, a16, 0, 0, width * 2, height, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy2DFromArray(r32.data(), width * 4, a32, 0, 0, width * 4, height, cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; i++) {
            EXPECT(r8[i] == static_cast<unsigned char>(i * 7 + 1), "uchar [%d] = %d", i, r8[i]);
            EXPECT(r16[i] == static_cast<unsigned short>(i * 7 + 1), "ushort [%d] = %d", i, r16[i]);
            const uchar4 c = r32[i];
            EXPECT(c.x == static_cast<unsigned char>(i) && c.y == static_cast<unsigned char>(i >> 8) &&
                   c.z == static_cast<unsigned char>(255 - i) && c.w == 9, "uchar4 [%d]", i);
        }
        for (cudaSurfaceObject_t s: {s8, s16, s32}) {
            CHECK(cudaDestroySurfaceObject(s));
        }
        for (cudaArray_t a: {a8, a16, a32}) {
            CHECK(cudaFreeArray(a));
        }
    }

    // 3D, layered and 1D
    {
        const int width = 10, height = 7, depth = 5, n3 = width * height * depth;
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
        cudaArray_t volume, layered, line;
        CHECK(cudaMalloc3DArray(&volume, &desc, make_cudaExtent(width, height, depth), cudaArraySurfaceLoadStore));
        CHECK(cudaMalloc3DArray(&layered, &desc, make_cudaExtent(width, height, depth),
                                cudaArrayLayered | cudaArraySurfaceLoadStore));
        CHECK(cudaMallocArray(&line, &desc, 200, 0, cudaArraySurfaceLoadStore));
        const cudaSurfaceObject_t sv = surfaceOf(volume), sl = surfaceOf(layered), s1 = surfaceOf(line);
        write3D<<<4, 128>>>(sv, width, height, depth);
        writeLayered<<<4, 128>>>(sl, width, height, depth);
        write1D<<<2, 128>>>(s1, 200);
        std::vector<float> rv(n3), rl(n3), r1(200);
        cudaMemcpy3DParms p{};
        p.dstPtr = make_cudaPitchedPtr(rv.data(), width * sizeof(float), width, height);
        p.srcArray = volume;
        p.extent = make_cudaExtent(width, height, depth);
        p.kind = cudaMemcpyDeviceToHost;
        CHECK(cudaMemcpy3D(&p));
        p.dstPtr = make_cudaPitchedPtr(rl.data(), width * sizeof(float), width, height);
        p.srcArray = layered;
        CHECK(cudaMemcpy3D(&p));
        CHECK(cudaMemcpy2DFromArray(r1.data(), 200 * sizeof(float), line, 0, 0, 200 * sizeof(float), 1,
                                    cudaMemcpyDeviceToHost));
        for (int i = 0; i < n3; i++) {
            EXPECT(rv[i] == i * 0.5f, "surf3Dwrite [%d] = %g", i, rv[i]);
            EXPECT(rl[i] == i * 3.0f, "surf2DLayeredwrite [%d] = %g", i, rl[i]);
        }
        for (int i = 0; i < 200; i++) {
            EXPECT(r1[i] == i - 50.0f, "surf1Dwrite [%d] = %g", i, r1[i]);
        }
        for (cudaSurfaceObject_t s: {sv, sl, s1}) {
            CHECK(cudaDestroySurfaceObject(s));
        }
        for (cudaArray_t a: {volume, layered, line}) {
            CHECK(cudaFreeArray(a));
        }
    }

    // cubemap faces
    {
        const int width = 9, n = width * width * 6;
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
        cudaArray_t cube;
        CHECK(cudaMalloc3DArray(&cube, &desc, make_cudaExtent(width, width, 6),
                                cudaArrayCubemap | cudaArraySurfaceLoadStore));
        const cudaSurfaceObject_t s = surfaceOf(cube);
        writeCubemap<<<4, 128>>>(s, width);
        std::vector<float> faces(n);
        cudaMemcpy3DParms p{};
        p.dstPtr = make_cudaPitchedPtr(faces.data(), width * sizeof(float), width, width);
        p.srcArray = cube;
        p.extent = make_cudaExtent(width, width, 6);
        p.kind = cudaMemcpyDeviceToHost;
        CHECK(cudaMemcpy3D(&p));
        for (int i = 0; i < n; i++) {
            EXPECT(faces[i] == i + 0.25f, "surfCubemapwrite [%d] = %g", i, faces[i]);
        }
        CHECK(cudaDestroySurfaceObject(s));
        CHECK(cudaFreeArray(cube));
    }

    CHECK(cudaDeviceSynchronize());
    PASS("surface_objects");
    return 0;
}
