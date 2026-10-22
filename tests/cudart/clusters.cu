// Thread block clusters: compile-time (__cluster_dims__) and launch-time (cudaLaunchKernelEx) cluster sizes, cluster
// barriers and distributed shared memory, cluster attributes and occupancy, invalid cluster launches, and non-portable
// cluster sizes. Checked against NVIDIA's runtime too.
#include "check.h"
#include <cooperative_groups.h>
#include <vector>

namespace cg = cooperative_groups;

constexpr int THREADS = 128;

// every block sums its slice; rank 0 of each cluster adds up the partial sums of all blocks of its cluster through
// distributed shared memory
__global__ void __cluster_dims__(4, 1, 1) clusterSum(const int *in, int *cluster_sums, int *ranks) {
    __shared__ int partial;
    cg::cluster_group cluster = cg::this_cluster();
    cg::thread_block block = cg::this_thread_block();
    if (threadIdx.x == 0) {
        partial = 0;
    }
    block.sync();
    atomicAdd(&partial, in[blockIdx.x * THREADS + threadIdx.x]);
    cluster.sync();
    if (threadIdx.x == 0) {
        ranks[blockIdx.x] = cluster.block_rank();
        if (cluster.block_rank() == 0) {
            int sum = 0;
            for (unsigned r = 0; r < cluster.num_blocks(); r++) {
                sum += *cluster.map_shared_rank(&partial, r);
            }
            cluster_sums[blockIdx.x / cluster.num_blocks()] = sum;
        }
    }
    // keep every block's shared memory alive until rank 0 read it
    cluster.sync();
}

// no compile-time cluster size: launched with clusters through cudaLaunchKernelEx. Each block exchanges a value with
// its partner (rank ^ 1) and records its position in the cluster.
__global__ void exchange(int *out) {
    __shared__ int value;
    cg::cluster_group cluster = cg::this_cluster();
    const int block = blockIdx.x + blockIdx.y * gridDim.x;
    if (threadIdx.x == 0) {
        value = block;
    }
    cluster.sync();
    const int partner = *cluster.map_shared_rank(&value, cluster.block_rank() ^ 1);
    cluster.sync();
    if (threadIdx.x == 0) {
        const dim3 index = cluster.block_index();
        out[block * 4 + 0] = cluster.block_rank();
        out[block * 4 + 1] = partner;
        out[block * 4 + 2] = index.x + 10 * index.y;
        out[block * 4 + 3] = cluster.num_blocks();
    }
}

// a ring through distributed shared memory: each block adds its successor's value (any cluster size)
__global__ void ring(int *out) {
    __shared__ int value;
    cg::cluster_group cluster = cg::this_cluster();
    if (threadIdx.x == 0) {
        value = static_cast<int>(cluster.block_rank()) + 1;
    }
    cluster.sync();
    const unsigned int next = (cluster.block_rank() + 1) % cluster.num_blocks();
    const int successor = *cluster.map_shared_rank(&value, next);
    cluster.sync();
    if (threadIdx.x == 0) {
        out[blockIdx.x] = static_cast<int>(cluster.num_blocks()) * 1000 + successor;
    }
}

// a regular launch: every block is a cluster of one
__global__ void noCluster(int *out) {
    if (threadIdx.x == 0) {
        cg::cluster_group cluster = cg::this_cluster();
        out[blockIdx.x * 2] = cluster.block_rank();
        out[blockIdx.x * 2 + 1] = cluster.num_blocks();
    }
}

int main() {
    int supported = 0;
    CHECK(cudaDeviceGetAttribute(&supported, cudaDevAttrClusterLaunch, 0));
    EXPECT(supported == 1, "cluster launch attribute is %d", supported);

    // compile-time cluster size
    {
        const int blocks = 64;
        std::vector<int> in(blocks * THREADS);
        for (size_t i = 0; i < in.size(); i++) {
            in[i] = static_cast<int>(i % 97);
        }
        int *d_in, *d_sums, *d_ranks;
        CHECK(cudaMalloc(&d_in, in.size() * sizeof(int)));
        CHECK(cudaMalloc(&d_sums, blocks / 4 * sizeof(int)));
        CHECK(cudaMalloc(&d_ranks, blocks * sizeof(int)));
        CHECK(cudaMemcpy(d_in, in.data(), in.size() * sizeof(int), cudaMemcpyHostToDevice));
        clusterSum<<<blocks, THREADS>>>(d_in, d_sums, d_ranks);
        CHECK(cudaGetLastError());
        std::vector<int> sums(blocks / 4), ranks(blocks);
        CHECK(cudaMemcpy(sums.data(), d_sums, sums.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(ranks.data(), d_ranks, ranks.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (int c = 0; c < blocks / 4; c++) {
            int expected = 0;
            for (int i = c * 4 * THREADS; i < (c + 1) * 4 * THREADS; i++) {
                expected += in[i];
            }
            EXPECT(sums[c] == expected, "cluster %d: sum %d, expected %d", c, sums[c], expected);
        }
        for (int b = 0; b < blocks; b++) {
            EXPECT(ranks[b] == b % 4, "block %d has cluster rank %d", b, ranks[b]);
        }

        cudaFuncAttributes attrs{};
        CHECK(cudaFuncGetAttributes(&attrs, clusterSum));
        EXPECT(attrs.requiredClusterWidth == 4 && attrs.requiredClusterHeight == 1 && attrs.requiredClusterDepth == 1,
               "required cluster size %dx%dx%d", attrs.requiredClusterWidth, attrs.requiredClusterHeight,
               attrs.requiredClusterDepth);

        // a different launch-time cluster size conflicts with the compiled one
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeClusterDimension;
        attribute.val.clusterDim.x = 2;
        attribute.val.clusterDim.y = 1;
        attribute.val.clusterDim.z = 1;
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(blocks);
        config.blockDim = dim3(THREADS);
        config.attrs = &attribute;
        config.numAttrs = 1;
        EXPECT(cudaLaunchKernelEx(&config, clusterSum, (const int *) d_in, d_sums, d_ranks) ==
               cudaErrorInvalidClusterSize, "conflicting cluster size accepted");
        cudaGetLastError();
        CHECK(cudaFree(d_in));
        CHECK(cudaFree(d_sums));
        CHECK(cudaFree(d_ranks));
    }

    // launch-time 2x2 clusters on an 8x4 grid
    {
        const dim3 grid(8, 4);
        const int blocks = grid.x * grid.y;
        int *d_out;
        CHECK(cudaMalloc(&d_out, blocks * 4 * sizeof(int)));
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeClusterDimension;
        attribute.val.clusterDim.x = 2;
        attribute.val.clusterDim.y = 2;
        attribute.val.clusterDim.z = 1;
        cudaLaunchConfig_t config{};
        config.gridDim = grid;
        config.blockDim = dim3(64);
        config.attrs = &attribute;
        config.numAttrs = 1;
        CHECK(cudaLaunchKernelEx(&config, exchange, d_out));
        std::vector<int> out(blocks * 4);
        CHECK(cudaMemcpy(out.data(), d_out, out.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (unsigned y = 0; y < grid.y; y++) {
            for (unsigned x = 0; x < grid.x; x++) {
                const int block = x + y * grid.x;
                const int rank = (x % 2) + 2 * (y % 2);
                // the partner (rank ^ 1) is the neighbour in x within the cluster
                const int partner = (x ^ 1) + y * grid.x;
                const int *r = &out[block * 4];
                EXPECT(r[0] == rank && r[1] == partner && r[2] == int(x % 2 + 10 * (y % 2)) && r[3] == 4,
                       "block (%u,%u): rank %d partner %d index %d size %d", x, y, r[0], r[1], r[2], r[3]);
            }
        }

        int clusters = 0, potential = 0;
        config.blockDim = dim3(64);
        CHECK(cudaOccupancyMaxActiveClusters(&clusters, exchange, &config));
        CHECK(cudaOccupancyMaxPotentialClusterSize(&potential, exchange, &config));
        printf("occupancy: %d active 2x2 clusters, potential cluster size %d\n", clusters, potential);
        EXPECT(clusters > 0 && potential >= 4, "cluster occupancy %d / %d", clusters, potential);

        // invalid: grid not a multiple of the cluster, and a cluster above the portable size of 8
        config.gridDim = dim3(7, 4);
        EXPECT(cudaLaunchKernelEx(&config, exchange, d_out) == cudaErrorInvalidClusterSize,
               "grid not divisible by the cluster accepted");
        cudaGetLastError();
        attribute.val.clusterDim.x = 16;
        attribute.val.clusterDim.y = 1;
        config.gridDim = dim3(16);
        EXPECT(cudaLaunchKernelEx(&config, exchange, d_out) == cudaErrorInvalidClusterSize,
               "cluster of 16 accepted");
        cudaGetLastError();
        CHECK(cudaFree(d_out));
    }

    // regular launch: clusters of one block
    {
        int *d_out;
        CHECK(cudaMalloc(&d_out, 8 * 2 * sizeof(int)));
        noCluster<<<8, 32>>>(d_out);
        std::vector<int> out(16);
        CHECK(cudaMemcpy(out.data(), d_out, out.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (int b = 0; b < 8; b++) {
            EXPECT(out[b * 2] == 0 && out[b * 2 + 1] == 1, "block %d: rank %d of %d", b, out[b * 2],
                   out[b * 2 + 1]);
        }
        CHECK(cudaFree(d_out));
    }

    // non-portable cluster sizes: more than 8 blocks (up to the SMs of a GPC) once the kernel allows them
    {
        cudaLaunchConfig_t config{};
        config.blockDim = 64;
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeClusterDimension;
        attribute.val.clusterDim.y = attribute.val.clusterDim.z = 1;
        config.attrs = &attribute;
        config.numAttrs = 1;
        int *d_out;
        CHECK(cudaMallocManaged(&d_out, 64 * sizeof(int)));
        // without a cluster size, no clusters are active
        cudaLaunchConfig_t plain = config;
        plain.gridDim = 32;
        plain.numAttrs = 0;
        int clusters = -1;
        CHECK(cudaOccupancyMaxActiveClusters(&clusters, (void *) ring, &plain));
        EXPECT(clusters == 0, "active clusters without a cluster size: %d", clusters);
        attribute.val.clusterDim.x = 2;
        config.gridDim = 32;
        CHECK(cudaOccupancyMaxActiveClusters(&clusters, (void *) ring, &config));
        EXPECT(clusters > 0, "active clusters of 2: %d", clusters);

        attribute.val.clusterDim.x = 16;
        config.gridDim = 16;
        EXPECT(cudaLaunchKernelEx(&config, ring, d_out) == cudaErrorInvalidClusterSize, "16 blocks, portable");
        cudaGetLastError();
        CHECK(cudaFuncSetAttribute((void *) ring, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        cudaFuncAttributes attrs{};
        CHECK(cudaFuncGetAttributes(&attrs, ring));
        EXPECT(attrs.nonPortableClusterSizeAllowed == 1, "non-portable attribute %d", attrs.nonPortableClusterSizeAllowed);
        attribute.val.clusterDim.x = 1;
        int potential = 0;
        CHECK(cudaOccupancyMaxPotentialClusterSize(&potential, (void *) ring, &config));
        printf("    largest non-portable cluster: %d blocks\n", potential);
        EXPECT(potential >= 8 && potential <= 16, "potential cluster size %d", potential);
        if (potential > 8) {
            attribute.val.clusterDim.x = potential;
            config.gridDim = 2 * potential;
            for (int i = 0; i < 64; i++) {
                d_out[i] = -1;
            }
            CHECK(cudaLaunchKernelEx(&config, ring, d_out));
            CHECK(cudaDeviceSynchronize());
            for (int b = 0; b < 2 * potential; b++) {
                const int rank = b % potential;
                EXPECT(d_out[b] == potential * 1000 + (rank + 1) % potential + 1, "block %d of a %d block cluster: %d",
                       b, potential, d_out[b]);
            }
        }
        CHECK(cudaFree(d_out));
    }

    CHECK(cudaDeviceSynchronize());
    PASS("clusters");
    return 0;
}
