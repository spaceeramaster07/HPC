#pragma once

#include "batch_update.cuh"
#include "hornet_graph.cuh"
#include "types.cuh"

#include <cub/cub.cuh>
#include <chrono>
#include <vector>

#ifndef HORNET_DS_NS
#define HORNET_DS_NS baseline
#endif

namespace hornet_alg {

using namespace hornet_common;
using namespace HORNET_DS_NS;

// =============================================================================
// SpMV (Hornet Tier 1 & Static CSR Reference)
// =============================================================================

__global__ void spmv_warp_kernel(vid_t nV,
                                 const vid_t* __restrict__ d_used,
                                 vid_t* const* __restrict__ d_ptr,
                                 const float* __restrict__ x,
                                 float* __restrict__ y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    vid_t v = tid >> 5;
    int lane = tid & 31;
    if (v >= nV) return;

    vid_t deg = d_used[v];
    const vid_t* adj = d_ptr[v];

    float sum = 0.0f;
    for (vid_t i = lane; i < deg; i += 32) {
        sum += x[adj[i]];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
    }

    if (lane == 0) {
        y[v] = sum;
    }
}

__global__ void spmv_warp_csr_kernel(vid_t nV,
                                     const eoff_t* __restrict__ csr_off,
                                     const vid_t* __restrict__ csr_edges,
                                     const float* __restrict__ x,
                                     float* __restrict__ y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    vid_t v = tid >> 5;
    int lane = tid & 31;
    if (v >= nV) return;

    eoff_t start = csr_off[v];
    eoff_t end = csr_off[v + 1];

    float sum = 0.0f;
    for (eoff_t i = start + lane; i < end; i += 32) {
        sum += x[csr_edges[i]];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
    }

    if (lane == 0) {
        y[v] = sum;
    }
}

struct SpMVResult {
    double time_ms = 0.0;
};

inline SpMVResult run_spmv(const HornetGraph& graph, int num_iters = 10) {
    vid_t nV = graph.nV();
    SpMVResult res;
    if (nV == 0) return res;

    float* d_x = nullptr;
    float* d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, nV * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, nV * sizeof(float)));

    std::vector<float> h_x(nV, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), nV * sizeof(float), cudaMemcpyHostToDevice));

    int blk = 256;
    int warps_per_blk = blk / 32;
    int grid = (nV + warps_per_blk - 1) / warps_per_blk;

    spmv_warp_kernel<<<grid, blk>>>(nV, graph.d_used(), graph.d_ptr(), d_x, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < num_iters; ++it) {
        spmv_warp_kernel<<<grid, blk>>>(nV, graph.d_used(), graph.d_ptr(), d_x, d_y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / num_iters;

    cudaFree(d_x);
    cudaFree(d_y);
    return res;
}

inline SpMVResult run_spmv_csr(const CSRGraph& csr, int num_iters = 10) {
    vid_t nV = csr.nV;
    SpMVResult res;
    if (nV == 0) return res;

    eoff_t* d_csr_off = nullptr;
    vid_t* d_csr_edges = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_csr_off, (nV + 1) * sizeof(eoff_t)));
    CUDA_CHECK(cudaMalloc(&d_csr_edges, std::max<eoff_t>(csr.nE, 1) * sizeof(vid_t)));
    CUDA_CHECK(cudaMalloc(&d_x, nV * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, nV * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_csr_off, csr.row_offsets.data(), (nV + 1) * sizeof(eoff_t), cudaMemcpyHostToDevice));
    if (csr.nE > 0) {
        CUDA_CHECK(cudaMemcpy(d_csr_edges, csr.col_indices.data(), csr.nE * sizeof(vid_t), cudaMemcpyHostToDevice));
    }
    std::vector<float> h_x(nV, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), nV * sizeof(float), cudaMemcpyHostToDevice));

    int blk = 256;
    int warps_per_blk = blk / 32;
    int grid = (nV + warps_per_blk - 1) / warps_per_blk;

    spmv_warp_csr_kernel<<<grid, blk>>>(nV, d_csr_off, d_csr_edges, d_x, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < num_iters; ++it) {
        spmv_warp_csr_kernel<<<grid, blk>>>(nV, d_csr_off, d_csr_edges, d_x, d_y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / num_iters;

    cudaFree(d_csr_off);
    cudaFree(d_csr_edges);
    cudaFree(d_x);
    cudaFree(d_y);
    return res;
}


} // namespace hornet_alg
