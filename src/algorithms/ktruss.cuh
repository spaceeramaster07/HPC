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
// Dynamic Maximal K-Truss (Paper Section IV-D, Fig. 9)
// =============================================================================

__global__ void ktruss_count_and_flag_kernel(vid_t nV,
                                             const eoff_t* __restrict__ d_offsets,
                                             eoff_t total_edges,
                                             const vid_t* __restrict__ d_used,
                                             vid_t* const* __restrict__ d_ptr,
                                             int min_triangles,
                                             vid_t* __restrict__ out_u,
                                             vid_t* __restrict__ out_v,
                                             uint8_t* __restrict__ out_prune) {
    eoff_t gid = static_cast<eoff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_edges) return;

    // Binary search vertex u owning edge gid
    vid_t lo = 0;
    vid_t hi = nV - 1;
    while (lo <= hi) {
        vid_t mid = lo + ((hi - lo) >> 1);
        if (d_offsets[mid + 1] <= gid) {
            lo = mid + 1;
        } else if (d_offsets[mid] > gid) {
            hi = mid - 1;
        } else {
            lo = mid;
            break;
        }
    }
    vid_t u = lo;
    eoff_t local_e = gid - d_offsets[u];
    const vid_t* adj_u = d_ptr[u];
    vid_t v = adj_u[local_e];

    out_u[gid] = u;
    out_v[gid] = v;

    vid_t deg_u = d_used[u];
    vid_t deg_v = d_used[v];
    const vid_t* adj_v = d_ptr[v];

    // Iterate over smaller sorted list and binary search larger sorted list
    const vid_t* small_adj = (deg_u <= deg_v) ? adj_u : adj_v;
    vid_t small_deg = (deg_u <= deg_v) ? deg_u : deg_v;
    const vid_t* large_adj = (deg_u <= deg_v) ? adj_v : adj_u;
    vid_t large_deg = (deg_u <= deg_v) ? deg_v : deg_u;

    int triangles = 0;
    for (vid_t i = 0; i < small_deg; ++i) {
        vid_t w = small_adj[i];
        vid_t l = 0;
        vid_t r = large_deg - 1;
        while (l <= r) {
            vid_t m = l + ((r - l) >> 1);
            vid_t val = large_adj[m];
            if (val == w) {
                ++triangles;
                if (triangles >= min_triangles) {
                    out_prune[gid] = 0;
                    return;
                }
                break;
            } else if (val < w) {
                l = m + 1;
            } else {
                if (m == 0) break;
                r = m - 1;
            }
        }
    }

    out_prune[gid] = (triangles < min_triangles) ? 1 : 0;
}

struct KTrussStep {
    int k = 0;
    int deleted_edges = 0;
    double step_ms = 0.0;
    double update_rate_per_sec = 0.0;
};

inline std::vector<KTrussStep> run_ktruss(const CSRGraph& sym_csr, int max_k = 100) {
    std::vector<KTrussStep> steps;
    vid_t nV = sym_csr.nV;
    eoff_t init_nE = sym_csr.nE;
    if (nV == 0 || init_nE == 0) return steps;

    HornetGraph hg(sym_csr);
    BatchWorkspace ws;
    ws.ensure_capacity(init_nE);

    eoff_t* d_offsets = nullptr;
    vid_t* d_all_u = nullptr;
    vid_t* d_all_v = nullptr;
    uint8_t* d_prune = nullptr;
    int* d_num_pruned = nullptr;

    CUDA_CHECK(cudaMalloc(&d_offsets, (nV + 1) * sizeof(eoff_t)));
    CUDA_CHECK(cudaMalloc(&d_all_u, init_nE * sizeof(vid_t)));
    CUDA_CHECK(cudaMalloc(&d_all_v, init_nE * sizeof(vid_t)));
    CUDA_CHECK(cudaMalloc(&d_prune, init_nE * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_num_pruned, sizeof(int)));

    void* d_scan_temp = nullptr;
    size_t scan_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_scan_temp, scan_temp_bytes, hg.d_used(), d_offsets, nV + 1);
    CUDA_CHECK(cudaMalloc(&d_scan_temp, scan_temp_bytes));

    int k = 3;
    int blk = 256;

    while (k <= max_k) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();

        // 1. Compute active edge offsets
        cub::DeviceScan::ExclusiveSum(d_scan_temp, scan_temp_bytes, hg.d_used(), d_offsets, nV + 1);
        eoff_t total_edges = 0;
        CUDA_CHECK(cudaMemcpy(&total_edges, &d_offsets[nV], sizeof(eoff_t), cudaMemcpyDeviceToHost));
        if (total_edges == 0) break;

        // 2. Count triangles per edge & flag edges with triangles < k - 2
        eoff_t grid = (total_edges + blk - 1) / blk;
        ktruss_count_and_flag_kernel<<<static_cast<unsigned int>(grid), blk>>>(
            nV, d_offsets, total_edges, hg.d_used(), hg.d_ptr(), k - 2,
            d_all_u, d_all_v, d_prune);
        CUDA_CHECK(cudaGetLastError());

        // 3. Compact flagged edges directly into ws.d_raw_src / ws.d_raw_dst
        cub::DeviceSelect::Flagged(
            ws.d_cub_temp, ws.cub_temp_bytes, d_all_u, d_prune, ws.d_raw_src, d_num_pruned, total_edges);
        cub::DeviceSelect::Flagged(
            ws.d_cub_temp, ws.cub_temp_bytes, d_all_v, d_prune, ws.d_raw_dst, d_num_pruned, total_edges);

        int num_pruned = 0;
        CUDA_CHECK(cudaMemcpy(&num_pruned, d_num_pruned, sizeof(int), cudaMemcpyDeviceToHost));

        if (num_pruned == 0) {
            ++k;
            continue;
        }

        // 4. Batch delete pruned edges from Hornet
        ws.batch_size = num_pruned;
        batch_delete(hg, ws);

        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        double step_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double rate = (step_ms > 0.0) ? (static_cast<double>(num_pruned) / (step_ms / 1000.0)) : 0.0;

        steps.push_back({k, num_pruned, step_ms, rate});
    }

    cudaFree(d_scan_temp);
    cudaFree(d_offsets);
    cudaFree(d_all_u);
    cudaFree(d_all_v);
    cudaFree(d_prune);
    cudaFree(d_num_pruned);

    return steps;
}


} // namespace hornet_alg
