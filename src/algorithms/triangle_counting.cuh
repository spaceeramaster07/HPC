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
// Dynamic Triangle Counting (Batch Insert + Batch Triangle Query)
// =============================================================================

struct DynamicTCResult {
    double insert_ms = 0.0;
    double query_ms = 0.0;
    double total_ms = 0.0;
    unsigned long long delta_triangles = 0;
};

__global__ void tc_batch_intersect_kernel(int num_edges,
                                            const vid_t* __restrict__ batch_u,
                                            const vid_t* __restrict__ batch_v,
                                            vid_t* const* __restrict__ d_ptrs,
                                            const vid_t* __restrict__ d_used,
                                            unsigned long long* __restrict__ d_total_tri) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    if (warp_id >= num_edges) return;

    vid_t u = batch_u[warp_id];
    vid_t v = batch_v[warp_id];
    if (u == v) return;

    vid_t du = d_used[u];
    vid_t dv = d_used[v];
    if (du == 0 || dv == 0) return;

    const vid_t* small_ptr = (du <= dv) ? d_ptrs[u] : d_ptrs[v];
    const vid_t* large_ptr = (du <= dv) ? d_ptrs[v] : d_ptrs[u];
    vid_t small_len = (du <= dv) ? du : dv;
    vid_t large_len = (du <= dv) ? dv : du;

    unsigned long long local_cnt = 0;
    for (vid_t i = lane; i < small_len; i += 32) {
        vid_t target = small_ptr[i];
        int lo = 0, hi = large_len - 1;
        while (lo <= hi) {
            int mid = lo + ((hi - lo) >> 1);
            vid_t val = large_ptr[mid];
            if (val == target) {
                ++local_cnt;
                break;
            }
            if (val < target) lo = mid + 1;
            else hi = mid - 1;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        local_cnt += __shfl_down_sync(0xffffffff, local_cnt, offset);
    }
    if (lane == 0 && local_cnt > 0) {
        atomicAdd(d_total_tri, local_cnt);
    }
}

inline DynamicTCResult run_dynamic_tc(const CSRGraph& sym_csr,
                                      int batch_size,
                                      uint64_t seed) {
    DynamicTCResult res;
    if (sym_csr.nV <= 1 || batch_size <= 0) return res;

    HornetGraph hg(sym_csr);

    // Generate undirected batch (u, v) and its symmetrized form for Hornet insertion
    std::vector<vid_t> raw_u, raw_v;
    generate_random_batch(sym_csr.nV, batch_size, seed, raw_u, raw_v);

    std::vector<vid_t> sym_u, sym_v;
    sym_u.reserve(batch_size * 2);
    sym_v.reserve(batch_size * 2);
    std::vector<vid_t> q_u, q_v;
    q_u.reserve(batch_size);
    q_v.reserve(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        if (raw_u[i] == raw_v[i]) continue;
        q_u.push_back(raw_u[i]);
        q_v.push_back(raw_v[i]);
        sym_u.push_back(raw_u[i]);
        sym_v.push_back(raw_v[i]);
        sym_u.push_back(raw_v[i]);
        sym_v.push_back(raw_u[i]);
    }
    int num_q = static_cast<int>(q_u.size());
    if (num_q == 0) return res;

    BatchWorkspace ws;
    ws.ensure_capacity(static_cast<int>(sym_u.size()));
    ws.prepare(sym_u, sym_v);

    vid_t *d_qu = nullptr, *d_qv = nullptr;
    unsigned long long* d_tri = nullptr;
    CUDA_CHECK(cudaMalloc(&d_qu, num_q * sizeof(vid_t)));
    CUDA_CHECK(cudaMalloc(&d_qv, num_q * sizeof(vid_t)));
    CUDA_CHECK(cudaMalloc(&d_tri, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_qu, q_u.data(), num_q * sizeof(vid_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qv, q_v.data(), num_q * sizeof(vid_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_tri, 0, sizeof(unsigned long long)));

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();

    batch_insert(hg, ws, true);

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (num_q + warps_per_block - 1) / warps_per_block;
    tc_batch_intersect_kernel<<<blocks, threads>>>(
        num_q, d_qu, d_qv, hg.d_ptr(), hg.d_used(), d_tri);

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(&res.delta_triangles, d_tri, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    res.insert_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.query_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    res.total_ms = res.insert_ms + res.query_ms;

    cudaFree(d_qu);
    cudaFree(d_qv);
    cudaFree(d_tri);
    return res;
}


} // namespace hornet_alg
