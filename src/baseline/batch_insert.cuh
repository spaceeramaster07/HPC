#pragma once

#include "batch_workspace.cuh"

namespace baseline {

__global__ void check_cross_dups_warp_kernel(int n_edges,
                                             const vid_t* __restrict__ batch_src,
                                             const vid_t* __restrict__ batch_dst,
                                             const vid_t* __restrict__ d_used,
                                             vid_t* const* __restrict__ d_ptr,
                                             uint8_t* __restrict__ d_keep) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    if (warp_id >= n_edges) return;

    vid_t u = batch_src[warp_id];
    vid_t v = batch_dst[warp_id];
    vid_t deg = d_used[u];
    const vid_t* adj = d_ptr[u];

    bool found = false;
    for (vid_t i = lane; i < deg; i += 32) {
        if (adj[i] == v) {
            found = true;
            break;
        }
    }
    unsigned mask = __ballot_sync(0xFFFFFFFFu, found);
    if (lane == 0) {
        d_keep[warp_id] = (mask == 0u) ? 1 : 0;
    }
}

__device__ __forceinline__ int dev_ceil_log2(int x) {
    if (x <= 1) return 0;
    return 32 - __clz(static_cast<unsigned int>(x - 1));
}

__global__ void check_insert_realloc_kernel(int num_unique,
                                            const vid_t* __restrict__ unique_srcs,
                                            const int* __restrict__ batch_deg,
                                            const vid_t* __restrict__ d_used,
                                            vid_t* const* __restrict__ d_ptr,
                                            uint8_t* __restrict__ need_realloc,
                                            vid_t* __restrict__ all_old_deg,
                                            vid_t* __restrict__ all_new_deg,
                                            vid_t** __restrict__ all_old_ptr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_unique) return;

    vid_t v = unique_srcs[i];
    vid_t old_deg = d_used[v];
    vid_t new_deg = old_deg + batch_deg[i];
    int old_cap = (old_deg == 0) ? 0 : (1 << dev_ceil_log2(old_deg));

    all_old_deg[i] = old_deg;
    all_new_deg[i] = new_deg;
    all_old_ptr[i] = d_ptr[v];
    need_realloc[i] = (new_deg > old_cap) ? 1 : 0;
}

// Binary-search load-balanced adjacency list copy (Paper Footnote 2)

__global__ void update_zero_deg_ptrs_kernel(int num_realloc,
                                            const vid_t* __restrict__ realloc_v,
                                            const vid_t* __restrict__ old_deg,
                                            vid_t* const* __restrict__ new_ptrs,
                                            vid_t** __restrict__ d_ptr) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < num_realloc && old_deg[r] == 0) {
        d_ptr[realloc_v[r]] = new_ptrs[r];
    }
}

__global__ void append_batch_edges_kernel(int num_unique,
                                          const vid_t* __restrict__ unique_srcs,
                                          const eoff_t* __restrict__ batch_offsets,
                                          const int* __restrict__ batch_deg,
                                          const vid_t* __restrict__ batch_dst,
                                          vid_t* __restrict__ d_used,
                                          vid_t* const* __restrict__ d_ptr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_unique) return;

    vid_t v = unique_srcs[i];
    vid_t old_deg = d_used[v];
    vid_t* adj = d_ptr[v];
    eoff_t b_off = batch_offsets[i];
    int add_deg = batch_deg[i];

    for (int k = 0; k < add_deg; ++k) {
        adj[old_deg + k] = batch_dst[b_off + k];
    }
    d_used[v] = old_deg + add_deg;
}


inline void batch_insert(HornetGraph& graph,
                         BatchWorkspace& ws,
                         bool remove_graph_dups) {
    ScopedPhaseTimer total_timer("insert_total", true);
    int batch_size = ws.batch_size;
    if (batch_size == 0) return;

    int num_unique = 0;
    vid_t* d_final_dst = nullptr;

    // -------------------------------------------------------------------------
    // Phase 1: Preprocessing (Sort, Deduplicate, Batch CSR)
    // -------------------------------------------------------------------------
    {
        ScopedPhaseTimer prep_timer("insert_preprocess", true);
        int blk = 256;

        pack_edges_kernel<<<(batch_size + blk - 1) / blk, blk>>>(
            batch_size, ws.d_raw_src, ws.d_raw_dst, ws.d_keys_in);
        CUDA_CHECK(cudaGetLastError());

        cub::DeviceRadixSort::SortKeys(
            ws.d_cub_temp, ws.cub_temp_bytes, ws.d_keys_in, ws.d_keys_sorted, batch_size);

        cub::DeviceSelect::Unique(
            ws.d_cub_temp, ws.cub_temp_bytes, ws.d_keys_sorted, ws.d_keys_uniq, ws.d_scalar_count, batch_size);

        int uniq_count = 0;
        CUDA_CHECK(cudaMemcpy(&uniq_count, ws.d_scalar_count, sizeof(int), cudaMemcpyDeviceToHost));
        if (uniq_count == 0) return;

        unpack_edges_kernel<<<(uniq_count + blk - 1) / blk, blk>>>(
            uniq_count, ws.d_keys_uniq, ws.d_src_uniq, ws.d_dst_uniq);
        CUDA_CHECK(cudaGetLastError());

        vid_t* d_final_src = ws.d_src_uniq;
        d_final_dst = ws.d_dst_uniq;
        int final_count = uniq_count;

        if (remove_graph_dups) {
            int warps_per_blk = blk / 32;
            int grid = (uniq_count + warps_per_blk - 1) / warps_per_blk;
            check_cross_dups_warp_kernel<<<grid, blk>>>(
                uniq_count, ws.d_src_uniq, ws.d_dst_uniq, graph.d_used(), graph.d_ptr(), ws.d_keep);
            CUDA_CHECK(cudaGetLastError());

            cub::DeviceSelect::Flagged(
                ws.d_cub_temp, ws.cub_temp_bytes, ws.d_src_uniq, ws.d_keep, ws.d_filt_src, ws.d_scalar_count, uniq_count);
            cub::DeviceSelect::Flagged(
                ws.d_cub_temp, ws.cub_temp_bytes, ws.d_dst_uniq, ws.d_keep, ws.d_filt_dst, ws.d_scalar_count, uniq_count);

            CUDA_CHECK(cudaMemcpy(&final_count, ws.d_scalar_count, sizeof(int), cudaMemcpyDeviceToHost));
            d_final_src = ws.d_filt_src;
            d_final_dst = ws.d_filt_dst;
        }

        if (final_count == 0) return;

        cub::DeviceRunLengthEncode::Encode(
            ws.d_cub_temp, ws.cub_temp_bytes, d_final_src, ws.d_unique_srcs, ws.d_batch_deg, ws.d_scalar_count, final_count);
        CUDA_CHECK(cudaMemcpy(&num_unique, ws.d_scalar_count, sizeof(int), cudaMemcpyDeviceToHost));

        cub::DeviceScan::ExclusiveSum(
            ws.d_cub_temp, ws.cub_temp_bytes, ws.d_batch_deg, ws.d_batch_offsets, num_unique);
    }

    // -------------------------------------------------------------------------
    // Phase 2: Build Reallocation List (Device -> Host)
    // -------------------------------------------------------------------------
    int num_realloc = 0;
    std::vector<vid_t> h_realloc_v;
    std::vector<vid_t> h_realloc_old_deg;
    std::vector<vid_t> h_realloc_new_deg;
    std::vector<vid_t*> h_realloc_old_ptr;

    {
        ScopedPhaseTimer realloc_timer("insert_build_realloc", true);
        int blk = 256;

        check_insert_realloc_kernel<<<(num_unique + blk - 1) / blk, blk>>>(
            num_unique, ws.d_unique_srcs, ws.d_batch_deg, graph.d_used(), graph.d_ptr(),
            ws.d_need_realloc, ws.d_all_old_deg, ws.d_all_new_deg, ws.d_all_old_ptr);
        CUDA_CHECK(cudaGetLastError());

        cub::DeviceSelect::Flagged(
            ws.d_cub_temp, ws.cub_temp_bytes, ws.d_unique_srcs, ws.d_need_realloc, ws.d_realloc_v, ws.d_scalar_count, num_unique);
        cub::DeviceSelect::Flagged(
            ws.d_cub_temp, ws.cub_temp_bytes, ws.d_all_old_deg, ws.d_need_realloc, ws.d_realloc_old_deg, ws.d_scalar_count, num_unique);
        cub::DeviceSelect::Flagged(
            ws.d_cub_temp, ws.cub_temp_bytes, ws.d_all_new_deg, ws.d_need_realloc, ws.d_realloc_new_deg, ws.d_scalar_count, num_unique);
        cub::DeviceSelect::Flagged(
            ws.d_cub_temp, ws.cub_temp_bytes, ws.d_all_old_ptr, ws.d_need_realloc, ws.d_realloc_old_ptr, ws.d_scalar_count, num_unique);

        CUDA_CHECK(cudaMemcpy(&num_realloc, ws.d_scalar_count, sizeof(int), cudaMemcpyDeviceToHost));

        if (num_realloc > 0) {
            h_realloc_v.resize(num_realloc);
            h_realloc_old_deg.resize(num_realloc);
            h_realloc_new_deg.resize(num_realloc);
            h_realloc_old_ptr.resize(num_realloc);

            CUDA_CHECK(cudaMemcpy(h_realloc_v.data(), ws.d_realloc_v, num_realloc * sizeof(vid_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_realloc_old_deg.data(), ws.d_realloc_old_deg, num_realloc * sizeof(vid_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_realloc_new_deg.data(), ws.d_realloc_new_deg, num_realloc * sizeof(vid_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_realloc_old_ptr.data(), ws.d_realloc_old_ptr, num_realloc * sizeof(vid_t*), cudaMemcpyDeviceToHost));
        }
    }

    if (num_realloc > 0) {
        std::vector<vid_t*> h_realloc_new_ptr(num_realloc, nullptr);

        // ---------------------------------------------------------------------
        // Phase 3: Host Block Allocation via B+Tree & VecTree
        // ---------------------------------------------------------------------
        {
            ScopedPhaseTimer alloc_timer("insert_host_alloc", false);
            auto& mgr = graph.memory_manager();
            for (int r = 0; r < num_realloc; ++r) {
                int log_b = ceil_log2(h_realloc_new_deg[r]);
                h_realloc_new_ptr[r] = mgr.allocate_block(log_b, false).ptr;
            }
        }

        // ---------------------------------------------------------------------
        // Phase 4: Copy New Pointers Host -> Device
        // ---------------------------------------------------------------------
        {
            ScopedPhaseTimer h2d_timer("insert_h2d_copy", true);
            CUDA_CHECK(cudaMemcpy(ws.d_realloc_new_ptr, h_realloc_new_ptr.data(),
                                  num_realloc * sizeof(vid_t*), cudaMemcpyHostToDevice));
        }

        // ---------------------------------------------------------------------
        // Phase 5: Adjacency List Move (Load-Balanced)
        // ---------------------------------------------------------------------
        {
            ScopedPhaseTimer move_timer("insert_adj_move", true);

            std::vector<eoff_t> h_move_offsets(num_realloc + 1, 0);
            for (int r = 0; r < num_realloc; ++r) {
                h_move_offsets[r + 1] = h_move_offsets[r] + h_realloc_old_deg[r];
            }
            eoff_t total_moves = h_move_offsets[num_realloc];

            int blk = 256;
            update_zero_deg_ptrs_kernel<<<(num_realloc + blk - 1) / blk, blk>>>(
                num_realloc, ws.d_realloc_v, ws.d_realloc_old_deg, ws.d_realloc_new_ptr, graph.d_ptr());
            CUDA_CHECK(cudaGetLastError());

            if (total_moves > 0) {
                CUDA_CHECK(cudaMemcpy(ws.d_move_offsets, h_move_offsets.data(),
                                      (num_realloc + 1) * sizeof(eoff_t), cudaMemcpyHostToDevice));

                eoff_t grid = (total_moves + blk - 1) / blk;
                adj_move_balanced_kernel<<<static_cast<unsigned int>(grid), blk>>>(
                    num_realloc, ws.d_move_offsets, total_moves,
                    ws.d_realloc_v, ws.d_realloc_old_ptr, ws.d_realloc_new_ptr, graph.d_ptr());
                CUDA_CHECK(cudaGetLastError());
            }
        }

        // ---------------------------------------------------------------------
        // Phase 6: Host Block Reclamation via B+Tree & VecTree
        // ---------------------------------------------------------------------
        {
            ScopedPhaseTimer free_timer("insert_host_free", false);
            auto& mgr = graph.memory_manager();
            for (int r = 0; r < num_realloc; ++r) {
                if (h_realloc_old_deg[r] > 0 && h_realloc_old_ptr[r] != nullptr) {
                    int old_log_b = ceil_log2(h_realloc_old_deg[r]);
                    mgr.reclaim_block(h_realloc_old_ptr[r], old_log_b);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Phase 7: Append New Edges
    // -------------------------------------------------------------------------
    {
        ScopedPhaseTimer append_timer("insert_append", true);
        int blk = 256;
        append_batch_edges_kernel<<<(num_unique + blk - 1) / blk, blk>>>(
            num_unique, ws.d_unique_srcs, ws.d_batch_offsets, ws.d_batch_deg, d_final_dst,
            graph.d_used(), graph.d_ptr());
        CUDA_CHECK(cudaGetLastError());
    }
}

/**
 * @brief Algorithm 1: Dynamic Batch Deletion (Paper Section III-E).
 */

} // namespace baseline
