#pragma once

#include "batch_workspace.cuh"

namespace baseline {

__global__ void delete_and_check_shrink_kernel(int num_unique,
                                               const vid_t* __restrict__ unique_srcs,
                                               const eoff_t* __restrict__ batch_offsets,
                                               const int* __restrict__ batch_deg,
                                               const vid_t* __restrict__ batch_dst,
                                               vid_t* __restrict__ d_used,
                                               vid_t** __restrict__ d_ptr,
                                               uint8_t* __restrict__ need_shrink,
                                               uint8_t* __restrict__ need_free_only,
                                               vid_t* __restrict__ all_old_deg,
                                               vid_t* __restrict__ all_new_deg,
                                               vid_t** __restrict__ all_old_ptr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_unique) return;

    vid_t v = unique_srcs[i];
    vid_t old_deg = d_used[v];
    vid_t* adj = d_ptr[v];
    all_old_deg[i] = old_deg;
    all_old_ptr[i] = adj;
    need_shrink[i] = 0;
    need_free_only[i] = 0;

    if (old_deg == 0 || adj == nullptr) {
        all_new_deg[i] = 0;
        return;
    }

    eoff_t b_off = batch_offsets[i];
    int del_deg = batch_deg[i];

    vid_t write_pos = 0;
    for (vid_t k = 0; k < old_deg; ++k) {
        vid_t nbr = adj[k];
        int lo = 0;
        int hi = del_deg - 1;
        bool deleted = false;
        while (lo <= hi) {
            int mid = lo + ((hi - lo) >> 1);
            vid_t val = batch_dst[b_off + mid];
            if (val == nbr) {
                deleted = true;
                break;
            } else if (val < nbr) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        if (!deleted) {
            adj[write_pos++] = nbr;
        }
    }

    vid_t new_deg = write_pos;
    all_new_deg[i] = new_deg;
    d_used[v] = new_deg;

    int old_cap = 1 << dev_ceil_log2(old_deg);
    if (new_deg == 0) {
        d_ptr[v] = nullptr;
        need_free_only[i] = 1;
    } else if (new_deg < old_cap / 2) {
        need_shrink[i] = 1;
    }
}

/**
 * @brief Reusable batch scratch workspace (matches original Hornet's BatchUpdate buffers).
 * Pre-allocates temporary CUB/batch arrays before insert()/erase() is timed.
 */

inline void batch_delete(HornetGraph& graph, BatchWorkspace& ws) {
    ScopedPhaseTimer total_timer("delete_total", true);
    int batch_size = ws.batch_size;
    if (batch_size == 0) return;

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

    cub::DeviceRunLengthEncode::Encode(
        ws.d_cub_temp, ws.cub_temp_bytes, ws.d_src_uniq, ws.d_unique_srcs, ws.d_batch_deg, ws.d_scalar_count, uniq_count);

    int num_unique = 0;
    CUDA_CHECK(cudaMemcpy(&num_unique, ws.d_scalar_count, sizeof(int), cudaMemcpyDeviceToHost));

    cub::DeviceScan::ExclusiveSum(
        ws.d_cub_temp, ws.cub_temp_bytes, ws.d_batch_deg, ws.d_batch_offsets, num_unique);

    delete_and_check_shrink_kernel<<<(num_unique + blk - 1) / blk, blk>>>(
        num_unique, ws.d_unique_srcs, ws.d_batch_offsets, ws.d_batch_deg, ws.d_dst_uniq,
        graph.d_used(), graph.d_ptr(), ws.d_need_realloc, ws.d_need_free_only,
        ws.d_all_old_deg, ws.d_all_new_deg, ws.d_all_old_ptr);
    CUDA_CHECK(cudaGetLastError());

    // Reclaim fully emptied blocks (new_deg == 0)
    cub::DeviceSelect::Flagged(
        ws.d_cub_temp, ws.cub_temp_bytes, ws.d_all_old_deg, ws.d_need_free_only, ws.d_realloc_old_deg, ws.d_scalar_count, num_unique);
    cub::DeviceSelect::Flagged(
        ws.d_cub_temp, ws.cub_temp_bytes, ws.d_all_old_ptr, ws.d_need_free_only, ws.d_realloc_old_ptr, ws.d_scalar_count, num_unique);

    int num_free_only = 0;
    CUDA_CHECK(cudaMemcpy(&num_free_only, ws.d_scalar_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (num_free_only > 0) {
        std::vector<vid_t> h_deg(num_free_only);
        std::vector<vid_t*> h_ptr(num_free_only);
        CUDA_CHECK(cudaMemcpy(h_deg.data(), ws.d_realloc_old_deg, num_free_only * sizeof(vid_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ptr.data(), ws.d_realloc_old_ptr, num_free_only * sizeof(vid_t*), cudaMemcpyDeviceToHost));
        auto& mgr = graph.memory_manager();
        for (int i = 0; i < num_free_only; ++i) {
            if (h_deg[i] > 0 && h_ptr[i]) {
                mgr.reclaim_block(h_ptr[i], ceil_log2(h_deg[i]));
            }
        }
    }

    // Shrink blocks where new_deg < bsize(old_deg) / 2
    cub::DeviceSelect::Flagged(
        ws.d_cub_temp, ws.cub_temp_bytes, ws.d_unique_srcs, ws.d_need_realloc, ws.d_realloc_v, ws.d_scalar_count, num_unique);
    cub::DeviceSelect::Flagged(
        ws.d_cub_temp, ws.cub_temp_bytes, ws.d_all_old_deg, ws.d_need_realloc, ws.d_realloc_old_deg, ws.d_scalar_count, num_unique);
    cub::DeviceSelect::Flagged(
        ws.d_cub_temp, ws.cub_temp_bytes, ws.d_all_new_deg, ws.d_need_realloc, ws.d_realloc_new_deg, ws.d_scalar_count, num_unique);
    cub::DeviceSelect::Flagged(
        ws.d_cub_temp, ws.cub_temp_bytes, ws.d_all_old_ptr, ws.d_need_realloc, ws.d_realloc_old_ptr, ws.d_scalar_count, num_unique);

    int num_shrink = 0;
    CUDA_CHECK(cudaMemcpy(&num_shrink, ws.d_scalar_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (num_shrink > 0) {
        std::vector<vid_t> h_old_deg(num_shrink), h_new_deg(num_shrink);
        std::vector<vid_t*> h_old_ptr(num_shrink), h_new_ptr(num_shrink);
        CUDA_CHECK(cudaMemcpy(h_old_deg.data(), ws.d_realloc_old_deg, num_shrink * sizeof(vid_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_new_deg.data(), ws.d_realloc_new_deg, num_shrink * sizeof(vid_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_old_ptr.data(), ws.d_realloc_old_ptr, num_shrink * sizeof(vid_t*), cudaMemcpyDeviceToHost));

        auto& mgr = graph.memory_manager();
        for (int i = 0; i < num_shrink; ++i) {
            h_new_ptr[i] = mgr.allocate_block(ceil_log2(h_new_deg[i]), false).ptr;
        }

        CUDA_CHECK(cudaMemcpy(ws.d_realloc_new_ptr, h_new_ptr.data(), num_shrink * sizeof(vid_t*), cudaMemcpyHostToDevice));

        std::vector<eoff_t> h_move_offsets(num_shrink + 1, 0);
        for (int i = 0; i < num_shrink; ++i) {
            h_move_offsets[i + 1] = h_move_offsets[i] + h_new_deg[i];
        }
        eoff_t total_moves = h_move_offsets[num_shrink];
        if (total_moves > 0) {
            CUDA_CHECK(cudaMemcpy(ws.d_move_offsets, h_move_offsets.data(),
                                  (num_shrink + 1) * sizeof(eoff_t), cudaMemcpyHostToDevice));
            eoff_t grid = (total_moves + blk - 1) / blk;
            adj_move_balanced_kernel<<<static_cast<unsigned int>(grid), blk>>>(
                num_shrink, ws.d_move_offsets, total_moves,
                ws.d_realloc_v, ws.d_realloc_old_ptr, ws.d_realloc_new_ptr, graph.d_ptr());
            CUDA_CHECK(cudaGetLastError());
        }

        for (int i = 0; i < num_shrink; ++i) {
            mgr.reclaim_block(h_old_ptr[i], ceil_log2(h_old_deg[i]));
        }
    }
}


} // namespace baseline
