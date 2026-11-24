#pragma once

#include "batch_workspace.cuh"
#include <thrust/iterator/counting_iterator.h>

namespace improved {

// DS-5: O(batch_deg * log old_deg) duplicate lookup over sorted adjacency lists
__global__ void check_cross_dups_sorted_kernel(int uniq_count,
                                               const vid_t* __restrict__ src_uniq,
                                               const vid_t* __restrict__ dst_uniq,
                                               const vid_t* __restrict__ d_used,
                                               vid_t* const* __restrict__ d_ptr,
                                               uint8_t* __restrict__ d_keep) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= uniq_count) return;

    vid_t u = src_uniq[gid];
    vid_t target = dst_uniq[gid];
    vid_t deg = d_used[u];
    if (deg <= 0) {
        d_keep[gid] = 1;
        return;
    }
    const vid_t* adj = d_ptr[u];
    if (adj == nullptr) {
        d_keep[gid] = 1;
        return;
    }

    vid_t idx = dev_lower_bound(adj, deg, target);
    d_keep[gid] = (idx < deg && adj[idx] == target) ? 0 : 1;
}

__global__ void check_insert_realloc_kernel(int num_unique,
                                            const vid_t* __restrict__ unique_srcs,
                                            const int* __restrict__ batch_deg,
                                            const vid_t* __restrict__ d_used,
                                            vid_t* const* __restrict__ d_ptr,
                                            uint8_t* __restrict__ need_realloc,
                                            vid_t* __restrict__ all_old_deg,
                                            vid_t* __restrict__ all_new_deg,
                                            vid_t** __restrict__ all_old_ptr,
                                            int* __restrict__ small_inplace_idx,
                                            int* __restrict__ med_inplace_idx,
                                            int* __restrict__ huge_inplace_idx,
                                            int* __restrict__ small_inplace_count,
                                            int* __restrict__ med_inplace_count,
                                            int* __restrict__ huge_inplace_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool in_bounds = (i < num_unique);
    bool realloc = false;
    vid_t new_deg = 0;

    if (in_bounds) {
        vid_t v = unique_srcs[i];
        vid_t old_deg = d_used[v];
        new_deg = old_deg + batch_deg[i];
        int old_cap = (old_deg == 0) ? 0 : (1 << dev_ceil_log2(old_deg));
        realloc = (new_deg > old_cap);

        all_old_deg[i] = old_deg;
        all_new_deg[i] = new_deg;
        all_old_ptr[i] = d_ptr[v];
        need_realloc[i] = realloc ? 1 : 0;
    }

    warp_append_idx(in_bounds && !realloc && new_deg <= 256,
                    i, small_inplace_idx, small_inplace_count);
    warp_append_idx(in_bounds && !realloc && new_deg > 256 && new_deg <= 4096,
                    i, med_inplace_idx, med_inplace_count);
    warp_append_idx(in_bounds && !realloc && new_deg > 4096,
                    i, huge_inplace_idx, huge_inplace_count);
}

__global__ void gather_insert_realloc_metadata_kernel(int num_realloc,
                                                      const int* __restrict__ selected_idx,
                                                      const vid_t* __restrict__ unique_srcs,
                                                      const vid_t* __restrict__ all_old_deg,
                                                      const vid_t* __restrict__ all_new_deg,
                                                      vid_t* const* __restrict__ all_old_ptr,
                                                      const eoff_t* __restrict__ batch_offsets,
                                                      vid_t* __restrict__ realloc_v,
                                                      vid_t* __restrict__ realloc_old_deg,
                                                      vid_t* __restrict__ realloc_new_deg,
                                                      vid_t** __restrict__ realloc_old_ptr,
                                                      eoff_t* __restrict__ realloc_batch_offset,
                                                      int* __restrict__ small_realloc_idx,
                                                      int* __restrict__ large_realloc_idx,
                                                      int* __restrict__ small_realloc_count,
                                                      int* __restrict__ large_realloc_count) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    bool in_bounds = (r < num_realloc);
    vid_t new_deg = 0;

    if (in_bounds) {
        int idx = selected_idx[r];
        new_deg = all_new_deg[idx];
        realloc_v[r] = unique_srcs[idx];
        realloc_old_deg[r] = all_old_deg[idx];
        realloc_new_deg[r] = new_deg;
        realloc_old_ptr[r] = all_old_ptr[idx];
        realloc_batch_offset[r] = batch_offsets[idx];
    }

    warp_append_idx(in_bounds && new_deg <= 256,
                    r, small_realloc_idx, small_realloc_count);
    warp_append_idx(in_bounds && new_deg > 256,
                    r, large_realloc_idx, large_realloc_count);
}

__device__ __forceinline__ void warp_merge_to_out(vid_t v,
                                                   const vid_t* __restrict__ A,
                                                   int lenA,
                                                   const vid_t* __restrict__ B,
                                                   int lenB,
                                                   vid_t* __restrict__ C,
                                                   int total,
                                                   int lane,
                                                   vid_t* s_buf,
                                                   vid_t* __restrict__ d_used,
                                                   vid_t** __restrict__ d_ptr,
                                                   bool write_ptr) {
    if (lenA == 0) {
        for (int k = lane; k < total; k += 32) {
            C[k] = B[k];
        }
        if (lane == 0) {
            if (write_ptr) d_ptr[v] = C;
            d_used[v] = total;
        }
        return;
    }

    vid_t* s_A = s_buf;
    vid_t* s_B = s_buf + lenA;

    for (int k = lane; k < lenA; k += 32) {
        s_A[k] = A[k];
    }
    for (int k = lane; k < lenB; k += 32) {
        s_B[k] = B[k];
    }
    __syncwarp(0xFFFFFFFFu);

    int k0 = (total * lane) >> 5;
    int k1 = (total * (lane + 1)) >> 5;
    int cnt = k1 - k0;

    vid_t out[8];
    if (cnt > 0) {
        int a_idx = dev_merge_path_partition(s_A, lenA, s_B, lenB, k0);
        int b_idx = k0 - a_idx;
        for (int t = 0; t < cnt; ++t) {
            if (a_idx < lenA && (b_idx >= lenB || s_A[a_idx] <= s_B[b_idx])) {
                out[t] = s_A[a_idx++];
            } else {
                out[t] = s_B[b_idx++];
            }
        }
    }
    __syncwarp(0xFFFFFFFFu);

    for (int t = 0; t < cnt; ++t) {
        s_buf[k0 + t] = out[t];
    }
    __syncwarp(0xFFFFFFFFu);

    for (int k = lane; k < total; k += 32) {
        C[k] = s_buf[k];
    }
    if (lane == 0) {
        if (write_ptr) d_ptr[v] = C;
        d_used[v] = total;
    }
}

__device__ __forceinline__ void tiled_merge_to_out(vid_t v,
                                                   const vid_t* __restrict__ A,
                                                   int lenA,
                                                   const vid_t* __restrict__ B,
                                                   int lenB,
                                                   vid_t* __restrict__ C,
                                                   int total_out,
                                                   int final_deg,
                                                   int tile_idx,
                                                   vid_t* s_mem,
                                                   int* s_bounds,
                                                   vid_t* __restrict__ d_used,
                                                   vid_t** __restrict__ d_ptr,
                                                   bool write_ptr) {
    int tid = threadIdx.x;
    int k_start = tile_idx * 1024;
    int k_end = min(k_start + 1024, total_out);
    int tile_total = k_end - k_start;

    if (lenA == 0) {
        for (int k = tid; k < tile_total; k += 256) {
            C[k_start + k] = B[k_start + k];
        }
        if (tile_idx == 0 && tid == 0) {
            if (write_ptr) d_ptr[v] = C;
            d_used[v] = final_deg;
        }
        return;
    }

    if (tid == 0) {
        s_bounds[0] = (k_start == 0) ? 0 : dev_merge_path_partition(A, lenA, B, lenB, k_start);
    }
    if (tid == 1) {
        s_bounds[1] = (k_end == total_out) ? lenA : dev_merge_path_partition(A, lenA, B, lenB, k_end);
    }
    __syncthreads();

    int a_start = s_bounds[0];
    int a_end = s_bounds[1];
    int b_start = k_start - a_start;
    int b_end = k_end - a_end;
    int tile_lenA = a_end - a_start;
    int tile_lenB = b_end - b_start;

    vid_t* s_A = s_mem;
    vid_t* s_B = s_mem + tile_lenA;

    for (int k = tid; k < tile_lenA; k += 256) {
        s_A[k] = A[a_start + k];
    }
    for (int k = tid; k < tile_lenB; k += 256) {
        s_B[k] = B[b_start + k];
    }
    __syncthreads();

    int lk0 = (tile_total * tid) >> 8;
    int lk1 = (tile_total * (tid + 1)) >> 8;
    int cnt = lk1 - lk0;

    vid_t out[4];
    if (cnt > 0) {
        int la = dev_merge_path_partition(s_A, tile_lenA, s_B, tile_lenB, lk0);
        int lb = lk0 - la;
        for (int t = 0; t < cnt; ++t) {
            if (la < tile_lenA && (lb >= tile_lenB || s_A[la] <= s_B[lb])) {
                out[t] = s_A[la++];
            } else {
                out[t] = s_B[lb++];
            }
        }
    }
    __syncthreads();

    for (int t = 0; t < cnt; ++t) {
        s_mem[lk0 + t] = out[t];
    }
    __syncthreads();

    for (int k = tid; k < tile_total; k += 256) {
        C[k_start + k] = s_mem[k];
    }
    if (tile_idx == 0 && tid == 0) {
        if (write_ptr) d_ptr[v] = C;
        d_used[v] = final_deg;
    }
}

__global__ void merge_realloc_warp_kernel(int num_small_realloc,
                                          const int* __restrict__ small_realloc_idx,
                                          const vid_t* __restrict__ realloc_v,
                                          const vid_t* __restrict__ realloc_old_deg,
                                          const vid_t* __restrict__ realloc_new_deg,
                                          vid_t* const* __restrict__ realloc_old_ptr,
                                          vid_t* const* __restrict__ realloc_new_ptr,
                                          const eoff_t* __restrict__ realloc_batch_offset,
                                          const vid_t* __restrict__ batch_dst,
                                          vid_t* __restrict__ d_used,
                                          vid_t** __restrict__ d_ptr) {
    __shared__ vid_t s_mem[8][256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int w = tid >> 5;
    if (w >= num_small_realloc) return;

    int warp_in_blk = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    int r = small_realloc_idx[w];
    vid_t v = realloc_v[r];
    int lenA = static_cast<int>(realloc_old_deg[r]);
    int total = static_cast<int>(realloc_new_deg[r]);
    int lenB = total - lenA;
    const vid_t* A = realloc_old_ptr[r];
    const vid_t* B = batch_dst + realloc_batch_offset[r];
    vid_t* C = realloc_new_ptr[r];

    warp_merge_to_out(v, A, lenA, B, lenB, C, total, lane,
                      s_mem[warp_in_blk], d_used, d_ptr, true);
}

__global__ void gather_realloc_tiles_kernel(int num_large_realloc,
                                            const int* __restrict__ large_realloc_idx,
                                            const vid_t* __restrict__ realloc_new_deg,
                                            eoff_t* __restrict__ tile_counts) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < num_large_realloc) {
        int r = large_realloc_idx[k];
        tile_counts[k] = (static_cast<int>(realloc_new_deg[r]) + 1023) >> 10;
    } else if (k == num_large_realloc) {
        tile_counts[num_large_realloc] = 0;
    }
}

__global__ void merge_realloc_tiled_kernel(int num_large_realloc,
                                           const eoff_t* __restrict__ tile_offsets,
                                           const int* __restrict__ large_realloc_idx,
                                           const vid_t* __restrict__ realloc_v,
                                           const vid_t* __restrict__ realloc_old_deg,
                                           const vid_t* __restrict__ realloc_new_deg,
                                           vid_t* const* __restrict__ realloc_old_ptr,
                                           vid_t* const* __restrict__ realloc_new_ptr,
                                           const eoff_t* __restrict__ realloc_batch_offset,
                                           const vid_t* __restrict__ batch_dst,
                                           vid_t* __restrict__ d_used,
                                           vid_t** __restrict__ d_ptr) {
    __shared__ int s_meta[2];
    __shared__ int s_bounds[2];
    __shared__ vid_t s_mem[1024];

    if (threadIdx.x == 0) {
        int m = dev_upper_bound_offsets(tile_offsets, num_large_realloc, static_cast<eoff_t>(blockIdx.x));
        s_meta[0] = m;
        s_meta[1] = static_cast<int>(static_cast<eoff_t>(blockIdx.x) - tile_offsets[m]);
    }
    __syncthreads();

    int m = s_meta[0];
    int tile_idx = s_meta[1];
    int r = large_realloc_idx[m];

    vid_t v = realloc_v[r];
    int lenA = static_cast<int>(realloc_old_deg[r]);
    int total = static_cast<int>(realloc_new_deg[r]);
    int lenB = total - lenA;
    const vid_t* A = realloc_old_ptr[r];
    const vid_t* B = batch_dst + realloc_batch_offset[r];
    vid_t* C = realloc_new_ptr[r];

    tiled_merge_to_out(v, A, lenA, B, lenB, C, total, total, tile_idx,
                       s_mem, s_bounds, d_used, d_ptr, true);
}

__global__ void merge_inplace_warp_kernel(int num_small_inplace,
                                          const int* __restrict__ small_inplace_idx,
                                          const vid_t* __restrict__ unique_srcs,
                                          const vid_t* __restrict__ all_old_deg,
                                          const vid_t* __restrict__ all_new_deg,
                                          vid_t* const* __restrict__ all_old_ptr,
                                          const eoff_t* __restrict__ batch_offsets,
                                          const vid_t* __restrict__ batch_dst,
                                          vid_t* __restrict__ d_used) {
    __shared__ vid_t s_mem[8][256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int w = tid >> 5;
    if (w >= num_small_inplace) return;

    int warp_in_blk = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    int pos = small_inplace_idx[w];
    vid_t v = unique_srcs[pos];
    int lenA = static_cast<int>(all_old_deg[pos]);
    int total = static_cast<int>(all_new_deg[pos]);
    int lenB = total - lenA;
    vid_t* adj = all_old_ptr[pos];
    const vid_t* B = batch_dst + batch_offsets[pos];

    warp_merge_to_out(v, adj, lenA, B, lenB, adj, total, lane,
                      s_mem[warp_in_blk], d_used, nullptr, false);
}

// Single-block 512-thread in-place merge for 256 < new_deg <= 4096 with prefix skip
__global__ void merge_inplace_block_kernel(int num_med_inplace,
                                           const int* __restrict__ med_inplace_idx,
                                           const vid_t* __restrict__ unique_srcs,
                                           const vid_t* __restrict__ all_old_deg,
                                           const vid_t* __restrict__ all_new_deg,
                                           vid_t* const* __restrict__ all_old_ptr,
                                           const eoff_t* __restrict__ batch_offsets,
                                           const vid_t* __restrict__ batch_dst,
                                           vid_t* __restrict__ d_used) {
    __shared__ int s_pstart;
    __shared__ vid_t s_mem[4096];

    if (blockIdx.x >= static_cast<unsigned int>(num_med_inplace)) return;
    int pos = med_inplace_idx[blockIdx.x];

    vid_t v = unique_srcs[pos];
    int old_deg = static_cast<int>(all_old_deg[pos]);
    int total = static_cast<int>(all_new_deg[pos]);
    int lenB = total - old_deg;
    vid_t* adj = all_old_ptr[pos];
    const vid_t* B = batch_dst + batch_offsets[pos];

    int tid = threadIdx.x;
    if (tid == 0) {
        s_pstart = static_cast<int>(dev_lower_bound(adj, static_cast<vid_t>(old_deg), B[0]));
    }
    __syncthreads();

    int p_start = s_pstart;
    int tail_lenA = old_deg - p_start;
    int tail_total = tail_lenA + lenB;
    vid_t* adj_tail = adj + p_start;

    if (tail_lenA == 0) {
        for (int k = tid; k < lenB; k += 512) {
            adj_tail[k] = B[k];
        }
        if (tid == 0) d_used[v] = total;
        return;
    }

    vid_t* s_A = s_mem;
    vid_t* s_B = s_mem + tail_lenA;
    for (int k = tid; k < tail_lenA; k += 512) {
        s_A[k] = adj_tail[k];
    }
    for (int k = tid; k < lenB; k += 512) {
        s_B[k] = B[k];
    }
    __syncthreads();

    int lk0 = (tail_total * tid) >> 9;
    int lk1 = (tail_total * (tid + 1)) >> 9;
    int cnt = lk1 - lk0;

    vid_t out[8];
    if (cnt > 0) {
        int la = dev_merge_path_partition(s_A, tail_lenA, s_B, lenB, lk0);
        int lb = lk0 - la;
        for (int t = 0; t < cnt; ++t) {
            if (la < tail_lenA && (lb >= lenB || s_A[la] <= s_B[lb])) {
                out[t] = s_A[la++];
            } else {
                out[t] = s_B[lb++];
            }
        }
    }
    __syncthreads();

    for (int t = 0; t < cnt; ++t) {
        s_mem[lk0 + t] = out[t];
    }
    __syncthreads();

    for (int k = tid; k < tail_total; k += 512) {
        adj_tail[k] = s_mem[k];
    }
    if (tid == 0) d_used[v] = total;
}

__global__ void gather_huge_inplace_prefix_kernel(int num_huge_inplace,
                                                  const int* __restrict__ huge_inplace_idx,
                                                  const vid_t* __restrict__ all_old_deg,
                                                  vid_t* const* __restrict__ all_old_ptr,
                                                  const eoff_t* __restrict__ batch_offsets,
                                                  const vid_t* __restrict__ batch_dst,
                                                  vid_t* __restrict__ huge_tail_old_deg,
                                                  vid_t* __restrict__ huge_p_start) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < num_huge_inplace) {
        int pos = huge_inplace_idx[k];
        vid_t old_deg = all_old_deg[pos];
        const vid_t* adj = all_old_ptr[pos];
        vid_t first_b = batch_dst[batch_offsets[pos]];
        vid_t p_start = dev_lower_bound(adj, old_deg, first_b);
        huge_p_start[k] = p_start;
        huge_tail_old_deg[k] = old_deg - p_start;
    } else if (k == num_huge_inplace) {
        huge_tail_old_deg[num_huge_inplace] = 0;
    }
}

__global__ void copy_huge_adj_to_scratch_kernel(int cnt,
                                                const eoff_t* __restrict__ scratch_offsets,
                                                eoff_t total_chunk_edges,
                                                const int* __restrict__ huge_inplace_idx,
                                                const vid_t* __restrict__ huge_p_start,
                                                vid_t* const* __restrict__ all_old_ptr,
                                                vid_t* __restrict__ d_merge_scratch) {
    int m = 0;
    eoff_t local_idx = 0;
    eoff_t gid = 0;
    if (!binarySearchLB(cnt, scratch_offsets, total_chunk_edges, m, local_idx, gid)) {
        return;
    }

    int pos = huge_inplace_idx[m];
    vid_t p_start = huge_p_start[m];
    d_merge_scratch[gid] = all_old_ptr[pos][p_start + local_idx];
}

__global__ void gather_huge_inplace_tiles_kernel(int cnt,
                                                 const int* __restrict__ huge_inplace_idx,
                                                 const vid_t* __restrict__ all_new_deg,
                                                 const vid_t* __restrict__ huge_p_start,
                                                 eoff_t* __restrict__ tile_counts) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < cnt) {
        int pos = huge_inplace_idx[k];
        int tail_total = static_cast<int>(all_new_deg[pos] - huge_p_start[k]);
        tile_counts[k] = (tail_total + 1023) >> 10;
    } else if (k == cnt) {
        tile_counts[cnt] = 0;
    }
}

__global__ void merge_huge_inplace_tiled_kernel(int cnt,
                                                const eoff_t* __restrict__ tile_offsets,
                                                const int* __restrict__ huge_inplace_idx,
                                                const vid_t* __restrict__ huge_p_start,
                                                const eoff_t* __restrict__ scratch_offsets,
                                                const vid_t* __restrict__ unique_srcs,
                                                const vid_t* __restrict__ all_old_deg,
                                                const vid_t* __restrict__ all_new_deg,
                                                vid_t* const* __restrict__ all_old_ptr,
                                                const eoff_t* __restrict__ batch_offsets,
                                                const vid_t* __restrict__ batch_dst,
                                                const vid_t* __restrict__ d_merge_scratch,
                                                vid_t* __restrict__ d_used) {
    __shared__ int s_meta[2];
    __shared__ int s_bounds[2];
    __shared__ vid_t s_mem[1024];

    if (threadIdx.x == 0) {
        int m = dev_upper_bound_offsets(tile_offsets, cnt, static_cast<eoff_t>(blockIdx.x));
        s_meta[0] = m;
        s_meta[1] = static_cast<int>(static_cast<eoff_t>(blockIdx.x) - tile_offsets[m]);
    }
    __syncthreads();

    int m = s_meta[0];
    int tile_idx = s_meta[1];
    int pos = huge_inplace_idx[m];

    vid_t v = unique_srcs[pos];
    vid_t p_start = huge_p_start[m];
    int old_deg = static_cast<int>(all_old_deg[pos]);
    int new_deg = static_cast<int>(all_new_deg[pos]);
    int lenA = old_deg - static_cast<int>(p_start);
    int lenB = new_deg - old_deg;
    int tail_total = lenA + lenB;
    const vid_t* A = d_merge_scratch + scratch_offsets[m];
    const vid_t* B = batch_dst + batch_offsets[pos];
    vid_t* C = all_old_ptr[pos] + p_start;

    tiled_merge_to_out(v, A, lenA, B, lenB, C, tail_total, new_deg, tile_idx,
                       s_mem, s_bounds, d_used, nullptr, false);
}

inline void launch_huge_inplace_merge_chunk(HornetGraph& graph,
                                            BatchWorkspace& ws,
                                            int c_start,
                                            int cnt,
                                            eoff_t total_chunk_edges,
                                            bool offsets_ready,
                                            const vid_t* d_final_dst) {
    if (cnt <= 0) return;

    int blk = 256;
    const int* d_chunk_idx = ws.huge_inplace_idx_buf() + c_start;
    vid_t* d_chunk_deg = ws.huge_old_deg_buf() + c_start;
    const vid_t* d_chunk_pstart = reinterpret_cast<const vid_t*>(ws.d_keys_in) + (ws.capacity + 1) + c_start;
    eoff_t* d_scratch_offsets = ws.d_graph_offsets;
    eoff_t* d_tile_counts = ws.tile_counts_buf();
    eoff_t* d_tile_offsets = ws.d_move_offsets;

    if (!offsets_ready) {
        cub::DeviceScan::ExclusiveSum(
            ws.d_cub_temp, ws.cub_temp_bytes, d_chunk_deg, d_scratch_offsets, cnt);
        CUDA_CHECK(cudaMemcpy(&d_scratch_offsets[cnt], &total_chunk_edges,
                              sizeof(eoff_t), cudaMemcpyHostToDevice));
    }

    if (total_chunk_edges > 0) {
        eoff_t copy_grid = (total_chunk_edges + blk - 1) / blk;
        copy_huge_adj_to_scratch_kernel<<<static_cast<unsigned int>(copy_grid), blk>>>(
            cnt, d_scratch_offsets, total_chunk_edges, d_chunk_idx,
            d_chunk_pstart, ws.d_all_old_ptr, ws.d_merge_scratch);
        CUDA_CHECK(cudaGetLastError());
    }

    gather_huge_inplace_tiles_kernel<<<(cnt + 1 + blk - 1) / blk, blk>>>(
        cnt, d_chunk_idx, ws.d_all_new_deg, d_chunk_pstart, d_tile_counts);
    CUDA_CHECK(cudaGetLastError());

    cub::DeviceScan::ExclusiveSum(
        ws.d_cub_temp, ws.cub_temp_bytes, d_tile_counts, d_tile_offsets, cnt + 1);

    eoff_t total_tiles = 0;
    CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, &d_tile_offsets[cnt],
                          sizeof(eoff_t), cudaMemcpyDeviceToHost));
    total_tiles = ws.h_pinned_counts[0];

    if (total_tiles > 0) {
        merge_huge_inplace_tiled_kernel<<<static_cast<unsigned int>(total_tiles), 256>>>(
            cnt, d_tile_offsets, d_chunk_idx, d_chunk_pstart, d_scratch_offsets,
            ws.d_unique_srcs, ws.d_all_old_deg, ws.d_all_new_deg,
            ws.d_all_old_ptr, ws.d_batch_offsets, d_final_dst,
            ws.d_merge_scratch, graph.d_used());
        CUDA_CHECK(cudaGetLastError());
    }
}

inline void record_zero_insert_remaining_phases(bool record_build_realloc) {
    if (!HornetProfiler::instance().is_enabled()) return;
    if (record_build_realloc) {
        HornetProfiler::instance().record_phase("insert_build_realloc", 0.0);
    }
    HornetProfiler::instance().record_phase("insert_host_alloc", 0.0);
    HornetProfiler::instance().record_phase("insert_h2d_copy", 0.0);
    HornetProfiler::instance().record_phase("insert_adj_move", 0.0);
    HornetProfiler::instance().record_phase("insert_host_free", 0.0);
    HornetProfiler::instance().record_phase("insert_append", 0.0);
}

inline void batch_insert(HornetGraph& graph,
                         BatchWorkspace& ws,
                         bool remove_graph_dups) {
    ScopedPhaseTimer total_timer("insert_total", true);
    int batch_size = ws.batch_size;
    if (batch_size == 0) return;

    const bool prof = HornetProfiler::instance().is_enabled();
    int blk = 256;
    int vid_bits = (graph.nV() > 1) ? hornet_common::ceil_log2(graph.nV()) : 1;
    if (vid_bits < 1) vid_bits = 1;
    if (vid_bits > 32) vid_bits = 32;

    int num_unique = 0;
    int final_count = 0;
    vid_t* d_final_dst = nullptr;

    // -------------------------------------------------------------------------
    // Phase 1: Preprocessing (Sort, Deduplicate, Batch CSR)
    // -------------------------------------------------------------------------
    {
        ScopedPhaseTimer prep_timer("insert_preprocess", true);
        int uniq_count = 0;

        if (batch_size <= 256) {
            // DS-3: Single-Block Shared-Memory Preprocessing Fast Path
            preprocess_small_batch_kernel<<<1, 256>>>(
                batch_size, ws.d_raw_src, ws.d_raw_dst, vid_bits,
                ws.d_src_uniq, ws.d_dst_uniq, ws.d_unique_srcs,
                ws.d_batch_deg, ws.d_batch_offsets, ws.d_scalar_count);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, ws.d_scalar_count,
                                  2 * sizeof(int), cudaMemcpyDeviceToHost));
            uniq_count = ws.h_pinned_counts[0];
            num_unique = ws.h_pinned_counts[1];
        } else {
            // DS-2: Tight Bit-Packing & Partial end_bit
            pack_edges_kernel<<<(batch_size + blk - 1) / blk, blk>>>(
                batch_size, ws.d_raw_src, ws.d_raw_dst, ws.d_keys_in, vid_bits);
            CUDA_CHECK(cudaGetLastError());

            int sort_bits = std::min(64, 2 * vid_bits);
            cub::DeviceRadixSort::SortKeys(
                ws.d_cub_temp, ws.cub_temp_bytes, ws.d_keys_in, ws.d_keys_sorted,
                batch_size, 0, sort_bits);

            cub::DeviceSelect::Unique(
                ws.d_cub_temp, ws.cub_temp_bytes, ws.d_keys_sorted, ws.d_keys_uniq,
                ws.d_scalar_count, batch_size);

            CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, ws.d_scalar_count,
                                  sizeof(int), cudaMemcpyDeviceToHost));
            uniq_count = ws.h_pinned_counts[0];
            if (uniq_count == 0) {
                record_zero_insert_remaining_phases(true);
                return;
            }

            unpack_edges_kernel<<<(uniq_count + blk - 1) / blk, blk>>>(
                uniq_count, ws.d_keys_uniq, ws.d_src_uniq, ws.d_dst_uniq, vid_bits);
            CUDA_CHECK(cudaGetLastError());

            cub::DeviceRunLengthEncode::Encode(
                ws.d_cub_temp, ws.cub_temp_bytes, ws.d_src_uniq, ws.d_unique_srcs,
                ws.d_batch_deg, ws.d_scalar_count, uniq_count);
            CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, ws.d_scalar_count,
                                  sizeof(int), cudaMemcpyDeviceToHost));
            num_unique = ws.h_pinned_counts[0];
            if (num_unique == 0) {
                record_zero_insert_remaining_phases(true);
                return;
            }

            CUDA_CHECK(cudaMemsetAsync(&ws.d_batch_deg[num_unique], 0, sizeof(int)));
            cub::DeviceScan::ExclusiveSum(
                ws.d_cub_temp, ws.cub_temp_bytes, ws.d_batch_deg, ws.d_batch_offsets,
                num_unique + 1);
        }

        if (uniq_count == 0 || num_unique == 0) {
            record_zero_insert_remaining_phases(true);
            return;
        }

        vid_t* d_final_src = ws.d_src_uniq;
        d_final_dst = ws.d_dst_uniq;
        final_count = uniq_count;

        if (remove_graph_dups) {
            // DS-5: O(batch_deg * log old_deg) cross-duplicate lookup over sorted adj
            check_cross_dups_sorted_kernel<<<(uniq_count + blk - 1) / blk, blk>>>(
                uniq_count, ws.d_src_uniq, ws.d_dst_uniq,
                graph.d_used(), graph.d_ptr(), ws.d_keep);
            CUDA_CHECK(cudaGetLastError());

            // DS-2: 1 Flagged pass on counting_iterator
            cub::DeviceSelect::Flagged(
                ws.d_cub_temp, ws.cub_temp_bytes, thrust::counting_iterator<int>(0),
                ws.d_keep, ws.d_selected_idx, ws.d_scalar_count, uniq_count);

            CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, ws.d_scalar_count,
                                  sizeof(int), cudaMemcpyDeviceToHost));
            final_count = ws.h_pinned_counts[0];
            if (final_count == 0) {
                record_zero_insert_remaining_phases(true);
                return;
            }

            if (final_count < uniq_count) {
                gather_selected_edges_kernel<<<(final_count + blk - 1) / blk, blk>>>(
                    final_count, ws.d_selected_idx, ws.d_src_uniq, ws.d_dst_uniq,
                    ws.d_filt_src, ws.d_filt_dst);
                CUDA_CHECK(cudaGetLastError());

                d_final_src = ws.d_filt_src;
                d_final_dst = ws.d_filt_dst;

                cub::DeviceRunLengthEncode::Encode(
                    ws.d_cub_temp, ws.cub_temp_bytes, d_final_src, ws.d_unique_srcs,
                    ws.d_batch_deg, ws.d_scalar_count, final_count);
                CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, ws.d_scalar_count,
                                      sizeof(int), cudaMemcpyDeviceToHost));
                num_unique = ws.h_pinned_counts[0];
                CUDA_CHECK(cudaMemsetAsync(&ws.d_batch_deg[num_unique], 0, sizeof(int)));
                cub::DeviceScan::ExclusiveSum(
                    ws.d_cub_temp, ws.cub_temp_bytes, ws.d_batch_deg, ws.d_batch_offsets,
                    num_unique + 1);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Phase 2-4: Build Realloc List, Pre-Check Deficit, GPU BitTree Allocation
    // -------------------------------------------------------------------------
    int num_realloc = 0;
    int num_small_inplace = 0;
    int num_med_inplace = 0;
    int num_huge_inplace = 0;
    int num_small_realloc = 0;
    int num_large_realloc = 0;

    {
        double build_realloc_ms = 0.0;
        std::chrono::high_resolution_clock::time_point br_t0;
        bool br_nvtx = false;
        if (prof) {
            if (HornetProfiler::instance().is_nvtx_enabled()) {
                nvtxRangePushA("insert_build_realloc");
                br_nvtx = true;
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            br_t0 = std::chrono::high_resolution_clock::now();
        }

        CUDA_CHECK(cudaMemsetAsync(ws.d_scalar_count, 0, 3 * sizeof(int)));
        check_insert_realloc_kernel<<<(num_unique + blk - 1) / blk, blk>>>(
            num_unique, ws.d_unique_srcs, ws.d_batch_deg, graph.d_used(), graph.d_ptr(),
            ws.d_need_realloc, ws.d_all_old_deg, ws.d_all_new_deg, ws.d_all_old_ptr,
            ws.small_inplace_idx_buf(), ws.med_inplace_idx_buf(), ws.huge_inplace_idx_buf(),
            &ws.d_scalar_count[0], &ws.d_scalar_count[1], &ws.d_scalar_count[2]);
        CUDA_CHECK(cudaGetLastError());

        // DS-2: 1 Flagged pass on counting_iterator + 1 gather kernel
        cub::DeviceSelect::Flagged(
            ws.d_cub_temp, ws.cub_temp_bytes, thrust::counting_iterator<int>(0),
            ws.d_need_realloc, ws.d_selected_idx, &ws.d_scalar_count[3], num_unique);

        CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, ws.d_scalar_count,
                              4 * sizeof(int), cudaMemcpyDeviceToHost));
        num_small_inplace = ws.h_pinned_counts[0];
        num_med_inplace = ws.h_pinned_counts[1];
        num_huge_inplace = ws.h_pinned_counts[2];
        num_realloc = ws.h_pinned_counts[3];

        if (num_realloc > 0) {
            CUDA_CHECK(cudaMemsetAsync(&ws.d_scalar_count[4], 0, 2 * sizeof(int)));
            gather_insert_realloc_metadata_kernel<<<(num_realloc + blk - 1) / blk, blk>>>(
                num_realloc, ws.d_selected_idx, ws.d_unique_srcs,
                ws.d_all_old_deg, ws.d_all_new_deg, ws.d_all_old_ptr, ws.d_batch_offsets,
                ws.d_realloc_v, ws.d_realloc_old_deg, ws.d_realloc_new_deg,
                ws.d_realloc_old_ptr, ws.realloc_batch_offset_buf(),
                ws.small_realloc_idx_buf(), ws.large_realloc_idx_buf(),
                &ws.d_scalar_count[4], &ws.d_scalar_count[5]);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpyAsync(&ws.h_pinned_counts[BatchWorkspace::PINNED_REALLOC_COUNTS_OFFSET],
                                       &ws.d_scalar_count[4], 2 * sizeof(int),
                                       cudaMemcpyDeviceToHost, 0));

            // DS-7: GPU bin-grouped pre-check, optional fallback, and GPU BitTree allocation
            ds7_gpu_allocate_blocks(
                graph.memory_manager(), ws.ds7_alloc_view(),
                num_realloc, ws.d_realloc_new_deg, ws.d_realloc_new_ptr,
                prof, "insert_gpu_alloc", "insert_host_alloc", "insert_h2d_copy",
                &br_t0, &br_nvtx, &build_realloc_ms);

            num_small_realloc = ws.h_pinned_counts[BatchWorkspace::PINNED_REALLOC_COUNTS_OFFSET];
            num_large_realloc = ws.h_pinned_counts[BatchWorkspace::PINNED_REALLOC_COUNTS_OFFSET + 1];
        } else if (prof) {
            HornetProfiler::instance().record_phase("insert_host_alloc", 0.0);
            HornetProfiler::instance().record_phase("insert_h2d_copy", 0.0);
            HornetProfiler::instance().record_phase("insert_adj_move", 0.0);
            HornetProfiler::instance().record_phase("insert_host_free", 0.0);
        }

        if (prof) {
            if (br_nvtx) {
                cudaDeviceSynchronize();
                auto br_t1 = std::chrono::high_resolution_clock::now();
                build_realloc_ms += std::chrono::duration<double, std::milli>(br_t1 - br_t0).count();
                nvtxRangePop();
            }
            HornetProfiler::instance().record_phase("insert_build_realloc", build_realloc_ms);
        }
    }

    if (num_realloc > 0) {
        // ---------------------------------------------------------------------
        // Phase 5-6: DS-5 Parallel Merge + Separate DS-7 GPU Block Reclamation
        // ---------------------------------------------------------------------
        {
            ScopedPhaseTimer move_timer("insert_adj_move", true);

            if (num_small_realloc > 0) {
                int warps_per_blk = blk / 32;
                int warp_grid = (num_small_realloc + warps_per_blk - 1) / warps_per_blk;
                merge_realloc_warp_kernel<<<warp_grid, blk>>>(
                    num_small_realloc, ws.small_realloc_idx_buf(),
                    ws.d_realloc_v, ws.d_realloc_old_deg, ws.d_realloc_new_deg,
                    ws.d_realloc_old_ptr, ws.d_realloc_new_ptr,
                    ws.realloc_batch_offset_buf(), d_final_dst,
                    graph.d_used(), graph.d_ptr());
                CUDA_CHECK(cudaGetLastError());
            }

            if (num_large_realloc > 0) {
                eoff_t* d_tile_counts = ws.d_graph_offsets;
                eoff_t* d_tile_offsets = ws.d_move_offsets;

                gather_realloc_tiles_kernel<<<(num_large_realloc + 1 + blk - 1) / blk, blk>>>(
                    num_large_realloc, ws.large_realloc_idx_buf(),
                    ws.d_realloc_new_deg, d_tile_counts);
                CUDA_CHECK(cudaGetLastError());

                cub::DeviceScan::ExclusiveSum(
                    ws.d_cub_temp, ws.cub_temp_bytes, d_tile_counts,
                    d_tile_offsets, num_large_realloc + 1);

                eoff_t total_tiles = 0;
                CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, &d_tile_offsets[num_large_realloc],
                                      sizeof(eoff_t), cudaMemcpyDeviceToHost));
                total_tiles = ws.h_pinned_counts[0];

                if (total_tiles > 0) {
                    merge_realloc_tiled_kernel<<<static_cast<unsigned int>(total_tiles), 256>>>(
                        num_large_realloc, d_tile_offsets, ws.large_realloc_idx_buf(),
                        ws.d_realloc_v, ws.d_realloc_old_deg, ws.d_realloc_new_deg,
                        ws.d_realloc_old_ptr, ws.d_realloc_new_ptr,
                        ws.realloc_batch_offset_buf(), d_final_dst,
                        graph.d_used(), graph.d_ptr());
                    CUDA_CHECK(cudaGetLastError());
                }
            }

            // Separate GPU reclamation phase after move/merge kernels complete
            if (prof && HornetProfiler::instance().is_nvtx_enabled()) {
                nvtxRangePushA("insert_gpu_reclaim");
            }
            ds7_gpu_reclaim_blocks(
                graph.memory_manager(), num_realloc, ws.d_realloc_old_deg, ws.d_realloc_old_ptr);
            if (prof && HornetProfiler::instance().is_nvtx_enabled()) {
                nvtxRangePop();
            }
        }

        if (prof) {
            HornetProfiler::instance().record_phase("insert_host_free", 0.0);
        }
    }

    // -------------------------------------------------------------------------
    // Phase 7: DS-5 Parallel Merge on Non-Reallocated Adjacency Lists
    // -------------------------------------------------------------------------
    if (num_small_inplace > 0 || num_med_inplace > 0 || num_huge_inplace > 0) {
        ScopedPhaseTimer append_timer("insert_append", true);

        if (num_small_inplace > 0) {
            int warps_per_blk = blk / 32;
            int warp_grid = (num_small_inplace + warps_per_blk - 1) / warps_per_blk;
            merge_inplace_warp_kernel<<<warp_grid, blk>>>(
                num_small_inplace, ws.small_inplace_idx_buf(),
                ws.d_unique_srcs, ws.d_all_old_deg, ws.d_all_new_deg,
                ws.d_all_old_ptr, ws.d_batch_offsets, d_final_dst, graph.d_used());
            CUDA_CHECK(cudaGetLastError());
        }

        if (num_med_inplace > 0) {
            merge_inplace_block_kernel<<<num_med_inplace, 512>>>(
                num_med_inplace, ws.med_inplace_idx_buf(),
                ws.d_unique_srcs, ws.d_all_old_deg, ws.d_all_new_deg,
                ws.d_all_old_ptr, ws.d_batch_offsets, d_final_dst, graph.d_used());
            CUDA_CHECK(cudaGetLastError());
        }

        if (num_huge_inplace > 0) {
            vid_t* d_huge_tail_deg = ws.huge_old_deg_buf();
            vid_t* d_huge_p_start = reinterpret_cast<vid_t*>(ws.d_keys_in) + (ws.capacity + 1);
            gather_huge_inplace_prefix_kernel<<<(num_huge_inplace + 1 + blk - 1) / blk, blk>>>(
                num_huge_inplace, ws.huge_inplace_idx_buf(), ws.d_all_old_deg,
                ws.d_all_old_ptr, ws.d_batch_offsets, d_final_dst,
                d_huge_tail_deg, d_huge_p_start);
            CUDA_CHECK(cudaGetLastError());

            cub::DeviceScan::ExclusiveSum(
                ws.d_cub_temp, ws.cub_temp_bytes, d_huge_tail_deg,
                ws.d_graph_offsets, num_huge_inplace + 1);

            eoff_t total_huge_tail_edges = 0;
            CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, &ws.d_graph_offsets[num_huge_inplace],
                                  sizeof(eoff_t), cudaMemcpyDeviceToHost));
            total_huge_tail_edges = ws.h_pinned_counts[0];

            if (total_huge_tail_edges <= static_cast<eoff_t>(BatchWorkspace::MAX_SCRATCH_EDGES)) {
                if (total_huge_tail_edges > 0) {
                    ws.ensure_scratch_capacity(static_cast<size_t>(total_huge_tail_edges));
                }
                launch_huge_inplace_merge_chunk(
                    graph, ws, 0, num_huge_inplace, total_huge_tail_edges, true, d_final_dst);
            } else {
                std::vector<vid_t> h_deg(num_huge_inplace);
                CUDA_CHECK(cudaMemcpy(h_deg.data(), d_huge_tail_deg,
                                      num_huge_inplace * sizeof(vid_t), cudaMemcpyDeviceToHost));
                vid_t max_single = *std::max_element(h_deg.begin(), h_deg.end());
                size_t cap = std::max(BatchWorkspace::MAX_SCRATCH_EDGES,
                                      static_cast<size_t>(max_single));
                ws.ensure_scratch_capacity(cap);

                int c_start = 0;
                while (c_start < num_huge_inplace) {
                    size_t sum = 0;
                    int c_end = c_start;
                    while (c_end < num_huge_inplace) {
                        size_t d = static_cast<size_t>(h_deg[c_end]);
                        if (c_end > c_start && sum + d > cap) break;
                        sum += d;
                        ++c_end;
                    }
                    launch_huge_inplace_merge_chunk(
                        graph, ws, c_start, c_end - c_start,
                        static_cast<eoff_t>(sum), false, d_final_dst);
                    c_start = c_end;
                }
            }
        }
    } else if (prof) {
        HornetProfiler::instance().record_phase("insert_append", 0.0);
    }
}

} // namespace improved
