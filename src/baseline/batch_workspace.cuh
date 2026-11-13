#pragma once

#include "hornet_graph.cuh"
#include "types.cuh"

#include <cub/cub.cuh>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace baseline {

// Pack/unpack (src, dst) into 64-bit sortable keys
__global__ void pack_edges_kernel(int n,
                                  const vid_t* __restrict__ src,
                                  const vid_t* __restrict__ dst,
                                  uint64_t* __restrict__ keys) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        keys[i] = (static_cast<uint64_t>(static_cast<uint32_t>(src[i])) << 32) |
                  static_cast<uint32_t>(dst[i]);
    }
}

__global__ void unpack_edges_kernel(int n,
                                    const uint64_t* __restrict__ keys,
                                    vid_t* __restrict__ src,
                                    vid_t* __restrict__ dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint64_t k = keys[i];
        src[i] = static_cast<vid_t>(static_cast<uint32_t>(k >> 32));
        dst[i] = static_cast<vid_t>(static_cast<uint32_t>(k & 0xFFFFFFFFULL));
    }
}

// 1 warp per batch edge: checks if dst already exists in d_ptr[src][0..d_used[src]-1]

__global__ void adj_move_balanced_kernel(int num_realloc,
                                         const eoff_t* __restrict__ move_offsets,
                                         eoff_t total_moves,
                                         const vid_t* __restrict__ realloc_v,
                                         vid_t* const* __restrict__ old_ptrs,
                                         vid_t* const* __restrict__ new_ptrs,
                                         vid_t** __restrict__ d_ptr) {
    eoff_t gid = static_cast<eoff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_moves) return;

    int lo = 0;
    int hi = num_realloc - 1;
    while (lo <= hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (move_offsets[mid + 1] <= gid) {
            lo = mid + 1;
        } else if (move_offsets[mid] > gid) {
            hi = mid - 1;
        } else {
            lo = mid;
            break;
        }
    }
    int r = lo;
    eoff_t local_idx = gid - move_offsets[r];
    new_ptrs[r][local_idx] = old_ptrs[r][local_idx];

    if (local_idx == 0) {
        d_ptr[realloc_v[r]] = new_ptrs[r];
    }
}


struct BatchWorkspace {
    int capacity = 0;
    int batch_size = 0;

    vid_t* d_raw_src = nullptr;
    vid_t* d_raw_dst = nullptr;
    uint64_t* d_keys_in = nullptr;
    uint64_t* d_keys_sorted = nullptr;
    uint64_t* d_keys_uniq = nullptr;
    vid_t* d_src_uniq = nullptr;
    vid_t* d_dst_uniq = nullptr;
    uint8_t* d_keep = nullptr;
    vid_t* d_filt_src = nullptr;
    vid_t* d_filt_dst = nullptr;

    vid_t* d_unique_srcs = nullptr;
    int* d_batch_deg = nullptr;
    eoff_t* d_batch_offsets = nullptr;

    uint8_t* d_need_realloc = nullptr;
    uint8_t* d_need_free_only = nullptr;
    vid_t* d_all_old_deg = nullptr;
    vid_t* d_all_new_deg = nullptr;
    vid_t** d_all_old_ptr = nullptr;

    vid_t* d_realloc_v = nullptr;
    vid_t* d_realloc_old_deg = nullptr;
    vid_t* d_realloc_new_deg = nullptr;
    vid_t** d_realloc_old_ptr = nullptr;
    vid_t** d_realloc_new_ptr = nullptr;
    eoff_t* d_move_offsets = nullptr;

    int* d_scalar_count = nullptr;
    void* d_cub_temp = nullptr;
    size_t cub_temp_bytes = 0;

    ~BatchWorkspace() { free_all(); }

    void free_all() {
        cudaFree(d_raw_src);
        cudaFree(d_raw_dst);
        cudaFree(d_keys_in);
        cudaFree(d_keys_sorted);
        cudaFree(d_keys_uniq);
        cudaFree(d_src_uniq);
        cudaFree(d_dst_uniq);
        cudaFree(d_keep);
        cudaFree(d_filt_src);
        cudaFree(d_filt_dst);
        cudaFree(d_unique_srcs);
        cudaFree(d_batch_deg);
        cudaFree(d_batch_offsets);
        cudaFree(d_need_realloc);
        cudaFree(d_need_free_only);
        cudaFree(d_all_old_deg);
        cudaFree(d_all_new_deg);
        cudaFree(d_all_old_ptr);
        cudaFree(d_realloc_v);
        cudaFree(d_realloc_old_deg);
        cudaFree(d_realloc_new_deg);
        cudaFree(d_realloc_old_ptr);
        cudaFree(d_realloc_new_ptr);
        cudaFree(d_move_offsets);
        cudaFree(d_scalar_count);
        cudaFree(d_cub_temp);

        d_raw_src = d_raw_dst = d_src_uniq = d_dst_uniq = nullptr;
        d_filt_src = d_filt_dst = d_unique_srcs = nullptr;
        d_keys_in = d_keys_sorted = d_keys_uniq = nullptr;
        d_keep = d_need_realloc = d_need_free_only = nullptr;
        d_batch_deg = nullptr;
        d_batch_offsets = d_move_offsets = nullptr;
        d_all_old_deg = d_all_new_deg = d_realloc_v = d_realloc_old_deg = d_realloc_new_deg = nullptr;
        d_all_old_ptr = d_realloc_old_ptr = d_realloc_new_ptr = nullptr;
        d_scalar_count = nullptr;
        d_cub_temp = nullptr;
        capacity = 0;
        cub_temp_bytes = 0;
    }

    void ensure_capacity(int n) {
        if (n <= capacity) return;
        free_all();
        capacity = std::max(n, 1);

        CUDA_CHECK(cudaMalloc(&d_raw_src, capacity * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_raw_dst, capacity * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_keys_in, capacity * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_keys_sorted, capacity * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_keys_uniq, capacity * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_src_uniq, capacity * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_dst_uniq, capacity * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_keep, capacity * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_filt_src, capacity * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_filt_dst, capacity * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_unique_srcs, capacity * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_batch_deg, capacity * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_batch_offsets, (capacity + 1) * sizeof(eoff_t)));
        CUDA_CHECK(cudaMalloc(&d_need_realloc, capacity * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_need_free_only, capacity * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_all_old_deg, capacity * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_all_new_deg, capacity * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_all_old_ptr, capacity * sizeof(vid_t*)));
        CUDA_CHECK(cudaMalloc(&d_realloc_v, capacity * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_realloc_old_deg, capacity * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_realloc_new_deg, capacity * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_realloc_old_ptr, capacity * sizeof(vid_t*)));
        CUDA_CHECK(cudaMalloc(&d_realloc_new_ptr, capacity * sizeof(vid_t*)));
        CUDA_CHECK(cudaMalloc(&d_move_offsets, (capacity + 1) * sizeof(eoff_t)));
        CUDA_CHECK(cudaMalloc(&d_scalar_count, sizeof(int)));

        // Pre-compute max CUB temporary storage needed across all operations
        size_t req = 0, tmp = 0;
        cub::DeviceRadixSort::SortKeys(nullptr, tmp, d_keys_in, d_keys_sorted, capacity);
        req = std::max(req, tmp);
        cub::DeviceSelect::Unique(nullptr, tmp, d_keys_sorted, d_keys_uniq, d_scalar_count, capacity);
        req = std::max(req, tmp);
        cub::DeviceSelect::Flagged(nullptr, tmp, d_all_old_ptr, d_need_realloc, d_realloc_old_ptr, d_scalar_count, capacity);
        req = std::max(req, tmp);
        cub::DeviceSelect::Flagged(nullptr, tmp, d_src_uniq, d_keep, d_filt_src, d_scalar_count, capacity);
        req = std::max(req, tmp);
        cub::DeviceRunLengthEncode::Encode(nullptr, tmp, d_filt_src, d_unique_srcs, d_batch_deg, d_scalar_count, capacity);
        req = std::max(req, tmp);
        cub::DeviceScan::ExclusiveSum(nullptr, tmp, d_batch_deg, d_batch_offsets, capacity);
        req = std::max(req, tmp);

        cub_temp_bytes = std::max<size_t>(req, 256);
        CUDA_CHECK(cudaMalloc(&d_cub_temp, cub_temp_bytes));
    }

    void prepare(const std::vector<vid_t>& h_src, const std::vector<vid_t>& h_dst) {
        batch_size = static_cast<int>(h_src.size());
        ensure_capacity(batch_size);
        if (batch_size > 0) {
            CUDA_CHECK(cudaMemcpy(d_raw_src, h_src.data(), batch_size * sizeof(vid_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_raw_dst, h_dst.data(), batch_size * sizeof(vid_t), cudaMemcpyHostToDevice));
        }
    }
};

/**
 * @brief Algorithm 1: Dynamic Batch Insertion (Paper Section III-E).
 */

} // namespace baseline
