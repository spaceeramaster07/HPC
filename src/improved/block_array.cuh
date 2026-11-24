#pragma once

#include "bit_tree.cuh"
#include "types.cuh"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace improved {

using namespace hornet_common;

/**
 * @brief DS-1 mempool-backed async device allocation helper with OOM trim retry.
 */
inline void cuda_malloc_async_with_pool(void** ptr, size_t bytes, cudaStream_t stream = 0) {
    static std::once_flag mempool_configured;
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaMemPool_t pool;
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, dev));
    std::call_once(mempool_configured, [&pool]() {
        uint64_t threshold = 256ULL << 20;
        CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
    });

    cudaError_t alloc_result = cudaMallocAsync(ptr, bytes, stream);
    if (alloc_result == cudaErrorMemoryAllocation) {
        CUDA_CHECK(cudaMemPoolTrimTo(pool, 0));
        alloc_result = cudaMallocAsync(ptr, bytes, stream);
    }
    CUDA_CHECK(alloc_result);
}

/**
 * @brief Physical BlockArray (Paper Section III-A, Fig. 1) + DS-7 GPU BitTree.
 *
 * Contiguous GPU slab of `num_blocks` equal-sized blocks of `2^log_bsize` edges.
 * Uses stream-ordered cudaMallocAsync/cudaFreeAsync (256 MiB mempool) and
 * mirrors its BitTree (Vec-Tree) into GPU VRAM (`d_bittree_`).
 */
class BlockArray {
public:
    BlockArray(int log_bsize, size_t max_edges_per_ba)
        : log_bsize_(log_bsize),
          block_size_(1ULL << log_bsize),
          num_blocks_(static_cast<int>(std::max<size_t>(1, max_edges_per_ba / (1ULL << log_bsize)))),
          total_edges_(static_cast<size_t>(num_blocks_) * block_size_),
          bit_tree_(num_blocks_) {
        cuda_malloc_async_with_pool(
            reinterpret_cast<void**>(&d_edges_), total_edges_ * sizeof(vid_t), 0);
        cuda_malloc_async_with_pool(
            reinterpret_cast<void**>(&d_bittree_), bit_tree_.total_bytes(), 0);
    }

    ~BlockArray() {
        if (d_bittree_) {
            cudaFreeAsync(d_bittree_, 0);
            d_bittree_ = nullptr;
        }
        if (d_edges_) {
            cudaFreeAsync(d_edges_, 0);
            d_edges_ = nullptr;
        }
    }

    BlockArray(const BlockArray&) = delete;
    BlockArray& operator=(const BlockArray&) = delete;

    void enable_host_staging(int staged_blocks = -1) {
        if (!h_staging_) {
            int blocks = (staged_blocks >= 0 && staged_blocks < num_blocks_) ? staged_blocks : num_blocks_;
            staged_edges_ = static_cast<size_t>(blocks) * block_size_;
            h_staging_.reset(new vid_t[staged_edges_]);
        }
    }

    void stage_block_data(int block_idx, const vid_t* src, int count) {
        size_t offset = static_cast<size_t>(block_idx) * block_size_;
        std::copy(src, src + count, h_staging_.get() + offset);
    }

    void flush_staging_to_device() {
        if (h_staging_) {
            if (staged_edges_ > 0) {
                CUDA_CHECK(cudaMemcpy(d_edges_, h_staging_.get(),
                                      staged_edges_ * sizeof(vid_t), cudaMemcpyHostToDevice));
            }
            h_staging_.reset();
            staged_edges_ = 0;
        }
    }

    void upload_bittree_to_device(cudaStream_t stream = 0) const {
        if (!d_bittree_ || bit_tree_.total_bytes() == 0) return;
        if (bit_tree_.free_count() == 0) {
            CUDA_CHECK(cudaMemsetAsync(d_bittree_, 0, bit_tree_.total_bytes(), stream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(d_bittree_, bit_tree_.words_data(),
                                       bit_tree_.total_bytes(), cudaMemcpyHostToDevice, stream));
        }
    }

    // Host BitTree reflects only the initial slab state uploaded to d_bittree_;
    // runtime allocations and frees operate exclusively on GPU d_bittree_.
    int free_count() const { return bit_tree_.free_count(); }
    int num_blocks() const { return num_blocks_; }
    void mark_prefix_used(int used_blocks) { bit_tree_.mark_prefix_used(used_blocks); }
    int log_bsize() const { return log_bsize_; }
    size_t block_size() const { return block_size_; }
    size_t total_edges() const { return total_edges_; }
    size_t total_bytes() const { return total_edges_ * sizeof(vid_t); }
    vid_t* device_base() const { return d_edges_; }
    uint64_t* device_bittree() const { return d_bittree_; }
    const BitTree& bit_tree() const { return bit_tree_; }
    vid_t* ptr_of_block(int block_idx) const {
        return d_edges_ + static_cast<size_t>(block_idx) * block_size_;
    }

    bool owns_ptr(const vid_t* ptr) const {
        return ptr >= d_edges_ && ptr < (d_edges_ + total_edges_);
    }

private:
    int log_bsize_;
    size_t block_size_;
    int num_blocks_;
    size_t total_edges_;
    BitTree bit_tree_;
    vid_t* d_edges_ = nullptr;
    uint64_t* d_bittree_ = nullptr;
    std::unique_ptr<vid_t[]> h_staging_;
    size_t staged_edges_ = 0;
};

} // namespace improved
