#pragma once

#include "bit_tree.cuh"
#include "types.cuh"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

namespace baseline {

using namespace hornet_common;

/**
 * @brief Physical BlockArray (Paper Section III-A, Fig. 1).
 *
 * Contiguous GPU slab of `num_blocks` equal-sized blocks of `2^log_bsize` edges.
 * Uses standard cudaMalloc/cudaFree and a host-side BitTree (Vec-Tree).
 */
class BlockArray {
public:
    BlockArray(int log_bsize, size_t max_edges_per_ba)
        : log_bsize_(log_bsize),
          block_size_(1ULL << log_bsize),
          num_blocks_(static_cast<int>(std::max<size_t>(1, max_edges_per_ba / (1ULL << log_bsize)))),
          total_edges_(static_cast<size_t>(num_blocks_) * block_size_),
          bit_tree_(num_blocks_) {
        CUDA_CHECK(cudaMalloc(&d_edges_, total_edges_ * sizeof(vid_t)));
    }

    ~BlockArray() {
        if (d_edges_) {
            cudaFree(d_edges_);
            d_edges_ = nullptr;
        }
    }

    BlockArray(const BlockArray&) = delete;
    BlockArray& operator=(const BlockArray&) = delete;

    void enable_host_staging() {
        if (h_staging_.empty()) {
            h_staging_.resize(total_edges_, 0);
        }
    }

    void stage_block_data(int block_idx, const vid_t* src, int count) {
        size_t offset = static_cast<size_t>(block_idx) * block_size_;
        std::copy(src, src + count, h_staging_.data() + offset);
    }

    void flush_staging_to_device() {
        if (!h_staging_.empty()) {
            CUDA_CHECK(cudaMemcpy(d_edges_, h_staging_.data(),
                                  total_edges_ * sizeof(vid_t), cudaMemcpyHostToDevice));
            std::vector<vid_t>().swap(h_staging_);
        }
    }

    bool has_free() const { return bit_tree_.has_free(); }
    int free_count() const { return bit_tree_.free_count(); }
    int num_blocks() const { return num_blocks_; }
    int used_blocks() const { return bit_tree_.used_count(); }
    int log_bsize() const { return log_bsize_; }
    size_t block_size() const { return block_size_; }
    size_t total_edges() const { return total_edges_; }
    size_t total_bytes() const { return total_edges_ * sizeof(vid_t); }
    vid_t* device_base() const { return d_edges_; }

    bool owns_ptr(const vid_t* ptr) const {
        return ptr >= d_edges_ && ptr < (d_edges_ + total_edges_);
    }

    vid_t* allocate_block(int* out_block_idx = nullptr) {
        int bidx = bit_tree_.allocate_block();
        if (bidx < 0) return nullptr;
        if (out_block_idx) *out_block_idx = bidx;
        return d_edges_ + static_cast<size_t>(bidx) * block_size_;
    }

    void free_block_by_ptr(const vid_t* ptr) {
        assert(owns_ptr(ptr));
        size_t offset = static_cast<size_t>(ptr - d_edges_);
        int bidx = static_cast<int>(offset >> log_bsize_);
        bit_tree_.free_block(bidx);
    }

private:
    int log_bsize_;
    size_t block_size_;
    int num_blocks_;
    size_t total_edges_;
    BitTree bit_tree_;
    vid_t* d_edges_ = nullptr;
    std::vector<vid_t> h_staging_;
};

} // namespace baseline
