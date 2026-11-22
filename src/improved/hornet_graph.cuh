#pragma once

#include "graph_io.hpp"
#include "memory_manager.cuh"
#include "types.cuh"

#include <numeric>
#include <vector>

namespace improved {

__global__ void scatter_hornet_to_coo_kernel(vid_t nV,
                                             const vid_t* __restrict__ d_used,
                                             vid_t* const* __restrict__ d_ptr,
                                             const eoff_t* __restrict__ d_offsets,
                                             vid_t* __restrict__ out_src,
                                             vid_t* __restrict__ out_dst) {
    vid_t v = static_cast<vid_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= nV) return;

    vid_t deg = d_used[v];
    if (deg <= 0) return;

    const vid_t* adj = d_ptr[v];
    eoff_t base = d_offsets[v];
    for (vid_t i = 0; i < deg; ++i) {
        out_src[base + i] = v;
        out_dst[base + i] = adj[i];
    }
}

/**
 * @brief Hornet Graph (Tier 1 User Interface + Tier 2 B+Tree/BlockArray Manager).
 *
 * Tier 1 exposes flat device arrays `d_used[nV]` and `d_ptr[nV]`.
 * Tier 2 manages physical BlockArrays via MemoryManager.
 */
class HornetGraph {
public:
    explicit HornetGraph(const CSRGraph& csr, size_t max_edges_per_ba = (1ULL << 23))
        : nV_(csr.nV), mem_mgr_(max_edges_per_ba) {
        init_from_csr(csr);
    }

    ~HornetGraph() {
        if (d_used_) cudaFree(d_used_);
        if (d_ptr_) cudaFree(d_ptr_);
    }

    HornetGraph(const HornetGraph&) = delete;
    HornetGraph& operator=(const HornetGraph&) = delete;

    vid_t nV() const { return nV_; }
    vid_t* d_used() const { return d_used_; }
    vid_t** d_ptr() const { return d_ptr_; }
    MemoryManager& memory_manager() { return mem_mgr_; }
    const MemoryManager& memory_manager() const { return mem_mgr_; }

    size_t get_ba_allocated_edges() const { return mem_mgr_.get_ba_allocated_edges(); }
    size_t get_ba_allocated_bytes() const { return mem_mgr_.get_ba_allocated_bytes(); }
    size_t get_ba_count() const { return mem_mgr_.get_ba_count(); }

    void get_coo(std::vector<vid_t>& out_src, std::vector<vid_t>& out_dst) const {
        std::vector<vid_t> h_deg(nV_);
        CUDA_CHECK(cudaMemcpy(h_deg.data(), d_used_, nV_ * sizeof(vid_t), cudaMemcpyDeviceToHost));

        std::vector<eoff_t> h_offsets(nV_ + 1, 0);
        for (vid_t i = 0; i < nV_; ++i) {
            h_offsets[i + 1] = h_offsets[i] + h_deg[i];
        }
        eoff_t total_edges = h_offsets[nV_];
        out_src.resize(total_edges);
        out_dst.resize(total_edges);
        if (total_edges == 0) return;

        eoff_t* d_offsets = nullptr;
        vid_t* d_out_src = nullptr;
        vid_t* d_out_dst = nullptr;
        CUDA_CHECK(cudaMalloc(&d_offsets, (nV_ + 1) * sizeof(eoff_t)));
        CUDA_CHECK(cudaMalloc(&d_out_src, total_edges * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_out_dst, total_edges * sizeof(vid_t)));

        CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(), (nV_ + 1) * sizeof(eoff_t),
                              cudaMemcpyHostToDevice));

        int block = 256;
        int grid = (nV_ + block - 1) / block;
        scatter_hornet_to_coo_kernel<<<grid, block>>>(
            nV_, d_used_, d_ptr_, d_offsets, d_out_src, d_out_dst);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(out_src.data(), d_out_src, total_edges * sizeof(vid_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out_dst.data(), d_out_dst, total_edges * sizeof(vid_t),
                              cudaMemcpyDeviceToHost));

        cudaFree(d_offsets);
        cudaFree(d_out_src);
        cudaFree(d_out_dst);
    }

private:
    void init_from_csr(const CSRGraph& csr) {
        ScopedPhaseTimer total_timer("init_total", true);

        CUDA_CHECK(cudaMalloc(&d_used_, static_cast<size_t>(nV_) * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_ptr_, static_cast<size_t>(nV_) * sizeof(vid_t*)));

        std::vector<vid_t> h_used(nV_, 0);
        std::vector<vid_t*> h_ptr(nV_, nullptr);

        {
            ScopedPhaseTimer vloop_timer("init_vertex_loop", false);
            for (vid_t v = 0; v < nV_; ++v) {
                int deg = csr.row_offsets[v + 1] - csr.row_offsets[v];
                h_used[v] = deg;
                if (deg == 0) {
                    h_ptr[v] = nullptr;
                    continue;
                }
                int log_b = ceil_log2(deg);
                auto res = mem_mgr_.allocate_block(log_b, true);
                h_ptr[v] = res.ptr;
                res.ba->stage_block_data(
                    res.block_idx, csr.col_indices.data() + csr.row_offsets[v], deg);
            }
        }

        {
            ScopedPhaseTimer dcopy_timer("init_device_copy_loop", true);
            mem_mgr_.flush_all_staging();
            CUDA_CHECK(cudaMemcpy(d_used_, h_used.data(),
                                  static_cast<size_t>(nV_) * sizeof(vid_t),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ptr_, h_ptr.data(),
                                  static_cast<size_t>(nV_) * sizeof(vid_t*),
                                  cudaMemcpyHostToDevice));
        }
    }

    vid_t nV_ = 0;
    vid_t* d_used_ = nullptr;
    vid_t** d_ptr_ = nullptr;
    MemoryManager mem_mgr_;
};

} // namespace improved
