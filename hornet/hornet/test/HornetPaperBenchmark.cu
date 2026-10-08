#include <Hornet.hpp>
#include "StandardAPI.hpp"
#include "Util/BatchFunctions.hpp"
#include "Util/RandomGraphData.cuh"
#include "Util/HornetProfiler.hpp"
#include <Host/FileUtil.hpp>
#include <Device/Util/CudaUtil.cuh>
#include <algorithm>
#include <chrono>
#include <random>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cassert>

using vert_t = int;
using eoff_t = int;
using HornetGPU = hornet::gpu::Hornet<vert_t>;
using Update = hornet::gpu::BatchUpdate<vert_t>;
using Init = hornet::HornetInit<vert_t>;
using hornet::DeviceType;

const std::string GIT_COMMIT = "2441ccabd44a09dbf16f521dbd0bc856ab5001f8";

// Helper to get graph basename
std::string get_graph_name(const std::string& path) {
    auto pos = path.find_last_of("/\\");
    std::string filename = (pos == std::string::npos) ? path : path.substr(pos + 1);
    auto dot = filename.find_last_of('.');
    return (dot == std::string::npos) ? filename : filename.substr(0, dot);
}

// High-performance correctness verification helper for insert
bool verify_insert_correctness(HornetGPU& hornet_gpu,
                               hornet::COO<DeviceType::DEVICE, vert_t, hornet::EMPTY, eoff_t>& init_coo,
                               hornet::COO<DeviceType::DEVICE, vert_t, hornet::EMPTY, eoff_t>& random_batch,
                               bool remove_graph_dups) {
    // Sort both on GPU for maximum throughput
    init_coo.sort();
    auto inst_coo = hornet_gpu.getCOO(true);
    inst_coo.sort();

    hornet::COO<DeviceType::HOST, vert_t, hornet::EMPTY, eoff_t> host_init = init_coo;
    hornet::COO<DeviceType::HOST, vert_t, hornet::EMPTY, eoff_t> host_batch = random_batch;
    hornet::COO<DeviceType::HOST, vert_t, hornet::EMPTY, eoff_t> host_inst = inst_coo;

    auto* init_s = host_init.srcPtr();
    auto* init_d = host_init.dstPtr();
    size_t init_len = host_init.size();

    // Fast binary search on sorted host_init
    auto exists_in_init = [&](vert_t s, vert_t d) -> bool {
        size_t low = 0, high = init_len;
        while (low < high) {
            size_t mid = low + (high - low) / 2;
            if (init_s[mid] < s || (init_s[mid] == s && init_d[mid] < d)) {
                low = mid + 1;
            } else if (init_s[mid] == s && init_d[mid] == d) {
                return true;
            } else {
                high = mid;
            }
        }
        return false;
    };

    // Filter batch edges against original graph
    std::vector<std::pair<vert_t, vert_t>> accepted_batch;
    accepted_batch.reserve(host_batch.size());
    for (size_t i = 0; i < host_batch.size(); ++i) {
        vert_t bs = host_batch.srcPtr()[i];
        vert_t bd = host_batch.dstPtr()[i];
        if (remove_graph_dups && exists_in_init(bs, bd)) {
            continue;
        }
        accepted_batch.emplace_back(bs, bd);
    }
    std::sort(accepted_batch.begin(), accepted_batch.end());

    size_t expected_total = init_len + accepted_batch.size();
    if (host_inst.size() != expected_total) {
        std::cerr << "[VERIFY ERROR] Size mismatch: Hornet has " << host_inst.size()
                  << " edges, expected " << expected_total
                  << " (init=" << init_len << ", batch_added=" << accepted_batch.size() << ")\n";
        return false;
    }

    // Linear 2-way merge comparison against host_inst
    auto* inst_s = host_inst.srcPtr();
    auto* inst_d = host_inst.dstPtr();
    size_t init_idx = 0;
    size_t batch_idx = 0;
    size_t inst_idx = 0;

    while (init_idx < init_len || batch_idx < accepted_batch.size()) {
        vert_t cur_s, cur_d;
        if (init_idx < init_len && batch_idx < accepted_batch.size()) {
            if (init_s[init_idx] < accepted_batch[batch_idx].first ||
                (init_s[init_idx] == accepted_batch[batch_idx].first &&
                 init_d[init_idx] <= accepted_batch[batch_idx].second)) {
                cur_s = init_s[init_idx];
                cur_d = init_d[init_idx];
                init_idx++;
            } else {
                cur_s = accepted_batch[batch_idx].first;
                cur_d = accepted_batch[batch_idx].second;
                batch_idx++;
            }
        } else if (init_idx < init_len) {
            cur_s = init_s[init_idx];
            cur_d = init_d[init_idx];
            init_idx++;
        } else {
            cur_s = accepted_batch[batch_idx].first;
            cur_d = accepted_batch[batch_idx].second;
            batch_idx++;
        }

        if (inst_s[inst_idx] != cur_s || inst_d[inst_idx] != cur_d) {
            std::cerr << "[VERIFY ERROR] Mismatch at inst_idx " << inst_idx
                      << ": Hornet has (" << inst_s[inst_idx] << "->" << inst_d[inst_idx]
                      << "), expected (" << cur_s << "->" << cur_d << ")\n";
            return false;
        }
        inst_idx++;
    }

    return true;
}

// Task 4 & 7: Dynamic Update Benchmark (Insert)
void run_bench_insert(const std::string& graph_path,
                      int batch_size,
                      int num_warmup,
                      int num_trials,
                      size_t base_seed,
                      bool remove_graph_dups,
                      bool verify,
                      bool enable_profiling) {
    std::string graph_name = get_graph_name(graph_path);
    std::string flags = remove_graph_dups ? "--graph-dedup" : "--no-dedup";

    graph::GraphStd<vert_t, vert_t> graph;
    graph.read(graph_path.c_str());

    Init hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());

    hornet::HornetProfiler::instance().set_enabled(enable_profiling);

    // Warm-up runs
    for (int w = 0; w < num_warmup; ++w) {
        HornetGPU hornet_gpu(hornet_init);
        hornet::RandomGenTraits<hornet::EMPTY> cooGenTraits(true, false, base_seed + 1000 + w);
        auto randomBatch = hornet::generateRandomCOO<vert_t, eoff_t>(graph.nV(), batch_size, cooGenTraits);
        Update batch_update(randomBatch);
        cudaDeviceSynchronize();
        hornet_gpu.insert(batch_update, false, remove_graph_dups);
        cudaDeviceSynchronize();
    }

    // Timed trials
    for (int t = 0; t < num_trials; ++t) {
        HornetGPU hornet_gpu(hornet_init);
        size_t trial_seed = base_seed + t;
        hornet::RandomGenTraits<hornet::EMPTY> cooGenTraits(true, false, trial_seed);
        auto randomBatch = hornet::generateRandomCOO<vert_t, eoff_t>(graph.nV(), batch_size, cooGenTraits);
        Update batch_update(randomBatch);

        hornet::COO<DeviceType::DEVICE, vert_t, hornet::EMPTY, eoff_t> init_coo;
        if (verify) {
            init_coo = hornet_gpu.getCOO(true);
        }

        if (enable_profiling) {
            hornet::HornetProfiler::instance().clear();
        }

        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();

        hornet_gpu.insert(batch_update, false, remove_graph_dups);

        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        double trial_total_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Print raw long-format row for total
        std::cout << graph_name << "," << batch_size << "," << t << ",insert_total,"
                  << std::fixed << std::setprecision(4) << trial_total_ms << ","
                  << trial_seed << "," << flags << "," << GIT_COMMIT << "\n";

        // Print detailed subphases if profiling enabled
        if (enable_profiling) {
            const auto& phases = hornet::HornetProfiler::instance().get_all_phases();
            for (const auto& kv : phases) {
                if (kv.first != "insert_total") {
                    std::cout << graph_name << "," << batch_size << "," << t << "," << kv.first << ","
                              << std::fixed << std::setprecision(4) << kv.second << ","
                              << trial_seed << "," << flags << "," << GIT_COMMIT << "\n";
                }
            }
        }

        if (verify) {
            bool ok = verify_insert_correctness(hornet_gpu, init_coo, randomBatch, remove_graph_dups);
            if (!ok) {
                std::cerr << "[FATAL] Correctness verification failed on trial " << t << "!\n";
                std::exit(1);
            }
        }
    }
}

// Task 4: Dynamic Delete Benchmark
void run_bench_delete(const std::string& graph_path,
                      int batch_size,
                      int num_warmup,
                      int num_trials,
                      size_t base_seed,
                      bool enable_profiling) {
    std::string graph_name = get_graph_name(graph_path);
    std::string flags = "--delete";

    graph::GraphStd<vert_t, vert_t> graph;
    graph.read(graph_path.c_str());

    Init hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());

    hornet::HornetProfiler::instance().set_enabled(enable_profiling);

    // Warm-up runs
    for (int w = 0; w < num_warmup; ++w) {
        HornetGPU hornet_gpu(hornet_init);
        auto init_coo = hornet_gpu.getCOO(true);
        hornet::RandomGenTraits<hornet::EMPTY> cooGenTraits(true, false, base_seed + 1000 + w);
        auto randomBatch = hornet::selectRandom(init_coo, batch_size, cooGenTraits);
        Update batch_update(randomBatch);
        cudaDeviceSynchronize();
        hornet_gpu.erase(batch_update);
        cudaDeviceSynchronize();
    }

    // Timed trials
    for (int t = 0; t < num_trials; ++t) {
        HornetGPU hornet_gpu(hornet_init);
        size_t trial_seed = base_seed + t;
        auto init_coo = hornet_gpu.getCOO(true);
        hornet::RandomGenTraits<hornet::EMPTY> cooGenTraits(true, false, trial_seed);
        auto randomBatch = hornet::selectRandom(init_coo, batch_size, cooGenTraits);
        Update batch_update(randomBatch);

        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();

        hornet_gpu.erase(batch_update);

        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        double trial_total_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << graph_name << "," << batch_size << "," << t << ",delete_total,"
                  << std::fixed << std::setprecision(4) << trial_total_ms << ","
                  << trial_seed << "," << flags << "," << GIT_COMMIT << "\n";
    }
}

// Task 6: Initialization from CSR Timing & Subphases
void run_bench_init(const std::string& graph_path, int num_trials) {
    std::string graph_name = get_graph_name(graph_path);

    graph::GraphStd<vert_t, vert_t> graph;
    graph.read(graph_path.c_str());

    hornet::HornetProfiler::instance().set_enabled(true);

    for (int t = 0; t < num_trials; ++t) {
        hornet::HornetProfiler::instance().clear();
        Init hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());
        
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();

        HornetGPU hornet_gpu(hornet_init);

        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

        double vloop_ms = hornet::HornetProfiler::instance().get_phase_ms("init_vertex_loop");
        double dcopy_ms = hornet::HornetProfiler::instance().get_phase_ms("init_device_copy_loop");

        std::cout << graph_name << ",0," << t << ",init_total,"
                  << std::fixed << std::setprecision(4) << total_ms << ",0,--init," << GIT_COMMIT << "\n";
        std::cout << graph_name << ",0," << t << ",init_vertex_loop,"
                  << std::fixed << std::setprecision(4) << vloop_ms << ",0,--init," << GIT_COMMIT << "\n";
        std::cout << graph_name << ",0," << t << ",init_device_copy_loop,"
                  << std::fixed << std::setprecision(4) << dcopy_ms << ",0,--init," << GIT_COMMIT << "\n";
    }
}

// Task 5: Memory Utilization Analysis
void run_bench_memory(const std::string& graph_path) {
    std::string graph_name = get_graph_name(graph_path);

    graph::GraphStd<vert_t, vert_t> graph;
    graph.read(graph_path.c_str());

    int nV = graph.nV();
    int nE = graph.nE();
    const int* offsets = graph.csr_out_offsets();

    // 1. Block-level utilization from CSR degrees
    long long sum_deg = 0;
    long long sum_pow2 = 0;
    for (int i = 0; i < nV; ++i) {
        int d = offsets[i+1] - offsets[i];
        sum_deg += d;
        int pow2_deg = (d <= 1) ? 1 : (1 << xlib::ceil_log2(d));
        sum_pow2 += pow2_deg;
    }
    double block_util = (double)sum_deg / sum_pow2;

    // 2. Measure VRAM before and after Hornet initialization
    cudaDeviceSynchronize();
    size_t free_before = 0, total_vram = 0;
    cudaMemGetInfo(&free_before, &total_vram);

    Init hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());
    HornetGPU hornet_gpu(hornet_init);
    cudaDeviceSynchronize();

    size_t free_after = 0;
    cudaMemGetInfo(&free_after, &total_vram);
    size_t vram_used_bytes = (free_before > free_after) ? (free_before - free_after) : 0;

    size_t ba_allocated_edges = hornet_gpu.get_ba_allocated_edges();
    size_t ba_allocated_bytes = hornet_gpu.get_ba_allocated_bytes();
    size_t ba_count = hornet_gpu.get_ba_count();

    double ba_util = (ba_allocated_edges > 0) ? ((double)sum_pow2 / ba_allocated_edges) : 0.0;
    double overall_util = (ba_allocated_edges > 0) ? ((double)sum_deg / ba_allocated_edges) : 0.0;

    // Analytical CSR & COO sizes
    size_t csr_bytes = ((size_t)nV + 1) * sizeof(int) + ((size_t)nE) * sizeof(int);
    size_t coo_bytes = ((size_t)nE) * 2 * sizeof(int);

    double hornet_bytes_per_edge = (nE > 0) ? ((double)vram_used_bytes / nE) : 0.0;
    double csr_bytes_per_edge = (nE > 0) ? ((double)csr_bytes / nE) : 0.0;
    double coo_bytes_per_edge = (nE > 0) ? ((double)coo_bytes / nE) : 0.0;

    std::cout << "[MEMORY_SUMMARY] "
              << "Graph=" << graph_name
              << " |V|=" << nV
              << " |E|=" << nE
              << " sum_deg=" << sum_deg
              << " sum_pow2=" << sum_pow2
              << " block_util=" << std::fixed << std::setprecision(4) << block_util
              << " ba_allocated_edges=" << ba_allocated_edges
              << " ba_util=" << ba_util
              << " overall_util=" << overall_util
              << " ba_count=" << ba_count
              << " vram_used_bytes=" << vram_used_bytes
              << " hornet_bpe=" << hornet_bytes_per_edge
              << " csr_bytes=" << csr_bytes
              << " csr_bpe=" << csr_bytes_per_edge
              << " coo_bytes=" << coo_bytes
              << " coo_bpe=" << coo_bytes_per_edge << "\n";
}

// Task 3: Measure Instrumentation Overhead
void run_bench_overhead(const std::string& graph_path, int batch_size, int num_trials) {
    std::string graph_name = get_graph_name(graph_path);
    graph::GraphStd<vert_t, vert_t> graph;
    graph.read(graph_path.c_str());
    Init hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());

    // 1. Without profiler (disabled)
    hornet::HornetProfiler::instance().set_enabled(false);
    double total_uninstrumented_ms = 0.0;
    for (int t = 0; t < num_trials; ++t) {
        HornetGPU hornet_gpu(hornet_init);
        hornet::RandomGenTraits<hornet::EMPTY> cooGenTraits(true, false, 42 + t);
        auto randomBatch = hornet::generateRandomCOO<vert_t, eoff_t>(graph.nV(), batch_size, cooGenTraits);
        Update batch_update(randomBatch);
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        hornet_gpu.insert(batch_update, false, true);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        total_uninstrumented_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }
    double avg_uninst = total_uninstrumented_ms / num_trials;

    // 2. With profiler enabled
    hornet::HornetProfiler::instance().set_enabled(true);
    double total_instrumented_ms = 0.0;
    for (int t = 0; t < num_trials; ++t) {
        HornetGPU hornet_gpu(hornet_init);
        hornet::RandomGenTraits<hornet::EMPTY> cooGenTraits(true, false, 42 + t);
        auto randomBatch = hornet::generateRandomCOO<vert_t, eoff_t>(graph.nV(), batch_size, cooGenTraits);
        Update batch_update(randomBatch);
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        hornet_gpu.insert(batch_update, false, true);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        total_instrumented_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }
    double avg_inst = total_instrumented_ms / num_trials;

    double overhead_ms = avg_inst - avg_uninst;
    double overhead_pct = (overhead_ms / avg_uninst) * 100.0;

    std::cout << "[OVERHEAD_SUMMARY] Graph=" << graph_name
              << " BatchSize=" << batch_size
              << " Trials=" << num_trials
              << " Uninstrumented_ms=" << std::fixed << std::setprecision(4) << avg_uninst
              << " Instrumented_ms=" << avg_inst
              << " Delta_ms=" << overhead_ms
              << " Overhead_pct=" << std::fixed << std::setprecision(2) << overhead_pct << "%\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: hornet_paper_benchmark <mode> <graph_path> [options...]\n"
                  << "Modes:\n"
                  << "  bench_insert <graph_path> <batch_size> <num_warmup> <num_trials> <seed> <dedup_flag: 1/0> <verify: 1/0> <profiling: 1/0>\n"
                  << "  bench_delete <graph_path> <batch_size> <num_warmup> <num_trials> <seed> <profiling: 1/0>\n"
                  << "  bench_init   <graph_path> <num_trials>\n"
                  << "  bench_memory <graph_path>\n"
                  << "  bench_overhead <graph_path> <batch_size> <num_trials>\n";
        return 1;
    }

    std::string mode = argv[1];
    std::string graph_path = argv[2];

    if (mode == "bench_insert") {
        int batch_size = std::stoi(argv[3]);
        int num_warmup = (argc > 4) ? std::stoi(argv[4]) : 3;
        int num_trials = (argc > 5) ? std::stoi(argv[5]) : 10;
        size_t base_seed = (argc > 6) ? std::stoull(argv[6]) : 42;
        bool remove_graph_dups = (argc > 7) ? (std::stoi(argv[7]) != 0) : true;
        bool verify = (argc > 8) ? (std::stoi(argv[8]) != 0) : false;
        bool enable_profiling = (argc > 9) ? (std::stoi(argv[9]) != 0) : false;

        run_bench_insert(graph_path, batch_size, num_warmup, num_trials, base_seed, remove_graph_dups, verify, enable_profiling);
    }
    else if (mode == "bench_delete") {
        int batch_size = std::stoi(argv[3]);
        int num_warmup = (argc > 4) ? std::stoi(argv[4]) : 3;
        int num_trials = (argc > 5) ? std::stoi(argv[5]) : 10;
        size_t base_seed = (argc > 6) ? std::stoull(argv[6]) : 42;
        bool enable_profiling = (argc > 7) ? (std::stoi(argv[7]) != 0) : false;

        run_bench_delete(graph_path, batch_size, num_warmup, num_trials, base_seed, enable_profiling);
    }
    else if (mode == "bench_init") {
        int num_trials = (argc > 3) ? std::stoi(argv[3]) : 10;
        run_bench_init(graph_path, num_trials);
    }
    else if (mode == "bench_memory") {
        run_bench_memory(graph_path);
    }
    else if (mode == "bench_overhead") {
        int batch_size = (argc > 3) ? std::stoi(argv[3]) : 10000;
        int num_trials = (argc > 4) ? std::stoi(argv[4]) : 10;
        run_bench_overhead(graph_path, batch_size, num_trials);
    }
    else {
        std::cerr << "Unknown mode: " << mode << "\n";
        return 1;
    }

    return 0;
}
