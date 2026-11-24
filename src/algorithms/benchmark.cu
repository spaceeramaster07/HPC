#include "analytics.cuh"
#include "batch_update.cuh"
#include "graph_io.hpp"
#include "hornet_graph.cuh"
#include "types.cuh"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#ifndef HORNET_DS_NS
#define HORNET_DS_NS baseline
#endif

using namespace hornet_common;
using namespace hornet_alg;
using namespace HORNET_DS_NS;

static std::string get_graph_name(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    std::string fname = (pos == std::string::npos) ? path : path.substr(pos + 1);
    size_t dot = fname.find_last_of('.');
    return (dot == std::string::npos) ? fname : fname.substr(0, dot);
}

static bool verify_insert_correctness(const HornetGraph& graph,
                                      std::vector<vid_t>& init_src,
                                      std::vector<vid_t>& init_dst,
                                      const std::vector<vid_t>& batch_src,
                                      const std::vector<vid_t>& batch_dst,
                                      bool remove_graph_dups) {
    std::vector<std::pair<vid_t, vid_t>> init_edges(init_src.size());
    for (size_t i = 0; i < init_src.size(); ++i) {
        init_edges[i] = {init_src[i], init_dst[i]};
    }
    std::sort(init_edges.begin(), init_edges.end());

    // Deduplicate batch intra-batch first (Hornet always deduplicates within a batch)
    std::vector<std::pair<vid_t, vid_t>> batch_edges(batch_src.size());
    for (size_t i = 0; i < batch_src.size(); ++i) {
        batch_edges[i] = {batch_src[i], batch_dst[i]};
    }
    std::sort(batch_edges.begin(), batch_edges.end());
    batch_edges.erase(std::unique(batch_edges.begin(), batch_edges.end()), batch_edges.end());

    std::vector<std::pair<vid_t, vid_t>> expected = init_edges;
    for (const auto& e : batch_edges) {
        if (remove_graph_dups && std::binary_search(init_edges.begin(), init_edges.end(), e)) {
            continue;
        }
        expected.push_back(e);
    }
    std::sort(expected.begin(), expected.end());

    std::vector<vid_t> got_src, got_dst;
    graph.get_coo(got_src, got_dst);
    if (got_src.size() != expected.size()) {
        std::cerr << "[VERIFY ERROR] Size mismatch: Hornet has " << got_src.size()
                  << " edges, expected " << expected.size() << "\n";
        return false;
    }

    std::vector<std::pair<vid_t, vid_t>> actual(got_src.size());
    for (size_t i = 0; i < got_src.size(); ++i) {
        actual[i] = {got_src[i], got_dst[i]};
    }
    std::sort(actual.begin(), actual.end());

    for (size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::cerr << "[VERIFY ERROR] Mismatch at sorted index " << i
                      << ": got (" << actual[i].first << ", " << actual[i].second << ")"
                      << ", expected (" << expected[i].first << ", " << expected[i].second << ")\n";
            return false;
        }
    }
    return true;
}

static bool verify_delete_correctness(const HornetGraph& graph,
                                      const std::vector<vid_t>& init_src,
                                      const std::vector<vid_t>& init_dst,
                                      const std::vector<vid_t>& del_src,
                                      const std::vector<vid_t>& del_dst) {
    std::vector<std::pair<vid_t, vid_t>> del_edges(del_src.size());
    for (size_t i = 0; i < del_src.size(); ++i) {
        del_edges[i] = {del_src[i], del_dst[i]};
    }
    std::sort(del_edges.begin(), del_edges.end());
    del_edges.erase(std::unique(del_edges.begin(), del_edges.end()), del_edges.end());

    std::vector<std::pair<vid_t, vid_t>> expected;
    expected.reserve(init_src.size());
    for (size_t i = 0; i < init_src.size(); ++i) {
        std::pair<vid_t, vid_t> e{init_src[i], init_dst[i]};
        if (!std::binary_search(del_edges.begin(), del_edges.end(), e)) {
            expected.push_back(e);
        }
    }
    std::sort(expected.begin(), expected.end());

    std::vector<vid_t> got_src, got_dst;
    graph.get_coo(got_src, got_dst);
    if (got_src.size() != expected.size()) {
        std::cerr << "[VERIFY DELETE ERROR] Size mismatch: Hornet has " << got_src.size()
                  << " edges, expected " << expected.size() << "\n";
        return false;
    }

    for (size_t i = 1; i < got_src.size(); ++i) {
        if (got_src[i] == got_src[i - 1] && got_dst[i] < got_dst[i - 1]) {
            std::cerr << "[VERIFY DELETE ERROR] Adjacency list for vertex " << got_src[i]
                      << " became unsorted at edge " << i << "\n";
            return false;
        }
    }

    std::vector<std::pair<vid_t, vid_t>> actual(got_src.size());
    for (size_t i = 0; i < got_src.size(); ++i) {
        actual[i] = {got_src[i], got_dst[i]};
    }
    std::sort(actual.begin(), actual.end());

    for (size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::cerr << "[VERIFY DELETE ERROR] Mismatch at index " << i << "\n";
            return false;
        }
    }
    return true;
}

static void log_vram_usage(const std::string& tag) {
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    double used_mb = static_cast<double>(total_b - free_b) / (1024.0 * 1024.0);
    double free_mb = static_cast<double>(free_b) / (1024.0 * 1024.0);
    double total_mb = static_cast<double>(total_b) / (1024.0 * 1024.0);
    std::cout << "[VRAM " << tag << "] used_mb=" << std::fixed << std::setprecision(1) << used_mb
              << " free_mb=" << free_mb << " total_mb=" << total_mb << "\n";
}

static void extract_coo_from_csr(const CSRGraph& csr,
                                 std::vector<vid_t>& out_src,
                                 std::vector<vid_t>& out_dst) {
    out_src.resize(csr.nE);
    out_dst.resize(csr.nE);
    eoff_t idx = 0;
    for (vid_t v = 0; v < csr.nV; ++v) {
        for (eoff_t e = csr.row_offsets[v]; e < csr.row_offsets[v + 1]; ++e) {
            out_src[idx] = v;
            out_dst[idx] = csr.col_indices[e];
            ++idx;
        }
    }
}

static void verify_insert_once_csr(const std::string& graph_name,
                                   const CSRGraph& csr,
                                   int batch_size,
                                   uint64_t seed,
                                   bool remove_graph_dups,
                                   int ba_exp) {
    bool prev_prof = HornetProfiler::instance().is_enabled();
    HornetProfiler::instance().set_enabled(false);

    HornetGraph hg(csr, 1ULL << ba_exp);
    BatchWorkspace ws;
    ws.ensure_capacity(batch_size);

    std::vector<vid_t> init_src, init_dst, b_src, b_dst;
    hg.get_coo(init_src, init_dst);
    generate_random_batch(csr.nV, batch_size, seed, b_src, b_dst);
    ws.prepare(b_src, b_dst);
    CUDA_CHECK(cudaDeviceSynchronize());
    batch_insert(hg, ws, remove_graph_dups);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (!verify_insert_correctness(hg, init_src, init_dst, b_src, b_dst, remove_graph_dups)) {
        std::cerr << "[FATAL] Insert verification failed on " << graph_name
                  << " batch=" << batch_size << "!\n";
        std::exit(1);
    }
    std::cout << "[VERIFY PASS] insert graph=" << graph_name << " batch=" << batch_size << "\n";
    HornetProfiler::instance().set_enabled(prev_prof);
}

static void verify_delete_once_csr(const std::string& graph_name,
                                   const CSRGraph& csr,
                                   int batch_size,
                                   uint64_t seed) {
    bool prev_prof = HornetProfiler::instance().is_enabled();
    HornetProfiler::instance().set_enabled(false);

    HornetGraph hg(csr);
    BatchWorkspace ws;
    ws.ensure_capacity(batch_size);

    std::vector<vid_t> init_src, init_dst, b_src, b_dst;
    hg.get_coo(init_src, init_dst);
    select_random_edges(init_src, init_dst, batch_size, seed, b_src, b_dst);
    ws.prepare(b_src, b_dst);
    CUDA_CHECK(cudaDeviceSynchronize());
    batch_delete(hg, ws);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (!verify_delete_correctness(hg, init_src, init_dst, b_src, b_dst)) {
        std::cerr << "[FATAL] Delete verification failed on " << graph_name
                  << " batch=" << batch_size << "!\n";
        std::exit(1);
    }
    std::cout << "[VERIFY PASS] delete graph=" << graph_name << " batch=" << batch_size << "\n";
    HornetProfiler::instance().set_enabled(prev_prof);
}

static void run_bench_insert_csr(const std::string& graph_name,
                                 const CSRGraph& csr,
                                 int batch_size,
                                 int num_warmup,
                                 int num_trials,
                                 uint64_t base_seed,
                                 bool remove_graph_dups,
                                 bool verify,
                                 bool enable_profiling,
                                 int ba_exp) {
    std::string flags = remove_graph_dups ? "--graph-dedup" : "--no-dedup";
    size_t max_edges_per_ba = 1ULL << ba_exp;

    HornetProfiler::instance().set_enabled(enable_profiling);
    BatchWorkspace ws;
    ws.ensure_capacity(batch_size);

    for (int w = 0; w < num_warmup; ++w) {
        HornetGraph hg(csr, max_edges_per_ba);
        std::vector<vid_t> b_src, b_dst;
        generate_random_batch(csr.nV, batch_size, base_seed + 1000 + w, b_src, b_dst);
        ws.prepare(b_src, b_dst);
        CUDA_CHECK(cudaDeviceSynchronize());
        batch_insert(hg, ws, remove_graph_dups);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (w == num_warmup - 1 && batch_size >= 1000000) {
            log_vram_usage("insert graph=" + graph_name + " batch=" + std::to_string(batch_size));
        }
    }

    for (int t = 0; t < num_trials; ++t) {
        HornetGraph hg(csr, max_edges_per_ba);
        uint64_t trial_seed = base_seed + t;
        std::vector<vid_t> b_src, b_dst;
        generate_random_batch(csr.nV, batch_size, trial_seed, b_src, b_dst);
        ws.prepare(b_src, b_dst);

        std::vector<vid_t> init_src, init_dst;
        if (verify) {
            hg.get_coo(init_src, init_dst);
        }

        if (enable_profiling) {
            HornetProfiler::instance().clear();
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();

        batch_insert(hg, ws, remove_graph_dups);

        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double trial_total_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << graph_name << "," << batch_size << "," << t << ",insert_total,"
                  << std::fixed << std::setprecision(4) << trial_total_ms << ","
                  << trial_seed << "," << flags << ",baseline_btree\n";

        if (enable_profiling) {
            for (const auto& kv : HornetProfiler::instance().get_all_phases()) {
                if (kv.first != "insert_total") {
                    std::cout << graph_name << "," << batch_size << "," << t << "," << kv.first << ","
                              << std::fixed << std::setprecision(4) << kv.second << ","
                              << trial_seed << "," << flags << ",baseline_btree\n";
                }
            }
        }

        if (verify) {
            if (!verify_insert_correctness(hg, init_src, init_dst, b_src, b_dst, remove_graph_dups)) {
                std::cerr << "[FATAL] Insert verification failed on trial " << t << "!\n";
                std::exit(1);
            }
        }
    }
}

static void run_bench_delete_csr(const std::string& graph_name,
                                 const CSRGraph& csr,
                                 int batch_size,
                                 int num_warmup,
                                 int num_trials,
                                 uint64_t base_seed,
                                 bool enable_profiling,
                                 bool verify) {
    std::string flags = "--delete";

    HornetProfiler::instance().set_enabled(enable_profiling);
    BatchWorkspace ws;
    ws.ensure_capacity(batch_size);

    std::vector<vid_t> init_src, init_dst;
    extract_coo_from_csr(csr, init_src, init_dst);

    for (int w = 0; w < num_warmup; ++w) {
        HornetGraph hg(csr);
        std::vector<vid_t> b_src, b_dst;
        select_random_edges(init_src, init_dst, batch_size, base_seed + 1000 + w, b_src, b_dst);
        ws.prepare(b_src, b_dst);
        CUDA_CHECK(cudaDeviceSynchronize());
        batch_delete(hg, ws);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (w == num_warmup - 1 && b_src.size() >= 1000000) {
            log_vram_usage("delete graph=" + graph_name + " batch=" + std::to_string(b_src.size()));
        }
    }

    for (int t = 0; t < num_trials; ++t) {
        HornetGraph hg(csr);
        uint64_t trial_seed = base_seed + t;
        std::vector<vid_t> b_src, b_dst;
        select_random_edges(init_src, init_dst, batch_size, trial_seed, b_src, b_dst);
        const size_t actual_batch = b_src.size();
        ws.prepare(b_src, b_dst);

        if (enable_profiling) {
            HornetProfiler::instance().clear();
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();

        batch_delete(hg, ws);

        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double trial_total_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << graph_name << "," << actual_batch << "," << t << ",delete_total,"
                  << std::fixed << std::setprecision(4) << trial_total_ms << ","
                  << trial_seed << "," << flags << ",baseline_btree\n";

        if (enable_profiling) {
            for (const auto& kv : HornetProfiler::instance().get_all_phases()) {
                if (kv.first != "delete_total") {
                    std::cout << graph_name << "," << actual_batch << "," << t << "," << kv.first << ","
                              << std::fixed << std::setprecision(4) << kv.second << ","
                              << trial_seed << "," << flags << ",baseline_btree\n";
                }
            }
        }

        if (verify) {
            if (!verify_delete_correctness(hg, init_src, init_dst, b_src, b_dst)) {
                std::cerr << "[FATAL] Delete verification failed on trial " << t << "!\n";
                std::exit(1);
            }
        }
    }
}

static void run_bench_init_csr(const std::string& graph_name, const CSRGraph& csr, int num_trials) {

    HornetProfiler::instance().set_enabled(true);

    for (int t = 0; t < num_trials; ++t) {
        HornetProfiler::instance().clear();

        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();

        HornetGraph hg(csr);

        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

        double vloop_ms = HornetProfiler::instance().get_phase_ms("init_vertex_loop");
        double dcopy_ms = HornetProfiler::instance().get_phase_ms("init_device_copy_loop");

        std::cout << graph_name << ",0," << t << ",init_total,"
                  << std::fixed << std::setprecision(4) << total_ms << ",0,--init,baseline_btree\n";
        std::cout << graph_name << ",0," << t << ",init_vertex_loop,"
                  << std::fixed << std::setprecision(4) << vloop_ms << ",0,--init,baseline_btree\n";
        std::cout << graph_name << ",0," << t << ",init_device_copy_loop,"
                  << std::fixed << std::setprecision(4) << dcopy_ms << ",0,--init,baseline_btree\n";
    }
}

static void run_bench_memory_csr(const std::string& graph_name, const CSRGraph& csr, int ba_exp) {

    vid_t nV = csr.nV;
    eoff_t nE = csr.nE;

    long long sum_deg = 0;
    long long sum_pow2 = 0;
    for (vid_t i = 0; i < nV; ++i) {
        int d = csr.row_offsets[i + 1] - csr.row_offsets[i];
        sum_deg += d;
        if (d > 0) {
            sum_pow2 += (d == 1) ? 1 : (1 << ceil_log2(d));
        }
    }
    double block_util = (sum_pow2 > 0) ? (static_cast<double>(sum_deg) / sum_pow2) : 0.0;

    CUDA_CHECK(cudaDeviceSynchronize());
    size_t free_before = 0, total_vram = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_vram));

    size_t max_edges_per_ba = 1ULL << ba_exp;
    HornetGraph hg(csr, max_edges_per_ba);
    CUDA_CHECK(cudaDeviceSynchronize());

    size_t free_after = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total_vram));
    size_t vram_used_bytes = (free_before > free_after) ? (free_before - free_after) : 0;

    size_t ba_allocated_edges = hg.get_ba_allocated_edges();
    size_t ba_count = hg.get_ba_count();
    double ba_util = (ba_allocated_edges > 0) ? (static_cast<double>(sum_pow2) / ba_allocated_edges) : 0.0;
    double overall_util = (ba_allocated_edges > 0) ? (static_cast<double>(sum_deg) / ba_allocated_edges) : 0.0;

    size_t csr_bytes = (static_cast<size_t>(nV) + 1) * sizeof(eoff_t) + static_cast<size_t>(nE) * sizeof(vid_t);
    size_t coo_bytes = static_cast<size_t>(nE) * 2 * sizeof(vid_t);

    double hornet_bpe = (nE > 0) ? (static_cast<double>(vram_used_bytes) / nE) : 0.0;
    double csr_bpe = (nE > 0) ? (static_cast<double>(csr_bytes) / nE) : 0.0;
    double coo_bpe = (nE > 0) ? (static_cast<double>(coo_bytes) / nE) : 0.0;

    std::cout << "[MEMORY_SUMMARY BA_EXP=" << ba_exp << "] "
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
              << " hornet_bpe=" << hornet_bpe
              << " csr_bytes=" << csr_bytes
              << " csr_bpe=" << csr_bpe
              << " coo_bytes=" << coo_bytes
              << " coo_bpe=" << coo_bpe << "\n";
}

static void run_bench_bfs_csr(const std::string& graph_name, const CSRGraph& csr, vid_t source, int num_trials) {

    // If default source=0 has degree 0, pick highest-degree vertex for meaningful traversal
    if (source == 0 && csr.nV > 0 && (csr.row_offsets[1] - csr.row_offsets[0]) == 0) {
        vid_t best = 0;
        int best_deg = 0;
        for (vid_t v = 0; v < csr.nV; ++v) {
            int d = csr.row_offsets[v + 1] - csr.row_offsets[v];
            if (d > best_deg) {
                best_deg = d;
                best = v;
            }
        }
        source = best;
    }

    HornetGraph hg(csr);
    // Warmup
    run_bfs(hg, source);
    run_bfs_csr(csr, source);

    double h_ms = 0.0, h_mteps = 0.0;
    double c_ms = 0.0, c_mteps = 0.0;
    long long traversed = 0;
    int visited = 0;
    for (int t = 0; t < num_trials; ++t) {
        auto rh = run_bfs(hg, source);
        auto rc = run_bfs_csr(csr, source);
        h_ms += rh.time_ms;
        h_mteps += rh.mteps;
        c_ms += rc.time_ms;
        c_mteps += rc.mteps;
        traversed = rh.traversed_edges;
        visited = rh.visited_vertices;
    }

    double avg_h_ms = h_ms / num_trials;
    double avg_c_ms = c_ms / num_trials;
    double ratio = (avg_h_ms > 0.0) ? (avg_c_ms / avg_h_ms) : 0.0;

    std::cout << "BFS," << graph_name
              << ",Hornet_ms=" << std::fixed << std::setprecision(3) << avg_h_ms
              << ",Hornet_MTEPS=" << std::setprecision(2) << (h_mteps / num_trials)
              << ",CSR_ms=" << std::setprecision(3) << avg_c_ms
              << ",CSR_MTEPS=" << std::setprecision(2) << (c_mteps / num_trials)
              << ",Speedup_vs_CSR=" << std::setprecision(2) << ratio << "x"
              << ",visited=" << visited << ",traversed=" << traversed << "\n";
}

static void run_bench_spmv_csr(const std::string& graph_name, const CSRGraph& csr, int num_trials) {

    HornetGraph hg(csr);
    auto rh = run_spmv(hg, num_trials);
    auto rc = run_spmv_csr(csr, num_trials);
    double ratio = (rh.time_ms > 0.0) ? (rc.time_ms / rh.time_ms) : 0.0;
    std::cout << "SPMV," << graph_name
              << ",Hornet_ms=" << std::fixed << std::setprecision(3) << rh.time_ms
              << ",CSR_ms=" << rc.time_ms
              << ",Speedup_vs_CSR=" << std::setprecision(2) << ratio << "x\n";
}

static void run_bench_ktruss_sym(const std::string& graph_name, const CSRGraph& sym, int max_k) {

    auto steps = run_ktruss(sym, max_k);
    long long total_deleted = 0;
    double total_ms = 0.0;
    double peak_rate = 0.0;
    for (const auto& s : steps) {
        total_deleted += s.deleted_edges;
        total_ms += s.step_ms;
        if (s.update_rate_per_sec > peak_rate) peak_rate = s.update_rate_per_sec;
        std::cout << "KTRUSS_STEP," << graph_name << ",k=" << s.k
                  << ",deleted=" << s.deleted_edges
                  << ",ms=" << std::fixed << std::setprecision(3) << s.step_ms
                  << ",rate_upd_s=" << std::setprecision(0) << s.update_rate_per_sec << "\n";
    }
    std::cout << "KTRUSS_SUMMARY," << graph_name
              << ",steps=" << steps.size()
              << ",total_deleted=" << total_deleted
              << ",total_ms=" << std::fixed << std::setprecision(2) << total_ms
              << ",peak_rate_upd_s=" << std::setprecision(0) << peak_rate << "\n";
}

static void run_bench_tc_sym(const std::string& graph_name, const CSRGraph& sym, int batch_size, int num_trials, uint64_t seed) {

    // Warmup
    run_dynamic_tc(sym, batch_size, seed + 999);

    double sum_ins = 0.0, sum_qry = 0.0, sum_tot = 0.0;
    unsigned long long last_tri = 0;
    for (int t = 0; t < num_trials; ++t) {
        auto r = run_dynamic_tc(sym, batch_size, seed + t);
        sum_ins += r.insert_ms;
        sum_qry += r.query_ms;
        sum_tot += r.total_ms;
        last_tri = r.delta_triangles;
    }
    double avg_ins = sum_ins / num_trials;
    double avg_qry = sum_qry / num_trials;
    double avg_tot = sum_tot / num_trials;
    double rate = (avg_tot > 0.0) ? (static_cast<double>(batch_size) / (avg_tot / 1000.0)) : 0.0;

    std::cout << "DYNAMIC_TC," << graph_name
              << ",batch=" << batch_size
              << ",insert_ms=" << std::fixed << std::setprecision(3) << avg_ins
              << ",query_ms=" << avg_qry
              << ",total_ms=" << avg_tot
              << ",rate_upd_s=" << std::setprecision(0) << rate
              << ",delta_triangles=" << last_tri << "\n";
}

static CSRGraph load_csr_or_exit(const std::string& graph_path) {
    CSRGraph csr;
    if (!read_mtx_to_csr(graph_path, csr)) std::exit(1);
    return csr;
}

static void run_bench_insert(const std::string& graph_path, int batch_size, int num_warmup, int num_trials,
                             uint64_t base_seed, bool remove_graph_dups, bool verify, bool enable_profiling, int ba_exp) {
    run_bench_insert_csr(get_graph_name(graph_path), load_csr_or_exit(graph_path),
                         batch_size, num_warmup, num_trials, base_seed, remove_graph_dups, verify, enable_profiling, ba_exp);
}

static void run_bench_delete(const std::string& graph_path, int batch_size, int num_warmup, int num_trials,
                             uint64_t base_seed, bool enable_profiling, bool verify) {
    run_bench_delete_csr(get_graph_name(graph_path), load_csr_or_exit(graph_path),
                         batch_size, num_warmup, num_trials, base_seed, enable_profiling, verify);
}

static void run_bench_init(const std::string& graph_path, int num_trials) {
    run_bench_init_csr(get_graph_name(graph_path), load_csr_or_exit(graph_path), num_trials);
}

static void run_bench_memory(const std::string& graph_path, int ba_exp) {
    run_bench_memory_csr(get_graph_name(graph_path), load_csr_or_exit(graph_path), ba_exp);
}

static void run_bench_bfs(const std::string& graph_path, vid_t source, int num_trials) {
    run_bench_bfs_csr(get_graph_name(graph_path), load_csr_or_exit(graph_path), source, num_trials);
}

static void run_bench_spmv(const std::string& graph_path, int num_trials) {
    run_bench_spmv_csr(get_graph_name(graph_path), load_csr_or_exit(graph_path), num_trials);
}

static void run_bench_ktruss(const std::string& graph_path, int max_k) {
    CSRGraph csr = load_csr_or_exit(graph_path);
    run_bench_ktruss_sym(get_graph_name(graph_path), make_undirected_simple(csr), max_k);
}

static void run_bench_tc(const std::string& graph_path, int batch_size, int num_trials, uint64_t seed) {
    CSRGraph csr = load_csr_or_exit(graph_path);
    run_bench_tc_sym(get_graph_name(graph_path), make_undirected_simple(csr), batch_size, num_trials, seed);
}

static std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start < s.size()) {
        size_t pos = s.find(',', start);
        if (pos == std::string::npos) {
            if (start < s.size()) out.push_back(s.substr(start));
            break;
        }
        if (pos > start) out.push_back(s.substr(start, pos - start));
        start = pos + 1;
    }
    return out;
}

/** Runs all requested benchmark suites for one graph from a single .mtx load. */
static void run_bench_suite(const std::string& graph_path, bool verify_small, const std::string& tasks_csv) {
    std::string graph_name = get_graph_name(graph_path);
    CSRGraph csr = load_csr_or_exit(graph_path);
    std::vector<std::string> tasks = split_csv(tasks_csv);
    auto wants = [&](const std::string& t) {
        return std::find(tasks.begin(), tasks.end(), t) != tasks.end();
    };
    const int batch_sizes[] = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000};

    log_vram_usage("suite_start graph=" + graph_name);
    std::cout.flush();

    if (wants("memory")) {
        for (int exp : {16, 19, 22, 23}) {
            run_bench_memory_csr(graph_name, csr, exp);
            std::cout.flush();
        }
    }
    if (wants("init")) {
        run_bench_init_csr(graph_name, csr, 10);
        std::cout.flush();
    }
    if (wants("insert")) {
        bool verify_graph = verify_small && (graph_name == "dblp-2010" || graph_name == "in-2004");
        if (verify_graph) {
            for (int b : batch_sizes) {
                if (b <= 1000000) {
                    verify_insert_once_csr(graph_name, csr, b, 42ULL, true, 23);
                    std::cout.flush();
                }
            }
        }
        for (int b : batch_sizes) {
            run_bench_insert_csr(graph_name, csr, b, 3, 10, 42ULL, true, false, true, 23);
            std::cout.flush();
            if (b == 10000 || b == 1000000) {
                run_bench_insert_csr(graph_name, csr, b, 3, 10, 42ULL, false, false, true, 23);
                std::cout.flush();
            }
        }
    }
    if (wants("delete")) {
        bool verify_graph = verify_small && (graph_name == "dblp-2010");
        if (verify_graph) {
            for (int b : batch_sizes) {
                if (b <= 1000000) {
                    verify_delete_once_csr(graph_name, csr, b, 42ULL);
                    std::cout.flush();
                }
            }
        }
        for (int b : batch_sizes) {
            run_bench_delete_csr(graph_name, csr, b, 3, 10, 42ULL, true, false);
            std::cout.flush();
        }
    }
    if (wants("traversal")) {
        run_bench_bfs_csr(graph_name, csr, 0, 5);
        std::cout.flush();
        run_bench_spmv_csr(graph_name, csr, 10);
        std::cout.flush();
    }
    if (wants("tc") || wants("ktruss")) {
        CSRGraph sym = make_undirected_simple(csr);
        if (wants("tc")) {
            for (int b : {10000, 100000}) {
                run_bench_tc_sym(graph_name, sym, b, 5, 42ULL);
                std::cout.flush();
            }
        }
        if (wants("ktruss")) {
            if (graph_name == "kron_g500-logn21") {
                std::cout << "[KTRUSS_SKIP] Skipping baseline ktruss on " << graph_name
                          << " (unoptimized static recount risks TDR/long runtime)\n";
            } else {
                run_bench_ktruss_sym(graph_name, sym, 50);
            }
            std::cout.flush();
        }
    }

    log_vram_usage("suite_end graph=" + graph_name);
    std::cout.flush();
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: baseline_benchmark <mode> <graph_path> [options...]\n"
                  << "Modes:\n"
                  << "  bench_insert <graph> <batch_size> <warmup> <trials> <seed> <dedup:1/0> <verify:1/0> <profile:1/0> [ba_exp=23]\n"
                  << "  bench_delete <graph> <batch_size> <warmup> <trials> <seed> <profile:1/0> [verify:1/0]\n"
                  << "  bench_init   <graph> <trials>\n"
                  << "  bench_memory <graph> [ba_exp=23]\n"
                  << "  bench_bfs    <graph> [source=0] [trials=5]\n"
                  << "  bench_spmv   <graph> [trials=10]\n"
                  << "  bench_ktruss <graph> [max_k=50]\n"
                  << "  bench_tc     <graph> [batch_size=100000] [trials=5] [seed=42]\n"
                  << "  bench_suite  <graph> [verify_small:1/0=1] [tasks_csv=memory,init,insert,delete,traversal,tc,ktruss]\n";
        return 1;
    }

    std::string mode = argv[1];
    std::string graph_path = argv[2];

    if (mode == "bench_insert") {
        int batch_size = std::stoi(argv[3]);
        int num_warmup = (argc > 4) ? std::stoi(argv[4]) : 3;
        int num_trials = (argc > 5) ? std::stoi(argv[5]) : 10;
        uint64_t seed = (argc > 6) ? std::stoull(argv[6]) : 42ULL;
        bool dedup = (argc > 7) ? (std::stoi(argv[7]) != 0) : true;
        bool verify = (argc > 8) ? (std::stoi(argv[8]) != 0) : false;
        bool profile = (argc > 9) ? (std::stoi(argv[9]) != 0) : false;
        int ba_exp = (argc > 10) ? std::stoi(argv[10]) : 23;
        run_bench_insert(graph_path, batch_size, num_warmup, num_trials, seed, dedup, verify, profile, ba_exp);
    } else if (mode == "bench_delete") {
        int batch_size = std::stoi(argv[3]);
        int num_warmup = (argc > 4) ? std::stoi(argv[4]) : 3;
        int num_trials = (argc > 5) ? std::stoi(argv[5]) : 10;
        uint64_t seed = (argc > 6) ? std::stoull(argv[6]) : 42ULL;
        bool profile = (argc > 7) ? (std::stoi(argv[7]) != 0) : false;
        bool verify = (argc > 8) ? (std::stoi(argv[8]) != 0) : false;
        run_bench_delete(graph_path, batch_size, num_warmup, num_trials, seed, profile, verify);
    } else if (mode == "bench_init") {
        int num_trials = (argc > 3) ? std::stoi(argv[3]) : 10;
        run_bench_init(graph_path, num_trials);
    } else if (mode == "bench_memory") {
        int ba_exp = (argc > 3) ? std::stoi(argv[3]) : 23;
        run_bench_memory(graph_path, ba_exp);
    } else if (mode == "bench_bfs") {
        vid_t source = (argc > 3) ? std::stoi(argv[3]) : 0;
        int trials = (argc > 4) ? std::stoi(argv[4]) : 5;
        run_bench_bfs(graph_path, source, trials);
    } else if (mode == "bench_spmv") {
        int trials = (argc > 3) ? std::stoi(argv[3]) : 10;
        run_bench_spmv(graph_path, trials);
    } else if (mode == "bench_ktruss") {
        int max_k = (argc > 3) ? std::stoi(argv[3]) : 50;
        run_bench_ktruss(graph_path, max_k);
    } else if (mode == "bench_tc") {
        int batch_size = (argc > 3) ? std::stoi(argv[3]) : 100000;
        int trials = (argc > 4) ? std::stoi(argv[4]) : 5;
        uint64_t seed = (argc > 5) ? std::stoull(argv[5]) : 42ULL;
        run_bench_tc(graph_path, batch_size, trials, seed);
    } else if (mode == "bench_suite") {
        bool verify_small = (argc > 3) ? (std::stoi(argv[3]) != 0) : true;
        std::string tasks_csv = (argc > 4) ? argv[4] : "memory,init,insert,delete,traversal,tc,ktruss";
        run_bench_suite(graph_path, verify_small, tasks_csv);
    } else {
        std::cerr << "Unknown mode: " << mode << "\n";
        return 1;
    }

    return 0;
}
