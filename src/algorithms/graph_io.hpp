#pragma once

#include "types.cuh"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace hornet_common {

struct CSRGraph {
    vid_t nV = 0;
    eoff_t nE = 0;
    std::vector<eoff_t> row_offsets;
    std::vector<vid_t> col_indices;
};

inline bool read_mtx_to_csr(const std::string& path, CSRGraph& out) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "[IO ERROR] Failed to open graph file: " << path << "\n";
        return false;
    }

    std::string line;
    bool is_symmetric = false;

    if (!std::getline(fin, line)) {
        std::cerr << "[IO ERROR] Empty file: " << path << "\n";
        return false;
    }

    if (line.rfind("%%MatrixMarket", 0) == 0) {
        std::string lower = line;
        for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        is_symmetric = (lower.find("symmetric") != std::string::npos);
    } else {
        fin.seekg(0);
    }

    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '%') continue;
        break;
    }

    long long rows = 0, cols = 0, nnz = 0;
    {
        std::istringstream iss(line);
        iss >> rows >> cols >> nnz;
    }
    if (rows <= 0 || nnz < 0) {
        std::cerr << "[IO ERROR] Invalid MatrixMarket header dimensions in " << path << "\n";
        return false;
    }

    vid_t nV = static_cast<vid_t>(std::max(rows, cols));
    std::vector<eoff_t> degrees(nV, 0);

    struct RawEdge {
        vid_t u;
        vid_t v;
    };
    std::vector<RawEdge> edges;
    edges.reserve(is_symmetric ? static_cast<size_t>(nnz) * 2 : static_cast<size_t>(nnz));

    std::FILE* fp = std::fopen(path.c_str(), "r");
    if (!fp) {
        std::cerr << "[IO ERROR] Failed to reopen file with fopen: " << path << "\n";
        return false;
    }

    char buf[4096];
    // Skip header/comment lines and dimension line
    while (std::fgets(buf, sizeof(buf), fp)) {
        if (buf[0] == '%' || buf[0] == '\n' || buf[0] == '\r') continue;
        break; // skipped dimension line
    }

    long long u_in = 0, v_in = 0;
    while (std::fgets(buf, sizeof(buf), fp)) {
        if (buf[0] == '%' || buf[0] == '\n' || buf[0] == '\r') continue;
        char* endptr = nullptr;
        u_in = std::strtoll(buf, &endptr, 10);
        if (endptr == buf) continue;
        v_in = std::strtoll(endptr, &endptr, 10);

        vid_t u = static_cast<vid_t>(u_in - 1);
        vid_t v = static_cast<vid_t>(v_in - 1);
        if (u < 0 || u >= nV || v < 0 || v >= nV) continue;

        edges.push_back({u, v});
        degrees[u]++;
        if (is_symmetric && u != v) {
            edges.push_back({v, u});
            degrees[v]++;
        }
    }
    std::fclose(fp);

    out.nV = nV;
    out.nE = static_cast<eoff_t>(edges.size());
    out.row_offsets.assign(nV + 1, 0);
    for (vid_t i = 0; i < nV; ++i) {
        out.row_offsets[i + 1] = out.row_offsets[i] + degrees[i];
    }

    out.col_indices.resize(out.nE);
    std::vector<eoff_t> cursor = out.row_offsets;
    for (const auto& e : edges) {
        out.col_indices[cursor[e.u]++] = e.v;
    }

    // Sort neighbor lists per vertex (preserves |V| and |E|, enables fast triangle intersection)
    for (vid_t v = 0; v < nV; ++v) {
        std::sort(out.col_indices.begin() + out.row_offsets[v],
                  out.col_indices.begin() + out.row_offsets[v + 1]);
    }

    return true;
}

inline CSRGraph make_undirected_simple(const CSRGraph& in) {
    std::vector<std::pair<vid_t, vid_t>> edges;
    edges.reserve(static_cast<size_t>(in.nE) * 2);
    for (vid_t u = 0; u < in.nV; ++u) {
        for (eoff_t e = in.row_offsets[u]; e < in.row_offsets[u + 1]; ++e) {
            vid_t v = in.col_indices[e];
            if (u == v) continue; // remove self-loops
            edges.emplace_back(u, v);
            edges.emplace_back(v, u);
        }
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    CSRGraph out;
    out.nV = in.nV;
    out.nE = static_cast<eoff_t>(edges.size());
    out.row_offsets.assign(out.nV + 1, 0);
    for (const auto& e : edges) {
        out.row_offsets[e.first + 1]++;
    }
    for (vid_t i = 0; i < out.nV; ++i) {
        out.row_offsets[i + 1] += out.row_offsets[i];
    }
    out.col_indices.resize(out.nE);
    for (size_t i = 0; i < edges.size(); ++i) {
        out.col_indices[i] = edges[i].second;
    }
    return out;
}

inline void generate_random_batch(vid_t nV,
                                  int batch_size,
                                  uint64_t seed,
                                  std::vector<vid_t>& out_src,
                                  std::vector<vid_t>& out_dst) {
    out_src.resize(batch_size);
    out_dst.resize(batch_size);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<vid_t> dist(0, nV - 1);
    for (int i = 0; i < batch_size; ++i) {
        vid_t u = dist(rng);
        vid_t v = dist(rng);
        if (nV > 1 && u == v) {
            v = (u + 1) % nV;
        }
        out_src[i] = u;
        out_dst[i] = v;
    }
}

inline void select_random_edges(const std::vector<vid_t>& all_src,
                                const std::vector<vid_t>& all_dst,
                                int batch_size,
                                uint64_t seed,
                                std::vector<vid_t>& out_src,
                                std::vector<vid_t>& out_dst) {
    size_t total = all_src.size();
    int actual = static_cast<int>(std::min<size_t>(batch_size, total));
    out_src.resize(actual);
    out_dst.resize(actual);
    if (actual == 0) return;

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, total - 1);
    for (int i = 0; i < actual; ++i) {
        size_t idx = dist(rng);
        out_src[i] = all_src[idx];
        out_dst[i] = all_dst[idx];
    }
}

} // namespace hornet_common
