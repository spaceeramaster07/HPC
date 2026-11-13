# Hornet: 2018 Paper Baseline & GPU Dynamic Graph Optimization

A clean, modular reproduction and optimization project based on **Hornet** (*Busato et al., IEEE HPEC 2018*): *"Hornet: An Efficient Data Structure for Dynamic Sparse Graphs and Matrices on GPUs"*, built for CUDA 13.3, GCC 12, and C++17 (tested on NVIDIA RTX 4060 Laptop GPU, `sm_89`).

---

## 1. Repository Structure

- `src/baseline/` — Faithful, unoptimized 2018 paper baseline data structure:
  - `btree.hpp` (B+Tree vertex-degree index), `bit_tree.cuh` (64-way BitTree), `block_array.cuh` (`cudaMalloc` per `BlockArray`), `memory_manager.cuh` (single-threaded host alloc/free), `hornet_graph.cuh`, `batch_update.cuh` (Algorithm 1 insert/delete with NVTX3 markers).
- `src/algorithms/` — Data-structure-agnostic analytics & benchmark harness (compiled against any `HornetGraph` implementation via `-DHORNET_DS_NS=...`):
  - **Static / Traversal:** BFS vs Static CSR (`bench_bfs`), SpMV vs Static CSR (`bench_spmv`).
  - **Dynamic Analytics:** Dynamic Maximal K-Truss (`bench_ktruss`), Dynamic Triangle Counting (`bench_tc`).
  - **Core Harness:** `types.cuh`, `graph_io.hpp`, `analytics.cuh`, `benchmark.cu`.
- `scripts/` — Automation for running the paper suite (`run_paper_experiments.py`), computing statistics & Amdahl bounds (`analyze_results.py`), and generating figures (`generate_plots.py`).
- `results/baseline/` — Baseline measurements, `nsys` profiles, `analysis_summary.md`, and plots.

---

## 2. Baseline Replication & Bottleneck Findings

Evaluated across 5 SuiteSparse graphs (`dblp-2010`, `in-2004`, `soc-LiveJournal1`, `cage15`, `kron_g500-logn21`):

- **Memory Utilization (Fig. 4):** **71.0%–75.0%** block-level utilization from power-of-2 rounding; `BA_EXP=16`/`19` achieves up to **99–100%** BlockArray utilization.
- **Static-CSR Traversal Parity (Fig. 7 & 8):** BFS achieves **0.86x–1.04x** of Static CSR (up to **11,408 MTEPS** on `kron_g500-logn21`); SpMV achieves **0.74x–0.99x** of Static CSR.
- **Dynamic Updates (Fig. 5 & 6):** Batch insert reaches **47.7M–239.0M upd/s** ($B = 10^7$); batch delete reaches **43.1M–828.8M upd/s** ($B = 10^7$).
- **Dynamic Maximal K-Truss (Fig. 9):** Up to **79.5M upd/s** (`dblp-2010`) and **41.9M upd/s** (`cage15`).
- **Primary Bottleneck (Amdahl & `nsys`):**
  - At $B = 10^6$, the single-threaded CPU B+Tree/BitTree allocator (`insert_host_alloc` + `insert_host_free`) consumes **47.6%–81.1%** of total insert time (**1.91x–5.28x** Amdahl ceiling), and `init_vertex_loop` consumes **71.7%–76.8%** of CSR initialization.
  - On the GPU side (`in-2004`, 1M batch), `check_cross_dups_warp_kernel` accounts for **50.4%** of GPU kernel time, while `cudaMalloc`/`cudaFree` drive CUDA API overhead.

---

## 3. Quick Start

### Build Baseline
```bash
cmake -S src/baseline -B src/baseline/build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc
cmake --build src/baseline/build -j8
```

### Run Benchmarks
```bash
# Single-command examples
./src/baseline/build/baseline_benchmark bench_memory datasets/dblp-2010.mtx 23
./src/baseline/build/baseline_benchmark bench_insert datasets/dblp-2010.mtx 10000 3 10 42 1 1 1
./src/baseline/build/baseline_benchmark bench_bfs datasets/dblp-2010.mtx 0 5
./src/baseline/build/baseline_benchmark bench_spmv datasets/dblp-2010.mtx 10
./src/baseline/build/baseline_benchmark bench_ktruss datasets/dblp-2010.mtx 50
./src/baseline/build/baseline_benchmark bench_tc datasets/dblp-2010.mtx 100000 5 42
```

### Run Full Suite, Analysis & Plots
```bash
.venv/bin/python3 scripts/run_paper_experiments.py --mode baseline
.venv/bin/python3 scripts/analyze_results.py > results/baseline/analysis_summary.md
.venv/bin/python3 scripts/generate_plots.py
```
