# Hornet: 2018 Paper Baseline & GPU Dynamic Graph Optimization

A modular reproduction and data-structure optimization project based on **Hornet** (*Busato et al., IEEE HPEC 2018*): *"Hornet: An Efficient Data Structure for Dynamic Sparse Graphs and Matrices on GPUs"*, written in CUDA/C++17 (tested on NVIDIA GeForce RTX 4060 Laptop GPU `sm_89` 8 GB VRAM, AMD Ryzen 7 7840HS, CUDA 13.3, GCC 12.4).

> **Pure Data-Structure-Only Experiment Design**
> `src/algorithms/` is shared untouched across implementations via `-DHORNET_DS_NS=baseline` and `-DHORNET_DS_NS=improved`. Every speedup reported in `src/improved/` comes strictly from optimizing the Hornet data structure itself (`MemoryManager`, `BitTree`, `BlockArray`, `HornetGraph`, `BatchWorkspace`, `batch_insert`, `batch_delete`), never from changing BFS, SpMV, Triangle Counting, or K-Truss algorithm kernels.

---

## 1. Repository & Branch Layout

* `src/baseline/` — Paper-faithful 2018 Hornet baseline (`baseline_benchmark`, `-DHORNET_DS_NS=baseline`):
  * `bit_tree.cuh`, `btree.hpp`, `block_array.cuh`, `memory_manager.cuh`, `hornet_graph.cuh`
  * `batch_workspace.cuh` (block-cooperative shared-memory `binarySearchLB`), `batch_insert.cuh`, `batch_delete.cuh` (with warp/block compaction & NVTX3/phase timers), `batch_update.cuh`
* `src/improved/` — Optimized Hornet data structure (`improved_benchmark`, `-DHORNET_DS_NS=improved`) implementing `DS-1`–`DS-5` and `DS-7` (`gpu_allocator.cuh` GPU-resident `BitTree` + device block allocator, OpenMP CSR init, sorted adjacency lists, tight radix-sort packing, small-batch fast path, CUDA mempool).
* `src/algorithms/` — Fixed, data-structure-agnostic analytics & benchmark harness:
  * **Static Traversal:** BFS vs Static CSR (`bfs.cuh`), SpMV vs Static CSR (`spmv.cuh`)
  * **Dynamic Analytics:** Maximal K-Truss (`ktruss.cuh`), Dynamic Triangle Counting (`triangle_counting.cuh`)
  * **Harness & I/O:** `benchmark.cu`, `types.cuh`, `graph_io.hpp`, `analytics.cuh`
* `scripts/` — Dataset download (`download_paper_datasets.py`), automated benchmark + `nsys` runner (`run_paper_experiments.py`), summary & Amdahl analyzer (`analyze_results.py`), per-mode figure generator (`generate_plots.py`), and side-by-side comparison generator (`compare_results.py`).
* `results/baseline/`, `results/improved/`, `results/comparison/` — Baseline measurements, `DS-1..DS-7` improved measurements, and side-by-side comparison report/plots (`comparison_summary.md`, `plots/*.png`).

**Primary Documents & Branches:**
* `main` — Baseline (`src/baseline/`), optimized `DS-1..DS-7` (`src/improved/`), shared harness (`src/algorithms/`), scripts, `HOTSPOTS.md`, and `results/{baseline,improved,comparison}/`.
* `analysis` — Initial profiling audit & `DS-1`–`DS-7` optimization roadmap (`HOTSPOTS.md`).

---

## 2. Prerequisites & Environment Setup

### Toolchain
* **CMake** `>= 3.18` (tested with `3.28.3`)
* **C++17 host compiler** (tested with GCC `12.4.0`)
* **CUDA Toolkit** (configured by default to `/usr/local/cuda-13.3/bin/nvcc`, targeting `sm_89` in `CMakeLists.txt`; override `-DCMAKE_CUDA_ARCHITECTURES=<sm>` if building on another GPU)
* **Nsight Systems (`nsys`)** (optional; `scripts/run_paper_experiments.py` checks `/usr/local/cuda-13.3/bin/nsys` for the `nsys` task and skips it if absent)

### Python Environment (for Datasets, Analysis, and Plots)
Create `.venv` and install the packages used by the analysis/plotting scripts:

```bash
python3 -m venv .venv
.venv/bin/python3 -m pip install --upgrade pip
.venv/bin/python3 -m pip install numpy matplotlib
```

---

## 3. Datasets Setup

Datasets live in `datasets/` as `.mtx` (Matrix Market) files and are gitignored (`datasets/*`).

Download SuiteSparse datasets via the helper script:

```bash
# Default downloads dblp-2010.mtx (~15 MB)
.venv/bin/python3 scripts/download_paper_datasets.py

# Or download all datasets configured in the downloader (dblp-2010, cantilever, in-2004, soc-LiveJournal1)
.venv/bin/python3 scripts/download_paper_datasets.py all
```

The full 5-graph suite in `scripts/run_paper_experiments.py` expects these 5 files in `datasets/`:
* `datasets/dblp-2010.mtx`
* `datasets/in-2004.mtx`
* `datasets/soc-LiveJournal1.mtx`
* `datasets/cage15.mtx` *(from SuiteSparse Matrix Collection `vanHeukelum/cage15` if running the full 5-graph suite)*
* `datasets/kron_g500-logn21.mtx` *(from SuiteSparse Matrix Collection `DIMACS10/kron_g500-logn21` if running the full 5-graph suite)*

---

## 4. Building `baseline` and `improved`

### Build Baseline (`src/baseline/build/baseline_benchmark`)
```bash
cmake -S src/baseline -B src/baseline/build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc
cmake --build src/baseline/build -j8
```

### Build Improved (`src/improved/build/improved_benchmark`)
```bash
cmake -S src/improved -B src/improved/build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc
cmake --build src/improved/build -j8
```

*(If running on a GPU other than RTX 40xx `sm_89`, add e.g. `-DCMAKE_CUDA_ARCHITECTURES=86` during the `cmake -S ... -B ...` configure step.)*

---

## 5. Running Benchmarks

### Single-Command Modes
Both `baseline_benchmark` and `improved_benchmark` share the exact same CLI (`benchmark.cu`):

```text
<binary> <mode> <graph_path> [options...]
  bench_insert <graph> <batch_size> <warmup> <trials> <seed> <dedup:1/0> <verify:1/0> <profile:1/0> [ba_exp=23]
  bench_delete <graph> <batch_size> <warmup> <trials> <seed> <profile:1/0> [verify:1/0]
  bench_init   <graph> <trials>
  bench_memory <graph> [ba_exp=23]
  bench_bfs    <graph> [source=0] [trials=5]
  bench_spmv   <graph> [trials=10]
  bench_ktruss <graph> [max_k=50]
  bench_tc     <graph> [batch_size=100000] [trials=5] [seed=42]
  bench_suite  <graph> [verify_small:1/0=1] [tasks_csv=memory,init,insert,delete,traversal,tc,ktruss]
```

#### Quick Correctness (`verify=1`) & Smoke Examples
Replace `./src/improved/build/improved_benchmark` with `./src/baseline/build/baseline_benchmark` to test the baseline binary:

```bash
# Verify dynamic insert & delete exactness + adjacency sortedness (keep verify=0 for pure timing)
./src/improved/build/improved_benchmark bench_insert datasets/dblp-2010.mtx 100000 1 1 42 1 1 0
./src/improved/build/improved_benchmark bench_delete datasets/dblp-2010.mtx 100000 1 1 42 0 1
./src/improved/build/improved_benchmark bench_insert datasets/kron_g500-logn21.mtx 100000 1 1 42 1 1 0
./src/improved/build/improved_benchmark bench_delete datasets/kron_g500-logn21.mtx 100000 1 1 42 0 1

# Single-operation / algorithm benchmarks
./src/improved/build/improved_benchmark bench_memory datasets/dblp-2010.mtx 23
./src/improved/build/improved_benchmark bench_init   datasets/dblp-2010.mtx 10
./src/improved/build/improved_benchmark bench_insert datasets/dblp-2010.mtx 10000 3 10 42 1 0 1
./src/improved/build/improved_benchmark bench_delete datasets/dblp-2010.mtx 10000 3 10 42 1 0
./src/improved/build/improved_benchmark bench_bfs    datasets/dblp-2010.mtx 0 5
./src/improved/build/improved_benchmark bench_spmv   datasets/dblp-2010.mtx 10
./src/improved/build/improved_benchmark bench_ktruss datasets/dblp-2010.mtx 50
./src/improved/build/improved_benchmark bench_tc     datasets/dblp-2010.mtx 100000 5 42
```

### Full Paper Suite, Analysis Tables & Plots
`run_paper_experiments.py` builds the selected target (`baseline` or `improved`), executes all tasks across the 5 graphs (skipping `ktruss` on `kron_g500-logn21` to avoid TDR/long static-recount runtime under the unmodified algorithm), runs `nsys` traces on `dblp-2010` and `in-2004` (`10k` and `1M`), and writes results into `results/<mode>/`.

```bash
# Run full suite (baseline or improved)
.venv/bin/python3 scripts/run_paper_experiments.py --mode improved
.venv/bin/python3 scripts/analyze_results.py --mode improved      # writes results/improved/analysis_summary.md
.venv/bin/python3 scripts/generate_plots.py --mode improved       # writes results/improved/plots/*.png

# Or run baseline
.venv/bin/python3 scripts/run_paper_experiments.py --mode baseline
.venv/bin/python3 scripts/analyze_results.py --mode baseline      # writes results/baseline/analysis_summary.md
.venv/bin/python3 scripts/generate_plots.py --mode baseline       # writes results/baseline/plots/*.png

# Generate side-by-side Baseline vs. Improved report & plots
.venv/bin/python3 scripts/compare_results.py                      # writes results/comparison/comparison_summary.md & plots/*.png
```

Useful options:
* `run_paper_experiments.py --tasks ...`: run a subset from `{memory, init, insert, delete, traversal, ktruss, tc, nsys}` (note: running any of `init`, `insert`, or `delete` overwrites `raw.csv`, so keep those three together when regenerating timing data).
* `run_paper_experiments.py --no-verify`: skip the `verify_small=1` correctness passes before timed trials in `bench_suite`.
* `analyze_results.py --no-write` / `-o <path>`: print to stdout only without overwriting `analysis_summary.md`, or specify a custom output path.

Generated outputs under `results/<mode>/`:
* `raw.csv` — Trial-level timing & phase breakdown rows (`init_*`, `insert_*`, `delete_*`)
* `memory_results.txt`, `bfs_spmv_results.txt`, `tc_results.txt`, `ktruss_results.txt`
* `nsys_{dblp-2010,in-2004}_{mid_10k,large_1m}.{nsys-rep,sqlite,_stats.txt}` *(binary `.nsys-rep`/`.sqlite` files are gitignored)*
* `analysis_summary.md` & `plots/*.png`

---

## 6. Faithfulness Baseline Snapshot (`results/baseline/`)

Measured across the 5 graphs on the faithful 2018 paper baseline ([`results/baseline/analysis_summary.md`](results/baseline/analysis_summary.md)):

* **Memory Utilization (Fig. 4):** **71.0%–75.9%** block-level utilization from power-of-2 rounding; at `BA_EXP=16`, BlockArray utilization reaches **92.79%–99.92%**.
* **CSR Initialization (Sec. III-D):** **142.60–503.70 ms**, with single-threaded host `init_vertex_loop` consuming **71.7%–79.5%** (**102.31–398.66 ms**) and BlockArray copy consuming **20.4%–27.9%**.
* **Static-CSR Traversal Parity (Fig. 7 & 8):** BFS achieves **0.94x–1.00x** of Static CSR (up to **11,503.26 MTEPS** on `kron_g500-logn21`); SpMV achieves **0.81x–0.96x** of Static CSR.
* **Dynamic Batch Updates (Fig. 5 & 6):**
  * **Insert ($B = 10^7$):** **53.7M–229.4M upd/s** median (up to **292.0M upd/s** peak on `dblp-2010`; **132.2M upd/s** median on `cage15`).
  * **Delete ($B = 10^7$):** **149.7M–259.9M upd/s** median on the 4 graphs with $|E| \ge 10^7$ (`149.7M` `cage15`, `175.7M` `in-2004`, `192.1M` `soc-LiveJournal1`, `259.9M` `kron_g500-logn21`; `dblp-2010` clamps to $|E| = 1,615,400$ at **117.2M upd/s** in `13.78 ms`).
* **Dynamic Maximal K-Truss (Fig. 9):** Up to **156.0M upd/s** peak (`cage15`: **854.14 ms** for **94,044,692** deleted edges across 12 steps), **62.4M upd/s** (`dblp-2010`), **44.0M upd/s** (`soc-LiveJournal1`), and **38.0M upd/s** (`in-2004`).
* **Primary Remaining Data-Structure Bottlenecks (see `HOTSPOTS.md`):**
  1. **Host CPU Alloc/Free + Metadata Ping-Pong:** At $B = 10^6$, `insert_host_alloc` + `insert_host_free` consume **34.7%–78.7%** of total insert time (**62.0%–78.7%** on 4 of 5 graphs).
  2. **CUDA Driver `cudaMalloc`/`cudaFree` Stalls & Zero-Filled Staging:** `cudaFree` takes **148.3 ms** (**54.4%** of CUDA API time) in `in-2004` $B = 10^6$ `nsys`.
  3. **$O(\text{deg})$ Unsorted-Adjacency Scans:** `check_cross_dups_lb_kernel` takes **44.7%** (**1,985.7 us**) of GPU kernel time on `in-2004` ($B = 10^6$ insert), and on `kron_g500-logn21` ($B = 10^6$ delete) `delete_mark` + `delete_compact` dominate GPU time (**21.03 ms** total).
  4. **Small-Batch Latency Floor & Preprocessing:** Separate CUB launches give a **~0.39–0.70 ms** floor for $B \le 10^2$, while `DeviceRadixSort` runs full 64-bit keys on `< 24`-bit vertex IDs.

---

## 7. Final Optimization Summary (`DS-1..DS-7` vs. Baseline)

Measured side by side in [`results/comparison/comparison_summary.md`](results/comparison/comparison_summary.md) with zero algorithm changes (`src/algorithms/` traversal/analytics kernels untouched):

* **CSR Initialization (`DS-1` + `DS-4`):** **3.10x–17.16x faster** (**8.31–152.28 ms** vs. **142.60–503.70 ms** baseline) via 2-pass OpenMP degree histogram + uninitialized prefix-sized staging + `cudaMallocAsync`.
* **Dynamic Batch Insert (`DS-2`, `DS-3`, `DS-5`, `DS-7`):**
  * **$B = 10^6$ (`--graph-dedup`):** **3.05x–10.67x faster** (**1.75–6.90 ms**, up to **571.9M upd/s** median on `dblp-2010`); `--no-dedup` reaches **652.6M upd/s** (**12.77x**).
  * **$B = 10^7$ (`--graph-dedup`):** **3.07x–7.33x faster** (**11.36–29.09 ms**, **343.8M–880.2M upd/s** median).
  * **Small-Slab `BA_EXP=16/19` ($B = 10^6$):** **4.81x–6.88x faster** (`in-2004`, `kron_g500-logn21`).
* **Dynamic Batch Delete (`DS-2`, `DS-3`, `DS-5`, `DS-7`):**
  * **$B = 10^6$:** **2.13x–7.36x faster** (**1.39–9.87 ms**, up to **720.0M upd/s** on `dblp-2010`).
  * **$B = 10^7$ (effective `min(B, |E|)`):** **1.78x–3.57x faster** on the 4 graphs with $|E| \ge 10^7$ (up to **626.5M upd/s** on `in-2004`); on `dblp-2010` (clamped by `select_random_edges` to $|E| = 1,615,400$), latency drops **8.79x** (**13.78 → 1.57 ms**, **1.03B upd/s** effective vs. **117.2M upd/s**).
* **Host-Serial Alloc/Free Elimination (`DS-7`):** Per-vertex CPU `BPlusTree`/`BitTree` alloc & free eliminated (`34.7%–78.7%` → **`0.0%` / `0.0000 ms`** at $B = 10^4$ and $10^6$, with **14.456 → 0.000 MB** D2H transfer in `in-2004` $B = 10^6$ `nsys`); only rare slab-exhaustion fallback (~**0.48–1.37 ms**) runs on $B = 10^7$.
* **Dynamic Analytics (Unchanged Kernels):**
  * **Dynamic Triangle Counting ($B = 10^5$):** **1.72x–5.78x faster**, and sorted adjacency lists (`DS-5`) restore exact post-insert triangle deltas (`dblp-2010`: `17 → 19`; `in-2004`: `1,900 → 1,904`).
  * **Dynamic Maximal K-Truss (`max_k = 50`):** **2.36x faster** on `cage15` (**854.14 → 361.79 ms**, **560.8M upd/s** peak) and **1.54x faster** on `dblp-2010` (**175.21 → 113.80 ms**, **186.9M upd/s** peak), with identical deleted edge counts across all 4 tested graphs (`kron_g500-logn21` skipped identically in both modes).
* **Static Traversal Parity:** BFS remains **0.94x–0.99x** of Static CSR (up to **11,418.11 MTEPS** on `kron_g500-logn21`); SpMV remains **0.81x–0.99x**.
