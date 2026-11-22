# Hornet: 2018 Paper Baseline & GPU Dynamic Graph Optimization

A modular reproduction and data-structure optimization project based on **Hornet** (*Busato et al., IEEE HPEC 2018*): *"Hornet: An Efficient Data Structure for Dynamic Sparse Graphs and Matrices on GPUs"*, written in CUDA/C++17 (tested on NVIDIA GeForce RTX 4060 Laptop GPU `sm_89` 8 GB VRAM, AMD Ryzen 7 7840HS, CUDA 13.3, GCC 12.4).

> **Pure Data-Structure-Only Experiment Design**
> `src/algorithms/` is shared untouched across implementations via `-DHORNET_DS_NS=baseline` and `-DHORNET_DS_NS=improved`. Every speedup reported in `src/improved/` comes strictly from optimizing the Hornet data structure itself (`MemoryManager`, `BitTree`, `BlockArray`, `HornetGraph`, `BatchWorkspace`, `batch_insert`, `batch_delete`), never from changing BFS, SpMV, Triangle Counting, or K-Truss algorithm kernels.

---

## 1. Repository & Branch Layout

* `src/baseline/` — Paper-faithful 2018 Hornet baseline (`baseline_benchmark`, `-DHORNET_DS_NS=baseline`):
  * `bit_tree.cuh`, `btree.hpp`, `block_array.cuh`, `memory_manager.cuh`, `hornet_graph.cuh`
  * `batch_workspace.cuh` (block-cooperative shared-memory `binarySearchLB`), `batch_insert.cuh`, `batch_delete.cuh` (with warp/block compaction & NVTX3/phase timers), `batch_update.cuh`
* `src/improved/` — Active optimization target (`improved_benchmark`, `-DHORNET_DS_NS=improved`), initialized from the faithful `src/baseline/` implementation for `DS-1`–`DS-7` improvements.
* `src/algorithms/` — Fixed, data-structure-agnostic analytics & benchmark harness:
  * **Static Traversal:** BFS vs Static CSR (`bfs.cuh`), SpMV vs Static CSR (`spmv.cuh`)
  * **Dynamic Analytics:** Maximal K-Truss (`ktruss.cuh`), Dynamic Triangle Counting (`triangle_counting.cuh`)
  * **Harness & I/O:** `benchmark.cu`, `types.cuh`, `graph_io.hpp`, `analytics.cuh`
* `scripts/` — Dataset download (`download_paper_datasets.py`), automated benchmark + `nsys` runner (`run_paper_experiments.py`), summary & Amdahl analyzer (`analyze_results.py`), and figure generator (`generate_plots.py`).
* `results/baseline/` — Committed baseline measurements (`raw.csv`, `analysis_summary.md`, `memory_results.txt`, `bfs_spmv_results.txt`, `tc_results.txt`, `ktruss_results.txt`, `nsys_*_stats.txt`, `plots/*.png`). `results/improved/` holds outputs from `--mode improved` runs.

**Git Branches:**
* `main` — Baseline (`src/baseline/`), improved scaffold (`src/improved/`), shared harness (`src/algorithms/`), scripts, and `results/baseline/`.
* `analysis` — Profiling audit & `DS-1`–`DS-7` optimization roadmap (`HOTSPOTS.md`).

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

## 6. Faithfulness Baseline Snapshot (`main@98fa57d`, `results/baseline/`)

Measured across the 5 graphs on the faithful 2018 paper baseline ([`results/baseline/analysis_summary.md`](results/baseline/analysis_summary.md)):

* **Memory Utilization (Fig. 4):** **71.0%–75.9%** block-level utilization from power-of-2 rounding; at `BA_EXP=16`, BlockArray utilization reaches **92.79%–99.92%**.
* **CSR Initialization (Sec. III-D):** **148.38–523.33 ms**, with single-threaded host `init_vertex_loop` consuming **71.5%–79.3%** (**106.16–406.19 ms**) and BlockArray host-to-device copy consuming **20.6%–27.2%**.
* **Static-CSR Traversal Parity (Fig. 7 & 8):** BFS achieves **0.93x–1.06x** of Static CSR (up to **10,802.54 MTEPS** on `kron_g500-logn21`); SpMV achieves **0.81x–0.97x** of Static CSR.
* **Dynamic Batch Updates (Fig. 5 & 6):**
  * **Insert ($B = 10^7$):** **50.78M–218.10M upd/s** median (up to **287.93M upd/s** peak on `dblp-2010`; **123.07M upd/s** median on `cage15`).
  * **Delete ($B = 10^7$):** **148.47M–697.02M upd/s** median (**715.01M upd/s** peak on `dblp-2010`; **247.11M upd/s** median on `kron_g500-logn21`).
* **Dynamic Maximal K-Truss (Fig. 9):** Up to **148.77M upd/s** peak (`cage15`: **925.86 ms** for **94,044,692** deleted edges across 12 steps), **53.77M upd/s** (`dblp-2010`), **40.16M upd/s** (`soc-LiveJournal1`), and **36.24M upd/s** (`in-2004`).
* **Primary Remaining Data-Structure Bottlenecks (see `HOTSPOTS.md` on `analysis`):**
  1. **Host CPU Alloc/Free + Metadata Ping-Pong:** At $B = 10^6$, `insert_host_alloc` + `insert_host_free` consume **40.7%–79.8%** of total insert time (**66.6%–79.8%** on 4 of 5 graphs; up to **4.95x** Amdahl ceiling on `soc-LiveJournal1` and **4.57x** on `dblp-2010`), and `delete_host_alloc` + `delete_host_free` consume **67.7%** of `dblp-2010` delete (**7.68 / 11.35 ms**).
  2. **CUDA Driver `cudaMalloc`/`cudaFree` Stalls & Zero-Filled Staging:** `cudaFree` accounts for **58.1%–65.6%** of CUDA API time in 10k `nsys` traces.
  3. **$O(\text{deg})$ Unsorted-Adjacency Scans:** `check_cross_dups_lb_kernel` takes **44.5%** of GPU kernel time on `in-2004` ($B = 10^6$ insert), and on `kron_g500-logn21` ($B = 10^6$ delete) `delete_mark` (**13.79 ms**, **64.4%**) + single-block `delete_compact` (**6.65 ms**, **31.0%**) dominate GPU time.
  4. **Small-Batch Latency Floor & Preprocessing:** ~15 separate CUB/CUDA launches give a **~0.38–0.83 ms** floor for $B \le 10^3$, while `DeviceRadixSort` runs full 64-bit keys on `< 24`-bit vertex IDs.
