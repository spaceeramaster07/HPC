#!/home/spaceeramaster/HPCproj/.venv/bin/python3
"""
Master reproduction script for:
"Hornet: An Efficient Data Structure for Dynamic Sparse Graphs and Matrices on GPUs"
(Busato, Green, Bombieri, Bader, IEEE HPEC 2018)

Author: HPC Performance Engineer
Target: RTX 4060 Laptop GPU (8 GB GDDR6, sm_89, WSL2 Ubuntu 24.04 LTS)
"""

import os
import sys
import time
import subprocess
import re
import csv
import shutil
import numpy as np
from pathlib import Path

WORKSPACE = Path("/home/spaceeramaster/HPCproj")
RESULTS_DIR = WORKSPACE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RAW_CSV = RESULTS_DIR / "raw.csv"
ENV_TXT = RESULTS_DIR / "env.txt"

HORNET_BUILD = WORKSPACE / "hornet" / "hornet" / "build"
HORNETSNEST_BUILD = WORKSPACE / "hornet" / "hornetsnest" / "build"
BENCHMARK_BIN = HORNET_BUILD / "hornet_paper_benchmark"
BFS2_BIN = HORNETSNEST_BUILD / "bfs2"
SPMV_BIN = HORNETSNEST_BUILD / "spmv"

NSYS_BIN = Path("/usr/local/cuda-13.3/bin/nsys")
GIT_COMMIT = "2441ccabd44a09dbf16f521dbd0bc856ab5001f8"

DATASETS = [
    ("dblp-2010", WORKSPACE / "datasets" / "dblp-2010.mtx"),
    ("in-2004", WORKSPACE / "datasets" / "in-2004.mtx"),
    ("soc-LiveJournal1", WORKSPACE / "datasets" / "soc-LiveJournal1.mtx"),
    ("cage15", WORKSPACE / "datasets" / "cage15.mtx"),
    ("kron_g500-logn21", WORKSPACE / "datasets" / "kron_g500-logn21.mtx")
]

def query_gpu_status():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.current.graphics,clocks.current.memory,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], text=True
        ).strip()
        clocks_g, clocks_m, temp, pwr = [x.strip() for x in out.split(",")]
        return {
            "gpu_clock_mhz": clocks_g,
            "mem_clock_mhz": clocks_m,
            "temp_c": temp,
            "power_w": pwr
        }
    except Exception as e:
        return {"gpu_clock_mhz": "N/A", "mem_clock_mhz": "N/A", "temp_c": "N/A", "power_w": "N/A"}

def log_event(msg):
    status = query_gpu_status()
    print(f"[{time.strftime('%X')}] [GPU: {status['gpu_clock_mhz']}MHz, {status['temp_c']}C, {status['power_w']}W] {msg}")

def ensure_binaries():
    if not BENCHMARK_BIN.exists():
        log_event(f"Building {BENCHMARK_BIN}...")
        subprocess.check_call(["make", "-j16", "hornet_paper_benchmark"], cwd=HORNET_BUILD)

def append_to_raw_csv(lines):
    with open(RAW_CSV, "a") as f:
        for line in lines:
            line = line.strip()
            if line and not line.startswith("Graph") and not line.startswith("@File") and not line.startswith("COO to CSR") and not line.startswith("[") and not line.startswith("Double"):
                f.write(line + "\n")

# ==============================================================================
# Task 4: Dynamic Insert Updates
# ==============================================================================
def run_dynamic_insert_benchmarks():
    log_event("=== RUNNING TASK 4: DYNAMIC INSERT UPDATES BENCHMARK ===")
    batch_sizes = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]

    for name, path in DATASETS:
        log_event(f"--- Graph: {name} ---")
        for b in batch_sizes:
            log_event(f"Running Insert | Graph={name} | Batch={b} | Deduplication=ON | TimedRuns=10 | Verify=ON...")
            cmd = [str(BENCHMARK_BIN), "bench_insert", str(path), str(b), "3", "10", "42", "1", "1", "1"]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                log_event(f"ERROR: bench_insert failed on {name} batch {b}!\n{proc.stderr}")
                sys.exit(1)
            append_to_raw_csv(proc.stdout.splitlines())

            # Also run a comparison trial with Dedup=OFF for Fig 6(b) comparison
            if b in [10000, 1000000]:
                log_event(f"Running Insert | Graph={name} | Batch={b} | Deduplication=OFF (GraphIn/Evograph baseline)...")
                cmd_nodedup = [str(BENCHMARK_BIN), "bench_insert", str(path), str(b), "3", "10", "42", "0", "1", "1"]
                proc_nd = subprocess.run(cmd_nodedup, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if proc_nd.returncode == 0:
                    append_to_raw_csv(proc_nd.stdout.splitlines())

# ==============================================================================
# Dynamic Delete Updates (Scope Task 4)
# ==============================================================================
def run_dynamic_delete_benchmarks():
    log_event("=== RUNNING DYNAMIC DELETE UPDATES BENCHMARK ===")
    batch_sizes = [1, 10, 100, 1000, 10000, 100000, 1000000]

    for name, path in DATASETS:
        log_event(f"--- Deletions on Graph: {name} ---")
        for b in batch_sizes:
            log_event(f"Running Delete | Graph={name} | Batch={b} | TimedRuns=10...")
            cmd = [str(BENCHMARK_BIN), "bench_delete", str(path), str(b), "3", "10", "42", "0"]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                log_event(f"ERROR: bench_delete failed on {name} batch {b}!\n{proc.stderr}")
                sys.exit(1)
            append_to_raw_csv(proc.stdout.splitlines())

# ==============================================================================
# Task 6: Initialization from CSR
# ==============================================================================
def run_initialization_benchmarks():
    log_event("=== RUNNING TASK 6: INITIALIZATION FROM CSR BENCHMARK ===")
    for name, path in DATASETS:
        log_event(f"Running Initialization | Graph={name} | TimedRuns=10...")
        cmd = [str(BENCHMARK_BIN), "bench_init", str(path), "10"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            log_event(f"ERROR: bench_init failed on {name}!\n{proc.stderr}")
            sys.exit(1)
        append_to_raw_csv(proc.stdout.splitlines())

# ==============================================================================
# Task 5: Memory Utilization Analysis
# ==============================================================================
def run_memory_benchmarks():
    log_event("=== RUNNING TASK 5: MEMORY UTILIZATION BENCHMARKS ===")
    ba_sizes = [(16, 1 << 16), (19, 1 << 19), (22, 1 << 22)]
    mem_results_file = RESULTS_DIR / "memory_results.txt"
    conf_file = WORKSPACE / "hornet" / "hornet" / "include" / "Core" / "Conf" / "MemoryManagerConf.cuh"

    with open(conf_file, "r") as f:
        orig_conf = f.read()

    with open(mem_results_file, "w") as out_f:
        # Run base memory stats
        log_event("Gathering base memory fragmentation stats...")
        for name, path in DATASETS:
            cmd = [str(BENCHMARK_BIN), "bench_memory", str(path)]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for line in proc.stdout.splitlines():
                if "[MEMORY_SUMMARY]" in line:
                    out_f.write(line + "\n")
                    log_event(f"Base Mem: {line}")

        # Run variations of EDGES_PER_BLOCKARRAY
        for exp, val in ba_sizes:
            log_event(f"Rebuilding with EDGES_PER_BLOCKARRAY = 2^{exp} ({val})...")
            new_conf = re.sub(r"const long EDGES_PER_BLOCKARRAY\s*=\s*[^;]+;",
                              f"const long EDGES_PER_BLOCKARRAY = 1 << {exp};", orig_conf)
            with open(conf_file, "w") as f:
                f.write(new_conf)

            # Recompile
            subprocess.check_call(["make", "-j16", "hornet_paper_benchmark"], cwd=HORNET_BUILD)

            for name, path in DATASETS:
                cmd = [str(BENCHMARK_BIN), "bench_memory", str(path)]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for line in proc.stdout.splitlines():
                    if "[MEMORY_SUMMARY]" in line:
                        record = line.replace("[MEMORY_SUMMARY]", f"[MEMORY_SUMMARY BA_EXP={exp}]")
                        out_f.write(record + "\n")
                        log_event(f"BA_EXP={exp} {name}: {record}")

    # Restore original configuration
    with open(conf_file, "w") as f:
        f.write(orig_conf)
    subprocess.check_call(["make", "-j16", "hornet_paper_benchmark"], cwd=HORNET_BUILD)
    log_event("Restored original configuration and rebuilt.")

# ==============================================================================
# Task 3: Instrumentation Overhead
# ==============================================================================
def run_overhead_benchmarks():
    log_event("=== RUNNING TASK 3: INSTRUMENTATION OVERHEAD BENCHMARKS ===")
    overhead_file = RESULTS_DIR / "overhead_results.txt"
    with open(overhead_file, "w") as f:
        for name, path in DATASETS:
            for b in [10000, 1000000]:
                log_event(f"Measuring overhead | Graph={name} | Batch={b} | Trials=10...")
                cmd = [str(BENCHMARK_BIN), "bench_overhead", str(path), str(b), "10"]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for line in proc.stdout.splitlines():
                    if "[OVERHEAD_SUMMARY]" in line:
                        f.write(line + "\n")
                        log_event(f"{line}")

# ==============================================================================
# Task 7: Nsight Systems Profiling
# ==============================================================================
def run_nsys_profiling():
    log_event("=== RUNNING TASK 7: NSYS PROFILING (MID & LARGE BATCH) ===")
    if not NSYS_BIN.exists():
        log_event(f"WARNING: Nsys binary {NSYS_BIN} not found. Skipping nsys trace.")
        return

    for name, path in [("dblp-2010", DATASETS[0][1]), ("in-2004", DATASETS[1][1])]:
        for b_name, b in [("mid_10k", 10000), ("large_1m", 1000000)]:
            out_prefix = RESULTS_DIR / f"nsys_{name}_{b_name}"
            log_event(f"Profiling with nsys: {name} batch={b} -> {out_prefix}...")
            cmd = [
                str(NSYS_BIN), "profile",
                "--trace=cuda,nvtx",
                "--stats=true",
                "-o", str(out_prefix),
                "--force-overwrite", "true",
                str(BENCHMARK_BIN), "bench_insert", str(path), str(b), "1", "1", "42", "1", "0", "1"
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            txt_log = RESULTS_DIR / f"nsys_{name}_{b_name}_stats.txt"
            with open(txt_log, "w") as f:
                f.write(proc.stdout + "\n" + proc.stderr)
            log_event(f"Nsys profile for {name} ({b_name}) saved.")

# ==============================================================================
# Algorithmic Benchmarks: BFS MTEPS & SpMV (Paper Fig. 7 & 8)
# ==============================================================================
def run_traversal_benchmarks():
    log_event("=== RUNNING BFS (MTEPS) AND SPMV BENCHMARKS ===")
    bfs_results_file = RESULTS_DIR / "bfs_spmv_results.txt"
    with open(bfs_results_file, "w") as out_f:
        for name, path in DATASETS:
            log_event(f"Running BFS2 on {name}...")
            cmd_bfs = [str(BFS2_BIN), str(path)]
            proc_bfs = subprocess.run(cmd_bfs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            time_ms = 0.0
            for line in proc_bfs.stdout.splitlines():
                if "TopDown2" in line:
                    m = re.search(r"TopDown2\s+([\d\.]+)\s+ms", line)
                    if m: time_ms = float(m.group(1))
            out_f.write(f"BFS2,{name},{time_ms}\n")
            log_event(f"BFS2 {name}: {time_ms} ms")

            log_event(f"Running SpMV on {name}...")
            cmd_spmv = [str(SPMV_BIN), str(path)]
            proc_spmv = subprocess.run(cmd_spmv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            spmv_ms = 0.0
            for line in proc_spmv.stdout.splitlines():
                if "SpMV" in line:
                    m = re.search(r"SpMV\s+([\d\.]+)\s+ms", line)
                    if m: spmv_ms = float(m.group(1))
            out_f.write(f"SPMV,{name},{spmv_ms}\n")
            log_event(f"SpMV {name}: {spmv_ms} ms")

def main():
    log_event("Starting full reproduction suite...")
    ensure_binaries()

    # Reset raw.csv
    with open(RAW_CSV, "w") as f:
        f.write("graph,batch_size,trial,phase,time_ms,seed,flags,commit\n")

    # Run tasks in order
    run_overhead_benchmarks()
    run_memory_benchmarks()
    run_initialization_benchmarks()
    run_dynamic_insert_benchmarks()
    run_dynamic_delete_benchmarks()
    run_nsys_profiling()
    run_traversal_benchmarks()

    log_event("=== ALL BENCHMARKS COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
