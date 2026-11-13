#!/home/spaceeramaster/HPCproj_backup/.venv/bin/python3
"""
Master Experiment Runner for Hornet 2018 Paper Replication (Clean Baseline)
Runs the full paper benchmark suite using src/baseline/build/baseline_benchmark:
  - Fig. 4: Memory utilization across BA_EXP in {16, 19, 22, 23}
  - Sec. III-D: Initialization from CSR
  - Fig. 5 & 6: Dynamic batch insert (B = 1 .. 10^7, dedup ON/OFF)
  - Dynamic batch delete (B = 1 .. 10^7)
  - Fig. 7 & 8: BFS (MTEPS) & SpMV vs Static CSR
  - Fig. 9: Dynamic Maximal K-Truss
  - Nsight Systems (nsys) CUDA + NVTX profiling
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
NSYS_BIN = Path("/usr/local/cuda-13.3/bin/nsys")

DATASETS = [
    ("dblp-2010", WORKSPACE / "datasets" / "dblp-2010.mtx"),
    ("in-2004", WORKSPACE / "datasets" / "in-2004.mtx"),
    ("soc-LiveJournal1", WORKSPACE / "datasets" / "soc-LiveJournal1.mtx"),
    ("cage15", WORKSPACE / "datasets" / "cage15.mtx"),
    ("kron_g500-logn21", WORKSPACE / "datasets" / "kron_g500-logn21.mtx"),
]


def query_gpu_status():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=clocks.gr,clocks.mem,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        clocks_g, clocks_m, temp, pwr = [x.strip() for x in out.split(",")]
        return {
            "gpu_clock_mhz": clocks_g,
            "mem_clock_mhz": clocks_m,
            "temp_c": temp,
            "power_w": pwr,
        }
    except Exception:
        return {"gpu_clock_mhz": "N/A", "mem_clock_mhz": "N/A", "temp_c": "N/A", "power_w": "N/A"}


def log_event(msg):
    status = query_gpu_status()
    print(
        f"[{time.strftime('%X')}] [GPU: {status['gpu_clock_mhz']}MHz, {status['temp_c']}C, {status['power_w']}W] {msg}",
        flush=True,
    )


def ensure_binaries(mode: str) -> Path:
    if mode == "baseline":
        build_dir = WORKSPACE / "src" / "baseline" / "build"
        bin_path = build_dir / "baseline_benchmark"
        if not bin_path.exists():
            log_event(f"Building {bin_path}...")
            subprocess.check_call(
                [
                    "cmake",
                    "-S",
                    str(WORKSPACE / "src" / "baseline"),
                    "-B",
                    str(build_dir),
                    "-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc",
                ]
            )
            subprocess.check_call(["cmake", "--build", str(build_dir), "-j8"])
        return bin_path
    raise ValueError(f"Unsupported mode: {mode}")


def append_to_raw_csv(raw_csv: Path, lines):
    with open(raw_csv, "a") as f:
        for line in lines:
            line = line.strip()
            if (
                line
                and not line.startswith("Graph")
                and not line.startswith("@File")
                and not line.startswith("COO to CSR")
                and not line.startswith("[")
                and not line.startswith("Double")
            ):
                f.write(line + "\n")


# ==============================================================================
# Fig. 4: Memory Utilization Analysis
# ==============================================================================
def run_memory_benchmarks(bin_path: Path, results_dir: Path):
    log_event("=== RUNNING FIG. 4: MEMORY UTILIZATION BENCHMARKS ===")
    ba_exps = [16, 19, 22, 23]
    mem_results_file = results_dir / "memory_results.txt"

    with open(mem_results_file, "w") as out_f:
        for exp in ba_exps:
            for name, path in DATASETS:
                cmd = [str(bin_path), "bench_memory", str(path), str(exp)]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if proc.returncode != 0:
                    log_event(f"WARNING: bench_memory failed on {name} BA_EXP={exp}: {proc.stderr.strip()}")
                    continue
                for line in proc.stdout.splitlines():
                    if "[MEMORY_SUMMARY" in line:
                        out_f.write(line + "\n")
                        log_event(f"BA_EXP={exp} {name}: {line}")


# ==============================================================================
# Sec. III-D: Initialization from CSR
# ==============================================================================
def run_initialization_benchmarks(bin_path: Path, raw_csv: Path):
    log_event("=== RUNNING SEC. III-D: INITIALIZATION FROM CSR BENCHMARK ===")
    for name, path in DATASETS:
        log_event(f"Running Initialization | Graph={name} | TimedRuns=10...")
        cmd = [str(bin_path), "bench_init", str(path), "10"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            log_event(f"WARNING: bench_init failed on {name}: {proc.stderr.strip()}")
            continue
        append_to_raw_csv(raw_csv, proc.stdout.splitlines())


# ==============================================================================
# Fig. 5 & 6: Dynamic Insert Updates
# ==============================================================================
def run_dynamic_insert_benchmarks(bin_path: Path, raw_csv: Path, verify_small: bool = True):
    log_event("=== RUNNING FIG. 5 & 6: DYNAMIC INSERT UPDATES BENCHMARK ===")
    batch_sizes = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]

    for name, path in DATASETS:
        log_event(f"--- Graph: {name} ---")
        # Only run full 10-trial CPU verification on smaller graphs to keep suite fast
        verify_flag = "1" if (verify_small and name in ("dblp-2010", "in-2004")) else "0"
        for b in batch_sizes:
            v_flag = verify_flag if b <= 1000000 else "0"
            log_event(f"Running Insert | Graph={name} | Batch={b} | Dedup=ON | TimedRuns=10 | Verify={v_flag}...")
            cmd = [str(bin_path), "bench_insert", str(path), str(b), "3", "10", "42", "1", v_flag, "1", "23"]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                log_event(f"WARNING: bench_insert failed on {name} batch {b}: {proc.stderr.strip()}")
                continue
            append_to_raw_csv(raw_csv, proc.stdout.splitlines())

            # Fig 6(b) comparison: Dedup=OFF (intra-batch dedup only)
            if b in [10000, 1000000]:
                log_event(f"Running Insert | Graph={name} | Batch={b} | Dedup=OFF...")
                cmd_nd = [str(bin_path), "bench_insert", str(path), str(b), "3", "10", "42", "0", "0", "1", "23"]
                proc_nd = subprocess.run(cmd_nd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if proc_nd.returncode == 0:
                    append_to_raw_csv(raw_csv, proc_nd.stdout.splitlines())
                else:
                    log_event(f"WARNING: bench_insert (dedup=0) failed on {name} batch {b}: {proc_nd.stderr.strip()}")


# ==============================================================================
# Dynamic Delete Updates
# ==============================================================================
def run_dynamic_delete_benchmarks(bin_path: Path, raw_csv: Path, verify_small: bool = True):
    log_event("=== RUNNING DYNAMIC DELETE UPDATES BENCHMARK ===")
    batch_sizes = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]

    for name, path in DATASETS:
        log_event(f"--- Deletions on Graph: {name} ---")
        verify_flag = "1" if (verify_small and name == "dblp-2010") else "0"
        for b in batch_sizes:
            v_flag = verify_flag if b <= 1000000 else "0"
            log_event(f"Running Delete | Graph={name} | Batch={b} | TimedRuns=10 | Verify={v_flag}...")
            cmd = [str(bin_path), "bench_delete", str(path), str(b), "3", "10", "42", "1", v_flag]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                log_event(f"WARNING: bench_delete failed on {name} batch {b}: {proc.stderr.strip()}")
                continue
            append_to_raw_csv(raw_csv, proc.stdout.splitlines())


# ==============================================================================
# Fig. 7 & 8: BFS (MTEPS) & SpMV vs Static CSR
# ==============================================================================
def run_traversal_benchmarks(bin_path: Path, results_dir: Path):
    log_event("=== RUNNING FIG. 7 & 8: BFS (MTEPS) AND SPMV VS STATIC CSR ===")
    bfs_results_file = results_dir / "bfs_spmv_results.txt"
    with open(bfs_results_file, "w") as out_f:
        for name, path in DATASETS:
            log_event(f"Running BFS vs CSR on {name}...")
            cmd_bfs = [str(bin_path), "bench_bfs", str(path), "0", "5"]
            proc_bfs = subprocess.run(cmd_bfs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc_bfs.returncode != 0:
                log_event(f"WARNING: bench_bfs failed on {name}: {proc_bfs.stderr.strip()}")
            else:
                for line in proc_bfs.stdout.splitlines():
                    if line.startswith("BFS,"):
                        out_f.write(line + "\n")
                        log_event(line)

            log_event(f"Running SpMV vs CSR on {name}...")
            cmd_spmv = [str(bin_path), "bench_spmv", str(path), "10"]
            proc_spmv = subprocess.run(cmd_spmv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc_spmv.returncode != 0:
                log_event(f"WARNING: bench_spmv failed on {name}: {proc_spmv.stderr.strip()}")
            else:
                for line in proc_spmv.stdout.splitlines():
                    if line.startswith("SPMV,"):
                        out_f.write(line + "\n")
                        log_event(line)


# ==============================================================================
# Fig. 9: Dynamic Maximal K-Truss
# ==============================================================================
def run_ktruss_benchmarks(bin_path: Path, results_dir: Path):
    log_event("=== RUNNING FIG. 9: DYNAMIC MAXIMAL K-TRUSS ===")
    ktruss_file = results_dir / "ktruss_results.txt"
    with open(ktruss_file, "w") as out_f:
        for name, path in DATASETS:
            log_event(f"Running Dynamic Maximal K-Truss on {name} (max_k=50)...")
            cmd = [str(bin_path), "bench_ktruss", str(path), "50"]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                log_event(f"WARNING: bench_ktruss failed on {name}: {proc.stderr.strip()}")
                continue
            for line in proc.stdout.splitlines():
                if line.startswith("KTRUSS_STEP,") or line.startswith("KTRUSS_SUMMARY,"):
                    out_f.write(line + "\n")
                if line.startswith("KTRUSS_SUMMARY,"):
                    log_event(line)


# ==============================================================================
# Nsight Systems Profiling
# ==============================================================================
def run_nsys_profiling(bin_path: Path, results_dir: Path):
    log_event("=== RUNNING NSYS PROFILING (MID & LARGE BATCH) ===")
    if not NSYS_BIN.exists():
        log_event(f"WARNING: Nsys binary {NSYS_BIN} not found. Skipping nsys trace.")
        return

    for name, path in [("dblp-2010", DATASETS[0][1]), ("in-2004", DATASETS[1][1])]:
        for b_name, b in [("mid_10k", 10000), ("large_1m", 1000000)]:
            out_prefix = results_dir / f"nsys_{name}_{b_name}"
            log_event(f"Profiling with nsys: {name} batch={b} -> {out_prefix}...")
            cmd = [
                str(NSYS_BIN),
                "profile",
                "--trace=cuda,nvtx",
                "--stats=true",
                "-o",
                str(out_prefix),
                "--force-overwrite",
                "true",
                str(bin_path),
                "bench_insert",
                str(path),
                str(b),
                "1",
                "1",
                "42",
                "1",
                "0",
                "1",
                "23",
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            txt_log = results_dir / f"nsys_{name}_{b_name}_stats.txt"
            with open(txt_log, "w") as f:
                f.write(proc.stdout + "\n" + proc.stderr)
            log_event(f"Nsys profile for {name} ({b_name}) saved.")


def main():
    parser = argparse.ArgumentParser(description="Run Hornet 2018 paper replication experiments.")
    parser.add_argument("--mode", default="baseline", choices=["baseline"], help="Implementation mode")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["memory", "init", "insert", "delete", "traversal", "ktruss", "nsys"],
        choices=["memory", "init", "insert", "delete", "traversal", "ktruss", "nsys"],
        help="Subset of benchmark tasks to run",
    )
    parser.add_argument("--no-verify", action="store_true", help="Disable CPU verification during insert/delete")
    args = parser.parse_args()

    results_dir = WORKSPACE / "results" / args.mode
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = results_dir / "raw.csv"

    log_event(f"Starting Hornet paper replication suite (mode={args.mode}, out={results_dir})...")
    bin_path = ensure_binaries(args.mode)

    if any(t in args.tasks for t in ("init", "insert", "delete")) or not raw_csv.exists():
        with open(raw_csv, "w") as f:
            f.write("graph,batch_size,trial,phase,time_ms,seed,flags,commit\n")

    if "memory" in args.tasks:
        run_memory_benchmarks(bin_path, results_dir)
    if "init" in args.tasks:
        run_initialization_benchmarks(bin_path, raw_csv)
    if "insert" in args.tasks:
        run_dynamic_insert_benchmarks(bin_path, raw_csv, verify_small=not args.no_verify)
    if "delete" in args.tasks:
        run_dynamic_delete_benchmarks(bin_path, raw_csv, verify_small=not args.no_verify)
    if "traversal" in args.tasks:
        run_traversal_benchmarks(bin_path, results_dir)
    if "ktruss" in args.tasks:
        run_ktruss_benchmarks(bin_path, results_dir)
    if "nsys" in args.tasks:
        run_nsys_profiling(bin_path, results_dir)

    log_event("=== ALL BENCHMARKS COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
