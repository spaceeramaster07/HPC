#!/home/spaceeramaster/HPCproj_backup/.venv/bin/python3
"""
Master Experiment Runner for Hornet 2018 Paper Replication & Improved Benchmarks
Runs the full paper benchmark suite using src/<mode>/build/<mode>_benchmark:
  - Fig. 4: Memory utilization across BA_EXP in {16, 19, 22, 23}
  - Sec. III-D: Initialization from CSR
  - Fig. 5 & 6: Dynamic batch insert (B = 1 .. 10^7, dedup ON/OFF)
  - Dynamic batch delete (B = 1 .. 10^7)
  - Fig. 7 & 8: BFS (MTEPS) & SpMV vs Static CSR
  - Fig. 9: Dynamic Maximal K-Truss
  - Nsight Systems (nsys) CUDA + NVTX profiling
"""

import argparse
from contextlib import ExitStack
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
    if mode not in ("baseline", "improved"):
        raise ValueError(f"Unsupported mode: {mode}")
    src_dir = WORKSPACE / "src" / mode
    build_dir = src_dir / "build"
    bin_path = build_dir / f"{mode}_benchmark"
    if not (build_dir / "CMakeCache.txt").exists() or not bin_path.exists():
        log_event(f"Building {bin_path}...")
        subprocess.check_call(
            [
                "cmake",
                "-S",
                str(src_dir),
                "-B",
                str(build_dir),
                "-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc",
            ]
        )
        subprocess.check_call(["cmake", "--build", str(build_dir), "-j8"])
    elif mode == "improved":
        log_event(f"Updating build for {bin_path}...")
        subprocess.check_call(["cmake", "--build", str(build_dir), "-j8"])
    return bin_path


def _is_raw_csv_row(line: str) -> bool:
    return (
        bool(line)
        and line.count(",") == 7
        and not line.startswith(("Graph", "@File", "COO to CSR", "[", "Double", "BFS,", "SPMV,", "DYNAMIC_TC,", "KTRUSS_"))
    )


def _route_suite_line(line: str, out_files: dict):
    line = line.strip()
    if not line:
        return
    if line.startswith("[MEMORY_SUMMARY"):
        if "memory" in out_files:
            out_files["memory"].write(line + "\n")
            out_files["memory"].flush()
        log_event(line)
    elif line.startswith(("BFS,", "SPMV,")):
        if "traversal" in out_files:
            out_files["traversal"].write(line + "\n")
            out_files["traversal"].flush()
        log_event(line)
    elif line.startswith("DYNAMIC_TC,"):
        if "tc" in out_files:
            out_files["tc"].write(line + "\n")
            out_files["tc"].flush()
        log_event(line)
    elif line.startswith(("KTRUSS_STEP,", "KTRUSS_SUMMARY,")):
        if "ktruss" in out_files:
            out_files["ktruss"].write(line + "\n")
            out_files["ktruss"].flush()
        if line.startswith("KTRUSS_SUMMARY,"):
            log_event(line)
    elif line.startswith(("[VRAM", "[VERIFY", "[KTRUSS_SKIP")):
        log_event(line)
    elif "raw_csv" in out_files and _is_raw_csv_row(line):
        out_files["raw_csv"].write(line + "\n")
        parts = line.split(",")
        if parts[2] == "0" and parts[3] in ("init_total", "insert_total", "delete_total"):
            out_files["raw_csv"].flush()
            log_event(f"{parts[0]} | {parts[3]} | batch={parts[1]} | flags={parts[6]} | trial0={parts[4]}ms")


def run_suite_benchmarks(bin_path: Path, results_dir: Path, raw_csv: Path, suite_tasks: list, verify_small: bool):
    """Runs all non-nsys benchmarks for each graph via a single bench_suite process load."""
    tasks_csv = ",".join(suite_tasks)
    verify_flag = "1" if verify_small else "0"
    log_event(f"=== RUNNING SINGLE-LOAD SUITE ({tasks_csv}, verify_small={verify_flag}) ===")

    with ExitStack() as stack:
        out_files = {}
        if "memory" in suite_tasks:
            out_files["memory"] = stack.enter_context(open(results_dir / "memory_results.txt", "w"))
        if any(t in suite_tasks for t in ("init", "insert", "delete")):
            out_files["raw_csv"] = stack.enter_context(open(raw_csv, "a"))
        if "traversal" in suite_tasks:
            out_files["traversal"] = stack.enter_context(open(results_dir / "bfs_spmv_results.txt", "w"))
        if "ktruss" in suite_tasks:
            out_files["ktruss"] = stack.enter_context(open(results_dir / "ktruss_results.txt", "w"))
        if "tc" in suite_tasks:
            out_files["tc"] = stack.enter_context(open(results_dir / "tc_results.txt", "w"))

        for name, path in DATASETS:
            per_graph_tasks = [
                t for t in suite_tasks
                if not (name == "kron_g500-logn21" and t == "ktruss")
            ]
            if not per_graph_tasks:
                continue
            per_graph_csv = ",".join(per_graph_tasks)
            log_event(f"--- Loading {name} once for [{per_graph_csv}] ---")
            cmd = [str(bin_path), "bench_suite", str(path), verify_flag, per_graph_csv]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            assert proc.stdout is not None and proc.stderr is not None
            for line in proc.stdout:
                _route_suite_line(line, out_files)
            stderr_text = proc.stderr.read().strip()
            rc = proc.wait()
            if rc != 0:
                log_event(f"WARNING: bench_suite exited with code {rc} on {name}: {stderr_text}")


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
    parser.add_argument("--mode", default="baseline", choices=["baseline", "improved"], help="Implementation mode")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["memory", "init", "insert", "delete", "traversal", "ktruss", "tc", "nsys"],
        choices=["memory", "init", "insert", "delete", "traversal", "ktruss", "tc", "nsys"],
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

    suite_tasks = [t for t in args.tasks if t != "nsys"]
    if suite_tasks:
        run_suite_benchmarks(bin_path, results_dir, raw_csv, suite_tasks, verify_small=not args.no_verify)
    if "nsys" in args.tasks:
        run_nsys_profiling(bin_path, results_dir)

    log_event("=== ALL BENCHMARKS COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
