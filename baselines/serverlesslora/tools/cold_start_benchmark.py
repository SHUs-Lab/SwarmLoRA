#!/usr/bin/env python3
"""
Cold Start Benchmark for ServerlessLoRA.

Measures the end-to-end cold start time of spawning worker containers,
mimicking exactly how the system spawns workers during benchmarking.

Phases measured per worker:
  1. Spawn (Popen) to healthy (/health returns "ready")
  2. First inference latency (optional, with --run-inference)

Usage:
    python tools/cold_start_benchmark.py \
        --config deployment_config_pckp_30.yaml \
        --num-workers 4 \
        --concurrent 4 \
        --output benchmark_results/cold_start.json
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import statistics
import requests as http_requests
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def load_config(config_path):
    """Load cluster config YAML."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def spawn_and_measure(worker_id, lora_path, device, bms_port, http_port,
                      hostname="localhost", run_inference=False):
    """Spawn a single worker and measure cold start time.

    Returns dict with timing breakdown.
    """
    result = {
        "worker_id": worker_id,
        "lora": os.path.basename(lora_path),
        "device": device,
        "port": http_port,
        "spawn_start": None,
        "spawn_end": None,
        "cold_start_ms": None,
        "health_polls": 0,
        "init_breakdown": {},
        "first_inference_ms": None,
        "error": None,
    }

    cmd = [
        sys.executable, "worker_batched.py",
        "--server-host", hostname,
        "--server-port", str(bms_port),
        "--http-port", str(http_port),
        "--worker-id", str(worker_id),
        "--lora", lora_path,
        "--device", device,
        "--slo-ms", "2000",
        "--max-batch-size", "8",
        "--base-ttft-ms", "400",
        "--marginal-cost-ms", "50",
    ]

    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"cold_bench_worker_{worker_id}.log")

    worker_env = os.environ.copy()
    worker_env["OMP_NUM_THREADS"] = "1"
    worker_env["MKL_NUM_THREADS"] = "1"

    process = None
    try:
        # Phase 1: Spawn
        t_spawn = time.time()
        result["spawn_start"] = t_spawn

        worker_log = open(log_path, "w")
        process = subprocess.Popen(
            cmd,
            stdout=worker_log,
            stderr=worker_log,
            env=worker_env,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
        )

        # Phase 2: Poll until healthy
        health_url = f"http://{hostname}:{http_port}/health"
        timeout = 180.0
        deadline = time.time() + timeout
        polls = 0

        while time.time() < deadline:
            polls += 1
            try:
                resp = http_requests.get(health_url, timeout=2.0)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "ready":
                        break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            result["error"] = "Timeout waiting for healthy"
            return result

        t_ready = time.time()
        result["spawn_end"] = t_ready
        result["cold_start_ms"] = (t_ready - t_spawn) * 1000
        result["health_polls"] = polls

        # Parse init breakdown from worker log
        worker_log.flush()
        try:
            with open(log_path) as f:
                for line in f:
                    if "INIT:" in line:
                        import re
                        m = re.search(
                            r'lib_import=([\d.]+)s cuda_init=([\d.]+)s '
                            r'connect=([\d.]+)s model=([\d.]+)s '
                            r'lora=([\d.]+)s tok=([\d.]+)s TOTAL=([\d.]+)s',
                            line
                        )
                        if m:
                            result["init_breakdown"] = {
                                "lib_import_s": float(m.group(1)),
                                "cuda_init_s": float(m.group(2)),
                                "connect_s": float(m.group(3)),
                                "model_s": float(m.group(4)),
                                "lora_s": float(m.group(5)),
                                "tokenizer_s": float(m.group(6)),
                                "total_s": float(m.group(7)),
                            }
                        break
        except Exception:
            pass

        # Phase 3: First inference (optional)
        if run_inference:
            try:
                t_inf_start = time.time()
                resp = http_requests.post(
                    f"http://{hostname}:{http_port}/batch_execute",
                    json={
                        "prompts": ["Hello, world!"],
                        "max_tokens": 16,
                        "batch_id": f"cold_bench_{worker_id}",
                        "function_id": f"lora_{worker_id % 10}",
                        "arrival_times": [time.time()],
                        "dispatch_time_wall": time.time(),
                    },
                    timeout=120.0,
                )
                t_inf_end = time.time()
                if resp.status_code == 200:
                    result["first_inference_ms"] = (t_inf_end - t_inf_start) * 1000
                else:
                    result["first_inference_ms"] = None
                    result["error"] = f"Inference HTTP {resp.status_code}"
            except Exception as e:
                result["error"] = f"Inference failed: {e}"

    except Exception as e:
        result["error"] = str(e)
    finally:
        # Kill worker
        if process and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except Exception:
                process.kill()
            process.wait()

    return result


def run_benchmark(config_path, num_workers, concurrent, gpus, run_inference,
                  output_path):
    """Run cold start benchmark."""
    config = load_config(config_path)
    hostname = config.get("hostname", "localhost")
    gpu_configs = config.get("gpus", [])
    if not gpu_configs:
        print("ERROR: No GPUs in config")
        return

    # Filter to requested GPUs
    if gpus:
        gpu_configs = [g for g in gpu_configs if g["device_id"] in gpus]

    if not gpu_configs:
        print("ERROR: No matching GPUs")
        return

    # Get adapter paths from config
    functions = config.get("functions", [])
    adapters = [f["adapter"] for f in functions if f.get("adapter")]
    if not adapters:
        print("ERROR: No adapters in config")
        return

    print(f"Cold Start Benchmark")
    print(f"  Workers to spawn: {num_workers}")
    print(f"  Concurrency: {concurrent}")
    print(f"  GPUs: {[g['device_id'] for g in gpu_configs]}")
    print(f"  Adapters: {len(adapters)}")
    print(f"  Run inference: {run_inference}")
    print()

    # Build spawn tasks: round-robin across GPUs and adapters
    tasks = []
    base_port = 9000  # Use high ports to avoid conflicts
    for i in range(num_workers):
        gpu = gpu_configs[i % len(gpu_configs)]
        adapter = adapters[i % len(adapters)]
        device = f"cuda:{gpu['device_id']}"
        bms_port = gpu["base_model_server_port"]
        http_port = base_port + i
        tasks.append({
            "worker_id": i,
            "lora_path": adapter,
            "device": device,
            "bms_port": bms_port,
            "http_port": http_port,
            "hostname": hostname,
            "run_inference": run_inference,
        })

    # Run spawns
    results = []
    t_bench_start = time.time()

    print(f"Spawning {num_workers} workers ({concurrent} concurrent)...")
    with ThreadPoolExecutor(max_workers=concurrent) as pool:
        futures = {
            pool.submit(spawn_and_measure, **t): t["worker_id"]
            for t in tasks
        }
        for future in as_completed(futures):
            wid = futures[future]
            r = future.result()
            results.append(r)
            if r["error"]:
                print(f"  Worker {wid}: FAILED — {r['error']}")
            else:
                cs = r["cold_start_ms"]
                inf = r.get("first_inference_ms")
                inf_str = f", first_inf={inf:.0f}ms" if inf else ""
                print(f"  Worker {wid}: cold_start={cs:.0f}ms "
                      f"(polls={r['health_polls']}){inf_str}")

    t_bench_end = time.time()

    # Compute stats
    successful = [r for r in results if not r["error"]]
    cold_starts = [r["cold_start_ms"] for r in successful]

    print()
    print(f"{'='*60}")
    print(f"Cold Start Benchmark Results")
    print(f"{'='*60}")
    print(f"  Workers spawned: {num_workers}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(results) - len(successful)}")
    print(f"  Benchmark duration: {t_bench_end - t_bench_start:.1f}s")
    print()

    if cold_starts:
        cold_sorted = sorted(cold_starts)
        n = len(cold_sorted)
        print(f"  Cold Start (spawn to ready):")
        print(f"    Mean:  {statistics.mean(cold_starts):.0f}ms")
        print(f"    P50:   {cold_sorted[int(n*0.5)]:.0f}ms")
        print(f"    P90:   {cold_sorted[int(n*0.9)]:.0f}ms")
        print(f"    P99:   {cold_sorted[min(int(n*0.99), n-1)]:.0f}ms")
        print(f"    Min:   {min(cold_starts):.0f}ms")
        print(f"    Max:   {max(cold_starts):.0f}ms")

    # Init breakdown stats
    breakdowns = [r["init_breakdown"] for r in successful if r["init_breakdown"]]
    if breakdowns:
        print()
        print(f"  Init Phase Breakdown (from worker logs):")
        for phase in ["lib_import_s", "cuda_init_s", "connect_s", "model_s",
                       "lora_s", "tokenizer_s", "total_s"]:
            vals = [b[phase] for b in breakdowns if phase in b]
            if vals:
                vals_sorted = sorted(vals)
                nn = len(vals_sorted)
                label = phase.replace("_s", "").replace("_", " ")
                print(f"    {label:<15} mean={statistics.mean(vals):.2f}s  "
                      f"p50={vals_sorted[int(nn*0.5)]:.2f}s  "
                      f"p90={vals_sorted[int(nn*0.9)]:.2f}s")

    # First inference stats
    if run_inference:
        inf_times = [r["first_inference_ms"] for r in successful
                     if r.get("first_inference_ms")]
        if inf_times:
            inf_sorted = sorted(inf_times)
            nn = len(inf_sorted)
            print()
            print(f"  First Inference Latency:")
            print(f"    Mean:  {statistics.mean(inf_times):.0f}ms")
            print(f"    P50:   {inf_sorted[int(nn*0.5)]:.0f}ms")
            print(f"    P90:   {inf_sorted[int(nn*0.9)]:.0f}ms")

    print(f"{'='*60}")

    # Save results
    output = {
        "config": config_path,
        "num_workers": num_workers,
        "concurrent": concurrent,
        "gpus": [g["device_id"] for g in gpu_configs],
        "benchmark_duration_s": t_bench_end - t_bench_start,
        "summary": {
            "total": num_workers,
            "successful": len(successful),
            "failed": len(results) - len(successful),
        },
        "cold_start_ms": {
            "mean": statistics.mean(cold_starts) if cold_starts else None,
            "p50": cold_sorted[int(n*0.5)] if cold_starts else None,
            "p90": cold_sorted[int(n*0.9)] if cold_starts else None,
            "p99": cold_sorted[min(int(n*0.99), n-1)] if cold_starts else None,
            "min": min(cold_starts) if cold_starts else None,
            "max": max(cold_starts) if cold_starts else None,
        } if cold_starts else {},
        "results": results,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Cold Start Benchmark")
    parser.add_argument("--config", required=True,
                        help="Cluster config YAML")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers to spawn (default: 4)")
    parser.add_argument("--concurrent", type=int, default=4,
                        help="Max concurrent spawns (default: 4)")
    parser.add_argument("--gpus", type=int, nargs="*",
                        help="GPU device IDs to use (default: all from config)")
    parser.add_argument("--run-inference", action="store_true",
                        help="Run a first inference after spawn")
    parser.add_argument("--output", default="benchmark_results/cold_start.json",
                        help="Output JSON path")
    args = parser.parse_args()

    run_benchmark(
        config_path=args.config,
        num_workers=args.num_workers,
        concurrent=args.concurrent,
        gpus=args.gpus,
        run_inference=args.run_inference,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
