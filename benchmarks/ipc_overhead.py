#!/usr/bin/env python3
"""Per-layer IPC overhead breakdown (GATHER/SCATTER/COMPUTE per decode token)."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

import aiohttp
import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from config import DEFAULT_PORT, AGGREGATOR_HEALTH_PORT


PROMPT = ("a Lambda-class T-4a shuttle and a few TIE-fighter escorts were being "
          "prepared for Lizzie, her bodyguard Mila, and Alyssa, along with a pair "
          "of imperial Deathtroopers, and 8 regular Imperial Army Troopers.")


def wait_for_health(url, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def launch_worker(worker_id, http_port, adapter, agg_host, agg_port, log_dir, mps_pipe=None):
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.path.join(project_root, "src") + ":" + project_root
    if mps_pipe:
        env["CUDA_MPS_PIPE_DIRECTORY"] = mps_pipe
    # PROFILE_LAYERS is inherited from parent env (set by shell script)

    cmd = [
        sys.executable, "-m", "worker.worker_sync",
        "--host", agg_host,
        "--port", str(agg_port),
        "--http-port", str(http_port),
        "--device", "cuda:0",
        "--lora", adapter,
    ]

    log_file = os.path.join(log_dir, f"worker_{worker_id}.log")
    fp = open(log_file, "w")
    proc = subprocess.Popen(cmd, stdout=fp, stderr=subprocess.STDOUT, env=env)
    return proc


async def send_inference(session, port, max_tokens):
    try:
        async with session.post(
            f"http://127.0.0.1:{port}/inference",
            json={"prompt": PROMPT, "max_tokens": max_tokens,
                  "do_sample": True, "temperature": 1.0, "top_k": 50},
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            result = await resp.json()
            return result
    except Exception as e:
        return {"success": False, "error": str(e)}


async def run_concurrent(worker_ports, max_tokens):
    async with aiohttp.ClientSession() as session:
        tasks = [send_inference(session, port, max_tokens) for port in worker_ports]
        return await asyncio.gather(*tasks)


def main():
    parser = argparse.ArgumentParser(description="RQ5: IPC overhead breakdown")
    parser.add_argument("--workers", default="1,4,8,16")
    parser.add_argument("--decode-tokens", type=int, default=100)
    parser.add_argument("--adapter", default="../sim-adapters/pool-10-r16/lora-0")
    parser.add_argument("--agg-host", default="localhost")
    parser.add_argument("--agg-port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--agg-health-port", type=int, default=AGGREGATOR_HEALTH_PORT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base-port", type=int, default=6000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    worker_counts = [int(x) for x in args.workers.split(",")]
    max_workers = max(worker_counts)
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)

    gpu_idx = int(args.device.split(":")[-1]) if ":" in args.device else 0
    mps_pipe = f"/tmp/mps_{gpu_idx}"

    print("=" * 72)
    print("RQ5: Per-Layer IPC Overhead Breakdown")
    print("=" * 72)
    print(f"Adapter:       {args.adapter}")
    print(f"Worker counts: {worker_counts}")
    print(f"Decode tokens: {args.decode_tokens}")
    print(f"Repeats:       {args.repeats}")
    print(f"PROFILE_LAYERS={os.environ.get('PROFILE_LAYERS', 'NOT SET')}")
    print()

    if os.environ.get("PROFILE_LAYERS") != "1":
        print("ERROR: PROFILE_LAYERS=1 must be set in environment.")
        print("Run via: bash scripts/run_overhead.sh")
        sys.exit(1)

    # Check aggregator
    if not wait_for_health(f"http://{args.agg_host}:{args.agg_health_port}/health", timeout=10):
        print("ERROR: Aggregator not healthy")
        sys.exit(1)
    print("Aggregator healthy.\n")

    all_data = {}

    for n_workers in worker_counts:
        print(f"\n{'=' * 72}")
        print(f"  N={n_workers} concurrent workers")
        print(f"{'=' * 72}")

        # Launch N workers
        worker_procs = []
        worker_ports = []
        print(f"  Launching {n_workers} workers...")
        for i in range(n_workers):
            port = args.base_port + i * 10
            worker_ports.append(port)
            proc = launch_worker(i, port, args.adapter,
                                 args.agg_host, args.agg_port,
                                 log_dir, mps_pipe)
            worker_procs.append(proc)

        # Wait for all workers
        print(f"  Waiting for workers to be ready...")
        all_ready = True
        for i, port in enumerate(worker_ports):
            if not wait_for_health(f"http://127.0.0.1:{port}/health", timeout=120):
                print(f"  ERROR: Worker {i} (port {port}) not ready")
                all_ready = False
                break
        if not all_ready:
            for p in worker_procs:
                p.kill()
            continue

        print(f"  All {n_workers} workers ready.")

        try:
            # Warmup: 1 request per worker
            print(f"  Warming up...")
            asyncio.run(run_concurrent(worker_ports, max_tokens=5))
            time.sleep(0.5)

            log_sizes_before = {}
            for i in range(n_workers):
                log_path = os.path.join(log_dir, f"worker_{i}.log")
                log_sizes_before[log_path] = os.path.getsize(log_path) if os.path.exists(log_path) else 0

            # Profile: send N concurrent requests, repeat
            print(f"  Profiling: {args.repeats} × {n_workers} concurrent, "
                  f"{args.decode_tokens} tokens each...")
            for rep in range(args.repeats):
                results = asyncio.run(run_concurrent(worker_ports, args.decode_tokens))
                ok = sum(1 for r in results if r.get("success"))
                tok = sum(r.get("tokens", r.get("output_tokens", 0))
                          for r in results if r.get("success"))
                print(f"    rep {rep+1}/{args.repeats}: {ok}/{n_workers} ok, {tok} tokens")
            time.sleep(0.5)

            # Parse profile data from worker logs (new lines only)
            all_records = []
            for i in range(n_workers):
                log_path = os.path.join(log_dir, f"worker_{i}.log")
                offset = log_sizes_before.get(log_path, 0)
                records = []
                if os.path.exists(log_path):
                    with open(log_path) as f:
                        f.seek(offset)
                        for line in f:
                            if "[LAYER_PROFILE]" not in line:
                                continue
                            parts = {}
                            for token in line.split():
                                if "=" in token:
                                    k, v = token.split("=", 1)
                                    try:
                                        parts[k] = float(v)
                                    except ValueError:
                                        parts[k] = v
                            if "gather_us" in parts:
                                records.append(parts)
                # Skip first 5 tokens per worker (decode warmup)
                all_records.extend(records[5:] if len(records) > 5 else records)

            if not all_records:
                print(f"  WARNING: No profile data found for N={n_workers}")
                continue

            gather = np.array([r["gather_us"] for r in all_records])
            scatter = np.array([r["scatter_us"] for r in all_records])
            compute = np.array([r["compute_us"] for r in all_records])
            wall = np.array([r["wall_us"] for r in all_records])
            ipc = gather + scatter

            all_data[n_workers] = {
                "gather_ms": float(np.mean(gather) / 1000),
                "scatter_ms": float(np.mean(scatter) / 1000),
                "compute_ms": float(np.mean(compute) / 1000),
                "total_ms": float(np.mean(wall) / 1000),
                "ipc_pct": float(np.mean(ipc / wall) * 100),
                "n_samples": len(all_records),
                "n_workers": n_workers,
            }

            print(f"  Parsed {len(all_records)} decode tokens from {n_workers} workers")
            print(f"  Total: {all_data[n_workers]['total_ms']:.2f}ms  "
                  f"IPC: {all_data[n_workers]['ipc_pct']:.1f}%")

        finally:
            # Kill workers
            print(f"  Stopping {n_workers} workers...")
            for p in worker_procs:
                try:
                    p.terminate()
                except Exception:
                    pass
            time.sleep(2)
            for p in worker_procs:
                try:
                    p.kill()
                except Exception:
                    pass

    if not all_data:
        print("\nNo data collected.")
        sys.exit(1)

    # ═══ Print results table ═══
    cols = sorted(all_data.keys())
    col_w = 14

    print()
    print("=" * 72)
    print("RESULTS: Per-Token Decode Breakdown")
    print("(ms per token, summed over 32 layers + embed + LM head)")
    print("=" * 72)
    print()

    header = f"{'Component':<12}" + "".join(f"{'N=' + str(c):>{col_w}}" for c in cols)
    print(header)
    print("-" * len(header))

    for key, label in [
        ("gather_ms", "GATHER"),
        ("scatter_ms", "SCATTER"),
        ("compute_ms", "COMPUTE"),
    ]:
        row = f"{label:<12}"
        for c in cols:
            val = all_data[c][key]
            row += f"{val:>{col_w - 3}.2f} ms"
        print(row)

    print("-" * len(header))

    row = f"{'TOTAL':<12}"
    for c in cols:
        row += f"{all_data[c]['total_ms']:>{col_w - 3}.2f} ms"
    print(row)

    row = f"{'IPC %':<12}"
    for c in cols:
        row += f"{all_data[c]['ipc_pct']:>{col_w - 3}.1f}  %"
    print(row)

    row = f"{'Samples':<12}"
    for c in cols:
        row += f"{all_data[c]['n_samples']:>{col_w}}"
    print(row)

    print()
    print("GATHER  = Worker → IPC buffer P2P copy (summed over 130 barriers)")
    print("SCATTER = IPC buffer → Worker P2P copy (summed over 130 barriers)")
    print("COMPUTE = TOTAL - GATHER - SCATTER (LoRA + base GEMM + attention + norms)")
    print("IPC overhead = (GATHER + SCATTER) / TOTAL")

    # Save
    if args.output:
        output = {
            "config": {
                "adapter": args.adapter,
                "decode_tokens": args.decode_tokens,
                "repeats": args.repeats,
                "device": args.device,
            },
            "results": {str(k): v for k, v in all_data.items()},
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
