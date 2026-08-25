#!/usr/bin/env python3
"""Throughput scaling: measures per-worker and total throughput as worker count increases."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from typing import List, Dict, Any

import aiohttp
import requests

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from config import DEFAULT_PORT, AGGREGATOR_HEALTH_PORT


PROMPT_TEMPLATE = "a Lambda-class T-4a shuttle and a few TIE-fighter escorts were being prepared for Lizzie, her bodyguard Mila, and Alyssa, along with a pair of imperial Deathtroopers, and 8 regular Imperial Army Troopers, so that the group could land planetside. however Lizzie was currently standing on the bridge, looking out the windows, Alyssa standing next to her, and Mila standing further away, but keeping an eye on Lizzie, since she is Lizzie's bodyguard."


def wait_for_health(url: str, timeout: int = 120) -> bool:
    """Wait for a health endpoint to respond."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def launch_worker(
    worker_id: int,
    http_port: int,
    adapter: str,
    agg_host: str,
    agg_port: int,
    log_dir: str,
    mps_pipe: str = None,
) -> subprocess.Popen:
    """Launch a worker process directly."""
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.path.join(project_root, "src") + ":" + project_root
    env["CUDA_VISIBLE_DEVICES"] = "0"  # MPS remaps
    if mps_pipe:
        env["CUDA_MPS_PIPE_DIRECTORY"] = mps_pipe

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


async def send_inference(
    session: aiohttp.ClientSession,
    port: int,
    prompt: str,
    max_tokens: int,
) -> Dict[str, Any]:
    """Send inference request to a worker and measure timing."""
    t_start = time.time()
    try:
        async with session.post(
            f"http://127.0.0.1:{port}/inference",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "do_sample": True,
                "temperature": 1.0,
                "top_k": 50,
            },
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            result = await resp.json()
    except Exception as e:
        return {"success": False, "error": str(e), "elapsed_ms": 0}

    elapsed_ms = (time.time() - t_start) * 1000
    result["elapsed_ms"] = round(elapsed_ms, 1)
    return result


async def run_scaling_point(
    num_workers: int,
    worker_ports: List[int],
    prompt: str,
    max_tokens: int,
) -> Dict[str, Any]:
    """Run all workers simultaneously and collect results."""
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Send one request to each worker simultaneously
        tasks = [
            send_inference(session, port, prompt, max_tokens)
            for port in worker_ports[:num_workers]
        ]

        t_start = time.time()
        results = await asyncio.gather(*tasks)
        wall_ms = (time.time() - t_start) * 1000

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    if not successful:
        return {
            "num_workers": num_workers,
            "success": 0,
            "failed": len(failed),
            "errors": [r.get("error", "unknown") for r in failed[:3]],
        }

    # Worker response uses 'tokens' or 'completion_tokens', not 'output_tokens'
    output_tokens = [r.get("tokens", r.get("completion_tokens", r.get("output_tokens", 0))) for r in successful]
    total_output = sum(output_tokens)
    elapsed_times = [r["elapsed_ms"] for r in successful]
    ttft_times = [r.get("ttft_ms", 0) for r in successful if r.get("ttft_ms")]

    # Per-worker throughput = output_tokens / elapsed_time per worker
    per_worker_tps = [
        r.get("tokens", r.get("completion_tokens", r.get("output_tokens", 0))) / (r["elapsed_ms"] / 1000)
        for r in successful if r["elapsed_ms"] > 0
    ]

    # Decode-only throughput: tokens after the first, over the time after TTFT.
    # Referenced by decode_tps_avg below; was never defined (NameError at N>=1).
    decode_tps = []
    for r in successful:
        toks = r.get("tokens", r.get("completion_tokens", r.get("output_tokens", 0)))
        ttft = r.get("ttft_ms", 0) or 0
        decode_ms = r["elapsed_ms"] - ttft
        if toks > 1 and decode_ms > 0:
            decode_tps.append((toks - 1) / (decode_ms / 1000))

    # Total throughput = total output tokens / wall time
    total_tps = total_output / (wall_ms / 1000) if wall_ms > 0 else 0

    return {
        "num_workers": num_workers,
        "success": len(successful),
        "failed": len(failed),
        "total_output_tokens": total_output,
        "wall_ms": round(wall_ms, 1),
        "total_tps": round(total_tps, 1),
        "per_worker_tps_avg": round(sum(per_worker_tps) / len(per_worker_tps), 1) if per_worker_tps else 0,
        "per_worker_tps_min": round(min(per_worker_tps), 1) if per_worker_tps else 0,
        "per_worker_tps_max": round(max(per_worker_tps), 1) if per_worker_tps else 0,
        "ttft_avg_ms": round(sum(ttft_times) / len(ttft_times), 1) if ttft_times else 0,
        "ttft_max_ms": round(max(ttft_times), 1) if ttft_times else 0,
        "decode_tps_avg": round(sum(decode_tps) / len(decode_tps), 1) if decode_tps else 0,
        "e2e_avg_ms": round(sum(elapsed_times) / len(elapsed_times), 1),
        "e2e_max_ms": round(max(elapsed_times), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Throughput scaling benchmark")
    parser.add_argument("--workers", default="1,2,4,8,12,16,24",
                        help="Comma-separated worker counts to test (default: 1,2,4,8,12,16,24)")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Max tokens to generate per request (default: 64)")
    parser.add_argument("--adapter", default="../sim-adapters/pool-10-r16/lora-0",
                        help="Adapter to use (default: ../sim-adapters/pool-10-r16/lora-0)")
    parser.add_argument("--agg-host", default="localhost",
                        help="Aggregator host")
    parser.add_argument("--agg-port", type=int, default=DEFAULT_PORT,
                        help=f"Aggregator TCP port (default: {DEFAULT_PORT})")
    parser.add_argument("--agg-health-port", type=int, default=AGGREGATOR_HEALTH_PORT,
                        help=f"Aggregator health port (default: {AGGREGATOR_HEALTH_PORT})")
    parser.add_argument("--device", default="cuda:0",
                        help="GPU device (default: cuda:0)")
    parser.add_argument("--base-port", type=int, default=6000,
                        help="Base HTTP port for workers (default: 6000)")
    parser.add_argument("--output", default=None,
                        help="Save results to JSON file")
    parser.add_argument("--no-launch", action="store_true",
                        help="Don't launch workers (assume already running)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Number of measurement repeats per worker count (default: 3)")
    args = parser.parse_args()

    worker_counts = [int(x) for x in args.workers.split(",")]
    max_workers = max(worker_counts)
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Use fixed prompt (~34 tokens)
    prompt = PROMPT_TEMPLATE

    # GPU index for MPS
    gpu_idx = int(args.device.split(":")[-1]) if ":" in args.device else 0
    mps_pipe = f"/tmp/mps_{gpu_idx}"

    # Check aggregator health
    print(f"Checking aggregator at {args.agg_host}:{args.agg_health_port}...")
    if not wait_for_health(f"http://{args.agg_host}:{args.agg_health_port}/health", timeout=10):
        print("ERROR: Aggregator not healthy. Start it first:")
        print(f"  PYTHONPATH=src python src/aggregator.py --device {args.device} --port {args.agg_port} --health-port {args.agg_health_port}")
        sys.exit(1)
    print("Aggregator healthy.")

    # Launch max_workers workers
    worker_procs = []
    worker_ports = []

    if not args.no_launch:
        print(f"\nLaunching {max_workers} workers...")
        for i in range(max_workers):
            port = args.base_port + i * 10
            worker_ports.append(port)
            proc = launch_worker(
                worker_id=i,
                http_port=port,
                adapter=args.adapter,
                agg_host=args.agg_host,
                agg_port=args.agg_port,
                log_dir=log_dir,
                mps_pipe=mps_pipe,
            )
            worker_procs.append(proc)
            print(f"  Worker {i}: port {port}, PID {proc.pid}")

        # Wait for all workers to be ready
        print(f"Waiting for {max_workers} workers to be ready...")
        for i, port in enumerate(worker_ports):
            if not wait_for_health(f"http://127.0.0.1:{port}/health", timeout=120):
                print(f"ERROR: Worker {i} (port {port}) not ready after 120s")
                # Cleanup
                for p in worker_procs:
                    p.kill()
                sys.exit(1)
        print(f"All {max_workers} workers ready.\n")
    else:
        worker_ports = [args.base_port + i * 10 for i in range(max_workers)]

    try:
        # Warmup
        print("Warming up (1 request to first worker)...")
        async def do_warmup():
            async with aiohttp.ClientSession() as s:
                return await send_inference(s, worker_ports[0], prompt, args.max_tokens)
        warmup_result = asyncio.run(do_warmup())
        if warmup_result.get("success"):
            print(f"  Warmup OK: {warmup_result.get('tokens', 0)} tokens in {warmup_result['elapsed_ms']:.0f}ms\n")
        else:
            print(f"  Warmup failed: {warmup_result.get('error')}\n")

        # Run scaling benchmark
        all_results = []
        print("=" * 90)
        print(f"{'Workers':>8} {'OK':>4} {'Total TPS':>10} {'Per-W TPS':>10} {'Per-W Min':>10} {'Per-W Max':>10} {'TTFT avg':>10} {'E2E avg':>10}")
        print("-" * 90)

        for n in worker_counts:
            if n > max_workers:
                print(f"  Skipping {n} workers (only {max_workers} launched)")
                continue

            best_result = None
            for rep in range(args.repeats):
                result = asyncio.run(run_scaling_point(n, worker_ports, prompt, args.max_tokens))
                if best_result is None or result.get("total_tps", 0) > best_result.get("total_tps", 0):
                    best_result = result

            r = best_result
            all_results.append(r)

            if r.get("total_tps"):
                print(f"{r['num_workers']:>8} {r['success']:>4} {r['total_tps']:>9.1f} {r['per_worker_tps_avg']:>9.1f} {r['per_worker_tps_min']:>9.1f} {r['per_worker_tps_max']:>9.1f} {r['ttft_avg_ms']:>8.0f}ms {r['e2e_avg_ms']:>8.0f}ms")
            else:
                print(f"{r['num_workers']:>8} {r.get('success',0):>4}   FAILED — {r.get('errors', ['unknown'])[:1]}")

        print("=" * 90)

        # Summary
        if all_results and all_results[0].get("total_tps"):
            base_tps = all_results[0]["total_tps"]
            print(f"\n{'Workers':>8} {'Total TPS':>10} {'Speedup':>10} {'Efficiency':>10}")
            print("-" * 45)
            for r in all_results:
                if r.get("total_tps"):
                    speedup = r["total_tps"] / base_tps
                    efficiency = speedup / r["num_workers"] * 100
                    print(f"{r['num_workers']:>8} {r['total_tps']:>9.1f} {speedup:>9.2f}x {efficiency:>9.1f}%")

        # Save results
        if args.output:
            output = {
                "config": {
                    "max_tokens": args.max_tokens,
                    "adapter": args.adapter,
                    "device": args.device,
                    "repeats": args.repeats,
                },
                "results": all_results,
            }
            with open(args.output, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nResults saved to {args.output}")

    finally:
        # Cleanup workers
        if worker_procs:
            print("\nCleaning up workers...")
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


if __name__ == "__main__":
    main()
