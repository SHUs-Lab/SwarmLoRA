#!/usr/bin/env python3
"""Measures the two SwarmLoRA cold-start numbers in Fig. 8: sequential worker
spawn (with per-phase init breakdown) and sequential adapter hot-swap. Requires
a running controller. "both" mode spawns N workers, times each spawn, then
hot-swaps those same workers and times each swap -- nothing else."""

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from typing import List, Dict, Any

import aiohttp

_INIT_TIMING_RE = re.compile(
    r"INIT TIMING: connect=([\d.]+)s cuda_init=([\d.]+)s client=([\d.]+)s "
    r"model=([\d.]+)s tokenizer=([\d.]+)s engine=([\d.]+)s TOTAL=([\d.]+)s"
)


def _parse_init_breakdown(worker_id) -> Dict[str, float]:
    """Read the worker's own log for its 'INIT TIMING' line (worker_sync.py's
    initialize()) and return the per-phase breakdown, mirroring how
    ServerlessLoRA's cold_start_benchmark.py extracts the same kind of data.

    Worker IDs are recycled (controller picks the lowest free ID) and the log
    file is opened in append mode and never truncated, so a reused ID's log
    can contain INIT TIMING lines from an earlier occupant. Take the last
    match, not the first -- appends are chronological, so it's always the
    current spawn's line."""
    log_path = os.path.join("logs", f"worker_{worker_id}.log")
    try:
        with open(log_path) as f:
            text = f.read()
    except OSError:
        return {}
    matches = list(_INIT_TIMING_RE.finditer(text))
    if not matches:
        return {}
    m = matches[-1]
    return {
        "connect_s": float(m.group(1)),
        "cuda_init_s": float(m.group(2)),
        "client_s": float(m.group(3)),
        "model_s": float(m.group(4)),
        "tokenizer_s": float(m.group(5)),
        "engine_s": float(m.group(6)),
        "total_s": float(m.group(7)),
    }


def _avg_breakdown(results: List[Dict]) -> Dict[str, float]:
    """Average the per-phase init_breakdown across spawns that have one."""
    breakdowns = [r["init_breakdown"] for r in results if r.get("init_breakdown")]
    if not breakdowns:
        return {}
    phases = ["connect_s", "cuda_init_s", "client_s", "model_s", "tokenizer_s", "engine_s", "total_s"]
    return {phase: round(statistics.mean(b[phase] for b in breakdowns), 3) for phase in phases}


async def get_workers(session: aiohttp.ClientSession, api_url: str) -> List[Dict]:
    """Get list of current workers from controller."""
    async with session.get(f"{api_url}/workers") as resp:
        data = await resp.json()
        return data.get("workers", [])


async def spawn_single(session: aiohttp.ClientSession, api_url: str, adapter_id: str) -> Dict[str, Any]:
    """Spawn a single worker and measure time."""
    start = time.time()
    async with session.post(
        f"{api_url}/workers/spawn",
        json={"adapter_id": adapter_id, "count": 1},
    ) as resp:
        result = await resp.json()
    elapsed_ms = (time.time() - start) * 1000

    success = result.get("success", False)
    spawned = result.get("spawned", [])
    worker_id = spawned[0]["worker_id"] if spawned else None

    return {
        "success": success,
        "worker_id": worker_id,
        "spawn_ms": round(elapsed_ms, 1),
        "init_breakdown": _parse_init_breakdown(worker_id) if worker_id is not None else {},
    }


async def swap_single(
    session: aiohttp.ClientSession, api_url: str, worker_id: int, adapter_id: str
) -> Dict[str, Any]:
    """Swap a single worker to a new adapter and measure time."""
    start = time.time()
    async with session.post(
        f"{api_url}/workers/{worker_id}/swap",
        json={"adapter_id": adapter_id},
    ) as resp:
        result = await resp.json()
    elapsed_ms = (time.time() - start) * 1000

    return {
        "success": result.get("success", False),
        "worker_id": worker_id,
        "swap_ms": round(elapsed_ms, 1),
    }


async def stop_worker(session: aiohttp.ClientSession, api_url: str, worker_id: int):
    """Stop a worker."""
    try:
        async with session.post(f"{api_url}/workers/{worker_id}/stop") as resp:
            await resp.json()
    except Exception:
        pass


def print_stats(label: str, times_ms: List[float]):
    """Print statistics for a list of timing measurements."""
    if not times_ms:
        print(f"  {label}: no data")
        return
    times_ms.sort()
    n = len(times_ms)
    p50_idx = int(n * 0.5)
    p90_idx = min(int(n * 0.9), n - 1)
    p99_idx = min(int(n * 0.99), n - 1)
    print(f"  {label} (n={n}):")
    print(f"    avg:  {statistics.mean(times_ms):>8.1f} ms")
    print(f"    min:  {min(times_ms):>8.1f} ms")
    print(f"    p50:  {times_ms[p50_idx]:>8.1f} ms")
    print(f"    p90:  {times_ms[p90_idx]:>8.1f} ms")
    print(f"    p99:  {times_ms[p99_idx]:>8.1f} ms")
    print(f"    max:  {max(times_ms):>8.1f} ms")


async def benchmark_spawn(
    session: aiohttp.ClientSession,
    api_url: str,
    adapter_prefix: str,
    count: int,
) -> Dict[str, Any]:
    """Measure sequential worker spawn (Fig. 8 'spawn' bar).

    Spawns `count` workers one at a time, timing each POST /workers/spawn
    round-trip and capturing the per-phase init breakdown. Workers are left
    running so the swap benchmark can reuse them (no redundant respawn) --
    the caller stops them when done."""
    adapter_id = f"{adapter_prefix}0"
    print(f"\n{'='*60}")
    print(f"  SPAWN BENCHMARK: {count} workers with {adapter_id}")
    print(f"{'='*60}")

    seq_results = []
    for i in range(count):
        r = await spawn_single(session, api_url, adapter_id)
        status = "OK" if r["success"] else "FAIL"
        print(f"    [{i+1}/{count}] worker {r['worker_id']}: {r['spawn_ms']:.0f}ms [{status}]")
        seq_results.append(r)

    seq_times = [r["spawn_ms"] for r in seq_results if r["success"]]
    print_stats("Sequential spawn", seq_times)

    worker_ids = [r["worker_id"] for r in seq_results if r["worker_id"] is not None]
    return {
        "sequential": {
            "count": len(seq_times),
            "times_ms": seq_times,
            "avg_ms": round(statistics.mean(seq_times), 1) if seq_times else 0,
            "p50_ms": round(sorted(seq_times)[len(seq_times)//2], 1) if seq_times else 0,
            "p90_ms": round(sorted(seq_times)[min(int(len(seq_times)*0.9), len(seq_times)-1)], 1) if seq_times else 0,
            "init_breakdown_avg": _avg_breakdown(seq_results),
        },
        "worker_ids": worker_ids,
    }


async def benchmark_swap(
    session: aiohttp.ClientSession,
    api_url: str,
    adapter_prefix: str,
    count: int,
    worker_ids: List[int] = None,
) -> Dict[str, Any]:
    """Benchmark adapter swap times."""
    # Get existing workers or spawn some
    if not worker_ids:
        workers = await get_workers(session, api_url)
        ready = [w for w in workers if w.get("status") == "ready"]
        if len(ready) < count:
            print(f"  Need {count} workers but only {len(ready)} ready. Spawning {count - len(ready)} more...")
            for _ in range(count - len(ready)):
                r = await spawn_single(session, api_url, f"{adapter_prefix}0")
                if r["success"]:
                    ready.append({"worker_id": r["worker_id"], "adapter_id": f"{adapter_prefix}0"})
            await asyncio.sleep(1)
        worker_ids = [w["worker_id"] for w in ready[:count]]

    if not worker_ids:
        print("  ERROR: No workers available for swap benchmark")
        return {}

    # Use two adapters to swap between
    adapter_a = f"{adapter_prefix}0"
    adapter_b = f"{adapter_prefix}1"

    print(f"\n{'='*60}")
    print(f"  SWAP BENCHMARK: {len(worker_ids)} workers, {adapter_a} -> {adapter_b}")
    print(f"{'='*60}")

    # --- Sequential swaps (Fig. 8 'hot swap' bar) ---
    print(f"\n  Sequential swaps (1 at a time):")
    seq_results = []
    for i, wid in enumerate(worker_ids):
        r = await swap_single(session, api_url, wid, adapter_b)  # swap to a different adapter
        status = "OK" if r["success"] else "FAIL"
        print(f"    [{i+1}/{len(worker_ids)}] worker {wid}: {r['swap_ms']:.0f}ms [{status}]")
        seq_results.append(r)

    seq_times = [r["swap_ms"] for r in seq_results if r["success"]]
    print_stats("Sequential swap", seq_times)

    return {
        "sequential": {
            "count": len(seq_times),
            "times_ms": seq_times,
            "avg_ms": round(statistics.mean(seq_times), 1) if seq_times else 0,
            "p50_ms": round(sorted(seq_times)[len(seq_times)//2], 1) if seq_times else 0,
            "p90_ms": round(sorted(seq_times)[min(int(len(seq_times)*0.9), len(seq_times)-1)], 1) if seq_times else 0,
        },
    }


async def main():
    parser = argparse.ArgumentParser(description="Cold start benchmark")
    parser.add_argument("--mode", choices=["spawn", "swap", "both"], default="both",
                        help="What to benchmark (default: both)")
    parser.add_argument("--count", type=int, default=5,
                        help="Number of workers to spawn/swap (default: 5)")
    parser.add_argument("--api-url", default="http://localhost:8344",
                        help="Controller API URL")
    parser.add_argument("--adapter-prefix", default="../sim-adapters/pool-10-r16/lora-",
                        help="Adapter path prefix")
    parser.add_argument("--output", default=None,
                        help="Save results to JSON file")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Don't cleanup spawned workers after spawn benchmark")
    args = parser.parse_args()

    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Verify controller is up
        try:
            async with session.get(f"{args.api_url}/health") as resp:
                if resp.status != 200:
                    print(f"ERROR: Controller not healthy at {args.api_url}")
                    sys.exit(1)
        except Exception as e:
            print(f"ERROR: Cannot reach controller at {args.api_url}: {e}")
            sys.exit(1)

        results = {"mode": args.mode, "count": args.count}
        worker_ids = []

        if args.mode in ("spawn", "both"):
            spawn_result = await benchmark_spawn(
                session, args.api_url, args.adapter_prefix, args.count,
            )
            worker_ids = spawn_result.pop("worker_ids", [])
            results["spawn"] = spawn_result

        if args.mode in ("swap", "both"):
            # In "both" mode reuse the just-spawned workers (no respawn); in
            # "swap" mode benchmark_swap spawns its own.
            swap_result = await benchmark_swap(
                session, args.api_url, args.adapter_prefix, args.count,
                worker_ids=worker_ids if args.mode == "both" else None,
            )
            results["swap"] = swap_result

        # Cleanup all workers we spawned
        if worker_ids and not args.no_cleanup:
            print(f"\n  Cleaning up {len(worker_ids)} workers...")
            for wid in worker_ids:
                await stop_worker(session, api_url=args.api_url, worker_id=wid)

        # Summary
        print(f"\n{'='*60}")
        print(f"  SUMMARY")
        print(f"{'='*60}")
        if "spawn" in results:
            s = results["spawn"]["sequential"]
            print(f"  Spawn:  avg={s['avg_ms']:.0f}ms  p50={s['p50_ms']:.0f}ms  p90={s['p90_ms']:.0f}ms")
        if "swap" in results:
            s = results["swap"]["sequential"]
            print(f"  Swap:   avg={s['avg_ms']:.0f}ms  p50={s['p50_ms']:.0f}ms  p90={s['p90_ms']:.0f}ms")

        if args.output:
            # Drop raw per-sample arrays for a compact file
            for key in ("spawn", "swap"):
                if key in results and "sequential" in results[key]:
                    results[key]["sequential"].pop("times_ms", None)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
