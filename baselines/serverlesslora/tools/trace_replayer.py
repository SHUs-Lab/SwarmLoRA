#!/usr/bin/env python3
"""Trace replayer for ServerlessLoRA experiments.

Replays request traces (synthetic or from CSV) against the controller's
/inference endpoint and records per-request metrics.

Synthetic generators (matching paper's evaluation patterns):
  - poisson:  Predictable arrivals  (CoV ~ 1)
  - bursty:   Bursty arrivals      (CoV > 4, on/off periods)
  - constant: Deterministic rate

Usage:
    # Synthetic Poisson
    python trace_replayer.py --synthetic poisson --rate 5.0 --duration 300 \\
        --functions llama8b_lora_A,llama8b_lora_B \\
        --controller http://localhost:8000 --output results/poisson_run.json

    # Bursty pattern
    python trace_replayer.py --synthetic bursty --rate 20.0 --duration 300 \\
        --burst-on-s 10 --burst-off-s 30 \\
        --functions llama8b_lora_A,llama8b_lora_B \\
        --controller http://localhost:8000 --output results/bursty_run.json

    # Replay from CSV file
    python trace_replayer.py --trace traces/azure.csv \\
        --controller http://localhost:8000 --output results/azure_run.json
"""

import argparse
import csv
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import requests

DEFAULT_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a short poem about the ocean.",
    "What are the main causes of climate change?",
    "Describe the process of photosynthesis.",
    "Summarize the plot of Romeo and Juliet.",
    "What is machine learning and how does it work?",
    "Explain the theory of relativity briefly.",
    "Write a haiku about autumn leaves.",
    "What are the benefits of regular exercise?",
    "Describe how a computer processor works.",
]


@dataclass
class TraceRequest:
    """A single request in the trace."""
    timestamp_ms: float
    function_id: str
    prompt: str
    max_tokens: int = 256


@dataclass
class RequestResult:
    """Result of a single inference request."""
    request_idx: int
    function_id: str
    submit_time: float
    ttft_ms: float
    e2e_ms: float
    tokens_generated: int
    success: bool
    error: str = ""
    batch_size: int = 1
    queue_wait_ms: float = 0.0
    container_id: str = ""
    is_cold_start: bool = False
    tpot_ms: float = 0.0


# ---------------------------------------------------------------
# Trace generators
# ---------------------------------------------------------------

class PoissonGenerator:
    """Poisson arrivals (CoV ~ 1)."""

    def __init__(self, rate: float, duration: float, functions: List[str],
                 max_tokens: int = 256):
        self.rate = rate
        self.duration = duration
        self.functions = functions
        self.max_tokens = max_tokens

    def generate(self) -> List[TraceRequest]:
        trace = []
        t = 0.0
        while t < self.duration * 1000:  # duration in ms
            gap = random.expovariate(self.rate) * 1000  # seconds -> ms
            t += gap
            if t >= self.duration * 1000:
                break
            trace.append(TraceRequest(
                timestamp_ms=t,
                function_id=random.choice(self.functions),
                prompt=random.choice(DEFAULT_PROMPTS),
                max_tokens=self.max_tokens,
            ))
        return trace


class BurstyGenerator:
    """Bursty arrivals (CoV > 4): alternating high-rate and silent periods."""

    def __init__(self, rate: float, duration: float, functions: List[str],
                 burst_on_s: float = 10.0, burst_off_s: float = 30.0,
                 max_tokens: int = 256):
        self.rate = rate
        self.duration = duration
        self.functions = functions
        self.burst_on_s = burst_on_s
        self.burst_off_s = burst_off_s
        self.max_tokens = max_tokens

    def generate(self) -> List[TraceRequest]:
        trace = []
        t = 0.0
        cycle = self.burst_on_s + self.burst_off_s
        while t < self.duration * 1000:
            cycle_pos = (t / 1000) % cycle
            if cycle_pos < self.burst_on_s:
                # Burst ON: generate at high rate
                gap = random.expovariate(self.rate) * 1000
                t += gap
                if t >= self.duration * 1000:
                    break
                trace.append(TraceRequest(
                    timestamp_ms=t,
                    function_id=random.choice(self.functions),
                    prompt=random.choice(DEFAULT_PROMPTS),
                    max_tokens=self.max_tokens,
                ))
            else:
                # Burst OFF: skip ahead to next burst
                remaining_off = (cycle - cycle_pos) * 1000
                t += remaining_off
        return trace


class ConstantRateGenerator:
    """Deterministic constant-rate arrivals."""

    def __init__(self, rate: float, duration: float, functions: List[str],
                 max_tokens: int = 256):
        self.rate = rate
        self.duration = duration
        self.functions = functions
        self.max_tokens = max_tokens

    def generate(self) -> List[TraceRequest]:
        trace = []
        interval_ms = 1000.0 / self.rate
        t = 0.0
        idx = 0
        while t < self.duration * 1000:
            trace.append(TraceRequest(
                timestamp_ms=t,
                function_id=self.functions[idx % len(self.functions)],
                prompt=DEFAULT_PROMPTS[idx % len(DEFAULT_PROMPTS)],
                max_tokens=self.max_tokens,
            ))
            t += interval_ms
            idx += 1
        return trace


# ---------------------------------------------------------------
# Trace loaders
# ---------------------------------------------------------------

def load_trace_csv(path: str) -> List[TraceRequest]:
    """Load trace from CSV: timestamp_ms,function_id,prompt,max_tokens"""
    trace = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            trace.append(TraceRequest(
                timestamp_ms=float(row["timestamp_ms"]),
                function_id=row["function_id"],
                prompt=row["prompt"],
                max_tokens=int(row.get("max_tokens", 256)),
            ))
    trace.sort(key=lambda r: r.timestamp_ms)
    return trace


def load_trace_jsonl(path: str, adapter_prefix: str = "lora") -> List[TraceRequest]:
    """Load trace from JSONL with fields: timestamp, adapter_id, input_text, output_tokens.

    Maps adapter_id N -> function_id '{adapter_prefix}_N'.
    Timestamp is in seconds, converted to milliseconds.
    """
    trace = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            adapter_id = row["adapter_id"]
            trace.append(TraceRequest(
                timestamp_ms=float(row["timestamp"]) * 1000,
                function_id=f"{adapter_prefix}_{adapter_id}",
                prompt=row.get("input_text", "Hello"),
                max_tokens=int(row.get("output_tokens", 256)),
            ))
    trace.sort(key=lambda r: r.timestamp_ms)
    return trace


# ---------------------------------------------------------------
# Request sender
# ---------------------------------------------------------------

def send_request(controller_url: str, req: TraceRequest,
                 idx: int, timeout: float) -> RequestResult:
    """Send a single inference request and measure latency."""
    url = f"{controller_url.rstrip('/')}/inference"
    payload = {
        "function_id": req.function_id,
        "prompt": req.prompt,
        "max_tokens": req.max_tokens,
    }

    submit_time = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        e2e_s = time.time() - submit_time
        data = resp.json()

        success = data.get("success", False)
        # Use server-reported token count if available, else estimate
        tokens = data.get("tokens", 0)
        if not tokens:
            text = data.get("text", "")
            tokens = len(text.split()) if text else 0
        ttft_ms = data.get("ttft_ms", e2e_s * 1000)
        e2e_ms = e2e_s * 1000

        return RequestResult(
            request_idx=idx,
            function_id=req.function_id,
            submit_time=submit_time,
            ttft_ms=ttft_ms,
            e2e_ms=e2e_ms,
            tokens_generated=tokens,
            success=success,
            error=data.get("error", "") if not success else "",
            batch_size=data.get("batch_size", 1),
            queue_wait_ms=data.get("queue_wait_ms", 0.0),
            container_id=data.get("container_id", ""),
            is_cold_start=data.get("is_cold_start", False),
            tpot_ms=data.get("tpot_ms", 0.0),
        )
    except Exception as e:
        e2e_s = time.time() - submit_time
        return RequestResult(
            request_idx=idx,
            function_id=req.function_id,
            submit_time=submit_time,
            ttft_ms=e2e_s * 1000,
            e2e_ms=e2e_s * 1000,
            tokens_generated=0,
            success=False,
            error=str(e),
        )


# ---------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------

def replay_trace(trace: List[TraceRequest], controller_url: str,
                 max_concurrent: int = 64,
                 request_timeout: float = 300.0) -> List[RequestResult]:
    """Replay a trace, respecting inter-arrival times."""
    if not trace:
        print("Empty trace, nothing to replay.")
        return []

    trace.sort(key=lambda r: r.timestamp_ms)
    results: List[RequestResult] = []
    futures = []

    total = len(trace)
    print(f"\nReplaying {total} requests over "
          f"{trace[-1].timestamp_ms / 1000:.1f}s...")

    start_time = time.time()
    executor = ThreadPoolExecutor(max_workers=max_concurrent)

    try:
        for idx, req in enumerate(trace):
            # Wait until the target arrival time
            target_time = start_time + (req.timestamp_ms / 1000.0)
            now = time.time()
            if target_time > now:
                time.sleep(target_time - now)

            # Submit request
            future = executor.submit(
                send_request, controller_url, req, idx, request_timeout
            )
            futures.append(future)

            # Progress
            if (idx + 1) % 50 == 0 or idx == total - 1:
                elapsed = time.time() - start_time
                print(f"  Submitted {idx + 1}/{total} "
                      f"({elapsed:.1f}s elapsed)")

        # Collect results
        print("\nWaiting for all requests to complete...")
        for future in as_completed(futures):
            results.append(future.result())

    finally:
        executor.shutdown(wait=True)

    results.sort(key=lambda r: r.request_idx)

    # Summary
    successes = sum(1 for r in results if r.success)
    successful_results = [r for r in results if r.success]
    if successful_results:
        avg_e2e = sum(r.e2e_ms for r in successful_results) / len(successful_results)
        avg_ttft = sum(r.ttft_ms for r in successful_results) / len(successful_results)
    else:
        avg_e2e = avg_ttft = 0

    print(f"\n=== Replay Complete ===")
    print(f"  Total requests: {len(results)}")
    print(f"  Successful: {successes}/{len(results)}")
    print(f"  Avg E2E (success only): {avg_e2e:.1f}ms")
    print(f"  Avg TTFT (success only): {avg_ttft:.1f}ms")

    return results


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ServerlessLoRA Trace Replayer"
    )

    # Trace source (mutually exclusive)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--trace", type=str,
        help="Path to CSV trace file",
    )
    source.add_argument(
        "--synthetic", type=str, choices=["poisson", "bursty", "constant"],
        help="Synthetic trace pattern",
    )

    # Synthetic parameters
    parser.add_argument("--rate", type=float, default=5.0,
                        help="Request rate (req/s)")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Duration in seconds")
    parser.add_argument("--burst-on-s", type=float, default=10.0,
                        help="Burst ON period (seconds, for bursty)")
    parser.add_argument("--burst-off-s", type=float, default=30.0,
                        help="Burst OFF period (seconds, for bursty)")
    parser.add_argument("--functions", type=str, default="default",
                        help="Comma-separated function IDs")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max tokens per request")

    # Controller
    parser.add_argument("--controller", type=str,
                        default="http://localhost:8000",
                        help="Controller URL")

    # Output
    parser.add_argument("--output", type=str, default="results/run.json",
                        help="Output JSON path for results")
    parser.add_argument("--save-trace", type=str, default=None,
                        help="Save generated trace to JSON (optional)")

    # Tuning
    parser.add_argument("--max-concurrent", type=int, default=64,
                        help="Max concurrent requests")
    parser.add_argument("--request-timeout", type=float, default=None,
                        help="Per-request timeout (seconds, None=no timeout)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    func_list = [f.strip() for f in args.functions.split(",")]

    # Generate or load trace
    if args.trace:
        print(f"Loading trace from {args.trace}...")
        if args.trace.endswith(".jsonl"):
            trace = load_trace_jsonl(args.trace)
        else:
            trace = load_trace_csv(args.trace)
    else:
        print(f"Generating {args.synthetic} trace: "
              f"rate={args.rate} req/s, duration={args.duration}s, "
              f"functions={func_list}")
        if args.synthetic == "poisson":
            gen = PoissonGenerator(args.rate, args.duration, func_list,
                                   args.max_tokens)
        elif args.synthetic == "bursty":
            gen = BurstyGenerator(args.rate, args.duration, func_list,
                                  args.burst_on_s, args.burst_off_s,
                                  args.max_tokens)
        else:
            gen = ConstantRateGenerator(args.rate, args.duration, func_list,
                                        args.max_tokens)
        trace = gen.generate()

    print(f"Trace contains {len(trace)} requests")

    # Optionally save the trace
    if args.save_trace:
        os.makedirs(os.path.dirname(args.save_trace) or ".", exist_ok=True)
        with open(args.save_trace, "w") as f:
            json.dump([{
                "timestamp_ms": r.timestamp_ms,
                "function_id": r.function_id,
                "prompt": r.prompt,
                "max_tokens": r.max_tokens,
            } for r in trace], f, indent=2)
        print(f"Trace saved to {args.save_trace}")

    # Replay
    results = replay_trace(
        trace, args.controller,
        max_concurrent=args.max_concurrent,
        request_timeout=args.request_timeout,
    )

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output_data = {
        "config": {
            "synthetic": args.synthetic,
            "trace": args.trace,
            "rate": args.rate,
            "duration": args.duration,
            "functions": func_list,
            "controller": args.controller,
            "max_tokens": args.max_tokens,
        },
        "summary": {
            "total_requests": len(results),
            "successful": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
        },
        "results": [asdict(r) for r in results],
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
