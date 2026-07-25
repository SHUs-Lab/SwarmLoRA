#!/usr/bin/env python3
"""Metrics collector for ServerlessLoRA experiments.

Reads trace_replayer output JSON and computes aggregate metrics matching
the paper's evaluation: TTFT, TPOT, E2E latency, throughput, SLO violations,
cold starts, monetary cost, cost-effectiveness, and per-function breakdowns.

Paper metrics (Section 6):
  - TTFT: Time To First Token
  - TPOT: Time Per Output Token = (E2E - TTFT) / tokens_generated
  - E2E:  End-to-end latency
  - Cost: Σ(container_alive_seconds × gpu_price_per_second)
  - Cost-Effectiveness: 1 / (E2E × Cost)

Usage:
    # Offline analysis
    python metrics_collector.py --input results/poisson_run.json \\
        --slo-ms 2000 --output results/poisson_metrics.json \\
        --text-report results/poisson_report.txt

    # With cost model
    python metrics_collector.py --input results/run.json \\
        --num-gpus 4 --gpu-price-per-hour 1.50

    # Live monitoring during experiment
    python metrics_collector.py --live --controller http://localhost:8000 \\
        --interval 10
"""

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import requests


# ---------------------------------------------------------------
# Percentile computation
# ---------------------------------------------------------------

def percentile(values: List[float], p: float) -> float:
    """Compute p-th percentile (0-100) of a list of values."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


# ---------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------

def compute_metrics(
    results: List[Dict],
    slo_ms: float,
    num_gpus: int = 0,
    gpu_price_per_hour: float = 0.0,
    pricing_model: str = "gpu-hour",
    cost_per_invocation: float = 0.0,
    cost_per_gpu_second: float = 0.0,
) -> Dict[str, Any]:
    """Compute aggregate metrics from a list of request results.

    Args:
        results: List of per-request result dicts from trace_replayer.
        slo_ms: SLO target for TTFT in milliseconds.
        num_gpus: Number of GPUs used (for cost model). 0 = skip cost.
        gpu_price_per_hour: Price per GPU-hour in USD (for gpu-hour model).
        pricing_model: "gpu-hour" or "serverless" (per-invocation + per-GPU-second).
        cost_per_invocation: Per-request fee in USD (serverless model).
        cost_per_gpu_second: Per-GPU-second fee in USD (serverless model).
    """
    if not results:
        return {"error": "no results"}

    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]

    # Extract latency arrays (successful only).
    # TTFT is reported as user-facing arrival -> first token, i.e.
    #   queue_wait_ms (arrival -> dispatch) + ttft_ms (dispatch -> first token).
    # This matches SwarmLoRA and ServerlessLLM, which both report arrival ->
    # first-token TTFT, so the three systems are compared on the same basis.
    # The prefill-only component and queue wait are kept separately below.
    ttft_prefill = [r["ttft_ms"] for r in successful]          # dispatch -> first token
    queue_waits = [r.get("queue_wait_ms", 0) for r in successful]  # arrival -> dispatch
    ttfts = [q + p for q, p in zip(queue_waits, ttft_prefill)]  # arrival -> first token
    e2es = [r["e2e_ms"] for r in successful]
    tokens = [r.get("tokens_generated", 0) for r in successful]

    # TPOT: Use worker-reported batch-level decode TPOT when available.
    # Fallback: (E2E - TTFT) / tokens (less accurate for batched requests).
    tpots = []
    for r in successful:
        tg = r.get("tokens_generated", 0)
        if tg > 0:
            worker_tpot = r.get("tpot_ms", 0)
            if worker_tpot > 0:
                tpots.append(worker_tpot)
            else:
                tpots.append((r["e2e_ms"] - r["ttft_ms"]) / tg)

    # Time span: first submit to last completion (actual wall-clock duration)
    submit_times = [r["submit_time"] for r in results]
    first_submit = min(submit_times)
    # Use submit_time + e2e_ms to find when the last request actually completed
    completion_times = [r["submit_time"] + r["e2e_ms"] / 1000.0
                        for r in successful if "e2e_ms" in r]
    if completion_times:
        last_complete = max(completion_times)
        duration_s = last_complete - first_submit
    elif len(submit_times) > 1:
        duration_s = max(submit_times) - first_submit
    else:
        duration_s = 1.0

    # Reported metrics only (paper): acceptance, throughput, TTFT/TPOT (ms).
    #   Acceptance   = successful / total submitted (8s-timeout = failed)
    #   Throughput   = output (completion) tokens per second
    #   TTFT         = arrival -> first token (queue_wait + prefill)
    #   TPOT         = per output token
    metrics = {
        "total_requests": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "duration_s": duration_s,
        "acceptance_rate": len(successful) / len(results) if results else 0,
        "tokens_per_second": sum(tokens) / duration_s if duration_s > 0 else 0,
        "ttft_p50_ms": percentile(ttfts, 50),
        "ttft_p90_ms": percentile(ttfts, 90),
        "tpot_p50_ms": percentile(tpots, 50),
        "tpot_p90_ms": percentile(tpots, 90),
    }
    return metrics


# ---------------------------------------------------------------
# Text report
# ---------------------------------------------------------------

def format_report(metrics: Dict[str, Any], config: Optional[Dict] = None,
                   fields: str = "all") -> str:
    """Format metrics as a human-readable text report.

    fields="all" (default, used by RQ3's trace-driven report, which needs
    acceptance/throughput/TTFT/TPOT for Table III and Figs. 7-9) or
    "throughput" (used by RQ1's throughput report, whose Fig. 5 plot only
    reads tokens_per_second -- acceptance is kept as a correctness check,
    TTFT/TPOT dropped since nothing consumes them here).
    """
    lines = []
    lines.append("=" * 60)
    lines.append("ServerlessLoRA Experiment Report")
    lines.append("=" * 60)

    if config:
        lines.append(f"\nTrace: {config.get('synthetic') or config.get('trace', 'N/A')}")
        if not config.get('trace'):
            # Rate/duration/functions are only meaningful for synthetic
            # traces -- a real --trace file carries its own timing, so these
            # would just be trace_replayer.py's unused argparse defaults.
            lines.append(f"Rate: {config.get('rate', 'N/A')} req/s")
            lines.append(f"Duration: {config.get('duration', 'N/A')}s")
            lines.append(f"Functions: {', '.join(config.get('functions', []))}")

    lines.append(f"\n--- Overview ---")
    lines.append(f"Total requests:  {metrics['total_requests']}")
    lines.append(f"Successful:      {metrics['successful']}")
    lines.append(f"Failed:          {metrics['failed']}")
    lines.append(f"Duration:        {metrics['duration_s']:.1f}s")

    # Canonical reported metrics (paper): acceptance, throughput, TTFT/TPOT (ms).
    lines.append(f"\n--- Reported metrics ---")
    lines.append(f"  Acceptance:  {metrics['acceptance_rate']*100:.1f}%")
    lines.append(f"  Throughput:  {metrics['tokens_per_second']:.1f} tok/s")
    if fields == "all":
        lines.append(f"  TTFT:        P50: {metrics['ttft_p50_ms']:.0f} ms   P90: {metrics['ttft_p90_ms']:.0f} ms")
        lines.append(f"  TPOT:        P50: {metrics['tpot_p50_ms']:.1f} ms   P90: {metrics['tpot_p90_ms']:.1f} ms")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------
# Live monitoring
# ---------------------------------------------------------------

def live_monitor(controller_url: str, interval: float, slo_ms: float):
    """Periodically poll controller /status and print metrics."""
    print(f"Live monitoring {controller_url} every {interval}s "
          f"(Ctrl-C to stop)\n")

    prev_total = 0
    prev_time = time.time()

    try:
        while True:
            try:
                r = requests.get(
                    f"{controller_url.rstrip('/')}/status", timeout=5
                )
                data = r.json()
                now = time.time()
                dt = now - prev_time

                total = data.get("total_requests", 0)
                new_reqs = total - prev_total
                rps = new_reqs / dt if dt > 0 else 0

                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"total={total}  "
                      f"success_rate={data.get('success_rate', 0):.1%}  "
                      f"avg_ttft={data.get('avg_ttft_ms', 0):.0f}ms  "
                      f"avg_e2e={data.get('avg_e2e_ms', 0):.0f}ms  "
                      f"rps={rps:.1f}")

                prev_total = total
                prev_time = now

            except requests.ConnectionError:
                print(f"[{time.strftime('%H:%M:%S')}] Controller unreachable")
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Error: {e}")

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ServerlessLoRA Metrics Collector"
    )

    # Mode
    parser.add_argument("--input", type=str, default=None,
                        help="Path to trace_replayer output JSON")
    parser.add_argument("--live", action="store_true",
                        help="Live monitoring mode")

    # Parameters
    parser.add_argument("--slo-ms", type=float, default=2000.0,
                        help="SLO target in milliseconds")
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Number of GPUs used (for cost model, 0=skip)")
    parser.add_argument("--gpu-price-per-hour", type=float, default=0.0,
                        help="Price per GPU-hour in USD (for gpu-hour model)")
    parser.add_argument("--pricing-model", choices=["gpu-hour", "serverless"],
                        default="gpu-hour",
                        help="Pricing model: gpu-hour or serverless (per-invocation + per-GPU-second)")
    parser.add_argument("--cost-per-invocation", type=float, default=0.0,
                        help="Per-request fee in USD (serverless model)")
    parser.add_argument("--cost-per-gpu-second", type=float, default=0.0,
                        help="Per-GPU-second fee in USD (serverless model)")
    parser.add_argument("--controller", type=str,
                        default="http://localhost:8000",
                        help="Controller URL (for live mode)")
    parser.add_argument("--interval", type=float, default=10.0,
                        help="Poll interval in seconds (for live mode)")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path for metrics")
    parser.add_argument("--text-report", type=str, default=None,
                        help="Output text report path")
    parser.add_argument("--fields", type=str, default="all",
                        choices=["all", "throughput"],
                        help="Report detail: 'all' (acceptance/throughput/"
                             "TTFT/TPOT, for RQ3) or 'throughput' (drops "
                             "TTFT/TPOT, for RQ1 which doesn't use them)")

    args = parser.parse_args()

    if args.live:
        live_monitor(args.controller, args.interval, args.slo_ms)
        return

    if not args.input:
        parser.error("--input is required in offline mode (or use --live)")

    # Load results
    with open(args.input) as f:
        data = json.load(f)

    results = data.get("results", [])
    config = data.get("config", {})

    if not results:
        print("No results found in input file.")
        return

    print(f"Loaded {len(results)} results from {args.input}")

    # Compute metrics
    metrics = compute_metrics(
        results, args.slo_ms,
        num_gpus=args.num_gpus,
        gpu_price_per_hour=args.gpu_price_per_hour,
        pricing_model=args.pricing_model,
        cost_per_invocation=args.cost_per_invocation,
        cost_per_gpu_second=args.cost_per_gpu_second,
    )

    # Text report
    report = format_report(metrics, config, fields=args.fields)
    print(f"\n{report}")

    if args.text_report:
        os.makedirs(os.path.dirname(args.text_report) or ".", exist_ok=True)
        with open(args.text_report, "w") as f:
            f.write(report)
        print(f"\nText report saved to {args.text_report}")

    # JSON output
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics JSON saved to {args.output}")


if __name__ == "__main__":
    main()
