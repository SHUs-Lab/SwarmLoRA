#!/usr/bin/env python3
"""
Trace replay benchmark client for ServerlessLLM.

Reads a JSONL trace file with timestamps, adapter IDs, and prompts,
then replays the requests against a running ServerlessLLM instance
at the correct timestamps. Collects per-request latency metrics and
outputs a JSON results file + summary report.

Trace format (JSONL):
    {"timestamp": 0.0, "adapter_id": 5, "input_text": "...", "input_tokens": 7, "output_tokens": 776}

Usage:
    python benchmarks/replay_trace.py \
        --server-url http://127.0.0.1:8343 \
        --model-name meta-llama/Llama-3.1-8B-Instruct \
        --trace-file ../traces/test_a0.8_cv2_r1_1min.jsonl \
        --output-file ./results/trace_replay_results.json
"""

import argparse
import asyncio
import contextlib
import dataclasses
import json
import os
import sys
import time
from collections import Counter
from typing import Optional


@dataclasses.dataclass
class TraceEntry:
    timestamp: float
    adapter_id: int
    input_text: str
    input_tokens: int
    output_tokens: int


@dataclasses.dataclass
class RequestResult:
    request_index: int
    adapter_id: int
    adapter_name: str
    scheduled_timestamp: float
    actual_send_time: float
    end_time: float
    latency: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    success: bool
    error_message: Optional[str]
    finish_reason: Optional[str]
    http_status_code: Optional[int]
    # Server-side timings (from backend "timings" field)
    queue_time_s: Optional[float] = None       # time waiting in batch queue
    adapter_wait_s: Optional[float] = None     # time waiting for other adapters in same batch round
    prefill_time_s: Optional[float] = None     # prefill (prompt processing until first token)
    decode_time_s: Optional[float] = None      # decoding (all tokens after first)
    generate_time_s: Optional[float] = None    # total model.generate() = prefill + decode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay a trace against ServerlessLLM and collect metrics"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:8343",
        help="ServerlessLLM head node URL",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Registered model name on ServerlessLLM",
    )
    parser.add_argument(
        "--trace-file",
        type=str,
        required=True,
        help="Path to JSONL trace file",
    )
    parser.add_argument(
        "--adapter-name-template",
        type=str,
        default="adapter_{adapter_id}",
        help="Template to map adapter_id to adapter name (default: adapter_{adapter_id})",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="./results/trace_replay_results.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=0,
        help="Max concurrent requests (0 = unlimited)",
    )
    parser.add_argument(
        "--timeout-per-request",
        type=int,
        default=600,
        help="Per-request timeout in seconds (default 600 to handle long queue waits)",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Don't send lora_adapter_name (for dummy backend testing)",
    )
    parser.add_argument(
        "--token-latency",
        type=float,
        default=None,
        help="Override token_latency in requests (for dummy backend, e.g., 0.001)",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Cap max_tokens per request (useful for dummy backend testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse trace and print schedule without sending requests",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Deployment mode label (original/batched/optimized) for result tagging",
    )
    parser.add_argument(
        "--gen-speed",
        type=float,
        default=17.7,
        help="Measured generation speed in tok/s for TTFT/TPOT estimation (default: 17.7)",
    )
    parser.add_argument(
        "--goodput-baseline-tokens",
        type=float,
        default=None,
        help="Avg completion tokens/successful-request from a low-load reference run "
             "(e.g. rps1), used to compute goodput_tok_per_s. Omit for the reference "
             "run itself -- it falls back to its own avg (goodput == raw throughput).",
    )
    parser.add_argument(
        "--fields",
        choices=["all", "throughput"],
        default="all",
        help="Report detail: 'all' (acceptance/throughput/goodput/TTFT/TPOT, "
             "for RQ3) or 'throughput' (drops TTFT/TPOT, for RQ1 which doesn't "
             "use them)",
    )
    return parser.parse_args()


def load_trace(trace_file_path):
    """Parse JSONL trace file into list of TraceEntry."""
    entries = []
    with open(trace_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            entries.append(
                TraceEntry(
                    timestamp=data["timestamp"],
                    adapter_id=data["adapter_id"],
                    input_text=data["input_text"],
                    input_tokens=data["input_tokens"],
                    output_tokens=data["output_tokens"],
                )
            )
    entries.sort(key=lambda e: e.timestamp)
    return entries


async def send_request(
    session,
    server_url,
    model_name,
    entry,
    request_index,
    adapter_name_template,
    timeout,
    semaphore,
    replay_start_time,
    no_lora=False,
    token_latency=None,
    max_output_tokens=None,
):
    """Send a single inference request and collect metrics."""
    import aiohttp
    adapter_name = adapter_name_template.replace(
        "{adapter_id}", str(entry.adapter_id)
    )
    url = f"{server_url.rstrip('/')}/v1/chat/completions"
    output_tokens = entry.output_tokens
    if max_output_tokens is not None:
        output_tokens = min(output_tokens, max_output_tokens)
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": entry.input_text}],
        "max_tokens": output_tokens,
    }
    if not no_lora:
        payload["lora_adapter_name"] = adapter_name
    if token_latency is not None:
        payload["token_latency"] = token_latency

    ctx = semaphore if semaphore else contextlib.AsyncExitStack()
    async with ctx:
        actual_send_time = time.monotonic() - replay_start_time
        send_wall = time.monotonic()
        try:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                status_code = resp.status
                body = await resp.json()
                end_wall = time.monotonic()
                latency = end_wall - send_wall
                end_time = end_wall - replay_start_time

                if status_code == 200 and "error" not in body:
                    usage = body.get("usage", {})
                    choices = body.get("choices", [{}])
                    finish_reason = (
                        choices[0].get("finish_reason") if choices else None
                    )
                    timings = body.get("timings", {})
                    return RequestResult(
                        request_index=request_index,
                        adapter_id=entry.adapter_id,
                        adapter_name=adapter_name,
                        scheduled_timestamp=entry.timestamp,
                        actual_send_time=actual_send_time,
                        end_time=end_time,
                        latency=latency,
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        total_tokens=usage.get("total_tokens", 0),
                        success=True,
                        error_message=None,
                        finish_reason=finish_reason,
                        http_status_code=status_code,
                        queue_time_s=timings.get("queue_time_s"),
                        adapter_wait_s=timings.get("adapter_wait_s"),
                        prefill_time_s=timings.get("prefill_time_s"),
                        decode_time_s=timings.get("decode_time_s"),
                        generate_time_s=timings.get("generate_time_s"),
                    )
                else:
                    error_msg = body.get("error", body.get("detail", str(body)))
                    return RequestResult(
                        request_index=request_index,
                        adapter_id=entry.adapter_id,
                        adapter_name=adapter_name,
                        scheduled_timestamp=entry.timestamp,
                        actual_send_time=actual_send_time,
                        end_time=end_wall - replay_start_time,
                        latency=latency,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        success=False,
                        error_message=str(error_msg),
                        finish_reason=None,
                        http_status_code=status_code,
                    )
        except Exception as e:
            end_wall = time.monotonic()
            return RequestResult(
                request_index=request_index,
                adapter_id=entry.adapter_id,
                adapter_name=adapter_name,
                scheduled_timestamp=entry.timestamp,
                actual_send_time=actual_send_time,
                end_time=end_wall - replay_start_time,
                latency=end_wall - send_wall,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                success=False,
                error_message=str(e),
                finish_reason=None,
                http_status_code=None,
            )


async def replay_trace(args):
    """Main async driver: dispatch requests at correct timestamps."""
    import aiohttp

    trace = load_trace(args.trace_file)
    print(f"Loaded {len(trace)} requests from {args.trace_file}")
    print(f"Trace duration: {trace[-1].timestamp:.2f}s")
    print(f"Target server: {args.server_url}")
    print(f"Model: {args.model_name}")
    print()

    semaphore = (
        asyncio.Semaphore(args.max_concurrent) if args.max_concurrent > 0 else None
    )

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        replay_start = time.monotonic()
        tasks = []

        for idx, entry in enumerate(trace):
            elapsed = time.monotonic() - replay_start
            delay = entry.timestamp - elapsed
            if delay > 0:
                await asyncio.sleep(delay)

            task = asyncio.create_task(
                send_request(
                    session=session,
                    server_url=args.server_url,
                    model_name=args.model_name,
                    entry=entry,
                    request_index=idx,
                    adapter_name_template=args.adapter_name_template,
                    timeout=args.timeout_per_request,
                    semaphore=semaphore,
                    replay_start_time=replay_start,
                    no_lora=args.no_lora,
                    token_latency=args.token_latency,
                    max_output_tokens=args.max_output_tokens,
                )
            )
            tasks.append(task)

            if (idx + 1) % 10 == 0:
                print(
                    f"  Dispatched {idx + 1}/{len(trace)} requests "
                    f"(t={entry.timestamp:.2f}s)"
                )

        print(f"All {len(trace)} requests dispatched. Waiting for completions...")
        results = await asyncio.gather(*tasks)

    return list(results)


def _percentile(sorted_list, p):
    """Return the p-th percentile from a sorted list (0-1)."""
    if not sorted_list:
        return 0
    idx = int(len(sorted_list) * p)
    idx = min(idx, len(sorted_list) - 1)
    return sorted_list[idx]


def compute_summary(results, trace_file, gen_tok_per_sec=17.7, goodput_baseline_tokens=None):
    """Compute aggregate statistics from results.

    Uses server-side timings if available (queue_time_s, adapter_wait_s,
    generate_time_s). Falls back to estimation if timings are missing.
    """
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    # Check if server-side timings are available
    has_timings = any(r.generate_time_s is not None for r in successful)

    # --- E2E latency ---
    e2e = sorted([r.latency for r in successful]) if successful else []

    # --- Server-side timing decomposition ---
    queue_times = []
    adapter_waits = []
    prefill_times = []
    decode_times = []
    gen_times = []
    tpots = []
    ttfts = []

    for r in successful:
        if has_timings and r.generate_time_s is not None:
            # Real measured timings from backend
            qt = r.queue_time_s or 0
            aw = r.adapter_wait_s or 0
            pf = r.prefill_time_s or 0
            dc = r.decode_time_s or 0
            gt = r.generate_time_s or 0
            queue_times.append(qt)
            adapter_waits.append(aw)
            prefill_times.append(pf)
            decode_times.append(dc)
            gen_times.append(gt)
            # TTFT = queue_wait + adapter_wait + prefill (time until first token)
            ttfts.append(qt + aw + pf)
            # TPOT = decode_time / (completion_tokens - 1), excluding first token
            if r.completion_tokens > 1 and dc > 0:
                tpots.append((dc / (r.completion_tokens - 1)) * 1000)  # ms
            elif r.completion_tokens > 0 and gt > 0:
                tpots.append((gt / r.completion_tokens) * 1000)
        else:
            # Fallback: estimate from gen speed
            gen_time = r.completion_tokens / gen_tok_per_sec if r.completion_tokens > 0 else 0
            gen_times.append(gen_time)
            ttfts.append(r.latency - gen_time)
            tpots.append((1.0 / gen_tok_per_sec) * 1000)

    ttfts_sorted = sorted(ttfts) if ttfts else []
    gen_times_sorted = sorted(gen_times) if gen_times else []
    queue_times_sorted = sorted(queue_times) if queue_times else []
    adapter_waits_sorted = sorted(adapter_waits) if adapter_waits else []
    prefill_times_sorted = sorted(prefill_times) if prefill_times else []
    decode_times_sorted = sorted(decode_times) if decode_times else []
    tpots_sorted = sorted(tpots) if tpots else []

    # --- Token throughput ---
    total_duration = max(r.end_time for r in results) if results else 0
    total_prompt_tokens = sum(r.prompt_tokens for r in successful)
    total_completion_tokens = sum(r.completion_tokens for r in successful)
    total_tokens = total_prompt_tokens + total_completion_tokens
    avg_completion_tokens = round(total_completion_tokens / len(successful), 2) if successful else 0

    # Goodput corrects a bias in raw tokens_per_second: this router's
    # admission-queue timeout kills requests on queue wait alone, and survivors
    # skew toward longer completions as load rises, so raw tok/s partly
    # reflects which requests survived rather than serving speed. Fixing the
    # per-request token count at a low-load baseline isolates requests-served.
    # Pass --goodput-baseline-tokens from a low-load run (e.g. rps1); that run
    # itself omits it and falls back to its own average (goodput == raw).
    goodput_baseline = (
        goodput_baseline_tokens if goodput_baseline_tokens is not None else avg_completion_tokens
    )
    goodput_tok_per_s = (
        round(len(successful) * goodput_baseline / total_duration, 1) if total_duration > 0 else 0
    )

    # Reported metrics only (paper): acceptance, throughput, TTFT/TPOT (ms).
    #   Acceptance = successful / total submitted.
    #   Throughput = output (completion) tokens per second (raw; see goodput_tok_per_s).
    #   TTFT = arrival -> first token (queue_time + adapter_wait + prefill);
    #          ttfts_sorted is in seconds -> *1000. TPOT (tpots_sorted) is ms.
    summary = {
        "trace_file": trace_file,
        "total_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "total_duration_s": round(total_duration, 3),
        "acceptance_rate": round(len(successful) / len(results), 4) if results else 0,
        "tokens_per_second": round(total_completion_tokens / total_duration, 1) if total_duration > 0 else 0,
        "avg_completion_tokens": avg_completion_tokens,
        "goodput_tok_per_s": goodput_tok_per_s,
        "ttft_p50_ms": round(_percentile(ttfts_sorted, 0.50) * 1000, 1) if ttfts_sorted else 0,
        "ttft_p90_ms": round(_percentile(ttfts_sorted, 0.90) * 1000, 1) if ttfts_sorted else 0,
        "tpot_p50_ms": round(_percentile(tpots_sorted, 0.50), 1) if tpots_sorted else 0,
        "tpot_p90_ms": round(_percentile(tpots_sorted, 0.90), 1) if tpots_sorted else 0,
    }

    return summary


def print_report(summary, fields="all"):
    """Print human-readable benchmark report to stdout.

    fields="all" (default, used by RQ3's trace-driven report, which needs
    acceptance/throughput/TTFT/TPOT for Table III and Figs. 7-9) or
    "throughput" (used by RQ1's throughput report, whose Fig. 5 plot only
    reads goodput_tok_per_s/tokens_per_second -- TTFT/TPOT dropped since
    nothing consumes them here).
    """
    print()
    print("=" * 80)
    print("  ServerlessLLM Trace Replay Benchmark Results")
    print("=" * 80)
    print(f"  Trace File:          {summary['trace_file']}")
    print(f"  Total Requests:      {summary['total_requests']}")
    print(f"  Successful:          {summary['successful_requests']}")
    print(f"  Failed:              {summary['failed_requests']}")
    print(f"  Total Duration:      {summary['total_duration_s']:.2f}s")
    print()
    # Reported metrics (paper): acceptance, throughput, TTFT/TPOT (ms).
    print("  Reported metrics:")
    print(f"    Acceptance:        {summary['acceptance_rate']*100:.1f}%")
    print(f"    Throughput:        {summary['goodput_tok_per_s']:.1f} tok/s")
    if fields == "all":
        print(f"    TTFT:              P50: {summary['ttft_p50_ms']:.0f} ms   P90: {summary['ttft_p90_ms']:.0f} ms")
        print(f"    TPOT:              P50: {summary['tpot_p50_ms']:.1f} ms   P90: {summary['tpot_p90_ms']:.1f} ms")
    print("=" * 80)


def save_results(results, summary, output_file, config=None):
    """Save per-request results and summary to JSON file."""
    output = {
        "summary": summary,
        "requests": [dataclasses.asdict(r) for r in results],
    }
    if config:
        output["config"] = config
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")


def main():
    args = parse_args()

    if args.dry_run:
        trace = load_trace(args.trace_file)
        print(f"Trace file: {args.trace_file}")
        print(f"Total requests: {len(trace)}")
        print(f"Trace duration: {trace[-1].timestamp:.2f}s")
        print(f"Avg rate: {len(trace) / trace[-1].timestamp:.2f} req/s")
        print()
        print("Per-adapter request counts:")
        counts = Counter(e.adapter_id for e in trace)
        for aid in sorted(counts):
            print(f"  adapter_{aid}: {counts[aid]} requests")
        print()
        print("Token statistics:")
        input_tokens = [e.input_tokens for e in trace]
        output_tokens = [e.output_tokens for e in trace]
        print(
            f"  Input tokens:  min={min(input_tokens)}, max={max(input_tokens)}, "
            f"avg={sum(input_tokens)/len(input_tokens):.1f}"
        )
        print(
            f"  Output tokens: min={min(output_tokens)}, max={max(output_tokens)}, "
            f"avg={sum(output_tokens)/len(output_tokens):.1f}"
        )
        return

    results = asyncio.run(replay_trace(args))
    summary = compute_summary(
        results, args.trace_file, gen_tok_per_sec=args.gen_speed,
        goodput_baseline_tokens=args.goodput_baseline_tokens,
    )
    print_report(summary, fields=args.fields)

    config = {
        "mode": args.mode,
        "trace_file": args.trace_file,
        "model_name": args.model_name,
        "timeout_per_request": args.timeout_per_request,
        "gen_speed_tok_per_s": args.gen_speed,
    }
    save_results(results, summary, args.output_file, config=config)


if __name__ == "__main__":
    main()
