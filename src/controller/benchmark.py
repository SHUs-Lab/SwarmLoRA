
import asyncio
import aiohttp
import os
import sys
import time
import json
import random
import argparse
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path
import statistics
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import subprocess
from config import CONTROLLER_PORT, AGGREGATOR_PORTS, AGGREGATOR_HEALTH_PORTS

try:
    from .slo_metrics import slo_metrics, ttft_percentiles_all
except ImportError:
    from controller.slo_metrics import slo_metrics, ttft_percentiles_all

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Data Classes

@dataclass
class RequestRecord:
    """Record of a single request"""
    request_id: str
    adapter_name: str
    prompt_tokens: int
    max_tokens: int

    # Timestamps (client-side)
    scheduled_at: float         # When request was supposed to be sent (from trace)
    sent_at: float              # When request was actually sent
    completed_at: float = 0     # When response fully received

    # Results
    output_tokens: int = 0
    success: bool = False
    error: Optional[str] = None

    # Server-reported metrics (from _serverless_metrics)
    routing_ms: float = 0       # Controller: time to find/spawn/swap worker
    e2e_ms: float = 0           # Worker: HTTP handler to completion
    queue_wait_ms: float = 0    # Worker: submit to slot allocation
    ttft_ms: float = 0          # TTFT: controller arrival to first token (includes routing+prefill)
    worker_ttft_ms: float = 0   # Raw worker-reported TTFT (same as ttft_ms with arrival_time fix)
    decode_time_ms: float = 0   # Worker: first token to last token
    decode_throughput_tps: float = 0  # Worker: tokens/sec (decode only)
    gen_throughput_tps: float = 0     # Worker: tokens/sec (from slot alloc)
    cold_start: bool = False
    worker_id: Optional[str] = None

    @property
    def total_latency(self) -> float:
        """Total end-to-end latency (client-measured)"""
        return self.completed_at - self.sent_at if self.completed_at else 0

    @property
    def ttft(self) -> float:
        """Time to first token (routing + queue_wait + prefill, in seconds)"""
        return self.ttft_ms / 1000.0 if self.ttft_ms > 0 else 0

    @property
    def tpot(self) -> float:
        """Time per output token (from server-reported decode_time_ms)"""
        if self.output_tokens <= 1 or self.decode_time_ms <= 0:
            return 0
        return (self.decode_time_ms / 1000.0) / (self.output_tokens - 1)

    @property
    def throughput(self) -> float:
        """Tokens per second (server-reported decode throughput)"""
        return self.decode_throughput_tps if self.decode_throughput_tps > 0 else 0


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    api_url: str = f"http://localhost:{CONTROLLER_PORT}"
    trace_file: Optional[str] = None

    # Synthetic workload settings (if no trace)
    num_requests: int = 100
    request_rate: float = 10.0      # Requests per second
    duration_seconds: float = 60.0   # Total duration

    # Request settings
    adapters: List[str] = field(default_factory=lambda: ["default"])
    adapter_weights: Optional[List[float]] = None  # Probability weights
    prompt_tokens: int = 50
    max_tokens: int = 64

    # Adapter mapping
    default_adapter: Optional[str] = None  # Override all trace adapters with this
    adapter_map_prefix: Optional[str] = None  # Map trace IDs to {prefix}{id}, e.g. "sim-adapters/lora-"

    # Output
    output_dir: str = "./benchmark_results"


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results"""
    config: BenchmarkConfig
    start_time: float
    end_time: float

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # ── Reported metrics (paper) ──────────────────────────────────────────
    # Acceptance rate: successful / total submitted (requests still queued
    # past the 8s timeout count as failed). Fraction in [0, 1].
    acceptance_rate: float = 0
    # Throughput: output (completion) tokens per second.
    total_tokens: int = 0
    tokens_per_second: float = 0
    # TTFT (arrival -> first token) and TPOT (per output token), in ms.
    # NOTE: over SUCCESSFUL requests only, so dropping a slow request improves
    # them. See the SLO-aware block below for the all-requests view.
    ttft_p50_ms: float = 0
    ttft_p90_ms: float = 0
    tpot_p50_ms: float = 0
    tpot_p90_ms: float = 0

    # ── SLO-aware metrics (src/controller/slo_metrics.py) ─────────────────
    # Denominator is ALL submitted requests, so a dropped or late request can
    # never flatter them: drops are deliberate under the SLO deadline policy.
    # Additive: nothing above changes.
    slo_s: float = 0
    slo_met_requests: int = 0
    slo_attainment: float = 0
    effective_throughput_tok_s: float = 0
    ttft_all_p50_ms: float = 0
    ttft_all_p90_ms: float = 0


# Trace Loading

def load_trace(trace_file: str) -> List[dict]:
    """Load trace from JSONL file."""
    trace = []
    with open(trace_file, 'r') as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            trace.append({
                "request_id": f"req-{i:06d}",
                "timestamp": float(row["timestamp"]),
                "adapter": str(row["adapter_id"]),
                "prompt": row["input_text"],
                "prompt_tokens": int(row["input_tokens"]),
                "max_tokens": int(row["output_tokens"]),
            })

    # Sort by timestamp
    trace.sort(key=lambda x: x.get("timestamp", 0))
    logger.info(f"Loaded {len(trace)} requests from {trace_file}")
    return trace


def generate_synthetic_trace(config: BenchmarkConfig) -> List[dict]:
    """Generate synthetic workload trace"""
    trace = []

    # Poisson arrival process
    current_time = 0
    inter_arrival = 1.0 / config.request_rate

    for i in range(config.num_requests):
        # Select adapter
        if config.adapter_weights:
            adapter = random.choices(config.adapters, weights=config.adapter_weights)[0]
        else:
            adapter = random.choice(config.adapters)

        trace.append({
            "request_id": f"req-{i:06d}",
            "timestamp": current_time,
            "adapter": adapter,
            "prompt": "Explain the concept of machine learning in simple terms.",
            "prompt_tokens": config.prompt_tokens,
            "max_tokens": config.max_tokens,
        })

        # Exponential inter-arrival time
        current_time += random.expovariate(1.0 / inter_arrival)

        if current_time > config.duration_seconds:
            break

    return trace


# Benchmark Runner

class BenchmarkRunner:
    """Run benchmarks against the controller platform"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.records: List[RequestRecord] = []
        self._session: Optional[aiohttp.ClientSession] = None

    async def run(self) -> BenchmarkResults:
        """Run the benchmark"""
        # Load or generate trace
        if self.config.trace_file:
            trace = load_trace(self.config.trace_file)
            if self.config.default_adapter:
                for req in trace:
                    req["adapter"] = self.config.default_adapter
                logger.info(f"Mapped all adapters to: {self.config.default_adapter}")
            elif self.config.adapter_map_prefix:
                for req in trace:
                    req["adapter"] = f"{self.config.adapter_map_prefix}{req['adapter']}"
                unique = len(set(r["adapter"] for r in trace))
                logger.info(f"Mapped adapters with prefix '{self.config.adapter_map_prefix}' -> {unique} unique adapters")
            logger.info(f"Loaded trace with {len(trace)} requests")
        else:
            trace = generate_synthetic_trace(self.config)
            logger.info(f"Generated synthetic trace with {len(trace)} requests")

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Run benchmark
        connector = aiohttp.TCPConnector(limit=0)  # Unlimited connections for burst traces
        async with aiohttp.ClientSession(connector=connector) as session:
            self._session = session
            start_time = time.time()

            await self._run_trace(trace, start_time)

            end_time = time.time()

        results = self._calculate_results(start_time, end_time)

        self._save_results(results)

        return results

    async def _run_trace(self, trace: List[dict], start_time: float):
        """Execute trace requests"""
        tasks = []

        for request in trace:
            # Calculate delay from trace start
            delay = request["timestamp"]

            # Schedule request
            task = asyncio.create_task(
                self._send_request_at(request, start_time, delay)
            )
            tasks.append(task)

        # Wait for all requests with progress
        total = len(tasks)
        completed = 0

        for coro in asyncio.as_completed(tasks):
            await coro
            completed += 1
            if completed % 10 == 0:
                logger.info(f"Progress: {completed}/{total} requests completed")

        logger.info(f"All {total} requests completed")

    async def _send_request_at(self, request: dict, start_time: float, delay: float):
        """Send request at specified time"""
        # Wait until scheduled time
        target_time = start_time + delay
        now = time.time()
        if target_time > now:
            await asyncio.sleep(target_time - now)

        # Create record
        record = RequestRecord(
            request_id=request.get("request_id", ""),
            adapter_name=request.get("adapter", "default"),
            prompt_tokens=request.get("prompt_tokens", 50),
            max_tokens=request.get("max_tokens", 64),
            scheduled_at=target_time,
            sent_at=time.time(),
        )

        # Send request
        try:
            payload = {
                "model": record.adapter_name,
                "prompt": request["prompt"],
                "max_tokens": record.max_tokens,
                "do_sample": False,
            }

            async with self._session.post(
                f"{self.config.api_url}/v1/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    record.completed_at = time.time()
                    record.success = True

                    # Use real token count from server usage stats
                    usage = result.get("usage", {})
                    record.output_tokens = usage.get("completion_tokens", 0)

                    # Extract server-reported metrics
                    if "_serverless_metrics" in result:
                        metrics = result["_serverless_metrics"]
                        record.routing_ms = metrics.get("routing_ms", 0)
                        record.e2e_ms = metrics.get("e2e_ms", 0)
                        record.queue_wait_ms = metrics.get("queue_wait_ms", 0)
                        record.worker_ttft_ms = metrics.get("ttft_ms", 0)
                        record.ttft_ms = record.worker_ttft_ms  # TTFT now measured from controller arrival
                        record.decode_time_ms = metrics.get("decode_time_ms", 0)
                        record.decode_throughput_tps = metrics.get("decode_throughput", 0)
                        record.gen_throughput_tps = metrics.get("gen_throughput", 0)
                        record.worker_id = metrics.get("worker_id")
                        record.cold_start = record.routing_ms > 5000  # >5s routing = likely cold start
                else:
                    record.completed_at = time.time()
                    record.success = False
                    try:
                        result = await response.json()
                        detail = result.get("detail", "")
                        record.error = f"HTTP {response.status}: {detail}" if detail else f"HTTP {response.status}"
                    except Exception:
                        record.error = f"HTTP {response.status}"

        except Exception as e:
            record.completed_at = time.time()
            record.success = False
            record.error = str(e)

        self.records.append(record)

    def _calculate_results(self, start_time: float, end_time: float) -> BenchmarkResults:
        """Calculate benchmark results from records"""
        results = BenchmarkResults(
            config=self.config,
            start_time=start_time,
            end_time=end_time,
        )

        results.total_requests = len(self.records)
        successful = [r for r in self.records if r.success]
        results.successful_requests = len(successful)
        results.failed_requests = results.total_requests - results.successful_requests
        results.acceptance_rate = (
            results.successful_requests / results.total_requests
            if results.total_requests else 0
        )

        # SLO-aware metrics, over EVERY submitted request (failures and drops
        # included) -- computed before the early return below so a run with no
        # successes still reports an honest 0 rather than an absent field.
        # r.ttft is seconds, which is what slo_metrics expects.
        samples = [(r.success, r.ttft, r.output_tokens) for r in self.records]
        duration = end_time - start_time
        slo = slo_metrics(samples, duration)
        results.slo_s = slo["slo_s"]
        results.slo_met_requests = slo["slo_met_requests"]
        results.slo_attainment = slo["slo_attainment"]
        results.effective_throughput_tok_s = slo["effective_throughput_tok_s"]
        allp = ttft_percentiles_all(samples)
        results.ttft_all_p50_ms = allp.get("ttft_all_p50_ms", 0)
        results.ttft_all_p90_ms = allp.get("ttft_all_p90_ms", 0)

        if not successful:
            return results

        # Throughput: output (completion) tokens per second.
        results.total_tokens = sum(r.output_tokens for r in successful)
        duration = end_time - start_time
        results.tokens_per_second = results.total_tokens / duration if duration else 0

        # TTFT (arrival -> first token). r.ttft is in seconds -> store as ms.
        ttfts = [r.ttft for r in successful if r.ttft > 0]
        if ttfts:
            results.ttft_p50_ms = statistics.median(ttfts) * 1000
            results.ttft_p90_ms = self._percentile(ttfts, 90) * 1000

        # TPOT (per output token). r.tpot is in seconds -> store as ms.
        tpots = [r.tpot for r in successful if r.tpot > 0]
        if tpots:
            results.tpot_p50_ms = statistics.median(tpots) * 1000
            results.tpot_p90_ms = self._percentile(tpots, 90) * 1000

        return results

    def _percentile(self, data: List[float], p: int) -> float:
        """Calculate percentile"""
        if not data:
            return 0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]

    def _save_results(self, results: BenchmarkResults):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save summary
        summary_file = Path(self.config.output_dir) / f"summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(self._results_to_dict(results), f, indent=2)
        logger.info(f"Summary saved to {summary_file}")

        # Save detailed records
        records_file = Path(self.config.output_dir) / f"records_{timestamp}.json"
        with open(records_file, 'w') as f:
            json.dump([asdict(r) for r in self.records], f, indent=2)
        logger.info(f"Records saved to {records_file}")

        self._print_summary(results)

    def _results_to_dict(self, results: BenchmarkResults) -> dict:
        """Convert results to dict (handle non-serializable fields)"""
        d = asdict(results)
        d["config"] = asdict(results.config)
        return d

    def _print_summary(self, results: BenchmarkResults):
        """Print results summary"""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)

        duration = results.end_time - results.start_time
        print(f"\nDuration: {duration:.2f}s")
        print(f"Total Requests: {results.total_requests}  "
              f"(successful: {results.successful_requests}, failed: {results.failed_requests})")

        # Reported metrics (paper): acceptance rate, throughput, TTFT/TPOT (ms).
        print(f"\nReported metrics:")
        print(f"  Acceptance:  {results.acceptance_rate*100:.1f}%")
        print(f"  Throughput:  {results.tokens_per_second:.1f} tok/s")
        print(f"  TTFT:        P50: {results.ttft_p50_ms:.0f} ms   P90: {results.ttft_p90_ms:.0f} ms")
        print(f"  TPOT:        P50: {results.tpot_p50_ms:.1f} ms   P90: {results.tpot_p90_ms:.1f} ms")
        # Denominator is every submitted request, so drops count against these.
        print(f"  SLO attain:  {results.slo_attainment*100:.1f}% "
              f"({results.slo_met_requests}/{results.total_requests} within {results.slo_s:.0f}s)")
        print(f"  Effective:   {results.effective_throughput_tok_s:.1f} tok/s "
              f"(SLO-meeting requests only)")
        print(f"  TTFT (all):  P50: {results.ttft_all_p50_ms:.0f} ms   P90: {results.ttft_all_p90_ms:.0f} ms")

        print("=" * 60 + "\n")


# CLI Entry Point

# Managed Lifecycle (MPS, Aggregator, Controller)

class ManagedLifecycle:
    """Manages the full lifecycle: MPS, aggregator, controller, cleanup."""

    def __init__(self, num_gpus: int = None, scheduler: str = "inter",
                 controller_port: int = 8344, adapter_map_prefix: str = None):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.logs_dir = os.path.join(self.project_root, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)

        # Auto-detect GPUs if not specified
        if num_gpus is None:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                )
                num_gpus = len(result.stdout.strip().split('\n')) if result.returncode == 0 else 1
            except Exception:
                num_gpus = 1

        self.num_gpus = num_gpus
        self.scheduler = scheduler
        self.controller_port = controller_port
        self.adapter_map_prefix = adapter_map_prefix

        # Port maps
        self.agg_tcp_ports = {i: AGGREGATOR_PORTS.get(i, 50056 + i) for i in range(num_gpus)}
        self.agg_health_ports = {i: AGGREGATOR_HEALTH_PORTS.get(i, 8000 + i) for i in range(num_gpus)}

        # Processes
        self._aggregator_proc = None
        self._controller_proc = None
        self._procs = []

    async def start_services(self):
        """Start MPS, primary aggregator, and controller."""
        logger.info("=" * 60)
        logger.info("Starting managed services")
        logger.info(f"  GPUs: {self.num_gpus}")
        logger.info(f"  Scheduler: {self.scheduler}")
        logger.info(f"  Controller port: {self.controller_port}")
        logger.info("=" * 60)

        # 1. Setup MPS (stop first to clean stale pipes, then start fresh)
        mps_script = os.path.join(self.project_root, "setup_mps.sh")
        if os.path.exists(mps_script):
            logger.info("Setting up MPS...")
            subprocess.run(["bash", mps_script, "stop"],
                           capture_output=True, text=True, timeout=60)
            result = subprocess.run(
                ["bash", mps_script, "start"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                logger.warning(f"MPS setup failed: {result.stderr}")
            else:
                logger.info("MPS setup complete")
        else:
            logger.info("No setup_mps.sh found, skipping MPS setup")

        # 2. Start primary aggregator (GPU 0, loads from disk)
        gpu_idx = 0
        tcp_port = self.agg_tcp_ports[gpu_idx]
        health_port = self.agg_health_ports[gpu_idx]

        mps_pipe_dir = f"/tmp/mps_{gpu_idx}"
        env = os.environ.copy()
        agg_device = f"cuda:{gpu_idx}"
        if os.path.exists(mps_pipe_dir):
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            env["CUDA_MPS_PIPE_DIRECTORY"] = mps_pipe_dir
            agg_device = "cuda:0"

        venv_python = os.path.join(self.project_root, "venv", "bin", "python")
        python_exe = venv_python if os.path.exists(venv_python) else sys.executable

        agg_cmd = [
            python_exe, os.path.join(self.project_root, "src", "aggregator.py"),
            "--device", agg_device,
            "--port", str(tcp_port),
            "--health-port", str(health_port),
        ]

        logger.info(f"Starting aggregator 0 on {agg_device} (tcp={tcp_port}, health={health_port})...")
        agg_log = open(os.path.join(self.logs_dir, "aggregator_0.log"), "w")
        self._aggregator_proc = subprocess.Popen(
            agg_cmd, cwd=self.project_root, stdout=agg_log, stderr=agg_log, env=env,
        )
        agg_log.close()
        self._procs.append(self._aggregator_proc)

        # Poll aggregator health until ready
        if not await self._wait_for_health(health_port, timeout=180, label="aggregator 0"):
            raise RuntimeError("Aggregator 0 failed to start")

        # 3. Start controller
        agg_port_map = ",".join(f"{i}:{self.agg_tcp_ports[i]}" for i in range(self.num_gpus))
        health_port_map = ",".join(f"{i}:{self.agg_health_ports[i]}" for i in range(self.num_gpus))

        ctrl_cmd = [
            python_exe, "-m", "controller.controller",
            "--port", str(self.controller_port),
            "--aggregator-port-map", agg_port_map,
            "--aggregator-health-port-map", health_port_map,
            "--scheduler", self.scheduler,
        ]

        logger.info(f"Starting controller on port {self.controller_port}...")
        ctrl_log = open(os.path.join(self.logs_dir, "controller.log"), "w")
        self._controller_proc = subprocess.Popen(
            ctrl_cmd, cwd=self.project_root, stdout=ctrl_log, stderr=ctrl_log,
        )
        ctrl_log.close()
        self._procs.append(self._controller_proc)

        # Poll controller health until ready
        if not await self._wait_for_health(self.controller_port, timeout=60,
                                           label="controller", path="/health"):
            raise RuntimeError("Controller failed to start")

        logger.info("All services started successfully")

    def stop_services(self):
        """Stop all managed services."""
        logger.info("Stopping managed services...")

        # Kill controller first (it will drain workers)
        if self._controller_proc and self._controller_proc.poll() is None:
            logger.info("Stopping controller...")
            self._controller_proc.terminate()
            try:
                self._controller_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._controller_proc.kill()
                self._controller_proc.wait()

        # Kill aggregator(s)
        if self._aggregator_proc and self._aggregator_proc.poll() is None:
            logger.info("Stopping aggregator...")
            self._aggregator_proc.terminate()
            try:
                self._aggregator_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._aggregator_proc.kill()
                self._aggregator_proc.wait()

        # Kill any remaining worker/aggregator processes
        for proc in self._procs:
            if proc and proc.poll() is None:
                proc.kill()
                proc.wait()

        logger.info("All managed services stopped")

    async def _wait_for_health(self, port: int, timeout: float = 120,
                               label: str = "service", path: str = "/health") -> bool:
        """Poll health endpoint until ready."""
        url = f"http://127.0.0.1:{port}{path}"
        start = time.time()
        async with aiohttp.ClientSession() as session:
            while time.time() - start < timeout:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=2.0)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            status = data.get('status', '')
                            if status in ('ready', 'healthy'):
                                elapsed = time.time() - start
                                logger.info(f"{label} ready ({elapsed:.1f}s)")
                                return True
                except Exception:
                    pass
                await asyncio.sleep(1.0)
        logger.error(f"{label} failed to become ready within {timeout}s")
        return False


async def main():
    parser = argparse.ArgumentParser(description="Benchmark Serverless LLM Platform")

    # API settings
    parser.add_argument("--api-url", type=str, default=f"http://localhost:{CONTROLLER_PORT}")

    # Trace settings
    parser.add_argument("--trace-file", type=str, help="Path to trace file")

    # Synthetic workload settings
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--request-rate", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=60.0)

    # Request settings
    parser.add_argument("--adapters", type=str, nargs="+", default=["default"])
    parser.add_argument("--default-adapter", type=str, default=None,
                        help="Override all trace adapter IDs with this adapter name")
    parser.add_argument("--adapter-map-prefix", type=str, default=None,
                        help="Map trace adapter IDs by prepending prefix, e.g. 'sim-adapters/lora-'")
    parser.add_argument("--prompt-tokens", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=64)

    # Output
    parser.add_argument("--output-dir", type=str, default="./benchmark_results")

    # Managed lifecycle
    parser.add_argument("--managed", action="store_true",
                        help="Manage full lifecycle: start MPS, aggregator, controller, run benchmark, cleanup")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs (default: auto-detect). Only used with --managed.")
    parser.add_argument("--scheduler", type=str, default="inter",
                        help="Scheduler algorithm (default: inter). Only used with --managed.")
    parser.add_argument("--controller-port", type=int, default=8344,
                        help="Controller port (default: 8344). Only used with --managed.")

    args = parser.parse_args()

    lifecycle = None
    if args.managed:
        # Use managed controller port for API URL
        api_url = f"http://localhost:{args.controller_port}"
    else:
        api_url = args.api_url

    config = BenchmarkConfig(
        api_url=api_url,
        trace_file=args.trace_file,
        num_requests=args.num_requests,
        request_rate=args.request_rate,
        duration_seconds=args.duration,
        adapters=args.adapters,
        default_adapter=args.default_adapter,
        adapter_map_prefix=args.adapter_map_prefix,
        prompt_tokens=args.prompt_tokens,
        max_tokens=args.max_tokens,
        output_dir=args.output_dir,
    )

    if args.managed:
        lifecycle = ManagedLifecycle(
            num_gpus=args.num_gpus,
            scheduler=args.scheduler,
            controller_port=args.controller_port,
            adapter_map_prefix=args.adapter_map_prefix,
        )

    try:
        if lifecycle:
            await lifecycle.start_services()

        runner = BenchmarkRunner(config)
        results = await runner.run()
    finally:
        if lifecycle:
            lifecycle.stop_services()

    return results


if __name__ == "__main__":
    asyncio.run(main())
