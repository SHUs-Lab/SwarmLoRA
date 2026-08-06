#!/usr/bin/env python3
"""Async Load Balancer / Serverless Controller for Multi-Adapter LoRA Inference."""

import os
import sys
import time
import json
import uuid
import signal
import socket
import asyncio
import subprocess
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, List
from collections import Counter, defaultdict, deque
from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

try:
    from .scheduling.base import LocalScheduler, WorkerSnapshot, RoutingAction, RoutingDecision, LocalSchedulerConfig
    from .scheduling.lorant import LoRantScheduler, LoRantNoSwapScheduler
    from .scheduling.random_sched import RandomScheduler
    from .admission import AdmissionConfig, post_admit_margin
except ImportError:
    from controller.scheduling.base import LocalScheduler, WorkerSnapshot, RoutingAction, RoutingDecision, LocalSchedulerConfig
    from controller.scheduling.lorant import LoRantScheduler, LoRantNoSwapScheduler
    from controller.scheduling.random_sched import RandomScheduler
    from controller.admission import AdmissionConfig, post_admit_margin

from config import (
    MAX_WORKERS,
    WORKER_BASE_PORT, DEFAULT_PORT as AGGREGATOR_PORT,
    CONTROLLER_PORT,
    WORKER_GPUS, MAX_WORKERS_PER_GPU,
    AGGREGATOR_PORTS, AGGREGATOR_HEALTH_PORTS,
)


def log(msg: str) -> None:
    print(f"[Controller] {msg}", flush=True)


# Optional pool-event trace, enabled by setting SWARM_EVENT_LOG to a path.
# Records worker lifecycle transitions with timestamps so that pool size and
# spawn/swap activity can be reconstructed over the course of a run. Written
# server-side only: worker identities are never exposed through the client
# API, which deliberately omits them to avoid leaking topology.
_EVENT_LOG_PATH = os.environ.get("SWARM_EVENT_LOG")
_event_log_fh = None


def event(kind: str, **fields) -> None:
    """Append one pool-lifecycle event. No-op unless SWARM_EVENT_LOG is set."""
    global _event_log_fh
    if not _EVENT_LOG_PATH:
        return
    try:
        if _event_log_fh is None:
            _event_log_fh = open(_EVENT_LOG_PATH, "a", buffering=1)
        _event_log_fh.write(
            json.dumps({"t": time.time(), "event": kind, **fields}) + "\n"
        )
    except Exception:
        pass  # instrumentation must never disturb the run


# CONFIGURATION

LB_PORT: int = CONTROLLER_PORT


# DATA MODELS

class WorkerStatus(Enum):
    """Worker lifecycle states."""
    STARTING = "starting"
    READY = "ready"
    SWAPPING = "swapping"
    UNHEALTHY = "unhealthy"
    STOPPING = "stopping"


@dataclass
class WorkerInfo:
    """Information about a managed worker."""
    worker_id: int
    http_port: int
    adapter_id: Optional[str] = None
    status: WorkerStatus = WorkerStatus.STARTING
    process: Optional[subprocess.Popen] = None
    device: str = "cuda:0"

    # Request tracking (N=1: always 0 or 1)
    active_requests: int = 0

    # Reported metrics from worker /health (updated each health check)
    reported_active_slots: int = 0
    reported_gpu_memory_free_mb: int = 0
    reported_gpu_memory_total_mb: int = 0

    # Stats
    last_health_check: float = 0.0
    spawned_at: float = field(default_factory=time.time)  # For reaping never-used workers
    last_request_at: float = 0.0  # For scale-to-zero idle detection
    requests_handled: int = 0
    total_inference_ms: float = 0.0
    total_swap_ms: float = 0.0

    def avg_inference_ms(self) -> float:
        if self.requests_handled == 0:
            return 0.0
        return self.total_inference_ms / self.requests_handled

    @property
    def idle_time(self) -> float:
        """Seconds since last request (or since spawn if never served)."""
        if self.last_request_at == 0:
            return time.time() - self.spawned_at
        return time.time() - self.last_request_at


class InferenceRequest(BaseModel):
    """Request body for inference endpoint."""
    prompt: str
    adapter_id: Optional[str] = None
    max_tokens: int = 256
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0


class SwapRequest(BaseModel):
    """Request body for adapter swap."""
    adapter_id: str


class SpawnRequest(BaseModel):
    """Request body for spawning workers."""
    adapter_id: Optional[str] = None
    count: int = 1


class PrewarmRequest(BaseModel):
    """Request body for prewarming workers with specific adapters."""
    adapters: Dict[str, int]  # adapter_id -> count
    devices: Optional[Dict[str, int]] = None  # device -> count, e.g. {"cuda:0": 12, "cuda:1": 18}


@dataclass
class PendingRequest:
    """A queued request waiting for a worker to become available."""
    adapter_id: str
    future: asyncio.Future
    deadline: float
    enqueued_at: float = field(default_factory=time.time)
    # Set when a newly spawned worker satisfies this request, so the caller can
    # exclude it from the capacity signal (a spawn is startup, not congestion).
    cold: bool = False


# OpenAI-compatible request models

class ChatMessage(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = False
    arrival_time: Optional[float] = None  # Set by global controller for cross-node TTFT


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = False
    arrival_time: Optional[float] = None  # Set by global controller for cross-node TTFT


# ADAPTER REGISTRY

class AdapterRegistry:
    """Maps adapters <-> workers for fast lookup."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self.adapter_to_workers: Dict[str, Set[int]] = defaultdict(set)
        self.worker_to_adapter: Dict[int, str] = {}

    async def register(self, worker_id: int, adapter_id: str):
        """Register a worker with an adapter."""
        async with self._lock:
            old_adapter = self.worker_to_adapter.get(worker_id)
            if old_adapter:
                self.adapter_to_workers[old_adapter].discard(worker_id)

            self.worker_to_adapter[worker_id] = adapter_id
            self.adapter_to_workers[adapter_id].add(worker_id)

    async def unregister(self, worker_id: int):
        """Remove a worker from the registry."""
        async with self._lock:
            adapter = self.worker_to_adapter.pop(worker_id, None)
            if adapter:
                self.adapter_to_workers[adapter].discard(worker_id)

    async def get_workers_for_adapter(self, adapter_id: str) -> Set[int]:
        """Get all workers with a specific adapter."""
        async with self._lock:
            return self.adapter_to_workers.get(adapter_id, set()).copy()

    async def get_adapter_for_worker(self, worker_id: int) -> Optional[str]:
        """Get the adapter loaded on a worker."""
        async with self._lock:
            return self.worker_to_adapter.get(worker_id)

    async def get_all_adapters(self) -> List[str]:
        """Get all known adapter IDs."""
        async with self._lock:
            return [a for a, workers in self.adapter_to_workers.items() if workers]


# ADAPTER FREQUENCY TRACKER

class AdapterFrequencyTracker:
    """Sliding-window tracker for per-adapter request rates."""

    def __init__(self, window_seconds: float = 60.0):
        self._window = window_seconds
        self._timestamps: Dict[str, List[float]] = {}  # adapter_id -> [timestamps]

    def record_request(self, adapter_id: str):
        """Record a request for an adapter."""
        now = time.time()
        if adapter_id not in self._timestamps:
            self._timestamps[adapter_id] = []
        ts = self._timestamps[adapter_id]
        ts.append(now)
        # Prune on write when lists grow large to prevent unbounded growth
        if len(ts) > 200:
            cutoff = now - self._window
            self._timestamps[adapter_id] = [t for t in ts if t > cutoff]

    def get_request_rate(self, adapter_id: str) -> float:
        """Get requests/sec for an adapter in the recent window."""
        if adapter_id not in self._timestamps:
            return 0.0
        now = time.time()
        cutoff = now - self._window
        ts = self._timestamps[adapter_id]
        self._timestamps[adapter_id] = ts = [t for t in ts if t > cutoff]
        if not ts:
            return 0.0
        return len(ts) / self._window


# FORK-BASED WORKER SPAWNING

class ForkedProcess:
    """Popen-compatible wrapper for fork()'d workers."""

    def __init__(self, pid: int):
        self.pid = pid
        self.returncode = None

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        try:
            os.kill(self.pid, 0)
            return None
        except ProcessLookupError:
            self.returncode = 0
            return 0
        except PermissionError:
            # Process exists but we can't signal it — treat as alive
            return None

    def terminate(self):
        try:
            os.kill(self.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def kill(self):
        try:
            os.kill(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def wait(self, timeout=None):
        start = time.time()
        while timeout is None or time.time() - start < timeout:
            if self.poll() is not None:
                return self.returncode
            time.sleep(0.1)
        raise subprocess.TimeoutExpired(f"forked-worker-{self.pid}", timeout)


# ASYNC LOAD BALANCER

class AsyncLoadBalancer:
    """Async load balancer: one request per worker, swap-first scheduling via LoRant."""

    def __init__(
        self,
        max_workers: int = MAX_WORKERS,
        base_port: int = WORKER_BASE_PORT,
        aggregator_host: str = "localhost",
        aggregator_port: int = AGGREGATOR_PORT,
        default_adapter: str = None,
        min_workers: int = 0,
        scale_down_delay: float = 60.0,
        worker_devices: list = None,
        scheduler: LocalScheduler = None,
        scheduler_config: LocalSchedulerConfig = None,
        use_fork: bool = True,
        gpu_worker_caps: Dict[str, int] = None,
        aggregator_port_map: Dict[str, int] = None,
        aggregator_health_port_map: Dict[str, int] = None,
    ):
        self.max_workers = max_workers
        self.base_port = base_port
        self.aggregator_host = aggregator_host
        self.aggregator_port = aggregator_port
        self.aggregator_port_map = aggregator_port_map  # {device: port} e.g. {"cuda:0": 50056, "cuda:1": 50057}
        self.default_adapter = default_adapter
        self.min_workers = min_workers
        self.scale_down_delay = scale_down_delay  # Seconds idle before scale-down
        self.worker_devices = worker_devices or ["cuda:0"]
        self.scheduler_config = scheduler_config or LocalSchedulerConfig()
        self.scheduler = scheduler or LoRantScheduler(cfg=self.scheduler_config)
        self.gpu_worker_caps = gpu_worker_caps  # Per-GPU max workers (None = no caps)
        self._gpu_worker_count: Dict[str, int] = {d: 0 for d in self.worker_devices}

        # Demand-driven aggregator tracking
        self.aggregator_health_port_map = aggregator_health_port_map  # {device: port} e.g. {"cuda:0": 8000, "cuda:1": 8001}
        self._aggregator_processes: Dict[str, subprocess.Popen] = {}  # device -> Popen
        self._aggregator_ready: Dict[str, bool] = {}  # device -> ready
        self._aggregator_launching: Dict[str, asyncio.Lock] = {}  # created lazily in event loop

        self._lock = asyncio.Lock()
        self._routing_lock = asyncio.Lock()  # Serializes snapshot+decide+reserve to prevent double-routing
        self.workers: Dict[int, WorkerInfo] = {}
        self.registry = AdapterRegistry()

        # HTTP session for async requests
        self._session: Optional[aiohttp.ClientSession] = None

        # Adapter frequency tracker for swap decisions
        self.frequency_tracker = AdapterFrequencyTracker()

        # Stats
        self.stats = {
            'requests_total': 0,
            'requests_matched': 0,
            'requests_swapped': 0,
            'requests_spawned': 0,
            'requests_queued': 0,
            'workers_scaled_down': 0,  # Scale-to-zero terminations
            'total_swap_ms': 0.0,
            'total_inference_ms': 0.0,
            'queue_depth': 0,
        }

        # Background tasks
        self._running = False
        self._health_task: Optional[asyncio.Task] = None
        self._idle_reaper_task: Optional[asyncio.Task] = None
        self._idempotency_cleanup_task: Optional[asyncio.Task] = None

        # Idempotency cache: key -> (timestamp, result)
        self._idempotency_cache: Dict[str, tuple] = {}
        self._idempotency_ttl = 3600.0
        self._idempotency_lock = asyncio.Lock()

        # Request deadline: each request has until arrival+SLO, enforced here
        # and again in the worker via slo_deadline_wall.
        self._admission = AdmissionConfig.from_env()

        # Timestamps of requests dropped for want of a worker, kept for a
        # short window and fed to the scaler so demand the pool failed to
        # serve stays visible to it.
        self._recent_drops: deque = deque()
        self._drop_window_s: float = 5.0

        # E1: measured work remaining AFTER a queued request is given a worker
        # (swap + worker-side queue + prefill). Drives the dynamic stage-1
        # deadline: a request is dropped once `elapsed + E1` would exceed its
        # target, rather than waiting out the full budget and then failing.
        # Mostly queueing, not compute, so it tracks how loaded the system is.
        self._e1_s = 0.0
        self._e1_n = 0

        # Event-driven dispatch queue (replaces per-request polling)
        self._pending_requests: deque = deque()
        self._worker_available = asyncio.Event()
        self._dispatcher_task: Optional[asyncio.Task] = None
        self._spawns_in_progress: List[str] = []  # tracks adapter_id per in-flight spawn

        # Burst detection
        self._recent_arrivals: List[float] = []
        self._burst_window: float = 2.0      # seconds to detect burst
        self._burst_threshold: int = 10       # requests in window to trigger burst
        self._burst_spawn_cap: int = 20       # max extra spawns per burst detection

        # Fork-based worker spawning
        self._use_fork = use_fork
        self._template_process: Optional[subprocess.Popen] = None
        self._template_socket_path: Optional[str] = None
        self._template_ready = False
        self._project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._logs_dir = os.path.join(self._project_root, "logs")

    async def start(self):
        """Start the load balancer."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300)  # Worker inference timeout (fixed, not queue_timeout)
        )
        self._running = True
        if self._use_fork:
            await self._start_template()
        self._health_task = asyncio.create_task(self._health_check_loop())
        self._idle_reaper_task = asyncio.create_task(self._idle_worker_reaper_loop())
        self._idempotency_cleanup_task = asyncio.create_task(self._idempotency_cleanup_loop())
        self._dispatcher_task = asyncio.create_task(self._dispatch_loop())
        self._proactive_task: Optional[asyncio.Task] = None
        if isinstance(self.scheduler, LoRantScheduler):
            self._proactive_task = asyncio.create_task(self._proactive_scaling_loop())
        log(f"Async load balancer started (min_workers={self.min_workers}, scale_down_delay={self.scale_down_delay}s, fork={'enabled' if self._template_ready else 'disabled'})")

        # GPU 1 aggregator launches on-demand when GPU 0 fills to 80%.

    async def stop(self):
        """Stop the load balancer and cleanup."""
        self._running = False

        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        if self._idle_reaper_task:
            self._idle_reaper_task.cancel()
            try:
                await self._idle_reaper_task
            except asyncio.CancelledError:
                pass

        if self._idempotency_cleanup_task:
            self._idempotency_cleanup_task.cancel()
            try:
                await self._idempotency_cleanup_task
            except asyncio.CancelledError:
                pass

        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                pass

        if self._proactive_task:
            self._proactive_task.cancel()
            try:
                await self._proactive_task
            except asyncio.CancelledError:
                pass

        # Cancel all pending queued requests
        while self._pending_requests:
            pr = self._pending_requests.popleft()
            if not pr.future.done():
                pr.future.cancel()

        # Drain in-flight requests before killing workers (max 30s)
        drain_start = time.time()
        drain_timeout = 30.0
        while time.time() - drain_start < drain_timeout:
            total_active = sum(w.active_requests for w in self.workers.values())
            if total_active == 0:
                break
            remaining = drain_timeout - (time.time() - drain_start)
            log(f"Draining {total_active} in-flight request(s)... ({remaining:.0f}s remaining)")
            await asyncio.sleep(1.0)

        # Stop all workers
        log("Stopping all workers...")
        stop_tasks = [self._stop_worker(w) for w in list(self.workers.values())]
        await asyncio.gather(*stop_tasks, return_exceptions=True)

        # Stop template process
        if self._template_process:
            self._stop_template()

        # Stop controller-launched aggregator processes
        for device, proc in self._aggregator_processes.items():
            if proc and proc.poll() is None:
                log(f"Terminating aggregator on {device} (pid={proc.pid})")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        self._aggregator_processes.clear()
        self._aggregator_ready.clear()

        if self._session:
            await self._session.close()

        log("Async load balancer stopped")

    # --------------------------------------------------------------------------
    # Fork-Based Worker Spawning
    # --------------------------------------------------------------------------

    async def _start_template(self):
        """Launch the template process and wait for it to be ready."""
        os.makedirs(self._logs_dir, exist_ok=True)
        self._template_socket_path = f"/tmp/template_{os.getpid()}.sock"

        # Clean stale socket
        if os.path.exists(self._template_socket_path):
            try:
                os.unlink(self._template_socket_path)
            except OSError:
                pass

        cmd = [
            sys.executable, os.path.join("src", "worker", "template_process.py"),
            "--socket", self._template_socket_path,
        ]
        template_log = open(os.path.join(self._logs_dir, "template.log"), "a")
        self._template_process = subprocess.Popen(
            cmd,
            cwd=self._project_root,
            stdout=template_log,
            stderr=template_log,
        )
        template_log.close()

        # Wait for socket file to appear (imports can take ~120s with vllm/torch)
        start = time.time()
        timeout = 180.0
        while time.time() - start < timeout:
            if self._template_process.poll() is not None:
                log(f"Template process died during startup (rc={self._template_process.returncode})")
                self._template_process = None
                return
            if os.path.exists(self._template_socket_path):
                self._template_ready = True
                log(f"Template process ready (pid={self._template_process.pid}, took {time.time() - start:.1f}s)")
                return
            await asyncio.sleep(0.2)

        log(f"Template process startup timed out after {timeout}s, falling back to subprocess spawning")
        self._template_process.terminate()
        self._template_process = None

    def _fork_worker(self, http_port: int, host: str, port: int, device: str,
                     adapter_id: str, worker_log_path: str) -> Optional[ForkedProcess]:
        """Send a fork command to the template process via Unix socket."""
        if not self._template_ready or not self._template_socket_path:
            return None

        import errno
        max_retries = 3
        for attempt in range(max_retries):
            sock = None
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect(self._template_socket_path)

                cmd = {
                    "http_port": http_port,
                    "host": host,
                    "port": port,
                    "device": device,
                    "lora": adapter_id,
                    "worker_log_path": worker_log_path,
                }
                sock.sendall(json.dumps(cmd).encode())
                sock.shutdown(socket.SHUT_WR)  # Signal end of message

                data = sock.recv(4096)
                sock.close()
                sock = None

                resp = json.loads(data.decode())
                if "pid" in resp:
                    return ForkedProcess(resp["pid"])
                else:
                    log(f"Template fork error: {resp.get('error', 'unknown')}")
                    return None

            except OSError as e:
                if sock:
                    sock.close()
                if e.errno in (errno.EAGAIN, errno.ECONNREFUSED) and attempt < max_retries - 1:
                    import time
                    time.sleep(0.05 * (attempt + 1))
                    continue
                if self._template_process and self._template_process.poll() is not None:
                    log(f"Template communication failed (process dead): {e}, disabling fork")
                    self._template_ready = False
                else:
                    log(f"Template fork failed (transient): {e}, falling back to subprocess")
                return None
            except Exception as e:
                if sock:
                    sock.close()
                if self._template_process and self._template_process.poll() is not None:
                    log(f"Template communication failed (process dead): {e}, disabling fork")
                    self._template_ready = False
                else:
                    log(f"Template fork failed (transient): {e}, falling back to subprocess")
                return None

    def _stop_template(self):
        """Stop the template process."""
        if not self._template_process:
            return

        # Try graceful shutdown via socket
        if self._template_socket_path and os.path.exists(self._template_socket_path):
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                sock.connect(self._template_socket_path)
                sock.sendall(json.dumps({"command": "shutdown"}).encode())
                sock.shutdown(socket.SHUT_WR)
                sock.recv(4096)
                sock.close()
            except Exception:
                pass

        # Terminate process
        try:
            self._template_process.terminate()
            self._template_process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._template_process.kill()
            self._template_process.wait(timeout=2.0)
        except Exception:
            pass

        self._template_process = None
        self._template_ready = False

        # Clean up socket file
        if self._template_socket_path:
            try:
                os.unlink(self._template_socket_path)
            except OSError:
                pass

        log("Template process stopped")

    def _launch_worker_process(self, worker_id: int, http_port: int,
                               device: str, adapter_id: str):
        """Launch a worker via fork (preferred) or subprocess (fallback)."""
        os.makedirs(self._logs_dir, exist_ok=True)
        worker_log_path = os.path.join(self._logs_dir, f"worker_{worker_id}.log")

        # Resolve aggregator port for this worker's GPU
        agg_port = self.aggregator_port
        if self.aggregator_port_map and device in self.aggregator_port_map:
            agg_port = self.aggregator_port_map[device]

        # Try fork first
        if self._template_ready:
            proc = self._fork_worker(
                http_port, self.aggregator_host, agg_port,
                device, adapter_id, worker_log_path,
            )
            if proc:
                return proc, "fork"

        # Subprocess fallback
        # Set per-GPU MPS env vars if per-GPU MPS daemons are running
        env = os.environ.copy()
        worker_device = device
        gpu_idx = device.replace("cuda:", "")
        mps_pipe_dir = f"/tmp/mps_{gpu_idx}"
        if os.path.exists(mps_pipe_dir):
            env["CUDA_VISIBLE_DEVICES"] = "0"
            env["CUDA_MPS_PIPE_DIRECTORY"] = mps_pipe_dir
            worker_device = "cuda:0"

        cmd = [
            sys.executable, os.path.join("src", "worker", "worker_sync.py"),
            "--http-port", str(http_port),
            "--host", self.aggregator_host,
            "--port", str(agg_port),
            "--device", worker_device,
        ]
        if adapter_id:
            cmd.extend(["--lora", adapter_id])

        worker_log = open(worker_log_path, "a")
        proc = subprocess.Popen(
            cmd,
            cwd=self._project_root,
            stdout=worker_log,
            stderr=worker_log,
            env=env,
        )
        worker_log.close()
        return proc, "subprocess"

    # --------------------------------------------------------------------------
    # Demand-Driven Aggregator Launch
    # --------------------------------------------------------------------------

    _AGG_PRELAUNACH_RATIO = 0.8  # Pre-launch next aggregator when GPU is this full

    def _maybe_prelaunach_next_aggregator(self, device: str):
        """If this GPU is ~80% full, fire-and-forget pre-launch of the next GPU's aggregator."""
        if self.gpu_worker_caps is None:
            return
        cap = self.gpu_worker_caps.get(device, self.max_workers)
        count = self._gpu_worker_count.get(device, 0)
        if count < int(cap * self._AGG_PRELAUNACH_RATIO):
            return
        # Find the next device in the list
        try:
            idx = self.worker_devices.index(device)
        except ValueError:
            return
        if idx + 1 >= len(self.worker_devices):
            return
        next_dev = self.worker_devices[idx + 1]
        if self._aggregator_ready.get(next_dev, False):
            return
        log(f"GPU {device} at {count}/{cap} — pre-launching aggregator on {next_dev}")
        asyncio.create_task(self._ensure_aggregator(next_dev))

    async def _ensure_aggregator(self, device: str) -> bool:
        """Ensure an aggregator is running for the given device. Returns True if ready."""
        # Fast path: already known ready
        if self._aggregator_ready.get(device, False):
            return True

        # Check if pre-launched (poll health endpoint)
        health_port = self._get_aggregator_health_port(device)
        if health_port and await self._poll_aggregator_health(health_port):
            self._aggregator_ready[device] = True
            log(f"Aggregator on {device} detected as pre-launched (health port {health_port})")
            return True

        # Not running — launch it (create lock lazily inside event loop)
        if device not in self._aggregator_launching:
            self._aggregator_launching[device] = asyncio.Lock()
        async with self._aggregator_launching[device]:
            # Re-check after acquiring lock (another task may have launched it)
            if self._aggregator_ready.get(device, False):
                return True
            return await self._launch_aggregator(device)

    async def _launch_aggregator(self, device: str) -> bool:
        """Launch an aggregator on the given device using P2P weight copy from a donor."""
        # Find any ready aggregator as donor
        donor_device = None
        for dev, ready in self._aggregator_ready.items():
            if ready:
                donor_device = dev
                break

        if donor_device is None:
            log(f"Cannot launch aggregator on {device}: no donor aggregator available")
            return False

        donor_tcp_port = self._get_aggregator_tcp_port(donor_device)
        target_tcp_port = self._get_aggregator_tcp_port(device)
        target_health_port = self._get_aggregator_health_port(device)

        if not all([donor_tcp_port, target_tcp_port, target_health_port]):
            log(f"Cannot launch aggregator on {device}: missing port config")
            return False

        gpu_idx = device.replace("cuda:", "")
        mps_pipe_dir = f"/tmp/mps_{gpu_idx}"

        env = os.environ.copy()
        # Per-GPU MPS: client CVD must be 0 (MPS server remaps physical GPU N to device 0)
        if os.path.exists(mps_pipe_dir):
            env["CUDA_VISIBLE_DEVICES"] = "0"
            env["CUDA_MPS_PIPE_DIRECTORY"] = mps_pipe_dir
            agg_device = "cuda:0"
        else:
            agg_device = device

        cmd = [
            sys.executable, os.path.join("src", "aggregator.py"),
            "--device", agg_device,
            "--port", str(target_tcp_port),
            "--health-port", str(target_health_port),
            "--donor-host", "localhost",
            "--donor-port", str(donor_tcp_port),
        ]

        log(f"Launching aggregator on {device} (donor={donor_device}, "
            f"tcp={target_tcp_port}, health={target_health_port})...")

        os.makedirs(self._logs_dir, exist_ok=True)
        log_path = os.path.join(self._logs_dir, f"aggregator_{gpu_idx}.log")
        log_file = open(log_path, "a")
        proc = subprocess.Popen(
            cmd,
            cwd=self._project_root,
            stdout=log_file,
            stderr=log_file,
            env=env,
        )
        log_file.close()
        self._aggregator_processes[device] = proc

        # Poll health endpoint until ready
        start = time.time()
        timeout = 120.0
        while time.time() - start < timeout:
            if proc.poll() is not None:
                log(f"Aggregator on {device} died during startup (exit={proc.returncode})")
                return False
            if await self._poll_aggregator_health(target_health_port):
                self._aggregator_ready[device] = True
                log(f"Aggregator on {device} ready via P2P ({time.time()-start:.1f}s)")
                return True
            await asyncio.sleep(0.1)

        log(f"Aggregator on {device} failed to start within {timeout}s")
        proc.terminate()
        return False

    async def _poll_aggregator_health(self, health_port: int) -> bool:
        """Check if an aggregator is healthy on the given port."""
        try:
            if self._session is None:
                return False
            url = f"http://127.0.0.1:{health_port}/health"
            async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=2.0)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('status') == 'ready'
        except Exception:
            pass
        return False

    def _get_aggregator_tcp_port(self, device: str) -> Optional[int]:
        """Get the TCP registration port for a device's aggregator."""
        if self.aggregator_port_map and device in self.aggregator_port_map:
            return self.aggregator_port_map[device]
        gpu_idx = int(device.replace("cuda:", ""))
        return AGGREGATOR_PORTS.get(gpu_idx, None)

    def _get_aggregator_health_port(self, device: str) -> Optional[int]:
        """Get the health port for a device's aggregator."""
        if self.aggregator_health_port_map and device in self.aggregator_health_port_map:
            return self.aggregator_health_port_map[device]
        gpu_idx = int(device.replace("cuda:", ""))
        return AGGREGATOR_HEALTH_PORTS.get(gpu_idx, None)

    # --------------------------------------------------------------------------
    # Health Monitoring
    # --------------------------------------------------------------------------

    async def _health_check_loop(self):
        """Periodically check worker health."""
        while self._running:
            await asyncio.sleep(self.scheduler_config.health_check_interval)
            await self._check_all_workers()
            # Restart template if it died
            if self._use_fork and not self._template_ready:
                if self._template_process is None or self._template_process.poll() is not None:
                    log("Template process died, restarting...")
                    await self._start_template()
                elif os.path.exists(self._template_socket_path):
                    log("Template process alive, re-enabling fork")
                    self._template_ready = True

    async def _check_all_workers(self):
        """Check health of all workers concurrently."""
        async with self._lock:
            workers = list(self.workers.values())

        tasks = [self._check_worker_health(w) for w in workers]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_worker_health(self, worker: WorkerInfo):
        """Check if a worker is healthy and update capacity info."""
        try:
            url = f"http://127.0.0.1:{worker.http_port}/health"
            async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=self.scheduler_config.worker_http_timeout)) as resp:
                if resp.status == 200:
                    data = await resp.json()

                    # Verify status field
                    if data.get('status') != 'ready':
                        if worker.status != WorkerStatus.STARTING:
                            worker.status = WorkerStatus.UNHEALTHY
                        return

                    # Ingest worker-reported metrics
                    worker.reported_active_slots = data.get('active_slots', 0)
                    worker.reported_gpu_memory_free_mb = data.get('gpu_memory_free_mb', 0)
                    worker.reported_gpu_memory_total_mb = data.get('gpu_memory_total_mb', 0)

                    worker.last_health_check = time.time()

                    if worker.status == WorkerStatus.UNHEALTHY:
                        worker.status = WorkerStatus.READY
                        log(f"Worker {worker.worker_id} recovered")
                        self._worker_available.set()  # Wake dispatcher — worker recovered
                    elif worker.status == WorkerStatus.STARTING:
                        worker.status = WorkerStatus.READY
                        self._worker_available.set()  # Wake dispatcher — worker became ready

                    # Get adapter info
                    await self._update_worker_adapter_info(worker)

                elif resp.status == 503:
                    # Still initializing
                    if worker.status != WorkerStatus.STARTING:
                        worker.status = WorkerStatus.UNHEALTHY
                else:
                    worker.status = WorkerStatus.UNHEALTHY

        except asyncio.TimeoutError:
            await self._handle_worker_failure(worker, "timeout")
        except Exception as e:
            await self._handle_worker_failure(worker, str(e))

    async def _update_worker_adapter_info(self, worker: WorkerInfo):
        """Update adapter info from worker."""
        try:
            url = f"http://127.0.0.1:{worker.http_port}/adapter_info"
            async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=self.scheduler_config.worker_http_timeout)) as resp:
                if resp.status == 200:
                    info = await resp.json()
                    adapter = info.get('adapter_id')
                    if adapter:
                        worker.adapter_id = adapter
                        await self.registry.register(worker.worker_id, adapter)
                    if info.get('is_swapping'):
                        worker.status = WorkerStatus.SWAPPING
        except Exception:
            pass

    async def _handle_worker_failure(self, worker: WorkerInfo, reason: str):
        """Handle worker failure - check if process died."""
        if worker.process and worker.process.poll() is not None:
            log(f"Worker {worker.worker_id} process died (exit={worker.process.returncode})")
            await self._remove_worker(worker.worker_id)
        else:
            worker.status = WorkerStatus.UNHEALTHY

    # --------------------------------------------------------------------------
    # Scale-to-Zero: Idle Worker Reaper
    # --------------------------------------------------------------------------

    async def _idle_worker_reaper_loop(self):
        """Periodically check for and terminate idle workers (scale-to-zero)."""
        while self._running:
            await asyncio.sleep(self.scheduler_config.health_check_interval)
            if self.scale_down_delay > 0:
                await self._reap_idle_workers()

    async def _reap_idle_workers(self):
        """Terminate workers that have been idle longer than scale_down_delay."""
        # Don't reap while requests are waiting — they need workers
        if self._pending_requests:
            return
        async with self._lock:
            snapshots = self._build_worker_snapshots()

        decision = self.scheduler.select_workers_to_reap(
            snapshots, self.min_workers, self.scale_down_delay,
        )
        if not decision.worker_ids:
            return

        to_stop = []
        # Hold both locks: _routing_lock prevents routing from incrementing
        # active_requests on a worker we're about to reap (race condition
        # that caused aggregator crashes when a worker died mid-slot).
        async with self._routing_lock:
            async with self._lock:
                current_count = len(self.workers)
                for wid in decision.worker_ids:
                    if current_count <= self.min_workers:
                        break
                    worker = self.workers.get(wid)
                    if worker is None:
                        continue
                    # Re-check live state inside lock — a request may have arrived
                    # between snapshot and now
                    if worker.active_requests > 0:
                        continue
                    if worker.status != WorkerStatus.READY:
                        continue

                    # Mark STOPPING before removing — prevents route_request from
                    # sending new requests to this worker
                    worker.status = WorkerStatus.STOPPING
                    log(f"Scale-to-zero: stopping worker {wid} (idle {worker.idle_time:.1f}s)")
                    self.stats['workers_scaled_down'] += 1
                    self.workers.pop(wid, None)
                    dev = worker.device
                    self._gpu_worker_count[dev] = max(0, self._gpu_worker_count.get(dev, 1) - 1)
                    current_count -= 1
                    to_stop.append(worker)

        # Unregister + kill outside lock to avoid blocking other operations
        for worker in to_stop:
            await self.registry.unregister(worker.worker_id)
            log(f"Worker {worker.worker_id} removed")
            asyncio.create_task(self._stop_worker_process(worker))

    # --------------------------------------------------------------------------
    # Worker Management
    # --------------------------------------------------------------------------

    def _get_next_worker_id(self) -> int:
        """Get the next available worker ID."""
        for i in range(self.max_workers):
            if i not in self.workers:
                return i
        return -1

    def _pick_device(self, worker_id: int) -> Optional[str]:
        """Pick a GPU device for a new worker, respecting per-GPU caps."""
        for dev in self.worker_devices:
            if self.gpu_worker_caps is None:
                return dev
            cap = self.gpu_worker_caps.get(dev, self.max_workers)
            if self._gpu_worker_count.get(dev, 0) < cap:
                return dev
        return None

    async def spawn_worker(self, adapter_id: str) -> Optional[WorkerInfo]:
        """Spawn a new worker with the given adapter."""
        async with self._lock:
            worker_id = self._get_next_worker_id()
            if worker_id < 0:
                log("Cannot spawn worker: at max capacity")
                return None

            device = self._pick_device(worker_id)
            if device is None:
                log("Cannot spawn worker: all GPUs at capacity")
                return None

            # Reserve GPU slot immediately so concurrent spawns see correct count
            http_port = self.base_port + worker_id * 10
            self._gpu_worker_count[device] = self._gpu_worker_count.get(device, 0) + 1

            # Pre-launch next GPU's aggregator when this GPU is ~80% full
            self._maybe_prelaunach_next_aggregator(device)

        # Ensure aggregator is running for this device (outside lock to avoid blocking)
        agg_ok = await self._ensure_aggregator(device)
        if not agg_ok:
            log(f"Cannot spawn worker: aggregator launch failed for {device}")
            async with self._lock:
                self._gpu_worker_count[device] -= 1
            return None

        async with self._lock:
            # Re-validate worker_id — another spawn may have claimed it while lock was released
            if worker_id in self.workers:
                worker_id = self._get_next_worker_id()
                if worker_id < 0:
                    log("Cannot spawn worker: at max capacity (after aggregator launch)")
                    self._gpu_worker_count[device] -= 1
                    return None
                http_port = self.base_port + worker_id * 10

            proc, method = self._launch_worker_process(worker_id, http_port, device, adapter_id)
            log(f"Spawning worker {worker_id} on port {http_port} device {device} "
                f"with adapter {adapter_id} via {method}...")
            event("spawn_start", worker_id=worker_id, adapter_id=adapter_id,
                  device=device, method=method, pool_size=len(self.workers) + 1)

            worker = WorkerInfo(
                worker_id=worker_id,
                http_port=http_port,
                adapter_id=adapter_id,
                status=WorkerStatus.STARTING,
                process=proc,
                device=device,
            )

            self.workers[worker_id] = worker

        # Wait for worker to be ready (outside lock)
        spawn_started = time.time()
        ready = await self._wait_for_worker_ready(worker)
        if ready:
            await self.registry.register(worker_id, adapter_id)
            event("spawn_ready", worker_id=worker_id, adapter_id=adapter_id,
                  spawn_ms=round((time.time() - spawn_started) * 1000, 1),
                  pool_size=len(self.workers))
            self._worker_available.set()  # Wake dispatcher — new worker ready
            return worker
        else:
            event("spawn_failed", worker_id=worker_id, adapter_id=adapter_id,
                  pool_size=len(self.workers))
            await self._stop_worker(worker)
            return None

    async def _wait_for_worker_ready(self, worker: WorkerInfo) -> bool:
        """Wait for worker to become ready."""
        start = time.time()
        while time.time() - start < self.scheduler_config.spawn_timeout:
            # Fast-fail: if the process has already exited (e.g. init error), don't
            # keep polling HTTP until the full spawn_timeout expires.
            if worker.process and worker.process.poll() is not None:
                log(f"Worker {worker.worker_id} process exited during startup "
                    f"(exit_code={worker.process.poll()})")
                return False
            try:
                url = f"http://127.0.0.1:{worker.http_port}/health"
                async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=self.scheduler_config.worker_http_timeout)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('status') == 'ready':
                            worker.status = WorkerStatus.READY
                            worker.spawned_at = time.time()  # Reset so idle_time starts from ready, not launch
                            log(f"Worker {worker.worker_id} ready (startup={time.time()-start:.1f}s)")
                            return True
            except Exception:
                pass
            await asyncio.sleep(self.scheduler_config.worker_ready_poll_interval)

        log(f"Worker {worker.worker_id} failed to start within timeout")
        return False

    async def _stop_worker(self, worker: WorkerInfo):
        """Stop a worker."""
        worker.status = WorkerStatus.STOPPING
        event("reap", worker_id=worker.worker_id, adapter_id=worker.adapter_id,
              pool_size=len(self.workers) - 1)

        try:
            url = f"http://127.0.0.1:{worker.http_port}/shutdown"
            async with self._session.post(url, timeout=aiohttp.ClientTimeout(total=self.scheduler_config.worker_shutdown_timeout)) as resp:
                pass
        except Exception:
            pass

        if worker.process:
            try:
                # Give the worker time to drain active slots (up to 30s)
                for _ in range(60):
                    if worker.process.poll() is not None:
                        break
                    await asyncio.sleep(0.5)
                else:
                    worker.process.terminate()
                    await asyncio.sleep(self.scheduler_config.worker_kill_grace_period)
                    if worker.process.poll() is None:
                        worker.process.kill()
                # Reap the process to avoid zombies
                worker.process.wait()
            except Exception:
                pass

        await self._remove_worker(worker.worker_id)

    async def _remove_worker(self, worker_id: int):
        """Remove worker from tracking."""
        async with self._lock:
            worker = self.workers.pop(worker_id, None)
            if worker:
                dev = worker.device
                self._gpu_worker_count[dev] = max(0, self._gpu_worker_count.get(dev, 1) - 1)
        await self.registry.unregister(worker_id)
        log(f"Worker {worker_id} removed")

    async def _stop_worker_process(self, worker: WorkerInfo):
        """Stop a worker's process (without lock, for use after removal from dict)."""
        try:
            url = f"http://127.0.0.1:{worker.http_port}/shutdown"
            async with self._session.post(url, timeout=aiohttp.ClientTimeout(total=self.scheduler_config.worker_shutdown_timeout)) as resp:
                pass
        except Exception:
            pass

        if worker.process:
            try:
                # Give the worker time to drain active slots (up to 30s)
                for _ in range(60):
                    if worker.process.poll() is not None:
                        break
                    await asyncio.sleep(0.5)
                else:
                    # Still alive after grace period — force kill
                    worker.process.terminate()
                    await asyncio.sleep(self.scheduler_config.worker_kill_grace_period)
                    if worker.process.poll() is None:
                        worker.process.kill()
                # Reap the process to avoid zombies
                worker.process.wait()
            except Exception:
                pass

        await self.registry.unregister(worker.worker_id)
        log(f"Worker {worker.worker_id} stopped")

    # --------------------------------------------------------------------------
    # Adapter Hot-Swap
    # --------------------------------------------------------------------------

    async def _swap_worker_adapter(self, worker: WorkerInfo, adapter_id: str) -> bool:
        """Hot-swap a worker to a new adapter."""
        worker.status = WorkerStatus.SWAPPING
        try:
            url = f"http://127.0.0.1:{worker.http_port}/swap_adapter"
            async with self._session.post(url, json={"adapter_id": adapter_id}) as resp:
                result = await resp.json()

            if result.get('success'):
                swap_ms = result.get('total_time_ms', 0)
                prev_adapter = worker.adapter_id
                worker.adapter_id = adapter_id
                worker.total_swap_ms += swap_ms
                self.stats['total_swap_ms'] += swap_ms
                await self.registry.register(worker.worker_id, adapter_id)
                log(f"Worker {worker.worker_id} swapped to {adapter_id} in {swap_ms:.1f}ms")
                event("swap", worker_id=worker.worker_id, adapter_id=adapter_id,
                      from_adapter=prev_adapter, swap_ms=round(swap_ms, 1),
                      pool_size=len(self.workers))
                self._worker_available.set()  # Wake dispatcher — adapter changed
                return True
            else:
                log(f"Worker {worker.worker_id} swap failed: {result.get('error')}")
                return False
        except Exception as e:
            log(f"Worker {worker.worker_id} swap error: {e}")
            return False
        finally:
            worker.status = WorkerStatus.READY

    # --------------------------------------------------------------------------
    # Request Routing (delegates to self.scheduler)
    # --------------------------------------------------------------------------

    def _build_worker_snapshots(self) -> Dict[int, WorkerSnapshot]:
        """Create read-only snapshots of all workers for the scheduler."""
        return {
            w.worker_id: WorkerSnapshot(
                worker_id=w.worker_id,
                adapter_id=w.adapter_id,
                status=w.status.value,
                active_requests=w.active_requests,
                idle_time=w.idle_time,
                device=w.device,
            )
            for w in self.workers.values()
        }

    async def route_request(self, req: InferenceRequest, arrival_wall_override: float = None) -> dict:
        """Route a request to the optimal worker with capacity tracking."""
        routing_start = time.perf_counter()
        # Use override from global controller (cross-node), else stamp locally
        arrival_wall = arrival_wall_override or time.time()
        cold_start = False

        self.stats['requests_total'] += 1

        adapter_id = req.adapter_id or self.default_adapter

        self.frequency_tracker.record_request(adapter_id)

        # Get workers with this adapter (await must happen before atomic block)
        workers_with_adapter = await self.registry.get_workers_for_adapter(adapter_id)

        # Atomic: snapshot + decide + reserve under routing lock
        # Prevents route_request and _dispatch_loop from double-routing to same worker
        async with self._routing_lock:
            snapshots = self._build_worker_snapshots()
            adapter_rate = self.frequency_tracker.get_request_rate(adapter_id)
            adapter_rates = {
                aid: self.frequency_tracker.get_request_rate(aid)
                for aid in self.frequency_tracker._timestamps
            }
            num_workers = len(self.workers)
            if isinstance(self.scheduler, LoRantScheduler):
                self.scheduler._pending_depth = len(self._pending_requests)
                self.scheduler._recent_drops = self._trim_recent_drops()
            decision = self.scheduler.route_request(
                adapter_id, snapshots, workers_with_adapter,
                adapter_rate, adapter_rates, num_workers, self.max_workers,
            )

            worker = None

            # --- Execute ROUTE action ---
            # Reserve capacity immediately (before any await) to prevent races
            if decision.action == RoutingAction.ROUTE:
                worker = self.workers.get(decision.worker_id)
                if worker:
                    worker.active_requests += 1

        if decision.action == RoutingAction.ROUTE:
            if worker:
                if decision.needs_swap:
                    self.stats['requests_swapped'] += 1
                    if not await self._swap_worker_adapter(worker, adapter_id):
                        worker.active_requests -= 1
                        # Swap failed — fall through to SPAWN
                        decision = RoutingDecision(action=RoutingAction.SPAWN)
                        worker = None
                else:
                    self.stats['requests_matched'] += 1

                # Background spawn: grow pool while serving via swap/match
                if decision.needs_spawn:
                    asyncio.create_task(self._background_grow(adapter_id))

        # --- Execute SPAWN action ---
        if decision.action == RoutingAction.SPAWN:
            # Served off a cold start: excluded from the capacity signal.
            cold_start = True
            self.stats['requests_spawned'] += 1
            worker = await self.spawn_worker(adapter_id)
            if not worker:
                log(f"Spawn failed for {adapter_id}, retrying once...")
                await asyncio.sleep(1.0)
                worker = await self.spawn_worker(adapter_id)
            if worker:
                worker.active_requests += 1
            else:
                decision = RoutingDecision(action=RoutingAction.QUEUE)

        # --- Execute QUEUE action ---
        if decision.action == RoutingAction.QUEUE:
            self.stats['requests_queued'] += 1
            self._detect_burst_and_spawn()
            worker, queued_cold = await self._wait_for_available_worker(
                adapter_id, arrival_wall)
            cold_start = cold_start or queued_cold
        # --- REJECT or exhausted fallbacks ---
        if decision.action == RoutingAction.REJECT or not worker:
            reason = f"action={decision.action.value}" if decision.action == RoutingAction.REJECT else "queue_exhausted"
            if reason == "queue_exhausted":
                self._recent_drops.append(time.time())
            log(f"Request failed for {adapter_id}: {reason} (workers={num_workers}/{self.max_workers})")
            return {"error": f"No workers available ({reason})", "success": False}

        routing_ms = (time.perf_counter() - routing_start) * 1000
        dispatch_wall = time.time()
        result = await self._send_inference(worker, adapter_id, req, arrival_wall)

        # Feed E1: from being handed a worker to producing a first token. The
        # worker anchors ttft_ms at controller ARRIVAL, so subtracting the time
        # spent queueing leaves exactly the post-assignment work.
        if result.get('success') and arrival_wall is not None:
            _obs = result.get('ttft_ms', 0) / 1000.0 - (dispatch_wall - arrival_wall)
            if 0 < _obs < self._admission.slo_s:
                self._e1_s = (_obs if self._e1_n == 0
                              else 0.2 * _obs + 0.8 * self._e1_s)
                self._e1_n += 1


        # Retry once on connection failure (worker died between route and send)
        if not result.get('success') and 'Connect call failed' in result.get('error', ''):
            log(f"Worker {worker.worker_id} unreachable, removing and retrying")
            await self._remove_worker(worker.worker_id)
            retry_worker, _ = await self._wait_for_available_worker(
                adapter_id, arrival_wall)
            if retry_worker:
                retry_worker.active_requests += 1
                routing_ms = (time.perf_counter() - routing_start) * 1000
                result = await self._send_inference(retry_worker, adapter_id, req, arrival_wall)

        result['routing_ms'] = round(routing_ms, 2)
        return result

    def _trim_recent_drops(self) -> int:
        """Drops within the trailing window, discarding older entries.

        A short window keeps this an estimate of CURRENT unmet demand: once the
        pool catches up the drops stop and the term decays to zero on its own,
        so the scaler settles instead of over-provisioning.
        """
        cutoff = time.time() - self._drop_window_s
        while self._recent_drops and self._recent_drops[0] < cutoff:
            self._recent_drops.popleft()
        return len(self._recent_drops)

    async def _wait_for_available_worker(self, adapter_id: str,
                                        arrival_wall: float = None):
        """Wait for a worker via centralized dispatch.

        Returns (worker, cold_start). cold_start is True when the request was
        satisfied by a newly spawned worker rather than an existing one, so the
        caller can exclude it from the capacity signal.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        # The request has until its target to obtain a worker.
        budget = self._admission.slo_s

        # Anchor at ARRIVAL, so stage 1 and stage 2 measure the same interval
        # and all three systems use one definition.
        if arrival_wall is not None:
            deadline = arrival_wall + budget
        else:
            deadline = time.time() + budget
        # Time actually left, not the full budget: a request that already spent
        # time upstream must not get a fresh budget here.
        wait_timeout = max(0.0, deadline - time.time())
        pr = PendingRequest(adapter_id=adapter_id, future=future, deadline=deadline)
        self._pending_requests.append(pr)
        self.stats['queue_depth'] = len(self._pending_requests)

        # Kick the dispatcher immediately for this new entry
        self._worker_available.set()

        try:
            worker = await asyncio.wait_for(future, timeout=wait_timeout)
            return worker, getattr(pr, "cold", False)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return None, False
        finally:
            self.stats['queue_depth'] = len(self._pending_requests)

    async def _dispatch_loop(self):
        """Centralized dispatcher: matches pending requests to workers."""
        while self._running:
            # Wait for a state-change signal (with fallback timeout for safety)
            try:
                await asyncio.wait_for(self._worker_available.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            self._worker_available.clear()

            if not self._pending_requests:
                continue

            # Prune expired/cancelled entries
            now = time.time()
            while self._pending_requests:
                pr = self._pending_requests[0]
                # Dynamic deadline: drop when the work still ahead of this
                # request no longer fits, not merely when the clock runs out.
                # Reserving E1 here avoids handing it a worker and paying for
                # an adapter swap on a request that cannot finish in time.
                _e1 = post_admit_margin(self._e1_s, self._e1_n,
                                        self._admission.slo_s)
                if (pr.future.done() or pr.future.cancelled()
                        or (now + _e1) > pr.deadline):
                    self._pending_requests.popleft()
                    if not pr.future.done():
                        pr.future.cancel()
                else:
                    break

            if not self._pending_requests:
                self.stats['queue_depth'] = 0
                continue

            # Iterate pending requests and try to match each
            still_pending = deque()
            # Same dynamic deadline as the prune above. This is the site that
            # matters most: it runs immediately before a worker is reserved and
            # possibly swapped, so dropping here is what avoids spending that
            # work on a request that can no longer finish in time.
            _e1 = post_admit_margin(self._e1_s, self._e1_n,
                                    self._admission.slo_s)
            for pr in self._pending_requests:
                if (pr.future.done() or pr.future.cancelled()
                        or (time.time() + _e1) > pr.deadline):
                    if not pr.future.done():
                        pr.future.cancel()
                    continue

                workers_with_adapter = await self.registry.get_workers_for_adapter(pr.adapter_id)

                # Atomic: snapshot + decide + reserve under routing lock
                async with self._routing_lock:
                    snapshots = self._build_worker_snapshots()
                    num_workers = len(self.workers)
                    adapter_rate = self.frequency_tracker.get_request_rate(pr.adapter_id)
                    adapter_rates = {
                        aid: self.frequency_tracker.get_request_rate(aid)
                        for aid in self.frequency_tracker._timestamps
                    }

                    if isinstance(self.scheduler, LoRantScheduler):
                        self.scheduler._pending_depth = len(self._pending_requests)
                        self.scheduler._recent_drops = self._trim_recent_drops()
                    decision = self.scheduler.route_request(
                        pr.adapter_id, snapshots, workers_with_adapter,
                        adapter_rate, adapter_rates, num_workers, self.max_workers,
                    )

                    routed_worker = None
                    if decision.action == RoutingAction.ROUTE:
                        routed_worker = self.workers.get(decision.worker_id)
                        if routed_worker:
                            routed_worker.active_requests += 1

                if decision.action == RoutingAction.ROUTE:
                    worker = routed_worker
                    if worker:
                        if decision.needs_swap:
                            # Fire-and-forget swap — don't block the dispatch loop.
                            # Resolve the future after swap completes so the loop
                            # can immediately reserve the next idle worker.
                            asyncio.create_task(
                                self._swap_and_resolve(worker, pr.adapter_id, pr.future)
                            )
                        elif not pr.future.done():
                            pr.future.set_result(worker)
                        else:
                            # Future was cancelled while we worked; undo reservation
                            worker.active_requests -= 1
                    else:
                        still_pending.append(pr)

                elif decision.action == RoutingAction.SPAWN:
                    self._spawns_in_progress.append(pr.adapter_id)
                    asyncio.create_task(self._spawn_for_dispatch(pr.adapter_id))
                    still_pending.append(pr)

                else:
                    # QUEUE or REJECT — keep pending for next cycle
                    still_pending.append(pr)

            self._pending_requests = still_pending
            self.stats['queue_depth'] = len(self._pending_requests)

    def _detect_burst_and_spawn(self):
        """Detect request burst and trigger extra spawns if needed."""
        now = time.time()
        self._recent_arrivals.append(now)
        cutoff = now - self._burst_window
        self._recent_arrivals = [t for t in self._recent_arrivals if t > cutoff]

        if len(self._recent_arrivals) < self._burst_threshold:
            return
        queue_depth = len(self._pending_requests)
        if queue_depth < self._burst_threshold:
            return

        # Count workers + in-flight spawns to avoid overshooting max
        num_workers = len(self.workers)
        spawns_pending = sum(
            1 for w in self.workers.values()
            if w.status == WorkerStatus.STARTING
        ) + len(self._spawns_in_progress)
        total_committed = num_workers + spawns_pending
        if total_committed >= self.max_workers:
            return

        # Burst detected — spawn extra workers for queued adapters
        # Count pending requests per adapter to spawn proportionally
        adapter_demand = Counter(pr.adapter_id for pr in self._pending_requests)
        spawned = 0
        for adapter_id, count in adapter_demand.most_common():
            # Spawn up to `count` workers for this adapter (skip already in-flight)
            in_flight = sum(1 for a in self._spawns_in_progress if a == adapter_id)
            to_spawn = min(count - in_flight, self._burst_spawn_cap - spawned)
            for _ in range(to_spawn):
                if total_committed + spawned >= self.max_workers:
                    break
                self._spawns_in_progress.append(adapter_id)
                asyncio.create_task(self._spawn_for_dispatch(adapter_id))
                spawned += 1
            if spawned >= self._burst_spawn_cap:
                break
            if total_committed + spawned >= self.max_workers:
                break

        if spawned > 0:
            log(f"Burst detected: {len(self._recent_arrivals)} reqs in {self._burst_window}s, "
                f"queue={queue_depth}, spawning {spawned} extra workers")

    async def _swap_and_resolve(self, worker: WorkerInfo, adapter_id: str, future: asyncio.Future):
        """Swap a worker's adapter and resolve the dispatch future when done."""
        if await self._swap_worker_adapter(worker, adapter_id):
            if not future.done():
                future.set_result(worker)
            else:
                worker.active_requests -= 1
        else:
            worker.active_requests -= 1
            if not future.done():
                # Swap failed — re-queue the request for another attempt
                self._pending_requests.append(PendingRequest(
                    adapter_id=adapter_id,
                    future=future,
                    deadline=time.time() + 60.0,
                ))
            self._worker_available.set()

    async def _background_grow(self, adapter_id: str):
        """Spawn a worker in the background to grow the pool (not for a specific request)."""
        try:
            worker = await self.spawn_worker(adapter_id)
            if worker:
                self._worker_available.set()
        except Exception:
            pass

    async def _spawn_for_dispatch(self, adapter_id: str):
        """Spawn a worker and directly assign to a pending request if possible."""
        try:
            worker = await self.spawn_worker(adapter_id)
            if not worker:
                log(f"Dispatch spawn failed for {adapter_id}, retrying once...")
                await asyncio.sleep(1.0)
                worker = await self.spawn_worker(adapter_id)
            # Directly resolve a pending future for this adapter (prevents starvation)
            if worker:
                for pr in self._pending_requests:
                    if not pr.future.done() and not pr.future.cancelled():
                        if pr.adapter_id == adapter_id or worker.adapter_id == pr.adapter_id:
                            worker.active_requests += 1
                            # Served off a spawn: exclude from the capacity signal.
                            pr.cold = True
                            pr.future.set_result(worker)
                            break
        finally:
            try:
                self._spawns_in_progress.remove(adapter_id)
            except ValueError:
                pass
            self._worker_available.set()

    async def _proactive_scaling_loop(self):
        """Background task: rebalance workers and grow pool as needed."""
        await asyncio.sleep(3.0)  # Let the system warm up first
        while self._running:
            try:
                await asyncio.sleep(2.0)

                adapter_rates = {
                    aid: self.frequency_tracker.get_request_rate(aid)
                    for aid in list(self.frequency_tracker._timestamps)
                }
                if not adapter_rates:
                    continue

                async with self._routing_lock:
                    snapshots = self._build_worker_snapshots()
                    num_workers = len(self.workers)
                    spawns_in_flight = sum(
                        1 for w in self.workers.values()
                        if w.status == WorkerStatus.STARTING
                    )

                # Phase 1: Rebalance via swaps (120ms each, no GPU contention)
                if isinstance(self.scheduler, LoRantScheduler):
                    rebalances = self.scheduler.get_proactive_rebalances(
                        snapshots, adapter_rates, num_workers, self.max_workers,
                    )
                    for wid, target_aid in rebalances:
                        worker = self.workers.get(wid)
                        if worker and worker.active_requests == 0:
                            log(f"Proactive rebalance: worker {wid} "
                                f"({worker.adapter_id} -> {target_aid})")
                            asyncio.create_task(
                                self._swap_worker_adapter(worker, target_aid)
                            )

                # Phase 2: Pool growth via spawn (only when pool too small)
                to_spawn = self.scheduler.get_proactive_spawns(
                    snapshots, adapter_rates, num_workers, self.max_workers,
                    spawns_in_flight=spawns_in_flight,
                )

                for adapter_id in to_spawn:
                    self._spawns_in_progress.append(adapter_id)
                    asyncio.create_task(self._spawn_for_dispatch(adapter_id))
                    log(f"Proactive spawn for {adapter_id} "
                        f"(workers={num_workers}/{self.max_workers})")

            except asyncio.CancelledError:
                break
            except Exception as e:
                log(f"Proactive scaling error: {e}")
                await asyncio.sleep(5.0)

    async def _send_inference(self, worker: WorkerInfo, adapter_id: str, req: InferenceRequest, arrival_time: float = None) -> dict:
        """Send inference request to a worker (async, non-blocking)."""
        # Guard: don't send to a worker being stopped
        if worker.status == WorkerStatus.STOPPING:
            worker.active_requests -= 1
            return {"error": "Worker is shutting down", "success": False}

        try:
            url = f"http://127.0.0.1:{worker.http_port}/inference"
            payload = {
                "prompt": req.prompt,
                "max_tokens": req.max_tokens,
                "do_sample": req.do_sample,
                "temperature": req.temperature,
                "top_k": req.top_k,
                "top_p": req.top_p,
                "arrival_time": arrival_time,  # controller perf_counter when request arrived
                # Absolute SLO deadline, so the worker's own queue (slot claim
                # + lock-step batch admission) is bounded by the same budget as
                # the controller queue. Without it the deadline governs only
                # the controller stage, as measured on ServerlessLoRA where the
                # unbounded second stage was 95% of TTFT.
                "slo_deadline_wall": (
                    arrival_time + self._admission.slo_s
                    if arrival_time is not None else None
                ),
            }

            async with self._session.post(url, json=payload) as resp:
                result = await resp.json()

            # Update stats
            worker.requests_handled += 1
            if result.get('success'):
                e2e = result.get('e2e_ms', 0)
                worker.total_inference_ms += e2e
                self.stats['total_inference_ms'] += e2e

            # Add routing info
            result['routed_to_worker'] = worker.worker_id
            result['routed_adapter'] = adapter_id

            return result

        except Exception as e:
            return {"error": f"Inference error: {e}", "success": False}
        finally:
            worker.active_requests -= 1
            worker.last_request_at = time.time()  # Set at END for accurate idle tracking
            # Inference completed — worker is alive, restore to READY immediately
            # (health checks may have marked it UNHEALTHY while Flask was blocked)
            if worker.status == WorkerStatus.UNHEALTHY:
                worker.status = WorkerStatus.READY
            self._worker_available.set()  # Wake dispatcher — capacity freed

    # --------------------------------------------------------------------------
    # Status / Stats
    # --------------------------------------------------------------------------

    def get_workers(self) -> List[dict]:
        """Get info about all workers."""
        return [
            {
                'worker_id': w.worker_id,
                'http_port': w.http_port,
                'adapter_id': w.adapter_id,
                'status': w.status.value,
                'active_requests': w.active_requests,
                'requests_handled': w.requests_handled,
                'avg_inference_ms': round(w.avg_inference_ms(), 2),
                'idle_time': round(w.idle_time, 1) if w.last_request_at > 0 else -1,
            }
            for w in self.workers.values()
        ]

    def get_stats(self) -> dict:
        """Get routing statistics."""
        total = self.stats['requests_total']
        self.stats['queue_depth'] = len(self._pending_requests)
        return {
            'workers_active': len(self.workers),
            'workers_max': self.max_workers,
            **self.stats,
            'avg_inference_ms': round(
                self.stats['total_inference_ms'] / max(1, total), 2
            ),
        }

    def get_node_metrics(self) -> dict:
        """Aggregate per-worker metrics into node-level summary."""
        workers = list(self.workers.values())
        num_active = sum(1 for w in workers if w.active_requests > 0)
        num_swapping = sum(1 for w in workers if w.status == WorkerStatus.SWAPPING)

        return {
            'total_active_requests': num_active,
            'num_workers': len(workers),
            'num_swapping': num_swapping,
            'swap_count': self.stats.get('requests_swapped', 0),
            'total_swap_ms': self.stats.get('total_swap_ms', 0.0),
        }

    # --------------------------------------------------------------------------
    # Idempotency Support
    # --------------------------------------------------------------------------

    async def check_idempotency(self, key: str) -> Optional[dict]:
        """Check if result for idempotency key is cached."""
        async with self._idempotency_lock:
            if key in self._idempotency_cache:
                ts, result = self._idempotency_cache[key]
                if time.time() - ts < self._idempotency_ttl:
                    return result
        return None

    async def cache_idempotency(self, key: str, result: dict):
        """Cache result for idempotency key."""
        async with self._idempotency_lock:
            self._idempotency_cache[key] = (time.time(), result)

    async def _idempotency_cleanup_loop(self):
        """Periodically clean expired idempotency entries."""
        while self._running:
            await asyncio.sleep(self.scheduler_config.idempotency_cleanup_interval)
            now = time.time()
            async with self._idempotency_lock:
                expired = [k for k, (ts, _) in self._idempotency_cache.items()
                           if now - ts > self._idempotency_ttl]
                for k in expired:
                    del self._idempotency_cache[k]
                if expired:
                    log(f"Cleaned {len(expired)} expired idempotency entries")


# HELPER: messages -> prompt

def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Convert chat messages to a prompt string."""
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            parts.append(f"Assistant: {msg.content}")
    parts.append("Assistant:")
    return "\n".join(parts)


# FASTAPI APPLICATION

# Global load balancer instance
lb: Optional[AsyncLoadBalancer] = None

# Number of workers to spawn on startup (set from CLI args)
_spawn_initial: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage load balancer lifecycle."""
    global lb
    await lb.start()

    # Spawn initial workers if requested (concurrently with timeout)
    if _spawn_initial > 0:
        log(f"Spawning {_spawn_initial} initial workers concurrently...")
        spawn_tasks = [lb.spawn_worker(lb.default_adapter) for _ in range(_spawn_initial)]
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*spawn_tasks, return_exceptions=True),
                timeout=lb.scheduler_config.initial_spawn_batch_timeout,
            )
            spawned = sum(1 for r in results if r is not None and not isinstance(r, Exception))
            log(f"Spawned {spawned}/{_spawn_initial} initial workers")
        except asyncio.TimeoutError:
            log("Warning: Initial worker spawn timed out, continuing with available workers...")

    yield

    await lb.stop()


app = FastAPI(
    title="Serverless Controller",
    description="Async load balancer: one request per worker, swap-first scheduling via LoRant",
    lifespan=lifespan
)


# Core endpoints (from load_balancer.py)

@app.post("/inference")
async def inference(req: InferenceRequest):
    """Route inference request to optimal worker."""
    result = await lb.route_request(req)
    if not result.get('success', False):
        raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
    return result


@app.get("/workers")
async def list_workers():
    """List all managed workers."""
    return {
        "workers": lb.get_workers(),
        "queue_depth": lb.stats.get('queue_depth', 0),
    }


@app.get("/node_metrics")
async def node_metrics():
    """Aggregate node-level metrics for heartbeat reporting."""
    return lb.get_node_metrics()


@app.post("/workers/{worker_id}/swap")
async def swap_worker(worker_id: int, req: SwapRequest):
    """Manually trigger adapter swap on a worker."""
    worker = lb.workers.get(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")

    success = await lb._swap_worker_adapter(worker, req.adapter_id)
    return {"success": success}


@app.post("/workers/spawn")
async def spawn_workers(req: SpawnRequest):
    """Spawn new workers."""
    adapter_id = req.adapter_id or lb.default_adapter
    spawned = []

    for _ in range(req.count):
        worker = await lb.spawn_worker(adapter_id)
        if worker:
            spawned.append({
                "worker_id": worker.worker_id,
                "http_port": worker.http_port,
            })
        else:
            break

    return {"success": len(spawned) > 0, "spawned": spawned}


@app.post("/prewarm")
async def prewarm_workers(req: PrewarmRequest):
    """Pre-warm workers with specified adapter counts (all launched concurrently)."""
    total_requested = sum(req.adapters.values())
    details = {}

    # Build device assignment list from req.devices (e.g. {"cuda:0": 12, "cuda:1": 18})
    # Workers are assigned devices in order: first 12 get cuda:0, next 18 get cuda:1, etc.
    device_list = []
    if req.devices:
        for dev, cnt in req.devices.items():
            device_list.extend([dev] * cnt)
    device_idx = 0

    # Every device prewarm may touch needs a ready aggregator *before* workers
    # launch onto it: Phase 1 calls _launch_worker_process() directly, bypassing
    # _ensure_aggregator(). Otherwise the first GPU's aggregator is never marked
    # ready, and once that GPU fills, the donor search for the next one fails
    # permanently with "no donor aggregator available".
    if req.devices:
        needed_devices = list(dict.fromkeys(req.devices.keys()))
    else:
        needed_devices = []
        remaining = total_requested
        for dev in lb.worker_devices:
            if remaining <= 0:
                break
            cap = (lb.gpu_worker_caps or {}).get(dev, lb.max_workers)
            needed_devices.append(dev)
            remaining -= cap
    for dev in needed_devices:
        await lb._ensure_aggregator(dev)

    # Phase 1: Launch all processes (fast, holds lock briefly per worker)
    pending = []  # list of (adapter_id, worker)
    for adapter_id, count in req.adapters.items():
        for _ in range(count):
            async with lb._lock:
                worker_id = lb._get_next_worker_id()
                if worker_id < 0:
                    break
                http_port = lb.base_port + worker_id * 10
                if device_list and device_idx < len(device_list):
                    device = device_list[device_idx]
                    device_idx += 1
                else:
                    device = lb._pick_device(worker_id)
                    if device is None:
                        break
                proc, method = lb._launch_worker_process(worker_id, http_port, device, adapter_id)
                log(f"Prewarm: launching worker {worker_id} on port {http_port} device {device} adapter {adapter_id} via {method}")

                worker = WorkerInfo(
                    worker_id=worker_id, http_port=http_port,
                    adapter_id=adapter_id, status=WorkerStatus.STARTING, process=proc,
                    device=device,
                )
                lb.workers[worker_id] = worker
                lb._gpu_worker_count[device] = lb._gpu_worker_count.get(device, 0) + 1
            pending.append((adapter_id, worker))

    log(f"Prewarm: {len(pending)} processes launched, waiting for ready...")

    # Phase 2: Wait for all workers concurrently
    async def wait_and_register(adapter_id, worker):
        ready = await lb._wait_for_worker_ready(worker)
        if ready:
            await lb.registry.register(worker.worker_id, adapter_id)
        return (adapter_id, worker.worker_id, ready)

    results = await asyncio.gather(*[wait_and_register(a, w) for a, w in pending])

    spawned = 0
    failed = 0
    for adapter_id, worker_id, ready in results:
        if adapter_id not in details:
            details[adapter_id] = {"requested": 0, "spawned": 0, "worker_ids": []}
        details[adapter_id]["requested"] += 1
        if ready:
            details[adapter_id]["spawned"] += 1
            details[adapter_id]["worker_ids"].append(worker_id)
            spawned += 1
        else:
            failed += 1

    log(f"Prewarm complete: {spawned}/{total_requested} spawned, {failed} failed")
    return {"success": failed == 0, "total_requested": total_requested, "spawned": spawned, "failed": failed, "details": details}


@app.post("/workers/{worker_id}/stop")
async def stop_worker(worker_id: int):
    """Stop a specific worker."""
    worker = lb.workers.get(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")

    await lb._stop_worker(worker)
    return {"success": True}


@app.post("/workers/stop-all")
async def stop_all_workers():
    """Stop all workers (for between-trace cleanup)."""
    worker_ids = list(lb.workers.keys())
    stopped = 0
    for wid in worker_ids:
        worker = lb.workers.get(wid)
        if worker:
            await lb._stop_worker(worker)
            stopped += 1
    log(f"Stopped {stopped}/{len(worker_ids)} workers")
    return {"stopped": stopped, "total": len(worker_ids)}


@app.get("/stats")
async def get_stats():
    """Get routing statistics."""
    return lb.get_stats()


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "workers": len(lb.workers),
        "used_capacity": sum(w.active_requests for w in lb.workers.values()),
    }


# OpenAI-compatible endpoints

@app.post("/v1/completions")
async def openai_completions(req: CompletionRequest, request: Request):
    """OpenAI-compatible completions endpoint with idempotency support."""
    idem_key = request.headers.get("X-Idempotency-Key")

    # Check idempotency cache
    if idem_key:
        cached = await lb.check_idempotency(idem_key)
        if cached:
            return {**cached, "_cached": True}

    # Build internal inference request
    inference_req = InferenceRequest(
        prompt=req.prompt,
        adapter_id=req.model,
        max_tokens=req.max_tokens,
        do_sample=req.do_sample,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
    )

    result = await lb.route_request(inference_req, arrival_wall_override=req.arrival_time)

    if not result.get('success', False):
        raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))

    response = {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "text": result.get("generated_text", result.get("text", "")),
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "total_tokens": result.get("prompt_tokens", 0) + result.get("completion_tokens", 0),
        },
        "_serverless_metrics": {
            # Quantised to 10ms / 1-decimal to reduce timing side-channel
            # precision for external clients. Internal benchmark uses the
            # precise values from the /inference endpoint directly.
            "e2e_ms": round(result.get("e2e_ms", 0) / 10) * 10,
            "queue_wait_ms": round(result.get("queue_wait_ms", 0) / 10) * 10,
            "ttft_ms": round(result.get("ttft_ms", 0) / 10) * 10,
            "decode_time_ms": round(result.get("decode_time_ms", 0) / 10) * 10,
            "decode_throughput": round(result.get("decode_throughput", 0), 1),
            "gen_throughput": round(result.get("gen_throughput", 0), 1),
            "total_throughput": round(result.get("total_throughput", 0), 1),
        },
    }

    # Cache result for idempotency
    if idem_key:
        await lb.cache_idempotency(idem_key, response)

    return response


@app.post("/v1/chat/completions")
async def openai_chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    prompt = messages_to_prompt(req.messages)

    inference_req = InferenceRequest(
        prompt=prompt,
        adapter_id=req.model,
        max_tokens=req.max_tokens,
        do_sample=req.do_sample,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
    )

    result = await lb.route_request(inference_req, arrival_wall_override=req.arrival_time)

    if not result.get('success', False):
        raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": result.get("generated_text", result.get("text", "")),
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "total_tokens": result.get("total_tokens", 0),
        },
        "_serverless_metrics": {
            # Same security policy as /v1/completions:
            # worker_id, adapter_id, routing_ms omitted (topology/timing leakage)
            # Timing quantised to 10ms buckets; throughput to 1 decimal
            "e2e_ms": round(result.get("e2e_ms", 0) / 10) * 10,
            "queue_wait_ms": round(result.get("queue_wait_ms", 0) / 10) * 10,
            "ttft_ms": round(result.get("ttft_ms", 0) / 10) * 10,
            "decode_time_ms": round(result.get("decode_time_ms", 0) / 10) * 10,
            "decode_throughput": round(result.get("decode_throughput", 0), 1),
            "gen_throughput": round(result.get("gen_throughput", 0), 1),
            "total_throughput": round(result.get("total_throughput", 0), 1),
        },
    }


@app.get("/v1/models")
async def openai_list_models():
    """OpenAI-compatible model listing (lists loaded adapters)."""
    adapters = await lb.registry.get_all_adapters()
    return {
        "object": "list",
        "data": [
            {
                "id": adapter_id,
                "object": "model",
                "owned_by": "user",
            }
            for adapter_id in adapters
        ],
    }


@app.get("/metrics")
async def metrics():
    """Alias for /stats (compatible with api_gateway metrics endpoint)."""
    return lb.get_stats()


# GPU AUTO-DETECTION

def _detect_gpus() -> list:
    """Auto-detect GPU info via nvidia-smi. Returns list of GPU dicts."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,index",
             "--format=csv,nounits,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []

        gpus = []
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                gpus.append({
                    'name': parts[0],
                    'memory_total_mb': int(parts[1]),
                    'memory_free_mb': int(parts[2]),
                    'index': int(parts[3]),
                })
        return gpus
    except Exception:
        return []


# MAIN

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Serverless Controller for Multi-Adapter LoRA Inference")
    parser.add_argument("--port", type=int, default=LB_PORT,
                        help=f"HTTP port for controller (default: {LB_PORT})")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Maximum number of workers to manage (default: 8)")
    parser.add_argument("--base-port", type=int, default=WORKER_BASE_PORT,
                        help=f"Base port for workers (default: {WORKER_BASE_PORT})")
    parser.add_argument("--aggregator-host", type=str, default="localhost",
                        help="Aggregator hostname (default: localhost)")
    parser.add_argument("--aggregator-port", type=int, default=AGGREGATOR_PORT,
                        help=f"Aggregator port (default: {AGGREGATOR_PORT})")
    parser.add_argument("--default-adapter", type=str, default=None,
                        help="Default LoRA adapter for new workers")
    parser.add_argument("--spawn-initial", type=int, default=0,
                        help="Number of workers to spawn on startup (default: 0)")
    parser.add_argument("--min-workers", type=int, default=0,
                        help="Minimum workers to keep alive (scale-to-zero if 0)")
    parser.add_argument("--scale-down-delay", type=float, default=20.0,
                        help="Seconds idle before scaling down a worker (default: 20)")
    parser.add_argument("--spawn-timeout", type=float, default=180.0,
                        help="Max seconds to wait for a spawned worker to become ready (default: 180)")
    parser.add_argument("--worker-devices", type=str, default=None,
                        help="Comma-separated GPU devices for workers, round-robin (e.g. cuda:0,cuda:1,cuda:2)")
    parser.add_argument("--scheduler", type=str, default="lorant",
                        help="Local scheduling algorithm (default: lorant)")
    parser.add_argument("--no-fork", action="store_true",
                        help="Disable fork-based worker spawning (use subprocess instead)")
    parser.add_argument("--aggregator-port-map", type=str, default=None,
                        help="Per-GPU aggregator port map, e.g. '0:50056,1:50057'. Workers on each GPU connect to the specified aggregator port.")
    parser.add_argument("--aggregator-health-port-map", type=str, default=None,
                        help="Per-GPU aggregator health port map, e.g. '0:8000,1:8001'. Used for demand-driven aggregator launch.")
    parser.add_argument("--max-workers-per-gpu", type=str, default=None,
                        help="Override per-GPU worker cap (default: from config.py MAX_WORKERS_PER_GPU)")
    args = parser.parse_args()

    # Store spawn_initial for lifespan
    _spawn_initial = args.spawn_initial

    # Create load balancer
    # Auto-detect GPUs and compute per-GPU capacity
    detected_gpus = _detect_gpus()

    # Parse worker devices and compute per-GPU caps
    # --max-workers-per-gpu accepts either a uniform int ("10") or
    # per-device map ("cuda:0:10,cuda:1:20") for asymmetric configs
    per_gpu_caps_map = {}
    per_gpu_cap = MAX_WORKERS_PER_GPU
    if args.max_workers_per_gpu:
        if "cuda:" in args.max_workers_per_gpu:
            for entry in args.max_workers_per_gpu.split(","):
                parts = entry.strip().split(":")
                dev = f"{parts[0]}:{parts[1]}"
                cap = int(parts[2])
                per_gpu_caps_map[dev] = cap
        else:
            per_gpu_cap = int(args.max_workers_per_gpu)
    gpu_worker_caps = None
    if args.worker_devices:
        worker_devices = [d.strip() for d in args.worker_devices.split(",")]
        if per_gpu_caps_map:
            gpu_worker_caps = {dev: per_gpu_caps_map.get(dev, per_gpu_cap) for dev in worker_devices}
        else:
            gpu_worker_caps = {dev: per_gpu_cap for dev in worker_devices}
        log(f"Using --worker-devices={worker_devices}, per-GPU caps: {gpu_worker_caps}")
    elif WORKER_GPUS:
        worker_devices = [f"cuda:{g}" for g in WORKER_GPUS]
        gpu_worker_caps = {dev: per_gpu_cap for dev in worker_devices}
        log(f"Using WORKER_GPUS={WORKER_GPUS}, per-GPU caps: {gpu_worker_caps}")
    elif detected_gpus:
        worker_devices = [f"cuda:{g['index']}" for g in detected_gpus]
        gpu_worker_caps = {dev: per_gpu_cap for dev in worker_devices}
        log(f"Auto-detected {len(detected_gpus)} GPUs, per-GPU caps: {gpu_worker_caps}")
    else:
        worker_devices = ["cuda:0"]

    max_workers = args.max_workers or (
        sum(gpu_worker_caps.values()) if gpu_worker_caps else 8
    )

    # Create scheduler config and scheduler
    scheduler_config = LocalSchedulerConfig()
    scheduler_config.spawn_timeout = args.spawn_timeout
    if args.scheduler == "lorant":
        scheduler = LoRantScheduler(cfg=scheduler_config)
    elif args.scheduler == "noswap":
        scheduler = LoRantNoSwapScheduler(cfg=scheduler_config)
    elif args.scheduler == "random":
        scheduler = RandomScheduler(cfg=scheduler_config)
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}. Available: lorant, noswap, random")

    # Parse per-GPU aggregator port map
    aggregator_port_map = None
    if args.aggregator_port_map:
        aggregator_port_map = {}
        for entry in args.aggregator_port_map.split(","):
            gpu_idx, port = entry.strip().split(":")
            aggregator_port_map[f"cuda:{gpu_idx}"] = int(port)
        log(f"Using per-GPU aggregator ports: {aggregator_port_map}")

    # Parse per-GPU aggregator health port map
    aggregator_health_port_map = None
    if args.aggregator_health_port_map:
        aggregator_health_port_map = {}
        for entry in args.aggregator_health_port_map.split(","):
            gpu_idx, port = entry.strip().split(":")
            aggregator_health_port_map[f"cuda:{gpu_idx}"] = int(port)
        log(f"Using per-GPU aggregator health ports: {aggregator_health_port_map}")

    lb = AsyncLoadBalancer(
        max_workers=max_workers,
        base_port=args.base_port,
        aggregator_host=args.aggregator_host,
        aggregator_port=args.aggregator_port,
        default_adapter=args.default_adapter,
        min_workers=args.min_workers,
        scale_down_delay=args.scale_down_delay,
        worker_devices=worker_devices,
        scheduler=scheduler,
        scheduler_config=scheduler_config,
        use_fork=not args.no_fork,
        gpu_worker_caps=gpu_worker_caps,
        aggregator_port_map=aggregator_port_map,
        aggregator_health_port_map=aggregator_health_port_map,
    )

    log("=" * 60)
    log("Serverless Controller (FastAPI + aiohttp)")
    log("=" * 60)
    log(f"Port: {args.port}")
    log(f"Max workers: {max_workers}")
    log(f"Min workers: {args.min_workers}")
    log(f"Scale-down delay: {args.scale_down_delay}s")
    log(f"Worker base port: {args.base_port}")
    log(f"Aggregator: {args.aggregator_host}:{args.aggregator_port}")
    log(f"Default adapter: {args.default_adapter}")
    log(f"Worker devices: {worker_devices}")
    log(f"GPU worker caps: {gpu_worker_caps or 'none (round-robin)'}")
    log(f"Scheduler: {args.scheduler}")
    log(f"Fork spawning: {'disabled' if args.no_fork else 'enabled'}")
    log("=" * 60)

    # Run with uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="info",
    )
