#  Node Agent - Runs on each node, manages aggregator and reports to global    #

import asyncio
import subprocess
import sys
import os
import time
import signal
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict
from pathlib import Path
import aiohttp
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn

try:
    from .resilience import RetryExecutor, RetryConfig
except ImportError:
    from resilience import RetryExecutor, RetryConfig

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import (
    WORKER_DEVICE, AGGREGATOR_DEVICE, WORKER_BASE_PORT,
    DEFAULT_PORT, AGGREGATOR_HEALTH_PORT, CONTROLLER_PORT,
    GLOBAL_CONTROLLER_PORT, NODE_AGENT_PORT, MAX_WORKERS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration

@dataclass
class NodeAgentConfig:
    # Identity
    node_id: str
    host: str = "0.0.0.0"
    port: int = NODE_AGENT_PORT

    # Global controller
    global_controller_url: str = f"http://localhost:{GLOBAL_CONTROLLER_PORT}"

    # Override registration address (for SSH tunnel setups where the global
    # controller reaches this node via a forwarded port on localhost)
    register_host: str = ""   # If set, register with this host instead of auto-detected IP
    register_port: int = 0    # If set, register with this port instead of self.port

    # GPU settings
    aggregator_device: str = AGGREGATOR_DEVICE
    worker_device: str = WORKER_DEVICE
    gpu_type: str = "unknown"
    gpu_memory_gb: float = 0

    # Paths
    project_root: str = ""
    aggregator_script: str = "src/aggregator.py"
    controller_module: str = "controller.controller"

    # Worker settings
    max_workers: int = MAX_WORKERS
    worker_base_port: int = WORKER_BASE_PORT
    controller_port: int = CONTROLLER_PORT
    aggregator_health_port: int = AGGREGATOR_HEALTH_PORT
    aggregator_tcp_port: int = DEFAULT_PORT

    # Scheduling
    scheduler: str = "default"
    scale_down_delay: float = 5.0  # Seconds idle before scale-down (0 = never)

    # Worker spawning
    use_fork: bool = True

    # Single aggregator mode: all workers (both GPUs) share one aggregator on GPU 0.
    # Use on nodes where GPU memory is too small for a second aggregator.
    single_aggregator: bool = False

    # Timeouts
    aggregator_startup_timeout: float = 600.0
    heartbeat_interval: float = 1.0
    health_check_timeout: float = 5.0


# Request/Response Models

class LaunchAggregatorRequest(BaseModel):
    force: bool = False              # Force restart if already running
    base_model: Optional[str] = None # Base model name (e.g., "llama_2_7b")
    model_id: Optional[str] = None   # HuggingFace model ID (e.g., "meta-llama/Llama-2-7b-hf")


class LaunchAggregatorResponse(BaseModel):
    success: bool
    aggregator_pid: Optional[int] = None
    base_model: Optional[str] = None
    message: str = ""


class StopAggregatorRequest(BaseModel):
    graceful: bool = True
    timeout: float = 30.0


class NodeStatusResponse(BaseModel):
    node_id: str
    status: str
    aggregator_running: bool
    aggregator_pid: Optional[int]
    base_model: Optional[str] = None
    controller_running: bool
    active_workers: int
    total_workers: int = 0       # Actually spawned and ready
    free_workers: int = 0        # Ready but not processing a request
    queue_depth: int = 0         # Requests waiting in local controller queue
    loaded_adapters: List[str]
    uptime: float
    node_metrics: Optional[Dict] = None  # Aggregate load metrics


class ForwardRequest(BaseModel):
    endpoint: str  # e.g., "/v1/completions"
    payload: dict


# Node Agent

def _ensure_subprocess_libs():
    """Ensure LD_LIBRARY_PATH includes torch/nvidia libs for subprocesses."""
    current = os.environ.get("LD_LIBRARY_PATH", "")
    paths = []
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if torch_lib not in current:
            paths.append(torch_lib)
    except ImportError:
        pass
    for pkg in ["nvidia.cuda_runtime", "nvidia.cublas", "nvidia.cudnn"]:
        try:
            mod = __import__(pkg, fromlist=["lib"])
            lib_dir = os.path.join(os.path.dirname(mod.__file__), "lib")
            if lib_dir not in current:
                paths.append(lib_dir)
        except ImportError:
            pass
    if paths:
        new_ld = ":".join(paths) + (":" + current if current else "")
        os.environ["LD_LIBRARY_PATH"] = new_ld
        return new_ld
    return current


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


class NodeAgent:
    """Node agent manages the aggregator and local controller on a single node."""

    def __init__(self, config: NodeAgentConfig):
        self.config = config
        self._ensure_ld_library_path()
        self.app = FastAPI(title=f"Node Agent - {config.node_id}")

        # Process handles
        self._aggregator_process: Optional[subprocess.Popen] = None
        self._controller_process: Optional[subprocess.Popen] = None

        # Base model state
        self._current_base_model: Optional[str] = None
        self._current_model_id: Optional[str] = None

        # Detected GPU info (for registration and monitoring)
        self._detected_gpus: list = []

        # Auto-detect GPUs (must be after _detected_gpus init)
        self._auto_detect_gpu()

        # State
        self._started_at = time.time()
        self._running = False
        self._registered = False

        # Background tasks
        self._tasks: List[asyncio.Task] = []
        self._session: Optional[aiohttp.ClientSession] = None

        self._setup_routes()

    def _setup_routes(self):
        app = self.app

        @app.on_event("startup")
        async def startup():
            await self.start()

        @app.on_event("shutdown")
        async def shutdown():
            await self.stop()

        # Status
        @app.get("/status", response_model=NodeStatusResponse)
        async def get_status():
            return await self._get_status()

        @app.get("/health")
        async def health():
            return {"status": "ok", "node_id": self.config.node_id}

        @app.get("/gpu")
        async def gpu_info():
            """Return detected GPU information."""
            return {
                "node_id": self.config.node_id,
                "gpu_type": self.config.gpu_type,
                "gpu_memory_gb": round(self.config.gpu_memory_gb, 1),
                "gpus": self._detected_gpus,
            }

        # Aggregator control
        @app.post("/aggregator/launch", response_model=LaunchAggregatorResponse)
        async def launch_aggregator(request: LaunchAggregatorRequest):
            return await self._launch_aggregator(
                force=request.force,
                base_model=request.base_model,
                model_id=request.model_id,
            )

        @app.post("/aggregator/stop")
        async def stop_aggregator(request: StopAggregatorRequest):
            return await self._stop_aggregator(request.graceful, request.timeout)

        @app.get("/aggregator/status")
        async def aggregator_status():
            return {
                "running": self._is_aggregator_running(),
                "pid": self._aggregator_process.pid if self._aggregator_process else None,
            }

        # Forward requests to local controller
        @app.post("/forward")
        async def forward_request(request: ForwardRequest):
            return await self._forward_to_controller(request.endpoint, request.payload)

        # Pre-warm workers with specific adapters
        @app.post("/prewarm")
        async def prewarm(request: Request):
            body = await request.json()
            return await self._forward_to_controller("/prewarm", body)

        # Worker info (proxied from local controller)
        @app.get("/workers")
        async def get_workers():
            return await self._get_workers()

    # =========================================================================
    # Environment Setup
    # =========================================================================

    def _ensure_ld_library_path(self):
        """Set LD_LIBRARY_PATH so subprocesses can load torch/CUDA .so files."""
        result = _ensure_subprocess_libs()
        if result:
            logger.info(f"LD_LIBRARY_PATH configured for subprocesses")

    # =========================================================================
    # GPU Auto-Detection
    # =========================================================================

    def _auto_detect_gpu(self):
        """Auto-detect GPU type and memory if not provided via CLI."""
        self._detected_gpus = _detect_gpus()
        if not self._detected_gpus:
            logger.warning("No GPUs detected via nvidia-smi, using config defaults")
            return

        logger.info(f"Detected {len(self._detected_gpus)} GPU(s):")
        for g in self._detected_gpus:
            logger.info(f"  GPU {g['index']}: {g['name']} "
                       f"({g['memory_total_mb']} MB total, {g['memory_free_mb']} MB free)")

        # Auto-fill config defaults from first GPU
        gpu = self._detected_gpus[0]
        if self.config.gpu_type == "unknown":
            self.config.gpu_type = gpu['name']
            logger.info(f"Auto-detected gpu_type: {self.config.gpu_type}")
        if self.config.gpu_memory_gb == 0:
            self.config.gpu_memory_gb = gpu['memory_total_mb'] / 1024.0
            logger.info(f"Auto-detected gpu_memory_gb: {self.config.gpu_memory_gb:.1f}")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self):
        """Start the node agent"""
        logger.info(f"Starting node agent {self.config.node_id}")
        self._running = True
        self._session = aiohttp.ClientSession()

        await self._register_with_global()

        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._process_monitor_loop()),
        ]

        logger.info(f"Node agent {self.config.node_id} started")

    async def stop(self):
        """Stop the node agent and cleanup"""
        if not self._running:
            return  # Already stopping
        logger.info(f"Stopping node agent {self.config.node_id}")
        self._running = False

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Stop processes
        await self._stop_aggregator(graceful=True, timeout=30.0)

        await self._unregister_from_global()

        if self._session:
            await self._session.close()

        logger.info(f"Node agent {self.config.node_id} stopped")

    # =========================================================================
    # Global Controller Communication
    # =========================================================================

    async def _register_with_global(self):
        """Register this node with the global controller"""
        try:
            reg_host = self.config.register_host or (
                self.config.host if self.config.host != "0.0.0.0" else self._get_external_ip()
            )
            reg_port = self.config.register_port or self.config.port
            payload = {
                "node_id": self.config.node_id,
                "host": reg_host,
                "port": reg_port,
                "gpu_type": self.config.gpu_type,
                "gpu_memory_gb": self.config.gpu_memory_gb,
                "num_gpus": len(self._detected_gpus),
                "max_workers": self.config.max_workers,
            }

            async with self._session.post(
                f"{self.config.global_controller_url}/nodes/register",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    try:
                        await response.json()  # Validate response is JSON
                    except Exception:
                        pass  # Response body not critical for registration
                    self._registered = True
                    logger.info(f"Registered with global controller")
                else:
                    logger.warning(f"Failed to register: {response.status}")

        except Exception as e:
            logger.warning(f"Could not register with global controller: {e}")

    async def _unregister_from_global(self):
        """Unregister from global controller"""
        if not self._registered:
            return

        try:
            async with self._session.post(
                f"{self.config.global_controller_url}/nodes/unregister",
                json={"node_id": self.config.node_id},
                timeout=aiohttp.ClientTimeout(total=5)
            ):
                pass
        except Exception as e:
            logger.warning(f"Could not unregister: {e}")

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to global controller"""
        while self._running:
            try:
                await self._send_heartbeat()
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")

            await asyncio.sleep(self.config.heartbeat_interval)

    async def _process_monitor_loop(self):
        """Monitor and restart crashed processes with exponential backoff."""
        agg_failures = 0
        ctrl_failures = 0
        max_retries = 5
        base_backoff = 5.0  # seconds

        while self._running:
            # Check aggregator
            if self._aggregator_process and self._aggregator_process.poll() is not None:
                exit_code = self._aggregator_process.returncode
                agg_failures += 1
                self._aggregator_process = None

                if agg_failures > max_retries:
                    logger.error(f"Aggregator crashed {agg_failures} times, giving up")
                else:
                    backoff = min(base_backoff * (2 ** (agg_failures - 1)), 120.0)
                    logger.warning(f"Aggregator crashed (exit={exit_code}), "
                                  f"retry {agg_failures}/{max_retries} in {backoff:.0f}s")
                    await asyncio.sleep(backoff)
                    try:
                        await self._launch_aggregator(
                            base_model=self._current_base_model,
                            model_id=self._current_model_id
                        )
                        agg_failures = 0  # Reset on success
                    except Exception as e:
                        logger.error(f"Failed to restart aggregator: {e}")

            # Check controller
            if self._controller_process and self._controller_process.poll() is not None:
                exit_code = self._controller_process.returncode
                ctrl_failures += 1
                self._controller_process = None

                if ctrl_failures > max_retries:
                    logger.error(f"Controller crashed {ctrl_failures} times, giving up")
                else:
                    backoff = min(base_backoff * (2 ** (ctrl_failures - 1)), 60.0)
                    logger.warning(f"Controller crashed (exit={exit_code}), "
                                  f"retry {ctrl_failures}/{max_retries} in {backoff:.0f}s")
                    await asyncio.sleep(backoff)
                    try:
                        await self._launch_controller()
                        ctrl_failures = 0  # Reset on success
                    except Exception as e:
                        logger.error(f"Failed to restart controller: {e}")

            await asyncio.sleep(5.0)

    async def _send_heartbeat(self):
        """Send heartbeat with current state"""
        status = await self._get_status()

        payload = {
            "node_id": self.config.node_id,
            "status": status.status,
            "aggregator_pid": status.aggregator_pid,
            "active_workers": status.active_workers,
            "total_workers": status.total_workers,
            "free_workers": status.free_workers,
            "queue_depth": status.queue_depth,
            "max_workers": self.config.max_workers,
            "num_gpus": len(self._detected_gpus),
            "gpu_type": self.config.gpu_type,
            "loaded_adapters": status.loaded_adapters,
            "base_model": self._current_base_model,
            "node_metrics": status.node_metrics,
        }

        try:
            async with self._session.post(
                f"{self.config.global_controller_url}/nodes/heartbeat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status != 200:
                    logger.warning(f"Heartbeat rejected: {response.status}")
        except Exception as e:
            # Global controller might be down, continue running
            logger.debug(f"Heartbeat failed: {e}")

    def _get_external_ip(self) -> str:
        """Get external IP address of this node"""
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Try to get external IP by connecting to a public DNS
            # This doesn't send any packets, just determines the route
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except Exception:
            # Fallback: try to get hostname IP
            try:
                return socket.gethostbyname(socket.gethostname())
            except Exception:
                return "localhost"
        finally:
            s.close()

    # =========================================================================
    # Aggregator Management
    # =========================================================================

    async def _launch_aggregator(
        self,
        force: bool = False,
        base_model: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> LaunchAggregatorResponse:
        """Launch the aggregator process with specified base model"""
        if self._is_aggregator_running():
            if not force:
                return LaunchAggregatorResponse(
                    success=True,
                    aggregator_pid=self._aggregator_process.pid,
                    base_model=self._current_base_model,
                    message="Aggregator already running"
                )
            else:
                await self._stop_aggregator(graceful=True, timeout=10.0)

        # Store the base model info
        self._current_base_model = base_model
        self._current_model_id = model_id

        logger.info(f"Launching aggregator on {self.config.aggregator_device} "
                   f"with base_model={base_model}, model_id={model_id}")

        try:
            # Build command
            aggregator_path = Path(self.config.project_root) / self.config.aggregator_script

            # CUDA_VISIBLE_DEVICES remaps the GPU to cuda:0 inside the subprocess
            cmd = [
                sys.executable, str(aggregator_path),
                "--device", "cuda:0",
                "--port", str(self.config.aggregator_tcp_port),
                "--health-port", str(self.config.aggregator_health_port),
            ]

            # Add model_id if provided (aggregator will load this model)
            if model_id:
                cmd.extend(["--model", model_id])

            # Set environment
            env = os.environ.copy()
            gpu_index = self.config.aggregator_device.replace("cuda:", "")
            env["CUDA_VISIBLE_DEVICES"] = gpu_index
            env["CUDA_MPS_PIPE_DIRECTORY"] = f"/tmp/mps_{gpu_index}"
            env["HF_HUB_OFFLINE"] = "1"  # Use cached model, skip online validation

            # Start process
            log_dir = Path(self.config.project_root) / "logs"
            log_dir.mkdir(exist_ok=True)
            agg_log = open(log_dir / f"aggregator-{self.config.node_id}.log", "a")
            self._aggregator_process = subprocess.Popen(
                cmd,
                stdout=agg_log,
                stderr=agg_log,
                env=env,
                cwd=self.config.project_root,
            )
            agg_log.close()  # Subprocess inherited the fd, safe to close

            ready = await self._wait_for_aggregator_ready()

            if ready:
                # Start the local controller
                await self._launch_controller()

                return LaunchAggregatorResponse(
                    success=True,
                    aggregator_pid=self._aggregator_process.pid,
                    base_model=base_model,
                    message=f"Aggregator launched with {base_model or 'default'} model"
                )
            else:
                # Cleanup
                await self._kill_process(self._aggregator_process)
                self._aggregator_process = None
                self._current_base_model = None
                self._current_model_id = None
                return LaunchAggregatorResponse(
                    success=False,
                    message="Aggregator startup timeout"
                )

        except Exception as e:
            logger.error(f"Failed to launch aggregator: {e}")
            return LaunchAggregatorResponse(
                success=False,
                message=str(e)
            )

    async def _wait_for_aggregator_ready(self) -> bool:
        """Wait for aggregator to be ready"""
        start_time = time.time()

        while time.time() - start_time < self.config.aggregator_startup_timeout:
            if not self._is_aggregator_running():
                return False

            # Try to connect to aggregator health endpoint
            try:
                async with self._session.get(
                    f"http://localhost:{self.config.aggregator_health_port}/health",
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('status') == 'ready':
                            logger.info(f"Aggregator is ready ({data.get('num_slots', 128)} slots)")
                            return True
            except Exception:
                pass

            await asyncio.sleep(1.0)

        logger.error("Aggregator startup timeout")
        return False

    def _per_gpu_cap(self, num_gpus: int) -> str:
        """Compute per-GPU worker cap string for the controller."""
        if not self.config.single_aggregator or num_gpus <= 1:
            return str(self.config.max_workers // max(num_gpus, 1))
        # Asymmetric: GPU 0 gets 10 (limited by aggregator memory), rest to GPU 1
        gpu0_cap = min(10, self.config.max_workers)
        gpu1_cap = self.config.max_workers - gpu0_cap
        return f"cuda:0:{gpu0_cap},cuda:1:{gpu1_cap}"

    async def _launch_controller(self):
        """Launch the local serverless controller"""
        if self._controller_process and self._controller_process.poll() is None:
            return  # Already running

        logger.info("Launching local controller")

        num_gpus = len(self._detected_gpus) if self._detected_gpus else 1

        # Build per-GPU port maps for multi-GPU demand-driven aggregator launch
        # GPU 0 uses the configured ports; GPU 1+ get sequential ports
        # In single_aggregator mode, all GPUs share GPU 0's aggregator port
        if self.config.single_aggregator:
            agg_port_map = ",".join(
                f"{i}:{self.config.aggregator_tcp_port}" for i in range(num_gpus)
            )
            agg_health_port_map = ",".join(
                f"{i}:{self.config.aggregator_health_port}" for i in range(num_gpus)
            )
        else:
            agg_port_map = ",".join(
                f"{i}:{self.config.aggregator_tcp_port + i}" for i in range(num_gpus)
            )
            agg_health_port_map = ",".join(
                f"{i}:{self.config.aggregator_health_port + i}" for i in range(num_gpus)
            )

        cmd = [
            sys.executable, "-m", self.config.controller_module,
            "--port", str(self.config.controller_port),
            "--max-workers", str(self.config.max_workers),
            "--base-port", str(self.config.worker_base_port),
            "--aggregator-port", str(self.config.aggregator_tcp_port),
            "--aggregator-port-map", agg_port_map,
            "--aggregator-health-port-map", agg_health_port_map,
            "--scheduler", self.config.scheduler,
            "--scale-down-delay", str(self.config.scale_down_delay),
            "--max-workers-per-gpu", str(self._per_gpu_cap(num_gpus)),
            "--worker-devices", ",".join(f"cuda:{g['index']}" for g in self._detected_gpus) or "cuda:0",
        ]
        if not self.config.use_fork:
            cmd.append("--no-fork")

        log_dir = Path(self.config.project_root) / "logs"
        log_dir.mkdir(exist_ok=True)
        ctrl_log = open(log_dir / f"controller-{self.config.node_id}.log", "a")

        # Controller + workers need per-GPU MPS pipe
        env = os.environ.copy()
        gpu_index = self.config.aggregator_device.replace("cuda:", "")
        env["CUDA_MPS_PIPE_DIRECTORY"] = f"/tmp/mps_{gpu_index}"

        self._controller_process = subprocess.Popen(
            cmd,
            stdout=ctrl_log,
            stderr=ctrl_log,
            env=env,
            cwd=self.config.project_root,
        )
        ctrl_log.close()  # Subprocess inherited the fd, safe to close

        # Wait briefly for controller to start
        await asyncio.sleep(2.0)

        if self._controller_process.poll() is None:
            logger.info(f"Local controller started on port {self.config.controller_port}")
        else:
            logger.error("Local controller failed to start")

    async def _stop_aggregator(self, graceful: bool = True, timeout: float = 30.0) -> dict:
        """Stop the aggregator and controller"""
        result = {"aggregator_stopped": False, "controller_stopped": False}

        # Stop controller first
        if self._controller_process:
            result["controller_stopped"] = await self._stop_process(
                self._controller_process, "controller", graceful, timeout / 2
            )
            self._controller_process = None

        # Stop aggregator
        if self._aggregator_process:
            result["aggregator_stopped"] = await self._stop_process(
                self._aggregator_process, "aggregator", graceful, timeout / 2
            )
            self._aggregator_process = None

        return result

    async def _stop_process(
        self,
        process: subprocess.Popen,
        name: str,
        graceful: bool,
        timeout: float
    ) -> bool:
        """Stop a process gracefully or forcefully (non-blocking)."""
        if process.poll() is not None:
            return True  # Already stopped

        logger.info(f"Stopping {name} (pid={process.pid})")

        if graceful:
            process.terminate()
            try:
                # Non-blocking wait: run in thread to avoid blocking event loop
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, process.wait),
                    timeout=timeout
                )
                return True
            except asyncio.TimeoutError:
                logger.warning(f"{name} didn't stop gracefully, killing")

        process.kill()
        try:
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, process.wait),
                timeout=5
            )
        except asyncio.TimeoutError:
            logger.error(f"{name} did not die after SIGKILL")
        return True

    async def _kill_process(self, process: Optional[subprocess.Popen]):
        """Force kill a process (non-blocking)."""
        if process and process.poll() is None:
            process.kill()
            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, process.wait),
                    timeout=5
                )
            except asyncio.TimeoutError:
                pass

    def _is_aggregator_running(self) -> bool:
        """Check if aggregator is running"""
        return self._aggregator_process is not None and self._aggregator_process.poll() is None

    def _is_controller_running(self) -> bool:
        """Check if local controller is running"""
        return self._controller_process is not None and self._controller_process.poll() is None

    # =========================================================================
    # Status & Forwarding
    # =========================================================================

    async def _get_status(self) -> NodeStatusResponse:
        """Get current node status"""
        # Determine status
        if self._is_aggregator_running() and self._is_controller_running():
            status = "active"
        elif self._is_aggregator_running():
            status = "starting"  # Aggregator up but controller not ready
        else:
            status = "idle"

        # Get worker info and node metrics from local controller
        active_workers = 0
        loaded_adapters = []
        node_metrics = None

        total_workers = 0
        free_workers = 0
        queue_depth = 0

        if self._is_controller_running():
            try:
                workers_info = await self._get_workers()
                all_workers = workers_info.get("workers", [])
                ready_workers = [w for w in all_workers if w.get("status") == "ready"]
                total_workers = len(ready_workers)
                active_workers = sum(
                    1 for w in ready_workers if w.get("active_requests", 0) > 0
                )
                free_workers = total_workers - active_workers
                queue_depth = workers_info.get("queue_depth", 0)
                loaded_adapters = list(set(
                    w.get("adapter_id", w.get("adapter", ""))
                    for w in ready_workers
                    if w.get("adapter_id") or w.get("adapter")
                ))
            except Exception as e:
                logger.debug(f"Failed to get worker info for status: {e}")

            # Collect aggregate load metrics
            try:
                node_metrics = await self._get_node_metrics()
            except Exception as e:
                logger.debug(f"Failed to get node metrics: {e}")

        return NodeStatusResponse(
            node_id=self.config.node_id,
            status=status,
            aggregator_running=self._is_aggregator_running(),
            aggregator_pid=self._aggregator_process.pid if self._aggregator_process else None,
            base_model=self._current_base_model,
            controller_running=self._is_controller_running(),
            active_workers=active_workers,
            total_workers=total_workers,
            free_workers=free_workers,
            queue_depth=queue_depth,
            loaded_adapters=loaded_adapters,
            uptime=time.time() - self._started_at,
            node_metrics=node_metrics,
        )

    async def _get_workers(self) -> dict:
        """Get worker info from local controller"""
        if not self._is_controller_running():
            return {"workers": []}

        try:
            async with self._session.get(
                f"http://localhost:{self.config.controller_port}/workers",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    try:
                        return await response.json()
                    except Exception as e:
                        logger.warning(f"Invalid JSON from controller /workers: {e}")
                        return {"workers": []}
                else:
                    logger.debug(f"Controller /workers returned {response.status}")
        except Exception as e:
            logger.debug(f"Failed to get workers from controller: {e}")

        return {"workers": []}

    async def _get_node_metrics(self) -> Optional[Dict]:
        """Get aggregate node metrics from local controller."""
        if not self._is_controller_running():
            return None

        try:
            async with self._session.get(
                f"http://localhost:{self.config.controller_port}/node_metrics",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.debug(f"Failed to get node metrics from controller: {e}")

        return None

    async def _forward_to_controller(self, endpoint: str, payload: dict) -> dict:
        """Forward a request to the local controller with retry."""
        if not self._is_controller_running():
            raise HTTPException(status_code=503, detail="Controller not running")

        retry = RetryExecutor(RetryConfig(max_attempts=2, base_delay=0.1))

        async def do_forward():
            url = f"http://localhost:{self.config.controller_port}{endpoint}"
            async with self._session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status != 200:
                    try:
                        body = await response.json()
                        detail = body.get("detail", str(body))
                    except Exception:
                        detail = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=detail[:200]
                    )
                try:
                    return await response.json()
                except Exception as e:
                    text = await response.text()
                    raise HTTPException(
                        status_code=502,
                        detail=f"Invalid JSON from controller: {e}, body: {text[:200]}"
                    )

        try:
            return await retry.execute(do_forward)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    # =========================================================================
    # Run
    # =========================================================================

    def run(self):
        """Run the node agent with signal handling for graceful shutdown."""
        def _signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            # Schedule async stop on the event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.stop())

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)

        uvicorn.run(self.app, host=self.config.host, port=self.config.port)


# CLI Entry Point

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Node Agent")
    parser.add_argument("--node-id", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=NODE_AGENT_PORT)
    parser.add_argument("--global-controller", type=str, default=f"http://localhost:{GLOBAL_CONTROLLER_PORT}")
    parser.add_argument("--aggregator-device", type=str, default=AGGREGATOR_DEVICE)
    parser.add_argument("--aggregator-health-port", type=int, default=AGGREGATOR_HEALTH_PORT,
                        help=f"HTTP port for aggregator health checks (default: {AGGREGATOR_HEALTH_PORT})")
    parser.add_argument("--aggregator-tcp-port", type=int, default=DEFAULT_PORT,
                        help=f"TCP port for aggregator worker registration (default: {DEFAULT_PORT})")
    parser.add_argument("--controller-port", type=int, default=CONTROLLER_PORT,
                        help=f"Port for local controller (default: {CONTROLLER_PORT})")
    parser.add_argument("--worker-base-port", type=int, default=WORKER_BASE_PORT,
                        help=f"Base port for workers (default: {WORKER_BASE_PORT})")
    parser.add_argument("--gpu-type", type=str, default="unknown")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--project-root", type=str, default="")
    parser.add_argument("--register-host", type=str, default="",
                        help="Override registration host (for SSH tunnel setups)")
    parser.add_argument("--register-port", type=int, default=0,
                        help="Override registration port (for SSH tunnel setups)")
    parser.add_argument("--scheduler", type=str, default="default",
                        help="Local scheduling algorithm (default, inter)")
    parser.add_argument("--no-fork", action="store_true",
                        help="Disable fork-based worker spawning")
    parser.add_argument("--scale-down-delay", type=float, default=5.0,
                        help="Seconds idle before worker scale-down (default: 5.0)")
    parser.add_argument("--single-aggregator", action="store_true",
                        help="All workers share GPU 0 aggregator (for small-memory GPUs)")

    args = parser.parse_args()

    if not args.project_root:
        # Default to the repo root (node_agent.py lives at src/controller/cluster/)
        args.project_root = str(Path(__file__).parent.parent.parent.parent)

    config = NodeAgentConfig(
        node_id=args.node_id,
        host=args.host,
        port=args.port,
        global_controller_url=args.global_controller,
        aggregator_device=args.aggregator_device,
        aggregator_health_port=args.aggregator_health_port,
        aggregator_tcp_port=args.aggregator_tcp_port,
        controller_port=args.controller_port,
        worker_base_port=args.worker_base_port,
        gpu_type=args.gpu_type,
        max_workers=args.max_workers,
        project_root=args.project_root,
        register_host=args.register_host,
        register_port=args.register_port,
        scheduler=args.scheduler,
        scale_down_delay=args.scale_down_delay,
        use_fork=not args.no_fork,
        single_aggregator=args.single_aggregator,
    )

    agent = NodeAgent(config)
    agent.run()
