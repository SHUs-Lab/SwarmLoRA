#!/usr/bin/env python3
"""
Pre-Loading Agent for ServerlessLoRA.

Implements Section 3.2 of the paper:
- Runs on each worker node
- Executes pre-loading decisions from scheduler
- Manages container lifecycle (spawn, terminate, keep-alive)
- Sends commands to containers and GPUs

REST API Endpoints:
- POST /preload   - Load artifact to container/GPU
- POST /offload   - Offload artifact from GPU
- POST /spawn     - Spawn new container
- POST /terminate - Terminate container
- GET  /status    - Node memory status
- GET  /containers - List containers

Port: 7000 + node_index
"""

import os
import sys
import time
import signal
import subprocess
import threading
import logging
import requests
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any

from artifact_registry import (
    get_registry,
    Artifact, ArtifactType, ArtifactLocation,
    Container, WorkerNode
)
from preload_scheduler import PreloadTarget
from config import (
    PRELOAD_AGENT_BASE_PORT,
    CONTAINER_KEEP_ALIVE_MS,
    CONTAINER_SPAWN_TIMEOUT_S,
    AGENT_POLL_INTERVAL_MS,
    BASELINE_WORKER_BASE_PORT,
    BASELINE_SERVER_PORT,
    WORKER_DEVICE,
)

# Feature-detect stream loader for CUDA optimizations (Paper Section 5) --
# this file only warns if unavailable, it doesn't use the module directly.
try:
    import utils.stream_loader  # noqa: F401
    STREAM_LOADER_AVAILABLE = True
except ImportError:
    STREAM_LOADER_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if not STREAM_LOADER_AVAILABLE:
    logger.warning("stream_loader not available - using standard loading")


@dataclass
class ContainerProcess:
    """Tracks a container subprocess."""
    container_id: str
    process: subprocess.Popen
    http_port: int
    function_id: str
    lora_id: str
    created_time: float = field(default_factory=time.time)
    last_health_check: float = 0.0
    healthy: bool = False


class PreloadAgent:
    """
    Per-node pre-loading agent.

    Paper Section 3.2: "A pre-loading agent runs in each worker node.
    It receives commands from the scheduler and manages local containers."
    """

    def __init__(
        self,
        node_id: str,
        hostname: str = "localhost",
        agent_port: int = PRELOAD_AGENT_BASE_PORT,
        scheduler_url: Optional[str] = None,
        base_model_server_port: int = BASELINE_SERVER_PORT,
        worker_base_port: int = BASELINE_WORKER_BASE_PORT,
        worker_device: str = WORKER_DEVICE,
        keep_alive_ms: int = CONTAINER_KEEP_ALIVE_MS,
        poll_interval_ms: int = AGENT_POLL_INTERVAL_MS,
        **kwargs,
    ):
        self.node_id = node_id
        self.hostname = hostname
        self.agent_port = agent_port
        self.scheduler_url = scheduler_url
        self.base_model_server_port = base_model_server_port
        self.worker_base_port = worker_base_port
        self.worker_device = worker_device
        self.keep_alive_ms = keep_alive_ms
        self.poll_interval_ms = poll_interval_ms

        # Batch scheduling parameters (forwarded to workers on spawn)
        self.batch_slo_ms = kwargs.get("batch_slo_ms", 2000.0)
        self.batch_max_batch_size = kwargs.get("batch_max_batch_size", 8)
        self.batch_base_ttft_ms = kwargs.get("batch_base_ttft_ms", 400.0)
        self.batch_marginal_cost_ms = kwargs.get("batch_marginal_cost_ms", 50.0)

        # Container tracking
        self._containers: Dict[str, ContainerProcess] = {}
        self._container_counter = 0
        self._containers_lock = threading.Lock()

        # Spawn concurrency: limit to 16 simultaneous spawns per GPU
        # (BMS is threaded so workers can init concurrently)
        self._spawn_semaphore = threading.Semaphore(16)

        # Port allocation
        self._allocated_ports: Set[int] = set()
        self._port_lock = threading.Lock()

        # Swap cooldown: container_id -> timestamp of last 503 failure
        # Prevents wasting 500ms per busy container on repeated attempts
        self._swap_busy_cooldown: Dict[str, float] = {}
        self._swap_busy_cooldown_s = 30.0  # skip for 30s after 503

        # Registry
        self.registry = get_registry()

        # Agent threads
        self._running = False
        self._keep_alive_thread: Optional[threading.Thread] = None
        self._scheduler_poll_thread: Optional[threading.Thread] = None

        # Statistics
        self._total_spawned = 0
        self._total_terminated = 0
        self._total_preloads = 0

    # -------------------------------------------------------------------------
    # Container Management
    # -------------------------------------------------------------------------

    def spawn_container(
        self,
        function_id: str,
        lora_id: str,
        http_port: Optional[int] = None,
        extra_args: Optional[List[str]] = None
    ) -> Optional[Container]:
        """
        Spawn a new container (worker process) for a function.

        Args:
            function_id: Function identifier
            lora_id: LoRA adapter ID to load
            http_port: Specific port (or auto-allocate)
            extra_args: Additional worker arguments

        Returns:
            Container object if successful, None otherwise
        """
        # Throttle concurrent spawns to avoid thundering herd
        self._spawn_semaphore.acquire()
        try:
            return self._spawn_container_inner(function_id, lora_id, http_port, extra_args)
        finally:
            self._spawn_semaphore.release()

    def _spawn_container_inner(
        self,
        function_id: str,
        lora_id: str,
        http_port: Optional[int] = None,
        extra_args: Optional[List[str]] = None
    ) -> Optional[Container]:
        if http_port is None:
            http_port = self._allocate_port()
            if http_port is None:
                logger.error("No available ports for new container")
                return None

        with self._containers_lock:
            self._container_counter += 1
            container_id = f"{self.node_id}_container_{self._container_counter}"

        logger.info(f"Spawning container {container_id} for {function_id} on port {http_port}")

        cmd = [
            sys.executable, "worker_batched.py",
            "--server-host", self.hostname,
            "--server-port", str(self.base_model_server_port),
            "--http-port", str(http_port),
            "--worker-id", str(self._container_counter),
            "--lora", lora_id,
            "--device", self.worker_device,
            "--slo-ms", str(self.batch_slo_ms),
            "--max-batch-size", str(self.batch_max_batch_size),
            "--base-ttft-ms", str(self.batch_base_ttft_ms),
            "--marginal-cost-ms", str(self.batch_marginal_cost_ms),
        ]

        if extra_args:
            cmd.extend(extra_args)

        try:
            # Start worker process — log stderr for crash diagnosis
            log_dir = os.path.join(os.path.dirname(__file__), "logs")
            os.makedirs(log_dir, exist_ok=True)
            worker_log = open(os.path.join(log_dir, f"worker_{container_id}.log"), "w")
            # Limit PyTorch CPU threads: workers do GPU compute, not CPU parallelism.
            # Without this, each worker spawns 128 threads (64 intra + 64 interop)
            # which hits the cgroup PID limit (31232) with 160+ workers.
            worker_env = os.environ.copy()
            worker_env["OMP_NUM_THREADS"] = "1"
            worker_env["MKL_NUM_THREADS"] = "1"
            process = subprocess.Popen(
                cmd,
                stdout=worker_log,
                stderr=worker_log,
                env=worker_env,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )

            container_proc = ContainerProcess(
                container_id=container_id,
                process=process,
                http_port=http_port,
                function_id=function_id,
                lora_id=lora_id
            )

            with self._containers_lock:
                self._containers[container_id] = container_proc

            # Wait for container to become healthy
            if self._wait_for_healthy(container_id, timeout=CONTAINER_SPAWN_TIMEOUT_S):
                # Create registry container object
                container = Container(
                    container_id=container_id,
                    function_id=function_id,
                    node_id=self.node_id,
                    http_port=http_port,
                    http_host=self.hostname,
                    status="ready",
                    keep_alive_ms=self.keep_alive_ms,
                    lora_id=lora_id
                )

                # Register in registry
                self.registry.register_container(container)
                self._total_spawned += 1

                logger.info(f"Container {container_id} ready on port {http_port}")
                return container
            else:
                logger.error(f"Container {container_id} failed to become healthy")
                self._cleanup_container(container_id)
                return None

        except Exception as e:
            logger.error(f"Failed to spawn container: {e}")
            self._release_port(http_port)
            return None

    def terminate_container(self, container_id: str) -> bool:
        """
        Terminate a container.

        Args:
            container_id: Container to terminate

        Returns:
            True if successful
        """
        logger.info(f"Terminating container {container_id}")

        with self._containers_lock:
            container_proc = self._containers.get(container_id)
            if not container_proc:
                logger.warning(f"Container {container_id} not found")
                return False

        try:
            shutdown_url = f"http://{self.hostname}:{container_proc.http_port}/shutdown"
            requests.post(shutdown_url, timeout=5.0)
            time.sleep(0.5)
        except Exception:
            pass

        return self._cleanup_container(container_id)

    def _cleanup_container(self, container_id: str) -> bool:
        """Clean up container process and resources."""
        with self._containers_lock:
            container_proc = self._containers.pop(container_id, None)
            if not container_proc:
                return False

        try:
            if container_proc.process.poll() is None:
                if hasattr(os, 'killpg'):
                    os.killpg(os.getpgid(container_proc.process.pid), signal.SIGTERM)
                else:
                    container_proc.process.terminate()
                container_proc.process.wait(timeout=5.0)
        except Exception as e:
            logger.warning(f"Error killing container process: {e}")
            try:
                container_proc.process.kill()
            except Exception:
                pass

        self._release_port(container_proc.http_port)

        self.registry.remove_container(container_id)
        self._total_terminated += 1

        logger.info(f"Container {container_id} terminated")
        return True

    def _wait_for_healthy(self, container_id: str, timeout: float) -> bool:
        """Wait for container to pass health check."""
        with self._containers_lock:
            container_proc = self._containers.get(container_id)
            if not container_proc:
                return False
            port = container_proc.http_port

        health_url = f"http://{self.hostname}:{port}/health"
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                resp = requests.get(health_url, timeout=2.0)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "ready":
                        with self._containers_lock:
                            if container_id in self._containers:
                                self._containers[container_id].healthy = True
                                self._containers[container_id].last_health_check = time.time()
                        return True
            except Exception:
                pass
            time.sleep(1.0)

        return False

    # -------------------------------------------------------------------------
    # Port Management
    # -------------------------------------------------------------------------

    def _allocate_port(self) -> Optional[int]:
        """Allocate an available port, skipping ports in use by other processes."""
        import socket
        with self._port_lock:
            for port in range(self.worker_base_port, self.worker_base_port + 100):
                if port not in self._allocated_ports:
                    # Probe if port is already bound by another process
                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            s.settimeout(0.1)
                            s.connect(("127.0.0.1", port))
                        # Connected — port is in use by someone else
                        continue
                    except (ConnectionRefusedError, OSError):
                        # Connection refused or timeout — port is free
                        pass
                    self._allocated_ports.add(port)
                    return port
        return None

    def _release_port(self, port: int):
        """Release an allocated port."""
        with self._port_lock:
            self._allocated_ports.discard(port)

    # -------------------------------------------------------------------------
    # Preloading Operations
    # -------------------------------------------------------------------------

    def preload_artifact(
        self,
        artifact_id: str,
        target_location: PreloadTarget,
        container_id: Optional[str] = None
    ) -> bool:
        """
        Preload an artifact to container or GPU.

        Args:
            artifact_id: Artifact to load
            target_location: CONTAINER or GPU
            container_id: Target container (optional)

        Returns:
            True if successful
        """
        logger.info(f"Preloading {artifact_id} to {target_location.value}")

        artifact = self.registry.get_artifact(artifact_id)
        if not artifact:
            logger.error(f"Artifact {artifact_id} not found")
            return False

        if artifact.artifact_type == ArtifactType.ADAPTER:
            return self._preload_adapter(artifact, target_location, container_id)

        # For backbones, they're shared via IPC - nothing to preload
        if artifact.artifact_type == ArtifactType.BACKBONE:
            logger.info(f"Backbone {artifact_id} shared via IPC, no preload needed")
            return True

        # For libraries/kernels, they're part of the container
        logger.info(f"Artifact type {artifact.artifact_type} handled at container spawn")
        return True

    def _preload_adapter(
        self,
        artifact: Artifact,
        target_location: PreloadTarget,
        container_id: Optional[str] = None
    ) -> bool:
        """Preload a LoRA adapter into a container's cache (not active swap)."""
        if container_id:
            container = self.registry.get_container(container_id)
        else:
            # Find container with fewest pending requests
            containers = self._get_local_containers()
            if containers:
                container = min(containers, key=lambda c: c.pending_requests)
            else:
                container = None

        if not container:
            logger.warning(f"No container available for adapter preload")
            return False

        try:
            adapter_id = artifact.artifact_id.replace("adapter_", "").replace("_", "/")

            # Use local URL (real bind port) to reach the worker on this node
            base_url = self._local_url(container.container_id) or container.get_url()
            preload_url = f"{base_url}/preload_adapter"
            payload = {
                "adapter_id": adapter_id,
                "to_gpu": target_location == PreloadTarget.GPU
            }

            resp = requests.post(preload_url, json=payload, timeout=60.0)
            if resp.status_code == 200:
                location = (ArtifactLocation.GPU if target_location == PreloadTarget.GPU
                           else ArtifactLocation.CONTAINER)
                self.registry.update_artifact_location(
                    artifact.artifact_id, location,
                    node_id=self.node_id,
                    container_id=container.container_id
                )

                container.loaded_artifacts.add(artifact.artifact_id)
                if target_location == PreloadTarget.GPU:
                    container.gpu_loaded_artifacts.add(artifact.artifact_id)

                self._total_preloads += 1
                logger.info(f"Preloaded adapter {adapter_id} to {target_location.value}")
                return True
            else:
                logger.error(f"Preload failed: {resp.text}")
                return False

        except Exception as e:
            logger.error(f"Preload error: {e}")
            return False

    def offload_artifact(
        self,
        artifact_id: str,
        to_container: bool = True
    ) -> bool:
        """
        Offload an artifact from GPU.

        Args:
            artifact_id: Artifact to offload
            to_container: If True, keep in container memory; if False, unload completely

        Returns:
            True if successful
        """
        logger.info(f"Offloading {artifact_id}, keep_in_container={to_container}")

        artifact = self.registry.get_artifact(artifact_id)
        if not artifact:
            logger.error(f"Artifact {artifact_id} not found")
            return False

        container = self.registry.get_container(artifact.loaded_at_container)
        if not container:
            logger.warning(f"Container for artifact not found")
            return False

        try:
            base_url = self._local_url(container.container_id) or container.get_url()
            offload_url = f"{base_url}/offload_adapter"
            payload = {
                "artifact_id": artifact_id,
                "keep_in_container": to_container
            }

            resp = requests.post(offload_url, json=payload, timeout=30.0)
            if resp.status_code == 200:
                new_location = ArtifactLocation.CONTAINER if to_container else ArtifactLocation.DISK
                self.registry.update_artifact_location(
                    artifact_id, new_location,
                    node_id=self.node_id if to_container else None,
                    container_id=container.container_id if to_container else None
                )

                container.gpu_loaded_artifacts.discard(artifact_id)
                if not to_container:
                    container.loaded_artifacts.discard(artifact_id)

                logger.info(f"Offloaded {artifact_id}")
                return True
            else:
                logger.error(f"Offload failed: {resp.text}")
                return False

        except Exception as e:
            logger.error(f"Offload error: {e}")
            return False

    # -------------------------------------------------------------------------
    # Background Loops
    # -------------------------------------------------------------------------

    def _keep_alive_loop(self):
        """Check container health and terminate expired containers."""
        while self._running:
            try:
                self._check_containers()
            except Exception as e:
                logger.error(f"Keep-alive loop error: {e}")

            time.sleep(5.0)  # Check every 5 seconds

    def _check_containers(self):
        """Check all containers for health and expiry."""
        now = time.time()

        with self._containers_lock:
            container_ids = list(self._containers.keys())

        for container_id in container_ids:
            with self._containers_lock:
                container_proc = self._containers.get(container_id)
                if not container_proc:
                    continue
                port = container_proc.http_port

            # Check if process is still running
            if container_proc.process.poll() is not None:
                exit_code = container_proc.process.returncode
                logger.warning(f"Container {container_id} process died (exit={exit_code})")
                self._cleanup_container(container_id)
                continue

            # Health check
            try:
                health_url = f"http://{self.hostname}:{port}/health"
                resp = requests.get(health_url, timeout=2.0)
                if resp.status_code == 200:
                    container_proc.healthy = True
                    container_proc.last_health_check = now
                else:
                    container_proc.healthy = False
            except Exception:
                container_proc.healthy = False

            # Check expiry
            registry_container = self.registry.get_container(container_id)
            if registry_container and registry_container.is_expired():
                logger.info(f"Container {container_id} expired, terminating")
                self.terminate_container(container_id)

    def _scheduler_poll_loop(self):
        """Poll scheduler for pending decisions."""
        if not self.scheduler_url:
            return

        while self._running:
            try:
                self._fetch_and_execute_decisions()
            except Exception as e:
                logger.error(f"Scheduler poll error: {e}")

            time.sleep(self.poll_interval_ms / 1000.0)

    def _fetch_and_execute_decisions(self):
        """Fetch and execute pending decisions from scheduler."""
        try:
            url = f"{self.scheduler_url}/decisions/{self.node_id}"
            resp = requests.get(url, timeout=5.0)
            if resp.status_code != 200:
                return

            data = resp.json()
            decisions = data.get("decisions", [])

            for d in decisions:
                success = self._execute_decision(d)

                # Report completion
                complete_url = f"{self.scheduler_url}/decisions/{d['decision_id']}/complete"
                requests.post(complete_url, json={"success": success}, timeout=5.0)

        except Exception as e:
            logger.debug(f"Failed to fetch decisions: {e}")

    def _execute_decision(self, decision: Dict) -> bool:
        """
        Execute a single preload decision.

        Paper Section 4.1: The agent receives preload decisions from the PCKP
        scheduler and executes them. When the target artifact (adapter) needs
        to be loaded on this node:
        1. If a specific container is targeted, swap its adapter if different
        2. Otherwise, select the best local container:
           - Prefer idle containers serving low-rate functions
           - Swap the selected container's adapter to the target function
        """
        artifact_id = decision.get("artifact_id")
        target = decision.get("target_location", "container")
        container_id = decision.get("target_container_id")
        function_id = decision.get("function_id")

        target_location = (PreloadTarget.GPU if target == "gpu"
                          else PreloadTarget.CONTAINER)

        adapter_id = artifact_id.replace("adapter_", "").replace("_", "/")

        if container_id:
            container = self.registry.get_container(container_id)
            if container and container.lora_id:
                if container.lora_id != adapter_id:
                    return self._swap_container_adapter(
                        container, adapter_id, function_id)
                else:
                    # Container already serves the target adapter — no action needed
                    logger.debug(f"Container {container_id} already has {adapter_id}")
                    return True
            return self.preload_artifact(artifact_id, target_location, container_id)

        # No specific container — agent selects the best one on this node
        # Paper: agent decides which container to use based on local state
        containers = self._get_local_containers()
        if not containers:
            logger.warning(f"No local containers for decision {decision.get('decision_id')}")
            return False

        for c in containers:
            if c.function_id == function_id:
                # Already have a container for this function — just preload to GPU
                return self.preload_artifact(artifact_id, target_location,
                                            c.container_id)

        # Select best container to swap: prefer idle, low-rate functions
        # Try multiple candidates if swap fails (e.g., container busy with inference)
        tried = set()
        while True:
            best_container = self._select_swap_candidate(
                [c for c in containers if c.container_id not in tried],
                function_id)
            if best_container is None:
                break

            tried.add(best_container.container_id)
            logger.info(f"Agent selecting {best_container.container_id} "
                        f"(serving {best_container.function_id}) for swap to {function_id}")
            result = self._swap_container_adapter(
                best_container, adapter_id, function_id)
            if result:
                return True
            # Swap failed (container busy) — try next candidate
            logger.info(f"Swap failed on {best_container.container_id}, trying next candidate")

        return False

    def _select_swap_candidate(
        self, containers: List[Container], target_function_id: str
    ) -> Optional[Container]:
        """
        Select the best container for adapter swap.

        Paper Section 4.1: The agent picks an idle container on this node.
        Only swaps if target function has significantly higher demand than
        the source function, and respects a cooldown period after swaps.
        """
        target_rate = (self.registry.get_request_rate(target_function_id)
                       if target_function_id else 0.0)

        candidates = []
        now = time.time()
        for c in containers:
            # Skip containers that are busy (pending requests > 0)
            if c.pending_requests > 0:
                continue
            # Skip if already serving the target function
            if c.function_id == target_function_id:
                continue
            # Skip if recently swapped (30s cooldown to prevent thrashing)
            last_swap = getattr(c, '_last_swap_time', 0.0)
            if now - last_swap < 30.0:
                continue
            # Skip containers that recently returned 503 (busy with inference)
            busy_until = self._swap_busy_cooldown.get(c.container_id, 0.0)
            if now < busy_until:
                continue
            # Only swap if target demand is meaningfully higher than source
            src_rate = self.registry.get_request_rate(c.function_id) if c.function_id else 0.0
            if src_rate > 0 and target_rate < src_rate * 2.0:
                continue  # target must have 2x higher demand
            candidates.append(c)

        if not candidates:
            return None

        # Score candidates: prefer containers serving LOW-demand functions
        def swap_score(c):
            rate = self.registry.get_request_rate(c.function_id) if c.function_id else 0.0
            return rate  # lower rate = better swap candidate

        candidates.sort(key=swap_score)
        return candidates[0]

    def _swap_container_adapter(
        self,
        container: Container,
        new_adapter_id: str,
        new_function_id: Optional[str] = None
    ) -> bool:
        """Swap a container's active adapter via its /swap_adapter endpoint."""
        try:
            logger.info(f"Swapping adapter on {container.container_id}: "
                       f"{container.lora_id} -> {new_adapter_id}")
            swap_start = time.time()
            base_url = self._local_url(container.container_id) or container.get_url()
            resp = requests.post(
                f"{base_url}/swap_adapter",
                json={
                    "adapter_id": new_adapter_id,
                    "function_id": new_function_id or "",
                },
                timeout=10.0
            )
            swap_ms = (time.time() - swap_start) * 1000

            # Fast-fail: worker returned 503 = model busy with inference
            if resp.status_code == 503:
                logger.info(f"Container {container.container_id} busy (model lock held), "
                           f"swap skipped after {swap_ms:.0f}ms — cooldown {self._swap_busy_cooldown_s}s")
                self._swap_busy_cooldown[container.container_id] = (
                    time.time() + self._swap_busy_cooldown_s)
                return False

            if resp.status_code == 200 and resp.json().get("success"):
                # Update registry binding
                if new_function_id:
                    self.registry.rebind_container(
                        container.container_id, new_function_id, new_adapter_id)
                # Update loaded_artifacts so controller scores this container
                # correctly for the new adapter
                old_adapter_artifact = (
                    f"adapter_{container.lora_id.replace('/', '_')}"
                    if container.lora_id else None
                )
                new_adapter_artifact = f"adapter_{new_adapter_id.replace('/', '_')}"
                if old_adapter_artifact:
                    container.loaded_artifacts.discard(old_adapter_artifact)
                    container.gpu_loaded_artifacts.discard(old_adapter_artifact)
                container.loaded_artifacts.add(new_adapter_artifact)
                container.gpu_loaded_artifacts.add(new_adapter_artifact)
                # Log worker-reported breakdown if available
                data = resp.json()
                worker_detail = ""
                if "swap_ms" in data:
                    worker_detail = (f" (worker: remove={data.get('remove_ms', 0):.0f}ms"
                                     f" apply={data.get('apply_ms', 0):.0f}ms)")
                logger.info(f"Adapter swap on {container.container_id}: "
                           f"{swap_ms:.0f}ms e2e{worker_detail}")
                # Set cooldown to prevent re-swapping this container too soon
                container._last_swap_time = time.time()
                return True
            else:
                logger.error(f"Adapter swap returned (after {swap_ms:.0f}ms): {resp.text}")
                return False
        except Exception as e:
            swap_ms = (time.time() - swap_start) * 1000
            logger.error(f"Adapter swap failed on {container.container_id} "
                        f"after {swap_ms:.0f}ms: {e}")
            return False

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def _local_url(self, container_id: str) -> Optional[str]:
        """Get the local URL for a container (using the bind port).
        Used by the agent to reach workers on the same node."""
        with self._containers_lock:
            cp = self._containers.get(container_id)
            if cp:
                return f"http://{self.hostname}:{cp.http_port}"
        return None

    def _get_local_containers(self) -> List[Container]:
        """Get all containers on this node."""
        with self._containers_lock:
            container_ids = list(self._containers.keys())
        return [self.registry.get_container(cid) for cid in container_ids
                if self.registry.get_container(cid)]

    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        with self._containers_lock:
            containers_info = [
                {
                    "container_id": cp.container_id,
                    "function_id": cp.function_id,
                    "http_port": cp.http_port,
                    "healthy": cp.healthy,
                    "uptime_s": time.time() - cp.created_time
                }
                for cp in self._containers.values()
            ]

        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "agent_port": self.agent_port,
            "running": self._running,
            "containers": containers_info,
            "total_spawned": self._total_spawned,
            "total_terminated": self._total_terminated,
            "total_preloads": self._total_preloads
        }

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def start(self):
        """Start agent background threads."""
        if self._running:
            return

        self._running = True

        node = WorkerNode(
            node_id=self.node_id,
            hostname=self.hostname,
            agent_port=self.agent_port
        )
        self.registry.register_node(node)

        self._keep_alive_thread = threading.Thread(
            target=self._keep_alive_loop,
            daemon=True,
            name="KeepAliveLoop"
        )
        self._keep_alive_thread.start()

        if self.scheduler_url:
            self._scheduler_poll_thread = threading.Thread(
                target=self._scheduler_poll_loop,
                daemon=True,
                name="SchedulerPollLoop"
            )
            self._scheduler_poll_thread.start()

        logger.info(f"PreloadAgent started for node {self.node_id}")

    def stop(self):
        """Stop agent and terminate all containers."""
        self._running = False

        if self._keep_alive_thread:
            self._keep_alive_thread.join(timeout=5.0)
        if self._scheduler_poll_thread:
            self._scheduler_poll_thread.join(timeout=5.0)

        with self._containers_lock:
            container_ids = list(self._containers.keys())

        for container_id in container_ids:
            self.terminate_container(container_id)

        logger.info(f"PreloadAgent stopped for node {self.node_id}")

    # -------------------------------------------------------------------------
    # HTTP API
    # -------------------------------------------------------------------------

    def run_server(self, port: Optional[int] = None):
        """Run as HTTP server."""
        try:
            from flask import Flask, request, jsonify
        except ImportError:
            logger.error("Flask not installed. Run: pip install flask")
            return

        if port is None:
            port = self.agent_port

        app = Flask(f"preload_agent_{self.node_id}")

        @app.route('/health', methods=['GET'])
        def health():
            return jsonify({"status": "running" if self._running else "stopped"})

        @app.route('/status', methods=['GET'])
        def status():
            return jsonify(self.get_status())

        @app.route('/containers', methods=['GET'])
        def list_containers():
            containers = self._get_local_containers()
            return jsonify({
                "containers": [
                    {
                        "container_id": c.container_id,
                        "function_id": c.function_id,
                        "http_port": c.http_port,
                        "status": c.status
                    }
                    for c in containers if c
                ]
            })

        @app.route('/spawn', methods=['POST'])
        def spawn():
            data = request.get_json() or {}
            function_id = data.get("function_id", "default")
            lora_id = data.get("lora_id", "")
            http_port = data.get("http_port")

            if not lora_id:
                return jsonify({"error": "lora_id required"}), 400

            container = self.spawn_container(function_id, lora_id, http_port)
            if container:
                return jsonify({
                    "success": True,
                    "container_id": container.container_id,
                    "http_port": container.http_port
                })
            else:
                return jsonify({"success": False, "error": "Failed to spawn"}), 500

        @app.route('/terminate', methods=['POST'])
        def terminate():
            data = request.get_json() or {}
            container_id = data.get("container_id")

            if not container_id:
                return jsonify({"error": "container_id required"}), 400

            success = self.terminate_container(container_id)
            return jsonify({"success": success})

        @app.route('/preload', methods=['POST'])
        def preload():
            data = request.get_json() or {}
            artifact_id = data.get("artifact_id")
            target = data.get("target_location", "container")
            container_id = data.get("container_id")

            if not artifact_id:
                return jsonify({"error": "artifact_id required"}), 400

            target_location = (PreloadTarget.GPU if target == "gpu"
                              else PreloadTarget.CONTAINER)
            success = self.preload_artifact(artifact_id, target_location, container_id)
            return jsonify({"success": success})

        @app.route('/offload', methods=['POST'])
        def offload():
            data = request.get_json() or {}
            artifact_id = data.get("artifact_id")
            keep_in_container = data.get("keep_in_container", True)

            if not artifact_id:
                return jsonify({"error": "artifact_id required"}), 400

            success = self.offload_artifact(artifact_id, keep_in_container)
            return jsonify({"success": success})

        @app.route('/shutdown', methods=['POST'])
        def shutdown():
            self.stop()
            threading.Thread(target=lambda: (time.sleep(0.5), os._exit(0)),
                           daemon=True).start()
            return jsonify({"status": "shutting_down"})

        # Start agent
        self.start()

        logger.info(f"PreloadAgent HTTP server on port {port}")
        app.run(host='0.0.0.0', port=port, threaded=True, use_reloader=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pre-Loading Agent")
    parser.add_argument("--node-id", type=str, default="node0",
                        help="Node identifier")
    parser.add_argument("--hostname", type=str, default="localhost",
                        help="Node hostname")
    parser.add_argument("--port", type=int, default=PRELOAD_AGENT_BASE_PORT,
                        help="Agent HTTP port")
    parser.add_argument("--scheduler-url", type=str, default=None,
                        help="Pre-Loading Scheduler URL")
    parser.add_argument("--server-port", type=int, default=BASELINE_SERVER_PORT,
                        help="Base model server port")
    parser.add_argument("--worker-base-port", type=int, default=BASELINE_WORKER_BASE_PORT,
                        help="Base port for worker containers")
    parser.add_argument("--device", type=str, default=WORKER_DEVICE,
                        help="GPU device for workers")
    parser.add_argument("--keep-alive-ms", type=int, default=CONTAINER_KEEP_ALIVE_MS,
                        help="Container keep-alive in milliseconds")
    # Batch scheduling parameters (forwarded to workers)
    parser.add_argument("--batch-slo-ms", type=float, default=2000.0,
                        help="SLO target for batch scheduling")
    parser.add_argument("--batch-max-batch-size", type=int, default=8,
                        help="Maximum batch size for workers")
    parser.add_argument("--batch-base-ttft-ms", type=float, default=400.0,
                        help="Base generation time (from profiling)")
    parser.add_argument("--batch-marginal-cost-ms", type=float, default=50.0,
                        help="Marginal cost per additional batch request")
    args = parser.parse_args()

    agent = PreloadAgent(
        node_id=args.node_id,
        hostname=args.hostname,
        agent_port=args.port,
        scheduler_url=args.scheduler_url,
        base_model_server_port=args.server_port,
        worker_base_port=args.worker_base_port,
        worker_device=args.device,
        keep_alive_ms=args.keep_alive_ms,
        batch_slo_ms=args.batch_slo_ms,
        batch_max_batch_size=args.batch_max_batch_size,
        batch_base_ttft_ms=args.batch_base_ttft_ms,
        batch_marginal_cost_ms=args.batch_marginal_cost_ms,
    )

    try:
        agent.run_server(port=args.port)
    except KeyboardInterrupt:
        agent.stop()


if __name__ == "__main__":
    main()
