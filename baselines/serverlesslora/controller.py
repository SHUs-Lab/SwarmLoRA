#!/usr/bin/env python3
"""
Controller/Frontend for ServerlessLoRA.

Implements Sections 3.2 and 3.3 of the paper:
- Central request router and coordinator
- Instance selection by pre-load score
- Request rate tracking
- Function and node registration
- Health monitoring

Request Flow (Paper Section 3.3 + 4.2):
    Client -> Controller:8000 -> Batch Scheduler (fill-or-expire)
    -> Select Instance (pre-load score) -> POST /batch_execute -> Worker

Instance Selection (by pre-load score):
    score = Σ(loaded_artifacts) × 10
    score += GPU_artifacts × bonus
    score -= busy_penalty
    Select: max(score)

REST API Endpoints:
    POST /inference        - Main inference endpoint
    POST /batch_inference  - Batch requests
    POST /register_function - Register new function
    POST /register_node    - Register worker node
    POST /scale            - Scale containers
    GET  /status           - System status
    GET  /functions        - List functions

Port: 8000
"""

import os
import time
import threading
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Any
import collections
from artifact_registry import (
    ArtifactRegistry, get_registry,
    Artifact,
    FunctionProfile, WorkerNode, Container,
    create_default_artifacts, create_adapter_artifact
)
from preload_scheduler import PreloadScheduler
from gpu_offloader import DynamicGPUOffloader
from utils.batch_scheduler import AdaptiveBatchScheduler, BatchConfig, Batch
from config import (
    CONTROLLER_PORT,
    CONTROLLER_RATE_UPDATE_INTERVAL_S,
    CONTROLLER_HEALTH_CHECK_INTERVAL_S,
    CONTROLLER_MAX_RETRIES,
    DEFAULT_BASE_TTFT_MS,
    DEFAULT_MARGINAL_COST_MS,
    DEFAULT_SLO_MS,
    MODEL_NAME,
    LORA_ADAPTER_ID,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RequestStats:
    """Statistics for a single request."""
    request_id: str
    function_id: str
    container_id: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    ttft_ms: Optional[float] = None
    e2e_ms: Optional[float] = None
    retries: int = 0


class Controller:
    """
    Central controller for ServerlessLoRA.

    Paper Section 3.2: "The Controller receives all inference requests
    and routes them to the appropriate worker containers."
    """

    def __init__(
        self,
        registry: Optional[ArtifactRegistry] = None,
        rate_update_interval_s: float = CONTROLLER_RATE_UPDATE_INTERVAL_S,
        health_check_interval_s: float = CONTROLLER_HEALTH_CHECK_INTERVAL_S,
        max_retries: int = CONTROLLER_MAX_RETRIES,
        enable_preload_scheduler: bool = True,
        enable_gpu_offloader: bool = False,
    ):
        self.registry = registry or get_registry()
        self.rate_update_interval_s = rate_update_interval_s
        self.health_check_interval_s = health_check_interval_s
        self.max_retries = max_retries

        # Request tracking
        self._request_counter = 0
        self._request_lock = threading.Lock()
        self._spawn_node_index = 0
        self._spawn_lock = threading.Lock()  # protects _pending_spawns counter
        self._pending_spawns: dict = {}  # node_id -> count of in-flight spawns
        self._swapping_containers: set = set()  # container IDs currently being swapped
        self._swapping_lock = threading.Lock()  # protects _swapping_containers
        self._selection_lock = threading.Lock()  # serializes container selection + claim
        # LIFO waiter stack: newest batch gets served first when a container frees up
        self._waiter_stack: list = []  # List of threading.Event
        self._waiter_lock = threading.Lock()  # protects _waiter_stack

        # Optional components
        self.preload_scheduler: Optional[PreloadScheduler] = None
        self.enable_gpu_offloader = enable_gpu_offloader
        # Per-GPU offloaders (Paper: "offloader detects whether a GPU has
        # enough remaining space" — one offloader per GPU device)
        self.gpu_offloaders: Dict[str, DynamicGPUOffloader] = {}

        if enable_preload_scheduler:
            self.preload_scheduler = PreloadScheduler(
                registry=self.registry,
                swapping_lock=self._swapping_lock,
                swapping_containers=self._swapping_containers,
            )

        # Controller-level batch scheduler (Paper Section 4.2, Steps 4-6)
        # No gpu_device — contention tracking stays at worker level where
        # it's meaningful (GPU-wide shared memory). Controller only handles
        # batching/priority.
        # 250ms window, max batch 16, no deferred queue-level batching.
        self._batch_scheduler = AdaptiveBatchScheduler(
            process_batch_fn=self._dispatch_batch,
        )

        # Paper Design Principle 1: disable on-demand spawning
        self._disable_spawn_on_demand = False
        self._config_path = os.path.join(os.path.dirname(__file__), "deployment_config.yaml")

        # Statistics (protected by _request_lock)
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._request_history: collections.deque = collections.deque(maxlen=1000)

        # Background threads
        self._running = False
        self._rate_update_thread: Optional[threading.Thread] = None
        self._health_check_thread: Optional[threading.Thread] = None

    # -------------------------------------------------------------------------
    # Request Routing
    # -------------------------------------------------------------------------

    def route_request(
        self,
        function_id: str,
        prompt: str,
        max_tokens: int = 256,
        timeout: float = None
    ) -> Dict[str, Any]:
        """
        Route a single inference request via controller-level batch scheduler.

        Paper Section 4.2: Controller collects requests per function,
        applies fill-or-expire, then dispatches the whole batch to the
        best container via /batch_inference.

        Args:
            function_id: Function to invoke
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            timeout: Request timeout

        Returns:
            Inference result dictionary
        """
        with self._request_lock:
            self._request_counter += 1
            request_id = f"req_{self._request_counter}"

        self.registry.record_request(function_id)
        with self._request_lock:
            self._total_requests += 1

        start_time = time.time()

        func_profile = self.registry.get_function(function_id)
        if not func_profile:
            logger.error(f"Unknown function: {function_id}")
            with self._request_lock:
                self._failed_requests += 1
            return {"success": False, "error": f"Unknown function: {function_id}"}

        # STEP 1: Submit to controller-level batch scheduler
        t_submit = time.time()
        try:
            inf_request = self._batch_scheduler.submit_request(
                function_id, prompt, max_tokens
            )
        except ValueError:
            # Function not registered with batch scheduler
            logger.error(f"Function {function_id} not registered with batch scheduler")
            with self._request_lock:
                self._failed_requests += 1
            return {"success": False, "error": f"Function not registered: {function_id}"}

        logger.info(f"[TRACE] {request_id} STEP1_SUBMITTED fn={function_id} "
                    f"max_tokens={max_tokens} t={t_submit:.3f}")

        # STEP 2+3+4: Block until batch is formed, dispatched, and result returned
        result = inf_request.wait(timeout=timeout)

        if result is None:
            result = {"success": False, "error": "Request timed out waiting for batch"}

        end_time = time.time()
        e2e_ms = (end_time - start_time) * 1000
        batch_wait_ms = (end_time - t_submit) * 1000
        stats = RequestStats(
            request_id=request_id,
            function_id=function_id,
            container_id=result.get("container_id", "unknown"),
            start_time=start_time,
            end_time=end_time,
            success=result.get("success", False),
            ttft_ms=result.get("ttft_ms"),
            e2e_ms=e2e_ms
        )

        with self._request_lock:
            self._request_history.append(stats)
            if stats.success:
                self._successful_requests += 1
            else:
                self._failed_requests += 1

        result["request_id"] = request_id
        result["e2e_ms"] = round(e2e_ms, 2)

        logger.info(f"[TRACE] {request_id} STEP5_COMPLETE fn={function_id} "
                    f"cid={result.get('container_id','?')} "
                    f"e2e={e2e_ms:.0f}ms batch_wait={batch_wait_ms:.0f}ms "
                    f"ttft={result.get('ttft_ms',0):.0f}ms "
                    f"tpot={result.get('tpot_ms',0):.0f}ms "
                    f"tokens={result.get('tokens',0)} "
                    f"batch_size={result.get('batch_size',0)} "
                    f"success={result.get('success')}")

        return result

    def batch_route_requests(
        self,
        function_id: str,
        prompts: List[str],
        max_tokens: int = 256,
        timeout: float = 120.0
    ) -> Dict[str, Any]:
        """
        Route a batch of requests through the controller-level batch scheduler.

        Each prompt is submitted individually; the scheduler forms optimal
        batches via fill-or-expire and dispatches them.
        """
        # Submit all prompts to the batch scheduler
        inf_requests = []
        for prompt in prompts:
            try:
                req = self._batch_scheduler.submit_request(
                    function_id, prompt, max_tokens
                )
                inf_requests.append(req)
            except ValueError:
                inf_requests.append(None)

        # Wait for all results
        results = []
        all_success = True
        for req in inf_requests:
            if req is None:
                results.append({"success": False, "error": "Function not registered"})
                all_success = False
                continue
            result = req.wait(timeout=timeout)
            if result is None:
                result = {"success": False, "error": "Request timed out"}
                all_success = False
            elif not result.get("success", False):
                all_success = False
            results.append(result)

        return {
            "success": all_success,
            "results": results,
        }

    def _send_request_with_retry(
        self,
        container: Container,
        prompt: str,
        max_tokens: int,
        timeout: float,
        stats: RequestStats
    ) -> Dict[str, Any]:
        """Send request with retry logic."""
        last_error = None

        for attempt in range(self.max_retries + 1):
            container.update_activity()
            container.last_request_time = time.time()
            with container._pending_lock:
                container.pending_requests += 1
            try:
                response = requests.post(
                    f"{container.get_url()}/inference",
                    json={"prompt": prompt, "max_tokens": max_tokens, "timeout": timeout},
                    timeout=timeout
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    last_error = f"HTTP {response.status_code}"

            except requests.exceptions.Timeout:
                last_error = "Timeout"
            except Exception as e:
                last_error = str(e)
            finally:
                with container._pending_lock:
                    container.pending_requests = max(0, container.pending_requests - 1)
                    if container.pending_requests == 0:
                        container._last_idle_time = time.time()
                self._notify_lifo_waiter()
                stats.retries = attempt

            if attempt < self.max_retries:
                logger.warning(f"Request retry {attempt + 1}/{self.max_retries}: {last_error}")
                time.sleep(0.1 * (2 ** attempt))  # Exponential backoff

        return {"success": False, "error": last_error}

    # -------------------------------------------------------------------------
    # Batch Dispatch (Paper Section 4.2, Steps 4-6)
    # -------------------------------------------------------------------------

    def _dispatch_batch(self, batch: Batch) -> List[Dict]:
        """
        Dispatch a formed batch to the best container.

        Called by the controller-level AdaptiveBatchScheduler when a batch
        is formed (fill-or-expire). Selects the best container by pre-load
        score and sends via POST /batch_execute for direct GPU execution
        (no worker-side re-batching).
        """
        function_id = batch.function_id

        # STEP 2: Select container. Anchor the queue-timeout to the batch's
        # OLDEST request arrival so total queue wait is bounded at 8s from
        # arrival (matches SwarmLoRA/ServerlessLLM). Under backlog, LIFO fill
        # makes a batch's requests homogeneously old, so min-arrival is a
        # faithful anchor.
        batch_arrival = min(
            (r.arrival_time_wall for r in batch.requests), default=time.time()
        )
        t_select_start = time.time()
        container, queue_wait_ms = self._select_and_claim(
            function_id, batch.size, batch_arrival
        )
        t_select_end = time.time()
        if not container:
            logger.info(f"[TRACE] {batch.batch_id} STEP2_SELECT_FAIL fn={function_id} "
                       f"queue_timeout_ms={queue_wait_ms:.0f}")
            return [{"success": False, "error": f"Queue timeout ({queue_wait_ms:.0f}ms)",
                     "queue_wait_ms": round(queue_wait_ms, 2)}] * batch.size

        # Count how many OTHER batches are inflight on this GPU
        node = self.registry.get_node(container.node_id)
        gpu_inflight = sum(c.pending_requests for c in node.containers.values()) if node else 0

        logger.info(f"[TRACE] {batch.batch_id} STEP2_SELECTED fn={function_id} "
                    f"cid={container.container_id} node={container.node_id} "
                    f"gpu_inflight={gpu_inflight} batch_size={batch.size} "
                    f"select_ms={(t_select_end - t_select_start)*1000:.1f}")

        # STEP 3: POST to worker
        t_post_start = time.time()
        try:
            response = requests.post(
                f"{container.get_url()}/batch_execute",
                json={
                    "prompts": batch.prompts,
                    "max_tokens": batch.max_tokens,
                    "batch_id": batch.batch_id,
                    "function_id": function_id,
                    # Wall-clock arrival times for accurate cross-process TTFT
                    "arrival_times": [r.arrival_time_wall for r in batch.requests],
                    # Wall-clock dispatch time (batch wait + container selection done)
                    "dispatch_time_wall": t_post_start,
                },
                timeout=None,
            )
            t_post_end = time.time()
            worker_ms = (t_post_end - t_post_start) * 1000

            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                for r in results:
                    r["container_id"] = container.container_id
                    r["batch_size"] = batch.size

                logger.info(f"[TRACE] {batch.batch_id} STEP4_WORKER_DONE fn={function_id} "
                           f"cid={container.container_id} worker_ms={worker_ms:.0f} "
                           f"tokens={results[0].get('tokens',0) if results else 0} "
                           f"ttft={results[0].get('ttft_ms',0):.0f}ms "
                           f"gen_ms={results[0].get('gen_time_ms',0):.0f}")
                return results
            else:
                logger.info(f"[TRACE] {batch.batch_id} STEP4_WORKER_FAIL fn={function_id} "
                           f"http={response.status_code} worker_ms={worker_ms:.0f}")
                return [{"success": False, "error": f"HTTP {response.status_code}"}] * batch.size
        except Exception as e:
            t_post_end = time.time()
            worker_ms = (t_post_end - t_post_start) * 1000
            logger.error(f"[TRACE] {batch.batch_id} STEP4_WORKER_ERROR fn={function_id} "
                        f"error={e} worker_ms={worker_ms:.0f}")
            return [{"success": False, "error": str(e)}] * batch.size
        finally:
            with container._pending_lock:
                container.pending_requests = max(0, container.pending_requests - batch.size)
                if container.pending_requests == 0:
                    container._last_idle_time = time.time()
            self._notify_lifo_waiter()

    # -------------------------------------------------------------------------
    # Container Selection
    # -------------------------------------------------------------------------

    def _select_and_claim(self, function_id: str, batch_size: int, arrival_wall=None):
        """
        Atomically select a container and mark it as busy.

        Priority: idle dedicated > inline swap > on-demand spawn (async).
        The 8s deadline is measured from the request's ARRIVAL (arrival_wall),
        not from when this method is entered -- otherwise a request that
        already sat in the batch-formation queue for minutes (LIFO leaves old
        requests at the back) would get a fresh 8s here and count as a
        "successful" multi-minute request, unlike SwarmLoRA/ServerlessLLM
        which cap total queue wait at 8s from arrival. Anchoring to arrival
        bounds total queue wait at 8s, matching the other two systems.
        Requests already past their deadline fail on the first loop iteration.
        Falls back to "now" if no arrival time is supplied.

        On-demand spawn is fired asynchronously — the spawned container
        joins the pool and notifies LIFO waiters when ready, so it's
        never wasted even if the triggering request times out.

        Returns (container, queue_wait_ms) — queue_wait_ms is 0 if no waiting.
        Returns (None, queue_wait_ms) on timeout.
        """
        QUEUE_TIMEOUT_S = 8.0
        start_time = time.time()
        # Deadline is anchored to ARRIVAL (not now) so total queue wait is
        # bounded at 8s from arrival; queue_wait_ms below is left measured from
        # method entry to keep the reported metric unchanged (only the timeout
        # behaviour changes).
        deadline = (arrival_wall if arrival_wall is not None else start_time) + QUEUE_TIMEOUT_S
        spawn_triggered = False

        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                queue_wait_ms = (time.time() - start_time) * 1000
                logger.warning(f"[TIMEOUT] {function_id}: queue timeout after {queue_wait_ms:.0f}ms")
                return None, queue_wait_ms

            # Fast path: try to find an idle dedicated container
            with self._selection_lock:
                container = self._select_idle_dedicated(function_id)
                if container:
                    with container._pending_lock:
                        container.pending_requests += batch_size
                    container.update_activity()
                    container.last_request_time = time.time()
                    queue_wait_ms = (time.time() - start_time) * 1000
                    if queue_wait_ms > 100:
                        logger.info(f"[WAIT_DONE] {function_id} got {container.container_id} "
                                    f"queue_wait={queue_wait_ms:.0f}ms")
                    return container, queue_wait_ms

            # Slow path: inline swap (HTTP call, ~100ms)
            swapped = self._try_inline_swap(function_id)
            if swapped:
                with swapped._pending_lock:
                    swapped.pending_requests += batch_size
                swapped.update_activity()
                swapped.last_request_time = time.time()
                queue_wait_ms = (time.time() - start_time) * 1000
                return swapped, queue_wait_ms

            # Async on-demand spawn — fire once, don't block.
            # Spawned container joins pool and notifies LIFO waiters.
            if not spawn_triggered:
                spawn_triggered = True
                threading.Thread(
                    target=self._async_spawn_on_demand,
                    args=(function_id,),
                    daemon=True
                ).start()

            logger.debug(f"[WAIT] {function_id}: LIFO queue, {remaining:.1f}s remaining")

            my_event = threading.Event()
            with self._waiter_lock:
                self._waiter_stack.append(my_event)

            my_event.wait(timeout=min(remaining, 0.5))

            with self._waiter_lock:
                try:
                    self._waiter_stack.remove(my_event)
                except ValueError:
                    pass  # Already removed by notifier

    def _async_spawn_on_demand(self, function_id: str):
        """Spawn a container asynchronously and notify LIFO waiters when ready."""
        container = self._spawn_on_demand(function_id)
        if container:
            self._notify_lifo_waiter()

    def _notify_lifo_waiter(self):
        """Notify the most recent waiter (LIFO) that a container is available."""
        with self._waiter_lock:
            if self._waiter_stack:
                event = self._waiter_stack.pop()  # LIFO — newest first
                event.set()

    def _select_idle_dedicated(self, function_id: str) -> Optional[Container]:
        """
        Select an idle dedicated container (fast path, called under lock).
        """
        dedicated = self.registry.get_containers_for_function(function_id)
        healthy = [c for c in dedicated
                   if c.status == "ready" or c.pending_requests > 0]
        idle = [c for c in healthy if c.pending_requests == 0]
        if idle:
            return self._pick_best(idle, function_id)
        return None

    def select_container(self, function_id: str) -> Optional[Container]:
        """
        Select the best container for a function by pre-load score.

        Paper Section 3.3: Instance selection by pre-load score.
        Never dispatch to a container that is already generating — use
        inline swap to claim an idle container from another function instead.
        """
        dedicated = self.registry.get_containers_for_function(function_id)
        # Accept both "ready" and containers with pending work (busy but alive)
        healthy = [c for c in dedicated
                   if c.status == "ready" or c.pending_requests > 0]

        # Prefer idle dedicated containers (not generating)
        idle = [c for c in healthy if c.pending_requests == 0]
        if idle:
            return self._pick_best(idle, function_id)

        # All dedicated containers are busy — swap an idle one in
        swapped = self._try_inline_swap(function_id)
        if swapped:
            return swapped

        # Last resort: spawn on-demand
        spawned = self._spawn_on_demand(function_id)
        if spawned:
            return spawned

        return None

    def _pick_best(self, containers: List[Container],
                   function_id: str) -> Container:
        """
        Pick the best container, preferring the least-loaded GPU.

        Computes pending requests per GPU across ALL containers, then
        picks the container on the least-busy GPU (breaking ties by
        preload score).
        """
        func_profile = self.registry.get_function(function_id)
        required = func_profile.required_artifacts if func_profile else set()

        gpu_pending: Dict[str, int] = {}
        for node in self.registry.get_healthy_nodes():
            total = sum(c.pending_requests for c in node.containers.values())
            gpu_pending[node.node_id] = total

        # Sort by: (gpu_pending ASC, preload_score DESC)
        return min(containers,
                   key=lambda c: (
                       gpu_pending.get(c.node_id, 0),
                       -self.compute_preload_score(c, required)
                   ))

    def _try_inline_swap(self, function_id: str) -> Optional[Container]:
        """
        Swap a truly idle container to serve this function.

        Idle = pending_requests == 0 AND (never received a request OR
        idle for >5 seconds). Filters candidates first, then tries
        the single best one. Concurrent swaps are allowed on different
        containers — only the per-container claim is serialized.
        """
        func = self.registry.get_function(function_id)
        if not func or not func.adapter_id:
            return None

        all_containers = self.registry.get_all_ready_containers()
        now = time.time()

        # Count containers per function to protect the last one
        fn_counts: Dict[str, int] = {}
        for c in all_containers:
            fn_counts[c.function_id] = fn_counts.get(c.function_id, 0) + 1

        # Filter truly idle containers (skip ones already being swapped)
        with self._swapping_lock:
            swapping = set(self._swapping_containers)

        idle = [c for c in all_containers
                if c.container_id not in swapping
                and c.pending_requests == 0
                and c.function_id != function_id
                and c.status == "ready"
                and fn_counts.get(c.function_id, 0) > 1]

        if not idle:
            # Debug: log why no candidates found
            all_idle_pending0 = [c for c in all_containers
                                 if c.pending_requests == 0
                                 and c.function_id != function_id
                                 and c.status == "ready"]
            blocked_by_swap = [c for c in all_idle_pending0 if c.container_id in swapping]
            blocked_by_count = [c for c in all_idle_pending0 if fn_counts.get(c.function_id, 0) <= 1]
            blocked_by_time = [c for c in all_idle_pending0
                               if c.container_id not in swapping
                               and fn_counts.get(c.function_id, 0) > 1
                               and not ((now - c.last_request_time) > 5.0
                                        or (now - getattr(c, '_last_idle_time', 0)) > 5.0)]
            logger.debug(f"Inline swap failed for {function_id}: "
                        f"total_ready={len(all_containers)}, "
                        f"idle_pending0={len(all_idle_pending0)}, "
                        f"blocked_by_swap={len(blocked_by_swap)}, "
                        f"blocked_by_fn_count={len(blocked_by_count)}, "
                        f"blocked_by_5s_timer={len(blocked_by_time)}")
            return None

        # Pick the best candidate: prefer lowest request rate function
        # Prefer: lowest-rate function, then from GPU with most same-function
        # containers (spreads swaps across GPUs evenly)
        gpu_fn_counts: Dict[str, int] = {}
        for c in all_containers:
            key = f"{c.node_id}:{c.function_id}"
            gpu_fn_counts[key] = gpu_fn_counts.get(key, 0) + 1
        idle.sort(key=lambda c: (
            self.registry.get_request_rate(c.function_id) if c.function_id else 0.0,
            -gpu_fn_counts.get(f"{c.node_id}:{c.function_id}", 0),
        ))
        candidate = idle[0]

        with self._swapping_lock:
            if candidate.container_id in self._swapping_containers:
                return None  # another thread claimed it
            self._swapping_containers.add(candidate.container_id)

        try:
            resp = requests.post(
                f"{candidate.get_url()}/swap_adapter",
                json={
                    "adapter_id": func.adapter_id,
                    "function_id": function_id,
                },
                timeout=2.0
            )
            if resp.status_code == 200 and resp.json().get("success"):
                old_artifact = (
                    f"adapter_{candidate.lora_id.replace('/', '_')}"
                    if candidate.lora_id else None
                )
                new_artifact = f"adapter_{func.adapter_id.replace('/', '_')}"
                if old_artifact:
                    candidate.loaded_artifacts.discard(old_artifact)
                    candidate.gpu_loaded_artifacts.discard(old_artifact)
                candidate.loaded_artifacts.add(new_artifact)
                candidate.gpu_loaded_artifacts.add(new_artifact)
                self.registry.rebind_container(
                    candidate.container_id, function_id, func.adapter_id)
                candidate._last_swap_time = time.time()
                swap_ms = resp.json().get("swap_ms", 0)
                logger.info(f"Inline swap: {candidate.container_id} "
                           f"-> {function_id} ({swap_ms:.0f}ms)")
                return candidate
        except Exception:
            pass
        finally:
            with self._swapping_lock:
                self._swapping_containers.discard(candidate.container_id)

        return None

    def _spawn_on_demand(self, function_id: str) -> Optional[Container]:
        """
        Spawn a new container for this function on-demand.

        Last resort when no free or idle containers exist. Blocks until
        the new container is ready (~30-60s) but avoids unbounded queueing.
        Capped at max_workers_per_gpu to prevent runaway GPU contention.

        Paper Design Principle 1: "LLM artifacts are only pre-loaded in
        existing idle container and GPU instances. We never proactively
        create new instances for pre-loading."
        """
        # Disabled: Paper Design Principle 1 — never create new instances.
        # All container placement handled by PCKP + inline swap.
        if self._disable_spawn_on_demand:
            return None

        func = self.registry.get_function(function_id)
        if not func:
            return None

        nodes = self.registry.get_healthy_nodes()
        if not nodes:
            return None

        # Reserve a spawn slot atomically, then spawn outside the lock.
        # Limit concurrent spawns per node to avoid flooding the preload agent.
        with self._spawn_lock:
            # Sort nodes by load (fewest containers + pending spawns first)
            sorted_nodes = sorted(nodes, key=lambda n: len(n.containers) + self._pending_spawns.get(n.node_id, 0))

            # Try each node until one has room under the cap
            node = None
            for candidate in sorted_nodes:
                total = len(candidate.containers) + self._pending_spawns.get(candidate.node_id, 0)
                if total < self._max_workers_per_gpu:
                    node = candidate
                    break

            if node is None:
                return None

            # Reserve the slot
            self._pending_spawns[node.node_id] = self._pending_spawns.get(node.node_id, 0) + 1

        logger.info(f"On-demand spawn for {function_id} on {node.node_id}")
        try:
            response = requests.post(
                f"{node.get_agent_url()}/spawn",
                json={"function_id": function_id,
                      "lora_id": func.adapter_id},
                timeout=120.0
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    container = Container(
                        container_id=data["container_id"],
                        function_id=function_id,
                        node_id=node.node_id,
                        http_port=data["http_port"],
                        http_host=node.hostname,
                        status="ready",
                        lora_id=func.adapter_id,
                    )
                    self.registry.register_container(container)
                    logger.info(f"On-demand spawned {data['container_id']} "
                               f"for {function_id}")
                    return container
        except Exception as e:
            logger.error(f"On-demand spawn failed for {function_id}: {e}")
        finally:
            with self._spawn_lock:
                self._pending_spawns[node.node_id] = self._pending_spawns.get(node.node_id, 0) - 1

        return None

    def compute_preload_score(
        self,
        container: Container,
        required_artifacts: Optional[Set[str]] = None
    ) -> float:
        """
        Compute pre-load score for a container.

        Paper: "Score components:
        - +10 for each required artifact loaded
        - +20 bonus for GPU-loaded artifacts
        - -5 penalty per pending request"
        """
        if required_artifacts is None:
            func = self.registry.get_function(container.function_id)
            required_artifacts = func.required_artifacts if func else set()

        return container.compute_preload_score(required_artifacts)

    # -------------------------------------------------------------------------
    # Function Management
    # -------------------------------------------------------------------------

    def register_function(
        self,
        function_id: str,
        backbone_id: str,
        adapter_id: Optional[str] = None,
        slo_ms: float = DEFAULT_SLO_MS,
        base_ttft_ms: float = DEFAULT_BASE_TTFT_MS,
        marginal_cost_ms: float = DEFAULT_MARGINAL_COST_MS
    ) -> FunctionProfile:
        """Register a new function."""
        logger.info(f"Registering function {function_id}")

        profile = FunctionProfile(
            function_id=function_id,
            backbone_id=backbone_id,
            adapter_id=adapter_id,
            base_ttft_ms=base_ttft_ms,
            marginal_cost_ms=marginal_cost_ms,
            slo_ms=slo_ms
        )

        profile.required_artifacts.add(f"backbone_{backbone_id}")
        if adapter_id:
            artifact_id = f"adapter_{adapter_id.replace('/', '_')}"
            profile.required_artifacts.add(artifact_id)

            # Create adapter artifact if not exists
            if not self.registry.get_artifact(artifact_id):
                artifact = create_adapter_artifact(
                    adapter_id, f"backbone_{backbone_id}"
                )
                self.registry.register_artifact(artifact)

        self.registry.register_function(profile)

        batch_cfg = BatchConfig(
            function_id=function_id,
            base_ttft_ms=base_ttft_ms,
            marginal_cost_ms=marginal_cost_ms,
            slo_ms=slo_ms,
        )
        self._batch_scheduler.register_function(batch_cfg)

        return profile

    def get_functions(self) -> List[Dict[str, Any]]:
        """Get all registered functions."""
        functions = self.registry.get_all_functions()
        return [
            {
                "function_id": f.function_id,
                "backbone_id": f.backbone_id,
                "adapter_id": f.adapter_id,
                "slo_ms": f.slo_ms,
                "request_rate": self.registry.get_request_rate(f.function_id),
                "max_batch_size": f.compute_max_batch_size()
            }
            for f in functions
        ]

    # -------------------------------------------------------------------------
    # Node Management
    # -------------------------------------------------------------------------

    def register_node(
        self,
        node_id: str,
        hostname: str,
        agent_port: int,
        container_memory_mb: float = 32768.0,
        gpu_memory_mb: float = 44400.0,
        gpu_device: Optional[str] = None
    ) -> WorkerNode:
        """Register a new worker node.

        Args:
            gpu_device: GPU device string (e.g. "cuda:0"). When provided and
                GPU offloading is enabled, a DynamicGPUOffloader is created for
                this GPU if one doesn't already exist.
        """
        logger.info(f"Registering node {node_id} at {hostname}:{agent_port}"
                     + (f" gpu={gpu_device}" if gpu_device else ""))

        gpu_devices = []
        if gpu_device:
            idx = int(gpu_device.split(":")[-1]) if ":" in gpu_device else 0
            gpu_devices = [idx]

        node = WorkerNode(
            node_id=node_id,
            hostname=hostname,
            agent_port=agent_port,
            container_memory_mb=container_memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            gpu_devices=gpu_devices if gpu_devices else [0]
        )

        self.registry.register_node(node)

        if self.enable_gpu_offloader and gpu_device:
            if gpu_device not in self.gpu_offloaders:
                offloader = DynamicGPUOffloader(
                    registry=self.registry, device=gpu_device
                )
                offloader.set_offload_callback(self._create_offload_callback())
                if self._running:
                    offloader.start()
                self.gpu_offloaders[gpu_device] = offloader
                logger.info(f"Created GPU offloader for {gpu_device}")

        return node

    def get_nodes(self) -> List[Dict[str, Any]]:
        """Get all registered nodes."""
        nodes = self.registry.get_all_nodes()
        return [
            {
                "node_id": n.node_id,
                "hostname": n.hostname,
                "agent_port": n.agent_port,
                "status": n.status,
                "healthy": n.is_healthy(),
                "containers": len(n.containers),
                "gpu_memory_used_mb": n.gpu_memory_used_mb,
                "container_memory_used_mb": n.container_memory_used_mb
            }
            for n in nodes
        ]

    # -------------------------------------------------------------------------
    # Scaling
    # -------------------------------------------------------------------------

    def scale_function(
        self,
        function_id: str,
        target_count: int,
        node_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Scale a function to target container count.

        This requests the PreloadAgent to spawn/terminate containers.
        """
        logger.info(f"Scaling {function_id} to {target_count} containers")

        func = self.registry.get_function(function_id)
        if not func:
            return {"success": False, "error": f"Unknown function: {function_id}"}

        current_containers = self.registry.get_containers_for_function(function_id)
        current_count = len(current_containers)

        if target_count == current_count:
            return {"success": True, "action": "none", "count": current_count}

        if target_count > current_count:
            # Scale up
            to_spawn = target_count - current_count
            spawned = self._spawn_containers(function_id, func.adapter_id, to_spawn, node_id)
            return {"success": True, "action": "scale_up", "spawned": spawned}
        else:
            # Scale down
            to_terminate = current_count - target_count
            terminated = self._terminate_containers(current_containers[:to_terminate])
            return {"success": True, "action": "scale_down", "terminated": terminated}

    def _spawn_containers(
        self,
        function_id: str,
        lora_id: str,
        count: int,
        node_id: Optional[str] = None
    ) -> int:
        """Request PreloadAgents to spawn containers (in parallel)."""
        nodes = self.registry.get_healthy_nodes()
        if node_id:
            nodes = [n for n in nodes if n.node_id == node_id]

        if not nodes:
            logger.error("No healthy nodes for spawning")
            return 0

        def _spawn_one(node, fn_id, adapter_id):
            try:
                response = requests.post(
                    f"{node.get_agent_url()}/spawn",
                    json={"function_id": fn_id, "lora_id": adapter_id},
                    timeout=600.0
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        container = Container(
                            container_id=data["container_id"],
                            function_id=fn_id,
                            node_id=node.node_id,
                            http_port=data["http_port"],
                            http_host=node.hostname,
                            status="ready",
                            lora_id=adapter_id,
                        )
                        self.registry.register_container(container)
                        logger.info(f"Spawned {data['container_id']} on {node.node_id}")
                        return True
            except Exception as e:
                logger.error(f"Failed to spawn on {node.node_id}: {e}")
            return False

        # Spawn across GPUs with bounded concurrency.
        # Agent-side semaphore (4 concurrent) throttles per-GPU, but we also
        # limit client-side threads to avoid HTTP timeout on queued requests.
        max_concurrent = len(nodes) * 8  # 8 in-flight per GPU; agent semaphore is the real throttle
        futures = []
        spawned = 0
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            for i in range(count):
                node = nodes[self._spawn_node_index % len(nodes)]
                self._spawn_node_index += 1
                futures.append(executor.submit(_spawn_one, node, function_id, lora_id))
            spawned = sum(1 for f in as_completed(futures) if f.result())
        return spawned

    def _terminate_containers(self, containers: List[Container]) -> int:
        """Request PreloadAgents to terminate containers."""
        terminated = 0
        for container in containers:
            node = self.registry.get_node(container.node_id)
            if not node:
                continue
            try:
                response = requests.post(
                    f"{node.get_agent_url()}/terminate",
                    json={"container_id": container.container_id},
                    timeout=30.0
                )
                if response.status_code == 200:
                    terminated += 1
            except Exception as e:
                logger.error(f"Failed to terminate {container.container_id}: {e}")

        return terminated

    # -------------------------------------------------------------------------
    # Background Loops
    # -------------------------------------------------------------------------

    def _rate_update_loop(self):
        """Update request rates periodically."""
        while self._running:
            try:
                for func in self.registry.get_all_functions():
                    rate = self.registry.get_request_rate(func.function_id)
                    logger.debug(f"Function {func.function_id} rate: {rate:.2f} req/s")
            except Exception as e:
                logger.error(f"Rate update error: {e}")

            time.sleep(self.rate_update_interval_s)

    def _health_check_loop(self):
        """Check container health and refresh node heartbeats."""
        while self._running:
            try:
                for node in self.registry.get_all_nodes():
                    any_healthy = False
                    for container in list(node.containers.values()):
                        self._check_container_health(container)
                        if container.status == "ready":
                            any_healthy = True
                    # Refresh node heartbeat so preload scheduler sees healthy nodes
                    if any_healthy:
                        node.update_heartbeat()
            except Exception as e:
                logger.error(f"Health check error: {e}")

            time.sleep(self.health_check_interval_s)

    def _check_container_health(self, container: Container):
        """Check health of a single container.

        When a worker is busy generating (model.generate() holds the GIL),
        Flask cannot serve /health within the timeout.  This is normal —
        don't mark the container unhealthy if it has pending requests
        (i.e. it's actively processing a batch).  Require 3 consecutive
        failures before marking a truly idle container unhealthy.
        """
        try:
            response = requests.get(
                f"{container.get_url()}/health",
                timeout=15.0
            )
            if response.status_code == 200:
                data = response.json()
                container.status = data.get("status", "ready")
                container._health_fail_count = 0
            else:
                container._health_fail_count = getattr(container, '_health_fail_count', 0) + 1
                if container.pending_requests == 0 and container._health_fail_count >= 6:
                    container.status = "unhealthy"
        except Exception:
            container._health_fail_count = getattr(container, '_health_fail_count', 0) + 1
            # Don't mark unhealthy if the worker is busy generating —
            # the GIL prevents Flask from responding during model.generate()
            if container.pending_requests == 0 and container._health_fail_count >= 6:
                container.status = "unhealthy"

    # -------------------------------------------------------------------------
    # Offload Callback
    # -------------------------------------------------------------------------

    def _create_offload_callback(self):
        """Create callback for GPU offloader to call worker /offload_adapter."""
        def offload_fn(artifact_id: str, container_id: str) -> bool:
            container = self.registry.get_container(container_id)
            if not container:
                logger.warning(f"Offload: container {container_id} not found")
                return False
            try:
                resp = requests.post(
                    f"{container.get_url()}/offload_adapter",
                    json={"artifact_id": artifact_id, "keep_in_container": True},
                    timeout=30.0
                )
                return resp.status_code == 200 and resp.json().get("success", False)
            except Exception as e:
                logger.error(f"Offload callback failed for {container_id}: {e}")
                return False
        return offload_fn

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def start(self):
        """Start controller background threads."""
        if self._running:
            return

        self._running = True

        # Initialize default artifacts
        for artifact in create_default_artifacts(MODEL_NAME):
            self.registry.register_artifact(artifact)

        # Start rate update thread
        self._rate_update_thread = threading.Thread(
            target=self._rate_update_loop,
            daemon=True,
            name="RateUpdateLoop"
        )
        self._rate_update_thread.start()

        # Start health check thread
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
            name="HealthCheckLoop"
        )
        self._health_check_thread.start()

        self._batch_scheduler.start()

        # Start optional components
        if self.preload_scheduler:
            self.preload_scheduler.start()
        # Start all per-GPU offloaders
        for device, offloader in self.gpu_offloaders.items():
            offloader.set_offload_callback(self._create_offload_callback())
            offloader.start()
            logger.info(f"Started GPU offloader for {device}")

        logger.info("Controller started")

    def stop(self):
        """Stop controller."""
        self._running = False

        if self._rate_update_thread:
            self._rate_update_thread.join(timeout=5.0)
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5.0)

        self._batch_scheduler.stop()

        if self.preload_scheduler:
            self.preload_scheduler.stop()
        for device, offloader in self.gpu_offloaders.items():
            offloader.stop()
            logger.info(f"Stopped GPU offloader for {device}")

        logger.info("Controller stopped")

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get controller statistics."""
        with self._request_lock:
            total = self._total_requests
            success_count = self._successful_requests
            fail_count = self._failed_requests
            recent_requests = list(self._request_history)[-100:]

        successful = [r for r in recent_requests if r.success]

        avg_e2e = 0.0
        avg_ttft = 0.0
        if successful:
            avg_e2e = sum(r.e2e_ms for r in successful if r.e2e_ms) / len(successful)
            ttfts = [r.ttft_ms for r in successful if r.ttft_ms]
            avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0

        return {
            "total_requests": total,
            "successful_requests": success_count,
            "failed_requests": fail_count,
            "success_rate": success_count / total if total > 0 else 0,
            "avg_e2e_ms": avg_e2e,
            "avg_ttft_ms": avg_ttft,
            "registry": self.registry.get_statistics(),
            "batch_scheduler": self._batch_scheduler.get_stats(),
            "preload_scheduler": self.preload_scheduler.get_statistics() if self.preload_scheduler else None,
            "gpu_offloaders": {
                device: offloader.get_statistics()
                for device, offloader in self.gpu_offloaders.items()
            } if self.gpu_offloaders else None,
        }

    # -------------------------------------------------------------------------
    # HTTP API
    # -------------------------------------------------------------------------

    def run_server(self, port: int = CONTROLLER_PORT):
        """Run as HTTP server."""
        try:
            from flask import Flask, request, jsonify
        except ImportError:
            logger.error("Flask not installed. Run: pip install flask")
            return

        app = Flask("serverless_lora_controller")

        @app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                "status": "running" if self._running else "stopped",
                "total_requests": self._total_requests
            })

        @app.route('/inference', methods=['POST'])
        def inference():
            """Main inference endpoint."""
            data = request.get_json() or {}
            function_id = data.get("function_id", "default")
            prompt = data.get("prompt", "")
            max_tokens = data.get("max_tokens", 256)
            timeout = data.get("timeout", None)

            if not prompt:
                return jsonify({"success": False, "error": "No prompt"}), 400

            # Paper Section 4.2: Controller batches requests via fill-or-expire,
            # then dispatches formed batches to workers via /batch_inference.
            result = self.route_request(
                function_id, prompt, max_tokens, timeout
            )
            return jsonify(result), 200 if result.get("success") else 500

        @app.route('/batch_inference', methods=['POST'])
        def batch_inference():
            """Batch inference endpoint."""
            data = request.get_json() or {}
            function_id = data.get("function_id", "default")
            prompts = data.get("prompts", [])
            max_tokens = data.get("max_tokens", 256)

            if not prompts:
                return jsonify({"success": False, "error": "No prompts"}), 400

            result = self.batch_route_requests(function_id, prompts, max_tokens)
            return jsonify(result)

        @app.route('/register_function', methods=['POST'])
        def register_function():
            data = request.get_json() or {}
            function_id = data.get("function_id")
            backbone_id = data.get("backbone_id", MODEL_NAME)
            adapter_id = data.get("adapter_id", LORA_ADAPTER_ID)
            slo_ms = data.get("slo_ms", DEFAULT_SLO_MS)

            if not function_id:
                return jsonify({"error": "function_id required"}), 400

            profile = self.register_function(
                function_id, backbone_id, adapter_id, slo_ms
            )
            return jsonify({
                "success": True,
                "function_id": profile.function_id
            })

        @app.route('/register_node', methods=['POST'])
        def register_node():
            data = request.get_json() or {}
            node_id = data.get("node_id")
            hostname = data.get("hostname", "localhost")
            agent_port = data.get("agent_port", 7000)
            gpu_memory_mb = data.get("gpu_memory_mb", 44400.0)
            container_memory_mb = data.get("container_memory_mb", 32768.0)
            gpu_device = data.get("gpu_device")

            if not node_id:
                return jsonify({"error": "node_id required"}), 400

            node = self.register_node(
                node_id, hostname, agent_port,
                container_memory_mb=container_memory_mb,
                gpu_memory_mb=gpu_memory_mb,
                gpu_device=gpu_device)
            return jsonify({
                "success": True,
                "node_id": node.node_id
            })

        @app.route('/scale', methods=['POST'])
        def scale():
            data = request.get_json() or {}
            function_id = data.get("function_id")
            target_count = data.get("target_count", 1)
            node_id = data.get("node_id")

            if not function_id:
                return jsonify({"error": "function_id required"}), 400

            result = self.scale_function(function_id, target_count, node_id)
            return jsonify(result)

        @app.route('/status', methods=['GET'])
        def status():
            return jsonify(self.get_statistics())

        @app.route('/functions', methods=['GET'])
        def functions():
            return jsonify({"functions": self.get_functions()})

        @app.route('/nodes', methods=['GET'])
        def nodes():
            return jsonify({"nodes": self.get_nodes()})

        @app.route('/containers', methods=['GET'])
        def containers():
            function_id = request.args.get('function_id')
            if function_id:
                containers = self.registry.get_containers_for_function(function_id)
            else:
                containers = []
                for node in self.registry.get_all_nodes():
                    containers.extend(node.containers.values())

            return jsonify({
                "containers": [
                    {
                        "container_id": c.container_id,
                        "function_id": c.function_id,
                        "node_id": c.node_id,
                        "http_port": c.http_port,
                        "status": c.status,
                        "pending_requests": c.pending_requests
                    }
                    for c in containers
                ]
            })

        @app.route('/reset_allocation', methods=['POST'])
        def reset_allocation():
            """Kill all containers and respawn from deployment_config.yaml.

            Clean restart guarantees correct allocation regardless of
            drift from inline swaps during the previous trace.
            POST body (optional): {"timeout": 300}
            """
            import yaml
            timeout = request.json.get("timeout", 300) if request.json else 300

            cfg_path = self._config_path
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            cfg_funcs = cfg.get("functions", [])
            default_cpf = cfg.get("defaults", {}).get("containers_per_function", 1)

            # Phase 1: Kill all existing containers (parallel)
            all_containers = []
            for node in self.registry.get_all_nodes():
                all_containers.extend(list(node.containers.values()))

            logger.info(f"[reset] Terminating {len(all_containers)} containers...")
            terminated = 0
            def _terminate_one(container):
                node = self.registry.get_node(container.node_id)
                if not node:
                    return False
                try:
                    resp = requests.post(
                        f"{node.get_agent_url()}/terminate",
                        json={"container_id": container.container_id},
                        timeout=30.0
                    )
                    return resp.status_code == 200
                except Exception:
                    return False
            with ThreadPoolExecutor(max_workers=32) as executor:
                results = list(executor.map(_terminate_one, all_containers))
            terminated = sum(1 for r in results if r)
            for c in all_containers:
                try:
                    node = self.registry.get_node(c.node_id)
                    if node and c.container_id in node.containers:
                        del node.containers[c.container_id]
                except Exception:
                    pass
            logger.info(f"[reset] Terminated {terminated}/{len(all_containers)}")

            # Phase 2: Respawn from config (parallel across functions)
            target = {}
            spawn_results = {}
            spawn_tasks = []
            for fn in cfg_funcs:
                fid = fn["function_id"]
                count = fn.get("containers", default_cpf)
                target[fid] = count
                func = self.registry.get_function(fid)
                if not func:
                    logger.error(f"[reset] Unknown function {fid}, skipping")
                    spawn_results[fid] = 0
                    continue
                spawn_tasks.append((fid, func.adapter_id, count))

            def _spawn_func(args):
                fid, adapter_id, count = args
                return fid, self._spawn_containers(fid, adapter_id, count)

            with ThreadPoolExecutor(max_workers=len(spawn_tasks) or 1) as executor:
                for fid, spawned in executor.map(_spawn_func, spawn_tasks):
                    spawn_results[fid] = spawned

            spawned_total = sum(spawn_results.values())
            failed_total = sum(target[fid] - spawn_results.get(fid, 0)
                               for fid in target)

            # Phase 3: Wait for all containers to be ready
            expected = sum(target.values())
            deadline = time.time() + timeout
            ready = 0
            while time.time() < deadline:
                ready = 0
                for node in self.registry.get_all_nodes():
                    for c in node.containers.values():
                        if c.status == "ready":
                            ready += 1
                if ready >= expected:
                    break
                time.sleep(2)

            final = {}
            for node in self.registry.get_all_nodes():
                for c in node.containers.values():
                    final[c.function_id] = final.get(c.function_id, 0) + 1

            logger.info(f"[reset] Done: {ready}/{expected} ready, "
                        f"spawned={spawned_total}, failed={failed_total}")

            return jsonify({
                "success": ready >= expected,
                "terminated": terminated,
                "spawned": spawned_total,
                "failed": failed_total,
                "ready": ready,
                "expected": expected,
                "allocation": {fid: {"target": target.get(fid, 0),
                                     "actual": final.get(fid, 0)}
                               for fid in sorted(target.keys())}
            })

        @app.route('/update_profile', methods=['POST'])
        def update_profile():
            """Update batch scheduling parameters from profiler results.

            Expects JSON:
                function_id: str
                base_ttft_ms: float  (T_0)
                marginal_cost_ms: float  (alpha)
                slo_ms: float (optional)
            """
            data = request.get_json() or {}
            function_id = data.get("function_id")
            if not function_id:
                return jsonify({"success": False, "error": "function_id required"}), 400

            base_ttft = data.get("base_ttft_ms")
            marginal_cost = data.get("marginal_cost_ms")
            slo = data.get("slo_ms")

            if base_ttft is None or marginal_cost is None:
                return jsonify({"success": False,
                                "error": "base_ttft_ms and marginal_cost_ms required"}), 400

            self.registry.update_function_profile(
                function_id,
                base_ttft_ms=base_ttft,
                marginal_cost_ms=marginal_cost,
                slo_ms=slo,
            )

            # Keep batch scheduler in sync with updated profile
            self._batch_scheduler.update_config_from_profile(
                function_id,
                base_ttft_ms=base_ttft,
                marginal_cost_ms=marginal_cost,
                slo_ms=slo
            )

            logger.info(f"Updated profile for {function_id}: "
                        f"T_0={base_ttft:.1f}ms, α={marginal_cost:.1f}ms"
                        + (f", SLO={slo:.1f}ms" if slo else ""))

            return jsonify({"success": True, "function_id": function_id,
                            "base_ttft_ms": base_ttft,
                            "marginal_cost_ms": marginal_cost,
                            "slo_ms": slo})

        @app.route('/decisions/<node_id>', methods=['GET'])
        def get_decisions(node_id):
            """Get pending preload decisions for a node (polled by agents)."""
            if not self.preload_scheduler:
                return jsonify({"decisions": []})
            decisions = self.preload_scheduler.get_pending_decisions(node_id)
            return jsonify({
                "decisions": [
                    {
                        "decision_id": d.decision_id,
                        "artifact_id": d.artifact_id,
                        "target_location": d.target_location.value if hasattr(d.target_location, 'value') else str(d.target_location),
                        "target_node_id": d.target_node_id,
                        "target_container_id": d.target_container_id,
                        "function_id": d.function_id,
                        "expected_value": d.expected_value,
                    }
                    for d in decisions
                ]
            })

        @app.route('/decisions/<decision_id>/complete', methods=['POST'])
        def complete_decision(decision_id):
            """Mark a preload decision as completed (called by agents)."""
            if not self.preload_scheduler:
                return jsonify({"success": False})
            data = request.get_json() or {}
            success = self.preload_scheduler.mark_decision_completed(
                decision_id, data.get("success", True)
            )
            return jsonify({"success": success})

        @app.route('/shutdown', methods=['POST'])
        def shutdown():
            self.stop()
            threading.Thread(target=lambda: (time.sleep(0.5), os._exit(0)),
                           daemon=True).start()
            return jsonify({"status": "shutting_down"})

        # Start controller
        self.start()

        logger.info(f"Controller HTTP server on port {port}")
        app.run(host='0.0.0.0', port=port, threaded=True, use_reloader=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ServerlessLoRA Controller")
    parser.add_argument("--port", type=int, default=CONTROLLER_PORT,
                        help="HTTP server port")
    parser.add_argument("--enable-preload", action="store_true",
                        help="Enable pre-loading scheduler")
    parser.add_argument("--enable-offload", action="store_true",
                        help="Enable GPU offloader")
    parser.add_argument("--max-workers-per-gpu", type=int, default=20,
                        help="Max containers per GPU (caps on-demand spawning)")
    parser.add_argument("--disable-spawn-on-demand", action="store_true",
                        help="Disable on-demand spawning (Paper Design Principle 1)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to cluster config YAML (used by /reset_allocation)")
    args = parser.parse_args()

    controller = Controller(
        enable_preload_scheduler=args.enable_preload,
        enable_gpu_offloader=args.enable_offload,
    )
    controller._max_workers_per_gpu = args.max_workers_per_gpu
    controller._disable_spawn_on_demand = args.disable_spawn_on_demand
    if args.config:
        controller._config_path = os.path.abspath(args.config)

    try:
        controller.run_server(port=args.port)
    except KeyboardInterrupt:
        controller.stop()


if __name__ == "__main__":
    main()
