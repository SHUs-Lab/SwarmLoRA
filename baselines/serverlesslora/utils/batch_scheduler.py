#!/usr/bin/env python3
"""
Adaptive Batching Scheduler for ServerlessLoRA.

Implements Section 4.2 of the paper:
- Fill-or-expire batching mechanism
- SLO-aware batch sizing
- Contention-aware scheduling across multiple functions

Key formulas from paper:
- TTFT with batch size: T_i(b) = T_0,i + α_i * (b - 1)
- Batch delay: d_i = SLO_i - T_i(N_i)
- Contention effect: T_eff_i(b) = M * T_i(b)
- Deadline margin: Δ_i = SLO_i - (w_i + M * T_i(b))
"""

import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import logging

from utils.gpu_contention import GPUContentionTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """A single inference request."""
    request_id: str
    prompt: str
    max_tokens: int
    arrival_time: float = field(default_factory=time.perf_counter)
    # Wall-clock arrival for cross-process TTFT (controller → worker)
    arrival_time_wall: float = field(default_factory=time.time)
    result: Optional[Dict] = None
    completed: threading.Event = field(default_factory=threading.Event)

    def wait(self, timeout: float = None) -> Optional[Dict]:
        """Wait for request to complete and return result."""
        self.completed.wait(timeout=timeout)
        return self.result

    def set_result(self, result: Dict):
        """Set the result and signal completion."""
        self.result = result
        self.completed.set()


@dataclass
class BatchConfig:
    """Configuration for a function's batching behavior."""
    function_id: str

    # Base inference time for single request (T_0,i) - from profiling
    base_ttft_ms: float = 400.0

    # Marginal cost per additional request in batch (α_i) - from profiling
    marginal_cost_ms: float = 50.0

    # SLO target for TTFT in milliseconds
    slo_ms: float = 2000.0

    # Maximum batch size (safety cap; SLO formula determines actual max)
    max_batch_size: int = 16

    # Minimum batch delay before dispatching (ms)
    min_batch_delay_ms: float = 50.0

    # Maximum batch delay (caps the SLO-based window)
    max_batch_delay_ms: float = 250.0

    # When True, batches are dispatched immediately (no fill-or-expire delay).
    # Used by the Offline Profiler to bypass queue delay so that measured
    # TTFT reflects actual GPU compute time, not scheduling overhead.
    profiling_mode: bool = False

    def compute_ttft(self, batch_size: int) -> float:
        """
        Compute expected TTFT for given batch size.
        Formula: T_i(b) = T_0,i + α_i * (b - 1)
        """
        return self.base_ttft_ms + self.marginal_cost_ms * (batch_size - 1)

    def compute_max_batch_size(self) -> int:
        """
        Compute maximum batch size that satisfies SLO.
        Solve: T_0 + α * (b - 1) <= SLO
        => b <= (SLO - T_0) / α + 1
        """
        if self.marginal_cost_ms <= 0:
            return self.max_batch_size
        max_b = int((self.slo_ms - self.base_ttft_ms) / self.marginal_cost_ms) + 1
        return min(max(1, max_b), self.max_batch_size)

    def compute_batch_delay(self, current_batch_size: int) -> float:
        """
        Compute maximum time to wait for more requests.
        Formula: d_i = min(max_batch_delay, SLO_i - T_i(N_i))
        """
        expected_ttft = self.compute_ttft(current_batch_size)
        delay = self.slo_ms - expected_ttft
        return max(self.min_batch_delay_ms, min(delay, self.max_batch_delay_ms))


@dataclass
class Batch:
    """A batch of requests to be processed together."""
    batch_id: str
    function_id: str
    requests: List[InferenceRequest]
    created_time: float = field(default_factory=time.perf_counter)
    dispatch_time: Optional[float] = None
    dispatch_time_wall: Optional[float] = None

    @property
    def size(self) -> int:
        return len(self.requests)

    @property
    def prompts(self) -> List[str]:
        return [r.prompt for r in self.requests]

    @property
    def max_tokens(self) -> int:
        # Use max of all requests' max_tokens
        return max(r.max_tokens for r in self.requests) if self.requests else 256

    def get_oldest_wait_time(self) -> float:
        """Get wait time of oldest request in batch."""
        if not self.requests:
            return 0.0
        oldest = min(r.arrival_time for r in self.requests)
        return (time.perf_counter() - oldest) * 1000  # ms

    def set_results(self, results: List[Dict]):
        """Set results for all requests in batch."""
        for req, result in zip(self.requests, results):
            req.set_result(result)


class FunctionBatchQueue:
    """
    Per-function batch queue implementing fill-or-expire logic.

    Paper: "The batch stops either N_i requests are collected or
    the batching delay d_i expires."
    """

    def __init__(self, config: BatchConfig):
        self.config = config
        self.queue: queue.Queue[InferenceRequest] = queue.Queue()
        self.lock = threading.Lock()
        self.current_batch: List[InferenceRequest] = []
        self.batch_start_time: Optional[float] = None
        self._batch_counter = 0

    def add_request(self, request: InferenceRequest):
        """Add a request to the queue."""
        self.queue.put(request)

    def _create_batch_id(self) -> str:
        self._batch_counter += 1
        return f"{self.config.function_id}_batch_{self._batch_counter}"

    def try_form_batch(self) -> Optional[Batch]:
        """
        Try to form a batch from queued requests.

        Returns a Batch if:
        1. Max batch size reached, OR
        2. Batch delay (250ms window) expired

        Always forms batches based on window — no deferred batching.
        Returns None if no batch ready yet.
        """
        with self.lock:
            # Collect pending requests
            while not self.queue.empty():
                try:
                    req = self.queue.get_nowait()
                    self.current_batch.append(req)
                    if self.batch_start_time is None:
                        self.batch_start_time = time.perf_counter()
                except queue.Empty:
                    break

            if not self.current_batch:
                return None

            # In profiling mode, dispatch immediately to avoid queue delay
            # polluting TTFT measurements (see Section 4.2 profiler fix).
            if self.config.profiling_mode:
                return self._dispatch_batch()

            current_size = len(self.current_batch)
            max_size = self.config.compute_max_batch_size()

            # Check if max batch size reached
            if current_size >= max_size:
                return self._dispatch_batch(max_size)

            # Check if batch delay expired
            if self.batch_start_time is not None:
                elapsed_ms = (time.perf_counter() - self.batch_start_time) * 1000
                max_delay_ms = self.config.compute_batch_delay(current_size)

                if elapsed_ms >= max_delay_ms:
                    return self._dispatch_batch()

            return None

    def _dispatch_batch(self, max_count: int = None) -> Batch:
        """Create and return a batch from current requests.

        Args:
            max_count: If set, dispatch at most this many requests and keep
                       the remainder in current_batch for the next batch.
        """
        if max_count and max_count < len(self.current_batch):
            dispatch_reqs = self.current_batch[:max_count]
            self.current_batch = self.current_batch[max_count:]
            # Keep batch_start_time for remainder
        else:
            dispatch_reqs = self.current_batch.copy()
            self.current_batch = []
            self.batch_start_time = None

        batch = Batch(
            batch_id=self._create_batch_id(),
            function_id=self.config.function_id,
            requests=dispatch_reqs
        )
        batch.dispatch_time = time.perf_counter()

        logger.info(f"Dispatching batch {batch.batch_id} with {batch.size} requests")
        return batch

    def get_pending_count(self) -> int:
        """Get number of pending requests."""
        return len(self.current_batch) + self.queue.qsize()

    def force_dispatch(self) -> Optional[Batch]:
        """Force dispatch current batch regardless of size/delay."""
        with self.lock:
            # Collect any remaining queued requests
            while not self.queue.empty():
                try:
                    req = self.queue.get_nowait()
                    self.current_batch.append(req)
                except queue.Empty:
                    break

            if self.current_batch:
                return self._dispatch_batch()
            return None


class AdaptiveBatchScheduler:
    """
    Global adaptive batch scheduler implementing contention-aware scheduling.

    Paper: "At the global level, the Batching Scheduler addresses resource
    contention that occurs when multiple batches compete for the same GPU resources."

    When M batches processed concurrently:
    - T_eff_i(b) = M * T_i(b)
    - Prioritize by deadline margin: Δ_i = SLO_i - (w_i + M * T_i(b))
    """

    def __init__(self, process_batch_fn: Callable[[Batch], List[Dict]],
                 gpu_device: Optional[str] = None):
        """
        Args:
            process_batch_fn: Function that processes a batch and returns results
            gpu_device: GPU device string (e.g. "cuda:0"). When provided,
                contention factor M is tracked GPU-wide across all worker
                processes sharing this GPU via POSIX shared memory, matching
                Paper Eq. 4-5. When None, falls back to process-local tracking.
        """
        self.process_batch_fn = process_batch_fn
        self.function_queues: Dict[str, FunctionBatchQueue] = {}
        self.configs: Dict[str, BatchConfig] = {}
        self.lock = threading.Lock()

        # GPU-wide contention tracking (Paper Eq. 4-5)
        # When gpu_device is set, M is shared across all workers on the same
        # GPU via POSIX shared memory. Otherwise, falls back to local counter.
        self._gpu_contention: Optional[GPUContentionTracker] = None
        if gpu_device:
            gpu_idx = int(gpu_device.split(":")[-1]) if ":" in gpu_device else 0
            try:
                self._gpu_contention = GPUContentionTracker(gpu_idx)
                logger.info(f"Using GPU-wide contention tracking for {gpu_device}")
            except Exception as e:
                logger.warning(f"Failed to create GPU contention tracker: {e}. "
                             f"Falling back to process-local M.")

        # Process-local fallback when shared memory is unavailable
        self.active_batches: int = 0
        self.active_lock = threading.Lock()

        # Scheduler thread
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._process_threads: List[threading.Thread] = []

        self._request_counter = 0

    def register_function(self, config: BatchConfig):
        """Register a function with its batching configuration."""
        with self.lock:
            self.configs[config.function_id] = config
            self.function_queues[config.function_id] = FunctionBatchQueue(config)
            logger.info(f"Registered function {config.function_id} with "
                       f"max_batch={config.compute_max_batch_size()}, "
                       f"SLO={config.slo_ms}ms")

    def submit_request(self, function_id: str, prompt: str,
                       max_tokens: int = 256) -> InferenceRequest:
        """
        Submit an inference request.

        Returns an InferenceRequest that can be waited on for result.
        """
        if function_id not in self.function_queues:
            raise ValueError(f"Unknown function: {function_id}")

        self._request_counter += 1
        request = InferenceRequest(
            request_id=f"req_{self._request_counter}",
            prompt=prompt,
            max_tokens=max_tokens
        )

        self.function_queues[function_id].add_request(request)
        logger.debug(f"Submitted request {request.request_id} to {function_id}")
        return request

    def _get_contention_M(self) -> int:
        """
        Get GPU-wide contention factor M (number of concurrent batches).

        Paper Eq. 4: M = concurrent batches on the same GPU across ALL
        functions sharing it. Uses shared memory when available, otherwise
        falls back to process-local counter.
        """
        if self._gpu_contention:
            return self._gpu_contention.get_count() + 1  # +1 for this batch
        with self.active_lock:
            return self.active_batches + 1

    def _compute_deadline_margin(self, function_id: str, batch_size: int,
                                  wait_time_ms: float,
                                  extra_contention: int = 0) -> float:
        """
        Compute deadline margin for prioritization.
        Formula: Δ_i = SLO_i - (w_i + M * T_i(b))

        Lower margin = higher priority (closer to deadline)

        extra_contention: number of other batches already formed in this
        scheduling cycle that haven't been dispatched yet but will add
        to contention.
        """
        config = self.configs[function_id]

        # M is GPU-wide when shared contention tracker is available,
        # plus any batches formed earlier in this same cycle
        M = self._get_contention_M() + extra_contention

        # Effective TTFT with contention
        base_ttft = config.compute_ttft(batch_size)
        effective_ttft = M * base_ttft

        # Deadline margin
        margin = config.slo_ms - (wait_time_ms + effective_ttft)
        return margin

    def _select_ready_batches(self) -> List[Batch]:
        """
        Collect ALL ready batches across function queues, sorted by
        deadline margin (smallest margin = highest priority = first).

        Paper: "Batches with smaller deadline margins are prioritized
        for immediate processing"

        Returns all ready batches because try_form_batch() is destructive
        (it pops requests from the queue). Returning only one would lose
        the others.
        """
        candidates: List[tuple] = []  # (margin, batch)

        with self.lock:
            for func_id, func_queue in self.function_queues.items():
                batch = func_queue.try_form_batch()
                if batch is not None:
                    # Each additional batch in this cycle adds to contention:
                    # first batch sees M+1, second sees M+2, etc.
                    margin = self._compute_deadline_margin(
                        func_id, batch.size, batch.get_oldest_wait_time(),
                        extra_contention=len(candidates)
                    )
                    candidates.append((margin, batch))

        if not candidates:
            return []

        # Sort by deadline margin (ascending - smaller margin = higher priority)
        candidates.sort(key=lambda x: x[0])

        return [batch for _, batch in candidates]

    def _process_batch_thread(self, batch: Batch):
        """Process a batch in a separate thread."""
        # Track contention GPU-wide (shared memory) and locally
        if self._gpu_contention:
            self._gpu_contention.increment()
        with self.active_lock:
            self.active_batches += 1

        try:
            oldest_wait = batch.get_oldest_wait_time()
            logger.info(f"[TRACE] {batch.batch_id} STEP3_BATCH_FORMED fn={batch.function_id} "
                       f"size={batch.size} active_inflight={self.active_batches} "
                       f"queue_wait_ms={oldest_wait:.0f}")

            # Call the actual processing function
            results = self.process_batch_fn(batch)

            if results is None:
                return  # Batch was requeued at controller, don't set results

            # Distribute results to requests
            if len(results) == batch.size:
                batch.set_results(results)
            else:
                # Handle mismatch - set error for all
                error_result = {"success": False, "error": "Batch processing failed"}
                batch.set_results([error_result] * batch.size)

        except Exception as e:
            logger.error(f"Error processing batch {batch.batch_id}: {e}")
            error_result = {"success": False, "error": str(e)}
            batch.set_results([error_result] * batch.size)
        finally:
            if self._gpu_contention:
                self._gpu_contention.decrement()
            with self.active_lock:
                self.active_batches -= 1

    def _scheduler_loop(self):
        """Main scheduler loop."""
        logger.info("Batch scheduler started")

        while self._running:
            batches = self._select_ready_batches()

            if batches:
                # Dispatch all ready batches (priority order)
                for batch in batches:
                    thread = threading.Thread(
                        target=self._process_batch_thread,
                        args=(batch,),
                        daemon=True
                    )
                    thread.start()
                    self._process_threads.append(thread)

                # Prune completed threads to prevent unbounded growth
                self._process_threads = [
                    t for t in self._process_threads if t.is_alive()
                ]
            else:
                # No batch ready, sleep briefly
                time.sleep(0.01)  # 10ms

        logger.info("Batch scheduler stopped")

    def start(self):
        """Start the scheduler."""
        if self._running:
            return

        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True
        )
        self._scheduler_thread.start()

    def stop(self):
        """Stop the scheduler and process remaining requests."""
        self._running = False

        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)

        # Force dispatch any remaining batches
        with self.lock:
            for func_queue in self.function_queues.values():
                batch = func_queue.force_dispatch()
                if batch:
                    self._process_batch_thread(batch)

        # Wait for processing threads
        for thread in self._process_threads:
            thread.join(timeout=5.0)

    def get_stats(self) -> Dict:
        """Get scheduler statistics."""
        with self.lock:
            stats = {
                "active_batches_local": self.active_batches,
                "active_batches_gpu": self._gpu_contention.get_count() if self._gpu_contention else self.active_batches,
                "gpu_contention_enabled": self._gpu_contention is not None,
                "functions": {}
            }
            for func_id, func_queue in self.function_queues.items():
                stats["functions"][func_id] = {
                    "pending_requests": func_queue.get_pending_count(),
                    "config": {
                        "max_batch_size": self.configs[func_id].compute_max_batch_size(),
                        "slo_ms": self.configs[func_id].slo_ms
                    }
                }
            return stats

    # -------------------------------------------------------------------------
    # Integration Methods (for Controller and Profiler)
    # -------------------------------------------------------------------------

    def update_config_from_profile(
        self,
        function_id: str,
        base_ttft_ms: float,
        marginal_cost_ms: float,
        slo_ms: Optional[float] = None
    ) -> bool:
        """
        Update function configuration from profiler results.

        This allows the Offline Profiler to update batch scheduling
        parameters after measuring actual performance.

        Args:
            function_id: Function to update
            base_ttft_ms: Measured T_0 (base TTFT)
            marginal_cost_ms: Measured α (marginal cost)
            slo_ms: Optional new SLO target

        Returns:
            True if update was successful
        """
        with self.lock:
            if function_id not in self.configs:
                logger.warning(f"Cannot update config: unknown function {function_id}")
                return False

            config = self.configs[function_id]
            old_base = config.base_ttft_ms
            old_marginal = config.marginal_cost_ms

            if base_ttft_ms is not None:
                config.base_ttft_ms = base_ttft_ms
            if marginal_cost_ms is not None:
                config.marginal_cost_ms = marginal_cost_ms
            if slo_ms is not None:
                config.slo_ms = slo_ms

            # Recalculate max batch size based on new parameters
            new_max_batch = config.compute_max_batch_size()

            logger.info(
                f"Updated {function_id} config: "
                f"T_0={old_base:.1f}->{config.base_ttft_ms:.1f}ms, "
                f"α={old_marginal:.1f}->{config.marginal_cost_ms:.1f}ms, "
                f"max_batch={new_max_batch}"
            )

            return True


def profile_function(inference_fn: Callable, prompt: str,
                     max_tokens: int = 32, num_samples: int = 3) -> BatchConfig:
    """
    Profile a function to determine batching parameters.

    This measures T_0 (base TTFT) and α (marginal cost per request).
    In practice, this would be done offline with various batch sizes.
    """
    import statistics

    # Measure single request TTFT
    ttfts = []
    for _ in range(num_samples):
        start = time.perf_counter()
        result = inference_fn(prompt, max_tokens)
        ttft = result.get("ttft_ms", (time.perf_counter() - start) * 1000)
        ttfts.append(ttft)

    base_ttft = statistics.mean(ttfts)

    # Estimate marginal cost (simplified - would need actual batch testing)
    # Paper suggests this increases linearly, typical values 30-100ms
    marginal_cost = base_ttft * 0.1  # Estimate 10% increase per request

    return BatchConfig(
        function_id="profiled",
        base_ttft_ms=base_ttft,
        marginal_cost_ms=marginal_cost,
        slo_ms=base_ttft * 5,  # Default SLO: 5x base TTFT
        max_batch_size=128
    )
