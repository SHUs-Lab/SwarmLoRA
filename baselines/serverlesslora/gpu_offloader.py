#!/usr/bin/env python3
"""
Dynamic GPU Offloader for ServerlessLoRA.

Implements Section 4.3 of the paper:
- Monitor GPU memory pressure
- Offload artifacts to container memory during bursty workloads
- Uses value density to select offload candidates (lowest first)

Paper Offloading Policy (Eq. 6-7):
- Trigger: GPU usage > 90%
- Goal: Free Q_g MB of memory

Algorithm:
1. Identify GPU-loaded artifacts not in current batch
2. Compute value density for each (same formula as pre-loading)
3. Sort by density ASCENDING (remove lowest value first)
4. Remove artifacts until freed >= Q_g

Execution: Microsecond-level (greedy selection)
"""

import time
import threading
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Any
import torch

from artifact_registry import (
    ArtifactRegistry, get_registry,
    Artifact, ArtifactType, ArtifactLocation,
    Container
)
from config import (
    GPU_MEMORY_PRESSURE_THRESHOLD,
    GPU_TARGET_FREE_MB,
    GPU_OFFLOAD_POLL_MS,
    GPU_OFFLOAD_COOLDOWN_MS,
    WORKER_DEVICE,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class OffloadCandidate:
    """Candidate artifact for offloading."""
    artifact: Artifact
    container_id: str
    value_density: float
    memory_mb: float


@dataclass
class OffloadEvent:
    """Record of an offload operation."""
    artifact_id: str
    container_id: str
    memory_freed_mb: float
    timestamp: float
    reason: str  # "pressure", "explicit", "cleanup"


class DynamicGPUOffloader:
    """
    Dynamic GPU offloader for handling memory pressure.

    Paper Section 4.3: "When GPU memory usage exceeds a threshold,
    the offloader moves low-value artifacts back to container memory."
    """

    def __init__(
        self,
        registry: Optional[ArtifactRegistry] = None,
        device: str = WORKER_DEVICE,
        pressure_threshold: float = GPU_MEMORY_PRESSURE_THRESHOLD,
        target_free_mb: float = GPU_TARGET_FREE_MB,
        poll_interval_ms: int = GPU_OFFLOAD_POLL_MS,
        cooldown_ms: int = GPU_OFFLOAD_COOLDOWN_MS
    ):
        self.registry = registry or get_registry()
        self.device = device
        self.device_idx = int(device.split(":")[-1]) if ":" in device else 0
        self.pressure_threshold = pressure_threshold
        self.target_free_mb = target_free_mb
        self.poll_interval_ms = poll_interval_ms
        self.cooldown_ms = cooldown_ms

        # Tracking
        self._current_batch_artifacts: Set[str] = set()
        self._batch_lock = threading.Lock()

        # Offload history
        self._offload_history: List[OffloadEvent] = []
        self._last_offload_time = 0.0

        # Background thread
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Statistics
        self._total_offloads = 0
        self._total_mb_freed = 0.0
        self._pressure_events = 0

        # Callbacks for offload execution
        self._offload_callback = None

    # -------------------------------------------------------------------------
    # Memory Monitoring
    # -------------------------------------------------------------------------

    def get_gpu_memory_status(self) -> Dict[str, float]:
        """
        Get current GPU memory status (device-wide, across all processes).

        Uses torch.cuda.mem_get_info() which calls CUDA's cudaMemGetInfo
        to report device-level free/total memory. This correctly shows
        memory used by ALL processes on the GPU (workers, BMS, etc.),
        unlike torch.cuda.memory_allocated() which is per-process.

        Returns:
            Dictionary with total_mb, used_mb, free_mb, usage_percent
        """
        try:
            if not torch.cuda.is_available():
                return {"total_mb": 0, "used_mb": 0, "free_mb": 0, "usage_percent": 0}

            # Device-wide memory: visible across all processes on this GPU
            free, total = torch.cuda.mem_get_info(self.device_idx)

            total_mb = total / (1024 * 1024)
            free_mb = free / (1024 * 1024)
            used_mb = total_mb - free_mb
            usage_percent = (total - free) / total if total > 0 else 0

            return {
                "total_mb": total_mb,
                "used_mb": used_mb,
                "free_mb": free_mb,
                "usage_percent": usage_percent
            }
        except Exception as e:
            logger.error(f"Failed to get GPU memory status: {e}")
            return {"total_mb": 0, "used_mb": 0, "free_mb": 0, "usage_percent": 0}

    def check_memory_pressure(self) -> bool:
        """
        Check if GPU memory usage exceeds threshold.

        Paper: "Trigger offload when usage > 90%"

        Returns:
            True if memory pressure detected
        """
        status = self.get_gpu_memory_status()
        return status["usage_percent"] > self.pressure_threshold

    def compute_required_free_mb(self) -> float:
        """
        Compute how much memory needs to be freed.

        Paper Eq. 6: Q_g = target amount to free

        Returns:
            Required memory to free in MB (0 if no pressure)
        """
        status = self.get_gpu_memory_status()

        if status["usage_percent"] <= self.pressure_threshold:
            return 0.0

        # Target: bring usage below threshold + buffer
        target_usage = self.pressure_threshold - 0.05  # 5% buffer
        target_used_mb = status["total_mb"] * target_usage
        current_used_mb = status["used_mb"]

        required = max(current_used_mb - target_used_mb, self.target_free_mb)
        return required

    # -------------------------------------------------------------------------
    # Candidate Selection
    # -------------------------------------------------------------------------

    def compute_offload_candidates(
        self,
        required_mb: float
    ) -> List[OffloadCandidate]:
        """
        Identify candidates for offloading.

        Paper Algorithm:
        1. Find GPU-loaded artifacts not in current batch
        2. Compute value density
        3. Sort by density ASCENDING (lowest first for offload)

        Args:
            required_mb: Amount of memory to free

        Returns:
            List of candidates sorted by density ascending
        """
        candidates = []

        with self._batch_lock:
            protected = self._current_batch_artifacts.copy()

        artifacts = self.registry.get_all_artifacts()
        for artifact in artifacts:
            if artifact.location != ArtifactLocation.GPU:
                continue

            if artifact.artifact_id in protected:
                continue

            func_rate = 0.0
            for func in self.registry.get_all_functions():
                if artifact.artifact_id in func.required_artifacts:
                    func_rate += self.registry.get_request_rate(func.function_id)

            density = artifact.compute_density(func_rate, ArtifactLocation.GPU)

            candidates.append(OffloadCandidate(
                artifact=artifact,
                container_id=artifact.loaded_at_container or "",
                value_density=density,
                memory_mb=artifact.get_effective_size(ArtifactLocation.GPU)
            ))

        # Sort by density ascending (lowest value first)
        candidates.sort(key=lambda c: c.value_density)

        return candidates

    def _find_coupled_artifacts(
        self,
        artifact: Artifact,
        candidates_by_id: Dict[str, OffloadCandidate],
    ) -> List[OffloadCandidate]:
        """
        Find artifacts that must be co-evicted due to coupling constraints.

        Paper Section 4.3, coupling constraint:
        - When adapter (model weights) is evicted, any artifact that
          depends on it (e.g. KV-cache) must also be evicted.
        - When backbone is evicted, all adapters on it must also be evicted.

        Returns:
            List of additional candidates that must be evicted together.
        """
        coupled = []

        # If evicting an adapter, also evict artifacts that depend on it
        # (e.g. KV-cache entries referencing this adapter)
        all_artifacts = self.registry.get_all_artifacts()
        for other in all_artifacts:
            if other.artifact_id == artifact.artifact_id:
                continue
            if other.location != ArtifactLocation.GPU:
                continue
            # If other depends on this artifact, it must go too
            if artifact.artifact_id in other.depends_on:
                if other.artifact_id in candidates_by_id:
                    coupled.append(candidates_by_id[other.artifact_id])

        # If evicting a backbone, also evict all its adapters
        if artifact.artifact_type == ArtifactType.BACKBONE:
            for other in all_artifacts:
                if other.artifact_id == artifact.artifact_id:
                    continue
                if other.location != ArtifactLocation.GPU:
                    continue
                if other.backbone_id == artifact.artifact_id:
                    if other.artifact_id in candidates_by_id:
                        coupled.append(candidates_by_id[other.artifact_id])

        return coupled

    def select_offload_set(
        self,
        required_mb: float
    ) -> List[OffloadCandidate]:
        """
        Select artifacts to offload to free required memory.

        Paper: "Greedily select lowest-density artifacts until freed >= Q_g"

        Coupling constraint (Paper Eq.7): When an artifact is selected for
        eviction, all coupled artifacts (dependents, adapters on same
        backbone) must also be evicted.

        Args:
            required_mb: Target memory to free

        Returns:
            List of artifacts to offload
        """
        candidates = self.compute_offload_candidates(required_mb)

        # Build lookup for coupling resolution
        candidates_by_id = {c.artifact.artifact_id: c for c in candidates}

        selected = []
        selected_ids: Set[str] = set()
        freed_mb = 0.0

        for candidate in candidates:
            if freed_mb >= required_mb:
                break
            if candidate.artifact.artifact_id in selected_ids:
                continue

            selected.append(candidate)
            selected_ids.add(candidate.artifact.artifact_id)
            freed_mb += candidate.memory_mb

            # Enforce coupling: co-evict dependent artifacts
            coupled = self._find_coupled_artifacts(
                candidate.artifact, candidates_by_id
            )
            for coupled_candidate in coupled:
                if coupled_candidate.artifact.artifact_id not in selected_ids:
                    selected.append(coupled_candidate)
                    selected_ids.add(coupled_candidate.artifact.artifact_id)
                    freed_mb += coupled_candidate.memory_mb

        return selected

    # -------------------------------------------------------------------------
    # Offload Execution
    # -------------------------------------------------------------------------

    def offload_to_container(
        self,
        artifact: Artifact,
        container_id: str
    ) -> bool:
        """
        Offload an artifact from GPU to container memory.

        Args:
            artifact: Artifact to offload
            container_id: Container that owns the artifact

        Returns:
            True if successful
        """
        logger.info(f"Offloading {artifact.artifact_id} from GPU to container {container_id}")

        if self._offload_callback:
            success = self._offload_callback(artifact.artifact_id, container_id)
        else:
            # Default: update registry only (actual offload done by agent)
            success = True

        if success:
            self.registry.update_artifact_location(
                artifact.artifact_id,
                ArtifactLocation.CONTAINER,
                node_id=artifact.loaded_at_node,
                container_id=container_id
            )

            container = self.registry.get_container(container_id)
            if container:
                container.gpu_loaded_artifacts.discard(artifact.artifact_id)

            event = OffloadEvent(
                artifact_id=artifact.artifact_id,
                container_id=container_id,
                memory_freed_mb=artifact.get_effective_size(ArtifactLocation.GPU),
                timestamp=time.time(),
                reason="pressure"
            )
            self._offload_history.append(event)
            self._total_offloads += 1
            self._total_mb_freed += event.memory_freed_mb

            return True
        else:
            logger.error(f"Failed to offload {artifact.artifact_id}")
            return False

    def handle_memory_pressure(self) -> int:
        """
        Handle memory pressure by offloading artifacts.

        Paper Section 4.3 main algorithm.

        Returns:
            Number of artifacts offloaded
        """
        # Check cooldown
        if (time.time() - self._last_offload_time) * 1000 < self.cooldown_ms:
            return 0

        # Check if pressure exists
        if not self.check_memory_pressure():
            return 0

        self._pressure_events += 1
        logger.warning(f"GPU memory pressure detected (event #{self._pressure_events})")

        # Compute required memory to free
        required_mb = self.compute_required_free_mb()
        logger.info(f"Need to free {required_mb:.1f} MB")

        # Select offload candidates
        to_offload = self.select_offload_set(required_mb)
        if not to_offload:
            logger.warning("No offload candidates available")
            return 0

        # Execute offloads
        offloaded = 0
        for candidate in to_offload:
            if self.offload_to_container(candidate.artifact, candidate.container_id):
                offloaded += 1

        self._last_offload_time = time.time()
        logger.info(f"Offloaded {offloaded} artifacts")

        return offloaded

    # -------------------------------------------------------------------------
    # Batch Protection
    # -------------------------------------------------------------------------

    def protect_artifacts(self, artifact_ids: Set[str]):
        """
        Mark artifacts as in-use by current batch (protected from offload).

        Called by batch scheduler before processing.
        """
        with self._batch_lock:
            self._current_batch_artifacts = artifact_ids.copy()

    def release_artifacts(self, artifact_ids: Set[str]):
        """
        Release artifacts after batch completion.
        """
        with self._batch_lock:
            self._current_batch_artifacts -= artifact_ids

    # -------------------------------------------------------------------------
    # Background Monitor
    # -------------------------------------------------------------------------

    def _monitor_loop(self):
        """
        Background monitoring loop.

        Paper: "Poll GPU memory every 100ms"
        """
        logger.info("GPU offloader monitor started")

        while self._running:
            try:
                self.handle_memory_pressure()
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

            time.sleep(self.poll_interval_ms / 1000.0)

        logger.info("GPU offloader monitor stopped")

    def start(self):
        """Start the background monitor."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="GPUOffloaderMonitor"
        )
        self._monitor_thread.start()
        logger.info("DynamicGPUOffloader started")

    def stop(self):
        """Stop the background monitor."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("DynamicGPUOffloader stopped")

    # -------------------------------------------------------------------------
    # Callback Registration
    # -------------------------------------------------------------------------

    def set_offload_callback(self, callback):
        """
        Set callback for offload execution.

        Callback signature: (artifact_id: str, container_id: str) -> bool
        """
        self._offload_callback = callback

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get offloader statistics."""
        status = self.get_gpu_memory_status()

        return {
            "running": self._running,
            "gpu_memory": status,
            "pressure_threshold": self.pressure_threshold,
            "target_free_mb": self.target_free_mb,
            "total_offloads": self._total_offloads,
            "total_mb_freed": self._total_mb_freed,
            "pressure_events": self._pressure_events,
            "last_offload_time": self._last_offload_time,
            "protected_artifacts": len(self._current_batch_artifacts),
            "recent_offloads": [
                {
                    "artifact_id": e.artifact_id,
                    "memory_freed_mb": e.memory_freed_mb,
                    "timestamp": e.timestamp,
                    "reason": e.reason
                }
                for e in self._offload_history[-10:]  # Last 10
            ]
        }

    # -------------------------------------------------------------------------
    # HTTP API (for standalone operation)
    # -------------------------------------------------------------------------

    def run_server(self, port: int = 7200):
        """Run as HTTP server for monitoring."""
        try:
            from flask import Flask, request, jsonify
        except ImportError:
            logger.error("Flask not installed. Run: pip install flask")
            return

        app = Flask("gpu_offloader")

        @app.route('/health', methods=['GET'])
        def health():
            return jsonify({"status": "running" if self._running else "stopped"})

        @app.route('/status', methods=['GET'])
        def status():
            return jsonify(self.get_statistics())

        @app.route('/memory', methods=['GET'])
        def memory():
            return jsonify(self.get_gpu_memory_status())

        @app.route('/pressure', methods=['GET'])
        def pressure():
            return jsonify({
                "under_pressure": self.check_memory_pressure(),
                "required_free_mb": self.compute_required_free_mb()
            })

        @app.route('/candidates', methods=['GET'])
        def candidates():
            required = request.args.get('required_mb', type=float, default=1024.0)
            candidates = self.compute_offload_candidates(required)
            return jsonify({
                "candidates": [
                    {
                        "artifact_id": c.artifact.artifact_id,
                        "container_id": c.container_id,
                        "value_density": c.value_density,
                        "memory_mb": c.memory_mb
                    }
                    for c in candidates
                ]
            })

        @app.route('/offload', methods=['POST'])
        def trigger_offload():
            """Manually trigger offload."""
            result = self.handle_memory_pressure()
            return jsonify({"offloaded": result})

        @app.route('/protect', methods=['POST'])
        def protect():
            data = request.get_json() or {}
            artifact_ids = set(data.get("artifact_ids", []))
            self.protect_artifacts(artifact_ids)
            return jsonify({"protected": len(artifact_ids)})

        @app.route('/release', methods=['POST'])
        def release():
            data = request.get_json() or {}
            artifact_ids = set(data.get("artifact_ids", []))
            self.release_artifacts(artifact_ids)
            return jsonify({"released": len(artifact_ids)})

        # Start monitor
        self.start()

        logger.info(f"GPUOffloader HTTP server on port {port}")
        app.run(host='0.0.0.0', port=port, threaded=True, use_reloader=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dynamic GPU Offloader")
    parser.add_argument("--port", type=int, default=7200,
                        help="HTTP server port")
    parser.add_argument("--device", type=str, default=WORKER_DEVICE,
                        help="GPU device")
    parser.add_argument("--threshold", type=float, default=GPU_MEMORY_PRESSURE_THRESHOLD,
                        help="Memory pressure threshold (0-1)")
    parser.add_argument("--target-free-mb", type=float, default=GPU_TARGET_FREE_MB,
                        help="Target memory to free in MB")
    parser.add_argument("--poll-ms", type=int, default=GPU_OFFLOAD_POLL_MS,
                        help="Polling interval in milliseconds")
    args = parser.parse_args()

    offloader = DynamicGPUOffloader(
        device=args.device,
        pressure_threshold=args.threshold,
        target_free_mb=args.target_free_mb,
        poll_interval_ms=args.poll_ms
    )

    try:
        offloader.run_server(port=args.port)
    except KeyboardInterrupt:
        offloader.stop()


if __name__ == "__main__":
    main()
