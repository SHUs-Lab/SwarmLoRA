#!/usr/bin/env python3
"""
Artifact Registry for ServerlessLoRA.

Provides shared data structures for all components:
- Artifact types and metadata
- Function profiles
- Worker node representation
- Container state tracking

This module is the foundation for the pre-loading scheduler, pre-loading agent,
GPU offloader, and controller components.
"""

import time
import threading
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from collections import defaultdict


class ArtifactType(Enum):
    """Types of artifacts that can be pre-loaded."""
    LIBRARY = auto()      # Python libraries, CUDA libraries
    BACKBONE = auto()     # Base LLM model weights
    ADAPTER = auto()      # LoRA adapter weights
    KERNEL = auto()       # Compiled CUDA kernels


class ArtifactLocation(Enum):
    """Where an artifact can be loaded."""
    NONE = auto()         # Not loaded
    DISK = auto()         # On disk only
    CONTAINER = auto()    # Loaded in container memory (CPU)
    GPU = auto()          # Loaded on GPU memory


@dataclass
class Artifact:
    """
    Represents a loadable artifact in the system.

    Paper Section 4.1: "Each artifact has an associated loading delay
    and memory footprint."
    """
    artifact_id: str
    artifact_type: ArtifactType
    size_mb: float                          # Memory footprint in MB
    loading_delay_ms: float                 # Time to load from disk/network

    # Dependencies (for precedence constraints)
    depends_on: Set[str] = field(default_factory=set)

    # For adapters: which backbone they require
    backbone_id: Optional[str] = None

    # Current state
    location: ArtifactLocation = ArtifactLocation.DISK
    loaded_at_node: Optional[str] = None
    loaded_at_container: Optional[str] = None

    def compute_value(self, request_rate: float) -> float:
        """
        Compute the value of pre-loading this artifact.

        Paper formula: v = delay × rate
        "The value of pre-loading an artifact is proportional to
        the loading delay saved times the request rate."

        Args:
            request_rate: Requests per second for functions using this artifact

        Returns:
            Value in delay-rate units
        """
        return self.loading_delay_ms * request_rate

    def compute_density(self, request_rate: float, location: ArtifactLocation) -> float:
        """
        Compute the value density for the greedy knapsack algorithm.

        Paper formula: ρ = v/w
        "We sort artifacts by value density and greedily select."

        Args:
            request_rate: Requests per second
            location: Target location (affects effective size)

        Returns:
            Value density (higher = better candidate for pre-loading)
        """
        value = self.compute_value(request_rate)
        # Weight is the memory footprint at the target location
        # (GPU has 1.05x overhead for alignment)
        weight = self.get_effective_size(location)
        if weight <= 0:
            return float('inf') if value > 0 else 0.0
        return value / weight

    def get_effective_size(self, location: ArtifactLocation) -> float:
        """Get memory footprint at given location."""
        # GPU memory might have different overhead
        if location == ArtifactLocation.GPU:
            # Assume slight overhead for GPU memory alignment
            return self.size_mb * 1.05
        return self.size_mb


@dataclass
class FunctionProfile:
    """
    Profile for a serverless function.

    Paper Section 4.2: "The offline profiler measures T_0 (base TTFT)
    and α (marginal cost per batch request) for each function."
    """
    function_id: str

    # Model/adapter configuration
    backbone_id: str
    adapter_id: Optional[str] = None

    # Profiled parameters (from offline profiler)
    base_ttft_ms: float = 400.0             # T_0: Base TTFT for single request
    marginal_cost_ms: float = 50.0          # α: Marginal cost per additional request

    # SLO configuration
    slo_ms: float = 2000.0                  # Target TTFT SLO

    # Runtime statistics (updated by controller)
    request_rate: float = 0.0               # Requests per second
    last_request_time: float = 0.0
    total_requests: int = 0

    # Required artifacts (computed from configuration)
    required_artifacts: Set[str] = field(default_factory=set)

    def compute_ttft(self, batch_size: int) -> float:
        """
        Compute expected TTFT for given batch size.

        Paper formula: T(b) = T_0 + α × (b - 1)
        """
        return self.base_ttft_ms + self.marginal_cost_ms * (batch_size - 1)

    def compute_max_batch_size(self) -> int:
        """
        Compute maximum batch size that satisfies SLO.

        Solve: T_0 + α × (b - 1) <= SLO
        => b <= (SLO - T_0) / α + 1
        """
        if self.marginal_cost_ms <= 0:
            return 128  # Safety cap
        max_b = int((self.slo_ms - self.base_ttft_ms) / self.marginal_cost_ms) + 1
        return min(max(1, max_b), 128)

    def update_request_stats(self, count: int = 1):
        """Update request statistics."""
        now = time.time()
        self.total_requests += count
        self.last_request_time = now


@dataclass
class Container:
    """
    Represents a container running a function instance.

    Paper Section 3.2: "Each container hosts a single function instance
    with its associated LoRA adapter."
    """
    container_id: str
    function_id: str
    node_id: str

    # Network configuration
    http_port: int
    http_host: str = "localhost"

    # State
    status: str = "starting"                # starting, ready, busy, stopping
    created_time: float = field(default_factory=time.time)
    last_request_time: float = field(default_factory=time.time)

    # Keep-alive configuration
    keep_alive_ms: int = 60000              # Default 60 seconds

    # Loaded artifacts (for preload score computation)
    loaded_artifacts: Set[str] = field(default_factory=set)
    gpu_loaded_artifacts: Set[str] = field(default_factory=set)

    # Current workload
    pending_requests: int = 0
    active_batch_size: int = 0

    # Lock for thread-safe pending_requests updates
    _pending_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # LoRA configuration
    lora_id: Optional[str] = None

    def is_expired(self) -> bool:
        """Check if container should be terminated due to inactivity."""
        if self.status == "busy":
            return False
        elapsed_ms = (time.time() - self.last_request_time) * 1000
        return elapsed_ms > self.keep_alive_ms

    def update_activity(self):
        """Update last request time."""
        self.last_request_time = time.time()

    def get_url(self) -> str:
        """Get the base URL for this container."""
        return f"http://{self.http_host}:{self.http_port}"

    def rebind_function(self, function_id: str, lora_id: str):
        """Rebind this container to serve a different function/adapter."""
        self.function_id = function_id
        self.lora_id = lora_id
        self.update_activity()

    def compute_preload_score(self, required_artifacts: Set[str]) -> float:
        """
        Compute pre-load score for routing decisions.

        Paper: "Route requests to containers with highest pre-load score."

        Score components:
        - +10 for each required artifact loaded in container
        - +20 bonus for artifacts loaded on GPU
        - -5 penalty for each pending request
        """
        score = 0.0

        # Score for loaded artifacts
        for artifact_id in required_artifacts:
            if artifact_id in self.loaded_artifacts:
                score += 10.0
                if artifact_id in self.gpu_loaded_artifacts:
                    score += 20.0  # GPU bonus

        # Penalty for busy container
        score -= self.pending_requests * 5.0

        return score


@dataclass
class WorkerNode:
    """
    Represents a worker node in the cluster.

    Paper Section 3.2: "Each worker node runs a pre-loading agent
    that manages container lifecycle and artifact loading."
    """
    node_id: str
    hostname: str

    # Agent endpoint
    agent_port: int = 7000

    # Memory limits
    container_memory_mb: float = 32768.0    # 32 GB container memory
    gpu_memory_mb: float = 44400.0          # L40S default; overridden by cluster config

    # Current usage
    container_memory_used_mb: float = 0.0
    gpu_memory_used_mb: float = 0.0

    # Containers on this node
    containers: Dict[str, Container] = field(default_factory=dict)

    # Status
    status: str = "active"                  # active, draining, offline
    last_heartbeat: float = field(default_factory=time.time)

    # GPU device indices available
    gpu_devices: List[int] = field(default_factory=lambda: [0])

    def get_container_memory_available(self) -> float:
        """Get available container memory in MB."""
        return max(0, self.container_memory_mb - self.container_memory_used_mb)

    def get_gpu_memory_available(self) -> float:
        """Get available GPU memory in MB."""
        return max(0, self.gpu_memory_mb - self.gpu_memory_used_mb)

    def can_allocate_container(self, required_mb: float) -> bool:
        """Check if node can allocate a new container."""
        return self.get_container_memory_available() >= required_mb

    def can_allocate_gpu(self, required_mb: float) -> bool:
        """Check if node can allocate GPU memory."""
        return self.get_gpu_memory_available() >= required_mb

    def get_agent_url(self) -> str:
        """Get the pre-loading agent URL."""
        return f"http://{self.hostname}:{self.agent_port}"

    def update_heartbeat(self):
        """Update last heartbeat time."""
        self.last_heartbeat = time.time()

    def is_healthy(self, timeout_s: float = 300.0) -> bool:
        """Check if node is healthy (received heartbeat recently)."""
        return (time.time() - self.last_heartbeat) < timeout_s


class ArtifactRegistry:
    """
    Central registry for all artifacts, functions, nodes, and containers.

    Thread-safe singleton that provides:
    - Artifact registration and lookup
    - Function profile management
    - Node and container tracking
    - Request rate statistics
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True

        # Thread-safe storage
        self._artifacts: Dict[str, Artifact] = {}
        self._functions: Dict[str, FunctionProfile] = {}
        self._nodes: Dict[str, WorkerNode] = {}

        # Locks for thread safety
        self._artifacts_lock = threading.RLock()
        self._functions_lock = threading.RLock()
        self._nodes_lock = threading.RLock()

        # Request rate tracking
        self._request_counts: Dict[str, List[float]] = defaultdict(list)
        self._rate_window_s: float = 60.0  # 1 minute window

    # -------------------------------------------------------------------------
    # Artifact Management
    # -------------------------------------------------------------------------

    def register_artifact(self, artifact: Artifact) -> None:
        """Register a new artifact."""
        with self._artifacts_lock:
            self._artifacts[artifact.artifact_id] = artifact

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Get artifact by ID."""
        with self._artifacts_lock:
            return self._artifacts.get(artifact_id)

    def get_all_artifacts(self) -> List[Artifact]:
        """Get all registered artifacts."""
        with self._artifacts_lock:
            return list(self._artifacts.values())

    def update_artifact_location(self, artifact_id: str,
                                  location: ArtifactLocation,
                                  node_id: Optional[str] = None,
                                  container_id: Optional[str] = None) -> bool:
        """Update artifact location."""
        with self._artifacts_lock:
            artifact = self._artifacts.get(artifact_id)
            if artifact:
                artifact.location = location
                artifact.loaded_at_node = node_id
                artifact.loaded_at_container = container_id
                return True
            return False

    # -------------------------------------------------------------------------
    # Function Management
    # -------------------------------------------------------------------------

    def register_function(self, profile: FunctionProfile) -> None:
        """Register a new function."""
        with self._functions_lock:
            self._functions[profile.function_id] = profile

    def get_function(self, function_id: str) -> Optional[FunctionProfile]:
        """Get function profile by ID."""
        with self._functions_lock:
            return self._functions.get(function_id)

    def get_all_functions(self) -> List[FunctionProfile]:
        """Get all registered functions."""
        with self._functions_lock:
            return list(self._functions.values())

    def update_function_profile(self, function_id: str,
                                 base_ttft_ms: Optional[float] = None,
                                 marginal_cost_ms: Optional[float] = None,
                                 slo_ms: Optional[float] = None) -> bool:
        """Update function profile from profiler."""
        with self._functions_lock:
            profile = self._functions.get(function_id)
            if profile:
                if base_ttft_ms is not None:
                    profile.base_ttft_ms = base_ttft_ms
                if marginal_cost_ms is not None:
                    profile.marginal_cost_ms = marginal_cost_ms
                if slo_ms is not None:
                    profile.slo_ms = slo_ms
                return True
            return False

    def record_request(self, function_id: str) -> None:
        """Record a request for rate calculation."""
        now = time.time()
        with self._functions_lock:
            self._request_counts[function_id].append(now)
            # Update function stats
            profile = self._functions.get(function_id)
            if profile:
                profile.update_request_stats()

    def get_request_rate(self, function_id: str) -> float:
        """
        Get current request rate for a function.

        Returns requests per second over the rate window.
        """
        now = time.time()
        cutoff = now - self._rate_window_s

        with self._functions_lock:
            # Clean old entries
            self._request_counts[function_id] = [
                t for t in self._request_counts[function_id] if t > cutoff
            ]

            count = len(self._request_counts[function_id])
            rate = count / self._rate_window_s

            # Update function profile
            profile = self._functions.get(function_id)
            if profile:
                profile.request_rate = rate

            return rate

    # -------------------------------------------------------------------------
    # Node Management
    # -------------------------------------------------------------------------

    def register_node(self, node: WorkerNode) -> None:
        """Register or update a worker node, preserving existing containers."""
        with self._nodes_lock:
            existing = self._nodes.get(node.node_id)
            if existing is not None:
                # Update heartbeat and metadata but keep existing containers
                existing.last_heartbeat = node.last_heartbeat or time.time()
                existing.hostname = node.hostname
                existing.agent_port = node.agent_port
                existing.gpu_memory_mb = node.gpu_memory_mb
                existing.container_memory_mb = node.container_memory_mb
            else:
                self._nodes[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[WorkerNode]:
        """Get node by ID."""
        with self._nodes_lock:
            return self._nodes.get(node_id)

    def get_all_nodes(self) -> List[WorkerNode]:
        """Get all registered nodes."""
        with self._nodes_lock:
            return list(self._nodes.values())

    def get_healthy_nodes(self) -> List[WorkerNode]:
        """Get all healthy nodes."""
        with self._nodes_lock:
            return [n for n in self._nodes.values()
                    if n.status == "active" and n.is_healthy()]

    # -------------------------------------------------------------------------
    # Container Management
    # -------------------------------------------------------------------------

    def register_container(self, container: Container) -> bool:
        """Register a new container."""
        with self._nodes_lock:
            node = self._nodes.get(container.node_id)
            if node:
                node.containers[container.container_id] = container
                return True
            return False

    def get_container(self, container_id: str) -> Optional[Container]:
        """Get container by ID."""
        with self._nodes_lock:
            for node in self._nodes.values():
                if container_id in node.containers:
                    return node.containers[container_id]
            return None

    def get_containers_for_function(self, function_id: str) -> List[Container]:
        """Get all containers running a function."""
        containers = []
        with self._nodes_lock:
            for node in self._nodes.values():
                for container in node.containers.values():
                    if container.function_id == function_id:
                        containers.append(container)
        return containers

    def get_all_ready_containers(self) -> List[Container]:
        """Get all containers with status == 'ready' across all nodes.

        Used by the controller's select_container() fallback to find any
        container with a matching adapter pre-loaded by the scheduler.
        """
        containers = []
        with self._nodes_lock:
            for node in self._nodes.values():
                for container in node.containers.values():
                    if container.status == "ready":
                        containers.append(container)
        return containers

    def rebind_container(self, container_id: str, function_id: str, lora_id: str) -> bool:
        """Update a container's function/adapter binding."""
        with self._nodes_lock:
            for node in self._nodes.values():
                if container_id in node.containers:
                    node.containers[container_id].rebind_function(function_id, lora_id)
                    return True
        return False

    def remove_container(self, container_id: str) -> bool:
        """Remove a container from the registry."""
        with self._nodes_lock:
            for node in self._nodes.values():
                if container_id in node.containers:
                    del node.containers[container_id]
                    return True
            return False

    # -------------------------------------------------------------------------
    # Statistics and Queries
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._artifacts_lock, self._functions_lock, self._nodes_lock:
            total_containers = sum(
                len(node.containers) for node in self._nodes.values()
            )
            total_gpu_mb = sum(
                node.gpu_memory_used_mb for node in self._nodes.values()
            )

            return {
                "artifacts": {
                    "total": len(self._artifacts),
                    "by_type": {
                        t.name: len([a for a in self._artifacts.values()
                                    if a.artifact_type == t])
                        for t in ArtifactType
                    }
                },
                "functions": {
                    "total": len(self._functions),
                    "active": len([f for f in self._functions.values()
                                  if f.request_rate > 0])
                },
                "nodes": {
                    "total": len(self._nodes),
                    "healthy": len(self.get_healthy_nodes())
                },
                "containers": {
                    "total": total_containers
                },
                "memory": {
                    "total_gpu_used_mb": total_gpu_mb
                }
            }

    def clear(self) -> None:
        """Clear all registry data (for testing)."""
        with self._artifacts_lock, self._functions_lock, self._nodes_lock:
            self._artifacts.clear()
            self._functions.clear()
            self._nodes.clear()
            self._request_counts.clear()


def get_registry() -> ArtifactRegistry:
    """Get the singleton ArtifactRegistry instance."""
    return ArtifactRegistry()


# Default artifact definitions for common components
def create_default_artifacts(model_name: str = "llama-3.1-8b") -> List[Artifact]:
    """
    Create default artifact definitions for a model.

    Returns artifacts for:
    - Python/CUDA libraries
    - Backbone model
    - Common kernels
    """
    artifacts = []

    # Library artifacts
    artifacts.append(Artifact(
        artifact_id="lib_torch",
        artifact_type=ArtifactType.LIBRARY,
        size_mb=500.0,
        loading_delay_ms=2000.0,
    ))

    artifacts.append(Artifact(
        artifact_id="lib_transformers",
        artifact_type=ArtifactType.LIBRARY,
        size_mb=100.0,
        loading_delay_ms=500.0,
        depends_on={"lib_torch"}
    ))

    # Backbone model (varies by model)
    backbone_sizes = {
        "llama-3.1-8b": 16000.0,    # 16 GB
        "llama-2-7b": 14000.0,      # 14 GB
        "mistral-7b": 14000.0,      # 14 GB
        "llama-2-13b": 26000.0,     # 26 GB
    }

    artifacts.append(Artifact(
        artifact_id=f"backbone_{model_name}",
        artifact_type=ArtifactType.BACKBONE,
        size_mb=backbone_sizes.get(model_name, 16000.0),
        loading_delay_ms=10000.0,   # 10 seconds to load
        depends_on={"lib_torch", "lib_transformers"}
    ))

    # Compiled kernels
    artifacts.append(Artifact(
        artifact_id="kernel_attention",
        artifact_type=ArtifactType.KERNEL,
        size_mb=10.0,
        loading_delay_ms=500.0,
        depends_on={f"backbone_{model_name}"}
    ))

    artifacts.append(Artifact(
        artifact_id="kernel_lora_fused",
        artifact_type=ArtifactType.KERNEL,
        size_mb=5.0,
        loading_delay_ms=300.0,
        depends_on={f"backbone_{model_name}"}
    ))

    return artifacts


def create_adapter_artifact(adapter_id: str, backbone_id: str,
                            size_mb: float = 100.0,
                            loading_delay_ms: float = 500.0) -> Artifact:
    """Create an adapter artifact definition."""
    return Artifact(
        artifact_id=f"adapter_{adapter_id.replace('/', '_')}",
        artifact_type=ArtifactType.ADAPTER,
        size_mb=size_mb,
        loading_delay_ms=loading_delay_ms,
        depends_on={backbone_id},
        backbone_id=backbone_id
    )
