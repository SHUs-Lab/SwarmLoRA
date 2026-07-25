"""Abstract interface for local scheduling policies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set


@dataclass(frozen=True)
class WorkerSnapshot:
    """Read-only view of a worker for scheduling decisions."""
    worker_id: int
    adapter_id: Optional[str]
    status: str                # "ready", "starting", "swapping", etc.
    active_requests: int
    idle_time: float
    device: str = "cuda:0"


# ---------------------------------------------------------------------------
# Routing types
# ---------------------------------------------------------------------------

class RoutingAction(Enum):
    """What the controller should do with this request."""
    ROUTE  = "route"   # Send to worker_id (with optional swap)
    SPAWN  = "spawn"   # Start a new worker
    QUEUE  = "queue"   # Wait for capacity
    REJECT = "reject"  # Return error


@dataclass(frozen=True)
class RoutingDecision:
    """Full routing decision returned by the scheduler."""
    action: RoutingAction
    worker_id: Optional[int] = None
    needs_swap: bool = False
    needs_spawn: bool = False  # Hint: pool should grow (background spawn)


@dataclass(frozen=True)
class ReapDecision:
    """Result of select_workers_to_reap(): which workers to stop."""
    worker_ids: List[int]


# ---------------------------------------------------------------------------
# Scheduler configuration
# ---------------------------------------------------------------------------

@dataclass
class LocalSchedulerConfig:
    """Tuning knobs for inter-adapter scheduling and controller infrastructure."""
    # --- Controller timeouts ---
    spawn_timeout: float = 180.0
    queue_timeout: float = 300.0
    queue_poll_interval: float = 0.1
    health_check_interval: float = 5.0
    worker_http_timeout: float = 2.0          # HTTP timeout for health/adapter/ready checks
    worker_ready_poll_interval: float = 0.2   # Poll interval while waiting for worker ready
    worker_shutdown_timeout: float = 5.0      # HTTP timeout for /shutdown
    worker_kill_grace_period: float = 0.5     # Grace period before SIGKILL after SIGTERM
    initial_spawn_batch_timeout: float = 120.0  # Timeout for concurrent initial worker spawns
    idempotency_cleanup_interval: float = 300.0  # How often to clean expired idempotency entries


# ---------------------------------------------------------------------------
# Local scheduler ABC
# ---------------------------------------------------------------------------

class LocalScheduler(ABC):
    """Pluggable scheduling interface for AsyncLoadBalancer."""

    @abstractmethod
    def route_request(
        self,
        adapter_id: str,
        workers: Dict[int, WorkerSnapshot],
        workers_with_adapter: Set[int],
        adapter_rate: float,
        adapter_rates: Dict[str, float],
        num_workers: int,
        max_workers: int,
    ) -> RoutingDecision:
        """Select a worker for this request."""

    @abstractmethod
    def select_workers_to_reap(
        self,
        workers: Dict[int, WorkerSnapshot],
        min_workers: int,
        scale_down_delay: float,
    ) -> ReapDecision:
        """Select idle workers to terminate for scale-to-zero."""


# ---------------------------------------------------------------------------
# Cluster scaling types (used by serverless/cluster/)
# ---------------------------------------------------------------------------

class ScalingAction(Enum):
    """What the global controller should do."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"


@dataclass(frozen=True)
class ClusterSnapshot:
    """Read-only view of cluster state for scaling decisions."""
    num_active_nodes: int
    num_idle_nodes: int
    min_active_nodes: int
    max_active_nodes: int
    pending_requests: int
    default_base_model: Optional[str] = None


@dataclass(frozen=True)
class ScalingDecision:
    """A single scaling action to execute."""
    action: ScalingAction
    node_id: Optional[str] = None
    base_model: Optional[str] = None


class ClusterScaler(ABC):
    """Pluggable auto-scaling interface for GlobalController."""

    @abstractmethod
    def make_scaling_decisions(
        self,
        snapshot: ClusterSnapshot,
        drainable_node_ids: Optional[List[str]] = None,
    ) -> List[ScalingDecision]:
        """Return zero or more scaling actions to execute."""
