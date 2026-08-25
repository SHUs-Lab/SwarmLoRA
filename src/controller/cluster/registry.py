#  Cluster Registry - Tracks nodes, aggregators, and adapter placement         #

import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    IDLE = "idle"              # No aggregator running
    STARTING = "starting"      # Aggregator loading base model
    ACTIVE = "active"          # Ready to serve requests
    DRAINING = "draining"      # Finishing work before shutdown
    FAILED = "failed"          # Health check failed


@dataclass
class NodeInfo:
    """Information about a node in the cluster"""
    node_id: str
    host: str
    port: int                          # Node agent port
    gpu_type: str = "unknown"          # "A100", "H100", etc.
    gpu_memory_gb: float = 0           # GPU memory in GB
    num_gpus: int = 0                  # Number of GPUs on this node
    status: NodeStatus = NodeStatus.IDLE

    # Base model this node is running (set when aggregator starts)
    base_model: Optional[str] = None   # e.g., "llama_2_7b" or "mistral_7b"
    base_model_id: Optional[str] = None  # e.g., "meta-llama/Llama-2-7b-hf"

    # Aggregator state
    aggregator_pid: Optional[int] = None
    aggregator_ready_at: Optional[float] = None

    # Worker state (reported by node agent)
    loaded_adapters: Set[str] = field(default_factory=set)
    active_workers: int = 0
    total_workers: int = 0       # Actually spawned and ready
    free_workers: int = 0        # Ready but idle
    queue_depth: int = 0         # Requests queued at local controller
    max_workers: int = 32

    # Reported load metrics (from node agent heartbeat)
    reported_active_requests: int = 0
    reported_total_capacity: int = 0
    reported_num_swapping: int = 0
    reported_max_seq_len: int = 0
    reported_total_seq_len: int = 0
    reported_num_prefill: int = 0
    reported_num_decode: int = 0
    reported_utilization: float = 0.0

    # GPU memory (from controller heartbeat)
    reported_gpu_memory_free_mb: int = 0
    reported_gpu_memory_total_mb: int = 0

    # Adapter swap stats (from controller heartbeat)
    reported_swap_count: int = 0
    reported_total_swap_ms: float = 0.0

    # Error tracking (synced from global controller circuit breakers)
    circuit_state: str = "closed"  # "closed", "open", "half_open"

    # Metrics
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    last_request_at: float = 0
    total_requests: int = 0

    @property
    def is_available(self) -> bool:
        """Can this node accept new requests?"""
        return self.status == NodeStatus.ACTIVE

    @property
    def has_capacity(self) -> bool:
        """Can this node load more adapters?"""
        return self.active_workers < self.max_workers

    @property
    def idle_time(self) -> float:
        """Time since last request"""
        if self.last_request_at == 0:
            return time.time() - self.aggregator_ready_at if self.aggregator_ready_at else 0
        return time.time() - self.last_request_at

    @property
    def url(self) -> str:
        """Node agent URL"""
        return f"http://{self.host}:{self.port}"


@dataclass
class AdapterInfo:
    """Information about an adapter in the cluster"""
    adapter_name: str
    adapter_path: Optional[str] = None
    loaded_on_nodes: Set[str] = field(default_factory=set)  # node_ids
    total_requests: int = 0
    last_request_at: float = 0

    @property
    def is_loaded(self) -> bool:
        """Is this adapter loaded on any node?"""
        return len(self.loaded_on_nodes) > 0


class ClusterRegistry:
    """Central registry for cluster state."""

    def __init__(self):
        self.nodes: Dict[str, NodeInfo] = {}
        self.adapters: Dict[str, AdapterInfo] = {}

        # Reverse index: adapter -> nodes
        self._adapter_to_nodes: Dict[str, Set[str]] = defaultdict(set)

        # Locks
        self._nodes_lock = asyncio.Lock()
        self._adapters_lock = asyncio.Lock()

    # =========================================================================
    # Node Management
    # =========================================================================

    async def register_node(
        self,
        node_id: str,
        host: str,
        port: int,
        gpu_type: str = "unknown",
        gpu_memory_gb: float = 0,
        num_gpus: int = 0,
        max_workers: int = 32,
    ) -> NodeInfo:
        """Register a new node in the cluster"""
        async with self._nodes_lock:
            if node_id in self.nodes:
                # Update existing node
                node = self.nodes[node_id]
                node.host = host
                node.port = port
                node.gpu_type = gpu_type
                node.gpu_memory_gb = gpu_memory_gb
                node.num_gpus = num_gpus
                node.max_workers = max_workers
                node.last_heartbeat = time.time()
                logger.info(f"Node {node_id} re-registered")
            else:
                # New node
                node = NodeInfo(
                    node_id=node_id,
                    host=host,
                    port=port,
                    gpu_type=gpu_type,
                    gpu_memory_gb=gpu_memory_gb,
                    num_gpus=num_gpus,
                    max_workers=max_workers,
                )
                self.nodes[node_id] = node
                logger.info(f"Node {node_id} registered: {host}:{port} ({gpu_type})")

            return node

    async def unregister_node(self, node_id: str):
        """Remove a node from the cluster"""
        adapters_to_clean = set()
        async with self._nodes_lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                adapters_to_clean = set(node.loaded_adapters)
                del self.nodes[node_id]
                logger.info(f"Node {node_id} unregistered")

        # Clean adapter mappings under correct lock
        if adapters_to_clean:
            async with self._adapters_lock:
                for adapter_name in adapters_to_clean:
                    self._adapter_to_nodes[adapter_name].discard(node_id)
                    if adapter_name in self.adapters:
                        self.adapters[adapter_name].loaded_on_nodes.discard(node_id)

    async def update_node_status(self, node_id: str, status: NodeStatus):
        """Update node status"""
        async with self._nodes_lock:
            if node_id in self.nodes:
                old_status = self.nodes[node_id].status
                self.nodes[node_id].status = status
                self.nodes[node_id].last_heartbeat = time.time()

                if status == NodeStatus.ACTIVE and old_status == NodeStatus.STARTING:
                    self.nodes[node_id].aggregator_ready_at = time.time()

                logger.debug(f"Node {node_id} status: {old_status.value} -> {status.value}")

    async def update_node_heartbeat(
        self,
        node_id: str,
        active_workers: int,
        loaded_adapters: List[str],
        aggregator_pid: Optional[int] = None,
        max_workers: Optional[int] = None,
        num_gpus: Optional[int] = None,
        gpu_type: Optional[str] = None,
        node_metrics: Optional[Dict] = None,
        total_workers: int = 0,
        free_workers: int = 0,
        queue_depth: int = 0,
    ):
        """Update node state from heartbeat"""
        # Acquire both locks to safely update both nodes and adapters
        async with self._nodes_lock:
            if node_id not in self.nodes:
                return

            node = self.nodes[node_id]
            node.last_heartbeat = time.time()
            node.active_workers = active_workers
            node.total_workers = total_workers
            node.free_workers = free_workers
            node.queue_depth = queue_depth
            node.aggregator_pid = aggregator_pid

            # Update effective max_workers if reported (GPU-memory-based cap)
            if max_workers is not None and max_workers != node.max_workers:
                logger.info(f"Node {node_id} max_workers updated: {node.max_workers} → {max_workers}")
                node.max_workers = max_workers

            # Dynamic GPU info refresh
            if num_gpus is not None and num_gpus > 0:
                node.num_gpus = num_gpus
            if gpu_type is not None and gpu_type != "unknown":
                node.gpu_type = gpu_type

            # Ingest load metrics if provided
            if node_metrics:
                node.reported_active_requests = node_metrics.get('total_active_requests', 0)
                node.reported_total_capacity = node_metrics.get('total_capacity', 0)
                node.reported_num_swapping = node_metrics.get('num_swapping', 0)
                node.reported_max_seq_len = node_metrics.get('max_seq_len', 0)
                node.reported_total_seq_len = node_metrics.get('total_seq_len', 0)
                node.reported_num_prefill = node_metrics.get('num_prefill', 0)
                node.reported_num_decode = node_metrics.get('num_decode', 0)
                node.reported_utilization = node_metrics.get('utilization', 0.0)
                # GPU memory
                node.reported_gpu_memory_free_mb = node_metrics.get('gpu_memory_free_mb', 0)
                node.reported_gpu_memory_total_mb = node_metrics.get('gpu_memory_total_mb', 0)
                # Swap stats
                node.reported_swap_count = node_metrics.get('swap_count', 0)
                node.reported_total_swap_ms = node_metrics.get('total_swap_ms', 0.0)

            # Calculate adapter changes
            old_adapters = node.loaded_adapters
            new_adapters = set(loaded_adapters)
            removed = old_adapters - new_adapters
            added = new_adapters - old_adapters

            node.loaded_adapters = new_adapters

        # Update adapter mappings under adapters lock
        async with self._adapters_lock:
            # Removed adapters
            for adapter_name in removed:
                self._adapter_to_nodes[adapter_name].discard(node_id)
                if adapter_name in self.adapters:
                    self.adapters[adapter_name].loaded_on_nodes.discard(node_id)

            # Added adapters
            for adapter_name in added:
                self._adapter_to_nodes[adapter_name].add(node_id)
                if adapter_name not in self.adapters:
                    self.adapters[adapter_name] = AdapterInfo(adapter_name=adapter_name)
                self.adapters[adapter_name].loaded_on_nodes.add(node_id)

    async def update_node_error_state(self, node_id: str, circuit_state: str):
        """Sync circuit breaker state from global controller."""
        async with self._nodes_lock:
            if node_id in self.nodes:
                self.nodes[node_id].circuit_state = circuit_state

    async def record_request(self, node_id: str, adapter_name: str):
        """Record that a request was sent to a node"""
        async with self._nodes_lock:
            if node_id in self.nodes:
                self.nodes[node_id].last_request_at = time.time()
                self.nodes[node_id].total_requests += 1

        async with self._adapters_lock:
            if adapter_name in self.adapters:
                self.adapters[adapter_name].last_request_at = time.time()
                self.adapters[adapter_name].total_requests += 1

    # =========================================================================
    # Queries
    # =========================================================================

    async def get_nodes_for_adapter(self, adapter_name: str) -> List[NodeInfo]:
        """Get all active nodes that have this adapter loaded"""
        async with self._nodes_lock:
            node_ids = self._adapter_to_nodes.get(adapter_name, set())
            return [
                self.nodes[nid] for nid in node_ids
                if nid in self.nodes and self.nodes[nid].is_available
            ]

    async def get_active_nodes(self) -> List[NodeInfo]:
        """Get all active nodes"""
        async with self._nodes_lock:
            return [n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE]

    async def get_idle_nodes(self) -> List[NodeInfo]:
        """Get all idle nodes (no aggregator)"""
        async with self._nodes_lock:
            return [n for n in self.nodes.values() if n.status == NodeStatus.IDLE]

    async def get_nodes_with_capacity(self) -> List[NodeInfo]:
        """Get active nodes that can accept more adapters"""
        async with self._nodes_lock:
            return [
                n for n in self.nodes.values()
                if n.is_available and n.has_capacity
            ]

    async def get_least_loaded_node(self) -> Optional[NodeInfo]:
        """Get the active node with fewest active workers"""
        async with self._nodes_lock:
            active = [n for n in self.nodes.values() if n.is_available]
            if not active:
                return None
            return min(active, key=lambda n: n.active_workers)

    async def get_node(self, node_id: str) -> Optional[NodeInfo]:
        """Get a specific node"""
        async with self._nodes_lock:
            return self.nodes.get(node_id)

    async def get_stale_nodes(self, timeout: float = 30.0) -> List[NodeInfo]:
        """Get nodes that haven't sent heartbeat recently."""
        async with self._nodes_lock:
            now = time.time()
            return [
                n for n in self.nodes.values()
                if n.status in (NodeStatus.ACTIVE, NodeStatus.DRAINING)
                and now - n.last_heartbeat > timeout
            ]

    async def get_drainable_nodes(self, idle_threshold: float = 300.0) -> List[NodeInfo]:
        """Get active nodes that have been idle and can be drained"""
        async with self._nodes_lock:
            return [
                n for n in self.nodes.values()
                if n.status == NodeStatus.ACTIVE
                and n.active_workers == 0
                and n.idle_time > idle_threshold
            ]

    # =========================================================================
    # Base Model Queries
    # =========================================================================

    async def set_node_base_model(
        self,
        node_id: str,
        base_model: str,
        base_model_id: Optional[str] = None,
    ):
        """Set the base model that a node is running."""
        async with self._nodes_lock:
            if node_id in self.nodes:
                self.nodes[node_id].base_model = base_model
                self.nodes[node_id].base_model_id = base_model_id
                logger.info(f"Node {node_id} set to base model: {base_model}")

    async def get_nodes_for_base_model(self, base_model: str) -> List[NodeInfo]:
        """Get all active nodes running a specific base model."""
        async with self._nodes_lock:
            return [
                n for n in self.nodes.values()
                if n.base_model == base_model and n.status == NodeStatus.ACTIVE
            ]

    async def get_nodes_for_base_model_with_capacity(self, base_model: str) -> List[NodeInfo]:
        """Get active nodes running a base model that have capacity for more workers."""
        async with self._nodes_lock:
            return [
                n for n in self.nodes.values()
                if n.base_model == base_model
                and n.status == NodeStatus.ACTIVE
                and n.has_capacity
            ]

    async def get_idle_nodes_for_base_model(self, base_model: str) -> List[NodeInfo]:
        """Get idle nodes that are assigned to a base model but not yet active."""
        async with self._nodes_lock:
            return [
                n for n in self.nodes.values()
                if n.base_model == base_model and n.status == NodeStatus.IDLE
            ]

    async def get_any_idle_node(self) -> Optional[NodeInfo]:
        """Get any idle node (no base model assigned or not active)."""
        async with self._nodes_lock:
            idle = [n for n in self.nodes.values() if n.status == NodeStatus.IDLE]
            return idle[0] if idle else None

    async def get_base_model_stats(self) -> Dict[str, dict]:
        """Get statistics per base model."""
        async with self._nodes_lock:
            stats = {}
            for node in self.nodes.values():
                if node.base_model:
                    if node.base_model not in stats:
                        stats[node.base_model] = {
                            "active_nodes": 0,
                            "idle_nodes": 0,
                            "total_workers": 0,
                            "total_requests": 0,
                        }
                    if node.status == NodeStatus.ACTIVE:
                        stats[node.base_model]["active_nodes"] += 1
                        stats[node.base_model]["total_workers"] += node.active_workers
                    elif node.status == NodeStatus.IDLE:
                        stats[node.base_model]["idle_nodes"] += 1
                    stats[node.base_model]["total_requests"] += node.total_requests
            return stats

    # =========================================================================
    # Adapter Management
    # =========================================================================

    async def register_adapter(
        self,
        adapter_name: str,
        adapter_path: Optional[str] = None,
    ) -> AdapterInfo:
        """Register an adapter"""
        async with self._adapters_lock:
            if adapter_name not in self.adapters:
                self.adapters[adapter_name] = AdapterInfo(
                    adapter_name=adapter_name,
                    adapter_path=adapter_path,
                )
            else:
                if adapter_path:
                    self.adapters[adapter_name].adapter_path = adapter_path

            return self.adapters[adapter_name]

    async def get_adapter(self, adapter_name: str) -> Optional[AdapterInfo]:
        """Get adapter info"""
        async with self._adapters_lock:
            return self.adapters.get(adapter_name)

    async def list_adapters(self) -> List[AdapterInfo]:
        """List all registered adapters"""
        async with self._adapters_lock:
            return list(self.adapters.values())

    # =========================================================================
    # Snapshot
    # =========================================================================

    async def get_cluster_state(self) -> dict:
        """Get full cluster state snapshot"""
        async with self._nodes_lock:
            nodes_state = {
                node_id: {
                    "host": node.host,
                    "port": node.port,
                    "gpu_type": node.gpu_type,
                    "num_gpus": node.num_gpus,
                    "gpu_memory_gb": node.gpu_memory_gb,
                    "status": node.status.value,
                    "active_workers": node.active_workers,
                    "max_workers": node.max_workers,
                    "loaded_adapters": list(node.loaded_adapters),
                    "idle_time": node.idle_time,
                    "total_requests": node.total_requests,
                    "reported_active_requests": node.reported_active_requests,
                    "reported_utilization": node.reported_utilization,
                    "reported_gpu_memory_free_mb": node.reported_gpu_memory_free_mb,
                    "reported_gpu_memory_total_mb": node.reported_gpu_memory_total_mb,
                    "reported_max_seq_len": node.reported_max_seq_len,
                    "reported_num_prefill": node.reported_num_prefill,
                    "reported_num_decode": node.reported_num_decode,
                    "reported_num_swapping": node.reported_num_swapping,
                    "reported_swap_count": node.reported_swap_count,
                    "reported_total_swap_ms": node.reported_total_swap_ms,
                    "circuit_state": node.circuit_state,
                }
                for node_id, node in self.nodes.items()
            }

        async with self._adapters_lock:
            adapters_state = {
                name: {
                    "loaded_on_nodes": list(info.loaded_on_nodes),
                    "total_requests": info.total_requests,
                }
                for name, info in self.adapters.items()
            }

        return {
            "nodes": nodes_state,
            "adapters": adapters_state,
            "summary": {
                "total_nodes": len(nodes_state),
                "active_nodes": len([n for n in nodes_state.values() if n["status"] == "active"]),
                "idle_nodes": len([n for n in nodes_state.values() if n["status"] == "idle"]),
                "total_adapters": len(adapters_state),
                "loaded_adapters": len([a for a in adapters_state.values() if a["loaded_on_nodes"]]),
            },
        }
