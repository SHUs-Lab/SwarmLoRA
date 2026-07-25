#  Adapter Placement Policies - Decides where to route requests and adapters   #

from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from collections import defaultdict
import logging

try:
    from .registry import ClusterRegistry, NodeInfo
except ImportError:
    from registry import ClusterRegistry, NodeInfo

logger = logging.getLogger(__name__)


# Base Policy Interface

class PlacementPolicy(ABC):
    """Abstract base class for adapter placement policies"""

    def __init__(self, registry: ClusterRegistry, model_config=None):
        self.registry = registry
        self.model_config = model_config

    def _get_base_model(self, adapter_name: str) -> Optional[str]:
        """Get the base model for an adapter, if model_config is set."""
        if self.model_config:
            return self.model_config.get_base_model_for_adapter(adapter_name)
        return None

    def _filter_by_base_model(self, nodes: List[NodeInfo], adapter_name: str) -> List[NodeInfo]:
        """Filter nodes to only those running the correct base model."""
        base_model = self._get_base_model(adapter_name)
        if base_model:
            return [n for n in nodes if n.base_model == base_model]
        return nodes

    @abstractmethod
    async def select_node_for_request(
        self, adapter_name: str, inflight: Dict[str, int] = None,
    ) -> Optional[NodeInfo]:
        """Select a node to handle a request for the given adapter.

        Parameters
        ----------
        adapter_name : str
            The adapter requested.
        inflight : dict, optional
            node_id -> number of in-flight requests not yet reflected in
            registry metrics.  Implementations should add this to the
            node's active_workers for load calculations.
        """
        pass

    @abstractmethod
    async def select_node_for_scale_up(self, base_model: Optional[str] = None) -> Optional[NodeInfo]:
        """Select an idle node to activate.

        Parameters
        ----------
        base_model : str, optional
            Preferred base model for the new node.
        """
        pass


# Affinity-First Policy (Default)

class AffinityFirstPolicy(PlacementPolicy):
    """
    Prioritizes routing to nodes that already have the adapter loaded,
    but spills to less-loaded nodes when the affinity node is saturated.

    Order:
    1. Node with adapter loaded (lowest load) — unless overloaded
    2. If affinity node load > spillover_threshold and another node has
       significantly lower load, route there (triggers adapter swap but
       balances load across the cluster)
    3. Active node with correct base model and capacity (lowest load)
    4. None (triggers scale up or queueing)
    """

    def __init__(self, registry: ClusterRegistry, model_config=None,
                 spillover_threshold: float = 0.5,
                 spillover_gap: float = 0.2):
        super().__init__(registry, model_config)
        # Spill to another node when affinity node load exceeds this
        self.spillover_threshold = spillover_threshold
        # Only spill if the other node is this much less loaded
        self.spillover_gap = spillover_gap

    async def select_node_for_request(
        self, adapter_name: str, inflight: Dict[str, int] = None,
    ) -> Optional[NodeInfo]:
        inflight = inflight or {}

        all_active = await self.registry.get_active_nodes()
        all_active = self._filter_by_base_model(all_active, adapter_name)

        if not all_active:
            return None

        # Weighted round-robin by max_workers capacity.
        # When GPUs are identical, heartbeat-based active_workers is
        # unreliable over high-latency tunnels (counter leaks inflate load).
        # Instead, distribute proportionally using total_requests / max_workers
        # so each GPU gets equal work per unit capacity.
        def requests_per_capacity(n: NodeInfo) -> float:
            return n.total_requests / max(n.max_workers, 1)

        return min(all_active, key=requests_per_capacity)

        return None

    async def select_node_for_scale_up(self, base_model: Optional[str] = None) -> Optional[NodeInfo]:
        if base_model:
            idle_nodes = await self.registry.get_idle_nodes_for_base_model(base_model)
            if idle_nodes:
                return idle_nodes[0]
        return await self.registry.get_any_idle_node()


# Least Loaded Policy

class LeastLoadedPolicy(PlacementPolicy):
    """
    Always routes to the least loaded node with the correct base model.
    Good for balanced load distribution, worse for adapter cache hits.
    """

    async def select_node_for_request(
        self, adapter_name: str, inflight: Dict[str, int] = None,
    ) -> Optional[NodeInfo]:
        inflight = inflight or {}

        def effective_load(n: NodeInfo) -> float:
            return (n.active_workers + inflight.get(n.node_id, 0)) / max(n.max_workers, 1)

        base_model = self._get_base_model(adapter_name)
        if base_model:
            nodes = await self.registry.get_nodes_for_base_model_with_capacity(base_model)
            if not nodes:
                return None
            return min(nodes, key=effective_load)
        return await self.registry.get_least_loaded_node()

    async def select_node_for_scale_up(self, base_model: Optional[str] = None) -> Optional[NodeInfo]:
        if base_model:
            idle_nodes = await self.registry.get_idle_nodes_for_base_model(base_model)
            if idle_nodes:
                return idle_nodes[0]
        return await self.registry.get_any_idle_node()


# Locality-Aware Policy

@dataclass
class LocalityStats:
    """Tracks request locality for an adapter"""
    request_count: int = 0
    last_node_id: Optional[str] = None
    node_request_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


class LocalityAwarePolicy(PlacementPolicy):
    """
    Tracks which nodes handle which adapters and maintains locality.
    Tries to "pin" adapters to specific nodes based on historical patterns.

    Good for reducing adapter loading overhead when workload has locality.
    """

    def __init__(self, registry: ClusterRegistry, model_config=None):
        super().__init__(registry, model_config)
        self._adapter_stats: Dict[str, LocalityStats] = defaultdict(LocalityStats)

    async def select_node_for_request(
        self, adapter_name: str, inflight: Dict[str, int] = None,
    ) -> Optional[NodeInfo]:
        inflight = inflight or {}
        stats = self._adapter_stats[adapter_name]

        def effective_load(n: NodeInfo) -> float:
            return (n.active_workers + inflight.get(n.node_id, 0)) / max(n.max_workers, 1)

        # Priority 1: Node with adapter loaded (correct base model)
        nodes_with_adapter = await self.registry.get_nodes_for_adapter(adapter_name)
        nodes_with_adapter = self._filter_by_base_model(nodes_with_adapter, adapter_name)
        if nodes_with_adapter:
            # Prefer the node that has handled the most requests for this adapter
            best_node = max(
                nodes_with_adapter,
                key=lambda n: stats.node_request_counts.get(n.node_id, 0)
            )
            self._record_request(adapter_name, best_node.node_id)
            return best_node

        # Priority 2: Last node used (sticky routing, if correct base model)
        if stats.last_node_id:
            node = await self.registry.get_node(stats.last_node_id)
            if node and node.is_available and node.has_capacity:
                base_model = self._get_base_model(adapter_name)
                if not base_model or node.base_model == base_model:
                    self._record_request(adapter_name, node.node_id)
                    return node

        # Priority 3: Any available node with correct base model
        nodes_with_capacity = await self.registry.get_nodes_with_capacity()
        nodes_with_capacity = self._filter_by_base_model(nodes_with_capacity, adapter_name)
        if nodes_with_capacity:
            node = min(nodes_with_capacity, key=effective_load)
            self._record_request(adapter_name, node.node_id)
            return node

        return None

    def _record_request(self, adapter_name: str, node_id: str):
        """Record that a request was routed to a node"""
        stats = self._adapter_stats[adapter_name]
        stats.request_count += 1
        stats.last_node_id = node_id
        stats.node_request_counts[node_id] += 1

    async def select_node_for_scale_up(self, base_model: Optional[str] = None) -> Optional[NodeInfo]:
        if base_model:
            idle_nodes = await self.registry.get_idle_nodes_for_base_model(base_model)
            if idle_nodes:
                return idle_nodes[0]
        return await self.registry.get_any_idle_node()


# GPU-Type Aware Policy

class GPUTypeAwarePolicy(PlacementPolicy):
    """
    Routes requests based on GPU type.
    Useful for heterogeneous clusters with different GPU capabilities.

    Configuration example:
    {
        "adapter_gpu_preferences": {
            "large_adapter": ["H100", "A100"],
            "small_adapter": ["A100", "V100"],
        }
    }
    """

    def __init__(
        self,
        registry: ClusterRegistry,
        adapter_gpu_preferences: Dict[str, List[str]] = None,
        model_config=None,
    ):
        super().__init__(registry, model_config)
        self.adapter_gpu_preferences = adapter_gpu_preferences or {}
        self._fallback = AffinityFirstPolicy(registry, model_config)

    async def select_node_for_request(
        self, adapter_name: str, inflight: Dict[str, int] = None,
    ) -> Optional[NodeInfo]:
        inflight = inflight or {}
        preferred_gpus = self.adapter_gpu_preferences.get(adapter_name)

        if not preferred_gpus:
            return await self._fallback.select_node_for_request(adapter_name, inflight=inflight)

        def effective_load(n: NodeInfo) -> float:
            return (n.active_workers + inflight.get(n.node_id, 0)) / max(n.max_workers, 1)

        # Try nodes with preferred GPU types (filtered by base model)
        for gpu_type in preferred_gpus:
            active_nodes = await self.registry.get_active_nodes()
            active_nodes = self._filter_by_base_model(active_nodes, adapter_name)
            matching_nodes = [
                n for n in active_nodes
                if n.gpu_type == gpu_type and n.has_capacity
            ]
            if matching_nodes:
                with_adapter = [
                    n for n in matching_nodes
                    if adapter_name in n.loaded_adapters
                ]
                if with_adapter:
                    return min(with_adapter, key=effective_load)
                return min(matching_nodes, key=effective_load)

        return await self._fallback.select_node_for_request(adapter_name, inflight=inflight)

    async def select_node_for_scale_up(self, base_model: Optional[str] = None) -> Optional[NodeInfo]:
        return await self._fallback.select_node_for_scale_up(base_model)


# Binpack Policy

class BinpackPolicy(PlacementPolicy):
    """
    Packs requests onto fewest nodes possible.
    Good for maximizing resource utilization and enabling scale-down.

    Fills up nodes to max capacity before using new nodes.
    """

    def __init__(self, registry: ClusterRegistry, target_utilization: float = 0.8, model_config=None):
        super().__init__(registry, model_config)
        self.target_utilization = target_utilization

    async def select_node_for_request(
        self, adapter_name: str, inflight: Dict[str, int] = None,
    ) -> Optional[NodeInfo]:
        inflight = inflight or {}
        active_nodes = await self.registry.get_active_nodes()
        active_nodes = self._filter_by_base_model(active_nodes, adapter_name)
        if not active_nodes:
            return None

        def effective_util(n: NodeInfo) -> float:
            return (n.active_workers + inflight.get(n.node_id, 0)) / max(n.max_workers, 1)

        # Sort by utilization (highest first)
        sorted_nodes = sorted(active_nodes, key=effective_util, reverse=True)

        # First pass: node with adapter that has room
        for node in sorted_nodes:
            if adapter_name in node.loaded_adapters and node.has_capacity:
                return node

        # Second pass: node with room below target utilization
        for node in sorted_nodes:
            if effective_util(node) < self.target_utilization and node.has_capacity:
                return node

        # Third pass: any node with capacity
        for node in sorted_nodes:
            if node.has_capacity:
                return node

        return None

    async def select_node_for_scale_up(self, base_model: Optional[str] = None) -> Optional[NodeInfo]:
        # Only scale up if all active nodes are above target utilization
        active_nodes = await self.registry.get_active_nodes()

        if active_nodes:
            min_utilization = min(
                n.active_workers / n.max_workers for n in active_nodes
            )
            if min_utilization < self.target_utilization:
                return None  # Don't scale up yet

        if base_model:
            idle_nodes = await self.registry.get_idle_nodes_for_base_model(base_model)
            if idle_nodes:
                return idle_nodes[0]
        return await self.registry.get_any_idle_node()



# Policy Factory

def create_placement_policy(
    policy_name: str,
    registry: ClusterRegistry,
    model_config=None,
    **kwargs
) -> PlacementPolicy:
    """Factory function to create placement policies"""

    policies = {
        "affinity": AffinityFirstPolicy,
        "least_loaded": LeastLoadedPolicy,
        "locality": LocalityAwarePolicy,
        "gpu_aware": GPUTypeAwarePolicy,
        "binpack": BinpackPolicy,
    }

    if policy_name not in policies:
        raise ValueError(f"Unknown policy: {policy_name}. Available: {list(policies.keys())}")

    return policies[policy_name](registry, model_config=model_config, **kwargs)
