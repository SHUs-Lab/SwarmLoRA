#!/usr/bin/env python3
"""
Pre-Loading Scheduler for ServerlessLoRA.

Implements Section 4.1 of the paper:
- Knapsack-based optimization for pre-loading decisions
- Greedy algorithm sorted by value density
- Respects precedence constraints (libs → models → kernels)
- Respects coupling constraints (adapter on same GPU as backbone)

Greedy Algorithm (Paper Section 4.1):
1. For each artifact, compute value density: ρ = v/w
   where v = loading_delay × request_rate
2. Sort artifacts by ρ descending
3. Greedily assign to containers/GPUs while respecting:
   - Capacity constraints (memory limits)
   - Precedence constraints
   - Coupling constraints

Time complexity: O(|F|² × (|C| + |G|)) - practical for serverless
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import requests as http_requests

from artifact_registry import (
    ArtifactRegistry, get_registry,
    Artifact, ArtifactType, ArtifactLocation,
    FunctionProfile, WorkerNode, Container
)
from config import (
    PRELOAD_SCHEDULER_PORT,
    PRELOAD_SCHEDULE_INTERVAL_MS,
    PRELOAD_VALUE_DENSITY_THRESHOLD,
    REBALANCE_RATE_RATIO_THRESHOLD,
    REBALANCE_MIN_CONTAINERS_PER_FUNCTION,
    REBALANCE_COOLDOWN_S,
    REBALANCE_MAX_SWAPS_PER_NODE,
    REBALANCE_MIN_TOTAL_REQUESTS,
    REBALANCE_MIN_RATE_DIFF,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PreloadTarget(Enum):
    """Target location for preloading."""
    CONTAINER = "container"
    GPU = "gpu"


@dataclass
class PreloadDecision:
    """
    A decision to preload an artifact.

    Used by the PreloadAgent to execute preloading.
    """
    decision_id: str
    artifact_id: str
    target_node_id: str
    target_container_id: Optional[str]
    target_location: PreloadTarget

    # Function this artifact belongs to (for adapter swap rebinding)
    function_id: Optional[str] = None

    # Priority (lower = higher priority)
    priority: int = 0

    # Computed values
    value_density: float = 0.0
    expected_value: float = 0.0

    # Status
    status: str = "pending"  # pending, executing, completed, failed
    created_at: float = field(default_factory=time.time)


@dataclass
class ScheduleResult:
    """Result of a scheduling round."""
    schedule_id: str
    timestamp: float
    decisions: List[PreloadDecision]
    total_value: float
    artifacts_scheduled: int
    nodes_affected: int


class PreloadScheduler:
    """
    Central pre-loading scheduler implementing knapsack optimization.

    Paper Section 4.1: "The Pre-Loading Scheduler runs centrally and
    computes optimal artifact placement across all worker nodes."
    """

    def __init__(
        self,
        registry: Optional[ArtifactRegistry] = None,
        schedule_interval_ms: int = PRELOAD_SCHEDULE_INTERVAL_MS,
        value_density_threshold: float = PRELOAD_VALUE_DENSITY_THRESHOLD,
        swapping_lock: Optional[threading.Lock] = None,
        swapping_containers: Optional[set] = None,
    ):
        self.registry = registry or get_registry()
        self.schedule_interval_ms = schedule_interval_ms
        self.value_density_threshold = value_density_threshold
        # Shared with controller for foreground/background swap coordination
        self._swapping_lock = swapping_lock or threading.Lock()
        self._swapping_containers = swapping_containers if swapping_containers is not None else set()

        # Pending decisions by node
        self._pending_decisions: Dict[str, List[PreloadDecision]] = {}
        self._decision_lock = threading.Lock()
        self._decision_counter = 0

        # Scheduler state
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._last_schedule_time = 0.0

        # Rebalance state
        self._last_swap_time: Dict[str, float] = {}  # container_id -> last swap timestamp
        self._total_rebalance_decisions = 0

        # Statistics
        self._total_schedules = 0
        self._total_decisions = 0

    # -------------------------------------------------------------------------
    # Core Scheduling Algorithm
    # -------------------------------------------------------------------------

    def compute_schedule(self) -> ScheduleResult:
        """
        Compute the next scheduling round.

        Paper Algorithm:
        1. Compute value density for each artifact
        2. Sort by density descending
        3. Greedily assign respecting constraints

        Returns:
            ScheduleResult with preload decisions
        """
        schedule_id = f"schedule_{self._total_schedules}"
        self._total_schedules += 1

        logger.debug(f"Computing schedule {schedule_id}")

        artifacts = self.registry.get_all_artifacts()
        functions = self.registry.get_all_functions()
        nodes = self.registry.get_healthy_nodes()

        if not nodes:
            logger.warning("No healthy nodes available")
            return ScheduleResult(
                schedule_id=schedule_id,
                timestamp=time.time(),
                decisions=[],
                total_value=0.0,
                artifacts_scheduled=0,
                nodes_affected=0
            )

        artifact_rates = self._compute_artifact_rates(functions, artifacts)

        # Debug: log nonzero rates
        nonzero_rates = {k: v for k, v in artifact_rates.items() if v > 0.001}
        if nonzero_rates:
            logger.info(f"Schedule {schedule_id}: {len(nonzero_rates)} artifacts with nonzero rates, "
                       f"{len(nodes)} healthy nodes, {len(artifacts)} artifacts")

        ranked = self._compute_greedy_ranking(artifacts, artifact_rates, nodes)

        if nonzero_rates and not ranked:
            # Debug why ranking is empty — log at INFO for visibility
            for art in artifacts:
                rate = artifact_rates.get(art.artifact_id, 0.0)
                if rate < 0.001:
                    continue
                for node in nodes:
                    loaded = self._artifact_loaded_on_node(art, node)
                    cap = self._check_capacity(art, node, PreloadTarget.GPU)
                    prec = self._check_precedence(art, node)
                    coup = self._check_coupling(art, node)
                    density = art.compute_density(rate, ArtifactLocation.GPU)
                    logger.info(
                        f"  PCKP-debug {art.artifact_id} on {node.node_id}: "
                        f"loaded={loaded} cap={cap} prec={prec} coup={coup} "
                        f"density={density:.4f} rate={rate:.4f}")

        if ranked:
            logger.info(f"Schedule {schedule_id}: {len(ranked)} ranked candidates")

        decisions = self._generate_decisions(ranked, nodes)

        # Demand-driven rebalancing (Section 4.1 extension)
        # Execute swaps directly (same process as controller, shared registry)
        rebalance_candidates = self._compute_rebalance_candidates(nodes)
        rebalance_count = self._execute_rebalance_swaps(rebalance_candidates)

        # Store pending preload decisions (for agents to poll)
        with self._decision_lock:
            for decision in decisions:
                if decision.target_node_id not in self._pending_decisions:
                    self._pending_decisions[decision.target_node_id] = []
                self._pending_decisions[decision.target_node_id].append(decision)

        total_value = sum(d.expected_value for d in decisions)
        nodes_affected = len(set(d.target_node_id for d in decisions))

        self._total_decisions += len(decisions) + rebalance_count
        self._last_schedule_time = time.time()

        if rebalance_count > 0 or decisions:
            logger.info(f"Schedule {schedule_id}: {len(decisions)} preload + "
                       f"{rebalance_count} rebalance swaps executed, "
                       f"value={total_value:.2f}, nodes={nodes_affected}")

        return ScheduleResult(
            schedule_id=schedule_id,
            timestamp=time.time(),
            decisions=decisions,
            total_value=total_value,
            artifacts_scheduled=len(decisions),
            nodes_affected=nodes_affected
        )

    def _compute_artifact_rates(
        self,
        functions: List[FunctionProfile],
        artifacts: List[Artifact]
    ) -> Dict[str, float]:
        """
        Compute effective request rate for each artifact.

        An artifact's rate is the sum of rates of all functions that require it.
        """
        rates: Dict[str, float] = {}

        for artifact in artifacts:
            rates[artifact.artifact_id] = 0.0

        for func in functions:
            func_rate = self.registry.get_request_rate(func.function_id)
            for artifact_id in func.required_artifacts:
                if artifact_id in rates:
                    rates[artifact_id] += func_rate

        return rates

    def _compute_greedy_ranking(
        self,
        artifacts: List[Artifact],
        artifact_rates: Dict[str, float],
        nodes: List[WorkerNode]
    ) -> List[Tuple[Artifact, str, PreloadTarget, float]]:
        """
        Compute greedy ranking of (artifact, node, location, density).

        Paper: "Sort artifacts by value density ρ = v/w descending"

        Returns:
            List of (artifact, node_id, target_location, density)
            sorted by density descending
        """
        candidates = []

        for artifact in artifacts:
            rate = artifact_rates.get(artifact.artifact_id, 0.0)

            if rate < 0.001:
                continue

            # Boost unserved functions: if this adapter's function has
            # demand but zero containers, multiply density by 10x to
            # prioritize getting it placed.
            unserved_boost = 1.0
            if artifact.artifact_type == ArtifactType.ADAPTER:
                for func in self.registry.get_all_functions():
                    if artifact.artifact_id in func.required_artifacts:
                        containers = self.registry.get_containers_for_function(func.function_id)
                        if not containers and func.request_rate > 0:
                            unserved_boost = 10.0
                        break

            for target in [PreloadTarget.CONTAINER, PreloadTarget.GPU]:
                location = (ArtifactLocation.CONTAINER if target == PreloadTarget.CONTAINER
                           else ArtifactLocation.GPU)
                density = artifact.compute_density(rate, location) * unserved_boost

                if density < self.value_density_threshold:
                    continue

                for node in nodes:
                    if self._artifact_loaded_on_node(artifact, node):
                        continue

                    if not self._check_capacity(artifact, node, target):
                        continue

                    if not self._check_precedence(artifact, node):
                        continue

                    if not self._check_coupling(artifact, node):
                        continue

                    value = artifact.compute_value(rate)
                    candidates.append((artifact, node.node_id, target, density, value))

        candidates.sort(key=lambda x: -x[3])

        return [(c[0], c[1], c[2], c[3]) for c in candidates]

    def _compute_rebalance_candidates(
        self,
        nodes: List[WorkerNode]
    ) -> List[Tuple[Artifact, str, PreloadTarget, float, str]]:
        """
        Compute demand-driven rebalance candidates.

        Paper Section 4.1 extension: When all adapters are already loaded
        (round-robin warm pool), rebalance containers from low-demand to
        high-demand functions based on proportional fair share.

        Returns:
            List of (adapter_artifact, node_id, PreloadTarget.GPU, density, donor_container_id)
            sorted by density descending.
        """
        candidates = []
        now = time.time()

        for node in nodes:
            if not node.containers:
                continue

            func_containers: Dict[str, List[Container]] = {}
            for container in node.containers.values():
                fid = container.function_id
                if fid:
                    func_containers.setdefault(fid, []).append(container)

            if not func_containers:
                continue

            func_rates: Dict[str, float] = {}
            total_rate = 0.0
            for fid in func_containers:
                rate = self.registry.get_request_rate(fid)
                func_rates[fid] = rate
                total_rate += rate

            # No demand yet — skip rebalancing
            if total_rate < 0.001:
                continue

            # Need enough requests for a stable signal (avoid noisy early swaps)
            # total_rate ≈ total_requests / 60s window, so check equivalent count
            if total_rate < REBALANCE_MIN_TOTAL_REQUESTS / 60.0:
                continue

            total_containers = sum(len(cs) for cs in func_containers.values())
            if total_containers < 2:
                continue

            # Compute desired count per function (proportional fair share)
            desired: Dict[str, int] = {}
            for fid, rate in func_rates.items():
                desired[fid] = max(
                    REBALANCE_MIN_CONTAINERS_PER_FUNCTION,
                    round(total_containers * rate / total_rate)
                )

            starving: List[Tuple[str, int]] = []   # (function_id, deficit)
            overstocked: List[Tuple[str, int]] = []  # (function_id, surplus)

            for fid in func_containers:
                actual = len(func_containers[fid])
                want = desired.get(fid, 1)
                if actual < want:
                    # Only starving if requests are actually queuing:
                    # more pending requests than containers on this node
                    total_pending = sum(
                        c.pending_requests for c in func_containers[fid]
                    )
                    if total_pending > actual:
                        starving.append((fid, want - actual))
                elif actual > want and actual > REBALANCE_MIN_CONTAINERS_PER_FUNCTION:
                    overstocked.append((fid, actual - want))

            if not starving or not overstocked:
                continue

            # Sort starving by rate descending (highest demand first)
            starving.sort(key=lambda x: -func_rates[x[0]])
            # Sort overstocked by rate ascending (lowest demand donates first)
            overstocked.sort(key=lambda x: func_rates[x[0]])

            # Pair starving with donors
            # Track which containers have already been selected as donors
            # within this node to avoid double-counting.
            selected_donors: Set[str] = set()
            node_swap_count = 0

            for starving_fid, deficit in starving:
                if node_swap_count >= REBALANCE_MAX_SWAPS_PER_NODE:
                    break
                starving_rate = func_rates[starving_fid]

                starving_func = self.registry.get_function(starving_fid)
                if not starving_func:
                    continue
                adapter_artifact = None
                for aid in starving_func.required_artifacts:
                    art = self.registry.get_artifact(aid)
                    if art and art.artifact_type == ArtifactType.ADAPTER:
                        adapter_artifact = art
                        break
                if not adapter_artifact:
                    continue

                for ov_idx in range(len(overstocked)):
                    if deficit <= 0:
                        break
                    donor_fid, surplus = overstocked[ov_idx]
                    if surplus <= 0:
                        continue

                    donor_rate = func_rates[donor_fid]

                    # Anti-thrashing: starving rate must be significantly higher
                    if starving_rate < donor_rate * REBALANCE_RATE_RATIO_THRESHOLD:
                        continue

                    # Anti-thrashing: minimum absolute rate difference
                    if starving_rate - donor_rate < REBALANCE_MIN_RATE_DIFF:
                        continue

                    # Find eligible donor containers (may consume multiple)
                    donor_containers = func_containers[donor_fid]
                    for container in donor_containers:
                        if deficit <= 0 or surplus <= 0:
                            break

                        if container.container_id in selected_donors:
                            continue

                        # Don't reduce below minimum
                        current_count = len(donor_containers) - len(
                            selected_donors & {c.container_id for c in donor_containers}
                        )
                        if current_count <= REBALANCE_MIN_CONTAINERS_PER_FUNCTION:
                            break  # No more donations possible from this function

                        # Skip busy containers
                        if container.pending_requests > 0:
                            continue

                        # Swap cooldown
                        last_swap = self._last_swap_time.get(container.container_id, 0.0)
                        if now - last_swap < REBALANCE_COOLDOWN_S:
                            continue

                        # Compute density: (rate_diff x loading_delay) / adapter_size
                        rate_diff = starving_rate - donor_rate
                        density = (rate_diff * adapter_artifact.loading_delay_ms) / max(
                            adapter_artifact.size_mb, 1.0
                        )

                        candidates.append((
                            adapter_artifact,
                            node.node_id,
                            PreloadTarget.GPU,
                            density,
                            container.container_id
                        ))

                        selected_donors.add(container.container_id)
                        self._last_swap_time[container.container_id] = now
                        node_swap_count += 1
                        surplus -= 1
                        deficit -= 1

                        if node_swap_count >= REBALANCE_MAX_SWAPS_PER_NODE:
                            break

                    # Write back updated surplus
                    overstocked[ov_idx] = (donor_fid, surplus)

        candidates.sort(key=lambda x: -x[3])

        return candidates

    def _execute_rebalance_swaps(
        self,
        candidates: List[Tuple[Artifact, str, PreloadTarget, float, str]],
    ) -> int:
        """
        Execute rebalance swaps directly via worker endpoints.

        Runs in the controller process — updates the shared registry
        immediately so request routing and future schedule rounds see
        the new container assignments.

        Returns:
            Number of successful swaps.
        """
        executed = 0
        targeted_containers: Set[str] = set()

        for artifact, node_id, target, density, donor_container_id in candidates:
            if donor_container_id in targeted_containers:
                continue

            container = self.registry.get_container(donor_container_id)
            if not container or container.status != "ready":
                continue

            func_id = next(
                (f.function_id for f in self.registry.get_all_functions()
                 if artifact.artifact_id in f.required_artifacts),
                ""
            )
            if not func_id:
                continue

            func = self.registry.get_function(func_id)
            if not func or not func.adapter_id:
                continue

            new_adapter_id = func.adapter_id

            if container.lora_id == new_adapter_id:
                continue

            # Skip busy containers (re-check at execution time)
            if container.pending_requests > 0:
                continue

            # Coordinate with controller's inline swaps
            with self._swapping_lock:
                if container.container_id in self._swapping_containers:
                    logger.debug(f"Rebalance skip {donor_container_id}: already being swapped by controller")
                    continue
                self._swapping_containers.add(container.container_id)

            try:
                resp = http_requests.post(
                    f"{container.get_url()}/swap_adapter",
                    json={
                        "adapter_id": new_adapter_id,
                        "function_id": func_id,
                    },
                    timeout=2.0
                )

                if resp.status_code == 200 and resp.json().get("success"):
                    # Update shared registry (controller sees it immediately)
                    old_artifact_id = (
                        f"adapter_{container.lora_id.replace('/', '_')}"
                        if container.lora_id else None
                    )
                    new_artifact_id = f"adapter_{new_adapter_id.replace('/', '_')}"
                    if old_artifact_id:
                        container.loaded_artifacts.discard(old_artifact_id)
                        container.gpu_loaded_artifacts.discard(old_artifact_id)
                    container.loaded_artifacts.add(new_artifact_id)
                    container.gpu_loaded_artifacts.add(new_artifact_id)
                    self.registry.rebind_container(
                        container.container_id, func_id, new_adapter_id)

                    swap_ms = resp.json().get("swap_ms", 0)
                    logger.info(
                        f"Rebalance swap: {container.container_id} "
                        f"{container.function_id}->{func_id} ({swap_ms:.0f}ms)")

                    targeted_containers.add(donor_container_id)
                    self._last_swap_time[donor_container_id] = time.time()
                    executed += 1

                elif resp.status_code == 503:
                    logger.debug(f"Rebalance skip {donor_container_id}: worker busy")

            except Exception as e:
                logger.debug(f"Rebalance swap failed {donor_container_id}: {e}")
            finally:
                with self._swapping_lock:
                    self._swapping_containers.discard(container.container_id)

        self._total_rebalance_decisions += executed
        return executed

    def _artifact_loaded_on_node(self, artifact: Artifact, node: WorkerNode) -> bool:
        """Check if artifact is already loaded on a specific node."""
        for container in node.containers.values():
            # Check active function's required artifacts
            if container.function_id:
                func = self.registry.get_function(container.function_id)
                if func and artifact.artifact_id in func.required_artifacts:
                    return True
            # Check loaded_artifacts and gpu_loaded_artifacts sets
            if artifact.artifact_id in container.loaded_artifacts:
                return True
            if artifact.artifact_id in container.gpu_loaded_artifacts:
                return True
        # For non-adapter artifacts (backbone, libraries), check if node has them
        if artifact.artifact_type != ArtifactType.ADAPTER:
            if artifact.loaded_at_node == node.node_id:
                return True
            # Backbone/libraries are loaded on all nodes that have containers
            if artifact.artifact_type in (ArtifactType.BACKBONE, ArtifactType.LIBRARY):
                if node.containers:
                    return True
        return False

    def _estimate_node_memory(
        self,
        node: WorkerNode,
        all_artifacts: Optional[List[Artifact]] = None
    ) -> Tuple[float, float]:
        """
        Estimate current container and GPU memory usage on a node
        from artifacts known to be loaded there.

        Returns:
            (container_mb_used, gpu_mb_used)
        """
        if all_artifacts is None:
            all_artifacts = self.registry.get_all_artifacts()

        container_used = 0.0
        gpu_used = 0.0
        for artifact in all_artifacts:
            if not self._artifact_loaded_on_node(artifact, node):
                continue
            if artifact.artifact_id in {
                aid for c in node.containers.values()
                for aid in c.gpu_loaded_artifacts
            }:
                gpu_used += artifact.get_effective_size(ArtifactLocation.GPU)
            elif artifact.artifact_id in {
                aid for c in node.containers.values()
                for aid in c.loaded_artifacts
            }:
                container_used += artifact.get_effective_size(ArtifactLocation.CONTAINER)
        return container_used, gpu_used

    def _check_capacity(
        self,
        artifact: Artifact,
        node: WorkerNode,
        target: PreloadTarget
    ) -> bool:
        """
        Check if node has capacity for artifact.

        Paper: "Respect capacity constraints (memory limits)"
        """
        if target == PreloadTarget.GPU:
            return node.can_allocate_gpu(artifact.size_mb * 1.05)
        else:
            return node.can_allocate_container(artifact.size_mb)

    def _check_precedence(
        self,
        artifact: Artifact,
        node: WorkerNode
    ) -> bool:
        """
        Check precedence constraints.

        Paper: "Respect precedence: libraries → backbones → adapters → kernels"

        An artifact can only be loaded if all its dependencies are already loaded.
        """
        if not artifact.depends_on:
            return True

        for dep_id in artifact.depends_on:
            dep = self.registry.get_artifact(dep_id)
            if dep is None:
                continue

            # Backbones and libraries are shared via IPC — available on all nodes
            if dep.artifact_type in (ArtifactType.BACKBONE, ArtifactType.LIBRARY):
                if node.containers:  # Node has active containers = shared artifacts
                    continue

            # Check if dependency is loaded on this node
            if dep.loaded_at_node != node.node_id:
                # Check if dependency is being scheduled
                with self._decision_lock:
                    pending = self._pending_decisions.get(node.node_id, [])
                    dep_scheduled = any(d.artifact_id == dep_id for d in pending)
                    if not dep_scheduled:
                        return False

        return True

    def _check_coupling(
        self,
        artifact: Artifact,
        node: WorkerNode
    ) -> bool:
        """
        Check coupling constraints.

        Paper: "Respect coupling: adapter must be on same GPU as backbone"
        """
        if artifact.artifact_type != ArtifactType.ADAPTER:
            return True

        if not artifact.backbone_id:
            return True

        backbone = self.registry.get_artifact(artifact.backbone_id)
        if backbone is None:
            return False

        # Backbone is shared via IPC — available on all nodes with containers
        if backbone.artifact_type == ArtifactType.BACKBONE and node.containers:
            return True

        # For non-IPC backbones, check explicit location
        if backbone.loaded_at_node != node.node_id:
            with self._decision_lock:
                pending = self._pending_decisions.get(node.node_id, [])
                backbone_scheduled = any(
                    d.artifact_id == artifact.backbone_id for d in pending
                )
                if not backbone_scheduled:
                    return False

        return True

    def _generate_decisions(
        self,
        ranked: List[Tuple[Artifact, str, PreloadTarget, float]],
        nodes: List[WorkerNode]
    ) -> List[PreloadDecision]:
        """
        Generate preload decisions from ranked candidates.

        Greedy selection: take highest density items that fit.
        """
        decisions = []
        allocated_container: Dict[str, float] = {}  # node_id -> MB used
        allocated_gpu: Dict[str, float] = {}  # node_id -> MB used
        # Key by (artifact_id, node_id) so the same adapter can be placed
        # on multiple GPUs/nodes in a single scheduling round.
        scheduled_placements: Set[Tuple[str, str]] = set()

        # Initialize from estimated usage (based on loaded artifacts)
        # node.container_memory_used_mb / gpu_memory_used_mb may be stale
        # since agents update their own registry, not the controller's.
        all_artifacts = self.registry.get_all_artifacts()
        for node in nodes:
            container_est, gpu_est = self._estimate_node_memory(node, all_artifacts)
            allocated_container[node.node_id] = max(node.container_memory_used_mb, container_est)
            allocated_gpu[node.node_id] = max(node.gpu_memory_used_mb, gpu_est)

        priority = 0
        for artifact, node_id, target, density in ranked:
            # Skip if already scheduled for THIS node
            if (artifact.artifact_id, node_id) in scheduled_placements:
                continue

            node = self.registry.get_node(node_id)
            if node is None:
                continue

            required_mb = artifact.get_effective_size(
                ArtifactLocation.GPU if target == PreloadTarget.GPU
                else ArtifactLocation.CONTAINER
            )

            if target == PreloadTarget.GPU:
                current = allocated_gpu.get(node_id, 0)
                if current + required_mb > node.gpu_memory_mb:
                    continue
                allocated_gpu[node_id] = current + required_mb
            else:
                current = allocated_container.get(node_id, 0)
                if current + required_mb > node.container_memory_mb:
                    continue
                allocated_container[node_id] = current + required_mb

            self._decision_counter += 1
            func_id = next((f.function_id for f in self.registry.get_all_functions()
                           if artifact.artifact_id in f.required_artifacts), "")
            rate = self.registry.get_request_rate(func_id)

            decision = PreloadDecision(
                decision_id=f"decision_{self._decision_counter}",
                artifact_id=artifact.artifact_id,
                target_node_id=node_id,
                target_container_id=None,  # Agent will decide
                target_location=target,
                function_id=func_id or None,
                priority=priority,
                value_density=density,
                expected_value=artifact.compute_value(rate)
            )

            decisions.append(decision)
            scheduled_placements.add((artifact.artifact_id, node_id))
            priority += 1

        return decisions

    # -------------------------------------------------------------------------
    # Decision Management
    # -------------------------------------------------------------------------

    def get_pending_decisions(self, node_id: str) -> List[PreloadDecision]:
        """
        Get pending decisions for a node.

        Called by PreloadAgent to fetch work.
        """
        with self._decision_lock:
            decisions = self._pending_decisions.get(node_id, [])
            return sorted(decisions, key=lambda d: d.priority)

    def mark_decision_completed(
        self,
        decision_id: str,
        success: bool = True
    ) -> bool:
        """Mark a decision as completed."""
        with self._decision_lock:
            for node_id, decisions in self._pending_decisions.items():
                for i, d in enumerate(decisions):
                    if d.decision_id == decision_id:
                        d.status = "completed" if success else "failed"
                        # Remove from pending
                        self._pending_decisions[node_id].pop(i)
                        return True
        return False

    # -------------------------------------------------------------------------
    # Scheduler Loop
    # -------------------------------------------------------------------------

    def start(self):
        """Start the scheduler loop."""
        if self._running:
            return

        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="PreloadScheduler"
        )
        self._scheduler_thread.start()
        logger.info("PreloadScheduler started")

    def stop(self):
        """Stop the scheduler loop."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
        logger.info("PreloadScheduler stopped")

    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                self.compute_schedule()
            except Exception as e:
                logger.error(f"Scheduling error: {e}")

            time.sleep(self.schedule_interval_ms / 1000.0)

    # -------------------------------------------------------------------------
    # HTTP API (for standalone operation)
    # -------------------------------------------------------------------------

    def run_server(self, port: int = PRELOAD_SCHEDULER_PORT):
        """Run as HTTP server for distributed deployment."""
        try:
            from flask import Flask, request, jsonify
        except ImportError:
            logger.error("Flask not installed. Run: pip install flask")
            return

        app = Flask("preload_scheduler")

        @app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                "status": "running" if self._running else "stopped",
                "total_schedules": self._total_schedules,
                "total_decisions": self._total_decisions,
                "last_schedule": self._last_schedule_time
            })

        @app.route('/schedule', methods=['POST'])
        def trigger_schedule():
            """Trigger immediate scheduling."""
            result = self.compute_schedule()
            return jsonify({
                "schedule_id": result.schedule_id,
                "decisions": len(result.decisions),
                "total_value": result.total_value
            })

        @app.route('/decisions/<node_id>', methods=['GET'])
        def get_decisions(node_id):
            """Get pending decisions for a node."""
            decisions = self.get_pending_decisions(node_id)
            return jsonify({
                "node_id": node_id,
                "decisions": [
                    {
                        "decision_id": d.decision_id,
                        "artifact_id": d.artifact_id,
                        "target_location": d.target_location.value,
                        "target_container_id": d.target_container_id,
                        "function_id": d.function_id,
                        "priority": d.priority,
                        "value_density": d.value_density
                    }
                    for d in decisions
                ]
            })

        @app.route('/decisions/<decision_id>/complete', methods=['POST'])
        def complete_decision(decision_id):
            """Mark decision as completed."""
            data = request.get_json() or {}
            success = data.get("success", True)
            result = self.mark_decision_completed(decision_id, success)
            return jsonify({"success": result})

        @app.route('/statistics', methods=['GET'])
        def statistics():
            """Get scheduler statistics."""
            return jsonify({
                "total_schedules": self._total_schedules,
                "total_decisions": self._total_decisions,
                "total_rebalance_decisions": self._total_rebalance_decisions,
                "pending_by_node": {
                    node_id: len(decisions)
                    for node_id, decisions in self._pending_decisions.items()
                },
                "registry": self.registry.get_statistics()
            })

        # Start scheduler loop
        self.start()

        logger.info(f"PreloadScheduler HTTP server on port {port}")
        app.run(host='0.0.0.0', port=port, threaded=True, use_reloader=False)

    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        with self._decision_lock:
            pending_count = sum(
                len(d) for d in self._pending_decisions.values()
            )

        return {
            "running": self._running,
            "total_schedules": self._total_schedules,
            "total_decisions": self._total_decisions,
            "total_rebalance_decisions": self._total_rebalance_decisions,
            "pending_decisions": pending_count,
            "last_schedule_time": self._last_schedule_time,
            "schedule_interval_ms": self.schedule_interval_ms
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pre-Loading Scheduler")
    parser.add_argument("--port", type=int, default=PRELOAD_SCHEDULER_PORT,
                        help="HTTP server port")
    parser.add_argument("--interval-ms", type=int, default=PRELOAD_SCHEDULE_INTERVAL_MS,
                        help="Scheduling interval in milliseconds")
    parser.add_argument("--threshold", type=float, default=PRELOAD_VALUE_DENSITY_THRESHOLD,
                        help="Minimum value density threshold")
    args = parser.parse_args()

    scheduler = PreloadScheduler(
        schedule_interval_ms=args.interval_ms,
        value_density_threshold=args.threshold
    )

    try:
        scheduler.run_server(port=args.port)
    except KeyboardInterrupt:
        scheduler.stop()


if __name__ == "__main__":
    main()
