"""Default cluster scaler — used by GlobalController."""

from typing import List, Optional

from ..scheduling.base import (
    ClusterScaler,
    ClusterSnapshot,
    ScalingAction,
    ScalingDecision,
)


class DefaultClusterScaler(ClusterScaler):
    """Default auto-scaling policy — extracted from GlobalController._make_scaling_decision."""

    def __init__(self, scale_up_threshold: int = 1):
        self.scale_up_threshold = scale_up_threshold

    def make_scaling_decisions(
        self,
        snapshot: ClusterSnapshot,
        drainable_node_ids: Optional[List[str]] = None,
    ) -> List[ScalingDecision]:
        decisions: List[ScalingDecision] = []

        # 1. Pre-launch: ensure min_active_nodes are running
        deficit = snapshot.min_active_nodes - snapshot.num_active_nodes
        for _ in range(min(deficit, snapshot.num_idle_nodes)):
            decisions.append(ScalingDecision(
                action=ScalingAction.SCALE_UP,
                base_model=snapshot.default_base_model,
            ))

        # 2. Scale-up: if pending requests exceed threshold
        if (snapshot.pending_requests > self.scale_up_threshold
                and snapshot.num_active_nodes < snapshot.max_active_nodes
                and snapshot.num_idle_nodes > 0):
            decisions.append(ScalingDecision(
                action=ScalingAction.SCALE_UP,
            ))

        # 3. Scale-down: drainable nodes above min floor
        if drainable_node_ids:
            active_after = snapshot.num_active_nodes
            for node_id in drainable_node_ids:
                if active_after <= snapshot.min_active_nodes:
                    break
                decisions.append(ScalingDecision(
                    action=ScalingAction.SCALE_DOWN,
                    node_id=node_id,
                ))
                active_after -= 1

        return decisions
