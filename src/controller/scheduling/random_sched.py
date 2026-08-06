"""Random scheduling policy: ablation baseline with no adapter awareness."""

import random
from typing import Dict, List, Optional, Set, Tuple

from .base import (
    LocalScheduler,
    LocalSchedulerConfig,
    WorkerSnapshot,
    RoutingAction,
    RoutingDecision,
    ReapDecision,
)


class RandomScheduler(LocalScheduler):
    """Ablation variant: picks any free worker at random with no adapter awareness.

    When a free worker is available, selects one uniformly at random regardless
    of which adapter it holds (triggering a swap if the adapter doesn't match).
    When no free worker exists, spawns up to max_workers. Isolates the
    contribution of LoRant's demand-proportional adapter-aware routing.

    Reaping uses the same idle-time logic as LoRantScheduler for a fair comparison.
    """

    def __init__(self, cfg: LocalSchedulerConfig = None):
        self.cfg = cfg or LocalSchedulerConfig()
        self._adapter_rates: Dict[str, float] = {}

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
        self._adapter_rates = dict(adapter_rates)

        free = [
            wid for wid, w in workers.items()
            if w.status == "ready" and w.active_requests == 0
        ]

        if free:
            chosen = random.choice(free)
            needs_swap = chosen not in workers_with_adapter
            return RoutingDecision(
                action=RoutingAction.ROUTE,
                worker_id=chosen,
                needs_swap=needs_swap,
            )

        if num_workers < max_workers:
            return RoutingDecision(action=RoutingAction.SPAWN)

        return RoutingDecision(action=RoutingAction.QUEUE)

    def get_proactive_spawns(self, *args, **kwargs):
        return []

    def select_workers_to_reap(
        self,
        workers: Dict[int, WorkerSnapshot],
        min_workers: int,
        scale_down_delay: float,
    ) -> ReapDecision:
        """Same idle-time reaping logic as LoRantScheduler for a fair comparison."""
        adapter_rates = self._adapter_rates
        num_workers = len(workers)

        busy = sum(1 for w in workers.values()
                   if w.status == "ready" and w.active_requests > 0)
        if num_workers > 0 and busy / num_workers > 0.3:
            return ReapDecision(worker_ids=[])

        worker_counts: Dict[Optional[str], int] = {}
        for w in workers.values():
            worker_counts[w.adapter_id] = worker_counts.get(w.adapter_id, 0) + 1

        eligible: List[Tuple[int, WorkerSnapshot]] = [
            (wid, w) for wid, w in workers.items()
            if w.status == "ready" and w.active_requests == 0
            and w.idle_time >= scale_down_delay
        ]
        if not eligible:
            return ReapDecision(worker_ids=[])

        eligible.sort(key=lambda item: -item[1].idle_time)

        current_count = num_workers
        reap_ids: List[int] = []
        for wid, w in eligible:
            if current_count <= min_workers:
                break
            adapter_count = worker_counts.get(w.adapter_id, 0)
            rate = adapter_rates.get(w.adapter_id, 0.0) if w.adapter_id else 0.0
            if adapter_count <= 1 and rate > 0:
                if w.idle_time < scale_down_delay * 4:
                    continue
            reap_ids.append(wid)
            current_count -= 1
            if w.adapter_id is not None:
                worker_counts[w.adapter_id] = worker_counts.get(w.adapter_id, 1) - 1

        return ReapDecision(worker_ids=reap_ids)
