"""LoRant scheduling policy: demand-proportional swap-first scheduling."""

from typing import Dict, List, Optional, Set, Tuple

from .base import (
    LocalScheduler,
    LocalSchedulerConfig,
    WorkerSnapshot,
    RoutingAction,
    RoutingDecision,
    ReapDecision,
)


class LoRantScheduler(LocalScheduler):
    """Demand-proportional swap-first scheduler for split-model LoRA serving."""

    def __init__(
        self,
        cfg: LocalSchedulerConfig = None,
        swap_protection: float = 0.1,
        proactive_headroom: float = 0.2,
    ):
        """Parameters"""
        self.cfg = cfg or LocalSchedulerConfig()
        self.swap_protection = swap_protection
        self.proactive_headroom = proactive_headroom
        self._adapter_rates: Dict[str, float] = {}
        self._pending_depth: int = 0  # Set by controller before route_request

    # ------------------------------------------------------------------
    # Demand-proportional helpers
    # ------------------------------------------------------------------

    def _compute_demand_targets(
        self,
        adapter_rates: Dict[str, float],
        busy_workers: int,
        max_workers: int,
    ) -> Dict[str, int]:
        """Compute target worker count per adapter based on demand share."""
        total_rate = sum(adapter_rates.values())
        if total_rate <= 0:
            return {}

        # Pool size = busy workers + pending queue depth, with headroom.
        # Using busy alone causes logarithmic growth (small pool → small target).
        # Adding pending_depth captures actual demand waiting for workers.
        effective_demand = busy_workers + getattr(self, '_pending_depth', 0)
        pool_size = min(
            int(effective_demand * (1 + self.proactive_headroom)),
            max_workers,
        )

        targets: Dict[str, int] = {}
        for aid, rate in adapter_rates.items():
            if rate <= 0:
                continue
            share = rate / total_rate
            target = max(1, int(round(pool_size * share)))
            targets[aid] = target

        # Clamp total to max_workers
        total_target = sum(targets.values())
        if total_target > max_workers:
            scale = max_workers / total_target
            targets = {aid: max(1, int(t * scale)) for aid, t in targets.items()}

        return targets

    def _count_workers_per_adapter(
        self, workers: Dict[int, WorkerSnapshot],
    ) -> Dict[Optional[str], int]:
        """Count total workers (busy + idle) per adapter."""
        counts: Dict[Optional[str], int] = {}
        for w in workers.values():
            counts[w.adapter_id] = counts.get(w.adapter_id, 0) + 1
        return counts

    def _overserve_score(
        self,
        adapter_id: Optional[str],
        adapter_rates: Dict[str, float],
        worker_counts: Dict[Optional[str], int],
    ) -> float:
        """How overserved an adapter is: actual_count / target_count."""
        if adapter_id is None:
            return float('inf')
        rate = adapter_rates.get(adapter_id, 0.0)
        if rate <= 0:
            return float('inf')
        total_rate = sum(adapter_rates.values())
        if total_rate <= 0:
            return float('inf')
        share = rate / total_rate
        count = worker_counts.get(adapter_id, 0)
        # target for this adapter (at least 1)
        target = max(1.0, share * sum(worker_counts.values()))
        return count / target

    # ------------------------------------------------------------------
    # ABC implementation
    # ------------------------------------------------------------------

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

        # Free = ready + not busy
        free = {
            wid: w for wid, w in workers.items()
            if w.status == "ready" and w.active_requests == 0
        }

        # Priority 1: Free worker with matching adapter (MRU)
        matching = {wid: w for wid, w in free.items()
                    if wid in workers_with_adapter}
        if matching:
            best_wid = min(matching, key=lambda wid: matching[wid].idle_time)
            return RoutingDecision(
                action=RoutingAction.ROUTE,
                worker_id=best_wid,
                needs_swap=False,
            )

        # Priority 2: Swap — pick most overserved adapter's worker
        # Use any free worker (no swap_protection delay — swaps are fast ~100ms)
        swappable = dict(free)
        if swappable:
            worker_counts = self._count_workers_per_adapter(workers)
            best_wid = max(
                swappable,
                key=lambda wid: self._overserve_score(
                    swappable[wid].adapter_id, adapter_rates, worker_counts
                ),
            )
            # Swap handles this request, but also signal that pool should
            # grow if utilization is high.  The controller checks
            # needs_spawn and fires a background spawn task.
            busy = num_workers - len(free)
            utilization = busy / max(num_workers, 1)
            needs_grow = (num_workers < max_workers and utilization > 0.7)
            return RoutingDecision(
                action=RoutingAction.ROUTE,
                worker_id=best_wid,
                needs_swap=True,
                needs_spawn=needs_grow,
            )

        # Priority 3: Spawn — always spawn if below capacity and no free worker
        if num_workers < max_workers:
            return RoutingDecision(action=RoutingAction.SPAWN)

        # At capacity, all busy — queue only as last resort
        return RoutingDecision(action=RoutingAction.QUEUE)

    # ------------------------------------------------------------------
    # Proactive rebalancing
    # ------------------------------------------------------------------

    def get_proactive_rebalances(
        self,
        workers: Dict[int, WorkerSnapshot],
        adapter_rates: Dict[str, float],
        num_workers: int,
        max_workers: int,
    ) -> List[Tuple[int, str]]:
        """Return (worker_id, target_adapter) pairs for proactive swaps."""
        if not adapter_rates or num_workers == 0:
            return []

        targets = self._compute_demand_targets(adapter_rates, num_workers, max_workers)
        if not targets:
            return []

        worker_counts = self._count_workers_per_adapter(workers)

        # Find underserved adapters (need more workers)
        underserved: List[Tuple[str, int]] = []
        for aid, target in targets.items():
            current = worker_counts.get(aid, 0)
            if target - current > 0:
                underserved.append((aid, target - current))
        if not underserved:
            return []
        underserved.sort(key=lambda x: x[1], reverse=True)

        # Find idle overserved workers eligible for swap
        swappable: List[Tuple[int, WorkerSnapshot]] = []
        for wid, w in workers.items():
            if w.status != "ready" or w.active_requests > 0:
                continue
            if w.idle_time < self.swap_protection:
                continue
            score = self._overserve_score(w.adapter_id, adapter_rates, worker_counts)
            if score > 1.0:
                swappable.append((wid, w))

        if not swappable:
            return []

        # Most overserved first
        swappable.sort(
            key=lambda item: self._overserve_score(
                item[1].adapter_id, adapter_rates, worker_counts
            ),
            reverse=True,
        )

        # Match overserved workers → underserved adapters
        rebalances: List[Tuple[int, str]] = []
        remaining = list(underserved)
        ridx = 0

        for wid, w in swappable:
            if ridx >= len(remaining):
                break
            target_aid, deficit = remaining[ridx]
            if w.adapter_id == target_aid:
                continue
            # Protect last worker for an active adapter
            adapter_count = worker_counts.get(w.adapter_id, 0)
            rate = adapter_rates.get(w.adapter_id, 0.0) if w.adapter_id else 0.0
            if adapter_count <= 1 and rate > 0:
                continue

            rebalances.append((wid, target_aid))
            if w.adapter_id is not None:
                worker_counts[w.adapter_id] = worker_counts.get(w.adapter_id, 1) - 1
            worker_counts[target_aid] = worker_counts.get(target_aid, 0) + 1
            deficit -= 1
            if deficit <= 0:
                ridx += 1
            else:
                remaining[ridx] = (target_aid, deficit)
            if len(rebalances) >= 2:
                break

        return rebalances

    def get_proactive_spawns(
        self,
        workers: Dict[int, WorkerSnapshot],
        adapter_rates: Dict[str, float],
        num_workers: int,
        max_workers: int,
        spawns_in_flight: int = 0,
    ) -> List[str]:
        """Return adapter IDs needing new workers (pool growth only)."""
        if not adapter_rates or num_workers == 0:
            return []
        if num_workers + spawns_in_flight >= max_workers:
            return []

        busy = sum(1 for w in workers.values()
                   if w.status == "ready" and w.active_requests > 0)
        targets = self._compute_demand_targets(adapter_rates, busy, max_workers)
        if not targets:
            return []

        total_target = sum(targets.values())
        if num_workers + spawns_in_flight >= total_target:
            return []  # Pool is large enough, rebalance handles distribution

        worker_counts = self._count_workers_per_adapter(workers)
        spawns: List[str] = []
        committed = num_workers + spawns_in_flight

        # Spawn for adapters with largest deficit first
        for aid, target in sorted(
            targets.items(),
            key=lambda kv: kv[1] - worker_counts.get(kv[0], 0),
            reverse=True,
        ):
            deficit = target - worker_counts.get(aid, 0)
            if deficit > 0 and committed + len(spawns) < max_workers:
                spawns.append(aid)
                if len(spawns) >= 40:
                    break

        return spawns

    # ------------------------------------------------------------------
    # Reaping
    # ------------------------------------------------------------------

    def select_workers_to_reap(
        self,
        workers: Dict[int, WorkerSnapshot],
        min_workers: int,
        scale_down_delay: float,
    ) -> ReapDecision:
        """Reap workers only during sustained low traffic (pool shrink)."""
        adapter_rates = self._adapter_rates
        num_workers = len(workers)

        # Only reap when utilization is sustainably low
        busy = sum(1 for w in workers.values()
                   if w.status == "ready" and w.active_requests > 0)
        if num_workers > 0 and busy / num_workers > 0.3:
            return ReapDecision(worker_ids=[])

        worker_counts = self._count_workers_per_adapter(workers)

        eligible: List[Tuple[int, WorkerSnapshot]] = []
        for wid, w in workers.items():
            if w.status != "ready" or w.active_requests > 0:
                continue
            if w.idle_time < scale_down_delay:
                continue
            eligible.append((wid, w))

        if not eligible:
            return ReapDecision(worker_ids=[])

        # Longest idle first
        eligible.sort(key=lambda item: -item[1].idle_time)

        current_count = num_workers
        reap_ids: List[int] = []

        for wid, w in eligible:
            if current_count <= min_workers:
                break

            # Protect last worker for an active adapter
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


class LoRantNoSwapScheduler(LoRantScheduler):
    """Ablation variant: LoRant without swap-first routing.

    Identical to LoRantScheduler except that when no matching worker is free,
    it spawns a new worker instead of swapping an existing one. Isolates the
    contribution of hot-swapping to overall performance.
    """

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

        free = {
            wid: w for wid, w in workers.items()
            if w.status == "ready" and w.active_requests == 0
        }

        # Priority 1: Free worker with matching adapter (same as LoRant)
        matching = {wid: w for wid, w in free.items()
                    if wid in workers_with_adapter}
        if matching:
            best_wid = min(matching, key=lambda wid: matching[wid].idle_time)
            return RoutingDecision(
                action=RoutingAction.ROUTE,
                worker_id=best_wid,
                needs_swap=False,
            )

        # Priority 2: Spawn — skip swap entirely (ablation: no hot-swap)
        if num_workers < max_workers:
            return RoutingDecision(action=RoutingAction.SPAWN)

        # At capacity — queue
        return RoutingDecision(action=RoutingAction.QUEUE)

    def get_proactive_rebalances(self, *args, **kwargs):
        return []
