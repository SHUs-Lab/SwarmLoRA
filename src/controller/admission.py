"""SLO-derived request deadlines.

A request is worth serving only while it can still meet its latency target:

    deadline = arrival + SLO

The deadline is enforced at every point the request waits. All three systems
queue a request twice -- once for a worker/container/instance, and again inside
it (batch queue, model lock, lock-step admission) -- so the same absolute
deadline is carried into the second stage and checked there too. This keeps the
whole of time-to-first-token inside the target rather than just the first wait.

Each check reserves the work that still remains after it passes: tokenize,
prefill, batch admission. The reservation is measured locally by each system,
so the check asks "will the deadline pass before this finishes?" rather than
"has it passed?". See post_admit_margin for the bounds applied to it.

One policy, one file, every system. The SLO is the only value chosen rather
than measured: 6s is the threshold S-LoRA defines for its own SLO attainment,
which the evaluation reports against. Override it with ADMISSION_SLO_S.
"""

from dataclasses import dataclass
from typing import Optional


DEFAULT_SLO_S: float = 6.0

# Bounds on the post-admit margin.
#
# MIN_SAMPLES: use the margin only once enough observations exist for it to be
#   an estimate. Below that it is treated as 0, which is the lenient direction:
#   a request admitted slightly late is scored a miss, while one wrongly pruned
#   is lost outright.
# MAX_FRAC:    keep the reservation a fraction of the target, so the check
#   stays a correction rather than becoming the deadline itself.
MARGIN_MIN_SAMPLES: int = 5
MARGIN_MAX_FRAC: float = 0.25


def post_admit_margin(observed_s: float, n_samples: int, slo_s: float) -> float:
    """Bounded estimate of the work remaining after a deadline check passes.

    Shared by every system so the bounds stay identical.
    """
    if n_samples < MARGIN_MIN_SAMPLES:
        return 0.0
    return min(max(0.0, observed_s), MARGIN_MAX_FRAC * slo_s)


@dataclass
class AdmissionConfig:
    """Per-deployment deadline settings.

    The SLO is the whole policy: a request has from arrival until arrival+slo_s
    to produce its first token, checked at both queueing stages.
    """
    slo_s: float = DEFAULT_SLO_S

    def __post_init__(self) -> None:
        # Validate at construction so a misconfigured target fails at startup
        # rather than silently running the wrong policy for a whole run.
        if self.slo_s <= 0:
            raise ValueError(f"slo_s must be positive, got {self.slo_s}")

    @classmethod
    def from_env(cls, env: Optional[dict] = None) -> "AdmissionConfig":
        """Build from environment, defaulting to the standard target."""
        import os
        e = env if env is not None else os.environ
        return cls(slo_s=float(e.get("ADMISSION_SLO_S", DEFAULT_SLO_S)))
