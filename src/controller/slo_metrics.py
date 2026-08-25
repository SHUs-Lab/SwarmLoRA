"""SLO-aware throughput metrics.

WHY THESE EXIST

The metrics the harnesses already report -- acceptance rate, tokens_per_second,
TTFT percentiles -- are all computed over *successful* requests. That is fine
when failures are rare and accidental, but it breaks down as soon as the system
drops requests deliberately:

  * TTFT percentiles improve when slow requests are dropped, because the
    dropped requests leave the distribution. The serving got no faster.
  * tokens_per_second counts tokens from a request that took 30s to first
    token exactly the same as one that took 0.3s.
  * acceptance_rate counts a deliberate, correct drop of a request that could
    never have met its target as equivalent to a crash.

Read together, those three can make a policy that improves user-visible
latency look like a regression, and a policy that merely hides its slow
requests look like an improvement. Both directions are wrong.

THE FIX

Two metrics whose denominator is *every submitted request*, so that dropping a
request can never flatter them:

    slo_attainment = |{r : r succeeded and ttft(r) <= SLO}| / |all submitted|

    effective_throughput = sum of completion tokens over requests meeting the
                           SLO, divided by wall-clock duration

A request that is dropped, fails, or answers too late contributes 0 to both. A
policy can only improve them by actually serving more requests within the
target. This is the "SLO attainment" S-LoRA reports, and the sense in which
"goodput" is normally meant in serving papers.

These are additive: nothing here replaces the existing metrics, which stay
comparable with previously published numbers. They are reported alongside.

FAIRNESS

One definition, one file, three call sites. Each system supplies its own
measured per-request samples; nothing here is parameterised per system. The
SLO comes from slo_from_env(), which reads ADMISSION_SLO_S exactly as the
admission policy does, so the number a request is *judged* against is always
the number admission was *targeting* -- if those two diverged, a system could
pass admission and still be scored as a miss.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from .admission import DEFAULT_SLO_S
except ImportError:  # loaded by path from a baseline tree
    try:
        from admission import DEFAULT_SLO_S  # type: ignore
    except ImportError:
        DEFAULT_SLO_S = 6.0


# One sample per SUBMITTED request: (succeeded, ttft_seconds, completion_tokens).
# ttft_s is ignored when succeeded is False; pass 0.0.
Sample = Tuple[bool, float, int]


def slo_from_env(env: Optional[dict] = None) -> float:
    """The SLO to judge against, read the same way the admission policy reads it.

    Callers must use this rather than DEFAULT_SLO_S directly. When this module
    is loaded by path from a baseline tree the relative import of the admission
    policy cannot resolve and DEFAULT_SLO_S falls back to a literal, so an
    ADMISSION_SLO_S override would silently judge requests against a different
    target than the one admission was aiming at.
    """
    import os
    e = env if env is not None else os.environ
    return float(e.get("ADMISSION_SLO_S", DEFAULT_SLO_S))


def slo_metrics(
    samples: Sequence[Sample],
    duration_s: float,
    slo_s: Optional[float] = None,
) -> Dict[str, float]:
    """Compute SLO attainment and effective throughput.

    Args:
        samples: one entry per submitted request, INCLUDING failures and
            drops. Omitting them is the exact bias this function exists to
            avoid, so the caller must pass the full population.
        duration_s: wall-clock span of the run.
        slo_s: time-to-first-token target; defaults to slo_from_env().

    Returns:
        Dict with slo_attainment (fraction of all submitted meeting the
        target), effective_throughput_tok_s, the SLO-meeting request count,
        and the slo_s used, so a result file is self-describing.
    """
    if slo_s is None:
        slo_s = slo_from_env()
    if slo_s <= 0:
        raise ValueError(f"slo_s must be positive, got {slo_s}")

    total = len(samples)
    met_tokens = 0
    met_count = 0
    for succeeded, ttft_s, tokens in samples:
        # A missing/zero TTFT cannot be shown to have met the target, so it is
        # counted as a miss rather than silently passing.
        if succeeded and 0 < ttft_s <= slo_s:
            met_count += 1
            met_tokens += max(0, tokens)

    return {
        "slo_s": slo_s,
        "slo_met_requests": met_count,
        "slo_attainment": round(met_count / total, 4) if total else 0.0,
        "effective_throughput_tok_s": (
            round(met_tokens / duration_s, 1) if duration_s > 0 else 0.0
        ),
    }


def ttft_percentiles_all(
    samples: Sequence[Sample],
    percentiles: Iterable[float] = (0.50, 0.90),
    miss_value_s: Optional[float] = None,
) -> Dict[str, float]:
    """TTFT percentiles over ALL submitted requests, not just survivors.

    The harnesses' existing ttft_p50/p90 are survivor-only and stay as they
    are. This is the companion view: a dropped request is charged
    `miss_value_s` (default: the largest observed TTFT, i.e. "at least this
    bad") so that dropping requests cannot improve the percentile.

    Returns keys like ttft_all_p50_ms.
    """
    if not samples:
        return {}

    observed: List[float] = [t for ok, t, _ in samples if ok and t > 0]
    if not observed:
        return {}
    charge = miss_value_s if miss_value_s is not None else max(observed)

    full = sorted(t if (ok and t > 0) else charge for ok, t, _ in samples)

    out: Dict[str, float] = {}
    for p in percentiles:
        idx = min(int(p * len(full)), len(full) - 1)
        out[f"ttft_all_p{int(p * 100)}_ms"] = round(full[idx] * 1000, 1)
    return out
