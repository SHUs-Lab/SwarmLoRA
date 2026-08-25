#!/usr/bin/env python3
"""RQ3: SwarmLoRA SLO-attainment CDF, single trace category (paper Fig. 11).

TTFT (2s SLO) and TPOT (100ms SLO) CDFs over SwarmLoRA's per-request records.
The paper draws one curve per traffic category; the required-path artifact runs
one category at a time, so one curve is drawn here. Defaults to steady_heavy,
matching plot_rq3_trace.py.

TPOT is reconstructed per request as 1000/decode_throughput_tps.

Usage: python3 analysis/plot_rq3_slo_cdf.py [--trace NAME] [--output FILE]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import CAT_COLORS, CAT_MARKERS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TTFT_SLO_MS = 2000
TPOT_SLO_MS = 100


def _records(category):
    """Newest records_*.json under benchmark_results/10min_lorant_pw20/<cat>/."""
    d = os.path.join(REPO_ROOT, "benchmark_results", "10min_lorant_pw20", category)
    files = sorted(glob.glob(os.path.join(d, "records_*.json")))
    if not files:
        alt = os.path.join(d, "records.json")
        files = [alt] if os.path.exists(alt) else []
    if not files:
        return None
    raw = json.load(open(files[-1]))
    recs = raw if isinstance(raw, list) else raw.get("records", raw.get("requests", []))
    ttft, tpot = [], []
    for r in recs:
        if not r.get("success"):
            continue
        if r.get("ttft_ms") is not None:
            ttft.append(float(r["ttft_ms"]))
        tps = r.get("decode_throughput_tps")
        if tps:
            tpot.append(1000.0 / float(tps))
    return sorted(ttft), sorted(tpot)


def _cdf(ax, vals, color, marker, slo_ms, xlabel, xmax, slo_label, category):
    vals = np.array(vals)
    cdf = np.arange(1, len(vals) + 1) / len(vals) * 100
    ax.plot(vals, cdf, color=color, linewidth=1.2, marker=marker,
            markevery=max(len(vals) // 12, 1), markersize=4, label=category)
    attain = (vals <= slo_ms).mean() * 100
    ax.axvline(slo_ms, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(slo_ms * 1.02, 12, slo_label, rotation=90, va="bottom",
            fontsize=10, fontweight="bold")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(100))
    ax.set_ylabel("Fraction of submitted requests")
    ax.set_xlabel(xlabel, fontweight="bold")
    return attain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default="steady_heavy",
                        help="Trace category to plot (default: steady_heavy)")
    parser.add_argument("--output", default=None,
                        help="Output path (default: analysis/rq3_slo_cdf_<trace>.pdf)")
    args = parser.parse_args()
    category = args.trace
    output = args.output or os.path.join(
        REPO_ROOT, "analysis", f"rq3_slo_cdf_{category}.pdf")

    data = _records(category)
    if data is None:
        print(f"ERROR: no SwarmLoRA records for the '{category}' trace under "
              f"benchmark_results/10min_lorant_pw20/{category}/. Run "
              f"scripts/run_trace_driven.sh {category} first.", file=sys.stderr)
        sys.exit(1)
    ttft, tpot = data

    color, marker = CAT_COLORS[category], CAT_MARKERS[category]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.16, 3.0))
    a_ttft = _cdf(ax_a, ttft, color, marker, TTFT_SLO_MS,
                  "(a) User-facing TTFT (ms)", 6000, "2s SLO", category)
    a_tpot = _cdf(ax_b, tpot, color, marker, TPOT_SLO_MS,
                  "(b) TPOT (ms)", 200, "100ms SLO", category)

    print(f"SwarmLoRA '{category}' trace ({len(ttft)} requests):")
    print(f"  TTFT <= {TTFT_SLO_MS}ms SLO attainment: {a_ttft:.1f}%")
    print(f"  TPOT <= {TPOT_SLO_MS}ms SLO attainment: {a_tpot:.1f}%")

    ax_a.legend(loc="lower right", fontsize=10)
    fig.suptitle("RQ3: SwarmLoRA SLO attainment CDF (Fig. 11)", y=1.02)
    fig.tight_layout()
    fig.savefig(output)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
