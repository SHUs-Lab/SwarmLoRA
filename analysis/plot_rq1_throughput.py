#!/usr/bin/env python3
"""RQ1: aggregate throughput vs. request rate (paper Fig. 5).

Grouped bars, five Poisson rates (1-12 RPS) x three systems.

Usage: python3 analysis/plot_rq1_throughput.py [--output FILE]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import COLOR

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RATES = (1, 2, 4, 8, 12)


def _load(path):
    with open(path) as f:
        return json.load(f)


def swarmlora_series():
    """benchmark_results/throughput/poisson_rps{N}/summary.json."""
    out = {}
    for n in RATES:
        p = os.path.join(REPO_ROOT, "benchmark_results", "throughput",
                          f"poisson_rps{n}", "summary.json")
        if os.path.exists(p):
            out[n] = _load(p).get("tokens_per_second")
    return out if len(out) == len(RATES) else None


def serverlesslora_series():
    """baselines/serverlesslora/benchmark_results/throughput/rps{N}_metrics.json."""
    out = {}
    for n in RATES:
        p = os.path.join(REPO_ROOT, "baselines", "serverlesslora",
                          "benchmark_results", "throughput", f"rps{n}_metrics.json")
        if os.path.exists(p):
            out[n] = _load(p).get("tokens_per_second")
    return out if len(out) == len(RATES) else None


def serverlessllm_series():
    """baselines/serverlessllm/.../sllm_optimized_poisson_rps{N}_*inst_*gpu.json.

    Reports raw completion tokens/s, the same measure as the other two systems.

    Instance/GPU counts are host-derived, hence wildcards; newest mtime wins.
    """
    base = os.path.join(REPO_ROOT, "baselines", "serverlessllm",
                         "benchmark_results", "trace_driven")
    out = {}
    for n in RATES:
        files = sorted(glob.glob(os.path.join(
            base, f"sllm_optimized_poisson_rps{n}_*inst_*gpu.json")),
            key=os.path.getmtime)
        if files:
            s = _load(files[-1]).get("summary", {})
            out[n] = s.get("tokens_per_second")
    return out if len(out) == len(RATES) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=os.path.join(
        REPO_ROOT, "analysis", "rq1_throughput.pdf"))
    args = parser.parse_args()

    data = {
        "ServerlessLLM": serverlessllm_series(),
        "ServerlessLoRA": serverlesslora_series(),
        "SwarmLoRA": swarmlora_series(),
    }
    missing = [n for n, v in data.items() if v is None]
    if missing:
        print(f"ERROR: no RQ1 throughput data for: {', '.join(missing)}. Run "
              f"scripts/run_throughput.sh and each missing baseline's "
              f"scripts/run_throughput.sh first.", file=sys.stderr)
        sys.exit(1)

    for name, series in data.items():
        print(f"{name}: " + ", ".join(
            f"{r}rps={series[r]:.1f}tok/s" for r in RATES))

    x = np.arange(len(RATES))
    w = 0.25
    order = ["ServerlessLLM", "ServerlessLoRA", "SwarmLoRA"]
    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    for i, name in enumerate(order):
        vals = [data[name][r] for r in RATES]
        ax.bar(x + (i - 1) * w, vals, w, color=COLOR[name],
               edgecolor="black", linewidth=0.5, label=name)

    ax.set_xlabel("Request rate (RPS)")
    ax.set_ylabel("Aggregate throughput (tok/s)")
    ax.set_xticks(x)
    ax.set_xticklabels(RATES)
    top = max(data["SwarmLoRA"][r] for r in RATES) * 1.15
    ax.set_ylim(0, top)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig(args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
