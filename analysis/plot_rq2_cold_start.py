#!/usr/bin/env python3
"""RQ2: cold-start and hot-swap latency (paper Fig. 6).

Horizontal bars. Each system measures cold start differently, so this plots the
most comparable number per system:

  Standard Load:   benchmarks/standard_load_cold_start.py (first_load, cold)
  ServerlessLoRA:  1 worker / 1 concurrent cold start
  ServerlessLLM:   sllm_store steady-state load (excludes one-time registration)
  SwarmLoRA:       sequential worker spawn, and adapter hot-swap

Usage: python3 analysis/plot_rq2_cold_start.py [--output FILE]
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


def _load(path):
    with open(path) as f:
        return json.load(f)


def standard_load_ms():
    p = os.path.join(REPO_ROOT, "benchmark_results", "cold_start",
                      "standard_load_cold_start.json")
    if not os.path.exists(p):
        return None
    s = _load(p).get("first_load", {}).get("total_s")
    return s * 1000 if s is not None else None


def swarmlora_spawn_swap_ms():
    """benchmark_results/cold_start/cold_start_both_10_*.json (newest);
    returns (spawn_avg_ms, swap_avg_ms)."""
    files = sorted(glob.glob(os.path.join(
        REPO_ROOT, "benchmark_results", "cold_start", "cold_start_both_10_*.json")))
    if not files:
        return None, None
    d = _load(files[-1])
    spawn = d.get("spawn", {}).get("sequential", {}).get("avg_ms")
    swap = d.get("swap", {}).get("sequential", {}).get("avg_ms")
    return spawn, swap


def serverlesslora_ms():
    p = os.path.join(REPO_ROOT, "baselines", "serverlesslora",
                      "benchmark_results", "cold_start", "cold_start_1w_1c.json")
    if not os.path.exists(p):
        return None
    return _load(p).get("cold_start_ms", {}).get("mean")


def serverlessllm_ms():
    """Steady-state rather than first_load: first_load additionally pays a
    one-time sllm_store registration, which is a deployment-time cost, not a
    per-cold-start one. Every other system here is measured with its setup
    already done (BMS/aggregator running), so steady-state is the comparable
    number."""
    p = os.path.join(REPO_ROOT, "baselines", "serverlessllm",
                      "benchmark_results", "cold_start",
                      "sllm_lora_cold_start_1gpu_reconstructed.json")
    if not os.path.exists(p):
        return None
    d = _load(p)
    s = d.get("avg_steady_state", d.get("first_load", {})).get("total_s")
    return s * 1000 if s is not None else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=os.path.join(
        REPO_ROOT, "analysis", "rq2_cold_start.pdf"))
    args = parser.parse_args()

    spawn, swap = swarmlora_spawn_swap_ms()
    rows = [
        ("Standard Load", standard_load_ms(), "#b0b0b0"),
        ("ServerlessLoRA", serverlesslora_ms(), COLOR["ServerlessLoRA"]),
        ("ServerlessLLM", serverlessllm_ms(), "#6baed6"),
        ("SwarmLoRA (spawn)", spawn, COLOR["SwarmLoRA"]),
        ("SwarmLoRA (hot swap)", swap, "#08519c"),
    ]
    missing = [name for name, v, _ in rows if v is None]
    if missing:
        print(f"ERROR: no RQ2 data for: {', '.join(missing)}. Run "
              f"scripts/run_cold_start.sh, scripts/run_standard_load_cold_start.sh, "
              f"and each baseline's scripts/run_cold_start.sh first.",
              file=sys.stderr)
        sys.exit(1)

    for name, v, _ in rows:
        print(f"{name}: {v:.0f} ms")

    labels = [r[0] for r in rows]
    vals = np.array([r[1] for r in rows])
    colors = [r[2] for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    bars = ax.barh(y, vals, height=0.72, edgecolor="black", linewidth=0.5)
    for bar, c in zip(bars, colors):
        bar.set_facecolor(c)
    for bar, v in zip(bars, vals):
        ax.text(v + vals.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.0f}", va="center", ha="left", fontsize=10)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cold-start latency (ms)")
    ax.invert_yaxis()
    ax.set_xlim(0, vals.max() * 1.12)
    fig.tight_layout()
    fig.savefig(args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
