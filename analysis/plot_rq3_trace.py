#!/usr/bin/env python3
"""RQ3: trace-driven evaluation, single trace category (paper Figs. 6-7, 9-10 subset).

Four panels across all three systems: acceptance, throughput, TTFT (P50/P90),
TPOT (P50/P90). The paper plots these as lines over all five traffic
categories; the required-path artifact runs one category at a time, so each
metric is a bar here. Defaults to steady_heavy, the required-path category --
it saturates all three systems, so the architectural differences the paper
reports are visible. Drop the trace argument from run_trace_driven.sh for the
full five-category version.

Usage: python3 analysis/plot_rq3_trace.py [--trace NAME] [--output FILE]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import COLOR

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ("acceptance_rate", "tokens_per_second",
        "ttft_p50_ms", "ttft_p90_ms", "tpot_p50_ms", "tpot_p90_ms")


def _fmt_dur(ms, _pos=None):
    """Log-axis tick labels: sub-second in ms, the rest in seconds."""
    if ms < 1000:
        return f"{ms:g}ms"
    return f"{ms / 1000:g}s"


def _load(path):
    with open(path) as f:
        return json.load(f)


def _pick(d):
    return {k: d[k] for k in KEYS} if all(k in d for k in KEYS) else None


def swarmlora(trace):
    p = os.path.join(REPO_ROOT, "benchmark_results", "10min_lorant_pw20",
                      trace, "summary.json")
    return _pick(_load(p)) if os.path.exists(p) else None


def serverlesslora(trace):
    p = os.path.join(REPO_ROOT, "baselines", "serverlesslora",
                      "benchmark_results", "trace_driven", f"{trace}_metrics.json")
    return _pick(_load(p)) if os.path.exists(p) else None


def serverlessllm(trace):
    """Newest sllm_optimized_{trace}_*inst_*gpu.json (host-derived wildcards).

    Reports goodput (paper convention, ServerlessLLM only -- see
    plot_rq1_throughput.py). A goodput correction needs a lighter-load
    baseline, which a standalone single-trace run lacks; there
    goodput_tok_per_s equals raw tokens/s. The full five-trace suite
    yields a corrected value.
    """
    files = sorted(glob.glob(os.path.join(
        REPO_ROOT, "baselines", "serverlessllm", "benchmark_results",
        "trace_driven", f"sllm_optimized_{trace}_*inst_*gpu.json")),
        key=os.path.getmtime)
    if not files:
        return None
    s = _load(files[-1]).get("summary", {})
    m = _pick(s)
    if m is not None:
        m["tokens_per_second"] = s.get("goodput_tok_per_s", m["tokens_per_second"])
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default="steady_heavy",
                         help="Trace category to plot (default: steady_heavy)")
    parser.add_argument("--output", default=None,
                         help="Output path (default: analysis/rq3_<trace>_trace.pdf)")
    args = parser.parse_args()
    output = args.output or os.path.join(
        REPO_ROOT, "analysis", f"rq3_{args.trace}_trace.pdf")

    data = {
        "ServerlessLLM": serverlessllm(args.trace),
        "ServerlessLoRA": serverlesslora(args.trace),
        "SwarmLoRA": swarmlora(args.trace),
    }
    missing = [n for n, v in data.items() if v is None]
    if missing:
        print(f"ERROR: no RQ3 '{args.trace}' trace data for: {', '.join(missing)}. "
              f"Run scripts/run_trace_driven.sh {args.trace} and each baseline's "
              f"equivalent first.", file=sys.stderr)
        sys.exit(1)

    order = ["ServerlessLLM", "ServerlessLoRA", "SwarmLoRA"]
    for n in order:
        m = data[n]
        print(f"{n}: accept={m['acceptance_rate']*100:.1f}%  "
              f"{m['tokens_per_second']:.1f}tok/s  "
              f"ttft(p50/p90)={m['ttft_p50_ms']:.0f}/{m['ttft_p90_ms']:.0f}ms  "
              f"tpot(p50/p90)={m['tpot_p50_ms']:.1f}/{m['tpot_p90_ms']:.1f}ms")

    x = np.arange(len(order))
    colors = [COLOR[n] for n in order]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0))

    # Panel (a): acceptance
    axes[0].bar(x, [data[n]["acceptance_rate"] * 100 for n in order],
                color=colors, edgecolor="black", linewidth=0.5, width=0.6)
    axes[0].set_ylabel("Acceptance rate (%)")
    axes[0].set_ylim(0, 108)
    axes[0].set_title("(a) Acceptance", fontsize=11, loc="left")

    # Panel (b): throughput. Floor the axis at 1000 tok/s so runs are visually
    # comparable across traces, but grow it if a system exceeds that.
    tput = [data[n]["tokens_per_second"] for n in order]
    axes[1].bar(x, tput, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
    axes[1].set_ylabel("Throughput (tok/s)")
    axes[1].set_ylim(0, max(1000, max(tput) * 1.1))
    axes[1].set_title("(b) Throughput", fontsize=11, loc="left")

    # Panels (c) TTFT, (d) TPOT: grouped P50 (solid) + P90 (hatched).
    # TTFT spans ~3 orders of magnitude under heavy load (a saturated baseline
    # reaches tens of seconds while SwarmLoRA stays sub-second), so it uses a
    # log axis -- matching the paper's TTFT figure. TPOT stays linear.
    for ax, (p50k, p90k, ylabel, title, log) in zip(
            axes[2:], [("ttft_p50_ms", "ttft_p90_ms", "TTFT", "(c) TTFT", True),
                        ("tpot_p50_ms", "tpot_p90_ms", "TPOT (ms)", "(d) TPOT", False)]):
        w = 0.38
        vals = [data[n][p50k] for n in order] + [data[n][p90k] for n in order]
        bottom = 10 ** np.floor(np.log10(min(vals))) if log else 0
        ax.bar(x - w / 2, [data[n][p50k] for n in order], w, color=colors,
               edgecolor="black", linewidth=0.5, bottom=bottom if log else 0)
        ax.bar(x + w / 2, [data[n][p90k] for n in order], w, color=colors,
               edgecolor="black", linewidth=0.5, hatch="////", alpha=0.85,
               bottom=bottom if log else 0)
        if log:
            ax.set_yscale("log")
            ax.set_ylim(bottom, max(vals) * 2.5)
            ax.yaxis.set_major_formatter(FuncFormatter(_fmt_dur))
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, loc="left")
        # P50/P90 key sits above the axes, alongside the panel title, so it
        # never overlaps the bars.
        ax.legend(handles=[Patch(facecolor="#cccccc", edgecolor="black", label="P50"),
                            Patch(facecolor="#cccccc", edgecolor="black",
                                  hatch="////", label="P90")],
                  loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2,
                  fontsize=8, frameon=False, handlelength=1.4,
                  columnspacing=1.0, borderpad=0.2)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=20, ha="right", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)

    fig.suptitle(f'RQ3: "{args.trace}" trace, 10-min replay', y=1.02)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
