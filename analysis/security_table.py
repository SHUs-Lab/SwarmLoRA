#!/usr/bin/env python3
"""RQ6: security isolation results, Table III (SwarmLoRA vs. S-LoRA).

Reads the four JSONs written by scripts/run_security.sh and renders Table III
both as a text table (stdout) and a PDF (analysis/security_table.pdf):

  Weight theft (cosine sim)   -- blind_attack_{sys}.json data_theft.wrs_auth
  KV cache & activations      -- data_theft.kcrr_live + aa_activations recovery
  Fault blast radius (N/7)    -- fault_{sys}.json, worst mode across kill/segv/abort

Usage: python3 analysis/security_table.py [--output FILE]
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEC_DIR = os.path.join(REPO_ROOT, "benchmark_results", "security")


def _load(name):
    p = os.path.join(SEC_DIR, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _weight_theft(attack):
    """Cosine similarity of stolen vs. real victim weights (0 = no theft)."""
    wrs = attack.get("data_theft", {}).get("wrs_auth", {})
    cos = wrs.get("victim_weight_cosine_sim_sorted")
    if cos is not None:
        return f"{cos:.3f}"
    return f"{wrs.get('recovery', 0.0):.1f}"


def _kv_act(attack):
    """KV cache + activation recovery -> Full / None / fraction."""
    dt = attack.get("data_theft", {})
    kv = dt.get("kcrr_live", {}).get("recovery", 0.0)
    aa = dt.get("aa_activations", {}).get("recovery", 0.0)
    lo, hi = min(kv, aa), max(kv, aa)
    if hi <= 0.01:
        return "None"
    if lo >= 0.99:
        return "Full"
    return f"{hi:.2f}"


def _blast(fault):
    """Worst blast radius across fault modes -> 'BR (survivors/total)'."""
    modes = fault.get("fault_modes", {})
    total = fault.get("num_tenants", 8) - 1
    worst_br, worst_affected = 0.0, 0
    for m in modes.values():
        br = m.get("blast_radius", 0.0)
        if br >= worst_br:
            worst_br = br
            worst_affected = m.get("affected_cotenants", 0)
    survivors = total - worst_affected
    return f"{worst_br:.1f} ({survivors}/{total})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=os.path.join(
        REPO_ROOT, "analysis", "security_table.pdf"))
    args = parser.parse_args()

    files = {
        "attack_swarm": "blind_attack_swarmlora.json",
        "attack_slora": "blind_attack_slora.json",
        "fault_swarm": "fault_swarmlora.json",
        "fault_slora": "fault_slora.json",
    }
    loaded = {k: _load(v) for k, v in files.items()}
    missing = [files[k] for k, v in loaded.items() if v is None]
    if missing:
        print(f"ERROR: missing security results: {', '.join(missing)}. Run "
              f"scripts/run_security.sh (and baselines/slora/setup.sh for the "
              f"S-LoRA column) first.", file=sys.stderr)
        sys.exit(1)

    rows = [
        ("Weight theft (cosine sim)",
         _weight_theft(loaded["attack_slora"]),
         _weight_theft(loaded["attack_swarm"])),
        ("KV cache & activations",
         _kv_act(loaded["attack_slora"]),
         _kv_act(loaded["attack_swarm"])),
        ("Fault blast radius",
         _blast(loaded["fault_slora"]),
         _blast(loaded["fault_swarm"])),
    ]

    header = ("Metric", "S-LoRA", "SwarmLoRA")
    w0 = max(len(header[0]), *(len(r[0]) for r in rows))
    w1 = max(len(header[1]), *(len(r[1]) for r in rows))
    w2 = max(len(header[2]), *(len(r[2]) for r in rows))
    line = f"+-{'-'*w0}-+-{'-'*w1}-+-{'-'*w2}-+"
    print("\nTable III -- Security isolation results")
    print(line)
    print(f"| {header[0]:<{w0}} | {header[1]:<{w1}} | {header[2]:<{w2}} |")
    print(line)
    for r in rows:
        print(f"| {r[0]:<{w0}} | {r[1]:<{w1}} | {r[2]:<{w2}} |")
    print(line)

    # PDF rendering
    fig, ax = plt.subplots(figsize=(6.4, 1.8))
    ax.axis("off")
    tbl = ax.table(cellText=[list(r) for r in rows],
                   colLabels=header, cellLoc="center", loc="center",
                   colWidths=[0.46, 0.27, 0.27])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("black")
        if row == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(fontweight="bold")
        elif col == 2:  # SwarmLoRA column
            cell.set_facecolor("#eaf2fb")
    ax.set_title("Table III: Security isolation results", pad=12)
    fig.tight_layout()
    fig.savefig(args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
