#!/usr/bin/env python3
"""Regenerate a deployment config's `gpus:` list to match the GPUs actually
visible on this host, leaving every other key untouched.

Port scheme (must stay consistent with the shipped configs):
  base_model_server_port = 50050 + device_id
  agent_port              = 7000  + device_id
  worker_base_port        = 6000  + device_id * 100

Usage: adapt_gpu_config.py <template.yaml> <output.yaml>
"""
import subprocess
import sys

import yaml


def detected_gpu_count() -> int:
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                          text=True).stdout
    return len([line for line in out.splitlines() if line.strip()])


def main():
    template, output = sys.argv[1], sys.argv[2]
    with open(template) as f:
        cfg = yaml.safe_load(f)

    n = detected_gpu_count()
    if n == 0:
        sys.exit("ERROR: no GPUs detected by nvidia-smi")

    cfg["gpus"] = [
        {
            "device_id": d,
            "base_model_server_port": 50050 + d,
            "agent_port": 7000 + d,
            "worker_base_port": 6000 + d * 100,
        }
        for d in range(n)
    ]

    with open(output, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[adapt_gpu_config] {template} -> {output}: {n} GPU(s) detected")


if __name__ == "__main__":
    main()
