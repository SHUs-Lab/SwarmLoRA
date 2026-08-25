#!/usr/bin/env python3
"""Standard-load cold-start baseline for Fig. 8 (the "Standard Load" bar).

Measures the cold start of the *default* serving path with no optimization at
all: plain HuggingFace `AutoModelForCausalLM.from_pretrained` reading the full
~16 GB base model from storage to GPU, then a PEFT adapter on top, then the
first token. This is the un-optimized reference point the three systems
(ServerlessLLM's checkpoint store, ServerlessLoRA's shared base model,
SwarmLoRA's Preload Engine) each improve on.

Breakdown schema matches baselines/serverlessllm/benchmarks/cold_start.py
(base_s / adapter_s / first_token_s / total_s) so the Fig. 8 bars are
apples-to-apples.

The headline number is the FIRST run (base_s dominated by real disk I/O of the
16 GB checkpoint) -- that is what "standard load" costs on a cold box. Later
runs read from the OS page cache and are reported only as context.

Usage:
  python standard_load_cold_start.py \
      --model-name meta-llama/Llama-3.1-8B-Instruct \
      --adapter-path ../sim-adapters/pool-10-r16/lora-0 \
      --num-runs 4 \
      --output benchmark_results/cold_start/standard_load_cold_start.json
"""
import argparse
import json
import os
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--adapter-path", default="../sim-adapters/pool-10-r16/lora-0")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-runs", type=int, default=4)
    p.add_argument("--output",
                   default="benchmark_results/cold_start/standard_load_cold_start.json")
    p.add_argument("--drop-caches", action="store_true",
                   help="Attempt to drop the OS page cache before each run so "
                        "every run pays real disk I/O (needs root / sudo; "
                        "silently skipped if unavailable).")
    return p.parse_args()


def _maybe_drop_caches(enabled):
    """Best-effort page-cache drop so base_s reflects true disk read, not the
    page cache. Only the first run is disk-cold otherwise; the paper's number
    is a genuine cold read, so offer this for parity. Silently no-ops without
    privileges."""
    if not enabled:
        return
    try:
        os.sync()
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except OSError:
        pass  # not root, or not Linux -- fall back to whatever the cache holds


def one_run(model_name, adapter_path, device, drop_caches):
    # Imported lazily and per-process-once; the import cost is not part of the
    # model-load measurement (it's amortized across every request the process
    # ever serves, exactly as in the other systems' breakdowns).
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    _maybe_drop_caches(drop_caches)

    # --- Base model load (the 16 GB checkpoint read + weight upload) ---
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to(device)
    model.eval()
    torch.cuda.synchronize(device)
    base_s = time.perf_counter() - t0

    # --- Adapter load (PEFT wrap of the 161 MB adapter) ---
    t1 = time.perf_counter()
    model = PeftModel.from_pretrained(model, adapter_path)
    torch.cuda.synchronize(device)
    adapter_s = time.perf_counter() - t1

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    # --- First token ---
    inputs = tokenizer("Hello", return_tensors="pt").to(device)
    t2 = time.perf_counter()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1, do_sample=False)
    torch.cuda.synchronize(device)
    first_token_s = time.perf_counter() - t2

    total_s = base_s + adapter_s + first_token_s

    # Free before the next run so each run starts from a clean allocator state.
    del model
    torch.cuda.empty_cache()

    return {
        "base_s": round(base_s, 3),
        "adapter_s": round(adapter_s, 3),
        "first_token_s": round(first_token_s, 3),
        "total_s": round(total_s, 3),
    }


def main():
    args = parse_args()
    runs = []
    for i in range(args.num_runs):
        # Each run reloads the model in a fresh state within this process; the
        # first is the cold, disk-dominated one that is the headline number.
        print(f"[run {i + 1}/{args.num_runs}] loading {args.model_name} ...")
        r = one_run(args.model_name, args.adapter_path, args.device, args.drop_caches)
        print(f"  base={r['base_s']}s adapter={r['adapter_s']}s "
              f"first_token={r['first_token_s']}s total={r['total_s']}s")
        runs.append(r)

    first_load = runs[0]
    steady = runs[1:] if len(runs) > 1 else []
    avg_steady = (
        {
            k: round(sum(r[k] for r in steady) / len(steady), 3)
            for k in ("base_s", "adapter_s", "first_token_s", "total_s")
        }
        if steady else None
    )

    out = {
        "model": args.model_name,
        "device": args.device,
        "adapter_path": args.adapter_path,
        "runs": runs,
        "first_load": first_load,
        "avg_steady_state": avg_steady,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nHeadline (cold, disk-dominated) total: {first_load['total_s']}s")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
