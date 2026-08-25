#!/usr/bin/env python3
"""Baseline TPOT measurement via HuggingFace + PEFT (no split architecture)."""

import argparse
import json
import os
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Baseline TPOT measurement")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter", default="../sim-adapters/pool-10-r16/lora-0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--decode-tokens", type=int, default=100)
    parser.add_argument("--warmup-tokens", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("=" * 72)
    print("Baseline TPOT: Standard LoRA Inference (no split architecture)")
    print("=" * 72)
    print(f"Model:         {args.model}")
    print(f"Adapter:       {args.adapter}")
    print(f"Device:        {args.device}")
    print(f"Decode tokens: {args.decode_tokens}")
    print(f"Warmup tokens: {args.warmup_tokens}")
    print(f"Repeats:       {args.repeats}")
    print()

    # Load model
    print("Loading base model...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        local_files_only=True,
    )
    base_model_load_s = time.time() - t0
    print(f"  Base model loaded in {base_model_load_s:.1f}s")

    # Load LoRA adapter
    print("Loading LoRA adapter...")
    t0 = time.time()
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    model.eval()
    adapter_load_s = time.time() - t0
    print(f"  Adapter loaded in {adapter_load_s:.1f}s")

    # Check adapter rank
    adapter_config_path = os.path.join(args.adapter, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path) as f:
            cfg = json.load(f)
        print(f"  Adapter rank: {cfg.get('r', '?')}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare prompt
    prompt = ("a Lambda-class T-4a shuttle and a few TIE-fighter escorts were being "
              "prepared for Lizzie, her bodyguard Mila, and Alyssa, along with a pair "
              "of imperial Deathtroopers, and 8 regular Imperial Army Troopers.")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(args.device)
    prompt_len = input_ids.shape[1]
    print(f"  Prompt: {prompt_len} tokens")
    print()

    # GPU memory
    mem_mb = torch.cuda.memory_allocated(args.device) / 1024 / 1024
    print(f"  GPU memory: {mem_mb:.0f} MB")
    print()

    all_tpots = []

    for rep in range(args.repeats):
        # Warmup: generate some tokens to fill KV cache
        print(f"  Rep {rep + 1}/{args.repeats}: warming up ({args.warmup_tokens} tokens)...")
        with torch.no_grad():
            _ = model.generate(
                input_ids,
                max_new_tokens=args.warmup_tokens,
                do_sample=False,
                use_cache=True,
            )

        # Profile: generate tokens one at a time, measuring each
        print(f"  Profiling ({args.decode_tokens} tokens)...")

        with torch.no_grad():
            # Prefill
            outputs = model(input_ids, use_cache=True)
            past_kv = outputs.past_key_values
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)

            token_times = []
            for i in range(args.decode_tokens):
                torch.cuda.synchronize()
                t_start = time.perf_counter()

                outputs = model(next_token, past_key_values=past_kv, use_cache=True)
                next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
                past_kv = outputs.past_key_values

                torch.cuda.synchronize()
                t_end = time.perf_counter()

                token_times.append((t_end - t_start) * 1000)  # ms

            # Skip first few tokens (warmup within decode)
            token_times = token_times[5:]
            avg_tpot = np.mean(token_times)
            p50_tpot = np.median(token_times)
            p90_tpot = np.percentile(token_times, 90)
            all_tpots.extend(token_times)

            print(f"    TPOT: avg={avg_tpot:.2f}ms  p50={p50_tpot:.2f}ms  p90={p90_tpot:.2f}ms")

    # Summary
    print()
    print("=" * 72)
    print("RESULTS: Baseline TPOT (standard LoRA inference)")
    print("=" * 72)
    print()
    print(f"  Samples:  {len(all_tpots)} decode tokens")
    print(f"  TPOT avg: {np.mean(all_tpots):.2f} ms")
    print(f"  TPOT p50: {np.median(all_tpots):.2f} ms")
    print(f"  TPOT p90: {np.percentile(all_tpots, 90):.2f} ms")
    print(f"  TPOT p99: {np.percentile(all_tpots, 99):.2f} ms")
    print(f"  Decode throughput: {1000 / np.mean(all_tpots):.1f} tok/s")
    print()

    # Save
    if args.output:
        result = {
            "config": {
                "model": args.model,
                "adapter": args.adapter,
                "device": args.device,
                "decode_tokens": args.decode_tokens,
                "repeats": args.repeats,
            },
            "base_model_load_s": float(base_model_load_s),
            "adapter_load_s": float(adapter_load_s),
            "tpot_avg_ms": float(np.mean(all_tpots)),
            "tpot_p50_ms": float(np.median(all_tpots)),
            "tpot_p90_ms": float(np.percentile(all_tpots, 90)),
            "tpot_p99_ms": float(np.percentile(all_tpots, 99)),
            "decode_tps": float(1000 / np.mean(all_tpots)),
            "n_samples": len(all_tpots),
            "token_times_ms": [float(t) for t in all_tpots],
        }
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
