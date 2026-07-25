"""Cold-start latency: base model load + LoRA adapter load + first token.

Best-effort reconstruction (SwarmLoRA artifact), not part of upstream
ServerlessLLM -- the original script behind the paper's cited 1.77s cold
start could not be recovered, so this rebuilds the same measurement via
sllm_store's own documented interface (sllm_store/examples/). Should land
in the same range but isn't guaranteed exact.

The sllm-store server keeps a loaded model registered/pinned in its memory
pool, so only the first --num-runs iteration is a genuine cold load (real
disk I/O); subsequent iterations reload from the warm pool and are much
faster. Results report both: "first_load" (one-time registration cost) and
"avg_steady_state" (average of runs 2..N, the recurring reload cost -- this
is the number comparable to the paper's cited figure).

Usage: bash scripts/run_cold_start.sh [--prepare]
"""
import argparse
import gc
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from sllm_store.transformers import load_lora, load_model, save_lora, save_model


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter-path", default=None,
                         help="Local safetensors LoRA adapter dir (only needed with --prepare)")
    parser.add_argument("--storage-path", default="./models",
                         help="sllm_store on-disk model directory")
    parser.add_argument("--adapter-name", default="cold_start_adapter")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-runs", type=int, default=4)
    parser.add_argument("--prepare", action="store_true",
                         help="One-time: convert model+adapter into sllm_store format, then exit")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def prepare(args):
    """Convert the base model and adapter into sllm_store's optimized format.
    Not timed -- this mirrors a one-time deployment step, not cold start itself."""
    print(f"[prepare] Loading {args.model_name} from HuggingFace...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, local_files_only=True
    )
    model_path = os.path.join(args.storage_path, args.model_name)
    save_model(base_model, model_path)
    print(f"[prepare] Base model saved to {model_path}")

    if args.adapter_path:
        print(f"[prepare] Loading adapter from {args.adapter_path}...")
        peft_model = PeftModel.from_pretrained(
            base_model, args.adapter_path, local_files_only=True
        )
        adapter_path = os.path.join(args.storage_path, args.adapter_name)
        save_lora(peft_model, adapter_path)
        print(f"[prepare] Adapter saved to {adapter_path}")

    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    print("[prepare] Done. Run again without --prepare to measure cold start.")


def measure_one_run(args, tokenizer):
    t0 = time.time()
    model = load_model(
        args.model_name,
        storage_path=args.storage_path,
        device_map=args.device,
        torch_dtype=torch.float16,
    )
    t1 = time.time()

    model = load_lora(
        model,
        args.adapter_name,
        args.adapter_name,
        device_map=args.device,
        storage_path=args.storage_path,
        torch_dtype=torch.float16,
    )
    t2 = time.time()

    inputs = tokenizer("The capital of France is", return_tensors="pt").to(args.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1, adapter_names=[args.adapter_name])
    torch.cuda.synchronize()
    t3 = time.time()

    run = {
        "base_s": round(t1 - t0, 3),
        "adapter_s": round(t2 - t1, 3),
        "first_token_s": round(t3 - t2, 3),
        "total_s": round(t3 - t0, 3),
    }

    del model, inputs
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return run


def main():
    args = get_args()

    if args.prepare:
        prepare(args)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    runs = []
    for i in range(args.num_runs):
        print(f"[cold_start] Run {i + 1}/{args.num_runs}...")
        run = measure_one_run(args, tokenizer)
        print(f"  base_s={run['base_s']} adapter_s={run['adapter_s']} "
              f"first_token_s={run['first_token_s']} total_s={run['total_s']}")
        runs.append(run)

    # The sllm-store server keeps a model registered/pinned in its memory
    # pool after the first load_model() call, so run 1 (real disk I/O, one
    # time per server lifetime) and runs 2+ (warm-pool reload) measure two
    # different things. Blending them into one average is misleading --
    # report both, and use the steady-state runs as the headline comparison
    # figure (that's the recurring cost a fair comparison cares about; the
    # one-time registration cost isn't part of what SwarmLoRA's Preload
    # Engine spawn latency is being compared against either).
    first_load = runs[0]
    steady_state_runs = runs[1:] if len(runs) > 1 else runs
    avg = {
        k: round(sum(r[k] for r in steady_state_runs) / len(steady_state_runs), 3)
        for k in ("base_s", "adapter_s", "first_token_s", "total_s")
    }
    result = {
        "model": args.model_name,
        "gpus": 1,
        "device": args.device,
        "runs": runs,
        "first_load": first_load,
        "avg_steady_state": avg,
    }
    print(f"\n[cold_start] First load (cold registration): {first_load}")
    print(f"[cold_start] Steady-state average (n={len(steady_state_runs)}): {avg}")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[cold_start] Results saved to {args.output}")


if __name__ == "__main__":
    main()
