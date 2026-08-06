#!/usr/bin/env python3
"""Pre-register the base model + all adapters into sllm_store format.

Run by setup.sh. The server's own registration path (sllm/model_downloader.py)
builds a throwaway base-model skeleton per adapter via `from_config`,
random-initializing 8B parameters on CPU -- ~5 min per adapter, since PyTorch's
fp16 CPU ops are poorly optimized.

This performs the same save_lora() registration but loads the real base model
once via `from_pretrained` and reuses it across adapters. Output lands where
the server's "already exists" check looks, so later runs skip registration.

Usage:
  python3 scripts/fast_register_adapters.py \
      --model-name meta-llama/Llama-3.1-8B-Instruct \
      --storage-path ./models
"""
import argparse
import gc
import os

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

from sllm_store.transformers import save_lora, save_model


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--storage-path", default="./models")
    p.add_argument("--raw-adapters-dir", default=None,
                    help="Defaults to <storage-path>/raw_adapters")
    return p.parse_args()


def main():
    args = get_args()
    storage_path = os.path.abspath(args.storage_path)
    raw_adapters_dir = args.raw_adapters_dir or os.path.join(storage_path, "raw_adapters")

    model_path = os.path.join(storage_path, "transformers", args.model_name)
    if os.path.exists(model_path):
        print(f"[skip] Base model already registered at {model_path}")
        base_model = None
    else:
        print(f"[load] Loading {args.model_name} from HuggingFace (real weights)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.float16, local_files_only=True
        )
        save_model(base_model, model_path)
        print(f"[save] Base model saved to {model_path}")

    adapter_names = sorted(os.listdir(raw_adapters_dir))
    pending = []
    for adapter_name in adapter_names:
        adapter_path = os.path.join(storage_path, "transformers", adapter_name)
        if os.path.exists(adapter_path):
            print(f"[skip] {adapter_name} already registered at {adapter_path}")
            continue
        pending.append(adapter_name)

    if pending and base_model is None:
        print(f"[load] Loading {args.model_name} from HuggingFace (real weights)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.float16, local_files_only=True
        )

    # save_lora() reads only PEFT's "default" adapter key, so each adapter must
    # be loaded under that name one at a time; unload() strips the injected
    # LoRA layers back to a clean base model in place (no disk reload) so the
    # next one doesn't stack on the previous.
    for adapter_name in pending:
        raw_path = os.path.join(raw_adapters_dir, adapter_name)
        adapter_path = os.path.join(storage_path, "transformers", adapter_name)
        print(f"[load] Attaching adapter {adapter_name} from {raw_path}...")
        peft_model = PeftModel.from_pretrained(
            base_model, raw_path, local_files_only=True
        )
        save_lora(peft_model, adapter_path)
        print(f"[save] {adapter_name} saved to {adapter_path}")
        base_model = peft_model.unload()

    if base_model is not None:
        del base_model
        gc.collect()
        torch.cuda.empty_cache()
    print("[done] All adapters registered.")


if __name__ == "__main__":
    main()
