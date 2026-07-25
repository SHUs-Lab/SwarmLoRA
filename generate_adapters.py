"""Generate synthetic LoRA adapters for benchmarking."""
import argparse
import json
import math
import os

import torch
from safetensors.torch import save_file

# Model dimension presets
MODEL_PRESETS = {
    "llama-3.1-8b": {
        "num_layers": 32,
        "modules": {
            "self_attn.q_proj": (4096, 4096),
            "self_attn.k_proj": (4096, 1024),
            "self_attn.v_proj": (4096, 1024),
            "self_attn.o_proj": (4096, 4096),
            "mlp.gate_proj":    (4096, 14336),
            "mlp.up_proj":      (4096, 14336),
            "mlp.down_proj":    (14336, 4096),
        },
    },
    "mistral-7b": {  # Same dims as Llama-3.1-8B
        "num_layers": 32,
        "modules": {
            "self_attn.q_proj": (4096, 4096),
            "self_attn.k_proj": (4096, 1024),
            "self_attn.v_proj": (4096, 1024),
            "self_attn.o_proj": (4096, 4096),
            "mlp.gate_proj":    (4096, 14336),
            "mlp.up_proj":      (4096, 14336),
            "mlp.down_proj":    (14336, 4096),
        },
    },
    "llama-2-13b": {
        "num_layers": 40,
        "modules": {
            "self_attn.q_proj": (5120, 5120),
            "self_attn.k_proj": (5120, 5120),
            "self_attn.v_proj": (5120, 5120),
            "self_attn.o_proj": (5120, 5120),
            "mlp.gate_proj":    (5120, 13824),
            "mlp.up_proj":      (5120, 13824),
            "mlp.down_proj":    (13824, 5120),
        },
    },
}

# Default (backward compat)
NUM_LAYERS = 32
MODULES = MODEL_PRESETS["llama-3.1-8b"]["modules"]


def get_model_dims(base_model: str):
    """Auto-detect model preset from base_model string."""
    bm = base_model.lower()
    if "13b" in bm:
        return MODEL_PRESETS["llama-2-13b"]
    elif "mistral" in bm:
        return MODEL_PRESETS["mistral-7b"]
    else:
        return MODEL_PRESETS["llama-3.1-8b"]

def generate_adapter(rank: int, seed: int = 42, num_layers: int = NUM_LAYERS, modules: dict = None) -> dict:
    """Generate one adapter's weights."""
    if modules is None:
        modules = MODULES
    torch.manual_seed(seed)
    weights = {}
    for layer_idx in range(num_layers):
        for module_name, (in_feat, out_feat) in modules.items():
            prefix = f"base_model.model.model.layers.{layer_idx}.{module_name}"

            # lora_A: Kaiming uniform (same as nn.Linear default)
            A = torch.empty(rank, in_feat)
            bound = 1.0 / math.sqrt(in_feat)
            A.uniform_(-bound, bound)

            # lora_B: zeros + small noise to simulate trained adapter
            B = torch.randn(out_feat, rank) * 0.004

            weights[f"{prefix}.lora_A.weight"] = A.to(torch.float32)
            weights[f"{prefix}.lora_B.weight"] = B.to(torch.float32)

    return weights


def make_config(rank: int, base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct") -> dict:
    return {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": rank,  # alpha = rank → scaling = 1.0
        "lora_dropout": 0,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "rank_pattern": {},
        "revision": None,
        "target_modules": [
            "gate_proj", "down_proj", "up_proj",
            "o_proj", "q_proj", "k_proj", "v_proj"
        ],
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }


"""Real-world rank popularity (by request volume):
  r8: 40-50%, r16: 20-25%, r4: 10-15%, r32: ~10%, r64: 5-10%

Round-robin pattern per 10 adapters ensures every popularity decile
in a Zipf-skewed trace sees the same rank mix:
  [8, 16, 8, 4, 8, 32, 8, 64, 8, 16]
  → 5×r8, 2×r16, 1×r4, 1×r32, 1×r64
"""
MIXED_PATTERN = [8, 16, 8, 4, 8, 32, 8, 64, 8, 16]


def get_mixed_rank(adapter_idx: int) -> int:
    return MIXED_PATTERN[adapter_idx % len(MIXED_PATTERN)]


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic LoRA adapters")
    parser.add_argument("--rank", type=int, default=None, help="LoRA rank for uniform pool (e.g., 8, 32, 64)")
    parser.add_argument("--mixed", action="store_true", help="Use round-robin mixed-rank pattern")
    parser.add_argument("--num", type=int, default=50, help="Number of adapter copies")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--start", type=int, default=0, help="Starting index for lora-N naming")
    parser.add_argument("--base-model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="Base model name for adapter config")
    parser.add_argument("--symlink", action="store_true",
                        help="Generate one real adapter per rank, symlink the rest (saves disk)")
    args = parser.parse_args()

    if not args.mixed and args.rank is None:
        parser.error("Either --rank or --mixed is required")

    # Auto-detect model dimensions from base model name
    dims = get_model_dims(args.base_model)
    num_layers = dims["num_layers"]
    modules = dims["modules"]
    print(f"Model: {args.base_model} ({num_layers} layers, hidden={list(modules.values())[0][0]})")

    # Pre-generate weights for each rank (one per rank, shared across adapters)
    if args.mixed:
        ranks_needed = sorted(set(MIXED_PATTERN))
    else:
        ranks_needed = [args.rank]

    weight_cache = {}
    for rank in ranks_needed:
        weight_cache[rank] = generate_adapter(rank, seed=42, num_layers=num_layers, modules=modules)

    from collections import Counter
    rank_counts = Counter()
    rank_source = {}  # rank -> first real adapter dir (for --symlink)

    for i in range(args.num):
        idx = args.start + i
        rank = get_mixed_rank(idx) if args.mixed else args.rank
        rank_counts[rank] += 1

        adapter_dir = os.path.join(args.out, f"lora-{idx}")

        if args.symlink and rank in rank_source:
            # Symlink to the first real adapter of this rank
            src = os.path.relpath(rank_source[rank], os.path.dirname(adapter_dir))
            if os.path.lexists(adapter_dir):
                if os.path.islink(adapter_dir):
                    os.unlink(adapter_dir)
                else:
                    import shutil
                    shutil.rmtree(adapter_dir)
            os.symlink(src, adapter_dir)
        else:
            os.makedirs(adapter_dir, exist_ok=True)
            weights = weight_cache[rank]
            save_file(weights, os.path.join(adapter_dir, "adapter_model.safetensors"))

            config = make_config(rank, base_model=args.base_model)
            with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
                json.dump(config, f, indent=2)

            rank_source[rank] = adapter_dir
            size_mb = os.path.getsize(os.path.join(adapter_dir, "adapter_model.safetensors")) / 1024 / 1024
            print(f"Rank {rank}: {size_mb:.1f} MB per adapter, {len(weights)} tensors")

    real = len(rank_source)
    dist = ", ".join(f"r{r}:{c}" for r, c in sorted(rank_counts.items()))
    print(f"Generated {args.num} adapters ({real} real, {args.num - real} symlinked) in {args.out} ({dist})")


if __name__ == "__main__":
    main()
