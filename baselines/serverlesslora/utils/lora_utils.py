#!/usr/bin/env python3
"""
Custom Unmerged LoRA Implementation for ServerlessLoRA.

This module implements LoRA inference WITHOUT using PEFT library,
keeping backbone and adapter computations completely separate.
This matches the paper's description of "unmerged inference atop Transformers".
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from typing import Dict, Optional, Tuple
from huggingface_hub import snapshot_download


class LoRALayer(nn.Module):
    """
    A single LoRA adapter layer that computes: output = base_output + B @ (A @ x) * scaling

    This keeps the base layer computation separate - we only add the LoRA delta.
    """

    def __init__(self, lora_A: torch.Tensor, lora_B: torch.Tensor,
                 scaling: float, dropout: float = 0.0):
        super().__init__()
        # lora_A: (rank, in_features) - down projection
        # lora_B: (out_features, rank) - up projection
        self.lora_A = nn.Parameter(lora_A, requires_grad=False)
        self.lora_B = nn.Parameter(lora_B, requires_grad=False)
        self.scaling = scaling
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute LoRA delta: B @ (A @ x) * scaling

        Args:
            x: Input tensor of shape (..., in_features)
        Returns:
            LoRA output delta of shape (..., out_features)
        """
        # x @ A.T @ B.T = x @ (B @ A).T
        # A: (rank, in_features), B: (out_features, rank)
        # x: (..., in_features)
        # Result: (..., out_features)
        x = self.dropout(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return lora_out * self.scaling


class LoRALinearWrapper(nn.Module):
    """
    Wraps a base Linear layer and adds LoRA computation.

    Forward: y = base_linear(x) + lora_layer(x)

    The base_linear weights are READ-ONLY (IPC-shared).
    Only the LoRA weights are owned by this process.
    """

    def __init__(self, base_linear: nn.Linear, lora_layer: LoRALayer):
        super().__init__()
        self.base_linear = base_linear
        self.lora_layer = lora_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = self.lora_layer(x)
        return base_out + lora_out


def load_lora_config(adapter_path: str) -> dict:
    """Load LoRA adapter configuration."""
    config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(config_path, 'r') as f:
        return json.load(f)


def load_lora_weights(adapter_path: str, device: str = "cuda:0",
                      dtype: torch.dtype = torch.bfloat16) -> Dict[str, torch.Tensor]:
    """
    Load LoRA weights from safetensors or bin file.

    Returns dict mapping layer names to tensors.
    """
    weights = {}

    # Try safetensors first
    safetensor_path = os.path.join(adapter_path, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_path, "adapter_model.bin")

    if os.path.exists(safetensor_path):
        with safe_open(safetensor_path, framework="pt", device=str(device)) as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key).to(dtype=dtype)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location=device)
        for key, tensor in state_dict.items():
            weights[key] = tensor.to(dtype=dtype)
    else:
        raise FileNotFoundError(f"No adapter weights found in {adapter_path}")

    return weights


def download_lora_adapter(adapter_id: str, local_dir: Optional[str] = None) -> str:
    """
    Download LoRA adapter from HuggingFace Hub.

    Returns path to downloaded adapter.
    """
    if local_dir is None:
        local_dir = os.path.join(os.path.expanduser("~"), ".cache", "lora_adapters",
                                  adapter_id.replace("/", "_"))

    if os.path.exists(os.path.join(local_dir, "adapter_config.json")):
        return local_dir

    # Download from HuggingFace
    snapshot_download(
        repo_id=adapter_id,
        local_dir=local_dir,
        allow_patterns=["adapter_config.json", "adapter_model.*"]
    )
    return local_dir


def parse_lora_weight_name(weight_name: str) -> Tuple[str, str, str]:
    """
    Parse LoRA weight name to extract module path and weight type.

    Example: "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    Returns: ("model.layers.0.self_attn.q_proj", "lora_A", "weight")
    """
    parts = weight_name.split(".")

    # Find lora_A or lora_B position
    lora_idx = None
    for i, p in enumerate(parts):
        if p in ("lora_A", "lora_B"):
            lora_idx = i
            break

    if lora_idx is None:
        raise ValueError(f"Cannot parse LoRA weight name: {weight_name}")

    # Extract module path (remove "base_model.model." prefix if present)
    module_parts = parts[:lora_idx]

    # Remove the PEFT "base_model.model." prefix if present.
    # Only strip exactly one "base_model" then one "model" to avoid
    # eating the real model submodule (e.g. LlamaForCausalLM.model).
    if module_parts and module_parts[0] == "base_model":
        module_parts = module_parts[1:]
    if module_parts and module_parts[0] == "model":
        module_parts = module_parts[1:]

    module_path = ".".join(module_parts)
    lora_type = parts[lora_idx]  # "lora_A" or "lora_B"
    weight_type = parts[lora_idx + 1] if lora_idx + 1 < len(parts) else "weight"

    return module_path, lora_type, weight_type


def get_module_by_path(model: nn.Module, path: str) -> nn.Module:
    """Get a submodule by dot-separated path."""
    parts = path.split(".")
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def set_module_by_path(model: nn.Module, path: str, new_module: nn.Module):
    """Set a submodule by dot-separated path."""
    parts = path.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    last_part = parts[-1]
    if last_part.isdigit():
        parent[int(last_part)] = new_module
    else:
        setattr(parent, last_part, new_module)


def apply_lora_to_model(model: nn.Module, adapter_id: str,
                        device: str = "cuda:0",
                        dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """
    Apply LoRA adapter to a model using custom unmerged inference.

    This does NOT use PEFT. Instead, it:
    1. Loads LoRA weights directly
    2. Creates LoRALayer modules for each target
    3. Wraps base Linear layers with LoRALinearWrapper

    The base model weights remain READ-ONLY (suitable for IPC sharing).

    Args:
        model: Base model with IPC-shared weights
        adapter_id: HuggingFace adapter ID or local path
        device: Target device
        dtype: Target dtype

    Returns:
        Model with LoRA layers applied (unmerged)
    """
    # Download adapter if needed
    if os.path.exists(adapter_id):
        adapter_path = adapter_id
    else:
        print(f"Downloading LoRA adapter: {adapter_id}")
        adapter_path = download_lora_adapter(adapter_id)

    # Load config and weights
    config = load_lora_config(adapter_path)
    weights = load_lora_weights(adapter_path, device=device, dtype=dtype)

    # Extract LoRA parameters from config
    rank = config.get("r", config.get("rank", 8))
    alpha = config.get("lora_alpha", rank)
    dropout = config.get("lora_dropout", 0.0)
    scaling = alpha / rank

    print(f"LoRA config: rank={rank}, alpha={alpha}, scaling={scaling:.4f}")

    # Group weights by module path
    lora_weights_by_module: Dict[str, Dict[str, torch.Tensor]] = {}

    for weight_name, tensor in weights.items():
        try:
            module_path, lora_type, _ = parse_lora_weight_name(weight_name)
            if module_path not in lora_weights_by_module:
                lora_weights_by_module[module_path] = {}
            lora_weights_by_module[module_path][lora_type] = tensor
        except ValueError as e:
            print(f"  Skipping weight: {weight_name} ({e})")

    # Apply LoRA to each target module
    applied_count = 0
    for module_path, lora_dict in lora_weights_by_module.items():
        if "lora_A" not in lora_dict or "lora_B" not in lora_dict:
            print(f"  Skipping {module_path}: missing lora_A or lora_B")
            continue

        try:
            # Get the base linear layer
            base_linear = get_module_by_path(model, module_path)

            if not isinstance(base_linear, nn.Linear):
                print(f"  Skipping {module_path}: not a Linear layer")
                continue

            # Create LoRA layer
            lora_A = lora_dict["lora_A"]  # (rank, in_features)
            lora_B = lora_dict["lora_B"]  # (out_features, rank)

            lora_layer = LoRALayer(lora_A, lora_B, scaling, dropout)
            lora_layer = lora_layer.to(device=device, dtype=dtype)

            # Wrap the base linear with LoRA
            wrapped = LoRALinearWrapper(base_linear, lora_layer)

            # Replace in model
            set_module_by_path(model, module_path, wrapped)
            applied_count += 1

        except Exception as e:
            print(f"  Error applying LoRA to {module_path}: {e}")

    print(f"Applied LoRA to {applied_count} layers")
    return model


def remove_lora_from_model(model: nn.Module) -> nn.Module:
    """
    Remove all LoRA layers, restoring original base Linear layers.

    Walks the model tree, finds LoRALinearWrapper instances,
    and replaces them with their inner base_linear (IPC-shared weights).
    """
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinearWrapper):
            replacements.append((name, module.base_linear))

    for name, base_linear in replacements:
        set_module_by_path(model, name, base_linear)

    print(f"Removed LoRA from {len(replacements)} layers")

    # Free LoRA weight memory
    torch.cuda.empty_cache()
    return model


def print_lora_info(model: nn.Module):
    """Print information about LoRA layers in a model."""
    lora_layers = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinearWrapper):
            lora_layers.append(name)

    print(f"Model has {len(lora_layers)} LoRA-wrapped layers:")
    for name in lora_layers[:5]:
        print(f"  - {name}")
    if len(lora_layers) > 5:
        print(f"  ... and {len(lora_layers) - 5} more")
