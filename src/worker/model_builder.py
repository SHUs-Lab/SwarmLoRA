"""Model building: LoRA loading, hollow model construction, sync module stubs."""

import os
import re
import time
import json
import numpy as np
import torch
import torch.nn as nn

from config import DTYPE, LORA_ADAPTER_ID
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from safetensors import safe_open
import ext_unified_barrier as ub
from worker.utils import log

# Operation constants
OP_O_PROJ = ub.OP_O_PROJ
OP_DOWN_PROJ = ub.OP_DOWN_PROJ
OP_QKV_FUSED = ub.OP_QKV_FUSED
OP_GATE_UP_FUSED = ub.OP_GATE_UP_FUSED
OP_EMBED = ub.OP_EMBED
OP_LM_HEAD = ub.OP_LM_HEAD


class FastLoRAModule:
    """Lightweight LoRA module that mimics peft's LoRALinear interface for FusedLoRA."""
    def __init__(self, lora_A: torch.Tensor, lora_B: torch.Tensor, scaling: float):
        # lora_A shape: (rank, in_features) - matches nn.Linear(in_features, rank)
        # lora_B shape: (out_features, rank) - matches nn.Linear(rank, out_features)
        rank, in_features = lora_A.shape
        out_features, _ = lora_B.shape

        # Create proper nn.Linear modules (callable)
        A_linear = nn.Linear(in_features, rank, bias=False, device=lora_A.device)
        A_linear.weight = nn.Parameter(lora_A, requires_grad=False)

        B_linear = nn.Linear(rank, out_features, bias=False, device=lora_B.device)
        B_linear.weight = nn.Parameter(lora_B, requires_grad=False)

        self.lora_A = {'default': A_linear}
        self.lora_B = {'default': B_linear}
        self.scaling = {'default': scaling}
        self.lora_dropout = {'default': nn.Identity()}


# Allowlist of base directories for local adapter loading.
# Populated at worker startup via register_adapter_base_dir().
_ALLOWED_ADAPTER_BASES: list = []

# Valid HF Hub repo ID: "owner/repo-name" with safe characters only
_HF_REPO_RE = re.compile(r'^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.\-]+$')


def register_adapter_base_dir(base_dir: str) -> None:
    """Register a trusted base directory for local adapter loading."""
    real = os.path.realpath(base_dir)
    if real not in _ALLOWED_ADAPTER_BASES:
        _ALLOWED_ADAPTER_BASES.append(real)


def _validate_adapter_id(lora_path: str) -> None:
    """Raise ValueError if the adapter ID is unsafe or untrusted."""
    if not lora_path or '\x00' in lora_path or len(lora_path) > 512:
        raise ValueError(f"Invalid adapter ID: {lora_path!r}")

    # Classify local-path vs HF Hub id the same way the resolver does (isdir
    # first): a bare "owner/repo" not on disk is a Hub id; anything existing
    # locally, or path-shaped (dot-prefixed, absolute, multi-segment), is a
    # local path. Every valid Hub id contains '/', so treating any string with
    # a separator as local misroutes Hub ids into the local branch.
    looks_like_hf = (_HF_REPO_RE.match(lora_path) is not None
                     and '..' not in lora_path.split('/'))
    is_local = os.path.isdir(lora_path) or not looks_like_hf
    if is_local:
        # realpath first (collapses '..', follows symlinks), then require the
        # result under a registered base. Allows legitimate relative paths
        # (../sim-adapters/pool/lora-0) while rejecting traversal escapes.
        real = os.path.realpath(lora_path)
        if _ALLOWED_ADAPTER_BASES:
            if not any(real.startswith(base + os.sep) or real == base
                       for base in _ALLOWED_ADAPTER_BASES):
                raise ValueError(
                    f"Adapter path outside allowed directories: {lora_path!r}")
        elif '..' + os.sep in lora_path or lora_path.endswith('..'):
            # No allowlist registered yet — fall back to rejecting explicit
            # traversal so an unconfigured worker can't be steered outside cwd.
            raise ValueError(f"Path traversal rejected: {lora_path!r}")
    # else: a valid HF Hub id with nothing on the local filesystem to guard.


def _resolve_lora_path(lora_path: str) -> str:
    """Resolve LoRA adapter path (local dir or HF Hub repo ID)."""
    if os.path.isdir(lora_path):
        real = os.path.realpath(lora_path)
        return real
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    config_file = try_to_load_from_cache(lora_path, 'adapter_config.json')
    if config_file is None:
        config_file = hf_hub_download(lora_path, 'adapter_config.json')
    adapter_path = os.path.dirname(config_file)
    safetensors_cached = try_to_load_from_cache(lora_path, 'adapter_model.safetensors')
    bin_cached = try_to_load_from_cache(lora_path, 'adapter_model.bin')
    if safetensors_cached is None and bin_cached is None:
        try:
            hf_hub_download(lora_path, 'adapter_model.safetensors')
        except Exception:
            hf_hub_download(lora_path, 'adapter_model.bin')
    return adapter_path


def _load_lora_raw(adapter_path: str, device: str, dtype: torch.dtype) -> tuple:
    """Load raw LoRA weight tensors and config from resolved path."""
    with open(os.path.join(adapter_path, 'adapter_config.json')) as f:
        config = json.load(f)

    weights = {}
    safetensors_file = os.path.join(adapter_path, 'adapter_model.safetensors')
    bin_file = os.path.join(adapter_path, 'adapter_model.bin')

    if os.path.exists(safetensors_file):
        with safe_open(safetensors_file, framework='pt', device=device) as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key).to(dtype=dtype)
    elif os.path.exists(bin_file):
        # Reject .bin — torch.load pickle has known weights_only bypass exploits.
        # All supported adapters use safetensors. Convert with:
        #   python -c "from safetensors.torch import save_file; import torch
        #   save_file(torch.load('adapter_model.bin', weights_only=True), 'adapter_model.safetensors')"
        raise ValueError(
            f"Adapter at {adapter_path!r} uses .bin format which is not allowed. "
            f"Convert to safetensors format."
        )
    else:
        raise FileNotFoundError(f"No adapter weights found in {adapter_path}")

    return weights, config


def load_lora_fast(lora_path: str, device: str, dtype: torch.dtype) -> tuple:
    """Load LoRA weights directly to the target GPU device."""
    _validate_adapter_id(lora_path)
    adapter_path = _resolve_lora_path(lora_path)
    weights, config = _load_lora_raw(adapter_path, device, dtype)

    rank = config['r']
    alpha = config.get('lora_alpha', rank)
    scaling = alpha / rank

    # Organize by layer and module
    lora_modules = {}  # {layer_idx: {module_name: FastLoRAModule}}

    for key, tensor in weights.items():
        # Parse: base_model.model.model.layers.0.mlp.down_proj.lora_A.weight
        parts = key.split('.')
        layer_idx = int(parts[4])
        # Get module name (e.g., 'self_attn.q_proj' or 'mlp.down_proj')
        if 'self_attn' in key:
            module_name = parts[5] + '.' + parts[6]  # self_attn.q_proj
        else:
            module_name = parts[5] + '.' + parts[6]  # mlp.down_proj
        lora_type = parts[-2]  # lora_A or lora_B

        if layer_idx not in lora_modules:
            lora_modules[layer_idx] = {}
        if module_name not in lora_modules[layer_idx]:
            lora_modules[layer_idx][module_name] = {'lora_A': None, 'lora_B': None}

        lora_modules[layer_idx][module_name][lora_type] = tensor

    # Convert to FastLoRAModule objects
    result = {}
    for layer_idx, modules in lora_modules.items():
        result[layer_idx] = {}
        for module_name, lora_weights in modules.items():
            result[layer_idx][module_name] = FastLoRAModule(
                lora_A=lora_weights['lora_A'],
                lora_B=lora_weights['lora_B'],
                scaling=scaling
            )

    return result, config


class SyncRemoteLinear(nn.Module):
    """Placeholder for non-LoRA linear layers (computation offloaded to aggregator)."""
    def __init__(self, op: int, layer: int, out_dim: int):
        super().__init__()
        self.op, self.layer, self.out_dim = op, layer, out_dim


class SyncRemoteLoRALinear(nn.Module):
    """LoRA module for aggregator-offloaded linear layers."""
    def __init__(self, op: int, layer: int, out_dim: int, lora_module):
        super().__init__()
        self.op, self.layer, self.out_dim = op, layer, out_dim
        self.lora_A = lora_module.lora_A['default']
        self.lora_B = lora_module.lora_B['default']
        self.scaling = lora_module.scaling['default']
        self.dropout = lora_module.lora_dropout['default']


class SyncEmbed(nn.Module):
    """Placeholder for embedding layer (computation offloaded to aggregator)."""
    def __init__(self):
        super().__init__()


class SyncLMHead(nn.Module):
    """Placeholder for LM head (computation offloaded to aggregator)."""
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size


class FusedLoRA(nn.Module):
    """Fused LoRA for multiple projections (QKV or GateUp)."""

    def __init__(self, lora_modules: list, out_dims: list, device: str, dtype: torch.dtype):
        super().__init__()
        self.out_dims = out_dims
        self.num_loras = len(lora_modules)

        valid_modules = [m for m in lora_modules if m is not None]
        if not valid_modules:
            self.has_lora = False
            return
        self.has_lora = True

        A_weights, B_weights = [], []
        for i, module in enumerate(lora_modules):
            if module is not None and hasattr(module, 'lora_A'):
                A = module.lora_A['default'].weight
                B = module.lora_B['default'].weight
                s = module.scaling['default']
                A_weights.append(A.t() * s)
                B_weights.append(B.t())
            else:
                rank = valid_modules[0].lora_A['default'].weight.size(0)
                in_features = valid_modules[0].lora_A['default'].weight.size(1)
                A_weights.append(torch.zeros(in_features, rank, device=device, dtype=dtype))
                B_weights.append(torch.zeros(rank, out_dims[i], device=device, dtype=dtype))

        self.fused_A = nn.Parameter(torch.cat(A_weights, dim=1), requires_grad=False)

        rank = A_weights[0].size(1)
        total_out = sum(out_dims)
        fused_B = torch.zeros(rank * self.num_loras, total_out, device=device, dtype=dtype)
        col_offset = 0
        for i, B in enumerate(B_weights):
            row_start, row_end = i * rank, (i + 1) * rank
            fused_B[row_start:row_end, col_offset:col_offset + out_dims[i]] = B
            col_offset += out_dims[i]
        self.fused_B = nn.Parameter(fused_B, requires_grad=False)
        self.dropout = valid_modules[0].lora_dropout['default']

    def forward(self, x: torch.Tensor) -> list:
        if not self.has_lora:
            return [None] * self.num_loras
        delta_fused = self.dropout(x) @ self.fused_A @ self.fused_B
        return delta_fused.split(self.out_dims, dim=-1)


class SyncFusedQKV(nn.Module):
    """Fused QKV projection holder for LoRA."""
    def __init__(self, layer: int, dims: list, fused_lora: FusedLoRA = None):
        super().__init__()
        self.layer, self.dims, self.fused_lora = layer, dims, fused_lora


class SyncFusedGateUp(nn.Module):
    """Fused GateUp projection holder for LoRA."""
    def __init__(self, layer: int, dims: list, fused_lora: FusedLoRA = None):
        super().__init__()
        self.layer, self.dims, self.fused_lora = layer, dims, fused_lora


def _load_norm_weights(model, norm_data: dict, device: str):
    """Load serialized norm weights into existing LlamaRMSNorm modules (created on meta device)."""

    def load_weight(norm_module, data_bytes):
        arr = np.frombuffer(data_bytes, dtype=np.float32)
        weight = torch.from_numpy(arr.copy()).to(device=device, dtype=DTYPE)
        norm_module.weight = nn.Parameter(weight, requires_grad=False)

    load_weight(model.model.norm, norm_data['final_norm'])
    for i, layer in enumerate(model.model.layers):
        load_weight(layer.input_layernorm, norm_data[f'layer_{i}_input_norm'])
        load_weight(layer.post_attention_layernorm, norm_data[f'layer_{i}_post_norm'])


def _setup_rotary_and_compat(model, cfg, device):
    """Create rotary embeddings on worker device and set per-layer attributes."""
    rotary_emb = LlamaRotaryEmbedding(config=cfg).to(device)
    model.model.rotary_emb = rotary_emb

    for layer in model.model.layers:
        attn = layer.self_attn
        attn.rotary_emb = rotary_emb
        attn.hidden_size = cfg.hidden_size
        attn.num_heads = cfg.num_attention_heads
        attn.num_kv_heads = cfg.num_key_value_heads


def build_hollow_model(info: dict, client, worker_id: int,
                        worker_device: str, lora_id: str = None):
    """Build hollow model: minimal structure + norm weights + LoRA, no base weights."""
    t0 = time.perf_counter()
    log(worker_id, "Building hollow model...")

    # Config from aggregator info (works for any model family: Llama, Mistral, etc.)
    if 'model_config' in info:
        config_dict = info['model_config'].copy()
        for key in ('transformers_version',):
            config_dict.pop(key, None)
        cfg = AutoConfig.for_model(**config_dict)
    else:
        cfg = AutoConfig.from_pretrained(info.get('model_id', 'meta-llama/Llama-2-7b-hf'), local_files_only=True)
    t1 = time.perf_counter()

    # Create model on meta device (full structure, zero memory)
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(cfg)
    t2 = time.perf_counter()

    _load_norm_weights(model, info['norm_weights'], worker_device)
    t3 = time.perf_counter()

    _setup_rotary_and_compat(model, cfg, worker_device)
    t4 = time.perf_counter()

    # Load LoRA weights directly to device
    lora_adapter = lora_id or LORA_ADAPTER_ID
    lora_layers, lora_config = load_lora_fast(lora_adapter, worker_device, DTYPE)
    t5 = time.perf_counter()

    decoder = model.model

    def get_lora_module(layer_idx: int, module_name: str):
        """Get FastLoRAModule from loaded weights."""
        if layer_idx not in lora_layers:
            return None
        return lora_layers[layer_idx].get(module_name)

    q_dim = info['hidden_size']
    k_dim = info['hidden_size'] // info['num_heads'] * info['num_kv_heads']
    v_dim = k_dim
    qkv_dims = [q_dim, k_dim, v_dim]
    gate_up_dims = [info['intermediate_size'], info['intermediate_size']]

    for i, layer in enumerate(decoder.layers):
        attn, mlp = layer.self_attn, layer.mlp

        # Get LoRA modules for this layer
        qkv_lora = [
            get_lora_module(i, 'self_attn.q_proj'),
            get_lora_module(i, 'self_attn.k_proj'),
            get_lora_module(i, 'self_attn.v_proj')
        ]
        fused_qkv_lora = FusedLoRA(qkv_lora, qkv_dims, worker_device, DTYPE) if any(qkv_lora) else None

        gate_up_lora = [
            get_lora_module(i, 'mlp.gate_proj'),
            get_lora_module(i, 'mlp.up_proj')
        ]
        fused_gate_up_lora = FusedLoRA(gate_up_lora, gate_up_dims, worker_device, DTYPE) if any(gate_up_lora) else None

        fused_qkv = SyncFusedQKV(i, qkv_dims, fused_qkv_lora)
        fused_gu = SyncFusedGateUp(i, gate_up_dims, fused_gate_up_lora)

        # Store references for hot-swap (AdapterManager needs these)
        attn._fused_qkv = fused_qkv
        mlp._fused_gate_up = fused_gu

        # Handle o_proj and down_proj LoRA
        o_proj_lora = get_lora_module(i, 'self_attn.o_proj')
        if o_proj_lora is not None:
            attn.o_proj = SyncRemoteLoRALinear(OP_O_PROJ, i, info['hidden_size'], o_proj_lora)
        else:
            attn.o_proj = SyncRemoteLinear(OP_O_PROJ, i, info['hidden_size'])

        down_proj_lora = get_lora_module(i, 'mlp.down_proj')
        if down_proj_lora is not None:
            mlp.down_proj = SyncRemoteLoRALinear(OP_DOWN_PROJ, i, info['hidden_size'], down_proj_lora)
        else:
            mlp.down_proj = SyncRemoteLinear(OP_DOWN_PROJ, i, info['hidden_size'])

    t6 = time.perf_counter()

    vocab_size = info.get('vocab_size', 128256)
    if hasattr(decoder, 'embed_tokens'):
        decoder.embed_tokens = SyncEmbed()
    if hasattr(model, 'lm_head'):
        model.lm_head = SyncLMHead(vocab_size)

    model.eval()
    t7 = time.perf_counter()

    log(worker_id, f"  INIT TIMING: config={t1-t0:.2f}s meta_model={t2-t1:.2f}s "
             f"norms={t3-t2:.2f}s rotary={t4-t3:.2f}s lora={t5-t4:.2f}s "
             f"stubs={t6-t5:.2f}s embed={t7-t6:.2f}s TOTAL={t7-t0:.2f}s")
    return model
