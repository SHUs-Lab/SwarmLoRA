#!/usr/bin/env python3
"""Worker for multi-GPU batched LLM inference."""

import time
_t_module_start = time.perf_counter()

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback
import threading
import uuid
import json
from collections import defaultdict
from dataclasses import dataclass, field
from queue import Queue, Empty
from concurrent.futures import Future
from typing import Dict, List, Optional

from config import (
    LORA_ADAPTER_ID,
    WORKER_DEVICE, DTYPE, DEFAULT_PORT,
    PAGED_ATTENTION_BLOCK_SIZE, MAX_CONCURRENT_REQUESTS_PER_WORKER, MAX_SEQ_LEN,
    PREFILL_CHUNK_SIZE,
)

MAX_SLOTS = int(os.environ.get("MAX_SLOTS_OVERRIDE", MAX_CONCURRENT_REQUESTS_PER_WORKER))

import ext_unified_barrier as ub
import ext_ipc_queue as extq

# flash_attn for unified prefill + decode attention
from vllm.vllm_flash_attn import flash_attn_with_kvcache
from vllm import _custom_ops as vllm_ops

# Pre-import Flask for accurate cold start timing
from flask import Flask, request, jsonify

# Extracted modules
from worker.utils import log, connect, signal_done, load_tokenizer_from_cache

# Guarded post-admit margin, shared with both baselines so the guards cannot
# drift apart. Loaded by path: this module runs with src/ on the path, not the
# controller package.
def _load_margin():
    import importlib.util, os
    _p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "controller", "admission.py")
    _sp = importlib.util.spec_from_file_location("swarm_admission_margin", _p)
    _m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m)
    return _m
# Every system reserves the same margin from this one policy, so a tree that
# cannot load it would run a different one. Fail at import instead.
_adm = _load_margin()
_guarded_margin = _adm.post_admit_margin
_SLO_S = float(os.environ.get("ADMISSION_SLO_S", _adm.DEFAULT_SLO_S))
from worker.model_builder import (
    SyncRemoteLoRALinear, build_hollow_model,
    OP_O_PROJ, OP_DOWN_PROJ, OP_QKV_FUSED, OP_GATE_UP_FUSED,
    OP_EMBED, OP_LM_HEAD,
    register_adapter_base_dir, _validate_adapter_id,
)
from worker.paged_attention import PagedKVCache, build_vllm_cos_sin_cache
from worker.adapter_manager import AdapterManager
from worker.slot_client import SlotClient, IntraAdapterSlotManager

_t_module_end = time.perf_counter()
_module_import_time = _t_module_end - _t_module_start

# Timeouts and limits
_SLOT_ACTIVATION_TIMEOUT = 30.0
_SLOT_LEAVE_TIMEOUT = 5.0
_SPIN_BEFORE_YIELD = 1000
_IDLE_POLL_INTERVAL = 0.01
_INFERENCE_TIMEOUT = 300.0
# PREFILL_CHUNK_SIZE imported from config


# INTRA-ADAPTER BATCHING: Data Structures

@dataclass
class InferenceRequest:
    """Request submitted by HTTP thread for inference."""
    request_id: str
    prompt: str
    max_tokens: int
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    future: Future = field(default_factory=Future)
    submit_time: float = field(default_factory=time.perf_counter)  # When request was submitted
    # Absolute wall-clock SLO deadline from the controller. None in static
    # mode, which leaves the pre-existing unbounded behaviour untouched.
    slo_deadline_wall: float = None


@dataclass
class SlotState:
    """State for an active slot in batched inference."""
    request: InferenceRequest
    slot_id: int
    input_ids: torch.Tensor          # Current input token(s)
    generated_tokens: List[int]      # Tokens generated so far
    hidden: torch.Tensor = None      # Current hidden state
    residual: torch.Tensor = None    # Residual for attention block
    mlp_residual: torch.Tensor = None  # Residual for MLP block
    joined: bool = False             # Whether join has been signaled to aggregator
    prefill_chunks: list = None       # Remaining chunks for chunked prefill
    prompt_tokens: int = 0             # Number of prompt tokens (for usage reporting)
    # Timing metrics
    slot_allocated_time: float = None  # When slot was assigned
    first_token_time: float = None     # When first token was generated
    join_time: float = None            # When join was signaled (for timeout detection)


# INTRA-ADAPTER BATCHING: Worker-Side Decode Batching

class WorkerBatchedOps:
    """Batched worker-side operations for decode slots."""

    @staticmethod
    def split_by_phase(slots: Dict[int, SlotState]) -> tuple:
        """Split the single active slot into prefill or decode group."""
        sid, s = next(iter(slots.items()))
        if s.input_ids.size(1) > 1:
            return slots, {}
        return {}, slots

    @staticmethod
    def batch_layernorm(decode_slots: Dict[int, SlotState],
                        norm_layer: nn.Module,
                        save_residual: bool = False,
                        save_mlp_residual: bool = False):
        """LayerNorm for the single decode slot (if present)."""
        if not decode_slots:
            return
        sid = next(iter(decode_slots))
        state = decode_slots[sid]
        if save_residual:
            state.residual = state.hidden
        if save_mlp_residual:
            state.mlp_residual = state.hidden
        state.hidden = norm_layer(state.hidden)

    @staticmethod
    def batch_mlp_activation(decode_slots: Dict[int, SlotState],
                             gate_up_results: Dict[int, tuple]) -> Dict[int, torch.Tensor]:
        """SiLU(gate) * up for the single decode slot (if present)."""
        if not decode_slots:
            return {}
        sid = next(iter(decode_slots))
        gate, up = gate_up_results[sid]
        return {sid: F.silu(gate) * up}

    @staticmethod
    def batch_final_norm(decode_slots: Dict[int, SlotState], norm_layer: nn.Module):
        """Final LayerNorm for the single decode slot (if present)."""
        if not decode_slots:
            return
        sid = next(iter(decode_slots))
        decode_slots[sid].hidden = norm_layer(decode_slots[sid].hidden)

    @staticmethod
    def batch_flash_attention(slot_ids: list,
                              qkv_results: dict,
                              paged_kv: PagedKVCache,
                              cos_sin_cache: torch.Tensor,
                              layer_idx: int,
                              num_heads: int, num_kv_heads: int,
                              head_dim: int,
                              attn_meta: dict = None) -> Dict[int, torch.Tensor]:
        """Attention for the single active slot using flash_attn_with_kvcache."""
        q_len = attn_meta['bufs_q'].shape[1]
        hidden_size = num_heads * head_dim

        # Zero-copy views of source tensors (skip copy_ calls). Safe: source
        # tensors from get_outputs_slot() are worker-owned, and RoPE modifies
        # them in-place which is fine.
        sid = slot_ids[0]
        q_4d = qkv_results[sid][0].reshape(1, q_len, num_heads, head_dim)
        k_4d = qkv_results[sid][1].reshape(1, q_len, num_kv_heads, head_dim)
        v_4d = qkv_results[sid][2].reshape(1, q_len, num_kv_heads, head_dim)

        # Apply RoPE in-place (fused CUDA kernel, pre-computed positions)
        q_flat = q_4d.reshape(q_len, num_heads * head_dim)
        k_flat = k_4d.reshape(q_len, num_kv_heads * head_dim)
        vllm_ops.rotary_embedding(attn_meta['positions'], q_flat, k_flat,
                                   head_dim, cos_sin_cache, True)

        # flash_attn: cache write + paged attention (RoPE already applied)
        out = flash_attn_with_kvcache(
            q_4d, paged_kv.key_caches[layer_idx], paged_kv.value_caches[layer_idx],
            k_4d, v_4d,
            cache_seqlens=attn_meta['cache_seqlens'],
            block_table=attn_meta['block_tables'],
            causal=True,
            softmax_scale=1.0 / (head_dim ** 0.5),
        )  # → (1, q_len, num_heads, head_dim)

        return {sid: out.reshape(1, q_len, hidden_size)}


# INTRA-ADAPTER BATCHING: Batched Operations (Aggregator Submit/Get)


class BatchedAggregatorOps:
    """Handles batched submit/signal/wait/get pattern for aggregator operations."""

    def __init__(self, slot_manager: IntraAdapterSlotManager, worker_id: int, dtype_code: int):
        self.slot_manager = slot_manager
        self.worker_id = worker_id
        self.dtype_code = dtype_code
        self.dtype = torch.bfloat16 if dtype_code == 15 else torch.float16
        self.hidden_dim = slot_manager.base_client.hidden_dim

    def barrier_sync(self, epoch: int):
        """Wait for aggregator to complete the specified epoch."""
        extq.wait_for_epoch(epoch)

    def _get_outputs_all(self, slots, dims):
        """Get aggregator outputs for the single active slot."""
        slot_id, state = next(iter(slots.items()))
        return {slot_id: ub.get_outputs_slot(slot_id, dims, state.hidden)}


    def embed_submit_all(self, slots: Dict[int, SlotState]) -> int:
        """Submit embed command for the single active slot. Returns epoch to wait for."""
        epoch = extq.get_barrier_done_epoch()
        slot_id, state = next(iter(slots.items()))
        ub.prepare_token_ids_slot(slot_id, state.input_ids)
        b, t = state.input_ids.shape
        extq.submit_slot_command(
            OP_EMBED, 0, slot_id, self.worker_id,
            b, t, self.hidden_dim, 0, 0, 0, 0, 0
        )
        extq.slot_signal_ready(slot_id)
        return epoch

    def embed_get_all(self, slots: Dict[int, SlotState]):
        """Get embedding for the single active slot."""
        slot_id, state = next(iter(slots.items()))
        state.hidden = ub.get_embeddings_slot(slot_id, self.dtype)


    def qkv_submit_all(self, slots: Dict[int, SlotState], layer_idx: int,
                       fused_qkv_modules: List) -> tuple:
        """Submit QKV command and compute LoRA delta. Returns (epoch, deltas)."""
        epoch = extq.get_barrier_done_epoch()
        deltas = {}
        slot_id, state = next(iter(slots.items()))
        ub.prepare_input_slot(slot_id, OP_QKV_FUSED, state.hidden)
        b, t, d = state.hidden.shape
        extq.submit_slot_command(
            OP_QKV_FUSED, layer_idx, slot_id, self.worker_id,
            b, t, d, 0, 0, 0, 0, 0
        )
        extq.slot_signal_ready(slot_id)
        fused_qkv = fused_qkv_modules[layer_idx]
        if fused_qkv.fused_lora and fused_qkv.fused_lora.has_lora:
            deltas[slot_id] = fused_qkv.fused_lora(state.hidden)
        else:
            deltas[slot_id] = (None, None, None)
        return epoch, deltas

    def qkv_get_all(self, slots: Dict[int, SlotState], layer_idx: int,
                    deltas: Dict[int, tuple], dims: List[int]) -> Dict[int, tuple]:
        """Get QKV results for all slots, add LoRA deltas."""
        raw = self._get_outputs_all(slots, dims)
        results = {}
        for slot_id in slots:
            q, k, v = raw[slot_id]
            q_delta, k_delta, v_delta = deltas.get(slot_id, (None, None, None))
            if q_delta is not None:
                q = q + q_delta
            if k_delta is not None:
                k = k + k_delta
            if v_delta is not None:
                v = v + v_delta
            results[slot_id] = (q, k, v)
        return results


    def o_proj_submit_all(self, slots: Dict[int, SlotState], layer_idx: int,
                          attn_outputs: Dict[int, torch.Tensor],
                          o_proj_modules: List) -> tuple:
        """Submit O_PROJ command and compute LoRA delta. Returns (epoch, deltas)."""
        epoch = extq.get_barrier_done_epoch()
        deltas = {}
        slot_id, state = next(iter(slots.items()))
        attn_out = attn_outputs[slot_id]
        ub.prepare_input_slot(slot_id, OP_O_PROJ, attn_out)
        b, t, d = attn_out.shape
        extq.submit_slot_command(
            OP_O_PROJ, layer_idx, slot_id, self.worker_id,
            b, t, d, 0, 0, 0, 0, 0
        )
        extq.slot_signal_ready(slot_id)
        o_proj = o_proj_modules[layer_idx]
        if isinstance(o_proj, SyncRemoteLoRALinear):
            deltas[slot_id] = o_proj.lora_B(o_proj.lora_A(o_proj.dropout(attn_out))) * o_proj.scaling
        else:
            deltas[slot_id] = None
        return epoch, deltas

    def o_proj_get_all(self, slots: Dict[int, SlotState], layer_idx: int,
                       deltas: Dict[int, torch.Tensor], out_dim: int):
        """Get O_PROJ results for all slots, add residual and LoRA delta."""
        raw = self._get_outputs_all(slots, [out_dim])
        for slot_id, state in slots.items():
            result = raw[slot_id][0]
            delta = deltas.get(slot_id)
            if delta is not None:
                result = result + delta
            state.hidden = state.residual + result


    def gate_up_submit_all(self, slots: Dict[int, SlotState], layer_idx: int,
                           fused_gate_up_modules: List) -> tuple:
        """Submit Gate/Up command and compute LoRA delta. Returns (epoch, deltas)."""
        epoch = extq.get_barrier_done_epoch()
        deltas = {}
        slot_id, state = next(iter(slots.items()))
        ub.prepare_input_slot(slot_id, OP_GATE_UP_FUSED, state.hidden)
        b, t, d = state.hidden.shape
        extq.submit_slot_command(
            OP_GATE_UP_FUSED, layer_idx, slot_id, self.worker_id,
            b, t, d, 0, 0, 0, 0, 0
        )
        extq.slot_signal_ready(slot_id)
        fused_gate_up = fused_gate_up_modules[layer_idx]
        if fused_gate_up.fused_lora and fused_gate_up.fused_lora.has_lora:
            deltas[slot_id] = fused_gate_up.fused_lora(state.hidden)
        else:
            deltas[slot_id] = (None, None)
        return epoch, deltas

    def gate_up_get_all(self, slots: Dict[int, SlotState], layer_idx: int,
                        deltas: Dict[int, tuple], dims: List[int]) -> Dict[int, tuple]:
        """Get Gate/Up results for all slots, add LoRA deltas."""
        raw = self._get_outputs_all(slots, dims)
        results = {}
        for slot_id in slots:
            gate, up = raw[slot_id]
            gate_delta, up_delta = deltas.get(slot_id, (None, None))
            if gate_delta is not None:
                gate = gate + gate_delta
            if up_delta is not None:
                up = up + up_delta
            results[slot_id] = (gate, up)
        return results


    def down_proj_submit_all(self, slots: Dict[int, SlotState], layer_idx: int,
                             mlp_intermediates: Dict[int, torch.Tensor],
                             down_proj_modules: List) -> tuple:
        """Submit Down_PROJ command and compute LoRA delta. Returns (epoch, deltas)."""
        epoch = extq.get_barrier_done_epoch()
        deltas = {}
        slot_id, state = next(iter(slots.items()))
        mlp_out = mlp_intermediates[slot_id]
        ub.prepare_input_slot(slot_id, OP_DOWN_PROJ, mlp_out)
        b, t, d = mlp_out.shape
        extq.submit_slot_command(
            OP_DOWN_PROJ, layer_idx, slot_id, self.worker_id,
            b, t, d, 0, 0, 0, 0, 0
        )
        extq.slot_signal_ready(slot_id)
        down_proj = down_proj_modules[layer_idx]
        if isinstance(down_proj, SyncRemoteLoRALinear):
            deltas[slot_id] = down_proj.lora_B(down_proj.lora_A(down_proj.dropout(mlp_out))) * down_proj.scaling
        else:
            deltas[slot_id] = None
        return epoch, deltas

    def down_proj_get_all(self, slots: Dict[int, SlotState], layer_idx: int,
                          deltas: Dict[int, torch.Tensor], out_dim: int):
        """Get Down_PROJ results for all slots, add residual and LoRA delta."""
        raw = self._get_outputs_all(slots, [out_dim])
        for slot_id, state in slots.items():
            result = raw[slot_id][0]
            delta = deltas.get(slot_id)
            if delta is not None:
                result = result + delta
            state.hidden = state.mlp_residual + result


    def lm_head_submit_all(self, slots: Dict[int, SlotState]) -> int:
        """Submit LM_HEAD command for the single active slot. Returns epoch to wait for."""
        epoch = extq.get_barrier_done_epoch()
        slot_id, state = next(iter(slots.items()))
        ub.prepare_hidden_for_lm_head_slot(slot_id, state.hidden)
        b, t, d = state.hidden.shape
        extq.submit_slot_command(
            OP_LM_HEAD, 0, slot_id, self.worker_id,
            b, t, d, 0, 0, 0, 0, 0
        )
        extq.slot_signal_ready(slot_id)
        return epoch

    def lm_head_get_all(self, slots: Dict[int, SlotState]):
        """Get next tokens for all slots."""
        for slot_id, state in slots.items():
            next_token = ub.get_next_token_slot(slot_id)

            if state.prefill_chunks:
                # Intermediate chunk: discard sampled token, load next chunk
                state.input_ids = state.prefill_chunks.pop(0)
                if not state.prefill_chunks:
                    state.prefill_chunks = None  # Next iteration is last chunk
            else:
                # Normal path: last chunk or decode — use sampled token
                if state.first_token_time is None:
                    state.first_token_time = time.perf_counter()
                    state.first_token_wall = time.time()
                state.generated_tokens.append(next_token)
                state.input_ids = torch.tensor([[next_token]], device=state.input_ids.device)


# FaaS Worker (HTTP Server)


class FaaSWorker:
    """HTTP server worker. Initializes once, serves multiple requests via slot-based join/leave."""

    def __init__(self, host: str = "localhost", port: int = DEFAULT_PORT,
                 device: str = WORKER_DEVICE, lora_id: str = None,
                 preloaded_tokenizer=None):
        self.host = host
        self.port = port
        self.device = device
        self.worker_device_idx = int(device.split(":")[-1]) if ":" in device else 0
        self.lora_id = lora_id or LORA_ADAPTER_ID

        # Register the adapter pool directory as trusted for path validation.
        # For local paths like "../sim-adapters/pool-10-r16/lora-0", trust the
        # parent directory so all sibling adapters in the same pool are allowed.
        _lora = self.lora_id
        if os.path.sep in _lora or _lora.startswith('.'):
            # Register the grandparent (e.g. sim-adapters/) so workers can be
            # swapped to any adapter pool, not just the one they started with.
            # Structure: sim-adapters/ → pool-X/ → lora-N  (3 levels)
            _real = os.path.realpath(_lora)
            register_adapter_base_dir(os.path.dirname(os.path.dirname(_real)))

        # Pre-loaded tokenizer from template process (inherited via fork COW)
        self._preloaded_tokenizer = preloaded_tokenizer

        # State (set during initialize)
        self.sock = None
        self.info = None
        self.worker_id = None
        self.client = None  # SlotClient
        self.model = None
        self.tokenizer = None
        self.initialized = False
        self.running = False

        # Hot-swap support
        self.adapter_manager = None

        # Intra-adapter batching
        self.inference_engine: Optional['BatchedInferenceEngine'] = None

    def initialize(self) -> bool:
        """Connect to aggregator, build model, load tokenizer."""
        if self.initialized:
            return True

        t_start = time.perf_counter()
        print(f"[FaaSWorker] Connecting to aggregator at {self.host}:{self.port}...", flush=True)

        try:
            self.sock, self.info = connect(self.host, self.port)
            self.worker_id = self.info['worker_id']
            t_connect = time.perf_counter()
            log(self.worker_id, "Connected to aggregator")

            torch.cuda.set_device(self.worker_device_idx)
            torch.cuda.synchronize(self.worker_device_idx)  # Force CUDA init
            t_cuda = time.perf_counter()

            # Create SlotClient (uses TCP socket for per-request slot allocation)
            qkv_dim = self.info['hidden_size'] + self.info['num_kv_heads'] * self.info['head_dim'] * 2
            self.client = SlotClient(
                worker_id=self.worker_id,
                aggregator_device_idx=self.info["aggregator_device_idx"],
                worker_device_idx=self.worker_device_idx,
                hidden_dim=self.info.get("hidden_size", 4096),
                intermediate_dim=self.info.get("intermediate_size", 14336),
                qkv_dim=qkv_dim,
                sock=self.sock,
                input_buffer_size=self.info.get("input_buffer_size"),
                output_buffer_size=self.info.get("output_buffer_size")
            )
            # Initialize slot manager and open queue
            extq.open_queue(
                self.info["queue"]["shm_name"],
                self.info["queue"]["capacity"],
                self.info["queue"]["max_tickets"],
            )
            self.client.initialize()
            t_client = time.perf_counter()

            self.model = build_hollow_model(
                self.info, self.client, self.worker_id,
                self.device, self.lora_id
            )
            t_model = time.perf_counter()

            # The template preloads the tokenizer from config.BASE_MODEL_ID; if
            # the aggregator runs a different model (--model), the vocab won't
            # match the sampled token IDs and decoded text is garbage, so
            # reload with the correct model.
            #
            # Compare len(tokenizer), not .vocab_size: for models with reserved
            # special tokens (e.g. Llama-3.1) .vocab_size is the raw BPE count
            # (128000) while len() is the usable vocab (128256) matching the
            # embedding size. Using .vocab_size mismatches on every spawn and
            # silently defeats the preload.
            agg_vocab = self.info.get('vocab_size')
            preloaded_len = len(self._preloaded_tokenizer) if self._preloaded_tokenizer is not None else None
            if (self._preloaded_tokenizer is not None
                    and (agg_vocab is None or preloaded_len == agg_vocab)):
                self.tokenizer = self._preloaded_tokenizer
                self._preloaded_tokenizer = None
                log(self.worker_id, "  TOKENIZER TIMING: preloaded (0.00s)")
            else:
                if self._preloaded_tokenizer is not None:
                    log(self.worker_id,
                        f"  Preloaded tokenizer vocab {preloaded_len}"
                        f" != aggregator vocab {agg_vocab}; reloading")
                    self._preloaded_tokenizer = None
                self.tokenizer = load_tokenizer_from_cache(self.info, self.worker_id)
            t_tokenizer = time.perf_counter()

            # Initialize adapter manager for hot-swap support
            self.adapter_manager = AdapterManager(
                worker_id=self.worker_id,
                device=self.device,
                dtype=DTYPE,
                info=self.info
            )
            # Set initial adapter info — check local dir first, then HF cache
            local_config = os.path.join(self.lora_id, 'adapter_config.json')
            if os.path.isfile(local_config):
                config_file = local_config
            else:
                from huggingface_hub import try_to_load_from_cache, hf_hub_download
                config_file = try_to_load_from_cache(self.lora_id, 'adapter_config.json')
                if config_file is None:
                    config_file = hf_hub_download(self.lora_id, 'adapter_config.json')
            with open(config_file) as f:
                lora_config = json.load(f)
            self.adapter_manager.set_adapter(self.lora_id, lora_config['r'], lora_config)

            # Initialize and start the batched inference engine
            self.inference_engine = BatchedInferenceEngine(worker=self)
            self.inference_engine.start()
            t_engine = time.perf_counter()

            log(self.worker_id,
                f"INIT TIMING: connect={t_connect-t_start:.2f}s cuda_init={t_cuda-t_connect:.2f}s "
                f"client={t_client-t_cuda:.2f}s model={t_model-t_client:.2f}s tokenizer={t_tokenizer-t_model:.2f}s "
                f"engine={t_engine-t_tokenizer:.2f}s TOTAL={t_engine-t_start:.2f}s")

            self.initialized = True
            log(self.worker_id, "FaaSWorker initialized (INTRA-ADAPTER BATCHING) - ready for requests!")
            return True

        except Exception as e:
            print(f"[FaaSWorker] Initialization failed: {e}", flush=True)
            traceback.print_exc()
            return False

    def start_http_server(self, http_port: int):
        """Start Flask HTTP server."""
        # Flask is pre-imported at module level for accurate cold start timing
        app = Flask(f"faas_worker_{self.worker_id}")

        @app.route('/health', methods=['GET'])
        def health():
            resp = {
                "status": "ready" if self.initialized else "initializing",
                "worker_id": self.worker_id,
                "device": self.device,
                "mode": "single-request",
                "active_slots": 0,
                "pending_slots": 0,
                "max_seq_len": 0,
                "total_seq_len": 0,
                "num_prefill": 0,
                "num_decode": 0,
            }
            if self.inference_engine:
                engine = self.inference_engine
                active = engine.slot_manager.active_slots

                resp["active_slots"] = len(active)

                with engine._pending_lock:
                    resp["pending_slots"] = len(engine.pending_slots)

                # KV cache sequence length stats
                seq_lens = list(engine.paged_kv_cache.slot_seq_lens.values())
                if seq_lens:
                    resp["max_seq_len"] = max(seq_lens)
                    resp["total_seq_len"] = sum(seq_lens)

                # Prefill vs decode counts from active slots
                num_prefill = 0
                num_decode = 0
                for state in active.values():
                    if state.prefill_chunks:
                        num_prefill += 1
                    else:
                        num_decode += 1
                resp["num_prefill"] = num_prefill
                resp["num_decode"] = num_decode

            # GPU memory info (torch CUDA context already active)
            try:
                free_mem, total_mem = torch.cuda.mem_get_info(self.device)
                resp["gpu_memory_free_mb"] = free_mem // (1024 * 1024)
                resp["gpu_memory_total_mb"] = total_mem // (1024 * 1024)
            except Exception:
                pass

            return jsonify(resp)

        _inference_semaphore = threading.Semaphore(MAX_SLOTS)

        @app.route('/inference', methods=['POST'])
        def inference():
            if not self.running:
                return jsonify({
                    "error": "Worker is shutting down",
                    "success": False
                }), 503
            if not _inference_semaphore.acquire(blocking=False):
                return jsonify({
                    "error": f"Worker at capacity ({MAX_SLOTS} slots)",
                    "success": False
                }), 503

            data = request.get_json() or {}
            prompt = data.get('prompt', '')
            max_tokens = max(1, min(int(data.get('max_tokens', 256)), MAX_SEQ_LEN))

            # Sampling parameters — clamp to safe ranges
            do_sample  = bool(data.get('do_sample', False))
            temperature = max(0.01, min(float(data.get('temperature', 1.0)), 100.0))
            top_k       = max(0, min(int(data.get('top_k', 0)), 100000))
            top_p       = max(0.0, min(float(data.get('top_p', 1.0)), 1.0))
            # Validate arrival_time — reject if >60s off to prevent metric skewing
            _now = time.perf_counter()
            arrival_time = data.get('arrival_time')
            if arrival_time is not None:
                try:
                    arrival_time = float(arrival_time)
                    if abs(_now - arrival_time) > 60.0:
                        arrival_time = _now
                except (TypeError, ValueError):
                    arrival_time = _now

            if not prompt:
                _inference_semaphore.release()
                return jsonify({"error": "No prompt provided", "success": False}), 400

            # Pre-tokenization character limit — prevents tokenizer DoS from huge strings
            _max_prompt_chars = MAX_SEQ_LEN * 6
            if len(prompt) > _max_prompt_chars:
                prompt = prompt[-_max_prompt_chars:]

            # Use batched inference engine
            t_start = time.perf_counter()

            req = InferenceRequest(
                request_id=str(uuid.uuid4()),
                prompt=prompt,
                max_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                submit_time=arrival_time if arrival_time is not None else _now,
                slo_deadline_wall=data.get('slo_deadline_wall'),
            )

            self.inference_engine.submit_request(req)

            # Wait for result (blocks until inference complete)
            try:
                result = req.future.result(timeout=_INFERENCE_TIMEOUT)
                t_end = time.perf_counter()
                result['e2e_ms'] = round((t_end - t_start) * 1000, 2)
                result['worker_id'] = self.worker_id
                result['adapter_id'] = self.adapter_manager.get_current_adapter() if self.adapter_manager else self.lora_id
                status = 200 if result.get('success') else 500
                return jsonify(result), status
            except Exception as e:
                return jsonify({"error": str(e), "success": False}), 500
            finally:
                _inference_semaphore.release()

        @app.route('/shutdown', methods=['POST'])
        def shutdown():
            self.running = False
            def do_shutdown():
                # Wait for active inference slots to drain before exiting
                # so we don't kill a worker mid-barrier and crash the aggregator
                deadline = time.time() + 120  # max wait 2 minutes
                while time.time() < deadline:
                    try:
                        engine = self.inference_engine
                        if engine and hasattr(engine, 'slot_manager'):
                            active = engine.slot_manager.get_active_slots()
                            if not active:
                                break
                        else:
                            break
                    except Exception:
                        break
                    time.sleep(0.1)
                time.sleep(0.2)  # brief grace for final cleanup
                os._exit(0)
            threading.Thread(target=do_shutdown, daemon=True).start()
            return jsonify({"status": "shutting_down"})

        @app.route('/swap_adapter', methods=['POST'])
        def swap_adapter():
            """Hot-swap to a new LoRA adapter."""
            data = request.get_json() or {}
            new_adapter_id = data.get('adapter_id')

            if not new_adapter_id:
                return jsonify({'error': 'adapter_id required', 'success': False}), 400

            try:
                _validate_adapter_id(new_adapter_id)
            except ValueError as e:
                return jsonify({'error': f'Invalid adapter_id: {e}', 'success': False}), 400

            if not self.initialized:
                return jsonify({'error': 'Worker not initialized', 'success': False}), 503

            if not self.adapter_manager:
                return jsonify({'error': 'Adapter manager not available', 'success': False}), 503

            # Check if already using this adapter
            if self.adapter_manager.get_current_adapter() == new_adapter_id:
                return jsonify({
                    'success': True,
                    'message': 'Already using this adapter',
                    'adapter_id': new_adapter_id,
                    'total_time_ms': 0
                })

            try:
                result = self.adapter_manager.swap_adapter(new_adapter_id, self.model)
                status = 200 if result.get('success') else 500
                return jsonify(result), status
            except Exception as e:
                return jsonify({'error': str(e), 'success': False}), 500

        @app.route('/adapter_info', methods=['GET'])
        def adapter_info():
            """Get current adapter information."""
            return jsonify({
                'adapter_id': self.adapter_manager.get_current_adapter() if self.adapter_manager else None,
                'rank': self.adapter_manager.current_rank if self.adapter_manager else 0,
                'is_swapping': self.adapter_manager.is_swapping() if self.adapter_manager else False,
                'worker_id': self.worker_id,
                'device': self.device,
                'initialized': self.initialized
            })

        self.running = True
        log(self.worker_id, f"Starting HTTP server on port {http_port}")
        app.run(host='0.0.0.0', port=http_port, threaded=True, use_reloader=False)

    def shutdown(self):
        """Clean up resources."""
        self.running = False
        if self.inference_engine:
            self.inference_engine.stop()
        if self.client:
            self.client.cleanup()
        if self.sock:
            try:
                signal_done(self.sock)
                self.sock.close()
            except Exception:
                pass  # Ignore socket errors during shutdown
        log(self.worker_id if self.worker_id else 0, "FaaSWorker shutdown complete")


# INTRA-ADAPTER BATCHING: Inference Engine


class BatchedInferenceEngine:
    """Single-threaded inference engine for one request per worker."""

    def __init__(self, worker: FaaSWorker):
        self.worker = worker
        self.running = False
        # Request queue: HTTP threads submit here
        self.request_queue: Queue[InferenceRequest] = Queue()

        # Event for instant wake-up when new requests arrive
        self._request_event = threading.Event()

        # Slot management (no worker-side limit; controller enforces concurrency)
        self.slot_manager = IntraAdapterSlotManager(
            base_client=worker.client,
        )

        # Pending slots: joined but not yet active (slot_id -> SlotState)
        self.pending_slots: Dict[int, SlotState] = {}
        self._pending_lock = threading.Lock()  # Protects pending_slots
        # EMA of work still owed after the deadline check passes: prune point
        # -> first token (slot claim, tokenize, prefill). Measured, not
        # guessed; 0 until observed so behaviour is unchanged at startup.
        self._post_admit_s = 0.0
        self._post_admit_n = 0

        # Batched operations helper
        self.batched_ops = BatchedAggregatorOps(
            slot_manager=self.slot_manager,
            worker_id=worker.worker_id,
            dtype_code=worker.client.dtype_code
        )
        # Model components (set after initialization)
        self.model = None
        self.layers = None
        self.final_norm = None
        self.num_layers = 0

        self._inference_thread = None

    def start(self):
        """Start the inference engine thread."""
        if self.running:
            return

        # Get model components
        self.model = self.worker.model
        decoder = self.model.model
        self.layers = list(decoder.layers)
        self.final_norm = decoder.norm
        self.num_layers = len(self.layers)

        # Initialize paged attention
        attn0 = self.layers[0].self_attn
        self.num_heads = attn0.num_heads
        self.num_kv_heads = attn0.num_kv_heads
        self.head_dim = attn0.head_dim
        self.attn_scale = 1.0 / (self.head_dim ** 0.5)

        self.paged_kv_cache = PagedKVCache(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            block_size=PAGED_ATTENTION_BLOCK_SIZE,
            max_slots=MAX_SLOTS,
            max_seq_len=MAX_SEQ_LEN,
            device=self.worker.device,
            dtype=DTYPE,
        )
        self.cos_sin_cache = build_vllm_cos_sin_cache(
            attn0.rotary_emb, MAX_SEQ_LEN,
            self.worker.device, DTYPE
        )

        # Pre-allocate reusable attention buffers for decode (q_len=1)
        max_decode = MAX_SLOTS
        dev = self.worker.device
        self._decode_attn_bufs = {
            'q': torch.empty(max_decode, 1, self.num_heads, self.head_dim,
                             device=dev, dtype=DTYPE),
            'k': torch.empty(max_decode, 1, self.num_kv_heads, self.head_dim,
                             device=dev, dtype=DTYPE),
            'v': torch.empty(max_decode, 1, self.num_kv_heads, self.head_dim,
                             device=dev, dtype=DTYPE),
        }
        # Pre-allocate reusable attention buffers for prefill (q_len=PREFILL_CHUNK_SIZE)
        self._prefill_attn_bufs = {
            'q': torch.empty(MAX_SLOTS, PREFILL_CHUNK_SIZE, self.num_heads, self.head_dim,
                             device=dev, dtype=DTYPE),
            'k': torch.empty(MAX_SLOTS, PREFILL_CHUNK_SIZE, self.num_kv_heads, self.head_dim,
                             device=dev, dtype=DTYPE),
            'v': torch.empty(MAX_SLOTS, PREFILL_CHUNK_SIZE, self.num_kv_heads, self.head_dim,
                             device=dev, dtype=DTYPE),
        }

        log(self.worker.worker_id,
            f"Mode: single-request (max_slots={MAX_SLOTS}), "
            f"PagedAttention: {self.paged_kv_cache.num_blocks} blocks, "
            f"block_size={PAGED_ATTENTION_BLOCK_SIZE}")

        self.running = True
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name=f"InferenceEngine-{self.worker.worker_id}",
            daemon=True
        )
        self._inference_thread.start()
        log(self.worker.worker_id, "BatchedInferenceEngine started")

    def stop(self):
        """Stop the inference engine."""
        self.running = False
        if self._inference_thread:
            self._inference_thread.join(timeout=5.0)

    def submit_request(self, request: InferenceRequest):
        """Submit a request for processing. Called by HTTP threads."""
        self.request_queue.put(request)
        self._request_event.set()

    def _inference_loop(self):
        """Main inference loop - runs in single thread."""
        eos_token_id = self.worker.tokenizer.eos_token_id
        while self.running:
            try:
                # BOUNDARY: accept + activate
                self._accept_pending_requests()
                with self._pending_lock:
                    has_pending = bool(self.pending_slots)
                if has_pending:
                    self._try_activate_pending_slots()

                active_slots = self.slot_manager.get_active_slots()
                if not active_slots:
                    if has_pending:
                        # Slots are joining — spin until aggregator activates them
                        continue
                    self._request_event.wait(timeout=_IDLE_POLL_INTERVAL)
                    self._request_event.clear()
                    continue

                # TIGHT LOOP: skip all boundary overhead per token
                while True:
                    self._generate_one_token_all(active_slots)

                    # Lightweight inline completion check (no lock, no dict copy)
                    any_completed = False
                    for state in active_slots.values():
                        n = len(state.generated_tokens)
                        if n >= state.request.max_tokens:
                            any_completed = True
                            break
                        if n > 0 and eos_token_id is not None and state.generated_tokens[-1] == eos_token_id:
                            any_completed = True
                            break

                    if any_completed:
                        break  # exit to full _handle_completions()
                    with self._pending_lock:
                        has_pending = bool(self.pending_slots)
                    if not self.request_queue.empty() or has_pending:
                        break  # new request arrived, handle at boundary

                self._handle_completions(active_slots)

            except Exception as e:
                log(self.worker.worker_id, f"[InferenceEngine] Error: {e}")
                traceback.print_exc()
                self._fail_all_active_slots(e)

    def _accept_pending_requests(self):
        """Accept pending requests from queue at token boundary."""
        # Accept all queued requests (controller enforces concurrency limit)
        while True:
            try:
                req = self.request_queue.get_nowait()
            except Empty:
                break  # No more pending requests

            # SLO deadline pruning. The controller's queue budget bounds only
            # the wait for a WORKER; this queue -- waiting for a slot and for
            # the lock-step batch to admit us -- is a second, unbounded stage.
            # A request whose deadline will have passed by the time we produce
            # a first token cannot meet its SLO however fast we run, so slot
            # and GPU time spent on it is spent on an already-lost request.
            # The margin is the measured prune-point -> first-token interval;
            # without it a request admitted just under its deadline lands just
            # over it. Mirrors ServerlessLoRA and ServerlessLLM exactly.
            if req.slo_deadline_wall is not None:
                _margin = _guarded_margin(self._post_admit_s,
                                          self._post_admit_n, _SLO_S)
                if (time.time() + _margin) > req.slo_deadline_wall:
                    req.future.set_result({
                        "success": False,
                        "error": "SLO deadline passed while queued at worker",
                    })
                    continue

            try:
                # Claim slot (TCP to aggregator, but don't signal join yet)
                slot_id = self.slot_manager.claim_slot()

                # Tokenize input
                inputs = self.worker.tokenizer(req.prompt, return_tensors="pt")
                input_ids = inputs.input_ids.to(self.worker.device)
                seq_len = input_ids.size(1)

                # Truncate input if it exceeds MAX_SEQ_LEN (keep last tokens for context)
                if seq_len >= MAX_SEQ_LEN:
                    input_ids = input_ids[:, -(MAX_SEQ_LEN - 1):]
                    seq_len = input_ids.size(1)

                # Cap max_tokens so total sequence fits in KV cache
                max_allowed = MAX_SEQ_LEN - seq_len
                if req.max_tokens > max_allowed:
                    req.max_tokens = max_allowed

                # Chunked prefill: split long prompts into chunks
                if seq_len > PREFILL_CHUNK_SIZE:
                    chunks = list(input_ids.split(PREFILL_CHUNK_SIZE, dim=1))
                    first_chunk = chunks[0]
                    remaining_chunks = chunks[1:]
                else:
                    first_chunk = input_ids
                    remaining_chunks = None

                # Set sampling state (out-of-band, before signal_join)
                extq.slot_set_pending_sampling(
                    slot_id, req.do_sample, req.temperature,
                    req.top_k, req.top_p
                )

                # Initialize paged cache tracking for this slot
                self.paged_kv_cache.slot_seq_lens[slot_id] = 0

                # Create slot state (but don't join yet)
                state = SlotState(
                    request=req,
                    slot_id=slot_id,
                    input_ids=first_chunk,
                    generated_tokens=[],
                    prefill_chunks=remaining_chunks,
                    prompt_tokens=seq_len,
                    slot_allocated_time=time.perf_counter(),
                )

                # Add to pending - will join when current batch is complete
                with self._pending_lock:
                    self.pending_slots[slot_id] = state
                chunks_info = f", chunks={len(remaining_chunks)+1}" if remaining_chunks else ""
                log(self.worker.worker_id,
                    f"Request {req.request_id[:8]} prepared slot {slot_id} "
                    f"(input={seq_len} tokens, first_chunk={first_chunk.size(1)}{chunks_info}, pending activation)")

            except Exception as e:
                req.future.set_exception(e)

    def _try_activate_pending_slots(self):
        """Try to activate pending slots at token boundary (non-blocking)."""
        with self._pending_lock:
            if not self.pending_slots:
                return

            activated = []
            timed_out = []
            current_time = time.perf_counter()

            for slot_id, state in list(self.pending_slots.items()):
                # Step 1: Signal join if not already done
                if not state.joined:
                    extq.slot_signal_join(slot_id)
                    state.joined = True
                    state.join_time = current_time
                    log(self.worker.worker_id,
                        f"Request {state.request.request_id[:8]} signaled join on slot {slot_id}")

                # Step 2: Check if active (non-blocking)
                if extq.slot_is_active(slot_id):
                    activated.append(slot_id)
                elif state.join_time and (current_time - state.join_time) > _SLOT_ACTIVATION_TIMEOUT:
                    timed_out.append((slot_id, state))

            # Handle timeouts
            for slot_id, state in timed_out:
                log(self.worker.worker_id,
                    f"[Slot-{slot_id}] activation TIMEOUT after {_SLOT_ACTIVATION_TIMEOUT}s")
                self.pending_slots.pop(slot_id)
                extq.slot_signal_leave(slot_id)
                self.slot_manager.release_slot(slot_id)
                state.request.future.set_exception(
                    RuntimeError(f"Slot activation timeout ({_SLOT_ACTIVATION_TIMEOUT}s)"))

            # Step 3: Move activated slots to active
            for slot_id in activated:
                state = self.pending_slots.pop(slot_id)
                self.slot_manager.add_active_slot(slot_id, state)
                log(self.worker.worker_id,
                    f"Request {state.request.request_id[:8]} joined slot {slot_id} "
                    f"(input={state.input_ids.size(1)} tokens)")

    def _handle_completions(self, active_slots=None):
        """Handle completed requests (max_tokens reached or EOS generated)."""
        completed = []
        eos_token_id = self.worker.tokenizer.eos_token_id

        if active_slots is None:
            active_slots = self.slot_manager.get_active_slots()

        for slot_id, state in active_slots.items():
            # Check completion conditions
            reached_max_tokens = len(state.generated_tokens) >= state.request.max_tokens
            hit_eos = (
                state.generated_tokens and
                eos_token_id is not None and
                state.generated_tokens[-1] == eos_token_id
            )

            if reached_max_tokens or hit_eos:
                # Record end time for metrics
                end_time = time.perf_counter()

                # Signal leave IMMEDIATELY — before any Python bookkeeping.
                # This prevents mid-op stalls in the aggregator which would
                # otherwise spin-wait up to 10ms per occurrence while we
                # decode tokens and compute metrics below.
                self.paged_kv_cache.free_slot(slot_id)
                extq.slot_signal_leave(slot_id)

                # Wait for leave acknowledgment (spin-then-yield)
                wait_start = time.perf_counter()
                spin_count = 0
                while extq.slot_is_active(slot_id):
                    if time.perf_counter() - wait_start > _SLOT_LEAVE_TIMEOUT:
                        break
                    spin_count += 1
                    if spin_count > _SPIN_BEFORE_YIELD:
                        time.sleep(0)  # yield, not 1ms sleep

                self.slot_manager.release_slot(slot_id)
                completed.append(slot_id)

                # Now do Python bookkeeping (tokenizer decode, metrics, future)
                # Decode output
                text = self.worker.tokenizer.decode(
                    state.generated_tokens,
                    skip_special_tokens=True
                )

                finish_reason = "eos" if hit_eos else "max_tokens"
                num_tokens = len(state.generated_tokens)

                # Calculate timing metrics (in milliseconds)
                submit_time = state.request.submit_time
                slot_allocated_time = state.slot_allocated_time or submit_time
                first_token_time = state.first_token_time or end_time

                # Post-admit margin: slot allocation (the prune point) -> first
                # token. Both are perf_counter, so the difference is a valid
                # duration even though the deadline it guards is wall-clock.
                # Lives here, not in BatchedAggregatorOps where first_token_time
                # is stamped: the EMA state belongs to this engine, and touching
                # self there raised AttributeError on every generated token.
                if state.first_token_time and state.slot_allocated_time:
                    _obs = max(0.0, state.first_token_time
                                    - state.slot_allocated_time)
                    self._post_admit_s = (
                        _obs if self._post_admit_n == 0
                        else 0.2 * _obs + 0.8 * self._post_admit_s)
                    self._post_admit_n += 1

                # Use wall clock for TTFT when arrival_time came from controller
                # (perf_counter is per-process, can't subtract across processes)
                first_token_wall = getattr(state, 'first_token_wall', None) or time.time()
                if submit_time > 1e9:  # Wall clock (time.time()) vs perf_counter
                    ttft_ms = (first_token_wall - submit_time) * 1000
                else:
                    ttft_ms = (first_token_time - submit_time) * 1000
                queue_wait_ms = (slot_allocated_time - submit_time) * 1000 if submit_time < 1e9 else 0
                decode_time_ms = (end_time - first_token_time) * 1000

                # Calculate throughput metrics (tokens per second)
                # decode_time starts AFTER first token, so use (num_tokens - 1) for decode phase
                decode_throughput = ((num_tokens - 1) / decode_time_ms * 1000) if decode_time_ms > 0 and num_tokens > 1 else 0
                # Time differences from perf_counter are in seconds, so no multiplication needed
                gen_time_s = end_time - slot_allocated_time
                total_time_s = end_time - submit_time
                gen_throughput = (num_tokens / gen_time_s) if gen_time_s > 0 else 0
                total_throughput = (num_tokens / total_time_s) if total_time_s > 0 else 0

                # Set result with all metrics
                state.request.future.set_result({
                    'success': True,
                    'text': text,
                    'tokens': num_tokens,
                    'prompt_tokens': state.prompt_tokens,
                    'completion_tokens': num_tokens,
                    'finish_reason': finish_reason,
                    # Timing metrics — precise for internal benchmark use
                    'ttft_ms': round(ttft_ms, 2),
                    'queue_wait_ms': round(queue_wait_ms, 2),
                    'decode_time_ms': round(decode_time_ms, 2),
                    # Throughput metrics
                    'decode_throughput': round(decode_throughput, 2),
                    'gen_throughput': round(gen_throughput, 2),
                    'total_throughput': round(total_throughput, 2),
                })

                log(self.worker.worker_id,
                    f"Request {state.request.request_id[:8]} completed ({finish_reason}): "
                    f"{len(state.generated_tokens)} tokens")

        for slot_id in completed:
            self.slot_manager.remove_active_slot(slot_id)

    def _fail_all_active_slots(self, error: Exception):
        """Clean up all active and pending slots on error."""
        # Fail active slots
        active_slots = self.slot_manager.get_active_slots()
        for slot_id, state in list(active_slots.items()):
            try:
                if state.request.future and not state.request.future.done():
                    state.request.future.set_exception(error)
                self.paged_kv_cache.free_slot(slot_id)
                extq.slot_signal_leave(slot_id)
                self.slot_manager.release_slot(slot_id)
            except Exception as cleanup_error:
                log(self.worker.worker_id, f"Cleanup error for slot {slot_id}: {cleanup_error}")
            self.slot_manager.remove_active_slot(slot_id)

        # Fail pending slots
        with self._pending_lock:
            for slot_id, state in list(self.pending_slots.items()):
                try:
                    if state.request.future and not state.request.future.done():
                        state.request.future.set_exception(error)
                    # Pending slots haven't joined yet, just release
                    self.slot_manager.release_slot(slot_id)
                except Exception as cleanup_error:
                    log(self.worker.worker_id, f"Cleanup error for pending slot {slot_id}: {cleanup_error}")
            self.pending_slots.clear()

    def _teardown_slot(self, sid, state, error=None):
        """Canonical slot teardown used by all abort/cleanup paths.

        Mirrors the normal-completion release sequence: free the KV blocks,
        signal the aggregator we're leaving, then release the slot. That last
        step resets aggregator-side ownership AND zeroes the slot's IPC buffers
        (residue prevention) — skipping it (as the eviction and KV-exhaustion
        paths previously did) permanently leaks the slot from the pool and
        leaves the departing tenant's hidden states resident in GPU memory.
        Finally drop it from the active set and fail the request future if it is
        still pending.
        """
        try:
            self.paged_kv_cache.free_slot(sid)
            extq.slot_signal_leave(sid)
            self.slot_manager.release_slot(sid)
        except Exception as cleanup_error:
            log(self.worker.worker_id, f"Teardown error for slot {sid}: {cleanup_error}")
        finally:
            self.slot_manager.remove_active_slot(sid)
            if error is not None and state.request.future and not state.request.future.done():
                state.request.future.set_exception(error)

    # Layer profiling (RQ5): set PROFILE_LAYERS=1 to enable
    _profile_layers = os.environ.get("PROFILE_LAYERS") == "1"
    _profile_token_idx = 0
    _profile_records = []  # list of dicts with per-token timings

    def _generate_one_token_all(self, slots: Dict[int, SlotState]):
        """Generate one token for all active slots using unified flash_attn_with_kvcache."""
        if self._profile_layers:
            return self._generate_one_token_all_profiled(slots)

        # Check for evicted slots — aggregator may have timed out and removed them.
        # Without this check a stuck-then-recovered worker would read stale output buffers
        # and produce garbage tokens. Abort evicted slots cleanly before doing any work.
        for sid in list(slots.keys()):
            if extq.is_slot_evicted(sid):
                extq.clear_evicted_slot(sid)
                state = slots.pop(sid)
                self._teardown_slot(sid, state, RuntimeError(
                    f"Slot {sid} evicted by aggregator (barrier timeout) — request aborted"))
                log(self.worker.worker_id, f"Slot {sid} detected as aggregator-evicted, request aborted")
        if not slots:
            return  # all slots evicted, nothing to generate

        # Split slots by phase (for norm/activation batching, not attention)
        prefill_slots, decode_slots = WorkerBatchedOps.split_by_phase(slots)

        for sid, state in list(slots.items()):
            needed = self.paged_kv_cache.slot_seq_lens.get(sid, 0) + state.input_ids.size(1)
            try:
                self.paged_kv_cache.ensure_blocks_for_length(sid, needed)
            except RuntimeError as e:
                # Fail only this over-budget request; co-resident slots continue.
                slots.pop(sid, None)
                self._teardown_slot(sid, state, RuntimeError(f"KV cache exhausted: {e}"))
        if not slots:
            return  # every active slot failed allocation — nothing to generate

        # Group all slots by q_len for flash_attn batching
        qlen_groups = defaultdict(list)
        for sid, state in slots.items():
            qlen_groups[state.input_ids.size(1)].append(sid)

        # Pre-compute attention metadata for ALL q_len groups (constant across all layers)
        attn_metas = {}
        dev = self.worker.device
        for q_len, group_sids in qlen_groups.items():
            n = len(group_sids)
            cache_seqlens = self.paged_kv_cache.get_seq_lens_tensor(group_sids)
            if q_len == 1:
                positions = cache_seqlens.long()
                bufs = self._decode_attn_bufs
            else:
                offsets = torch.arange(q_len, device=dev)
                positions = (cache_seqlens.long().unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1)
                bufs = self._prefill_attn_bufs
            attn_metas[q_len] = {
                'cache_seqlens': cache_seqlens,
                'block_tables': self.paged_kv_cache.get_block_table_tensor(group_sids),
                'positions': positions,
                'bufs_q': bufs['q'][:n, :q_len],
                'bufs_k': bufs['k'][:n, :q_len],
                'bufs_v': bufs['v'][:n, :q_len],
            }

        epoch = self.batched_ops.embed_submit_all(slots)
        self.batched_ops.barrier_sync(epoch)
        self.batched_ops.embed_get_all(slots)

        for layer_idx in range(self.num_layers):
            layer = self.layers[layer_idx]
            attn = layer.self_attn
            mlp = layer.mlp
            fused_qkv = attn._fused_qkv
            fused_gate_up = mlp._fused_gate_up

            for sid, state in prefill_slots.items():
                state.residual = state.hidden
                state.hidden = layer.input_layernorm(state.hidden)

            WorkerBatchedOps.batch_layernorm(
                decode_slots, layer.input_layernorm, save_residual=True
            )

            epoch, qkv_deltas = self.batched_ops.qkv_submit_all(
                slots, layer_idx, [l.self_attn._fused_qkv for l in self.layers]
            )
            self.batched_ops.barrier_sync(epoch)
            qkv_results = self.batched_ops.qkv_get_all(
                slots, layer_idx, qkv_deltas, fused_qkv.dims
            )

            attn_outputs = {}
            for q_len, group_sids in qlen_groups.items():
                group_out = WorkerBatchedOps.batch_flash_attention(
                    group_sids, qkv_results, self.paged_kv_cache,
                    self.cos_sin_cache, layer_idx,
                    self.num_heads, self.num_kv_heads, self.head_dim,
                    attn_meta=attn_metas[q_len],
                )
                attn_outputs.update(group_out)

            epoch, o_deltas = self.batched_ops.o_proj_submit_all(
                slots, layer_idx, attn_outputs, [l.self_attn.o_proj for l in self.layers]
            )
            self.batched_ops.barrier_sync(epoch)
            self.batched_ops.o_proj_get_all(
                slots, layer_idx, o_deltas, attn.hidden_size
            )

            for sid, state in prefill_slots.items():
                state.mlp_residual = state.hidden
                state.hidden = layer.post_attention_layernorm(state.hidden)

            WorkerBatchedOps.batch_layernorm(
                decode_slots, layer.post_attention_layernorm, save_mlp_residual=True
            )

            epoch, gate_up_deltas = self.batched_ops.gate_up_submit_all(
                slots, layer_idx, [l.mlp._fused_gate_up for l in self.layers]
            )
            self.batched_ops.barrier_sync(epoch)
            gate_up_results = self.batched_ops.gate_up_get_all(
                slots, layer_idx, gate_up_deltas, fused_gate_up.dims
            )

            mlp_intermediates = {}
            for sid, state in prefill_slots.items():
                gate, up = gate_up_results[sid]
                mlp_intermediates[sid] = F.silu(gate) * up

            mlp_intermediates.update(
                WorkerBatchedOps.batch_mlp_activation(decode_slots, gate_up_results)
            )

            epoch, down_deltas = self.batched_ops.down_proj_submit_all(
                slots, layer_idx, mlp_intermediates, [l.mlp.down_proj for l in self.layers]
            )
            self.batched_ops.barrier_sync(epoch)
            self.batched_ops.down_proj_get_all(
                slots, layer_idx, down_deltas, attn.hidden_size
            )

        all_sids = list(slots.keys())
        all_counts = [slots[sid].input_ids.size(1) for sid in all_sids]
        self.paged_kv_cache.advance_seq_lens_by(all_sids, all_counts)

        for sid, state in prefill_slots.items():
            state.hidden = self.final_norm(state.hidden)
            if state.prefill_chunks:
                state.hidden = state.hidden[:, -1:, :]

        WorkerBatchedOps.batch_final_norm(decode_slots, self.final_norm)

        # LM_HEAD (Aggregator batches - no LoRA)
        epoch = self.batched_ops.lm_head_submit_all(slots)
        self.batched_ops.barrier_sync(epoch)
        self.batched_ops.lm_head_get_all(slots)

    def _generate_one_token_all_profiled(self, slots: Dict[int, SlotState]):
        """Profiled version: times GATHER, BASE_GEMM, SCATTER, LORA, ATTN per token."""
        sync = torch.cuda.synchronize
        pc = time.perf_counter

        # Wall-clock the entire function
        sync()
        wall_start = pc()

        prefill_slots, decode_slots = WorkerBatchedOps.split_by_phase(slots)
        n_workers = len(slots)

        # Skip profiling for prefill tokens (mixed seq_len complicates timing)
        has_prefill = bool(prefill_slots)

        # Pre-layer setup (identical to original)
        for sid, state in slots.items():
            needed = self.paged_kv_cache.slot_seq_lens.get(sid, 0) + state.input_ids.size(1)
            self.paged_kv_cache.ensure_blocks_for_length(sid, needed)

        qlen_groups = defaultdict(list)
        for sid, state in slots.items():
            qlen_groups[state.input_ids.size(1)].append(sid)

        dev = self.worker.device
        attn_metas = {}
        for q_len, group_sids in qlen_groups.items():
            n = len(group_sids)
            cache_seqlens = self.paged_kv_cache.get_seq_lens_tensor(group_sids)
            if q_len == 1:
                positions = cache_seqlens.long()
                bufs = self._decode_attn_bufs
            else:
                offsets = torch.arange(q_len, device=dev)
                positions = (cache_seqlens.long().unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1)
                bufs = self._prefill_attn_bufs
            attn_metas[q_len] = {
                'cache_seqlens': cache_seqlens,
                'block_tables': self.paged_kv_cache.get_block_table_tensor(group_sids),
                'positions': positions,
                'bufs_q': bufs['q'][:n, :q_len],
                'bufs_k': bufs['k'][:n, :q_len],
                'bufs_v': bufs['v'][:n, :q_len],
            }

        gather_us = 0.0
        gemm_us = 0.0
        scatter_us = 0.0
        lora_us = 0.0
        attn_us = 0.0

        sync(); t0 = pc()
        epoch = self.batched_ops.embed_submit_all(slots)
        sync(); t1 = pc()
        self.batched_ops.barrier_sync(epoch)
        t2 = pc()
        sync(); t2b = pc()
        self.batched_ops.embed_get_all(slots)
        sync(); t3 = pc()
        gather_us += (t1 - t0) * 1e6
        gemm_us += (t2 - t1) * 1e6
        scatter_us += (t3 - t2b) * 1e6

        for layer_idx in range(self.num_layers):
            layer = self.layers[layer_idx]
            attn_mod = layer.self_attn
            mlp = layer.mlp
            fused_qkv = attn_mod._fused_qkv
            fused_gate_up = mlp._fused_gate_up

            # --- Input LayerNorm (ATTN) ---
            sync(); ta0 = pc()
            for sid, state in prefill_slots.items():
                state.residual = state.hidden
                state.hidden = layer.input_layernorm(state.hidden)
            WorkerBatchedOps.batch_layernorm(
                decode_slots, layer.input_layernorm, save_residual=True)
            sync(); ta1 = pc()
            attn_us += (ta1 - ta0) * 1e6

            # --- QKV: separate GATHER from LORA ---
            # GATHER: prepare_input + submit_command + signal_ready
            sync(); tg0 = pc()
            epoch = extq.get_barrier_done_epoch()
            qkv_deltas = {}
            slot_id, state = next(iter(slots.items()))
            ub.prepare_input_slot(slot_id, OP_QKV_FUSED, state.hidden)
            b, t, d = state.hidden.shape
            extq.submit_slot_command(
                OP_QKV_FUSED, layer_idx, slot_id, self.batched_ops.worker_id,
                b, t, d, 0, 0, 0, 0, 0)
            extq.slot_signal_ready(slot_id)
            sync(); tg1 = pc()
            gather_us += (tg1 - tg0) * 1e6

            # LORA: compute LoRA delta
            sync(); tl0 = pc()
            fqkv = [l.self_attn._fused_qkv for l in self.layers][layer_idx]
            if fqkv.fused_lora and fqkv.fused_lora.has_lora:
                qkv_deltas[slot_id] = fqkv.fused_lora(state.hidden)
            else:
                qkv_deltas[slot_id] = (None, None, None)
            sync(); tl1 = pc()
            lora_us += (tl1 - tl0) * 1e6

            # BARRIER (BASE_GEMM)
            sync(); tb0 = pc()
            self.batched_ops.barrier_sync(epoch)
            tb1 = pc()
            gemm_us += (tb1 - tb0) * 1e6

            # SCATTER: P2P copy only
            sync(); ts0 = pc()
            raw_qkv = self.batched_ops._get_outputs_all(slots, fused_qkv.dims)
            sync(); ts1 = pc()
            scatter_us += (ts1 - ts0) * 1e6

            # LORA: delta addition (q + q_delta)
            sync(); tla0 = pc()
            qkv_results = {}
            for slot_id in slots:
                q, k, v = raw_qkv[slot_id]
                q_delta, k_delta, v_delta = qkv_deltas.get(slot_id, (None, None, None))
                if q_delta is not None:
                    q = q + q_delta
                if k_delta is not None:
                    k = k + k_delta
                if v_delta is not None:
                    v = v + v_delta
                qkv_results[slot_id] = (q, k, v)
            sync(); tla1 = pc()
            lora_us += (tla1 - tla0) * 1e6

            # --- Attention (ATTN) ---
            sync(); ta2 = pc()
            attn_outputs = {}
            for q_len, group_sids in qlen_groups.items():
                group_out = WorkerBatchedOps.batch_flash_attention(
                    group_sids, qkv_results, self.paged_kv_cache,
                    self.cos_sin_cache, layer_idx,
                    self.num_heads, self.num_kv_heads, self.head_dim,
                    attn_meta=attn_metas[q_len])
                attn_outputs.update(group_out)
            sync(); ta3 = pc()
            attn_us += (ta3 - ta2) * 1e6

            # --- O_PROJ: GATHER ---
            sync(); tg0 = pc()
            epoch_o = extq.get_barrier_done_epoch()
            o_deltas = {}
            slot_id, state = next(iter(slots.items()))
            ao = attn_outputs[slot_id]
            ub.prepare_input_slot(slot_id, OP_O_PROJ, ao)
            b, t, d = ao.shape
            extq.submit_slot_command(
                OP_O_PROJ, layer_idx, slot_id, self.batched_ops.worker_id,
                b, t, d, 0, 0, 0, 0, 0)
            extq.slot_signal_ready(slot_id)
            sync(); tg1 = pc()
            gather_us += (tg1 - tg0) * 1e6

            # O_PROJ LORA
            sync(); tl0 = pc()
            o_mod = [l.self_attn.o_proj for l in self.layers][layer_idx]
            if isinstance(o_mod, SyncRemoteLoRALinear):
                o_deltas[slot_id] = o_mod.lora_B(o_mod.lora_A(o_mod.dropout(ao))) * o_mod.scaling
            else:
                o_deltas[slot_id] = None
            sync(); tl1 = pc()
            lora_us += (tl1 - tl0) * 1e6

            # O_PROJ BARRIER
            sync(); tb0 = pc()
            self.batched_ops.barrier_sync(epoch_o)
            tb1 = pc()
            gemm_us += (tb1 - tb0) * 1e6

            # O_PROJ SCATTER: P2P copy only
            sync(); ts0 = pc()
            raw_o = self.batched_ops._get_outputs_all(slots, [attn_mod.hidden_size])
            sync(); ts1 = pc()
            scatter_us += (ts1 - ts0) * 1e6

            # O_PROJ: delta addition + residual
            sync(); tla0 = pc()
            for slot_id, state in slots.items():
                result = raw_o[slot_id][0]
                delta = o_deltas.get(slot_id)
                if delta is not None:
                    result = result + delta
                state.hidden = state.residual + result
            sync(); tla1 = pc()
            # Delta add → LORA, residual add → ATTN (combined is small, attribute to LORA)
            lora_us += (tla1 - tla0) * 1e6

            # --- Post-Attention LayerNorm (ATTN) ---
            sync(); ta4 = pc()
            for sid, state in prefill_slots.items():
                state.mlp_residual = state.hidden
                state.hidden = layer.post_attention_layernorm(state.hidden)
            WorkerBatchedOps.batch_layernorm(
                decode_slots, layer.post_attention_layernorm, save_mlp_residual=True)
            sync(); ta5 = pc()
            attn_us += (ta5 - ta4) * 1e6

            # --- GATE_UP: GATHER ---
            sync(); tg0 = pc()
            epoch_gu = extq.get_barrier_done_epoch()
            gate_up_deltas = {}
            slot_id, state = next(iter(slots.items()))
            ub.prepare_input_slot(slot_id, OP_GATE_UP_FUSED, state.hidden)
            b, t, d = state.hidden.shape
            extq.submit_slot_command(
                OP_GATE_UP_FUSED, layer_idx, slot_id, self.batched_ops.worker_id,
                b, t, d, 0, 0, 0, 0, 0)
            extq.slot_signal_ready(slot_id)
            sync(); tg1 = pc()
            gather_us += (tg1 - tg0) * 1e6

            # GATE_UP LORA
            sync(); tl0 = pc()
            fgu = [l.mlp._fused_gate_up for l in self.layers][layer_idx]
            if fgu.fused_lora and fgu.fused_lora.has_lora:
                gate_up_deltas[slot_id] = fgu.fused_lora(state.hidden)
            else:
                gate_up_deltas[slot_id] = (None, None)
            sync(); tl1 = pc()
            lora_us += (tl1 - tl0) * 1e6

            # GATE_UP BARRIER
            sync(); tb0 = pc()
            self.batched_ops.barrier_sync(epoch_gu)
            tb1 = pc()
            gemm_us += (tb1 - tb0) * 1e6

            # GATE_UP SCATTER: P2P copy only
            sync(); ts0 = pc()
            raw_gu = self.batched_ops._get_outputs_all(slots, fused_gate_up.dims)
            sync(); ts1 = pc()
            scatter_us += (ts1 - ts0) * 1e6

            # GATE_UP: delta addition
            sync(); tla0 = pc()
            gate_up_results = {}
            for slot_id in slots:
                gate, up = raw_gu[slot_id]
                gate_delta, up_delta = gate_up_deltas.get(slot_id, (None, None))
                if gate_delta is not None:
                    gate = gate + gate_delta
                if up_delta is not None:
                    up = up + up_delta
                gate_up_results[slot_id] = (gate, up)
            sync(); tla1 = pc()
            lora_us += (tla1 - tla0) * 1e6

            # --- MLP Activation (ATTN) ---
            sync(); ta6 = pc()
            mlp_intermediates = {}
            for sid, state in prefill_slots.items():
                gate, up = gate_up_results[sid]
                mlp_intermediates[sid] = F.silu(gate) * up
            mlp_intermediates.update(
                WorkerBatchedOps.batch_mlp_activation(decode_slots, gate_up_results))
            sync(); ta7 = pc()
            attn_us += (ta7 - ta6) * 1e6

            # --- DOWN_PROJ: GATHER ---
            sync(); tg0 = pc()
            epoch_dp = extq.get_barrier_done_epoch()
            down_deltas = {}
            slot_id, state = next(iter(slots.items()))
            dp_input = mlp_intermediates[slot_id]
            ub.prepare_input_slot(slot_id, OP_DOWN_PROJ, dp_input)
            b, t, d = dp_input.shape
            extq.submit_slot_command(
                OP_DOWN_PROJ, layer_idx, slot_id, self.batched_ops.worker_id,
                b, t, d, 0, 0, 0, 0, 0)
            extq.slot_signal_ready(slot_id)
            sync(); tg1 = pc()
            gather_us += (tg1 - tg0) * 1e6

            # DOWN_PROJ LORA
            sync(); tl0 = pc()
            dp_mod = [l.mlp.down_proj for l in self.layers][layer_idx]
            if isinstance(dp_mod, SyncRemoteLoRALinear):
                down_deltas[slot_id] = dp_mod.lora_B(dp_mod.lora_A(dp_mod.dropout(dp_input))) * dp_mod.scaling
            else:
                down_deltas[slot_id] = None
            sync(); tl1 = pc()
            lora_us += (tl1 - tl0) * 1e6

            # DOWN_PROJ BARRIER
            sync(); tb0 = pc()
            self.batched_ops.barrier_sync(epoch_dp)
            tb1 = pc()
            gemm_us += (tb1 - tb0) * 1e6

            # DOWN_PROJ SCATTER: P2P copy only
            sync(); ts0 = pc()
            raw_dp = self.batched_ops._get_outputs_all(slots, [attn_mod.hidden_size])
            sync(); ts1 = pc()
            scatter_us += (ts1 - ts0) * 1e6

            # DOWN_PROJ: delta addition + residual
            sync(); tla0 = pc()
            for slot_id, state in slots.items():
                result = raw_dp[slot_id][0]
                delta = down_deltas.get(slot_id)
                if delta is not None:
                    result = result + delta
                state.hidden = state.mlp_residual + result
            sync(); tla1 = pc()
            lora_us += (tla1 - tla0) * 1e6

        all_sids = list(slots.keys())
        all_counts = [slots[sid].input_ids.size(1) for sid in all_sids]
        self.paged_kv_cache.advance_seq_lens_by(all_sids, all_counts)

        sync(); ta8 = pc()
        for sid, state in prefill_slots.items():
            state.hidden = self.final_norm(state.hidden)
            if state.prefill_chunks:
                state.hidden = state.hidden[:, -1:, :]
        WorkerBatchedOps.batch_final_norm(decode_slots, self.final_norm)
        sync(); ta9 = pc()
        attn_us += (ta9 - ta8) * 1e6

        sync(); t0 = pc()
        epoch = self.batched_ops.lm_head_submit_all(slots)
        sync(); t1 = pc()
        self.batched_ops.barrier_sync(epoch)
        t2 = pc()
        sync(); t2b = pc()
        self.batched_ops.lm_head_get_all(slots)
        sync(); t3 = pc()
        gather_us += (t1 - t0) * 1e6
        gemm_us += (t2 - t1) * 1e6
        scatter_us += (t3 - t2b) * 1e6

        # Log profile
        sync()
        wall_us = (pc() - wall_start) * 1e6
        self._profile_token_idx += 1
        if not has_prefill:  # Only log decode tokens
            ipc_us = gather_us + scatter_us
            compute_us = wall_us - ipc_us
            ipc_pct = ipc_us / wall_us * 100 if wall_us > 0 else 0
            print(f"[LAYER_PROFILE] worker={self.worker.worker_id} token={self._profile_token_idx} "
                  f"gather_us={gather_us:.0f} scatter_us={scatter_us:.0f} "
                  f"compute_us={compute_us:.0f} wall_us={wall_us:.0f} ipc_pct={ipc_pct:.1f}",
                  flush=True)
            self._profile_records.append({
                "batch": n_workers, "token": self._profile_token_idx,
                "gather_us": gather_us, "scatter_us": scatter_us,
                "compute_us": compute_us, "wall_us": wall_us,
            })


def main():
    """Entry point for worker with HTTP server."""
    t_main_start = time.perf_counter()

    import argparse
    t_argparse = time.perf_counter()

    parser = argparse.ArgumentParser(description="Multi-GPU Worker (HTTP Server)")
    parser.add_argument("--host", default="localhost",
                        help="Aggregator hostname")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Aggregator port (default: {DEFAULT_PORT})")
    parser.add_argument("--http-port", type=int, required=True,
                        help="HTTP server port for this worker")
    parser.add_argument("--device", default=WORKER_DEVICE,
                        help=f"Worker device (default: {WORKER_DEVICE})")
    parser.add_argument("--lora", default=LORA_ADAPTER_ID,
                        help="LoRA adapter ID")
    args = parser.parse_args()
    t_args_parsed = time.perf_counter()

    worker = FaaSWorker(
        host=args.host,
        port=args.port,
        device=args.device,
        lora_id=args.lora
    )
    t_worker_created = time.perf_counter()

    if not worker.initialize():
        print("[Worker] Initialization failed, exiting", flush=True)
        sys.exit(1)
    t_initialized = time.perf_counter()

    # Report comprehensive cold start timing
    print(f"[Worker-{worker.worker_id}] COLD_START_TIMING: "
          f"module_imports={_module_import_time:.2f}s "
          f"argparse={t_argparse-t_main_start:.2f}s "
          f"args_parse={t_args_parsed-t_argparse:.2f}s "
          f"worker_create={t_worker_created-t_args_parsed:.2f}s "
          f"initialize={t_initialized-t_worker_created:.2f}s "
          f"TOTAL={t_initialized-_t_module_start:.2f}s", flush=True)

    try:
        worker.start_http_server(args.http_port)
    except KeyboardInterrupt:
        print("\n[Worker] Interrupted", flush=True)
    finally:
        worker.shutdown()


def fork_main(http_port, host="localhost", port=DEFAULT_PORT, device=WORKER_DEVICE,
              lora=None, preloaded_tokenizer=None):
    """Entry point for workers forked from template process."""
    t_start = time.perf_counter()

    worker = FaaSWorker(host=host, port=port, device=device, lora_id=lora,
                        preloaded_tokenizer=preloaded_tokenizer)
    t_worker_created = time.perf_counter()

    if not worker.initialize():
        print("[Worker] Initialization failed, exiting", flush=True)
        os._exit(1)
    t_initialized = time.perf_counter()

    print(f"[Worker-{worker.worker_id}] COLD_START_TIMING: "
          f"module_imports=0.00s "
          f"worker_create={t_worker_created - t_start:.2f}s "
          f"initialize={t_initialized - t_worker_created:.2f}s "
          f"TOTAL={t_initialized - t_start:.2f}s (forked)", flush=True)

    try:
        worker.start_http_server(http_port)
    except KeyboardInterrupt:
        print("\n[Worker] Interrupted", flush=True)
    finally:
        worker.shutdown()


if __name__ == "__main__":
    main()
