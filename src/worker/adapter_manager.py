"""Adapter lifecycle management with thread-safe hot-swap support."""

import time
import threading
import traceback
import torch
import torch.nn as nn

from worker.utils import log
from worker.model_builder import load_lora_fast, FusedLoRA, SyncRemoteLoRALinear


class AdapterManager:
    """Manages LoRA adapter lifecycle with thread-safe hot-swap support."""

    def __init__(self, worker_id: int, device: str, dtype: torch.dtype, info: dict):
        self.worker_id = worker_id
        self.device = device
        self.dtype = dtype
        self.info = info

        # Current adapter state
        self.current_adapter_id = None
        self.current_rank = 0
        self.current_config = None

        # Thread safety
        self._swap_lock = threading.Lock()
        self._is_swapping = threading.Event()

    def get_current_adapter(self):
        """Thread-safe getter for current adapter ID."""
        with self._swap_lock:
            return self.current_adapter_id

    def is_swapping(self) -> bool:
        """Check if a swap is currently in progress."""
        return self._is_swapping.is_set()

    def set_adapter(self, adapter_id: str, rank: int, config: dict):
        """Set initial adapter info (called during build_hollow_model)."""
        with self._swap_lock:
            self.current_adapter_id = adapter_id
            self.current_rank = rank
            self.current_config = config

    def swap_adapter(self, new_adapter_id: str, model: nn.Module) -> dict:
        """Atomically swap to a new LoRA adapter."""
        t_start = time.perf_counter()

        with self._swap_lock:
            self._is_swapping.set()

            try:
                old_adapter = self.current_adapter_id
                old_rank = self.current_rank

                # Load new adapter
                t_load = time.perf_counter()
                new_lora_layers, new_config = load_lora_fast(new_adapter_id, self.device, self.dtype)
                new_rank = new_config['r']
                t_load_done = time.perf_counter()

                # Check if rank changed (requires module recreation)
                rank_changed = (old_rank != new_rank and old_rank != 0)

                # Update modules
                t_update = time.perf_counter()
                if rank_changed:
                    self._recreate_lora_modules(model, new_lora_layers, new_config)
                else:
                    self._update_lora_weights(model, new_lora_layers, new_config)
                torch.cuda.synchronize()
                t_update_done = time.perf_counter()

                # Update state
                self.current_adapter_id = new_adapter_id
                self.current_rank = new_rank
                self.current_config = new_config

                t_end = time.perf_counter()

                log(self.worker_id, f"Adapter swap: {old_adapter} -> {new_adapter_id} "
                    f"(rank {old_rank} -> {new_rank}, {'recreated' if rank_changed else 'in-place'})")

                return {
                    'success': True,
                    'old_adapter': old_adapter,
                    'new_adapter': new_adapter_id,
                    'old_rank': old_rank,
                    'new_rank': new_rank,
                    'rank_changed': rank_changed,
                    'load_time_ms': round((t_load_done - t_load) * 1000, 2),
                    'update_time_ms': round((t_update_done - t_update) * 1000, 2),
                    'total_time_ms': round((t_end - t_start) * 1000, 2),
                }

            except Exception as e:
                log(self.worker_id, f"Adapter swap failed: {e}")
                traceback.print_exc()
                return {
                    'success': False,
                    'error': str(e),
                }
            finally:
                self._is_swapping.clear()

    def _update_lora_weights(self, model, new_lora_layers: dict, new_config: dict):
        """Fast path: same rank, update tensor data in-place."""
        new_scaling = new_config.get('lora_alpha', new_config['r']) / new_config['r']
        decoder = model.model

        for layer_idx, layer in enumerate(decoder.layers):
            attn = layer.self_attn
            mlp = layer.mlp

            # Update fused QKV LoRA
            if hasattr(attn, '_fused_qkv') and attn._fused_qkv.fused_lora is not None:
                self._update_fused_lora(
                    attn._fused_qkv.fused_lora,
                    ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
                    layer_idx, new_lora_layers, new_scaling
                )

            # Update fused GateUp LoRA
            if hasattr(mlp, '_fused_gate_up') and mlp._fused_gate_up.fused_lora is not None:
                self._update_fused_lora(
                    mlp._fused_gate_up.fused_lora,
                    ['mlp.gate_proj', 'mlp.up_proj'],
                    layer_idx, new_lora_layers, new_scaling
                )

            # Update o_proj LoRA
            if isinstance(attn.o_proj, SyncRemoteLoRALinear):
                self._update_single_lora(attn.o_proj, 'self_attn.o_proj',
                                         layer_idx, new_lora_layers, new_scaling)

            # Update down_proj LoRA
            if isinstance(mlp.down_proj, SyncRemoteLoRALinear):
                self._update_single_lora(mlp.down_proj, 'mlp.down_proj',
                                         layer_idx, new_lora_layers, new_scaling)

    def _update_fused_lora(self, fused_lora, module_names: list,
                           layer_idx: int, new_lora_layers: dict, new_scaling: float):
        """Update weights in a FusedLoRA module."""
        if not fused_lora.has_lora:
            return

        # Get new modules for this layer
        new_modules = []
        for name in module_names:
            if layer_idx in new_lora_layers and name in new_lora_layers[layer_idx]:
                new_modules.append(new_lora_layers[layer_idx][name])
            else:
                new_modules.append(None)

        # Find a valid module to get rank info
        valid_modules = [m for m in new_modules if m is not None]
        if not valid_modules:
            return

        # Rebuild fused tensors
        A_weights, B_weights = [], []
        rank = valid_modules[0].lora_A['default'].weight.size(0)

        for i, module in enumerate(new_modules):
            if module is not None:
                A = module.lora_A['default'].weight
                B = module.lora_B['default'].weight
                A_weights.append(A.t() * new_scaling)
                B_weights.append(B.t())
            else:
                in_features = fused_lora.fused_A.size(0)
                A_weights.append(torch.zeros(in_features, rank,
                                            device=self.device, dtype=self.dtype))
                B_weights.append(torch.zeros(rank, fused_lora.out_dims[i],
                                            device=self.device, dtype=self.dtype))

        # Update fused_A
        fused_lora.fused_A.data.copy_(torch.cat(A_weights, dim=1))

        # Rebuild and update fused_B
        total_out = sum(fused_lora.out_dims)
        new_fused_B = torch.zeros(rank * len(module_names), total_out,
                                  device=self.device, dtype=self.dtype)
        col_offset = 0
        for i, B in enumerate(B_weights):
            row_start, row_end = i * rank, (i + 1) * rank
            new_fused_B[row_start:row_end, col_offset:col_offset + fused_lora.out_dims[i]] = B
            col_offset += fused_lora.out_dims[i]
        fused_lora.fused_B.data.copy_(new_fused_B)

    def _update_single_lora(self, sync_lora, module_name: str,
                            layer_idx: int, new_lora_layers: dict, new_scaling: float):
        """Update weights in a SyncRemoteLoRALinear module."""
        if layer_idx not in new_lora_layers or module_name not in new_lora_layers[layer_idx]:
            return

        new_module = new_lora_layers[layer_idx][module_name]
        sync_lora.lora_A.weight.data.copy_(new_module.lora_A['default'].weight)
        sync_lora.lora_B.weight.data.copy_(new_module.lora_B['default'].weight)
        sync_lora.scaling = new_scaling

    def _zero_lora_modules(self, model):
        """Zero all current LoRA weight tensors before rank-change deallocation."""
        decoder = model.model
        for layer in decoder.layers:
            attn = layer.self_attn
            mlp  = layer.mlp
            if hasattr(attn, '_fused_qkv') and attn._fused_qkv.fused_lora is not None:
                fl = attn._fused_qkv.fused_lora
                if hasattr(fl, 'fused_A'): fl.fused_A.data.zero_()
                if hasattr(fl, 'fused_B'): fl.fused_B.data.zero_()
            if hasattr(mlp, '_fused_gate_up') and mlp._fused_gate_up.fused_lora is not None:
                fl = mlp._fused_gate_up.fused_lora
                if hasattr(fl, 'fused_A'): fl.fused_A.data.zero_()
                if hasattr(fl, 'fused_B'): fl.fused_B.data.zero_()
            if isinstance(attn.o_proj, SyncRemoteLoRALinear):
                attn.o_proj.lora_A.weight.data.zero_()
                attn.o_proj.lora_B.weight.data.zero_()
            if isinstance(mlp.down_proj, SyncRemoteLoRALinear):
                mlp.down_proj.lora_A.weight.data.zero_()
                mlp.down_proj.lora_B.weight.data.zero_()

    def _recreate_lora_modules(self, model, new_lora_layers: dict, new_config: dict):
        """Slow path: different rank, recreate nn.Linear modules."""
        self._zero_lora_modules(model)
        new_scaling = new_config.get('lora_alpha', new_config['r']) / new_config['r']
        new_rank = new_config['r']
        decoder = model.model

        def get_new_module(layer_idx: int, module_name: str):
            if layer_idx in new_lora_layers and module_name in new_lora_layers[layer_idx]:
                return new_lora_layers[layer_idx][module_name]
            return None

        q_dim = self.info['hidden_size']
        k_dim = self.info['hidden_size'] // self.info['num_heads'] * self.info['num_kv_heads']
        v_dim = k_dim
        qkv_dims = [q_dim, k_dim, v_dim]
        gate_up_dims = [self.info['intermediate_size'], self.info['intermediate_size']]

        for layer_idx, layer in enumerate(decoder.layers):
            attn = layer.self_attn
            mlp = layer.mlp

            # Recreate fused QKV LoRA
            if hasattr(attn, '_fused_qkv'):
                qkv_lora = [
                    get_new_module(layer_idx, 'self_attn.q_proj'),
                    get_new_module(layer_idx, 'self_attn.k_proj'),
                    get_new_module(layer_idx, 'self_attn.v_proj')
                ]
                new_fused = FusedLoRA(qkv_lora, qkv_dims, self.device, self.dtype) if any(qkv_lora) else None
                attn._fused_qkv.fused_lora = new_fused

            # Recreate fused GateUp LoRA
            if hasattr(mlp, '_fused_gate_up'):
                gate_up_lora = [
                    get_new_module(layer_idx, 'mlp.gate_proj'),
                    get_new_module(layer_idx, 'mlp.up_proj')
                ]
                new_fused = FusedLoRA(gate_up_lora, gate_up_dims, self.device, self.dtype) if any(gate_up_lora) else None
                mlp._fused_gate_up.fused_lora = new_fused

            # Recreate o_proj LoRA
            if isinstance(attn.o_proj, SyncRemoteLoRALinear):
                new_module = get_new_module(layer_idx, 'self_attn.o_proj')
                if new_module:
                    attn.o_proj.lora_A = new_module.lora_A['default']
                    attn.o_proj.lora_B = new_module.lora_B['default']
                    attn.o_proj.scaling = new_scaling

            # Recreate down_proj LoRA
            if isinstance(mlp.down_proj, SyncRemoteLoRALinear):
                new_module = get_new_module(layer_idx, 'mlp.down_proj')
                if new_module:
                    mlp.down_proj.lora_A = new_module.lora_A['default']
                    mlp.down_proj.lora_B = new_module.lora_B['default']
                    mlp.down_proj.scaling = new_scaling
