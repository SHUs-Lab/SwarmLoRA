"""Paged Attention: KV cache block manager for flash_attn_with_kvcache."""

import torch
from typing import Dict, List


class PagedKVCache:
    """Block manager for paged KV cache (flash_attn format).

    Pre-allocates KV cache blocks per layer. Each block holds `block_size` tokens.
    Both key and value caches use the same layout:
        (num_blocks, block_size, num_kv_heads, head_dim)
    """

    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int,
                 block_size: int, max_slots: int, max_seq_len: int,
                 device: str, dtype: torch.dtype):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.device = device
        self.dtype = dtype

        # Calculate total blocks needed
        max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        self.max_blocks_per_seq = max_blocks_per_seq
        self.num_blocks = max_slots * max_blocks_per_seq

        # Pre-allocate KV cache for all layers
        # flash_attn format: (num_blocks, block_size, num_kv_heads, head_dim)
        self.key_caches = []
        self.value_caches = []
        for _ in range(num_layers):
            self.key_caches.append(torch.zeros(
                self.num_blocks, block_size, num_kv_heads, head_dim,
                device=device, dtype=dtype
            ))
            self.value_caches.append(torch.zeros(
                self.num_blocks, block_size, num_kv_heads, head_dim,
                device=device, dtype=dtype
            ))

        # Block free-list (all blocks start free)
        self.free_blocks = list(range(self.num_blocks - 1, -1, -1))  # stack

        # Per-slot tracking
        self.slot_block_tables: Dict[int, List[int]] = {}  # slot_id -> [block_ids]
        self.slot_seq_lens: Dict[int, int] = {}  # slot_id -> current seq length

        # Pre-allocated metadata buffers (avoid per-call tensor creation)
        self._block_tables_buf = torch.zeros(
            max_slots, max_blocks_per_seq, dtype=torch.int32, device=device
        )
        self._seq_lens_buf = torch.zeros(max_slots, dtype=torch.int32, device=device)

    def allocate_blocks(self, slot_id: int, num_blocks: int):
        """Allocate blocks for a slot. Raises if not enough free blocks."""
        if len(self.free_blocks) < num_blocks:
            raise RuntimeError(
                f"PagedKVCache: need {num_blocks} blocks but only {len(self.free_blocks)} free"
            )
        blocks = [self.free_blocks.pop() for _ in range(num_blocks)]
        if slot_id in self.slot_block_tables:
            self.slot_block_tables[slot_id].extend(blocks)
        else:
            self.slot_block_tables[slot_id] = blocks
            self.slot_seq_lens[slot_id] = 0

    def ensure_blocks_for_length(self, slot_id: int, needed_len: int):
        """Ensure slot has enough blocks for needed_len tokens."""
        needed_blocks = (needed_len + self.block_size - 1) // self.block_size
        current_blocks = len(self.slot_block_tables.get(slot_id, []))
        if needed_blocks > current_blocks:
            self.allocate_blocks(slot_id, needed_blocks - current_blocks)

    def free_slot(self, slot_id: int):
        """Return all blocks for a slot to the free list, zeroing KV data first."""
        blocks = self.slot_block_tables.pop(slot_id, [])
        if blocks:
            # Zero freed blocks to prevent cross-request KV leakage.
            # index_fill_ zeros all selected blocks in one kernel per layer
            # (64 launches) vs per-block loops (1024+ launches).
            block_idx = torch.tensor(blocks, dtype=torch.long, device=self.device)
            for layer_idx in range(self.num_layers):
                self.key_caches[layer_idx].index_fill_(0, block_idx, 0.0)
                self.value_caches[layer_idx].index_fill_(0, block_idx, 0.0)
        self.free_blocks.extend(blocks)
        self.slot_seq_lens.pop(slot_id, None)

    def get_block_table_tensor(self, slot_ids: List[int]) -> torch.Tensor:
        """Build block_tables tensor for flash_attn_with_kvcache.

        Returns: (num_seqs, max_blocks_per_seq) int32 tensor.
        Returns a clone to avoid aliasing when multiple q_len groups
        share the same pre-allocated buffer.
        """
        n = len(slot_ids)
        bt = self._block_tables_buf[:n]
        bt.zero_()
        for i, sid in enumerate(slot_ids):
            blocks = self.slot_block_tables[sid]
            for j, blk in enumerate(blocks):
                bt[i, j] = blk
        return bt.clone()

    def get_seq_lens_tensor(self, slot_ids: List[int]) -> torch.Tensor:
        """Build cache_seqlens tensor for flash_attn_with_kvcache.

        Returns: (num_seqs,) int32 tensor.
        Returns a clone to avoid aliasing when multiple q_len groups
        share the same pre-allocated buffer.
        """
        n = len(slot_ids)
        sl = self._seq_lens_buf[:n]
        for i, sid in enumerate(slot_ids):
            sl[i] = self.slot_seq_lens[sid]
        return sl.clone()

    def advance_seq_lens(self, slot_ids: List[int], count: int = 1):
        """Advance sequence lengths for slots by a uniform count."""
        for sid in slot_ids:
            self.slot_seq_lens[sid] += count

    def advance_seq_lens_by(self, slot_ids: List[int], counts: List[int]):
        """Advance sequence lengths by different amounts per slot."""
        for sid, count in zip(slot_ids, counts):
            self.slot_seq_lens[sid] += count


def build_vllm_cos_sin_cache(rotary_emb, max_seq_len: int,
                             device: str, dtype: torch.dtype):
    """Pre-compute cos_sin_cache for vllm_ops.rotary_embedding.

    Returns: (max_seq_len, head_dim) tensor with [cos, sin] concatenated.
    """
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    inv_freq = rotary_emb.inv_freq.to(device=device)
    freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim // 2)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return torch.cat([cos, sin], dim=-1)  # (max_seq_len, head_dim)
