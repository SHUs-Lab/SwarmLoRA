"""Slot-based GPU client for synchronized batched inference."""

import socket
import threading
import torch
from typing import Optional

from config import DTYPE
import ext_unified_barrier as ub
from worker.utils import log, send_msg, recv_msg

_SOCKET_RECV_BUFFER = 65536
_SOCKET_OP_TIMEOUT = 30.0
_MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100MB


class SlotClient:
    """Slot-based GPU client for synchronized batched inference."""

    def __init__(self, worker_id: int, aggregator_device_idx: int, worker_device_idx: int,
                 hidden_dim: int, intermediate_dim: int, qkv_dim: int,
                 sock: socket.socket, input_buffer_size: int, output_buffer_size: int):
        self.worker_id = worker_id
        self.aggregator_device_idx = aggregator_device_idx
        self.worker_device_idx = worker_device_idx
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.qkv_dim = qkv_dim
        self.sock = sock  # TCP socket to aggregator for slot requests
        self.input_buffer_size = input_buffer_size
        self.output_buffer_size = output_buffer_size
        self.sock_lock = threading.Lock()  # Protect socket for concurrent requests
        self.dtype_code = 15 if DTYPE == torch.bfloat16 else 16
        self.initialized = False

    def _send(self, obj: dict) -> None:
        """Send a JSON message to aggregator."""
        try:
            send_msg(self.sock, obj)
        except (socket.error, BrokenPipeError, ConnectionResetError) as e:
            raise ConnectionError(f"Lost connection to aggregator: {e}") from e

    def _recv(self) -> Optional[dict]:
        """Receive a JSON message from aggregator with timeout."""
        self.sock.settimeout(_SOCKET_OP_TIMEOUT)
        try:
            return recv_msg(self.sock)
        except socket.timeout:
            log(self.worker_id, "Socket recv timeout")
            return None
        except ConnectionError:
            log(self.worker_id, "Connection closed by aggregator")
            return None

    def initialize(self):
        """Initialize the slot manager (call once per worker)."""
        if self.initialized:
            return

        # Initialize the worker slot manager (once per worker)
        ub.init_worker_slot_manager(
            worker_id=self.worker_id,
            worker_device=self.worker_device_idx,
            aggregator_device=self.aggregator_device_idx,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            qkv_dim=self.qkv_dim
        )
        self.initialized = True

    def cleanup(self):
        """Cleanup slot resources."""
        if self.initialized:
            ub.cleanup_worker_slots()
            self.initialized = False

    def release_and_close_by_id(self, slot_id: int):
        """Close IPC handles and release a specific slot by ID."""
        if slot_id < 0:
            return

        ub.close_slot(slot_id)

        with self.sock_lock:
            self._send({
                'type': 'slot_request',
                'slot_request': 'release',
                'worker_id': self.worker_id,
                'slot_id': slot_id,
            })
            response = self._recv()
            if response is None:
                log(self.worker_id, "Lost connection during slot release")
                raise ConnectionError("Aggregator connection lost")
            if response.get('type') == 'slot_error':
                log(self.worker_id, f"Warning: slot release error: {response.get('error')}")


class IntraAdapterSlotManager:
    """Manages slot claim/release and active slot state for a worker."""

    def __init__(self, base_client: SlotClient):
        self.base_client = base_client
        self.active_slots: dict = {}  # slot_id -> SlotState
        self._lock = threading.Lock()

    def claim_slot(self) -> int:
        """Claim a slot from aggregator."""
        with self.base_client.sock_lock:
            self.base_client._send({
                'type': 'slot_request',
                'slot_request': 'claim',
                'worker_id': self.base_client.worker_id,
            })
            response = self.base_client._recv()
            if response is None:
                raise ConnectionError("Aggregator connection lost during slot claim")
            if response.get('type') == 'slot_error':
                raise RuntimeError(f"Failed to claim slot: {response.get('error')}")

            slot_id = response['slot_id']
            input_handle = response['input_handle']
            output_handle = response['output_handle']

        # Open IPC handles
        ub.open_slot(
            slot_id=slot_id,
            input_handle=input_handle,
            output_handle=output_handle,
            input_buffer_size=self.base_client.input_buffer_size,
            output_buffer_size=self.base_client.output_buffer_size
        )

        return slot_id

    def release_slot(self, slot_id: int):
        """Release a slot back to aggregator."""
        if slot_id >= 0:
            self.base_client.release_and_close_by_id(slot_id)

    def add_active_slot(self, slot_id: int, state):
        """Add a slot to active slots."""
        with self._lock:
            self.active_slots[slot_id] = state

    def remove_active_slot(self, slot_id: int):
        """Remove and return a slot from active slots."""
        with self._lock:
            return self.active_slots.pop(slot_id, None)

    def get_active_slots(self) -> dict:
        """Get a snapshot of active slots."""
        with self._lock:
            return dict(self.active_slots)
