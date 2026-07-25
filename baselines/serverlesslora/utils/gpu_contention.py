#!/usr/bin/env python3
"""
GPU-wide contention tracker for ServerlessLoRA.

Fixes Paper Deviation #1: Contention factor M (Eq. 4-5) must be
GPU-wide, not process-local. Multiple worker processes share a GPU
via CUDA IPC, so the contention count must be visible across all of them.

Uses POSIX shared memory for cross-process counter sharing with file
locking for atomic read-modify-write operations.
"""

import os
import struct
import logging
from multiprocessing import shared_memory
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_SHM_PREFIX = "slora_gpu"
_LOCK_DIR = "/tmp"


class GPUContentionTracker:
    """
    Cross-process counter tracking active batches on a specific GPU.

    Paper Eq. 4: T_eff_i(b) = M * T_i(b) where M is the number of
    batches running concurrently on the same GPU across ALL functions.

    Each GPU gets a named shared memory segment holding a single int64.
    Workers on the same GPU read/write the same counter with file locking.
    """

    def __init__(self, gpu_index: int):
        self.gpu_index = gpu_index
        self._shm_name = f"{_SHM_PREFIX}_{gpu_index}_batches"
        self._lock_path = os.path.join(_LOCK_DIR, f"{self._shm_name}.lock")
        self._shm = None
        self._lock_fd = None
        self._attached = False

        self._attach()

    def _attach(self):
        """Create or attach to shared memory for this GPU."""
        import fcntl
        self._fcntl = fcntl

        # Create/open lock file (append mode avoids truncation on re-attach)
        self._lock_fd = open(self._lock_path, "a")

        try:
            # Try to create (first process on this GPU)
            self._shm = shared_memory.SharedMemory(
                name=self._shm_name, create=True, size=8
            )
            struct.pack_into("q", self._shm.buf, 0, 0)
            logger.info(
                f"Created GPU contention tracker for gpu:{self.gpu_index}"
            )
        except FileExistsError:
            # Attach to existing (subsequent processes)
            self._shm = shared_memory.SharedMemory(
                name=self._shm_name, create=False
            )
            logger.info(
                f"Attached to GPU contention tracker for gpu:{self.gpu_index}"
            )

        # Prevent Python's resource_tracker from auto-unlinking the shared
        # memory when this process exits — the segment is intentionally
        # long-lived and shared across independent worker processes.
        try:
            from multiprocessing.resource_tracker import unregister
            unregister(self._shm._name, "shared_memory")
        except Exception:
            pass

        self._attached = True

    def _lock(self):
        self._fcntl.flock(self._lock_fd, self._fcntl.LOCK_EX)

    def _unlock(self):
        self._fcntl.flock(self._lock_fd, self._fcntl.LOCK_UN)

    def get_count(self) -> int:
        """Get current GPU-wide active batch count."""
        if not self._attached:
            return 1
        self._lock()
        try:
            return struct.unpack_from("q", self._shm.buf, 0)[0]
        finally:
            self._unlock()

    def increment(self) -> int:
        """Atomically increment active batch count. Returns new value."""
        if not self._attached:
            return 1
        self._lock()
        try:
            count = struct.unpack_from("q", self._shm.buf, 0)[0] + 1
            struct.pack_into("q", self._shm.buf, 0, count)
            return count
        finally:
            self._unlock()

    def decrement(self) -> int:
        """Atomically decrement active batch count. Returns new value."""
        if not self._attached:
            return 0
        self._lock()
        try:
            count = max(0, struct.unpack_from("q", self._shm.buf, 0)[0] - 1)
            struct.pack_into("q", self._shm.buf, 0, count)
            return count
        finally:
            self._unlock()

    @contextmanager
    def batch_active(self):
        """Context manager that tracks a batch as active on this GPU."""
        self.increment()
        try:
            yield
        finally:
            self.decrement()

    def close(self):
        """Detach from shared memory (does NOT destroy it)."""
        if self._shm:
            self._shm.close()
            self._shm = None
        if self._lock_fd:
            self._lock_fd.close()
            self._lock_fd = None
        self._attached = False

    @staticmethod
    def cleanup(gpu_index: int):
        """Destroy shared memory for a GPU. Call once during cluster shutdown."""
        name = f"{_SHM_PREFIX}_{gpu_index}_batches"
        try:
            shm = shared_memory.SharedMemory(name=name, create=False)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass
        lock_path = os.path.join(_LOCK_DIR, f"{name}.lock")
        if os.path.exists(lock_path):
            os.unlink(lock_path)
