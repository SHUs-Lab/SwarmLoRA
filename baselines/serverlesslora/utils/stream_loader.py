#!/usr/bin/env python3
"""
Stream Loader Module for ServerlessLoRA.

Implements Paper Section 5:
- CUDA Streams for concurrent tensor loading
- CUDA Asynchronous Memory Transfer to overlap loading and GPU transferring

This module provides a high-level Python interface for:
1. Concurrent model weight loading using multiple CUDA streams
2. Asynchronous memory transfers between CPU and GPU
3. Batch tensor operations for efficient pre-loading

Usage:
    from stream_loader import StreamLoader, async_load_state_dict

    # Initialize stream loader
    loader = StreamLoader(device_id=0, num_streams=4)

    # Load model weights concurrently
    gpu_state_dict = loader.load_state_dict(cpu_state_dict)

    # Or use the convenience function
    gpu_weights = async_load_state_dict(cpu_weights, device_id=0)
"""

import time
import logging
from typing import Dict, List, Any
from collections import OrderedDict
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import the CUDA extension
_ext_available = False
try:
    import ext_stream_loader as ext
    _ext_available = True
    logger.debug("ext_stream_loader loaded successfully")
except ImportError:
    logger.warning("ext_stream_loader not found. Using fallback implementation. "
                   "Run: python setup.py build_ext --inplace")


class StreamLoader:
    """
    High-level interface for concurrent tensor loading using CUDA Streams.

    Paper Section 5: "To accelerate the pre-loading of backbone LLM, we utilize
    CUDA Streams to load tensors concurrently and CUDA Asynchronous Memory
    Transfer to overlap loading and GPU transferring."

    Features:
    - Multi-stream parallel loading
    - Automatic stream pool management
    - Support for state_dict loading
    - Pinned memory optimization
    """

    def __init__(
        self,
        device_id: int = 0,
        num_streams: int = 4,
        use_pinned_memory: bool = True
    ):
        """
        Initialize the StreamLoader.

        Args:
            device_id: CUDA device to load tensors to
            num_streams: Number of CUDA streams for parallel loading
            use_pinned_memory: Whether to use pinned memory for faster transfers
        """
        self.device_id = device_id
        self.num_streams = num_streams
        self.use_pinned_memory = use_pinned_memory
        self._initialized = False

        # Initialize stream pool if extension available
        if _ext_available:
            try:
                ext.init_stream_pool(device_id, num_streams)
                self._initialized = True
                logger.info(f"StreamLoader initialized with {num_streams} streams "
                           f"on device {device_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize stream pool: {e}")

    def load_tensors(
        self,
        tensors: List[torch.Tensor],
        synchronize: bool = True
    ) -> List[torch.Tensor]:
        """
        Load multiple tensors to GPU concurrently.

        Args:
            tensors: List of CPU tensors to load
            synchronize: Whether to wait for all transfers to complete

        Returns:
            List of GPU tensors
        """
        if not tensors:
            return []

        cpu_tensors = []
        gpu_indices = []
        result_tensors = [None] * len(tensors)

        for i, t in enumerate(tensors):
            if t.is_cuda and t.device.index == self.device_id:
                result_tensors[i] = t
            else:
                cpu_tensors.append(t.cpu() if t.is_cuda else t)
                gpu_indices.append(i)

        if not cpu_tensors:
            return tensors

        if _ext_available and self._initialized:
            try:
                gpu_tensors = ext.batch_load_tensors_async(
                    cpu_tensors, self.device_id, self.num_streams
                )
                for idx, gpu_t in zip(gpu_indices, gpu_tensors):
                    result_tensors[idx] = gpu_t
                return result_tensors
            except Exception as e:
                logger.warning(f"Extension batch load failed, using fallback: {e}")

        return self._fallback_load_tensors(tensors, synchronize)

    def _fallback_load_tensors(
        self,
        tensors: List[torch.Tensor],
        synchronize: bool = True
    ) -> List[torch.Tensor]:
        """Fallback implementation using PyTorch's non_blocking transfers."""
        device = torch.device(f'cuda:{self.device_id}')
        result = []

        for t in tensors:
            if t.is_cuda and t.device.index == self.device_id:
                result.append(t)
            else:
                if self.use_pinned_memory and not t.is_pinned():
                    t = t.pin_memory()
                result.append(t.to(device, non_blocking=True))

        if synchronize:
            torch.cuda.synchronize(self.device_id)

        return result

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        synchronize: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Load a state_dict to GPU concurrently.

        This is the main method for loading model weights efficiently.

        Args:
            state_dict: Dictionary of parameter name -> CPU tensor
            synchronize: Whether to wait for all transfers to complete

        Returns:
            Dictionary of parameter name -> GPU tensor
        """
        if not state_dict:
            return {}

        names = list(state_dict.keys())
        tensors = list(state_dict.values())

        t_start = time.perf_counter()

        gpu_tensors = self.load_tensors(tensors, synchronize)

        t_end = time.perf_counter()
        total_mb = sum(t.numel() * t.element_size() for t in tensors) / (1024 * 1024)
        logger.debug(f"Loaded {len(tensors)} tensors ({total_mb:.1f} MB) "
                    f"in {(t_end - t_start) * 1000:.1f} ms")

        return OrderedDict(zip(names, gpu_tensors))

    def offload_tensors(
        self,
        tensors: List[torch.Tensor],
        synchronize: bool = True
    ) -> List[torch.Tensor]:
        """
        Offload multiple tensors from GPU to CPU concurrently.

        Args:
            tensors: List of GPU tensors to offload
            synchronize: Whether to wait for all transfers to complete

        Returns:
            List of CPU tensors
        """
        if not tensors:
            return []

        # Filter out already-CPU tensors
        gpu_tensors = [t for t in tensors if t.is_cuda]
        if not gpu_tensors:
            return tensors

        if _ext_available and self._initialized:
            try:
                cpu_tensors = ext.batch_offload_tensors_async(
                    gpu_tensors, self.num_streams
                )
                # Merge results
                result = []
                gpu_idx = 0
                for t in tensors:
                    if t.is_cuda:
                        result.append(cpu_tensors[gpu_idx])
                        gpu_idx += 1
                    else:
                        result.append(t)
                return result
            except Exception as e:
                logger.warning(f"Extension batch offload failed, using fallback: {e}")

        return [t.cpu() for t in tensors]

    def offload_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        synchronize: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Offload a state_dict from GPU to CPU concurrently.

        Args:
            state_dict: Dictionary of parameter name -> GPU tensor
            synchronize: Whether to wait for all transfers to complete

        Returns:
            Dictionary of parameter name -> CPU tensor
        """
        if not state_dict:
            return {}

        names = list(state_dict.keys())
        tensors = list(state_dict.values())

        cpu_tensors = self.offload_tensors(tensors, synchronize)

        return OrderedDict(zip(names, cpu_tensors))

    def synchronize(self):
        """Synchronize all CUDA streams."""
        if _ext_available and self._initialized:
            ext.synchronize_streams(self.device_id)
        else:
            torch.cuda.synchronize(self.device_id)

    def cleanup(self):
        """Cleanup resources."""
        if _ext_available:
            try:
                ext.cleanup_stream_pools()
            except Exception:
                pass

    def get_info(self) -> Dict[str, Any]:
        """Get stream loader information."""
        info = {
            "device_id": self.device_id,
            "num_streams": self.num_streams,
            "use_pinned_memory": self.use_pinned_memory,
            "extension_available": _ext_available,
            "initialized": self._initialized
        }

        if _ext_available:
            try:
                pool_info = ext.get_stream_pool_info(self.device_id)
                info["stream_pool"] = dict(pool_info)
            except Exception:
                pass

        return info


# =============================================================================
# Convenience Functions
# =============================================================================

def async_load_state_dict(
    state_dict: Dict[str, torch.Tensor],
    device_id: int = 0,
    num_streams: int = 4
) -> Dict[str, torch.Tensor]:
    """
    Convenience function to load a state_dict using CUDA streams.

    Args:
        state_dict: Dictionary of CPU tensors
        device_id: Target CUDA device
        num_streams: Number of streams for parallel loading

    Returns:
        Dictionary of GPU tensors
    """
    loader = StreamLoader(device_id, num_streams)
    return loader.load_state_dict(state_dict)


def async_offload_state_dict(
    state_dict: Dict[str, torch.Tensor],
    num_streams: int = 4
) -> Dict[str, torch.Tensor]:
    """
    Convenience function to offload a state_dict using CUDA streams.

    Args:
        state_dict: Dictionary of GPU tensors
        num_streams: Number of streams for parallel offloading

    Returns:
        Dictionary of CPU tensors
    """
    if not state_dict:
        return {}

    # Get device from first tensor
    first_tensor = next(iter(state_dict.values()))
    if not first_tensor.is_cuda:
        return state_dict

    device_id = first_tensor.device.index
    loader = StreamLoader(device_id, num_streams)
    return loader.offload_state_dict(state_dict)
