#!/usr/bin/env python3
"""
Batched Worker - Worker with Adaptive Batching support.

Implements Section 4.2 of the paper:
- Fill-or-expire batching mechanism
- SLO-aware batch sizing
- Contention-aware scheduling

This worker can process multiple requests in a single forward pass.
"""

import time
_t_module_start = time.perf_counter()

import sys
import os
import socket
import pickle
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
# Pre-import concrete model class to work around broken Auto mapping in
# some transformers installs (e.g. LlamaForCausalLM not resolvable via
# AutoModelForCausalLM.from_config).
try:
    from transformers.models.llama.modeling_llama import LlamaForCausalLM as _LlamaForCausalLM
except ImportError:
    _LlamaForCausalLM = None
import threading
import traceback
from typing import List, Dict, Optional

from transformers import LogitsProcessor

class FirstTokenTimer(LogitsProcessor):
    """Records timestamp when first called (= after prefill)."""
    def __init__(self):
        self.t_first_token = None      # wall clock for cross-process TTFT
        self.t_first_token_perf = None  # perf_counter for TPOT calculation

    def __call__(self, input_ids, scores):
        if self.t_first_token is None:
            self.t_first_token = time.time()
            self.t_first_token_perf = time.perf_counter()
        return scores

from config import (
    BASE_MODEL_ID, LORA_ADAPTER_ID, DTYPE,
    BASELINE_SERVER_PORT, MAX_NEW_TOKENS, WORKER_DEVICE
)
from utils.lora_utils import apply_lora_to_model, remove_lora_from_model, print_lora_info
from utils.batch_scheduler import (
    AdaptiveBatchScheduler, BatchConfig, Batch, InferenceRequest
)

try:
    import ext_ipc_wrap as ext
except ImportError:
    print("ERROR: ext_ipc_wrap not found. Run: python setup.py build_ext --inplace")
    sys.exit(1)

# Import stream loader for CUDA optimized loading (Paper Section 5)
try:
    from utils.stream_loader import async_load_state_dict
    STREAM_LOADER_AVAILABLE = True
except ImportError:
    STREAM_LOADER_AVAILABLE = False

_t_module_end = time.perf_counter()
_module_import_time = _t_module_end - _t_module_start


def log(worker_id: int, msg: str) -> None:
    print(f"[BatchedWorker-{worker_id}] {msg}", flush=True)


class BatchedWorker:
    """
    Worker with adaptive batching support.

    Key features:
    - Batches multiple requests for efficient GPU utilization
    - SLO-aware batch sizing
    - Uses custom unmerged LoRA inference
    """

    def __init__(self, worker_id: int = 0, server_host: str = "localhost",
                 server_port: int = BASELINE_SERVER_PORT, lora_id: str = None,
                 device: str = WORKER_DEVICE,
                 # Batching parameters
                 slo_ms: float = 2000.0,
                 max_batch_size: int = 8,
                 base_ttft_ms: float = 400.0,
                 marginal_cost_ms: float = 50.0):

        self.worker_id = worker_id
        self.server_host = server_host
        self.server_port = server_port
        self.lora_id = lora_id or LORA_ADAPTER_ID
        self.device = device
        self.device_idx = int(device.split(":")[-1]) if ":" in device else 0
        self.initialized = False
        self.running = False

        # Lock to prevent inference during adapter swaps
        self._model_lock = threading.RLock()

        # Batching configuration
        self.batch_config = BatchConfig(
            function_id=f"worker_{worker_id}",
            base_ttft_ms=base_ttft_ms,
            marginal_cost_ms=marginal_cost_ms,
            slo_ms=slo_ms,
            max_batch_size=max_batch_size
        )

        # Scheduler will be created after model initialization
        self.scheduler: Optional[AdaptiveBatchScheduler] = None

        # Cold-start ground-truth tracking (Paper Section 6.1)
        # True after init or adapter swap, cleared after first successful batch
        self._is_cold = True

    def initialize(self) -> bool:
        """Connect to server and build model from IPC handles."""
        if self.initialized:
            return True

        t_start = time.perf_counter()
        log(self.worker_id, f"Connecting to {self.server_host}:{self.server_port}...")
        log(self.worker_id, f"Using device: {self.device}")
        log(self.worker_id, "Batched worker with adaptive scheduling")

        try:
            torch.cuda.set_device(self.device_idx)
            torch.cuda.synchronize(self.device_idx)
            t_cuda = time.perf_counter()

            self.ipc_metadata = self._fetch_ipc_metadata()
            t_connect = time.perf_counter()
            log(self.worker_id, f"  Received IPC metadata ({self.ipc_metadata['num_tensors']} tensors)")

            self.model = self._reconstruct_model_from_ipc()
            t_model = time.perf_counter()

            # Apply LoRA using custom implementation
            log(self.worker_id, f"  Applying custom LoRA: {self.lora_id}")
            self.model = apply_lora_to_model(
                self.model, self.lora_id,
                device=self.device, dtype=DTYPE
            )
            self.model.eval()
            t_lora = time.perf_counter()

            print_lora_info(self.model)

            model_name = self.ipc_metadata.get('model_name', BASE_MODEL_ID)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"  # For batch generation
            t_tokenizer = time.perf_counter()

            # Initialize batch scheduler with GPU-wide contention tracking
            self.scheduler = AdaptiveBatchScheduler(
                process_batch_fn=self._process_batch,
                gpu_device=self.device
            )
            self.scheduler.register_function(self.batch_config)

            log(self.worker_id,
                f"INIT: lib_import={_module_import_time:.2f}s cuda_init={t_cuda-t_start:.2f}s "
                f"connect={t_connect-t_cuda:.2f}s model={t_model-t_connect:.2f}s "
                f"lora={t_lora-t_model:.2f}s tok={t_tokenizer-t_lora:.2f}s TOTAL={t_tokenizer-t_start:.2f}s")

            log(self.worker_id, f"Batch config: max_size={self.batch_config.compute_max_batch_size()}, "
                               f"SLO={self.batch_config.slo_ms}ms")

            self.initialized = True
            log(self.worker_id, "Ready for requests!")
            return True

        except Exception as e:
            log(self.worker_id, f"Initialization failed: {e}")
            traceback.print_exc()
            return False

    def _fetch_ipc_metadata(self):
        """Get IPC handles from server via TCP, with retry."""
        max_retries = 5
        for attempt in range(max_retries):
            sock = None
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(60.0)
                sock.connect((self.server_host, self.server_port))
                sock.sendall(b"GET_IPC_METADATA")
                size = int.from_bytes(sock.recv(4), 'big')
                data = b''
                while len(data) < size:
                    chunk = sock.recv(min(size - len(data), 65536))
                    if not chunk:
                        raise RuntimeError("Connection closed")
                    data += chunk
                return pickle.loads(data)
            except (ConnectionRefusedError, socket.timeout, OSError, RuntimeError) as e:
                if attempt < max_retries - 1:
                    delay = 1.0 * (2 ** attempt)  # 1, 2, 4, 8, 16s
                    log(self.worker_id, f"BMS connect attempt {attempt+1}/{max_retries} failed: {e}, retry in {delay:.0f}s")
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"Failed to connect to BMS after {max_retries} attempts: {e}")
            finally:
                if sock:
                    try: sock.close()
                    except: pass

    def _reconstruct_model_from_ipc(self):
        """Build model shell and replace tensors with IPC handles."""
        tensors_meta = self.ipc_metadata["tensors"]
        model_name = self.ipc_metadata.get("model_name", BASE_MODEL_ID)

        param_map, buffer_map = {}, {}
        for info in tensors_meta:
            try:
                t = ext.open_ipc_as_tensor(
                    info["handle"], info["shape"], info.get("stride", []),
                    info["dtype_code"], info.get("device_index", 0), info.get("offset_bytes", 0)
                )
                if info["kind"] == "param":
                    param_map[info["name"]] = t
                else:
                    buffer_map[info["name"]] = t
            except Exception as e:
                log(self.worker_id, f"  Warning: Could not open {info['name']}: {e}")

        log(self.worker_id, f"  Opened {len(param_map)} params, {len(buffer_map)} buffers via IPC")

        if 'model_config' in self.ipc_metadata:
            config_dict = self.ipc_metadata['model_config'].copy()
            model_type = config_dict.pop('model_type', 'llama')
            cfg = AutoConfig.for_model(model_type, **config_dict)
        else:
            cfg = AutoConfig.from_pretrained(model_name, local_files_only=True)

        with torch.device("meta"):
            try:
                model = AutoModelForCausalLM.from_config(cfg, torch_dtype=DTYPE)
            except (ValueError, KeyError):
                # Same fallback as the import above.
                if _LlamaForCausalLM is not None and getattr(cfg, "model_type", "") == "llama":
                    model = _LlamaForCausalLM(cfg).to(dtype=DTYPE)
                else:
                    raise

        def _replace_tensors(module, prefix=""):
            for name, _ in list(module.named_parameters(recurse=False)):
                full_name = f"{prefix}.{name}" if prefix else name
                if full_name in param_map:
                    delattr(module, name)
                    module.register_parameter(name, nn.Parameter(param_map[full_name], requires_grad=False))

            for name, _ in list(module.named_buffers(recurse=False)):
                full_name = f"{prefix}.{name}" if prefix else name
                if full_name in buffer_map:
                    delattr(module, name)
                    module.register_buffer(name, buffer_map[full_name])

            for name, child in module.named_children():
                _replace_tensors(child, f"{prefix}.{name}" if prefix else name)

        _replace_tensors(model)
        if hasattr(model, 'tie_weights'):
            model.tie_weights()
        return model

    def _process_batch(self, batch: Batch) -> List[Dict]:
        """
        Process a batch of requests.

        This is called by the scheduler when a batch is ready.
        Returns a list of results, one per request in the batch.

        Acquires _model_lock to prevent concurrent adapter swaps.
        """
        try:
            prompts = batch.prompts
            max_tokens = batch.max_tokens
            batch_size = batch.size

            log(self.worker_id, f"Processing batch {batch.batch_id}: "
                               f"{batch_size} requests, max_tokens={max_tokens}")

            t_start = time.perf_counter()

            with self._model_lock:
                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_NEW_TOKENS
                ).to(self.device)

                input_lens = [
                    (inputs.attention_mask[i] == 1).sum().item()
                    for i in range(batch_size)
                ]

                # Cap total sequence length (input + output) to MAX_NEW_TOKENS
                max_input_len = max(input_lens)
                effective_max_tokens = max(1, min(max_tokens, MAX_NEW_TOKENS - max_input_len))

                torch.cuda.synchronize(self.device_idx)
                t_gen_start = time.perf_counter()

                ttft_timer = FirstTokenTimer()
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=effective_max_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        logits_processor=[ttft_timer],
                    )

                torch.cuda.synchronize(self.device_idx)
                t_gen_end = time.perf_counter()

            # Process results for each request
            gen_time_ms = (t_gen_end - t_gen_start) * 1000

            # First pass: decode outputs and count total tokens for TPOT
            decoded = []
            for i in range(batch_size):
                output_ids = outputs[i, inputs.input_ids.shape[1]:]
                output_len = (output_ids != self.tokenizer.pad_token_id).sum().item()
                text = self.tokenizer.decode(output_ids[:output_len], skip_special_tokens=True)
                decoded.append((output_len, text))

            total_tokens = sum(ol for ol, _ in decoded)

            # TPOT: decode time per output token (excludes prefill)
            # FirstTokenTimer marks the boundary between prefill and decoding.
            if ttft_timer.t_first_token_perf and total_tokens > 0:
                decode_time_ms = (t_gen_end - ttft_timer.t_first_token_perf) * 1000
                tpot_ms = decode_time_ms / total_tokens
            elif total_tokens > 0:
                tpot_ms = gen_time_ms / total_tokens
            else:
                tpot_ms = 0.0
            batch_throughput = round(total_tokens / (gen_time_ms / 1000), 2) if gen_time_ms > 0 else 0

            # Wall-clock timestamps for cross-process TTFT/E2E
            t_gen_start_wall = time.time() - gen_time_ms / 1000  # approximate
            t_gen_end_wall = time.time()

            # Second pass: build per-request results
            results = []
            for i in range(batch_size):
                input_len = input_lens[i]
                output_len, text = decoded[i]

                req = batch.requests[i]
                # Use wall-clock times for TTFT and E2E (cross-process accurate)
                e2e_ms = (t_gen_end_wall - req.arrival_time_wall) * 1000
                # queue_wait: time from controller arrival to container dispatch
                # (includes batch formation + container selection)
                if batch.dispatch_time_wall and req.arrival_time_wall:
                    queue_wait_ms = (batch.dispatch_time_wall - req.arrival_time_wall) * 1000
                else:
                    queue_wait_ms = 0.0

                # TTFT: pure prefill time (dispatch to first token)
                if ttft_timer.t_first_token and batch.dispatch_time_wall:
                    ttft_ms = (ttft_timer.t_first_token - batch.dispatch_time_wall) * 1000
                elif ttft_timer.t_first_token and req.arrival_time_wall:
                    ttft_ms = (ttft_timer.t_first_token - req.arrival_time_wall) * 1000
                else:
                    ttft_ms = (t_gen_start_wall - req.arrival_time_wall) * 1000

                results.append({
                    "success": True,
                    "text": text,
                    "tokens": output_len,
                    "input_tokens": input_len,
                    "worker_id": self.worker_id,
                    "batch_id": batch.batch_id,
                    "batch_size": batch_size,
                    "e2e_ms": round(e2e_ms, 2),
                    "queue_wait_ms": round(queue_wait_ms, 2),
                    "ttft_ms": round(ttft_ms, 2),
                    "tpot_ms": round(tpot_ms, 2),
                    "gen_time_ms": round(gen_time_ms, 2),
                    "gen_throughput": round(output_len / (gen_time_ms / 1000), 2) if gen_time_ms > 0 else 0,
                    "batch_throughput": batch_throughput,
                    "is_cold_start": self._is_cold,
                    "lora_type": "custom_unmerged_batched"
                })

            # Clear cold-start flag after first successful batch
            self._is_cold = False
            log(self.worker_id, f"Batch {batch.batch_id} complete: "
                               f"{total_tokens} tokens in {gen_time_ms:.1f}ms "
                               f"({total_tokens / (gen_time_ms / 1000):.1f} tok/s)")

            return results

        except Exception as e:
            log(self.worker_id, f"Batch processing failed: {e}")
            traceback.print_exc()
            return [{"success": False, "error": str(e)} for _ in range(batch.size)]

    def process_request(self, prompt: str, max_tokens: int = MAX_NEW_TOKENS,
                        timeout: float = 30.0) -> Dict:
        """
        Submit a single request and wait for result.

        The request will be batched with others by the scheduler.
        """
        if not self.initialized:
            return {"error": "Worker not initialized", "success": False}

        request = self.scheduler.submit_request(
            function_id=self.batch_config.function_id,
            prompt=prompt,
            max_tokens=max_tokens
        )

        result = request.wait(timeout=timeout)
        if result is None:
            return {"error": "Request timed out", "success": False}
        return result

    def start_http_server(self, http_port: int):
        """Start HTTP server with batching support."""
        try:
            from flask import Flask, request, jsonify
        except ImportError:
            print("Flask not installed. Run: pip install flask")
            return

        app = Flask(f"batched_worker_{self.worker_id}")

        # Start the batch scheduler
        self.scheduler.start()

        @app.route('/health', methods=['GET'])
        def health():
            stats = self.scheduler.get_stats() if self.scheduler else {}
            return jsonify({
                "status": "ready" if self.initialized else "initializing",
                "worker_id": self.worker_id,
                "device": self.device,
                "type": "batched",
                "batch_config": {
                    "max_batch_size": self.batch_config.compute_max_batch_size(),
                    "slo_ms": self.batch_config.slo_ms
                },
                "scheduler_stats": stats
            })

        @app.route('/inference', methods=['POST'])
        def inference():
            data = request.get_json() or {}
            prompt = data.get('prompt', '')
            max_tokens = min(data.get('max_tokens', MAX_NEW_TOKENS), MAX_NEW_TOKENS)
            timeout = data.get('timeout', 300.0)

            if not prompt:
                return jsonify({"error": "No prompt", "success": False}), 400

            result = self.process_request(prompt, max_tokens, timeout)
            return jsonify(result), 200 if result.get('success') else 500

        @app.route('/batch_inference', methods=['POST'])
        def batch_inference():
            """Submit multiple requests at once."""
            data = request.get_json() or {}
            prompts = data.get('prompts', [])
            max_tokens = data.get('max_tokens', MAX_NEW_TOKENS)
            timeout = data.get('timeout', 60.0)

            if not prompts:
                return jsonify({"error": "No prompts", "success": False}), 400

            # Submit all requests
            requests_list = [
                self.scheduler.submit_request(
                    function_id=self.batch_config.function_id,
                    prompt=p,
                    max_tokens=max_tokens
                )
                for p in prompts
            ]

            # Wait for all results
            results = []
            for req in requests_list:
                r = req.wait(timeout=timeout)
                if r is None:
                    r = {"success": False, "error": "Request timed out in worker batch queue"}
                results.append(r)
            return jsonify({"results": results, "count": len(results)})

        @app.route('/batch_execute', methods=['POST'])
        def batch_execute():
            """
            Execute a pre-formed batch directly, bypassing worker scheduler.

            Called by the Controller which has already formed the batch
            via its own AdaptiveBatchScheduler. This avoids double-batching
            delay (Paper Section 4.2: batching is controller's responsibility).
            """
            data = request.get_json() or {}
            prompts = data.get('prompts', [])
            max_tokens = min(data.get('max_tokens', MAX_NEW_TOKENS), MAX_NEW_TOKENS)
            batch_id = data.get('batch_id', f'direct_{int(time.time()*1000)}')
            function_id = data.get('function_id', '')
            # Wall-clock arrival times from controller (for cross-process TTFT)
            arrival_times = data.get('arrival_times', [])
            # Wall-clock dispatch time from controller (batch wait + selection done)
            dispatch_time_wall = data.get('dispatch_time_wall')

            if not prompts:
                return jsonify({"error": "No prompts", "success": False}), 400

            now = time.perf_counter()
            now_wall = time.time()
            inf_requests = []
            for i, p in enumerate(prompts):
                req = InferenceRequest(
                    request_id=f"{batch_id}_req_{i}",
                    prompt=p,
                    max_tokens=max_tokens,
                    arrival_time=now,
                )
                # Use controller's wall-clock arrival if provided
                if i < len(arrival_times):
                    req.arrival_time_wall = arrival_times[i]
                else:
                    req.arrival_time_wall = now_wall
                inf_requests.append(req)

            batch_obj = Batch(
                batch_id=batch_id,
                function_id=function_id or self.batch_config.function_id,
                requests=inf_requests,
            )
            batch_obj.dispatch_time = now
            batch_obj.dispatch_time_wall = dispatch_time_wall

            results = self._process_batch(batch_obj)
            return jsonify({"results": results, "count": len(results)})

        @app.route('/stats', methods=['GET'])
        def stats():
            return jsonify(self.scheduler.get_stats())

        @app.route('/preload_adapter', methods=['POST'])
        def preload_adapter():
            """
            Preload a LoRA adapter to this worker.

            Used by the PreloadAgent to load adapters in advance.

            Request body:
                adapter_id: HuggingFace adapter ID
                to_gpu: Whether to load to GPU (default: True)
            """
            data = request.get_json() or {}
            adapter_id = data.get('adapter_id', '')
            to_gpu = data.get('to_gpu', True)

            if not adapter_id:
                return jsonify({"success": False, "error": "No adapter_id"}), 400

            try:
                log(self.worker_id, f"Preloading adapter {adapter_id}, to_gpu={to_gpu}")

                # Check if already loaded
                if hasattr(self, 'preloaded_adapters') and adapter_id in self.preloaded_adapters:
                    return jsonify({"success": True, "status": "already_loaded"})

                # Load the adapter
                from utils.lora_utils import load_lora_weights, download_lora_adapter

                # Download if needed
                adapter_path = download_lora_adapter(adapter_id)

                lora_weights = load_lora_weights(adapter_path)

                # Store in preloaded cache
                if not hasattr(self, 'preloaded_adapters'):
                    self.preloaded_adapters = {}

                if to_gpu:
                    # Move to GPU using CUDA Streams for concurrent loading (Paper Section 5)
                    if STREAM_LOADER_AVAILABLE:
                        log(self.worker_id, f"Using CUDA Streams for concurrent GPU transfer")
                        lora_weights = async_load_state_dict(
                            lora_weights,
                            device_id=self.device_idx,
                            num_streams=4
                        )
                    else:
                        # Fallback: sequential transfer
                        lora_weights = {k: v.to(self.device) for k, v in lora_weights.items()}

                self.preloaded_adapters[adapter_id] = {
                    'weights': lora_weights,
                    'on_gpu': to_gpu,
                    'loaded_at': time.time()
                }

                log(self.worker_id, f"Preloaded adapter {adapter_id}")
                return jsonify({
                    "success": True,
                    "adapter_id": adapter_id,
                    "on_gpu": to_gpu,
                    "weight_count": len(lora_weights)
                })

            except Exception as e:
                log(self.worker_id, f"Failed to preload adapter: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @app.route('/offload_adapter', methods=['POST'])
        def offload_adapter():
            """
            Offload a LoRA adapter from GPU to CPU or unload completely.

            Used by the GPU Offloader during memory pressure.
            """
            data = request.get_json() or {}
            artifact_id = data.get('artifact_id', '')
            keep_in_container = data.get('keep_in_container', True)

            # Extract adapter ID from artifact ID
            adapter_id = artifact_id.replace('adapter_', '').replace('_', '/')

            if not hasattr(self, 'preloaded_adapters'):
                return jsonify({"success": False, "error": "No preloaded adapters"})

            if adapter_id not in self.preloaded_adapters:
                return jsonify({"success": False, "error": f"Adapter {adapter_id} not found"})

            try:
                if keep_in_container:
                    # Move to CPU using CUDA Streams for concurrent offloading (Paper Section 5)
                    weights = self.preloaded_adapters[adapter_id]['weights']
                    if STREAM_LOADER_AVAILABLE:
                        from utils.stream_loader import async_offload_state_dict
                        log(self.worker_id, f"Using CUDA Streams for concurrent CPU offload")
                        self.preloaded_adapters[adapter_id]['weights'] = async_offload_state_dict(
                            weights,
                            num_streams=4
                        )
                    else:
                        # Fallback: sequential transfer
                        self.preloaded_adapters[adapter_id]['weights'] = {
                            k: v.cpu() for k, v in weights.items()
                        }
                    self.preloaded_adapters[adapter_id]['on_gpu'] = False
                    log(self.worker_id, f"Offloaded adapter {adapter_id} to CPU")
                else:
                    # Completely unload
                    del self.preloaded_adapters[adapter_id]
                    log(self.worker_id, f"Unloaded adapter {adapter_id}")

                torch.cuda.empty_cache()

                return jsonify({"success": True})

            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 500

        @app.route('/swap_adapter', methods=['POST'])
        def swap_adapter():
            """
            Swap the active LoRA adapter at runtime.

            Removes current LoRA layers, applies new adapter, and updates
            worker identity. Enables dynamic container reuse across functions.

            Request body:
                adapter_id: Path or HuggingFace ID of new adapter
                function_id: New function ID (optional)
            """
            data = request.get_json() or {}
            new_adapter_id = data.get('adapter_id', '')
            new_function_id = data.get('function_id', '')

            if not new_adapter_id:
                return jsonify({"success": False, "error": "No adapter_id"}), 400

            if new_adapter_id == self.lora_id:
                return jsonify({"success": True, "status": "already_active"})

            try:
                old_adapter = self.lora_id
                log(self.worker_id, f"Swapping adapter: {old_adapter} -> {new_adapter_id}")
                swap_start = time.time()

                # Try to acquire model lock with short timeout — fail fast if busy
                acquired = self._model_lock.acquire(timeout=0.5)
                if not acquired:
                    return jsonify({
                        "success": False,
                        "error": "model_busy",
                        "detail": "Model lock held by inference, retry later"
                    }), 503

                try:
                    # 1. Remove current LoRA layers (restores base Linear layers)
                    t0 = time.time()
                    self.model = remove_lora_from_model(self.model)
                    remove_ms = (time.time() - t0) * 1000

                    # 2. Apply new adapter
                    t0 = time.time()
                    self.model = apply_lora_to_model(
                        self.model, new_adapter_id,
                        device=self.device, dtype=DTYPE
                    )
                    self.model.eval()
                    apply_ms = (time.time() - t0) * 1000
                finally:
                    self._model_lock.release()

                swap_ms = (time.time() - swap_start) * 1000

                # 3. Update identity
                self.lora_id = new_adapter_id
                if new_function_id:
                    self.function_id = new_function_id

                # Mark next inference as cold start (adapter just changed)
                self._is_cold = True

                log(self.worker_id,
                    f"Adapter swap complete: {old_adapter} -> {new_adapter_id} "
                    f"in {swap_ms:.0f}ms (remove={remove_ms:.0f}ms, apply={apply_ms:.0f}ms)")
                return jsonify({
                    "success": True,
                    "old_adapter": old_adapter,
                    "new_adapter": new_adapter_id,
                    "swap_ms": swap_ms,
                    "remove_ms": remove_ms,
                    "apply_ms": apply_ms,
                })

            except Exception as e:
                log(self.worker_id, f"Adapter swap failed: {e}")
                traceback.print_exc()
                return jsonify({"success": False, "error": str(e)}), 500

        @app.route('/memory_status', methods=['GET'])
        def memory_status():
            """
            Get current memory status.

            Used by the Controller and GPU Offloader to monitor memory.
            """
            try:
                # GPU memory
                gpu_total = torch.cuda.get_device_properties(self.device_idx).total_memory
                gpu_reserved = torch.cuda.memory_reserved(self.device_idx)
                gpu_allocated = torch.cuda.memory_allocated(self.device_idx)

                # Preloaded adapters
                preloaded_count = len(getattr(self, 'preloaded_adapters', {}))
                gpu_adapters = sum(
                    1 for a in getattr(self, 'preloaded_adapters', {}).values()
                    if a.get('on_gpu', False)
                )

                return jsonify({
                    "gpu": {
                        "total_mb": gpu_total / (1024 * 1024),
                        "reserved_mb": gpu_reserved / (1024 * 1024),
                        "allocated_mb": gpu_allocated / (1024 * 1024),
                        "free_mb": (gpu_total - gpu_allocated) / (1024 * 1024),
                        "usage_percent": gpu_allocated / gpu_total if gpu_total > 0 else 0
                    },
                    "adapters": {
                        "preloaded_count": preloaded_count,
                        "gpu_loaded_count": gpu_adapters
                    },
                    "device": self.device,
                    "worker_id": self.worker_id
                })

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @app.route('/set_profiling_mode', methods=['POST'])
        def set_profiling_mode():
            """Toggle profiling mode on the batch scheduler.

            When enabled, batches are dispatched immediately (no
            fill-or-expire delay) so the profiler measures real GPU
            compute time instead of queue-polluted TTFT.
            """
            data = request.get_json() or {}
            enabled = bool(data.get("enabled", False))

            if self.scheduler:
                with self.scheduler.lock:
                    for cfg in self.scheduler.configs.values():
                        cfg.profiling_mode = enabled
                # Also update the worker's own BatchConfig reference
                self.batch_config.profiling_mode = enabled
                log(self.worker_id, f"Profiling mode {'enabled' if enabled else 'disabled'}")
                return jsonify({"success": True, "profiling_mode": enabled})
            return jsonify({"success": False, "error": "no scheduler"}), 500

        @app.route('/update_batch_config', methods=['POST'])
        def update_batch_config():
            """Update adaptive batch scheduler parameters from profiler."""
            data = request.get_json() or {}
            base_ttft = data.get("base_ttft_ms")
            marginal_cost = data.get("marginal_cost_ms")
            slo = data.get("slo_ms")

            if self.scheduler:
                # Update all function configs in the scheduler
                fid = getattr(self, 'function_id', None) or self.batch_config.function_id
                updated = self.scheduler.update_config_from_profile(
                    function_id=fid,
                    base_ttft_ms=base_ttft,
                    marginal_cost_ms=marginal_cost,
                    slo_ms=slo,
                )
                # Also update the base config
                if base_ttft is not None:
                    self.batch_config.base_ttft_ms = base_ttft
                if marginal_cost is not None:
                    self.batch_config.marginal_cost_ms = marginal_cost
                if slo is not None:
                    self.batch_config.slo_ms = slo
                return jsonify({"success": updated})
            return jsonify({"success": False, "error": "no scheduler"})

        @app.route('/shutdown', methods=['POST'])
        def shutdown():
            self.running = False
            self.scheduler.stop()
            threading.Thread(target=lambda: (time.sleep(0.5), os._exit(0)), daemon=True).start()
            return jsonify({"status": "shutting_down"})

        self.running = True
        log(self.worker_id, f"HTTP server on port {http_port} (batched)")
        app.run(host='0.0.0.0', port=http_port, threaded=True, use_reloader=False)

    def shutdown(self):
        self.running = False
        if self.scheduler:
            self.scheduler.stop()
        log(self.worker_id, "Shutdown complete")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batched Worker with Adaptive Scheduling")
    parser.add_argument("--server-host", default="localhost")
    parser.add_argument("--server-port", type=int, default=BASELINE_SERVER_PORT)
    parser.add_argument("--http-port", type=int, required=True)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--lora", default=LORA_ADAPTER_ID)
    parser.add_argument("--device", default=WORKER_DEVICE)
    # Batching parameters
    parser.add_argument("--slo-ms", type=float, default=2000.0,
                        help="SLO target for TTFT in milliseconds")
    parser.add_argument("--max-batch-size", type=int, default=8,
                        help="Maximum batch size")
    parser.add_argument("--base-ttft-ms", type=float, default=400.0,
                        help="Base TTFT for single request (from profiling)")
    parser.add_argument("--marginal-cost-ms", type=float, default=50.0,
                        help="Marginal TTFT cost per additional request")
    args = parser.parse_args()

    worker = BatchedWorker(
        worker_id=args.worker_id,
        server_host=args.server_host,
        server_port=args.server_port,
        lora_id=args.lora,
        device=args.device,
        slo_ms=args.slo_ms,
        max_batch_size=args.max_batch_size,
        base_ttft_ms=args.base_ttft_ms,
        marginal_cost_ms=args.marginal_cost_ms
    )

    if not worker.initialize():
        sys.exit(1)

    try:
        worker.start_http_server(args.http_port)
    except KeyboardInterrupt:
        pass
    finally:
        worker.shutdown()


if __name__ == "__main__":
    main()
