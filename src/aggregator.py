#!/usr/bin/env python3
"""Aggregator for multi-GPU batched LLM inference."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hmac
import torch
import time
import threading
import socket
import traceback
import gc
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

# Reuse JSON wire helpers from worker.utils (same codebase, same PYTHONPATH)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'worker'))
from utils import send_msg, recv_msg, AGG_SECRET

from config import (
    BASE_MODEL_ID, AGGREGATOR_DEVICE, WORKER_DEVICE, DTYPE,
    QUEUE_CAPACITY, MAX_TICKETS, MAX_WORKERS, MAX_WORKERS_PER_GPU,
    DEFAULT_PORT, AGGREGATOR_HEALTH_PORT, PREFILL_CHUNK_SIZE,
)


def log(msg: str) -> None:
    print(f"[Aggregator] {msg}", flush=True)


def get_device_index(device_str: str) -> int:
    """Extract device index from device string like 'cuda:1'."""
    return int(device_str.split(":")[-1]) if ":" in device_str else 0


class AggregatorHealthHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for health checks."""

    def __init__(self, aggregator, *args, **kwargs):
        self.aggregator = aggregator
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        # Suppress default logging
        pass

    def do_GET(self):
        if self.path == '/health':
            response = {
                'status': 'ready' if self.aggregator.is_ready else 'starting',
                'num_registered': self.aggregator.num_registered,
                'num_slots': self.aggregator.num_slots,
                'device': self.aggregator.aggregator_device,
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        elif self.path == '/status':
            response = {
                'status': 'ready' if self.aggregator.is_ready else 'starting',
                'num_registered': self.aggregator.num_registered,
                'max_workers': self.aggregator.max_workers,
                'num_slots': self.aggregator.num_slots,
                'device': self.aggregator.aggregator_device,
                'port': self.aggregator.port,
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()


class BatchedAggregator:
    """Lock-step synchronized aggregator with C++ batched GEMM execution."""

    NUM_SLOTS = MAX_WORKERS_PER_GPU  # One slot per worker; MAX_SLOTS (128) is the compile-time upper bound

    def __init__(self, port: int = DEFAULT_PORT, device: str = None,
                 health_port: int = AGGREGATOR_HEALTH_PORT, model_id: str = None,
                 donor_host: str = None, donor_port: int = None,
                 host: str = '127.0.0.1', num_slots: int = None):
        self.port = port
        self.host = host
        self.health_port = health_port
        self.max_workers = MAX_WORKERS
        self.num_registered = 0
        self.worker_lock = threading.Lock()

        # Defaults to MAX_WORKERS_PER_GPU. Callers packing more workers onto
        # one GPU (e.g. RQ5's overhead sweep) must override it, or the excess
        # workers contend for slots.
        self.num_slots = num_slots if num_slots is not None else self.NUM_SLOTS
        self.slot_pool_info = None

        # Health server state
        self.is_ready = False
        self._health_server = None
        self._health_thread = None

        # Donor mode: copy weights from another aggregator via P2P
        self.donor_host = donor_host
        self.donor_port = donor_port
        self._donor_mode = donor_host is not None and donor_port is not None

        # Model to load (can override config)
        self.model_id = model_id if model_id else BASE_MODEL_ID

        # Use provided device or fall back to config
        self.aggregator_device = device if device else AGGREGATOR_DEVICE
        self.aggregator_device_idx = get_device_index(self.aggregator_device)

        # Import extensions
        try:
            import ext_ipc_queue as extq
            import ext_aggregator
            self.extq = extq
            self.ext_agg = ext_aggregator
        except ImportError as e:
            log(f"ERROR: Extension not found: {e}")
            log("Build extensions: pip install -e .")
            sys.exit(1)

        log("=" * 60)
        log("Multi-GPU Batched Aggregator")
        log("=" * 60)
        log(f"Aggregator: {self.aggregator_device}")
        log(f"Workers: {WORKER_DEVICE}")
        if self._donor_mode:
            log(f"Donor mode: copying weights from {donor_host}:{donor_port}")

        torch.cuda.set_device(self.aggregator_device_idx)
        torch.cuda.init()  # Force CUDA context init (avoids 3s lazy init during slot pool creation)

        # Initialize with timing
        t0 = time.perf_counter()

        if self._donor_mode:
            self._init_from_donor()
        else:
            self._load_model()
            t1 = time.perf_counter()
            log(f"  Model load time: {t1-t0:.1f}s")

            self._serialize_norm_weights()
            t2 = time.perf_counter()
            log(f"  Norm serialization time: {t2-t1:.2f}s")

            self._cache_config()
            t3 = time.perf_counter()
            log(f"  Config cache time: {t3-t2:.2f}s")

        t_qb = time.perf_counter()
        self._create_queue_and_buffers()
        t4 = time.perf_counter()
        log(f"  Queue/buffer creation time: {t4-t_qb:.2f}s")

        self._start_server()

        log("\n" + "=" * 60)
        log(f"Aggregator ready! Total init: {time.perf_counter()-t0:.1f}s")
        log("=" * 60 + "\n")

    def _load_model(self):
        """Load base model on aggregator device."""
        from transformers import AutoModelForCausalLM

        log(f"\n[1/4] Loading base model: {self.model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=DTYPE, device_map=None, local_files_only=True
        )
        self.model.to(self.aggregator_device).eval()

        # Extract config from loaded model
        cfg = self.model.config
        self.num_layers = cfg.num_hidden_layers
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = getattr(cfg, 'num_key_value_heads', self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.intermediate_size = cfg.intermediate_size
        self.vocab_size = cfg.vocab_size
        self.rms_norm_eps = getattr(cfg, 'rms_norm_eps', 1e-6)

        # Store layer references
        self.layers = self.model.model.layers

        mem = torch.cuda.memory_allocated(self.aggregator_device_idx) / 1024**3
        log(f"  {self.num_layers} layers, {self.hidden_size} hidden, {self.vocab_size} vocab")
        log(f"  Memory: {mem:.2f} GB")

    def _init_from_donor(self):
        """Initialize by copying weights from a donor aggregator via P2P."""
        log(f"\n[1/2] Connecting to donor aggregator at {self.donor_host}:{self.donor_port}...")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(120.0)
        sock.connect((self.donor_host, self.donor_port))

        # Request weight export (JSON, include secret if configured)
        req = {'type': 'export_weights'}
        if AGG_SECRET:
            req['secret'] = AGG_SECRET
        send_msg(sock, req)

        export = self._recv(sock, timeout=120.0)
        sock.close()

        if not export:
            raise RuntimeError("Failed to receive weight export from donor")
        if export.get('type') != 'weight_export':
            raise RuntimeError(f"Unexpected response from donor: {export.get('type')}")

        t0 = time.perf_counter()

        # Import weights via P2P copy in C++
        ipc_handles = export['ipc_handles']
        self.ext_agg.import_weights_from_ipc(ipc_handles, self.aggregator_device_idx)

        t1 = time.perf_counter()
        log(f"  P2P weight import: {t1-t0:.1f}s")

        # Receive norm_weights, model_config, model metadata from donor
        self.norm_weights = export['norm_weights']
        self.model_config_dict = export['model_config']
        self.model_id = export['model_name']
        self.num_layers = export['num_layers']
        self.hidden_size = export['hidden_size']
        self.num_heads = export['num_heads']
        self.num_kv_heads = export['num_kv_heads']
        self.head_dim = export['head_dim']
        self.intermediate_size = export['intermediate_size']
        self.vocab_size = export['vocab_size']
        self.rms_norm_eps = export.get('rms_norm_eps', 1e-6)

        log(f"  Imported model: {self.num_layers} layers, {self.hidden_size} hidden, {self.vocab_size} vocab")

    def _serialize_norm_weights(self):
        """Serialize norm weights to send to workers."""
        log("\n[2/4] Serializing norm weights...")

        def to_bytes(t):
            t_cpu = t.detach().cpu()
            if t_cpu.dtype == torch.bfloat16:
                t_cpu = t_cpu.to(torch.float32)
            return t_cpu.numpy().tobytes()

        self.norm_weights = {
            'final_norm': to_bytes(self.model.model.norm.weight),
            'rms_norm_eps': self.rms_norm_eps,
        }

        # Per-layer norms
        for i, layer in enumerate(self.layers):
            self.norm_weights[f'layer_{i}_input_norm'] = to_bytes(layer.input_layernorm.weight)
            self.norm_weights[f'layer_{i}_post_norm'] = to_bytes(layer.post_attention_layernorm.weight)

        total_kb = sum(len(v) for v in self.norm_weights.values() if isinstance(v, bytes)) / 1024
        log(f"  Norm weights: {total_kb:.1f} KB")

    def _cache_config(self):
        """Cache config to send to workers."""
        log("\n[3/4] Caching config...")

        self.model_config_dict = self.model.config.to_dict()
        log(f"  Config: {len(str(self.model_config_dict))} bytes")

    def _create_queue_and_buffers(self):
        """Create shared memory queue and slot pool."""
        log("\n[4/4] Creating queue and slot pool...")
        self.queue_info = self.extq.create_queue(QUEUE_CAPACITY, MAX_TICKETS)

        # Create slot pool with contiguous buffers (sized dynamically from model dims + chunk size)
        self.slot_pool_info = self.ext_agg.init_slot_pool(
            self.num_slots, self.aggregator_device_idx,
            self.hidden_size, self.intermediate_size,
            self.num_kv_heads, self.head_dim,
            PREFILL_CHUNK_SIZE,
            2 if DTYPE == torch.bfloat16 else 4
        )

        in_kb = self.slot_pool_info['input_buffer_size'] / 1024
        out_kb = self.slot_pool_info['output_buffer_size'] / 1024
        total_mb = self.num_slots * (self.slot_pool_info['input_buffer_size'] + self.slot_pool_info['output_buffer_size']) / (1024*1024)
        log(f"  Queue: capacity={QUEUE_CAPACITY}")
        log(f"  Slot pool: {self.num_slots} slots, {in_kb:.0f}KB input + {out_kb:.0f}KB output per slot ({total_mb:.1f}MB total)")
        log(f"  QKV dim: {self.hidden_size + self.num_kv_heads * self.head_dim * 2} (hidden={self.hidden_size}, kv_heads={self.num_kv_heads})")

    def _start_server(self):
        """Start TCP server for worker registration."""
        log(f"\nStarting server on {self.host}:{self.port}...")
        if AGG_SECRET:
            log("  Auth: HMAC shared secret enabled (AGG_SECRET)")
        else:
            log("  Auth: DISABLED — set AGG_SECRET env var to enable")
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.max_workers)

    def _send(self, sock: socket.socket, obj: dict) -> None:
        """Send a JSON message to a worker."""
        send_msg(sock, obj)

    def _recv(self, sock: socket.socket, timeout: float = 30.0) -> Optional[dict]:
        """Receive a JSON message from a worker with timeout."""
        old_timeout = sock.gettimeout()
        try:
            sock.settimeout(timeout)
            return recv_msg(sock)
        except socket.timeout:
            raise
        except ConnectionError:
            return None
        finally:
            sock.settimeout(old_timeout)

    def _verify_secret(self, received: str) -> bool:
        """Constant-time comparison of shared secret."""
        if not AGG_SECRET:
            return True  # Auth disabled — backward compatible
        return hmac.compare_digest(received, AGG_SECRET)

    def _init_cpp_aggregator(self):
        """Initialize C++ extension with model weights."""
        num_layers = len(self.layers)
        self.ext_agg.init_aggregator_begin(num_layers, self.aggregator_device_idx)

        freed_params = 0
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            mlp = layer.mlp

            # Pass one layer's weights to C++ for fusion
            self.ext_agg.init_aggregator_layer(
                i,
                attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight,
                attn.o_proj.weight,
                attn.o_proj.bias if attn.o_proj.bias is not None else torch.empty(0),
                mlp.gate_proj.weight, mlp.up_proj.weight,
                mlp.down_proj.weight,
                mlp.down_proj.bias if mlp.down_proj.bias is not None else torch.empty(0),
            )

            # Free originals that C++ just fused (Q, K, V, gate, up)
            # C++ holds fused copies; O/down are shared via detach()
            for proj in [attn.q_proj, attn.k_proj, attn.v_proj,
                         mlp.gate_proj, mlp.up_proj]:
                freed_params += proj.weight.numel()
                proj.weight = None

            # Release GPU memory before next layer
            torch.cuda.empty_cache()

        self.ext_agg.init_aggregator_finalize()

        self.ext_agg.init_embed_lm_head_weights(
            self.model.model.embed_tokens.weight,
            self.model.lm_head.weight
        )

        self.ext_agg.init_aggregator_queue(
            self.queue_info['shm_name'],
            self.queue_info['capacity']
        )

        freed_mb = freed_params * 2 / (1024 * 1024)  # bf16 = 2 bytes
        gc.collect()
        torch.cuda.empty_cache()
        log(f"C++ aggregator initialized (freed {freed_mb:.0f} MiB fused-duplicate weights)")

    def _handle_registration(self, sock: socket.socket) -> Optional[int]:
        """Handle worker registration or weight export requests (JSON, no pickle)."""
        try:
            msg = self._recv(sock)
        except Exception:
            return None
        if not msg:
            return None

        # Verify shared secret before doing anything
        if not self._verify_secret(msg.get('secret', '')):
            log("Registration rejected: invalid or missing secret")
            self._send(sock, {'type': 'auth_error', 'error': 'invalid secret'})
            sock.close()
            return None

        # Handle weight export request (from a donor-mode aggregator)
        if msg.get('type') == 'export_weights':
            log("Received weight export request from receiver aggregator")
            try:
                ipc_handles = self.ext_agg.export_weight_ipc_handles()
                self._send(sock, {
                    'type': 'weight_export',
                    'ipc_handles': ipc_handles,
                    'norm_weights': self.norm_weights,
                    'model_config': self.model_config_dict,
                    'model_name': self.model_id,
                    'num_layers': self.num_layers,
                    'hidden_size': self.hidden_size,
                    'num_heads': self.num_heads,
                    'num_kv_heads': self.num_kv_heads,
                    'head_dim': self.head_dim,
                    'intermediate_size': self.intermediate_size,
                    'vocab_size': self.vocab_size,
                    'rms_norm_eps': self.rms_norm_eps,
                })
                log("Weight export sent successfully")
            except Exception as e:
                log(f"Weight export failed: {e}")
                traceback.print_exc()
            sock.close()
            return None

        if msg.get('type') != 'register':
            return None

        with self.worker_lock:
            if self.num_registered >= self.max_workers:
                log(f"Registration rejected: max workers ({self.max_workers}) reached")
                return None
            worker_id = self.num_registered
            self.num_registered += 1

        self._send(sock, {
            'type': 'registered',
            'worker_id': worker_id,
            'norm_weights': self.norm_weights,
            'model_name': self.model_id,
            'model_config': self.model_config_dict,
            'queue': self.queue_info,
            'num_layers': self.num_layers,
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'num_kv_heads': self.num_kv_heads,
            'head_dim': self.head_dim,
            'intermediate_size': self.intermediate_size,
            'vocab_size': self.vocab_size,
            'aggregator_device_idx': self.aggregator_device_idx,
            'input_buffer_size': self.slot_pool_info['input_buffer_size'],
            'output_buffer_size': self.slot_pool_info['output_buffer_size'],
        })
        return worker_id

    def _handle_slot_request(self, sock: socket.socket, msg: dict) -> bool:
        """Handle slot claim/release requests from workers (JSON, no pickle)."""
        req_type = msg.get('slot_request')
        worker_id = msg.get('worker_id', -1)

        if req_type == 'claim':
            try:
                slot_info = self.ext_agg.claim_slot(worker_id)
                self._send(sock, {
                    'type': 'slot_claimed',
                    'slot_id': slot_info['slot_id'],
                    'input_handle': slot_info['input_handle'],
                    'output_handle': slot_info['output_handle'],
                })
                return True
            except Exception as e:
                log(f"Slot claim failed for worker {worker_id}: {e}")
                self._send(sock, {'type': 'slot_error', 'error': str(e)})
                return False

        elif req_type == 'release':
            slot_id = msg.get('slot_id', -1)
            try:
                self.ext_agg.release_slot(slot_id, worker_id)
                self._send(sock, {'type': 'slot_released', 'slot_id': slot_id})
                return True
            except Exception as e:
                log(f"Slot release failed for slot {slot_id}: {e}")
                self._send(sock, {'type': 'slot_error', 'error': str(e)})
                return False

        return False

    def _wait_for_done(self, sock: socket.socket, worker_id: int):
        """Wait for worker to signal done, handle slot requests (JSON, no pickle)."""
        try:
            while True:
                try:
                    msg = self._recv(sock)
                except socket.timeout:
                    continue  # Idle timeout — keep waiting for next message
                if not msg:
                    break  # Real disconnect
                msg_type = msg.get('type')
                if msg_type == 'done':
                    log(f"Worker {worker_id} disconnected")
                    break
                elif msg_type == 'slot_request':
                    self._handle_slot_request(sock, msg)
        except Exception as e:
            log(f"Worker {worker_id} connection error: {e}")
        finally:
            # Signal barrier to stop waiting for slots owned by this worker.
            # Only signal leave — do NOT release to free pool yet, because
            # the aggregator's in-flight GEMM/embedding kernel may still be
            # reading from the slot buffer.  The main loop will release the
            # slot after the current epoch finishes.
            stale_slots = []
            for sid in range(self.num_slots):
                try:
                    owner = self.ext_agg.get_slot_owner(sid)
                    if owner == worker_id:
                        self.extq.slot_signal_leave(sid)
                        stale_slots.append(sid)
                        log(f"Signaled leave for stale slot {sid} from worker {worker_id}")
                except Exception:
                    pass
            # Deferred release: wait briefly for the barrier epoch to finish,
            # then release the slots back to the free pool.
            if stale_slots:
                time.sleep(0.5)  # Allow current barrier epoch to complete
                for sid in stale_slots:
                    try:
                        self.ext_agg.release_slot(sid, worker_id)
                        log(f"Deferred-released slot {sid} from worker {worker_id}")
                    except Exception:
                        pass  # Slot may have been released by another path
            sock.close()

    def _start_health_server(self):
        """Start HTTP health server in background thread."""
        aggregator = self

        def handler(*args, **kwargs):
            return AggregatorHealthHandler(aggregator, *args, **kwargs)

        try:
            # Bind the health endpoint to the same interface as the TCP server
            # (default 127.0.0.1). Hardcoding 0.0.0.0 exposed topology/load info
            # on every interface even when the operator chose a local-only host.
            self._health_server = HTTPServer((self.host, self.health_port), handler)
            self._health_thread = threading.Thread(
                target=self._health_server.serve_forever,
                daemon=True
            )
            self._health_thread.start()
            log(f"Health server started on port {self.health_port}")
        except Exception as e:
            log(f"Warning: Could not start health server: {e}")

    def _stop_health_server(self):
        """Stop the health server."""
        if self._health_server:
            self._health_server.shutdown()
            self._health_server = None
            log("Health server stopped")

    def run(self):
        """Run aggregator with slot-based batching."""
        log("=" * 60)
        log("Starting aggregator (SLOT-BASED batching)")
        log(f"Slots: {self.num_slots}")
        log("=" * 60)
        log("Press Ctrl+C to stop")

        self.extq.init_batch_control(self.num_slots)
        if self._donor_mode:
            # Weights already initialized by import_weights_from_ipc
            # Just need to attach the queue
            self.ext_agg.init_aggregator_queue(
                self.queue_info['shm_name'],
                self.queue_info['capacity']
            )
            log("C++ aggregator initialized (donor mode, weights already imported)")
        else:
            self._init_cpp_aggregator()

        self.registration_running = True
        registration_thread = threading.Thread(
            target=self._registration_loop,
            daemon=True
        )
        registration_thread.start()

        # Start health server and mark as ready
        self._start_health_server()
        self.is_ready = True

        log("Aggregator ready! Waiting for workers...")
        log(f"Health endpoint: http://localhost:{self.health_port}/health")

        dtype_code = 15 if DTYPE == torch.bfloat16 else 16

        # Slot-based main loop
        self.ext_agg.aggregator_main_loop_slots(self.num_slots, dtype_code)

        self.registration_running = False
        self.is_ready = False
        log("Aggregator stopped")

    def _registration_loop(self):
        """Accept worker registrations in background thread."""
        log("Registration loop started")
        while self.registration_running:
            try:
                self.server_socket.settimeout(0.5)
                conn, addr = self.server_socket.accept()
                log(f"Connection accepted from {addr}")
                conn.settimeout(600.0)
                try:
                    worker_id = self._handle_registration(conn)
                    if worker_id is not None:
                        t = threading.Thread(
                            target=self._wait_for_done,
                            args=(conn, worker_id),
                            daemon=True
                        )
                        t.start()
                        log(f"Worker {worker_id} registered successfully from {addr}")
                    else:
                        log(f"Registration returned None for {addr}")
                        conn.close()
                except Exception as e:
                    log(f"Registration handler error: {e}")
                    traceback.print_exc()
                    conn.close()
            except socket.timeout:
                continue
            except Exception as e:
                log(f"Accept error: {e}")
                traceback.print_exc()

    def shutdown(self):
        """Clean shutdown."""
        self.registration_running = False
        self.is_ready = False
        self._stop_health_server()
        self.extq.set_quit(1)
        self.server_socket.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-GPU Aggregator with Slot-Based Batching")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"TCP port for worker registration (default: {DEFAULT_PORT})")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Bind address for TCP server (default: 127.0.0.1; use 0.0.0.0 for multi-node)")
    parser.add_argument("--health-port", type=int, default=AGGREGATOR_HEALTH_PORT,
                        help=f"HTTP port for health checks (default: {AGGREGATOR_HEALTH_PORT})")
    parser.add_argument("--device", type=str, default=None,
                        help=f"Device for aggregator (default: {AGGREGATOR_DEVICE} from config)")
    parser.add_argument("--model", type=str, default=None,
                        help=f"HuggingFace model ID to load (default: {BASE_MODEL_ID} from config)")
    parser.add_argument("--donor-host", type=str, default=None,
                        help="Host of donor aggregator for P2P weight copy (enables donor mode)")
    parser.add_argument("--donor-port", type=int, default=None,
                        help="TCP port of donor aggregator for P2P weight copy")
    parser.add_argument("--num-slots", type=int, default=None,
                        help="Slot pool size (default: MAX_WORKERS_PER_GPU from config). "
                             "Raise this for standalone benchmarks that pack more concurrent "
                             "workers onto this GPU than the scheduler's normal per-GPU cap "
                             "(e.g. RQ5's IPC-overhead sweep) -- otherwise slot contention "
                             "forces some requests to wait for a slot to free up mid-run.")
    args = parser.parse_args()

    server = BatchedAggregator(
        port=args.port,
        host=args.host,
        health_port=args.health_port,
        device=args.device,
        model_id=args.model,
        donor_host=args.donor_host,
        donor_port=args.donor_port,
        num_slots=args.num_slots,
    )

    try:
        server.run()
    except KeyboardInterrupt:
        log("Interrupted")
    finally:
        server.shutdown()
