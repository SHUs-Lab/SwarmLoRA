#!/usr/bin/env python3
"""
Base Model Server - Loads model and exports weights via CUDA IPC.
Workers connect to fetch IPC handles and reconstruct the model.
"""

import sys
import os
import socket
import pickle
import threading
import torch
from transformers import AutoModelForCausalLM
import time

from config import BASE_MODEL_ID, DTYPE, BASELINE_SERVER_PORT, WORKER_DEVICE

try:
    import ext_ipc_wrap as ext
except ImportError:
    print("ERROR: ext_ipc_wrap not found. Run: python setup.py build_ext --inplace")
    sys.exit(1)


class BaseModelServer:
    """Server that loads model once and provides IPC access to weights."""

    def __init__(self, host: str = "0.0.0.0", port: int = BASELINE_SERVER_PORT,
                 model_name: str = BASE_MODEL_ID, device: str = WORKER_DEVICE):
        self.host = host
        self.port = port
        self.model_name = model_name
        self.device = device
        self.active_workers = 0

        print("=" * 60)
        print("BASE MODEL SERVER")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Device: {device}")
        print(f"Port: {port}")
        print("=" * 60)

        print(f"\n[1/3] Loading base model to {device}...")
        t0 = time.perf_counter()
        self.model = self._load_base_model(device)
        print(f"  Model loaded in {time.perf_counter()-t0:.1f}s")

        print(f"\n[2/3] Exporting weights via CUDA IPC...")
        t1 = time.perf_counter()
        self.ipc_metadata = self._export_ipc_metadata()
        print(f"  Exported {self.ipc_metadata['num_tensors']} tensors in {time.perf_counter()-t1:.1f}s")

        # Pre-serialize metadata once (read-only after init)
        self._serialized_metadata = pickle.dumps(self.ipc_metadata)
        self._serialized_len_bytes = len(self._serialized_metadata).to_bytes(4, 'big')
        self._counter_lock = threading.Lock()
        print(f"  Pre-serialized metadata: {len(self._serialized_metadata)} bytes")

        print(f"\n[3/3] Starting server on {host}:{port}...")
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.server_socket.listen(256)

        print("\n" + "=" * 60)
        print("SERVER READY - Workers can now connect")
        print("=" * 60)

    def _load_base_model(self, device: str):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=DTYPE, device_map=device, local_files_only=True
        )
        model.eval()
        return model

    def _export_ipc_metadata(self):
        """Get IPC handles for all model tensors."""
        tensors_meta = []

        for name, param in self.model.named_parameters():
            if param.is_cuda:
                try:
                    handle, dtype_code, offset, dev_idx, _ = ext.get_ipc_details(param)
                    tensors_meta.append({
                        "name": name, "handle": handle, "shape": list(param.shape),
                        "stride": list(param.stride()), "dtype_code": int(dtype_code),
                        "device_index": int(dev_idx), "offset_bytes": int(offset), "kind": "param"
                    })
                except Exception as e:
                    print(f"  Warning: Could not export {name}: {e}")

        for name, buf in self.model.named_buffers():
            if buf.is_cuda:
                try:
                    handle, dtype_code, offset, dev_idx, _ = ext.get_ipc_details(buf)
                    tensors_meta.append({
                        "name": name, "handle": handle, "shape": list(buf.shape),
                        "stride": list(buf.stride()), "dtype_code": int(dtype_code),
                        "device_index": int(dev_idx), "offset_bytes": int(offset), "kind": "buffer"
                    })
                except Exception as e:
                    print(f"  Warning: Could not export {name}: {e}")

        return {
            "tensors": tensors_meta,
            "model_name": self.model_name,
            "num_tensors": len(tensors_meta),
            "model_config": self.model.config.to_dict()
        }

    def handle_client(self, client_socket, addr):
        try:
            client_socket.settimeout(10.0)
            request = client_socket.recv(1024).decode('utf-8').strip()

            if request == "GET_IPC_METADATA":
                client_socket.sendall(self._serialized_len_bytes)
                client_socket.sendall(self._serialized_metadata)
                with self._counter_lock:
                    self.active_workers += 1
                    worker_num = self.active_workers
                print(f"[{time.strftime('%H:%M:%S')}] Served IPC metadata to {addr[0]}:{addr[1]} (worker #{worker_num})")

            elif request == "HEALTH":
                status = {
                    "status": "healthy",
                    "active_workers": self.active_workers,
                    "model_name": self.model_name,
                    "memory_gb": round(torch.cuda.memory_allocated() / (1024**3), 2)
                }
                client_socket.sendall(pickle.dumps(status))
            else:
                client_socket.sendall(f"Unknown request: {request}".encode())

        except socket.timeout:
            pass
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Error handling {addr}: {e}")
        finally:
            client_socket.close()

    def run(self):
        print(f"\nListening for worker connections... (Ctrl+C to stop)\n")
        try:
            while True:
                client_socket, addr = self.server_socket.accept()
                t = threading.Thread(target=self.handle_client, args=(client_socket, addr), daemon=True)
                t.start()
        except KeyboardInterrupt:
            print("\n\nShutting down...")
        finally:
            self.server_socket.close()
            print("Server stopped.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Base Model Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=BASELINE_SERVER_PORT)
    parser.add_argument("--model", default=BASE_MODEL_ID)
    parser.add_argument("--device", default=WORKER_DEVICE)
    args = parser.parse_args()

    server = BaseModelServer(host=args.host, port=args.port, model_name=args.model, device=args.device)
    server.run()
