#!/usr/bin/env python3
"""Template process for fork-based worker spawning."""

import os
import sys
import json
import signal
import socket
import time
import threading

# Prevent CUDA runtime initialization before fork.
# On Python 3.12, PyTorch refuses to re-initialize CUDA in forked children
# if the parent loaded libcudart.  Setting CUDA_VISIBLE_DEVICES="" prevents
# any CUDA driver/runtime init during import.  The forked child restores it
# before calling torch.cuda.set_device().
_saved_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Ensure project root is on sys.path (same as worker_sync.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- Pre-import all heavy worker modules ----
# This is the whole point: these imports take ~5s via subprocess but are
# inherited for free via fork().
_t_import_start = time.perf_counter()
import worker.worker_sync  # noqa: F401 — triggers all transitive imports
_t_import_end = time.perf_counter()

# Restore CUDA_VISIBLE_DEVICES so forked children can use GPU
if _saved_cvd is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _saved_cvd
else:
    del os.environ["CUDA_VISIBLE_DEVICES"]

# ---- Pre-load tokenizer (CPU only, no CUDA) ----
_t_tok_start = time.perf_counter()
from config import BASE_MODEL_ID
from transformers import AutoTokenizer
_preloaded_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, local_files_only=True)
if _preloaded_tokenizer.pad_token_id is None:
    _preloaded_tokenizer.pad_token_id = _preloaded_tokenizer.eos_token_id
_t_tok_end = time.perf_counter()


def _setup_per_gpu_mps(device_str):
    """Set per-GPU MPS env vars before CUDA init."""
    gpu_idx = device_str.replace("cuda:", "")
    pipe_dir = f"/tmp/mps_{gpu_idx}"
    if os.path.exists(pipe_dir):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir
        return "cuda:0"
    return device_str


def _run_child(cmd, listener_fd):
    """Child process after fork: close inherited socket, run worker."""
    # Close the listener socket inherited from parent
    os.close(listener_fd)

    # New session so parent's SIGTERM doesn't cascade to us
    os.setsid()

    # Prevent privilege escalation — worker can never gain new privs via
    # setuid/setgid executables or capability-raising after this point.
    try:
        import ctypes
        PR_SET_NO_NEW_PRIVS = 38
        ctypes.CDLL('libc.so.6', use_errno=True).prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
    except Exception:
        pass  # Non-fatal: prctl may not be available on all platforms

    # Sanitize environment — keep only what the worker actually needs.
    # Removes inherited secrets (HF_TOKEN, API keys, passwords, etc.)
    _ALLOWED_ENV = {
        'CUDA_VISIBLE_DEVICES', 'CUDA_MPS_PIPE_DIRECTORY',
        'CUDA_MPS_LOG_DIRECTORY', 'PYTHONPATH', 'PATH',
        'HOME', 'USER', 'LANG', 'LC_ALL',
        'LD_LIBRARY_PATH', 'TORCH_HOME', 'HF_HOME',
        'PREFILL_CHUNK_SIZE', 'AGG_SECRET',
        'PROFILE_LAYERS',   # RQ5 IPC overhead benchmark (read after fork)
    }
    for key in list(os.environ.keys()):
        if key not in _ALLOWED_ENV:
            del os.environ[key]

    # Redirect stdout/stderr to worker log file (owner-only: 0o600)
    log_path = cmd.get("worker_log_path", "/dev/null")
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.dup2(log_fd, 1)  # stdout
    os.dup2(log_fd, 2)  # stderr
    os.close(log_fd)

    # On Python 3.12+, PyTorch detects native threads from C libraries (libgomp,
    # libtorch_cpu) and refuses CUDA init in forked children.  Reset the internal
    # flag so CUDA init succeeds — safe because the fork serialized on fork_lock
    # and the child has its own address space.
    import torch.cuda
    torch.cuda._initialized = False
    if hasattr(torch.cuda, '_is_in_bad_fork'):
        torch.cuda._is_in_bad_fork = lambda: False
    # Also reset the C++ level flag if available
    try:
        torch._C._cuda_resetAccumulatedMemoryStats(0)
    except Exception:
        pass

    # Set per-GPU MPS env vars before any CUDA call
    device = _setup_per_gpu_mps(cmd.get("device", "cuda:0"))

    # Run the worker
    try:
        worker.worker_sync.fork_main(
            http_port=cmd["http_port"],
            host=cmd.get("host", "localhost"),
            port=cmd.get("port", 50056),
            device=device,
            lora=cmd.get("lora", None),
            preloaded_tokenizer=_preloaded_tokenizer,
        )
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        os._exit(0)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Template process for fork-based worker spawning")
    parser.add_argument("--socket", required=True, help="Path to Unix domain socket")
    args = parser.parse_args()

    socket_path = args.socket

    # Auto-reap zombie children
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)

    # Clean up stale socket
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    # Bind Unix domain socket
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(socket_path)
    srv.listen(128)
    srv.settimeout(1.0)  # Allow periodic SIGTERM checks

    print(f"[Template] Ready (imports={_t_import_end - _t_import_start:.2f}s, "
          f"tokenizer={_t_tok_end - _t_tok_start:.2f}s, "
          f"socket={socket_path})", flush=True)

    def _shutdown(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    running = True
    listener_fd = srv.fileno()

    # Serialize fork() calls — fork is not thread-safe, but we can
    # overlap the socket I/O (recv/send) with fork in another thread.
    fork_lock = threading.Lock()

    def _handle_connection(conn):
        nonlocal running
        try:
            # Read command (single JSON message, max 4KB)
            data = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                if len(data) > 4096:
                    break

            if not data:
                conn.close()
                return

            cmd = json.loads(data.decode())

            if cmd.get("command") == "shutdown":
                conn.sendall(json.dumps({"status": "ok"}).encode())
                conn.close()
                running = False
                return

            # Fork a new worker (serialized — fork is not thread-safe)
            with fork_lock:
                pid = os.fork()
            if pid == 0:
                # Child — inherits _preloaded_tokenizer via COW
                conn.close()
                _run_child(cmd, listener_fd)
                # _run_child calls os._exit(), but just in case:
                os._exit(1)
            else:
                # Parent: send back child PID
                conn.sendall(json.dumps({"pid": pid}).encode())
                conn.close()

        except Exception as e:
            print(f"[Template] Error handling connection: {e}", flush=True)
            try:
                conn.sendall(json.dumps({"error": str(e)}).encode())
                conn.close()
            except Exception:
                pass

    while running:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        except OSError:
            break

        threading.Thread(target=_handle_connection, args=(conn,), daemon=True).start()

    # Cleanup
    srv.close()
    try:
        os.unlink(socket_path)
    except OSError:
        pass
    print("[Template] Shutdown", flush=True)


if __name__ == "__main__":
    main()
