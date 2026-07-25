"""Worker utility functions: logging, networking, tokenizer loading, profiling."""

import os
import json
import base64
import time
import struct
import socket

from config import BASE_MODEL_ID

# Optional shared secret — set AGG_SECRET env var on aggregator AND all workers
# to enable authentication. If unset, auth is skipped (backward compatible).
AGG_SECRET: str = os.environ.get('AGG_SECRET', '')

_MAX_MSG_BYTES = 200 * 1024 * 1024  # 200 MB safety cap


def _encode(obj):
    """Recursively base64-encode bytes so the object is JSON-serialisable."""
    if isinstance(obj, bytes):
        return {'__b64__': base64.b64encode(obj).decode('ascii')}
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_encode(v) for v in obj]
    return obj


def _decode(obj):
    """Reverse _encode: turn base64 markers back into bytes."""
    if isinstance(obj, dict):
        if list(obj.keys()) == ['__b64__']:
            return base64.b64decode(obj['__b64__'])
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


def send_msg(sock: socket.socket, obj: dict) -> None:
    """Send a length-prefixed JSON message — no pickle."""
    data = json.dumps(_encode(obj)).encode('utf-8')
    sock.sendall(struct.pack('>I', len(data)) + data)


def recv_msg(sock: socket.socket) -> dict:
    """Receive a length-prefixed JSON message — no pickle."""
    raw_len = b''
    while len(raw_len) < 4:
        chunk = sock.recv(4 - len(raw_len))
        if not chunk:
            raise ConnectionError("Connection closed during recv")
        raw_len += chunk
    size = struct.unpack('>I', raw_len)[0]
    if size > _MAX_MSG_BYTES:
        raise ValueError(f"Message too large: {size} bytes")
    data = b''
    while len(data) < size:
        chunk = sock.recv(min(size - len(data), _SOCKET_RECV_BUFFER))
        if not chunk:
            # Peer closed after the length prefix but before the full body —
            # without this check the loop would append b'' forever (100% CPU).
            raise ConnectionError("Connection closed mid-message")
        data += chunk
    return _decode(json.loads(data.decode('utf-8')))

_SOCKET_RECV_BUFFER = 65536
_SOCKET_CONNECT_TIMEOUT = 120.0


def log(worker_id: int, msg: str) -> None:
    print(f"[Worker-{worker_id}] {msg}", flush=True)


def connect(host: str, port: int):
    """Connect to aggregator and register (JSON protocol, no pickle)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(_SOCKET_CONNECT_TIMEOUT)
    sock.connect((host, port))

    msg = {'type': 'register'}
    if AGG_SECRET:
        msg['secret'] = AGG_SECRET
    send_msg(sock, msg)

    info = recv_msg(sock)
    if info.get('type') == 'auth_error':
        sock.close()
        raise ConnectionError("Aggregator rejected connection: invalid secret")
    return sock, info


def signal_done(sock: socket.socket) -> None:
    """Signal completion to aggregator (JSON, no pickle)."""
    send_msg(sock, {'type': 'done'})


def load_tokenizer_from_cache(info: dict, worker_id: int):
    """Load tokenizer for the model the aggregator is actually serving.

    The aggregator may be running a different base model than the config
    default (via --model). Use the model_name it reports in the registration
    info so the worker's tokenizer always matches the vocabulary the
    aggregator samples from.
    """
    from transformers import AutoTokenizer

    model_id = info.get('model_name') or BASE_MODEL_ID
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    t_end = time.perf_counter()
    log(worker_id, f"  TOKENIZER TIMING: load={t_end-t0:.2f}s model={model_id}")
    return tokenizer
