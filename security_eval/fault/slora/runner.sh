#!/usr/bin/env bash
# S-LoRA fault-isolation suite — BR / RT / RLR across SIGKILL / SIGSEGV / SIGABRT.
# All tenants share one model-worker; each signal kills every co-batched request.
# Full server relaunch per mode.
set -u
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
# S-LoRA (serverful baseline) ships in-tree under baselines/slora, with its own
# venv (built separately — see baselines/slora/README.md). Override SLORA to
# point elsewhere if you have a separate checkout.
SLORA="${SLORA:-$ROOT/baselines/slora}"
[ -x "$SLORA/venv/bin/python" ] || {
  echo "[runner] ERROR: $SLORA/venv/bin/python not found."
  echo "[runner]        Build the S-LoRA venv first: cd $SLORA && python3 -m venv venv && venv/bin/pip install -e ."
  exit 1
}
PY=$SLORA/venv/bin/python
SEC=$ROOT/security_eval
OUT=${1:-$SEC/fault/results/slora.json}
NTEN=${NTEN:-8}
PORT="${PORT:-8100}"
WORK=/tmp/fault_slora_work
TMPD=/tmp/fault_slora_modes
mkdir -p "$(dirname "$OUT")" "$TMPD" "$WORK"
rm -f "$TMPD"/*.json

# Resolve real llama-7b model (shared symlink with malicious_adapter/slora)
MODEL="$SEC/malicious_adapter/slora/llama-7b"
[ -f "$MODEL/config.json" ] || {
  SNAP_PATH="${SNAP:-}"
  [ -n "$SNAP_PATH" ] && [ -f "$SNAP_PATH/config.json" ] && ln -sfn "$SNAP_PATH" "$MODEL"
  [ -f "$MODEL/config.json" ] || {
    echo "[runner] llama-7b not found at $MODEL -- downloading (one-time, ~26 GB, public/non-gated)..." >&2
    "$SLORA/venv/bin/huggingface-cli" download huggyllama/llama-7b --local-dir "$MODEL" || {
      echo "[runner] ERROR: automatic download failed. Retry manually:" >&2
      echo "[runner]          baselines/slora/venv/bin/huggingface-cli download huggyllama/llama-7b --local-dir $MODEL" >&2
      exit 1
    }
    [ -f "$MODEL/config.json" ] || {
      echo "[runner] ERROR: download completed but $MODEL/config.json still missing." >&2
      exit 1
    }
  }
}

# Build 2 real safetensors adapters (rank-8 and rank-16) for llama-7b
ADAPTER_R8="$WORK/adapter-rank8"
ADAPTER_R16="$WORK/adapter-rank16"
"$PY" - "$ADAPTER_R8" "$ADAPTER_R16" <<'PY'
import sys, os, json, torch
from safetensors.torch import save_file
H=4096; I=11008; L=32
MODS=[("self_attn.q_proj",H,H),("self_attn.k_proj",H,H),("self_attn.v_proj",H,H),
      ("self_attn.o_proj",H,H),("mlp.gate_proj",H,I),("mlp.up_proj",H,I),("mlp.down_proj",I,H)]
def build(path, rank, seed):
    torch.manual_seed(seed)
    os.makedirs(path, exist_ok=True)
    cfg={"base_model_name_or_path":"huggyllama/llama-7b","bias":"none","inference_mode":True,
         "lora_alpha":16,"lora_dropout":0.0,"peft_type":"LORA","r":rank,
         "target_modules":["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
         "task_type":"CAUSAL_LM"}
    with open(os.path.join(path,"adapter_config.json"),"w") as f: json.dump(cfg,f)
    w={}
    for i in range(L):
        for mod,in_f,out_f in MODS:
            base=f"base_model.model.model.layers.{i}.{mod}"
            w[f"{base}.lora_A.weight"]=torch.randn(rank,in_f,dtype=torch.float16)*0.02
            w[f"{base}.lora_B.weight"]=torch.zeros(out_f,rank,dtype=torch.float16)
    save_file(w, os.path.join(path,"adapter_model.safetensors"))
    print(f"  built {path} (rank={rank})")
build(sys.argv[1], 8,  42)
build(sys.argv[2], 16, 99)
PY
echo "[runner] adapters built: rank-8 and rank-16"

SRV=""
cleanup() {
  echo "[runner] teardown"
  [ -n "${SRV:-}" ] && kill -9 -- -"$SRV" 2>/dev/null || true
  sleep 1
}
trap cleanup EXIT

# Launch the S-LoRA server (setsid -> own process group for clean tree-kill).
launch_server() {
  local logf=$1
  cd "$SLORA"
  setsid env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    "$PY" -m slora.server.api_server --model_dir "$MODEL" --tokenizer_mode auto \
    --lora-dirs "$ADAPTER_R8" --lora-dirs "$ADAPTER_R16" \
    --max_total_token_num 6000 --swap --host 127.0.0.1 --port "$PORT" \
    > "$logf" 2>&1 &
  SRV=$!
  for i in $(seq 1 300); do
    kill -0 "$SRV" 2>/dev/null || { echo "[runner] SERVER DIED"; tail -30 "$logf"; return 1; }
    curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "[runner] health OK ($i s)"; return 0; }
    [ $((i % 30)) -eq 0 ] && echo "[runner] still waiting (${i}s)..."
    sleep 1
  done
  echo "[runner] never healthy"; tail -40 "$logf"; return 1
}

find_mwpid() {
  $PY - <<'PY'
import subprocess, os
out = subprocess.check_output(
    ["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader,nounits"]
).decode().strip().splitlines()
cands=[]
for line in out:
    pid,mem=[x.strip() for x in line.split(",")]; pid=int(pid); mem=int(mem)
    try:
        cl=open(f"/proc/{pid}/cmdline","rb").read().replace(b"\0",b" ").decode(errors="ignore")
        exe=os.path.realpath(f"/proc/{pid}/exe")
    except Exception:
        cl=exe=""
    blob=(cl+" "+exe).lower()
    if "ray" in blob: continue
    if "s-lora" in blob or "slora" in blob: cands.append((mem,pid))
cands.sort(reverse=True)
print(cands[0][1] if cands else 0)
PY
}

build_tenants() {  # echoes 8-tenant JSON (victim=index0)
  $PY - "$NTEN" "$PORT" "$ADAPTER_R8" "$ADAPTER_R16" <<'PY'
import json,sys
n=int(sys.argv[1]); port=sys.argv[2]; ad=[sys.argv[3],sys.argv[4]]
t=[{"id":(f"victim" if k==0 else f"cotenant{k}"),
    "url":f"http://127.0.0.1:{port}/generate",
    "payload":{"inputs":"Tell me a long detailed story about a robot exploring a distant planet.",
               "lora_dir":ad[k%2],
               "parameters":{"do_sample":False,"max_new_tokens":200}}}
   for k in range(n)]
print(json.dumps(t))
PY
}

# ── One fault mode: relaunch server -> drive -> kill shared worker -> measure ──
run_mode() {
  local mode=$1 out=$2 logf=/tmp/fault_slora.$mode.log
  echo ""
  echo "══════════════════════════════════════════════════"
  echo "[runner] FAULT MODE: $mode  (S-LoRA full relaunch)"
  echo "══════════════════════════════════════════════════"
  launch_server "$logf" || return 1
  local MWPID; MWPID=$(find_mwpid)
  echo "[runner] model-worker pid = $MWPID"
  [ "${MWPID:-0}" != "0" ] || { echo "[runner] could not find model worker"; return 1; }
  local TENANTS; TENANTS=$(build_tenants)
  echo "[runner] driving $NTEN tenants; victim=tenant0; killing shared worker pid=$MWPID ($mode)"
  PYTHONPATH=$SEC $PY -m common.fault_driver \
      --system slora --fault-type crash --fault-signal "$mode" \
      --tenants "$TENANTS" --victim-index 0 --victim-pid "$MWPID" \
      --fault-delay 3.0 --req-timeout 20 --recovery-timeout 5 \
      --recovery-url "http://127.0.0.1:$PORT/generate" \
      --out "$out"
  echo "[runner] mode $mode done (rc=$?)"
  [ -n "${SRV:-}" ] && kill -9 -- -"$SRV" 2>/dev/null || true
  SRV=""
  sleep 1
}

for mode in kill segv abort; do
  run_mode "$mode" "$TMPD/$mode.json" || { echo "[runner] mode $mode failed"; }
done

# ── Combine the 3 per-mode results ────────────────────────────────────────────
NTEN=$NTEN "$PY" - "$OUT" "$TMPD/kill.json" "$TMPD/segv.json" "$TMPD/abort.json" <<'PY'
import json,sys,os
out=sys.argv[1]; paths=sys.argv[2:]
modes={}
for p in paths:
    name=os.path.splitext(os.path.basename(p))[0]
    try: modes[name]=json.load(open(p))
    except Exception as e: modes[name]={"error":str(e)}
brs=[m.get("blast_radius") for m in modes.values() if isinstance(m.get("blast_radius"),(int,float))]
all_surv=all(m.get("params",{}).get("all_survivors_completed") for m in modes.values())
res={
  "experiment":"Fault isolation across 3 abrupt-death modes (SIGKILL/SIGSEGV/SIGABRT)",
  "system":"slora",
  "num_tenants":int(os.environ.get("NTEN","8")),
  "fault_modes":modes,
  "summary":{
    "max_blast_radius":max(brs) if brs else None,
    "min_blast_radius":min(brs) if brs else None,
    "all_modes_isolated":all(b==0.0 for b in brs) if brs else False,
    "all_survivors_completed":bool(all_surv),
    "recovered_all_modes":all(m.get("params",{}).get("recovered") for m in modes.values()),
  },
  "note":"S-LoRA shares one model-worker process across all tenants; each fatal signal kills every co-batched request. Full server relaunch per mode.",
}
json.dump(res,open(out,"w"),indent=2)
print("\n=== S-LoRA fault-isolation suite ===")
for name,m in modes.items():
    p=m.get("params",{})
    print(f"  {name:5s}: BR={m.get('blast_radius')}  RLR={m.get('request_loss_rate')}  "
          f"affected={m.get('affected_cotenants')}  recovery_ms={m.get('recovery_time_ms')}")
print(f"  SUMMARY: max_BR={res['summary']['max_blast_radius']}  "
      f"all_isolated={res['summary']['all_modes_isolated']}  "
      f"recovered_all={res['summary']['recovered_all_modes']}")
print(f"  wrote {out}")
PY
echo "[runner] suite complete -> $OUT"
