#!/usr/bin/env bash
# SwarmLoRA fault-isolation suite — BR / RT / RLR across SIGKILL / SIGSEGV / SIGABRT.
# Cluster: 1 aggregator + 7 persistent co-tenant workers + 1 idle spare.
# Each mode kills a fresh victim worker; co-tenants + aggregator persist across all 3 modes.
set -u
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
PY=$ROOT/venv/bin/python
SEC=$ROOT/security_eval
OUT=${1:-$SEC/fault/results/swarmlora.json}
NCOTEN=${NCOTEN:-7}          # co-tenant workers (+1 victim = 8 tenants)
LIB=$($PY -c "import torch,os;print(os.path.dirname(torch.__file__)+'/lib')")
export LD_LIBRARY_PATH=$LIB:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset CUDA_MPS_PIPE_DIRECTORY 2>/dev/null || true

LOG=/tmp/fault_swarm
TMPD=/tmp/fault_swarm_modes
mkdir -p "$(dirname "$OUT")" "$TMPD"
rm -f "$TMPD"/*.json
PIDS=()
cleanup() {
  echo "[runner] teardown"
  for p in "${PIDS[@]:-}"; do kill -9 "$p" 2>/dev/null || true; done
  [ -n "${AGG_PID:-}" ] && kill -9 "$AGG_PID" 2>/dev/null || true
  pkill -9 -f "_launch_worker.py" 2>/dev/null || true
  sleep 1
}
trap cleanup EXIT

AGG_PORT=50056
AGG_HEALTH=8000

echo "[runner] starting aggregator (cuda:0 :$AGG_PORT health:$AGG_HEALTH) ..."
cd "$ROOT"
PYTHONPATH=$ROOT/src:$ROOT $PY src/aggregator.py --device cuda:0 --port $AGG_PORT --health-port $AGG_HEALTH \
    > "$LOG.agg.log" 2>&1 &
AGG_PID=$!
for i in $(seq 1 120); do
  curl -sf http://localhost:$AGG_HEALTH/health >/dev/null 2>&1 && { echo "[runner] aggregator healthy ($i s)"; break; }
  kill -0 "$AGG_PID" 2>/dev/null || { echo "[runner] AGG DIED"; tail -30 "$LOG.agg.log"; exit 1; }
  sleep 1
done
curl -sf http://localhost:$AGG_HEALTH/health >/dev/null 2>&1 || { echo "[runner] aggregator never healthy"; tail -40 "$LOG.agg.log"; exit 1; }

wait_ready() {
  local port=$1 pid=$2 logf=$3 ok=0
  for i in $(seq 1 90); do
    st=$(curl -s "http://localhost:$port/health" 2>/dev/null | grep -o '"status": *"ready"')
    [ -n "$st" ] && { echo "[runner] worker $port ready ($i s)"; ok=1; break; }
    kill -0 "$pid" 2>/dev/null || { echo "[runner] worker $port DIED"; tail -25 "$logf"; return 1; }
    sleep 1
  done
  [ "$ok" = 1 ] || { echo "[runner] worker $port never ready"; tail -25 "$logf"; return 1; }
}

launch_worker() {  # args: port lora_idx  -> echoes pid
  local port=$1 idx=$2
  PYTHONPATH=$ROOT/src:$ROOT $PY "$SEC/common/_launch_worker.py" \
      --http-port "$port" --agg-host localhost --agg-port $AGG_PORT \
      --device cuda:0 --lora "../sim-adapters/pool-10-r16/lora-$idx" > "$LOG.w$port.log" 2>&1 &
  echo $!
}

# ── Persistent co-tenant workers (lora-1..lora-NCOTEN on ports 5002..) ─────────
declare -a COPORT COPID
for k in $(seq 1 "$NCOTEN"); do
  port=$((5001+k))
  pid=$(launch_worker "$port" "$k")
  COPORT[$k]=$port; COPID[$k]=$pid; PIDS+=("$pid")
  echo "[runner] launched co-tenant worker port=$port pid=$pid lora=../sim-adapters/pool-10-r16/lora-$k"
done
for k in $(seq 1 "$NCOTEN"); do wait_ready "${COPORT[$k]}" "${COPID[$k]}" "$LOG.w${COPORT[$k]}.log" || exit 1; done

COTENANT_FRAG=$($PY - "$NCOTEN" <<'PY'
import json,sys
n=int(sys.argv[1])
t=[{"id":f"cotenant{k}","url":f"http://localhost:{5001+k}/inference",
    "payload":{"prompt":"Tell me a long detailed story about a robot exploring a distant planet.","max_tokens":60,"do_sample":False}}
   for k in range(1,n+1)]
print(json.dumps(t))
PY
)

# ── Run each fault mode with a fresh victim worker ────────────────────────────
MODES=(kill segv abort)
VICTIM_BASE_PORT=5001
round=0
for mode in "${MODES[@]}"; do
  round=$((round+1))
  vport=$((VICTIM_BASE_PORT + (round-1)*10))   # 5001, 5011, 5021 (avoid socket reuse)
  echo ""
  echo "══════════════════════════════════════════════════"
  echo "[runner] FAULT MODE: $mode  (fresh victim port=$vport lora-0)"
  echo "══════════════════════════════════════════════════"
  VPID=$(launch_worker "$vport" 0)
  PIDS+=("$VPID")
  wait_ready "$vport" "$VPID" "$LOG.w$vport.log" || exit 1

  TENANTS=$($PY - "$vport" "$COTENANT_FRAG" <<'PY'
import json,sys
vport=sys.argv[1]; co=json.loads(sys.argv[2])
victim={"id":"victim","url":f"http://localhost:{vport}/inference",
        "payload":{"prompt":"Tell me a long detailed story about a robot exploring a distant planet.","max_tokens":60,"do_sample":False}}
print(json.dumps([victim]+co))
PY
)
  PYTHONPATH=$SEC $PY -m common.fault_driver \
      --system swarmlora --fault-type crash --fault-signal "$mode" \
      --tenants "$TENANTS" --victim-index 0 --victim-pid "$VPID" \
      --fault-delay 3.0 --req-timeout 65 --recovery-timeout 0 \
      --out "$TMPD/$mode.json"
  echo "[runner] mode $mode done (rc=$?)"
done

# ── Combine the 3 per-mode results ────────────────────────────────────────────
NTEN=$((NCOTEN+1)) "$PY" - "$OUT" "$TMPD/kill.json" "$TMPD/segv.json" "$TMPD/abort.json" <<'PY'
import json,sys,os
out=sys.argv[1]; paths=sys.argv[2:]
modes={}
for p in paths:
    name=os.path.splitext(os.path.basename(p))[0]
    try: modes[name]=json.load(open(p))
    except Exception as e: modes[name]={"error":str(e)}
brs=[m.get("blast_radius") for m in modes.values() if isinstance(m.get("blast_radius"),(int,float))]
all_isolated=all(b==0.0 for b in brs) if brs else False
all_surv=all(m.get("params",{}).get("all_survivors_completed") for m in modes.values())
all_rec=all(m.get("params",{}).get("recovered") for m in modes.values())
res={
  "experiment":"Fault isolation across 3 abrupt-death modes (SIGKILL/SIGSEGV/SIGABRT)",
  "system":"swarmlora",
  "num_tenants":int(os.environ.get("NTEN","8")),
  "fault_modes":modes,
  "summary":{
    "max_blast_radius":max(brs) if brs else None,
    "all_modes_isolated":all_isolated,
    "all_survivors_completed":bool(all_surv),
    "recovered_all_modes":bool(all_rec),
  },
  "note":"Each mode kills a fresh victim worker mid-generation with a different fatal signal; co-tenants + aggregator persist across all modes.",
}
json.dump(res,open(out,"w"),indent=2)
print("\n=== SwarmLoRA fault-isolation suite ===")
for name,m in modes.items():
    p=m.get("params",{})
    print(f"  {name:5s}: BR={m.get('blast_radius')}  RLR={m.get('request_loss_rate')}  "
          f"survivors={p.get('survivors_completed')}/{p.get('cotenants_total')}  "
          f"recovery_ms={m.get('recovery_time_ms')}")
print(f"  SUMMARY: max_BR={res['summary']['max_blast_radius']}  "
      f"all_isolated={res['summary']['all_modes_isolated']}  "
      f"all_survivors_completed={res['summary']['all_survivors_completed']}  "
      f"recovered_all={res['summary']['recovered_all_modes']}")
print(f"  wrote {out}")
PY
echo "[runner] suite complete -> $OUT"
